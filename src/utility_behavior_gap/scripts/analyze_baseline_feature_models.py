#!/usr/bin/env python3
"""Analyze arm-vs-baseline rubric and generic feature differences."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.paths import ANALYSIS
from utility_behavior_gap.scripts.analyze_task_rubric_feature_models import (
    GENERIC_FEATURES,
    evaluate_model,
    markdown_table,
)
from utility_behavior_gap.scripts.run_task_rubric_pilot import TASK_DIMENSIONS


OUTPUT_FEATURES = ANALYSIS / "final_text_analysis_by_output.csv"
DEFAULT_OUT_PREFIX = ANALYSIS / "baseline_feature_models"


def winner_to_ab_delta(winner: str) -> float:
    if winner == "A":
        return 1.0
    if winner == "B":
        return -1.0
    return 0.0


def winner_role_to_arm_delta(role: str) -> float:
    if role == "arm":
        return 1.0
    if role == "baseline":
        return -1.0
    return 0.0


def infer_dimensions(row: dict[str, Any]) -> list[str]:
    stored = row.get("dimensions")
    if isinstance(stored, list) and stored:
        return [str(value) for value in stored]
    task = str(row.get("task", ""))
    if task in TASK_DIMENSIONS:
        return TASK_DIMENSIONS[task]
    codes = row.get("codes", {})
    if isinstance(codes, dict) and codes:
        return list(codes)
    return []


def load_codes(paths: list[Path]) -> pd.DataFrame:
    rows = []
    all_dimensions: list[str] = []
    for path in paths:
        for raw in read_jsonl(path):
            if raw.get("success") is False:
                continue
            dimensions = infer_dimensions(raw)
            for dimension in dimensions:
                if dimension not in all_dimensions:
                    all_dimensions.append(dimension)
            row: dict[str, Any] = {
                "baseline_pair_uid": raw["baseline_pair_uid"],
                "actor": raw.get("actor", ""),
                "task": raw.get("task", ""),
                "arm_condition": raw.get("arm_condition", ""),
                "baseline_condition": raw.get("baseline_condition", ""),
                "item_label": raw.get("item_label", ""),
                "repeat": raw.get("repeat", ""),
                "a_role": raw.get("a_role", ""),
                "b_role": raw.get("b_role", ""),
                "output_a_id": raw.get("output_a_id", ""),
                "output_b_id": raw.get("output_b_id", ""),
                "arm_output_id": raw.get("arm_output_id", ""),
                "baseline_output_id": raw.get("baseline_output_id", ""),
                "coder_model": raw.get("coder_model", ""),
                "codes_path": str(path),
            }
            codes = raw.get("codes", {})
            for dimension in dimensions:
                code = codes.get(dimension, {})
                winner = str(code.get("winner", "unresolved"))
                role = str(code.get("winner_role", "unresolved"))
                row[f"rubric_ab__{dimension}"] = winner_to_ab_delta(winner)
                row[f"rubric_arm__{dimension}"] = winner_role_to_arm_delta(role)
            rows.append(row)
    if not rows:
        raise SystemExit("No successful baseline rubric codes found.")
    df = pd.DataFrame(rows)
    for dimension in all_dimensions:
        df[f"rubric_ab__{dimension}"] = df.get(f"rubric_ab__{dimension}", 0.0).fillna(0.0)
        df[f"rubric_arm__{dimension}"] = df.get(f"rubric_arm__{dimension}", 0.0).fillna(0.0)
    return df


def add_generic_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.read_csv(OUTPUT_FEATURES, low_memory=False)
    cols = ["output_id"] + [feature for feature in GENERIC_FEATURES if feature in features.columns]
    features = features[cols].dropna(subset=["output_id"]).drop_duplicates("output_id")
    features["output_id"] = features["output_id"].astype(str)
    features = features.set_index("output_id")

    out = df.copy()
    for feature in GENERIC_FEATURES:
        if feature not in features.columns:
            continue
        a = pd.to_numeric(out["output_a_id"].map(features[feature]), errors="coerce")
        b = pd.to_numeric(out["output_b_id"].map(features[feature]), errors="coerce")
        arm = pd.to_numeric(out["arm_output_id"].map(features[feature]), errors="coerce")
        baseline = pd.to_numeric(out["baseline_output_id"].map(features[feature]), errors="coerce")
        out[f"generic_ab__{feature}"] = a - b
        out[f"generic_arm__{feature}"] = arm - baseline
    return out


def feature_columns(df: pd.DataFrame, prefix: str, feature_set: str) -> list[str]:
    rubric = sorted([col for col in df.columns if col.startswith(f"rubric_{prefix}__")])
    generic = sorted([col for col in df.columns if col.startswith(f"generic_{prefix}__")])
    if feature_set == "rubric":
        return rubric
    if feature_set == "generic":
        return generic
    if feature_set == "combined":
        return rubric + generic
    raise ValueError(feature_set)


def group_iterator(df: pd.DataFrame, group_by: str):
    if group_by == "overall":
        yield {"group": "overall", "group_label": "overall"}, df
        return
    keys = group_by.split("+")
    for values, group in df.groupby(keys, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        meta = {"group": group_by, "group_label": ", ".join(f"{k}={v}" for k, v in zip(keys, values, strict=False))}
        for key, value in zip(keys, values, strict=False):
            meta[key] = value
        yield meta, group


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return math.nan, math.nan, math.nan
    mean = float(clean.mean())
    if len(clean) < 2:
        return mean, math.nan, math.nan
    se = float(clean.std(ddof=1) / math.sqrt(len(clean)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def feature_means(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = sorted([col for col in df.columns if col.startswith(("rubric_arm__", "generic_arm__"))])
    for group_by in ["overall", "baseline_condition", "baseline_condition+arm_condition", "baseline_condition+task", "baseline_condition+task+arm_condition"]:
        for meta, group in group_iterator(df, group_by):
            for col in cols:
                mean, low, high = mean_ci(group[col])
                if pd.isna(mean):
                    continue
                family, feature = col.split("__", 1)
                rows.append(
                    {
                        **meta,
                        "feature_family": family.replace("_arm", ""),
                        "feature": feature,
                        "n": int(pd.to_numeric(group[col], errors="coerce").notna().sum()),
                        "mean_arm_minus_baseline": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("codes_paths", nargs="+", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--ridge-alpha", type=float, default=0.1)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["overall", "baseline_condition", "baseline_condition+task", "baseline_condition+arm_condition"],
        choices=[
            "overall",
            "baseline_condition",
            "baseline_condition+task",
            "baseline_condition+arm_condition",
            "baseline_condition+task+arm_condition",
        ],
    )
    args = parser.parse_args()

    df = add_generic_features(load_codes(args.codes_paths))
    df["a_is_arm"] = df["a_role"].astype(str).eq("arm").astype(int)

    model_rows = []
    coef_rows = []
    for group_by in args.group_by:
        for meta, group in group_iterator(df, group_by):
            for feature_set in ["rubric", "generic", "combined"]:
                cols = feature_columns(group, "ab", feature_set)
                metrics, coef = evaluate_model(
                    group,
                    outcome_col="a_is_arm",
                    feature_cols=cols,
                    folds=args.folds,
                    seed=args.seed,
                    alpha=args.ridge_alpha,
                )
                model_rows.append(
                    {
                        **meta,
                        "outcome": "arm_vs_baseline_identity",
                        "feature_set": feature_set,
                        **metrics,
                    }
                )
                if not coef.empty:
                    coef = coef[coef["term"].ne("const")].copy()
                    for _, coef_row in coef.iterrows():
                        coef_rows.append(
                            {
                                **meta,
                                "outcome": "arm_vs_baseline_identity",
                                "feature_set": feature_set,
                                "term": coef_row["term"],
                                "beta_log_odds_per_sd": coef_row["beta_log_odds"],
                                "odds_ratio_per_sd": coef_row["odds_ratio_per_sd"],
                            }
                        )

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    flat_path = out_prefix.with_name(out_prefix.name + "_flat_pairs.csv")
    metrics_path = out_prefix.with_name(out_prefix.name + "_metrics.csv")
    coef_path = out_prefix.with_name(out_prefix.name + "_coefficients.csv")
    means_path = out_prefix.with_name(out_prefix.name + "_feature_means.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.md")

    df.to_csv(flat_path, index=False)
    metrics = pd.DataFrame(model_rows)
    coefs = pd.DataFrame(coef_rows)
    means = feature_means(df)
    metrics.to_csv(metrics_path, index=False)
    coefs.to_csv(coef_path, index=False)
    means.to_csv(means_path, index=False)

    overall = metrics[metrics["group"].eq("overall")].copy()
    by_baseline = metrics[metrics["group"].eq("baseline_condition")].copy()
    lines = [
        "# Baseline Feature Models",
        "",
        f"Pairs: {len(df)}",
        "",
        "Outcome: whether blinded output A is the experimental arm rather than the baseline.",
        "",
        "Feature means report arm-minus-baseline deltas; positive values mean the arm has more of that feature than the baseline.",
        "",
        "## Overall Metrics",
        "",
        markdown_table(
            overall[
                [
                    "feature_set",
                    "n",
                    "positive_rate",
                    "baseline_accuracy",
                    "cv_accuracy",
                    "cv_auc",
                    "cv_brier",
                    "n_features",
                    "status",
                ]
            ]
        ),
        "",
        "## By Baseline Metrics",
        "",
        markdown_table(
            by_baseline[
                [
                    "baseline_condition",
                    "feature_set",
                    "n",
                    "positive_rate",
                    "baseline_accuracy",
                    "cv_accuracy",
                    "cv_auc",
                    "cv_brier",
                    "n_features",
                    "status",
                ]
            ]
        ),
        "",
        "## Outputs",
        "",
        f"- Flat pairs: `{flat_path}`",
        f"- Metrics: `{metrics_path}`",
        f"- Coefficients: `{coef_path}`",
        f"- Feature means: `{means_path}`",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"flat pairs: {flat_path}")
    print(f"metrics: {metrics_path}")
    print(f"coefficients: {coef_path}")
    print(f"feature means: {means_path}")
    print(f"summary: {summary_path}")
    print(markdown_table(overall[["feature_set", "n", "baseline_accuracy", "cv_accuracy", "cv_auc", "cv_brier", "status"]]))


if __name__ == "__main__":
    main()
