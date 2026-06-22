#!/usr/bin/env python3
"""Paired feature deltas for the finalized direct-instruction contrast.

This local analysis uses the final text-feature pair catalog. It compares each
strong direct-instruction output against its matched framed-neutral output:

    direct_high - direct_low

For the current canonical data, direct_high is the exhortative user-prompt arm
and direct_low is the framed-neutral arm. The headline summaries use equal
actor-task cell means and bootstrap confidence intervals over those cells so
the larger translation sample does not dominate the overall estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utility_behavior_gap.constants import ACTOR_LABEL, TASK_LABEL
from utility_behavior_gap.feature_specs import generic_feature_info, standard_generic_feature_ids
from utility_behavior_gap.output_exclusions import (
    filter_semantic_excluded_pair_rows,
    filter_valid_pair_rows,
)
from utility_behavior_gap.paths import ANALYSIS


PAIR_DELTAS = ANALYSIS / "final_text_analysis_pair_deltas.csv"
FEATURE_DEFINITIONS = ANALYSIS / "final_text_analysis_feature_definitions.csv"
OUT_PREFIX = "direct_instruction_feature_deltas"
RANDOM_SEED = 20260615

GENERIC_FEATURE_INFO = generic_feature_info()
STANDARD_GENERIC_FEATURES = standard_generic_feature_ids()
COMPOSITE_FEATURE_DEFINITIONS = {
    feature_id: str(info.get("definition", ""))
    for feature_id, info in GENERIC_FEATURE_INFO.items()
    if "formula" in info
}


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in df.to_numpy()
    ]
    return "\n".join([header, sep] + rows)


def bootstrap_cell_ci(
    cell_values: pd.DataFrame,
    *,
    value_col: str,
    cell_cols: list[str],
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    clean = cell_values.dropna(subset=[value_col]).copy()
    if clean.empty:
        return math.nan, math.nan
    if iterations <= 0:
        return math.nan, math.nan

    if cell_cols == ["actor", "task"]:
        actors = sorted(clean["actor"].dropna().unique().tolist())
        tasks = sorted(clean["task"].dropna().unique().tolist())
        lookup: dict[tuple[str, str], float] = {}
        for row in clean[["actor", "task", value_col]].itertuples(index=False):
            value = float(getattr(row, value_col))
            if np.isfinite(value):
                lookup[(str(row.actor), str(row.task))] = value
        if len(actors) <= 1 or len(tasks) <= 1:
            return math.nan, math.nan
        estimates: list[float] = []
        for _ in range(iterations):
            sampled_actors = rng.choice(actors, size=len(actors), replace=True)
            sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
            vals = [
                lookup[(actor, task)]
                for actor in sampled_actors
                for task in sampled_tasks
                if (actor, task) in lookup
            ]
            if vals:
                estimates.append(float(np.mean(vals)))
    else:
        vals = clean[value_col].to_numpy(dtype=float)
        if len(vals) <= 1:
            return math.nan, math.nan
        estimates = [
            float(np.mean(rng.choice(vals, size=len(vals), replace=True)))
            for _ in range(iterations)
        ]

    if not estimates:
        return math.nan, math.nan
    lo, hi = np.quantile(np.asarray(estimates, dtype=float), [0.025, 0.975])
    return float(lo), float(hi)


def load_feature_definitions(path: Path) -> tuple[list[str], dict[str, str]]:
    definitions = pd.read_csv(path)
    features = definitions.loc[definitions["type"].eq("feature"), "name"].tolist()
    definition_by_name = dict(zip(definitions["name"], definitions["definition"]))
    return features, definition_by_name


def validate_direct_rows(df: pd.DataFrame) -> None:
    expected = df[
        df["high_raw_condition"].eq("framed_user_strong")
        & df["low_raw_condition"].eq("framed_neutral")
        & df["high_condition"].eq("direct_high")
        & df["low_condition"].eq("direct_low")
    ]
    if len(expected) != len(df):
        bad = df.loc[
            ~df.index.isin(expected.index),
            [
                "actor",
                "task",
                "source_dataset",
                "high_condition",
                "low_condition",
                "high_raw_condition",
                "low_raw_condition",
                "run_id",
            ],
        ].head(20)
        raise ValueError(
            "Direct-instruction feature rows are not all framed_user_strong vs framed_neutral.\n"
            + bad.to_string(index=False)
        )


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = [
        "high_numbers",
        "low_numbers",
        "high_percentages",
        "low_percentages",
        "high_words",
        "low_words",
    ]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Cannot compute quantitative_detail; missing columns: {missing}")

    for side in ("high", "low"):
        numeric = pd.to_numeric(out[f"{side}_numbers"], errors="coerce").fillna(0)
        percentages = pd.to_numeric(out[f"{side}_percentages"], errors="coerce").fillna(0)
        words = pd.to_numeric(out[f"{side}_words"], errors="coerce")
        out[f"{side}_quantitative_detail"] = (numeric + percentages) / words * 1000

    out["delta_quantitative_detail"] = (
        out["high_quantitative_detail"] - out["low_quantitative_detail"]
    )
    return out


def available_features(df: pd.DataFrame, definitions_path: Path) -> tuple[list[str], dict[str, str], list[str]]:
    features, definition_by_name = load_feature_definitions(definitions_path)
    features = [feature for feature in STANDARD_GENERIC_FEATURES if feature in features or feature in COMPOSITE_FEATURE_DEFINITIONS]
    definition_by_name.update(COMPOSITE_FEATURE_DEFINITIONS)
    usable: list[str] = []
    skipped: list[str] = []
    for feature in features:
        col = f"delta_{feature}"
        if col not in df.columns:
            skipped.append(feature)
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() == 0:
            skipped.append(feature)
            continue
        usable.append(feature)
    return usable, definition_by_name, skipped


def cell_table(df: pd.DataFrame, *, feature: str, cell_cols: list[str]) -> pd.DataFrame:
    cols = cell_cols + [f"high_{feature}", f"low_{feature}", f"delta_{feature}"]
    clean = df[cols].copy()
    for col in cols:
        if col not in cell_cols:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.dropna(subset=[f"delta_{feature}"])
    if clean.empty:
        return pd.DataFrame(columns=cell_cols + ["n_pairs", "high_mean", "low_mean", "delta_mean"])
    return (
        clean.groupby(cell_cols, dropna=False, sort=True)
        .agg(
            n_pairs=(f"delta_{feature}", "size"),
            high_mean=(f"high_{feature}", "mean"),
            low_mean=(f"low_{feature}", "mean"),
            delta_mean=(f"delta_{feature}", "mean"),
        )
        .reset_index()
    )


def summarize_feature(
    df: pd.DataFrame,
    *,
    feature: str,
    group_cols: list[str],
    cell_cols: list[str],
    iterations: int,
    seed: int,
    definition: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = [((), df)] if not group_cols else list(df.groupby(group_cols, dropna=False, sort=True))
    for group_key, sub in groups:
        group_key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        raw_delta = pd.to_numeric(sub[f"delta_{feature}"], errors="coerce")
        valid = sub.loc[raw_delta.notna()].copy()
        if valid.empty:
            continue
        raw_delta = raw_delta.loc[valid.index].to_numpy(dtype=float)
        raw_high = pd.to_numeric(valid[f"high_{feature}"], errors="coerce").to_numpy(dtype=float)
        raw_low = pd.to_numeric(valid[f"low_{feature}"], errors="coerce").to_numpy(dtype=float)

        cells = cell_table(valid, feature=feature, cell_cols=group_cols + cell_cols)
        value_cols = group_cols + cell_cols + ["delta_mean", "high_mean", "low_mean", "n_pairs"]
        cells = cells[value_cols].copy()
        if group_cols:
            for col, value in zip(group_cols, group_key_tuple):
                cells = cells[cells[col].eq(value)]
        seed_material = repr((feature, group_key_tuple)).encode("utf-8")
        seed_offset = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
        rng_seed = seed + seed_offset
        rng = np.random.default_rng(rng_seed)
        ci_low, ci_high = bootstrap_cell_ci(
            cells.rename(columns={"delta_mean": "value"}),
            value_col="value",
            cell_cols=cell_cols,
            iterations=iterations,
            rng=rng,
        )

        row: dict[str, Any] = {
            "feature": feature,
            "definition": definition,
            "n_pairs": int(len(raw_delta)),
            "n_cells": int(cells["delta_mean"].notna().sum()),
            "strong_mean_equal_cell": float(cells["high_mean"].mean()),
            "neutral_mean_equal_cell": float(cells["low_mean"].mean()),
            "mean_delta_strong_minus_neutral": float(cells["delta_mean"].mean()),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_excludes_zero": bool(
                np.isfinite(ci_low)
                and np.isfinite(ci_high)
                and ((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
            ),
            "raw_pair_mean_delta": float(np.mean(raw_delta)),
            "raw_pair_sd_delta": float(np.std(raw_delta, ddof=1)) if len(raw_delta) > 1 else math.nan,
            "standardized_delta_pair_sd": (
                float(cells["delta_mean"].mean() / np.std(raw_delta, ddof=1))
                if len(raw_delta) > 1 and np.std(raw_delta, ddof=1) > 0
                else math.nan
            ),
            "raw_strong_mean": float(np.nanmean(raw_high)),
            "raw_neutral_mean": float(np.nanmean(raw_low)),
            "pct_pairs_strong_greater": float(np.mean(raw_delta > 0)),
            "pct_pairs_neutral_greater": float(np.mean(raw_delta < 0)),
            "pct_pairs_equal": float(np.mean(raw_delta == 0)),
        }
        for col, value in zip(group_cols, group_key_tuple):
            row[col] = value
            if col == "actor":
                row["actor_label"] = ACTOR_LABEL.get(str(value), str(value))
            if col == "task":
                row["task_label"] = TASK_LABEL.get(str(value), str(value))
        rows.append(row)
    return rows


def summarize_scope(
    df: pd.DataFrame,
    *,
    features: list[str],
    definition_by_name: dict[str, str],
    group_cols: list[str],
    cell_cols: list[str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        rows.extend(
            summarize_feature(
                df,
                feature=feature,
                group_cols=group_cols,
                cell_cols=cell_cols,
                iterations=iterations,
                seed=seed,
                definition=definition_by_name.get(feature, ""),
            )
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    group_prefix = group_cols + [col for col in ("actor_label", "task_label") if col in out.columns]
    ordered = group_prefix + [
        "feature",
        "definition",
        "n_pairs",
        "n_cells",
        "strong_mean_equal_cell",
        "neutral_mean_equal_cell",
        "mean_delta_strong_minus_neutral",
        "ci_low",
        "ci_high",
        "ci_excludes_zero",
        "standardized_delta_pair_sd",
        "raw_pair_mean_delta",
        "raw_pair_sd_delta",
        "raw_strong_mean",
        "raw_neutral_mean",
        "pct_pairs_strong_greater",
        "pct_pairs_neutral_greater",
        "pct_pairs_equal",
    ]
    return out[[col for col in ordered if col in out.columns]].sort_values(
        group_cols + ["feature"] if group_cols else ["feature"]
    )


def compact_table(overall: pd.DataFrame) -> pd.DataFrame:
    selected = overall[overall["feature"].isin(STANDARD_GENERIC_FEATURES)].copy()
    selected["abs_std_delta"] = selected["standardized_delta_pair_sd"].abs()
    selected = selected.sort_values("abs_std_delta", ascending=False)
    return pd.DataFrame(
        {
            "feature": selected["feature"],
            "strong mean": selected["strong_mean_equal_cell"].map(lambda x: fmt(x, 3)),
            "neutral mean": selected["neutral_mean_equal_cell"].map(lambda x: fmt(x, 3)),
            "delta": selected["mean_delta_strong_minus_neutral"].map(lambda x: fmt(x, 3)),
            "95% CI": [
                f"[{fmt(lo, 3)}, {fmt(hi, 3)}]"
                for lo, hi in zip(selected["ci_low"], selected["ci_high"])
            ],
            "std delta": selected["standardized_delta_pair_sd"].map(lambda x: fmt(x, 3)),
        }
    )


def write_summary(
    *,
    out_dir: Path,
    overall: pd.DataFrame,
    by_task: pd.DataFrame,
    skipped_features: list[str],
    input_path: Path,
    definitions_path: Path,
    iterations: int,
    valid_output_filter_report: dict[str, Any],
    semantic_exclusion_filter_report: dict[str, Any],
) -> Path:
    top_pos = overall.sort_values("standardized_delta_pair_sd", ascending=False).head(10)
    top_neg = overall.sort_values("standardized_delta_pair_sd", ascending=True).head(10)
    significant = overall[overall["ci_excludes_zero"]].copy()

    def display(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": df["feature"],
                "delta": df["mean_delta_strong_minus_neutral"].map(lambda x: fmt(x, 3)),
                "95% CI": [
                    f"[{fmt(lo, 3)}, {fmt(hi, 3)}]"
                    for lo, hi in zip(df["ci_low"], df["ci_high"])
                ],
                "std delta": df["standardized_delta_pair_sd"].map(lambda x: fmt(x, 3)),
            }
        )

    task_rows: list[dict[str, str]] = []
    for task, sub in by_task.groupby("task", sort=True):
        sig = sub[sub["ci_excludes_zero"]].copy()
        pos = sig[sig["mean_delta_strong_minus_neutral"].gt(0)]
        neg = sig[sig["mean_delta_strong_minus_neutral"].lt(0)]
        task_rows.append(
            {
                "task": TASK_LABEL.get(str(task), str(task)),
                "significant increases": str(len(pos)),
                "significant decreases": str(len(neg)),
                "largest increase": (
                    pos.sort_values("standardized_delta_pair_sd", ascending=False)["feature"].iloc[0]
                    if not pos.empty
                    else ""
                ),
                "largest decrease": (
                    neg.sort_values("standardized_delta_pair_sd", ascending=True)["feature"].iloc[0]
                    if not neg.empty
                    else ""
                ),
            }
        )

    lines = [
        "# Direct-Instruction Feature Deltas",
        "",
        "Comparison: `direct_high - direct_low`, where `direct_high` is the finalized exhortative user-prompt arm and `direct_low` is framed neutral.",
        "",
        "The primary overall estimate is the equal actor-task-cell mean. Its confidence interval uses a nonparametric bootstrap over actor and task cells, so translation's larger sample does not dominate the aggregate.",
        "",
        "Generic features use the standard paper-facing set from `analysis_specs/feature_definitions.yaml`. Quantitative detail is numeric tokens plus percentage expressions per 1,000 words; MATTR-50 measures fixed-window lexical variety; rare-word rate uses the standardized `wordfreq` English Zipf-frequency scale.",
        "",
        f"- input pair catalog: `{input_path}`",
        f"- feature definitions: `{definitions_path}`",
        f"- bootstrap iterations: `{iterations}`",
        f"- valid-output filter: `{valid_output_filter_report}`",
        f"- semantic exclusion filter: `{semantic_exclusion_filter_report}`",
        f"- direct-instruction pairs: `{int(overall['n_pairs'].max()) if not overall.empty else 0}`",
        f"- actor-task cells: `{int(overall['n_cells'].max()) if not overall.empty else 0}`",
        f"- standard generic features analyzed: `{len(overall)}`",
        f"- skipped feature columns: `{', '.join(skipped_features) if skipped_features else 'none'}`",
        "",
        "## Compact Overall Table",
        "",
        markdown_table(compact_table(overall).head(22)),
        "",
        "## Largest Increases",
        "",
        markdown_table(display(top_pos)),
        "",
        "## Largest Decreases",
        "",
        markdown_table(display(top_neg)),
        "",
        "## Task Summary",
        "",
        markdown_table(pd.DataFrame(task_rows)),
        "",
        "## Outputs",
        "",
        f"- overall: `{out_dir / (OUT_PREFIX + '_overall.csv')}`",
        f"- by task: `{out_dir / (OUT_PREFIX + '_by_task.csv')}`",
        f"- by actor: `{out_dir / (OUT_PREFIX + '_by_actor.csv')}`",
        f"- by actor-task: `{out_dir / (OUT_PREFIX + '_by_actor_task.csv')}`",
        f"- significant overall features: `{out_dir / (OUT_PREFIX + '_overall_significant.csv')}`",
    ]

    summary_path = out_dir / f"{OUT_PREFIX}_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    significant.to_csv(out_dir / f"{OUT_PREFIX}_overall_significant.csv", index=False)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PAIR_DELTAS)
    parser.add_argument("--definitions", type=Path, default=FEATURE_DEFINITIONS)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = pd.read_csv(args.input, low_memory=False)
    direct = pairs[pairs["contrast"].eq("direct_instruction")].copy()
    if direct.empty:
        raise ValueError("No direct_instruction rows found in pair-delta catalog.")
    validate_direct_rows(direct)
    direct, valid_output_filter_report = filter_valid_pair_rows(direct)
    if direct.empty:
        raise ValueError("No direct_instruction rows remain after valid-output filtering.")
    direct, semantic_exclusion_filter_report = filter_semantic_excluded_pair_rows(direct)
    if direct.empty:
        raise ValueError("No direct_instruction rows remain after semantic exclusion filtering.")
    direct = add_composite_features(direct)

    features, definition_by_name, skipped = available_features(direct, args.definitions)

    overall = summarize_scope(
        direct,
        features=features,
        definition_by_name=definition_by_name,
        group_cols=[],
        cell_cols=["actor", "task"],
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    by_task = summarize_scope(
        direct,
        features=features,
        definition_by_name=definition_by_name,
        group_cols=["task"],
        cell_cols=["actor"],
        iterations=args.bootstrap_iterations,
        seed=args.seed + 100,
    )
    by_actor = summarize_scope(
        direct,
        features=features,
        definition_by_name=definition_by_name,
        group_cols=["actor"],
        cell_cols=["task"],
        iterations=args.bootstrap_iterations,
        seed=args.seed + 200,
    )
    by_actor_task = summarize_scope(
        direct,
        features=features,
        definition_by_name=definition_by_name,
        group_cols=["actor", "task"],
        cell_cols=[],
        iterations=args.bootstrap_iterations,
        seed=args.seed + 300,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.out_dir / f"{OUT_PREFIX}_overall.csv", index=False)
    by_task.to_csv(args.out_dir / f"{OUT_PREFIX}_by_task.csv", index=False)
    by_actor.to_csv(args.out_dir / f"{OUT_PREFIX}_by_actor.csv", index=False)
    by_actor_task.to_csv(args.out_dir / f"{OUT_PREFIX}_by_actor_task.csv", index=False)

    summary_path = write_summary(
        out_dir=args.out_dir,
        overall=overall,
        by_task=by_task,
        skipped_features=skipped,
        input_path=args.input,
        definitions_path=args.definitions,
        iterations=args.bootstrap_iterations,
        valid_output_filter_report=valid_output_filter_report,
        semantic_exclusion_filter_report=semantic_exclusion_filter_report,
    )

    print(f"valid-output filter: {valid_output_filter_report}")
    print(f"semantic exclusion filter: {semantic_exclusion_filter_report}")
    print(f"direct pairs: {len(direct)}")
    print(f"features analyzed: {len(features)}")
    print(f"summary: {summary_path}")
    print(f"overall: {args.out_dir / (OUT_PREFIX + '_overall.csv')}")


if __name__ == "__main__":
    main()
