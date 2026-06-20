#!/usr/bin/env python3
"""Model panel preference and condition identity from rubric + generic features.

Inputs are one or more ``task_rubric_codes.jsonl`` files produced by
``run_task_rubric_pilot``. The script joins rubric codes to the existing
pair-level generic text features and evaluates two questions:

1. Panel preference: do high-minus-low feature deltas predict whether the panel
   preferred the high side?
2. Condition identity: do randomized A-minus-B feature deltas predict whether
   output A was the high side?

The second question is the non-tautological version of "can features identify
the high/low arm": A/B presentation was randomized before rubric coding.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.paths import ANALYSIS
from utility_behavior_gap.scripts.run_task_rubric_pilot import TASK_DIMENSIONS


PANEL_DATA = ANALYSIS / "panel_feature_lasso_model_data.csv"
DEFAULT_OUT_PREFIX = ANALYSIS / "task_rubric_feature_models"

GENERIC_FEATURES = [
    "words",
    "sentences",
    "paragraphs",
    "avg_sentence_words",
    "unique_word_ratio",
    "numbers",
    "percentages",
    "textstat_flesch_kincaid_grade",
    "counterargument_markers_per_1k",
    "dash_per_1k",
    "spacy_adjective_rate",
    "positive_words_per_1k",
    "negative_words_per_1k",
]


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace("-", "_")


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    total = 0
    for p in pos:
        wins += np.sum(p > neg)
        wins += 0.5 * np.sum(p == neg)
        total += len(neg)
    return float(wins / total)


def stratified_folds(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in sorted(set(y.tolist())):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        for i, value in enumerate(idx):
            folds[i % k].append(int(value))
    return [np.asarray(sorted(fold), dtype=int) for fold in folds if fold]


def clean_x(x: pd.DataFrame) -> pd.DataFrame:
    out = x.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    keep = []
    for col in out.columns:
        values = out[col].dropna()
        if values.empty:
            continue
        if float(values.std(ddof=0)) == 0.0:
            continue
        keep.append(col)
    out = out[keep]
    for col in out.columns:
        median = out[col].median()
        out[col] = out[col].fillna(0.0 if pd.isna(median) else median)
    return out


def standardize_train_test(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0).replace(0, 1.0)
    return (x_train - mean) / std, (x_test - mean) / std


def ridge_cv_predictions(x: pd.DataFrame, y: np.ndarray, *, folds: int, seed: int, alpha: float) -> np.ndarray:
    predictions = np.full(len(y), np.nan, dtype=float)
    for test_idx in stratified_folds(y, folds, seed):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        if len(set(y[train_mask].tolist())) < 2:
            predictions[test_idx] = float(np.mean(y[train_mask]))
            continue
        x_train, x_test = standardize_train_test(x.iloc[train_mask], x.iloc[test_idx])
        x_train = sm.add_constant(x_train.astype(float), has_constant="add")
        x_test = sm.add_constant(x_test.astype(float), has_constant="add")
        model = sm.GLM(y[train_mask], x_train, family=sm.families.Binomial())
        try:
            result = model.fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=1000)
            predictions[test_idx] = result.predict(x_test)
        except Exception:
            predictions[test_idx] = float(np.mean(y[train_mask]))
    return predictions


def ridge_coefficients(x: pd.DataFrame, y: np.ndarray, *, alpha: float) -> pd.DataFrame:
    x = clean_x(x)
    if x.empty or len(set(y.tolist())) < 2:
        return pd.DataFrame()
    x_std, _ = standardize_train_test(x, x)
    design = sm.add_constant(x_std.astype(float), has_constant="add")
    model = sm.GLM(y, design, family=sm.families.Binomial())
    try:
        result = model.fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=1000)
    except Exception:
        return pd.DataFrame()
    rows = []
    for term, beta in zip(design.columns, result.params, strict=False):
        beta = float(beta)
        rows.append(
            {
                "term": term,
                "beta_log_odds": beta,
                "odds_ratio_per_sd": math.exp(beta) if -700 < beta < 700 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    feature_cols: list[str],
    folds: int,
    seed: int,
    alpha: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    x = clean_x(df[feature_cols])
    y = df[outcome_col].astype(int).to_numpy()
    if len(df) < max(12, folds * 2) or x.empty or len(set(y.tolist())) < 2:
        metrics = {
            "n": len(df),
            "positive_rate": float(np.mean(y)) if len(y) else np.nan,
            "baseline_accuracy": np.nan,
            "cv_accuracy": np.nan,
            "cv_auc": np.nan,
            "cv_brier": np.nan,
            "n_features": int(x.shape[1]),
            "status": "skipped_insufficient_variation",
        }
        return metrics, pd.DataFrame()

    cv_pred = ridge_cv_predictions(x, y, folds=folds, seed=seed, alpha=alpha)
    majority = int(np.mean(y) >= 0.5)
    metrics = {
        "n": len(df),
        "positive_rate": float(np.mean(y)),
        "baseline_accuracy": float(np.mean(y == majority)),
        "cv_accuracy": float(np.mean((cv_pred >= 0.5).astype(int) == y)),
        "cv_auc": auc_score(y, cv_pred),
        "cv_brier": float(np.mean((cv_pred - y) ** 2)),
        "n_features": int(x.shape[1]),
        "status": "ok",
    }
    coef = ridge_coefficients(x, y, alpha=alpha)
    return metrics, coef


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


def winner_to_ab_delta(winner: str) -> float:
    if winner == "A":
        return 1.0
    if winner == "B":
        return -1.0
    return 0.0


def winner_side_to_hl_delta(winner_side: str) -> float:
    if winner_side == "high":
        return 1.0
    if winner_side == "low":
        return -1.0
    return 0.0


def load_rubric_codes(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
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
                "pair_uid": raw["pair_uid"],
                "actor": raw.get("actor", ""),
                "task": raw.get("task", ""),
                "contrast": raw.get("contrast", ""),
                "panel_signature": raw.get("panel_signature", ""),
                "n_high": int(raw.get("n_high", 0)),
                "n_low": int(raw.get("n_low", 0)),
                "a_side": raw.get("a_side", ""),
                "b_side": raw.get("b_side", ""),
                "coder_model": raw.get("coder_model", ""),
                "codes_path": str(path),
            }
            codes = raw.get("codes", {})
            for dimension in dimensions:
                code = codes.get(dimension, {})
                winner = str(code.get("winner", "unresolved"))
                winner_side = str(code.get("winner_side", "unresolved"))
                row[f"rubric_ab__{dimension}"] = winner_to_ab_delta(winner)
                row[f"rubric_hl__{dimension}"] = winner_side_to_hl_delta(winner_side)
            rows.append(row)
    if not rows:
        raise SystemExit("No successful rubric codes found.")
    df = pd.DataFrame(rows)
    for dimension in all_dimensions:
        df[f"rubric_ab__{dimension}"] = df.get(f"rubric_ab__{dimension}", 0.0).fillna(0.0)
        df[f"rubric_hl__{dimension}"] = df.get(f"rubric_hl__{dimension}", 0.0).fillna(0.0)
    return df


def add_generic_features(df: pd.DataFrame) -> pd.DataFrame:
    panel = pd.read_csv(PANEL_DATA, low_memory=False)
    wanted = ["pair_uid"] + [
        col
        for feature in GENERIC_FEATURES
        for col in (f"high_{feature}", f"low_{feature}")
        if col in panel.columns
    ]
    panel = panel[wanted].drop_duplicates("pair_uid")
    out = df.merge(panel, on="pair_uid", how="left")
    a_is_high = out["a_side"].astype(str).eq("high")
    for feature in GENERIC_FEATURES:
        high_col = f"high_{feature}"
        low_col = f"low_{feature}"
        if high_col not in out.columns or low_col not in out.columns:
            continue
        high_minus_low = pd.to_numeric(out[high_col], errors="coerce") - pd.to_numeric(out[low_col], errors="coerce")
        out[f"generic_hl__{feature}"] = high_minus_low
        out[f"generic_ab__{feature}"] = np.where(a_is_high, high_minus_low, -high_minus_low)
    return out


def group_iterator(df: pd.DataFrame, group_by: str):
    if group_by == "overall":
        yield {"group": "overall"}, df
        return
    keys = group_by.split("+")
    for values, group in df.groupby(keys, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        label = ", ".join(f"{key}={value}" for key, value in zip(keys, values, strict=False))
        meta = {"group": group_by, "group_label": label}
        for key, value in zip(keys, values, strict=False):
            meta[key] = value
        yield meta, group


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


def summarize_feature_means(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    hl_cols = sorted([col for col in df.columns if col.startswith(("rubric_hl__", "generic_hl__"))])
    for group_by in ["overall", "task", "contrast", "task+contrast"]:
        for meta, group in group_iterator(df, group_by):
            for col in hl_cols:
                values = pd.to_numeric(group[col], errors="coerce").dropna()
                if values.empty:
                    continue
                rows.append(
                    {
                        **meta,
                        "feature": col,
                        "n": int(values.shape[0]),
                        "mean_high_minus_low": float(values.mean()),
                        "sd_high_minus_low": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    }
                )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    out = [
        "| " + " | ".join(shown.columns) + " |",
        "| " + " | ".join("---" for _ in shown.columns) + " |",
    ]
    for _, row in shown.iterrows():
        out.append("| " + " | ".join(str(row[col]) for col in shown.columns) + " |")
    return "\n".join(out)


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
        default=["overall", "task", "contrast", "task+contrast"],
        choices=["overall", "task", "contrast", "task+contrast"],
    )
    args = parser.parse_args()

    df = load_rubric_codes(args.codes_paths)
    df = add_generic_features(df)
    df["panel_high_win"] = (df["n_high"].astype(int) > df["n_low"].astype(int)).astype(int)
    df["a_is_high"] = df["a_side"].astype(str).eq("high").astype(int)

    model_rows = []
    coef_rows = []
    for outcome_name, outcome_col, prefix in [
        ("panel_preference", "panel_high_win", "hl"),
        ("condition_identity", "a_is_high", "ab"),
    ]:
        for group_by in args.group_by:
            for meta, group in group_iterator(df, group_by):
                for feature_set in ["rubric", "generic", "combined"]:
                    cols = feature_columns(group, prefix, feature_set)
                    metrics, coef = evaluate_model(
                        group,
                        outcome_col=outcome_col,
                        feature_cols=cols,
                        folds=args.folds,
                        seed=args.seed,
                        alpha=args.ridge_alpha,
                    )
                    row = {
                        **meta,
                        "outcome": outcome_name,
                        "feature_set": feature_set,
                        **metrics,
                    }
                    model_rows.append(row)
                    if not coef.empty:
                        coef = coef[coef["term"].ne("const")].copy()
                        for _, coef_row in coef.iterrows():
                            coef_rows.append(
                                {
                                    **meta,
                                    "outcome": outcome_name,
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
    metrics.to_csv(metrics_path, index=False)
    coefs.to_csv(coef_path, index=False)
    summarize_feature_means(df, means_path)

    overall = metrics[metrics["group"].eq("overall")].copy()
    by_task = metrics[metrics["group"].eq("task")].copy()
    lines = [
        "# Task Rubric Feature Models",
        "",
        f"Pairs: {len(df)}",
        "",
        "Panel preference uses high-minus-low feature deltas to predict whether the panel preferred the high side.",
        "",
        "Condition identity uses randomized A-minus-B feature deltas to predict whether output A was the high side.",
        "",
        "## Overall Metrics",
        "",
        markdown_table(
            overall[
                [
                    "outcome",
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
        "## By Task Metrics",
        "",
        markdown_table(
            by_task[
                [
                    "task",
                    "outcome",
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
    print(markdown_table(overall[["outcome", "feature_set", "n", "baseline_accuracy", "cv_accuracy", "cv_auc", "cv_brier", "status"]]))


if __name__ == "__main__":
    main()
