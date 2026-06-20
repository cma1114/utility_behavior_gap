#!/usr/bin/env python3
"""Analyze whether translation rubric codes predict panel decisions."""

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
from utility_behavior_gap.scripts.run_translation_rubric_pilot import DIMENSIONS


def finite_exp(value: float) -> float:
    if value > 700:
        return math.inf
    if value < -700:
        return 0.0
    return float(math.exp(value))


def flatten_codes(path: Path) -> pd.DataFrame:
    rows = []
    for row in read_jsonl(path):
        if row.get("success") is False:
            continue
        flat: dict[str, Any] = {
            "pair_uid": row["pair_uid"],
            "actor": row.get("actor", ""),
            "contrast": row.get("contrast", ""),
            "panel_signature": row.get("panel_signature", ""),
            "n_high": int(row.get("n_high", 0)),
            "n_low": int(row.get("n_low", 0)),
            "panel_score_high_minus_low": float(row.get("panel_score_high_minus_low", 0.0)),
            "coder_model": row.get("coder_model", ""),
        }
        flat["panel_high_wins"] = int(flat["n_high"] > flat["n_low"])
        flat["panel_low_wins"] = int(flat["n_low"] > flat["n_high"])
        codes = row.get("codes", {})
        for dimension in DIMENSIONS:
            code = codes.get(dimension, {})
            side = str(code.get("winner_side", "unresolved"))
            flat[f"{dimension}_side"] = side
            if side == "high":
                delta = 1.0
            elif side == "low":
                delta = -1.0
            else:
                delta = 0.0
            flat[f"{dimension}_delta"] = delta
            flat[f"{dimension}_reason"] = str(code.get("reason", ""))
        rows.append(flat)
    if not rows:
        raise SystemExit(f"No successful rubric codes found in {path}")
    return pd.DataFrame(rows)


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


def ridge_cv_predictions(x: pd.DataFrame, y: np.ndarray, *, folds: int, seed: int, alpha: float) -> np.ndarray:
    predictions = np.full(len(y), np.nan, dtype=float)
    for test_idx in stratified_folds(y, folds, seed):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        x_train = sm.add_constant(x.loc[train_mask].astype(float), has_constant="add")
        x_test = sm.add_constant(x.loc[test_idx].astype(float), has_constant="add")
        model = sm.GLM(y[train_mask], x_train, family=sm.families.Binomial())
        try:
            result = model.fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=1000)
            predictions[test_idx] = result.predict(x_test)
        except Exception:
            predictions[test_idx] = float(np.mean(y[train_mask]))
    return predictions


def fit_full_logit(x: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    design = sm.add_constant(x.astype(float), has_constant="add")
    model = sm.GLM(y, design, family=sm.families.Binomial())

    def ridge_rows() -> pd.DataFrame:
        result = model.fit_regularized(alpha=0.1, L1_wt=0.0, maxiter=1000)
        return pd.DataFrame(
            [
                {
                    "term": term,
                    "beta_log_odds": float(beta),
                    "odds_ratio": finite_exp(float(beta)),
                    "odds_ratio_ci_low": np.nan,
                    "odds_ratio_ci_high": np.nan,
                    "p_value": np.nan,
                    "fit_type": "ridge_no_inference_glm_unstable",
                }
                for term, beta in zip(design.columns, result.params, strict=False)
            ]
        )

    try:
        result = model.fit()
        ci = result.conf_int()
        rows = []
        for term in result.params.index:
            beta = float(result.params[term])
            rows.append(
                {
                    "term": term,
                    "beta_log_odds": beta,
                    "odds_ratio": finite_exp(beta),
                    "odds_ratio_ci_low": finite_exp(float(ci.loc[term, 0])),
                    "odds_ratio_ci_high": finite_exp(float(ci.loc[term, 1])),
                    "p_value": float(result.pvalues[term]),
                    "fit_type": "glm",
                }
            )
        out = pd.DataFrame(rows)
        cluster_rows = out[out["term"].ne("const")]
        unstable = (
            not np.isfinite(cluster_rows[["beta_log_odds", "odds_ratio_ci_low", "odds_ratio_ci_high"]].to_numpy()).all()
            or cluster_rows["odds_ratio_ci_low"].eq(0).any()
            or np.isinf(cluster_rows["odds_ratio_ci_high"]).any()
        )
        if unstable:
            return ridge_rows()
        return out
    except Exception:
        return ridge_rows()


def dimension_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    panel_direction = np.where(df["panel_high_wins"].eq(1), "high", "low")
    for dimension in DIMENSIONS:
        side = df[f"{dimension}_side"].astype(str)
        directional = side.isin(["high", "low"])
        coverage = float(directional.mean())
        accuracy_when_directional = (
            float((side[directional].to_numpy() == panel_direction[directional]).mean())
            if directional.any()
            else np.nan
        )
        rows.append(
            {
                "dimension": dimension,
                "n": int(len(df)),
                "high_count": int(side.eq("high").sum()),
                "low_count": int(side.eq("low").sum()),
                "tie_count": int(side.eq("tie").sum()),
                "not_applicable_count": int(side.eq("not_applicable").sum()),
                "unresolved_count": int(side.eq("unresolved").sum()),
                "directional_coverage": coverage,
                "matches_panel_when_directional": accuracy_when_directional,
                "mean_delta_high_minus_low": float(df[f"{dimension}_delta"].mean()),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    out = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(out)


def fmt(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("codes_path", type=Path, help="translation_rubric_codes.jsonl from run_translation_rubric_pilot")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--ridge-alpha", type=float, default=0.1)
    parser.add_argument("--out-prefix", type=Path)
    args = parser.parse_args()

    df = flatten_codes(args.codes_path)
    x_cols = [f"{dimension}_delta" for dimension in DIMENSIONS]
    x = df[x_cols].copy()
    y = df["panel_high_wins"].astype(int).to_numpy()

    rubric_sum = x.sum(axis=1).to_numpy()
    rubric_sum_pred = (rubric_sum > 0).astype(int)
    rubric_sum_nonzero = rubric_sum != 0
    cv_pred = ridge_cv_predictions(x, y, folds=args.folds, seed=args.seed, alpha=args.ridge_alpha)
    majority = int(np.mean(y) >= 0.5)

    metrics = pd.DataFrame(
        [
            {
                "metric": "n_pairs",
                "value": len(df),
            },
            {
                "metric": "panel_high_win_rate",
                "value": float(np.mean(y)),
            },
            {
                "metric": "majority_baseline_accuracy",
                "value": float(np.mean(y == majority)),
            },
            {
                "metric": "rubric_sum_accuracy_all_pairs",
                "value": float(np.mean(rubric_sum_pred == y)),
            },
            {
                "metric": "rubric_sum_directional_coverage",
                "value": float(np.mean(rubric_sum_nonzero)),
            },
            {
                "metric": "rubric_sum_accuracy_when_directional",
                "value": float(np.mean(rubric_sum_pred[rubric_sum_nonzero] == y[rubric_sum_nonzero]))
                if rubric_sum_nonzero.any()
                else np.nan,
            },
            {
                "metric": f"{args.folds}_fold_ridge_logit_accuracy",
                "value": float(np.mean((cv_pred >= 0.5).astype(int) == y)),
            },
            {
                "metric": f"{args.folds}_fold_ridge_logit_auc",
                "value": auc_score(y, cv_pred),
            },
            {
                "metric": f"{args.folds}_fold_ridge_logit_brier",
                "value": float(np.mean((cv_pred - y) ** 2)),
            },
        ]
    )
    dim = dimension_summary(df)
    coef = fit_full_logit(x, y)

    out_prefix = args.out_prefix or args.codes_path.with_name("translation_rubric_pilot_analysis")
    flat_path = out_prefix.with_name(out_prefix.name + "_flat_codes.csv")
    metrics_path = out_prefix.with_name(out_prefix.name + "_predictive_metrics.csv")
    dim_path = out_prefix.with_name(out_prefix.name + "_dimension_summary.csv")
    coef_path = out_prefix.with_name(out_prefix.name + "_logit_coefficients.csv")
    md_path = out_prefix.with_name(out_prefix.name + "_summary.md")

    df.to_csv(flat_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    dim.to_csv(dim_path, index=False)
    coef.to_csv(coef_path, index=False)

    metrics_md = metrics.copy()
    metrics_md["value"] = metrics_md["value"].map(lambda v: fmt(v) if isinstance(v, float) else str(v))
    dim_md = dim.copy()
    for col in ["directional_coverage", "matches_panel_when_directional", "mean_delta_high_minus_low"]:
        dim_md[col] = dim_md[col].map(fmt)
    coef_md = coef.copy()
    for col in ["beta_log_odds", "odds_ratio", "odds_ratio_ci_low", "odds_ratio_ci_high", "p_value"]:
        coef_md[col] = coef_md[col].map(fmt)

    lines = [
        "# Translation Rubric Pilot Analysis",
        "",
        f"Codes file: `{args.codes_path}`",
        "",
        "## Predictive Metrics",
        "",
        markdown_table(metrics_md),
        "",
        "## Dimension Summary",
        "",
        markdown_table(dim_md),
        "",
        "## Logistic Coefficients",
        "",
        "Outcome is whether the panel preferred the high/strong side. Predictors are pair-level rubric deltas: +1 if high/strong is better on that dimension, -1 if low/weak is better, 0 for tie/not-applicable.",
        "",
        markdown_table(coef_md),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"flat codes: {flat_path}")
    print(f"metrics: {metrics_path}")
    print(f"dimension summary: {dim_path}")
    print(f"coefficients: {coef_path}")
    print(f"summary: {md_path}")
    print(markdown_table(metrics_md))


if __name__ == "__main__":
    main()
