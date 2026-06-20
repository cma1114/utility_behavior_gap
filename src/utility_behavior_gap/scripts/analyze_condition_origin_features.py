#!/usr/bin/env python3
"""Predict high-vs-low condition origin from generated response features.

This is a response-level companion to the panel-preference feature models.

For each experimental contrast, the script asks which generated-text features
distinguish the high-condition outputs from the low-condition outputs. It fits:

1. A grouped-cross-validated logistic elastic-net screen using all text
   features from the final text-analysis catalog.
2. An unpenalized logistic model on the lasso-selected features, with
   cluster-robust standard errors by pair id, to report beta estimates and
   p-values.

The analysis is run both aggregated across tasks and separately within each
task. Aggregated models standardize feature values within task so task scale
differences do not dominate the classifier.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from statsmodels.stats.multitest import multipletests

from utility_behavior_gap.paths import ANALYSIS


BY_OUTPUT = ANALYSIS / "final_text_analysis_by_output.csv"
FEATURE_DEFINITIONS = ANALYSIS / "final_text_analysis_feature_definitions.csv"
OUT_PREFIX = "condition_origin_features"
RANDOM_SEED = 20260614

CONTRASTS = ["direct_instruction", "amount", "moral", "utility"]
HIGH_CONDITIONS = {
    "direct_high",
    "amount_high",
    "moral_high",
    "utility_high",
}
LOW_CONDITIONS = {
    "direct_low",
    "amount_low",
    "moral_low",
    "utility_low",
}


def log(message: str) -> None:
    print(message, flush=True)


def text_feature_names() -> list[str]:
    defs = pd.read_csv(FEATURE_DEFINITIONS)
    return defs.loc[defs["type"].eq("feature"), "name"].tolist()


def clean_response_rows(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df[df["contrast"].isin(CONTRASTS)].copy()
    out = out[out["condition"].isin(HIGH_CONDITIONS | LOW_CONDITIONS)].copy()
    out["condition_high"] = out["condition"].isin(HIGH_CONDITIONS).astype(int)
    artifact_cols = [
        "missing_output",
        "generation_success_false",
        "empty_output",
    ]
    for col in artifact_cols:
        if col in out.columns:
            out = out[pd.to_numeric(out[col], errors="coerce").fillna(0).eq(0)].copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=feature_cols + ["pair_uid", "condition_high", "task", "contrast"])

    # Keep only complete high/low pairs within each contrast/scope source.
    pair_counts = (
        out.groupby(["contrast", "pair_uid"])["condition_high"]
        .agg(["count", "sum"])
        .reset_index()
    )
    complete = pair_counts[(pair_counts["count"].eq(2)) & (pair_counts["sum"].eq(1))]
    keep = set(zip(complete["contrast"], complete["pair_uid"]))
    out = out[out[["contrast", "pair_uid"]].apply(lambda row: (row["contrast"], row["pair_uid"]) in keep, axis=1)]
    return out.reset_index(drop=True)


def standardize_features(df: pd.DataFrame, feature_cols: list[str], *, by_task: bool) -> pd.DataFrame:
    out = df.copy()
    groups = out.groupby("task", dropna=False) if by_task else [(None, out)]
    for col in feature_cols:
        z = pd.Series(index=out.index, dtype=float)
        for _key, sub in groups:
            values = pd.to_numeric(sub[col], errors="coerce")
            mean = values.mean()
            sd = values.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                z.loc[sub.index] = 0.0
            else:
                z.loc[sub.index] = (values - mean) / sd
        out[col] = z
    return out


def intercept_loss(y_train: np.ndarray, y_test: np.ndarray) -> float:
    p = float(np.mean(y_train))
    p = min(max(p, 1e-6), 1 - 1e-6)
    probs = np.column_stack([np.full_like(y_test, 1 - p, dtype=float), np.full_like(y_test, p, dtype=float)])
    return float(log_loss(y_test, probs, labels=[0, 1]))


def fit_lasso_screen(
    *,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    l1_ratios: list[float],
    c_values: list[float],
    n_splits: int,
) -> tuple[LogisticRegression, dict[str, Any], pd.DataFrame, float]:
    splits = list(GroupKFold(n_splits=n_splits).split(x, y, groups=groups))
    baseline_losses = [intercept_loss(y[train], y[test]) for train, test in splits]
    baseline_loss = float(np.mean(baseline_losses))

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for l1_ratio in l1_ratios:
        for c_value in c_values:
            losses: list[float] = []
            for train, test in splits:
                model = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=l1_ratio,
                    C=c_value,
                    max_iter=8000,
                    random_state=RANDOM_SEED,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", FutureWarning)
                    model.fit(x[train], y[train])
                probs = model.predict_proba(x[test])
                losses.append(float(log_loss(y[test], probs, labels=[0, 1])))
            row = {
                "l1_ratio": l1_ratio,
                "C": c_value,
                "mean_cv_log_loss": float(np.mean(losses)),
                "sd_cv_log_loss": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
            }
            rows.append(row)
            if best is None or row["mean_cv_log_loss"] < best["mean_cv_log_loss"]:
                best = row
    assert best is not None

    final_model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=float(best["l1_ratio"]),
        C=float(best["C"]),
        max_iter=10000,
        random_state=RANDOM_SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        final_model.fit(x, y)
    return final_model, best, pd.DataFrame(rows), baseline_loss


def finite_exp(value: float) -> float:
    if value > 700:
        return math.inf
    if value < -700:
        return 0.0
    return float(math.exp(value))


def fit_inference(
    *,
    df: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    if not selected_features:
        return pd.DataFrame()
    x = sm.add_constant(df[selected_features].astype(float), has_constant="add")
    y = df["condition_high"].astype(int).to_numpy()
    groups = df["pair_uid"].astype(str).to_numpy()
    model = sm.GLM(y, x, family=sm.families.Binomial())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = result.conf_int()
    rows: list[dict[str, Any]] = []
    for term in result.params.index:
        beta = float(result.params[term])
        ci_low = float(ci.loc[term, 0])
        ci_high = float(ci.loc[term, 1])
        rows.append(
            {
                "term": term,
                "term_type": "intercept" if term == "const" else "feature",
                "beta_log_odds": beta,
                "std_error_cluster_pair": float(result.bse[term]),
                "z": float(result.tvalues[term]),
                "p_value": float(result.pvalues[term]),
                "odds_ratio": finite_exp(beta),
                "odds_ratio_ci_low": finite_exp(ci_low),
                "odds_ratio_ci_high": finite_exp(ci_high),
            }
        )
    out = pd.DataFrame(rows)
    feature_mask = out["term_type"].eq("feature")
    out["p_holm_features_within_model"] = np.nan
    out["p_bh_features_within_model"] = np.nan
    if feature_mask.any():
        pvals = out.loc[feature_mask, "p_value"].to_numpy()
        out.loc[feature_mask, "p_holm_features_within_model"] = multipletests(pvals, method="holm")[1]
        out.loc[feature_mask, "p_bh_features_within_model"] = multipletests(pvals, method="fdr_bh")[1]
    return out


def analysis_slices(df: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    slices: list[tuple[str, str, pd.DataFrame]] = []
    for contrast in CONTRASTS:
        cdf = df[df["contrast"].eq(contrast)].copy()
        slices.append((contrast, "all_tasks", cdf))
        for task, tdf in sorted(cdf.groupby("task", dropna=False), key=lambda item: str(item[0])):
            slices.append((contrast, str(task), tdf.copy()))
    return slices


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    show = df.loc[:, columns].head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join("---" for _ in show.columns) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in show.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"| ... | {len(df) - max_rows} more rows omitted |" + " |" * (len(show.columns) - 2))
    return "\n".join(lines)


def write_summary(
    *,
    path: Path,
    performance: pd.DataFrame,
    selected: pd.DataFrame,
    inference: pd.DataFrame,
    features: list[str],
) -> None:
    significant = inference[
        inference["term_type"].eq("feature")
        & inference["p_holm_features_within_model"].lt(0.05)
    ].sort_values(["contrast", "task_scope", "p_holm_features_within_model"])
    lines = [
        "# Condition-Origin Feature Models",
        "",
        "This analysis predicts whether an individual generated response came from the high or low side of its contrast.",
        "It is a condition-origin analysis, not a judge-preference analysis.",
        "",
        "For each contrast, models are fit once across all tasks and once within each task.",
        "The aggregate all-task models standardize features within task before fitting; task-specific models standardize within the task slice.",
        "",
        "Each model first runs a grouped elastic-net logistic lasso screen using pair id as the cross-validation group.",
        "Then an unpenalized logistic model is fit on the lasso-selected features, with cluster-robust standard errors by pair id.",
        "",
        "Candidate features:",
        "",
    ]
    lines.extend(f"- `{feature}`" for feature in features)
    lines.extend(
        [
            "",
            "## Cross-Validated Performance",
            "",
            markdown_table(
                performance,
                [
                    "contrast",
                    "task_scope",
                    "n_pairs",
                    "n_responses",
                    "baseline_cv_log_loss",
                    "elastic_net_cv_log_loss",
                    "log_loss_improvement",
                    "selected_feature_count",
                ],
                max_rows=40,
            ),
            "",
            "## Lasso-Selected Features",
            "",
            markdown_table(
                selected.sort_values(["contrast", "task_scope", "abs_coefficient"], ascending=[True, True, False]),
                ["contrast", "task_scope", "feature", "coefficient"],
                max_rows=80,
            ),
            "",
            "## Holm-Significant Inference Features",
            "",
            markdown_table(
                significant,
                [
                    "contrast",
                    "task_scope",
                    "term",
                    "beta_log_odds",
                    "std_error_cluster_pair",
                    "p_value",
                    "p_holm_features_within_model",
                    "odds_ratio",
                ],
                max_rows=120,
            ),
            "",
            "Interpretation caution: p-values are post-screening and exploratory. Use this to identify response features that distinguish prompt conditions, not as a causal claim that a feature caused the condition effect.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1-ratios", default="0.1,0.5,1.0")
    parser.add_argument("--c-values", default="0.01,0.03,0.1,0.3,1,3,10")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    l1_ratios = [float(value) for value in args.l1_ratios.split(",") if value.strip()]
    c_values = [float(value) for value in args.c_values.split(",") if value.strip()]

    features = text_feature_names()
    feature_cols = features
    log(f"Loading response features: {BY_OUTPUT}")
    raw = pd.read_csv(BY_OUTPUT, low_memory=False)
    df = clean_response_rows(raw, feature_cols)
    log(f"Clean complete-pair responses: {len(df)}")

    perf_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    inference_frames: list[pd.DataFrame] = []
    cv_frames: list[pd.DataFrame] = []

    for contrast, task_scope, slice_df in analysis_slices(df):
        by_task = task_scope == "all_tasks"
        model_df = standardize_features(slice_df, feature_cols, by_task=by_task)
        n_pairs = model_df["pair_uid"].nunique()
        n_responses = len(model_df)
        if n_pairs < args.n_splits:
            log(f"Skipping {contrast}/{task_scope}: only {n_pairs} complete pairs")
            continue
        log(f"Fitting {contrast}/{task_scope}: {n_pairs} pairs, {n_responses} responses")
        x = model_df[feature_cols].to_numpy(dtype=float)
        y = model_df["condition_high"].astype(int).to_numpy()
        groups = model_df["pair_uid"].astype(str).to_numpy()
        lasso, best, cv_grid, baseline_loss = fit_lasso_screen(
            x=x,
            y=y,
            groups=groups,
            feature_names=features,
            l1_ratios=l1_ratios,
            c_values=c_values,
            n_splits=args.n_splits,
        )
        coefs = lasso.coef_[0]
        selected_features = [
            feature for feature, coef in zip(features, coefs) if abs(float(coef)) > 1e-7
        ]
        for feature, coef in zip(features, coefs):
            if abs(float(coef)) > 1e-7:
                selected_rows.append(
                    {
                        "contrast": contrast,
                        "task_scope": task_scope,
                        "feature": feature,
                        "coefficient": float(coef),
                        "abs_coefficient": float(abs(coef)),
                        "best_l1_ratio": float(best["l1_ratio"]),
                        "best_C": float(best["C"]),
                    }
                )
        inf = fit_inference(df=model_df, selected_features=selected_features)
        if not inf.empty:
            inf.insert(0, "contrast", contrast)
            inf.insert(1, "task_scope", task_scope)
            inf.insert(2, "n_pairs", n_pairs)
            inf.insert(3, "n_responses", n_responses)
            inference_frames.append(inf)

        cv_grid.insert(0, "contrast", contrast)
        cv_grid.insert(1, "task_scope", task_scope)
        cv_frames.append(cv_grid)
        perf_rows.append(
            {
                "contrast": contrast,
                "task_scope": task_scope,
                "n_pairs": n_pairs,
                "n_responses": n_responses,
                "best_l1_ratio": float(best["l1_ratio"]),
                "best_C": float(best["C"]),
                "baseline_cv_log_loss": baseline_loss,
                "elastic_net_cv_log_loss": float(best["mean_cv_log_loss"]),
                "log_loss_improvement": baseline_loss - float(best["mean_cv_log_loss"]),
                "pseudo_r2_vs_baseline": 1 - float(best["mean_cv_log_loss"]) / baseline_loss if baseline_loss else math.nan,
                "selected_feature_count": len(selected_features),
            }
        )

    performance = pd.DataFrame(perf_rows)
    selected = pd.DataFrame(selected_rows)
    inference = pd.concat(inference_frames, ignore_index=True) if inference_frames else pd.DataFrame()
    cv_grid = pd.concat(cv_frames, ignore_index=True) if cv_frames else pd.DataFrame()

    performance_path = ANALYSIS / f"{OUT_PREFIX}_cv_performance.csv"
    selected_path = ANALYSIS / f"{OUT_PREFIX}_selected_features.csv"
    inference_path = ANALYSIS / f"{OUT_PREFIX}_inference_coefficients.csv"
    significant_path = ANALYSIS / f"{OUT_PREFIX}_significant_features.csv"
    cv_grid_path = ANALYSIS / f"{OUT_PREFIX}_cv_grid.csv"
    summary_path = ANALYSIS / f"{OUT_PREFIX}_summary.md"

    performance.to_csv(performance_path, index=False)
    selected.to_csv(selected_path, index=False)
    inference.to_csv(inference_path, index=False)
    significant = inference[
        inference["term_type"].eq("feature")
        & inference["p_holm_features_within_model"].lt(0.05)
    ].copy()
    significant.to_csv(significant_path, index=False)
    cv_grid.to_csv(cv_grid_path, index=False)
    write_summary(
        path=summary_path,
        performance=performance,
        selected=selected,
        inference=inference,
        features=features,
    )

    print(f"summary: {summary_path}")
    print(f"performance: {performance_path}")
    print(f"selected features: {selected_path}")
    print(f"inference: {inference_path}")
    print(f"significant features: {significant_path}")
    print(performance.to_string(index=False))


if __name__ == "__main__":
    main()

