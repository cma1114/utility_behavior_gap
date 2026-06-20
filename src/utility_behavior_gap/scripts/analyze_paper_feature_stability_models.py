#!/usr/bin/env python3
"""Paper-facing text-feature models with data-driven redundancy control.

This script replaces the earlier minimum-loss lasso screen for paper tables.
It uses a stricter model-selection rule:

1. Fit grouped-CV elastic-net logistic models over the full text-feature set.
2. Use the one-standard-error rule: among models whose CV loss is within one
   SE of the best model, choose the sparsest model.
3. Run stability selection at the chosen elastic-net hyperparameters over
   repeated grouped folds.
4. Refit an unpenalized clustered logistic model on the stable selected
   features and report Holm-corrected significant predictors.

The analysis is run for:

* judging-panel preference, using decisive panels and high-side judge wins out
  of three as a binomial outcome;
* condition-origin models, predicting whether a response came from the high or
  low side of each contrast.

Outputs are separate from the exploratory lasso files and use the
`paper_feature_stability_*` prefix.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
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


FEATURE_DEFINITIONS = ANALYSIS / "final_text_analysis_feature_definitions.csv"
PANEL_MODEL_DATA = ANALYSIS / "panel_feature_lasso_model_data.csv"
BY_OUTPUT = ANALYSIS / "final_text_analysis_by_output.csv"

OUT_PREFIX = "paper_feature_stability"
RANDOM_SEED = 20260614
P_THRESHOLD = 0.05

CONTRASTS = ["direct_instruction", "amount", "moral", "utility"]
HIGH_CONDITIONS = {"direct_high", "amount_high", "moral_high", "utility_high"}
LOW_CONDITIONS = {"direct_low", "amount_low", "moral_low", "utility_low"}

CONTRAST_LABELS = {
    "direct_instruction": "Direct instruction",
    "amount": "Amount",
    "moral": "Moral cue",
    "utility": "Utility high-low",
    "judge_panel": "Judging panel",
}

TASK_LABELS = {
    "all_tasks": "All tasks",
    "essay": "Essay",
    "grant_proposal_abstract": "Grant abstract",
    "incident_postmortem": "Incident postmortem",
    "translation": "Translation",
}

FEATURE_LABELS = {
    "words": "Word count",
    "characters": "Character count",
    "sentences": "Sentence count",
    "paragraphs": "Paragraph count",
    "avg_sentence_words": "Avg. sentence length",
    "mean_word_length_chars": "Mean word length",
    "unique_word_ratio": "Unique-word ratio",
    "numbers": "Numeric tokens",
    "percentages": "Percentage expressions",
    "currency_mentions": "Currency mentions",
    "numeric_specificity_per_1k": "Numeric specificity / 1k words",
    "percentages_per_1k": "Percentages / 1k words",
    "semicolon_colon_per_1k": "Semicolon/colon / 1k words",
    "dash_per_1k": "Dash punctuation / 1k words",
    "contrast_markers_per_1k": "Contrast markers / 1k words",
    "counterargument_markers_per_1k": "Counterargument markers / 1k words",
    "example_markers_per_1k": "Example markers / 1k words",
    "transition_markers_per_1k": "Transition markers / 1k words",
    "qualification_markers_per_1k": "Qualification markers / 1k words",
    "method_markers_per_1k": "Method/evaluation markers / 1k words",
    "specificity_markers_per_1k": "Specificity markers / 1k words",
    "positive_words_per_1k": "Positive-word rate",
    "negative_words_per_1k": "Negative-word rate",
    "textstat_flesch_reading_ease": "Flesch reading ease",
    "textstat_flesch_kincaid_grade": "Flesch-Kincaid grade",
    "textstat_gunning_fog": "Gunning Fog index",
    "spacy_adjective_rate": "Adjective rate",
    "spacy_adverb_rate": "Adverb rate",
    "spacy_modifier_rate": "Modifier rate",
}

WIDE_COLUMNS = [
    ("direct_instruction", "Direct instruction"),
    ("amount", "Amount"),
    ("moral", "Moral cue"),
    ("utility", "Utility high-low"),
    ("judge_panel", "Judging panel"),
]

FEATURE_ORDER = [
    "words",
    "characters",
    "sentences",
    "paragraphs",
    "avg_sentence_words",
    "mean_word_length_chars",
    "unique_word_ratio",
    "numbers",
    "percentages",
    "currency_mentions",
    "numeric_specificity_per_1k",
    "percentages_per_1k",
    "semicolon_colon_per_1k",
    "dash_per_1k",
    "contrast_markers_per_1k",
    "counterargument_markers_per_1k",
    "example_markers_per_1k",
    "transition_markers_per_1k",
    "qualification_markers_per_1k",
    "method_markers_per_1k",
    "specificity_markers_per_1k",
    "positive_words_per_1k",
    "negative_words_per_1k",
    "textstat_flesch_reading_ease",
    "textstat_flesch_kincaid_grade",
    "textstat_gunning_fog",
    "spacy_adjective_rate",
    "spacy_adverb_rate",
    "spacy_modifier_rate",
]


@dataclass(frozen=True)
class ModelInput:
    analysis: str
    contrast: str
    task_scope: str
    x: pd.DataFrame
    y: np.ndarray
    weights: np.ndarray
    selection_groups: np.ndarray
    inference_groups: np.ndarray
    n_pairs: int
    n_rows: int


def log(message: str) -> None:
    print(message, flush=True)


def text_feature_names() -> list[str]:
    defs = pd.read_csv(FEATURE_DEFINITIONS)
    return defs.loc[defs["type"].eq("feature"), "name"].tolist()


def finite_exp(value: float) -> float:
    if value > 700:
        return math.inf
    if value < -700:
        return 0.0
    return float(math.exp(value))


def format_float(value: float | str | None) -> str:
    if value is None or value == "" or pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def format_p(value: float | str | None) -> str:
    if value is None or value == "" or pd.isna(value):
        return ""
    value = float(value)
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def format_or_ci(or_value: float | str | None, low: float | str | None, high: float | str | None) -> str:
    if or_value is None or low is None or high is None or pd.isna(or_value) or pd.isna(low) or pd.isna(high):
        return ""
    return f"{float(or_value):.3f} [{float(low):.3f}, {float(high):.3f}]"


def stable_group_splits(
    groups: np.ndarray,
    *,
    n_splits: int,
    repeats: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Repeated grouped folds with deterministic shuffled group order."""
    unique_groups = np.asarray(sorted(pd.unique(groups).tolist(), key=str))
    if len(unique_groups) < n_splits:
        raise ValueError(f"need at least {n_splits} groups, got {len(unique_groups)}")
    rng = np.random.default_rng(seed)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    indices = np.arange(len(groups))
    for repeat in range(repeats):
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)
        fold_groups = np.array_split(shuffled, n_splits)
        for test_groups in fold_groups:
            test_mask = np.isin(groups, test_groups)
            train_idx = indices[~test_mask]
            test_idx = indices[test_mask]
            splits.append((train_idx, test_idx))
    return splits


def simple_group_splits(groups: np.ndarray, *, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    dummy_x = np.zeros((len(groups), 1))
    dummy_y = np.zeros(len(groups))
    return list(GroupKFold(n_splits=n_splits).split(dummy_x, dummy_y, groups=groups))


def intercept_log_loss(
    y_train: np.ndarray,
    w_train: np.ndarray,
    y_test: np.ndarray,
    w_test: np.ndarray,
) -> float:
    p = float(np.average(y_train, weights=w_train))
    p = min(max(p, 1e-6), 1 - 1e-6)
    probs = np.column_stack([np.full_like(y_test, 1 - p, dtype=float), np.full_like(y_test, p, dtype=float)])
    return float(log_loss(y_test, probs, sample_weight=w_test, labels=[0, 1]))


def fit_elastic_net(
    *,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    l1_ratio: float,
    c_value: float,
    seed: int,
) -> LogisticRegression:
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        C=c_value,
        max_iter=6000,
        random_state=seed,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(x, y, sample_weight=weights)
    return model


def grid_search_one_se(
    *,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    l1_ratios: list[float],
    c_values: list[float],
    n_splits: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    splits = simple_group_splits(groups, n_splits=n_splits)
    rows: list[dict[str, Any]] = []
    for l1_ratio in l1_ratios:
        for c_value in c_values:
            losses: list[float] = []
            for train_idx, test_idx in splits:
                model = fit_elastic_net(
                    x=x[train_idx],
                    y=y[train_idx],
                    weights=weights[train_idx],
                    l1_ratio=l1_ratio,
                    c_value=c_value,
                    seed=seed,
                )
                probs = model.predict_proba(x[test_idx])
                losses.append(float(log_loss(y[test_idx], probs, sample_weight=weights[test_idx], labels=[0, 1])))
            full_model = fit_elastic_net(
                x=x,
                y=y,
                weights=weights,
                l1_ratio=l1_ratio,
                c_value=c_value,
                seed=seed,
            )
            selected_count = int(np.sum(np.abs(full_model.coef_[0]) > 1e-7))
            rows.append(
                {
                    "l1_ratio": float(l1_ratio),
                    "C": float(c_value),
                    "mean_cv_log_loss": float(np.mean(losses)),
                    "sd_cv_log_loss": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
                    "se_cv_log_loss": float(np.std(losses, ddof=1) / math.sqrt(len(losses))) if len(losses) > 1 else 0.0,
                    "selected_feature_count_full_data": selected_count,
                    "selected_features_full_data": ";".join(
                        feature
                        for feature, coef in zip(feature_names, full_model.coef_[0], strict=False)
                        if abs(float(coef)) > 1e-7
                    ),
                }
            )
    grid = pd.DataFrame(rows)
    best_idx = int(grid["mean_cv_log_loss"].idxmin())
    best = grid.loc[best_idx].to_dict()
    threshold = float(best["mean_cv_log_loss"] + best["se_cv_log_loss"])
    eligible = grid[grid["mean_cv_log_loss"].le(threshold)].copy()
    eligible = eligible.sort_values(
        ["selected_feature_count_full_data", "C", "l1_ratio", "mean_cv_log_loss"],
        ascending=[True, True, False, True],
    )
    chosen = eligible.iloc[0].to_dict()
    chosen["best_mean_cv_log_loss"] = float(best["mean_cv_log_loss"])
    chosen["best_se_cv_log_loss"] = float(best["se_cv_log_loss"])
    chosen["one_se_threshold"] = threshold
    chosen["one_se_eligible_models"] = int(len(eligible))
    return chosen, grid


def stability_selection(
    *,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    l1_ratio: float,
    c_value: float,
    n_splits: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    splits = stable_group_splits(groups, n_splits=n_splits, repeats=repeats, seed=seed)
    selected_counts = {feature: 0 for feature in feature_names}
    coefficient_sums = {feature: 0.0 for feature in feature_names}
    for split_idx, (train_idx, _test_idx) in enumerate(splits):
        model = fit_elastic_net(
            x=x[train_idx],
            y=y[train_idx],
            weights=weights[train_idx],
            l1_ratio=l1_ratio,
            c_value=c_value,
            seed=seed + split_idx,
        )
        for feature, coef in zip(feature_names, model.coef_[0], strict=False):
            coef = float(coef)
            if abs(coef) > 1e-7:
                selected_counts[feature] += 1
                coefficient_sums[feature] += coef
    n_fits = len(splits)
    rows = []
    for feature in feature_names:
        count = selected_counts[feature]
        rows.append(
            {
                "term": feature,
                "selection_frequency": count / n_fits,
                "selected_fit_count": count,
                "stability_fit_count": n_fits,
                "mean_nonzero_coefficient": coefficient_sums[feature] / count if count else 0.0,
            }
        )
    return pd.DataFrame(rows)


def fit_clustered_inference(
    *,
    model_input: ModelInput,
    selected_features: list[str],
) -> pd.DataFrame:
    if not selected_features:
        return pd.DataFrame()
    repeat_counts = model_input.weights.astype(int)
    if not np.allclose(model_input.weights, repeat_counts):
        raise ValueError("clustered inference expects integer binomial weights")
    x_base = sm.add_constant(model_input.x[selected_features].astype(float), has_constant="add")
    x = x_base.loc[x_base.index.repeat(repeat_counts)].reset_index(drop=True)
    y = np.repeat(model_input.y, repeat_counts)
    groups = np.repeat(model_input.inference_groups, repeat_counts)
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
                "analysis": model_input.analysis,
                "contrast": model_input.contrast,
                "task_scope": model_input.task_scope,
                "term": term,
                "term_type": "intercept" if term == "const" else "feature",
                "beta_log_odds": beta,
                "std_error_cluster": float(result.bse[term]),
                "z": float(result.tvalues[term]),
                "p_value": float(result.pvalues[term]),
                "odds_ratio": finite_exp(beta),
                "odds_ratio_ci_low": finite_exp(ci_low),
                "odds_ratio_ci_high": finite_exp(ci_high),
                "n_pairs": model_input.n_pairs,
                "n_rows": model_input.n_rows,
                "n_clusters": int(len(pd.unique(model_input.inference_groups))),
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


def drop_zero_variance_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if np.isfinite(values.std(ddof=0)) and not np.isclose(float(values.std(ddof=0)), 0.0):
            keep.append(col)
    return df[keep].copy()


def standardize_frame(df: pd.DataFrame, feature_cols: list[str], *, by_task: bool) -> pd.DataFrame:
    out = df.copy()
    groups = out.groupby("task", dropna=False) if by_task else [(None, out)]
    for col in feature_cols:
        z = pd.Series(index=out.index, dtype=float)
        for _key, sub in groups:
            values = pd.to_numeric(sub[col], errors="coerce")
            mean = values.mean()
            sd = values.std(ddof=0)
            if not np.isfinite(sd) or np.isclose(float(sd), 0.0):
                z.loc[sub.index] = 0.0
            else:
                z.loc[sub.index] = (values - mean) / sd
        out[col] = z
    return out


def panel_inputs(feature_names: list[str]) -> list[ModelInput]:
    df = pd.read_csv(PANEL_MODEL_DATA, low_memory=False)
    allowed = {"HHH", "HHL", "HLL", "LLL"}
    bad = sorted(set(df["panel_signature"]) - allowed)
    if bad:
        raise ValueError(f"panel model data contains non-decisive signatures: {bad}")
    if "run_id" not in df.columns:
        df["run_id"] = df["run_dir"].astype(str)
    feature_cols = [f"delta_{feature}" for feature in feature_names if f"delta_{feature}" in df.columns]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=feature_cols + ["n_high", "n_low", "run_id"]).copy()

    inputs: list[ModelInput] = []
    for task_scope, sub in [("all_tasks", df)] + [(str(task), tdf.copy()) for task, tdf in sorted(df.groupby("task"))]:
        x = sub[feature_cols].copy()
        x.columns = [col.removeprefix("delta_") for col in feature_cols]
        x = drop_zero_variance_columns(x)
        y_rows: list[int] = []
        w_rows: list[float] = []
        x_rows: list[pd.Series] = []
        selection_groups: list[str] = []
        inference_groups: list[str] = []
        for idx, row in sub.iterrows():
            high = int(row["n_high"])
            low = int(row["n_low"])
            pair_group = str(row.get("pair_uid", idx))
            run_group = str(row["run_id"])
            if high:
                x_rows.append(x.loc[idx])
                y_rows.append(1)
                w_rows.append(float(high))
                selection_groups.append(pair_group)
                inference_groups.append(run_group)
            if low:
                x_rows.append(x.loc[idx])
                y_rows.append(0)
                w_rows.append(float(low))
                selection_groups.append(pair_group)
                inference_groups.append(run_group)
        inputs.append(
            ModelInput(
                analysis="Judging panel",
                contrast="judge_panel",
                task_scope=task_scope,
                x=pd.DataFrame(x_rows).reset_index(drop=True),
                y=np.asarray(y_rows, dtype=int),
                weights=np.asarray(w_rows, dtype=float),
                selection_groups=np.asarray(selection_groups),
                inference_groups=np.asarray(inference_groups),
                n_pairs=int(len(sub)),
                n_rows=int(sum(sub["n_high"].astype(int) + sub["n_low"].astype(int))),
            )
        )
    return inputs


def clean_condition_rows(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    out = df[df["contrast"].isin(CONTRASTS)].copy()
    out = out[out["condition"].isin(HIGH_CONDITIONS | LOW_CONDITIONS)].copy()
    out["condition_high"] = out["condition"].isin(HIGH_CONDITIONS).astype(int)
    for col in ["missing_output", "generation_success_false", "empty_output"]:
        if col in out.columns:
            out = out[pd.to_numeric(out[col], errors="coerce").fillna(0).eq(0)].copy()
    for feature in feature_names:
        out[feature] = pd.to_numeric(out[feature], errors="coerce")
    out = out.dropna(subset=feature_names + ["pair_uid", "condition_high", "task", "contrast"])
    pair_counts = out.groupby(["contrast", "pair_uid"])["condition_high"].agg(["count", "sum"]).reset_index()
    complete = pair_counts[(pair_counts["count"].eq(2)) & (pair_counts["sum"].eq(1))]
    keep = set(zip(complete["contrast"], complete["pair_uid"], strict=False))
    out = out[out[["contrast", "pair_uid"]].apply(lambda row: (row["contrast"], row["pair_uid"]) in keep, axis=1)]
    return out.reset_index(drop=True)


def condition_inputs(feature_names: list[str]) -> list[ModelInput]:
    raw = pd.read_csv(BY_OUTPUT, low_memory=False)
    df = clean_condition_rows(raw, feature_names)
    inputs: list[ModelInput] = []
    for contrast in CONTRASTS:
        contrast_df = df[df["contrast"].eq(contrast)].copy()
        slices = [("all_tasks", contrast_df)]
        slices.extend((str(task), tdf.copy()) for task, tdf in sorted(contrast_df.groupby("task")))
        for task_scope, sub in slices:
            by_task = task_scope == "all_tasks"
            model_df = standardize_frame(sub, feature_names, by_task=by_task)
            x = drop_zero_variance_columns(model_df[feature_names].copy())
            inputs.append(
                ModelInput(
                    analysis="Condition origin",
                    contrast=contrast,
                    task_scope=task_scope,
                    x=x.reset_index(drop=True),
                    y=model_df["condition_high"].astype(int).to_numpy(),
                    weights=np.ones(len(model_df), dtype=float),
                    selection_groups=model_df["pair_uid"].astype(str).to_numpy(),
                    inference_groups=model_df["pair_uid"].astype(str).to_numpy(),
                    n_pairs=int(model_df["pair_uid"].nunique()),
                    n_rows=int(len(model_df)),
                )
            )
    return inputs


def run_stability_model(
    *,
    model_input: ModelInput,
    l1_ratios: list[float],
    c_values: list[float],
    n_splits: int,
    stability_repeats: int,
    stability_threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_names = model_input.x.columns.tolist()
    x = model_input.x.to_numpy(dtype=float)
    chosen, grid = grid_search_one_se(
        x=x,
        y=model_input.y,
        weights=model_input.weights,
        groups=model_input.selection_groups,
        feature_names=feature_names,
        l1_ratios=l1_ratios,
        c_values=c_values,
        n_splits=n_splits,
        seed=RANDOM_SEED,
    )
    stability = stability_selection(
        x=x,
        y=model_input.y,
        weights=model_input.weights,
        groups=model_input.selection_groups,
        feature_names=feature_names,
        l1_ratio=float(chosen["l1_ratio"]),
        c_value=float(chosen["C"]),
        n_splits=n_splits,
        repeats=stability_repeats,
        seed=RANDOM_SEED,
    )
    selected = stability[stability["selection_frequency"].ge(stability_threshold)].copy()
    selected_features = selected["term"].tolist()
    inference = fit_clustered_inference(model_input=model_input, selected_features=selected_features)

    baseline_losses = []
    for train_idx, test_idx in simple_group_splits(model_input.selection_groups, n_splits=n_splits):
        baseline_losses.append(
            intercept_log_loss(
                model_input.y[train_idx],
                model_input.weights[train_idx],
                model_input.y[test_idx],
                model_input.weights[test_idx],
            )
        )
    perf = {
        "analysis": model_input.analysis,
        "contrast": model_input.contrast,
        "task_scope": model_input.task_scope,
        "n_pairs": model_input.n_pairs,
        "n_rows": model_input.n_rows,
        "candidate_features": len(feature_names),
        "baseline_cv_log_loss": float(np.mean(baseline_losses)),
        "best_mean_cv_log_loss": float(chosen["best_mean_cv_log_loss"]),
        "best_se_cv_log_loss": float(chosen["best_se_cv_log_loss"]),
        "one_se_threshold": float(chosen["one_se_threshold"]),
        "chosen_mean_cv_log_loss": float(chosen["mean_cv_log_loss"]),
        "chosen_l1_ratio": float(chosen["l1_ratio"]),
        "chosen_C": float(chosen["C"]),
        "one_se_eligible_models": int(chosen["one_se_eligible_models"]),
        "chosen_full_data_feature_count": int(chosen["selected_feature_count_full_data"]),
        "stable_selected_feature_count": int(len(selected_features)),
        "stable_selected_features": ";".join(selected_features),
    }
    for frame in [grid, stability, inference]:
        if not frame.empty:
            if "analysis" not in frame.columns:
                frame.insert(0, "analysis", model_input.analysis)
            if "contrast" not in frame.columns:
                frame.insert(1, "contrast", model_input.contrast)
            if "task_scope" not in frame.columns:
                frame.insert(2, "task_scope", model_input.task_scope)
    return perf, grid, stability, inference


def add_none_rows(inference: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sig_keys = set()
    if not inference.empty:
        sig = inference[inference["term_type"].eq("feature") & inference["p_holm_features_within_model"].lt(P_THRESHOLD)]
        sig_keys = set(zip(sig["analysis"], sig["contrast"], sig["task_scope"], strict=False))
    for _, row in performance.iterrows():
        key = (row["analysis"], row["contrast"], row["task_scope"])
        if key not in sig_keys:
            rows.append(
                {
                    "analysis": row["analysis"],
                    "contrast": row["contrast"],
                    "task_scope": row["task_scope"],
                    "term": "none",
                    "term_type": "feature",
                    "beta_log_odds": np.nan,
                    "std_error_cluster": np.nan,
                    "z": np.nan,
                    "p_value": np.nan,
                    "odds_ratio": np.nan,
                    "odds_ratio_ci_low": np.nan,
                    "odds_ratio_ci_high": np.nan,
                    "n_pairs": int(row["n_pairs"]),
                    "n_rows": int(row["n_rows"]),
                    "n_clusters": np.nan,
                    "p_holm_features_within_model": np.nan,
                    "p_bh_features_within_model": np.nan,
                }
            )
    return pd.DataFrame(rows)


def significant_with_none(inference: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    sig = pd.DataFrame()
    if not inference.empty:
        sig = inference[
            inference["term_type"].eq("feature")
            & inference["p_holm_features_within_model"].lt(P_THRESHOLD)
        ].copy()
    none = add_none_rows(inference, performance)
    return pd.concat([sig, none], ignore_index=True, sort=False)


def paper_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Analysis"] = out["analysis"]
    out["Contrast"] = out["contrast"].map(CONTRAST_LABELS).fillna(out["contrast"])
    out["Task"] = out["task_scope"].map(TASK_LABELS).fillna(out["task_scope"])
    out["Predictor"] = out["term"].map(FEATURE_LABELS).fillna(out["term"])
    out.loc[out["term"].eq("none"), "Predictor"] = "None significant"
    out["Direction"] = np.where(out["beta_log_odds"].gt(0), "positive", np.where(out["beta_log_odds"].lt(0), "negative", ""))
    out["Beta"] = out["beta_log_odds"].map(format_float)
    out["OR"] = out["odds_ratio"].map(format_float)
    out["OR 95% CI"] = [
        format_or_ci(or_value, low, high)
        for or_value, low, high in zip(
            out["odds_ratio"],
            out["odds_ratio_ci_low"],
            out["odds_ratio_ci_high"],
            strict=False,
        )
    ]
    out["Holm p"] = out["p_holm_features_within_model"].map(format_p)
    out["n pairs"] = out["n_pairs"].astype("Int64")
    return out[
        [
            "Analysis",
            "Contrast",
            "Task",
            "Predictor",
            "Direction",
            "Beta",
            "OR",
            "OR 95% CI",
            "Holm p",
            "n pairs",
        ]
    ]


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def wide_table(sig: pd.DataFrame, *, task_scope: str | None) -> pd.DataFrame:
    if task_scope is not None:
        sig = sig[sig["task_scope"].eq(task_scope)].copy()
    sig = sig[sig["term"].ne("none")].copy()
    terms = set(sig["term"])
    ordered_terms = [term for term in FEATURE_ORDER if term in terms]
    ordered_terms.extend(sorted(terms - set(ordered_terms)))
    rows: list[dict[str, str]] = []
    for term in ordered_terms:
        row = {"Predictor": FEATURE_LABELS.get(term, term)}
        for contrast, label in WIDE_COLUMNS:
            matches = sig[sig["contrast"].eq(contrast) & sig["term"].eq(term)]
            if matches.empty:
                row[label] = ""
            else:
                match = matches.iloc[0]
                row[label] = format_or_ci(
                    match["odds_ratio"],
                    match["odds_ratio_ci_low"],
                    match["odds_ratio_ci_high"],
                )
        rows.append(row)
    return pd.DataFrame(rows, columns=["Predictor"] + [label for _contrast, label in WIDE_COLUMNS])


def wide_by_task_table(sig: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for task_scope in ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]:
        task_wide = wide_table(sig, task_scope=task_scope)
        if task_wide.empty:
            task_wide = pd.DataFrame({"Predictor": ["None significant"]})
            for _contrast, label in WIDE_COLUMNS:
                task_wide[label] = ""
        task_wide.insert(0, "Task", TASK_LABELS.get(task_scope, task_scope))
        rows.append(task_wide)
    return pd.concat(rows, ignore_index=True, sort=False)


def write_wide_task_markdown(path: Path, table: pd.DataFrame) -> None:
    lines = [
        "# Stability-Selected Task-Specific Predictor Table",
        "",
        "Cells show odds ratio [95% CI] for predictors significant after Holm correction in the stability-selected refit.",
        "Empty cells indicate that the predictor was not significant in that task/column.",
        "",
    ]
    for task, sub in table.groupby("Task", sort=False):
        lines.extend([f"## {task}", "", markdown_table(sub.drop(columns=["Task"]).reset_index(drop=True)), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    *,
    path: Path,
    performance: pd.DataFrame,
    selected: pd.DataFrame,
    significant: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Paper Feature Stability Models",
        "",
        "This analysis uses grouped-CV elastic-net models with one-standard-error selection and stability selection.",
        "It is intended for paper-facing text-feature predictor tables, where redundant features should be handled statistically rather than hidden after fitting.",
        "",
        "Selection rule:",
        "",
        f"- Grid l1 ratios: `{args.l1_ratios}`.",
        f"- Grid C values: `{args.c_values}`.",
        f"- CV folds: `{args.n_splits}` grouped by pair.",
        "- One-SE rule: choose the sparsest model within one SE of the minimum CV log loss.",
        f"- Stability selection: `{args.stability_repeats}` repeated grouped fold partitions.",
        f"- Stable predictor threshold: selection frequency >= `{args.stability_threshold}`.",
        "- Inference: unpenalized logistic refit on stable selected predictors, with clustered standard errors.",
        "- Table inclusion: Holm-corrected p < .05 within the refit model.",
        "",
        "## Performance And Selection Counts",
        "",
        markdown_table(
            performance[
                [
                    "analysis",
                    "contrast",
                    "task_scope",
                    "n_pairs",
                    "candidate_features",
                    "baseline_cv_log_loss",
                    "best_mean_cv_log_loss",
                    "chosen_mean_cv_log_loss",
                    "chosen_l1_ratio",
                    "chosen_C",
                    "chosen_full_data_feature_count",
                    "stable_selected_feature_count",
                ]
            ]
        ),
        "",
        "## Stable Selected Features",
        "",
        markdown_table(
            selected[
                [
                    "analysis",
                    "contrast",
                    "task_scope",
                    "term",
                    "selection_frequency",
                    "mean_nonzero_coefficient",
                ]
            ].sort_values(["analysis", "contrast", "task_scope", "selection_frequency"], ascending=[True, True, True, False])
        ),
        "",
        "## Holm-Significant Refit Features",
        "",
        markdown_table(
            paper_rows(significant).sort_values(["Analysis", "Contrast", "Task", "Holm p", "Predictor"])
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--stability-repeats", type=int, default=20)
    parser.add_argument("--stability-threshold", type=float, default=0.70)
    parser.add_argument("--l1-ratios", default="0.5,0.9,1.0")
    parser.add_argument("--c-values", default="0.01,0.03,0.1,0.3,1,3")
    args = parser.parse_args()

    l1_ratios = [float(value) for value in args.l1_ratios.split(",") if value.strip()]
    c_values = [float(value) for value in args.c_values.split(",") if value.strip()]
    feature_names = text_feature_names()

    log("Preparing model inputs")
    inputs = panel_inputs(feature_names) + condition_inputs(feature_names)
    log(f"Prepared {len(inputs)} model slices")

    performance_rows: list[dict[str, Any]] = []
    grid_frames: list[pd.DataFrame] = []
    stability_frames: list[pd.DataFrame] = []
    inference_frames: list[pd.DataFrame] = []

    for index, model_input in enumerate(inputs, start=1):
        log(
            f"[{index}/{len(inputs)}] {model_input.analysis} / {model_input.contrast} / "
            f"{model_input.task_scope}: {model_input.n_pairs} pairs, {model_input.x.shape[1]} candidates"
        )
        perf, grid, stability, inference = run_stability_model(
            model_input=model_input,
            l1_ratios=l1_ratios,
            c_values=c_values,
            n_splits=args.n_splits,
            stability_repeats=args.stability_repeats,
            stability_threshold=args.stability_threshold,
        )
        performance_rows.append(perf)
        grid_frames.append(grid)
        stability_frames.append(stability)
        if not inference.empty:
            inference_frames.append(inference)

    performance = pd.DataFrame(performance_rows)
    grid = pd.concat(grid_frames, ignore_index=True, sort=False)
    stability = pd.concat(stability_frames, ignore_index=True, sort=False)
    selected = stability[stability["selection_frequency"].ge(args.stability_threshold)].copy()
    inference = pd.concat(inference_frames, ignore_index=True, sort=False) if inference_frames else pd.DataFrame()
    significant = significant_with_none(inference, performance)
    detailed = paper_rows(significant).sort_values(["Analysis", "Contrast", "Task", "Holm p", "Predictor"])
    main_sig = significant[significant["task_scope"].eq("all_tasks")].copy()
    task_sig = significant[significant["task_scope"].ne("all_tasks")].copy()
    wide_main = wide_table(main_sig, task_scope="all_tasks")
    wide_task = wide_by_task_table(task_sig)

    paths = {
        "performance": ANALYSIS / f"{OUT_PREFIX}_performance.csv",
        "grid": ANALYSIS / f"{OUT_PREFIX}_cv_grid.csv",
        "stability": ANALYSIS / f"{OUT_PREFIX}_stability.csv",
        "selected": ANALYSIS / f"{OUT_PREFIX}_selected_features.csv",
        "inference": ANALYSIS / f"{OUT_PREFIX}_inference_coefficients.csv",
        "significant": ANALYSIS / f"{OUT_PREFIX}_significant_features.csv",
        "detailed_csv": ANALYSIS / f"{OUT_PREFIX}_paper_detailed.csv",
        "detailed_tex": ANALYSIS / f"{OUT_PREFIX}_paper_detailed.tex",
        "wide_main_csv": ANALYSIS / f"{OUT_PREFIX}_paper_wide_main.csv",
        "wide_main_tex": ANALYSIS / f"{OUT_PREFIX}_paper_wide_main.tex",
        "wide_main_md": ANALYSIS / f"{OUT_PREFIX}_paper_wide_main.md",
        "wide_task_csv": ANALYSIS / f"{OUT_PREFIX}_paper_wide_by_task.csv",
        "wide_task_tex": ANALYSIS / f"{OUT_PREFIX}_paper_wide_by_task.tex",
        "wide_task_md": ANALYSIS / f"{OUT_PREFIX}_paper_wide_by_task.md",
        "summary": ANALYSIS / f"{OUT_PREFIX}_summary.md",
    }

    performance.to_csv(paths["performance"], index=False)
    grid.to_csv(paths["grid"], index=False)
    stability.to_csv(paths["stability"], index=False)
    selected.to_csv(paths["selected"], index=False)
    inference.to_csv(paths["inference"], index=False)
    significant.to_csv(paths["significant"], index=False)
    detailed.to_csv(paths["detailed_csv"], index=False)
    detailed.to_latex(paths["detailed_tex"], index=False, escape=True)
    wide_main.to_csv(paths["wide_main_csv"], index=False)
    wide_main.to_latex(paths["wide_main_tex"], index=False, escape=True)
    wide_task.to_csv(paths["wide_task_csv"], index=False)
    wide_task.to_latex(paths["wide_task_tex"], index=False, escape=True)

    wide_main_lines = [
        "# Stability-Selected Main Predictor Table",
        "",
        "Cells show odds ratio [95% CI] for predictors significant after Holm correction in the stability-selected refit.",
        "Empty cells indicate that the predictor was not significant in that column.",
        "",
        markdown_table(wide_main),
        "",
    ]
    paths["wide_main_md"].write_text("\n".join(wide_main_lines), encoding="utf-8")
    write_wide_task_markdown(paths["wide_task_md"], wide_task)
    write_summary(path=paths["summary"], performance=performance, selected=selected, significant=significant, args=args)

    for label, path in paths.items():
        print(f"{label}: {path}")
    print(f"stable selected rows: {len(selected)}")
    print(f"significant-or-none rows: {len(significant)}")
    print(f"wide main rows: {len(wide_main)}")
    print(f"wide task rows: {len(wide_task)}")


if __name__ == "__main__":
    main()
