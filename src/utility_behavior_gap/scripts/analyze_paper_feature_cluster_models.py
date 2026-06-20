#!/usr/bin/env python3
"""Paper-facing text-feature models based on correlation clusters.

This script handles redundant text features statistically rather than by
manually choosing which variables to keep.

For each scope (`all_tasks` and each individual task), it:

1. Builds feature clusters from absolute Spearman correlations among generated
   response features.
2. Replaces every cluster by its first principal-component score.
3. Uses those same scope-specific cluster scores in all five paper columns:
   direct instruction, amount, moral cue, utility high-low, and judging panel.
4. Fits clustered logistic regressions on all cluster scores and reports
   Holm-significant clusters.

The task-specific fits use task-specific clusters. The aggregate fits use
clusters learned on all tasks after within-task feature standardization.
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
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests

from utility_behavior_gap.paths import ANALYSIS


FEATURE_DEFINITIONS = ANALYSIS / "final_text_analysis_feature_definitions.csv"
BY_OUTPUT = ANALYSIS / "final_text_analysis_by_output.csv"
PANEL_MODEL_DATA = ANALYSIS / "panel_feature_lasso_model_data.csv"

OUT_PREFIX = "paper_feature_clusters"
P_THRESHOLD = 0.05
DEFAULT_CORR_THRESHOLD = 0.70

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


@dataclass(frozen=True)
class ClusterDefinition:
    scope: str
    cluster_id: str
    features: tuple[str, ...]
    loadings: tuple[float, ...]
    score_mean: float
    score_sd: float
    label: str
    top_loading_features: tuple[str, ...]


@dataclass(frozen=True)
class ClusterModel:
    scope: str
    corr_threshold: float
    feature_stats: dict[tuple[str, str], tuple[float, float]]
    clusters: tuple[ClusterDefinition, ...]


@dataclass(frozen=True)
class ModelInput:
    analysis: str
    contrast: str
    task_scope: str
    x: pd.DataFrame
    y: np.ndarray
    weights: np.ndarray
    groups: np.ndarray
    group_label: str
    n_pairs: int
    n_rows: int
    fallback_groups: np.ndarray | None = None
    fallback_group_label: str = ""


def log(message: str) -> None:
    print(message, flush=True)


def text_feature_names() -> list[str]:
    defs = pd.read_csv(FEATURE_DEFINITIONS)
    return defs.loc[defs["type"].eq("feature"), "name"].tolist()


def available_feature_names(df: pd.DataFrame, feature_names: list[str]) -> list[str]:
    available: list[str] = []
    unavailable: list[str] = []
    for feature in feature_names:
        if feature not in df.columns:
            unavailable.append(feature)
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.notna().any():
            available.append(feature)
        else:
            unavailable.append(feature)
    if unavailable:
        log("Skipping unavailable/all-NaN features: " + ", ".join(unavailable))
    if not available:
        raise SystemExit("No usable text features found.")
    return available


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def finite_exp(value: float) -> float:
    if value > 700:
        return math.inf
    if value < -700:
        return 0.0
    return float(math.exp(value))


def format_p(value: float | str | None) -> str:
    if value is None or value == "" or pd.isna(value):
        return ""
    value = float(value)
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def format_float(value: float | str | None) -> str:
    if value is None or value == "" or pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def format_or_ci(or_value: float | str | None, low: float | str | None, high: float | str | None) -> str:
    if or_value is None or low is None or high is None or pd.isna(or_value) or pd.isna(low) or pd.isna(high):
        return ""
    return f"{float(or_value):.3f} [{float(low):.3f}, {float(high):.3f}]"


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


def clean_response_rows(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
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


def scope_response_rows(clean_rows: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all_tasks":
        return clean_rows.copy()
    return clean_rows[clean_rows["task"].eq(scope)].copy()


def feature_stats_for_scope(rows: pd.DataFrame, feature_names: list[str], scope: str) -> dict[tuple[str, str], tuple[float, float]]:
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    if scope == "all_tasks":
        grouped = rows.groupby("task", dropna=False)
    else:
        grouped = [(scope, rows)]
    for task, task_rows in grouped:
        for feature in feature_names:
            values = pd.to_numeric(task_rows[feature], errors="coerce")
            mean = float(values.mean())
            sd = float(values.std(ddof=0))
            if not np.isfinite(sd) or np.isclose(sd, 0.0):
                sd = 1.0
            stats[(str(task), feature)] = (mean, sd)
    return stats


def standardize_response_features(
    rows: pd.DataFrame,
    feature_names: list[str],
    *,
    scope: str,
    stats: dict[tuple[str, str], tuple[float, float]],
) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    for feature in feature_names:
        values = pd.Series(index=rows.index, dtype=float)
        for idx, row in rows.iterrows():
            task = str(row["task"]) if scope == "all_tasks" else scope
            mean, sd = stats[(task, feature)]
            values.loc[idx] = (float(row[feature]) - mean) / sd
        out[feature] = values
    return out


def standardized_feature_row(
    *,
    row: pd.Series,
    feature: str,
    side: str,
    scope: str,
    stats: dict[tuple[str, str], tuple[float, float]],
) -> float:
    task = str(row["task"]) if scope == "all_tasks" else scope
    mean, sd = stats[(task, feature)]
    value = float(row[f"{side}_{feature}"])
    return (value - mean) / sd


def cluster_feature_names(x: pd.DataFrame, corr_threshold: float) -> list[list[str]]:
    retained = [col for col in x.columns if not np.isclose(float(x[col].std(ddof=0)), 0.0)]
    if not retained:
        return []
    if len(retained) == 1:
        return [retained]
    corr = x[retained].corr(method="spearman").abs().fillna(0.0)
    corr_values = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_values, 1.0)
    distance = pd.DataFrame(1.0 - corr_values, index=retained, columns=retained)
    condensed = squareform(distance.to_numpy(), checks=False)
    tree = linkage(condensed, method="average")
    labels = fcluster(tree, t=1.0 - corr_threshold, criterion="distance")
    clusters: dict[int, list[str]] = {}
    for feature, label in zip(retained, labels, strict=False):
        clusters.setdefault(int(label), []).append(feature)
    ordered = sorted(clusters.values(), key=lambda features: min(retained.index(feature) for feature in features))
    return ordered


def first_pc(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[1] == 1:
        return x.iloc[:, 0].to_numpy(dtype=float), np.asarray([1.0])
    matrix = x.to_numpy(dtype=float)
    _u, _s, vt = np.linalg.svd(matrix, full_matrices=False)
    loadings = vt[0].copy()
    max_idx = int(np.argmax(np.abs(loadings)))
    if loadings[max_idx] < 0:
        loadings *= -1
    scores = matrix @ loadings
    return scores, loadings


def cluster_label(features: list[str], loadings: np.ndarray) -> tuple[str, tuple[str, ...]]:
    order = np.argsort(-np.abs(loadings))
    top = [features[int(idx)] for idx in order[: min(4, len(features))]]
    if len(features) == 1:
        return feature_label(features[0]), tuple(top)
    label = "Cluster: " + ", ".join(feature_label(feature) for feature in top)
    if len(features) > len(top):
        label += f" (+{len(features) - len(top)} more)"
    return label, tuple(top)


def build_cluster_model(
    *,
    clean_rows: pd.DataFrame,
    feature_names: list[str],
    scope: str,
    corr_threshold: float,
) -> ClusterModel:
    rows = scope_response_rows(clean_rows, scope)
    stats = feature_stats_for_scope(rows, feature_names, scope)
    standardized = standardize_response_features(rows, feature_names, scope=scope, stats=stats)
    standardized = standardized.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    clusters = cluster_feature_names(standardized, corr_threshold)
    definitions: list[ClusterDefinition] = []
    for cluster_index, features in enumerate(clusters, start=1):
        cluster_x = standardized[features]
        scores, loadings = first_pc(cluster_x)
        score_mean = float(np.mean(scores))
        score_sd = float(np.std(scores, ddof=0))
        if not np.isfinite(score_sd) or np.isclose(score_sd, 0.0):
            score_sd = 1.0
        label, top = cluster_label(features, loadings)
        definitions.append(
            ClusterDefinition(
                scope=scope,
                cluster_id=f"cluster_{cluster_index:02d}",
                features=tuple(features),
                loadings=tuple(float(value) for value in loadings),
                score_mean=score_mean,
                score_sd=score_sd,
                label=label,
                top_loading_features=top,
            )
        )
    return ClusterModel(scope=scope, corr_threshold=corr_threshold, feature_stats=stats, clusters=tuple(definitions))


def transform_response_rows(rows: pd.DataFrame, model: ClusterModel) -> pd.DataFrame:
    feature_names = sorted({feature for cluster in model.clusters for feature in cluster.features})
    standardized = standardize_response_features(rows, feature_names, scope=model.scope, stats=model.feature_stats)
    out = pd.DataFrame(index=rows.index)
    for cluster in model.clusters:
        raw = standardized[list(cluster.features)].to_numpy(dtype=float) @ np.asarray(cluster.loadings, dtype=float)
        out[cluster.cluster_id] = (raw - cluster.score_mean) / cluster.score_sd
    return out.reset_index(drop=True)


def transform_panel_rows(rows: pd.DataFrame, model: ClusterModel) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    for cluster in model.clusters:
        high_scores = []
        low_scores = []
        for _idx, row in rows.iterrows():
            high_values = [
                standardized_feature_row(row=row, feature=feature, side="high", scope=model.scope, stats=model.feature_stats)
                for feature in cluster.features
            ]
            low_values = [
                standardized_feature_row(row=row, feature=feature, side="low", scope=model.scope, stats=model.feature_stats)
                for feature in cluster.features
            ]
            loadings = np.asarray(cluster.loadings, dtype=float)
            high_scores.append((float(np.dot(high_values, loadings)) - cluster.score_mean) / cluster.score_sd)
            low_scores.append((float(np.dot(low_values, loadings)) - cluster.score_mean) / cluster.score_sd)
        out[cluster.cluster_id] = np.asarray(high_scores) - np.asarray(low_scores)
    return out.reset_index(drop=True)


def standardize_panel_scores(x: pd.DataFrame, rows: pd.DataFrame, scope: str) -> pd.DataFrame:
    out = x.copy()
    if scope == "all_tasks":
        groups = rows.groupby("task", dropna=False).groups
    else:
        groups = {scope: rows.index}
    for col in out.columns:
        standardized = pd.Series(index=out.index, dtype=float)
        for _task, idx in groups.items():
            values = pd.to_numeric(out.loc[idx, col], errors="coerce")
            mean = values.mean()
            sd = values.std(ddof=0)
            if not np.isfinite(sd) or np.isclose(float(sd), 0.0):
                standardized.loc[idx] = 0.0
            else:
                standardized.loc[idx] = (values - mean) / sd
        out[col] = standardized
    return out


def condition_inputs(clean_rows: pd.DataFrame, cluster_models: dict[str, ClusterModel]) -> list[ModelInput]:
    inputs: list[ModelInput] = []
    for contrast in CONTRASTS:
        contrast_rows = clean_rows[clean_rows["contrast"].eq(contrast)].copy()
        slices = [("all_tasks", contrast_rows)]
        slices.extend((str(task), task_rows.copy()) for task, task_rows in sorted(contrast_rows.groupby("task")))
        for scope, rows in slices:
            cluster_model = cluster_models[scope]
            x = transform_response_rows(rows, cluster_model)
            x = drop_zero_variance_columns(x)
            inputs.append(
                ModelInput(
                    analysis="Condition origin",
                    contrast=contrast,
                    task_scope=scope,
                    x=x,
                    y=rows["condition_high"].astype(int).to_numpy(),
                    weights=np.ones(len(rows), dtype=int),
                    groups=rows["pair_uid"].astype(str).to_numpy(),
                    group_label="pair_uid",
                    n_pairs=int(rows["pair_uid"].nunique()),
                    n_rows=int(len(rows)),
                )
            )
    return inputs


def panel_inputs(cluster_models: dict[str, ClusterModel]) -> list[ModelInput]:
    df = pd.read_csv(PANEL_MODEL_DATA, low_memory=False)
    allowed = {"HHH", "HHL", "HLL", "LLL"}
    bad = sorted(set(df["panel_signature"]) - allowed)
    if bad:
        raise ValueError(f"panel model data contains non-decisive signatures: {bad}")
    if "run_id" not in df.columns:
        df["run_id"] = df["run_dir"].astype(str)
    inputs: list[ModelInput] = []
    slices = [("all_tasks", df)]
    slices.extend((str(task), task_rows.copy()) for task, task_rows in sorted(df.groupby("task")))
    for scope, rows in slices:
        cluster_model = cluster_models[scope]
        required = [f"{side}_{feature}" for cluster in cluster_model.clusters for feature in cluster.features for side in ["high", "low"]]
        rows = rows.dropna(subset=required + ["n_high", "n_low", "run_id"]).copy().reset_index(drop=True)
        x_pair = transform_panel_rows(rows, cluster_model)
        x_pair = standardize_panel_scores(x_pair, rows, scope)
        x_pair = drop_zero_variance_columns(x_pair)
        x_rows: list[pd.Series] = []
        y_rows: list[int] = []
        weights: list[int] = []
        groups: list[str] = []
        fallback_groups: list[str] = []
        for idx, row in rows.iterrows():
            high = int(row["n_high"])
            low = int(row["n_low"])
            if high:
                x_rows.append(x_pair.loc[idx])
                y_rows.append(1)
                weights.append(high)
                groups.append(str(row["run_id"]))
                fallback_groups.append(str(row["pair_uid"]))
            if low:
                x_rows.append(x_pair.loc[idx])
                y_rows.append(0)
                weights.append(low)
                groups.append(str(row["run_id"]))
                fallback_groups.append(str(row["pair_uid"]))
        inputs.append(
            ModelInput(
                analysis="Judging panel",
                contrast="judge_panel",
                task_scope=scope,
                x=pd.DataFrame(x_rows).reset_index(drop=True),
                y=np.asarray(y_rows, dtype=int),
                weights=np.asarray(weights, dtype=int),
                groups=np.asarray(groups),
                group_label="run_id",
                n_pairs=int(len(rows)),
                n_rows=int(sum(rows["n_high"].astype(int) + rows["n_low"].astype(int))),
                fallback_groups=np.asarray(fallback_groups),
                fallback_group_label="pair_uid",
            )
        )
    return inputs


def drop_zero_variance_columns(x: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for col in x.columns:
        values = pd.to_numeric(x[col], errors="coerce")
        if np.isfinite(values.std(ddof=0)) and not np.isclose(float(values.std(ddof=0)), 0.0):
            keep.append(col)
    return x[keep].reset_index(drop=True)


def fit_clustered_glm(model_input: ModelInput) -> pd.DataFrame:
    if model_input.x.empty:
        return pd.DataFrame()
    repeat_counts = model_input.weights.astype(int)
    if not np.allclose(model_input.weights, repeat_counts):
        raise ValueError("weights must be integer counts for expansion")
    x_base = sm.add_constant(model_input.x.astype(float), has_constant="add")
    x = x_base.loc[x_base.index.repeat(repeat_counts)].reset_index(drop=True)
    y = np.repeat(model_input.y, repeat_counts)
    model = sm.GLM(y, x, family=sm.families.Binomial())

    def fit_with_groups(group_values: np.ndarray, group_label: str, status: str) -> pd.DataFrame:
        groups = np.repeat(group_values, repeat_counts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
        ci = result.conf_int()
        rows = []
        for term in result.params.index:
            beta = float(result.params[term])
            lo = float(ci.loc[term, 0])
            hi = float(ci.loc[term, 1])
            rows.append(
                {
                    "analysis": model_input.analysis,
                    "contrast": model_input.contrast,
                    "task_scope": model_input.task_scope,
                    "term": term,
                    "term_type": "intercept" if term == "const" else "cluster",
                    "beta_log_odds": beta,
                    "std_error_cluster": float(result.bse[term]),
                    "z": float(result.tvalues[term]),
                    "p_value": float(result.pvalues[term]),
                    "odds_ratio": finite_exp(beta),
                    "odds_ratio_ci_low": finite_exp(lo),
                    "odds_ratio_ci_high": finite_exp(hi),
                    "n_pairs": model_input.n_pairs,
                    "n_rows": model_input.n_rows,
                    "n_clusters": int(len(pd.unique(groups))),
                    "cluster_unit": group_label,
                    "inference_status": status,
                }
            )
        return pd.DataFrame(rows)

    def valid_inference(out: pd.DataFrame) -> bool:
        cluster_rows = out[out["term_type"].eq("cluster")]
        if cluster_rows.empty:
            return True
        required = ["std_error_cluster", "z", "p_value", "odds_ratio_ci_low", "odds_ratio_ci_high"]
        return bool(np.isfinite(cluster_rows[required].to_numpy(dtype=float)).all())

    out = fit_with_groups(model_input.groups, model_input.group_label, "ok")
    if not valid_inference(out) and model_input.fallback_groups is not None:
        out = fit_with_groups(
            model_input.fallback_groups,
            model_input.fallback_group_label,
            f"fallback_{model_input.fallback_group_label}_after_invalid_{model_input.group_label}",
        )
    elif not valid_inference(out):
        out["inference_status"] = f"invalid_{model_input.group_label}"

    cluster_mask = out["term_type"].eq("cluster")
    out["p_holm_clusters_within_model"] = np.nan
    out["p_bh_clusters_within_model"] = np.nan
    if cluster_mask.any():
        pvals = out.loc[cluster_mask, "p_value"]
        finite = pvals.notna() & np.isfinite(pvals)
        if finite.all():
            out.loc[cluster_mask, "p_holm_clusters_within_model"] = multipletests(pvals.to_numpy(), method="holm")[1]
            out.loc[cluster_mask, "p_bh_clusters_within_model"] = multipletests(pvals.to_numpy(), method="fdr_bh")[1]
    return out


def cluster_definition_rows(cluster_models: dict[str, ClusterModel]) -> pd.DataFrame:
    rows = []
    for scope, model in cluster_models.items():
        for cluster in model.clusters:
            rows.append(
                {
                    "scope": scope,
                    "cluster_id": cluster.cluster_id,
                    "label": cluster.label,
                    "features": ";".join(cluster.features),
                    "top_loading_features": ";".join(cluster.top_loading_features),
                    "loadings": ";".join(f"{value:.6g}" for value in cluster.loadings),
                    "score_mean": cluster.score_mean,
                    "score_sd": cluster.score_sd,
                    "cluster_size": len(cluster.features),
                    "corr_threshold": model.corr_threshold,
                }
            )
    return pd.DataFrame(rows)


def significant_with_none(inference: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    sig = inference[
        inference["term_type"].eq("cluster")
        & inference["p_holm_clusters_within_model"].lt(P_THRESHOLD)
    ].copy()
    sig_keys = set(zip(sig["analysis"], sig["contrast"], sig["task_scope"], strict=False))
    rows = []
    for _, row in performance.iterrows():
        key = (row["analysis"], row["contrast"], row["task_scope"])
        if key not in sig_keys:
            if "finite_inference" in row and not bool(row["finite_inference"]):
                term = "inference_failed"
            else:
                term = "none"
            rows.append(
                {
                    "analysis": row["analysis"],
                    "contrast": row["contrast"],
                    "task_scope": row["task_scope"],
                    "term": term,
                    "term_type": "cluster",
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
                    "p_holm_clusters_within_model": np.nan,
                    "p_bh_clusters_within_model": np.nan,
                }
            )
    return pd.concat([sig, pd.DataFrame(rows)], ignore_index=True, sort=False)


def add_cluster_labels(df: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    labels = {
        (row["scope"], row["cluster_id"]): row["label"]
        for _, row in definitions.iterrows()
    }
    features = {
        (row["scope"], row["cluster_id"]): row["features"]
        for _, row in definitions.iterrows()
    }
    out["cluster_label"] = [
        labels.get(
            (row["task_scope"], row["term"]),
            "None significant"
            if row["term"] == "none"
            else "Inference failed"
            if row["term"] == "inference_failed"
            else row["term"],
        )
        for _, row in out.iterrows()
    ]
    out["cluster_features"] = [
        features.get((row["task_scope"], row["term"]), "")
        for _, row in out.iterrows()
    ]
    return out


def paper_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Analysis"] = out["analysis"]
    out["Contrast"] = out["contrast"].map(CONTRAST_LABELS).fillna(out["contrast"])
    out["Task"] = out["task_scope"].map(TASK_LABELS).fillna(out["task_scope"])
    out["Predictor cluster"] = out["cluster_label"]
    out["Features"] = out["cluster_features"]
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
    out["Holm p"] = out["p_holm_clusters_within_model"].map(format_p)
    out["n pairs"] = out["n_pairs"].astype("Int64")
    return out[
        [
            "Analysis",
            "Contrast",
            "Task",
            "Predictor cluster",
            "Features",
            "Direction",
            "Beta",
            "OR",
            "OR 95% CI",
            "Holm p",
            "n pairs",
        ]
    ]


def wide_table(significant: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    sig = significant[(significant["task_scope"].eq(scope)) & significant["term"].ne("none")].copy()
    clusters = []
    for term in sig["term"]:
        if term not in clusters:
            clusters.append(term)
    rows = []
    for cluster_id in clusters:
        cluster_rows = sig[sig["term"].eq(cluster_id)]
        label = str(cluster_rows["cluster_label"].iloc[0])
        features = str(cluster_rows["cluster_features"].iloc[0])
        row = {"Predictor cluster": label, "Features": features}
        for contrast, label_col in WIDE_COLUMNS:
            match = cluster_rows[cluster_rows["contrast"].eq(contrast)]
            if match.empty:
                row[label_col] = ""
            else:
                m = match.iloc[0]
                if m["term"] == "inference_failed":
                    row[label_col] = "inference failed"
                else:
                    row[label_col] = format_or_ci(m["odds_ratio"], m["odds_ratio_ci_low"], m["odds_ratio_ci_high"])
        rows.append(row)
    if not rows:
        rows = [{"Predictor cluster": "None significant", "Features": "", **{label: "" for _c, label in WIDE_COLUMNS}}]
    return pd.DataFrame(rows, columns=["Predictor cluster", "Features"] + [label for _c, label in WIDE_COLUMNS])


def wide_by_task(significant: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for scope in ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]:
        table = wide_table(significant, scope=scope)
        table.insert(0, "Task", TASK_LABELS.get(scope, scope))
        frames.append(table)
    return pd.concat(frames, ignore_index=True, sort=False)


def write_wide_task_markdown(path: Path, table: pd.DataFrame) -> None:
    lines = [
        "# Cluster-Score Task-Specific Predictor Table",
        "",
        "Cells show odds ratio [95% CI] for predictor clusters significant after Holm correction.",
        "Task-specific rows use task-specific feature clusters.",
        "",
    ]
    for task, sub in table.groupby("Task", sort=False):
        lines.extend([f"## {task}", "", markdown_table(sub.drop(columns=["Task"]).reset_index(drop=True)), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    *,
    path: Path,
    performance: pd.DataFrame,
    definitions: pd.DataFrame,
    detailed: pd.DataFrame,
    corr_threshold: float,
) -> None:
    lines = [
        "# Paper Feature Cluster Models",
        "",
        "This analysis replaces correlated text features with data-driven cluster scores.",
        f"Features are clustered within each scope using absolute Spearman correlation >= {corr_threshold:.2f}.",
        "Each cluster is represented by its first principal-component score, oriented so the top-loading feature has positive loading.",
        "Aggregate models use aggregate clusters learned after within-task standardization; task-specific models use task-specific clusters.",
        "",
        "## Model Slices",
        "",
        markdown_table(performance),
        "",
        "## Cluster Definitions",
        "",
        markdown_table(definitions[["scope", "cluster_id", "label", "features", "cluster_size"]]),
        "",
        "## Significant Clusters",
        "",
        markdown_table(detailed),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corr-threshold", type=float, default=DEFAULT_CORR_THRESHOLD)
    args = parser.parse_args()

    raw_rows = pd.read_csv(BY_OUTPUT, low_memory=False)
    feature_names = available_feature_names(raw_rows, text_feature_names())
    clean_rows = clean_response_rows(raw_rows, feature_names)
    if clean_rows.empty:
        raise SystemExit("No complete high/low response pairs remain after feature filtering.")
    scopes = ["all_tasks"] + sorted(clean_rows["task"].dropna().astype(str).unique().tolist())

    log(f"Building feature clusters for {len(scopes)} scopes")
    cluster_models = {
        scope: build_cluster_model(
            clean_rows=clean_rows,
            feature_names=feature_names,
            scope=scope,
            corr_threshold=args.corr_threshold,
        )
        for scope in scopes
    }
    definitions = cluster_definition_rows(cluster_models)

    log("Preparing model inputs")
    inputs = condition_inputs(clean_rows, cluster_models) + panel_inputs(cluster_models)
    log(f"Fitting {len(inputs)} cluster-score models")
    inference_frames = []
    performance_rows = []
    for index, model_input in enumerate(inputs, start=1):
        log(
            f"[{index}/{len(inputs)}] {model_input.analysis} / {model_input.contrast} / "
            f"{model_input.task_scope}: {model_input.n_pairs} pairs, {model_input.x.shape[1]} clusters"
        )
        inf = fit_clustered_glm(model_input)
        if not inf.empty:
            inference_frames.append(inf)
            cluster_rows = inf[inf["term_type"].eq("cluster")]
            finite_inference = bool(
                not cluster_rows.empty
                and np.isfinite(
                    cluster_rows[
                        ["std_error_cluster", "z", "p_value", "odds_ratio_ci_low", "odds_ratio_ci_high"]
                    ].to_numpy(dtype=float)
                ).all()
            )
            cluster_unit = str(inf["cluster_unit"].dropna().iloc[0]) if "cluster_unit" in inf.columns else ""
            inference_status = str(inf["inference_status"].dropna().iloc[0]) if "inference_status" in inf.columns else ""
        else:
            finite_inference = False
            cluster_unit = ""
            inference_status = "no_predictors"
        performance_rows.append(
            {
                "analysis": model_input.analysis,
                "contrast": model_input.contrast,
                "task_scope": model_input.task_scope,
                "n_pairs": model_input.n_pairs,
                "n_rows": model_input.n_rows,
                "n_predictor_clusters": int(model_input.x.shape[1]),
                "cluster_unit": cluster_unit,
                "inference_status": inference_status,
                "finite_inference": finite_inference,
            }
        )
    inference = pd.concat(inference_frames, ignore_index=True, sort=False) if inference_frames else pd.DataFrame()
    performance = pd.DataFrame(performance_rows)
    significant = significant_with_none(inference, performance)
    significant = add_cluster_labels(significant, definitions)
    detailed = paper_rows(significant).sort_values(["Analysis", "Contrast", "Task", "Holm p", "Predictor cluster"])
    wide_main = wide_table(significant, scope="all_tasks")
    wide_tasks = wide_by_task(significant)

    paths = {
        "definitions": ANALYSIS / f"{OUT_PREFIX}_definitions.csv",
        "performance": ANALYSIS / f"{OUT_PREFIX}_performance.csv",
        "inference": ANALYSIS / f"{OUT_PREFIX}_inference_coefficients.csv",
        "significant": ANALYSIS / f"{OUT_PREFIX}_significant_clusters.csv",
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

    definitions.to_csv(paths["definitions"], index=False)
    performance.to_csv(paths["performance"], index=False)
    inference.to_csv(paths["inference"], index=False)
    significant.to_csv(paths["significant"], index=False)
    detailed.to_csv(paths["detailed_csv"], index=False)
    detailed.to_latex(paths["detailed_tex"], index=False, escape=True)
    wide_main.to_csv(paths["wide_main_csv"], index=False)
    wide_main.to_latex(paths["wide_main_tex"], index=False, escape=True)
    wide_tasks.to_csv(paths["wide_task_csv"], index=False)
    wide_tasks.to_latex(paths["wide_task_tex"], index=False, escape=True)

    main_lines = [
        "# Cluster-Score Main Predictor Table",
        "",
        "Cells show odds ratio [95% CI] for predictor clusters significant after Holm correction.",
        "Rows use aggregate feature clusters learned after within-task standardization.",
        "",
        markdown_table(wide_main),
        "",
    ]
    paths["wide_main_md"].write_text("\n".join(main_lines), encoding="utf-8")
    write_wide_task_markdown(paths["wide_task_md"], wide_tasks)
    write_summary(
        path=paths["summary"],
        performance=performance,
        definitions=definitions,
        detailed=detailed,
        corr_threshold=args.corr_threshold,
    )

    for label, path in paths.items():
        print(f"{label}: {path}")
    print(f"cluster definitions: {len(definitions)}")
    print(f"significant-or-none rows: {len(significant)}")
    print(f"wide main rows: {len(wide_main)}")
    print(f"wide task rows: {len(wide_tasks)}")


if __name__ == "__main__":
    main()
