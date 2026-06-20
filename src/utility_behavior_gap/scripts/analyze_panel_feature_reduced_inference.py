#!/usr/bin/env python3
"""Inferential models for the reduced panel-feature set.

This script complements `analyze_panel_feature_lasso.py`. The lasso analysis is
used for feature screening; this script fits unpenalized logistic regressions on
the reduced feature set so beta estimates and p-values are available.

Two outcomes are fit:

* `panel_strength`: expanded judge-level outcome from decisive panels. `HHH`
  contributes three high-side successes; `HHL` contributes two high-side
  successes and one low-side success.
* `binary_panel_winner`: one row per decisive panel, high-side majority winner
  coded as 1 for `HHH` or `HHL` and 0 for `HLL` or `LLL`.

Standard errors are cluster-robust by run id.
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
from statsmodels.stats.multitest import multipletests

from utility_behavior_gap.paths import ANALYSIS


MODEL_DATA = ANALYSIS / "panel_feature_lasso_model_data.csv"
OUT_PREFIX = "panel_feature_reduced_inference"

REDUCED_FEATURES = [
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

MODEL_SPECS = [
    ("feature_only", []),
    ("task_adjusted", ["task"]),
    ("task_condition_adjusted", ["task", "contrast"]),
]


def log(message: str) -> None:
    print(message, flush=True)


def read_model_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    allowed = {"HHH", "HHL", "HLL", "LLL"}
    bad = sorted(set(df["panel_signature"]) - allowed)
    if bad:
        raise ValueError(f"model data contains non-decisive signatures: {bad}")
    for feature in REDUCED_FEATURES:
        col = f"delta_{feature}"
        if col not in df.columns:
            raise ValueError(f"missing reduced feature column: {col}")
    if "run_id" not in df.columns:
        df["run_id"] = df["run_dir"].astype(str)
    return df


def design_matrix(df: pd.DataFrame, control_cols: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    feature_cols = [f"delta_{feature}" for feature in REDUCED_FEATURES]
    x_features = df[feature_cols].copy()
    x_features.columns = REDUCED_FEATURES

    term_types = {feature: "feature" for feature in REDUCED_FEATURES}
    pieces = [x_features.reset_index(drop=True)]
    if control_cols:
        controls = pd.get_dummies(df[control_cols].astype(str), columns=control_cols, drop_first=True, dtype=float)
        pieces.append(controls.reset_index(drop=True))
        term_types.update({col: "control" for col in controls.columns})
    x = pd.concat(pieces, axis=1)
    x = sm.add_constant(x, has_constant="add")
    term_types["const"] = "intercept"
    return x.astype(float), term_types


def expand_panel_strength(df: pd.DataFrame, x: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[pd.Series] = []
    y: list[int] = []
    groups: list[str] = []
    for idx, row in df.iterrows():
        high = int(row["n_high"])
        low = int(row["n_low"])
        for _ in range(high):
            rows.append(x.loc[idx])
            y.append(1)
            groups.append(str(row["run_id"]))
        for _ in range(low):
            rows.append(x.loc[idx])
            y.append(0)
            groups.append(str(row["run_id"]))
    return pd.DataFrame(rows).reset_index(drop=True), np.asarray(y), np.asarray(groups)


def binary_panel_winner(df: pd.DataFrame, x: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    y = (df["n_high"].astype(int) >= 2).astype(int).to_numpy()
    groups = df["run_id"].astype(str).to_numpy()
    return x.reset_index(drop=True), y, groups


def fit_glm_binomial(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> Any:
    model = sm.GLM(y, x, family=sm.families.Binomial())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


def finite_exp(value: float) -> float:
    if value > 700:
        return math.inf
    if value < -700:
        return 0.0
    return float(math.exp(value))


def coefficient_rows(
    *,
    result: Any,
    outcome: str,
    model_name: str,
    term_types: dict[str, str],
    n_rows: int,
    n_pairs: int,
    n_clusters: int,
) -> list[dict[str, Any]]:
    ci = result.conf_int()
    rows = []
    for term in result.params.index:
        beta = float(result.params[term])
        se = float(result.bse[term])
        ci_low = float(ci.loc[term, 0])
        ci_high = float(ci.loc[term, 1])
        rows.append(
            {
                "outcome": outcome,
                "model": model_name,
                "term": term,
                "term_type": term_types.get(term, "control"),
                "beta_log_odds": beta,
                "std_error_cluster_run": se,
                "z": float(result.tvalues[term]),
                "p_value": float(result.pvalues[term]),
                "odds_ratio": finite_exp(beta),
                "odds_ratio_ci_low": finite_exp(ci_low),
                "odds_ratio_ci_high": finite_exp(ci_high),
                "n_rows": n_rows,
                "n_pairs": n_pairs,
                "n_clusters": n_clusters,
            }
        )
    return rows


def add_multiple_testing_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p_holm_features_within_outcome_model"] = np.nan
    out["p_bh_features_within_outcome_model"] = np.nan
    for (_outcome, _model), idx in out[out["term_type"].eq("feature")].groupby(["outcome", "model"]).groups.items():
        pvals = out.loc[idx, "p_value"].to_numpy()
        out.loc[idx, "p_holm_features_within_outcome_model"] = multipletests(pvals, method="holm")[1]
        out.loc[idx, "p_bh_features_within_outcome_model"] = multipletests(pvals, method="fdr_bh")[1]
    return out


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
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


def write_summary(path: Path, coefficients: pd.DataFrame) -> None:
    feature_rows = coefficients[coefficients["term_type"].eq("feature")].copy()
    top = feature_rows.sort_values(["outcome", "model", "p_value"]).copy()
    lines = [
        "# Reduced Panel-Feature Inference",
        "",
        "This analysis fits unpenalized logistic regressions on the reduced feature set selected from the panel-feature lasso screen.",
        "All beta values are log-odds coefficients for one within-task-standard-deviation increase in the high-minus-low feature delta.",
        "Standard errors and p-values are cluster-robust by run id.",
        "",
        "Outcomes:",
        "",
        "- `panel_strength`: expanded judge-level outcome from decisive panels, so `HHL` contributes two high-side successes and one low-side success.",
        "- `binary_panel_winner`: one row per decisive panel, with `HHH`/`HHL` coded as high-side panel win and `HLL`/`LLL` coded as low-side panel win.",
        "",
        "Models:",
        "",
        "- `feature_only`: reduced text features only.",
        "- `task_adjusted`: reduced text features plus task controls.",
        "- `task_condition_adjusted`: reduced text features plus task and contrast controls.",
        "",
        "## Reduced Feature Set",
        "",
    ]
    lines.extend(f"- `{feature}`" for feature in REDUCED_FEATURES)
    lines.extend(
        [
            "",
            "## Feature Coefficients Ranked By P-Value",
            "",
            markdown_table(
                top,
                [
                    "outcome",
                    "model",
                    "term",
                    "beta_log_odds",
                    "std_error_cluster_run",
                    "p_value",
                    "p_holm_features_within_outcome_model",
                    "odds_ratio",
                ],
                max_rows=80,
            ),
            "",
            "Interpretation caution: these are exploratory predictive models over correlated text features. The p-values say whether a feature has predictive value conditional on the other included features and controls, not whether that feature is a causal mechanism.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data", type=Path, default=MODEL_DATA)
    args = parser.parse_args()

    log(f"Loading decisive model data: {args.model_data}")
    df = read_model_data(args.model_data)
    rows: list[dict[str, Any]] = []

    for model_name, control_cols in MODEL_SPECS:
        log(f"Fitting {model_name}")
        x, term_types = design_matrix(df, control_cols)

        panel_x, panel_y, panel_groups = expand_panel_strength(df, x)
        panel_result = fit_glm_binomial(panel_x, panel_y, panel_groups)
        rows.extend(
            coefficient_rows(
                result=panel_result,
                outcome="panel_strength",
                model_name=model_name,
                term_types=term_types,
                n_rows=len(panel_y),
                n_pairs=len(df),
                n_clusters=len(set(panel_groups)),
            )
        )

        binary_x, binary_y, binary_groups = binary_panel_winner(df, x)
        binary_result = fit_glm_binomial(binary_x, binary_y, binary_groups)
        rows.extend(
            coefficient_rows(
                result=binary_result,
                outcome="binary_panel_winner",
                model_name=model_name,
                term_types=term_types,
                n_rows=len(binary_y),
                n_pairs=len(df),
                n_clusters=len(set(binary_groups)),
            )
        )

    coefficients = add_multiple_testing_columns(pd.DataFrame(rows))
    coefficient_path = ANALYSIS / f"{OUT_PREFIX}_coefficients.csv"
    feature_path = ANALYSIS / f"{OUT_PREFIX}_feature_coefficients.csv"
    summary_path = ANALYSIS / f"{OUT_PREFIX}_summary.md"

    coefficients.to_csv(coefficient_path, index=False)
    coefficients[coefficients["term_type"].eq("feature")].to_csv(feature_path, index=False)
    write_summary(summary_path, coefficients)

    print(f"summary: {summary_path}")
    print(f"coefficients: {coefficient_path}")
    print(f"feature coefficients: {feature_path}")
    print(
        coefficients[coefficients["term_type"].eq("feature")]
        .sort_values(["outcome", "model", "p_value"])
        .groupby(["outcome", "model"])
        .head(5)[
            [
                "outcome",
                "model",
                "term",
                "beta_log_odds",
                "std_error_cluster_run",
                "p_value",
                "p_holm_features_within_outcome_model",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

