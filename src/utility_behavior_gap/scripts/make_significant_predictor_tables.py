#!/usr/bin/env python3
"""Create paper-facing significant-predictor tables.

Inputs are the already-run panel-preference and condition-origin feature models.
The output tables use Holm-corrected p < .05 as the inclusion rule.

Main table:
* overall panel-preference predictors;
* overall condition-origin predictors for each contrast.

Appendix table:
* task-specific panel-preference predictors;
* task-specific condition-origin predictors for each contrast.

Task-specific panel-preference rows are refit within task using the reduced
feature set. Predictors with zero variance inside a task are dropped.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from utility_behavior_gap.paths import ANALYSIS


PANEL_COEFFICIENTS = ANALYSIS / "panel_feature_reduced_inference_feature_coefficients.csv"
PANEL_MODEL_DATA = ANALYSIS / "panel_feature_lasso_model_data.csv"
CONDITION_PERFORMANCE = ANALYSIS / "condition_origin_features_cv_performance.csv"
CONDITION_SIGNIFICANT = ANALYSIS / "condition_origin_features_significant_features.csv"

OUT_MAIN_CSV = ANALYSIS / "paper_significant_predictors_main.csv"
OUT_APPENDIX_CSV = ANALYSIS / "paper_significant_predictors_appendix.csv"
OUT_MAIN_TEX = ANALYSIS / "paper_significant_predictors_main.tex"
OUT_APPENDIX_TEX = ANALYSIS / "paper_significant_predictors_appendix.tex"
OUT_MD = ANALYSIS / "paper_significant_predictor_tables.md"
OUT_WIDE_MAIN_CSV = ANALYSIS / "paper_significant_predictors_wide_main.csv"
OUT_WIDE_MAIN_TEX = ANALYSIS / "paper_significant_predictors_wide_main.tex"
OUT_WIDE_MAIN_MD = ANALYSIS / "paper_significant_predictors_wide_main.md"
OUT_WIDE_TASK_CSV = ANALYSIS / "paper_significant_predictors_wide_by_task.csv"
OUT_WIDE_TASK_TEX = ANALYSIS / "paper_significant_predictors_wide_by_task.tex"
OUT_WIDE_TASK_MD = ANALYSIS / "paper_significant_predictors_wide_by_task.md"

P_THRESHOLD = 0.05

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

CONTRAST_LABELS = {
    "all_contrasts": "All contrasts",
    "direct_instruction": "Direct instruction",
    "amount": "Amount",
    "moral": "Moral cue",
    "utility": "Utility high-low",
}

TASK_LABELS = {
    "all_tasks": "All tasks",
    "essay": "Essay",
    "grant_proposal_abstract": "Grant abstract",
    "incident_postmortem": "Incident postmortem",
    "translation": "Translation",
}

WIDE_TABLE_CONTRASTS = [
    ("direct_instruction", "Direct instruction"),
    ("amount", "Amount"),
    ("moral", "Moral cue"),
    ("utility", "Utility high-low"),
    ("judge_panel", "Judging panel"),
]

WIDE_FEATURE_ORDER = [
    "numbers",
    "unique_word_ratio",
    "words",
    "characters",
    "sentences",
    "paragraphs",
    "mean_word_length_chars",
    "textstat_flesch_reading_ease",
    "textstat_flesch_kincaid_grade",
    "semicolon_colon_per_1k",
    "dash_per_1k",
    "counterargument_markers_per_1k",
    "specificity_markers_per_1k",
    "transition_markers_per_1k",
    "currency_mentions",
    "positive_words_per_1k",
    "negative_words_per_1k",
]


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


def format_ci(low: float | str | None, high: float | str | None) -> str:
    if low is None or high is None or low == "" or high == "" or pd.isna(low) or pd.isna(high):
        return ""
    return f"[{float(low):.3f}, {float(high):.3f}]"


def format_or_ci(or_value: float | str | None, low: float | str | None, high: float | str | None) -> str:
    if or_value is None or low is None or high is None or pd.isna(or_value) or pd.isna(low) or pd.isna(high):
        return ""
    return f"{float(or_value):.3f} [{float(low):.3f}, {float(high):.3f}]"


def direction(beta: float | str | None) -> str:
    if beta is None or beta == "" or pd.isna(beta):
        return ""
    return "positive" if float(beta) > 0 else "negative"


def finite_exp(value: float) -> float:
    if value > 700:
        return math.inf
    if value < -700:
        return 0.0
    return float(math.exp(value))


def panel_strength_task_coefficients() -> pd.DataFrame:
    df = pd.read_csv(PANEL_MODEL_DATA, low_memory=False)
    rows: list[dict[str, Any]] = []
    for task, task_df in sorted(df.groupby("task", dropna=False), key=lambda item: str(item[0])):
        delta_cols = [f"delta_{feature}" for feature in REDUCED_FEATURES]
        x_features = task_df[delta_cols].astype(float).copy()
        x_features.columns = REDUCED_FEATURES
        retained_features = [
            feature
            for feature in REDUCED_FEATURES
            if not np.isclose(float(x_features[feature].std()), 0.0)
        ]
        x_features = x_features[retained_features]
        x = sm.add_constant(x_features, has_constant="add")
        y_rows: list[int] = []
        x_rows: list[pd.Series] = []
        groups: list[str] = []
        for idx, row in task_df.iterrows():
            for _ in range(int(row["n_high"])):
                x_rows.append(x.loc[idx])
                y_rows.append(1)
                groups.append(str(row["run_id"]))
            for _ in range(int(row["n_low"])):
                x_rows.append(x.loc[idx])
                y_rows.append(0)
                groups.append(str(row["run_id"]))
        x_expanded = pd.DataFrame(x_rows).reset_index(drop=True)
        y = np.asarray(y_rows)
        group_arr = np.asarray(groups)
        model = sm.GLM(y, x_expanded, family=sm.families.Binomial())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = model.fit(cov_type="cluster", cov_kwds={"groups": group_arr})
        ci = result.conf_int()
        feature_pvals = []
        feature_terms = []
        for term in retained_features:
            feature_terms.append(term)
            feature_pvals.append(float(result.pvalues[term]))
        holm = multipletests(feature_pvals, method="holm")[1]
        holm_by_term = dict(zip(feature_terms, holm))
        for term in retained_features:
            beta = float(result.params[term])
            term_ci = ci.loc[term]
            rows.append(
                {
                    "analysis": "Judging panel",
                    "contrast": "all_contrasts",
                    "task_scope": str(task),
                    "term": term,
                    "beta_log_odds": beta,
                    "std_error": float(result.bse[term]),
                    "p_value": float(result.pvalues[term]),
                    "p_holm": float(holm_by_term[term]),
                    "odds_ratio": finite_exp(beta),
                    "odds_ratio_ci_low": finite_exp(float(term_ci[0])),
                    "odds_ratio_ci_high": finite_exp(float(term_ci[1])),
                    "n_pairs": int(len(task_df)),
                    "n_rows": int(len(y)),
                }
            )
    return pd.DataFrame(rows)


def table_rows_from_coefficients(df: pd.DataFrame, *, analysis: str, contrast: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out = out[out["p_holm"].lt(P_THRESHOLD)].copy()
    if out.empty:
        return pd.DataFrame()
    out["analysis"] = analysis
    if contrast is not None:
        out["contrast"] = contrast
    return out


def panel_overall_rows() -> pd.DataFrame:
    df = pd.read_csv(PANEL_COEFFICIENTS)
    df = df[
        df["outcome"].eq("panel_strength")
        & df["model"].eq("feature_only")
        & df["term_type"].eq("feature")
    ].copy()
    df = df.rename(
        columns={
            "p_holm_features_within_outcome_model": "p_holm",
            "std_error_cluster_run": "std_error",
        }
    )
    df["analysis"] = "Judging panel"
    df["contrast"] = "all_contrasts"
    df["task_scope"] = "all_tasks"
    df["n_rows"] = df["n_rows"]
    return df[df["p_holm"].lt(P_THRESHOLD)].copy()


def condition_rows(*, task_specific: bool) -> pd.DataFrame:
    sig = pd.read_csv(CONDITION_SIGNIFICANT)
    if sig.empty:
        return pd.DataFrame()
    if task_specific:
        sig = sig[~sig["task_scope"].eq("all_tasks")].copy()
    else:
        sig = sig[sig["task_scope"].eq("all_tasks")].copy()
    sig = sig.rename(
        columns={
            "p_holm_features_within_model": "p_holm",
            "std_error_cluster_pair": "std_error",
        }
    )
    sig["analysis"] = "Condition origin"
    return sig


def none_rows_for_missing_conditions(existing: pd.DataFrame, *, task_specific: bool) -> pd.DataFrame:
    perf = pd.read_csv(CONDITION_PERFORMANCE)
    perf = perf[perf["contrast"].isin(["direct_instruction", "amount", "moral", "utility"])].copy()
    if task_specific:
        perf = perf[~perf["task_scope"].eq("all_tasks")].copy()
    else:
        perf = perf[perf["task_scope"].eq("all_tasks")].copy()
    existing_keys = set(zip(existing.get("contrast", []), existing.get("task_scope", [])))
    rows = []
    for _, row in perf.iterrows():
        key = (row["contrast"], row["task_scope"])
        if key not in existing_keys:
            rows.append(
                {
                    "analysis": "Condition origin",
                    "contrast": row["contrast"],
                    "task_scope": row["task_scope"],
                    "term": "none",
                    "beta_log_odds": np.nan,
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "p_holm": np.nan,
                    "odds_ratio": np.nan,
                    "n_pairs": int(row["n_pairs"]),
                    "n_rows": int(row["n_responses"]),
                }
            )
    return pd.DataFrame(rows)


def none_rows_for_missing_panel_tasks(existing: pd.DataFrame, all_task_rows: pd.DataFrame) -> pd.DataFrame:
    existing_tasks = set(existing["task_scope"]) if not existing.empty else set()
    rows = []
    for task, task_df in all_task_rows.groupby("task_scope", dropna=False):
        if task not in existing_tasks:
            rows.append(
                {
                    "analysis": "Judging panel",
                    "contrast": "all_contrasts",
                    "task_scope": task,
                    "term": "none",
                    "beta_log_odds": np.nan,
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "p_holm": np.nan,
                    "odds_ratio": np.nan,
                    "n_pairs": int(task_df["n_pairs"].iloc[0]),
                    "n_rows": int(task_df["n_rows"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def paper_format(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["odds_ratio_ci_low", "odds_ratio_ci_high"]:
        if col not in out.columns:
            out[col] = np.nan
    out["Analysis"] = out["analysis"]
    out["Contrast"] = out["contrast"].map(CONTRAST_LABELS).fillna(out["contrast"])
    out["Task"] = out["task_scope"].map(TASK_LABELS).fillna(out["task_scope"])
    out["Predictor"] = out["term"].map(FEATURE_LABELS).fillna(out["term"])
    out.loc[out["term"].eq("none"), "Predictor"] = "None significant"
    out["Direction"] = out["beta_log_odds"].map(direction)
    out["Beta"] = out["beta_log_odds"].map(format_float)
    out["OR"] = out["odds_ratio"].map(format_float)
    out["OR 95% CI"] = [
        format_ci(low, high)
        for low, high in zip(out["odds_ratio_ci_low"], out["odds_ratio_ci_high"], strict=False)
    ]
    out["Holm p"] = out["p_holm"].map(format_p)
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


def make_wide_main_table(main_rows: pd.DataFrame) -> pd.DataFrame:
    sig = main_rows[
        main_rows["term"].ne("none")
        & main_rows["task_scope"].eq("all_tasks")
        & main_rows["p_holm"].lt(P_THRESHOLD)
    ].copy()
    terms = set(sig["term"])
    ordered_terms = [term for term in WIDE_FEATURE_ORDER if term in terms]
    ordered_terms.extend(sorted(terms - set(ordered_terms)))

    rows: list[dict[str, str]] = []
    for term in ordered_terms:
        row = {"Predictor": FEATURE_LABELS.get(term, term)}
        for contrast_key, label in WIDE_TABLE_CONTRASTS:
            if contrast_key == "judge_panel":
                matches = sig[(sig["analysis"].eq("Judging panel")) & (sig["term"].eq(term))]
            else:
                matches = sig[
                    sig["analysis"].eq("Condition origin")
                    & sig["contrast"].eq(contrast_key)
                    & sig["term"].eq(term)
                ]
            if matches.empty:
                row[label] = ""
                continue
            match = matches.iloc[0]
            row[label] = format_or_ci(
                match["odds_ratio"],
                match.get("odds_ratio_ci_low", np.nan),
                match.get("odds_ratio_ci_high", np.nan),
            )
        rows.append(row)

    return pd.DataFrame(rows, columns=["Predictor"] + [label for _, label in WIDE_TABLE_CONTRASTS])


def make_wide_task_table(appendix_rows: pd.DataFrame) -> pd.DataFrame:
    sig = appendix_rows[
        appendix_rows["term"].ne("none")
        & appendix_rows["task_scope"].ne("all_tasks")
        & appendix_rows["p_holm"].lt(P_THRESHOLD)
    ].copy()

    rows: list[dict[str, str]] = []
    for task in sorted(sig["task_scope"].unique(), key=lambda value: TASK_LABELS.get(value, value)):
        task_sig = sig[sig["task_scope"].eq(task)].copy()
        terms = set(task_sig["term"])
        ordered_terms = [term for term in WIDE_FEATURE_ORDER if term in terms]
        ordered_terms.extend(sorted(terms - set(ordered_terms)))

        for term in ordered_terms:
            row = {
                "Task": TASK_LABELS.get(task, task),
                "Predictor": FEATURE_LABELS.get(term, term),
            }
            for contrast_key, label in WIDE_TABLE_CONTRASTS:
                if contrast_key == "judge_panel":
                    matches = task_sig[(task_sig["analysis"].eq("Judging panel")) & (task_sig["term"].eq(term))]
                else:
                    matches = task_sig[
                        task_sig["analysis"].eq("Condition origin")
                        & task_sig["contrast"].eq(contrast_key)
                        & task_sig["term"].eq(term)
                    ]
                if matches.empty:
                    row[label] = ""
                    continue
                match = matches.iloc[0]
                row[label] = format_or_ci(
                    match["odds_ratio"],
                    match.get("odds_ratio_ci_low", np.nan),
                    match.get("odds_ratio_ci_high", np.nan),
                )
            rows.append(row)

    return pd.DataFrame(rows, columns=["Task", "Predictor"] + [label for _, label in WIDE_TABLE_CONTRASTS])


def write_wide_markdown(wide: pd.DataFrame) -> None:
    lines = [
        "# Main-Text Significant Predictor Table",
        "",
        "Cells show odds ratio [95% CI] for predictors significant after Holm correction within the relevant model.",
        "Empty cells indicate that the predictor was not significant in that column.",
        "Rows are the union of significant overall predictors across the judging-panel model and the four contrast-origin models.",
        "",
        markdown_table(wide),
        "",
    ]
    OUT_WIDE_MAIN_MD.write_text("\n".join(lines), encoding="utf-8")


def write_wide_task_markdown(wide: pd.DataFrame) -> None:
    lines = [
        "# Task-Specific Significant Predictor Tables",
        "",
        "Cells show odds ratio [95% CI] for predictors significant after Holm correction within the relevant task-specific model.",
        "Empty cells indicate that the predictor was not significant in that column.",
        "Rows are the union of significant task-specific predictors within each task.",
        "",
    ]
    if wide.empty:
        lines.extend(["_No task-specific significant predictors._", ""])
    else:
        for task, task_df in wide.groupby("Task", sort=False):
            lines.extend([f"## {task}", "", markdown_table(task_df.drop(columns=["Task"]).reset_index(drop=True)), ""])
    OUT_WIDE_TASK_MD.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(main: pd.DataFrame, appendix: pd.DataFrame) -> None:
    lines = [
        "# Significant Predictor Tables",
        "",
        "Inclusion rule: Holm-corrected p < .05 within each model.",
        "",
        "Judging-panel rows use the feature-only panel-strength model.",
        "Task-specific judging-panel rows refit that model within task using the reduced feature set, dropping zero-variance predictors.",
        "Condition-origin rows use high-vs-low condition origin as the binary outcome.",
        "",
        "## Main Table: Overall Predictors",
        "",
        markdown_table(main),
        "",
        "## Appendix Table: Task-Specific Predictors",
        "",
        markdown_table(appendix),
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    panel_main = panel_overall_rows()
    panel_task_all = panel_strength_task_coefficients()
    panel_task_sig = panel_task_all[panel_task_all["p_holm"].lt(P_THRESHOLD)].copy()
    panel_task_none = none_rows_for_missing_panel_tasks(panel_task_sig, panel_task_all)

    condition_main = condition_rows(task_specific=False)
    condition_main_none = none_rows_for_missing_conditions(condition_main, task_specific=False)
    condition_appendix = condition_rows(task_specific=True)
    condition_appendix_none = none_rows_for_missing_conditions(condition_appendix, task_specific=True)

    main_rows = pd.concat([panel_main, condition_main, condition_main_none], ignore_index=True, sort=False)
    appendix_rows = pd.concat(
        [panel_task_sig, panel_task_none, condition_appendix, condition_appendix_none],
        ignore_index=True,
        sort=False,
    )
    sort_cols = ["analysis", "contrast", "task_scope", "p_holm", "term"]
    main_rows = main_rows.sort_values(sort_cols, na_position="last")
    appendix_rows = appendix_rows.sort_values(sort_cols, na_position="last")

    main_table = paper_format(main_rows)
    appendix_table = paper_format(appendix_rows)
    wide_main_table = make_wide_main_table(main_rows)
    wide_task_table = make_wide_task_table(appendix_rows)

    main_table.to_csv(OUT_MAIN_CSV, index=False)
    appendix_table.to_csv(OUT_APPENDIX_CSV, index=False)
    main_table.to_latex(OUT_MAIN_TEX, index=False, escape=True)
    appendix_table.to_latex(OUT_APPENDIX_TEX, index=False, escape=True)
    wide_main_table.to_csv(OUT_WIDE_MAIN_CSV, index=False)
    wide_main_table.to_latex(OUT_WIDE_MAIN_TEX, index=False, escape=True)
    wide_task_table.to_csv(OUT_WIDE_TASK_CSV, index=False)
    wide_task_table.to_latex(OUT_WIDE_TASK_TEX, index=False, escape=True)
    write_markdown(main_table, appendix_table)
    write_wide_markdown(wide_main_table)
    write_wide_task_markdown(wide_task_table)

    print(f"wide main csv: {OUT_WIDE_MAIN_CSV}")
    print(f"wide main tex: {OUT_WIDE_MAIN_TEX}")
    print(f"wide main markdown: {OUT_WIDE_MAIN_MD}")
    print(f"wide task csv: {OUT_WIDE_TASK_CSV}")
    print(f"wide task tex: {OUT_WIDE_TASK_TEX}")
    print(f"wide task markdown: {OUT_WIDE_TASK_MD}")
    print(f"main csv: {OUT_MAIN_CSV}")
    print(f"appendix csv: {OUT_APPENDIX_CSV}")
    print(f"main tex: {OUT_MAIN_TEX}")
    print(f"appendix tex: {OUT_APPENDIX_TEX}")
    print(f"markdown: {OUT_MD}")
    print(f"wide main rows: {len(wide_main_table)}")
    print(f"wide task rows: {len(wide_task_table)}")
    print(f"main rows: {len(main_table)}")
    print(f"appendix rows: {len(appendix_table)}")


if __name__ == "__main__":
    main()
