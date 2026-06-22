#!/usr/bin/env python3
"""Select length-robust feature dimensions for paper tables.

This local-only analysis combines generic text features and optional
task-specific LLM rubric features. A feature is selected when:

1. The left arm differs from the right arm after controlling for word-count
   difference. Word count itself is tested without this control.
2. The judging panel's left-minus-right score is associated with that feature
   after controlling for word-count difference. For word count itself, the
   association is the word-count coefficient.
3. The two signs align, so the feature difference is in the direction favored
   by the panel.

No model APIs are called.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from utility_behavior_gap.constants import TASK_LABEL
from utility_behavior_gap.feature_specs import rubric_dimension_labels
from utility_behavior_gap.output_exclusions import (
    filter_semantic_excluded_pair_rows,
    filter_valid_pair_rows,
)
from utility_behavior_gap.paths import ANALYSIS


DEFAULT_PAIR_DELTAS = ANALYSIS / "final_text_analysis_pair_deltas.csv"
DEFAULT_DIRECT_RUBRIC_DIR = (
    ANALYSIS
    / "task_rubric_feature_coding"
    / "direct_instruction__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash"
)

GENERIC_FEATURES = [
    {
        "feature_id": "words",
        "feature_label": "Words",
        "definition": "Word count.",
        "delta_col": "delta_words",
        "is_words": True,
    },
    {
        "feature_id": "paragraphs",
        "feature_label": "Paragraphs",
        "definition": "Paragraph count split on blank lines.",
        "delta_col": "delta_paragraphs",
        "is_words": False,
    },
    {
        "feature_id": "mattr_50",
        "feature_label": "MATTR-50",
        "definition": "Moving-average type-token ratio computed in 50-word windows.",
        "delta_col": "delta_mattr_50",
        "is_words": False,
    },
    {
        "feature_id": "rare_word_rate_per_1k",
        "feature_label": "Rare-word rate per 1k words",
        "definition": "Eligible alphabetic tokens with wordfreq English Zipf frequency below 3.5, per 1,000 eligible tokens.",
        "delta_col": "delta_rare_word_rate_per_1k",
        "is_words": False,
    },
    {
        "feature_id": "quantitative_detail_rate",
        "feature_label": "Quantitative detail rate",
        "definition": "Numeric tokens plus percentage expressions per 1,000 words.",
        "delta_col": "delta_quantitative_detail_rate",
        "is_words": False,
    },
    {
        "feature_id": "textstat_flesch_kincaid_grade",
        "feature_label": "Flesch-Kincaid grade",
        "definition": "Flesch-Kincaid grade level.",
        "delta_col": "delta_textstat_flesch_kincaid_grade",
        "is_words": False,
    },
    {
        "feature_id": "positive_words_per_1k",
        "feature_label": "Positive-word rate per 1k words",
        "definition": "VADER positive-lexicon tokens per 1,000 words.",
        "delta_col": "delta_positive_words_per_1k",
        "is_words": False,
    },
    {
        "feature_id": "negative_words_per_1k",
        "feature_label": "Negative-word rate per 1k words",
        "definition": "VADER negative-lexicon tokens per 1,000 words.",
        "delta_col": "delta_negative_words_per_1k",
        "is_words": False,
    },
]


def normal_p_two_sided(z_value: float) -> float:
    if not math.isfinite(z_value):
        return math.nan
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def fmt(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    rounded = round(float(value), digits)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:+.{digits}f}"


def fmt_p(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    return "\n".join(lines)


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    sd = numeric.std(ddof=0)
    if not math.isfinite(float(sd)) or float(sd) <= 0:
        return pd.Series(np.nan, index=series.index)
    return (numeric - numeric.mean()) / sd


def fit_adjusted_gap(df: pd.DataFrame, *, feature_col: str, is_words: bool) -> dict[str, float | int]:
    work = df.dropna(subset=[feature_col, "delta_words", "actor"]).copy()
    if len(work) < 8:
        return empty_result(len(work))

    y = pd.to_numeric(work[feature_col], errors="coerce")
    parts = [pd.Series(1.0, index=work.index, name="const")]
    if not is_words:
        parts.append(pd.to_numeric(work["delta_words"], errors="coerce").rename("delta_words"))
    if work["actor"].nunique() > 1:
        parts.append(pd.get_dummies(work["actor"].astype(str), prefix="actor", drop_first=True, dtype=float))
    x = pd.concat(parts, axis=1).astype(float)
    valid = y.notna() & x.notna().all(axis=1)
    y = y[valid]
    x = x.loc[valid]
    if len(y) < 8 or y.nunique(dropna=True) <= 1:
        return empty_result(len(y))
    pair_delta_sd = float(y.std(ddof=1))

    result = sm.OLS(y.to_numpy(dtype=float), x.to_numpy(dtype=float)).fit(cov_type="HC3")
    x_equal = x.copy()
    if "delta_words" in x_equal.columns:
        x_equal["delta_words"] = 0.0
    contrast = x_equal.to_numpy(dtype=float).mean(axis=0)
    point = float(np.dot(contrast, result.params))
    se = float(np.sqrt(np.dot(contrast, np.dot(result.cov_params(), contrast))))
    ci_low = point - 1.96 * se
    ci_high = point + 1.96 * se
    p_value = normal_p_two_sided(point / se) if se > 0 else math.nan
    if math.isfinite(pair_delta_sd) and pair_delta_sd > 0:
        gap_effect_sd = point / pair_delta_sd
        gap_ci_low_sd = ci_low / pair_delta_sd
        gap_ci_high_sd = ci_high / pair_delta_sd
    else:
        gap_effect_sd = math.nan
        gap_ci_low_sd = math.nan
        gap_ci_high_sd = math.nan
    return {
        "n_gap": int(len(y)),
        "gap_effect": point,
        "gap_ci_low": ci_low,
        "gap_ci_high": ci_high,
        "gap_p": p_value,
        "gap_pair_delta_sd": pair_delta_sd,
        "gap_effect_sd": gap_effect_sd,
        "gap_ci_low_sd": gap_ci_low_sd,
        "gap_ci_high_sd": gap_ci_high_sd,
    }


def fit_panel_association(df: pd.DataFrame, *, feature_col: str, is_words: bool) -> dict[str, float | int]:
    work = df.dropna(subset=[feature_col, "delta_words", "panel_score", "actor"]).copy()
    if len(work) < 8:
        return empty_panel_result(len(work))

    y = pd.to_numeric(work["panel_score"], errors="coerce")
    if is_words:
        predictor = zscore(work["delta_words"]).rename("feature_z")
        parts = [pd.Series(1.0, index=work.index, name="const"), predictor]
    else:
        predictor = zscore(work[feature_col]).rename("feature_z")
        words = zscore(work["delta_words"]).rename("words_z")
        parts = [pd.Series(1.0, index=work.index, name="const"), predictor, words]
    if work["actor"].nunique() > 1:
        parts.append(pd.get_dummies(work["actor"].astype(str), prefix="actor", drop_first=True, dtype=float))
    x = pd.concat(parts, axis=1).astype(float)
    valid = y.notna() & x.notna().all(axis=1)
    y = y[valid]
    x = x.loc[valid]
    if len(y) < 8 or x["feature_z"].std(ddof=0) <= 0:
        return empty_panel_result(len(y))

    result = sm.OLS(y.to_numpy(dtype=float), x.to_numpy(dtype=float)).fit(cov_type="HC3")
    coef = float(result.params[list(x.columns).index("feature_z")])
    se = float(result.bse[list(x.columns).index("feature_z")])
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    p_value = float(result.pvalues[list(x.columns).index("feature_z")])
    return {
        "n_panel": int(len(y)),
        "panel_coef_per_sd": coef,
        "panel_ci_low": ci_low,
        "panel_ci_high": ci_high,
        "panel_p": p_value,
    }


def empty_result(n: int = 0) -> dict[str, float | int]:
    return {
        "n_gap": int(n),
        "gap_effect": math.nan,
        "gap_ci_low": math.nan,
        "gap_ci_high": math.nan,
        "gap_p": math.nan,
        "gap_pair_delta_sd": math.nan,
        "gap_effect_sd": math.nan,
        "gap_ci_low_sd": math.nan,
        "gap_ci_high_sd": math.nan,
    }


def empty_panel_result(n: int = 0) -> dict[str, float | int]:
    return {
        "n_panel": int(n),
        "panel_coef_per_sd": math.nan,
        "panel_ci_low": math.nan,
        "panel_ci_high": math.nan,
        "panel_p": math.nan,
    }


def ci_excludes_zero(lo: float, hi: float) -> bool:
    return bool(math.isfinite(float(lo)) and math.isfinite(float(hi)) and (lo > 0 or hi < 0))


def signs_align(gap: float, panel: float) -> bool:
    if not math.isfinite(float(gap)) or not math.isfinite(float(panel)):
        return False
    return bool(gap * panel > 0)


def add_quantitative_detail_rate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = ["high_numbers", "low_numbers", "high_percentages", "low_percentages", "high_words", "low_words"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Cannot compute quantitative detail rate; missing columns: {missing}")
    for side in ("high", "low"):
        numeric = pd.to_numeric(out[f"{side}_numbers"], errors="coerce").fillna(0)
        percentages = pd.to_numeric(out[f"{side}_percentages"], errors="coerce").fillna(0)
        words = pd.to_numeric(out[f"{side}_words"], errors="coerce")
        out[f"{side}_quantitative_detail_rate"] = (numeric + percentages) / words * 1000
    out["delta_quantitative_detail_rate"] = (
        out["high_quantitative_detail_rate"] - out["low_quantitative_detail_rate"]
    )
    return out


def load_pair_deltas(path: Path, contrast: str) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    data = pd.read_csv(path, low_memory=False)
    data = data[data["contrast"].eq(contrast)].copy()
    if data.empty:
        raise ValueError(f"No pair-delta rows found for contrast {contrast!r} in {path}")
    data, valid_report = filter_valid_pair_rows(data)
    data, semantic_report = filter_semantic_excluded_pair_rows(data)
    data = add_quantitative_detail_rate(data)
    data["panel_score"] = pd.to_numeric(data["effect_score_high_minus_low"], errors="coerce")
    return data, valid_report, semantic_report


def generic_rows(pair_deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, task_data in pair_deltas.groupby("task", sort=True):
        for info in GENERIC_FEATURES:
            feature_col = str(info["delta_col"])
            if feature_col not in task_data.columns:
                continue
            gap = fit_adjusted_gap(task_data, feature_col=feature_col, is_words=bool(info["is_words"]))
            panel = fit_panel_association(task_data, feature_col=feature_col, is_words=bool(info["is_words"]))
            rows.append(
                make_row(
                    task=str(task),
                    family="Generic text feature",
                    feature_id=str(info["feature_id"]),
                    feature_label=str(info["feature_label"]),
                    definition=str(info["definition"]),
                    controls_words_for_gap=not bool(info["is_words"]),
                    controls_words_for_panel=not bool(info["is_words"]),
                    gap=gap,
                    panel=panel,
                )
            )
    return pd.DataFrame(rows)


def load_rubric_data(rubric_dir: Path, pair_deltas: pd.DataFrame) -> pd.DataFrame:
    flat = pd.read_csv(rubric_dir / "rubric_feature_analysis_flat_codes.csv")
    sample = pd.read_csv(rubric_dir / "rubric_feature_sample.csv")
    sample_cols = [col for col in [
        "coding_pair_uid",
        "source_pair_uid",
        "left_output_id",
        "right_output_id",
        "effect_score_left_minus_right",
        "panel_winner_condition",
    ] if col in sample.columns]
    sample = sample[sample_cols].drop_duplicates()
    sample["source_pair_uid"] = sample.get("source_pair_uid", pd.Series("", index=sample.index))
    sample["source_pair_uid"] = sample["source_pair_uid"].fillna("").astype(str)

    word_by_pair = pair_deltas[
        ["pair_uid", "high_output_id", "low_output_id", "delta_words", "effect_score_high_minus_low"]
    ].drop_duplicates()
    keyed = sample[sample["source_pair_uid"].str.strip().ne("")].merge(
        word_by_pair[["pair_uid", "delta_words", "effect_score_high_minus_low"]],
        left_on="source_pair_uid",
        right_on="pair_uid",
        how="left",
        validate="many_to_one",
    )
    unkeyed = sample[sample["source_pair_uid"].str.strip().eq("")].copy()
    if not unkeyed.empty:
        needed = {"left_output_id", "right_output_id"}
        if not needed.issubset(unkeyed.columns):
            raise ValueError(
                f"Rubric sample has blank source_pair_uid but lacks columns: {sorted(needed - set(unkeyed.columns))}"
            )
        unkeyed = unkeyed.merge(
            word_by_pair,
            left_on=["left_output_id", "right_output_id"],
            right_on=["high_output_id", "low_output_id"],
            how="left",
            validate="many_to_one",
        )
    sample = pd.concat([keyed, unkeyed], ignore_index=True, sort=False)
    missing_word = sample["delta_words"].isna()
    if missing_word.any():
        first = sample.loc[missing_word].iloc[0].to_dict()
        raise ValueError(f"Could not attach delta_words to {int(missing_word.sum())} rubric sample rows; first={first}")
    if "effect_score_left_minus_right" not in sample.columns:
        sample["effect_score_left_minus_right"] = pd.NA
    sample["effect_score_left_minus_right"] = pd.to_numeric(
        sample["effect_score_left_minus_right"], errors="coerce"
    )
    sample["effect_score_left_minus_right"] = sample["effect_score_left_minus_right"].fillna(
        pd.to_numeric(sample["effect_score_high_minus_low"], errors="coerce")
    )
    merged = flat.merge(
        sample[
            [
                "coding_pair_uid",
                "source_pair_uid",
                "effect_score_left_minus_right",
                "panel_winner_condition",
                "delta_words",
            ]
        ],
        on="coding_pair_uid",
        how="left",
        validate="many_to_one",
    )
    merged["rubric_score"] = pd.to_numeric(merged["left_minus_right_score"], errors="coerce")
    merged["panel_score"] = pd.to_numeric(merged["effect_score_left_minus_right"], errors="coerce")
    merged["delta_words"] = pd.to_numeric(merged["delta_words"], errors="coerce")
    return merged


def rubric_rows(rubric_dir: Path, pair_deltas: pd.DataFrame) -> pd.DataFrame:
    labels = rubric_dimension_labels()
    data = load_rubric_data(rubric_dir, pair_deltas)
    rows: list[dict[str, object]] = []
    for (task, dimension), group in data.groupby(["task", "dimension"], sort=True):
        gap = fit_adjusted_gap(group, feature_col="rubric_score", is_words=False)
        panel = fit_panel_association(group, feature_col="rubric_score", is_words=False)
        rows.append(
            make_row(
                task=str(task),
                family="LLM rubric marker",
                feature_id=str(dimension),
                feature_label=labels.get(str(dimension), str(dimension)),
                definition=str(group["description"].dropna().iloc[0]) if group["description"].notna().any() else "",
                controls_words_for_gap=True,
                controls_words_for_panel=True,
                gap=gap,
                panel=panel,
            )
        )
    return pd.DataFrame(rows)


def make_row(
    *,
    task: str,
    family: str,
    feature_id: str,
    feature_label: str,
    definition: str,
    controls_words_for_gap: bool,
    controls_words_for_panel: bool,
    gap: dict[str, float | int],
    panel: dict[str, float | int],
) -> dict[str, object]:
    gap_clear = ci_excludes_zero(float(gap["gap_ci_low"]), float(gap["gap_ci_high"]))
    panel_clear = ci_excludes_zero(float(panel["panel_ci_low"]), float(panel["panel_ci_high"]))
    aligned = signs_align(float(gap["gap_effect"]), float(panel["panel_coef_per_sd"]))
    selected = bool(gap_clear and panel_clear and aligned)
    return {
        "task": task,
        "task_label": TASK_LABEL.get(task, task),
        "family": family,
        "feature_id": feature_id,
        "feature_label": feature_label,
        "n_gap": gap["n_gap"],
        "gap_effect": gap["gap_effect"],
        "gap_ci_low": gap["gap_ci_low"],
        "gap_ci_high": gap["gap_ci_high"],
        "gap_p": gap["gap_p"],
        "gap_pair_delta_sd": gap["gap_pair_delta_sd"],
        "gap_effect_sd": gap["gap_effect_sd"],
        "gap_ci_low_sd": gap["gap_ci_low_sd"],
        "gap_ci_high_sd": gap["gap_ci_high_sd"],
        "gap_ci_excludes_zero": gap_clear,
        "n_panel": panel["n_panel"],
        "panel_coef_per_sd": panel["panel_coef_per_sd"],
        "panel_ci_low": panel["panel_ci_low"],
        "panel_ci_high": panel["panel_ci_high"],
        "panel_p": panel["panel_p"],
        "panel_ci_excludes_zero": panel_clear,
        "signs_align": aligned,
        "selected": selected,
        "controls_words_for_gap": controls_words_for_gap,
        "controls_words_for_panel": controls_words_for_panel,
        "definition": definition,
    }


def write_markdown(results: pd.DataFrame, path: Path, *, contrast: str, pair_deltas: Path, rubric_dir: Path | None) -> None:
    lines = [
        f"# Length-Controlled Feature Selection: {contrast}",
        "",
        "Selection rule: keep features whose left-minus-right arm gap excludes zero, whose panel association excludes zero, and whose signs align. Non-word features control for word-count difference in both tests. Word count itself is included when its raw arm gap and panel association are both clear.",
        "",
        "Gap model for non-word features: `feature_delta ~ delta_words + actor fixed effects`.",
        "Panel model for non-word features: `panel_score ~ standardized(feature_delta) + standardized(delta_words) + actor fixed effects`.",
        "Word-count models omit the word-count control because word count is the feature.",
        "The standardized arm gap divides the adjusted arm gap by the observed SD of the paired feature deltas within the same task. It is descriptive scale information, not a separate significance test.",
        "",
        f"Pair deltas: `{pair_deltas}`",
    ]
    if rubric_dir is not None:
        lines.append(f"Rubric run: `{rubric_dir}`")
    lines.append("")

    task_order = ["Essay writing", "Grant abstract", "Incident postmortem", "Translation"]
    for task_label in task_order:
        task_rows = results[results["task_label"].eq(task_label) & results["selected"]].copy()
        if task_rows.empty:
            continue
        task_rows["gap"] = [
            f"{fmt(row.gap_effect)} [{fmt(row.gap_ci_low)}, {fmt(row.gap_ci_high)}]"
            for row in task_rows.itertuples()
        ]
        task_rows["panel"] = [
            f"{fmt(row.panel_coef_per_sd)} [{fmt(row.panel_ci_low)}, {fmt(row.panel_ci_high)}]"
            for row in task_rows.itertuples()
        ]
        task_rows["gap_sd"] = [
            f"{fmt(row.gap_effect_sd)} [{fmt(row.gap_ci_low_sd)}, {fmt(row.gap_ci_high_sd)}]"
            for row in task_rows.itertuples()
        ]
        show = task_rows[
            [
                "family",
                "feature_label",
                "n_gap",
                "gap",
                "gap_sd",
                "n_panel",
                "panel",
            ]
        ].rename(
            columns={
                "family": "family",
                "feature_label": "feature",
                "n_gap": "n gap",
                "gap": "arm gap",
                "gap_sd": "arm gap, SD",
                "n_panel": "n panel",
                "panel": "panel assoc.",
            }
        )
        lines.extend([f"## {task_label}", "", markdown_table(show), ""])

    all_selected = results[results["selected"]].copy()
    lines.extend(
        [
            "## Output Files",
            "",
            f"- selected/full CSV: `{path.with_suffix('.csv')}`",
            "",
            f"Selected rows: `{len(all_selected)}` of `{len(results)}` tested feature-task rows.",
            "",
        ]
    )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_selected_table(results: pd.DataFrame, out_prefix: Path) -> tuple[Path, Path]:
    selected = results[results["selected"]].copy()
    selected["arm_gap"] = [
        f"{fmt(row.gap_effect)} [{fmt(row.gap_ci_low)}, {fmt(row.gap_ci_high)}]"
        for row in selected.itertuples()
    ]
    selected["panel_assoc"] = [
        f"{fmt(row.panel_coef_per_sd)} [{fmt(row.panel_ci_low)}, {fmt(row.panel_ci_high)}]"
        for row in selected.itertuples()
    ]
    selected["arm_gap_sd"] = [
        f"{fmt(row.gap_effect_sd)} [{fmt(row.gap_ci_low_sd)}, {fmt(row.gap_ci_high_sd)}]"
        for row in selected.itertuples()
    ]
    table = selected[
        [
            "task_label",
            "family",
            "feature_label",
            "n_gap",
            "arm_gap",
            "arm_gap_sd",
            "gap_p",
            "n_panel",
            "panel_assoc",
            "panel_p",
        ]
    ].rename(
        columns={
            "task_label": "Task",
            "family": "Family",
            "feature_label": "Feature",
            "n_gap": "N gap",
            "arm_gap": "Arm gap (95% CI)",
            "arm_gap_sd": "Arm gap, SD units (95% CI)",
            "gap_p": "Gap p",
            "n_panel": "N panel",
            "panel_assoc": "Panel assoc. per SD (95% CI)",
            "panel_p": "Panel p",
        }
    )
    table["Gap p"] = table["Gap p"].map(fmt_p)
    table["Panel p"] = table["Panel p"].map(fmt_p)

    csv_path = out_prefix.with_name(out_prefix.name + "_selected_table.csv")
    md_path = out_prefix.with_name(out_prefix.name + "_selected_table.md")
    table.to_csv(csv_path, index=False)

    lines = [
        f"# {out_prefix.name.replace('_', ' ').title()} Selected Feature Table",
        "",
        f"Rows shown are exactly the selected rows from `{out_prefix.with_suffix('.csv')}`.",
        "",
        "Arm gap is left-minus-right at equal word count for non-word features. Panel association is the standardized feature coefficient in `panel_score ~ feature + words + actor`; for Words, it is the word-count coefficient without word-count control.",
        "",
        "Arm gap in SD units divides the adjusted arm gap by the observed SD of the paired feature deltas within the same task. Use it to compare effect sizes across features with different raw units.",
        "",
    ]
    for task_label in ["Essay writing", "Grant abstract", "Incident postmortem", "Translation"]:
        task_rows = table[table["Task"].eq(task_label)]
        if task_rows.empty:
            continue
        lines.extend([f"## {task_label}", "", markdown_table(task_rows), ""])
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return csv_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast", required=True)
    parser.add_argument("--pair-deltas", type=Path, default=DEFAULT_PAIR_DELTAS)
    parser.add_argument("--rubric-dir", type=Path, default=None)
    parser.add_argument("--out-prefix", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rubric_dir = args.rubric_dir
    if rubric_dir is None and args.contrast == "direct_instruction" and DEFAULT_DIRECT_RUBRIC_DIR.exists():
        rubric_dir = DEFAULT_DIRECT_RUBRIC_DIR

    pair_deltas, valid_report, semantic_report = load_pair_deltas(args.pair_deltas, args.contrast)
    pieces = [generic_rows(pair_deltas)]
    if rubric_dir is not None:
        pieces.append(rubric_rows(rubric_dir, pair_deltas))
    results = pd.concat(pieces, ignore_index=True)

    task_order = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]
    family_order = {"Generic text feature": 0, "LLM rubric marker": 1}
    results["task_order"] = results["task"].map({task: idx for idx, task in enumerate(task_order)}).fillna(99)
    results["family_order"] = results["family"].map(family_order).fillna(99)
    results = results.sort_values(
        ["task_order", "selected", "family_order", "gap_effect"],
        ascending=[True, False, True, False],
    ).drop(columns=["task_order", "family_order"])

    out_prefix = args.out_prefix or (ANALYSIS / f"{args.contrast}_length_controlled_feature_selection")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    results.to_csv(csv_path, index=False)
    write_markdown(results, md_path, contrast=args.contrast, pair_deltas=args.pair_deltas, rubric_dir=rubric_dir)
    selected_csv_path, selected_md_path = write_selected_table(results, out_prefix)

    print(f"valid-output filter: {valid_report}")
    print(f"semantic exclusion filter: {semantic_report}")
    print(f"tested rows: {len(results)}")
    print(f"selected rows: {int(results['selected'].sum())}")
    print(f"csv: {csv_path}")
    print(f"summary: {md_path}")
    print(f"selected table csv: {selected_csv_path}")
    print(f"selected table summary: {selected_md_path}")


if __name__ == "__main__":
    main()
