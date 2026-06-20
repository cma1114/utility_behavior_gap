#!/usr/bin/env python3
"""Summarize low-vs-R0 text features into high-level families."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utility_behavior_gap.io_utils import write_csv_rows
from utility_behavior_gap.paths import ANALYSIS, ROOT


RUN_NAME = "highlow-r0-bridge__2026-06-13_21-12-00Z__hash-222ecc29ee05"
TASKS = [
    ("essay", "Essay"),
    ("grant_proposal_abstract", "Grant abstract"),
    ("incident_postmortem", "Incident postmortem"),
    ("translation", "Translation"),
]
FEATURE_PATH_TEMPLATE = (
    "highlow_r0_text_features__low_vs_r0__{task}__political__"
    f"{RUN_NAME}__feature_summary.csv"
)
PAIR_PATH = ANALYSIS / f"highlow_r0_bridge__{RUN_NAME}__pair_outcomes.csv"
OUT_CSV = ANALYSIS / f"highlow_r0_low_vs_r0_text_feature_families__{RUN_NAME}.csv"
NOTE_MD = ROOT / "notes" / "highlow_r0_low_vs_r0_text_feature_families.md"


FAMILIES = [
    (
        "Length/verbosity",
        ["words", "characters", "sentences", "paragraphs"],
        "How much output the model produced.",
    ),
    (
        "Sentence density",
        ["avg_sentence_words"],
        "Whether prose is packed into longer or shorter sentences.",
    ),
    (
        "Lexical profile",
        ["mean_word_length_chars", "unique_word_ratio"],
        "Word length and non-repetition; not a direct quality score.",
    ),
    (
        "Quantitative specificity",
        ["numbers", "percentages"],
        "Explicit numeric detail, including percentages.",
    ),
    (
        "Task-method specificity",
        ["method_markers", "specificity_markers"],
        "Predefined markers of evaluation/method language and concrete implementation detail.",
    ),
    (
        "Readability complexity",
        [
            "textstat_flesch_reading_ease",
            "textstat_flesch_kincaid_grade",
            "textstat_gunning_fog",
        ],
        "Automated readability indices; higher reading ease is easier, higher FK/fog is harder.",
    ),
    (
        "Modifier style",
        ["spacy_adjective_rate", "spacy_adverb_rate", "spacy_modifier_rate"],
        "spaCy adjective/adverb rates among alphabetic tokens.",
    ),
]


def feature_summary_path(task: str) -> Path:
    return ANALYSIS / FEATURE_PATH_TEMPLATE.format(task=task)


def row_for(df: pd.DataFrame, feature: str) -> dict[str, Any]:
    rows = df[(df["group"].eq("all_pairs")) & (df["feature"].eq(feature))]
    if rows.empty:
        raise KeyError(feature)
    return rows.iloc[0].to_dict()


def fmt(value: Any, digits: int = 2) -> str:
    if value == "" or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def metric_cell(row: dict[str, Any], feature_label: str) -> str:
    return (
        f"{feature_label}: Δ {fmt(row['delta_low_minus_r0'])} "
        f"[{fmt(row['ci_lo'])}, {fmt(row['ci_hi'])}]"
    )


def panel_outcomes() -> dict[str, dict[str, int]]:
    df = pd.read_csv(PAIR_PATH)
    df = df[(df["side"].eq("low")) & (df["domain"].eq("political"))]
    out: dict[str, dict[str, int]] = {}
    for task, _label in TASKS:
        sub = df[df["task"].eq(task)]
        out[task] = {
            "low_wins": int(sub["outcome_vs_r0"].eq("side").sum()),
            "r0_wins": int(sub["outcome_vs_r0"].eq("r0").sum()),
            "ties": int(sub["outcome_vs_r0"].eq("tie").sum()),
            "pairs": int(len(sub)),
        }
    return out


def build_family_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, task_label in TASKS:
        df = pd.read_csv(feature_summary_path(task))
        for family, features, definition in FAMILIES:
            for feature in features:
                row = row_for(df, feature)
                rows.append(
                    {
                        "task": task,
                        "task_label": task_label,
                        "family": family,
                        "family_definition": definition,
                        "feature": feature,
                        "feature_definition": row["definition"],
                        "n_pairs": int(row["n_pairs"]),
                        "low_mean": row["low_mean"],
                        "r0_mean": row["r0_mean"],
                        "delta_low_minus_r0": row["mean_delta_low_minus_r0"],
                        "ci_lo": row["bootstrap_ci_lo"],
                        "ci_hi": row["bootstrap_ci_hi"],
                        "low_greater_pairs": int(row["low_greater_pairs"]),
                        "r0_greater_pairs": int(row["r0_greater_pairs"]),
                    }
                )
    return rows


def task_table_lines(family_rows: list[dict[str, Any]], outcomes: dict[str, dict[str, int]]) -> list[str]:
    by_task_feature = {(row["task"], row["feature"]): row for row in family_rows}
    lines = [
        "| task | low/R0/tie | length | quantitative specificity | task-method specificity | readability/modifier notes |",
        "|---|---:|---|---|---|---|",
    ]
    for task, task_label in TASKS:
        out = outcomes[task]
        words = by_task_feature[(task, "words")]
        numbers = by_task_feature[(task, "numbers")]
        percentages = by_task_feature[(task, "percentages")]
        methods = by_task_feature[(task, "method_markers")]
        specificity = by_task_feature[(task, "specificity_markers")]
        reading = by_task_feature[(task, "textstat_flesch_reading_ease")]
        modifiers = by_task_feature[(task, "spacy_modifier_rate")]
        lines.append(
            "| {task_label} | {low}/{r0}/{tie} | {words} | {nums}; {pct} | {meth}; {spec} | {read}; {mod} |".format(
                task_label=task_label,
                low=out["low_wins"],
                r0=out["r0_wins"],
                tie=out["ties"],
                words=metric_cell(words, "words"),
                nums=metric_cell(numbers, "numbers"),
                pct=metric_cell(percentages, "percentages"),
                meth=metric_cell(methods, "method markers"),
                spec=metric_cell(specificity, "specificity markers"),
                read=metric_cell(reading, "reading ease"),
                mod=metric_cell(modifiers, "modifier rate"),
            )
        )
    return lines


def write_note(family_rows: list[dict[str, Any]], outcomes: dict[str, dict[str, int]]) -> None:
    lines = [
        "# High-Low Political Low Utility Versus R0 Text Features",
        "",
        f"Source run: `{RUN_NAME}`",
        "",
        "This note summarizes paired text-feature diagnostics for political-domain low-utility outputs versus matched R0 bare-task outputs. Deltas are `low minus R0`; negative values mean the low-utility output has less of that feature than R0.",
        "",
        "The confidence intervals are row-level paired bootstrap intervals over the 259 actor/task/item/repeat pairs for each task. They are descriptive diagnostics, not confirmatory causal tests.",
        "",
        "## Short Takeaway",
        "",
        "- Grant abstracts are the only task where low utility is clearly more verbose than R0. Essays and translations are essentially flat on length, and incident postmortems are shorter under low utility.",
        "- The grant-abstract result should not be summarized as low utility producing more numbers. In the full grant set, R0 has more numeric tokens and more percentages despite being shorter.",
        "- Quantitative specificity is task-dependent: R0 has more numbers/percentages in grant abstracts, low utility has somewhat more numbers in incident postmortems, and the other two tasks are close to flat.",
        "- No task shows a simple general pattern where low utility globally degrades all surface quality proxies. The effect looks more like task-specific shifts in how the model allocates detail.",
        "",
        "## High-Level Feature Families",
        "",
    ]
    lines.extend(task_table_lines(family_rows, outcomes))
    lines.extend(
        [
            "",
            "## Family Definitions",
            "",
        ]
    )
    for family, features, definition in FAMILIES:
        lines.append(f"- **{family}**: {definition} Components: `{', '.join(features)}`.")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Family CSV: `{OUT_CSV}`",
        ]
    )
    for task, _label in TASKS:
        lines.append(f"- {task} feature summary: `{feature_summary_path(task)}`")
    NOTE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    family_rows = build_family_rows()
    outcomes = panel_outcomes()
    write_csv_rows(OUT_CSV, family_rows)
    write_note(family_rows, outcomes)
    print(f"family csv: {OUT_CSV}")
    print(f"note: {NOTE_MD}")


if __name__ == "__main__":
    main()
