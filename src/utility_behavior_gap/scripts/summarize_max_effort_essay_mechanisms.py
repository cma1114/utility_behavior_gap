#!/usr/bin/env python3
"""Write compact human-readable summaries of max-effort essay mechanisms."""

from __future__ import annotations

import csv
from pathlib import Path

from utility_behavior_gap.io_utils import write_csv_rows
from utility_behavior_gap.paths import ANALYSIS


SUMMARY_IN = ANALYSIS / "max_effort_essay_mechanism_summary.csv"
COMPACT_CSV = ANALYSIS / "max_effort_essay_mechanism_compact_summary.csv"
COMPACT_MD = ANALYSIS / "max_effort_essay_mechanism_compact_summary.md"


METRICS = [
    ("Win rate", "strong_win_rate_ties_excluded", "%"),
    ("Word count", "mean_delta_words", "words"),
    ("Mean word length", "mean_delta_spacy_mean_word_length_chars", "chars"),
    ("FK grade", "mean_delta_textstat_flesch_kincaid_grade", "grade"),
    ("Reading ease", "mean_delta_textstat_flesch_reading_ease", "points"),
    ("Gunning fog", "mean_delta_textstat_gunning_fog", "grade"),
    ("ADJ rate", "mean_delta_spacy_adjective_rate", "pp"),
    ("ADV rate", "mean_delta_spacy_adverb_rate", "pp"),
    ("ADJ+ADV rate", "mean_delta_spacy_modifier_rate", "pp"),
    ("Contrast framing", "mean_delta_contrast_framing", "count"),
    ("Counterargument", "mean_delta_counterargument", "count"),
    ("Example markers", "mean_delta_example_markers", "count"),
    ("Colon/semicolon", "mean_delta_colon_semicolon", "count"),
    ("Imagistic framing", "mean_delta_imagistic_framing", "count"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def fmt(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "pp":
        return f"{100 * value:+.2f} pp"
    if unit in {"words", "count"}:
        return f"{value:+.2f}"
    if unit in {"grade", "points", "chars"}:
        return f"{value:+.3f}"
    return f"{value:+.3f}"


def compact_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        compact = {
            "actor": row["actor"],
            "pairs": row["pairs"],
            "panel_result": f"{row['strong_wins']} strong / {row['normal_wins']} normal / "
            f"{row['ties']} tie / {row['no_majority']} no-majority",
        }
        for label, field, unit in METRICS:
            compact[label] = fmt(as_float(row, field), unit)
        out.append(compact)
    return out


def markdown_table(rows: list[dict[str, str]]) -> str:
    columns = [
        "actor",
        "panel_result",
        "Win rate",
        "Word count",
        "Mean word length",
        "FK grade",
        "Reading ease",
        "Gunning fog",
        "ADJ rate",
        "ADV rate",
        "ADJ+ADV rate",
        "Contrast framing",
        "Counterargument",
        "Example markers",
        "Colon/semicolon",
        "Imagistic framing",
    ]
    lines = [
        "# Max-Effort Essay Mechanism Compact Summary",
        "",
        "All deltas are strong minus normal within paired clean direct-instruction essay prompts.",
        "Rates in percentage points use spaCy POS tags from `en_core_web_sm`; readability uses `textstat`.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "",
            "- Higher FK grade and Gunning fog mean more complex text; lower reading ease also means more complex text.",
            "- ADJ/ADV rates are real spaCy POS-tag rates, not suffix proxies.",
            "- Hand-coded rhetorical counts are exploratory descriptors, not validated quality measures.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = compact_rows(read_rows(SUMMARY_IN))
    write_csv_rows(COMPACT_CSV, rows)
    COMPACT_MD.write_text(markdown_table(rows), encoding="utf-8")
    print(f"wrote {COMPACT_CSV}")
    print(f"wrote {COMPACT_MD}")


if __name__ == "__main__":
    main()
