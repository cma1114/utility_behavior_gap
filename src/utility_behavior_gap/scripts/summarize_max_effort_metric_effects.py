#!/usr/bin/env python3
"""Summarize paired effect sizes for headline max-effort essay metrics."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scipy import stats

from utility_behavior_gap.io_utils import write_csv_rows
from utility_behavior_gap.paths import ANALYSIS


BY_PAIR_IN = ANALYSIS / "max_effort_essay_mechanism_by_pair.csv"
EFFECTS_CSV = ANALYSIS / "max_effort_essay_metric_effects.csv"
EFFECTS_MD = ANALYSIS / "max_effort_essay_metric_effects.md"


METRICS = [
    ("Word count", "delta_words", "words", "more words"),
    ("Mean word length", "delta_spacy_mean_word_length_chars", "chars", "longer words"),
    ("Flesch-Kincaid grade", "delta_textstat_flesch_kincaid_grade", "grade", "higher grade level"),
    ("Flesch reading ease", "delta_textstat_flesch_reading_ease", "points", "easier to read"),
    ("Gunning fog", "delta_textstat_gunning_fog", "grade", "higher grade level"),
    ("spaCy adjective rate", "delta_spacy_adjective_rate", "pp", "more adjectives"),
    ("spaCy adverb rate", "delta_spacy_adverb_rate", "pp", "more adverbs"),
    ("spaCy ADJ+ADV rate", "delta_spacy_modifier_rate", "pp", "more adjectives/adverbs"),
    ("Contrast framing", "delta_contrast_framing", "count", "more contrast markers"),
    ("Counterargument", "delta_counterargument", "count", "more counterargument markers"),
    ("Example markers", "delta_example_markers", "count", "more example markers"),
    ("Colon/semicolon", "delta_colon_semicolon", "count", "more colon/semicolon structure"),
    ("Imagistic framing", "delta_imagistic_framing", "count", "more image/metaphor words"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ci95(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n < 2:
        return (math.nan, math.nan)
    sd = stats.tstd(vals)
    se = sd / math.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=mean(vals), scale=se)
    return float(lo), float(hi)


def paired_dz(vals: list[float]) -> float:
    sd = stats.tstd(vals)
    if sd == 0:
        return math.nan
    return float(mean(vals) / sd)


def sign_test_p(vals: list[float]) -> float:
    positive = sum(value > 0 for value in vals)
    negative = sum(value < 0 for value in vals)
    if positive + negative == 0:
        return math.nan
    return float(stats.binomtest(max(positive, negative), positive + negative, 0.5).pvalue)


def fmt(value: float, unit: str) -> str:
    if math.isnan(value):
        return ""
    if unit == "pp":
        return f"{100 * value:+.2f} pp"
    if unit in {"words", "count"}:
        return f"{value:+.2f}"
    if unit in {"grade", "points", "chars"}:
        return f"{value:+.3f}"
    return f"{value:+.3f}"


def format_p(value: float) -> str:
    if math.isnan(value):
        return ""
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}"


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_actor: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_actor[row["actor"]].append(row)

    out: list[dict[str, str]] = []
    for actor in sorted(by_actor):
        actor_rows = by_actor[actor]
        for metric_label, field, unit, interpretation in METRICS:
            vals = [float(row[field]) for row in actor_rows]
            lo, hi = ci95(vals)
            positive = sum(value > 0 for value in vals)
            negative = sum(value < 0 for value in vals)
            zero = sum(value == 0 for value in vals)
            out.append(
                {
                    "actor": actor,
                    "metric": metric_label,
                    "interpretation_if_positive": interpretation,
                    "n_pairs": str(len(vals)),
                    "mean_delta": fmt(mean(vals), unit),
                    "ci95_low": fmt(lo, unit),
                    "ci95_high": fmt(hi, unit),
                    "cohen_dz": f"{paired_dz(vals):+.3f}",
                    "strong_higher": str(positive),
                    "normal_higher": str(negative),
                    "equal": str(zero),
                    "sign_test_p": format_p(sign_test_p(vals)),
                }
            )
    return out


def markdown(rows: list[dict[str, str]]) -> str:
    priority_metrics = {
        "Word count",
        "Mean word length",
        "Flesch-Kincaid grade",
        "Flesch reading ease",
        "Gunning fog",
        "spaCy ADJ+ADV rate",
        "Contrast framing",
        "Counterargument",
        "Colon/semicolon",
    }
    columns = [
        "actor",
        "metric",
        "mean_delta",
        "ci95_low",
        "ci95_high",
        "cohen_dz",
        "strong_higher",
        "normal_higher",
        "sign_test_p",
    ]
    lines = [
        "# Max-Effort Essay Metric Effect Sizes",
        "",
        "Each row summarizes paired strong-minus-normal deltas over 200 essay pairs for one actor.",
        "The 95% CI is a paired t interval over pair-level deltas. The sign test counts how often",
        "the strong essay had a higher metric value than the normal essay, ignoring exact ties.",
        "",
        "These are exploratory pair-level summaries. Because the design repeats five essay topics,",
        "pair-level p-values should not be treated as final generalization tests over topics.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        if row["metric"] not in priority_metrics:
            continue
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = summarize(read_rows(BY_PAIR_IN))
    write_csv_rows(EFFECTS_CSV, rows)
    EFFECTS_MD.write_text(markdown(rows), encoding="utf-8")
    print(f"wrote {EFFECTS_CSV}")
    print(f"wrote {EFFECTS_MD}")


if __name__ == "__main__":
    main()
