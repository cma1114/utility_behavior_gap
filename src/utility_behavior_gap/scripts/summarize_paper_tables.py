#!/usr/bin/env python3
"""Regenerate compact CSV summaries for paper tables and text claims."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

from utility_behavior_gap.paths import ANALYSIS, PROCESSED

OUT = ANALYSIS


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_int(value: str) -> int:
    return int(float(value))


def as_float(value: str) -> float:
    return float(value)


def is_positive(value: str) -> bool:
    return value.strip().lower() in {"yes", "true", "1"}


def wilson(wins: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals)


def summarize_comparison(
    *,
    label: str,
    predicted_side: str,
    other_side: str,
    source: str,
    rows: list[dict[str, str]],
    predicted_col: str,
    other_col: str,
    tie_col: str,
    rate_col: str,
    positive_col: str,
) -> dict:
    predicted = sum(as_int(row[predicted_col]) for row in rows)
    other = sum(as_int(row[other_col]) for row in rows)
    ties = sum(as_int(row[tie_col]) for row in rows)
    non_tied = predicted + other
    rate, ci_lo, ci_hi = wilson(predicted, non_tied)
    return {
        "comparison": label,
        "predicted_side": predicted_side,
        "other_side": other_side,
        "positive_cells": sum(1 for row in rows if is_positive(row[positive_col])),
        "total_cells": len(rows),
        "predicted_wins": predicted,
        "other_wins": other,
        "ties": ties,
        "non_tied_n": non_tied,
        "total_n": non_tied + ties,
        "pooled_win_rate": round(rate, 4),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "mean_cell_win_rate": round(mean(as_float(row[rate_col]) for row in rows), 4),
        "source": source,
    }


def build_cue_summary() -> list[dict]:
    highlow = read_rows(PROCESSED / "highlow_main_data.csv")
    same_count = read_rows(PROCESSED / "highlow_within_count_data.csv")
    system = read_rows(PROCESSED / "system_prompt_calibration_data.csv")
    moral = read_rows(PROCESSED / "moral_nolabel_main_data.csv")
    amount = [
        row
        for row in read_rows(PROCESSED / "incentive_channel_data.csv")
        if row["condition"] == "amount"
    ]

    return [
        summarize_comparison(
            label="High-low utility main",
            predicted_side="high utility",
            other_side="low utility",
            source="outputs/processed/highlow_main_data.csv",
            rows=highlow,
            predicted_col="n_high",
            other_col="n_low",
            tie_col="n_tie",
            rate_col="high_win_rate",
            positive_col="ci_positive",
        ),
        summarize_comparison(
            label="Same-count high-low",
            predicted_side="same-count high utility",
            other_side="same-count low utility",
            source="outputs/processed/highlow_within_count_data.csv",
            rows=same_count,
            predicted_col="n_high",
            other_col="n_low",
            tie_col="n_tie",
            rate_col="high_win_rate",
            positive_col="ci_positive",
        ),
        summarize_comparison(
            label="Strong vs normal prompt",
            predicted_side="strong prompt",
            other_side="normal prompt",
            source="outputs/processed/system_prompt_calibration_data.csv",
            rows=system,
            predicted_col="strong_wins",
            other_col="normal_wins",
            tie_col="ties",
            rate_col="strong_win_rate",
            positive_col="ci_positive",
        ),
        summarize_comparison(
            label="Moral no-label",
            predicted_side="good cause",
            other_side="harmful cause",
            source="outputs/processed/moral_nolabel_main_data.csv",
            rows=moral,
            predicted_col="n_good",
            other_col="n_bad",
            tie_col="n_tie",
            rate_col="win_rate",
            positive_col="ci_positive",
        ),
        summarize_comparison(
            label="Larger vs smaller amount",
            predicted_side="$1,000,000",
            other_side="$100",
            source="outputs/processed/incentive_channel_data.csv",
            rows=amount,
            predicted_col="n_left_wins",
            other_col="n_right_wins",
            tie_col="n_ties",
            rate_col="left_winrate_excl_tie",
            positive_col="positive_ci_excludes_50",
        ),
    ]


def build_judging_tie_summary(cue_rows: list[dict]) -> list[dict]:
    keep = {
        "High-low utility main",
        "Same-count high-low",
        "Strong vs normal prompt",
        "Moral no-label",
    }
    return [
        {
            "comparison": row["comparison"],
            "predicted_side": row["predicted_wins"],
            "other_side": row["other_wins"],
            "ties": row["ties"],
            "non_tied_n": row["non_tied_n"],
            "total_n": row["total_n"],
        }
        for row in cue_rows
        if row["comparison"] in keep
    ]


def build_utility_replication_tables() -> tuple[list[dict], list[dict]]:
    rows = read_rows(OUT / "utility_replication_diagnostics.csv")
    holdout = []
    monotonicity = []
    for row in rows:
        holdout.append(
            {
                "actor": row["actor"],
                "religions": row["religions_holdout_accuracy"],
                "animals": row["animals_holdout_accuracy"],
                "countries": row["countries_holdout_accuracy"],
                "political": row["political_holdout_accuracy"],
            }
        )
        monotonicity.append(
            {
                "actor": row["actor"],
                "religions_entity_mean_spearman": row["religions_entity_mean_spearman"],
                "animals_entity_mean_spearman": row["animals_entity_mean_spearman"],
                "countries_entity_mean_spearman": row["countries_entity_mean_spearman"],
            }
        )
    return holdout, monotonicity


def main() -> None:
    cue_summary = build_cue_summary()
    judging_ties = build_judging_tie_summary(cue_summary)
    holdout, monotonicity = build_utility_replication_tables()

    write_rows(OUT / "cue_summary.csv", cue_summary)
    write_rows(OUT / "judging_tie_summary.csv", judging_ties)
    write_rows(OUT / "utility_replication_holdout.csv", holdout)
    write_rows(OUT / "utility_replication_monotonicity.csv", monotonicity)

    print(f"wrote {OUT / 'cue_summary.csv'}")
    print(f"wrote {OUT / 'judging_tie_summary.csv'}")
    print(f"wrote {OUT / 'utility_replication_holdout.csv'}")
    print(f"wrote {OUT / 'utility_replication_monotonicity.csv'}")


if __name__ == "__main__":
    main()
