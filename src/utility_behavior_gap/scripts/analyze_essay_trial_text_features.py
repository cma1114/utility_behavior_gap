#!/usr/bin/env python3
"""Mechanical text-feature analysis for trial-level essay pairs.

The goal is diagnostic, not causal: identify reproducible surface features that
move with condition arms or judge choices in the essay trial-level data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from utility_behavior_gap.paths import ANALYSIS, ROOT


ESSAY_DATA = ROOT / "essay_all_conditions"
BY_PAIR_CSV = ANALYSIS / "essay_trial_text_features_by_pair.csv"
SUMMARY_CSV = ANALYSIS / "essay_trial_text_features_summary.csv"
TOP_CSV = ANALYSIS / "essay_trial_text_features_top_deltas.csv"
NOTE_MD = ROOT / "notes" / "essay_trial_text_feature_analysis.md"

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SENTENCE_RE = re.compile(r"[.!?]+(?:\s+|$)")
NUMERIC_RE = re.compile(r"(?:\$|#)?\b\d[\d,]*(?:\.\d+)?%?\b")


def regex_counter(pattern: str) -> Callable[[str], int]:
    compiled = re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
    return lambda text: len(compiled.findall(text))


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def word_count(text: str) -> int:
    return len(words(text))


def sentence_count(text: str) -> int:
    count = len(SENTENCE_RE.findall(text))
    return max(1, count) if text.strip() else 0


def paragraph_count(text: str) -> int:
    return sum(1 for para in re.split(r"\n\s*\n+", text.strip()) if para.strip())


def per_1000_words(text: str, count_fn: Callable[[str], int]) -> float:
    n_words = word_count(text)
    if n_words == 0:
        return 0.0
    return count_fn(text) * 1000.0 / n_words


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    definition: str
    value: Callable[[str], float]
    unit: str


CONTRAST_COUNT = regex_counter(
    r"\b("
    r"rather than|instead of|by contrast|on the other hand|"
    r"not merely|not only|not just|more than|less than|"
    r"not\b.{0,80}\bbut"
    r")\b"
)

COUNTERARGUMENT_COUNT = regex_counter(
    r"\b("
    r"critics|opponents|some may argue|some argue|"
    r"while critics|while opponents|although|though|objection|objections"
    r")\b"
)

EXAMPLE_COUNT = regex_counter(
    r"\b("
    r"for example|for instance|such as|including|"
    r"a clear example|a concrete example|consider"
    r")\b"
)

TRANSITION_COUNT = regex_counter(
    r"\b("
    r"therefore|consequently|as a result|in turn|ultimately|"
    r"in conclusion|in short|moreover|furthermore|however"
    r")\b"
)

QUALIFICATION_COUNT = regex_counter(
    r"\b("
    r"may|might|could|perhaps|likely|often|sometimes|in some cases|"
    r"to some extent|nuance|nuanced"
    r")\b"
)


FEATURES = [
    FeatureSpec(
        "word_count",
        "Total regex word tokens.",
        lambda text: float(word_count(text)),
        "words",
    ),
    FeatureSpec(
        "paragraph_count",
        "Number of non-empty paragraph blocks split on blank lines.",
        lambda text: float(paragraph_count(text)),
        "paragraphs",
    ),
    FeatureSpec(
        "avg_sentence_words",
        "Word count divided by rough sentence count.",
        lambda text: word_count(text) / sentence_count(text) if sentence_count(text) else 0.0,
        "words_per_sentence",
    ),
    FeatureSpec(
        "unique_word_rate",
        "Unique lowercased word types divided by total word tokens.",
        lambda text: (len({w.lower() for w in words(text)}) / word_count(text)) if word_count(text) else 0.0,
        "proportion",
    ),
    FeatureSpec(
        "semicolon_colon_per_1k",
        "Semicolon plus colon characters per 1,000 words.",
        lambda text: per_1000_words(text, lambda value: value.count(";") + value.count(":")),
        "per_1000_words",
    ),
    FeatureSpec(
        "dash_per_1k",
        "Hyphen, en dash, and em dash characters per 1,000 words.",
        lambda text: per_1000_words(
            text,
            lambda value: value.count("-") + value.count("\u2013") + value.count("\u2014"),
        ),
        "per_1000_words",
    ),
    FeatureSpec(
        "contrast_framing_per_1k",
        "Contrast/framing markers per 1,000 words: rather than, instead of, by contrast, on the other hand, not merely, not only, not just, more than, less than, and not...but.",
        lambda text: per_1000_words(text, CONTRAST_COUNT),
        "per_1000_words",
    ),
    FeatureSpec(
        "counterargument_per_1k",
        "Counterargument markers per 1,000 words: critics, opponents, some may argue, while critics/opponents, although, though, objection(s).",
        lambda text: per_1000_words(text, COUNTERARGUMENT_COUNT),
        "per_1000_words",
    ),
    FeatureSpec(
        "example_marker_per_1k",
        "Example/detail markers per 1,000 words: for example, for instance, such as, including, concrete example, consider.",
        lambda text: per_1000_words(text, EXAMPLE_COUNT),
        "per_1000_words",
    ),
    FeatureSpec(
        "transition_marker_per_1k",
        "Reasoning/transition markers per 1,000 words: therefore, consequently, as a result, ultimately, moreover, furthermore, however, and related phrases.",
        lambda text: per_1000_words(text, TRANSITION_COUNT),
        "per_1000_words",
    ),
    FeatureSpec(
        "numeric_specificity_per_1k",
        "Numeric tokens per 1,000 words, including percentages and currency-like numbers.",
        lambda text: per_1000_words(text, lambda value: len(NUMERIC_RE.findall(value))),
        "per_1000_words",
    ),
    FeatureSpec(
        "qualification_per_1k",
        "Qualification/modality markers per 1,000 words: may, might, could, perhaps, likely, often, sometimes, in some cases, to some extent, nuance(d).",
        lambda text: per_1000_words(text, QUALIFICATION_COUNT),
        "per_1000_words",
    ),
]


def read_trials(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*/*.json")):
        condition = path.parent.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        for idx, trial in enumerate(payload["trials"]):
            row = dict(trial)
            row["condition"] = condition
            row["actor_id"] = path.stem
            row["trial_index"] = idx
            rows.append(row)
    return rows


def feature_values(text: str) -> dict[str, float]:
    return {feature.name: feature.value(text) for feature in FEATURES}


def outcome_label(winner_arm: str) -> str:
    if winner_arm == "A":
        return "A"
    if winner_arm == "B":
        return "B"
    return winner_arm


def paired_feature_rows(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        values_a = feature_values(str(trial.get("essay_A", "")))
        values_b = feature_values(str(trial.get("essay_B", "")))
        winner = outcome_label(str(trial.get("winner_arm", "")))
        out: dict[str, Any] = {
            "condition": trial["condition"],
            "actor": trial["actor"],
            "actor_id": trial["actor_id"],
            "trial_index": trial["trial_index"],
            "domain": trial.get("domain") or "",
            "essay_topic": trial.get("essay_topic") or "",
            "framing": trial.get("framing") or "",
            "arm_A": trial.get("arm_A") or "",
            "arm_B": trial.get("arm_B") or "",
            "winner_arm": winner,
        }
        for feature in FEATURES:
            a_value = values_a[feature.name]
            b_value = values_b[feature.name]
            delta = a_value - b_value
            if winner == "A":
                winner_minus_loser = delta
            elif winner == "B":
                winner_minus_loser = -delta
            else:
                winner_minus_loser = ""
            out[f"{feature.name}_A"] = round(a_value, 6)
            out[f"{feature.name}_B"] = round(b_value, 6)
            out[f"{feature.name}_A_minus_B"] = round(delta, 6)
            out[f"{feature.name}_winner_minus_loser"] = (
                round(winner_minus_loser, 6) if winner_minus_loser != "" else ""
            )
        rows.append(out)
    return rows


def safe_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def standardized_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    sd = safe_stdev(values)
    return mean(values) / sd if sd else 0.0


def gt_counts(values: list[float]) -> tuple[int, int, int]:
    greater = sum(1 for value in values if value > 0)
    equal = sum(1 for value in values if value == 0)
    lesser = sum(1 for value in values if value < 0)
    return greater, equal, lesser


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    conditions = sorted({str(row["condition"]) for row in rows})
    for condition in conditions:
        sub = [row for row in rows if row["condition"] == condition]
        resolved = [row for row in sub if row["winner_arm"] in {"A", "B"}]
        counts = Counter(str(row["winner_arm"]) for row in sub)
        for feature in FEATURES:
            arm_deltas = [float(row[f"{feature.name}_A_minus_B"]) for row in sub]
            winner_deltas = [
                float(row[f"{feature.name}_winner_minus_loser"])
                for row in resolved
                if row[f"{feature.name}_winner_minus_loser"] != ""
            ]
            a_gt_b, equal, b_gt_a = gt_counts(arm_deltas)
            winner_gt_loser, winner_equal, loser_gt_winner = gt_counts(winner_deltas)
            out.append(
                {
                    "condition": condition,
                    "feature": feature.name,
                    "unit": feature.unit,
                    "definition": feature.definition,
                    "n_pairs": len(sub),
                    "n_resolved": len(resolved),
                    "wins_A": counts["A"],
                    "wins_B": counts["B"],
                    "ties": counts["TIE"],
                    "disagreements": counts["disagree"],
                    "mean_A": round(mean(float(row[f"{feature.name}_A"]) for row in sub), 6),
                    "mean_B": round(mean(float(row[f"{feature.name}_B"]) for row in sub), 6),
                    "mean_A_minus_B": round(mean(arm_deltas), 6),
                    "median_A_minus_B": round(median(arm_deltas), 6),
                    "std_A_minus_B": round(standardized_mean(arm_deltas), 6),
                    "A_gt_B": a_gt_b,
                    "A_eq_B": equal,
                    "B_gt_A": b_gt_a,
                    "mean_winner_minus_loser": round(mean(winner_deltas), 6),
                    "median_winner_minus_loser": round(median(winner_deltas), 6),
                    "std_winner_minus_loser": round(standardized_mean(winner_deltas), 6),
                    "winner_gt_loser": winner_gt_loser,
                    "winner_eq_loser": winner_equal,
                    "loser_gt_winner": loser_gt_winner,
                }
            )
    return out


def top_rows(summary_rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in sorted({str(row["condition"]) for row in summary_rows}):
        sub = [row for row in summary_rows if row["condition"] == condition]
        for metric in ["std_A_minus_B", "std_winner_minus_loser"]:
            ranked = sorted(sub, key=lambda row: abs(float(row[metric])), reverse=True)
            for rank, row in enumerate(ranked[:top_n], start=1):
                rows.append(
                    {
                        "condition": condition,
                        "ranking": metric,
                        "rank": rank,
                        "feature": row["feature"],
                        "unit": row["unit"],
                        "standardized_mean": row[metric],
                        "mean_A_minus_B": row["mean_A_minus_B"],
                        "mean_winner_minus_loser": row["mean_winner_minus_loser"],
                        "A_gt_B": row["A_gt_B"],
                        "B_gt_A": row["B_gt_A"],
                        "winner_gt_loser": row["winner_gt_loser"],
                        "loser_gt_winner": row["loser_gt_winner"],
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: int, denom: int) -> str:
    return f"{100 * value / denom:.1f}%" if denom else ""


def feature_lookup(summary_rows: list[dict[str, Any]], condition: str, feature: str) -> dict[str, Any]:
    for row in summary_rows:
        if row["condition"] == condition and row["feature"] == feature:
            return row
    raise KeyError((condition, feature))


def feature_contingency(
    by_pair: list[dict[str, Any]],
    condition: str,
    feature: str,
) -> dict[str, int]:
    out: Counter[str] = Counter()
    for row in by_pair:
        if row["condition"] != condition or row["winner_arm"] not in {"A", "B"}:
            continue
        delta = float(row[f"{feature}_A_minus_B"])
        if delta > 0:
            sign = "A_gt_B"
        elif delta < 0:
            sign = "B_gt_A"
        else:
            sign = "equal"
        out[f"{sign}_winner_{row['winner_arm']}"] += 1
    return dict(out)


def markdown_note(
    summary_rows: list[dict[str, Any]],
    top: list[dict[str, Any]],
    by_pair: list[dict[str, Any]],
) -> str:
    lines = [
        "# Essay Trial Text Feature Analysis",
        "",
        "This note summarizes mechanical text features computed from `essay_all_conditions/<condition>/<actor>.json`.",
        "The analysis compares arm A against arm B within each pair, and separately compares the judge-panel winner against the loser for resolved A/B pairs.",
        "These features are diagnostic surface probes, not a causal model of essay quality.",
        "",
        "## Features",
        "",
    ]
    for feature in FEATURES:
        lines.append(f"- `{feature.name}` ({feature.unit}): {feature.definition}")
    lines.extend(
        [
            "",
            "## Main Pattern",
            "",
            "The strongest judge-choice correlate across conditions is usually `word_count`: resolved winners are longer than losers.",
            "Length is therefore a plausible proximal reason judges prefer one essay over another, but it is not an explanation of why a condition arm shifts quality unless the arm itself also changes length or other features.",
            "",
            "For the high-low condition, arm A is the high-utility side. It does not show a clear positive shift on the features judges tend to reward.",
            "",
            "## Selected Rows",
            "",
            "| Condition | Feature | Mean A-B | A>B / B>A | Mean winner-loser | Winner>loser / Loser>winner |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    selected = [
        ("highlow", "word_count"),
        ("highlow", "paragraph_count"),
        ("highlow", "contrast_framing_per_1k"),
        ("highlow", "counterargument_per_1k"),
        ("direct", "word_count"),
        ("direct", "contrast_framing_per_1k"),
        ("moral", "word_count"),
        ("amount", "word_count"),
    ]
    for condition, feature in selected:
        row = feature_lookup(summary_rows, condition, feature)
        lines.append(
            "| {condition} | `{feature}` | {mean_ab} | {a_gt}/{b_gt} | {mean_win} | {w_gt}/{l_gt} |".format(
                condition=condition,
                feature=feature,
                mean_ab=row["mean_A_minus_B"],
                a_gt=row["A_gt_B"],
                b_gt=row["B_gt_A"],
                mean_win=row["mean_winner_minus_loser"],
                w_gt=row["winner_gt_loser"],
                l_gt=row["loser_gt_winner"],
            )
        )

    lines.extend(
        [
            "",
            "## Length Contingency",
            "",
            "For resolved A/B pairs, this table asks whether arm A won more often when arm A was longer, and whether arm B won more often when arm B was longer.",
            "",
            "| Condition | A longer: A wins / B wins | B longer: A wins / B wins | Equal length: A wins / B wins |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in ["direct", "highlow", "moral", "amount"]:
        counts = feature_contingency(by_pair, condition, "word_count")
        lines.append(
            "| {condition} | {agta} / {agtb} | {bgta} / {bgtb} | {eqa} / {eqb} |".format(
                condition=condition,
                agta=counts.get("A_gt_B_winner_A", 0),
                agtb=counts.get("A_gt_B_winner_B", 0),
                bgta=counts.get("B_gt_A_winner_A", 0),
                bgtb=counts.get("B_gt_A_winner_B", 0),
                eqa=counts.get("equal_winner_A", 0),
                eqb=counts.get("equal_winner_B", 0),
            )
        )

    lines.extend(
        [
            "",
            "## Top Standardized Deltas",
            "",
            "`std_A_minus_B` ranks features by how much the condition arm A differs from arm B.",
            "`std_winner_minus_loser` ranks features by how much the panel winner differs from the loser.",
            "The CSV `outputs/analysis/essay_trial_text_features_top_deltas.csv` contains the full top-feature list.",
            "",
            "## Output Files",
            "",
            "- `outputs/analysis/essay_trial_text_features_by_pair.csv`",
            "- `outputs/analysis/essay_trial_text_features_summary.csv`",
            "- `outputs/analysis/essay_trial_text_features_top_deltas.csv`",
            "",
            "## Command",
            "",
            "```bash",
            "python -m utility_behavior_gap.scripts.analyze_essay_trial_text_features",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def write_note(
    path: Path,
    summary_rows: list[dict[str, Any]],
    top: list[dict[str, Any]],
    by_pair: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_note(summary_rows, top, by_pair), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ESSAY_DATA)
    parser.add_argument("--by-pair-out", type=Path, default=BY_PAIR_CSV)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--top-out", type=Path, default=TOP_CSV)
    parser.add_argument("--note-out", type=Path, default=NOTE_MD)
    parser.add_argument("--top-n", type=int, default=6)
    args = parser.parse_args()

    trials = read_trials(args.data_dir)
    by_pair = paired_feature_rows(trials)
    summary_rows = summarize(by_pair)
    top = top_rows(summary_rows, args.top_n)

    write_csv(args.by_pair_out, by_pair)
    write_csv(args.summary_out, summary_rows)
    write_csv(args.top_out, top)
    write_note(args.note_out, summary_rows, top, by_pair)

    print(f"wrote {len(by_pair)} pair rows to {args.by_pair_out}")
    print(f"wrote {len(summary_rows)} summary rows to {args.summary_out}")
    print(f"wrote {len(top)} top-delta rows to {args.top_out}")
    print(f"wrote note to {args.note_out}")

    for condition in ["direct", "highlow", "moral", "amount"]:
        word = feature_lookup(summary_rows, condition, "word_count")
        contrast = feature_lookup(summary_rows, condition, "contrast_framing_per_1k")
        print(
            f"{condition}: word A-B={word['mean_A_minus_B']}, "
            f"winner-loser={word['mean_winner_minus_loser']}; "
            f"contrast A-B={contrast['mean_A_minus_B']}, "
            f"winner-loser={contrast['mean_winner_minus_loser']}"
        )


if __name__ == "__main__":
    main()
