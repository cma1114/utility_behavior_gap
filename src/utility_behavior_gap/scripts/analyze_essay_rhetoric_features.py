#!/usr/bin/env python3
"""Analyze simple rhetorical features in paired essay outputs.

This is deliberately transparent: the feature definitions are fixed regular
expressions or literal punctuation counts. They are not meant to be a complete
theory of essay quality; they are reproducible probes for differences between
the strong-system and normal-system essay outputs.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from utility_behavior_gap.io_utils import read_csv_rows, read_jsonl, write_csv_rows
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API, OUTPUT_RAW


JUDGED_PAIRS = OUTPUT_RAW / "judged_pairs.csv"
JUDGE_VOTES = OUTPUT_RAW / "judge_votes.csv"
GENERATIONS = OUTPUT_API / "generations.jsonl"

BY_PAIR_OUT = ANALYSIS / "essay_rhetoric_features_by_pair.csv"
SUMMARY_OUT = ANALYSIS / "essay_rhetoric_features_summary.csv"


WORD_RE = re.compile(r"[A-Za-z0-9']+")


def regex_counter(pattern: str) -> Callable[[str], int]:
    compiled = re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
    return lambda text: len(compiled.findall(text))


def punctuation_counter(chars: str) -> Callable[[str], int]:
    return lambda text: sum(text.count(char) for char in chars)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    definition: str
    count: Callable[[str], int]


FEATURES = [
    FeatureSpec(
        name="contrast_framing_language",
        definition=(
            "Regex count of: rather than, instead of, by contrast, on the other hand, "
            "not merely, not only, not just, more than, less than. This is the feature "
            "behind the earlier 26 strong-favored vs 9 normal-favored unanimous-pair count."
        ),
        count=regex_counter(
            r"\b("
            r"rather than|instead of|by contrast|on the other hand|"
            r"not merely|not only|not just|more than|less than"
            r")\b"
        ),
    ),
    FeatureSpec(
        name="counterargument_markers",
        definition=(
            "Regex count of: critics, opponents, some may argue, some argue, "
            "while critics, while opponents, although, though."
        ),
        count=regex_counter(
            r"\b("
            r"critics|opponents|some may argue|some argue|"
            r"while critics|while opponents|although|though"
            r")\b"
        ),
    ),
    FeatureSpec(
        name="semicolon_colon_structuring",
        definition="Literal count of semicolon and colon characters: ';' plus ':'.",
        count=punctuation_counter(";:"),
    ),
    FeatureSpec(
        name="not_but_contrast",
        definition=(
            "Regex count of 'not' followed by 'but' within 80 characters. "
            "Tracked separately from contrast_framing_language."
        ),
        count=regex_counter(r"\bnot\b.{0,80}\bbut\b"),
    ),
    FeatureSpec(
        name="example_markers",
        definition=(
            "Regex count of: for example, for instance, such as, including, "
            "a clear example, a concrete example, consider."
        ),
        count=regex_counter(
            r"\b("
            r"for example|for instance|such as|including|"
            r"a clear example|a concrete example|consider"
            r")\b"
        ),
    ),
]


def generations_by_pair(path: Path) -> dict[str, dict[str, str]]:
    by_pair: dict[str, dict[str, str]] = defaultdict(dict)
    for row in read_jsonl(path):
        condition = str(row["condition"])
        if condition in {"sys_strong", "sys_normal"}:
            by_pair[str(row["pair_uid"])][condition] = str(row["output_text"])
    return by_pair


def votes_by_pair(path: Path) -> dict[str, list[dict[str, str]]]:
    by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv_rows(path):
        by_pair[row["pair_uid"]].append(row)
    return by_pair


def direction(diff: int) -> str:
    if diff > 0:
        return "strong_gt"
    if diff < 0:
        return "normal_gt"
    return "equal"


def pair_feature_rows(
    judged_pairs: list[dict[str, str]],
    votes: dict[str, list[dict[str, str]]],
    generations: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in judged_pairs:
        pair_uid = pair["pair_uid"]
        outputs = generations.get(pair_uid, {})
        strong = outputs.get("sys_strong", "")
        normal = outputs.get("sys_normal", "")
        if not strong or not normal:
            continue

        pair_votes = votes.get(pair_uid, [])
        unanimous_strong = (
            len(pair_votes) == 3
            and all(vote.get("winner_condition") == "sys_strong" for vote in pair_votes)
        )
        row: dict[str, Any] = {
            "pair_uid": pair_uid,
            "item_label": pair["item_label"],
            "repeat": pair.get("repeat", ""),
            "panel_winner_condition": pair["panel_winner_condition"],
            "counted_winner_condition": pair["counted_winner_condition"],
            "unanimous_strong": int(unanimous_strong),
            "strong_words": word_count(strong),
            "normal_words": word_count(normal),
            "word_diff": word_count(strong) - word_count(normal),
        }
        for feature in FEATURES:
            strong_count = feature.count(strong)
            normal_count = feature.count(normal)
            diff = strong_count - normal_count
            row[f"{feature.name}_strong"] = strong_count
            row[f"{feature.name}_normal"] = normal_count
            row[f"{feature.name}_diff"] = diff
            row[f"{feature.name}_direction"] = direction(diff)
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], *, group_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature in FEATURES:
        diffs = [int(row[f"{feature.name}_diff"]) for row in rows]
        dirs = [str(row[f"{feature.name}_direction"]) for row in rows]
        out.append(
            {
                "group": group_name,
                "pairs": len(rows),
                "feature": feature.name,
                "definition": feature.definition,
                "strong_gt": dirs.count("strong_gt"),
                "equal": dirs.count("equal"),
                "normal_gt": dirs.count("normal_gt"),
                "mean_diff": round(mean(diffs), 6) if diffs else "",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judged-pairs", type=Path, default=JUDGED_PAIRS)
    parser.add_argument("--judge-votes", type=Path, default=JUDGE_VOTES)
    parser.add_argument("--generations", type=Path, default=GENERATIONS)
    parser.add_argument("--by-pair-out", type=Path, default=BY_PAIR_OUT)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_OUT)
    args = parser.parse_args()

    judged_pairs = read_csv_rows(args.judged_pairs)
    votes = votes_by_pair(args.judge_votes)
    generations = generations_by_pair(args.generations)
    by_pair_rows = pair_feature_rows(judged_pairs, votes, generations)
    if not by_pair_rows:
        raise ValueError("No paired essay feature rows were generated.")

    summary_rows: list[dict[str, Any]] = []
    summary_rows.extend(summarize(by_pair_rows, group_name="all_pairs"))
    unanimous_rows = [row for row in by_pair_rows if int(row["unanimous_strong"]) == 1]
    summary_rows.extend(summarize(unanimous_rows, group_name="unanimous_strong_pairs"))

    write_csv_rows(args.by_pair_out, by_pair_rows)
    write_csv_rows(args.summary_out, summary_rows)

    print(f"wrote {len(by_pair_rows)} pair rows to {args.by_pair_out}")
    print(f"wrote {len(summary_rows)} summary rows to {args.summary_out}")
    for row in summary_rows:
        if row["group"] == "unanimous_strong_pairs" and row["feature"] in {
            "contrast_framing_language",
            "counterargument_markers",
            "semicolon_colon_structuring",
        }:
            print(
                f"{row['feature']}: strong_gt={row['strong_gt']} "
                f"equal={row['equal']} normal_gt={row['normal_gt']} "
                f"mean_diff={row['mean_diff']}"
            )


if __name__ == "__main__":
    main()
