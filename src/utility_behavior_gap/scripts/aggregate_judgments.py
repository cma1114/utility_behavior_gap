#!/usr/bin/env python3
"""Aggregate live judge votes into pair-level judged records."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl, write_csv_rows
from utility_behavior_gap.job_builder import read_generation_jobs
from utility_behavior_gap.judging import (
    derive_counted_winner_condition,
    derive_judge_verdict,
    derive_panel_winner_condition,
)
from utility_behavior_gap.openrouter import judge_model_ids
from utility_behavior_gap.paths import OUTPUT_API, OUTPUT_RAW


JUDGE_VOTES_JSONL = OUTPUT_API / "judge_votes.jsonl"
GENERATIONS_JSONL = OUTPUT_API / "generations.jsonl"
JUDGED_PAIRS_CSV = OUTPUT_RAW / "judged_pairs.csv"
JUDGE_VOTES_CSV = OUTPUT_RAW / "judge_votes.csv"


def generation_map() -> dict[str, dict[str, Any]]:
    return {row["output_id"]: row for row in read_jsonl(GENERATIONS_JSONL)}


def current_pair_hashes(
    jobs: list[dict[str, Any]],
    generations: dict[str, dict[str, Any]],
) -> dict[str, tuple[str, str]]:
    pair_hashes: dict[str, tuple[str, str]] = {}
    for job in jobs:
        out_a = generations.get(f"{job['pair_uid']}::a")
        out_b = generations.get(f"{job['pair_uid']}::b")
        if out_a is None or out_b is None:
            continue
        pair_hashes[job["pair_uid"]] = (
            output_text_fingerprint(out_a),
            output_text_fingerprint(out_b),
        )
    return pair_hashes


def vote_matches_current_outputs(row: dict[str, Any], expected_hashes: tuple[str, str]) -> bool:
    return (
        row.get("source_output_a_hash") == expected_hashes[0]
        and row.get("source_output_b_hash") == expected_hashes[1]
    )


def votes_by_pair(
    path: Path,
    pair_hashes: dict[str, tuple[str, str]] | None = None,
    allowed_judges: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    # Deduplication is per (pair, judge, presentation orientation): with
    # both-orders judging each judge legitimately casts two votes per pair.
    legacy_latest: dict[tuple[str, str, bool], dict[str, Any]] = {}
    current_latest: dict[tuple[str, str, bool], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("success") is False:
            continue
        if allowed_judges is not None and row["judge_model"] not in allowed_judges:
            continue
        pair_uid = row["pair_uid"]
        key = (pair_uid, row["judge_model"], bool(row.get("flipped")))
        expected_hashes = pair_hashes.get(pair_uid) if pair_hashes else None
        if expected_hashes is not None and (
            "source_output_a_hash" in row or "source_output_b_hash" in row
        ):
            if vote_matches_current_outputs(row, expected_hashes):
                current_latest[key] = row
            continue
        legacy_latest[key] = row

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, row in legacy_latest.items():
        if key not in current_latest:
            by_pair[row["pair_uid"]].append(row)
    for row in current_latest.values():
        by_pair[row["pair_uid"]].append(row)
    return by_pair


def judge_verdicts(pair_votes: list[dict[str, Any]]) -> dict[str, str]:
    """Collapse a pair's votes to one verdict per judge (orders-aware)."""
    by_judge: dict[str, list[str]] = defaultdict(list)
    for row in pair_votes:
        by_judge[row["judge_model"]].append(str(row["winner_condition"]))
    return {judge: derive_judge_verdict(conds) for judge, conds in by_judge.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-votes", type=int, default=3,
        help="Minimum number of JUDGES with a verdict per pair (not raw votes).",
    )
    args = parser.parse_args()

    jobs = read_generation_jobs()
    votes = votes_by_pair(
        JUDGE_VOTES_JSONL,
        current_pair_hashes(jobs, generation_map()),
        allowed_judges=set(judge_model_ids()),
    )
    judged_rows: list[dict[str, Any]] = []
    vote_rows: list[dict[str, Any]] = []
    for job in jobs:
        pair_votes = votes.get(job["pair_uid"], [])
        verdicts = judge_verdicts(pair_votes)
        if len(verdicts) < args.min_votes:
            continue
        conditions = list(verdicts.values())
        panel = derive_panel_winner_condition(job, conditions)
        counted = derive_counted_winner_condition(job, conditions)
        judged_rows.append(
            {
                "pair_uid": job["pair_uid"],
                "comparison": job["comparison"],
                "source_run": "live_openrouter",
                "actor": job["actor"],
                "actor_label": job["actor_label"],
                "task": job["task"],
                "task_label": job["task_label"],
                "domain": job.get("domain", ""),
                "domain_label": job.get("domain_label", ""),
                "pair_idx": job.get("pair_idx", ""),
                "item_id": job["item_id"],
                "item_label": job["item_label"],
                "repeat": job.get("repeat", ""),
                "framing": job.get("framing") or job.get("sample_k", ""),
                "condition_a": job["condition_a"],
                "condition_b": job["condition_b"],
                "predicted_condition": job["predicted_condition"],
                "other_condition": job["other_condition"],
                "panel_winner_condition": panel,
                "counted_winner_condition": counted,
                "panel_winner_raw": panel,
                "high_description": job.get("high_description", ""),
                "low_description": job.get("low_description", ""),
                "high_consequence": job.get("high_consequence", ""),
                "low_consequence": job.get("low_consequence", ""),
                "high_utility": job.get("high_utility", ""),
                "low_utility": job.get("low_utility", ""),
                "delta_u": job.get("delta_u", ""),
                "cause_pair_label": job.get("cause_pair_label", ""),
                "good_text": job.get("good_text", ""),
                "bad_text": job.get("bad_text", ""),
                "amount_high": job.get("amount_high", ""),
                "amount_low": job.get("amount_low", ""),
                "outcome": job.get("outcome", ""),
                "source_trial_index": job.get("source_trial_index", ""),
                "source_note": job.get("source_note", ""),
                "source_outcome_index": job.get("source_outcome_index", ""),
                "source_topic_index": job.get("source_topic_index", ""),
                "topic_design": job.get("topic_design", ""),
                "gap_bin": job.get("gap_bin", ""),
                "gap_bin_count": job.get("gap_bin_count", ""),
                "gap_bin_min_delta_u": job.get("gap_bin_min_delta_u", ""),
                "gap_bin_max_delta_u": job.get("gap_bin_max_delta_u", ""),
                "gap_bin_sample_index": job.get("gap_bin_sample_index", ""),
                "gap_sampling_seed": job.get("gap_sampling_seed", ""),
            }
        )
        for vote in pair_votes:
            vote_rows.append(
                {
                    "pair_uid": vote["pair_uid"],
                    "judge_index": vote["judge_index"],
                    "judge_model": vote["judge_model"],
                    "flipped": vote.get("flipped", ""),
                    "vote_raw": vote["vote_raw"],
                    "winner_condition": vote["winner_condition"],
                    "judge_verdict": verdicts.get(vote["judge_model"], ""),
                    "success": vote["success"],
                    "source_output_a_hash": vote.get("source_output_a_hash", ""),
                    "source_output_b_hash": vote.get("source_output_b_hash", ""),
                }
            )

    if not judged_rows:
        raise ValueError("No judged pairs were aggregated. Run generation and judging first.")
    write_csv_rows(JUDGED_PAIRS_CSV, judged_rows)
    write_csv_rows(JUDGE_VOTES_CSV, vote_rows)
    print(f"wrote {len(judged_rows)} judged pairs to {JUDGED_PAIRS_CSV}")
    print(f"wrote {len(vote_rows)} judge votes to {JUDGE_VOTES_CSV}")


if __name__ == "__main__":
    main()
