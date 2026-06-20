#!/usr/bin/env python3
"""Show compact status for the current live OpenRouter manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from utility_behavior_gap.openrouter import judge_model_ids
from utility_behavior_gap.paths import OUTPUT_API


GENERATION_JOBS = OUTPUT_API / "generation_jobs.jsonl"
GENERATIONS = OUTPUT_API / "generations.jsonl"
JUDGE_VOTES = OUTPUT_API / "judge_votes.jsonl"
GENERATION_FAILURES = OUTPUT_API / "generation_failures.jsonl"


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_log_path(jobs: list[dict[str, Any]], field: str, fallback: Path) -> Path:
    values = {str(job.get(field) or "") for job in jobs}
    values.discard("")
    if len(values) == 1:
        return Path(values.pop())
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory to inspect instead of current manifest.")
    parser.add_argument(
        "--orders",
        choices=["single", "both"],
        default="single",
        help="Use both for runs judged with run_judging --orders both.",
    )
    args = parser.parse_args()

    jobs_path = args.run_dir / "generation_jobs.jsonl" if args.run_dir else GENERATION_JOBS
    jobs = read_jsonl_if_exists(jobs_path)
    if not jobs:
        print(f"no generation manifest found at {jobs_path}")
        return

    if args.run_dir:
        run_generations = args.run_dir / "generations.jsonl"
        run_generation_failures = args.run_dir / "generation_failures.jsonl"
        run_judge_votes = args.run_dir / "judge_votes.jsonl"
    else:
        run_generations = run_log_path(jobs, "run_generations_path", GENERATIONS)
        run_generation_failures = run_log_path(jobs, "run_generation_failures_path", GENERATION_FAILURES)
        run_judge_votes = run_log_path(jobs, "run_judge_votes_path", JUDGE_VOTES)

    pair_uids = {str(job["pair_uid"]) for job in jobs}
    expected_output_ids = {f"{pair_uid}::{suffix}" for pair_uid in pair_uids for suffix in ("a", "b")}
    judges = judge_model_ids()

    generations = [
        row for row in read_jsonl_if_exists(run_generations) if str(row.get("output_id", "")) in expected_output_ids
    ]
    valid_generation_ids = {
        str(row["output_id"])
        for row in generations
        if row.get("success") is not False and str(row.get("output_text", "")).strip()
    }
    unresolved_generation_failures = [
        row
        for row in read_jsonl_if_exists(run_generation_failures)
        if str(row.get("output_id", "")) in expected_output_ids
    ]

    votes = [row for row in read_jsonl_if_exists(run_judge_votes) if str(row.get("pair_uid", "")) in pair_uids]
    latest_successful_vote_by_key: dict[tuple[str, str, bool], dict[str, Any]] = {}
    failed_vote_keys: set[tuple[str, str, bool]] = set()
    for row in votes:
        judge_model = str(row.get("judge_model", ""))
        if judge_model not in judges:
            continue
        flipped = bool(row.get("flipped")) if args.orders == "both" else False
        key = (str(row["pair_uid"]), judge_model, flipped)
        if row.get("success") is False:
            failed_vote_keys.add(key)
        else:
            latest_successful_vote_by_key[key] = row
    successful_vote_keys = set(latest_successful_vote_by_key)
    unresolved_failed_vote_keys = failed_vote_keys - successful_vote_keys
    vote_split = Counter(str(row.get("winner_condition", "")) for row in latest_successful_vote_by_key.values())
    unresolved_generation_failures = [
        row for row in unresolved_generation_failures if str(row.get("output_id", "")) not in valid_generation_ids
    ]

    complete_pairs = 0
    pairs_with_generations = 0
    pair_vote_counts: Counter[int] = Counter()
    expected_votes_per_pair = len(judges) * (2 if args.orders == "both" else 1)
    by_pair_votes: dict[str, set[tuple[str, bool]]] = defaultdict(set)
    for pair_uid, judge_model, flipped in successful_vote_keys:
        by_pair_votes[pair_uid].add((judge_model, flipped))
    for pair_uid in pair_uids:
        has_generations = all(f"{pair_uid}::{suffix}" in valid_generation_ids for suffix in ("a", "b"))
        if has_generations:
            pairs_with_generations += 1
        vote_count = len(by_pair_votes[pair_uid])
        pair_vote_counts[vote_count] += 1
        if has_generations and vote_count == expected_votes_per_pair:
            complete_pairs += 1

    first_job = jobs[0]
    print(f"manifest:    {len(jobs)} pairs, {first_job.get('comparison', '')}, {first_job.get('actor', '')}")
    if first_job.get("run_id"):
        print(f"run_id:      {first_job['run_id']}")
        print(f"run_dir:     {first_job.get('run_dir', '')}")
    print(f"generations: {len(valid_generation_ids)}/{len(expected_output_ids)} outputs complete")
    print(f"judging:     {len(successful_vote_keys)}/{len(pair_uids) * expected_votes_per_pair} votes complete")
    print(f"pairs:       {complete_pairs}/{len(pair_uids)} complete; {pairs_with_generations}/{len(pair_uids)} have both outputs")
    print(f"vote split:  {dict(vote_split)}")
    incomplete_pairs = sum(count for votes_done, count in pair_vote_counts.items() if votes_done < expected_votes_per_pair)
    if unresolved_generation_failures or unresolved_failed_vote_keys or incomplete_pairs:
        print(
            "issues:      "
            f"{len(unresolved_generation_failures)} generation failures, "
            f"{len(unresolved_failed_vote_keys)} failed judge calls, "
            f"{incomplete_pairs} incomplete pairs"
        )


if __name__ == "__main__":
    main()
