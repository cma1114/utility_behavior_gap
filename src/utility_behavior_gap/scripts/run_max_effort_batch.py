#!/usr/bin/env python3
"""Run the 200-pair user-prompt max-effort essay contrast for multiple actors."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from utility_behavior_gap.constants import ACTORS
from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.job_builder import build_generation_jobs
from utility_behavior_gap.openrouter import judge_model_ids
from utility_behavior_gap.paths import OUTPUT_API, OUTPUT_RAW, ROOT


DEFAULT_COMPARISON = "essay_direct_user_prompt_max_effort_full_topics"
CURRENT_GENERATION_JOBS = OUTPUT_API / "generation_jobs.jsonl"
RUNS_DIR = OUTPUT_API / "runs"
GENERATIONS_JSONL = OUTPUT_API / "generations.jsonl"
JUDGE_VOTES_JSONL = OUTPUT_API / "judge_votes.jsonl"
JUDGED_PAIRS_CSV = OUTPUT_RAW / "judged_pairs.csv"
JUDGE_VOTES_CSV = OUTPUT_RAW / "judge_votes.csv"


def parse_actors(value: str) -> list[str]:
    if not value:
        return []
    return [actor.strip() for actor in value.split(",") if actor.strip()]


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def run_actor_step(actor: str, step: str, command: list[str]) -> bool:
    try:
        run(command)
    except subprocess.CalledProcessError as exc:
        print(
            f"error: {step} failed for {actor} with exit code {exc.returncode}; "
            "skipping this actor and continuing",
            flush=True,
        )
        return False
    return True


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def valid_generation_row(row: dict[str, Any]) -> bool:
    return row.get("success") is not False and bool(str(row.get("output_text", "")).strip())


def valid_generation_map(paths: list[Path] | None = None) -> dict[str, dict[str, Any]]:
    paths = paths or [GENERATIONS_JSONL]
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl_if_exists(path):
            if valid_generation_row(row):
                out[str(row["output_id"])] = row
    return out


def jobs_from_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    return read_jsonl(manifest_path)


def latest_manifest_path(actor: str, comparison: str) -> Path | None:
    if not RUNS_DIR.exists():
        return None
    candidates: list[Path] = []
    for manifest in RUNS_DIR.glob("*/generation_jobs.jsonl"):
        try:
            jobs = jobs_from_manifest(manifest)
        except Exception:
            continue
        if not jobs:
            continue
        actors = {str(job.get("actor", "")) for job in jobs}
        comparisons = {str(job.get("comparison", "")) for job in jobs}
        if actors == {actor} and comparisons == {comparison}:
            candidates.append(manifest)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def current_manifest_path() -> Path:
    jobs = read_jsonl(CURRENT_GENERATION_JOBS)
    values = {str(job.get("run_manifest_path") or "") for job in jobs}
    values.discard("")
    if len(values) == 1:
        return Path(values.pop())
    return CURRENT_GENERATION_JOBS


def activate_manifest(manifest_path: Path) -> None:
    shutil.copy2(manifest_path, CURRENT_GENERATION_JOBS)


def jobs_for(actor: str, comparison: str, manifest_path: Path | None = None) -> list[dict[str, Any]]:
    if manifest_path is not None:
        return jobs_from_manifest(manifest_path)
    return build_generation_jobs(comparisons={comparison}, tasks={"essay"}, actors={actor})


def run_log_path(jobs: list[dict[str, Any]], field: str, fallback: Path) -> Path:
    values = {str(job.get(field) or "") for job in jobs}
    values.discard("")
    if len(values) == 1:
        return Path(values.pop())
    return fallback


def run_specific_paths(jobs: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    return (
        run_log_path(jobs, "run_generations_path", GENERATIONS_JSONL),
        run_log_path(jobs, "run_generation_failures_path", OUTPUT_API / "generation_failures.jsonl"),
        run_log_path(jobs, "run_judge_votes_path", JUDGE_VOTES_JSONL),
    )


def unresolved_generation_failures(jobs: list[dict[str, Any]], generations: dict[str, dict[str, Any]]) -> int:
    _, failures_path, _ = run_specific_paths(jobs)
    expected_outputs = {f"{job['pair_uid']}::{suffix}" for job in jobs for suffix in ("a", "b")}
    failures = [
        row
        for row in read_jsonl_if_exists(failures_path)
        if str(row.get("output_id", "")) in expected_outputs
    ]
    return sum(1 for row in failures if str(row.get("output_id", "")) not in generations)


def latest_current_votes(
    pair_hashes: dict[str, tuple[str, str]],
    paths: list[Path] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    paths = paths or [JUDGE_VOTES_JSONL]
    if not paths:
        return {}
    judges = set(judge_model_ids())
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl_if_exists(path):
            pair_uid = str(row.get("pair_uid", ""))
            judge_model = str(row.get("judge_model", ""))
            if judge_model not in judges or row.get("success") is False:
                continue
            expected = pair_hashes.get(pair_uid)
            if expected is None:
                continue
            if (
                row.get("source_output_a_hash") != expected[0]
                or row.get("source_output_b_hash") != expected[1]
            ):
                continue
            out[(pair_uid, judge_model)] = row
    return out


def completion_counts(
    actor: str,
    comparison: str,
    manifest_path: Path | None = None,
) -> tuple[int, int, int, int, int, int, int]:
    jobs = jobs_for(actor, comparison, manifest_path)
    run_generations, _, run_votes = run_specific_paths(jobs)
    generation_paths = [run_generations] if manifest_path is not None else [GENERATIONS_JSONL]
    vote_paths = [run_votes] if manifest_path is not None else [JUDGE_VOTES_JSONL]
    generations = valid_generation_map(generation_paths)
    expected_outputs = {f"{job['pair_uid']}::{suffix}" for job in jobs for suffix in ("a", "b")}
    complete_outputs = sum(1 for output_id in expected_outputs if output_id in generations)
    unresolved_failures = unresolved_generation_failures(jobs, generations) if manifest_path is not None else 0

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

    current_votes = latest_current_votes(pair_hashes, vote_paths)
    expected_votes = len(jobs) * len(judge_model_ids())
    complete_votes = len(current_votes)
    complete_pairs = 0
    for job in jobs:
        pair_uid = job["pair_uid"]
        has_outputs = all(f"{pair_uid}::{suffix}" in generations for suffix in ("a", "b"))
        has_votes = sum((pair_uid, judge) in current_votes for judge in judge_model_ids()) == len(judge_model_ids())
        if has_outputs and has_votes:
            complete_pairs += 1
    return complete_outputs, len(expected_outputs), complete_votes, expected_votes, complete_pairs, len(jobs), unresolved_failures


def is_complete(actor: str, comparison: str, manifest_path: Path | None = None) -> bool:
    complete_outputs, expected_outputs, complete_votes, expected_votes, complete_pairs, expected_pairs, _ = (
        completion_counts(actor, comparison, manifest_path)
    )
    return (
        complete_outputs == expected_outputs
        and complete_votes == expected_votes
        and complete_pairs == expected_pairs
    )


def snapshot_outputs(actor: str, comparison: str) -> None:
    if not JUDGED_PAIRS_CSV.exists() or not JUDGE_VOTES_CSV.exists():
        return
    prefix = f"{comparison}__{actor}"
    shutil.copy2(JUDGED_PAIRS_CSV, OUTPUT_RAW / f"{prefix}__judged_pairs.csv")
    shutil.copy2(JUDGE_VOTES_CSV, OUTPUT_RAW / f"{prefix}__judge_votes.csv")


def actor_status_line(actor: str, comparison: str, manifest_path: Path | None = None) -> str:
    manifest_path = manifest_path or latest_manifest_path(actor, comparison)
    complete_outputs, expected_outputs, complete_votes, expected_votes, complete_pairs, expected_pairs, failures = (
        completion_counts(actor, comparison, manifest_path)
    )
    run_part = ""
    if manifest_path is not None:
        run_part = f", run {manifest_path.parent.name}"
    failure_part = f", unresolved generation failures {failures}" if failures else ""
    return (
        f"{actor}: generations {complete_outputs}/{expected_outputs}, "
        f"votes {complete_votes}/{expected_votes}, pairs {complete_pairs}/{expected_pairs}"
        f"{failure_part}{run_part}"
    )


def print_batch_status(actors: list[str], comparison: str) -> None:
    print(f"comparison: {comparison}")
    for actor in actors:
        print(actor_status_line(actor, comparison))


def prepare_or_resume_actor(actor: str, comparison: str, *, fresh: bool) -> Path | None:
    existing = None if fresh else latest_manifest_path(actor, comparison)
    if existing is not None and not is_complete(actor, comparison, existing):
        activate_manifest(existing)
        print(f"resuming existing run: {existing.parent}", flush=True)
        return existing
    if existing is not None and is_complete(actor, comparison, existing):
        activate_manifest(existing)
        print(f"using complete existing run: {existing.parent}", flush=True)
        return existing
    prepared = run_actor_step(
        actor,
        "prepare_generation_jobs",
        [
            sys.executable,
            "-m",
            "utility_behavior_gap.scripts.prepare_generation_jobs",
            "--comparisons",
            comparison,
            "--tasks",
            "essay",
            "--actors",
            actor,
        ],
    )
    if not prepared:
        return None
    return current_manifest_path()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default=DEFAULT_COMPARISON)
    parser.add_argument(
        "--actors",
        default="",
        help="Comma-separated actor ids. Default: all incomplete actors on the roster.",
    )
    parser.add_argument(
        "--include-completed",
        action="store_true",
        help="Do not skip actors that already have complete generations and judge votes.",
    )
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--generation-retries", type=int, default=3)
    parser.add_argument(
        "--retry-sleep-s",
        type=float,
        default=60.0,
        help="Seconds to wait before retrying missing generations or judge votes.",
    )
    parser.add_argument("--generation-limit", type=int, default=None)
    parser.add_argument("--judging-limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Prepare a new run directory instead of resuming the latest incomplete run for each actor.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show per-actor progress for the latest matching runs and exit without API calls.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print which actors would run, without making API calls.",
    )
    args = parser.parse_args()

    actors = parse_actors(args.actors) or list(ACTORS)
    unknown = sorted(set(actors) - set(ACTORS))
    if unknown:
        raise ValueError(f"unknown actor ids: {', '.join(unknown)}")

    if args.status:
        print_batch_status(actors, args.comparison)
        return

    if not args.include_completed:
        actors = [
            actor
            for actor in actors
            if not is_complete(actor, args.comparison, latest_manifest_path(actor, args.comparison))
        ]

    if not actors:
        print("all requested actors are already complete")
        return

    print("batch actors:")
    for actor in actors:
        print(f"  {actor} ({actor_status_line(actor, args.comparison)})")
    if args.plan_only:
        return

    failed_actors: list[tuple[str, str]] = []
    for actor in actors:
        print(f"\n=== {actor} ===", flush=True)
        manifest_path = prepare_or_resume_actor(actor, args.comparison, fresh=args.fresh)
        if manifest_path is None:
            failed_actors.append((actor, "prepare_generation_jobs"))
            continue
        print(actor_status_line(actor, args.comparison, manifest_path), flush=True)

        generation_args = [sys.executable, "-m", "utility_behavior_gap.scripts.run_generation"]
        if args.dry_run:
            generation_args.append("--dry-run")
        if args.generation_limit is not None:
            generation_args += ["--limit", str(args.generation_limit)]
        for attempt in range(args.generation_retries + 1):
            complete_outputs, expected_outputs, *_ = completion_counts(actor, args.comparison, manifest_path)
            if complete_outputs == expected_outputs:
                break
            if attempt:
                print(
                    f"retrying missing generations for {actor} "
                    f"({attempt}/{args.generation_retries})",
                    flush=True,
                )
                if args.retry_sleep_s > 0:
                    time.sleep(args.retry_sleep_s)
            generated = run_actor_step(actor, "run_generation", generation_args)
            if not generated:
                failed_actors.append((actor, "run_generation"))
                break
            print(actor_status_line(actor, args.comparison, manifest_path), flush=True)

        complete_outputs, expected_outputs, *_ = completion_counts(actor, args.comparison, manifest_path)
        if complete_outputs != expected_outputs:
            print(
                f"warning: skipping judging for {actor}; generations are still incomplete "
                f"({complete_outputs}/{expected_outputs}). Re-run the batch to resume.",
                flush=True,
            )
            continue

        judging_args = [sys.executable, "-m", "utility_behavior_gap.scripts.run_judging"]
        if args.dry_run:
            judging_args.append("--dry-run")
        if args.judging_limit is not None:
            judging_args += ["--limit", str(args.judging_limit)]

        for attempt in range(args.judge_retries + 1):
            if is_complete(actor, args.comparison, manifest_path):
                break
            if attempt:
                print(f"retrying missing judge votes for {actor} ({attempt}/{args.judge_retries})", flush=True)
                if args.retry_sleep_s > 0:
                    time.sleep(args.retry_sleep_s)
            judged = run_actor_step(actor, "run_judging", judging_args)
            if not judged:
                failed_actors.append((actor, "run_judging"))
                break
            print(actor_status_line(actor, args.comparison, manifest_path), flush=True)

        aggregated = run_actor_step(
            actor,
            "aggregate_judgments",
            [sys.executable, "-m", "utility_behavior_gap.scripts.aggregate_judgments"],
        )
        if not aggregated:
            failed_actors.append((actor, "aggregate_judgments"))
            continue
        snapshot_outputs(actor, args.comparison)
        if not is_complete(actor, args.comparison, manifest_path):
            print(
                f"warning: {actor} still incomplete after aggregation: "
                f"{actor_status_line(actor, args.comparison, manifest_path)}"
            )

    if failed_actors:
        print("\nfailed actors:")
        for actor, step in failed_actors:
            print(f"  {actor}: {step}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
