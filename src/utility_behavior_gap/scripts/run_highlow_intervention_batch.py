#!/usr/bin/env python3
"""Run cleaned high-low intervention jobs across actors and tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from utility_behavior_gap.constants import ACTORS, TASK_LABEL
from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.job_builder import build_generation_jobs
from utility_behavior_gap.openrouter import judge_model_ids
from utility_behavior_gap.paths import OUTPUT_API, OUTPUT_RAW, ROOT


DEFAULT_COMPARISON = "highlow_intervention"
DEFAULT_TASKS = ["essay", "translation", "incident_postmortem", "grant_proposal_abstract"]
TASK_GENERATION_MAX_TOKENS = {
    "essay": 900,
    "translation": 600,
    "incident_postmortem": 3000,
    "grant_proposal_abstract": 1000,
}
CURRENT_GENERATION_JOBS = OUTPUT_API / "generation_jobs.jsonl"
RUNS_DIR = OUTPUT_API / "runs"
GENERATIONS_JSONL = OUTPUT_API / "generations.jsonl"
JUDGE_VOTES_JSONL = OUTPUT_API / "judge_votes.jsonl"
GENERATION_FAILURES_JSONL = OUTPUT_API / "generation_failures.jsonl"
JUDGED_PAIRS_CSV = OUTPUT_RAW / "judged_pairs.csv"
JUDGE_VOTES_CSV = OUTPUT_RAW / "judge_votes.csv"


def parse_csv(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def run_step(label: str, command: list[str]) -> bool:
    try:
        run(command)
    except subprocess.CalledProcessError as exc:
        print(f"error: {label} failed with exit code {exc.returncode}; continuing", flush=True)
        return False
    return True


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def valid_generation_row(row: dict[str, Any]) -> bool:
    return row.get("success") is not False and bool(str(row.get("output_text", "")).strip())


def valid_generation_map(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["output_id"]): row
        for row in read_jsonl_if_exists(path)
        if row.get("output_id") and valid_generation_row(row)
    }


def jobs_from_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    return read_jsonl(manifest_path)


def manifest_matches(manifest: Path, *, actor: str, task: str, comparison: str) -> bool:
    try:
        jobs = jobs_from_manifest(manifest)
    except Exception:
        return False
    if not jobs:
        return False
    return (
        {str(job.get("actor", "")) for job in jobs} == {actor}
        and {str(job.get("task", "")) for job in jobs} == {task}
        and {str(job.get("comparison", "")) for job in jobs} == {comparison}
    )


def latest_manifest_path(actor: str, task: str, comparison: str) -> Path | None:
    if not RUNS_DIR.exists():
        return None
    candidates = [
        manifest
        for manifest in RUNS_DIR.glob("*/generation_jobs.jsonl")
        if manifest_matches(manifest, actor=actor, task=task, comparison=comparison)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def current_manifest_path() -> Path:
    jobs = read_jsonl(CURRENT_GENERATION_JOBS)
    values = {str(job.get("run_manifest_path") or "") for job in jobs}
    values.discard("")
    return Path(values.pop()) if len(values) == 1 else CURRENT_GENERATION_JOBS


def activate_manifest(manifest_path: Path) -> None:
    shutil.copy2(manifest_path, CURRENT_GENERATION_JOBS)


def job_signature(jobs: list[dict[str, Any]]) -> str:
    run_fields = {
        "run_id",
        "run_dir",
        "run_manifest_path",
        "run_generations_path",
        "run_generation_failures_path",
        "run_judge_votes_path",
    }
    stripped = [
        {key: value for key, value in job.items() if key not in run_fields}
        for job in jobs
    ]
    payload = json.dumps(stripped, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def expected_jobs(
    *,
    actor: str,
    task: str,
    comparison: str,
    pairs_per_actor_domain: int | None,
    items_per_task: int | None,
    system_repeats: int | None,
) -> list[dict[str, Any]]:
    return build_generation_jobs(
        comparisons={comparison},
        tasks={task},
        actors={actor},
        pairs_per_actor_domain=pairs_per_actor_domain,
        items_per_task=items_per_task,
        system_repeats=system_repeats,
    )


def manifest_matches_current_builder(
    manifest_path: Path,
    *,
    actor: str,
    task: str,
    comparison: str,
    pairs_per_actor_domain: int | None,
    items_per_task: int | None,
    system_repeats: int | None,
) -> tuple[bool, str]:
    actual = jobs_from_manifest(manifest_path)
    expected = expected_jobs(
        actor=actor,
        task=task,
        comparison=comparison,
        pairs_per_actor_domain=pairs_per_actor_domain,
        items_per_task=items_per_task,
        system_repeats=system_repeats,
    )
    if len(actual) != len(expected):
        return False, f"expected {len(expected)} pairs, found {len(actual)}"
    if job_signature(actual) != job_signature(expected):
        return False, "manifest content does not match the current job builder"
    return True, ""


def run_log_path(jobs: list[dict[str, Any]], field: str, fallback: Path) -> Path:
    values = {str(job.get(field) or "") for job in jobs}
    values.discard("")
    if len(values) == 1:
        return Path(values.pop())
    return fallback


def run_specific_paths(jobs: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    return (
        run_log_path(jobs, "run_generations_path", GENERATIONS_JSONL),
        run_log_path(jobs, "run_generation_failures_path", GENERATION_FAILURES_JSONL),
        run_log_path(jobs, "run_judge_votes_path", JUDGE_VOTES_JSONL),
    )


def unresolved_generation_failures(jobs: list[dict[str, Any]], generations: dict[str, dict[str, Any]]) -> int:
    _, failures_path, _ = run_specific_paths(jobs)
    expected_outputs = {f"{job['pair_uid']}::{suffix}" for job in jobs for suffix in ("a", "b")}
    failures = [
        row for row in read_jsonl_if_exists(failures_path) if str(row.get("output_id", "")) in expected_outputs
    ]
    return sum(1 for row in failures if str(row.get("output_id", "")) not in generations)


def latest_current_votes(
    pair_hashes: dict[str, tuple[str, str]],
    votes_path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    judges = set(judge_model_ids())
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl_if_exists(votes_path):
        pair_uid = str(row.get("pair_uid", ""))
        judge_model = str(row.get("judge_model", ""))
        if judge_model not in judges or row.get("success") is False:
            continue
        expected = pair_hashes.get(pair_uid)
        if expected is None:
            continue
        if row.get("source_output_a_hash") != expected[0] or row.get("source_output_b_hash") != expected[1]:
            continue
        out[(pair_uid, judge_model)] = row
    return out


def completion_counts(manifest_path: Path) -> tuple[int, int, int, int, int, int, int]:
    jobs = jobs_from_manifest(manifest_path)
    run_generations, _, run_votes = run_specific_paths(jobs)
    generations = valid_generation_map(run_generations)
    expected_outputs = {f"{job['pair_uid']}::{suffix}" for job in jobs for suffix in ("a", "b")}
    complete_outputs = sum(1 for output_id in expected_outputs if output_id in generations)
    failures = unresolved_generation_failures(jobs, generations)

    pair_hashes: dict[str, tuple[str, str]] = {}
    for job in jobs:
        out_a = generations.get(f"{job['pair_uid']}::a")
        out_b = generations.get(f"{job['pair_uid']}::b")
        if out_a and out_b:
            pair_hashes[job["pair_uid"]] = (
                output_text_fingerprint(out_a),
                output_text_fingerprint(out_b),
            )

    current_votes = latest_current_votes(pair_hashes, run_votes)
    expected_votes = len(jobs) * len(judge_model_ids())
    complete_pairs = 0
    for job in jobs:
        pair_uid = job["pair_uid"]
        has_outputs = all(f"{pair_uid}::{suffix}" in generations for suffix in ("a", "b"))
        has_votes = sum((pair_uid, judge) in current_votes for judge in judge_model_ids()) == len(judge_model_ids())
        if has_outputs and has_votes:
            complete_pairs += 1
    return complete_outputs, len(expected_outputs), len(current_votes), expected_votes, complete_pairs, len(jobs), failures


def is_complete(manifest_path: Path | None) -> bool:
    if manifest_path is None:
        return False
    complete_outputs, expected_outputs, complete_votes, expected_votes, complete_pairs, expected_pairs, _ = (
        completion_counts(manifest_path)
    )
    return (
        complete_outputs == expected_outputs
        and complete_votes == expected_votes
        and complete_pairs == expected_pairs
    )


def status_line(actor: str, task: str, comparison: str, manifest_path: Path | None = None) -> str:
    manifest_path = manifest_path or latest_manifest_path(actor, task, comparison)
    label = TASK_LABEL.get(task, task)
    if manifest_path is None:
        return f"{actor} / {label}: no run"
    complete_outputs, expected_outputs, complete_votes, expected_votes, complete_pairs, expected_pairs, failures = (
        completion_counts(manifest_path)
    )
    failure_part = f", unresolved generation failures {failures}" if failures else ""
    return (
        f"{actor} / {label}: generations {complete_outputs}/{expected_outputs}, "
        f"votes {complete_votes}/{expected_votes}, pairs {complete_pairs}/{expected_pairs}"
        f"{failure_part}, run {manifest_path.parent.name}"
    )


def print_status(actors: list[str], tasks: list[str], comparison: str) -> None:
    print(f"comparison: {comparison}")
    for actor in actors:
        for task in tasks:
            print(status_line(actor, task, comparison))


def compact_cell_status(actor: str, task: str, comparison: str) -> str:
    manifest_path = latest_manifest_path(actor, task, comparison)
    if manifest_path is None:
        return "not started"
    complete_outputs, expected_outputs, complete_votes, expected_votes, complete_pairs, expected_pairs, failures = (
        completion_counts(manifest_path)
    )
    if complete_outputs == expected_outputs and complete_votes == expected_votes and complete_pairs == expected_pairs:
        status = "done"
    elif complete_outputs == 0 and complete_votes == 0:
        status = "queued"
    elif complete_outputs < expected_outputs:
        status = f"gen {complete_outputs}/{expected_outputs}"
    elif complete_votes < expected_votes:
        status = f"judge {complete_votes}/{expected_votes}"
    else:
        status = f"pairs {complete_pairs}/{expected_pairs}"
    if failures:
        status += f" ({failures} fail)"
    return status


def print_compact_status(actors: list[str], tasks: list[str], comparison: str) -> None:
    short_task = {
        "essay": "essay",
        "translation": "translation",
        "incident_postmortem": "incident",
        "grant_proposal_abstract": "grant",
    }
    for actor in actors:
        cell_statuses = [(short_task.get(task, task), compact_cell_status(actor, task, comparison)) for task in tasks]
        done = sum(1 for _, status in cell_statuses if status == "done")
        parts = "; ".join(f"{task}={status}" for task, status in cell_statuses)
        print(f"{actor}: {done}/{len(cell_statuses)} done; {parts}")


def prepare_or_resume(
    *,
    actor: str,
    task: str,
    comparison: str,
    pairs_per_actor_domain: int | None,
    items_per_task: int | None,
    system_repeats: int | None,
    fresh: bool,
) -> Path | None:
    existing = None if fresh else latest_manifest_path(actor, task, comparison)
    if existing is not None:
        current, reason = manifest_matches_current_builder(
            existing,
            actor=actor,
            task=task,
            comparison=comparison,
            pairs_per_actor_domain=pairs_per_actor_domain,
            items_per_task=items_per_task,
            system_repeats=system_repeats,
        )
        if not current:
            print(f"ignoring stale existing run: {existing.parent} ({reason})", flush=True)
            existing = None
        else:
            activate_manifest(existing)
            print(("using complete" if is_complete(existing) else "resuming") + f" existing run: {existing.parent}")
            return existing

    if existing is not None:
        activate_manifest(existing)
        print(("using complete" if is_complete(existing) else "resuming") + f" existing run: {existing.parent}")
        return existing

    command = [
        sys.executable,
        "-m",
        "utility_behavior_gap.scripts.prepare_generation_jobs",
        "--comparisons",
        comparison,
        "--tasks",
        task,
        "--actors",
        actor,
    ]
    if pairs_per_actor_domain is not None:
        command += ["--pairs-per-actor-domain", str(pairs_per_actor_domain)]
    if items_per_task is not None:
        command += ["--items-per-task", str(items_per_task)]
    if system_repeats is not None:
        command += ["--system-repeats", str(system_repeats)]
    if not run_step(f"prepare_generation_jobs {actor}/{task}", command):
        return None
    return current_manifest_path()


def snapshot_outputs(actor: str, task: str, comparison: str) -> None:
    if not JUDGED_PAIRS_CSV.exists() or not JUDGE_VOTES_CSV.exists():
        return
    prefix = f"{comparison}__{actor}__{task}"
    shutil.copy2(JUDGED_PAIRS_CSV, OUTPUT_RAW / f"{prefix}__judged_pairs.csv")
    shutil.copy2(JUDGE_VOTES_CSV, OUTPUT_RAW / f"{prefix}__judge_votes.csv")


def generation_max_tokens(task: str, value: str) -> int:
    if value == "auto":
        return TASK_GENERATION_MAX_TOKENS[task]
    try:
        max_tokens = int(value)
    except ValueError as exc:
        raise ValueError("--generation-max-tokens must be an integer or 'auto'") from exc
    if max_tokens <= 0:
        raise ValueError("--generation-max-tokens must be positive")
    return max_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default=DEFAULT_COMPARISON)
    parser.add_argument("--actors", default="", help="Comma-separated actor ids. Default: current roster.")
    parser.add_argument("--tasks", default="", help="Comma-separated task ids. Default: all four tasks.")
    parser.add_argument("--pairs-per-actor-domain", type=int, default=None)
    parser.add_argument("--items-per-task", type=int, default=None)
    parser.add_argument("--system-repeats", type=int, default=None)
    parser.add_argument("--generation-retries", type=int, default=3)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-s", type=float, default=60.0)
    parser.add_argument("--generation-limit", type=int, default=None)
    parser.add_argument("--judging-limit", type=int, default=None)
    parser.add_argument("--generation-workers", type=int, default=1, help="Parallel actor-generation API calls per cell.")
    parser.add_argument("--judge-workers", type=int, default=1, help="Parallel judge API calls per cell.")
    parser.add_argument(
        "--generation-max-tokens",
        default="auto",
        help="Generation cap. Use 'auto' for task-specific defaults, or pass an integer override.",
    )
    parser.add_argument("--include-completed", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--compact-status", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.generation_workers < 1:
        raise ValueError("--generation-workers must be at least 1")
    if args.judge_workers < 1:
        raise ValueError("--judge-workers must be at least 1")

    actors = parse_csv(args.actors) or list(ACTORS)
    tasks = parse_csv(args.tasks) or list(DEFAULT_TASKS)
    unknown_actors = sorted(set(actors) - set(ACTORS))
    unknown_tasks = sorted(set(tasks) - set(DEFAULT_TASKS))
    if unknown_actors:
        raise ValueError(f"unknown actor ids: {', '.join(unknown_actors)}")
    if unknown_tasks:
        raise ValueError(f"unknown task ids: {', '.join(unknown_tasks)}")

    if args.compact_status:
        print_compact_status(actors, tasks, args.comparison)
        return
    if args.status:
        print_status(actors, tasks, args.comparison)
        return

    cells = [(actor, task) for actor in actors for task in tasks]
    if not args.include_completed:
        cells = [
            (actor, task)
            for actor, task in cells
            if not is_complete(latest_manifest_path(actor, task, args.comparison))
        ]
    if not cells:
        print("all requested actor/task cells are already complete")
        return

    print("batch cells:")
    for actor, task in cells:
        print(f"  {status_line(actor, task, args.comparison)}")
    if args.plan_only:
        return

    failed: list[tuple[str, str, str]] = []
    for actor, task in cells:
        print(f"\n=== {actor} / {task} ===", flush=True)
        max_tokens = generation_max_tokens(task, args.generation_max_tokens)
        manifest_path = prepare_or_resume(
            actor=actor,
            task=task,
            comparison=args.comparison,
            pairs_per_actor_domain=args.pairs_per_actor_domain,
            items_per_task=args.items_per_task,
            system_repeats=args.system_repeats,
            fresh=args.fresh,
        )
        if manifest_path is None:
            failed.append((actor, task, "prepare_generation_jobs"))
            continue
        print(status_line(actor, task, args.comparison, manifest_path), flush=True)

        generation_args = [
            sys.executable,
            "-m",
            "utility_behavior_gap.scripts.run_generation",
            "--max-tokens",
            str(max_tokens),
        ]
        if args.dry_run:
            generation_args.append("--dry-run")
        if args.generation_limit is not None:
            generation_args += ["--limit", str(args.generation_limit)]
        if args.generation_workers > 1:
            generation_args += ["--workers", str(args.generation_workers)]
        for attempt in range(args.generation_retries + 1):
            complete_outputs, expected_outputs, *_ = completion_counts(manifest_path)
            if complete_outputs == expected_outputs:
                break
            if attempt:
                print(f"retrying missing generations for {actor}/{task} ({attempt}/{args.generation_retries})")
                if args.retry_sleep_s > 0:
                    time.sleep(args.retry_sleep_s)
            if not run_step(f"run_generation {actor}/{task}", generation_args):
                failed.append((actor, task, "run_generation"))
                break
            print(status_line(actor, task, args.comparison, manifest_path), flush=True)

        complete_outputs, expected_outputs, *_ = completion_counts(manifest_path)
        if complete_outputs != expected_outputs:
            print(
                f"warning: skipping judging for {actor}/{task}; generations are incomplete "
                f"({complete_outputs}/{expected_outputs})"
            )
            failed.append((actor, task, "incomplete_generations"))
            continue

        judging_args = [sys.executable, "-m", "utility_behavior_gap.scripts.run_judging"]
        if args.dry_run:
            judging_args.append("--dry-run")
        if args.judging_limit is not None:
            judging_args += ["--limit", str(args.judging_limit)]
        if args.judge_workers > 1:
            judging_args += ["--workers", str(args.judge_workers)]
        for attempt in range(args.judge_retries + 1):
            if is_complete(manifest_path):
                break
            if attempt:
                print(f"retrying missing judge votes for {actor}/{task} ({attempt}/{args.judge_retries})")
                if args.retry_sleep_s > 0:
                    time.sleep(args.retry_sleep_s)
            if not run_step(f"run_judging {actor}/{task}", judging_args):
                failed.append((actor, task, "run_judging"))
                break
            print(status_line(actor, task, args.comparison, manifest_path), flush=True)

        if not is_complete(manifest_path):
            print(f"warning: skipping aggregation for {actor}/{task}; judging is incomplete")
            print(status_line(actor, task, args.comparison, manifest_path), flush=True)
            failed.append((actor, task, "incomplete_judging"))
            continue

        if not run_step(
            f"aggregate_judgments {actor}/{task}",
            [sys.executable, "-m", "utility_behavior_gap.scripts.aggregate_judgments"],
        ):
            failed.append((actor, task, "aggregate_judgments"))
            continue
        snapshot_outputs(actor, task, args.comparison)

    if failed:
        print("\nfailed cells:")
        for actor, task, step in failed:
            print(f"  {actor} / {task}: {step}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
