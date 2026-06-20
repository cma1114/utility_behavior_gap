#!/usr/bin/env python3
"""Run clean direct-instruction max-effort jobs on the high-low task grid."""

from __future__ import annotations

import argparse
import subprocess
import sys

from utility_behavior_gap.constants import ACTORS
from utility_behavior_gap.paths import ROOT


DEFAULT_COMPARISON = "direct_user_prompt_max_effort"
DEFAULT_TASKS = ["essay", "translation", "incident_postmortem", "grant_proposal_abstract"]


def parse_csv(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def run_step(label: str, command: list[str]) -> bool:
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        print(f"error: {label} failed with exit code {result.returncode}; continuing", flush=True)
        return False
    return True


def base_batch_command(args: argparse.Namespace, tasks: list[str]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "utility_behavior_gap.scripts.run_highlow_intervention_batch",
        "--comparison",
        args.comparison,
        "--tasks",
        ",".join(tasks),
    ]
    if args.actors:
        command += ["--actors", args.actors]
    if args.generation_retries is not None:
        command += ["--generation-retries", str(args.generation_retries)]
    if args.judge_retries is not None:
        command += ["--judge-retries", str(args.judge_retries)]
    if args.retry_sleep_s is not None:
        command += ["--retry-sleep-s", str(args.retry_sleep_s)]
    if args.generation_limit is not None:
        command += ["--generation-limit", str(args.generation_limit)]
    if args.judging_limit is not None:
        command += ["--judging-limit", str(args.judging_limit)]
    if args.generation_workers is not None:
        command += ["--generation-workers", str(args.generation_workers)]
    if args.judge_workers is not None:
        command += ["--judge-workers", str(args.judge_workers)]
    if args.generation_max_tokens:
        command += ["--generation-max-tokens", args.generation_max_tokens]
    if args.include_completed:
        command.append("--include-completed")
    if args.fresh:
        command.append("--fresh")
    if args.dry_run:
        command.append("--dry-run")
    if args.plan_only:
        command.append("--plan-only")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default=DEFAULT_COMPARISON)
    parser.add_argument("--actors", default="", help="Comma-separated actor ids. Default: current roster.")
    parser.add_argument("--tasks", default="", help="Comma-separated task ids. Default: all four high-low tasks.")
    parser.add_argument("--generation-retries", type=int, default=5)
    parser.add_argument("--judge-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-s", type=float, default=15.0)
    parser.add_argument("--generation-limit", type=int, default=None)
    parser.add_argument("--judging-limit", type=int, default=None)
    parser.add_argument("--generation-workers", type=int, default=4)
    parser.add_argument("--judge-workers", type=int, default=8)
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

    tasks = parse_csv(args.tasks) or list(DEFAULT_TASKS)
    actors = parse_csv(args.actors) or list(ACTORS)
    unknown_tasks = sorted(set(tasks) - set(DEFAULT_TASKS))
    unknown_actors = sorted(set(actors) - set(ACTORS))
    if unknown_tasks:
        raise ValueError(f"unknown direct-instruction task ids: {', '.join(unknown_tasks)}")
    if unknown_actors:
        raise ValueError(f"unknown actor ids: {', '.join(unknown_actors)}")
    args.actors = ",".join(actors)

    if args.status or args.compact_status:
        command = base_batch_command(args, tasks)
        command.append("--compact-status" if args.compact_status else "--status")
        ok = run_step("status", command)
        raise SystemExit(0 if ok else 1)

    print("direct-instruction batch design: high-low matched, 320 pairs per actor/task", flush=True)

    failures: list[str] = []
    for task in tasks:
        ok = run_step(task, base_batch_command(args, [task]))
        if not ok:
            failures.append(task)

    if failures:
        print("\nfailed task batches:")
        for task in failures:
            print(f"  {task}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
