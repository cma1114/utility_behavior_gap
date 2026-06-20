#!/usr/bin/env python3
"""Prepare judging-only MiMo framed-neutral versus old direct-output comparisons.

This script does not call any model. It compares existing outputs:

* latest corrected ``framed_neutral`` outputs from the fund-wording rerun
* older clean ``direct_user_prompt_max_effort`` outputs for MiMo V2.5

Matching is by task and item label. The older direct runs did not preserve
repeat IDs, so within each item label both sides are sorted deterministically
and paired in order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utility_behavior_gap.fingerprints import text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl, write_jsonl
from utility_behavior_gap.paths import OUTPUT_API


ACTOR = "mimo-v25-pro-or"
RUNS_DIR = OUTPUT_API / "runs"
LATEST_PATH = RUNS_DIR / "mimo_framed_neutral_bridge_latest.txt"
FUND_MANIFEST = RUNS_DIR / f"fund_wording_rerun_manifests__{ACTOR}.tsv"
TASKS = ("essay", "translation", "incident_postmortem", "grant_proposal_abstract")
OLD_CONDITIONS = ("user_strong", "user_normal")
OLD_LABELS = {
    "user_strong": "old_user_strong_max_effort",
    "user_normal": "old_user_normal_good_job",
}
NEUTRAL_CONDITION = "framed_neutral"


def csv_arg(value: str | None, allowed: tuple[str, ...] | None = None) -> list[str]:
    if value is None or value.strip().lower() == "all":
        return list(allowed or ())
    values = [part.strip() for part in value.split(",") if part.strip()]
    if allowed is not None:
        bad = sorted(set(values) - set(allowed))
        if bad:
            raise ValueError(f"unknown value(s): {', '.join(bad)}")
    return values


def digest_payload(payload: dict[str, Any], n: int = 12) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:n]


def output_side(job: dict[str, Any], condition: str) -> str:
    if job.get("condition_a") == condition:
        return "a"
    if job.get("condition_b") == condition:
        return "b"
    raise ValueError(f"{job.get('pair_uid')} does not contain condition {condition!r}")


def output_id(job: dict[str, Any], condition: str) -> str:
    return f"{job['pair_uid']}::{output_side(job, condition)}"


def read_fund_manifest_paths() -> dict[str, Path]:
    if not FUND_MANIFEST.exists():
        raise FileNotFoundError(FUND_MANIFEST)
    paths: dict[str, Path] = {}
    for line in FUND_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        actor, task, manifest = line.split("\t")
        if actor != ACTOR:
            continue
        paths[task] = Path(manifest)
    return paths


def direct_run_dirs(task: str) -> list[Path]:
    pattern = f"{task}__direct_user_prompt_max_effort__{ACTOR}__*"
    return sorted(path for path in RUNS_DIR.glob(pattern) if (path / "generation_jobs.jsonl").exists())


def generated_outputs_by_label(run_dir: Path, condition: str) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    jobs = read_jsonl(run_dir / "generation_jobs.jsonl")
    generations = {row["output_id"]: row for row in read_jsonl(run_dir / "generations.jsonl")}
    by_label: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for job in jobs:
        if job.get("condition_a") != "user_strong" or job.get("condition_b") != "user_normal":
            continue
        oid = output_id(job, condition)
        generation = generations.get(oid)
        if generation is None or not str(generation.get("output_text", "")).strip():
            continue
        by_label[str(job.get("item_label", ""))].append((job, generation))
    for rows in by_label.values():
        rows.sort(key=lambda pair: str(pair[0].get("pair_uid", "")))
    return dict(by_label)


def latest_neutral_outputs_by_label(manifest_path: Path) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    jobs = read_jsonl(manifest_path)
    generations = {row["output_id"]: row for row in read_jsonl(manifest_path.parent / "generations.jsonl")}
    by_label: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for job in jobs:
        if not str(job.get("comparison", "")).endswith("_headroom"):
            continue
        oid = output_id(job, NEUTRAL_CONDITION)
        generation = generations.get(oid)
        if generation is None or not str(generation.get("output_text", "")).strip():
            raise ValueError(f"missing framed-neutral output for {job['pair_uid']}")
        by_label[str(job.get("item_label", ""))].append((job, generation))
    for rows in by_label.values():
        rows.sort(key=lambda pair: (str(pair[0].get("item_label", "")), str(pair[0].get("repeat", "")), str(pair[0].get("pair_uid", ""))))
    return dict(by_label)


def overlap_count(
    neutral_by_label: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    old_by_label: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
) -> int:
    return sum(min(len(neutral_by_label.get(label, [])), len(old_by_label.get(label, []))) for label in neutral_by_label)


def select_old_run(task: str, condition: str, neutral_by_label: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]) -> tuple[Path, dict[str, list[tuple[dict[str, Any], dict[str, Any]]]], int]:
    candidates: list[tuple[int, Path, dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]]] = []
    for run_dir in direct_run_dirs(task):
        try:
            old_by_label = generated_outputs_by_label(run_dir, condition)
        except FileNotFoundError:
            continue
        candidates.append((overlap_count(neutral_by_label, old_by_label), run_dir, old_by_label))
    if not candidates:
        raise FileNotFoundError(f"no prior direct_user_prompt_max_effort run found for {task}")
    candidates.sort(key=lambda row: (row[0], row[1].name), reverse=True)
    best_overlap, best_run, best_by_label = candidates[0]
    expected = sum(len(rows) for rows in neutral_by_label.values())
    if best_overlap < expected:
        raise ValueError(
            f"best prior run for {task}/{condition} has only {best_overlap}/{expected} matched outputs: {best_run}"
        )
    return best_run, best_by_label, best_overlap


def bridge_pair_uid(neutral_job: dict[str, Any], old_job: dict[str, Any], old_condition: str) -> str:
    digest = digest_payload(
        {
            "neutral_pair_uid": neutral_job["pair_uid"],
            "old_pair_uid": old_job["pair_uid"],
            "old_condition": old_condition,
        }
    )
    return (
        f"mimo_framed_neutral_bridge:{old_condition}:{neutral_job['task']}:"
        f"r{neutral_job.get('repeat', '')}:v{digest}"
    )


def copied_generation(
    *,
    source: dict[str, Any],
    job: dict[str, Any],
    suffix: str,
    condition: str,
    source_role: str,
) -> dict[str, Any]:
    text = str(source.get("output_text", ""))
    return {
        "actor": ACTOR,
        "condition": condition,
        "finish_reason": source.get("finish_reason", ""),
        "job": job,
        "model": source.get("model", ""),
        "output_id": f"{job['pair_uid']}::{suffix}",
        "output_text": text,
        "pair_uid": job["pair_uid"],
        "run_id": job["run_id"],
        "success": True,
        "source_condition": source.get("condition", ""),
        "source_output_hash": text_fingerprint(text),
        "source_output_id": source.get("output_id", ""),
        "source_pair_uid": source.get("pair_uid", ""),
        "source_role": source_role,
        "source_run_dir": source.get("job", {}).get("run_dir", ""),
        "source_run_id": source.get("run_id", ""),
        "source_note": "Copied for judging-only MiMo framed-neutral bridge; no generation call was made.",
    }


def bridge_job(
    *,
    neutral_job: dict[str, Any],
    neutral_output: dict[str, Any],
    old_job: dict[str, Any],
    old_output: dict[str, Any],
    old_condition: str,
    old_run_dir: Path,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    old_label = OLD_LABELS[old_condition]
    pair_uid = bridge_pair_uid(neutral_job, old_job, old_condition)
    neutral_prompt_field = f"prompt_{output_side(neutral_job, NEUTRAL_CONDITION)}"
    old_prompt_field = f"prompt_{output_side(old_job, old_condition)}"
    return {
        "actor": ACTOR,
        "actor_label": neutral_job.get("actor_label", "MiMo V2.5 Pro"),
        "axis": neutral_job.get("axis", old_job.get("axis", "")),
        "axis_definition": neutral_job.get("axis_definition", old_job.get("axis_definition", "")),
        "base_prompt": neutral_job.get("base_prompt", old_job.get("base_prompt", "")),
        "comparison": f"mimo_framed_neutral_vs_{old_label}",
        "condition_a": NEUTRAL_CONDITION,
        "condition_b": old_label,
        "item_id": neutral_job.get("item_id", ""),
        "item_index": neutral_job.get("item_index", ""),
        "item_label": neutral_job.get("item_label", ""),
        "old_condition": old_condition,
        "old_label": old_label,
        "other_condition": old_label,
        "pair_uid": pair_uid,
        "predicted_condition": NEUTRAL_CONDITION,
        "prompt_a": neutral_job.get(neutral_prompt_field, ""),
        "prompt_b": old_job.get(old_prompt_field, ""),
        "repeat": neutral_job.get("repeat", ""),
        "run_dir": str(run_dir),
        "run_generation_failures_path": str(run_dir / "generation_failures.jsonl"),
        "run_generations_path": str(run_dir / "generations.jsonl"),
        "run_id": run_id,
        "run_judge_votes_path": str(run_dir / "judge_votes.jsonl"),
        "run_manifest_path": str(run_dir / "generation_jobs.jsonl"),
        "source_neutral_output_id": neutral_output.get("output_id", ""),
        "source_neutral_pair_uid": neutral_job["pair_uid"],
        "source_neutral_prompt": neutral_job.get(neutral_prompt_field, ""),
        "source_neutral_run_dir": neutral_job.get("run_dir", ""),
        "source_old_output_id": old_output.get("output_id", ""),
        "source_old_pair_uid": old_job["pair_uid"],
        "source_old_prompt": old_job.get(old_prompt_field, ""),
        "source_old_run_dir": str(old_run_dir),
        "source_note": "judging-only bridge: latest framed-neutral versus older clean direct MiMo output",
        "system_prompt_a": neutral_job.get(f"system_prompt_{output_side(neutral_job, NEUTRAL_CONDITION)}", ""),
        "system_prompt_b": old_job.get(f"system_prompt_{output_side(old_job, old_condition)}", ""),
        "task": neutral_job["task"],
        "task_label": neutral_job.get("task_label", neutral_job["task"]),
        "topic_design": neutral_job.get("topic_design", ""),
    }


def prepare(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], Counter, dict[tuple[str, str], Path]]:
    selected_tasks = set(csv_arg(args.tasks, TASKS))
    selected_old_conditions = set(csv_arg(args.old_conditions, OLD_CONDITIONS))
    fund_paths = read_fund_manifest_paths()

    run_seed = {
        "actor": ACTOR,
        "tasks": sorted(selected_tasks),
        "old_conditions": sorted(selected_old_conditions),
        "source": "latest fund-wording framed-neutral plus prior direct_user_prompt_max_effort",
    }
    run_hash = digest_payload(run_seed)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    run_id = f"mimo-framed-neutral-bridge__{timestamp}__hash-{run_hash}"
    run_dir = RUNS_DIR / run_id

    jobs_out: list[dict[str, Any]] = []
    generations_out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    selected_runs: dict[tuple[str, str], Path] = {}
    seen: set[str] = set()

    for task in TASKS:
        if task not in selected_tasks:
            continue
        if task not in fund_paths:
            raise ValueError(f"no latest fund-wording manifest for task {task}")
        neutral_by_label = latest_neutral_outputs_by_label(fund_paths[task])
        for old_condition in OLD_CONDITIONS:
            if old_condition not in selected_old_conditions:
                continue
            old_run_dir, old_by_label, _overlap = select_old_run(task, old_condition, neutral_by_label)
            selected_runs[(task, old_condition)] = old_run_dir
            for label, neutral_rows in sorted(neutral_by_label.items()):
                old_rows = old_by_label.get(label, [])
                if len(old_rows) < len(neutral_rows):
                    raise ValueError(
                        f"{task}/{old_condition}/{label} has only {len(old_rows)} old rows for "
                        f"{len(neutral_rows)} neutral rows"
                    )
                for index, ((neutral_job, neutral_output), (old_job, old_output)) in enumerate(zip(neutral_rows, old_rows)):
                    job = bridge_job(
                        neutral_job=neutral_job,
                        neutral_output=neutral_output,
                        old_job=old_job,
                        old_output=old_output,
                        old_condition=old_condition,
                        old_run_dir=old_run_dir,
                        run_id=run_id,
                        run_dir=run_dir,
                    )
                    job["match_index_within_item_label"] = str(index)
                    if job["pair_uid"] in seen:
                        raise ValueError(f"duplicate pair_uid: {job['pair_uid']}")
                    seen.add(job["pair_uid"])
                    jobs_out.append(job)
                    generations_out.append(
                        copied_generation(
                            source=neutral_output,
                            job=job,
                            suffix="a",
                            condition=NEUTRAL_CONDITION,
                            source_role=NEUTRAL_CONDITION,
                        )
                    )
                    generations_out.append(
                        copied_generation(
                            source=old_output,
                            job=job,
                            suffix="b",
                            condition=job["condition_b"],
                            source_role=old_condition,
                        )
                    )
                    counts[(task, old_condition)] += 1

    if not jobs_out:
        raise ValueError("no bridge jobs were prepared")
    return run_dir, jobs_out, generations_out, counts, selected_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks, or all. Default: all.")
    parser.add_argument(
        "--old-conditions",
        default="user_strong",
        help="Comma-separated old direct conditions: user_strong,user_normal. Default: user_strong.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing files.")
    args = parser.parse_args()

    run_dir, jobs, generations, counts, selected_runs = prepare(args)
    print(f"bridge judging jobs: {len(jobs)}")
    print(f"copied outputs: {len(generations)}")
    print(f"run_dir: {run_dir}")
    print("selected old runs:")
    for (task, condition), path in sorted(selected_runs.items()):
        print(f"  {task} / {condition}: {path}")
    print("breakdown:")
    for key, count in sorted(counts.items()):
        task, condition = key
        print(f"  {task} / framed_neutral vs {condition}: {count}")

    if args.dry_run:
        print("dry run: wrote nothing")
        return

    write_jsonl(run_dir / "generation_jobs.jsonl", jobs)
    write_jsonl(run_dir / "generations.jsonl", generations)
    LATEST_PATH.write_text(str(run_dir) + "\n", encoding="utf-8")
    print(f"wrote {run_dir / 'generation_jobs.jsonl'}")
    print(f"wrote {run_dir / 'generations.jsonl'}")
    print(f"wrote latest pointer: {LATEST_PATH}")
    print(f"next: python -m utility_behavior_gap.scripts.run_judging --run-dir {run_dir} --orders both --workers 10")


if __name__ == "__main__":
    main()
