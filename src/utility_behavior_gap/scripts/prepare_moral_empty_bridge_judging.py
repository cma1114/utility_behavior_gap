#!/usr/bin/env python3
"""Prepare judging-only moral-low versus framed-empty comparisons.

This script makes no API calls. It copies current clean moral-low outputs and
matched framed-empty outputs into a run-local judging manifest. Matching is
exact on actor, task, item_id, and repeat.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utility_behavior_gap.constants import ACTOR_LABEL, TASK_LABEL
from utility_behavior_gap.fingerprints import text_fingerprint
from utility_behavior_gap.io_utils import write_jsonl
from utility_behavior_gap.paths import OUTPUT_API
from utility_behavior_gap.scripts.prepare_amount_neutral_bridge_judging import (
    csv_arg,
    output_id,
    output_side,
    read_manifest_entries,
    selected_entries,
)
from utility_behavior_gap.scripts.prepare_highlow_framed_empty_bridge_judging import (
    FRAMED_EMPTY_CONDITION,
    load_framed_empty_outputs,
    match_key_from_job,
)
from utility_behavior_gap.scripts.prepare_moral_neutral_bridge_judging import (
    MORAL_BAD,
    MORAL_BASE_GLOB,
    MORAL_HIGHN_GLOB,
    TASKS,
    digest_payload,
    load_classification_labels,
    load_run,
    moral_label_for_output,
)


RUNS_DIR = OUTPUT_API / "runs"
LATEST_PATH = RUNS_DIR / "moral_empty_bridge_latest.txt"


def bridge_pair_uid(moral_job: dict[str, Any], empty_job: dict[str, Any]) -> str:
    digest = digest_payload(
        {
            "source_moral_pair_uid": moral_job["pair_uid"],
            "source_empty_pair_uid": empty_job["pair_uid"],
            "source_moral_condition": MORAL_BAD,
            "source_empty_condition": FRAMED_EMPTY_CONDITION,
        }
    )
    return (
        f"moral_empty_bridge:{moral_job['actor']}:{moral_job['task']}:"
        f"i{moral_job.get('item_id', '')}:r{moral_job.get('repeat', '')}:v{digest}"
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
        "actor": job["actor"],
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
        "source_note": "Copied for judging-only moral-low versus framed-empty bridge; no generation call was made.",
    }


def bridge_job(
    *,
    moral_job: dict[str, Any],
    empty_job: dict[str, Any],
    moral_output: dict[str, Any],
    empty_output: dict[str, Any],
    moral_label: str,
    source_block: str,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    moral_side = output_side(moral_job, MORAL_BAD)
    empty_side = output_side(empty_job, FRAMED_EMPTY_CONDITION)
    pair_uid = bridge_pair_uid(moral_job, empty_job)
    return {
        "actor": moral_job["actor"],
        "actor_label": moral_job.get("actor_label") or ACTOR_LABEL.get(moral_job["actor"], moral_job["actor"]),
        "axis": moral_job.get("axis", ""),
        "axis_definition": moral_job.get("axis_definition", ""),
        "base_prompt": moral_job.get("base_prompt", ""),
        "cause_pair_label": moral_job.get("cause_pair_label", ""),
        "comparison": "moral_bad_vs_framed_empty",
        "condition_a": MORAL_BAD,
        "condition_b": FRAMED_EMPTY_CONDITION,
        "domain": moral_job.get("domain", ""),
        "domain_label": moral_job.get("domain_label", ""),
        "framing": moral_job.get("framing", ""),
        "item_id": moral_job.get("item_id", ""),
        "item_index": moral_job.get("item_index", ""),
        "item_label": moral_job.get("item_label", ""),
        "moral_bad_label": moral_label,
        "other_condition": FRAMED_EMPTY_CONDITION,
        "pair_uid": pair_uid,
        "predicted_condition": MORAL_BAD,
        "prompt_a": moral_job.get(f"prompt_{moral_side}", ""),
        "prompt_b": empty_job.get(f"prompt_{empty_side}", ""),
        "repeat": moral_job.get("repeat", ""),
        "run_dir": str(run_dir),
        "run_generation_failures_path": str(run_dir / "generation_failures.jsonl"),
        "run_generations_path": str(run_dir / "generations.jsonl"),
        "run_id": run_id,
        "run_judge_votes_path": str(run_dir / "judge_votes.jsonl"),
        "run_manifest_path": str(run_dir / "generation_jobs.jsonl"),
        "source_block": source_block,
        "source_empty_output_id": empty_output.get("output_id", ""),
        "source_empty_pair_uid": empty_job["pair_uid"],
        "source_empty_prompt": empty_job.get(f"prompt_{empty_side}", ""),
        "source_empty_run_dir": empty_job.get("run_dir", ""),
        "source_moral_output_id": moral_output.get("output_id", ""),
        "source_moral_pair_uid": moral_job["pair_uid"],
        "source_moral_prompt": moral_job.get(f"prompt_{moral_side}", ""),
        "source_moral_run_dir": moral_job.get("run_dir", ""),
        "source_note": "judging-only bridge from current clean moral-low and framed-empty outputs",
        "system_prompt_a": moral_job.get(f"system_prompt_{moral_side}", ""),
        "system_prompt_b": empty_job.get(f"system_prompt_{empty_side}", ""),
        "task": moral_job["task"],
        "task_label": moral_job.get("task_label") or TASK_LABEL.get(moral_job["task"], moral_job["task"]),
        "topic_design": moral_job.get("topic_design", ""),
    }


def prepare(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], Counter]:
    selected_tasks = set(csv_arg(args.tasks, TASKS))
    selected_actors = set(csv_arg(args.actors)) if args.actors else set()
    labels = load_classification_labels()
    empty_by_key = load_framed_empty_outputs(selected_tasks=selected_tasks, selected_actors=selected_actors)

    moral_entries = selected_entries(
        read_manifest_entries(MORAL_BASE_GLOB),
        selected_tasks=selected_tasks,
        selected_actors=selected_actors,
    )
    if args.source_scope == "current":
        moral_entries.extend(
            selected_entries(
                read_manifest_entries(MORAL_HIGHN_GLOB),
                selected_tasks=selected_tasks,
                selected_actors=selected_actors,
            )
        )
    if not moral_entries:
        raise ValueError("no moral manifests matched the requested actor/task filters")

    run_seed = {
        "actors": sorted({entry["actor"] for entry in moral_entries}),
        "tasks": sorted(selected_tasks),
        "source_scope": args.source_scope,
        "bridge": "moral_bad_vs_framed_empty",
        "moral_bad_clean_only": True,
    }
    run_hash = digest_payload(run_seed)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    run_id = f"moral-empty-bridge-{args.source_scope}__{timestamp}__hash-{run_hash}"
    run_dir = RUNS_DIR / run_id

    jobs_out: list[dict[str, Any]] = []
    generations_out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    seen_pair_uids: set[str] = set()

    for entry in moral_entries:
        jobs, generations = load_run(entry["manifest_path"])
        source_block = "base_fund_wording_r0_r4" if "fund_wording_rerun" in entry["manifest_list"] else "highn_extension_r5_r9"
        for moral_job in jobs:
            if moral_job.get("condition_a") != "moral_good" or moral_job.get("condition_b") != MORAL_BAD:
                continue
            if not str(moral_job.get("comparison", "")).endswith("_moral"):
                continue
            empty_match = empty_by_key.get(match_key_from_job(moral_job))
            if empty_match is None:
                raise ValueError(f"no matched framed_empty job for {moral_job['pair_uid']}")
            empty_job, empty_output = empty_match
            moral_output = generations.get(output_id(moral_job, MORAL_BAD))
            if moral_output is None or not str(moral_output.get("output_text", "")).strip():
                counts[(moral_job["actor"], moral_job["task"], source_block, "missing_moral_bad_output")] += 1
                continue
            moral_label = moral_label_for_output(labels, moral_output)
            if not moral_label:
                raise ValueError(f"missing moral classifier label for {moral_output.get('output_id')}")
            if moral_label != "clean":
                counts[(moral_job["actor"], moral_job["task"], source_block, f"excluded_{moral_label}")] += 1
                continue

            job = bridge_job(
                moral_job=moral_job,
                empty_job=empty_job,
                moral_output=moral_output,
                empty_output=empty_output,
                moral_label=moral_label,
                source_block=source_block,
                run_id=run_id,
                run_dir=run_dir,
            )
            if job["pair_uid"] in seen_pair_uids:
                raise ValueError(f"duplicate bridge pair_uid: {job['pair_uid']}")
            seen_pair_uids.add(job["pair_uid"])
            jobs_out.append(job)
            generations_out.append(
                copied_generation(source=moral_output, job=job, suffix="a", condition=MORAL_BAD, source_role=MORAL_BAD)
            )
            generations_out.append(
                copied_generation(
                    source=empty_output,
                    job=job,
                    suffix="b",
                    condition=FRAMED_EMPTY_CONDITION,
                    source_role=FRAMED_EMPTY_CONDITION,
                )
            )
            counts[(job["actor"], job["task"], source_block, "included")] += 1

    if not jobs_out:
        raise ValueError("no moral-low versus framed-empty bridge comparisons were prepared")
    return run_dir, jobs_out, generations_out, counts


def write_actor_manifest_pointers(jobs: list[dict[str, Any]], run_dir: Path) -> None:
    actors = sorted({str(job["actor"]) for job in jobs})
    if len(actors) == 1:
        actor = actors[0]
        pointer = RUNS_DIR / f"moral_empty_bridge_manifests__{actor}.tsv"
        pointer.write_text(f"{actor}\t{run_dir / 'generation_jobs.jsonl'}\n", encoding="utf-8")
        return
    pointer = RUNS_DIR / "moral_empty_bridge_manifests__all.tsv"
    pointer.write_text("\n".join(f"{actor}\t{run_dir / 'generation_jobs.jsonl'}" for actor in actors) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", default=None, help="Comma-separated actor ids, or all. Default: all.")
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks, or all. Default: all.")
    parser.add_argument(
        "--source-scope",
        choices=("base", "current"),
        default="current",
        help="base uses repeats 0-4 only. current uses moral repeats 0-9 and matched framed-empty controls.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing files.")
    args = parser.parse_args()

    run_dir, jobs, generations, counts = prepare(args)
    print(f"bridge judging jobs: {len(jobs)}")
    print(f"copied outputs: {len(generations)}")
    print(f"run_dir: {run_dir}")
    print("breakdown:")
    for key, count in sorted(counts.items()):
        actor, task, source_block, status = key
        print(f"  {actor} / {task} / {source_block} / {status}: {count}")

    if args.dry_run:
        print("dry run: wrote nothing")
        return

    write_jsonl(run_dir / "generation_jobs.jsonl", jobs)
    write_jsonl(run_dir / "generations.jsonl", generations)
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(str(run_dir) + "\n", encoding="utf-8")
    write_actor_manifest_pointers(jobs, run_dir)
    print(f"wrote {run_dir / 'generation_jobs.jsonl'}")
    print(f"wrote {run_dir / 'generations.jsonl'}")
    print(f"wrote latest pointer: {LATEST_PATH}")
    print(f"next: python -m utility_behavior_gap.scripts.run_judging --run-dir {run_dir} --orders single --workers 4")


if __name__ == "__main__":
    main()
