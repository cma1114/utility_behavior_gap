#!/usr/bin/env python3
"""Prepare judging-only moral-bad versus framed-neutral comparisons.

This script makes no API calls. It copies existing outputs into a run-local
bridge manifest:

* current canonical ``moral_bad`` outputs from repeats 0-9;
* matched ``framed_neutral`` outputs from the direct-instruction headroom runs.

Matching is exact on actor, task, item_id, and repeat. Repeats 5-9 match the
first framed-neutral repeat block with a +5 repeat offset.

Only clean moral-bad outputs are included. Labels are loaded by output id and
hash from ``canonical_highn_moral_refusal_classifications.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utility_behavior_gap.constants import ACTOR_LABEL, TASK_LABEL
from utility_behavior_gap.fingerprints import output_text_fingerprint, text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl, write_jsonl
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API
from utility_behavior_gap.scripts.prepare_amount_neutral_bridge_judging import (
    csv_arg,
    load_neutral_index,
    load_run,
    match_key,
    output_id,
    output_side,
    read_manifest_entries,
    selected_entries,
)


RUNS_DIR = OUTPUT_API / "runs"
LATEST_PATH = RUNS_DIR / "moral_neutral_bridge_latest.txt"
MORAL_BASE_GLOB = "fund_wording_rerun_manifests__*.tsv"
MORAL_HIGHN_GLOB = "canonical_highn10_manifests__*.tsv"
NEUTRAL_BASE_GLOB = "framed_user_strong_manifests__*.tsv"
NEUTRAL_REPEAT_BLOCK_GLOB = "framed_user_strong_manifests__repeat-block01__*.tsv"
CLASSIFICATIONS = ANALYSIS / "canonical_highn_moral_refusal_classifications.jsonl"
TASKS = ("essay", "translation", "incident_postmortem", "grant_proposal_abstract")
MORAL_BAD = "moral_bad"
FRAMED_NEUTRAL = "framed_neutral"
VALID_LABELS = {"clean", "partial_refusal", "full_refusal", "degenerate"}


def digest_payload(payload: dict[str, Any], n: int = 12) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:n]


def load_classification_labels() -> dict[tuple[str, str], str]:
    labels: dict[tuple[str, str], str] = {}
    if not CLASSIFICATIONS.exists():
        raise FileNotFoundError(CLASSIFICATIONS)
    for row in read_jsonl(CLASSIFICATIONS):
        output_id_value = str(row.get("output_id", ""))
        output_hash = str(row.get("output_hash", ""))
        label = str(row.get("label", ""))
        if output_id_value and output_hash and label in VALID_LABELS:
            labels[(output_id_value, output_hash)] = label
    return labels


def moral_label_for_output(labels: dict[tuple[str, str], str], output: dict[str, Any]) -> str:
    key = (str(output.get("output_id", "")), output_text_fingerprint(output))
    return labels.get(key, "")


def bridge_pair_uid(moral_job: dict[str, Any], neutral_job: dict[str, Any]) -> str:
    digest = digest_payload(
        {
            "source_moral_pair_uid": moral_job["pair_uid"],
            "source_neutral_pair_uid": neutral_job["pair_uid"],
            "source_moral_condition": MORAL_BAD,
            "source_neutral_condition": FRAMED_NEUTRAL,
        }
    )
    return (
        f"moral_neutral_bridge:{moral_job['actor']}:{moral_job['task']}:"
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
        "source_note": "Copied for judging-only moral-bad versus framed-neutral bridge; no generation call was made.",
    }


def bridge_job(
    *,
    moral_job: dict[str, Any],
    neutral_job: dict[str, Any],
    moral_output: dict[str, Any],
    neutral_output: dict[str, Any],
    moral_label: str,
    source_block: str,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    moral_side = output_side(moral_job, MORAL_BAD)
    neutral_side = output_side(neutral_job, FRAMED_NEUTRAL)
    pair_uid = bridge_pair_uid(moral_job, neutral_job)
    return {
        "actor": moral_job["actor"],
        "actor_label": moral_job.get("actor_label") or ACTOR_LABEL.get(moral_job["actor"], moral_job["actor"]),
        "axis": moral_job.get("axis", ""),
        "axis_definition": moral_job.get("axis_definition", ""),
        "base_prompt": moral_job.get("base_prompt", ""),
        "cause_pair_label": moral_job.get("cause_pair_label", ""),
        "comparison": "moral_bad_vs_framed_neutral",
        "condition_a": MORAL_BAD,
        "condition_b": FRAMED_NEUTRAL,
        "domain": moral_job.get("domain", ""),
        "domain_label": moral_job.get("domain_label", ""),
        "framing": moral_job.get("framing", ""),
        "item_id": moral_job.get("item_id", ""),
        "item_index": moral_job.get("item_index", ""),
        "item_label": moral_job.get("item_label", ""),
        "moral_bad_label": moral_label,
        "other_condition": FRAMED_NEUTRAL,
        "pair_uid": pair_uid,
        "predicted_condition": MORAL_BAD,
        "prompt_a": moral_job.get(f"prompt_{moral_side}", ""),
        "prompt_b": neutral_job.get(f"prompt_{neutral_side}", ""),
        "repeat": moral_job.get("repeat", ""),
        "run_dir": str(run_dir),
        "run_generation_failures_path": str(run_dir / "generation_failures.jsonl"),
        "run_generations_path": str(run_dir / "generations.jsonl"),
        "run_id": run_id,
        "run_judge_votes_path": str(run_dir / "judge_votes.jsonl"),
        "run_manifest_path": str(run_dir / "generation_jobs.jsonl"),
        "source_block": source_block,
        "source_moral_output_id": moral_output.get("output_id", ""),
        "source_moral_pair_uid": moral_job["pair_uid"],
        "source_moral_prompt": moral_job.get(f"prompt_{moral_side}", ""),
        "source_moral_run_dir": moral_job.get("run_dir", ""),
        "source_neutral_output_id": neutral_output.get("output_id", ""),
        "source_neutral_pair_uid": neutral_job["pair_uid"],
        "source_neutral_prompt": neutral_job.get(f"prompt_{neutral_side}", ""),
        "source_neutral_run_dir": neutral_job.get("run_dir", ""),
        "source_note": "judging-only bridge from current clean moral-bad and framed-neutral outputs",
        "system_prompt_a": moral_job.get(f"system_prompt_{moral_side}", ""),
        "system_prompt_b": neutral_job.get(f"system_prompt_{neutral_side}", ""),
        "task": moral_job["task"],
        "task_label": moral_job.get("task_label") or TASK_LABEL.get(moral_job["task"], moral_job["task"]),
        "topic_design": moral_job.get("topic_design", ""),
    }


def prepare(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], Counter]:
    selected_tasks = set(csv_arg(args.tasks, TASKS))
    selected_actors = set(csv_arg(args.actors)) if args.actors else set()
    labels = load_classification_labels()

    moral_entries = selected_entries(
        read_manifest_entries(MORAL_BASE_GLOB),
        selected_tasks=selected_tasks,
        selected_actors=selected_actors,
    )
    neutral_base_entries = selected_entries(
        read_manifest_entries(NEUTRAL_BASE_GLOB, exclude_repeat_blocks=True),
        selected_tasks=selected_tasks,
        selected_actors=selected_actors,
    )
    neutral_by_key = load_neutral_index(neutral_base_entries)
    source_manifest_globs = [MORAL_BASE_GLOB, NEUTRAL_BASE_GLOB]

    if args.source_scope == "current":
        highn_entries = selected_entries(
            read_manifest_entries(MORAL_HIGHN_GLOB),
            selected_tasks=selected_tasks,
            selected_actors=selected_actors,
        )
        moral_entries.extend(highn_entries)
        repeat_block_entries = selected_entries(
            read_manifest_entries(NEUTRAL_REPEAT_BLOCK_GLOB),
            selected_tasks=selected_tasks,
            selected_actors=selected_actors,
        )
        repeat_block_neutrals = load_neutral_index(repeat_block_entries, repeat_offset=5)
        conflicts = set(neutral_by_key).intersection(repeat_block_neutrals)
        if conflicts:
            raise ValueError(f"neutral key collision between base and repeat block: {sorted(conflicts)[:3]}")
        neutral_by_key.update(repeat_block_neutrals)
        source_manifest_globs.extend([MORAL_HIGHN_GLOB, NEUTRAL_REPEAT_BLOCK_GLOB])

    if not moral_entries:
        raise ValueError("no moral manifests matched the requested actor/task filters")

    run_seed = {
        "actors": sorted({entry["actor"] for entry in moral_entries}),
        "tasks": sorted(selected_tasks),
        "source_scope": args.source_scope,
        "source_manifest_globs": source_manifest_globs,
        "bridge": "moral_bad_vs_framed_neutral",
        "moral_bad_clean_only": True,
    }
    run_hash = digest_payload(run_seed)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    run_id = f"moral-neutral-bridge-{args.source_scope}__{timestamp}__hash-{run_hash}"
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
            neutral_match = neutral_by_key.get(match_key(moral_job))
            if neutral_match is None:
                raise ValueError(f"no matched framed-neutral job for {moral_job['pair_uid']}")
            neutral_job, neutral_generations = neutral_match
            moral_output = generations.get(output_id(moral_job, MORAL_BAD))
            neutral_output = neutral_generations.get(output_id(neutral_job, FRAMED_NEUTRAL))
            if moral_output is None or not str(moral_output.get("output_text", "")).strip():
                counts[(moral_job["actor"], moral_job["task"], source_block, "missing_moral_bad_output")] += 1
                continue
            if neutral_output is None or not str(neutral_output.get("output_text", "")).strip():
                raise ValueError(f"missing framed_neutral output for {neutral_job['pair_uid']}")

            moral_label = moral_label_for_output(labels, moral_output)
            if not moral_label:
                raise ValueError(f"missing moral classifier label for {moral_output.get('output_id')}")
            if moral_label != "clean":
                counts[(moral_job["actor"], moral_job["task"], source_block, f"excluded_{moral_label}")] += 1
                continue

            job = bridge_job(
                moral_job=moral_job,
                neutral_job=neutral_job,
                moral_output=moral_output,
                neutral_output=neutral_output,
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
                copied_generation(
                    source=moral_output,
                    job=job,
                    suffix="a",
                    condition=MORAL_BAD,
                    source_role=MORAL_BAD,
                )
            )
            generations_out.append(
                copied_generation(
                    source=neutral_output,
                    job=job,
                    suffix="b",
                    condition=FRAMED_NEUTRAL,
                    source_role=FRAMED_NEUTRAL,
                )
            )
            counts[(job["actor"], job["task"], source_block, "included")] += 1

    if not jobs_out:
        raise ValueError("no moral-bad versus framed-neutral bridge comparisons were prepared")
    return run_dir, jobs_out, generations_out, counts


def write_actor_manifest_pointers(jobs: list[dict[str, Any]], run_dir: Path) -> None:
    actors = sorted({str(job["actor"]) for job in jobs})
    if len(actors) == 1:
        actor = actors[0]
        pointer = RUNS_DIR / f"moral_neutral_bridge_manifests__{actor}.tsv"
        pointer.write_text(f"{actor}\t{run_dir / 'generation_jobs.jsonl'}\n", encoding="utf-8")
        return
    pointer = RUNS_DIR / "moral_neutral_bridge_manifests__all.tsv"
    pointer.write_text("\n".join(f"{actor}\t{run_dir / 'generation_jobs.jsonl'}" for actor in actors) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", default=None, help="Comma-separated actor ids, or all. Default: all.")
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks, or all. Default: all.")
    parser.add_argument(
        "--source-scope",
        choices=("base", "current"),
        default="current",
        help="base uses repeats 0-4 only. current uses moral repeats 0-9 and matched neutral controls.",
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
    print(f"next: python -m utility_behavior_gap.scripts.run_judging --run-dir {run_dir} --orders both --workers 5")


if __name__ == "__main__":
    main()
