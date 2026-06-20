#!/usr/bin/env python3
"""Prepare judging-only amount-high versus framed-neutral comparisons.

This script does not call any model. It builds a run-local judging manifest by
copying existing outputs:

* current canonical ``amount_high`` outputs from repeats 0-9;
* matched ``framed_neutral`` outputs from the direct-instruction headroom runs.

The match is exact on actor, task, item_id, and repeat. For amount repeats 5-9,
the script matches to the first framed-neutral repeat block with a +5 repeat
offset, mirroring the high/low-utility-vs-neutral bridge design.
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
from utility_behavior_gap.fingerprints import text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl, write_jsonl
from utility_behavior_gap.paths import OUTPUT_API


RUNS_DIR = OUTPUT_API / "runs"
LATEST_PATH = RUNS_DIR / "amount_neutral_bridge_latest.txt"
AMOUNT_BASE_GLOB = "canonical_amount_base_manifests__*.tsv"
AMOUNT_HIGHN_GLOB = "canonical_highn10_manifests__*.tsv"
NEUTRAL_BASE_GLOB = "framed_user_strong_manifests__*.tsv"
NEUTRAL_REPEAT_BLOCK_GLOB = "framed_user_strong_manifests__repeat-block01__*.tsv"
TASKS = ("essay", "translation", "incident_postmortem", "grant_proposal_abstract")
AMOUNT_HIGH = "amount_high"
FRAMED_NEUTRAL = "framed_neutral"


def csv_arg(value: str | None, allowed: tuple[str, ...] | None = None) -> list[str]:
    if value is None or value.strip().lower() == "all":
        return list(allowed or ())
    out = [part.strip() for part in value.split(",") if part.strip()]
    if allowed is not None:
        bad = sorted(set(out) - set(allowed))
        if bad:
            raise ValueError(f"unknown value(s): {', '.join(bad)}")
    return out


def read_manifest_entries(glob: str, *, exclude_repeat_blocks: bool = False) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for tsv in sorted(RUNS_DIR.glob(glob)):
        if exclude_repeat_blocks and "repeat-block" in tsv.name:
            continue
        with tsv.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                actor, task, manifest_path = line.rstrip("\n").split("\t")
                path = Path(manifest_path)
                entries.append(
                    {
                        "actor": actor,
                        "task": task,
                        "manifest_path": path,
                        "run_dir": path.parent,
                        "manifest_list": tsv.name,
                    }
                )
    if not entries:
        raise FileNotFoundError(f"no {glob} files found under {RUNS_DIR}")
    return entries


def selected_entries(
    entries: list[dict[str, Any]],
    *,
    selected_tasks: set[str],
    selected_actors: set[str],
) -> list[dict[str, Any]]:
    return [
        entry
        for entry in entries
        if entry["task"] in selected_tasks and (not selected_actors or entry["actor"] in selected_actors)
    ]


def output_side(job: dict[str, Any], condition: str) -> str:
    if job.get("condition_a") == condition:
        return "a"
    if job.get("condition_b") == condition:
        return "b"
    raise ValueError(f"{job.get('pair_uid')} does not contain condition {condition!r}")


def output_id(job: dict[str, Any], condition: str) -> str:
    return f"{job['pair_uid']}::{output_side(job, condition)}"


def shifted_repeat(value: Any, offset: int = 0) -> str:
    try:
        return str(int(value) + offset)
    except (TypeError, ValueError):
        if offset:
            raise ValueError(f"cannot apply repeat offset {offset} to non-integer repeat {value!r}")
        return str(value)


def match_key(job: dict[str, Any], *, repeat_offset: int = 0) -> tuple[str, str, str, str]:
    return (
        str(job.get("actor", "")),
        str(job.get("task", "")),
        str(job.get("item_id", "")),
        shifted_repeat(job.get("repeat", ""), repeat_offset),
    )


def load_run(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    jobs = read_jsonl(path)
    generations_path = path.parent / "generations.jsonl"
    if not generations_path.exists():
        raise FileNotFoundError(generations_path)
    generations = {str(row["output_id"]): row for row in read_jsonl(generations_path)}
    return jobs, generations


def digest_payload(payload: dict[str, Any], n: int = 12) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:n]


def bridge_pair_uid(amount_job: dict[str, Any], neutral_job: dict[str, Any]) -> str:
    digest = digest_payload(
        {
            "source_amount_pair_uid": amount_job["pair_uid"],
            "source_neutral_pair_uid": neutral_job["pair_uid"],
            "source_amount_condition": AMOUNT_HIGH,
            "source_neutral_condition": FRAMED_NEUTRAL,
        }
    )
    return (
        f"amount_neutral_bridge:{amount_job['actor']}:{amount_job['task']}:"
        f"i{amount_job.get('item_id', '')}:r{amount_job.get('repeat', '')}:v{digest}"
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
        "source_note": "Copied for judging-only amount-high versus framed-neutral bridge; no generation call was made.",
    }


def bridge_job(
    *,
    amount_job: dict[str, Any],
    neutral_job: dict[str, Any],
    amount_output: dict[str, Any],
    neutral_output: dict[str, Any],
    source_block: str,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    amount_side = output_side(amount_job, AMOUNT_HIGH)
    neutral_side = output_side(neutral_job, FRAMED_NEUTRAL)
    pair_uid = bridge_pair_uid(amount_job, neutral_job)
    return {
        "actor": amount_job["actor"],
        "actor_label": amount_job.get("actor_label") or ACTOR_LABEL.get(amount_job["actor"], amount_job["actor"]),
        "amount_high": amount_job.get("amount_high", ""),
        "amount_low": amount_job.get("amount_low", ""),
        "axis": amount_job.get("axis", ""),
        "axis_definition": amount_job.get("axis_definition", ""),
        "base_prompt": amount_job.get("base_prompt", ""),
        "comparison": "amount_high_vs_framed_neutral",
        "condition_a": AMOUNT_HIGH,
        "condition_b": FRAMED_NEUTRAL,
        "domain": amount_job.get("domain", ""),
        "domain_label": amount_job.get("domain_label", ""),
        "framing": amount_job.get("framing", ""),
        "item_id": amount_job.get("item_id", ""),
        "item_index": amount_job.get("item_index", ""),
        "item_label": amount_job.get("item_label", ""),
        "other_condition": FRAMED_NEUTRAL,
        "pair_uid": pair_uid,
        "predicted_condition": AMOUNT_HIGH,
        "prompt_a": amount_job.get(f"prompt_{amount_side}", ""),
        "prompt_b": neutral_job.get(f"prompt_{neutral_side}", ""),
        "repeat": amount_job.get("repeat", ""),
        "run_dir": str(run_dir),
        "run_generation_failures_path": str(run_dir / "generation_failures.jsonl"),
        "run_generations_path": str(run_dir / "generations.jsonl"),
        "run_id": run_id,
        "run_judge_votes_path": str(run_dir / "judge_votes.jsonl"),
        "run_manifest_path": str(run_dir / "generation_jobs.jsonl"),
        "source_amount_output_id": amount_output.get("output_id", ""),
        "source_amount_pair_uid": amount_job["pair_uid"],
        "source_amount_prompt": amount_job.get(f"prompt_{amount_side}", ""),
        "source_amount_run_dir": amount_job.get("run_dir", ""),
        "source_block": source_block,
        "source_neutral_output_id": neutral_output.get("output_id", ""),
        "source_neutral_pair_uid": neutral_job["pair_uid"],
        "source_neutral_prompt": neutral_job.get(f"prompt_{neutral_side}", ""),
        "source_neutral_run_dir": neutral_job.get("run_dir", ""),
        "source_note": "judging-only bridge from current amount-high and framed-neutral outputs",
        "system_prompt_a": amount_job.get(f"system_prompt_{amount_side}", ""),
        "system_prompt_b": neutral_job.get(f"system_prompt_{neutral_side}", ""),
        "task": amount_job["task"],
        "task_label": amount_job.get("task_label") or TASK_LABEL.get(amount_job["task"], amount_job["task"]),
        "topic_design": amount_job.get("topic_design", ""),
    }


def load_neutral_index(
    entries: list[dict[str, Any]],
    *,
    repeat_offset: int = 0,
) -> dict[tuple[str, str, str, str], tuple[dict[str, Any], dict[str, dict[str, Any]]]]:
    neutral_by_key: dict[tuple[str, str, str, str], tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}
    for entry in entries:
        jobs, generations = load_run(entry["manifest_path"])
        for job in jobs:
            if FRAMED_NEUTRAL not in {job.get("condition_a"), job.get("condition_b")}:
                continue
            if str(job.get("comparison", "")) != "framed_user_strong_headroom":
                continue
            key = match_key(job, repeat_offset=repeat_offset)
            if key in neutral_by_key:
                raise ValueError(f"duplicate framed-neutral job for key {key}")
            neutral_by_key[key] = (job, generations)
    return neutral_by_key


def prepare(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], Counter]:
    selected_tasks = set(csv_arg(args.tasks, TASKS))
    selected_actors = set(csv_arg(args.actors)) if args.actors else set()

    amount_entries = selected_entries(
        read_manifest_entries(AMOUNT_BASE_GLOB),
        selected_tasks=selected_tasks,
        selected_actors=selected_actors,
    )
    neutral_base_entries = selected_entries(
        read_manifest_entries(NEUTRAL_BASE_GLOB, exclude_repeat_blocks=True),
        selected_tasks=selected_tasks,
        selected_actors=selected_actors,
    )
    neutral_by_key = load_neutral_index(neutral_base_entries)
    source_manifest_globs = [AMOUNT_BASE_GLOB, NEUTRAL_BASE_GLOB]

    if args.source_scope == "current":
        highn_entries = selected_entries(
            read_manifest_entries(AMOUNT_HIGHN_GLOB),
            selected_tasks=selected_tasks,
            selected_actors=selected_actors,
        )
        amount_entries.extend(highn_entries)
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
        source_manifest_globs.extend([AMOUNT_HIGHN_GLOB, NEUTRAL_REPEAT_BLOCK_GLOB])

    if not amount_entries:
        raise ValueError("no amount manifests matched the requested actor/task filters")

    run_seed = {
        "actors": sorted({entry["actor"] for entry in amount_entries}),
        "tasks": sorted(selected_tasks),
        "source_scope": args.source_scope,
        "source_manifest_globs": source_manifest_globs,
        "bridge": "amount_high_vs_framed_neutral",
    }
    run_hash = digest_payload(run_seed)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    run_id = f"amount-neutral-bridge-{args.source_scope}__{timestamp}__hash-{run_hash}"
    run_dir = RUNS_DIR / run_id

    jobs_out: list[dict[str, Any]] = []
    generations_out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    seen_pair_uids: set[str] = set()

    for entry in amount_entries:
        jobs, generations = load_run(entry["manifest_path"])
        source_block = "base_amount_current_target_r0_r4" if "canonical_amount_base" in entry["manifest_list"] else "highn_extension_r5_r9"
        for amount_job in jobs:
            if amount_job.get("condition_a") != AMOUNT_HIGH or amount_job.get("condition_b") != "amount_low":
                continue
            if not str(amount_job.get("comparison", "")).endswith("_amount"):
                continue
            neutral_match = neutral_by_key.get(match_key(amount_job))
            if neutral_match is None:
                raise ValueError(f"no matched framed-neutral job for {amount_job['pair_uid']}")
            neutral_job, neutral_generations = neutral_match
            amount_output = generations.get(output_id(amount_job, AMOUNT_HIGH))
            neutral_output = neutral_generations.get(output_id(neutral_job, FRAMED_NEUTRAL))
            if amount_output is None or not str(amount_output.get("output_text", "")).strip():
                raise ValueError(f"missing amount_high output for {amount_job['pair_uid']}")
            if neutral_output is None or not str(neutral_output.get("output_text", "")).strip():
                raise ValueError(f"missing framed_neutral output for {neutral_job['pair_uid']}")

            job = bridge_job(
                amount_job=amount_job,
                neutral_job=neutral_job,
                amount_output=amount_output,
                neutral_output=neutral_output,
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
                    source=amount_output,
                    job=job,
                    suffix="a",
                    condition=AMOUNT_HIGH,
                    source_role=AMOUNT_HIGH,
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
            counts[(job["actor"], job["task"], source_block)] += 1

    if not jobs_out:
        raise ValueError("no amount-high versus framed-neutral bridge comparisons were prepared")
    return run_dir, jobs_out, generations_out, counts


def write_actor_manifest_pointers(jobs: list[dict[str, Any]], run_dir: Path) -> None:
    actors = sorted({str(job["actor"]) for job in jobs})
    if len(actors) == 1:
        actor = actors[0]
        pointer = RUNS_DIR / f"amount_neutral_bridge_manifests__{actor}.tsv"
        pointer.write_text(f"{actor}\t{run_dir / 'generation_jobs.jsonl'}\n", encoding="utf-8")
        return
    pointer = RUNS_DIR / "amount_neutral_bridge_manifests__all.tsv"
    pointer.write_text("\n".join(f"{actor}\t{run_dir / 'generation_jobs.jsonl'}" for actor in actors) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", default=None, help="Comma-separated actor ids, or all. Default: all.")
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks, or all. Default: all.")
    parser.add_argument(
        "--source-scope",
        choices=("base", "current"),
        default="current",
        help="base uses repeats 0-4 only. current uses amount repeats 0-9 and matched neutral controls.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing files.")
    args = parser.parse_args()

    run_dir, jobs, generations, counts = prepare(args)
    print(f"bridge judging jobs: {len(jobs)}")
    print(f"copied outputs: {len(generations)}")
    print(f"run_dir: {run_dir}")
    print("breakdown:")
    for key, count in sorted(counts.items()):
        actor, task, source_block = key
        print(f"  {actor} / {task} / {source_block}: {count}")

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
