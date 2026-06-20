#!/usr/bin/env python3
"""Prepare essay direct-instruction jobs with a mid-effort user prompt.

This is an exploratory comparison, not part of the canonical prompt book unless
explicitly promoted later. It matches the current framed user-prompt direct
instruction setup for essays:

- control: framed-neutral user prompt
- treatment: same framed-neutral user prompt, plus the mid-effort sentence at
  the end
- both system prompts blank

The script reuses the corrected framed-neutral control outputs from the latest
fund-wording headroom manifests, so running generation on the output manifest
generates only the treatment side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utility_behavior_gap.constants import ACTOR_LABEL, ACTORS, TASK_LABEL
from utility_behavior_gap.fingerprints import text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl, write_jsonl
from utility_behavior_gap.job_builder import prompt_variant_id
from utility_behavior_gap.paths import OUTPUT_API
from utility_behavior_gap.scripts.prepare_framed_user_strong_jobs import (
    output_id,
    output_side,
    read_fund_manifest_paths,
)


RUNS_DIR = OUTPUT_API / "runs"
TASK = "essay"
COMPARISON = "framed_user_mid_effort_essay"
TREATMENT_CONDITION = "framed_user_mid_effort"
NEUTRAL_CONDITION = "framed_neutral"
MID_EFFORT_TEXT = "This essay is extremely important - give your absolute best effort!"


def digest_payload(payload: dict[str, Any], n: int = 12) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:n]


def moved_user_prompt(base_prompt: str, instruction: str) -> str:
    return f"{base_prompt.rstrip()}\n\n{instruction.strip()}"


def csv_arg(value: str | None, allowed: tuple[str, ...] | list[str] | None = None) -> list[str]:
    if value is None or value.strip().lower() == "all":
        return list(allowed or ())
    values = [part.strip() for part in value.split(",") if part.strip()]
    if allowed is not None:
        bad = sorted(set(values) - set(allowed))
        if bad:
            raise ValueError(f"unknown value(s): {', '.join(bad)}")
    return values


def new_pair_uid(source_job: dict[str, Any]) -> str:
    digest = digest_payload(
        {
            "source_pair_uid": source_job["pair_uid"],
            "comparison": COMPARISON,
            "mid_effort_text": MID_EFFORT_TEXT,
        }
    )
    return (
        f"{COMPARISON}:{source_job['actor']}:{source_job['task']}:"
        f"i{source_job.get('item_id', '')}:r{source_job.get('repeat', '')}:v{digest}"
    )


def copied_neutral_generation(source: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    text = str(source.get("output_text", ""))
    return {
        "actor": job["actor"],
        "condition": NEUTRAL_CONDITION,
        "finish_reason": source.get("finish_reason", ""),
        "job": job,
        "model": source.get("model", ""),
        "output_id": f"{job['pair_uid']}::b",
        "output_text": text,
        "pair_uid": job["pair_uid"],
        "run_id": job["run_id"],
        "success": True,
        "source_condition": source.get("condition", ""),
        "source_output_hash": text_fingerprint(text),
        "source_output_id": source.get("output_id", ""),
        "source_pair_uid": source.get("pair_uid", ""),
        "source_role": NEUTRAL_CONDITION,
        "source_run_dir": source.get("job", {}).get("run_dir", ""),
        "source_run_id": source.get("run_id", ""),
        "source_note": (
            "Copied framed-neutral control for exploratory framed user-prompt "
            "mid-effort essay run; no generation call was made."
        ),
    }


def build_job(
    source_job: dict[str, Any],
    neutral_output: dict[str, Any],
    *,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    neutral_side = output_side(source_job, NEUTRAL_CONDITION)
    neutral_prompt = str(source_job.get(f"prompt_{neutral_side}", ""))
    prompt_a = moved_user_prompt(neutral_prompt, MID_EFFORT_TEXT)
    prompt_b = neutral_prompt
    pair_uid = new_pair_uid(source_job)
    actor = str(source_job["actor"])
    task = str(source_job["task"])
    return {
        "actor": actor,
        "actor_label": source_job.get("actor_label", ACTOR_LABEL.get(actor, actor)),
        "axis": source_job.get("axis", ""),
        "axis_definition": source_job.get("axis_definition", ""),
        "base_prompt": source_job.get("base_prompt", ""),
        "comparison": COMPARISON,
        "condition_a": TREATMENT_CONDITION,
        "condition_b": NEUTRAL_CONDITION,
        "domain": source_job.get("domain", ""),
        "domain_label": source_job.get("domain_label", ""),
        "framing": source_job.get("framing", ""),
        "item_id": source_job.get("item_id", ""),
        "item_index": source_job.get("item_index", ""),
        "item_label": source_job.get("item_label", ""),
        "mid_effort_text": MID_EFFORT_TEXT,
        "other_condition": NEUTRAL_CONDITION,
        "pair_uid": pair_uid,
        "predicted_condition": TREATMENT_CONDITION,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "prompt_variant_id": prompt_variant_id(
            system_prompt_a="",
            system_prompt_b="",
            prompt_a=prompt_a,
            prompt_b=prompt_b,
        ),
        "repeat": source_job.get("repeat", ""),
        "run_dir": str(run_dir),
        "run_generation_failures_path": str(run_dir / "generation_failures.jsonl"),
        "run_generations_path": str(run_dir / "generations.jsonl"),
        "run_id": run_id,
        "run_judge_votes_path": str(run_dir / "judge_votes.jsonl"),
        "run_manifest_path": str(run_dir / "generation_jobs.jsonl"),
        "source_headroom_pair_uid": source_job["pair_uid"],
        "source_neutral_output_id": neutral_output.get("output_id", ""),
        "source_neutral_prompt": neutral_prompt,
        "source_neutral_run_dir": source_job.get("run_dir", ""),
        "source_note": (
            "Exploratory mid-effort direct-instruction essay run; treatment "
            "uses the framed-neutral user prompt plus the mid-effort sentence."
        ),
        "system_prompt_a": "",
        "system_prompt_b": "",
        "task": task,
        "task_label": source_job.get("task_label", TASK_LABEL.get(task, task)),
        "topic_design": source_job.get("topic_design", ""),
    }


def run_id_for(actor: str, run_hash: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    return f"{TASK}__{COMPARISON}__{actor}__{timestamp}__hash-{run_hash}"


def prepare(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], Counter]:
    actors = csv_arg(args.actors, ACTORS)
    if len(actors) != 1:
        raise ValueError("prepare exactly one actor per run; pass a single --actors value")
    actor = actors[0]
    fund_paths = read_fund_manifest_paths(actor)
    if TASK not in fund_paths:
        raise ValueError(f"no latest fund-wording manifest for {actor}/{TASK}")

    run_seed = {
        "actor": actor,
        "task": TASK,
        "comparison": COMPARISON,
        "mid_effort_text": MID_EFFORT_TEXT,
        "source": "latest fund-wording framed-neutral controls",
    }
    run_hash = digest_payload(run_seed)
    run_id = run_id_for(actor, run_hash)
    run_dir = RUNS_DIR / run_id

    manifest_path = fund_paths[TASK]
    source_jobs = read_jsonl(manifest_path)
    source_generations = {
        row["output_id"]: row for row in read_jsonl(manifest_path.parent / "generations.jsonl")
    }

    jobs_out: list[dict[str, Any]] = []
    generations_out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    seen: set[str] = set()
    for source_job in source_jobs:
        if not str(source_job.get("comparison", "")).endswith("_headroom"):
            continue
        if source_job.get("condition_a") != "framed_strong" or source_job.get("condition_b") != NEUTRAL_CONDITION:
            continue
        neutral_output = source_generations.get(output_id(source_job, NEUTRAL_CONDITION))
        if neutral_output is None or not str(neutral_output.get("output_text", "")).strip():
            raise ValueError(f"missing framed-neutral output for {source_job['pair_uid']}")
        job = build_job(source_job, neutral_output, run_id=run_id, run_dir=run_dir)
        if job["pair_uid"] in seen:
            raise ValueError(f"duplicate pair_uid: {job['pair_uid']}")
        seen.add(job["pair_uid"])
        jobs_out.append(job)
        generations_out.append(copied_neutral_generation(neutral_output, job))
        counts[TASK] += 1

    if not jobs_out:
        raise ValueError(f"no mid-effort essay jobs were prepared for {actor}")
    return run_dir, jobs_out, generations_out, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", required=True, help="One actor id.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing files.")
    args = parser.parse_args()

    run_dir, jobs, generations, counts = prepare(args)
    print(f"generation jobs: {len(jobs)}")
    print(f"prefilled neutral outputs: {len(generations)}")
    print(f"new treatment generations needed: {len(jobs)}")
    print(f"new total generations needed: {2 * len(jobs) - len(generations)}")
    print(f"run_dir: {run_dir}")
    print(f"mid_effort_text: {MID_EFFORT_TEXT}")
    print("breakdown:")
    for task, count in sorted(counts.items()):
        print(f"  {task}: {count}")

    if args.dry_run:
        print("dry run: wrote nothing")
        return

    write_jsonl(run_dir / "generation_jobs.jsonl", jobs)
    write_jsonl(run_dir / "generations.jsonl", generations)
    print(f"wrote {run_dir / 'generation_jobs.jsonl'}")
    print(f"prefilled {run_dir / 'generations.jsonl'}")


if __name__ == "__main__":
    main()
