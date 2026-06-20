#!/usr/bin/env python3
"""Prepare framed-neutral direct-instruction jobs with the strong cue in the user prompt.

By default this reuses the corrected fund-wording framed-neutral controls from
the modgrid headroom runs. The treatment side uses the same framed-neutral user
prompt, with the task-specific max-effort text appended to the end of the user
prompt. Both sides use blank system prompts.

The script writes one run-local manifest and pre-populates its generations log
with the copied framed-neutral controls. Running run_generation on the manifest
therefore generates only the treatment side.

For additional repeat blocks, pass --repeat-block and --fresh-neutral. That
keeps the exact same prompts but gives every pair a new id and generates both
the strong and neutral arms from scratch, rather than reusing the old neutral
outputs.
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


RUNS_DIR = OUTPUT_API / "runs"
TASKS = ("essay", "translation", "incident_postmortem", "grant_proposal_abstract")
COMPARISON = "framed_user_strong_headroom"
STRONG_CONDITION = "framed_user_strong"
NEUTRAL_CONDITION = "framed_neutral"


def csv_arg(value: str | None, allowed: tuple[str, ...] | list[str] | None = None) -> list[str]:
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


def read_fund_manifest_paths(actor: str) -> dict[str, Path]:
    manifest_list = RUNS_DIR / f"fund_wording_rerun_manifests__{actor}.tsv"
    if not manifest_list.exists():
        raise FileNotFoundError(manifest_list)
    paths: dict[str, Path] = {}
    for line in manifest_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row_actor, task, manifest = line.split("\t")
        if row_actor != actor:
            continue
        paths[task] = Path(manifest)
    return paths


def moved_user_prompt(base_prompt: str, strong_text: str) -> str:
    return f"{base_prompt.rstrip()}\n\n{strong_text.strip()}"


def new_pair_uid(source_job: dict[str, Any], *, repeat_block: str = "") -> str:
    digest = digest_payload(
        {
            "source_pair_uid": source_job["pair_uid"],
            "comparison": COMPARISON,
            "repeat_block": repeat_block,
        }
    )
    block_part = f":block{repeat_block}" if repeat_block else ""
    return (
        f"{COMPARISON}:{source_job['actor']}:{source_job['task']}:"
        f"i{source_job.get('item_id', '')}:r{source_job.get('repeat', '')}{block_part}:v{digest}"
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
            "Copied framed-neutral control for framed user-prompt-strong headroom run; "
            "no generation call was made."
        ),
    }


def build_job(
    source_job: dict[str, Any],
    neutral_output: dict[str, Any],
    *,
    run_id: str,
    run_dir: Path,
    repeat_block: str = "",
    fresh_neutral: bool = False,
) -> dict[str, Any]:
    neutral_side = output_side(source_job, NEUTRAL_CONDITION)
    strong_side = output_side(source_job, "framed_strong")
    neutral_prompt = str(source_job.get(f"prompt_{neutral_side}", ""))
    strong_text = str(source_job.get(f"system_prompt_{strong_side}", ""))
    prompt_a = moved_user_prompt(neutral_prompt, strong_text)
    prompt_b = neutral_prompt
    pair_uid = new_pair_uid(source_job, repeat_block=repeat_block)
    actor = str(source_job["actor"])
    task = str(source_job["task"])
    source_note = (
        "Fresh framed user-prompt-strong repeat block; both framed-user-strong "
        "and framed-neutral arms are generated from scratch."
        if fresh_neutral
        else (
            "Fund-wording headroom rerun with the former system-prompt max-effort cue "
            "appended to the user prompt."
        )
    )
    return {
        "actor": actor,
        "actor_label": source_job.get("actor_label", ACTOR_LABEL.get(actor, actor)),
        "axis": source_job.get("axis", ""),
        "axis_definition": source_job.get("axis_definition", ""),
        "base_prompt": source_job.get("base_prompt", ""),
        "comparison": COMPARISON,
        "condition_a": STRONG_CONDITION,
        "condition_b": NEUTRAL_CONDITION,
        "domain": source_job.get("domain", ""),
        "domain_label": source_job.get("domain_label", ""),
        "framing": source_job.get("framing", ""),
        "item_id": source_job.get("item_id", ""),
        "item_index": source_job.get("item_index", ""),
        "item_label": source_job.get("item_label", ""),
        "other_condition": NEUTRAL_CONDITION,
        "pair_uid": pair_uid,
        "predicted_condition": STRONG_CONDITION,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "prompt_variant_id": prompt_variant_id(
            system_prompt_a="",
            system_prompt_b="",
            prompt_a=prompt_a,
            prompt_b=prompt_b,
        ),
        "repeat": source_job.get("repeat", ""),
        "repeat_block": repeat_block,
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
        "source_strong_system_prompt": strong_text,
        "source_note": source_note,
        "system_prompt_a": "",
        "system_prompt_b": "",
        "task": task,
        "task_label": source_job.get("task_label", TASK_LABEL.get(task, task)),
        "topic_design": source_job.get("topic_design", ""),
    }


def run_id_for(actor: str, task: str, run_hash: str, *, repeat_block: str = "") -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    block = f"__repeat-{repeat_block}" if repeat_block else ""
    return f"{task}__{COMPARISON}{block}__{actor}__{timestamp}__hash-{run_hash}"


def prepare(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], Counter]:
    actors = csv_arg(args.actors, ACTORS)
    tasks = csv_arg(args.tasks, TASKS)
    if len(actors) != 1:
        raise ValueError("prepare exactly one actor per run; pass a single --actors value")
    if len(tasks) != 1:
        raise ValueError("prepare exactly one task per run; pass a single --tasks value")
    actor = actors[0]
    task = tasks[0]
    repeat_block = args.repeat_block.strip()
    if args.fresh_neutral and not repeat_block:
        raise ValueError("--fresh-neutral requires --repeat-block so new pair ids cannot collide with existing runs")
    fund_paths = read_fund_manifest_paths(actor)
    if task not in fund_paths:
        raise ValueError(f"no latest fund-wording manifest for {actor}/{task}")

    run_seed = {
        "actor": actor,
        "task": task,
        "comparison": COMPARISON,
        "source": "latest fund-wording headroom controls; max-effort cue moved from system to user prompt",
        "repeat_block": repeat_block,
        "fresh_neutral": bool(args.fresh_neutral),
    }
    run_hash = digest_payload(run_seed)
    run_id = run_id_for(actor, task, run_hash, repeat_block=repeat_block)
    run_dir = RUNS_DIR / run_id

    manifest_path = fund_paths[task]
    source_jobs = read_jsonl(manifest_path)
    source_generations = (
        {}
        if args.fresh_neutral
        else {row["output_id"]: row for row in read_jsonl(manifest_path.parent / "generations.jsonl")}
    )

    jobs_out: list[dict[str, Any]] = []
    generations_out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    seen: set[str] = set()
    for source_job in source_jobs:
        if not str(source_job.get("comparison", "")).endswith("_headroom"):
            continue
        if source_job.get("condition_a") != "framed_strong" or source_job.get("condition_b") != NEUTRAL_CONDITION:
            continue
        neutral_output = {} if args.fresh_neutral else source_generations.get(output_id(source_job, NEUTRAL_CONDITION))
        if not args.fresh_neutral and (neutral_output is None or not str(neutral_output.get("output_text", "")).strip()):
            raise ValueError(f"missing framed-neutral output for {source_job['pair_uid']}")
        job = build_job(
            source_job,
            neutral_output,
            run_id=run_id,
            run_dir=run_dir,
            repeat_block=repeat_block,
            fresh_neutral=bool(args.fresh_neutral),
        )
        if job["pair_uid"] in seen:
            raise ValueError(f"duplicate pair_uid: {job['pair_uid']}")
        seen.add(job["pair_uid"])
        jobs_out.append(job)
        if not args.fresh_neutral:
            generations_out.append(copied_neutral_generation(neutral_output, job))
        counts[task] += 1

    if not jobs_out:
        raise ValueError(f"no framed-user-strong jobs were prepared for {actor}/{task}")
    return run_dir, jobs_out, generations_out, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", required=True, help="One actor id.")
    parser.add_argument("--tasks", required=True, help="One task id.")
    parser.add_argument("--repeat-block", default="", help="Unique id for an additional repeat block.")
    parser.add_argument(
        "--fresh-neutral",
        action="store_true",
        help="Generate both arms from scratch instead of copying the existing framed-neutral output.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing files.")
    args = parser.parse_args()

    run_dir, jobs, generations, counts = prepare(args)
    print(f"generation jobs: {len(jobs)}")
    print(f"prefilled neutral outputs: {len(generations)}")
    print(f"new treatment generations needed: {len(jobs)}")
    print(f"new total generations needed: {2 * len(jobs) - len(generations)}")
    print(f"run_dir: {run_dir}")
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
