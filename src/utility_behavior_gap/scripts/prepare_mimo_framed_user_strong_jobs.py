#!/usr/bin/env python3
"""Prepare MiMo framed-neutral headroom jobs with the strong cue in the user prompt.

This is the same corrected fund-wording headroom setup, except the treatment
side has the task-specific max-effort text appended to the end of the user
prompt and uses a blank system prompt. The control side is the existing
framed-neutral output from the corrected fund-wording rerun.

The script writes a normal generation manifest, but pre-populates the run-local
generations.jsonl with copied framed-neutral controls. Running run_generation on
the manifest should therefore generate only the treatment side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utility_behavior_gap.fingerprints import text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl, write_jsonl
from utility_behavior_gap.job_builder import prompt_variant_id
from utility_behavior_gap.paths import OUTPUT_API


ACTOR = "mimo-v25-pro-or"
RUNS_DIR = OUTPUT_API / "runs"
LATEST_PATH = RUNS_DIR / "mimo_framed_user_strong_latest.txt"
FUND_MANIFEST = RUNS_DIR / f"fund_wording_rerun_manifests__{ACTOR}.tsv"
TASKS = ("essay", "translation", "incident_postmortem", "grant_proposal_abstract")
STRONG_CONDITION = "framed_user_strong"
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


def moved_user_prompt(base_prompt: str, strong_text: str) -> str:
    return f"{base_prompt.rstrip()}\n\n{strong_text.strip()}"


def new_pair_uid(source_job: dict[str, Any]) -> str:
    digest = digest_payload(
        {
            "source_pair_uid": source_job["pair_uid"],
            "comparison": "mimo_framed_user_strong_headroom",
        }
    )
    return (
        f"mimo_framed_user_strong:{source_job['task']}:"
        f"i{source_job.get('item_id', '')}:r{source_job.get('repeat', '')}:v{digest}"
    )


def copied_neutral_generation(source: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    text = str(source.get("output_text", ""))
    return {
        "actor": ACTOR,
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
        "source_note": "Copied framed-neutral control for MiMo user-prompt-strong headroom run; no generation call was made.",
    }


def build_job(source_job: dict[str, Any], neutral_output: dict[str, Any], *, run_id: str, run_dir: Path) -> dict[str, Any]:
    neutral_side = output_side(source_job, NEUTRAL_CONDITION)
    strong_side = output_side(source_job, "framed_strong")
    neutral_prompt = str(source_job.get(f"prompt_{neutral_side}", ""))
    strong_text = str(source_job.get(f"system_prompt_{strong_side}", ""))
    prompt_a = moved_user_prompt(neutral_prompt, strong_text)
    prompt_b = neutral_prompt
    pair_uid = new_pair_uid(source_job)
    return {
        "actor": ACTOR,
        "actor_label": source_job.get("actor_label", "MiMo V2.5 Pro"),
        "axis": source_job.get("axis", ""),
        "axis_definition": source_job.get("axis_definition", ""),
        "base_prompt": source_job.get("base_prompt", ""),
        "comparison": "mimo_framed_user_strong_headroom",
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
        "source_note": "MiMo headroom rerun with the former system-prompt max-effort cue appended to the user prompt.",
        "system_prompt_a": "",
        "system_prompt_b": "",
        "task": source_job["task"],
        "task_label": source_job.get("task_label", source_job["task"]),
        "topic_design": source_job.get("topic_design", ""),
    }


def prepare(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], Counter]:
    selected_tasks = set(csv_arg(args.tasks, TASKS))
    fund_paths = read_fund_manifest_paths()
    missing_tasks = sorted(selected_tasks - set(fund_paths))
    if missing_tasks:
        raise ValueError(f"no latest fund-wording manifest for task(s): {', '.join(missing_tasks)}")

    run_seed = {
        "actor": ACTOR,
        "tasks": sorted(selected_tasks),
        "source": "latest fund-wording headroom controls; max-effort cue moved from system to user prompt",
    }
    run_hash = digest_payload(run_seed)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    run_id = f"mimo-framed-user-strong__{timestamp}__hash-{run_hash}"
    run_dir = RUNS_DIR / run_id

    jobs_out: list[dict[str, Any]] = []
    generations_out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    seen: set[str] = set()

    for task in TASKS:
        if task not in selected_tasks:
            continue
        manifest_path = fund_paths[task]
        jobs = read_jsonl(manifest_path)
        generations = {row["output_id"]: row for row in read_jsonl(manifest_path.parent / "generations.jsonl")}
        for source_job in jobs:
            if not str(source_job.get("comparison", "")).endswith("_headroom"):
                continue
            if source_job.get("condition_a") != "framed_strong" or source_job.get("condition_b") != NEUTRAL_CONDITION:
                continue
            neutral_output = generations.get(output_id(source_job, NEUTRAL_CONDITION))
            if neutral_output is None or not str(neutral_output.get("output_text", "")).strip():
                raise ValueError(f"missing framed-neutral output for {source_job['pair_uid']}")
            job = build_job(source_job, neutral_output, run_id=run_id, run_dir=run_dir)
            if job["pair_uid"] in seen:
                raise ValueError(f"duplicate pair_uid: {job['pair_uid']}")
            seen.add(job["pair_uid"])
            jobs_out.append(job)
            generations_out.append(copied_neutral_generation(neutral_output, job))
            counts[task] += 1

    if not jobs_out:
        raise ValueError("no MiMo user-prompt-strong jobs were prepared")
    return run_dir, jobs_out, generations_out, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks, or all. Default: all.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing files.")
    args = parser.parse_args()

    run_dir, jobs, generations, counts = prepare(args)
    print(f"generation jobs: {len(jobs)}")
    print(f"prefilled neutral outputs: {len(generations)}")
    print(f"new treatment generations needed: {len(jobs)}")
    print(f"run_dir: {run_dir}")
    print("breakdown:")
    for task, count in sorted(counts.items()):
        print(f"  {task}: {count}")

    if args.dry_run:
        print("dry run: wrote nothing")
        return

    write_jsonl(run_dir / "generation_jobs.jsonl", jobs)
    write_jsonl(run_dir / "generations.jsonl", generations)
    LATEST_PATH.write_text(str(run_dir) + "\n", encoding="utf-8")
    print(f"wrote {run_dir / 'generation_jobs.jsonl'}")
    print(f"prefilled {run_dir / 'generations.jsonl'}")
    print(f"wrote latest pointer: {LATEST_PATH}")
    print(f"next generation: python -m utility_behavior_gap.scripts.run_generation --jobs {run_dir / 'generation_jobs.jsonl'} --workers 10")
    print(f"next judging: python -m utility_behavior_gap.scripts.run_judging --run-dir {run_dir} --orders both --workers 10")


if __name__ == "__main__":
    main()
