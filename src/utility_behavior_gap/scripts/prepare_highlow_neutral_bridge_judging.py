#!/usr/bin/env python3
"""Prepare judging-only high/low utility versus framed-neutral comparisons.

This script does not call any model. It builds a run-local judging manifest by
copying existing generated outputs from the corrected fund-wording rerun:

* high utility output versus the matched framed-neutral output
* low utility output versus the matched framed-neutral output

The matched framed-neutral output is the headroom/control output with the same
actor, task, item, and repeat as the high-low output.
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
from utility_behavior_gap.paths import OUTPUT_API


RUNS_DIR = OUTPUT_API / "runs"
LATEST_PATH = RUNS_DIR / "highlow_neutral_bridge_latest.txt"
BASE_MANIFEST_GLOB = "fund_wording_rerun_manifests__*.tsv"
HIGHN_MANIFEST_GLOB = "canonical_highn10_manifests__*.tsv"
NEUTRAL_REPEAT_BLOCK_GLOB = "framed_user_strong_manifests__repeat-block01__*.tsv"
TASKS = ("essay", "translation", "incident_postmortem", "grant_proposal_abstract")
SIDES = ("high", "low")
SIDE_CONDITIONS = {"high": "hl_high", "low": "hl_low"}
NEUTRAL_CONDITION = "framed_neutral"


def csv_arg(value: str | None, allowed: tuple[str, ...] | None = None) -> list[str]:
    if value is None or value.strip().lower() == "all":
        return list(allowed or ())
    out = [part.strip() for part in value.split(",") if part.strip()]
    if allowed is not None:
        bad = sorted(set(out) - set(allowed))
        if bad:
            raise ValueError(f"unknown value(s): {', '.join(bad)}")
    return out


def read_manifest_entries(glob: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for tsv in sorted(RUNS_DIR.glob(glob)):
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


def bridge_pair_uid(highlow_job: dict[str, Any], neutral_job: dict[str, Any], side: str) -> str:
    digest = digest_payload(
        {
            "source_highlow_pair_uid": highlow_job["pair_uid"],
            "source_neutral_pair_uid": neutral_job["pair_uid"],
            "side": side,
        }
    )
    return (
        f"highlow_neutral_bridge:{side}:{highlow_job['actor']}:{highlow_job['task']}:"
        f"p{highlow_job.get('pair_idx', '')}:r{highlow_job.get('repeat', '')}:v{digest}"
    )


def copied_generation(
    *,
    source: dict[str, Any],
    job: dict[str, Any],
    output_suffix: str,
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
        "output_id": f"{job['pair_uid']}::{output_suffix}",
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
        "source_note": "Copied for judging-only high/low versus framed-neutral bridge; no generation call was made.",
    }


def bridge_job(
    *,
    highlow_job: dict[str, Any],
    neutral_job: dict[str, Any],
    highlow_output: dict[str, Any],
    neutral_output: dict[str, Any],
    side: str,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    side_condition = SIDE_CONDITIONS[side]
    pair_uid = bridge_pair_uid(highlow_job, neutral_job, side)
    prompt_field = f"prompt_{output_side(highlow_job, side_condition)}"
    neutral_prompt_field = f"prompt_{output_side(neutral_job, NEUTRAL_CONDITION)}"
    return {
        "actor": highlow_job["actor"],
        "actor_label": highlow_job.get("actor_label", highlow_job["actor"]),
        "axis": highlow_job.get("axis", ""),
        "axis_definition": highlow_job.get("axis_definition", ""),
        "base_prompt": highlow_job.get("base_prompt", ""),
        "comparison": f"highlow_{side}_vs_framed_neutral",
        "condition_a": side_condition,
        "condition_b": NEUTRAL_CONDITION,
        "delta_u": highlow_job.get("delta_u", ""),
        "domain": highlow_job.get("domain", ""),
        "domain_label": highlow_job.get("domain_label", ""),
        "framing": highlow_job.get("framing", ""),
        "high_consequence": highlow_job.get("high_consequence", ""),
        "high_description": highlow_job.get("high_description", ""),
        "high_utility": highlow_job.get("high_utility", ""),
        "item_id": highlow_job.get("item_id", ""),
        "item_index": highlow_job.get("item_index", ""),
        "item_label": highlow_job.get("item_label", ""),
        "low_consequence": highlow_job.get("low_consequence", ""),
        "low_description": highlow_job.get("low_description", ""),
        "low_utility": highlow_job.get("low_utility", ""),
        "other_condition": NEUTRAL_CONDITION,
        "pair_idx": highlow_job.get("pair_idx", ""),
        "pair_set": highlow_job.get("pair_set", ""),
        "pair_uid": pair_uid,
        "predicted_condition": side_condition,
        "prompt_a": highlow_job.get(prompt_field, ""),
        "prompt_b": neutral_job.get(neutral_prompt_field, ""),
        "repeat": highlow_job.get("repeat", ""),
        "run_dir": str(run_dir),
        "run_generation_failures_path": str(run_dir / "generation_failures.jsonl"),
        "run_generations_path": str(run_dir / "generations.jsonl"),
        "run_id": run_id,
        "run_judge_votes_path": str(run_dir / "judge_votes.jsonl"),
        "run_manifest_path": str(run_dir / "generation_jobs.jsonl"),
        "side": side,
        "side_condition": side_condition,
        "source_highlow_output_id": highlow_output.get("output_id", ""),
        "source_highlow_pair_uid": highlow_job["pair_uid"],
        "source_highlow_prompt": highlow_job.get(prompt_field, ""),
        "source_highlow_run_dir": highlow_job.get("run_dir", ""),
        "source_neutral_output_id": neutral_output.get("output_id", ""),
        "source_neutral_pair_uid": neutral_job["pair_uid"],
        "source_neutral_prompt": neutral_job.get(neutral_prompt_field, ""),
        "source_neutral_run_dir": neutral_job.get("run_dir", ""),
        "source_note": "judging-only bridge from corrected fund-wording rerun outputs",
        "system_prompt_a": highlow_job.get(f"system_prompt_{output_side(highlow_job, side_condition)}", ""),
        "system_prompt_b": neutral_job.get(f"system_prompt_{output_side(neutral_job, NEUTRAL_CONDITION)}", ""),
        "task": highlow_job["task"],
        "task_label": highlow_job.get("task_label", highlow_job["task"]),
        "topic_design": highlow_job.get("topic_design", ""),
    }


def selected_entries(
    entries: list[dict[str, Any]],
    *,
    selected_tasks: set[str],
    selected_actors: set[str],
) -> list[dict[str, Any]]:
    return [
        entry
        for entry in entries
        if entry["task"] in selected_tasks
        and (not selected_actors or entry["actor"] in selected_actors)
    ]


def load_neutral_index(
    entries: list[dict[str, Any]],
    *,
    repeat_offset: int = 0,
) -> dict[tuple[str, str, str, str], tuple[dict[str, Any], dict[str, dict[str, Any]]]]:
    neutral_by_key: dict[tuple[str, str, str, str], tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}
    for entry in entries:
        jobs, generations = load_run(entry["manifest_path"])
        for job in jobs:
            if NEUTRAL_CONDITION not in {job.get("condition_a"), job.get("condition_b")}:
                continue
            if not str(job.get("comparison", "")).endswith("_headroom"):
                continue
            key = match_key(job, repeat_offset=repeat_offset)
            if key in neutral_by_key:
                raise ValueError(f"duplicate neutral/headroom job for key {key}")
            neutral_by_key[key] = (job, generations)
    return neutral_by_key


def prepare(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], Counter]:
    selected_tasks = set(csv_arg(args.tasks, TASKS))
    selected_sides = set(csv_arg(args.sides, SIDES))
    selected_domains = set(csv_arg(args.domains)) if args.domains else set()
    selected_actors = set(csv_arg(args.actors)) if args.actors else set()

    base_entries = selected_entries(
        read_manifest_entries(BASE_MANIFEST_GLOB),
        selected_tasks=selected_tasks,
        selected_actors=selected_actors,
    )
    highlow_entries = list(base_entries)
    neutral_by_key = load_neutral_index(base_entries)
    source_manifest_globs = [BASE_MANIFEST_GLOB]

    if args.source_scope == "current":
        highn_entries = selected_entries(
            read_manifest_entries(HIGHN_MANIFEST_GLOB),
            selected_tasks=selected_tasks,
            selected_actors=selected_actors,
        )
        repeat_block_entries = selected_entries(
            read_manifest_entries(NEUTRAL_REPEAT_BLOCK_GLOB),
            selected_tasks=selected_tasks,
            selected_actors=selected_actors,
        )
        highlow_entries.extend(highn_entries)
        source_manifest_globs.extend([HIGHN_MANIFEST_GLOB, NEUTRAL_REPEAT_BLOCK_GLOB])
        repeat_block_neutrals = load_neutral_index(repeat_block_entries, repeat_offset=5)
        conflicts = set(neutral_by_key).intersection(repeat_block_neutrals)
        if conflicts:
            preview = sorted(conflicts)[:3]
            raise ValueError(f"neutral key collision between base and repeat-block sources: {preview}")
        neutral_by_key.update(repeat_block_neutrals)

    if not highlow_entries:
        raise ValueError("no rerun manifests matched the requested actor/task filters")

    run_seed = {
        "actors": sorted({entry["actor"] for entry in highlow_entries}),
        "tasks": sorted(selected_tasks),
        "domains": sorted(selected_domains) or ["all"],
        "sides": sorted(selected_sides),
        "source_scope": args.source_scope,
        "source_manifest_globs": source_manifest_globs,
    }
    run_hash = digest_payload(run_seed)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    scope_part = "current" if args.source_scope == "current" else "base"
    run_id = f"highlow-neutral-bridge-{scope_part}__{timestamp}__hash-{run_hash}"
    run_dir = RUNS_DIR / run_id

    jobs_out: list[dict[str, Any]] = []
    generations_out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    seen_pair_uids: set[str] = set()

    for entry in highlow_entries:
        jobs, generations = load_run(entry["manifest_path"])
        for highlow_job in jobs:
            if highlow_job.get("condition_a") != "hl_high" or highlow_job.get("condition_b") != "hl_low":
                continue
            if not str(highlow_job.get("comparison", "")).endswith("_highlow"):
                continue
            if selected_domains and highlow_job.get("domain") not in selected_domains:
                continue
            neutral_match = neutral_by_key.get(match_key(highlow_job))
            if neutral_match is None:
                raise ValueError(f"no matched framed-neutral job for {highlow_job['pair_uid']}")
            neutral_job, neutral_generations = neutral_match
            neutral_output = neutral_generations.get(output_id(neutral_job, NEUTRAL_CONDITION))
            if neutral_output is None or not str(neutral_output.get("output_text", "")).strip():
                raise ValueError(f"missing framed-neutral output for {neutral_job['pair_uid']}")

            for side in selected_sides:
                side_condition = SIDE_CONDITIONS[side]
                highlow_output = generations.get(output_id(highlow_job, side_condition))
                if highlow_output is None or not str(highlow_output.get("output_text", "")).strip():
                    raise ValueError(f"missing {side_condition} output for {highlow_job['pair_uid']}")
                job = bridge_job(
                    highlow_job=highlow_job,
                    neutral_job=neutral_job,
                    highlow_output=highlow_output,
                    neutral_output=neutral_output,
                    side=side,
                    run_id=run_id,
                    run_dir=run_dir,
                )
                if job["pair_uid"] in seen_pair_uids:
                    raise ValueError(f"duplicate bridge pair_uid: {job['pair_uid']}")
                seen_pair_uids.add(job["pair_uid"])
                jobs_out.append(job)
                generations_out.append(
                    copied_generation(
                        source=highlow_output,
                        job=job,
                        output_suffix="a",
                        condition=side_condition,
                        source_role=side,
                    )
                )
                generations_out.append(
                    copied_generation(
                        source=neutral_output,
                        job=job,
                        output_suffix="b",
                        condition=NEUTRAL_CONDITION,
                        source_role="framed_neutral",
                    )
                )
                counts[(job["actor"], job["task"], job["domain"], side)] += 1

    if not jobs_out:
        raise ValueError("no bridge comparisons were prepared")
    return run_dir, jobs_out, generations_out, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", default=None, help="Comma-separated actor ids, or all. Default: all.")
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks, or all. Default: all.")
    parser.add_argument(
        "--domains",
        default="all",
        help="Comma-separated high-low domains, or all. Default: all.",
    )
    parser.add_argument("--sides", default="high,low", help="high, low, or high,low. Default: high,low.")
    parser.add_argument(
        "--source-scope",
        choices=("base", "current"),
        default="base",
        help=(
            "base uses fund-wording repeats 0-4 only. current adds canonical high-N repeats 5-9 "
            "and matches them to fresh framed-neutral repeat-block controls."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing files.")
    args = parser.parse_args()

    run_dir, jobs, generations, counts = prepare(args)

    print(f"bridge judging jobs: {len(jobs)}")
    print(f"copied outputs: {len(generations)}")
    print(f"run_dir: {run_dir}")
    print("breakdown:")
    for key, count in sorted(counts.items()):
        actor, task, domain, side = key
        print(f"  {actor} / {task} / {domain} / {side}: {count}")

    if args.dry_run:
        print("dry run: wrote nothing")
        return

    write_jsonl(run_dir / "generation_jobs.jsonl", jobs)
    write_jsonl(run_dir / "generations.jsonl", generations)
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(str(run_dir) + "\n", encoding="utf-8")
    print(f"wrote {run_dir / 'generation_jobs.jsonl'}")
    print(f"wrote {run_dir / 'generations.jsonl'}")
    print(f"wrote latest pointer: {LATEST_PATH}")
    print(f"next: python -m utility_behavior_gap.scripts.run_judging --run-dir {run_dir} --orders both --workers 5")


if __name__ == "__main__":
    main()
