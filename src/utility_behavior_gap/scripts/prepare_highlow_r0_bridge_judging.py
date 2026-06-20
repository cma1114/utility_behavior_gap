#!/usr/bin/env python3
"""Prepare judging-only high/low utility versus R0 bare-task comparisons.

This script does not call any model. It builds a run-local judging manifest by
copying existing generated outputs:

* corrected fund-wording high/low utility outputs
* matched R0 bare-task outputs from outputs/api/r0_generations.jsonl

The match is exact on actor, task, item_id, and repeat. By default the script
fails if any requested high/low row lacks a matched R0 output.
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
LATEST_PATH = RUNS_DIR / "highlow_r0_bridge_latest.txt"
MANIFEST_GLOB = "fund_wording_rerun_manifests__*.tsv"
R0_PATH = OUTPUT_API / "r0_generations.jsonl"
TASKS = ("essay", "translation", "incident_postmortem", "grant_proposal_abstract")
SIDES = ("high", "low")
SIDE_CONDITIONS = {"high": "hl_high", "low": "hl_low"}
R0_CONDITION = "r0"


def csv_arg(value: str | None, allowed: tuple[str, ...] | None = None) -> list[str]:
    if value is None or value.strip().lower() == "all":
        return list(allowed or ())
    out = [part.strip() for part in value.split(",") if part.strip()]
    if allowed is not None:
        bad = sorted(set(out) - set(allowed))
        if bad:
            raise ValueError(f"unknown value(s): {', '.join(bad)}")
    return out


def read_rerun_manifest_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for tsv in sorted(RUNS_DIR.glob(MANIFEST_GLOB)):
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
                    }
                )
    if not entries:
        raise FileNotFoundError(f"no {MANIFEST_GLOB} files found under {RUNS_DIR}")
    return entries


def output_side(job: dict[str, Any], condition: str) -> str:
    if job.get("condition_a") == condition:
        return "a"
    if job.get("condition_b") == condition:
        return "b"
    raise ValueError(f"{job.get('pair_uid')} does not contain condition {condition!r}")


def output_id(job: dict[str, Any], condition: str) -> str:
    return f"{job['pair_uid']}::{output_side(job, condition)}"


def match_key_from_job(job: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(job.get("actor", "")),
        str(job.get("task", "")),
        str(job.get("item_id", "")),
        str(job.get("repeat", "")),
    )


def match_key_from_r0(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("actor", "")),
        str(row.get("task", "")),
        str(row.get("item_id", "")),
        str(row.get("repeat", "")),
    )


def load_run(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    jobs = read_jsonl(path)
    generations_path = path.parent / "generations.jsonl"
    if not generations_path.exists():
        raise FileNotFoundError(generations_path)
    generations = {str(row["output_id"]): row for row in read_jsonl(generations_path)}
    return jobs, generations


def load_r0_outputs() -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if not R0_PATH.exists():
        raise FileNotFoundError(R0_PATH)
    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    duplicates: list[tuple[str, str, str, str]] = []
    for row in read_jsonl(R0_PATH):
        if not row.get("ok"):
            continue
        if not str(row.get("output_text", "")).strip():
            continue
        key = match_key_from_r0(row)
        if key in rows:
            duplicates.append(key)
            continue
        rows[key] = row
    if duplicates:
        sample = ", ".join(map(str, duplicates[:5]))
        raise ValueError(f"duplicate valid R0 outputs for exact match keys, examples: {sample}")
    return rows


def digest_payload(payload: dict[str, Any], n: int = 12) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:n]


def bridge_pair_uid(highlow_job: dict[str, Any], r0_output: dict[str, Any], side: str) -> str:
    digest = digest_payload(
        {
            "source_highlow_pair_uid": highlow_job["pair_uid"],
            "source_r0_output_id": r0_output["output_id"],
            "side": side,
        }
    )
    return (
        f"highlow_r0_bridge:{side}:{highlow_job['actor']}:{highlow_job['task']}:"
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
        "source_note": "Copied for judging-only high/low versus R0 bridge; no generation call was made.",
    }


def bridge_job(
    *,
    highlow_job: dict[str, Any],
    highlow_output: dict[str, Any],
    r0_output: dict[str, Any],
    side: str,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    side_condition = SIDE_CONDITIONS[side]
    pair_uid = bridge_pair_uid(highlow_job, r0_output, side)
    prompt_field = f"prompt_{output_side(highlow_job, side_condition)}"
    return {
        "actor": highlow_job["actor"],
        "actor_label": highlow_job.get("actor_label", highlow_job["actor"]),
        "axis": highlow_job.get("axis", ""),
        "axis_definition": highlow_job.get("axis_definition", ""),
        "base_prompt": highlow_job.get("base_prompt", ""),
        "comparison": f"highlow_{side}_vs_r0",
        "condition_a": side_condition,
        "condition_b": R0_CONDITION,
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
        "other_condition": R0_CONDITION,
        "pair_idx": highlow_job.get("pair_idx", ""),
        "pair_set": highlow_job.get("pair_set", ""),
        "pair_uid": pair_uid,
        "predicted_condition": side_condition,
        "prompt_a": highlow_job.get(prompt_field, ""),
        "prompt_b": r0_output.get("user_prompt", highlow_job.get("base_prompt", "")),
        "repeat": highlow_job.get("repeat", ""),
        "r0_output_id": r0_output.get("output_id", ""),
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
        "source_r0_output_id": r0_output.get("output_id", ""),
        "source_r0_prompt": r0_output.get("user_prompt", ""),
        "source_note": "judging-only bridge from corrected fund-wording high/low outputs and matched R0 outputs",
        "system_prompt_a": highlow_job.get(f"system_prompt_{output_side(highlow_job, side_condition)}", ""),
        "system_prompt_b": "",
        "task": highlow_job["task"],
        "task_label": highlow_job.get("task_label", highlow_job["task"]),
        "topic_design": highlow_job.get("topic_design", ""),
    }


def prepare(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], Counter, list[dict[str, Any]]]:
    selected_tasks = set(csv_arg(args.tasks, TASKS))
    selected_sides = set(csv_arg(args.sides, SIDES))
    selected_domains = set(csv_arg(args.domains)) if args.domains else set()
    selected_actors = set(csv_arg(args.actors)) if args.actors else set()
    r0_outputs = load_r0_outputs()

    entries = [
        entry
        for entry in read_rerun_manifest_entries()
        if entry["task"] in selected_tasks
        and (not selected_actors or entry["actor"] in selected_actors)
    ]
    if not entries:
        raise ValueError("no rerun manifests matched the requested actor/task filters")

    run_seed = {
        "actors": sorted({entry["actor"] for entry in entries}),
        "tasks": sorted(selected_tasks),
        "domains": sorted(selected_domains) or ["all"],
        "sides": sorted(selected_sides),
        "source": f"{MANIFEST_GLOB} + {R0_PATH.name}",
    }
    run_hash = digest_payload(run_seed)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    run_id = f"highlow-r0-bridge__{timestamp}__hash-{run_hash}"
    run_dir = RUNS_DIR / run_id

    jobs_out: list[dict[str, Any]] = []
    generations_out: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    counts: Counter = Counter()
    seen_pair_uids: set[str] = set()

    for entry in entries:
        jobs, generations = load_run(entry["manifest_path"])
        for highlow_job in jobs:
            if highlow_job.get("condition_a") != "hl_high" or highlow_job.get("condition_b") != "hl_low":
                continue
            if not str(highlow_job.get("comparison", "")).endswith("_highlow"):
                continue
            if selected_domains and highlow_job.get("domain") not in selected_domains:
                continue
            key = match_key_from_job(highlow_job)
            r0_output = r0_outputs.get(key)
            if r0_output is None:
                missing.append(
                    {
                        "actor": key[0],
                        "task": key[1],
                        "item_id": key[2],
                        "repeat": key[3],
                        "source_highlow_pair_uid": highlow_job["pair_uid"],
                    }
                )
                continue

            for side in selected_sides:
                side_condition = SIDE_CONDITIONS[side]
                highlow_output = generations.get(output_id(highlow_job, side_condition))
                if highlow_output is None or not str(highlow_output.get("output_text", "")).strip():
                    raise ValueError(f"missing {side_condition} output for {highlow_job['pair_uid']}")
                job = bridge_job(
                    highlow_job=highlow_job,
                    highlow_output=highlow_output,
                    r0_output=r0_output,
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
                        source=r0_output,
                        job=job,
                        output_suffix="b",
                        condition=R0_CONDITION,
                        source_role=R0_CONDITION,
                    )
                )
                counts[(job["actor"], job["task"], job["domain"], side)] += 1

    if missing and not args.allow_missing_r0:
        examples = "\n".join(
            f"  {row['actor']} / {row['task']} / item {row['item_id']} / repeat {row['repeat']}"
            for row in missing[:20]
        )
        raise ValueError(
            f"{len(missing)} requested high/low rows lack matched R0 outputs. "
            f"Run the R0 generator for those actor/task cells or pass --allow-missing-r0.\n{examples}"
        )
    if not jobs_out:
        raise ValueError("no bridge comparisons were prepared")
    return run_dir, jobs_out, generations_out, counts, missing


def write_actor_manifest_pointer(jobs: list[dict[str, Any]], run_dir: Path) -> list[Path]:
    actors = sorted({str(job["actor"]) for job in jobs})
    sides = sorted({str(job["side"]) for job in jobs})
    if len(actors) != 1:
        return []

    actor = actors[0]
    side_slug = "+".join(sides)
    written: list[Path] = []
    pointer = RUNS_DIR / f"highlow_r0_bridge_manifests__{actor}__{side_slug}.tsv"
    pointer.write_text(f"{actor}\t{run_dir / 'generation_jobs.jsonl'}\n", encoding="utf-8")
    written.append(pointer)

    if sides == ["high"]:
        alias = RUNS_DIR / f"high_utility_r0_bridge_manifests__{actor}.tsv"
        alias.write_text(f"{actor}\t{run_dir / 'generation_jobs.jsonl'}\n", encoding="utf-8")
        written.append(alias)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", default=None, help="Comma-separated actor ids, or all. Default: all.")
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks, or all. Default: all.")
    parser.add_argument(
        "--domains",
        default="political",
        help="Comma-separated high-low domains. Default: political. Use all for all domains.",
    )
    parser.add_argument("--sides", default="high,low", help="high, low, or high,low. Default: high,low.")
    parser.add_argument("--allow-missing-r0", action="store_true", help="Skip high/low rows lacking matched R0 outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing files.")
    args = parser.parse_args()

    run_dir, jobs, generations, counts, missing = prepare(args)

    print(f"bridge judging jobs: {len(jobs)}")
    print(f"copied outputs: {len(generations)}")
    print(f"missing matched R0 rows: {len(missing)}")
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
    manifest_pointers = write_actor_manifest_pointer(jobs, run_dir)
    print(f"wrote {run_dir / 'generation_jobs.jsonl'}")
    print(f"wrote {run_dir / 'generations.jsonl'}")
    for pointer in manifest_pointers:
        print(f"wrote actor manifest list: {pointer}")
    print(f"wrote latest pointer: {LATEST_PATH}")
    print(f"next: python -m utility_behavior_gap.scripts.run_judging --run-dir {run_dir} --orders both --workers 10")


if __name__ == "__main__":
    main()
