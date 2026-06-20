#!/usr/bin/env python3
"""Prepare judging-only moral-low versus R0 bare-task comparisons.

This script makes no model calls. It builds a run-local judging manifest by
copying existing outputs:

* hash-classified clean ``moral_bad`` outputs;
* matched ``r0`` bare-task outputs from ``outputs/api/r0_generations.jsonl``.

Matching is exact on actor, task, item_id, and repeat. The default source scope
is ``base`` because the current R0 file covers the base repeat block; use
``--source-scope current`` only after generating matched R0 outputs for the
high-N extension repeats.
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
    load_run,
    output_id,
    output_side,
    read_manifest_entries,
    selected_entries,
)


RUNS_DIR = OUTPUT_API / "runs"
LATEST_PATH = RUNS_DIR / "moral_r0_bridge_latest.txt"
MORAL_BASE_GLOB = "fund_wording_rerun_manifests__*.tsv"
MORAL_HIGHN_GLOB = "canonical_highn10_manifests__*.tsv"
R0_PATH = OUTPUT_API / "r0_generations.jsonl"
CLASSIFICATIONS = ANALYSIS / "canonical_highn_moral_refusal_classifications.jsonl"
TASKS = ("essay", "translation", "incident_postmortem", "grant_proposal_abstract")
MORAL_BAD = "moral_bad"
R0 = "r0"
VALID_LABELS = {"clean", "partial_refusal", "full_refusal", "degenerate"}


def digest_payload(payload: dict[str, Any], n: int = 12) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:n]


def normalize_repeat(value: Any) -> str:
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def match_key_from_job(job: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(job.get("actor", "")),
        str(job.get("task", "")),
        str(job.get("item_id", "")),
        normalize_repeat(job.get("repeat", "")),
    )


def match_key_from_r0(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("actor", "")),
        str(row.get("task", "")),
        str(row.get("item_id", "")),
        normalize_repeat(row.get("repeat", "")),
    )


def load_r0_outputs() -> dict[tuple[str, str, str, str], dict[str, Any]]:
    if not R0_PATH.exists():
        raise FileNotFoundError(R0_PATH)
    rows: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    duplicates: list[tuple[str, str, str, str]] = []
    for row in read_jsonl(R0_PATH):
        if not row.get("ok", True):
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


def bridge_pair_uid(moral_job: dict[str, Any], r0_output: dict[str, Any]) -> str:
    digest = digest_payload(
        {
            "source_moral_pair_uid": moral_job["pair_uid"],
            "source_r0_output_id": r0_output["output_id"],
            "source_moral_condition": MORAL_BAD,
            "source_r0_condition": R0,
        }
    )
    return (
        f"moral_r0_bridge:{moral_job['actor']}:{moral_job['task']}:"
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
        "source_note": "Copied for judging-only moral-low versus R0 bridge; no generation call was made.",
    }


def bridge_job(
    *,
    moral_job: dict[str, Any],
    moral_output: dict[str, Any],
    r0_output: dict[str, Any],
    moral_label: str,
    source_block: str,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    moral_side = output_side(moral_job, MORAL_BAD)
    pair_uid = bridge_pair_uid(moral_job, r0_output)
    return {
        "actor": moral_job["actor"],
        "actor_label": moral_job.get("actor_label") or ACTOR_LABEL.get(moral_job["actor"], moral_job["actor"]),
        "axis": moral_job.get("axis", ""),
        "axis_definition": moral_job.get("axis_definition", ""),
        "base_prompt": moral_job.get("base_prompt") or r0_output.get("user_prompt", ""),
        "cause_pair_label": moral_job.get("cause_pair_label", ""),
        "comparison": "moral_bad_vs_r0",
        "condition_a": MORAL_BAD,
        "condition_b": R0,
        "domain": moral_job.get("domain", ""),
        "domain_label": moral_job.get("domain_label", ""),
        "framing": "moral_bad_vs_bare_task_bridge",
        "item_id": moral_job.get("item_id", r0_output.get("item_id", "")),
        "item_index": moral_job.get("item_index", ""),
        "item_label": moral_job.get("item_label", r0_output.get("item_label", "")),
        "moral_bad_label": moral_label,
        "other_condition": R0,
        "pair_uid": pair_uid,
        "predicted_condition": MORAL_BAD,
        "prompt_a": moral_job.get(f"prompt_{moral_side}", ""),
        "prompt_b": r0_output.get("user_prompt", moral_job.get("base_prompt", "")),
        "repeat": moral_job.get("repeat", ""),
        "r0_output_id": r0_output.get("output_id", ""),
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
        "source_note": "judging-only bridge from clean moral-low and matched R0 outputs",
        "source_r0_output_id": r0_output.get("output_id", ""),
        "source_r0_prompt": r0_output.get("user_prompt", ""),
        "system_prompt_a": moral_job.get(f"system_prompt_{moral_side}", ""),
        "system_prompt_b": r0_output.get("system_prompt", ""),
        "task": moral_job["task"],
        "task_label": moral_job.get("task_label") or TASK_LABEL.get(moral_job["task"], moral_job["task"]),
        "topic_design": moral_job.get("topic_design", ""),
    }


def prepare(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], Counter, list[dict[str, Any]]]:
    selected_tasks = set(csv_arg(args.tasks, TASKS))
    selected_actors = set(csv_arg(args.actors)) if args.actors else set()
    if not selected_actors:
        raise ValueError("pass --actors for this bridge; use one actor per terminal")
    if len(selected_actors) != 1:
        raise ValueError("this bridge intentionally prepares one actor per run; pass exactly one actor")

    labels = load_classification_labels()
    r0_outputs = load_r0_outputs()
    moral_entries = selected_entries(
        read_manifest_entries(MORAL_BASE_GLOB),
        selected_tasks=selected_tasks,
        selected_actors=selected_actors,
    )
    source_manifest_globs = [MORAL_BASE_GLOB]

    if args.source_scope == "current":
        moral_entries.extend(
            selected_entries(
                read_manifest_entries(MORAL_HIGHN_GLOB),
                selected_tasks=selected_tasks,
                selected_actors=selected_actors,
            )
        )
        source_manifest_globs.append(MORAL_HIGHN_GLOB)

    if not moral_entries:
        raise ValueError("no moral manifests matched the requested actor/task filters")

    run_seed = {
        "actors": sorted(selected_actors),
        "tasks": sorted(selected_tasks),
        "source_scope": args.source_scope,
        "source_manifest_globs": source_manifest_globs,
        "bridge": "moral_bad_vs_r0",
        "moral_bad_clean_only": True,
        "r0_source": R0_PATH.name,
    }
    run_hash = digest_payload(run_seed)
    actor_slug = "+".join(sorted(selected_actors))
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    run_id = f"moral-r0-bridge-{args.source_scope}__{actor_slug}__{timestamp}__hash-{run_hash}"
    run_dir = RUNS_DIR / run_id

    jobs_out: list[dict[str, Any]] = []
    generations_out: list[dict[str, Any]] = []
    missing_r0: list[dict[str, Any]] = []
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

            key = match_key_from_job(moral_job)
            r0_output = r0_outputs.get(key)
            if r0_output is None:
                counts[(moral_job["actor"], moral_job["task"], source_block, "missing_r0_output")] += 1
                missing_r0.append(
                    {
                        "actor": key[0],
                        "task": key[1],
                        "item_id": key[2],
                        "repeat": key[3],
                        "source_moral_pair_uid": moral_job["pair_uid"],
                        "source_block": source_block,
                    }
                )
                continue

            job = bridge_job(
                moral_job=moral_job,
                moral_output=moral_output,
                r0_output=r0_output,
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
                    source=r0_output,
                    job=job,
                    suffix="b",
                    condition=R0,
                    source_role=R0,
                )
            )
            counts[(job["actor"], job["task"], source_block, "included")] += 1

    if missing_r0 and args.require_complete_r0:
        examples = "\n".join(
            f"  {row['actor']} / {row['task']} / item {row['item_id']} / repeat {row['repeat']}"
            for row in missing_r0[:20]
        )
        raise ValueError(
            f"{len(missing_r0)} requested clean moral-low rows lack matched R0 outputs. "
            f"Generate R0 outputs for those rows or omit --require-complete-r0.\n{examples}"
        )
    if not jobs_out:
        raise ValueError("no moral-low versus R0 bridge comparisons were prepared")
    return run_dir, jobs_out, generations_out, counts, missing_r0


def write_actor_manifest_pointer(jobs: list[dict[str, Any]], run_dir: Path) -> None:
    actor = str(jobs[0]["actor"])
    pointer = RUNS_DIR / f"moral_r0_bridge_manifests__{actor}.tsv"
    pointer.write_text(f"{actor}\t{run_dir / 'generation_jobs.jsonl'}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", required=True, help="Comma-separated actor ids. Use one actor per run.")
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks, or all. Default: all.")
    parser.add_argument(
        "--source-scope",
        choices=("base", "current"),
        default="base",
        help="base uses moral repeats 0-4. current also tries extension repeats 5-9.",
    )
    parser.add_argument(
        "--require-complete-r0",
        action="store_true",
        help="Fail instead of writing a manifest when clean moral-low rows lack matched R0 outputs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing files.")
    args = parser.parse_args()

    run_dir, jobs, generations, counts, missing_r0 = prepare(args)
    print(f"bridge judging pairs: {len(jobs)}")
    print(f"copied outputs: {len(generations)}")
    print(f"missing matched R0 rows: {len(missing_r0)}")
    print(f"run_dir: {run_dir}")
    print("breakdown:")
    for key, count in sorted(counts.items()):
        actor, task, source_block, status = key
        print(f"  {actor} / {task} / {source_block} / {status}: {count}")
    print(f"expected judge votes with --orders both and 3 judges: {len(jobs) * 3 * 2}")

    if args.dry_run:
        print("dry run: wrote nothing")
        return

    write_jsonl(run_dir / "generation_jobs.jsonl", jobs)
    write_jsonl(run_dir / "generations.jsonl", generations)
    if missing_r0:
        write_jsonl(run_dir / "missing_r0_rows.jsonl", missing_r0)
        print(f"wrote missing R0 audit: {run_dir / 'missing_r0_rows.jsonl'}")
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(str(run_dir) + "\n", encoding="utf-8")
    write_actor_manifest_pointer(jobs, run_dir)
    print(f"wrote {run_dir / 'generation_jobs.jsonl'}")
    print(f"wrote {run_dir / 'generations.jsonl'}")
    actor_file = RUNS_DIR / f"moral_r0_bridge_manifests__{jobs[0]['actor']}.tsv"
    print(f"wrote actor manifest list: {actor_file}")
    print(f"wrote latest pointer: {LATEST_PATH}")
    print(f"next: python -m utility_behavior_gap.scripts.run_judging --run-dir {run_dir} --orders both --workers 6")


if __name__ == "__main__":
    main()
