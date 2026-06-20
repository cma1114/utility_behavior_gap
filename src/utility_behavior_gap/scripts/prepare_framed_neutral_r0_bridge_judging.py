#!/usr/bin/env python3
"""Prepare judging-only framed-neutral versus R0 comparisons.

This script makes no model calls. It builds a run-local judging manifest by
copying existing outputs:

* `direct_low` / framed-neutral outputs from the manifest-driven text catalog;
* matched `r0` / bare-task outputs from `outputs/api/r0_generations.jsonl`.

The match is exact on actor, task, item_label, and repeat. The resulting run
can be judged with `run_judging --orders both` just like the other bridge runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from utility_behavior_gap.constants import ACTOR_LABEL, TASK_LABEL
from utility_behavior_gap.fingerprints import text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl, write_jsonl
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API


RUNS_DIR = OUTPUT_API / "runs"
TEXT_CATALOG = ANALYSIS / "final_text_analysis_by_output.csv"
R0_PATH = OUTPUT_API / "r0_generations.jsonl"
LATEST_PATH = RUNS_DIR / "framed_neutral_r0_bridge_latest.txt"
TASKS = ("essay", "translation", "incident_postmortem", "grant_proposal_abstract")
CONDITION_FRAMED_NEUTRAL = "direct_low"
RAW_CONDITION_FRAMED_NEUTRAL = "framed_neutral"
CONDITION_R0 = "r0"
PAIR_KEYS = ["actor", "task", "item_label", "repeat"]


def csv_arg(value: str | None, allowed: tuple[str, ...] | None = None) -> list[str]:
    if value is None or value.strip().lower() == "all":
        return list(allowed or ())
    out = [part.strip() for part in value.split(",") if part.strip()]
    if allowed is not None:
        bad = sorted(set(out) - set(allowed))
        if bad:
            raise ValueError(f"unknown value(s): {', '.join(bad)}")
    return out


def digest_payload(payload: dict[str, Any], n: int = 12) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:n]


def normalize_repeat(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if text.endswith(".0"):
        return text[:-2]
    return text


def pair_key(row: dict[str, Any] | pd.Series) -> tuple[str, str, str, str]:
    return (
        str(row.get("actor", "")),
        str(row.get("task", "")),
        str(row.get("item_label", "")),
        normalize_repeat(row.get("repeat", "")),
    )


def check_unique(df: pd.DataFrame, condition: str) -> None:
    sub = df[df["condition"].eq(condition)].copy()
    sub["repeat_key"] = sub["repeat"].map(normalize_repeat)
    key_cols = ["actor", "task", "item_label", "repeat_key"]
    duplicates = sub[sub.duplicated(key_cols, keep=False)]
    if not duplicates.empty:
        sample = duplicates[key_cols + ["output_id", "run_dir"]].head(20).to_string(index=False)
        raise ValueError(f"{condition} has duplicate match keys; refusing ambiguous bridge.\n{sample}")


def output_side(job: dict[str, Any], raw_condition: str) -> str:
    if job.get("condition_a") == raw_condition:
        return "a"
    if job.get("condition_b") == raw_condition:
        return "b"
    raise ValueError(f"{job.get('pair_uid')} does not contain condition {raw_condition!r}")


def load_generation_by_output_id(run_dir: Path, output_id: str) -> dict[str, Any]:
    generations_path = run_dir / "generations.jsonl"
    if not generations_path.exists():
        raise FileNotFoundError(generations_path)
    for row in read_jsonl(generations_path):
        if str(row.get("output_id", "")) == output_id:
            return row
    raise KeyError(f"{output_id} not found in {generations_path}")


def load_r0_outputs() -> dict[str, dict[str, Any]]:
    if not R0_PATH.exists():
        raise FileNotFoundError(R0_PATH)
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(R0_PATH):
        if not row.get("ok", True):
            continue
        if not str(row.get("output_text", "")).strip():
            continue
        output_id = str(row.get("output_id", ""))
        if not output_id:
            continue
        if output_id in rows:
            raise ValueError(f"duplicate R0 output_id: {output_id}")
        rows[output_id] = row
    return rows


def bridge_pair_uid(framed_output_id: str, r0_output_id: str, actor: str, task: str, repeat: str) -> str:
    digest = digest_payload(
        {
            "framed_neutral_output_id": framed_output_id,
            "r0_output_id": r0_output_id,
            "actor": actor,
            "task": task,
            "repeat": repeat,
        }
    )
    return f"framed_neutral_r0_bridge:{actor}:{task}:r{repeat}:v{digest}"


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
        "source_note": "Copied for judging-only framed-neutral versus R0 bridge; no generation call was made.",
    }


def bridge_job(
    *,
    catalog_row: pd.Series,
    framed_generation: dict[str, Any],
    r0_generation: dict[str, Any],
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    actor = str(catalog_row["actor"])
    task = str(catalog_row["task"])
    item_label = str(catalog_row["item_label"])
    repeat = normalize_repeat(catalog_row["repeat"])
    framed_job = dict(framed_generation.get("job") or {})
    framed_side = output_side(framed_job, RAW_CONDITION_FRAMED_NEUTRAL)
    pair_uid = bridge_pair_uid(
        str(framed_generation["output_id"]),
        str(r0_generation["output_id"]),
        actor,
        task,
        repeat,
    )
    return {
        "actor": actor,
        "actor_label": framed_job.get("actor_label") or ACTOR_LABEL.get(actor, actor),
        "axis": framed_job.get("axis", ""),
        "axis_definition": framed_job.get("axis_definition", ""),
        "base_prompt": framed_job.get("base_prompt") or r0_generation.get("user_prompt", ""),
        "comparison": "framed_neutral_vs_r0",
        "condition_a": RAW_CONDITION_FRAMED_NEUTRAL,
        "condition_b": CONDITION_R0,
        "domain": framed_job.get("domain", ""),
        "domain_label": framed_job.get("domain_label", ""),
        "framing": "framed_neutral_vs_bare_task_bridge",
        "item_id": framed_job.get("item_id", r0_generation.get("item_id", item_label)),
        "item_index": framed_job.get("item_index", catalog_row.get("item_index", "")),
        "item_label": item_label,
        "other_condition": CONDITION_R0,
        "pair_uid": pair_uid,
        "predicted_condition": RAW_CONDITION_FRAMED_NEUTRAL,
        "prompt_a": framed_job.get(f"prompt_{framed_side}", ""),
        "prompt_b": r0_generation.get("user_prompt", ""),
        "repeat": repeat,
        "r0_output_id": r0_generation.get("output_id", ""),
        "run_dir": str(run_dir),
        "run_generation_failures_path": str(run_dir / "generation_failures.jsonl"),
        "run_generations_path": str(run_dir / "generations.jsonl"),
        "run_id": run_id,
        "run_judge_votes_path": str(run_dir / "judge_votes.jsonl"),
        "run_manifest_path": str(run_dir / "generation_jobs.jsonl"),
        "source_framed_neutral_output_id": framed_generation.get("output_id", ""),
        "source_framed_neutral_pair_uid": framed_job.get("pair_uid", ""),
        "source_framed_neutral_prompt": framed_job.get(f"prompt_{framed_side}", ""),
        "source_framed_neutral_run_dir": framed_job.get("run_dir", framed_generation.get("run_id", "")),
        "source_note": "judging-only bridge from matched framed-neutral and R0 outputs",
        "source_r0_output_id": r0_generation.get("output_id", ""),
        "source_r0_prompt": r0_generation.get("user_prompt", ""),
        "system_prompt_a": framed_job.get(f"system_prompt_{framed_side}", ""),
        "system_prompt_b": r0_generation.get("system_prompt", ""),
        "task": task,
        "task_label": framed_job.get("task_label") or TASK_LABEL.get(task, task),
        "topic_design": framed_job.get("topic_design", ""),
    }


def load_matched_catalog(*, actors: set[str], tasks: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TEXT_CATALOG.exists():
        raise FileNotFoundError(f"{TEXT_CATALOG} does not exist; run analyze_final_text_features first")
    catalog = pd.read_csv(TEXT_CATALOG, low_memory=False)
    check_unique(catalog, CONDITION_FRAMED_NEUTRAL)
    check_unique(catalog, CONDITION_R0)
    sub = catalog[catalog["condition"].isin([CONDITION_FRAMED_NEUTRAL, CONDITION_R0])].copy()
    if actors:
        sub = sub[sub["actor"].isin(actors)]
    if tasks:
        sub = sub[sub["task"].isin(tasks)]
    sub["repeat_key"] = sub["repeat"].map(normalize_repeat)
    keys = ["actor", "task", "item_label", "repeat_key"]
    framed = sub[sub["condition"].eq(CONDITION_FRAMED_NEUTRAL)].copy()
    r0 = sub[sub["condition"].eq(CONDITION_R0)].copy()
    matched = framed.merge(
        r0[keys + ["output_id"]],
        on=keys,
        how="inner",
        suffixes=("", "_r0"),
        validate="one_to_one",
    )

    framed_keys = set(map(tuple, framed[keys].to_numpy()))
    r0_keys = set(map(tuple, r0[keys].to_numpy()))
    missing_rows: list[dict[str, Any]] = []
    for key in sorted(framed_keys - r0_keys):
        missing_rows.append(dict(zip(keys, key), missing_side="r0"))
    for key in sorted(r0_keys - framed_keys):
        missing_rows.append(dict(zip(keys, key), missing_side="framed_neutral"))
    return matched, pd.DataFrame(missing_rows)


def prepare(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], Counter, pd.DataFrame]:
    selected_tasks = set(csv_arg(args.tasks, TASKS))
    selected_actors = set(csv_arg(args.actors)) if args.actors else set()
    if not selected_actors:
        raise ValueError("pass --actors for this bridge; use one actor per terminal")
    if len(selected_actors) != 1:
        raise ValueError("this bridge intentionally prepares one actor per run; pass exactly one actor")

    matched, missing = load_matched_catalog(actors=selected_actors, tasks=selected_tasks)
    if matched.empty:
        raise ValueError("no matched framed-neutral/R0 rows found for requested filters")

    r0_outputs = load_r0_outputs()
    run_seed = {
        "actors": sorted(selected_actors),
        "tasks": sorted(selected_tasks),
        "source": f"{TEXT_CATALOG.name} + {R0_PATH.name}",
        "comparison": "framed_neutral_vs_r0",
    }
    run_hash = digest_payload(run_seed)
    actor_slug = "+".join(sorted(selected_actors))
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    run_id = f"framed-neutral-r0-bridge__{actor_slug}__{timestamp}__hash-{run_hash}"
    run_dir = RUNS_DIR / run_id

    jobs: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    counts: Counter = Counter()
    seen_pair_uids: set[str] = set()
    generation_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for _, row in matched.sort_values(PAIR_KEYS).iterrows():
        framed_output_id = str(row["output_id"])
        r0_output_id = str(row["output_id_r0"])
        run_dir_value = Path(str(row["run_dir"]))
        cache_key = (str(run_dir_value), framed_output_id)
        framed_generation = generation_cache.get(cache_key)
        if framed_generation is None:
            framed_generation = load_generation_by_output_id(run_dir_value, framed_output_id)
            generation_cache[cache_key] = framed_generation
        r0_generation = r0_outputs.get(r0_output_id)
        if r0_generation is None:
            raise ValueError(f"R0 output {r0_output_id} is listed in the text catalog but missing from {R0_PATH}")

        job = bridge_job(
            catalog_row=row,
            framed_generation=framed_generation,
            r0_generation=r0_generation,
            run_id=run_id,
            run_dir=run_dir,
        )
        if job["pair_uid"] in seen_pair_uids:
            raise ValueError(f"duplicate bridge pair_uid: {job['pair_uid']}")
        seen_pair_uids.add(job["pair_uid"])
        jobs.append(job)
        generations.append(
            copied_generation(
                source=framed_generation,
                job=job,
                suffix="a",
                condition=RAW_CONDITION_FRAMED_NEUTRAL,
                source_role="framed_neutral",
            )
        )
        generations.append(
            copied_generation(
                source=r0_generation,
                job=job,
                suffix="b",
                condition=CONDITION_R0,
                source_role=CONDITION_R0,
            )
        )
        counts[(job["actor"], job["task"])] += 1

    return run_dir, jobs, generations, counts, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", required=True, help="Comma-separated actor ids. Use one actor per run.")
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks, or all. Default: all.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing files.")
    args = parser.parse_args()

    run_dir, jobs, generations, counts, missing = prepare(args)
    print(f"bridge judging pairs: {len(jobs)}")
    print(f"copied outputs: {len(generations)}")
    print(f"unmatched catalog rows under filters: {len(missing)}")
    print(f"run_dir: {run_dir}")
    print("breakdown:")
    for (actor, task), count in sorted(counts.items()):
        print(f"  {actor} / {task}: {count}")
    expected_votes = len(jobs) * 3 * 2
    print(f"expected judge votes with --orders both and 3 judges: {expected_votes}")

    if args.dry_run:
        print("dry run: wrote nothing")
        return

    write_jsonl(run_dir / "generation_jobs.jsonl", jobs)
    write_jsonl(run_dir / "generations.jsonl", generations)
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(str(run_dir) + "\n", encoding="utf-8")
    actor_file = RUNS_DIR / f"framed_neutral_r0_bridge_manifests__{jobs[0]['actor']}.tsv"
    actor_file.write_text(f"{jobs[0]['actor']}\t{run_dir / 'generation_jobs.jsonl'}\n", encoding="utf-8")
    print(f"wrote {run_dir / 'generation_jobs.jsonl'}")
    print(f"wrote {run_dir / 'generations.jsonl'}")
    print(f"wrote actor manifest list: {actor_file}")
    print(f"wrote latest pointer: {LATEST_PATH}")
    print(f"next: python -m utility_behavior_gap.scripts.run_judging --run-dir {run_dir} --orders both --workers 10")


if __name__ == "__main__":
    main()
