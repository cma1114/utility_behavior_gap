#!/usr/bin/env python3
"""Prepare judging-only pairs comparing high-low essays with clean direct outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utility_behavior_gap.io_utils import read_jsonl, write_jsonl
from utility_behavior_gap.paths import OUTPUT_API, ROOT


HIGHLOW_DIR = ROOT / "essay_all_conditions" / "highlow"
RUNS_DIR = OUTPUT_API / "runs"
CURRENT_JOBS = OUTPUT_API / "generation_jobs.jsonl"
CURRENT_GENERATIONS = OUTPUT_API / "generations.jsonl"
AXIS = "essay_quality"
AXIS_DEFINITION = "Which essay is better written, more compelling, clearer, and better structured?"
DEFAULT_DIRECT_CONDITION = "user_normal"
DEFAULT_DIRECT_CUES = {
    "user_normal": "Do a good job",
    "user_strong": "maximum care and effort",
}
DEFAULT_DIRECT_LABELS = {
    "user_normal": "direct_good_job",
    "user_strong": "direct_max_effort",
}


def default_direct_label(direct_condition: str) -> str:
    return DEFAULT_DIRECT_LABELS.get(direct_condition, "direct_" + direct_condition.replace("-", "_"))


def bridge_comparison(highlow_side: str, direct_label: str = "direct_good_job") -> str:
    return f"essay_{highlow_side}_utility_vs_{direct_label}"


def read_highlow_trials(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    trials = data.get("trials")
    if not isinstance(trials, list):
        raise ValueError(f"{path} does not contain a trials list")
    return trials


def highlow_side_output(trial: dict[str, Any], side: str) -> tuple[str, str]:
    needle = f"{side}-utility"
    for arm in ("A", "B"):
        label = str(trial.get(f"arm_{arm}", ""))
        text = str(trial.get(f"essay_{arm}", ""))
        if needle in label and text.strip():
            return arm, text
    raise ValueError(f"could not find {side}-utility essay in trial with topic={trial.get('essay_topic')!r}")


def highlow_outputs_by_topic(trials: list[dict[str, Any]], side: str) -> dict[str, list[dict[str, Any]]]:
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial_index, trial in enumerate(trials):
        topic = str(trial.get("essay_topic", "")).strip()
        if not topic:
            raise ValueError(f"high-low trial {trial_index} has no essay_topic")
        arm, text = highlow_side_output(trial, side)
        by_topic[topic].append(
            {
                "source": "essay_all_conditions/highlow",
                "source_trial_index": trial_index,
                "source_arm": arm,
                "source_arm_label": trial.get(f"arm_{arm}", ""),
                "topic": topic,
                "domain": trial.get("domain", ""),
                "framing": trial.get("framing", ""),
                "output_text": text,
                "source_winner_arm": trial.get("winner_arm", ""),
                "source_majority_vote": trial.get("majority_vote", ""),
            }
        )
    return dict(by_topic)


def direct_outputs_by_topic(
    run_dir: Path,
    *,
    actor: str,
    direct_condition: str,
    required_direct_cue: str,
) -> dict[str, list[dict[str, Any]]]:
    jobs_path = run_dir / "generation_jobs.jsonl"
    generations_path = run_dir / "generations.jsonl"
    if not jobs_path.exists():
        raise FileNotFoundError(jobs_path)
    if not generations_path.exists():
        raise FileNotFoundError(generations_path)

    jobs = {str(job["pair_uid"]): job for job in read_jsonl(jobs_path)}
    outputs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(generations_path):
        if row.get("success") is False:
            continue
        if row.get("actor") != actor:
            continue
        if row.get("condition") != direct_condition:
            continue
        if not str(row.get("output_text", "")).strip():
            continue
        pair_uid = str(row.get("pair_uid", ""))
        job = jobs.get(pair_uid) or row.get("job", {})
        if not job:
            raise ValueError(f"direct output {row.get('output_id')} has no matching job metadata")
        topic = str(job.get("item_label", "")).strip()
        if not topic:
            raise ValueError(f"direct output {row.get('output_id')} has no item_label")
        prompt_field = "prompt_a" if job.get("condition_a") == direct_condition else "prompt_b"
        prompt = str(job.get(prompt_field, ""))
        if required_direct_cue and required_direct_cue not in prompt:
            raise ValueError(
                f"direct output {row.get('output_id')} did not come from a prompt containing "
                f"{required_direct_cue!r}"
            )
        outputs[topic].append(
            {
                "source": "outputs/api/runs",
                "source_run_dir": str(run_dir),
                "source_run_id": row.get("run_id", run_dir.name),
                "source_output_id": row.get("output_id", ""),
                "source_pair_uid": row.get("pair_uid", ""),
                "topic": topic,
                "base_prompt": job.get("base_prompt", ""),
                "prompt": prompt,
                "output_text": row["output_text"],
                "source_model": row.get("model", ""),
                "source_finish_reason": row.get("finish_reason", ""),
            }
        )
    return dict(outputs)


def direct_control_outputs_by_topic(
    run_dir: Path,
    *,
    actor: str,
    control_condition: str,
    required_control_cue: str,
) -> dict[str, list[dict[str, Any]]]:
    return direct_outputs_by_topic(
        run_dir,
        actor=actor,
        direct_condition=control_condition,
        required_direct_cue=required_control_cue,
    )


def pair_digest(highlow: dict[str, Any], direct: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "highlow_source_trial_index": highlow["source_trial_index"],
            "highlow_source_arm": highlow["source_arm"],
            "direct_source_output_id": direct["source_output_id"],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def select_pairs(
    *,
    actor: str,
    highlow_by_topic: dict[str, list[dict[str, Any]]],
    direct_by_topic: dict[str, list[dict[str, Any]]],
    highlow_side: str,
    comparison: str,
    direct_label: str,
    seed: int,
    pairs_per_topic: int | None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    topics = sorted(set(highlow_by_topic) & set(direct_by_topic))
    if not topics:
        raise ValueError("no overlapping essay topics between high-low trials and direct-control run")

    pairs: list[dict[str, Any]] = []
    side_condition = f"{highlow_side}_utility"
    for topic_index, topic in enumerate(topics):
        highlow_rows = list(highlow_by_topic[topic])
        direct_rows = list(direct_by_topic[topic])
        rng.shuffle(highlow_rows)
        rng.shuffle(direct_rows)
        n = min(len(highlow_rows), len(direct_rows))
        if pairs_per_topic is not None:
            n = min(n, pairs_per_topic)
        if n <= 0:
            continue
        for local_index, (highlow, direct) in enumerate(zip(highlow_rows[:n], direct_rows[:n])):
            digest = pair_digest(highlow, direct)
            pair_uid = f"{comparison}:{actor}:topic{topic_index:02d}:pair{local_index:03d}:v{digest}"
            pairs.append(
                {
                    "actor": actor,
                    "actor_label": actor,
                    "task": "essay",
                    "task_label": "Essay writing",
                    "comparison": comparison,
                    "pair_uid": pair_uid,
                    "condition_a": side_condition,
                    "condition_b": direct_label,
                    "predicted_condition": side_condition,
                    "other_condition": direct_label,
                    "axis": AXIS,
                    "axis_definition": AXIS_DEFINITION,
                    "item_id": topic,
                    "item_index": str(topic_index),
                    "item_label": topic,
                    "base_prompt": direct.get("base_prompt", ""),
                    "prompt_a": "",
                    "prompt_b": direct.get("prompt", ""),
                    "system_prompt_a": "",
                    "system_prompt_b": "",
                    "domain": highlow.get("domain", ""),
                    "framing": highlow.get("framing", ""),
                    "source_highlow_side": highlow_side,
                    "source_highlow_trial_index": str(highlow.get("source_trial_index", "")),
                    "source_highlow_arm": highlow.get("source_arm", ""),
                    "source_highlow_arm_label": highlow.get("source_arm_label", ""),
                    "source_highlow_winner_arm": highlow.get("source_winner_arm", ""),
                    "source_highlow_majority_vote": highlow.get("source_majority_vote", ""),
                    "source_direct_run_id": direct.get("source_run_id", ""),
                    "source_direct_run_dir": direct.get("source_run_dir", ""),
                    "source_direct_output_id": direct.get("source_output_id", ""),
                    "source_direct_pair_uid": direct.get("source_pair_uid", ""),
                    "source_note": "judging-only bridge; essays were generated in prior runs",
                }
            )
    if not pairs:
        raise ValueError("no bridge pairs selected")
    return pairs


def run_id_for(*, actor: str, highlow_side: str, direct_label: str, digest: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    return f"bridge_{highlow_side}_utility_vs_{direct_label}__essay__{actor}__{timestamp}__hash-{digest}"


def add_run_paths(jobs: list[dict[str, Any]], run_dir: Path, run_id: str) -> list[dict[str, Any]]:
    manifest_path = run_dir / "generation_jobs.jsonl"
    return [
        {
            **job,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "run_manifest_path": str(manifest_path),
            "run_generations_path": str(run_dir / "generations.jsonl"),
            "run_generation_failures_path": str(run_dir / "generation_failures.jsonl"),
            "run_judge_votes_path": str(run_dir / "judge_votes.jsonl"),
        }
        for job in jobs
    ]


def source_generations_for_jobs(
    jobs: list[dict[str, Any]],
    highlow_by_trial: dict[tuple[int, str], dict[str, Any]],
    direct_by_output_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        highlow_key = (int(job["source_highlow_trial_index"]), str(job["source_highlow_arm"]))
        highlow = highlow_by_trial[highlow_key]
        direct = direct_by_output_id[job["source_direct_output_id"]]
        rows.extend(
            [
                {
                    "output_id": f"{job['pair_uid']}::a",
                    "pair_uid": job["pair_uid"],
                    "actor": job["actor"],
                    "model": job["actor"],
                    "condition": job["condition_a"],
                    "output_text": highlow["output_text"],
                    "success": True,
                    "latency_s": 0,
                    "temperature": "",
                    "max_tokens": "",
                    "reasoning": {},
                    "provider": {},
                    "finish_reason": "stop",
                    "usage": {},
                    "raw_response": {},
                    "request": {
                        "script": "utility_behavior_gap.scripts.prepare_highlow_vs_direct_judging",
                        "argv": sys.argv,
                        "source": highlow["source"],
                        "source_trial_index": highlow["source_trial_index"],
                        "source_arm": highlow["source_arm"],
                        "exact_api_request_available": False,
                    },
                    "job": job,
                    "run_id": job["run_id"],
                },
                {
                    "output_id": f"{job['pair_uid']}::b",
                    "pair_uid": job["pair_uid"],
                    "actor": job["actor"],
                    "model": direct.get("source_model", ""),
                    "condition": job["condition_b"],
                    "output_text": direct["output_text"],
                    "success": True,
                    "latency_s": 0,
                    "temperature": "",
                    "max_tokens": "",
                    "reasoning": {},
                    "provider": {},
                    "finish_reason": "stop",
                    "usage": {},
                    "raw_response": {},
                    "request": {
                        "script": "utility_behavior_gap.scripts.prepare_highlow_vs_direct_judging",
                        "argv": sys.argv,
                        "source": direct["source"],
                        "source_run_dir": direct["source_run_dir"],
                        "source_output_id": direct["source_output_id"],
                        "source_prompt": direct["prompt"],
                        "exact_api_request_available": True,
                    },
                    "job": job,
                    "run_id": job["run_id"],
                },
            ]
        )
    return rows


def write_manifest_and_sources(
    *,
    jobs: list[dict[str, Any]],
    source_generations: list[dict[str, Any]],
    activate: bool,
) -> None:
    run_dir = Path(jobs[0]["run_dir"])
    write_jsonl(run_dir / "generation_jobs.jsonl", jobs)
    write_jsonl(run_dir / "generations.jsonl", source_generations)
    if activate:
        write_jsonl(CURRENT_JOBS, jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True, help="Actor id, e.g. gpt-5.4-mini-or")
    parser.add_argument("--direct-run-dir", type=Path, required=True, help="Run dir containing clean direct outputs")
    parser.add_argument("--highlow-dir", type=Path, default=HIGHLOW_DIR)
    parser.add_argument("--highlow-side", choices=["high", "low"], default="high")
    parser.add_argument(
        "--direct-condition",
        "--control-condition",
        dest="direct_condition",
        default=DEFAULT_DIRECT_CONDITION,
        help="Condition from the direct run to compare against the utility essay.",
    )
    parser.add_argument(
        "--direct-label",
        default=None,
        help="Condition label to use in this bridge run, e.g. direct_max_effort.",
    )
    parser.add_argument(
        "--required-direct-cue",
        "--required-control-cue",
        dest="required_direct_cue",
        default=None,
        help="Literal prompt text that must appear in the selected direct condition.",
    )
    parser.add_argument("--pairs-per-topic", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Also write outputs/api/generation_jobs.jsonl. Avoid while another batch is running.",
    )
    args = parser.parse_args()

    direct_label = args.direct_label or default_direct_label(args.direct_condition)
    required_direct_cue = (
        args.required_direct_cue
        if args.required_direct_cue is not None
        else DEFAULT_DIRECT_CUES.get(args.direct_condition, "")
    )
    comparison = bridge_comparison(args.highlow_side, direct_label)

    highlow_path = args.highlow_dir / f"{args.actor}.json"
    highlow_trials = read_highlow_trials(highlow_path)
    highlow_by_topic = highlow_outputs_by_topic(highlow_trials, args.highlow_side)
    direct_by_topic = direct_outputs_by_topic(
        args.direct_run_dir,
        actor=args.actor,
        direct_condition=args.direct_condition,
        required_direct_cue=required_direct_cue,
    )
    jobs_without_run = select_pairs(
        actor=args.actor,
        highlow_by_topic=highlow_by_topic,
        direct_by_topic=direct_by_topic,
        highlow_side=args.highlow_side,
        comparison=comparison,
        direct_label=direct_label,
        seed=args.seed,
        pairs_per_topic=args.pairs_per_topic,
    )
    payload = "\n".join(json.dumps(job, ensure_ascii=False, sort_keys=True) for job in jobs_without_run)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    run_id = run_id_for(actor=args.actor, highlow_side=args.highlow_side, direct_label=direct_label, digest=digest)
    run_dir = RUNS_DIR / run_id
    jobs = add_run_paths(jobs_without_run, run_dir, run_id)

    highlow_by_trial: dict[tuple[int, str], dict[str, Any]] = {}
    for topic_rows in highlow_by_topic.values():
        for row in topic_rows:
            highlow_by_trial[(int(row["source_trial_index"]), str(row["source_arm"]))] = row
    direct_by_output_id = {row["source_output_id"]: row for topic_rows in direct_by_topic.values() for row in topic_rows}
    source_generations = source_generations_for_jobs(jobs, highlow_by_trial, direct_by_output_id)
    write_manifest_and_sources(jobs=jobs, source_generations=source_generations, activate=args.activate)

    topic_counts = Counter(job["item_label"] for job in jobs)
    print(f"wrote {len(jobs)} bridge judging pairs")
    print(f"run_dir: {run_dir}")
    print(f"manifest: {run_dir / 'generation_jobs.jsonl'}")
    print(f"source generations: {run_dir / 'generations.jsonl'}")
    if args.activate:
        print(f"current manifest: {CURRENT_JOBS}")
    else:
        print("current manifest: unchanged")
    print(f"topics: {dict(topic_counts)}")
    print(f"next: python -m utility_behavior_gap.scripts.run_judging --run-dir {run_dir}")


if __name__ == "__main__":
    main()
