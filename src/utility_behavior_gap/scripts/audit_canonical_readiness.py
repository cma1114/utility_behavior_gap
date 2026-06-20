#!/usr/bin/env python3
"""Audit canonical run readiness without making API calls.

The audit is deliberately source-aware. It checks the current approved run
families separately instead of globbing every historical experiment:

* direct instruction: framed-user-strong manifest lists;
* utility/moral rerun: fund-wording manifest lists;
* high-N repeat block: canonical_highn10 manifest lists;
* corrected amount base block, when available; otherwise the legacy amount base
  block from the four-comparison modgrid runs.

For each manifest it reports prompt-rule failures, generation completeness,
judge-vote completeness, hash-mismatched votes, duplicate log keys, and
possible truncation indicators. It writes only analysis/audit artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from utility_behavior_gap.constants import ACTORS, JUDGE_MODEL_IDS, TASK_LABEL
from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.io_utils import write_csv_rows
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API
from utility_behavior_gap.prompts import (
    MAX_EFFORT_STRONG_SYSTEM_PROMPTS,
    MODGRID_NEUTRAL_TARGET,
)


RUNS = OUTPUT_API / "runs"
DIRECT_MANIFEST_GLOB = "framed_user_strong_manifests__*.tsv"
FUND_MANIFEST_GLOB = "fund_wording_rerun_manifests__*.tsv"
CANONICAL_HIGHN_MANIFEST_GLOB = "canonical_highn10_manifests__*.tsv"
CANONICAL_AMOUNT_BASE_MANIFEST_GLOB = "canonical_amount_base_manifests__*.tsv"
FOUR_COMPARISON_GLOB = "*__4-comparisons__*/generation_jobs.jsonl"

TASKS = ["essay", "translation", "incident_postmortem", "grant_proposal_abstract"]
EXPECTED_JUDGES = len(JUDGE_MODEL_IDS)
EXPECTED_VOTES_BOTH_ORDERS = EXPECTED_JUDGES * 2

FUND_SENTENCE = "the sponsor will fund the following intervention:"
OLD_1000_DONATION_SENTENCE = "donate $1,000 toward"


@dataclass(frozen=True)
class SourceSpec:
    source_family: str
    manifest: Path
    comparison_kind: str
    expected_votes_per_pair: int = EXPECTED_VOTES_BOTH_ORDERS


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def manifest_paths(pattern: str) -> list[Path]:
    paths: list[Path] = []
    for tsv in sorted(RUNS.glob(pattern)):
        for line in tsv.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                paths.append(Path(parts[2]))
    return sorted(set(paths))


def source_specs() -> list[SourceSpec]:
    specs: list[SourceSpec] = []
    specs.extend(
        SourceSpec("direct_instruction", path, "direct")
        for path in manifest_paths(DIRECT_MANIFEST_GLOB)
    )
    specs.extend(
        SourceSpec("fund_wording_base", path, "fund_highlow_moral")
        for path in manifest_paths(FUND_MANIFEST_GLOB)
    )
    specs.extend(
        SourceSpec("canonical_highn10", path, "highn_amount_moral_utility")
        for path in manifest_paths(CANONICAL_HIGHN_MANIFEST_GLOB)
    )
    amount_base_paths = manifest_paths(CANONICAL_AMOUNT_BASE_MANIFEST_GLOB)
    if amount_base_paths:
        specs.extend(
            SourceSpec("canonical_amount_base", path, "amount_only")
            for path in amount_base_paths
        )
    else:
        specs.extend(
            SourceSpec("legacy_amount_base", path, "amount_only")
            for path in sorted(RUNS.glob(FOUR_COMPARISON_GLOB))
        )
    return specs


def canonical_jobs(jobs: list[dict[str, Any]], comparison_kind: str) -> list[dict[str, Any]]:
    if comparison_kind == "direct":
        return [job for job in jobs if job.get("comparison") == "framed_user_strong_headroom"]
    if comparison_kind == "fund_highlow_moral":
        return [
            job
            for job in jobs
            if str(job.get("comparison", "")).endswith(("_highlow", "_moral"))
        ]
    if comparison_kind == "highn_amount_moral_utility":
        return [
            job
            for job in jobs
            if str(job.get("comparison", "")).endswith(("_highlow", "_moral", "_amount"))
        ]
    if comparison_kind == "amount_only":
        return [job for job in jobs if str(job.get("comparison", "")).endswith("_amount")]
    raise ValueError(f"unknown comparison_kind: {comparison_kind}")


def prompt_errors(job: dict[str, Any], source_family: str) -> list[str]:
    errors: list[str] = []
    comparison = str(job.get("comparison", ""))
    task = str(job.get("task", ""))
    prompt_a = str(job.get("prompt_a", ""))
    prompt_b = str(job.get("prompt_b", ""))
    both_prompts = prompt_a + "\n" + prompt_b

    if job.get("system_prompt_a") or job.get("system_prompt_b"):
        errors.append("nonblank_system_prompt")

    if source_family == "direct_instruction":
        if comparison != "framed_user_strong_headroom":
            errors.append("wrong_direct_comparison")
        if job.get("condition_a") != "framed_user_strong":
            errors.append("direct_condition_a_not_strong")
        if job.get("condition_b") != "framed_neutral":
            errors.append("direct_condition_b_not_neutral")
        strong_text = MAX_EFFORT_STRONG_SYSTEM_PROMPTS.get(task, "")
        if strong_text and strong_text not in prompt_a:
            errors.append("direct_strong_text_missing_from_user_prompt")
        if strong_text and strong_text in prompt_b:
            errors.append("direct_strong_text_in_neutral_prompt")
        if FUND_SENTENCE not in both_prompts:
            errors.append("direct_missing_fund_sentence")
        if MODGRID_NEUTRAL_TARGET not in both_prompts:
            errors.append("direct_missing_neutral_target")
    elif source_family in {"fund_wording_base", "canonical_highn10"}:
        if comparison.endswith(("_highlow", "_moral")):
            if FUND_SENTENCE not in both_prompts:
                errors.append("missing_fund_sentence")
            if OLD_1000_DONATION_SENTENCE in both_prompts:
                errors.append("old_1000_donation_wording_present")
        elif comparison.endswith("_amount"):
            if "$1,000,000" not in prompt_a:
                errors.append("amount_high_missing_1000000")
            if "$100" not in prompt_b:
                errors.append("amount_low_missing_100")
            if FUND_SENTENCE in both_prompts:
                errors.append("amount_uses_fund_sentence")
    elif source_family in {"legacy_amount_base", "canonical_amount_base"}:
        if not comparison.endswith("_amount"):
            errors.append("amount_base_non_amount_comparison")
        if "$1,000,000" not in prompt_a:
            errors.append("amount_high_missing_1000000")
        if "$100" not in prompt_b:
            errors.append("amount_low_missing_100")
        if FUND_SENTENCE in both_prompts:
            errors.append("amount_uses_fund_sentence")
        if source_family == "canonical_amount_base" and MODGRID_NEUTRAL_TARGET not in both_prompts:
            errors.append("amount_missing_current_neutral_target")
    return errors


def generation_log_paths(jobs: list[dict[str, Any]], manifest: Path) -> tuple[Path, Path]:
    run_dir = manifest.parent
    gen_paths = {str(job.get("run_generations_path") or "") for job in jobs}
    fail_paths = {str(job.get("run_generation_failures_path") or "") for job in jobs}
    gen_paths.discard("")
    fail_paths.discard("")
    generations = Path(next(iter(gen_paths))) if len(gen_paths) == 1 else run_dir / "generations.jsonl"
    failures = Path(next(iter(fail_paths))) if len(fail_paths) == 1 else run_dir / "generation_failures.jsonl"
    return generations, failures


def vote_log_path(jobs: list[dict[str, Any]], manifest: Path) -> Path:
    run_dir = manifest.parent
    vote_paths = {str(job.get("run_judge_votes_path") or "") for job in jobs}
    vote_paths.discard("")
    return Path(next(iter(vote_paths))) if len(vote_paths) == 1 else run_dir / "judge_votes.jsonl"


def generation_finish_reason(row: dict[str, Any]) -> str:
    reason = row.get("finish_reason")
    return "" if reason is None else str(reason)


def native_finish_reason(row: dict[str, Any]) -> str:
    try:
        return str(row["raw_response"]["choices"][0].get("native_finish_reason") or "")
    except (KeyError, IndexError, TypeError):
        return ""


def completion_tokens(row: dict[str, Any]) -> int | None:
    usage = row.get("usage") or row.get("raw_response", {}).get("usage") or {}
    value = usage.get("completion_tokens")
    return int(value) if isinstance(value, int | float) else None


def max_tokens(row: dict[str, Any]) -> int | None:
    request = row.get("request") if isinstance(row.get("request"), dict) else {}
    value = request.get("max_tokens")
    return int(value) if isinstance(value, int | float) else None


def valid_generation(row: dict[str, Any]) -> bool:
    return (
        row.get("success") is not False
        and bool(str(row.get("output_text") or "").strip())
        and generation_finish_reason(row) in {"", "stop"}
    )


def summarize_manifest(spec: SourceSpec) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    all_jobs = read_jsonl_if_exists(spec.manifest)
    jobs = canonical_jobs(all_jobs, spec.comparison_kind)
    if not jobs:
        return None, []

    prompt_error_counter: Counter[str] = Counter()
    bad_prompt_jobs = 0
    for job in jobs:
        errors = prompt_errors(job, spec.source_family)
        if errors:
            bad_prompt_jobs += 1
            prompt_error_counter.update(errors)

    run_dir = spec.manifest.parent
    generations_path, generation_failures_path = generation_log_paths(jobs, spec.manifest)
    votes_path = vote_log_path(jobs, spec.manifest)
    generations = read_jsonl_if_exists(generations_path)
    generation_failures = read_jsonl_if_exists(generation_failures_path)
    votes = read_jsonl_if_exists(votes_path)

    pair_uids = {str(job["pair_uid"]) for job in jobs}
    expected_output_ids = {f"{pair_uid}::{suffix}" for pair_uid in pair_uids for suffix in ("a", "b")}
    generation_rows = [row for row in generations if str(row.get("output_id", "")) in expected_output_ids]
    output_ids = [str(row.get("output_id", "")) for row in generation_rows]
    duplicate_output_ids = len(output_ids) - len(set(output_ids))
    generation_by_id = {str(row.get("output_id", "")): row for row in generation_rows if row.get("output_id")}
    valid_generation_ids = {
        output_id for output_id, row in generation_by_id.items() if valid_generation(row)
    }
    complete_generation_pairs = sum(
        all(f"{pair_uid}::{suffix}" in valid_generation_ids for suffix in ("a", "b"))
        for pair_uid in pair_uids
    )

    unresolved_failures = [
        row
        for row in generation_failures
        if str(row.get("output_id", "")) in expected_output_ids
        and str(row.get("output_id", "")) not in valid_generation_ids
    ]

    non_stop_generations = [
        row
        for row in generation_rows
        if row.get("success") is not False
        and generation_finish_reason(row)
        and generation_finish_reason(row) != "stop"
    ]
    length_finish_generations = [
        row for row in generation_rows if generation_finish_reason(row) == "length"
    ]
    near_cap_95 = 0
    near_cap_98 = 0
    for row in generation_rows:
        complete = completion_tokens(row)
        requested = max_tokens(row)
        if complete is None or not requested:
            continue
        ratio = complete / requested
        near_cap_95 += ratio >= 0.95
        near_cap_98 += ratio >= 0.98

    current_hashes: dict[str, tuple[str, str]] = {}
    for pair_uid in pair_uids:
        out_a = generation_by_id.get(f"{pair_uid}::a")
        out_b = generation_by_id.get(f"{pair_uid}::b")
        if out_a is None or out_b is None or not valid_generation(out_a) or not valid_generation(out_b):
            continue
        current_hashes[pair_uid] = (output_text_fingerprint(out_a), output_text_fingerprint(out_b))

    relevant_votes = [row for row in votes if str(row.get("pair_uid", "")) in pair_uids]
    successful_matching_vote_keys: set[tuple[str, str, bool]] = set()
    failed_vote_keys: set[tuple[str, str, bool]] = set()
    hash_mismatched_votes = 0
    unresolved_successful_votes = 0
    vote_key_counts: Counter[tuple[str, str, bool]] = Counter()
    for vote in relevant_votes:
        pair_uid = str(vote.get("pair_uid", ""))
        judge = str(vote.get("judge_model", ""))
        flipped = bool(vote.get("flipped"))
        key = (pair_uid, judge, flipped)
        if vote.get("success") is False:
            failed_vote_keys.add(key)
            continue
        expected_hashes = current_hashes.get(pair_uid)
        if expected_hashes is None:
            continue
        vote_hashes = (vote.get("source_output_a_hash"), vote.get("source_output_b_hash"))
        if all(vote_hashes) and vote_hashes != expected_hashes:
            hash_mismatched_votes += 1
            continue
        successful_matching_vote_keys.add(key)
        vote_key_counts[key] += 1
        if vote.get("winner_condition") == "unresolved":
            unresolved_successful_votes += 1
    duplicate_vote_keys = sum(max(0, count - 1) for count in vote_key_counts.values())
    unresolved_failed_vote_keys = failed_vote_keys - successful_matching_vote_keys

    expected_votes_total = len(pair_uids) * spec.expected_votes_per_pair
    complete_vote_pairs = 0
    complete_pairs = 0
    votes_per_pair: Counter[int] = Counter()
    for pair_uid in pair_uids:
        vote_count = sum(1 for key in successful_matching_vote_keys if key[0] == pair_uid)
        votes_per_pair[vote_count] += 1
        has_all_votes = vote_count >= spec.expected_votes_per_pair
        if has_all_votes:
            complete_vote_pairs += 1
        if pair_uid in current_hashes and has_all_votes:
            complete_pairs += 1

    tasks = sorted({str(job.get("task", "")) for job in jobs})
    actors = sorted({str(job.get("actor", "")) for job in jobs})
    comparisons = sorted({str(job.get("comparison", "")) for job in jobs})
    repeats = sorted({str(job.get("repeat", "")) for job in jobs if str(job.get("repeat", ""))})
    domains = sorted({str(job.get("domain", "")) for job in jobs if str(job.get("domain", ""))})

    status = "PASS"
    if bad_prompt_jobs:
        status = "FAIL"
    elif complete_pairs < len(pair_uids):
        status = "INCOMPLETE"
    elif unresolved_successful_votes or unresolved_failed_vote_keys or hash_mismatched_votes:
        status = "WARN"
    elif non_stop_generations or length_finish_generations:
        status = "WARN"

    row = {
        "status": status,
        "source_family": spec.source_family,
        "run_id": run_dir.name,
        "manifest": str(spec.manifest),
        "tasks": ",".join(tasks),
        "task_labels": ",".join(TASK_LABEL.get(task, task) for task in tasks),
        "actors": ",".join(actors),
        "comparisons": ",".join(comparisons),
        "repeats": ",".join(repeats),
        "domains": ",".join(domains),
        "jobs": len(jobs),
        "expected_outputs": len(expected_output_ids),
        "valid_outputs": len(valid_generation_ids),
        "complete_generation_pairs": complete_generation_pairs,
        "expected_votes": expected_votes_total,
        "valid_matching_votes": len(successful_matching_vote_keys),
        "complete_vote_pairs": complete_vote_pairs,
        "complete_pairs": complete_pairs,
        "prompt_failed_jobs": bad_prompt_jobs,
        "prompt_errors": ";".join(f"{key}:{value}" for key, value in sorted(prompt_error_counter.items())),
        "unresolved_generation_failures": len(unresolved_failures),
        "non_stop_generations": len(non_stop_generations),
        "length_finish_generations": len(length_finish_generations),
        "near_token_cap_95": near_cap_95,
        "near_token_cap_98": near_cap_98,
        "duplicate_output_ids": duplicate_output_ids,
        "failed_judge_keys_unresolved": len(unresolved_failed_vote_keys),
        "hash_mismatched_votes": hash_mismatched_votes,
        "unresolved_successful_votes": unresolved_successful_votes,
        "duplicate_vote_keys": duplicate_vote_keys,
        "votes_per_pair": json.dumps(dict(sorted(votes_per_pair.items()))),
        "generations_path": str(generations_path),
        "generation_failures_path": str(generation_failures_path),
        "judge_votes_path": str(votes_path),
    }

    cell_rows = []
    for (actor, task, comparison), group in group_jobs(jobs, ("actor", "task", "comparison")).items():
        group_pair_uids = {str(job["pair_uid"]) for job in group}
        group_complete_pairs = sum(
            pair_uid in current_hashes
            and sum(1 for key in successful_matching_vote_keys if key[0] == pair_uid) >= spec.expected_votes_per_pair
            for pair_uid in group_pair_uids
        )
        group_valid_outputs = sum(
            f"{pair_uid}::{suffix}" in valid_generation_ids
            for pair_uid in group_pair_uids
            for suffix in ("a", "b")
        )
        group_valid_votes = sum(1 for key in successful_matching_vote_keys if key[0] in group_pair_uids)
        cell_rows.append(
            {
                "status": "PASS" if group_complete_pairs == len(group_pair_uids) else "INCOMPLETE",
                "source_family": spec.source_family,
                "run_id": run_dir.name,
                "actor": actor,
                "task": task,
                "task_label": TASK_LABEL.get(task, task),
                "comparison": comparison,
                "jobs": len(group),
                "valid_outputs": group_valid_outputs,
                "expected_outputs": 2 * len(group),
                "valid_matching_votes": group_valid_votes,
                "expected_votes": spec.expected_votes_per_pair * len(group),
                "complete_pairs": group_complete_pairs,
                "manifest": str(spec.manifest),
            }
        )
    return row, cell_rows


def group_jobs(jobs: list[dict[str, Any]], fields: Iterable[str]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        key = tuple(str(job.get(field, "")) for field in fields)
        grouped[key].append(job)
    return grouped


def actor_task_matrix(cell_df: pd.DataFrame) -> pd.DataFrame:
    if cell_df.empty:
        return pd.DataFrame()
    grouped = (
        cell_df.groupby(["source_family", "actor", "task"], dropna=False)
        .agg(
            jobs=("jobs", "sum"),
            valid_outputs=("valid_outputs", "sum"),
            expected_outputs=("expected_outputs", "sum"),
            valid_matching_votes=("valid_matching_votes", "sum"),
            expected_votes=("expected_votes", "sum"),
            complete_pairs=("complete_pairs", "sum"),
        )
        .reset_index()
    )
    grouped["status"] = grouped.apply(
        lambda row: "PASS"
        if row["valid_outputs"] == row["expected_outputs"]
        and row["valid_matching_votes"] == row["expected_votes"]
        and row["complete_pairs"] == row["jobs"]
        else "INCOMPLETE",
        axis=1,
    )
    grouped["task_label"] = grouped["task"].map(TASK_LABEL).fillna(grouped["task"])
    return grouped[
        [
            "status",
            "source_family",
            "actor",
            "task",
            "task_label",
            "jobs",
            "complete_pairs",
            "valid_outputs",
            "expected_outputs",
            "valid_matching_votes",
            "expected_votes",
        ]
    ]


def write_markdown(summary: pd.DataFrame, cells: pd.DataFrame, matrix: pd.DataFrame, path: Path) -> None:
    status_counts = summary["status"].value_counts().to_dict() if not summary.empty else {}
    source_counts = (
        summary.groupby(["source_family", "status"]).size().reset_index(name="runs")
        if not summary.empty
        else pd.DataFrame()
    )
    incomplete = summary[summary["status"].ne("PASS")].copy() if not summary.empty else pd.DataFrame()
    lines = [
        "# Canonical Readiness Audit",
        "",
        "This audit is read-only on generation/judging logs. It checks prompt compliance, generation completeness, judge-vote completeness, hash-matched votes, duplicates, and truncation indicators.",
        "",
        f"Run manifests checked: {len(summary)}",
        f"Status counts: `{status_counts}`",
        "",
        "## By Source",
        "",
    ]
    if source_counts.empty:
        lines.append("No canonical manifests found.")
    else:
        lines.extend(markdown_table(source_counts).splitlines())
    lines += ["", "## Non-Passing Runs", ""]
    if incomplete.empty:
        lines.append("All checked runs pass.")
    else:
        cols = [
            "status",
            "source_family",
            "run_id",
            "jobs",
            "complete_pairs",
            "valid_outputs",
            "expected_outputs",
            "valid_matching_votes",
            "expected_votes",
            "prompt_errors",
        ]
        lines.extend(markdown_table(incomplete[cols]).splitlines())
    lines += ["", "## Actor-Task Incomplete Cells", ""]
    if matrix.empty or matrix["status"].eq("PASS").all():
        lines.append("All actor-task cells pass.")
    else:
        cols = [
            "status",
            "source_family",
            "actor",
            "task_label",
            "jobs",
            "complete_pairs",
            "valid_outputs",
            "expected_outputs",
            "valid_matching_votes",
            "expected_votes",
        ]
        lines.extend(markdown_table(matrix[matrix["status"].ne("PASS")][cols]).splitlines())
    lines += [
        "",
        "## Notes",
        "",
        "- `canonical_highn10` is expected to be incomplete while the high-N runs are still running.",
        "- `canonical_amount_base` is used for amount repeats 0-4 when those manifests exist; otherwise the audit falls back to `legacy_amount_base` rows from old four-comparison runs.",
        "- `legacy_amount_base` audits only `_amount` jobs from the old four-comparison runs; non-amount rows in those manifests are ignored.",
        "- `fund_wording_base` audits only high-low and moral jobs; headroom/framing rows in those manifests are ignored for canonical outcome readiness.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    shown = df.copy()
    for col in shown.columns:
        shown[col] = shown[col].map(format_cell)
    lines = [
        "| " + " | ".join(shown.columns) + " |",
        "| " + " | ".join("---" for _ in shown.columns) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in shown.columns) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4g}"
    text = str(value)
    return text.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="canonical_readiness_audit")
    args = parser.parse_args()

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for spec in source_specs():
        row, cells = summarize_manifest(spec)
        if row is None:
            continue
        summary_rows.append(row)
        cell_rows.extend(cells)

    if not summary_rows:
        raise SystemExit("No canonical run manifests found.")

    summary = pd.DataFrame(summary_rows)
    cells = pd.DataFrame(cell_rows)
    matrix = actor_task_matrix(cells)

    summary_path = ANALYSIS / f"{args.output_prefix}_runs.csv"
    cell_path = ANALYSIS / f"{args.output_prefix}_cells.csv"
    matrix_path = ANALYSIS / f"{args.output_prefix}_actor_task.csv"
    markdown_path = ANALYSIS / f"{args.output_prefix}.md"

    summary.to_csv(summary_path, index=False)
    cells.to_csv(cell_path, index=False)
    matrix.to_csv(matrix_path, index=False)
    write_markdown(summary, cells, matrix, markdown_path)

    print(f"runs: {summary_path}")
    print(f"cells: {cell_path}")
    print(f"actor-task: {matrix_path}")
    print(f"summary: {markdown_path}")
    print(summary["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
