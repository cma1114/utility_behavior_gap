#!/usr/bin/env python3
"""Check whether saved generation rows contain returned reasoning traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utility_behavior_gap.paths import ROOT


RUNS = ROOT / "outputs" / "api" / "runs"
DEFAULT_MANIFEST_GLOB = "highlow_reasoning_traces_medium_manifests__*.tsv"
TASKS = {"essay", "translation", "grant_proposal_abstract", "incident_postmortem"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_manifest_list(path: Path) -> list[Path]:
    manifests: list[Path] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                manifests.append(Path(parts[2]))
    return manifests


def selected_tasks(value: str) -> set[str] | None:
    value = value.strip()
    if not value or value == "all":
        return None
    out = {part.strip() for part in value.split(",") if part.strip()}
    unknown = out - TASKS
    if unknown:
        raise ValueError(f"unknown task(s): {', '.join(sorted(unknown))}")
    return out


def manifest_paths(args: argparse.Namespace) -> list[Path]:
    task_filter = selected_tasks(str(args.tasks or ""))
    paths: list[Path] = []
    if args.manifest_list:
        for manifest_list in args.manifest_list:
            paths.extend(read_manifest_list(manifest_list))
    else:
        for manifest_list in sorted(RUNS.glob(DEFAULT_MANIFEST_GLOB)):
            paths.extend(read_manifest_list(manifest_list))
    if task_filter is not None:
        paths = [path for path in paths if path.parent.name.split("__", 1)[0] in task_filter]
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def response_message(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_response", {})
    choices = raw.get("choices", []) if isinstance(raw, dict) else []
    if not choices or not isinstance(choices[0], dict):
        return {}
    message = choices[0].get("message", {})
    return message if isinstance(message, dict) else {}


def reasoning_tokens(row: dict[str, Any]) -> int:
    details = row.get("usage", {}).get("completion_tokens_details", {})
    try:
        return int(details.get("reasoning_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def trace_fields(row: dict[str, Any]) -> dict[str, Any]:
    message = response_message(row)
    reasoning = message.get("reasoning")
    reasoning_content = message.get("reasoning_content")
    reasoning_details = message.get("reasoning_details")
    reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
    reasoning_content_chars = len(reasoning_content) if isinstance(reasoning_content, str) else 0
    reasoning_details_chars = (
        len(json.dumps(reasoning_details, ensure_ascii=False, sort_keys=True)) if reasoning_details else 0
    )
    return {
        "has_reasoning": bool(reasoning_chars),
        "has_reasoning_content": bool(reasoning_content_chars),
        "has_reasoning_details": bool(reasoning_details_chars),
        "trace_chars": reasoning_chars + reasoning_content_chars + reasoning_details_chars,
        "message_keys": ",".join(sorted(message)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-list", type=Path, action="append", help="Manifest-list TSV to inspect.")
    parser.add_argument("--tasks", default="all", help="Comma-separated task ids to include, or all.")
    args = parser.parse_args()

    manifests = manifest_paths(args)
    if not manifests:
        raise FileNotFoundError(f"No manifest lists found matching {RUNS / DEFAULT_MANIFEST_GLOB}")

    totals = {
        "successful_outputs": 0,
        "failed_outputs": 0,
        "rows": 0,
        "outputs_with_reasoning_tokens": 0,
        "outputs_with_trace": 0,
        "outputs_with_message_reasoning": 0,
        "outputs_with_reasoning_content": 0,
        "outputs_with_reasoning_details": 0,
        "trace_chars": 0,
    }
    first_trace: dict[str, Any] | None = None
    first_no_trace_with_tokens: dict[str, Any] | None = None

    for manifest in manifests:
        run_dir = manifest.parent
        rows = [
            *read_jsonl(run_dir / "generations.jsonl"),
            *read_jsonl(run_dir / "generation_failures.jsonl"),
        ]
        for row in rows:
            fields = trace_fields(row)
            tokens = reasoning_tokens(row)
            has_trace = fields["trace_chars"] > 0
            totals["rows"] += 1
            totals["successful_outputs"] += int(row.get("success") is not False)
            totals["failed_outputs"] += int(row.get("success") is False)
            totals["outputs_with_reasoning_tokens"] += int(tokens > 0)
            totals["outputs_with_trace"] += int(has_trace)
            totals["outputs_with_message_reasoning"] += int(fields["has_reasoning"])
            totals["outputs_with_reasoning_content"] += int(fields["has_reasoning_content"])
            totals["outputs_with_reasoning_details"] += int(fields["has_reasoning_details"])
            totals["trace_chars"] += int(fields["trace_chars"])
            sample = {
                "run_dir": str(run_dir),
                "output_id": row.get("output_id", ""),
                "model": row.get("model", ""),
                "reasoning_tokens": tokens,
                "trace_chars": fields["trace_chars"],
                "message_keys": fields["message_keys"],
            }
            if has_trace and first_trace is None:
                first_trace = sample
            if tokens > 0 and not has_trace and first_no_trace_with_tokens is None:
                first_no_trace_with_tokens = sample

    rows = totals["rows"]
    mean_trace_chars = totals["trace_chars"] / rows if rows else 0
    print(f"successful_outputs: {totals['successful_outputs']}")
    print(f"failed_outputs: {totals['failed_outputs']}")
    print(f"rows_inspected: {rows}")
    print(f"outputs_with_reasoning_tokens: {totals['outputs_with_reasoning_tokens']}")
    print(f"outputs_with_trace: {totals['outputs_with_trace']}")
    print(f"outputs_with_message.reasoning: {totals['outputs_with_message_reasoning']}")
    print(f"outputs_with_message.reasoning_content: {totals['outputs_with_reasoning_content']}")
    print(f"outputs_with_message.reasoning_details: {totals['outputs_with_reasoning_details']}")
    print(f"mean_trace_chars_per_output: {mean_trace_chars:.1f}")
    if first_trace:
        print("first_trace_row:")
        print(json.dumps(first_trace, indent=2, sort_keys=True))
    if first_no_trace_with_tokens:
        print("first_no_trace_with_tokens_row:")
        print(json.dumps(first_no_trace_with_tokens, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
