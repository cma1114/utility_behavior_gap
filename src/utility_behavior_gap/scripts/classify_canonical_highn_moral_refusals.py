#!/usr/bin/env python3
"""Classify current canonical moral-condition outputs for refusal content.

This is the current-data version of ``classify_modgrid_moral_refusals``. It
reads only:

- fund-wording base manifests, repeats 0-4;
- canonical high-N extension manifests, repeats 5-9.

It writes hash-checked labels to:

``outputs/analysis/canonical_highn_moral_refusal_classifications.jsonl``

The output hash is stored and used by downstream analysis so labels cannot be
silently reused across prompt variants that recycled output IDs.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.paths import ROOT
from utility_behavior_gap.scripts.classify_moral_refusals import (
    CLASSIFIER_MODEL,
    CLASSIFY_TEMPLATE,
    TASK_ARTIFACT,
    parse_label,
)


RUNS = ROOT / "outputs" / "api" / "runs"
OUT_PATH = ROOT / "outputs" / "analysis" / "canonical_highn_moral_refusal_classifications.jsonl"
MANIFEST_GLOBS = [
    "fund_wording_rerun_manifests__*.tsv",
    "canonical_highn10_manifests__*.tsv",
]
TASK_KEY = {
    "essay": "essay",
    "translation": "translation",
    "grant_proposal_abstract": "grant",
    "incident_postmortem": "incident",
}
VALID_LABELS = {"clean", "partial_refusal", "full_refusal", "degenerate"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def manifest_paths() -> list[Path]:
    paths: list[Path] = []
    for pattern in MANIFEST_GLOBS:
        for tsv in sorted(RUNS.glob(pattern)):
            with tsv.open(encoding="utf-8") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) == 3:
                        paths.append(Path(parts[2]))
    return sorted(set(paths))


def load_units() -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for manifest in manifest_paths():
        run_dir = manifest.parent
        jobs = read_jsonl(manifest)
        generations = {row["output_id"]: row for row in read_jsonl(run_dir / "generations.jsonl")}
        for job in jobs:
            comparison = str(job.get("comparison", ""))
            if not comparison.endswith("_moral"):
                continue
            for side in ("a", "b"):
                output_id = f"{job['pair_uid']}::{side}"
                generation = generations.get(output_id)
                if not generation:
                    continue
                output_hash = output_text_fingerprint(generation)
                key = (output_id, output_hash)
                if key in seen:
                    continue
                seen.add(key)
                units.append(
                    {
                        "output_id": output_id,
                        "output_hash": output_hash,
                        "pair_uid": str(job["pair_uid"]),
                        "task": TASK_KEY.get(str(job["task"]), str(job["task"])),
                        "task_raw": str(job["task"]),
                        "actor": str(job["actor"]),
                        "condition": str(generation.get("condition", job.get(f"condition_{side}", ""))),
                        "source_run_id": run_dir.name,
                        "manifest": str(manifest),
                        "text": str(generation.get("output_text") or ""),
                    }
                )
    return units


def existing_keys() -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    for row in read_jsonl(OUT_PATH):
        label = row.get("label")
        output_id = str(row.get("output_id", ""))
        output_hash = str(row.get("output_hash", ""))
        if output_id and output_hash and label in VALID_LABELS:
            done.add((output_id, output_hash))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    units = load_units()
    done = existing_keys()
    todo = [unit for unit in units if (unit["output_id"], unit["output_hash"]) not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"{len(units)} current moral outputs; {len(units) - len(todo)} classified; {len(todo)} to classify with {CLASSIFIER_MODEL}")
    by_task = Counter(unit["task_raw"] for unit in todo)
    by_condition = Counter(unit["condition"] for unit in todo)
    print(f"to classify by task: {dict(sorted(by_task.items()))}")
    print(f"to classify by condition: {dict(sorted(by_condition.items()))}")
    print(f"output: {OUT_PATH}")
    if args.plan_only or not todo:
        return

    from utility_behavior_gap.openrouter import OpenRouterClient, response_text

    client = OpenRouterClient(timeout_s=120.0, max_retries=3)
    lock = threading.Lock()
    counts: Counter[str] = Counter()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = OUT_PATH.open("a", encoding="utf-8")

    def classify(unit: dict[str, Any]) -> dict[str, Any]:
        text = str(unit["text"])
        if not text.strip():
            return {**unit, "label": "degenerate", "evidence": "empty output", "classifier": "rule:empty"}
        prompt = CLASSIFY_TEMPLATE.format(artifact=TASK_ARTIFACT[unit["task"]], output=text[:12000])
        response = client.chat_completion(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
            reasoning={"effort": "minimal", "exclude": True},
        )
        parsed = parse_label(response_text(response))
        if parsed is None:
            return {
                **unit,
                "label": "parse_error",
                "evidence": response_text(response)[:200],
                "classifier": CLASSIFIER_MODEL,
            }
        return {
            **unit,
            "label": parsed["label"],
            "evidence": str(parsed.get("evidence", ""))[:200],
            "classifier": CLASSIFIER_MODEL,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(classify, unit): unit for unit in todo}
        for idx, future in enumerate(as_completed(futures), 1):
            unit = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {**unit, "label": "api_error", "evidence": str(exc)[:200], "classifier": CLASSIFIER_MODEL}
            row = {key: value for key, value in row.items() if key != "text"}
            counts[str(row["label"])] += 1
            with lock:
                out.write(json.dumps(row, sort_keys=True) + "\n")
                out.flush()
            if idx % 250 == 0 or idx == len(todo):
                print(f"  {idx}/{len(todo)} {dict(counts)}", flush=True)
    out.close()
    print(f"done: {dict(counts)}; rerun to retry api_error/parse_error rows")


if __name__ == "__main__":
    main()
