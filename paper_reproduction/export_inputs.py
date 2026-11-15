"""Maintainer-only export of the explicitly selected paper inputs; no API calls.

Readers use reproduce.py. This records source checksums and compresses the
missing inputs without copying raw API caches or exploratory experiments.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_FILES = [
    "framed_user_strong_panel_pairs.csv",
    "canonical_highn_utility_gap_trend__gap-actor-z_trials.csv",
    "final_text_analysis_pair_deltas.csv",
    "framed_user_prompt_role_pair_deltas.csv",
    "moral_low_vs_framed_empty_current_pair_deltas_for_selection.csv",
]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def export_policy_framed_empty(source: Path, dest: Path) -> None:
    """Lossless JSON evidence plus small paired measurements, never raw archives."""
    panel = source / "outputs/api/policy_low_framed_empty_2026-11-14"
    features = source / "outputs/api/policy_low_framed_empty_features"
    inputs = {"panel/" + name: panel / name for name in (
        "sources.jsonl.gz", "pairs.jsonl.gz", "pair_accounting.jsonl.gz", "reference_votes.jsonl.gz",
        "judge_requests.jsonl.gz", "judge_responses.jsonl", "screen_requests.jsonl.gz",
        "screen_responses.jsonl", "manual_review.json", "archive_audit.json", "input_snapshot.json")}
    inputs.update({"features/" + name: features / name for name in (
        "protocol.jsonl", "rubric_requests.jsonl.gz", "rubric_responses.jsonl")})
    inputs["features/pair_measurements.csv"] = source / "docs/reviewer_response/policy_framed_empty/features/pair_features.csv"
    for name in ("run_policy_framed_empty", "run_screened_pair_features", "complete_policy_baseline_analysis",
                 "analyze_policy_baseline_features", "run_task_rubric_pilot", "run_task_rubric_feature_coding"):
        inputs[f"collection_code/{name}.py"] = source / f"src/utility_behavior_gap/scripts/{name}.py"
    inputs["collection_code/request_batch.py"] = source / "src/utility_behavior_gap/request_batch.py"
    target = dest / "policy_framed_empty.jsonl.gz"
    provenance_path = dest / "policy_framed_empty_provenance.json"
    if target.exists() or provenance_path.exists():
        raise FileExistsError("Policy export already exists; preserve it rather than silently replacing evidence")
    records = []
    with target.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        for key, path in inputs.items():
            count = 0
            if path.suffix == ".csv":
                rows = pd.read_csv(path).to_dict("records")
            elif path.suffix == ".py":
                rows = [{"source_text": path.read_text()}]
            elif path.suffix == ".json":
                rows = [json.loads(path.read_text())]
            else:
                opener = gzip.open if path.suffix == ".gz" else open
                with opener(path, "rt") as handle:
                    rows = [json.loads(line) for line in handle if line.strip()]
            for row in rows:
                zipped.write((json.dumps({"file": key, "row": row}, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
                count += 1
            records.append(dict(member=key, source=str(path.relative_to(source)), source_sha256=digest(path), rows=count))
    provenance_path.write_text(json.dumps(dict(path=str(target.relative_to(ROOT)), sha256=digest(target),
        sources=records, transformation="JSON objects preserved; CSV measurements parsed without filtering; collection source text preserved",
        excluded="No credentials, account-credit records, or unrelated raw run archives"), indent=2) + "\n")
    print(f"Exported policy evidence: {target.stat().st_size / 1024**2:.2f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--supplement", choices=["policy_framed_empty"])
    args = parser.parse_args()
    source = args.research_root.resolve()
    dest = ROOT / "data" / "processed"
    dest.mkdir(parents=True, exist_ok=True)
    if args.supplement == "policy_framed_empty":
        export_policy_framed_empty(source, dest)
        return
    records = []

    def export(path: Path, name: str, frame: pd.DataFrame | None = None) -> None:
        target = dest / name
        with tempfile.NamedTemporaryFile(dir=dest, suffix=".gz", delete=False) as temp:
            staging = Path(temp.name)
        if frame is None and path.suffix == ".gz":
            shutil.copy2(path, staging)
        elif frame is None:
            with path.open("rb") as inp, staging.open("wb") as out:
                with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as zipped:
                    shutil.copyfileobj(inp, zipped)
        else:
            with staging.open("wb") as out:
                with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as zipped:
                    zipped.write(frame.to_csv(index=False).encode("utf-8"))
        os.replace(staging, target)
        records.append({
            "path": str(target.relative_to(ROOT)), "sha256": digest(target),
            "source": str(path.relative_to(source)), "source_sha256": digest(path),
            "transformation": ("unchanged gzip" if path.suffix == ".gz" else "gzip only") if frame is None else "selected columns; rows and order unchanged",
            "rows": None if frame is None else len(frame),
        })

    for name in ANALYSIS_FILES:
        export(source / "outputs" / "analysis" / name, name + ".gz")
    catalog = source / "outputs/analysis/final_text_analysis_by_output.csv"
    flags = {"output_id", "missing_output", "generation_success_false", "finish_reason_missing",
             "non_stop_finish", "explicit_length_truncation", "empty_output"}
    export(catalog, "output_validity.csv.gz", pd.read_csv(catalog, usecols=lambda c: c in flags))
    labels = source / "outputs/analysis/canonical_highn_moral_refusal_classifications.jsonl"
    export(labels, labels.name + ".gz")
    followup = source / "outputs/api/runs/essay__qwen35_reasoning_stance_followup__qwen3.5-9b-or__seed-20260915__hash-2994eb3b2711/analysis/qwen35_reasoning_stance_pair_outcomes.csv"
    frame = pd.read_csv(followup)
    # The labels are joined afresh by the reproduction step, not treated as outcomes.
    remove = {f"{side}_{key}" for side in ("high", "low")
              for key in ("believes", "disbelieves", "suspicious", "dismisses", "label_source")}
    remove |= {"both_traces_classified", "either_disbelieves", "either_suspicious_or_disbelieves"}
    export(followup, "qwen35_followup_pair_inputs.csv.gz", frame.drop(columns=list(remove), errors="ignore"))
    export(source / "outputs/analysis/incentive_belief_qwen9b_followup_traces.jsonl",
           "qwen35_followup_local_labels.jsonl.gz")
    export(followup.parent.parent / "qwen35_old_plus_followup_reasoning_traces.jsonl.gz",
           "qwen35_old_plus_followup_reasoning_traces.jsonl.gz")
    (dest / "provenance.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Exported {len(records)} inputs, {sum((ROOT / r['path']).stat().st_size for r in records)/1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
