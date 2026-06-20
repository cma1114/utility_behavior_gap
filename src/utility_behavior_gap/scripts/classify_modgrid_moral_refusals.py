"""LLM-classify the modgrid moral-condition outputs for refusal content.

Applies the moral-audit classifier (same prompt, labels, and model as
``classify_moral_refusals``) to the new modgrid moral arms: every
``modgrid_{task}_moral`` output (moral_good and moral_bad, all actors, all
tasks) from ``outputs/api/runs/*__4-comparisons__*/generations.jsonl``.

Per the protocol's exclusion rules, this is the ONLY model-based screen, and
it applies to moral arms only. Results feed the strict screen (pair dropped if
either arm carries refusal content) and the reported refusal rates.

Output: ``outputs/analysis/modgrid_moral_refusal_classifications.jsonl``
(append-only; keyed by output_id; rerun to retry failures).

Usage:
  python -m utility_behavior_gap.scripts.classify_modgrid_moral_refusals --plan-only
  python -m utility_behavior_gap.scripts.classify_modgrid_moral_refusals --workers 12
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from utility_behavior_gap.paths import ROOT
from utility_behavior_gap.scripts.classify_moral_refusals import (
    CLASSIFIER_MODEL,
    CLASSIFY_TEMPLATE,
    TASK_ARTIFACT,
    parse_label,
)

OUT_PATH = ROOT / "outputs" / "analysis" / "modgrid_moral_refusal_classifications.jsonl"
RUNS_GLOB = "*__4-comparisons__*/generations.jsonl"
TASK_KEY = {"essay": "essay", "translation": "translation",
            "grant_proposal_abstract": "grant", "incident_postmortem": "incident"}


def load_units() -> list[dict]:
    units, seen = [], set()
    for path in sorted((ROOT / "outputs" / "api" / "runs").glob(RUNS_GLOB)):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = r.get("pair_uid", "")
            if "_moral:" not in uid or r["output_id"] in seen:
                continue
            seen.add(r["output_id"])
            task = uid.split(":")[2]
            units.append(dict(
                output_id=r["output_id"], pair_uid=uid,
                task=TASK_KEY.get(task, task), actor=r["actor"],
                condition=r["condition"], text=r.get("output_text") or ""))
    return units


def existing_ids() -> set[str]:
    done = set()
    if OUT_PATH.exists():
        for line in open(OUT_PATH):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("label") in {"clean", "partial_refusal", "full_refusal", "degenerate"}:
                done.add(row["output_id"])
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    units = load_units()
    done = existing_ids()
    todo = [u for u in units if u["output_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(units)} moral outputs; {len(units) - len(todo)} classified; "
          f"{len(todo)} to classify with {CLASSIFIER_MODEL}")
    if args.plan_only or not todo:
        return

    from utility_behavior_gap.openrouter import OpenRouterClient, response_text

    client = OpenRouterClient(timeout_s=120.0, max_retries=3)
    lock = threading.Lock()
    out = open(OUT_PATH, "a")
    counts: Counter = Counter()

    def classify(unit: dict) -> dict:
        if not unit["text"].strip():
            return {**unit, "label": "degenerate", "evidence": "empty output",
                    "classifier": "rule:empty"}
        prompt = CLASSIFY_TEMPLATE.format(
            artifact=TASK_ARTIFACT[unit["task"]], output=unit["text"][:12000])
        response = client.chat_completion(
            model=CLASSIFIER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=150,
            reasoning={"effort": "minimal", "exclude": True})
        obj = parse_label(response_text(response))
        if obj is None:
            return {**unit, "label": "parse_error",
                    "evidence": response_text(response)[:200],
                    "classifier": CLASSIFIER_MODEL}
        return {**unit, "label": obj["label"],
                "evidence": str(obj.get("evidence", ""))[:200],
                "classifier": CLASSIFIER_MODEL}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(classify, u): u for u in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            unit = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {**unit, "label": "api_error", "evidence": str(exc)[:200],
                       "classifier": CLASSIFIER_MODEL}
            row = {k: v for k, v in row.items() if k != "text"}
            counts[row["label"]] += 1
            with lock:
                out.write(json.dumps(row, sort_keys=True) + "\n")
                out.flush()
            if n % 500 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {dict(counts)}", flush=True)
    out.close()
    print(f"done: {dict(counts)}; rerun to retry any api_error/parse_error rows")


if __name__ == "__main__":
    main()
