"""Generate R0 (bare-task) outputs: blank system prompt, task instruction only.

R0 is the reference arm of the designed modulation experiment
(notes/designed_modulation_experiment_protocol.md): no system prompt, no
suffix, no framing — the unmodulated floor. This script generates arm-level
outputs (NOT pairs); judged contrasts referencing them are built separately.

Conventions mirrored from run_generation:
- append-only, resumable: outputs keyed by stable output_id
  ``r0:{task}:{actor}:i{item_index}:r{repeat}``; existing valid ids skipped
- temperature 1.0, reasoning disabled ({"effort": "none"}); responses that
  spent reasoning tokens or did not finish with ``stop`` are logged as
  failures, not stored
- transient API failures logged to ``outputs/api/r0_generation_failures.jsonl``
  and retried on the next invocation

Essay max-tokens default to 900 to match the existing gpt-5.4-mini R0 outputs
from the 2026-06-10 pilot; other tasks use the paper's Table 6 budgets.

Usage:
  python -m utility_behavior_gap.scripts.run_r0_generations \
      --tasks essay --actors deepseek-v3.2-or,... --repeats 5 --workers 8
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from utility_behavior_gap.constants import ACTOR_MODEL_ID, ACTORS
from utility_behavior_gap.io_utils import read_csv_rows
from utility_behavior_gap.openrouter import OpenRouterClient, response_text
from utility_behavior_gap.paths import INPUTS, OUTPUT_API
from utility_behavior_gap.job_builder import clean_essay_direct_prompt

OUT_PATH = OUTPUT_API / "r0_generations.jsonl"
FAIL_PATH = OUTPUT_API / "r0_generation_failures.jsonl"

GENERATION_REASONING = {"effort": "none"}
MAX_TOKENS = {
    "essay": 900,  # matches the 2026-06-10 pilot's bare arm
    "translation": 600,
    "grant_proposal_abstract": 1000,
    "incident_postmortem": 3000,
}


def bare_prompt(task_row: dict[str, str]) -> str:
    if task_row["task"] == "essay":
        return clean_essay_direct_prompt(task_row["item_label"])
    return task_row["base_prompt"]


def build_units(tasks: set[str], actors: set[str], repeats: int) -> list[dict]:
    units = []
    for task_row in read_csv_rows(INPUTS / "task_items.csv"):
        if task_row["task"] not in tasks:
            continue
        for actor in sorted(actors):
            for rep in range(repeats):
                units.append(dict(
                    output_id=(
                        f"r0:{task_row['task']}:{actor}:"
                        f"i{task_row['item_id']}:r{rep}"
                    ),
                    actor=actor,
                    task=task_row["task"],
                    item_id=task_row["item_id"],
                    item_label=task_row["item_label"],
                    repeat=rep,
                    system_prompt="",
                    user_prompt=bare_prompt(task_row),
                    max_tokens=MAX_TOKENS[task_row["task"]],
                ))
    return units


def existing_valid_ids() -> set[str]:
    done = set()
    if OUT_PATH.exists():
        for line in open(OUT_PATH):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok"):
                done.add(row["output_id"])
    return done


def reasoning_tokens(response: dict) -> int:
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return int(details.get("reasoning_tokens") or 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="essay", help="comma-separated task ids")
    ap.add_argument("--actors", default="", help="comma-separated actor ids (default: all)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    actors = {a.strip() for a in args.actors.split(",") if a.strip()} or set(ACTORS)
    unknown = actors - set(ACTOR_MODEL_ID)
    if unknown:
        raise SystemExit(f"unknown actors: {sorted(unknown)}")

    units = build_units(tasks, actors, args.repeats)
    done = existing_valid_ids()
    todo = [u for u in units if u["output_id"] not in done]
    print(f"{len(units)} R0 units; {len(done & {u['output_id'] for u in units})} already done; {len(todo)} to generate")
    if args.plan_only or not todo:
        return

    client = OpenRouterClient(timeout_s=180.0, max_retries=3)
    lock = threading.Lock()
    out = open(OUT_PATH, "a")
    fail = open(FAIL_PATH, "a")
    counts = {"ok": 0, "failed": 0}

    def generate(unit: dict) -> tuple[dict, bool]:
        response = client.chat_completion(
            model=ACTOR_MODEL_ID[unit["actor"]],
            messages=[{"role": "user", "content": unit["user_prompt"]}],
            temperature=1.0,
            max_tokens=unit["max_tokens"],
            reasoning=GENERATION_REASONING,
        )
        text = response_text(response)
        finish = (response.get("choices") or [{}])[0].get("finish_reason")
        rtok = reasoning_tokens(response)
        record = dict(unit, comparison="r0_bare", condition="r0",
                      model=ACTOR_MODEL_ID[unit["actor"]],
                      output_text=text, finish_reason=finish,
                      reasoning_tokens=rtok,
                      provider=response.get("provider"),
                      usage=response.get("usage"),
                      ok=bool(text.strip()) and finish == "stop" and rtok == 0)
        return record, record["ok"]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate, u): u for u in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            unit = futures[fut]
            try:
                record, ok = fut.result()
            except Exception as exc:
                record, ok = dict(unit, error=str(exc)[:300], ok=False), False
            with lock:
                if ok:
                    out.write(json.dumps(record, sort_keys=True) + "\n")
                    out.flush()
                    counts["ok"] += 1
                else:
                    fail.write(json.dumps(record, sort_keys=True) + "\n")
                    fail.flush()
                    counts["failed"] += 1
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} {counts}", flush=True)
    out.close()
    fail.close()
    print(f"done: {counts}; rerun this command to retry failures")


if __name__ == "__main__":
    main()
