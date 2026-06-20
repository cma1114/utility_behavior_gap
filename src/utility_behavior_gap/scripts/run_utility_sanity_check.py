#!/usr/bin/env python3
"""Check whether rewritten intervention consequences preserve high-low utility order."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utility_behavior_gap.api_errors import is_transient_api_error
from utility_behavior_gap.io_utils import append_jsonl, read_csv_rows, write_jsonl
from utility_behavior_gap.job_builder import read_selected_pairs, rewritten_consequence, run_id_timestamp, slug_part
from utility_behavior_gap.openrouter import (
    MalformedOpenRouterResponse,
    OpenRouterClient,
    actor_model_id,
    response_without_message_content,
    response_text,
)
from utility_behavior_gap.paths import INPUTS, OUTPUT_API
from utility_behavior_gap.prompts import UTILITY_SANITY_SYSTEM_PROMPT, build_utility_sanity_prompt


UTILITY_SANITY_DIR = OUTPUT_API / "utility_sanity"
MANIFEST_NAME = "utility_sanity_jobs.jsonl"
RESULTS_NAME = "utility_sanity_results.jsonl"
UTILITY_SANITY_REASONING = {"effort": "none"}
UTILITY_SANITY_PROVIDER = {"require_parameters": True}


def selected_pairs_for_actor(actor: str, pair_set: str) -> list[dict[str, str]]:
    rows = [row for row in read_selected_pairs() if row["actor"] == actor]
    if pair_set != "all":
        rows = [row for row in rows if row["pair_set"] == pair_set]
    return rows


def utility_option_pairs_for_actor(
    *,
    actor: str,
    pairs_per_domain: int,
    seed: int,
    domains: set[str] | None,
) -> list[dict[str, str]]:
    rows = [
        row
        for row in read_csv_rows(INPUTS / "utility_options.csv")
        if row["actor"] == actor and (domains is None or row["domain"] in domains)
    ]
    by_domain: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row)
    pairs: list[dict[str, str]] = []
    rng = random.Random(seed)
    for domain in sorted(by_domain):
        domain_rows = sorted(by_domain[domain], key=lambda row: float(row["utility_mean"]))
        if len(domain_rows) < 3:
            raise ValueError(f"not enough utility options for {actor} {domain}: {len(domain_rows)}")
        tercile = len(domain_rows) // 3
        low_pool = domain_rows[:tercile]
        high_pool = domain_rows[-tercile:]
        possible = [(high, low) for high in high_pool for low in low_pool]
        if pairs_per_domain > len(possible):
            raise ValueError(
                f"requested {pairs_per_domain} pairs for {actor} {domain}, "
                f"but only {len(possible)} high-low pairs are available"
            )
        sampled = rng.sample(possible, pairs_per_domain)
        for idx, (high, low) in enumerate(sampled):
            high_u = float(high["utility_mean"])
            low_u = float(low["utility_mean"])
            pairs.append(
                {
                    "pair_set": "utility_options_sample",
                    "domain": domain,
                    "domain_label": high["domain_label"],
                    "pair_idx": str(idx),
                    "high_option_id": high["option_id"],
                    "low_option_id": low["option_id"],
                    "high_description": high["description"],
                    "low_description": low["description"],
                    "high_utility": high["utility_mean"],
                    "low_utility": low["utility_mean"],
                    "delta_u": str(high_u - low_u),
                }
            )
    return pairs


def ordered_conditions(order: str, pair_uid: str) -> tuple[str, str]:
    if order == "high-first":
        return "high", "low"
    if order == "low-first":
        return "low", "high"
    if order == "random":
        bit = int(hashlib.sha256(pair_uid.encode("utf-8")).hexdigest(), 16) % 2
        return ("high", "low") if bit == 0 else ("low", "high")
    raise ValueError(f"unknown order: {order}")


def build_jobs(
    *,
    actor: str,
    source: str,
    pair_set: str,
    orders: str,
    pairs_per_domain: int,
    seed: int,
    domains: set[str] | None,
) -> list[dict[str, Any]]:
    if source == "selected-pairs":
        pairs = selected_pairs_for_actor(actor, pair_set)
    elif source == "utility-options":
        pairs = utility_option_pairs_for_actor(
            actor=actor,
            pairs_per_domain=pairs_per_domain,
            seed=seed,
            domains=domains,
        )
    else:
        raise ValueError(f"unknown source: {source}")
    order_names = ["high-first", "low-first"] if orders == "both" else [orders]
    jobs: list[dict[str, Any]] = []
    for pair in pairs:
        high_consequence = rewritten_consequence(pair["high_description"])
        low_consequence = rewritten_consequence(pair["low_description"])
        base_uid = (
            f"utility_sanity:{source}:{actor}:{pair['pair_set']}:{pair['domain']}:"
            f"{pair['pair_idx']}:{pair.get('high_option_id', '')}:{pair.get('low_option_id', '')}"
        )
        for order_name in order_names:
            option_a_condition, option_b_condition = ordered_conditions(order_name, base_uid)
            consequence_by_condition = {
                "high": high_consequence,
                "low": low_consequence,
            }
            prompt = build_utility_sanity_prompt(
                consequence_a=consequence_by_condition[option_a_condition],
                consequence_b=consequence_by_condition[option_b_condition],
            )
            jobs.append(
                {
                    "job_uid": f"{base_uid}:{order_name}",
                    "base_pair_uid": base_uid,
                    "actor": actor,
                    "model": actor_model_id(actor),
                    "pair_set": pair["pair_set"],
                    "domain": pair["domain"],
                    "domain_label": pair["domain_label"],
                    "pair_idx": pair["pair_idx"],
                    "high_description": pair["high_description"],
                    "low_description": pair["low_description"],
                    "high_option_id": pair.get("high_option_id", ""),
                    "low_option_id": pair.get("low_option_id", ""),
                    "high_consequence": high_consequence,
                    "low_consequence": low_consequence,
                    "high_utility": pair["high_utility"],
                    "low_utility": pair["low_utility"],
                    "delta_u": pair["delta_u"],
                    "order": order_name,
                    "source": source,
                    "option_a_condition": option_a_condition,
                    "option_b_condition": option_b_condition,
                    "system_prompt": UTILITY_SANITY_SYSTEM_PROMPT,
                    "prompt": prompt,
                }
            )
    return jobs


def request_snapshot(*, job: dict[str, Any], temperature: float | None, max_tokens: int) -> dict[str, Any]:
    return {
        "script": "utility_behavior_gap.scripts.run_utility_sanity_check",
        "argv": sys.argv,
        "model": job["model"],
        "messages": [
            {"role": "system", "content": job["system_prompt"]},
            {"role": "user", "content": job["prompt"]},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": UTILITY_SANITY_REASONING,
        "provider": UTILITY_SANITY_PROVIDER,
    }


def parse_choice(text: str) -> str:
    normalized = text.strip().upper()
    if normalized in {"A", "B"}:
        return normalized
    match = re.search(r"\b(?:ANSWER|OPTION|CHOICE)\s*[:\-]?\s*([AB])\b", normalized)
    if match:
        return match.group(1)
    match = re.search(r"\b([AB])\b", normalized)
    return match.group(1) if match else "unparseable"


def result_condition(job: dict[str, Any], parsed_choice: str) -> str:
    if parsed_choice == "A":
        return job["option_a_condition"]
    if parsed_choice == "B":
        return job["option_b_condition"]
    return "unparseable"


def run_id_for_jobs(jobs: list[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(job, ensure_ascii=False, sort_keys=True) for job in jobs)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    actor = slug_part(jobs[0]["actor"] if jobs else "no-actor")
    pair_sets = "+".join(sorted({slug_part(job["pair_set"]) for job in jobs})) if jobs else "no-pairs"
    return f"utility_sanity__{actor}__{pair_sets}__{run_id_timestamp()}__hash-{digest}"


def print_plan(jobs: list[dict[str, Any]], run_dir: Path) -> None:
    print(f"utility sanity jobs: {len(jobs)}")
    print(f"run_dir: {run_dir}")
    if not jobs:
        return
    first = jobs[0]
    print(f"actor: {first['actor']} ({first['model']})")
    print(f"pair sets: {', '.join(sorted({job['pair_set'] for job in jobs}))}")
    print("first prompt:")
    print("--- system ---")
    print(first["system_prompt"])
    print("--- user ---")
    print(first["prompt"])


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def latest_run_dir() -> Path:
    dirs = [path for path in UTILITY_SANITY_DIR.glob("utility_sanity__*") if path.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"no utility sanity run directories found in {UTILITY_SANITY_DIR}")
    return max(dirs, key=lambda path: path.stat().st_mtime)


def print_status(run_dir: Path | None = None) -> None:
    run_dir = run_dir or latest_run_dir()
    manifest_path = run_dir / MANIFEST_NAME
    results_path = run_dir / RESULTS_NAME
    total = count_jsonl(manifest_path)
    done = count_jsonl(results_path)
    remaining = max(total - done, 0)
    pct = (done / total * 100) if total else 0.0
    print(f"run_dir: {run_dir}")
    print(f"completed: {done}/{total} ({pct:.1f}%)")
    print(f"remaining: {remaining}")
    print(f"results: {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", default="gpt-5.4-mini-or")
    parser.add_argument("--source", choices=["utility-options", "selected-pairs"], default="utility-options")
    parser.add_argument("--pair-set", choices=["default", "same_count", "all"], default="default")
    parser.add_argument("--pairs-per-domain", type=int, default=40)
    parser.add_argument("--domains", default="", help="Optional comma-separated domains.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--orders", choices=["both", "high-first", "low-first", "random"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--omit-temperature",
        action="store_true",
        help="Omit the temperature field from the OpenRouter payload. This is a routing diagnostic, not the paper-faithful default.",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--run", action="store_true", help="Actually call OpenRouter. Without this, only write a plan.")
    parser.add_argument("--status", action="store_true", help="Print progress for the latest utility sanity run and exit.")
    parser.add_argument("--run-dir", default="", help="Run directory to use with --status. Defaults to latest run.")
    args = parser.parse_args()
    if args.status:
        print_status(Path(args.run_dir) if args.run_dir else None)
        return
    temperature = None if args.omit_temperature else args.temperature

    domains = {item.strip() for item in args.domains.split(",") if item.strip()} or None
    jobs = build_jobs(
        actor=args.actor,
        source=args.source,
        pair_set=args.pair_set,
        orders=args.orders,
        pairs_per_domain=args.pairs_per_domain,
        seed=args.seed,
        domains=domains,
    )
    if args.limit is not None:
        jobs = jobs[: args.limit]
    run_id = run_id_for_jobs(jobs)
    run_dir = UTILITY_SANITY_DIR / run_id
    manifest_path = run_dir / MANIFEST_NAME
    results_path = run_dir / RESULTS_NAME
    write_jsonl(manifest_path, jobs)
    print_plan(jobs, run_dir)
    print(f"manifest: {manifest_path}")
    if not args.run:
        print("not running API calls; pass --run to execute")
        return

    client = OpenRouterClient()
    counts: Counter[str] = Counter()
    pair_votes: dict[str, list[str]] = {}
    for job in jobs:
        started = time.time()
        request = request_snapshot(job=job, temperature=temperature, max_tokens=args.max_tokens)
        raw_response: dict[str, Any]
        try:
            raw_response = client.chat_completion(
                model=job["model"],
                messages=request["messages"],
                temperature=temperature,
                max_tokens=args.max_tokens,
                reasoning=UTILITY_SANITY_REASONING,
                provider=UTILITY_SANITY_PROVIDER,
            )
            text = response_text(raw_response)
        except MalformedOpenRouterResponse as exc:
            raw_response = exc.response
            text = ""
        except RuntimeError as exc:
            if not is_transient_api_error(exc):
                raise
            raw_response = {"error": str(exc)}
            text = ""
        parsed_choice = parse_choice(text)
        condition = result_condition(job, parsed_choice)
        counts[condition] += 1
        pair_votes.setdefault(job["base_pair_uid"], []).append(condition)
        append_jsonl(
            results_path,
            {
                "job_uid": job["job_uid"],
                "base_pair_uid": job["base_pair_uid"],
                "actor": job["actor"],
                "model": job["model"],
                "domain": job["domain"],
                "pair_set": job["pair_set"],
                "order": job["order"],
                "option_a_condition": job["option_a_condition"],
                "option_b_condition": job["option_b_condition"],
                "parsed_choice": parsed_choice,
                "winner_condition": condition,
                "raw_text": text,
                "latency_s": round(time.time() - started, 3),
                "request": request,
                "raw_response": response_without_message_content(raw_response),
                "job": job,
            },
        )
    print(f"wrote {sum(counts.values())} utility sanity results to {results_path}")
    print(f"vote split: {dict(counts)}")
    if counts["high"] + counts["low"]:
        rate = counts["high"] / (counts["high"] + counts["low"])
        print(f"high-side preference rate excluding unparseable: {rate:.3f}")
    pair_counts: Counter[str] = Counter()
    for votes in pair_votes.values():
        resolved = [vote for vote in votes if vote in {"high", "low"}]
        if not resolved:
            pair_counts["all_unparseable"] += 1
        elif all(vote == "high" for vote in resolved) and len(resolved) == len(votes):
            pair_counts["high_all_orders"] += 1
        elif all(vote == "low" for vote in resolved) and len(resolved) == len(votes):
            pair_counts["low_all_orders"] += 1
        elif len(resolved) < len(votes):
            pair_counts["partly_unparseable"] += 1
        else:
            pair_counts["order_split"] += 1
    print(f"pair-level split: {dict(pair_counts)}")


if __name__ == "__main__":
    main()
