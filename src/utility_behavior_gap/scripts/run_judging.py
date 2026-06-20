#!/usr/bin/env python3
"""Run blind pairwise judging through OpenRouter."""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from utility_behavior_gap.api_errors import is_backoff_api_error, is_transient_api_error
from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.io_utils import append_jsonl, read_jsonl
from utility_behavior_gap.job_builder import read_generation_jobs
from utility_behavior_gap.openrouter import (
    MalformedOpenRouterResponse,
    OpenRouterClient,
    judge_model_ids,
    response_without_message_content,
    response_text,
)
from utility_behavior_gap.paths import OUTPUT_API
from utility_behavior_gap.prompts import build_essay_judge_prompt, build_pairwise_judge_prompt


GENERATIONS = OUTPUT_API / "generations.jsonl"
JUDGE_VOTES = OUTPUT_API / "judge_votes.jsonl"
JUDGE_REASONING = {"effort": "minimal", "exclude": True}


def parse_winner(text: str) -> str:
    match = re.search(r"winner\s*:\s*(A|B|tie)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    match = re.search(r"answer\s*:\s*(A|B|X|Y|tie)\b", text, flags=re.IGNORECASE)
    if match:
        value = match.group(1).lower()
        return {"x": "a", "y": "b"}.get(value, value)
    return "unresolved"


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def run_log_path(jobs: list[dict[str, Any]], field: str, fallback: Path) -> Path:
    values = {str(job.get(field) or "") for job in jobs}
    values.discard("")
    if len(values) == 1:
        return Path(values.pop())
    return fallback


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def read_jobs_from_args(run_dir: Path | None, manifest: Path | None) -> list[dict[str, Any]]:
    if run_dir is not None and manifest is not None:
        raise ValueError("use either --run-dir or --manifest, not both")
    if run_dir is not None:
        return read_jsonl(run_dir / "generation_jobs.jsonl")
    if manifest is not None:
        return read_jsonl(manifest)
    return read_generation_jobs()


def generation_map(paths: list[Path] | None = None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in unique_paths(paths or [GENERATIONS]):
        for row in read_jsonl_if_exists(path):
            rows[row["output_id"]] = row
    return rows


def current_pair_hashes(
    jobs: list[dict[str, Any]],
    generations: dict[str, dict[str, Any]],
) -> dict[str, tuple[str, str]]:
    pair_hashes: dict[str, tuple[str, str]] = {}
    for job in jobs:
        out_a = generations.get(f"{job['pair_uid']}::a")
        out_b = generations.get(f"{job['pair_uid']}::b")
        if out_a is None or out_b is None:
            continue
        pair_hashes[job["pair_uid"]] = (
            output_text_fingerprint(out_a),
            output_text_fingerprint(out_b),
        )
    return pair_hashes


def vote_matches_current_outputs(row: dict[str, Any], expected_hashes: tuple[str, str]) -> bool:
    return (
        row.get("source_output_a_hash") == expected_hashes[0]
        and row.get("source_output_b_hash") == expected_hashes[1]
    )


def existing_vote_keys(
    pair_hashes: dict[str, tuple[str, str]],
    *,
    retry_unresolved: bool = False,
    paths: list[Path] | None = None,
) -> set[tuple[str, str, bool]]:
    """Keys of valid existing votes, including presentation orientation."""
    keys: set[tuple[str, str, bool]] = set()
    for path in unique_paths(paths or [JUDGE_VOTES]):
        for row in read_jsonl_if_exists(path):
            if row.get("success") is False:
                continue
            pair_uid = row["pair_uid"]
            expected_hashes = pair_hashes.get(pair_uid)
            if expected_hashes is None:
                continue
            if not vote_matches_current_outputs(row, expected_hashes):
                continue
            if not retry_unresolved or row.get("winner_condition") != "unresolved":
                keys.add((pair_uid, row["judge_model"], bool(row.get("flipped"))))
    return keys


def judged_in_any_orientation(done: set[tuple[str, str, bool]], pair_uid: str, judge_model: str) -> bool:
    return (pair_uid, judge_model, False) in done or (pair_uid, judge_model, True) in done


def finish_reason(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    return str(choices[0].get("finish_reason", ""))


def judge_prompt(job: dict[str, Any], output_a: dict[str, Any], output_b: dict[str, Any]) -> str:
    if job["task"] == "essay":
        return build_essay_judge_prompt(
            topic=job["item_label"],
            response_x=output_a["output_text"],
            response_y=output_b["output_text"],
        )
    return build_pairwise_judge_prompt(
        axis=job["axis"],
        axis_def=job["axis_definition"],
        base_prompt=job["base_prompt"],
        output_a=output_a["output_text"],
        output_b=output_b["output_text"],
    )


def judging_request_snapshot(
    *,
    judge_model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    flipped: bool,
    source_output_a_id: str,
    source_output_b_id: str,
    displayed_output_a_id: str,
    displayed_output_b_id: str,
) -> dict[str, Any]:
    return {
        "script": "utility_behavior_gap.scripts.run_judging",
        "argv": sys.argv,
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": JUDGE_REASONING,
        "seed": seed,
        "flipped": flipped,
        "source_output_a_id": source_output_a_id,
        "source_output_b_id": source_output_b_id,
        "displayed_output_a_id": displayed_output_a_id,
        "displayed_output_b_id": displayed_output_b_id,
    }


def judge_vote_row(
    *,
    job: dict[str, Any],
    judge_idx: int,
    judge_model: str,
    flip: bool,
    raw_text: str,
    response: dict[str, Any],
    reason: str,
    success: bool,
    latency_s: float,
    temperature: float,
    max_tokens: int,
    request_snapshot: dict[str, Any],
    out_a: dict[str, Any],
    out_b: dict[str, Any],
    output_a: dict[str, Any],
    output_b: dict[str, Any],
    source_output_a_hash: str,
    source_output_b_hash: str,
    dry_run: bool,
) -> dict[str, Any]:
    parsed = parse_winner(raw_text)
    if parsed == "a":
        winner_condition = output_a["condition"]
    elif parsed == "b":
        winner_condition = output_b["condition"]
    elif parsed == "tie":
        winner_condition = "tie"
    else:
        winner_condition = "unresolved"
    return {
        "pair_uid": job["pair_uid"],
        "judge_index": judge_idx + 1,
        "judge_model": judge_model,
        "flipped": flip,
        "vote_raw": raw_text,
        "parsed_winner": parsed,
        "winner_condition": winner_condition,
        "success": success,
        "latency_s": round(latency_s, 3),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": JUDGE_REASONING if not dry_run else {},
        "finish_reason": reason,
        "source_output_a_id": out_a["output_id"],
        "source_output_b_id": out_b["output_id"],
        "source_output_a_hash": source_output_a_hash,
        "source_output_b_hash": source_output_b_hash,
        "displayed_output_a_id": output_a["output_id"],
        "displayed_output_b_id": output_b["output_id"],
        "usage": response.get("usage", {}),
        "raw_response": response_without_message_content(response),
        "request": request_snapshot,
        "job": job,
        "run_id": job.get("run_id", ""),
    }


def run_judge_request(
    *,
    job: dict[str, Any],
    out_a: dict[str, Any],
    out_b: dict[str, Any],
    judge_idx: int,
    judge_model: str,
    flip: bool,
    temperature: float,
    max_tokens: int,
    seed: int,
    dry_run: bool,
) -> tuple[dict[str, Any], bool]:
    source_output_a_hash = output_text_fingerprint(out_a)
    source_output_b_hash = output_text_fingerprint(out_b)
    output_a = out_b if flip else out_a
    output_b = out_a if flip else out_b
    prompt = judge_prompt(job, output_a, output_b)
    request_snapshot = judging_request_snapshot(
        judge_model=judge_model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        flipped=flip,
        source_output_a_id=out_a["output_id"],
        source_output_b_id=out_b["output_id"],
        displayed_output_a_id=output_a["output_id"],
        displayed_output_b_id=output_b["output_id"],
    )
    started = time.time()
    stop_after_backoff_error = False
    if dry_run:
        response: dict[str, Any] = {}
        raw_text = "winner: tie\nreason: dry run"
        reason = "dry_run"
        success = True
    else:
        client = OpenRouterClient()
        try:
            response = client.chat_completion(
                model=judge_model,
                messages=request_snapshot["messages"],
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=JUDGE_REASONING,
            )
            raw_text = response_text(response)
            reason = finish_reason(response)
            success = True
        except MalformedOpenRouterResponse as exc:
            response = exc.response
            raw_text = ""
            reason = "malformed_response"
            success = False
        except RuntimeError as exc:
            if not is_transient_api_error(exc):
                raise
            response = {}
            raw_text = ""
            reason = "api_error"
            success = False
            stop_after_backoff_error = is_backoff_api_error(exc)

    row = judge_vote_row(
        job=job,
        judge_idx=judge_idx,
        judge_model=judge_model,
        flip=flip,
        raw_text=raw_text,
        response=response,
        reason=reason,
        success=success,
        latency_s=time.time() - started,
        temperature=temperature,
        max_tokens=max_tokens,
        request_snapshot=request_snapshot,
        out_a=out_a,
        out_b=out_b,
        output_a=output_a,
        output_b=output_b,
        source_output_a_hash=source_output_a_hash,
        source_output_b_hash=source_output_b_hash,
        dry_run=dry_run,
    )
    return row, stop_after_backoff_error


def pending_judge_requests(
    *,
    jobs: list[dict[str, Any]],
    generations: dict[str, dict[str, Any]],
    done: set[tuple[str, str, bool]],
    judges: list[str],
    seed: int,
    limit: int | None,
    orders: str = "single",
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, str, bool]]:
    """Build the judge-call queue.

    orders="single": one vote per (pair, judge), random presentation order.
    orders="both": two votes per (pair, judge), one in each presentation
    order; resumability is per orientation, so a single-order history is
    topped up with the missing orientation rather than redone.
    """
    rng = random.Random(seed)
    pending: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, str, bool]] = []
    for job in jobs:
        out_a = generations.get(f"{job['pair_uid']}::a")
        out_b = generations.get(f"{job['pair_uid']}::b")
        if out_a is None or out_b is None:
            continue
        for judge_idx, judge_model in enumerate(judges):
            if orders == "both":
                flips = [flip for flip in (False, True)
                         if (job["pair_uid"], judge_model, flip) not in done]
            else:
                if judged_in_any_orientation(done, job["pair_uid"], judge_model):
                    flips = []
                else:
                    flips = [rng.random() < 0.5]
            for flip in flips:
                if limit is not None and len(pending) >= limit:
                    return pending
                pending.append((job, out_a, out_b, judge_idx, judge_model, flip))
    return pending


def run_parallel_judging(
    *,
    jobs: list[dict[str, Any]],
    generations: dict[str, dict[str, Any]],
    done: set[tuple[str, str]],
    judges: list[str],
    vote_paths: list[Path],
    run_votes: Path,
    mirror_global: bool,
    dry_run: bool,
    temperature: float,
    max_tokens: int,
    seed: int,
    workers: int,
    limit: int | None,
    orders: str = "single",
) -> None:
    pending = pending_judge_requests(
        jobs=jobs,
        generations=generations,
        done=done,
        judges=judges,
        seed=seed,
        limit=limit,
        orders=orders,
    )
    written = 0
    stop_after_backoff_error = False
    next_index = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}

        def submit_next() -> None:
            nonlocal next_index
            job, out_a, out_b, judge_idx, judge_model, flip = pending[next_index]
            future = executor.submit(
                run_judge_request,
                job=job,
                out_a=out_a,
                out_b=out_b,
                judge_idx=judge_idx,
                judge_model=judge_model,
                flip=flip,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                dry_run=dry_run,
            )
            futures[future] = (job["pair_uid"], judge_model)
            next_index += 1

        while next_index < len(pending) and len(futures) < workers:
            submit_next()

        while futures:
            done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done_futures:
                futures.pop(future)
                row, backoff_error = future.result()
                for path in vote_paths:
                    append_jsonl(path, row)
                written += 1
                stop_after_backoff_error = stop_after_backoff_error or backoff_error

            while not stop_after_backoff_error and next_index < len(pending) and len(futures) < workers:
                submit_next()

    print(f"wrote {written} new judge votes to {run_votes}")
    if mirror_global and run_votes != JUDGE_VOTES:
        print(f"mirrored judge votes to {JUDGE_VOTES}")
    if stop_after_backoff_error:
        print("stopping after transient API/network error; re-run to resume missing judge votes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Maximum new judge votes to run.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--retry-unresolved", action="store_true", help="Retry existing unresolved judge votes.")
    parser.add_argument("--dry-run", action="store_true", help="Write deterministic placeholder votes without API calls.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing generation_jobs.jsonl.")
    parser.add_argument("--manifest", type=Path, default=None, help="Specific generation_jobs.jsonl manifest to judge.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel API calls. Default: 1.")
    parser.add_argument(
        "--orders",
        choices=["single", "both"],
        default="single",
        help=(
            "single: one vote per pair x judge in a random presentation order. "
            "both: two votes per pair x judge, one in each order — position "
            "bias cancels within every pair; judge-level orientation flips are "
            "collapsed to ties at aggregation."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    jobs = read_jobs_from_args(args.run_dir, args.manifest)
    run_generations = run_log_path(jobs, "run_generations_path", GENERATIONS)
    run_votes = run_log_path(jobs, "run_judge_votes_path", JUDGE_VOTES)
    mirror_global = args.run_dir is None and args.manifest is None
    vote_paths = unique_paths([run_votes, JUDGE_VOTES]) if mirror_global else [run_votes]
    generations = generation_map([run_generations])
    pair_hashes = current_pair_hashes(jobs, generations)
    done = existing_vote_keys(pair_hashes, retry_unresolved=args.retry_unresolved, paths=[run_votes])
    judges = ["dry-run-judge-a", "dry-run-judge-b", "dry-run-judge-c"] if args.dry_run else judge_model_ids()
    client = None if args.dry_run else OpenRouterClient()
    if args.workers > 1:
        run_parallel_judging(
            jobs=jobs,
            generations=generations,
            done=done,
            judges=judges,
            vote_paths=vote_paths,
            run_votes=run_votes,
            mirror_global=mirror_global,
            dry_run=args.dry_run,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            workers=args.workers,
            limit=args.limit,
            orders=args.orders,
        )
        return
    written = 0
    pending = pending_judge_requests(
        jobs=jobs,
        generations=generations,
        done=done,
        judges=judges,
        seed=args.seed,
        limit=args.limit,
        orders=args.orders,
    )
    for job, out_a, out_b, judge_idx, judge_model, flip in pending:
        row, backoff_error = run_judge_request(
            job=job,
            out_a=out_a,
            out_b=out_b,
            judge_idx=judge_idx,
            judge_model=judge_model,
            flip=flip,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        for path in vote_paths:
            append_jsonl(path, row)
        written += 1
        if not args.dry_run and backoff_error:
            print(f"wrote {written} new judge votes to {run_votes}")
            if mirror_global and run_votes != JUDGE_VOTES:
                print(f"mirrored judge votes to {JUDGE_VOTES}")
            print("stopping after transient API/network error; re-run to resume missing judge votes")
            return
    print(f"wrote {written} new judge votes to {run_votes}")
    if mirror_global and run_votes != JUDGE_VOTES:
        print(f"mirrored judge votes to {JUDGE_VOTES}")


if __name__ == "__main__":
    main()
