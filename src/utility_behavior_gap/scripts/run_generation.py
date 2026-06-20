#!/usr/bin/env python3
"""Run actor generations through OpenRouter."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from utility_behavior_gap.api_errors import is_rate_limit_api_error, is_transient_api_error
from utility_behavior_gap.io_utils import append_jsonl, read_jsonl
from utility_behavior_gap.job_builder import read_generation_jobs
from utility_behavior_gap.openrouter import (
    MalformedOpenRouterResponse,
    OpenRouterClient,
    actor_model_id,
    response_without_message_content,
    response_text,
)
from utility_behavior_gap.paths import OUTPUT_API


GENERATIONS = OUTPUT_API / "generations.jsonl"
GENERATION_FAILURES = OUTPUT_API / "generation_failures.jsonl"
GENERATION_REASONING = {"effort": "none"}
GENERATION_PROVIDER = {"require_parameters": True}
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    rows = []
    if system_prompt:
        rows.append({"role": "system", "content": system_prompt})
    rows.append({"role": "user", "content": user_prompt})
    return rows


def generation_request_snapshot(
    *,
    model: str,
    request: dict[str, str],
    temperature: float | None,
    max_tokens: int,
    reasoning: dict[str, Any],
    provider: dict[str, Any],
) -> dict[str, Any]:
    return {
        "script": "utility_behavior_gap.scripts.run_generation",
        "argv": sys.argv,
        "model": model,
        "messages": messages(request.get("system_prompt", ""), request.get("prompt", "")),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": reasoning,
        "provider": provider,
    }


def generation_reasoning(effort: str, *, exclude: bool) -> dict[str, Any]:
    if effort not in REASONING_EFFORTS:
        raise ValueError(f"unsupported reasoning effort: {effort!r}")
    if effort == "none":
        return dict(GENERATION_REASONING)
    config: dict[str, Any] = {"effort": effort}
    if exclude:
        config["exclude"] = True
    return config


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


def existing_output_ids(paths: list[Path] | None = None, *, allow_reasoning_tokens: bool = False) -> set[str]:
    ids: set[str] = set()
    for path in unique_paths(paths or [GENERATIONS]):
        ids.update(
            row["output_id"]
            for row in read_jsonl_if_exists(path)
            if is_valid_generation_row(row, allow_reasoning_tokens=allow_reasoning_tokens)
        )
    return ids


def output_requests(job: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "output_id": f"{job['pair_uid']}::a",
            "pair_uid": job["pair_uid"],
            "condition": job["condition_a"],
            "system_prompt": job.get("system_prompt_a", ""),
            "prompt": job["prompt_a"],
        },
        {
            "output_id": f"{job['pair_uid']}::b",
            "pair_uid": job["pair_uid"],
            "condition": job["condition_b"],
            "system_prompt": job.get("system_prompt_b", ""),
            "prompt": job["prompt_b"],
        },
    ]


def finish_reason(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    reason = choices[0].get("finish_reason", "")
    return "" if reason is None else str(reason)


def reasoning_tokens(response: dict[str, Any]) -> int:
    details = response.get("usage", {}).get("completion_tokens_details", {})
    return int(details.get("reasoning_tokens") or 0)


def validate_generation_response(
    *,
    output_id: str,
    text: str,
    response: dict[str, Any],
    reason: str,
    allow_reasoning_tokens: bool,
) -> None:
    used_reasoning = reasoning_tokens(response)
    if used_reasoning and not allow_reasoning_tokens:
        raise RuntimeError(
            f"{output_id} used {used_reasoning} reasoning tokens. "
            "Actor generation must run with reasoning disabled."
        )
    if reason and reason != "stop":
        raise RuntimeError(f"{output_id} ended with finish_reason={reason!r}; not writing a partial output.")
    if not text.strip():
        raise RuntimeError(f"{output_id} returned empty output_text; not writing an unusable generation.")


def append_generation_failure(
    *,
    job: dict[str, Any],
    request: dict[str, str],
    model: str,
    text: str,
    raw_response: dict[str, Any],
    reason: str,
    error: str,
    latency_s: float,
    temperature: float | None,
    max_tokens: int,
    reasoning: dict[str, Any],
    provider: dict[str, Any],
    paths: list[Path] | None = None,
) -> None:
    row = generation_failure_row(
        job=job,
        request=request,
        model=model,
        text=text,
        raw_response=raw_response,
        reason=reason,
        error=error,
        latency_s=latency_s,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning=reasoning,
        provider=provider,
    )
    for path in unique_paths(paths or [GENERATION_FAILURES]):
        append_jsonl(path, row)


def generation_failure_row(
    *,
    job: dict[str, Any],
    request: dict[str, str],
    model: str,
    text: str,
    raw_response: dict[str, Any],
    reason: str,
    error: str,
    latency_s: float,
    temperature: float | None,
    max_tokens: int,
    reasoning: dict[str, Any],
    provider: dict[str, Any],
) -> dict[str, Any]:
    return {
        "output_id": request["output_id"],
        "pair_uid": request["pair_uid"],
        "actor": job["actor"],
        "model": model,
        "condition": request["condition"],
        "success": False,
        "error": error,
        "latency_s": round(latency_s, 3),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": reasoning,
        "provider": provider,
        "finish_reason": reason,
        "output_text": text,
        "usage": raw_response.get("usage", {}),
        "raw_response": response_without_message_content(raw_response),
        "request": generation_request_snapshot(
            model=model,
            request=request,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
            provider=provider,
        ),
        "job": job,
        "run_id": job.get("run_id", ""),
    }


def is_valid_generation_row(row: dict[str, Any], *, allow_reasoning_tokens: bool = False) -> bool:
    if reasoning_tokens(row) and not allow_reasoning_tokens:
        return False
    if row.get("finish_reason") not in {"", "stop", "dry_run"}:
        return False
    return bool(str(row.get("output_text", "")).strip())


def generation_success_row(
    *,
    job: dict[str, Any],
    request: dict[str, str],
    model: str,
    text: str,
    raw_response: dict[str, Any],
    reason: str,
    latency_s: float,
    temperature: float | None,
    max_tokens: int,
    request_snapshot: dict[str, Any],
    dry_run: bool,
    reasoning: dict[str, Any],
    provider: dict[str, Any],
) -> dict[str, Any]:
    return {
        "output_id": request["output_id"],
        "pair_uid": request["pair_uid"],
        "actor": job["actor"],
        "model": model,
        "condition": request["condition"],
        "output_text": text,
        "success": True,
        "latency_s": round(latency_s, 3),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": reasoning if not dry_run else {},
        "provider": provider if not dry_run else {},
        "finish_reason": reason,
        "usage": raw_response.get("usage", {}),
        "raw_response": response_without_message_content(raw_response),
        "request": request_snapshot,
        "job": job,
        "run_id": job.get("run_id", ""),
    }


def run_generation_request(
    *,
    job: dict[str, Any],
    request: dict[str, str],
    model: str,
    dry_run: bool,
    temperature: float | None,
    max_tokens: int,
    reasoning: dict[str, Any],
    provider: dict[str, Any],
    allow_reasoning_tokens: bool,
) -> tuple[bool, bool, dict[str, Any]]:
    started = time.time()
    request_snapshot = generation_request_snapshot(
        model=model,
        request=request,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning=reasoning,
        provider=provider,
    )
    if dry_run:
        text = f"[dry run] {job['comparison']} {request['condition']} output for {job['actor']}"
        raw_response: dict[str, Any] = {}
        reason = "dry_run"
    else:
        client = OpenRouterClient()
        try:
            raw_response = client.chat_completion(
                model=model,
                messages=request_snapshot["messages"],
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
                provider=provider,
            )
            text = response_text(raw_response)
            reason = finish_reason(raw_response)
        except MalformedOpenRouterResponse as exc:
            return (
                False,
                False,
                generation_failure_row(
                    job=job,
                    request=request,
                    model=model,
                    text="",
                    raw_response=exc.response,
                    reason="malformed_response",
                    error=str(exc),
                    latency_s=time.time() - started,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning=reasoning,
                    provider=provider,
                ),
            )
        except RuntimeError as exc:
            if not is_transient_api_error(exc):
                raise
            return (
                False,
                is_rate_limit_api_error(exc),
                generation_failure_row(
                    job=job,
                    request=request,
                    model=model,
                    text="",
                    raw_response={},
                    reason="api_error",
                    error=str(exc),
                    latency_s=time.time() - started,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning=reasoning,
                    provider=provider,
                ),
            )
        try:
            validate_generation_response(
                output_id=request["output_id"],
                text=text,
                response=raw_response,
                reason=reason,
                allow_reasoning_tokens=allow_reasoning_tokens,
            )
        except RuntimeError as exc:
            return (
                False,
                False,
                generation_failure_row(
                    job=job,
                    request=request,
                    model=model,
                    text=text,
                    raw_response=raw_response,
                    reason=reason,
                    error=str(exc),
                    latency_s=time.time() - started,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning=reasoning,
                    provider=provider,
                ),
            )
    return (
        True,
        False,
        generation_success_row(
            job=job,
            request=request,
            model=model,
            text=text,
            raw_response=raw_response,
            reason=reason,
            latency_s=time.time() - started,
            temperature=temperature,
            max_tokens=max_tokens,
            request_snapshot=request_snapshot,
            dry_run=dry_run,
            reasoning=reasoning,
            provider=provider,
        ),
    )


def pending_generation_requests(
    jobs: list[dict[str, Any]],
    done: set[str],
    *,
    dry_run: bool,
    limit: int | None,
    conditions: set[str] | None = None,
) -> list[tuple[dict[str, Any], dict[str, str], str]]:
    pending: list[tuple[dict[str, Any], dict[str, str], str]] = []
    for job in jobs:
        model = "dry-run-model" if dry_run else actor_model_id(job["actor"])
        for request in output_requests(job):
            if conditions is not None and request["condition"] not in conditions:
                continue
            if request["output_id"] in done:
                continue
            pending.append((job, request, model))
            if limit is not None and len(pending) >= limit:
                return pending
    return pending


def run_parallel_generation(
    *,
    jobs: list[dict[str, Any]],
    done: set[str],
    generation_paths: list[Path],
    failure_paths: list[Path],
    run_generations: Path,
    run_failures: Path,
    dry_run: bool,
    temperature: float | None,
    max_tokens: int,
    reasoning: dict[str, Any],
    provider: dict[str, Any],
    allow_reasoning_tokens: bool,
    workers: int,
    limit: int | None,
    conditions: set[str] | None,
) -> None:
    pending = pending_generation_requests(
        jobs, done, dry_run=dry_run, limit=limit, conditions=conditions
    )
    written = 0
    failed = 0
    stop_after_rate_limit = False
    next_index = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}

        def submit_next() -> None:
            nonlocal next_index
            job, request, model = pending[next_index]
            future = executor.submit(
                run_generation_request,
                job=job,
                request=request,
                model=model,
                dry_run=dry_run,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
                provider=provider,
                allow_reasoning_tokens=allow_reasoning_tokens,
            )
            futures[future] = request["output_id"]
            next_index += 1

        while next_index < len(pending) and len(futures) < workers:
            submit_next()

        while futures:
            done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done_futures:
                futures.pop(future)
                success, rate_limited, row = future.result()
                if success:
                    for path in generation_paths:
                        append_jsonl(path, row)
                    written += 1
                else:
                    for path in failure_paths:
                        append_jsonl(path, row)
                    failed += 1
                    stop_after_rate_limit = stop_after_rate_limit or rate_limited

            while not stop_after_rate_limit and next_index < len(pending) and len(futures) < workers:
                submit_next()

    print(f"wrote {written} new generations to {run_generations}")
    if run_generations != GENERATIONS:
        print(f"mirrored generations to {GENERATIONS}")
    if failed:
        print(f"logged {failed} invalid generations to {run_failures}")
        if run_failures != GENERATION_FAILURES:
            print(f"mirrored invalid generations to {GENERATION_FAILURES}")
    if stop_after_rate_limit:
        print("stopping after rate-limit error; re-run to resume missing generations")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Maximum new generations to run.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Omitted by default to use the provider default.",
    )
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--reasoning-effort",
        choices=sorted(REASONING_EFFORTS),
        default="none",
        help=(
            "Actor reasoning effort. Default is none, preserving the canonical "
            "reasoning-disabled generation protocol."
        ),
    )
    parser.add_argument(
        "--reasoning-exclude",
        action="store_true",
        help="When reasoning is enabled, request that reasoning content be excluded from the response.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel API calls. Default: 1.")
    parser.add_argument(
        "--conditions",
        default="",
        help="Optional comma-separated condition names to generate from paired jobs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write deterministic placeholder outputs without API calls.")
    parser.add_argument(
        "--jobs",
        default=None,
        help=(
            "Path to a generation-jobs JSONL (e.g. an immutable run manifest). "
            "Defaults to the shared outputs/api/generation_jobs.jsonl. Use this "
            "to run several generation processes concurrently without racing "
            "on the shared jobs file."
        ),
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    reasoning = generation_reasoning(args.reasoning_effort, exclude=args.reasoning_exclude)
    allow_reasoning_tokens = args.reasoning_effort != "none"
    provider = dict(GENERATION_PROVIDER)
    condition_filter = {part.strip() for part in args.conditions.split(",") if part.strip()} or None

    jobs = read_jsonl(Path(args.jobs)) if args.jobs else read_generation_jobs()
    run_generations = run_log_path(jobs, "run_generations_path", GENERATIONS)
    run_failures = run_log_path(jobs, "run_generation_failures_path", GENERATION_FAILURES)
    generation_paths = unique_paths([run_generations, GENERATIONS])
    failure_paths = unique_paths([run_failures, GENERATION_FAILURES])
    done = existing_output_ids([run_generations], allow_reasoning_tokens=allow_reasoning_tokens)
    client = None if args.dry_run else OpenRouterClient()
    if args.workers > 1:
        run_parallel_generation(
            jobs=jobs,
            done=done,
            generation_paths=generation_paths,
            failure_paths=failure_paths,
            run_generations=run_generations,
            run_failures=run_failures,
            dry_run=args.dry_run,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning=reasoning,
            provider=provider,
            allow_reasoning_tokens=allow_reasoning_tokens,
            workers=args.workers,
            limit=args.limit,
            conditions=condition_filter,
        )
        return
    attempted = 0
    written = 0
    failed = 0
    for job in jobs:
        model = "dry-run-model" if args.dry_run else actor_model_id(job["actor"])
        for request in output_requests(job):
            if condition_filter is not None and request["condition"] not in condition_filter:
                continue
            if request["output_id"] in done:
                continue
            if args.limit is not None and attempted >= args.limit:
                print(f"wrote {written} new generations to {run_generations}")
                if run_generations != GENERATIONS:
                    print(f"mirrored generations to {GENERATIONS}")
                if failed:
                    print(f"logged {failed} invalid generations to {run_failures}")
                    if run_failures != GENERATION_FAILURES:
                        print(f"mirrored invalid generations to {GENERATION_FAILURES}")
                return
            attempted += 1
            started = time.time()
            request_snapshot = generation_request_snapshot(
                model=model,
                request=request,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                reasoning=reasoning,
                provider=provider,
            )
            if args.dry_run:
                text = f"[dry run] {job['comparison']} {request['condition']} output for {job['actor']}"
                raw_response: dict[str, Any] = {}
                reason = "dry_run"
            else:
                assert client is not None
                try:
                    raw_response = client.chat_completion(
                        model=model,
                        messages=request_snapshot["messages"],
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        reasoning=reasoning,
                        provider=provider,
                    )
                    text = response_text(raw_response)
                    reason = finish_reason(raw_response)
                except MalformedOpenRouterResponse as exc:
                    append_generation_failure(
                        job=job,
                        request=request,
                        model=model,
                        text="",
                        raw_response=exc.response,
                        reason="malformed_response",
                        error=str(exc),
                        latency_s=time.time() - started,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        reasoning=reasoning,
                        provider=provider,
                        paths=failure_paths,
                    )
                    failed += 1
                    continue
                except RuntimeError as exc:
                    if not is_transient_api_error(exc):
                        raise
                    append_generation_failure(
                        job=job,
                        request=request,
                        model=model,
                        text="",
                        raw_response={},
                        reason="api_error",
                        error=str(exc),
                        latency_s=time.time() - started,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        reasoning=reasoning,
                        provider=provider,
                        paths=failure_paths,
                    )
                    failed += 1
                    if is_rate_limit_api_error(exc):
                        print(f"wrote {written} new generations to {run_generations}")
                        print(f"mirrored generations to {GENERATIONS}")
                        print(f"logged {failed} invalid generations to {run_failures}")
                        print("stopping after rate-limit error; re-run to resume missing generations")
                        return
                    continue
                try:
                    validate_generation_response(
                        output_id=request["output_id"],
                        text=text,
                        response=raw_response,
                        reason=reason,
                        allow_reasoning_tokens=allow_reasoning_tokens,
                    )
                except RuntimeError as exc:
                    append_generation_failure(
                        job=job,
                        request=request,
                        model=model,
                        text=text,
                        raw_response=raw_response,
                        reason=reason,
                        error=str(exc),
                        latency_s=time.time() - started,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        reasoning=reasoning,
                        provider=provider,
                        paths=failure_paths,
                    )
                    failed += 1
                    continue
            row = {
                "output_id": request["output_id"],
                "pair_uid": request["pair_uid"],
                "actor": job["actor"],
                "model": model,
                "condition": request["condition"],
                "output_text": text,
                "success": True,
                "latency_s": round(time.time() - started, 3),
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "reasoning": reasoning if not args.dry_run else {},
                "provider": provider if not args.dry_run else {},
                "finish_reason": reason,
                "usage": raw_response.get("usage", {}),
                "raw_response": response_without_message_content(raw_response),
                "request": request_snapshot,
                "job": job,
                "run_id": job.get("run_id", ""),
            }
            for path in generation_paths:
                append_jsonl(path, row)
            written += 1
    print(f"wrote {written} new generations to {run_generations}")
    if run_generations != GENERATIONS:
        print(f"mirrored generations to {GENERATIONS}")
    if failed:
        print(f"logged {failed} invalid generations to {run_failures}")
        if run_failures != GENERATION_FAILURES:
            print(f"mirrored invalid generations to {GENERATION_FAILURES}")


if __name__ == "__main__":
    main()
