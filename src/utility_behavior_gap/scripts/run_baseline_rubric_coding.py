#!/usr/bin/env python3
"""Code task-specific rubric dimensions for arm-vs-baseline pairs.

This answers the absolute-baseline version of the mechanism question:
does a given arm elicit more judge-valued task features than a baseline
such as framed neutral (`direct_low`) or bare task (`r0`)?

Live API calls are disabled unless --run is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from utility_behavior_gap.io_utils import append_jsonl, read_jsonl, write_csv_rows
from utility_behavior_gap.openrouter import (
    MalformedOpenRouterResponse,
    OpenRouterClient,
    response_text,
    response_without_message_content,
)
from utility_behavior_gap.paths import ANALYSIS
from utility_behavior_gap.scripts.run_task_rubric_pilot import (
    DEFAULT_CODER_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    TASK_DIMENSIONS,
    extract_json_object,
    normalize_codes,
    rubric_prompt,
    slug,
    task_context,
)


OUTPUT_FEATURES = ANALYSIS / "final_text_analysis_by_output.csv"
DEFAULT_BASELINE_ARMS = {
    "direct_low": [
        "direct_high",
        "amount_low",
        "amount_high",
        "moral_low",
        "moral_high",
        "utility_low",
        "utility_high",
    ],
    "r0": [
        "direct_low",
        "direct_high",
        "amount_low",
        "amount_high",
        "moral_low",
        "moral_high",
        "utility_low",
        "utility_high",
    ],
}


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def output_text_map(run_dirs: list[str]) -> dict[str, str]:
    by_id: dict[str, str] = {}
    for raw_run_dir in sorted(set(run_dirs)):
        if not raw_run_dir or raw_run_dir == "nan":
            continue
        path = Path(raw_run_dir)
        if path.is_file():
            rows = read_jsonl_if_exists(path)
        else:
            rows = read_jsonl_if_exists(path / "generations.jsonl")
        for row in rows:
            output_id = str(row.get("output_id") or "")
            text = str(row.get("output_text") or "").strip()
            if output_id and text:
                by_id[output_id] = text
    return by_id


def stable_pair_uid(row: pd.Series, arm: str, baseline: str) -> str:
    key = {
        "actor": row["actor"],
        "task": row["task"],
        "item_label": row["item_label"],
        "repeat": str(row["repeat"]),
        "arm": arm,
        "baseline": baseline,
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"baseline:{row['task']}:{arm}:vs:{baseline}:{row['actor']}:h{digest}"


def clean_outputs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["missing_output"].fillna(1).astype(int).eq(0)]
    out = out[out["empty_output"].fillna(1).astype(int).eq(0)]
    out = out[out["output_id"].notna()]
    out = out[out["output_id"].astype(str).str.len().gt(0)]
    return out


def default_arms_for_baseline(baseline: str, conditions: list[str]) -> list[str]:
    if baseline in DEFAULT_BASELINE_ARMS:
        return DEFAULT_BASELINE_ARMS[baseline]
    return [condition for condition in conditions if condition != baseline and condition != "r0"]


def build_pairs(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_FEATURES, low_memory=False)
    df = clean_outputs(df)
    df = df[df["task"].eq(args.task)].copy()
    if df.empty:
        raise SystemExit(f"No outputs found for task {args.task!r}.")

    all_conditions = sorted(df["condition"].dropna().astype(str).unique())
    arms = args.arms or default_arms_for_baseline(args.baseline, all_conditions)
    if args.baseline in arms:
        raise SystemExit("Baseline cannot also be an arm.")

    keys = ["actor", "task", "item_label", "repeat"]
    baseline = df[df["condition"].eq(args.baseline)].copy()
    if baseline.empty:
        raise SystemExit(f"No baseline outputs found for {args.baseline!r}.")
    baseline = baseline.drop_duplicates(keys)

    records: list[dict[str, Any]] = []
    for arm in arms:
        arm_df = df[df["condition"].eq(arm)].copy()
        if arm_df.empty:
            print(f"warning: no outputs for arm {arm!r}; skipping", flush=True)
            continue
        arm_df = arm_df.drop_duplicates(keys)
        merged = arm_df.merge(
            baseline,
            on=keys,
            how="inner",
            suffixes=("_arm", "_baseline"),
        )
        for _, row in merged.iterrows():
            records.append(
                {
                    "baseline_pair_uid": stable_pair_uid(row, arm, args.baseline),
                    "actor": row["actor"],
                    "actor_label": row.get("actor_label_arm", ""),
                    "task": row["task"],
                    "task_label": row.get("task_label_arm", ""),
                    "item_label": row["item_label"],
                    "item_index": row.get("item_index_arm", ""),
                    "repeat": row["repeat"],
                    "arm_condition": arm,
                    "baseline_condition": args.baseline,
                    "arm_output_id": row["output_id_arm"],
                    "baseline_output_id": row["output_id_baseline"],
                    "arm_run_dir": row.get("run_dir_arm", ""),
                    "baseline_run_dir": row.get("run_dir_baseline", ""),
                }
            )
    if not records:
        raise SystemExit("No matched arm-vs-baseline pairs found.")

    pairs = pd.DataFrame(records)
    if args.sample_size > 0 and args.sample_size < len(pairs):
        if args.sample_design == "arm_balanced":
            rng = random.Random(args.seed)
            groups = [(arm, group.copy()) for arm, group in pairs.groupby("arm_condition", sort=True)]
            per_arm = max(1, args.sample_size // len(groups))
            sampled = []
            for _arm, group in groups:
                n = min(len(group), per_arm)
                sampled.append(group.sample(n=n, random_state=rng.randrange(1_000_000_000)))
            pairs = pd.concat(sampled, ignore_index=True).sample(frac=1.0, random_state=args.seed + 1)
            if len(pairs) > args.sample_size:
                pairs = pairs.sample(n=args.sample_size, random_state=args.seed + 2)
            pairs = pairs.reset_index(drop=True)
        else:
            pairs = pairs.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)
    else:
        pairs = pairs.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    texts = output_text_map(
        pairs["arm_run_dir"].astype(str).tolist() + pairs["baseline_run_dir"].astype(str).tolist()
    )
    rng = random.Random(args.seed)
    flip_flags = [True] * (len(pairs) // 2) + [False] * (len(pairs) - len(pairs) // 2)
    rng.shuffle(flip_flags)

    rows = []
    for i, row in pairs.iterrows():
        arm_text = texts.get(str(row["arm_output_id"]), "")
        baseline_text = texts.get(str(row["baseline_output_id"]), "")
        if not arm_text or not baseline_text:
            continue
        flip = flip_flags[i]
        if flip:
            output_a, output_b = baseline_text, arm_text
            output_a_id, output_b_id = row["baseline_output_id"], row["arm_output_id"]
            a_role, b_role = "baseline", "arm"
        else:
            output_a, output_b = arm_text, baseline_text
            output_a_id, output_b_id = row["arm_output_id"], row["baseline_output_id"]
            a_role, b_role = "arm", "baseline"
        rows.append(
            {
                **row.to_dict(),
                "sample_index": len(rows),
                "a_role": a_role,
                "b_role": b_role,
                "output_a_id": output_a_id,
                "output_b_id": output_b_id,
                "task_context": task_context(str(row["task"]), str(row["item_label"])),
                "output_a": output_a,
                "output_b": output_b,
            }
        )
    if not rows:
        raise SystemExit("No matched pairs had both output texts available.")
    return pd.DataFrame(rows)


def code_to_role(winner: str, row: pd.Series) -> str:
    if winner == "A":
        return str(row["a_role"])
    if winner == "B":
        return str(row["b_role"])
    if winner in {"tie", "not_applicable"}:
        return winner
    return "unresolved"


def existing_keys(path: Path, coder_model: str) -> set[str]:
    keys = set()
    for row in read_jsonl_if_exists(path):
        if row.get("coder_model") == coder_model and row.get("success") is not False:
            keys.add(str(row.get("baseline_pair_uid")))
    return keys


def output_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return args.out_dir
    arms_slug = "all-default" if not args.arms else "+".join(slug(arm) for arm in args.arms)
    return ANALYSIS / (
        f"baseline_rubric__task-{slug(args.task)}__baseline-{slug(args.baseline)}__"
        f"arms-{arms_slug}__sample-{slug(args.sample_design)}__n-{args.sample_size}__"
        f"seed-{args.seed}__coder-{slug(args.coder_model)}"
    )


def request_snapshot(*, model: str, prompt: str, temperature: float | None, max_tokens: int) -> dict[str, Any]:
    return {
        "script": "utility_behavior_gap.scripts.run_baseline_rubric_coding",
        "argv": sys.argv,
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def code_baseline_row(
    *,
    row_dict: dict[str, Any],
    dimensions: list[str],
    coder_model: str,
    temperature: float | None,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    row = pd.Series(row_dict)
    pair_uid = str(row["baseline_pair_uid"])
    prompt = rubric_prompt(row, dimensions)
    request = request_snapshot(
        model=coder_model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    client = OpenRouterClient(timeout_s=timeout_s, max_retries=3)
    started = time.time()
    try:
        response = client.chat_completion(
            model=coder_model,
            messages=request["messages"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_text = response_text(response)
        parsed = extract_json_object(raw_text)
        codes = normalize_codes(parsed, dimensions)
        success = all(codes[dimension]["winner"] != "unresolved" for dimension in dimensions)
        error = ""
    except (MalformedOpenRouterResponse, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        response = getattr(exc, "response", {})
        raw_text = ""
        codes = {dimension: {"winner": "unresolved", "reason": str(exc)} for dimension in dimensions}
        success = False
        error = str(exc)
    latency_s = time.time() - started
    mapped_codes = {
        dimension: {
            **codes[dimension],
            "winner_role": code_to_role(codes[dimension]["winner"], row),
        }
        for dimension in dimensions
    }
    return {
        "baseline_pair_uid": pair_uid,
        "sample_index": int(row["sample_index"]),
        "actor": row["actor"],
        "task": row["task"],
        "arm_condition": row["arm_condition"],
        "baseline_condition": row["baseline_condition"],
        "item_label": row["item_label"],
        "repeat": row["repeat"],
        "a_role": row["a_role"],
        "b_role": row["b_role"],
        "output_a_id": row["output_a_id"],
        "output_b_id": row["output_b_id"],
        "arm_output_id": row["arm_output_id"],
        "baseline_output_id": row["baseline_output_id"],
        "dimensions": dimensions,
        "coder_model": coder_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "success": success,
        "error": error,
        "latency_s": round(latency_s, 3),
        "raw_text": raw_text,
        "codes": mapped_codes,
        "usage": response.get("usage", {}) if isinstance(response, dict) else {},
        "raw_response": response_without_message_content(response) if isinstance(response, dict) else {},
        "request": request,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_DIMENSIONS))
    parser.add_argument("--baseline", required=True, help="Baseline condition, e.g. direct_low or r0.")
    parser.add_argument("--arms", nargs="*", default=[])
    parser.add_argument("--sample-size", type=int, default=120, help="0 means all matched pairs.")
    parser.add_argument("--sample-design", choices=["random", "arm_balanced"], default="arm_balanced")
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--coder-model", default=DEFAULT_CODER_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--workers", type=int, default=1, help="Parallel rubric API calls. Default: 1.")
    parser.add_argument("--limit", type=int, help="Optional live-call limit for testing/resuming.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--run", action="store_true", help="Actually make OpenRouter calls.")
    args = parser.parse_args()

    dimensions = TASK_DIMENSIONS[args.task]
    out_dir = output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "baseline_rubric_sample.csv"
    codes_path = out_dir / "baseline_rubric_codes.jsonl"

    sample = build_pairs(args)
    write_csv_rows(manifest_path, sample.to_dict(orient="records"))
    first_prompt = rubric_prompt(sample.iloc[0], dimensions)
    (out_dir / "first_rubric_prompt.txt").write_text(first_prompt, encoding="utf-8")

    print(f"task: {args.task}")
    print(f"baseline: {args.baseline}")
    print(f"sampled pairs: {len(sample)}")
    print(f"by arm: {sample['arm_condition'].value_counts().sort_index().to_dict()}")
    print(f"A role counts: {sample['a_role'].value_counts().sort_index().to_dict()}")
    print(f"manifest: {manifest_path}")
    print(f"codes: {codes_path}")
    print(f"first prompt: {out_dir / 'first_rubric_prompt.txt'}")
    if not args.run:
        print("dry run only; pass --run to make paid OpenRouter calls")
        return
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    done = existing_keys(codes_path, args.coder_model)
    pending = [
        row.to_dict()
        for _, row in sample.iterrows()
        if str(row["baseline_pair_uid"]) not in done
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    total = len(pending)
    if total == 0:
        print(f"wrote 0 new baseline rubric codes to {codes_path}")
        return

    written = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                code_baseline_row,
                row_dict=row_dict,
                dimensions=dimensions,
                coder_model=args.coder_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_s=120.0,
            )
            for row_dict in pending
        ]
        for future in as_completed(futures):
            result = future.result()
            append_jsonl(codes_path, result)
            written += 1
            print(
                f"coded {written}/{total}: {result['baseline_pair_uid']} success={result['success']}",
                flush=True,
            )
    print(f"wrote {written} new baseline rubric codes to {codes_path}")


if __name__ == "__main__":
    main()
