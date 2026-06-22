#!/usr/bin/env python3
"""Blind task-rubric feature coding for matched output pairs.

This is the reusable qualitative feature workflow. It supports two kinds of
matched pairs:

1. Existing within-condition contrasts from ``final_text_analysis_pair_deltas``,
   e.g. direct_high vs direct_low or utility_high vs utility_low.
2. Arbitrary arm-vs-baseline matches from ``final_text_analysis_by_output``,
   e.g. moral_low vs direct_low or utility_high vs r0.

The sampled A/B prompt hides condition labels from the rubric coder. The output
JSONL stores the exact request prompt and sanitized API response for audit.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from utility_behavior_gap.io_utils import append_jsonl, read_jsonl, write_csv_rows, write_jsonl
from utility_behavior_gap.output_exclusions import (
    filter_semantic_excluded_output_catalog,
    filter_semantic_excluded_pair_rows,
    filter_valid_output_catalog,
    filter_valid_pair_rows,
)
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
    task_context,
)


PAIR_DELTAS = ANALYSIS / "final_text_analysis_pair_deltas.csv"
BY_OUTPUT = ANALYSIS / "final_text_analysis_by_output.csv"
OUT_ROOT = ANALYSIS / "task_rubric_feature_coding"
PAIR_KEYS = ["actor", "task", "item_label", "repeat"]
TASKS = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]


def complete_string_mask(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        values = df[col]
        mask &= values.notna()
        text = values.astype(str).str.strip()
        mask &= text.ne("")
        mask &= ~text.str.lower().isin({"nan", "none", "null"})
    return mask


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "none"


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def output_text_map(run_dirs: list[str]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for run_dir in sorted(set(run_dirs)):
        run_path = Path(run_dir)
        path = run_path if run_path.is_file() else run_path / "generations.jsonl"
        for row in read_jsonl_if_exists(path):
            if row.get("success") is False:
                continue
            text = str(row.get("output_text") or "").strip()
            if text:
                outputs[str(row["output_id"])] = text
    return outputs


def comparison_name(args: argparse.Namespace) -> str:
    if args.comparison_name:
        return args.comparison_name
    if args.contrast:
        return args.contrast
    return f"{args.left_condition}_vs_{args.right_condition}"


def default_out_dir(args: argparse.Namespace) -> Path:
    source = "pair-deltas" if args.contrast else "arm-match"
    n = "all" if args.sample_size_per_task == 0 else str(args.sample_size_per_task)
    return OUT_ROOT / (
        f"{slug(comparison_name(args))}__source-{source}__"
        f"n-per-task-{n}__seed-{args.seed}__coder-{slug(args.coder_model)}"
    )


def parse_tasks(values: list[str]) -> list[str]:
    if not values or values == ["all"]:
        return TASKS
    unknown = sorted(set(values) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown task(s): {', '.join(unknown)}")
    return values


def source_from_pair_deltas(args: argparse.Namespace, tasks: list[str]) -> pd.DataFrame:
    pair_deltas_path = args.pair_deltas or PAIR_DELTAS
    df = pd.read_csv(pair_deltas_path, low_memory=False)
    df = df[df["contrast"].eq(args.contrast)].copy()
    df = df[df["task"].isin(tasks)].copy()
    if args.actors:
        df = df[df["actor"].isin(args.actors)].copy()
    if df.empty:
        raise SystemExit(f"No pair-delta rows found for contrast={args.contrast!r}.")
    df, valid_output_filter_report = filter_valid_pair_rows(df)
    if df.empty:
        raise SystemExit(
            f"No pair-delta rows remain for contrast={args.contrast!r} after valid-output filtering."
        )
    df, semantic_exclusion_filter_report = filter_semantic_excluded_pair_rows(df)
    if df.empty:
        raise SystemExit(
            f"No pair-delta rows remain for contrast={args.contrast!r} after semantic exclusion filtering."
        )
    before_complete = len(df)
    df = df[
        complete_string_mask(df, ["high_output_id", "low_output_id", "run_dir", "pair_uid"])
    ].copy()
    incomplete_dropped = before_complete - len(df)
    if df.empty:
        raise SystemExit(f"No complete pair-delta rows found for contrast={args.contrast!r}.")

    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        rows.append(
            {
                "coding_pair_uid": f"{comparison_name(args)}:{row['pair_uid']}",
                "source_mode": "pair_deltas",
                "comparison_name": comparison_name(args),
                "actor": row["actor"],
                "task": row["task"],
                "task_label": row.get("task_label", row["task"]),
                "item_label": row["item_label"],
                "item_index": row.get("item_index", ""),
                "repeat": row.get("repeat", ""),
                "source_pair_uid": row["pair_uid"],
                "left_condition": row["high_condition"],
                "right_condition": row["low_condition"],
                "left_raw_condition": row.get("high_raw_condition", row["high_condition"]),
                "right_raw_condition": row.get("low_raw_condition", row["low_condition"]),
                "left_output_id": row["high_output_id"],
                "right_output_id": row["low_output_id"],
                "left_run_dir": row["run_dir"],
                "right_run_dir": row["run_dir"],
                "panel_winner_condition": row.get("panel_winner_condition", ""),
                "effect_score_left_minus_right": row.get("effect_score_high_minus_low", ""),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["incomplete_source_rows_dropped"] = int(incomplete_dropped)
    out.attrs["valid_output_filter_report"] = valid_output_filter_report
    out.attrs["semantic_exclusion_filter_report"] = semantic_exclusion_filter_report
    return out


def dedupe_condition(df: pd.DataFrame, condition: str) -> tuple[pd.DataFrame, int, int]:
    sub = df[df["condition"].eq(condition)].copy()
    if sub.empty:
        raise SystemExit(f"No rows found for condition={condition!r}.")
    before_complete = len(sub)
    sub = sub[complete_string_mask(sub, ["output_id", "run_dir"])].copy()
    incomplete_dropped = before_complete - len(sub)
    if sub.empty:
        raise SystemExit(f"No complete rows found for condition={condition!r}.")
    before = len(sub)
    run_id = sub.get("run_id", pd.Series("", index=sub.index)).fillna("").astype(str)
    run_dir = sub.get("run_dir", pd.Series("", index=sub.index)).fillna("").astype(str)
    sub["_repeat_block_rank"] = (
        run_id.str.contains("__repeat-block", regex=False)
        | run_dir.str.contains("__repeat-block", regex=False)
    ).astype(int)
    sub["_sort_run_id"] = run_id
    sub["_sort_output_id"] = sub["output_id"].astype(str)
    sub = sub.sort_values(PAIR_KEYS + ["_repeat_block_rank", "_sort_run_id", "_sort_output_id"])
    sub = sub.drop_duplicates(PAIR_KEYS, keep="first").copy()
    return (
        sub.drop(columns=["_repeat_block_rank", "_sort_run_id", "_sort_output_id"]),
        before - len(sub),
        incomplete_dropped,
    )


def source_from_arm_match(args: argparse.Namespace, tasks: list[str]) -> pd.DataFrame:
    if not args.left_condition or not args.right_condition:
        raise SystemExit("Arm-match mode requires --left-condition and --right-condition.")
    df = pd.read_csv(BY_OUTPUT, low_memory=False)
    df = df[df["task"].isin(tasks)].copy()
    if args.actors:
        df = df[df["actor"].isin(args.actors)].copy()
    df, valid_output_filter_report = filter_valid_output_catalog(df)
    df, semantic_exclusion_filter_report = filter_semantic_excluded_output_catalog(df)

    left, left_dropped, left_incomplete_dropped = dedupe_condition(df, args.left_condition)
    right, right_dropped, right_incomplete_dropped = dedupe_condition(df, args.right_condition)
    merged = left.merge(
        right,
        on=PAIR_KEYS,
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if merged.empty:
        raise SystemExit(
            f"No exact actor/task/item/repeat matches for {args.left_condition} vs {args.right_condition}."
        )

    rows: list[dict[str, Any]] = []
    for row in merged.to_dict(orient="records"):
        rows.append(
            {
                "coding_pair_uid": (
                    f"{comparison_name(args)}:{row['actor']}:{row['task']}:"
                    f"i{slug(str(row['item_label']))}:r{row['repeat']}:"
                    f"{row['output_id_left']}:{row['output_id_right']}"
                ),
                "source_mode": "arm_match",
                "comparison_name": comparison_name(args),
                "actor": row["actor"],
                "task": row["task"],
                "task_label": row.get("task_label_left") or row.get("task_label_right") or row["task"],
                "item_label": row["item_label"],
                "item_index": row.get("item_index_left", ""),
                "repeat": row.get("repeat", ""),
                "source_pair_uid": "",
                "left_condition": args.left_condition,
                "right_condition": args.right_condition,
                "left_raw_condition": row.get("raw_condition_left", args.left_condition),
                "right_raw_condition": row.get("raw_condition_right", args.right_condition),
                "left_output_id": row["output_id_left"],
                "right_output_id": row["output_id_right"],
                "left_run_dir": row["run_dir_left"],
                "right_run_dir": row["run_dir_right"],
                "panel_winner_condition": "",
                "effect_score_left_minus_right": "",
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["left_duplicates_dropped"] = left_dropped
    out.attrs["right_duplicates_dropped"] = right_dropped
    out.attrs["left_incomplete_rows_dropped"] = left_incomplete_dropped
    out.attrs["right_incomplete_rows_dropped"] = right_incomplete_dropped
    out.attrs["valid_output_filter_report"] = valid_output_filter_report
    out.attrs["semantic_exclusion_filter_report"] = semantic_exclusion_filter_report
    return out


def actor_balanced_sample(df: pd.DataFrame, *, sample_size_per_task: int, seed: int) -> pd.DataFrame:
    if sample_size_per_task == 0:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = random.Random(seed)
    sampled_frames: list[pd.DataFrame] = []
    for task, task_df in df.groupby("task", sort=True):
        target = min(sample_size_per_task, len(task_df))
        actors = sorted(task_df["actor"].dropna().unique().tolist())
        if not actors:
            continue
        base = target // len(actors)
        remainder = target % len(actors)
        actor_targets = {actor: base for actor in actors}
        for actor in rng.sample(actors, remainder):
            actor_targets[actor] += 1

        chosen_parts: list[pd.DataFrame] = []
        chosen_indices: set[int] = set()
        for actor in actors:
            actor_df = task_df[task_df["actor"].eq(actor)]
            n = min(actor_targets[actor], len(actor_df))
            if n:
                part = actor_df.sample(n=n, random_state=rng.randrange(1_000_000_000))
                chosen_parts.append(part)
                chosen_indices.update(part.index.tolist())
        chosen = pd.concat(chosen_parts, ignore_index=False) if chosen_parts else task_df.iloc[0:0]

        if len(chosen) < target:
            remaining = task_df.loc[~task_df.index.isin(chosen_indices)]
            fill = remaining.sample(n=target - len(chosen), random_state=rng.randrange(1_000_000_000))
            chosen = pd.concat([chosen, fill], ignore_index=False)
        sampled_frames.append(chosen)

    if not sampled_frames:
        raise SystemExit("No rows sampled.")
    return pd.concat(sampled_frames, ignore_index=True).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)


def add_outputs_and_ab(sample: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    run_dirs = sample["left_run_dir"].astype(str).tolist() + sample["right_run_dir"].astype(str).tolist()
    text_by_id = output_text_map(run_dirs)

    rng = random.Random(seed)
    flip_flags = [True] * (len(sample) // 2) + [False] * (len(sample) - len(sample) // 2)
    rng.shuffle(flip_flags)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for flip, row in zip(flip_flags, sample.to_dict(orient="records"), strict=True):
        left_text = text_by_id.get(str(row["left_output_id"]), "")
        right_text = text_by_id.get(str(row["right_output_id"]), "")
        if not left_text:
            missing.append(str(row["left_output_id"]))
        if not right_text:
            missing.append(str(row["right_output_id"]))
        if flip:
            a_side, b_side = "left", "right"
            output_a, output_b = left_text, right_text
            output_a_id, output_b_id = row["left_output_id"], row["right_output_id"]
        else:
            a_side, b_side = "right", "left"
            output_a, output_b = right_text, left_text
            output_a_id, output_b_id = row["right_output_id"], row["left_output_id"]
        rows.append(
            {
                **row,
                "a_side": a_side,
                "b_side": b_side,
                "output_a_id": output_a_id,
                "output_b_id": output_b_id,
                "task_context": task_context(str(row["task"]), str(row["item_label"])),
                "output_a": output_a,
                "output_b": output_b,
            }
        )
    if missing:
        raise SystemExit(f"Missing output text for {len(missing)} sampled outputs; first: {missing[0]}")
    return pd.DataFrame(rows)


def attach_outputs_from_manifest(sample: pd.DataFrame) -> pd.DataFrame:
    """Reload output text while preserving the manifest's stored A/B assignment."""

    run_dirs = sample["left_run_dir"].astype(str).tolist() + sample["right_run_dir"].astype(str).tolist()
    text_by_id = output_text_map(run_dirs)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in sample.to_dict(orient="records"):
        left_text = text_by_id.get(str(row["left_output_id"]), "")
        right_text = text_by_id.get(str(row["right_output_id"]), "")
        if not left_text:
            missing.append(str(row["left_output_id"]))
        if not right_text:
            missing.append(str(row["right_output_id"]))
        if row["a_side"] == "left":
            output_a, output_b = left_text, right_text
        elif row["a_side"] == "right":
            output_a, output_b = right_text, left_text
        else:
            raise SystemExit(f"Invalid a_side in manifest: {row['a_side']!r}")
        rows.append(
            {
                **row,
                "task_context": task_context(str(row["task"]), str(row["item_label"])),
                "output_a": output_a,
                "output_b": output_b,
            }
        )
    if missing:
        raise SystemExit(f"Missing output text for {len(missing)} manifest outputs; first: {missing[0]}")
    return pd.DataFrame(rows)


def code_to_left_right(winner: str, row: pd.Series) -> str:
    if winner == "A":
        return str(row["a_side"])
    if winner == "B":
        return str(row["b_side"])
    if winner in {"tie", "not_applicable"}:
        return winner
    return "unresolved"


def request_snapshot(*, model: str, prompt: str, temperature: float | None, max_tokens: int) -> dict[str, Any]:
    return {
        "script": "utility_behavior_gap.scripts.run_task_rubric_feature_coding",
        "argv": sys.argv,
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def code_sample_row(
    *,
    row_dict: dict[str, Any],
    coder_model: str,
    temperature: float | None,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    row = pd.Series(row_dict)
    dimensions = TASK_DIMENSIONS[str(row["task"])]
    prompt = rubric_prompt(row, dimensions)
    request = request_snapshot(
        model=coder_model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    client = OpenRouterClient(timeout_s=timeout_s, max_retries=3)
    started = time.time()
    response: dict[str, Any] = {}
    raw_text = ""
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
        response = getattr(exc, "response", response)
        codes = {dimension: {"winner": "unresolved", "reason": str(exc)} for dimension in dimensions}
        success = False
        error = str(exc)
    latency_s = time.time() - started

    mapped_codes = {
        dimension: {
            **codes[dimension],
            "winner_side": code_to_left_right(codes[dimension]["winner"], row),
        }
        for dimension in dimensions
    }
    return {
        "coding_pair_uid": row["coding_pair_uid"],
        "comparison_name": row["comparison_name"],
        "source_mode": row["source_mode"],
        "actor": row["actor"],
        "task": row["task"],
        "left_condition": row["left_condition"],
        "right_condition": row["right_condition"],
        "left_output_id": row["left_output_id"],
        "right_output_id": row["right_output_id"],
        "a_side": row["a_side"],
        "b_side": row["b_side"],
        "output_a_id": row["output_a_id"],
        "output_b_id": row["output_b_id"],
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


def existing_keys(path: Path, coder_model: str) -> set[str]:
    return {
        str(row.get("coding_pair_uid"))
        for row in read_jsonl_if_exists(path)
        if row.get("coder_model") == coder_model and row.get("success") is not False
    }


def build_or_load_sample(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    manifest_path = out_dir / "rubric_feature_sample.csv"
    prompts_path = out_dir / "rubric_feature_prompts.jsonl"
    if manifest_path.exists() and prompts_path.exists() and not args.rebuild:
        manifest = pd.read_csv(manifest_path, low_memory=False)
        manifest, _valid_output_filter_report = filter_valid_pair_rows(manifest)
        manifest, _semantic_exclusion_filter_report = filter_semantic_excluded_pair_rows(
            manifest
        )
        return attach_outputs_from_manifest(manifest)

    tasks = parse_tasks(args.tasks)
    if args.contrast:
        source = source_from_pair_deltas(args, tasks)
        duplicate_note = {
            "incomplete_source_rows_dropped": int(
                source.attrs.get("incomplete_source_rows_dropped", 0)
            ),
            "left_duplicates_dropped": 0,
            "right_duplicates_dropped": 0,
            "left_incomplete_rows_dropped": 0,
            "right_incomplete_rows_dropped": 0,
            "valid_output_filter_report": source.attrs.get("valid_output_filter_report", {}),
            "semantic_exclusion_filter_report": source.attrs.get(
                "semantic_exclusion_filter_report", {}
            ),
        }
    else:
        source = source_from_arm_match(args, tasks)
        duplicate_note = {
            "incomplete_source_rows_dropped": 0,
            "left_duplicates_dropped": int(source.attrs.get("left_duplicates_dropped", 0)),
            "right_duplicates_dropped": int(source.attrs.get("right_duplicates_dropped", 0)),
            "left_incomplete_rows_dropped": int(
                source.attrs.get("left_incomplete_rows_dropped", 0)
            ),
            "right_incomplete_rows_dropped": int(
                source.attrs.get("right_incomplete_rows_dropped", 0)
            ),
            "valid_output_filter_report": source.attrs.get("valid_output_filter_report", {}),
            "semantic_exclusion_filter_report": source.attrs.get(
                "semantic_exclusion_filter_report", {}
            ),
        }
    sample = actor_balanced_sample(source, sample_size_per_task=args.sample_size_per_task, seed=args.seed)
    sample = add_outputs_and_ab(sample, seed=args.seed + 101)
    sample.insert(0, "sample_index", range(len(sample)))

    manifest_cols = [
        "sample_index",
        "coding_pair_uid",
        "comparison_name",
        "source_mode",
        "actor",
        "task",
        "task_label",
        "item_label",
        "item_index",
        "repeat",
        "source_pair_uid",
        "left_condition",
        "right_condition",
        "left_raw_condition",
        "right_raw_condition",
        "left_output_id",
        "right_output_id",
        "left_run_dir",
        "right_run_dir",
        "a_side",
        "b_side",
        "output_a_id",
        "output_b_id",
        "panel_winner_condition",
        "effect_score_left_minus_right",
    ]
    write_csv_rows(manifest_path, sample[manifest_cols].to_dict(orient="records"), fields=manifest_cols)

    prompt_rows: list[dict[str, Any]] = []
    for row in sample.to_dict(orient="records"):
        dimensions = TASK_DIMENSIONS[str(row["task"])]
        prompt_rows.append(
            {
                "sample_index": row["sample_index"],
                "coding_pair_uid": row["coding_pair_uid"],
                "comparison_name": row["comparison_name"],
                "actor": row["actor"],
                "task": row["task"],
                "left_condition": row["left_condition"],
                "right_condition": row["right_condition"],
                "a_side": row["a_side"],
                "b_side": row["b_side"],
                "output_a_id": row["output_a_id"],
                "output_b_id": row["output_b_id"],
                "prompt": rubric_prompt(pd.Series(row), dimensions),
            }
        )
    write_jsonl(prompts_path, prompt_rows)
    (out_dir / "first_rubric_prompt.txt").write_text(prompt_rows[0]["prompt"], encoding="utf-8")
    (out_dir / "sample_metadata.json").write_text(
        json.dumps(
            {
                "comparison_name": comparison_name(args),
                "contrast": args.contrast,
                "pair_deltas": str(args.pair_deltas or PAIR_DELTAS) if args.contrast else "",
                "left_condition": args.left_condition,
                "right_condition": args.right_condition,
                "tasks": tasks,
                "actors": args.actors,
                "sample_size_per_task": args.sample_size_per_task,
                "seed": args.seed,
                "coder_model": args.coder_model,
                **duplicate_note,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--contrast", help="Use existing high-vs-low pair-delta rows for this contrast.")
    source.add_argument("--left-condition", help="Left arm for arbitrary arm-vs-arm matching.")
    parser.add_argument("--pair-deltas", type=Path, help="Pair-delta CSV to use with --contrast.")
    parser.add_argument("--right-condition", help="Right arm for arbitrary arm-vs-arm matching.")
    parser.add_argument("--comparison-name", help="Human-readable comparison id for output paths.")
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--actors", nargs="*", default=[])
    parser.add_argument(
        "--sample-size-per-task",
        type=int,
        default=120,
        help="Random actor-balanced sample per task; use 0 for all matched pairs.",
    )
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--coder-model", default=DEFAULT_CODER_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild sample manifest even if it exists.")
    parser.add_argument("--run", action="store_true", help="Make paid OpenRouter calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.left_condition and not args.right_condition:
        raise SystemExit("--right-condition is required with --left-condition.")
    if args.right_condition and not args.left_condition:
        raise SystemExit("--left-condition is required with --right-condition.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")

    out_dir = args.out_dir or default_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = build_or_load_sample(args, out_dir)
    codes_path = out_dir / "rubric_feature_codes.jsonl"

    print(f"comparison: {comparison_name(args)}")
    print(f"sampled pairs: {len(sample)}")
    print(f"by task: {sample.groupby('task').size().to_dict()}")
    print(f"manifest: {out_dir / 'rubric_feature_sample.csv'}")
    print(f"prompts: {out_dir / 'rubric_feature_prompts.jsonl'}")
    print(f"codes: {codes_path}")
    print(f"first prompt: {out_dir / 'first_rubric_prompt.txt'}")
    if not args.run:
        print("dry run only; pass --run to make paid OpenRouter calls")
        return

    done = existing_keys(codes_path, args.coder_model)
    pending = [
        row.to_dict()
        for _, row in sample.iterrows()
        if str(row["coding_pair_uid"]) not in done
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        print(f"wrote 0 new rubric codes to {codes_path}")
        return

    written = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                code_sample_row,
                row_dict=row,
                coder_model=args.coder_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_s=120.0,
            )
            for row in pending
        ]
        for future in as_completed(futures):
            result = future.result()
            append_jsonl(codes_path, result)
            written += 1
            print(
                f"coded {written}/{len(pending)}: {result['coding_pair_uid']} success={result['success']}",
                flush=True,
            )
    print(f"wrote {written} new rubric codes to {codes_path}")


if __name__ == "__main__":
    main()
