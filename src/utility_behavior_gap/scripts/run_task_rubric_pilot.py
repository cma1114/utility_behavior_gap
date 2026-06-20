#!/usr/bin/env python3
"""Code task-specific quality dimensions with a blind rubric.

This is a pilot mechanism-analysis script. It samples completed judged pairs
from the panel-analysis dataset, hides condition labels from the coder, and
asks one LLM call to code task-specific quality dimensions for one task.

Live API calls are disabled unless --run is passed.
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

from utility_behavior_gap.constants import TASK_LABEL as TASK_LABELS
from utility_behavior_gap.feature_specs import (
    rubric_dimension_descriptions,
    task_rubric_dimensions,
)
from utility_behavior_gap.io_utils import append_jsonl, read_jsonl, write_csv_rows
from utility_behavior_gap.openrouter import (
    MalformedOpenRouterResponse,
    OpenRouterClient,
    response_text,
    response_without_message_content,
)
from utility_behavior_gap.paths import ANALYSIS


PANEL_DATA = ANALYSIS / "panel_feature_lasso_model_data.csv"
DEFAULT_CODER_MODEL = "google/gemini-2.5-flash"
DEFAULT_MAX_TOKENS = 1400
DEFAULT_TEMPERATURE = 0.0

TASK_DIMENSIONS = task_rubric_dimensions()
DIMENSION_DESCRIPTIONS = rubric_dimension_descriptions()

VALID_WINNERS = {"A", "B", "tie", "not_applicable"}
VALID_JSON_ESCAPE_CHARS = set('"\\/bfnrtu')


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "none"


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def output_map(run_dirs: list[str]) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(set(run_dirs)):
        path = Path(run_dir) / "generations.jsonl"
        for row in read_jsonl_if_exists(path):
            if row.get("success") is False:
                continue
            text = str(row.get("output_text") or "").strip()
            if not text:
                continue
            outputs[str(row["output_id"])] = row
    return outputs


def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = random.Random(seed)
    groups = [(key, group.copy()) for key, group in df.groupby("panel_signature", sort=True)]
    target_by_group: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for key, group in groups:
        raw = sample_size * len(group) / len(df)
        base = min(len(group), int(raw))
        target_by_group[str(key)] = base
        remainders.append((raw - base, str(key)))

    allocated = sum(target_by_group.values())
    for _rem, key in sorted(remainders, reverse=True):
        if allocated >= sample_size:
            break
        group_len = int((df["panel_signature"] == key).sum())
        if target_by_group[key] < group_len:
            target_by_group[key] += 1
            allocated += 1

    rows = []
    for key, group in groups:
        n = target_by_group[str(key)]
        if n:
            rows.append(group.sample(n=n, random_state=rng.randrange(1_000_000_000)))
    return pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)


def panel_direction_balanced_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Sample equal high-panel-win and low-panel-win pairs without replacement."""

    high = df[df["n_high"].astype(int) > df["n_low"].astype(int)].copy()
    low = df[df["n_low"].astype(int) > df["n_high"].astype(int)].copy()
    if high.empty or low.empty:
        raise SystemExit("Cannot create a panel-direction-balanced sample: one direction has no rows.")
    max_per_direction = min(len(high), len(low))
    if sample_size > 0:
        max_per_direction = min(max_per_direction, sample_size // 2)
    high_sample = high.sample(n=max_per_direction, random_state=seed)
    low_sample = low.sample(n=max_per_direction, random_state=seed + 1)
    return pd.concat([high_sample, low_sample], ignore_index=True).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def task_context(task: str, item_label: str) -> str:
    if task == "essay":
        return f"""Task: Write a compelling essay of 250-400 words on this topic:
"{item_label}"

The essay should include a clear thesis, 2-3 supporting arguments, and a concrete example."""
    if task == "grant_proposal_abstract":
        return f"""Task: Write a 500-700 word grant proposal abstract for a project on:
"{item_label}"

The abstract should include problem, approach, evaluation plan, feasibility, risks, and expected impact."""
    if task == "incident_postmortem":
        return f"""Task: Write a 500-750 word blameless incident postmortem for this incident:
"{item_label}"

The postmortem should include impact, timeline, contributing factors, root cause analysis, what went well, what went poorly, and concrete action items."""
    if task == "translation":
        return f"""Task: Translate the following Chinese passage into English.

Chinese passage:
{item_label}"""
    raise ValueError(f"Unsupported task: {task}")


def build_sample(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(PANEL_DATA, low_memory=False)
    df = df[df["task"].eq(args.task)].copy()
    if args.contrast and args.contrast != "all":
        df = df[df["contrast"].eq(args.contrast)].copy()
    if args.actors:
        actor_set = set(args.actors)
        df = df[df["actor"].isin(actor_set)].copy()
    df = df[df["panel_signature"].isin(["HHH", "HHL", "HLL", "LLL"])].copy()
    if df.empty:
        raise SystemExit(f"No matching {args.task} pairs found.")
    if args.sample_design == "panel_direction_balanced":
        sampled = panel_direction_balanced_sample(df, args.sample_size, args.seed)
    elif args.stratify_panel_signature:
        sampled = stratified_sample(df, args.sample_size, args.seed)
    elif args.sample_size > 0 and args.sample_size < len(df):
        sampled = df.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)
    else:
        sampled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    outputs = output_map(sampled["run_dir"].astype(str).tolist())
    records = []
    rng = random.Random(args.seed)
    flip_flags = [True] * (len(sampled) // 2) + [False] * (len(sampled) - (len(sampled) // 2))
    rng.shuffle(flip_flags)
    for sample_index, row in sampled.reset_index(drop=True).iterrows():
        high_output = outputs.get(str(row["high_output_id"]))
        low_output = outputs.get(str(row["low_output_id"]))
        if high_output is None or low_output is None:
            continue
        flip = flip_flags[sample_index]
        if flip:
            output_a, output_b = low_output, high_output
            a_side, b_side = "low", "high"
        else:
            output_a, output_b = high_output, low_output
            a_side, b_side = "high", "low"
        records.append(
            {
                "sample_index": sample_index,
                "pair_uid": row["pair_uid"],
                "actor": row["actor"],
                "actor_label": row.get("actor_label", ""),
                "task": row["task"],
                "task_label": row.get("task_label", TASK_LABELS.get(args.task, args.task)),
                "contrast": row["contrast"],
                "run_dir": row["run_dir"],
                "item_label": row.get("item_label", ""),
                "panel_signature": row["panel_signature"],
                "n_high": int(row["n_high"]),
                "n_low": int(row["n_low"]),
                "panel_score_high_minus_low": (int(row["n_high"]) - int(row["n_low"])) / 3.0,
                "a_side": a_side,
                "b_side": b_side,
                "output_a_id": output_a["output_id"],
                "output_b_id": output_b["output_id"],
                "high_output_id": row["high_output_id"],
                "low_output_id": row["low_output_id"],
                "task_context": task_context(args.task, str(row.get("item_label", ""))),
                "output_a": output_a["output_text"],
                "output_b": output_b["output_text"],
            }
        )
    if not records:
        raise SystemExit("No sampled pairs had both outputs available.")
    return pd.DataFrame(records)


def rubric_prompt(row: pd.Series, dimensions: list[str]) -> str:
    dimension_lines = "\n".join(
        f"{i}. {dimension}: {DIMENSION_DESCRIPTIONS[dimension]}"
        for i, dimension in enumerate(dimensions, start=1)
    )
    task_label = str(row.get("task_label") or TASK_LABELS.get(str(row["task"]), str(row["task"])))
    return f"""You are coding task-specific quality dimensions for a research audit.

You will see the task and two outputs, labeled A and B.
You must compare A and B on each dimension separately.

Important constraints:
- Do not infer anything about how A or B was produced.
- Do not try to guess which one a previous judge preferred.
- If a dimension is not relevant to this pair, use "not_applicable".
- Use "tie" when neither output is meaningfully better on that dimension.
- Return exactly one JSON object and no surrounding prose.

Task type: {task_label}

Dimensions:
{dimension_lines}

For each dimension, use this exact schema:
{{"winner": "A" | "B" | "tie" | "not_applicable", "reason": "brief reason"}}

Return JSON with exactly these top-level keys:
{json.dumps(dimensions, ensure_ascii=False)}

TASK:
\"\"\"
{row["task_context"]}
\"\"\"

OUTPUT A:
\"\"\"
{row["output_a"]}
\"\"\"

OUTPUT B:
\"\"\"
{row["output_b"]}
\"\"\"
"""


def _json_loads_lenient(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some coder responses include literal backslashes in brief reasons,
        # which is invalid JSON unless escaped. Repair only odd-length
        # backslash runs before characters that are not legal JSON escapes.
        return json.loads(repair_invalid_json_escapes(text))


def repair_invalid_json_escapes(text: str) -> str:
    parts: list[str] = []
    i = 0
    while i < len(text):
        if text[i] != "\\":
            parts.append(text[i])
            i += 1
            continue

        j = i
        while j < len(text) and text[j] == "\\":
            j += 1
        run_length = j - i
        next_char = text[j] if j < len(text) else ""
        if run_length % 2 == 1 and next_char not in VALID_JSON_ESCAPE_CHARS:
            run_length += 1
        parts.append("\\" * run_length)
        i = j
    return "".join(parts)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return _json_loads_lenient(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    return _json_loads_lenient(match.group(0))


def normalize_codes(parsed: dict[str, Any], dimensions: list[str]) -> dict[str, dict[str, str]]:
    codes: dict[str, dict[str, str]] = {}
    for dimension in dimensions:
        value = parsed.get(dimension)
        if not isinstance(value, dict):
            codes[dimension] = {"winner": "unresolved", "reason": "missing or non-object dimension"}
            continue
        winner = str(value.get("winner", "")).strip()
        winner = {"a": "A", "b": "B", "na": "not_applicable", "n/a": "not_applicable"}.get(
            winner.lower(),
            winner,
        )
        if winner not in VALID_WINNERS:
            winner = "unresolved"
        reason = str(value.get("reason", "")).strip()
        codes[dimension] = {"winner": winner, "reason": reason}
    return codes


def code_to_high_low(winner: str, row: pd.Series) -> str:
    if winner == "A":
        return str(row["a_side"])
    if winner == "B":
        return str(row["b_side"])
    if winner in {"tie", "not_applicable"}:
        return winner
    return "unresolved"


def existing_keys(path: Path, coder_model: str) -> set[str]:
    keys = set()
    for row in read_jsonl_if_exists(path):
        if row.get("coder_model") == coder_model and row.get("success") is not False:
            keys.add(str(row.get("pair_uid")))
    return keys


def output_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return args.out_dir
    return ANALYSIS / (
        f"task_rubric_pilot__task-{slug(args.task)}__contrast-{slug(args.contrast or 'all')}__"
        f"sample-{slug(args.sample_design)}__n-{args.sample_size}__seed-{args.seed}__coder-{slug(args.coder_model)}"
    )


def request_snapshot(*, model: str, prompt: str, temperature: float | None, max_tokens: int) -> dict[str, Any]:
    return {
        "script": "utility_behavior_gap.scripts.run_task_rubric_pilot",
        "argv": sys.argv,
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def code_sample_row(
    *,
    row_dict: dict[str, Any],
    dimensions: list[str],
    coder_model: str,
    temperature: float | None,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    row = pd.Series(row_dict)
    pair_uid = str(row["pair_uid"])
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
            "winner_side": code_to_high_low(codes[dimension]["winner"], row),
        }
        for dimension in dimensions
    }
    return {
        "pair_uid": pair_uid,
        "sample_index": int(row["sample_index"]),
        "actor": row["actor"],
        "task": row["task"],
        "contrast": row["contrast"],
        "panel_signature": row["panel_signature"],
        "n_high": int(row["n_high"]),
        "n_low": int(row["n_low"]),
        "panel_score_high_minus_low": float(row["panel_score_high_minus_low"]),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_DIMENSIONS))
    parser.add_argument("--contrast", default="direct_instruction")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument(
        "--sample-design",
        choices=["proportional", "panel_direction_balanced"],
        default="proportional",
        help="proportional preserves panel-signature proportions; panel_direction_balanced samples equal high- and low-panel wins.",
    )
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--actors", nargs="*", default=[])
    parser.add_argument("--coder-model", default=DEFAULT_CODER_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--workers", type=int, default=1, help="Parallel rubric API calls. Default: 1.")
    parser.add_argument("--limit", type=int, help="Optional live-call limit for resuming/testing.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--no-stratify-panel-signature", dest="stratify_panel_signature", action="store_false")
    parser.add_argument("--run", action="store_true", help="Actually make OpenRouter calls.")
    args = parser.parse_args()

    dimensions = TASK_DIMENSIONS[args.task]
    out_dir = output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "task_rubric_sample.csv"
    codes_path = out_dir / "task_rubric_codes.jsonl"

    sample = build_sample(args)
    write_csv_rows(manifest_path, sample.to_dict(orient="records"))

    first_prompt = rubric_prompt(sample.iloc[0], dimensions)
    (out_dir / "first_rubric_prompt.txt").write_text(first_prompt, encoding="utf-8")

    print(f"task: {args.task}")
    print(f"sampled pairs: {len(sample)}")
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
        if str(row["pair_uid"]) not in done
    ]
    if args.limit is not None:
        pending = pending[: args.limit]
    total = len(pending)
    if total == 0:
        print(f"wrote 0 new rubric codes to {codes_path}")
        return

    written = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                code_sample_row,
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
                f"coded {written}/{total}: {result['pair_uid']} success={result['success']}",
                flush=True,
            )
    print(f"wrote {written} new rubric codes to {codes_path}")


if __name__ == "__main__":
    main()
