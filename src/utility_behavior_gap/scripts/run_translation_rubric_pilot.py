#!/usr/bin/env python3
"""Code translation-pair quality dimensions with a blind rubric.

This is a pilot mechanism-analysis script. It samples completed translation
pairs from the panel-analysis dataset, hides condition labels from the coder,
and asks one LLM call to code several top-down translation-quality dimensions.

Live API calls are disabled unless --run is passed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
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


PANEL_DATA = ANALYSIS / "panel_feature_lasso_model_data.csv"
DEFAULT_CODER_MODEL = "google/gemini-3.1-flash-lite-preview"
DEFAULT_MAX_TOKENS = 1200
DEFAULT_TEMPERATURE = 0.0
DIMENSIONS = [
    "fluency_idiomaticity",
    "terminology_precision",
    "named_entity_fidelity",
    "numeric_factual_fidelity",
    "additions_omissions",
    "structural_clarity",
    "register_match",
]
VALID_WINNERS = {"A", "B", "tie", "not_applicable"}


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


def build_sample(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(PANEL_DATA, low_memory=False)
    df = df[df["task"].eq("translation")].copy()
    if args.contrast and args.contrast != "all":
        df = df[df["contrast"].eq(args.contrast)].copy()
    if args.actors:
        actor_set = set(args.actors)
        df = df[df["actor"].isin(actor_set)].copy()
    df = df[df["panel_signature"].isin(["HHH", "HHL", "HLL", "LLL"])].copy()
    if df.empty:
        raise SystemExit("No matching translation pairs found.")
    if args.stratify_panel_signature:
        sampled = stratified_sample(df, args.sample_size, args.seed)
    elif args.sample_size > 0 and args.sample_size < len(df):
        sampled = df.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)
    else:
        sampled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    outputs = output_map(sampled["run_dir"].astype(str).tolist())
    records = []
    rng = random.Random(args.seed)
    for sample_index, row in sampled.reset_index(drop=True).iterrows():
        high_output = outputs.get(str(row["high_output_id"]))
        low_output = outputs.get(str(row["low_output_id"]))
        if high_output is None or low_output is None:
            continue
        flip = rng.random() < 0.5
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
                "source_chinese": row.get("item_label", ""),
                "translation_a": output_a["output_text"],
                "translation_b": output_b["output_text"],
            }
        )
    if not records:
        raise SystemExit("No sampled pairs had both outputs available.")
    return pd.DataFrame(records)


def rubric_prompt(row: pd.Series) -> str:
    return f"""You are coding translation-quality dimensions for a research audit.

You will see a Chinese source passage and two English translations, labeled A and B.
You must compare A and B on each dimension separately.

Important constraints:
- Do not infer anything about how A or B was produced.
- Do not try to guess which one a previous judge preferred.
- If a dimension is not relevant to this passage, use "not_applicable".
- Use "tie" when neither translation is meaningfully better on that dimension.
- Return exactly one JSON object and no surrounding prose.

Dimensions:
1. fluency_idiomaticity: Which translation reads more naturally and idiomatically in English?
2. terminology_precision: Which translation uses more domain-appropriate terminology?
3. named_entity_fidelity: Which translation better handles names, organizations, places, titles, acronyms, and transliterations?
4. numeric_factual_fidelity: Which translation better preserves numbers, dates, currencies, percentages, units, and factual details?
5. additions_omissions: Which translation has fewer unsupported additions and fewer omissions of source meaning?
6. structural_clarity: Which translation more clearly handles clauses, lists, appositives, parentheticals, quotation, and sentence breaks while preserving source relations?
7. register_match: Which translation better matches the source genre and tone?

For each dimension, use this exact schema:
{{"winner": "A" | "B" | "tie" | "not_applicable", "reason": "brief reason"}}

Return JSON with exactly these top-level keys:
{json.dumps(DIMENSIONS, ensure_ascii=False)}

CHINESE SOURCE:
\"\"\"
{row["source_chinese"]}
\"\"\"

TRANSLATION A:
\"\"\"
{row["translation_a"]}
\"\"\"

TRANSLATION B:
\"\"\"
{row["translation_b"]}
\"\"\"
"""


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group(0))


def normalize_codes(parsed: dict[str, Any]) -> dict[str, dict[str, str]]:
    codes: dict[str, dict[str, str]] = {}
    for dimension in DIMENSIONS:
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
        f"translation_rubric_pilot__contrast-{slug(args.contrast or 'all')}__"
        f"n-{args.sample_size}__seed-{args.seed}__coder-{slug(args.coder_model)}"
    )


def request_snapshot(*, model: str, prompt: str, temperature: float | None, max_tokens: int) -> dict[str, Any]:
    return {
        "script": "utility_behavior_gap.scripts.run_translation_rubric_pilot",
        "argv": sys.argv,
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast", default="direct_instruction")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--actors", nargs="*", default=[])
    parser.add_argument("--coder-model", default=DEFAULT_CODER_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--limit", type=int, help="Optional live-call limit for resuming/testing.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--no-stratify-panel-signature", dest="stratify_panel_signature", action="store_false")
    parser.add_argument("--run", action="store_true", help="Actually make OpenRouter calls.")
    args = parser.parse_args()

    out_dir = output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "translation_rubric_sample.csv"
    codes_path = out_dir / "translation_rubric_codes.jsonl"

    sample = build_sample(args)
    write_csv_rows(manifest_path, sample.to_dict(orient="records"))

    first_prompt = rubric_prompt(sample.iloc[0])
    (out_dir / "first_rubric_prompt.txt").write_text(first_prompt, encoding="utf-8")

    print(f"sampled pairs: {len(sample)}")
    print(f"manifest: {manifest_path}")
    print(f"codes: {codes_path}")
    print(f"first prompt: {out_dir / 'first_rubric_prompt.txt'}")
    if not args.run:
        print("dry run only; pass --run to make paid OpenRouter calls")
        return

    done = existing_keys(codes_path, args.coder_model)
    client = OpenRouterClient(timeout_s=120.0, max_retries=3)
    written = 0
    for _, row in sample.iterrows():
        pair_uid = str(row["pair_uid"])
        if pair_uid in done:
            continue
        if args.limit is not None and written >= args.limit:
            break
        prompt = rubric_prompt(row)
        request = request_snapshot(
            model=args.coder_model,
            prompt=prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        started = time.time()
        try:
            response = client.chat_completion(
                model=args.coder_model,
                messages=request["messages"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            raw_text = response_text(response)
            parsed = extract_json_object(raw_text)
            codes = normalize_codes(parsed)
            success = all(codes[dimension]["winner"] != "unresolved" for dimension in DIMENSIONS)
            error = ""
        except (MalformedOpenRouterResponse, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            response = getattr(exc, "response", {})
            raw_text = ""
            codes = {dimension: {"winner": "unresolved", "reason": str(exc)} for dimension in DIMENSIONS}
            success = False
            error = str(exc)
        latency_s = time.time() - started

        mapped_codes = {
            dimension: {
                **codes[dimension],
                "winner_side": code_to_high_low(codes[dimension]["winner"], row),
            }
            for dimension in DIMENSIONS
        }
        append_jsonl(
            codes_path,
            {
                "pair_uid": pair_uid,
                "sample_index": int(row["sample_index"]),
                "actor": row["actor"],
                "contrast": row["contrast"],
                "panel_signature": row["panel_signature"],
                "n_high": int(row["n_high"]),
                "n_low": int(row["n_low"]),
                "panel_score_high_minus_low": float(row["panel_score_high_minus_low"]),
                "a_side": row["a_side"],
                "b_side": row["b_side"],
                "output_a_id": row["output_a_id"],
                "output_b_id": row["output_b_id"],
                "coder_model": args.coder_model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "success": success,
                "error": error,
                "latency_s": round(latency_s, 3),
                "raw_text": raw_text,
                "codes": mapped_codes,
                "usage": response.get("usage", {}) if isinstance(response, dict) else {},
                "raw_response": response_without_message_content(response) if isinstance(response, dict) else {},
                "request": request,
            },
        )
        written += 1
        print(f"coded {written}: {pair_uid} success={success}", flush=True)
    print(f"wrote {written} new rubric codes to {codes_path}")


if __name__ == "__main__":
    main()
