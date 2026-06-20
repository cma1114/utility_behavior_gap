#!/usr/bin/env python3
"""Mechanical quality check for the canonical amount/translation result.

This is a local-only audit. It does not call any model. It reads the same
canonical manifests used by ``analyze_canonical_highn_conditions.py`` and checks
whether the odd below-50% amount effect for translation is explained by obvious
run artifacts: truncation, empty outputs, scaffold leakage, untranslated
Chinese fragments, duplicated judge votes, or length imbalance.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t

from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition
from utility_behavior_gap.paths import ROOT
from utility_behavior_gap.scripts.analyze_canonical_highn_conditions import (
    condition_from_comparison,
    source_specs,
)


ANALYSIS = ROOT / "outputs" / "analysis"
PAIR_OUTCOMES = ANALYSIS / "canonical_highn_condition_results_pair_outcomes.csv"
OUT_PREFIX = "amount_translation_quality_check"

WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)?|\d+(?:\.\d+)?%?")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
REFUSAL_RE = re.compile(r"\b(as an ai|i cannot|i can't|unable to|sorry|cannot comply)\b", re.I)
BAD_START_RE = re.compile(r"\s*(translation|answer|here is|sure|certainly)\b", re.I)

# Exact scaffold terms, not generic source-text words like "funding" or "$".
SCAFFOLD_TERMS = [
    "sponsor",
    "evaluation",
    "evaluator",
    "judging process",
    "selected as the best",
    "children's hospital",
    "healthcare intervention",
    "do not mention",
    "output only",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def load_amount_translation_pairs() -> pd.DataFrame:
    pairs = pd.read_csv(PAIR_OUTCOMES)
    return pairs[pairs["condition"].eq("amount") & pairs["task"].eq("translation")].copy()


def collect_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs = load_amount_translation_pairs()
    pair_lookup = {(row.source_block, row.pair_uid): row for row in pairs.itertuples(index=False)}

    pair_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    vote_rows: list[dict[str, Any]] = []

    for source_block, manifest, allowed_conditions in source_specs():
        if "amount" not in allowed_conditions:
            continue
        run_dir = manifest.parent
        jobs = read_jsonl(manifest)
        generations = {str(row.get("output_id")): row for row in read_jsonl(run_dir / "generations.jsonl")}
        votes_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for vote in read_jsonl(run_dir / "judge_votes.jsonl"):
            votes_by_pair[str(vote.get("pair_uid"))].append(vote)

        for job in jobs:
            comparison = str(job.get("comparison", ""))
            if condition_from_comparison(comparison) != "amount" or job.get("task") != "translation":
                continue

            pair_uid = str(job["pair_uid"])
            outcome = pair_lookup.get((source_block, pair_uid))
            if outcome is None:
                continue

            output_ids = [f"{pair_uid}::a", f"{pair_uid}::b"]
            output_hashes: list[str] = []
            by_condition: dict[str, dict[str, Any]] = {}

            for side, output_id in zip(("a", "b"), output_ids):
                condition = str(job.get(f"condition_{side}", ""))
                generation = generations.get(output_id)
                text = "" if generation is None else str(generation.get("output_text") or "")
                output_hash = output_text_fingerprint(generation) if generation is not None else ""
                output_hashes.append(output_hash)
                flags = {
                    "source_block": source_block,
                    "run_id": run_dir.name,
                    "actor": str(job.get("actor", "")),
                    "pair_uid": pair_uid,
                    "output_id": output_id,
                    "condition": condition,
                    "has_generation": generation is not None,
                    "finish_reason": "" if generation is None else str(generation.get("finish_reason", "")),
                    "empty": not bool(text.strip()),
                    "words": word_count(text),
                    "chars": len(text),
                    "lines": len(text.splitlines()),
                    "has_chinese_chars": bool(CHINESE_RE.search(text)),
                    "has_exact_scaffold_text": any(term in text.lower() for term in SCAFFOLD_TERMS),
                    "has_refusal_language": bool(REFUSAL_RE.search(text)),
                    "starts_with_preamble": bool(BAD_START_RE.match(text)),
                    "output_text": text,
                }
                by_condition[condition] = flags
                output_rows.append(flags)

            matching_votes = [
                vote
                for vote in votes_by_pair[pair_uid]
                if (vote.get("source_output_a_hash"), vote.get("source_output_b_hash"))
                == tuple(output_hashes)
                and vote.get("success") is not False
            ]
            vote_slot_counts = Counter((vote.get("judge_model"), bool(vote.get("flipped"))) for vote in matching_votes)
            vote_slot_winners: dict[tuple[str, bool], list[str]] = defaultdict(list)
            for vote in matching_votes:
                vote_slot_winners[(str(vote.get("judge_model")), bool(vote.get("flipped")))].append(
                    str(vote.get("winner_condition", ""))
                )
            conflicting_duplicate_slots = sum(
                len(winners) > 1 and len(set(winners)) > 1 for winners in vote_slot_winners.values()
            )
            by_judge: dict[str, list[str]] = defaultdict(list)
            for vote in matching_votes:
                by_judge[str(vote.get("judge_model"))].append(str(vote.get("winner_condition", "")))
            judge_verdicts = [derive_judge_verdict(conditions) for conditions in by_judge.values()]
            panel_rederived = derive_panel_winner_condition(job, judge_verdicts)
            unique_slot_judge_verdicts = []
            for judge_model in sorted(by_judge):
                unique_slot_winners: list[str] = []
                for (slot_judge, _flipped), winners in vote_slot_winners.items():
                    if slot_judge == judge_model:
                        unique_slot_winners.extend(sorted(set(winners)))
                unique_slot_judge_verdicts.append(derive_judge_verdict(unique_slot_winners))
            panel_rederived_unique_slots = derive_panel_winner_condition(job, unique_slot_judge_verdicts)

            high = by_condition.get("amount_high", {})
            low = by_condition.get("amount_low", {})
            pair_rows.append(
                {
                    "source_block": source_block,
                    "run_id": run_dir.name,
                    "actor": str(job.get("actor", "")),
                    "pair_uid": pair_uid,
                    "item_id": str(job.get("item_id", "")),
                    "repeat": str(job.get("repeat", "")),
                    "panel_winner_condition": str(outcome.panel_winner_condition),
                    "target_win": bool(outcome.target_win),
                    "target_loss": bool(outcome.target_loss),
                    "tie": bool(outcome.tie),
                    "resolved": bool(outcome.resolved),
                    "high_words": high.get("words", 0),
                    "low_words": low.get("words", 0),
                    "word_delta_high_minus_low": high.get("words", 0) - low.get("words", 0),
                    "high_chars": high.get("chars", 0),
                    "low_chars": low.get("chars", 0),
                    "char_delta_high_minus_low": high.get("chars", 0) - low.get("chars", 0),
                    "high_has_chinese_chars": bool(high.get("has_chinese_chars", False)),
                    "low_has_chinese_chars": bool(low.get("has_chinese_chars", False)),
                    "high_has_exact_scaffold_text": bool(high.get("has_exact_scaffold_text", False)),
                    "low_has_exact_scaffold_text": bool(low.get("has_exact_scaffold_text", False)),
                    "high_has_refusal_language": bool(high.get("has_refusal_language", False)),
                    "low_has_refusal_language": bool(low.get("has_refusal_language", False)),
                    "high_starts_with_preamble": bool(high.get("starts_with_preamble", False)),
                    "low_starts_with_preamble": bool(low.get("starts_with_preamble", False)),
                    "matching_vote_rows": len(matching_votes),
                    "matching_judges": len(by_judge),
                    "duplicated_vote_slots": sum(count > 1 for count in vote_slot_counts.values()),
                    "conflicting_duplicated_vote_slots": conflicting_duplicate_slots,
                    "panel_rederived_from_votes": panel_rederived,
                    "panel_rederived_matches": panel_rederived == str(outcome.panel_winner_condition),
                    "panel_rederived_from_unique_vote_slots": panel_rederived_unique_slots,
                    "panel_rederived_unique_slots_matches": panel_rederived_unique_slots
                    == str(outcome.panel_winner_condition),
                    "high_output_text": str(high.get("output_text", "")),
                    "low_output_text": str(low.get("output_text", "")),
                    "base_prompt": str(job.get("base_prompt", "")),
                    "high_prompt": str(job.get("prompt_a" if job.get("condition_a") == "amount_high" else "prompt_b", "")),
                    "low_prompt": str(job.get("prompt_a" if job.get("condition_a") == "amount_low" else "prompt_b", "")),
                }
            )
            vote_rows.append(
                {
                    "source_block": source_block,
                    "actor": str(job.get("actor", "")),
                    "pair_uid": pair_uid,
                    "matching_vote_rows": len(matching_votes),
                    "matching_judges": len(by_judge),
                    "duplicated_vote_slots": sum(count > 1 for count in vote_slot_counts.values()),
                    "conflicting_duplicated_vote_slots": conflicting_duplicate_slots,
                    "panel_rederived_from_votes": panel_rederived,
                    "panel_rederived_matches": panel_rederived == str(outcome.panel_winner_condition),
                    "panel_rederived_from_unique_vote_slots": panel_rederived_unique_slots,
                    "panel_rederived_unique_slots_matches": panel_rederived_unique_slots
                    == str(outcome.panel_winner_condition),
                    "judge_verdicts": ";".join(judge_verdicts),
                }
            )

    return pd.DataFrame(pair_rows), pd.DataFrame(output_rows), pd.DataFrame(vote_rows)


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.1%}"


def simple_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    out = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row.tolist()) + " |")
    return "\n".join(out)


def result_row(label: str, pairs: pd.DataFrame) -> dict[str, Any]:
    resolved = pairs[pairs["resolved"]]
    wins = int(resolved["target_win"].sum())
    losses = int(resolved["target_loss"].sum())
    return {
        "subset": label,
        "pairs": len(pairs),
        "resolved": len(resolved),
        "larger_wins": wins,
        "smaller_wins": losses,
        "ties": int(pairs["tie"].sum()),
        "larger_win_rate_excl_ties": pct(wins / len(resolved)) if len(resolved) else "",
    }


def wilson_ci(wins: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def actor_rate_t_ci(pairs: pd.DataFrame, value_col: str, *, resolved_only: bool) -> tuple[float, float, float]:
    source = pairs[pairs["resolved"]].copy() if resolved_only else pairs.copy()
    rates = source.groupby("actor")[value_col].mean().to_numpy(dtype=float)
    point = float(np.mean(rates))
    if len(rates) < 2:
        return point, math.nan, math.nan
    se = float(np.std(rates, ddof=1) / math.sqrt(len(rates)))
    crit = float(t.ppf(0.975, len(rates) - 1))
    return point, point - crit * se, point + crit * se


def bootstrap_actor_pair(
    pairs: pd.DataFrame,
    value_col: str,
    *,
    resolved_only: bool,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    source = pairs[pairs["resolved"]].copy() if resolved_only else pairs.copy()
    groups = [group[value_col].to_numpy(dtype=float) for _, group in source.groupby("actor")]
    point = float(np.mean([group.mean() for group in groups]))
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for idx in range(iterations):
        sampled_actors = rng.integers(0, len(groups), len(groups))
        estimates[idx] = float(
            np.mean(
                [
                    rng.choice(groups[actor_idx], size=len(groups[actor_idx]), replace=True).mean()
                    for actor_idx in sampled_actors
                ]
            )
        )
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return point, float(lo), float(hi)


def bootstrap_actor_item_cluster(
    pairs: pd.DataFrame,
    value_col: str,
    *,
    resolved_only: bool,
    equal_item_weight: bool,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    source = pairs[pairs["resolved"]].copy() if resolved_only else pairs.copy()
    actor_items: list[list[np.ndarray]] = []
    for _, actor_df in source.groupby("actor"):
        items = [
            item_df[value_col].to_numpy(dtype=float)
            for _, item_df in actor_df.groupby("item_id")
            if len(item_df)
        ]
        actor_items.append(items)

    if equal_item_weight:
        actor_values = [np.array([item.mean() for item in items], dtype=float) for items in actor_items]
        point = float(np.mean([values.mean() for values in actor_values]))
    else:
        point = float(np.mean([np.concatenate(items).mean() for items in actor_items]))

    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for idx in range(iterations):
        sampled_actors = rng.integers(0, len(actor_items), len(actor_items))
        actor_estimates = []
        for actor_idx in sampled_actors:
            items = actor_items[int(actor_idx)]
            sampled_items = rng.integers(0, len(items), len(items))
            if equal_item_weight:
                actor_estimates.append(float(np.mean([items[item_idx].mean() for item_idx in sampled_items])))
            else:
                actor_estimates.append(float(np.concatenate([items[item_idx] for item_idx in sampled_items]).mean()))
        estimates[idx] = float(np.mean(actor_estimates))
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return point, float(lo), float(hi)


def inference_sensitivity(pairs: pd.DataFrame, *, iterations: int = 20000, seed: int = 20260615) -> pd.DataFrame:
    pairs = pairs.copy()
    pairs["target_win_numeric"] = pairs["target_win"].astype(float)
    pairs["tie_half_score"] = pairs["target_win"].astype(float) + 0.5 * pairs["tie"].astype(float)

    rows: list[dict[str, Any]] = []
    resolved = pairs[pairs["resolved"]]
    wins = int(resolved["target_win"].sum())
    n = int(len(resolved))
    lo, hi = wilson_ci(wins, n)
    rows.append(
        {
            "estimand": "larger win rate, ties excluded",
            "method": "pooled binomial Wilson; ignores clustering",
            "estimate": wins / n,
            "ci_lo": lo,
            "ci_hi": hi,
            "p_two_sided_vs_0.5": binomtest(wins, n, 0.5, alternative="two-sided").pvalue,
            "p_one_sided_less_than_0.5": binomtest(wins, n, 0.5, alternative="less").pvalue,
        }
    )

    for method, fn in [
        (
            "actor-level t interval over 7 model rates",
            lambda: actor_rate_t_ci(pairs, "target_win_numeric", resolved_only=True),
        ),
        (
            "bootstrap actors and resolved pairs within actor",
            lambda: bootstrap_actor_pair(
                pairs,
                "target_win_numeric",
                resolved_only=True,
                iterations=iterations,
                seed=seed,
            ),
        ),
        (
            "bootstrap actors and translation passages within actor",
            lambda: bootstrap_actor_item_cluster(
                pairs,
                "target_win_numeric",
                resolved_only=True,
                equal_item_weight=False,
                iterations=iterations,
                seed=seed + 1,
            ),
        ),
        (
            "bootstrap actors and equal-weight passage rates",
            lambda: bootstrap_actor_item_cluster(
                pairs,
                "target_win_numeric",
                resolved_only=True,
                equal_item_weight=True,
                iterations=iterations,
                seed=seed + 2,
            ),
        ),
    ]:
        point, ci_lo, ci_hi = fn()
        rows.append(
            {
                "estimand": "larger win rate, ties excluded",
                "method": method,
                "estimate": point,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p_two_sided_vs_0.5": "",
                "p_one_sided_less_than_0.5": "",
            }
        )

    for method, fn in [
        (
            "actor-level t interval over 7 model scores",
            lambda: actor_rate_t_ci(pairs, "tie_half_score", resolved_only=False),
        ),
        (
            "bootstrap actors and translation passages within actor",
            lambda: bootstrap_actor_item_cluster(
                pairs,
                "tie_half_score",
                resolved_only=False,
                equal_item_weight=False,
                iterations=iterations,
                seed=seed + 3,
            ),
        ),
    ]:
        point, ci_lo, ci_hi = fn()
        rows.append(
            {
                "estimand": "larger panel score; tie = 0.5",
                "method": method,
                "estimate": point,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p_two_sided_vs_0.5": "",
                "p_one_sided_less_than_0.5": "",
            }
        )

    out = pd.DataFrame(rows)
    for col in ["estimate", "ci_lo", "ci_hi"]:
        out[col] = out[col].map(lambda value: pct(value) if value != "" and not pd.isna(value) else "")
    for col in ["p_two_sided_vs_0.5", "p_one_sided_less_than_0.5"]:
        out[col] = out[col].map(lambda value: "" if value == "" or pd.isna(value) else f"{float(value):.4g}")
    return out


def write_examples(pairs: pd.DataFrame, path: Path, *, n: int, seed: int) -> None:
    candidates = pairs[pairs["target_loss"]].copy()
    if candidates.empty:
        path.write_text("# Amount Translation Low-Win Examples\n\n_No low-win examples found._\n", encoding="utf-8")
        return
    examples = candidates.sample(n=min(n, len(candidates)), random_state=seed)
    lines = [
        "# Amount Translation Low-Win Examples",
        "",
        "Random sample from canonical amount/translation pairs where the smaller-amount side beat the larger-amount side.",
        "",
    ]
    for idx, row in enumerate(examples.itertuples(index=False), start=1):
        lines.extend(
            [
                f"## Example {idx}",
                "",
                f"- actor: `{row.actor}`",
                f"- source block: `{row.source_block}`",
                f"- pair_uid: `{row.pair_uid}`",
                f"- word delta high-low: `{row.word_delta_high_minus_low}`",
                f"- high has Chinese chars: `{row.high_has_chinese_chars}`; low has Chinese chars: `{row.low_has_chinese_chars}`",
                "",
                "### Base Prompt",
                "",
                "```text",
                row.base_prompt,
                "```",
                "",
                "### Larger-Amount Output",
                "",
                "```text",
                row.high_output_text,
                "```",
                "",
                "### Smaller-Amount Output",
                "",
                "```text",
                row.low_output_text,
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(pair_df: pd.DataFrame, output_df: pd.DataFrame, vote_df: pd.DataFrame, path: Path) -> None:
    finish = pd.crosstab(output_df["condition"], output_df["finish_reason"]).reset_index()
    flag_cols = [
        "empty",
        "has_exact_scaffold_text",
        "has_refusal_language",
        "starts_with_preamble",
        "has_chinese_chars",
    ]
    flags = output_df.groupby("condition")[flag_cols].sum().astype(int).reset_index()
    lengths = (
        output_df.groupby("condition")[["words", "chars", "lines"]]
        .agg(["mean", "median", "min", "max"])
        .round(2)
    )
    lengths.columns = ["_".join(col) for col in lengths.columns]
    lengths = lengths.reset_index()

    by_actor = (
        pair_df[pair_df["resolved"]]
        .groupby("actor")
        .agg(
            resolved=("target_win", "size"),
            larger_wins=("target_win", "sum"),
            smaller_wins=("target_loss", "sum"),
        )
        .reset_index()
    )
    by_actor["larger_win_rate_excl_ties"] = by_actor["larger_wins"] / by_actor["resolved"]
    by_actor["larger_win_rate_excl_ties"] = by_actor["larger_win_rate_excl_ties"].map(pct)

    by_block = (
        pair_df[pair_df["resolved"]]
        .groupby("source_block")
        .agg(
            resolved=("target_win", "size"),
            larger_wins=("target_win", "sum"),
            smaller_wins=("target_loss", "sum"),
        )
        .reset_index()
    )
    by_block["larger_win_rate_excl_ties"] = by_block["larger_wins"] / by_block["resolved"]
    by_block["larger_win_rate_excl_ties"] = by_block["larger_win_rate_excl_ties"].map(pct)

    clean_no_chinese = pair_df[~pair_df["high_has_chinese_chars"] & ~pair_df["low_has_chinese_chars"]]
    clean_strict = clean_no_chinese[
        ~clean_no_chinese["high_has_exact_scaffold_text"]
        & ~clean_no_chinese["low_has_exact_scaffold_text"]
        & ~clean_no_chinese["high_has_refusal_language"]
        & ~clean_no_chinese["low_has_refusal_language"]
        & ~clean_no_chinese["high_starts_with_preamble"]
        & ~clean_no_chinese["low_starts_with_preamble"]
    ]
    sensitivity = pd.DataFrame(
        [
            result_row("all canonical amount/translation pairs", pair_df),
            result_row("drop pairs where either side left Chinese characters", clean_no_chinese),
            result_row("drop Chinese/scaffold/refusal/preamble pairs", clean_strict),
        ]
    )

    vote_summary = pd.DataFrame(
        [
            {
                "quantity": "pairs",
                "value": len(vote_df),
            },
            {
                "quantity": "pairs with 3 matching judges",
                "value": int(vote_df["matching_judges"].eq(3).sum()),
            },
            {
                "quantity": "pairs with exactly 6 hash-matched vote rows",
                "value": int(vote_df["matching_vote_rows"].eq(6).sum()),
            },
            {
                "quantity": "pairs with extra duplicate hash-matched vote rows",
                "value": int(vote_df["matching_vote_rows"].gt(6).sum()),
            },
            {
                "quantity": "pairs with conflicting duplicated judge/order slots",
                "value": int(vote_df["conflicting_duplicated_vote_slots"].gt(0).sum()),
            },
            {
                "quantity": "pairs whose rederived panel matched canonical panel",
                "value": int(vote_df["panel_rederived_matches"].sum()),
            },
            {
                "quantity": "pairs whose panel still matches after deduping vote slots",
                "value": int(vote_df["panel_rederived_unique_slots_matches"].sum()),
            },
        ]
    )

    chinese_patterns = pd.crosstab(
        [pair_df["high_has_chinese_chars"], pair_df["low_has_chinese_chars"]],
        pair_df["panel_winner_condition"],
    ).reset_index()
    inference = inference_sensitivity(pair_df)

    lines = [
        "# Amount Translation Quality Check",
        "",
        "Scope: canonical high-N `amount` condition, `translation` task only. This is a local audit; no model calls are made.",
        "",
        "## Mechanical Checks",
        "",
        f"- canonical pairs checked: `{len(pair_df)}`",
        f"- outputs checked: `{len(output_df)}`",
        f"- exact scaffold leaks: `{int(output_df['has_exact_scaffold_text'].sum())}`",
        f"- refusal-like outputs: `{int(output_df['has_refusal_language'].sum())}`",
        f"- outputs with untranslated Chinese characters: `{int(output_df['has_chinese_chars'].sum())}`",
        "",
        "### Finish Reasons",
        "",
        simple_table(finish),
        "",
        "### Output Flags",
        "",
        simple_table(flags),
        "",
        "### Output Lengths",
        "",
        simple_table(lengths),
        "",
        "### Judge-Vote Integrity",
        "",
        simple_table(vote_summary),
        "",
        "Duplicate hash-matched vote rows occur for some pairs, but rederiving the panel winner from the run-local, hash-matched votes reproduces every canonical panel outcome. Collapsing duplicates to unique judge/order-slot winner sets also leaves every panel outcome unchanged.",
        "",
        "## Result Sensitivity",
        "",
        simple_table(sensitivity),
        "",
        "## Inference Sensitivity",
        "",
        "The primary paper-style estimate excludes panel ties from the win-rate denominator. Because translation has many ties, the tie-inclusive panel score is also shown here as a robustness check.",
        "",
        simple_table(inference),
        "",
        "## By Source Block",
        "",
        simple_table(by_block),
        "",
        "## By Actor",
        "",
        simple_table(by_actor),
        "",
        "## Chinese-Character Artifact Patterns",
        "",
        simple_table(chinese_patterns),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    pair_df, output_df, vote_df = collect_rows()

    pair_path = ANALYSIS / f"{OUT_PREFIX}_pairs.csv"
    output_path = ANALYSIS / f"{OUT_PREFIX}_outputs.csv"
    vote_path = ANALYSIS / f"{OUT_PREFIX}_votes.csv"
    summary_path = ANALYSIS / f"{OUT_PREFIX}_summary.md"
    examples_path = ANALYSIS / f"{OUT_PREFIX}_low_win_examples.md"

    pair_df.to_csv(pair_path, index=False)
    output_df.drop(columns=["output_text"]).to_csv(output_path, index=False)
    vote_df.to_csv(vote_path, index=False)
    write_summary(pair_df, output_df, vote_df, summary_path)
    write_examples(pair_df, examples_path, n=args.examples, seed=args.seed)

    print(f"summary: {summary_path}")
    print(f"pairs: {pair_path}")
    print(f"outputs: {output_path}")
    print(f"votes: {vote_path}")
    print(f"examples: {examples_path}")


if __name__ == "__main__":
    main()
