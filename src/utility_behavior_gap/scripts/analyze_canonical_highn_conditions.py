#!/usr/bin/env python3
"""Analyze the canonical high-N amount, moral, and utility conditions.

This script is intentionally local and manifest-driven. It reads:

- base high-low and moral runs from ``fund_wording_rerun_manifests__*.tsv``;
- corrected base amount runs from ``canonical_amount_base_manifests__*.tsv``
  when present; otherwise, legacy amount runs from
  ``canonical_readiness_audit_cells.csv`` rows marked ``legacy_amount_base``;
- extension repeats 5-9 from ``canonical_highn10_manifests__*.tsv``.

The high-N design planned the final dataset as repeats 0-9. The script records
source blocks so prompt or screen caveats stay visible in the outputs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

from utility_behavior_gap.constants import ACTOR_LABEL, ACTORS, TASK_LABEL
from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition
from utility_behavior_gap.paths import ROOT


RUNS = ROOT / "outputs" / "api" / "runs"
ANALYSIS = ROOT / "outputs" / "analysis"

OUT_PREFIX = "canonical_highn_condition_results"
BASE_FUND_GLOB = "fund_wording_rerun_manifests__*.tsv"
AMOUNT_BASE_GLOB = "canonical_amount_base_manifests__*.tsv"
HIGHN_GLOB = "canonical_highn10_manifests__*.tsv"
READINESS_CELLS = ANALYSIS / "canonical_readiness_audit_cells.csv"

TASK_ORDER = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]
CONDITION_ORDER = ["amount", "moral", "utility"]
CONDITION_LABEL = {
    "amount": "Amount",
    "moral": "Moral",
    "utility": "High vs low utility",
}
TARGET_LABEL = {
    "amount": "larger amount",
    "moral": "good cause",
    "utility": "high utility",
}

# Do not use modgrid_moral_refusal_classifications.jsonl here. Those labels are
# keyed by output_id, and output_ids were reused across old prompt variants.
# Applying them to current fund-wording/high-N outputs would silently mix
# refusal labels across different generated texts.
REFUSAL_FILES = [
    ANALYSIS / "fund_wording_moral_refusal_classifications.jsonl",
    ANALYSIS / "canonical_highn_moral_refusal_classifications.jsonl",
]
EXCLUDE_LABELS = {"partial_refusal", "full_refusal", "degenerate"}
VALID_LABELS = {"clean", "partial_refusal", "full_refusal", "degenerate"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def manifest_paths_from_tsv(pattern: str) -> list[Path]:
    paths: list[Path] = []
    for tsv in sorted(RUNS.glob(pattern)):
        with tsv.open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    paths.append(Path(parts[2]))
    return sorted(set(paths))


def amount_base_manifest_paths() -> list[Path]:
    cells = pd.read_csv(READINESS_CELLS)
    sub = cells[
        cells["source_family"].eq("legacy_amount_base")
        & cells["comparison"].astype(str).str.endswith("_amount")
    ].copy()
    return sorted({Path(path) for path in sub["manifest"].tolist()})


def condition_from_comparison(comparison: str) -> str | None:
    if comparison.endswith("_amount"):
        return "amount"
    if comparison.endswith("_moral"):
        return "moral"
    if comparison.endswith("_highlow"):
        return "utility"
    return None


def side_output_ids(job: dict[str, Any]) -> tuple[str, str]:
    pair_uid = str(job["pair_uid"])
    return f"{pair_uid}::a", f"{pair_uid}::b"


def load_classifications() -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    labels: dict[tuple[str, str], str] = {}
    evidence: dict[tuple[str, str], str] = {}
    for path in REFUSAL_FILES:
        for row in read_jsonl(path):
            label = str(row.get("label", ""))
            output_id = str(row.get("output_id", ""))
            output_hash = str(row.get("output_hash", ""))
            if output_id and output_hash and label in VALID_LABELS:
                key = (output_id, output_hash)
                labels[key] = label
                evidence[key] = str(row.get("evidence", ""))
    return labels, evidence


def load_vote_rows(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    votes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote in read_jsonl(run_dir / "judge_votes.jsonl"):
        if vote.get("success"):
            votes[str(vote["pair_uid"])].append(vote)
    return votes


def load_generation_rows(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {str(row["output_id"]): row for row in read_jsonl(run_dir / "generations.jsonl")}


def source_specs() -> list[tuple[str, Path, set[str]]]:
    specs: list[tuple[str, Path, set[str]]] = []
    for path in manifest_paths_from_tsv(BASE_FUND_GLOB):
        specs.append(("base_fund_wording_r0_r4", path, {"moral", "utility"}))
    corrected_amount_paths = manifest_paths_from_tsv(AMOUNT_BASE_GLOB)
    if corrected_amount_paths:
        for path in corrected_amount_paths:
            specs.append(("base_amount_current_target_r0_r4", path, {"amount"}))
    else:
        for path in amount_base_manifest_paths():
            specs.append(("base_amount_legacy_target_r0_r4", path, {"amount"}))
    for path in manifest_paths_from_tsv(HIGHN_GLOB):
        specs.append(("highn_extension_r5_r9", path, {"amount", "moral", "utility"}))
    unique: dict[tuple[str, str], tuple[str, Path, set[str]]] = {}
    for source_block, path, allowed in specs:
        unique[(source_block, str(path))] = (source_block, path, allowed)
    return sorted(unique.values(), key=lambda x: (x[0], str(x[1])))


def is_amount_target_mismatch(job: dict[str, Any], source_block: str) -> bool:
    if condition_from_comparison(str(job.get("comparison", ""))) != "amount":
        return False
    text = str(job.get("prompt_a", "")) + "\n" + str(job.get("prompt_b", ""))
    has_current = "A healthcare intervention at a children's hospital." in text
    return source_block.startswith("base_amount") and not has_current


def build_pair_rows(*, moral_refusal_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels, _ = load_classifications()
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for source_block, manifest, allowed_conditions in source_specs():
        run_dir = manifest.parent
        jobs = read_jsonl(manifest)
        gens = load_generation_rows(run_dir)
        votes_by_pair = load_vote_rows(run_dir)
        for job in jobs:
            comparison = str(job.get("comparison", ""))
            condition = condition_from_comparison(comparison)
            if condition is None or condition not in allowed_conditions:
                continue

            output_a_id, output_b_id = side_output_ids(job)
            out_a = gens.get(output_a_id)
            out_b = gens.get(output_b_id)
            audit_base = {
                "source_block": source_block,
                "run_id": run_dir.name,
                "manifest": str(manifest),
                "condition": condition,
                "comparison": comparison,
                "actor": job.get("actor", ""),
                "task": job.get("task", ""),
                "pair_uid": job.get("pair_uid", ""),
            }
            if out_a is None or out_b is None:
                audit_rows.append({**audit_base, "status": "missing_generation"})
                continue
            if any(
                row.get("finish_reason") != "stop" or not str(row.get("output_text", "")).strip()
                for row in (out_a, out_b)
            ):
                audit_rows.append({**audit_base, "status": "mechanical_exclusion"})
                continue

            hash_a = output_text_fingerprint(out_a)
            hash_b = output_text_fingerprint(out_b)
            label_a = labels.get((output_a_id, hash_a), "")
            label_b = labels.get((output_b_id, hash_b), "")
            moral_refusal_screen = "not_applicable"
            if condition == "moral":
                missing = not label_a or not label_b
                excluded = label_a in EXCLUDE_LABELS or label_b in EXCLUDE_LABELS
                if excluded:
                    moral_refusal_screen = "excluded_refusal_or_degenerate"
                elif missing:
                    moral_refusal_screen = "missing_llm_labels"
                else:
                    moral_refusal_screen = "clean_llm_labels"

                if moral_refusal_mode == "required" and missing:
                    audit_rows.append({**audit_base, "status": "missing_moral_llm_label"})
                    continue
                if excluded:
                    audit_rows.append({**audit_base, "status": "moral_refusal_or_degenerate_exclusion"})
                    continue

            hashes = (hash_a, hash_b)
            valid_votes = [
                vote
                for vote in votes_by_pair.get(str(job["pair_uid"]), [])
                if (vote.get("source_output_a_hash"), vote.get("source_output_b_hash")) == hashes
            ]
            by_judge: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for vote in valid_votes:
                by_judge[str(vote["judge_model"])].append(vote)
            verdicts = [
                derive_judge_verdict([vote["winner_condition"] for vote in judge_votes])
                for judge_votes in by_judge.values()
            ]
            if len(verdicts) < 3:
                audit_rows.append({**audit_base, "status": "insufficient_matching_judge_votes"})
                continue
            panel = derive_panel_winner_condition(job, verdicts)
            predicted = str(job["predicted_condition"])
            other = str(job.get("other_condition", ""))
            resolved = panel not in {"", "tie", "unresolved"}
            rows.append(
                {
                    "source_block": source_block,
                    "condition": condition,
                    "condition_label": CONDITION_LABEL[condition],
                    "target_label": TARGET_LABEL[condition],
                    "comparison": comparison,
                    "actor": str(job["actor"]),
                    "actor_label": ACTOR_LABEL.get(str(job["actor"]), str(job["actor"])),
                    "task": str(job["task"]),
                    "task_label": TASK_LABEL.get(str(job["task"]), str(job["task"])),
                    "domain": str(job.get("domain", "")),
                    "domain_label": str(job.get("domain_label", "")),
                    "item_label": str(job.get("item_label", "")),
                    "repeat": int(job.get("repeat", -1)),
                    "pair_uid": str(job["pair_uid"]),
                    "predicted_condition": predicted,
                    "other_condition": other,
                    "panel_winner_condition": panel,
                    "target_win": bool(resolved and panel == predicted),
                    "target_loss": bool(resolved and panel != predicted),
                    "tie": bool(panel == "tie"),
                    "resolved": bool(resolved),
                    "delta_u": pd.to_numeric(job.get("delta_u", ""), errors="coerce"),
                    "cause_pair_label": str(job.get("cause_pair_label", "")),
                    "moral_label_a": label_a,
                    "moral_label_b": label_b,
                    "moral_refusal_screen": moral_refusal_screen,
                    "amount_target_mismatch_from_base": is_amount_target_mismatch(job, source_block),
                }
            )
            audit_rows.append({**audit_base, "status": "included"})

    pairs = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    return pairs, audit


def wilson(wins: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def bootstrap_equal_cell(
    resolved: pd.DataFrame,
    cell_cols: list[str],
    *,
    seed: int,
    iterations: int,
) -> tuple[float, float, float, int]:
    if resolved.empty:
        return math.nan, math.nan, math.nan, 0
    grouped = list(resolved.groupby(cell_cols, dropna=False, sort=True)) if cell_cols else [((), resolved)]
    cell_arrays = [group["target_win"].to_numpy(dtype=float) for _, group in grouped if len(group)]
    if not cell_arrays:
        return math.nan, math.nan, math.nan, 0
    point = float(np.mean([arr.mean() for arr in cell_arrays]))
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    n_cells = len(cell_arrays)
    for idx in range(iterations):
        sampled = rng.integers(0, n_cells, size=n_cells)
        rates = []
        for cell_idx in sampled:
            arr = cell_arrays[int(cell_idx)]
            rates.append(float(rng.choice(arr, size=len(arr), replace=True).mean()))
        draws[idx] = float(np.mean(rates))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return point, float(lo), float(hi), n_cells


def exact_familywise_ci(wins: int, total: int, family_size: int, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    tail_alpha = alpha / (2 * family_size)
    lo = 0.0 if wins == 0 else float(beta.ppf(tail_alpha, wins, total - wins + 1))
    hi = 1.0 if wins == total else float(beta.ppf(1.0 - tail_alpha, wins + 1, total - wins))
    return lo, hi


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def summarize_subset(
    df: pd.DataFrame,
    filters: dict[str, str],
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    sub = df.copy()
    for key, value in filters.items():
        sub = sub[sub[key].eq(value)]
    resolved = sub[sub["resolved"]].copy()
    wins = int(resolved["target_win"].sum())
    n = int(len(resolved))
    ties = int(sub["tie"].sum())
    losses = n - wins
    pooled = wins / n if n else math.nan
    pooled_lo, pooled_hi = wilson(wins, n)

    condition = filters.get("condition", str(sub["condition"].iloc[0]) if not sub.empty else "")
    cell_cols = ["actor", "task"]
    if condition == "utility":
        cell_cols = ["actor", "task", "domain"]
    for fixed in ["actor", "task", "domain"]:
        if fixed in filters and fixed in cell_cols:
            cell_cols.remove(fixed)
    point, ci_lo, ci_hi, n_cells = bootstrap_equal_cell(
        resolved,
        cell_cols,
        seed=seed,
        iterations=iterations,
    )
    return {
        **filters,
        "resolved_n": n,
        "target_wins": wins,
        "target_losses": losses,
        "ties": ties,
        "all_pairs_after_screen": int(len(sub)),
        "pooled_win_rate": pooled,
        "pooled_wilson_ci_lo": pooled_lo,
        "pooled_wilson_ci_hi": pooled_hi,
        "equal_cell_mean": point,
        "equal_cell_ci_lo": ci_lo,
        "equal_cell_ci_hi": ci_hi,
        "bootstrap_cells": n_cells,
        "bootstrap_iterations": iterations,
    }


def make_summaries(pairs: pd.DataFrame, *, iterations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    idx = 0
    for condition in CONDITION_ORDER:
        rows.append(summarize_subset(pairs, {"condition": condition}, seed=seed + idx, iterations=iterations))
        idx += 1
        for task in TASK_ORDER:
            rows.append(
                summarize_subset(
                    pairs,
                    {"condition": condition, "task": task},
                    seed=seed + idx,
                    iterations=iterations,
                )
            )
            idx += 1
        for actor in ACTORS:
            rows.append(
                summarize_subset(
                    pairs,
                    {"condition": condition, "actor": actor},
                    seed=seed + idx,
                    iterations=iterations,
                )
            )
            idx += 1
        if condition == "utility":
            for domain in sorted(pairs.loc[pairs["condition"].eq("utility"), "domain"].dropna().unique()):
                if domain:
                    rows.append(
                        summarize_subset(
                            pairs,
                            {"condition": condition, "domain": domain},
                            seed=seed + idx,
                            iterations=iterations,
                        )
                    )
                    idx += 1
    summary = pd.DataFrame(rows)
    summary["condition_label"] = summary["condition"].map(CONDITION_LABEL)
    summary["task_label"] = summary["task"].map(TASK_LABEL).fillna(summary.get("task", ""))
    summary["actor_label"] = summary["actor"].map(ACTOR_LABEL).fillna(summary.get("actor", ""))

    cell_rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        cdf = pairs[pairs["condition"].eq(condition)].copy()
        for (actor, task), sub in cdf.groupby(["actor", "task"], sort=True):
            resolved = sub[sub["resolved"]]
            wins = int(resolved["target_win"].sum())
            n = int(len(resolved))
            lo, hi = exact_familywise_ci(wins, n, 28)
            p_value = binomtest(wins, n, 0.5, alternative="two-sided").pvalue if n else math.nan
            cell_rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABEL[condition],
                    "actor": actor,
                    "actor_label": ACTOR_LABEL.get(actor, actor),
                    "task": task,
                    "task_label": TASK_LABEL.get(task, task),
                    "resolved_n": n,
                    "target_wins": wins,
                    "target_losses": n - wins,
                    "ties": int(sub["tie"].sum()),
                    "target_win_rate_excluding_ties": wins / n if n else math.nan,
                    "familywise_ci_lo": lo,
                    "familywise_ci_hi": hi,
                    "familywise_ci_positive": bool(np.isfinite(lo) and lo > 0.5),
                    "p_two_sided_exact": p_value,
                }
            )
    cells = pd.DataFrame(cell_rows)
    adjusted_rows = []
    for condition, cdf in cells.groupby("condition", sort=False):
        adjusted = holm_adjust(cdf["p_two_sided_exact"].fillna(1.0).tolist())
        tmp = cdf.copy()
        tmp["holm_p_two_sided"] = adjusted
        tmp["holm_positive"] = (
            tmp["target_win_rate_excluding_ties"].gt(0.5)
            & tmp["holm_p_two_sided"].lt(0.05)
        )
        adjusted_rows.append(tmp)
    cells = pd.concat(adjusted_rows, ignore_index=True)
    return summary, cells


def pct(x: float) -> str:
    return "" if not np.isfinite(x) else f"{x:.1%}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    rows = []
    rows.append("| " + " | ".join(str(col) for col in df.columns) + " |")
    rows.append("| " + " | ".join("---" for _ in df.columns) + " |")
    for _, row in df.iterrows():
        rows.append(
            "| "
            + " | ".join("" if pd.isna(value) else str(value) for value in row.tolist())
            + " |"
        )
    return "\n".join(rows)


def write_markdown(
    pairs: pd.DataFrame,
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    cells: pd.DataFrame,
    *,
    moral_refusal_mode: str,
) -> None:
    total = summary[summary["task"].isna() & summary["actor"].isna() & summary["domain"].isna()].copy()
    by_task = summary[summary["task"].notna() & summary["actor"].isna() & summary["domain"].isna()].copy()
    by_domain = summary[summary["condition"].eq("utility") & summary["domain"].notna() & summary["task"].isna()].copy()
    source_counts = pairs.groupby(["condition", "source_block"]).size().reset_index(name="included_pairs")
    audit_counts = audit.groupby(["condition", "status"]).size().reset_index(name="rows")
    moral = pairs[pairs["condition"].eq("moral")]
    moral_screen_counts = (
        moral["moral_refusal_screen"].value_counts(dropna=False).rename_axis("screen_status").reset_index(name="pairs")
        if not moral.empty
        else pd.DataFrame(columns=["screen_status", "pairs"])
    )
    amount_mismatch = int(pairs["amount_target_mismatch_from_base"].sum()) if not pairs.empty else 0
    amount_source_blocks = set(
        pairs.loc[pairs["condition"].eq("amount"), "source_block"].dropna().astype(str)
    )

    lines = [
        "# Canonical High-N Condition Results",
        "",
        "Conditions: amount, moral, and high-low utility.",
        "Rows combine base repeats 0-4 with high-N extension repeats 5-9 where available.",
        "",
        "Primary estimate: equal-cell mean. Amount and moral average actor x task cells; high-low utility averages actor x task x domain cells. Panel ties are excluded from the win-rate denominator and reported separately.",
        "",
        f"Moral refusal mode used for this run: `{moral_refusal_mode}`.",
        "",
    ]
    if not moral_screen_counts.empty:
        lines += ["## Moral Refusal Screen Coverage", "", markdown_table(moral_screen_counts), ""]
    if amount_mismatch:
        lines += [
            "## Amount Prompt Caveat",
            "",
            f"{amount_mismatch} included amount pairs do not use the current neutral target text `A healthcare intervention at a children's hospital.`",
            "That makes the combined amount high-N estimate useful as a power-expanded diagnostic, but not as clean as moral and utility until the corrected amount base block is complete.",
            "",
        ]
    elif "base_amount_current_target_r0_r4" in amount_source_blocks:
        lines += [
            "## Amount Prompt Status",
            "",
            "Amount uses the corrected base block for repeats 0-4 plus the high-N extension block for repeats 5-9. Both use the current neutral target text `A healthcare intervention at a children's hospital.`",
            "",
        ]

    lines += [
        "## Total",
        "",
        "| condition | resolved | wins | losses | ties | pooled | equal-cell mean | equal-cell 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in total.sort_values("condition").iterrows():
        lines.append(
            f"| {row['condition_label']} | {int(row['resolved_n'])} | {int(row['target_wins'])} | "
            f"{int(row['target_losses'])} | {int(row['ties'])} | {pct(row['pooled_win_rate'])} | "
            f"{pct(row['equal_cell_mean'])} | {pct(row['equal_cell_ci_lo'])}-{pct(row['equal_cell_ci_hi'])} |"
        )

    lines += [
        "",
        "## By Task",
        "",
        "| condition | task | resolved | wins | losses | ties | equal-cell mean | equal-cell 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_task.sort_values(["condition", "task"]).iterrows():
        lines.append(
            f"| {row['condition_label']} | {row['task_label']} | {int(row['resolved_n'])} | "
            f"{int(row['target_wins'])} | {int(row['target_losses'])} | {int(row['ties'])} | "
            f"{pct(row['equal_cell_mean'])} | {pct(row['equal_cell_ci_lo'])}-{pct(row['equal_cell_ci_hi'])} |"
        )

    if not by_domain.empty:
        lines += [
            "",
            "## Utility By Domain",
            "",
            "| domain | resolved | wins | losses | ties | equal-cell mean | equal-cell 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in by_domain.sort_values("domain").iterrows():
            lines.append(
                f"| {row['domain']} | {int(row['resolved_n'])} | {int(row['target_wins'])} | "
                f"{int(row['target_losses'])} | {int(row['ties'])} | "
                f"{pct(row['equal_cell_mean'])} | {pct(row['equal_cell_ci_lo'])}-{pct(row['equal_cell_ci_hi'])} |"
            )

    model_task_counts = (
        cells.groupby("condition")[["familywise_ci_positive", "holm_positive"]]
        .sum()
        .reset_index()
    )
    lines += [
        "",
        "## Model-Task Positive Counts",
        "",
        "| condition | familywise-CI positive cells | Holm-positive cells |",
        "|---|---:|---:|",
    ]
    for _, row in model_task_counts.sort_values("condition").iterrows():
        lines.append(
            f"| {CONDITION_LABEL[row['condition']]} | {int(row['familywise_ci_positive'])}/28 | "
            f"{int(row['holm_positive'])}/28 |"
        )

    lines += [
        "",
        "## Source Blocks",
        "",
        markdown_table(source_counts),
        "",
        "## Audit Counts",
        "",
        markdown_table(audit_counts),
        "",
        "## Output Files",
        "",
        f"- `{ANALYSIS / f'{OUT_PREFIX}_pair_outcomes.csv'}`",
        f"- `{ANALYSIS / f'{OUT_PREFIX}_summary.csv'}`",
        f"- `{ANALYSIS / f'{OUT_PREFIX}_model_task.csv'}`",
        f"- `{ANALYSIS / f'{OUT_PREFIX}_audit.csv'}`",
        f"- `{ANALYSIS / f'{OUT_PREFIX}_summary.md'}`",
        "",
    ]
    (ANALYSIS / f"{OUT_PREFIX}_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument(
        "--moral-refusal-mode",
        choices=["available", "required"],
        default="available",
        help=(
            "available: apply available LLM labels and mark missing labels; "
            "required: exclude unlabeled moral pairs and report them in the audit"
        ),
    )
    args = parser.parse_args()

    pairs, audit = build_pair_rows(moral_refusal_mode=args.moral_refusal_mode)
    if pairs.empty:
        raise SystemExit("No analyzable pairs found.")

    summary, cells = make_summaries(pairs, iterations=args.bootstraps, seed=args.seed)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(ANALYSIS / f"{OUT_PREFIX}_pair_outcomes.csv", index=False)
    audit.to_csv(ANALYSIS / f"{OUT_PREFIX}_audit.csv", index=False)
    summary.to_csv(ANALYSIS / f"{OUT_PREFIX}_summary.csv", index=False)
    cells.to_csv(ANALYSIS / f"{OUT_PREFIX}_model_task.csv", index=False)
    write_markdown(
        pairs,
        audit,
        summary,
        cells,
        moral_refusal_mode=args.moral_refusal_mode,
    )
    print(f"wrote {len(pairs)} pair outcomes to {ANALYSIS / f'{OUT_PREFIX}_pair_outcomes.csv'}")
    print(f"summary: {ANALYSIS / f'{OUT_PREFIX}_summary.md'}")


if __name__ == "__main__":
    main()
