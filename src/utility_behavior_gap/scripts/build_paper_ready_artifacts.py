#!/usr/bin/env python3
"""Build canonical prompt, lineage, and paper-ready analysis artifacts.

This script is intentionally narrow. It only reads the approved current
lineages:

* direct instruction: framed_user_strong_headroom manifests;
* high-low and moral: June 13 fund-wording manifests;
* amount: June 10 modgrid amount manifests, because amount deliberately
  manipulates explicit dollar donation wording.

It writes a small curated artifact set under outputs/paper_ready.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

from utility_behavior_gap.constants import ACTOR_LABEL, ACTORS, TASK_LABEL
from utility_behavior_gap.paths import ROOT
from utility_behavior_gap.prompts import MAX_EFFORT_STRONG_SYSTEM_PROMPTS


ANALYSIS = ROOT / "outputs" / "analysis"
RUNS = ROOT / "outputs" / "api" / "runs"
OUT = ROOT / "outputs" / "paper_ready"
FIGURES = OUT / "figures"

DIRECT_MANIFEST_GLOB = "framed_user_strong_manifests__*.tsv"
FUND_MANIFEST_GLOB = "fund_wording_rerun_manifests__*.tsv"

TASK_ORDER = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]
TASK_SHORT = {
    "essay": "Essay",
    "grant_proposal_abstract": "Grant",
    "incident_postmortem": "Postmortem",
    "translation": "Translation",
}

CONDITION_LABEL = {
    "direct_instruction": "Direct instruction",
    "amount": "Amount",
    "moral": "Moral",
    "high_low": "High vs low utility",
}

TARGET_COLUMN = {
    "direct_instruction": "strong_win",
    "amount": "target_win",
    "moral": "target_win",
    "high_low": "target_win",
}

PAPER_BG = "#FFFFFF"
PANEL_BG = "#FFFFFF"
INK = "#1A1A1F"
SUBTLE = "#6B7280"
GRID = "#E9EDF5"
CHANCE_LINE = "#9CA3AF"
CI_POS_BAND = "#E8F4ED"
CI_POS_PILL_BG = "#DDF1E3"
CI_POS_PILL_INK = "#2F7A4F"
NEUTRAL_PILL_BG = "#F1F2F5"
NEUTRAL_PILL_INK = "#4B5563"

MODEL_COLORS = {
    "DeepSeek V3.2": "#2A8C9E",
    "GPT-5.4 mini": "#3A66C9",
    "GLM-5.1": "#5B6068",
    "Kimi K2.5": "#D4711B",
    "MiMo V2.5 Pro": "#2E8C5C",
    "Qwen3.5 9B": "#6E45BD",
    "Qwen3.6 Plus": "#C2304A",
}

X_AXIS_LABEL = {
    "direct_instruction": "Max-effort-side win rate (ties excluded)",
    "amount": "Larger-amount-side win rate (ties excluded)",
    "moral": "Good-cause-side win rate (ties excluded)",
    "high_low": "High-utility-side win rate (ties excluded)",
}

FIGURE_STEM = {
    "direct_instruction": "direct-instruction",
    "amount": "amount",
    "moral": "moral",
    "high_low": "high-low",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_manifest_paths(pattern: str) -> list[Path]:
    paths: list[Path] = []
    for tsv in sorted(RUNS.glob(pattern)):
        with tsv.open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    paths.append(Path(parts[2]))
    return sorted(set(paths))


def read_jobs(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def first_job(path: Path, predicate) -> dict[str, Any]:
    for row in read_jobs(path):
        if predicate(row):
            return row
    raise ValueError(f"no matching job in {path}")


def clean_prompt(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def prompt_block(label: str, prompt: str) -> str:
    return f"### {label}\n\n```text\n{clean_prompt(prompt)}\n```\n"


def write_prompt_book() -> None:
    """Write the canonical prompt book from actual verified manifests."""
    direct_paths = read_manifest_paths(DIRECT_MANIFEST_GLOB)
    fund_paths = read_manifest_paths(FUND_MANIFEST_GLOB)
    amount_paths = sorted(RUNS.glob("*__4-comparisons__*/generation_jobs.jsonl"))

    lines: list[str] = [
        "# Canonical Prompt Book",
        "",
        "Status: paper-ready prompt specification for the current analysis lineages.",
        "",
        "Authorized revisions relative to the earlier modgrid prompt book:",
        "",
        "- Direct instruction strong text is placed at the end of the user prompt, not in the system prompt.",
        "- Direct instruction, high-low utility, and moral prompts use `the sponsor will fund the following intervention:` for the $1,000 consequence slot.",
        "- The neutral framed target is `A healthcare intervention at a children's hospital.`",
        "- Amount remains an explicit dollar-amount manipulation and therefore keeps donation wording.",
        "",
        "All canonical generation system prompts are blank.",
        "",
        "The examples below are read from the actual run manifests, not hand-retyped.",
        "",
    ]

    for task in TASK_ORDER:
        lines += [f"## {TASK_LABEL.get(task, task)}", ""]

        direct_path = next(p for p in direct_paths if f"/{task}__" in str(p))
        direct = first_job(direct_path, lambda j: j.get("comparison") == "framed_user_strong_headroom")
        lines.append("### Direct instruction: framed neutral vs user-prompt max effort\n")
        lines.append("System prompt, both sides: blank.\n")
        lines.append(prompt_block("Direct low / framed neutral user prompt", direct["prompt_b"]))
        lines.append(prompt_block("Direct high / user-prompt max-effort user prompt", direct["prompt_a"]))

        fund_path = next(p for p in fund_paths if f"/{task}__" in str(p))
        highlow = first_job(fund_path, lambda j: str(j.get("comparison", "")).endswith("_highlow"))
        moral = first_job(fund_path, lambda j: str(j.get("comparison", "")).endswith("_moral"))
        lines.append("### High-low utility\n")
        lines.append("System prompt, both sides: blank.\n")
        lines.append(prompt_block("Utility high user prompt", highlow["prompt_a"]))
        lines.append(prompt_block("Utility low user prompt", highlow["prompt_b"]))
        lines.append("### Moral\n")
        lines.append("System prompt, both sides: blank.\n")
        lines.append(prompt_block("Moral high / good-cause user prompt", moral["prompt_a"]))
        lines.append(prompt_block("Moral low / bad-cause user prompt", moral["prompt_b"]))

        amount_path = next(p for p in amount_paths if f"/{task}__" in str(p))
        amount = first_job(amount_path, lambda j: str(j.get("comparison", "")).endswith("_amount"))
        lines.append("### Amount\n")
        lines.append("System prompt, both sides: blank.\n")
        lines.append("Amount intentionally retains donation wording because the manipulation is `$1,000,000` vs `$100`.\n")
        lines.append(prompt_block("Amount high user prompt", amount["prompt_a"]))
        lines.append(prompt_block("Amount low user prompt", amount["prompt_b"]))

    (OUT / "canonical_prompt_book.md").write_text("\n".join(lines), encoding="utf-8")


def verify_job(job: dict[str, Any], source_family: str) -> list[str]:
    errors: list[str] = []
    sys_a = job.get("system_prompt_a", "")
    sys_b = job.get("system_prompt_b", "")
    if sys_a or sys_b:
        errors.append("nonblank_system_prompt")
    prompt_a = str(job.get("prompt_a", ""))
    prompt_b = str(job.get("prompt_b", ""))
    comparison = str(job.get("comparison", ""))
    condition_a = str(job.get("condition_a", ""))
    condition_b = str(job.get("condition_b", ""))

    if source_family == "direct_instruction":
        if comparison != "framed_user_strong_headroom":
            errors.append("wrong_direct_comparison")
        if condition_a != "framed_user_strong" or condition_b != "framed_neutral":
            errors.append("wrong_direct_conditions")
        if "the sponsor will fund the following intervention:" not in prompt_a + prompt_b:
            errors.append("direct_missing_fund_sentence")
        if "A healthcare intervention at a children's hospital." not in prompt_a + prompt_b:
            errors.append("direct_missing_healthcare_neutral_target")
        if MAX_EFFORT_STRONG_SYSTEM_PROMPTS[job["task"]] not in prompt_a:
            errors.append("direct_strong_text_not_in_user_prompt")
        if MAX_EFFORT_STRONG_SYSTEM_PROMPTS[job["task"]] in prompt_b:
            errors.append("direct_strong_text_in_neutral_prompt")
    elif source_family == "fund_wording":
        if comparison.endswith("_highlow") or comparison.endswith("_moral"):
            if "the sponsor will fund the following intervention:" not in prompt_a + prompt_b:
                errors.append("fund_family_missing_fund_sentence")
            if "donate $1,000 toward the following" in prompt_a + prompt_b:
                errors.append("fund_family_has_old_1000_donation_sentence")
        elif comparison.endswith("_headroom") or comparison.endswith("_framing"):
            # These rows exist in the fund rerun but are not canonical direct
            # instruction. They are excluded from paper-ready reporting.
            errors.append("excluded_noncanonical_headroom_or_framing")
        else:
            errors.append("unexpected_fund_family_comparison")
    elif source_family == "amount":
        if not comparison.endswith("_amount"):
            errors.append("wrong_amount_comparison")
        if "$1,000,000" not in prompt_a or "$100" not in prompt_b:
            errors.append("amount_missing_dollar_manipulation")
        if "the sponsor will fund the following intervention:" in prompt_a + prompt_b:
            errors.append("amount_uses_fund_sentence")
    return errors


def audit_manifests() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for source_family, paths in [
        ("direct_instruction", read_manifest_paths(DIRECT_MANIFEST_GLOB)),
        ("fund_wording", read_manifest_paths(FUND_MANIFEST_GLOB)),
        ("amount", sorted(RUNS.glob("*__4-comparisons__*/generation_jobs.jsonl"))),
    ]:
        for path in paths:
            jobs = read_jobs(path)
            checked = 0
            failed = 0
            errors: set[str] = set()
            comparisons: set[str] = set()
            tasks: set[str] = set()
            actors: set[str] = set()
            for job in jobs:
                comparison = str(job.get("comparison", ""))
                if source_family == "amount" and not comparison.endswith("_amount"):
                    continue
                if source_family == "fund_wording" and not (
                    comparison.endswith("_highlow") or comparison.endswith("_moral")
                ):
                    continue
                if source_family == "direct_instruction" and comparison != "framed_user_strong_headroom":
                    continue
                checked += 1
                comparisons.add(comparison)
                tasks.add(str(job.get("task", "")))
                actors.add(str(job.get("actor", "")))
                job_errors = verify_job(job, source_family)
                if job_errors:
                    failed += 1
                    errors.update(job_errors)
            if checked == 0:
                continue
            rows.append(
                {
                    "source_family": source_family,
                    "manifest": str(path),
                    "run_id": path.parent.name,
                    "actors": ",".join(sorted(actors)),
                    "tasks": ",".join(sorted(tasks)),
                    "comparisons": ",".join(sorted(comparisons)),
                    "checked_jobs": checked,
                    "failed_jobs": failed,
                    "status": "PASS" if checked and failed == 0 else "FAIL",
                    "errors": ",".join(sorted(errors)),
                }
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "canonical_prompt_audit.csv", index=False)

    status_counts = audit["status"].value_counts().to_dict()
    lines = [
        "# Canonical Prompt Audit",
        "",
        f"Manifests checked: {len(audit)}",
        f"Status counts: {status_counts}",
        "",
        "Audit rules:",
        "",
        "- Direct instruction must be `framed_user_strong_headroom`, blank system prompts, fund-intervention wording, healthcare neutral target, and max-effort text appended to the strong user prompt only.",
        "- High-low and moral must come from the fund-wording manifests, use blank system prompts, and use fund-intervention wording.",
        "- Amount must come from `*_amount` rows, use blank system prompts, and retain explicit `$1,000,000` vs `$100` donation wording.",
        "",
    ]
    if (audit["status"] != "PASS").any():
        failures = audit[audit["status"] != "PASS"]
        lines += [
            "## Failures",
            "",
            "| source_family | run_id | checked_jobs | failed_jobs | errors |",
            "|---|---|---:|---:|---|",
        ]
        for _, row in failures.iterrows():
            lines.append(
                f"| {row['source_family']} | `{row['run_id']}` | "
                f"{int(row['checked_jobs'])} | {int(row['failed_jobs'])} | {row['errors']} |"
            )
    else:
        lines += ["All checked canonical manifests passed.", ""]
    (OUT / "canonical_prompt_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return audit


def load_condition_pairs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    direct = pd.read_csv(ANALYSIS / "framed_user_strong_panel_pairs.csv", keep_default_na=False)
    direct["condition"] = "direct_instruction"
    direct["source_family"] = "direct_instruction"
    direct["target_win"] = direct["outcome"].eq("strong")
    direct["resolved"] = direct["outcome"].isin(["strong", "neutral"])
    direct["tie"] = direct["outcome"].eq("tie")
    frames.append(
        direct[
            [
                "condition",
                "source_family",
                "actor",
                "task",
                "item_label",
                "pair_uid",
                "panel_winner_condition",
                "target_win",
                "resolved",
                "tie",
            ]
        ]
    )

    fund = pd.read_csv(ANALYSIS / "fund_wording_judged_pairs.csv", keep_default_na=False)
    for suffix, condition in [("_highlow", "high_low"), ("_moral", "moral")]:
        sub = fund[fund["comparison"].astype(str).str.endswith(suffix)].copy()
        sub["condition"] = condition
        sub["source_family"] = "fund_wording"
        sub["target_win"] = sub["panel_winner_condition"].eq(sub["predicted_condition"])
        sub["resolved"] = ~sub["panel_winner_condition"].isin(["tie", "unresolved", ""])
        sub["tie"] = sub["panel_winner_condition"].eq("tie")
        frames.append(
            sub[
                [
                    "condition",
                    "source_family",
                    "actor",
                    "task",
                    "item_label",
                    "pair_uid",
                    "domain",
                    "panel_winner_condition",
                    "target_win",
                    "resolved",
                    "tie",
                ]
            ]
        )

    amount = pd.read_csv(ANALYSIS / "modgrid_judged_pairs.csv", keep_default_na=False)
    amount = amount[amount["comparison"].astype(str).str.endswith("_amount")].copy()
    amount["condition"] = "amount"
    amount["source_family"] = "amount"
    amount["target_win"] = amount["panel_winner_condition"].eq(amount["predicted_condition"])
    amount["resolved"] = ~amount["panel_winner_condition"].isin(["tie", "unresolved", ""])
    amount["tie"] = amount["panel_winner_condition"].eq("tie")
    frames.append(
        amount[
            [
                "condition",
                "source_family",
                "actor",
                "task",
                "item_label",
                "pair_uid",
                "panel_winner_condition",
                "target_win",
                "resolved",
                "tie",
            ]
        ]
    )

    pairs = pd.concat(frames, ignore_index=True)
    pairs["domain"] = pairs.get("domain", "").fillna("")
    pairs["target_win"] = pairs["target_win"].astype(bool)
    pairs["resolved"] = pairs["resolved"].astype(bool)
    pairs["tie"] = pairs["tie"].astype(bool)
    pairs.to_csv(OUT / "canonical_pair_level_outcomes.csv", index=False)
    return pairs


def wilson(wins: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


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
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def bootstrap_equal_cell(
    resolved: pd.DataFrame,
    group_cols: list[str],
    *,
    seed: int,
    iterations: int = 1000,
) -> tuple[float, float, float]:
    if resolved.empty:
        return math.nan, math.nan, math.nan
    arrays = {
        key if isinstance(key, tuple) else (key,): group["target_win"].to_numpy(dtype=float)
        for key, group in resolved.groupby(group_cols, sort=True)
    }
    point = float(np.mean([arr.mean() for arr in arrays.values()]))
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    keys = list(arrays)
    for i in range(iterations):
        sampled_keys = rng.choice(len(keys), size=len(keys), replace=True)
        cell_rates = []
        for idx in sampled_keys:
            arr = arrays[keys[int(idx)]]
            cell_rates.append(float(rng.choice(arr, size=len(arr), replace=True).mean()))
        estimates[i] = float(np.mean(cell_rates))
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return point, float(lo), float(hi)


def summarize_pairs(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []

    breakouts: list[tuple[str, list[str]]] = [
        ("total", []),
        ("by_condition_task", ["task"]),
        ("by_condition_actor", ["actor"]),
        ("by_condition_actor_task", ["actor", "task"]),
    ]
    for condition, cdf in pairs.groupby("condition", sort=True):
        for breakout, cols in breakouts:
            groups = [((), cdf)] if not cols else cdf.groupby(cols, sort=True)
            for key, sub in groups:
                key_tuple = key if isinstance(key, tuple) else (key,)
                if not cols:
                    key_tuple = ()
                resolved = sub[sub["resolved"]].copy()
                wins = int(resolved["target_win"].sum())
                n = int(len(resolved))
                ties = int(sub["tie"].sum())
                pooled = wins / n if n else math.nan
                pooled_lo, pooled_hi = wilson(wins, n)
                boot_cols = ["actor", "task"]
                if "actor" in cols:
                    boot_cols = [c for c in boot_cols if c != "actor"]
                if "task" in cols:
                    boot_cols = [c for c in boot_cols if c != "task"]
                if boot_cols:
                    mean, ci_lo, ci_hi = bootstrap_equal_cell(
                        resolved,
                        boot_cols,
                        seed=1000 + len(rows),
                    )
                else:
                    # Within a single actor-task cell the equal-cell estimand
                    # is just the resolved-pair win rate, so use the Wilson
                    # interval already computed for the plotted cell.
                    mean, ci_lo, ci_hi = pooled, pooled_lo, pooled_hi
                row = {
                    "condition": condition,
                    "condition_label": CONDITION_LABEL[condition],
                    "breakout": breakout,
                    "actor": "all",
                    "task": "all",
                    "resolved_n": n,
                    "target_wins": wins,
                    "target_losses": n - wins,
                    "ties": ties,
                    "pooled_win_rate": pooled,
                    "pooled_wilson_ci_lo": pooled_lo,
                    "pooled_wilson_ci_hi": pooled_hi,
                    "equal_cell_mean": mean,
                    "equal_cell_ci_lo": ci_lo,
                    "equal_cell_ci_hi": ci_hi,
                    "source_family": ",".join(sorted(sub["source_family"].unique())),
                }
                for col, val in zip(cols, key_tuple):
                    row[col] = val
                rows.append(row)
                if breakout == "by_condition_actor_task":
                    cell_rows.append(row.copy())

    summary = pd.DataFrame(rows)
    cells = pd.DataFrame(cell_rows)
    summary.to_csv(OUT / "canonical_aggregate_results.csv", index=False)
    cells.to_csv(OUT / "canonical_model_task_results.csv", index=False)
    return summary, cells


def write_summary_md(summary: pd.DataFrame) -> None:
    total = summary[summary["breakout"].eq("total")].copy()
    total = total.sort_values("condition")
    by_task = summary[summary["breakout"].eq("by_condition_task")].copy()
    by_task["task_label"] = by_task["task"].map(TASK_LABEL).fillna(by_task["task"])
    lines = [
        "# Paper-Ready Canonical Results",
        "",
        "All rows here are produced only after manifest prompt audit.",
        "",
        "Primary reported estimate: equal-cell mean. For total rows, actor and task cells are equally weighted; for task rows, actors are equally weighted. Panel ties are excluded from the win-rate denominator and reported separately.",
        "",
        "## Total",
        "",
        "| condition | source | resolved | wins | ties | pooled win rate | equal-cell mean | equal-cell CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in total.iterrows():
        lines.append(
            f"| {row['condition_label']} | {row['source_family']} | {int(row['resolved_n'])} | "
            f"{int(row['target_wins'])} | {int(row['ties'])} | "
            f"{row['pooled_win_rate']:.1%} | {row['equal_cell_mean']:.1%} | "
            f"{row['equal_cell_ci_lo']:.1%}-{row['equal_cell_ci_hi']:.1%} |"
        )
    lines += ["", "## By Task", "", "| condition | task | resolved | wins | ties | equal-cell mean | equal-cell CI |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, row in by_task.sort_values(["condition", "task"]).iterrows():
        lines.append(
            f"| {row['condition_label']} | {row['task_label']} | {int(row['resolved_n'])} | "
            f"{int(row['target_wins'])} | {int(row['ties'])} | "
            f"{row['equal_cell_mean']:.1%} | {row['equal_cell_ci_lo']:.1%}-{row['equal_cell_ci_hi']:.1%} |"
        )
    lines += [
        "",
        "## Important Exclusions",
        "",
        "- `highlow_intervention_final_*` artifacts are not canonical because those runs used nonblank neutral system prompts.",
        "- Old `modgrid_*_headroom` rows are not canonical direct-instruction results because the strong cue was in the system prompt.",
        "- Amount rows are intentionally from the June 10 modgrid amount runs, because amount was not part of the fund-wording rerun and must keep explicit dollar donation wording.",
        "",
    ]
    (OUT / "paper_ready_results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_model_task_plot_data(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, cdf in cells.groupby("condition", sort=True):
        family_size = len(cdf)
        condition_rows: list[dict[str, Any]] = []
        for _, row in cdf.iterrows():
            wins = int(row["target_wins"])
            total = int(row["resolved_n"])
            rate = wins / total if total else math.nan
            lo, hi = exact_familywise_ci(wins, total, family_size)
            p_value = (
                binomtest(wins, total, 0.5, alternative="two-sided").pvalue
                if total
                else math.nan
            )
            condition_rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABEL[condition],
                    "actor": row["actor"],
                    "actor_label": ACTOR_LABEL.get(row["actor"], row["actor"]),
                    "task": row["task"],
                    "task_label": TASK_LABEL.get(row["task"], row["task"]),
                    "target_wins": wins,
                    "target_losses": int(row["target_losses"]),
                    "ties": int(row["ties"]),
                    "n_excl_tie": total,
                    "target_win_rate_excluding_ties": rate,
                    "familywise_ci_method": (
                        "Bonferroni exact binomial 95% familywise CI "
                        f"across {family_size} model-task cells"
                    ),
                    "familywise_ci_lo": lo,
                    "familywise_ci_hi": hi,
                    "familywise_ci_positive": lo > 0.50,
                    "p_two_sided_exact": p_value,
                    "source_family": row["source_family"],
                }
            )
        adjusted = holm_adjust([float(row["p_two_sided_exact"]) for row in condition_rows])
        for row, p_holm in zip(condition_rows, adjusted):
            row["holm_p_two_sided"] = p_holm
            row["holm_positive"] = (
                row["target_win_rate_excluding_ties"] > 0.5 and p_holm < 0.05
            )
        rows.extend(condition_rows)

    out = pd.DataFrame(rows)
    out["_actor_order"] = out["actor"].map({actor: idx for idx, actor in enumerate(ACTORS)})
    out["_task_order"] = out["task"].map({task: idx for idx, task in enumerate(TASK_ORDER)})
    out = out.sort_values(["condition", "_task_order", "_actor_order"]).drop(
        columns=["_actor_order", "_task_order"]
    )
    out.to_csv(OUT / "canonical_model_task_plot_data.csv", index=False)
    return out


def strip_spines(ax, keep=()) -> None:
    for spine in ("top", "right", "left", "bottom"):
        if spine in keep:
            ax.spines[spine].set_color("#9CA3AF")
            ax.spines[spine].set_linewidth(0.7)
        else:
            ax.spines[spine].set_visible(False)


def render_standard_panel(
    ax,
    df_task: pd.DataFrame,
    condition: str,
    task_label: str,
    panel_letter: str,
    *,
    show_xlabel: bool,
) -> None:
    ax.set_facecolor(PANEL_BG)

    actor_labels = [ACTOR_LABEL[actor] for actor in ACTORS if actor in set(df_task["actor"])]
    df_task = df_task.set_index("actor_label").loc[actor_labels].reset_index()
    n = len(actor_labels)

    for i, row in df_task.iterrows():
        y = n - 1 - i
        if bool(row["familywise_ci_positive"]):
            ax.add_patch(
                mpatches.Rectangle(
                    (0.005, y - 0.42),
                    0.99,
                    0.84,
                    facecolor=CI_POS_BAND,
                    edgecolor="none",
                    zorder=0,
                )
            )

    ax.axvline(0.50, color=CHANCE_LINE, lw=1.2, ls=(0, (4, 3)), alpha=0.9, zorder=1)

    for i, row in df_task.iterrows():
        y = n - 1 - i
        actor_label = row["actor_label"]
        color = MODEL_COLORS[actor_label]
        rate = float(row["target_win_rate_excluding_ties"])
        lo = float(row["familywise_ci_lo"])
        hi = float(row["familywise_ci_hi"])
        ci_pos = bool(row["familywise_ci_positive"])

        ax.hlines(
            y,
            lo,
            hi,
            color=color,
            lw=4.0,
            alpha=0.42 if ci_pos else 0.30,
            capstyle="round",
            zorder=2,
        )
        for x in (lo, hi):
            ax.vlines(
                x,
                y - 0.16,
                y + 0.16,
                color=color,
                lw=1.3,
                alpha=0.55 if ci_pos else 0.40,
                zorder=2,
            )
        ax.scatter(rate, y, s=145, color=color, edgecolor="white", linewidth=1.6, zorder=4)
        ax.text(
            min(hi + 0.025, 1.02),
            y,
            f"{rate:.2f}",
            ha="left",
            va="center",
            fontsize=10.0,
            color=color,
            fontweight="semibold",
            zorder=5,
        )

    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels(actor_labels, fontsize=10.5)
    for tick, actor_label in zip(ax.get_yticklabels(), actor_labels):
        tick.set_color(MODEL_COLORS[actor_label])
        tick.set_fontweight("semibold")

    ax.set_xlim(0.005, 1.06)
    ax.set_ylim(-0.55, n - 0.10)
    ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.tick_params(axis="x", labelsize=10)
    if show_xlabel:
        ax.set_xlabel(X_AXIS_LABEL[condition], color=INK, labelpad=4, fontsize=11)
    else:
        ax.set_xlabel("")

    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    strip_spines(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=SUBTLE, length=3, width=0.6)

    ax.text(
        -0.005,
        1.20,
        task_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        color=INK,
        fontweight="bold",
    )

    n_pos = int(df_task["familywise_ci_positive"].sum())
    n_total = len(df_task)
    chip_bg = CI_POS_PILL_BG if n_pos > 0 else NEUTRAL_PILL_BG
    chip_fg = CI_POS_PILL_INK if n_pos > 0 else NEUTRAL_PILL_INK
    ax.text(
        1.0,
        1.20,
        f"{n_pos} / {n_total} CI-positive",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color=chip_fg,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.30", facecolor=chip_bg, edgecolor="none"),
    )

    n_excl = df_task["n_excl_tie"].dropna().astype(int)
    if len(n_excl) > 0:
        n_lo, n_hi = int(n_excl.min()), int(n_excl.max())
        n_note = (
            f"n = {n_lo} pairs / actor"
            if n_lo == n_hi
            else f"n = {n_lo}-{n_hi} pairs / actor"
        )
        ax.text(
            -0.005,
            1.06,
            f"{n_note}; FWER 95% CIs",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            color=SUBTLE,
        )

    ax.text(
        -0.30,
        1.18,
        panel_letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=INK,
        va="top",
        ha="left",
    )


def plot_condition_cells(cells: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.titleweight": "regular",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10.5,
            "axes.edgecolor": "#9CA3AF",
            "axes.linewidth": 0.6,
            "figure.dpi": 200,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    plot_data = prepare_model_task_plot_data(cells)
    for condition, cdf in plot_data.groupby("condition", sort=True):
        fig = plt.figure(figsize=(12.8, 7.0), facecolor=PAPER_BG)
        gs = fig.add_gridspec(
            2,
            2,
            top=0.92,
            bottom=0.10,
            left=0.10,
            right=0.985,
            hspace=0.55,
            wspace=0.55,
        )
        panel_letters = ["A", "B", "C", "D"]
        for k, task in enumerate(TASK_ORDER):
            ax = fig.add_subplot(gs[k // 2, k % 2])
            sub = cdf[cdf["task"].eq(task)].copy()
            if sub.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"no data: {TASK_LABEL.get(task, task)}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
                continue
            render_standard_panel(
                ax,
                sub,
                condition,
                TASK_LABEL.get(task, task),
                panel_letters[k],
                show_xlabel=(k // 2 == 1),
            )
        stem = FIGURE_STEM[condition]
        fig.savefig(FIGURES / f"{stem}_model_task.png", dpi=240)
        fig.savefig(FIGURES / f"{stem}_model_task.pdf", facecolor=PAPER_BG)
        plt.close(fig)


def write_artifact_index(audit: pd.DataFrame) -> None:
    lines = [
        "# Paper-Ready Artifact Index",
        "",
        "This directory is the canonical entry point for paper-ready analysis artifacts.",
        "",
        "## Use These",
        "",
        "- `canonical_prompt_book.md`: corrected prompt book.",
        "- `canonical_prompt_audit.md` and `.csv`: manifest-level prompt verification.",
        "- `canonical_pair_level_outcomes.csv`: verified pair-level outcomes.",
        "- `canonical_aggregate_results.csv`: aggregate and breakout estimates.",
        "- `canonical_model_task_results.csv`: model-by-task estimates used for figures.",
        "- `canonical_model_task_plot_data.csv`: model-by-task plot estimates with familywise CIs and Holm-adjusted p-values.",
        "- `paper_ready_results_summary.md`: human-readable aggregate summary.",
        "- `figures/*_model_task.{png,pdf}`: paper-style model-by-task figures for each condition.",
        "",
        "## Do Not Treat As Canonical Without Re-Audit",
        "",
        "- `outputs/analysis/highlow_intervention_final_*`: neutral-system high-low variant.",
        "- `outputs/analysis/modgrid_prompt_book.md`: superseded by the corrected prompt book here.",
        "- `outputs/analysis/modgrid_full_breakout_results.*`: mixes current rows with legacy amount and old system-prompt headroom; useful history, not the canonical table.",
        "- one-off exploratory files in `outputs/analysis/`: keep for provenance, but do not cite unless added to this index after audit.",
        "",
        f"Prompt audit status counts: {audit['status'].value_counts().to_dict()}",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_prompt_book()
    audit = audit_manifests()
    if (audit["status"] != "PASS").any():
        raise SystemExit("Prompt audit failed; refusing to build paper-ready results.")
    pairs = load_condition_pairs()
    summary, cells = summarize_pairs(pairs)
    write_summary_md(summary)
    plot_condition_cells(cells)
    write_artifact_index(audit)
    print(f"wrote paper-ready artifacts to {OUT}")


if __name__ == "__main__":
    main()
