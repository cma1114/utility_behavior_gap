#!/usr/bin/env python3
"""Reusable model-by-task lollipop analysis for two-arm judged runs.

This is local-only: it reads stored generation jobs, generations, and judge
votes. It does not make API calls.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition
from utility_behavior_gap.openrouter import judge_model_ids
from utility_behavior_gap.paths import ANALYSIS, FIGURES, ROOT

RUNS = ROOT / "outputs" / "api" / "runs"
FAMILY_ALPHA = 0.05

INK = "#1A1A1F"
SUBTLE = "#6B7280"
GRID = "#E9EDF5"
PANEL_BG = "#FFFFFF"
PAPER_BG = "#FFFFFF"
CHANCE_LINE = "#9CA3AF"
CI_POS_BAND = "#E8F4ED"
CI_POS_PILL_BG = "#DDF1E3"
CI_POS_PILL_INK = "#2F7A4F"
NEUTRAL_PILL_BG = "#F1F2F5"
NEUTRAL_PILL_INK = "#4B5563"
MISSING_INK = "#9CA3AF"

MODEL_COLORS = {
    "DeepSeek V3.2": "#2A8C9E",
    "GPT-5.4 mini": "#3A66C9",
    "GLM-5.1": "#5B6068",
    "Kimi K2.5": "#D4711B",
    "MiMo V2.5 Pro": "#2E8C5C",
    "Qwen3.5 9B": "#6E45BD",
    "Qwen3.6 Plus": "#C2304A",
}

DEFAULT_ACTORS = [
    ("deepseek-v3.2-or", "DeepSeek V3.2"),
    ("gpt-5.4-mini-or", "GPT-5.4 mini"),
    ("glm-5.1-or", "GLM-5.1"),
    ("kimi-k2.5-or", "Kimi K2.5"),
    ("mimo-v25-pro-or", "MiMo V2.5 Pro"),
    ("qwen3.5-9b-or", "Qwen3.5 9B"),
    ("qwen3.6-plus-or", "Qwen3.6 Plus"),
]

DEFAULT_TASKS = [
    ("essay", "Essay writing"),
    ("grant_proposal_abstract", "Grant abstract"),
    ("incident_postmortem", "Incident postmortem"),
    ("translation", "Translation"),
]

JUDGE_MODELS = set(judge_model_ids())


def output_is_valid(row: dict[str, Any] | None) -> bool:
    if row is None or row.get("success") is False:
        return False
    finish = str(row.get("finish_reason") or "")
    if finish and finish != "stop":
        return False
    return bool(str(row.get("output_text") or "").strip())


def bonferroni_exact_ci(wins: int, total: int, family_size: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    tail_alpha = FAMILY_ALPHA / (2 * family_size)
    lo = 0.0 if wins == 0 else float(beta.ppf(tail_alpha, wins, total - wins + 1))
    hi = 1.0 if wins == total else float(beta.ppf(1.0 - tail_alpha, wins + 1, total - wins))
    return lo, hi


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    headers = [str(col) for col in show.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    return "\n".join(lines)


def pct(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{100 * float(value):.1f}%"


def judge_verdicts(votes: list[dict[str, Any]]) -> list[str]:
    by_judge: dict[str, list[str]] = defaultdict(list)
    for vote in votes:
        by_judge[str(vote.get("judge_model") or "")].append(str(vote.get("winner_condition") or ""))
    return [derive_judge_verdict(values) for _, values in sorted(by_judge.items())]


def valid_votes_for_pair(
    *,
    pair_uid: str,
    votes_by_pair: dict[str, list[dict[str, Any]]],
    output_a: dict[str, Any] | None,
    output_b: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if output_a is None or output_b is None:
        return []
    expected_hashes = (output_text_fingerprint(output_a), output_text_fingerprint(output_b))
    out = []
    for vote in votes_by_pair.get(pair_uid, []):
        if vote.get("success") is False:
            continue
        if str(vote.get("judge_model") or "") not in JUDGE_MODELS:
            continue
        if (vote.get("source_output_a_hash"), vote.get("source_output_b_hash")) != expected_hashes:
            continue
        out.append(vote)
    return out


def read_run(run_dir: Path, *, left_condition: str, right_condition: str) -> dict[str, Any]:
    jobs_path = run_dir / "generation_jobs.jsonl"
    generations_path = run_dir / "generations.jsonl"
    if not jobs_path.exists() or not generations_path.exists():
        raise FileNotFoundError(f"missing run artifacts in {run_dir}")

    jobs = read_jsonl(jobs_path)
    generations = {
        str(row.get("output_id") or ""): row
        for row in read_jsonl(generations_path)
        if row.get("output_id")
    }
    votes_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    votes_path = run_dir / "judge_votes.jsonl"
    if votes_path.exists():
        for vote in read_jsonl(votes_path):
            votes_by_pair[str(vote.get("pair_uid") or "")].append(vote)

    panel_counts = {left_condition: 0, right_condition: 0, "tie": 0, "unresolved": 0}
    pair_rows: list[dict[str, Any]] = []
    valid_pairs = 0
    complete_pairs = 0

    for job in jobs:
        pair_uid = str(job["pair_uid"])
        output_a = generations.get(f"{pair_uid}::a")
        output_b = generations.get(f"{pair_uid}::b")
        outputs_valid = output_is_valid(output_a) and output_is_valid(output_b)
        if outputs_valid:
            valid_pairs += 1
        votes = valid_votes_for_pair(
            pair_uid=pair_uid,
            votes_by_pair=votes_by_pair,
            output_a=output_a,
            output_b=output_b,
        )
        complete = outputs_valid and len(votes) >= len(JUDGE_MODELS) * 2
        if complete:
            complete_pairs += 1
            panel = derive_panel_winner_condition(job, judge_verdicts(votes))
        else:
            panel = "unresolved"
        if panel not in {left_condition, right_condition, "tie"}:
            panel = "unresolved"
        panel_counts[panel] += 1
        pair_rows.append(
            {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "actor": job.get("actor", ""),
                "actor_label": job.get("actor_label", ""),
                "task": job.get("task", ""),
                "task_label": job.get("task_label", ""),
                "pair_uid": pair_uid,
                "panel_winner_condition": panel,
                "complete": complete,
                "n_valid_vote_rows": len(votes),
            }
        )

    return {
        "run_dir": str(run_dir),
        "expected_pairs": len(jobs),
        "valid_pairs": valid_pairs,
        "complete_pairs": complete_pairs,
        "panel_counts": panel_counts,
        "pair_rows": pair_rows,
    }


def candidate_run_dirs(comparison: str, explicit: list[Path]) -> list[Path]:
    if explicit:
        return [path.resolve() for path in explicit]
    return sorted(RUNS.glob(f"*__{comparison}__*__hash-*"))


def parse_run_name(run_dir: Path) -> tuple[str, str, str] | None:
    parts = run_dir.name.split("__")
    if len(parts) < 5:
        return None
    return parts[0], parts[1], parts[2]


def infer_run_cell(run_dir: Path) -> tuple[str, str, str] | None:
    parsed = parse_run_name(run_dir)
    if parsed is not None:
        return parsed
    jobs_path = run_dir / "generation_jobs.jsonl"
    if not jobs_path.exists():
        return None
    jobs = read_jsonl(jobs_path)
    if not jobs:
        return None
    first = jobs[0]
    task = str(first.get("task") or "")
    comparison = str(first.get("comparison") or "")
    actor = str(first.get("actor") or "")
    if not task or not comparison or not actor:
        return None
    return task, comparison, actor


def select_runs(
    *,
    comparison: str,
    explicit_run_dirs: list[Path],
    actor_order: list[tuple[str, str]],
    task_order: list[tuple[str, str]],
    left_condition: str,
    right_condition: str,
    combine_runs: bool,
) -> tuple[dict[tuple[str, str], list[Path]], pd.DataFrame]:
    actors = {actor for actor, _ in actor_order}
    tasks = {task for task, _ in task_order}
    rows: list[dict[str, Any]] = []
    by_cell: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)

    for run_dir in candidate_run_dirs(comparison, explicit_run_dirs):
        parsed = infer_run_cell(run_dir)
        if parsed is None:
            continue
        task, run_comparison, actor = parsed
        if run_comparison != comparison or task not in tasks or actor not in actors:
            continue
        if not (run_dir / "generation_jobs.jsonl").exists() or not (run_dir / "generations.jsonl").exists():
            continue
        status = read_run(run_dir, left_condition=left_condition, right_condition=right_condition)
        row = {
            "actor": actor,
            "task": task,
            "run_dir": str(run_dir),
            "expected_pairs": status["expected_pairs"],
            "valid_pairs": status["valid_pairs"],
            "complete_pairs": status["complete_pairs"],
        }
        row.update({f"panel_{key}": value for key, value in status["panel_counts"].items()})
        rows.append(row)
        by_cell[(actor, task)].append((run_dir, status))

    selected: dict[tuple[str, str], list[Path]] = {}
    for cell, candidates in by_cell.items():
        candidates = sorted(
            candidates,
            key=lambda item: (
                item[1]["complete_pairs"],
                item[1]["valid_pairs"],
                item[1]["expected_pairs"],
                item[0].name,
            ),
            reverse=True,
        )
        if combine_runs:
            selected[cell] = sorted([run_dir for run_dir, _ in candidates], key=lambda path: path.name)
        else:
            selected[cell] = [candidates[0][0]]
    return selected, pd.DataFrame(rows)


def cells_from_runs(
    selected: dict[tuple[str, str], list[Path]],
    *,
    actor_order: list[tuple[str, str]],
    task_order: list[tuple[str, str]],
    left_condition: str,
    right_condition: str,
    family_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actor_labels = dict(actor_order)
    task_labels = dict(task_order)
    cell_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for (actor, task), run_dirs in sorted(selected.items()):
        combined_pair_rows: list[dict[str, Any]] = []
        seen_pair_uids: set[str] = set()
        for run_dir in run_dirs:
            status = read_run(run_dir, left_condition=left_condition, right_condition=right_condition)
            for row in status["pair_rows"]:
                pair_uid = str(row.get("pair_uid") or "")
                if pair_uid in seen_pair_uids:
                    continue
                seen_pair_uids.add(pair_uid)
                combined_pair_rows.append(row)
                pair_rows.append(row)
        pairs = pd.DataFrame(combined_pair_rows)
        left_wins = int(pairs["panel_winner_condition"].eq(left_condition).sum())
        right_wins = int(pairs["panel_winner_condition"].eq(right_condition).sum())
        ties = int(pairs["panel_winner_condition"].eq("tie").sum())
        unresolved = int(pairs["panel_winner_condition"].eq("unresolved").sum())
        complete_pairs = int(pairs["complete"].sum())
        non_ties = left_wins + right_wins
        rate = left_wins / non_ties if non_ties else math.nan
        ci_lo, ci_hi = bonferroni_exact_ci(left_wins, non_ties, family_size)
        p_value = (
            binomtest(left_wins, non_ties, 0.5, alternative="two-sided").pvalue
            if non_ties
            else math.nan
        )
        cell_rows.append(
            {
                "actor": actor,
                "actor_label": actor_labels.get(actor, actor),
                "task": task,
                "task_label": task_labels.get(task, task),
                "run_id": ";".join(run_dir.name for run_dir in run_dirs),
                "run_dir": ";".join(str(run_dir) for run_dir in run_dirs),
                "n_pairs": len(combined_pair_rows),
                "complete_pairs": complete_pairs,
                "left_wins": left_wins,
                "right_wins": right_wins,
                "ties": ties,
                "unresolved": unresolved,
                "n_excluding_ties": non_ties,
                "left_win_rate_excluding_ties": rate,
                "familywise_ci_lo": ci_lo,
                "familywise_ci_hi": ci_hi,
                "familywise_ci_positive": bool(math.isfinite(ci_lo) and ci_lo > 0.5),
                "p_two_sided_exact": p_value,
            }
        )

    cells = pd.DataFrame(cell_rows)
    if not cells.empty:
        cells["holm_p_two_sided"] = holm_adjust(cells["p_two_sided_exact"].fillna(1.0).tolist())
        cells["holm_positive"] = cells["left_win_rate_excluding_ties"].gt(0.5) & cells["holm_p_two_sided"].lt(0.05)
        cells["familywise_ci_method"] = (
            f"Bonferroni exact binomial 95% familywise CI across {family_size} plotted/planned cells"
        )
    return cells, pd.DataFrame(pair_rows)


def wilson_ci(wins: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def bootstrap_equal_cell_mean(
    pairs: pd.DataFrame,
    *,
    task: str | None,
    left_condition: str,
    right_condition: str,
    iterations: int,
    seed: int,
) -> tuple[float, float, float, str]:
    work = pairs[pairs["complete"]].copy()
    if task is not None:
        work = work[work["task"].eq(task)].copy()
    work = work[work["panel_winner_condition"].isin([left_condition, right_condition])].copy()
    if work.empty:
        return math.nan, math.nan, math.nan, "no non-tied completed pairs"
    work["target_win"] = work["panel_winner_condition"].eq(left_condition).astype(float)

    arrays: dict[tuple[str, str], np.ndarray] = {}
    for key, group in work.groupby(["actor", "task"], sort=True):
        arrays[(str(key[0]), str(key[1]))] = group["target_win"].to_numpy(dtype=float)

    point = float(np.mean([array.mean() for array in arrays.values()]))
    rng = np.random.default_rng(seed)
    actors = sorted(work["actor"].dropna().astype(str).unique().tolist())
    tasks = [task] if task is not None else sorted(work["task"].dropna().astype(str).unique().tolist())
    estimates = np.empty(iterations, dtype=float)

    for index in range(iterations):
        sampled_actors = rng.choice(actors, size=len(actors), replace=True) if len(actors) > 1 else np.asarray(actors)
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True) if task is None and len(tasks) > 1 else np.asarray(tasks)
        cell_rates: list[float] = []
        for actor in sampled_actors:
            for task_key in sampled_tasks:
                array = arrays.get((str(actor), str(task_key)))
                if array is None or len(array) == 0:
                    continue
                sampled = rng.choice(array, size=len(array), replace=True)
                cell_rates.append(float(sampled.mean()))
        estimates[index] = float(np.mean(cell_rates)) if cell_rates else math.nan

    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) == 0:
        return point, math.nan, math.nan, "bootstrap failed"
    if task is None:
        method = (
            f"equal actor-task-cell mean; crossed bootstrap over actors and tasks; "
            f"raw non-tied trials resampled within cells; B={iterations}"
        )
    else:
        method = (
            f"equal actor-cell mean within fixed task; bootstrap over actors; "
            f"raw non-tied trials resampled within actor-task cells; B={iterations}"
        )
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return point, float(lo), float(hi), method


def aggregate_rows(
    pairs: pd.DataFrame,
    *,
    task_order: list[tuple[str, str]],
    left_condition: str,
    right_condition: str,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, str | None]] = [("overall", "All tasks", None)]
    scopes.extend(("task", label, task) for task, label in task_order)

    for offset, (scope, label, task) in enumerate(scopes):
        sub = pairs[pairs["complete"]].copy()
        if task is not None:
            sub = sub[sub["task"].eq(task)].copy()
        resolved = sub[sub["panel_winner_condition"].isin([left_condition, right_condition])].copy()
        left_wins = int(resolved["panel_winner_condition"].eq(left_condition).sum())
        right_wins = int(resolved["panel_winner_condition"].eq(right_condition).sum())
        ties = int(sub["panel_winner_condition"].eq("tie").sum())
        unresolved = int(sub["panel_winner_condition"].eq("unresolved").sum())
        n = left_wins + right_wins
        pooled = left_wins / n if n else math.nan
        pooled_lo, pooled_hi = wilson_ci(left_wins, n)
        mean, ci_lo, ci_hi, method = bootstrap_equal_cell_mean(
            sub,
            task=task,
            left_condition=left_condition,
            right_condition=right_condition,
            iterations=bootstrap_iterations,
            seed=seed + offset,
        )
        rows.append(
            {
                "scope": scope,
                "task": task or "all",
                "task_label": label,
                "left_wins": left_wins,
                "right_wins": right_wins,
                "ties": ties,
                "unresolved": unresolved,
                "n_excluding_ties": n,
                "pooled_win_rate": pooled,
                "pooled_wilson_ci_lo": pooled_lo,
                "pooled_wilson_ci_hi": pooled_hi,
                "equal_cell_mean_win_rate": mean,
                "equal_cell_bootstrap_ci_lo": ci_lo,
                "equal_cell_bootstrap_ci_hi": ci_hi,
                "ci_method": method,
            }
        )
    return pd.DataFrame(rows)


def strip_spines(ax, keep=()) -> None:
    for spine in ("top", "right", "left", "bottom"):
        if spine in keep:
            ax.spines[spine].set_color("#9CA3AF")
            ax.spines[spine].set_linewidth(0.7)
        else:
            ax.spines[spine].set_visible(False)


def render_panel(
    ax,
    cells: pd.DataFrame,
    *,
    actor_order: list[tuple[str, str]],
    task: str,
    task_label: str,
    panel_letter: str,
    family_actor_count: int,
    x_label: str,
    show_xlabel: bool,
) -> None:
    ax.set_facecolor(PANEL_BG)
    n = len(actor_order)
    task_cells = cells[cells["task"].eq(task)].set_index("actor") if not cells.empty else pd.DataFrame()

    for i, (actor, actor_label) in enumerate(actor_order):
        y = n - 1 - i
        if actor in task_cells.index and bool(task_cells.loc[actor, "familywise_ci_positive"]):
            ax.add_patch(mpatches.Rectangle((0.005, y - 0.42), 0.99, 0.84, facecolor=CI_POS_BAND, edgecolor="none", zorder=0))

    ax.axvline(0.50, color=CHANCE_LINE, lw=1.2, ls=(0, (4, 3)), alpha=0.9, zorder=1)

    for i, (actor, actor_label) in enumerate(actor_order):
        y = n - 1 - i
        color = MODEL_COLORS.get(actor_label, "#5B6068")
        if actor not in task_cells.index:
            ax.text(0.52, y, "missing", ha="left", va="center", fontsize=9.5, color=MISSING_INK)
            continue
        row = task_cells.loc[actor]
        rate = float(row["left_win_rate_excluding_ties"])
        lo = float(row["familywise_ci_lo"])
        hi = float(row["familywise_ci_hi"])
        if not math.isfinite(rate) or not math.isfinite(lo) or not math.isfinite(hi):
            ax.text(0.52, y, "unresolved", ha="left", va="center", fontsize=9.5, color=MISSING_INK)
            continue
        adjusted_pos = bool(row["familywise_ci_positive"])
        ax.hlines(y, lo, hi, color=color, lw=4.0, alpha=0.42 if adjusted_pos else 0.30, capstyle="round", zorder=2)
        for x in (lo, hi):
            ax.vlines(x, y - 0.16, y + 0.16, color=color, lw=1.3, alpha=0.55 if adjusted_pos else 0.40, zorder=2)
        ax.scatter(rate, y, s=145, color=color, edgecolor="white", linewidth=1.6, zorder=4)
        ax.text(min(hi + 0.025, 1.02), y, f"{rate:.2f}", ha="left", va="center", fontsize=10.0, color=color, fontweight="semibold", zorder=5)

    actor_labels = [label for _, label in actor_order]
    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels(actor_labels, fontsize=10.5)
    for tick, actor_label in zip(ax.get_yticklabels(), actor_labels):
        tick.set_color(MODEL_COLORS.get(actor_label, "#5B6068"))
        tick.set_fontweight("semibold")

    ax.set_xlim(0.005, 1.06)
    ax.set_ylim(-0.55, n - 0.10)
    ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel(x_label if show_xlabel else "", color=INK, labelpad=4, fontsize=11)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    strip_spines(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=SUBTLE, length=3, width=0.6)

    ax.text(-0.005, 1.20, task_label, transform=ax.transAxes, ha="left", va="top", fontsize=13, color=INK, fontweight="bold")

    present = cells[cells["task"].eq(task)] if not cells.empty else pd.DataFrame()
    n_pos = int(present["familywise_ci_positive"].sum()) if not present.empty else 0
    chip_bg = CI_POS_PILL_BG if n_pos > 0 else NEUTRAL_PILL_BG
    chip_fg = CI_POS_PILL_INK if n_pos > 0 else NEUTRAL_PILL_INK
    ax.text(
        1.0,
        1.20,
        f"{n_pos} / {family_actor_count} adj-CI > 0.5",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color=chip_fg,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.30", facecolor=chip_bg, edgecolor="none"),
    )

    observed = present[present["n_excluding_ties"].gt(0)] if not present.empty else pd.DataFrame()
    if observed.empty:
        note = "no complete cells; FWER-adjusted 95% CIs"
    else:
        n_lo = int(observed["n_excluding_ties"].min())
        n_hi = int(observed["n_excluding_ties"].max())
        note = f"n = {n_lo} pairs / actor" if n_lo == n_hi else f"n = {n_lo}-{n_hi} pairs / actor"
        missing = family_actor_count - len(present)
        if missing:
            note += f"; {missing} missing"
        note += "; FWER-adjusted 95% CIs"
    ax.text(-0.005, 1.06, note, transform=ax.transAxes, ha="left", va="top", fontsize=9.0, color=SUBTLE)
    ax.text(-0.30, 1.18, panel_letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, va="top", ha="left")


def plot(cells: pd.DataFrame, *, actor_order: list[tuple[str, str]], task_order: list[tuple[str, str]], x_label: str, png: Path, paper_png: Path | None) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if paper_png is not None:
        paper_png.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 200})
    n_tasks = len(task_order)
    n_cols = 2 if n_tasks > 1 else 1
    n_rows = math.ceil(n_tasks / n_cols)
    fig = plt.figure(figsize=(12.8, 3.65 * n_rows), facecolor=PAPER_BG)
    gs = fig.add_gridspec(n_rows, n_cols, top=0.90, bottom=0.12, left=0.10, right=0.985, hspace=0.55, wspace=0.55)
    for index, (task, task_label) in enumerate(task_order):
        ax = fig.add_subplot(gs[index // n_cols, index % n_cols])
        render_panel(
            ax,
            cells,
            actor_order=actor_order,
            task=task,
            task_label=task_label,
            panel_letter=chr(ord("A") + index),
            family_actor_count=len(actor_order),
            x_label=x_label,
            show_xlabel=index // n_cols == n_rows - 1,
        )
    for index in range(n_tasks, n_rows * n_cols):
        ax = fig.add_subplot(gs[index // n_cols, index % n_cols])
        ax.axis("off")
    fig.savefig(png, dpi=360, bbox_inches="tight", facecolor=PAPER_BG)
    if paper_png is not None:
        fig.savefig(paper_png, dpi=360, bbox_inches="tight", facecolor=PAPER_BG)
    plt.close(fig)


def write_outputs(
    *,
    cells: pd.DataFrame,
    pairs: pd.DataFrame,
    aggregate: pd.DataFrame,
    run_audit: pd.DataFrame,
    output_stem: str,
    comparison: str,
    left_condition: str,
    right_condition: str,
    left_label: str,
    right_label: str,
    family_size: int,
) -> tuple[Path, Path, Path]:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    plot_data = ANALYSIS / f"{output_stem}_model_task_plot_data.csv"
    pair_data = ANALYSIS / f"{output_stem}_pair_outcomes.csv"
    aggregate_data = ANALYSIS / f"{output_stem}_aggregate.csv"
    aggregate_md = ANALYSIS / f"{output_stem}_aggregate.md"
    audit_data = ANALYSIS / f"{output_stem}_run_audit.csv"
    summary = ANALYSIS / f"{output_stem}_model_task_summary.md"
    cells.to_csv(plot_data, index=False)
    pairs.to_csv(pair_data, index=False)
    aggregate.to_csv(aggregate_data, index=False)
    run_audit.to_csv(audit_data, index=False)
    observed = cells[cells["n_excluding_ties"].gt(0)] if not cells.empty else cells
    lines = [
        f"# {left_label} Versus {right_label}",
        "",
        f"Comparison: `{comparison}`.",
        f"Left condition: `{left_condition}`. Right condition: `{right_condition}`.",
        "",
        f"Model-task CIs use exact binomial intervals Bonferroni-adjusted over {family_size} plotted/planned cells. Ties are excluded from displayed win rates.",
        "",
        f"Selected model-task run directories: {len(cells)} / {family_size}.",
        f"Analyzable model-task cells with non-tied outcomes: {int(cells['n_excluding_ties'].gt(0).sum()) if not cells.empty else 0} / {family_size}.",
        f"Total completed pairs: {int(cells['complete_pairs'].sum()) if not cells.empty else 0}.",
        f"Total non-tied pairs: {int(cells['n_excluding_ties'].sum()) if not cells.empty else 0}.",
        "",
    ]
    if not observed.empty:
        lines.extend(
            [
                f"Equal-cell mean win rate: {observed['left_win_rate_excluding_ties'].mean():.3f}.",
                f"FWER-positive cells: {int(observed['familywise_ci_positive'].sum())} / {family_size}.",
                "",
            ]
        )
    show_cols = [
        "task_label",
        "actor_label",
        "left_wins",
        "right_wins",
        "ties",
        "unresolved",
        "n_excluding_ties",
        "left_win_rate_excluding_ties",
        "familywise_ci_lo",
        "familywise_ci_hi",
        "familywise_ci_positive",
        "holm_p_two_sided",
    ]
    lines.extend(["## Model-Task Cells", "", markdown_table(cells[show_cols].sort_values(["task_label", "actor_label"]) if not cells.empty else pd.DataFrame()), ""])
    aggregate_show = aggregate.copy()
    if not aggregate_show.empty:
        aggregate_show["pooled"] = [
            f"{pct(row.pooled_win_rate)} [{pct(row.pooled_wilson_ci_lo)}, {pct(row.pooled_wilson_ci_hi)}]"
            for row in aggregate_show.itertuples()
        ]
        aggregate_show["equal_cell"] = [
            f"{pct(row.equal_cell_mean_win_rate)} [{pct(row.equal_cell_bootstrap_ci_lo)}, {pct(row.equal_cell_bootstrap_ci_hi)}]"
            for row in aggregate_show.itertuples()
        ]
        aggregate_show = aggregate_show[
            [
                "scope",
                "task_label",
                "left_wins",
                "right_wins",
                "ties",
                "n_excluding_ties",
                "pooled",
                "equal_cell",
            ]
        ]
    lines.extend(["## Aggregate and Per-Task Means", "", markdown_table(aggregate_show), ""])
    lines.extend(["## Run Audit", "", markdown_table(run_audit.sort_values(["task", "actor"]) if not run_audit.empty else run_audit), ""])
    summary.write_text("\n".join(lines), encoding="utf-8")
    aggregate_lines = [
        f"# {left_label} Versus {right_label}: Aggregate Analysis",
        "",
        "Primary estimate: equal-cell mean win rate. For the overall row, cells are actor x task. For task rows, cells are actors within that fixed task.",
        "CIs are nonparametric bootstraps over the relevant design cells with non-tied raw trials resampled within cells. Pooled Wilson intervals are included only as descriptive checks.",
        "",
        markdown_table(aggregate_show),
        "",
    ]
    aggregate_md.write_text("\n".join(aggregate_lines), encoding="utf-8")
    return plot_data, pair_data, aggregate_data, aggregate_md, summary


def subset_pairs(all_pairs: list[tuple[str, str]], wanted: list[str]) -> list[tuple[str, str]]:
    if not wanted:
        return all_pairs
    wanted_set = set(wanted)
    return [item for item in all_pairs if item[0] in wanted_set or item[1] in wanted_set]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--left-condition", required=True)
    parser.add_argument("--right-condition", required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--x-label", default="")
    parser.add_argument("--actor", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--family-size", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--paper-copy", action="store_true")
    parser.add_argument(
        "--combine-runs",
        action="store_true",
        help="combine all complete matching run blocks per actor-task instead of selecting one best run",
    )
    args = parser.parse_args()

    actor_order = subset_pairs(DEFAULT_ACTORS, args.actor)
    task_order = subset_pairs(DEFAULT_TASKS, args.task)
    if not actor_order:
        raise SystemExit("no actors selected")
    if not task_order:
        raise SystemExit("no tasks selected")
    family_size = args.family_size or (len(actor_order) * len(task_order))

    selected, run_audit = select_runs(
        comparison=args.comparison,
        explicit_run_dirs=args.run_dir,
        actor_order=actor_order,
        task_order=task_order,
        left_condition=args.left_condition,
        right_condition=args.right_condition,
        combine_runs=args.combine_runs,
    )
    cells, pairs = cells_from_runs(
        selected,
        actor_order=actor_order,
        task_order=task_order,
        left_condition=args.left_condition,
        right_condition=args.right_condition,
        family_size=family_size,
    )
    aggregate = aggregate_rows(
        pairs,
        task_order=task_order,
        left_condition=args.left_condition,
        right_condition=args.right_condition,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )

    plot_data, pair_data, aggregate_data, aggregate_md, summary = write_outputs(
        cells=cells,
        pairs=pairs,
        aggregate=aggregate,
        run_audit=run_audit,
        output_stem=args.output_stem,
        comparison=args.comparison,
        left_condition=args.left_condition,
        right_condition=args.right_condition,
        left_label=args.left_label,
        right_label=args.right_label,
        family_size=family_size,
    )
    png = FIGURES / f"{args.output_stem}_model_task_lollipop.png"
    paper_png = (ROOT / "CURRENT_PAPER" / "figures" / f"{args.output_stem}_model_task_lollipop.png") if args.paper_copy else None
    x_label = args.x_label or f"{args.left_label} win rate (ties excluded)"
    plot(cells, actor_order=actor_order, task_order=task_order, x_label=x_label, png=png, paper_png=paper_png)

    print(f"plot data: {plot_data}")
    print(f"pair outcomes: {pair_data}")
    print(f"aggregate: {aggregate_data}")
    print(f"aggregate summary: {aggregate_md}")
    print(f"summary: {summary}")
    print(f"figure: {png}")
    if paper_png is not None:
        print(f"paper figure: {paper_png}")
    analyzable = int(cells["n_excluding_ties"].gt(0).sum()) if not cells.empty else 0
    print(f"selected cells: {len(cells)}/{family_size}")
    print(f"analyzable cells: {analyzable}/{family_size}")
    if len(cells) < family_size:
        missing = [
            f"{actor_label} / {task_label}"
            for actor, actor_label in actor_order
            for task, task_label in task_order
            if not ((cells["actor"].eq(actor)) & (cells["task"].eq(task))).any()
        ]
        print("missing cells: " + "; ".join(missing))
    incomplete = cells[cells["n_excluding_ties"].le(0)] if not cells.empty else pd.DataFrame()
    if not incomplete.empty:
        print(
            "incomplete/unresolved cells: "
            + "; ".join(f"{row.actor_label} / {row.task_label}" for row in incomplete.itertuples())
        )


if __name__ == "__main__":
    main()
