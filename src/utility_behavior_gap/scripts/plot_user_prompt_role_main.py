#!/usr/bin/env python3
"""Paper-style lollipop figure for user-prompt role-cue runs."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import beta, binomtest

from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition
from utility_behavior_gap.openrouter import judge_model_ids
from utility_behavior_gap.paths import ANALYSIS, FIGURES, ROOT


RUNS = ROOT / "outputs" / "api" / "runs"
PLOT_DATA = ANALYSIS / "user_prompt_role_model_task_plot_data.csv"
SUMMARY = ANALYSIS / "user_prompt_role_model_task_summary.md"
PNG = FIGURES / "user_prompt_role_model_task_lollipop.png"
PAPER_PNG = ROOT / "CURRENT_PAPER" / "figures" / "user_prompt_role_model_task_lollipop.png"

FAMILY_ALPHA = 0.05
PLANNED_FAMILY_SIZE = 28

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

ACTOR_ORDER = [
    ("deepseek-v3.2-or", "DeepSeek V3.2"),
    ("gpt-5.4-mini-or", "GPT-5.4 mini"),
    ("glm-5.1-or", "GLM-5.1"),
    ("kimi-k2.5-or", "Kimi K2.5"),
    ("mimo-v25-pro-or", "MiMo V2.5 Pro"),
    ("qwen3.5-9b-or", "Qwen3.5 9B"),
    ("qwen3.6-plus-or", "Qwen3.6 Plus"),
]

TASK_ORDER = [
    ("essay", "Essay writing"),
    ("grant_proposal_abstract", "Grant abstract"),
    ("incident_postmortem", "Incident postmortem"),
    ("translation", "Translation"),
]


def output_is_valid(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    if row.get("success") is False:
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
        if (
            vote.get("source_output_a_hash"),
            vote.get("source_output_b_hash"),
        ) != expected_hashes:
            continue
        if str(vote.get("judge_model") or "") not in JUDGE_MODELS:
            continue
        out.append(vote)
    return out


JUDGE_MODELS = set(judge_model_ids())


def run_status(run_dir: Path) -> dict[str, Any]:
    jobs = read_jsonl(run_dir / "generation_jobs.jsonl")
    generations = {
        str(row.get("output_id") or ""): row
        for row in read_jsonl(run_dir / "generations.jsonl")
        if row.get("output_id")
    }
    votes_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    votes_path = run_dir / "judge_votes.jsonl"
    if votes_path.exists():
        for vote in read_jsonl(votes_path):
            votes_by_pair[str(vote.get("pair_uid") or "")].append(vote)

    expected_pairs = len(jobs)
    valid_pairs = 0
    complete_pairs = 0
    panel_counts = {"user_strong": 0, "user_normal": 0, "tie": 0, "unresolved": 0}
    pair_rows: list[dict[str, Any]] = []

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
        if panel not in panel_counts:
            panel = "unresolved"
        panel_counts[panel] += 1
        pair_rows.append(
            {
                "run_id": run_dir.name,
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
        "expected_pairs": expected_pairs,
        "valid_pairs": valid_pairs,
        "complete_pairs": complete_pairs,
        "panel_counts": panel_counts,
        "pair_rows": pair_rows,
    }


def candidate_run_dirs() -> list[Path]:
    return sorted(RUNS.glob("*__user_prompt_role__*__hash-*"))


def select_runs() -> tuple[dict[tuple[str, str], Path], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    by_cell: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for run_dir in candidate_run_dirs():
        parts = run_dir.name.split("__")
        if len(parts) < 5:
            continue
        task, comparison, actor = parts[0], parts[1], parts[2]
        if comparison != "user_prompt_role":
            continue
        if task not in {task for task, _ in TASK_ORDER}:
            continue
        if actor not in {actor for actor, _ in ACTOR_ORDER}:
            continue
        if not (run_dir / "generation_jobs.jsonl").exists():
            continue
        if not (run_dir / "generations.jsonl").exists():
            continue
        status = run_status(run_dir)
        row = {
            "actor": actor,
            "task": task,
            "run_dir": str(run_dir),
            "expected_pairs": status["expected_pairs"],
            "valid_pairs": status["valid_pairs"],
            "complete_pairs": status["complete_pairs"],
            **{f"panel_{key}": value for key, value in status["panel_counts"].items()},
        }
        rows.append(row)
        by_cell[(actor, task)].append((run_dir, status))

    selected: dict[tuple[str, str], Path] = {}
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
        selected[cell] = candidates[0][0]
    return selected, pd.DataFrame(rows)


def cells_from_runs(selected: dict[tuple[str, str], Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for (actor, task), run_dir in sorted(selected.items()):
        status = run_status(run_dir)
        pairs = pd.DataFrame(status["pair_rows"])
        pair_rows.extend(status["pair_rows"])
        strong_wins = int(pairs["panel_winner_condition"].eq("user_strong").sum())
        normal_wins = int(pairs["panel_winner_condition"].eq("user_normal").sum())
        ties = int(pairs["panel_winner_condition"].eq("tie").sum())
        unresolved = int(pairs["panel_winner_condition"].eq("unresolved").sum())
        non_ties = strong_wins + normal_wins
        rate = strong_wins / non_ties if non_ties else math.nan
        ci_lo, ci_hi = bonferroni_exact_ci(strong_wins, non_ties, PLANNED_FAMILY_SIZE)
        actor_label = next(label for key, label in ACTOR_ORDER if key == actor)
        task_label = next(label for key, label in TASK_ORDER if key == task)
        p_value = (
            binomtest(strong_wins, non_ties, 0.5, alternative="two-sided").pvalue
            if non_ties
            else math.nan
        )
        cell_rows.append(
            {
                "actor": actor,
                "actor_label": actor_label,
                "task": task,
                "task_label": task_label,
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "n_pairs": int(status["expected_pairs"]),
                "complete_pairs": int(status["complete_pairs"]),
                "strong_wins": strong_wins,
                "normal_wins": normal_wins,
                "ties": ties,
                "unresolved": unresolved,
                "n_excluding_ties": non_ties,
                "strong_win_rate_excluding_ties": rate,
                "familywise_ci_lo": ci_lo,
                "familywise_ci_hi": ci_hi,
                "familywise_ci_positive": bool(math.isfinite(ci_lo) and ci_lo > 0.5),
                "p_two_sided_exact": p_value,
            }
        )

    cells = pd.DataFrame(cell_rows)
    if not cells.empty:
        p_values = cells["p_two_sided_exact"].fillna(1.0).tolist()
        cells["holm_p_two_sided"] = holm_adjust(p_values)
        cells["holm_positive"] = (
            cells["strong_win_rate_excluding_ties"].gt(0.5)
            & cells["holm_p_two_sided"].lt(0.05)
        )
        cells["familywise_ci_method"] = (
            "Bonferroni exact binomial 95% familywise CI across "
            f"{PLANNED_FAMILY_SIZE} planned actor-task cells"
        )
    return cells, pd.DataFrame(pair_rows)


def strip_spines(ax, keep=()) -> None:
    for spine in ("top", "right", "left", "bottom"):
        if spine in keep:
            ax.spines[spine].set_color("#9CA3AF")
            ax.spines[spine].set_linewidth(0.7)
        else:
            ax.spines[spine].set_visible(False)


def render_panel(ax, cells: pd.DataFrame, task: str, task_label: str, panel_letter: str, *, show_xlabel: bool) -> None:
    ax.set_facecolor(PANEL_BG)
    actor_labels = [label for _, label in ACTOR_ORDER]
    n = len(actor_labels)
    task_cells = cells[cells["task"].eq(task)].set_index("actor")

    for i, (actor, actor_label) in enumerate(ACTOR_ORDER):
        y = n - 1 - i
        if actor in task_cells.index and bool(task_cells.loc[actor, "familywise_ci_positive"]):
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

    for i, (actor, actor_label) in enumerate(ACTOR_ORDER):
        y = n - 1 - i
        color = MODEL_COLORS[actor_label]
        if actor not in task_cells.index:
            ax.text(0.52, y, "missing", ha="left", va="center", fontsize=9.5, color=MISSING_INK)
            continue
        row = task_cells.loc[actor]
        rate = float(row["strong_win_rate_excluding_ties"])
        lo = float(row["familywise_ci_lo"])
        hi = float(row["familywise_ci_hi"])
        adjusted_pos = bool(row["familywise_ci_positive"])
        if not math.isfinite(rate) or not math.isfinite(lo) or not math.isfinite(hi):
            ax.text(0.52, y, "unresolved", ha="left", va="center", fontsize=9.5, color=MISSING_INK)
            continue

        ax.hlines(y, lo, hi, color=color, lw=4.0, alpha=0.42 if adjusted_pos else 0.30, capstyle="round", zorder=2)
        for x in (lo, hi):
            ax.vlines(x, y - 0.16, y + 0.16, color=color, lw=1.3, alpha=0.55 if adjusted_pos else 0.40, zorder=2)
        ax.scatter(rate, y, s=145, color=color, edgecolor="white", linewidth=1.6, zorder=4)
        ax.text(min(hi + 0.025, 1.02), y, f"{rate:.2f}", ha="left", va="center", fontsize=10.0, color=color, fontweight="semibold", zorder=5)

    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels(actor_labels, fontsize=10.5)
    for tick, actor_label in zip(ax.get_yticklabels(), actor_labels):
        tick.set_color(MODEL_COLORS[actor_label])
        tick.set_fontweight("semibold")

    ax.set_xlim(0.005, 1.06)
    ax.set_ylim(-0.55, n - 0.10)
    ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("World-class-role-side win rate (ties excluded)" if show_xlabel else "", color=INK, labelpad=4, fontsize=11)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    strip_spines(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=SUBTLE, length=3, width=0.6)

    ax.text(-0.005, 1.20, task_label, transform=ax.transAxes, ha="left", va="top", fontsize=13, color=INK, fontweight="bold")

    present = cells[cells["task"].eq(task)]
    n_pos = int(present["familywise_ci_positive"].sum()) if not present.empty else 0
    chip_bg = CI_POS_PILL_BG if n_pos > 0 else NEUTRAL_PILL_BG
    chip_fg = CI_POS_PILL_INK if n_pos > 0 else NEUTRAL_PILL_INK
    ax.text(
        1.0,
        1.20,
        f"{n_pos} / 7 adj-CI > 0.5",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color=chip_fg,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.30", facecolor=chip_bg, edgecolor="none"),
    )

    observed = present[present["n_excluding_ties"].gt(0)]
    if observed.empty:
        note = "no complete cells; FWER-adjusted 95% CIs"
    else:
        n_lo = int(observed["n_excluding_ties"].min())
        n_hi = int(observed["n_excluding_ties"].max())
        note = f"n = {n_lo} pairs / actor" if n_lo == n_hi else f"n = {n_lo}-{n_hi} pairs / actor"
        missing = 7 - len(present)
        if missing:
            note += f"; {missing} missing"
        note += "; FWER-adjusted 95% CIs"
    ax.text(-0.005, 1.06, note, transform=ax.transAxes, ha="left", va="top", fontsize=9.0, color=SUBTLE)

    ax.text(-0.30, 1.18, panel_letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, va="top", ha="left")


def write_summary(cells: pd.DataFrame, run_audit: pd.DataFrame, pair_rows: pd.DataFrame) -> None:
    observed = cells[cells["n_excluding_ties"].gt(0)].copy()
    lines = [
        "# User-Prompt Role Cue",
        "",
        "Contrast: `user_strong` / world-class role cue versus `user_normal` / skilled role cue.",
        "",
        "Model-task CIs use exact binomial intervals Bonferroni-adjusted over the 28 planned actor-task cells. Ties are excluded from the displayed win-rate denominator.",
        "",
        f"Observed model-task cells: {len(cells)} / {PLANNED_FAMILY_SIZE}.",
        f"Total completed pairs: {int(cells['complete_pairs'].sum()) if not cells.empty else 0}.",
        f"Total non-tied pairs: {int(cells['n_excluding_ties'].sum()) if not cells.empty else 0}.",
        "",
    ]
    if not observed.empty:
        lines.extend(
            [
                f"Equal-cell mean win rate: {observed['strong_win_rate_excluding_ties'].mean():.3f}.",
                f"FWER-positive cells: {int(observed['familywise_ci_positive'].sum())} / {PLANNED_FAMILY_SIZE}.",
                "",
            ]
        )

    missing = [
        {"actor": actor, "actor_label": actor_label, "task": task, "task_label": task_label}
        for actor, actor_label in ACTOR_ORDER
        for task, task_label in TASK_ORDER
        if not ((cells["actor"].eq(actor)) & (cells["task"].eq(task))).any()
    ]
    if missing:
        lines.append("## Missing Planned Cells")
        lines.append("")
        lines.append(markdown_table(pd.DataFrame(missing)[["actor_label", "task_label"]]))
        lines.append("")

    show_cols = [
        "task_label",
        "actor_label",
        "strong_wins",
        "normal_wins",
        "ties",
        "unresolved",
        "n_excluding_ties",
        "strong_win_rate_excluding_ties",
        "familywise_ci_lo",
        "familywise_ci_hi",
        "familywise_ci_positive",
        "holm_p_two_sided",
    ]
    lines.append("## Model-Task Cells")
    lines.append("")
    lines.append(markdown_table(cells[show_cols].sort_values(["task_label", "actor_label"])))
    lines.append("")
    lines.append("## Selected Run Audit")
    lines.append("")
    lines.append(markdown_table(run_audit.sort_values(["task", "actor"])))
    lines.append("")

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    pair_rows.to_csv(ANALYSIS / "user_prompt_role_pair_outcomes.csv", index=False)


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


def plot(cells: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_PNG.parent.mkdir(parents=True, exist_ok=True)
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
        }
    )

    fig = plt.figure(figsize=(12.8, 7.0), facecolor=PAPER_BG)
    gs = fig.add_gridspec(2, 2, top=0.92, bottom=0.10, left=0.10, right=0.985, hspace=0.55, wspace=0.55)
    for index, (task, task_label) in enumerate(TASK_ORDER):
        ax = fig.add_subplot(gs[index // 2, index % 2])
        render_panel(ax, cells, task, task_label, "ABCD"[index], show_xlabel=index // 2 == 1)

    fig.savefig(PNG, dpi=360, bbox_inches="tight", facecolor=PAPER_BG)
    fig.savefig(PAPER_PNG, dpi=360, bbox_inches="tight", facecolor=PAPER_BG)
    plt.close(fig)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    selected, run_audit = select_runs()
    cells, pair_rows = cells_from_runs(selected)
    cells.to_csv(PLOT_DATA, index=False)
    write_summary(cells, run_audit, pair_rows)
    plot(cells)
    print(f"plot data: {PLOT_DATA}")
    print(f"summary: {SUMMARY}")
    print(f"figure: {PNG}")
    print(f"paper figure: {PAPER_PNG}")
    print(f"observed cells: {len(cells)}/{PLANNED_FAMILY_SIZE}")
    if len(cells) < PLANNED_FAMILY_SIZE:
        missing = [
            f"{actor_label} / {task_label}"
            for actor, actor_label in ACTOR_ORDER
            for task, task_label in TASK_ORDER
            if not ((cells["actor"].eq(actor)) & (cells["task"].eq(task))).any()
        ]
        print("missing cells: " + "; ".join(missing))
    if not cells.empty:
        print(
            cells.sort_values(["task", "actor"])[
                [
                    "task_label",
                    "actor_label",
                    "strong_wins",
                    "normal_wins",
                    "ties",
                    "n_excluding_ties",
                    "strong_win_rate_excluding_ties",
                    "familywise_ci_lo",
                    "familywise_ci_hi",
                    "familywise_ci_positive",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
