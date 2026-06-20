#!/usr/bin/env python3
"""Paper-style lollipop figures for canonical high-N high-low utility results."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

from utility_behavior_gap.constants import ACTOR_LABEL, ACTORS, TASK_LABEL
from utility_behavior_gap.paths import ROOT


ANALYSIS = ROOT / "outputs" / "analysis"
FIGURES = ANALYSIS / "figures"
PAIR_PATH = ANALYSIS / "canonical_highn_condition_results_pair_outcomes.csv"
OUT_PREFIX = "canonical_highn_highlow"

TASK_ORDER = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]
DOMAIN_ORDER = ["animals", "countries", "political", "religions"]

PAPER_BG = "#FFFFFF"
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


def pct(x: float) -> str:
    return "" if not np.isfinite(x) else f"{x:.1%}"


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


def bootstrap_equal_cell(
    resolved: pd.DataFrame,
    cell_cols: list[str],
    *,
    iterations: int = 5000,
    seed: int = 20260615,
) -> tuple[float, float, float, int]:
    if resolved.empty:
        return math.nan, math.nan, math.nan, 0
    arrays = [
        group["target_win"].to_numpy(dtype=float)
        for _, group in resolved.groupby(cell_cols, dropna=False, sort=True)
        if len(group)
    ]
    point = float(np.mean([arr.mean() for arr in arrays]))
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for idx in range(iterations):
        sample = rng.integers(0, len(arrays), size=len(arrays))
        rates = [
            float(rng.choice(arrays[int(cell_idx)], size=len(arrays[int(cell_idx)]), replace=True).mean())
            for cell_idx in sample
        ]
        estimates[idx] = float(np.mean(rates))
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return point, float(lo), float(hi), len(arrays)


def build_aggregate_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [("overall", {})]
    specs += [("task", {"task": task}) for task in TASK_ORDER]
    specs += [("domain", {"domain": domain}) for domain in DOMAIN_ORDER]
    specs += [
        ("task_domain", {"task": task, "domain": domain})
        for task in TASK_ORDER
        for domain in DOMAIN_ORDER
    ]
    for idx, (breakout, filters) in enumerate(specs):
        sub = pairs.copy()
        for key, value in filters.items():
            sub = sub[sub[key].eq(value)]
        resolved = sub[sub["resolved"]].copy()
        wins = int(resolved["target_win"].sum())
        n = int(len(resolved))
        point, lo, hi, cells = bootstrap_equal_cell(
            resolved,
            ["actor", "task", "domain"],
            seed=20260615 + idx,
        )
        rows.append(
            {
                "breakout": breakout,
                **filters,
                "resolved_n": n,
                "target_wins": wins,
                "target_losses": n - wins,
                "ties": int(sub["tie"].sum()),
                "pooled_win_rate": wins / n if n else math.nan,
                "equal_cell_mean": point,
                "equal_cell_ci_lo": lo,
                "equal_cell_ci_hi": hi,
                "bootstrap_cells": cells,
            }
        )
    out = pd.DataFrame(rows)
    out["task_label"] = out["task"].map(TASK_LABEL).fillna(out.get("task", ""))
    return out


def build_cell_rows(pairs: pd.DataFrame, *, domain: str | None = None) -> pd.DataFrame:
    sub = pairs.copy()
    family_size = 28 if domain is None else 28
    if domain is not None:
        sub = sub[sub["domain"].eq(domain)].copy()
    rows = []
    for actor in ACTORS:
        for task in TASK_ORDER:
            cell = sub[sub["actor"].eq(actor) & sub["task"].eq(task)]
            resolved = cell[cell["resolved"]]
            wins = int(resolved["target_win"].sum())
            n = int(len(resolved))
            lo, hi = exact_familywise_ci(wins, n, family_size)
            p_value = binomtest(wins, n, 0.5, alternative="two-sided").pvalue if n else math.nan
            rows.append(
                {
                    "domain": domain or "all",
                    "actor": actor,
                    "actor_label": ACTOR_LABEL.get(actor, actor),
                    "task": task,
                    "task_label": TASK_LABEL.get(task, task),
                    "resolved_n": n,
                    "target_wins": wins,
                    "target_losses": n - wins,
                    "ties": int(cell["tie"].sum()),
                    "target_win_rate_excluding_ties": wins / n if n else math.nan,
                    "familywise_ci_lo": lo,
                    "familywise_ci_hi": hi,
                    "familywise_ci_positive": bool(np.isfinite(lo) and lo > 0.5),
                    "p_two_sided_exact": p_value,
                }
            )
    out = pd.DataFrame(rows)
    adjusted = holm_adjust(out["p_two_sided_exact"].fillna(1.0).tolist())
    out["holm_p_two_sided"] = adjusted
    out["holm_positive"] = out["target_win_rate_excluding_ties"].gt(0.5) & out["holm_p_two_sided"].lt(0.05)
    return out


def strip_spines(ax) -> None:
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.spines["bottom"].set_linewidth(0.7)


def render_panel(ax, df_task: pd.DataFrame, task_label: str, panel_letter: str, *, show_xlabel: bool) -> None:
    actor_labels = [ACTOR_LABEL[actor] for actor in ACTORS]
    df_task = df_task.set_index("actor_label").reindex(actor_labels).reset_index()
    n = len(actor_labels)
    ax.set_facecolor(PAPER_BG)

    for i, row in df_task.iterrows():
        y = n - 1 - i
        if bool(row.get("familywise_ci_positive", False)):
            ax.add_patch(
                mpatches.Rectangle((0.005, y - 0.42), 0.99, 0.84, facecolor=CI_POS_BAND, edgecolor="none", zorder=0)
            )
    ax.axvline(0.50, color=CHANCE_LINE, lw=1.2, ls=(0, (4, 3)), alpha=0.9, zorder=1)

    for i, row in df_task.iterrows():
        y = n - 1 - i
        actor_label = row["actor_label"]
        color = MODEL_COLORS.get(actor_label, "#555555")
        rate = float(row["target_win_rate_excluding_ties"]) if np.isfinite(row["target_win_rate_excluding_ties"]) else math.nan
        lo = float(row["familywise_ci_lo"]) if np.isfinite(row["familywise_ci_lo"]) else math.nan
        hi = float(row["familywise_ci_hi"]) if np.isfinite(row["familywise_ci_hi"]) else math.nan
        if not np.isfinite(rate):
            continue
        ax.hlines(y, lo, hi, color=color, lw=4.0, alpha=0.34, capstyle="round", zorder=2)
        ax.vlines([lo, hi], y - 0.16, y + 0.16, color=color, lw=1.2, alpha=0.45, zorder=2)
        ax.scatter(rate, y, s=135, color=color, edgecolor="white", linewidth=1.5, zorder=4)
        ax.text(min(hi + 0.025, 1.02), y, f"{rate:.2f}", ha="left", va="center", fontsize=9.7, color=color, fontweight="semibold")

    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels(actor_labels, fontsize=10.2)
    for tick, actor_label in zip(ax.get_yticklabels(), actor_labels):
        tick.set_color(MODEL_COLORS.get(actor_label, INK))
        tick.set_fontweight("semibold")
    ax.set_xlim(0.005, 1.06)
    ax.set_ylim(-0.55, n - 0.10)
    ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.tick_params(axis="x", labelsize=10, colors=SUBTLE, length=3, width=0.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    strip_spines(ax)
    ax.set_xlabel("High-utility-side win rate (ties excluded)" if show_xlabel else "", color=INK, labelpad=4, fontsize=11)
    ax.text(-0.005, 1.20, task_label, transform=ax.transAxes, ha="left", va="top", fontsize=13, color=INK, fontweight="bold")
    n_pos = int(df_task["familywise_ci_positive"].sum())
    chip_bg = CI_POS_PILL_BG if n_pos else NEUTRAL_PILL_BG
    chip_fg = CI_POS_PILL_INK if n_pos else NEUTRAL_PILL_INK
    ax.text(1.0, 1.20, f"{n_pos} / {len(df_task)} CI-positive", transform=ax.transAxes, ha="right", va="top", fontsize=10.2, color=chip_fg, fontweight="semibold", bbox=dict(boxstyle="round,pad=0.30", facecolor=chip_bg, edgecolor="none"))
    n_excl = df_task["resolved_n"].dropna().astype(int)
    if len(n_excl):
        n_lo, n_hi = int(n_excl.min()), int(n_excl.max())
        note = f"n = {n_lo} pairs / actor" if n_lo == n_hi else f"n = {n_lo}-{n_hi} pairs / actor"
        ax.text(-0.005, 1.06, f"{note}; FWER 95% CIs", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=SUBTLE)
    ax.text(-0.30, 1.18, panel_letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, va="top", ha="left")


def plot_lollipop(cells: pd.DataFrame, path_stem: Path, *, title: str, write_pdf: bool = True) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(12.8, 7.0), facecolor=PAPER_BG)
    gs = fig.add_gridspec(2, 2, top=0.88, bottom=0.10, left=0.10, right=0.985, hspace=0.55, wspace=0.55)
    fig.suptitle(title, fontsize=14, fontweight="bold", color=INK, y=0.975)
    letters = ["A", "B", "C", "D"]
    for idx, task in enumerate(TASK_ORDER):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        render_panel(ax, cells[cells["task"].eq(task)].copy(), TASK_LABEL.get(task, task), letters[idx], show_xlabel=(idx // 2 == 1))
    fig.savefig(path_stem.with_suffix(".png"), dpi=240)
    if write_pdf:
        fig.savefig(path_stem.with_suffix(".pdf"), facecolor=PAPER_BG)
    plt.close(fig)


def write_summary_md(aggregate: pd.DataFrame, figure_paths: list[Path]) -> None:
    total = aggregate[aggregate["breakout"].eq("overall")]
    by_task = aggregate[aggregate["breakout"].eq("task")]
    by_domain = aggregate[aggregate["breakout"].eq("domain")]
    lines = [
        "# Canonical High-N High-Low Utility",
        "",
        "Source: `canonical_highn_condition_results_pair_outcomes.csv`, condition `utility` only.",
        "This is the fund-intervention wording with blank system prompts: base repeats 0-4 from June 13 plus high-N extension repeats 5-9 from June 15.",
        "",
        "Primary estimate: equal-cell mean over actor x task x domain cells. Panel ties are excluded from the win-rate denominator and reported separately.",
        "",
        "## Overall",
        "",
        "| resolved | high wins | low wins | ties | pooled | equal-cell mean | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    row = total.iloc[0]
    lines.append(
        f"| {int(row['resolved_n'])} | {int(row['target_wins'])} | {int(row['target_losses'])} | {int(row['ties'])} | "
        f"{pct(row['pooled_win_rate'])} | {pct(row['equal_cell_mean'])} | {pct(row['equal_cell_ci_lo'])}-{pct(row['equal_cell_ci_hi'])} |"
    )
    lines += [
        "",
        "## By Task",
        "",
        "| task | resolved | high wins | low wins | ties | equal-cell mean | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_task.sort_values("task").iterrows():
        lines.append(
            f"| {row['task_label']} | {int(row['resolved_n'])} | {int(row['target_wins'])} | {int(row['target_losses'])} | "
            f"{int(row['ties'])} | {pct(row['equal_cell_mean'])} | {pct(row['equal_cell_ci_lo'])}-{pct(row['equal_cell_ci_hi'])} |"
        )
    lines += [
        "",
        "## By Domain",
        "",
        "| domain | resolved | high wins | low wins | ties | equal-cell mean | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_domain.sort_values("domain").iterrows():
        lines.append(
            f"| {row['domain']} | {int(row['resolved_n'])} | {int(row['target_wins'])} | {int(row['target_losses'])} | "
            f"{int(row['ties'])} | {pct(row['equal_cell_mean'])} | {pct(row['equal_cell_ci_lo'])}-{pct(row['equal_cell_ci_hi'])} |"
        )
    lines += ["", "## Figures", ""]
    for path in figure_paths:
        lines.append(f"- `{path}`")
    (ANALYSIS / f"{OUT_PREFIX}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(PAIR_PATH)
    highlow = pairs[pairs["condition"].eq("utility")].copy()
    if highlow.empty:
        raise SystemExit("No high-low rows found. Run analyze_canonical_highn_conditions first.")
    aggregate = build_aggregate_rows(highlow)
    aggregate.to_csv(ANALYSIS / f"{OUT_PREFIX}_aggregate.csv", index=False)

    all_cells = build_cell_rows(highlow)
    all_cells.to_csv(ANALYSIS / f"{OUT_PREFIX}_model_task_cells.csv", index=False)
    figure_paths: list[Path] = []
    overall_stem = FIGURES / f"{OUT_PREFIX}_model_task_lollipop"
    plot_lollipop(all_cells, overall_stem, title="High-Low Utility: All Domains")
    figure_paths += [overall_stem.with_suffix(".png"), overall_stem.with_suffix(".pdf")]

    domain_frames = []
    for domain in DOMAIN_ORDER:
        cells = build_cell_rows(highlow, domain=domain)
        domain_frames.append(cells)
        stem = FIGURES / f"{OUT_PREFIX}_{domain}_model_task_lollipop"
        plot_lollipop(cells, stem, title=f"High-Low Utility: {domain.capitalize()} Domain")
        figure_paths += [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    pd.concat(domain_frames, ignore_index=True).to_csv(ANALYSIS / f"{OUT_PREFIX}_domain_model_task_cells.csv", index=False)
    write_summary_md(aggregate, figure_paths)
    print(f"summary: {ANALYSIS / f'{OUT_PREFIX}_summary.md'}")
    print(f"figures: {FIGURES / f'{OUT_PREFIX}_model_task_lollipop.png'} and domain breakouts")


if __name__ == "__main__":
    main()
