#!/usr/bin/env python3
"""Paper-style lollipop figure for the canonical high-N moral condition."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utility_behavior_gap.constants import ACTOR_LABEL, ACTORS, TASK_LABEL
from utility_behavior_gap.paths import ROOT


ANALYSIS = ROOT / "outputs" / "analysis"
FIGURES = ANALYSIS / "figures"
CELL_PATH = ANALYSIS / "canonical_highn_condition_results_model_task.csv"
OUT_PREFIX = "canonical_highn_moral"
TASK_ORDER = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]

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
        color = MODEL_COLORS.get(actor_label, "#555555")
        rate = float(row["target_win_rate_excluding_ties"])
        lo = float(row["familywise_ci_lo"])
        hi = float(row["familywise_ci_hi"])
        if not np.isfinite(rate):
            continue
        ax.hlines(y, lo, hi, color=color, lw=4.0, alpha=0.34, capstyle="round", zorder=2)
        ax.vlines([lo, hi], y - 0.16, y + 0.16, color=color, lw=1.2, alpha=0.45, zorder=2)
        ax.scatter(rate, y, s=135, color=color, edgecolor="white", linewidth=1.5, zorder=4)
        ax.text(
            min(hi + 0.025, 1.02),
            y,
            f"{rate:.2f}",
            ha="left",
            va="center",
            fontsize=9.7,
            color=color,
            fontweight="semibold",
        )

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
    ax.set_xlabel("Good-cause-side win rate (ties excluded)" if show_xlabel else "", color=INK, labelpad=4, fontsize=11)
    ax.text(-0.005, 1.20, task_label, transform=ax.transAxes, ha="left", va="top", fontsize=13, color=INK, fontweight="bold")
    n_pos = int(df_task["familywise_ci_positive"].sum())
    chip_bg = CI_POS_PILL_BG if n_pos else NEUTRAL_PILL_BG
    chip_fg = CI_POS_PILL_INK if n_pos else NEUTRAL_PILL_INK
    ax.text(
        1.0,
        1.20,
        f"{n_pos} / {len(df_task)} CI-positive",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.2,
        color=chip_fg,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.30", facecolor=chip_bg, edgecolor="none"),
    )
    n_excl = df_task["resolved_n"].dropna().astype(int)
    if len(n_excl):
        n_lo, n_hi = int(n_excl.min()), int(n_excl.max())
        note = f"n = {n_lo} pairs / actor" if n_lo == n_hi else f"n = {n_lo}-{n_hi} pairs / actor"
        ax.text(-0.005, 1.06, f"{note}; FWER 95% CIs", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=SUBTLE)
    ax.text(-0.30, 1.18, panel_letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, va="top", ha="left")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(CELL_PATH)
    cells = cells[cells["condition"].eq("moral")].copy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 200,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(12.8, 7.0), facecolor=PAPER_BG)
    gs = fig.add_gridspec(2, 2, top=0.88, bottom=0.10, left=0.10, right=0.985, hspace=0.55, wspace=0.55)
    fig.suptitle("Moral: Good Versus Bad Cause", fontsize=14, fontweight="bold", color=INK, y=0.975)
    for idx, task in enumerate(TASK_ORDER):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        render_panel(ax, cells[cells["task"].eq(task)].copy(), TASK_LABEL.get(task, task), "ABCD"[idx], show_xlabel=(idx // 2 == 1))

    path_stem = FIGURES / f"{OUT_PREFIX}_model_task_lollipop"
    fig.savefig(path_stem.with_suffix(".png"), dpi=240)
    fig.savefig(path_stem.with_suffix(".pdf"), facecolor=PAPER_BG)
    plt.close(fig)
    print(path_stem.with_suffix(".png"))
    print(path_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
