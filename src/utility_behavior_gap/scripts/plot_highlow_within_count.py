#!/usr/bin/env python3
"""Within-N high-low utility figure - same-count cross-entity essay pairs.

3 panels (Religion / Animals / Countries) x 7 actor rows. Lollipop + Wilson
95% CI in the same visual grammar as plot_highlow_main.py. The pair design
isolates entity preference from numerosity: both sides save the same N, only
the entity differs (e.g., "300 Hindu saved" vs "300 Christian saved"). Policy
domain has no count structure so it is excluded.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from utility_behavior_gap.paths import FIGURES, PROCESSED

DATA_DIR = PROCESSED
OUT_DIR = FIGURES
CSV_PATH = DATA_DIR / "highlow_within_count_data.csv"
PNG_PATH = OUT_DIR / "highlow_within_count.png"
PDF_PATH = OUT_DIR / "highlow_within_count.pdf"


INK = "#1A1A1F"
SUBTLE = "#6B7280"
GRID = "#E9EDF5"
PANEL_BG = "#FFFFFF"
PAPER_BG = "#FFFFFF"

MODEL_COLORS = {
    "DeepSeek V3.2": "#2A8C9E", "GPT-5.4 mini": "#3A66C9",
    "GLM-5.1": "#5B6068", "Kimi K2.5": "#D4711B",
    "MiMo V2 Pro": "#2E8C5C", "Qwen3.5 9B": "#6E45BD",
    "Qwen3.6 Plus": "#C2304A",
}
ACTOR_ORDER = ["DeepSeek V3.2", "GPT-5.4 mini", "GLM-5.1", "Kimi K2.5",
               "MiMo V2 Pro", "Qwen3.5 9B", "Qwen3.6 Plus"]

CHANCE_LINE = "#9CA3AF"
CI_POS_BAND = "#E8F4ED"
CI_POS_PILL_BG = "#DDF1E3"
CI_POS_PILL_INK = "#2F7A4F"
NEUTRAL_PILL_BG = "#F1F2F5"
NEUTRAL_PILL_INK = "#4B5563"

DOMAIN_ORDER = ["Religion", "Animals", "Countries"]


def _strip_spines(ax, keep=()):
    for s in ("top", "right", "left", "bottom"):
        if s in keep:
            ax.spines[s].set_color("#9CA3AF")
            ax.spines[s].set_linewidth(0.7)
        else:
            ax.spines[s].set_visible(False)


def render_panel(ax, df_dom: pd.DataFrame, domain: str, panel_letter: str,
                 *, show_xlabel: bool) -> None:
    ax.set_facecolor(PANEL_BG)
    actors = [a for a in ACTOR_ORDER if a in set(df_dom["actor"])]
    df_dom = df_dom.set_index("actor").loc[actors].reset_index()
    n = len(actors)

    for i, row in df_dom.iterrows():
        y = n - 1 - i
        if str(row["ci_positive"]).strip().lower() == "yes":
            ax.add_patch(mpatches.Rectangle(
                (0.005, y - 0.42), 0.99, 0.84,
                facecolor=CI_POS_BAND, edgecolor="none", zorder=0,
            ))

    ax.axvline(0.50, color=CHANCE_LINE, lw=1.2, ls=(0, (4, 3)), alpha=0.9, zorder=1)

    for i, row in df_dom.iterrows():
        y = n - 1 - i
        actor = row["actor"]
        c = MODEL_COLORS[actor]
        rate = float(row["high_win_rate"])
        lo = float(row["ci_lo"])
        hi = float(row["ci_hi"])
        ci_pos = str(row["ci_positive"]).strip().lower() == "yes"

        ax.hlines(y, lo, hi, color=c, lw=4.0,
                  alpha=0.42 if ci_pos else 0.30, capstyle="round", zorder=2)
        for x in (lo, hi):
            ax.vlines(x, y - 0.16, y + 0.16, color=c,
                      lw=1.3, alpha=0.55 if ci_pos else 0.40, zorder=2)
        ax.scatter(rate, y, s=145, color=c, edgecolor="white",
                   linewidth=1.6, zorder=4)
        ax.text(min(hi + 0.025, 1.02), y, f"{rate:.2f}",
                ha="left", va="center",
                fontsize=10.0, color=c, fontweight="semibold", zorder=5)

    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels(actors, fontsize=10.5)
    for tick, actor in zip(ax.get_yticklabels(), actors):
        tick.set_color(MODEL_COLORS[actor]); tick.set_fontweight("semibold")

    ax.set_xlim(0.005, 1.06)
    ax.set_ylim(-0.55, n - 0.10)
    ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.tick_params(axis="x", labelsize=10)
    if show_xlabel:
        ax.set_xlabel("High-utility-side win rate (same-count pairs)",
                      color=INK, labelpad=4, fontsize=11)
    else:
        ax.set_xlabel("")

    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    _strip_spines(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=SUBTLE, length=3, width=0.6)

    ax.text(-0.005, 1.20, domain, transform=ax.transAxes,
            ha="left", va="top", fontsize=13, color=INK, fontweight="bold")

    n_pos = int((df_dom["ci_positive"].astype(str).str.lower() == "yes").sum())
    n_total = len(df_dom)
    chip_text = f"{n_pos} / {n_total} CI-positive"
    chip_bg = CI_POS_PILL_BG if n_pos > 0 else NEUTRAL_PILL_BG
    chip_fg = CI_POS_PILL_INK if n_pos > 0 else NEUTRAL_PILL_INK
    ax.text(1.0, 1.20, chip_text, transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10.5, color=chip_fg, fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.30",
                      facecolor=chip_bg, edgecolor="none"))

    n_excl = df_dom["n_excl_tie"].astype(int)
    n_lo, n_hi = int(n_excl.min()), int(n_excl.max())
    n_note = (f"n = {n_lo} pairs / actor"
              if n_lo == n_hi else f"n = {n_lo}-{n_hi} pairs / actor")
    ax.text(-0.005, 1.06, n_note, transform=ax.transAxes,
            ha="left", va="top", fontsize=9.0, color=SUBTLE)

    ax.text(-0.30, 1.18, panel_letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=INK, va="top", ha="left")


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12, "axes.titleweight": "regular",
        "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10.5, "axes.edgecolor": "#9CA3AF",
        "axes.linewidth": 0.6, "figure.dpi": 200,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(16.0, 4.0), facecolor=PAPER_BG)
    gs = fig.add_gridspec(1, 3,
                          top=0.85, bottom=0.18,
                          left=0.06, right=0.985,
                          wspace=0.45)

    panel_letters = ["A", "B", "C"]
    for k, domain in enumerate(DOMAIN_ORDER):
        ax = fig.add_subplot(gs[0, k])
        df_dom = df[df["domain"] == domain]
        if df_dom.empty:
            ax.text(0.5, 0.5, f"no data: {domain}",
                    transform=ax.transAxes, ha="center", va="center")
            continue
        render_panel(ax, df_dom, domain, panel_letters[k], show_xlabel=True)

    fig.savefig(PNG_PATH, dpi=360, bbox_inches="tight", facecolor=PAPER_BG)
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor=PAPER_BG)
    print(f"Wrote {PNG_PATH}\nWrote {PDF_PATH}")


if __name__ == "__main__":
    main()
