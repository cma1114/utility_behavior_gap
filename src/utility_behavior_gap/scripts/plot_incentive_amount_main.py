#!/usr/bin/env python3
"""Amount manipulation ($1M vs $100) forest plot.

This uses the same four-panel lollipop grammar as the current
direct-instruction plot: per-actor win rates with Bonferroni exact binomial
familywise CIs across the 28 actor-task cells, plus Holm-adjusted p-values in
the companion CSV.

Data source: outputs/analysis/modgrid_judged_pairs.csv.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import beta, binomtest

from utility_behavior_gap.constants import ACTOR_LABEL, ACTORS, TASK_LABEL
from utility_behavior_gap.paths import ANALYSIS, FIGURES

OUT_DIR = FIGURES
CSV_PATH = ANALYSIS / "modgrid_judged_pairs.csv"
PLOT_DATA_PATH = ANALYSIS / "incentive_amount_main_plot_data.csv"
PNG_PATH = OUT_DIR / "incentive_amount_main.png"
PDF_PATH = OUT_DIR / "incentive_amount_main.pdf"

FAMILY_ALPHA = 0.05

INK = "#1A1A1F"
SUBTLE = "#6B7280"
GRID = "#E9EDF5"
PANEL_BG = "#FFFFFF"
PAPER_BG = "#FFFFFF"

MODEL_COLORS = {
    "DeepSeek V3.2": "#2A8C9E",
    "GPT-5.4 mini": "#3A66C9",
    "GLM-5.1": "#5B6068",
    "Kimi K2.5": "#D4711B",
    "MiMo V2 Pro": "#2E8C5C",
    "MiMo V2.5 Pro": "#2E8C5C",
    "Qwen3.5 9B": "#6E45BD",
    "Qwen3.6 Plus": "#C2304A",
}

ACTOR_ORDER = [ACTOR_LABEL[actor] for actor in ACTORS]

CHANCE_LINE = "#9CA3AF"
CI_POS_BAND = "#E8F4ED"
CI_POS_PILL_BG = "#DDF1E3"
CI_POS_PILL_INK = "#2F7A4F"
NEUTRAL_PILL_BG = "#F1F2F5"
NEUTRAL_PILL_INK = "#4B5563"

TASK_ORDER = [
    ("essay",                    "Essay writing"),
    ("grant_proposal_abstract",  "Grant abstract"),
    ("incident_postmortem",      "Incident postmortem"),
    ("translation",              "Translation"),
]


def exact_familywise_ci(
    wins: int,
    total: int,
    family_size: int,
    alpha: float = FAMILY_ALPHA,
) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
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


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df[df["comparison"].astype(str).str.endswith("_amount")].copy()
    rows = []
    for (actor, task), sub in df.groupby(["actor", "task"], sort=True):
        target = "amount_high"
        wins = int(sub["panel_winner_condition"].eq(target).sum())
        losses = int(sub["panel_winner_condition"].eq("amount_low").sum())
        ties = int(sub["panel_winner_condition"].eq("tie").sum())
        unresolved = int(sub["panel_winner_condition"].isin(["", "unresolved"]).sum())
        total = wins + losses
        rate = wins / total if total else float("nan")
        p_value = (
            binomtest(wins, total, 0.5, alternative="two-sided").pvalue
            if total
            else float("nan")
        )
        rows.append(
            {
                "actor": actor,
                "actor_label": ACTOR_LABEL.get(actor, actor),
                "task": task,
                "task_label": TASK_LABEL.get(task, task),
                "amount_high_wins": wins,
                "amount_low_wins": losses,
                "ties": ties,
                "unresolved": unresolved,
                "n_excl_tie": total,
                "larger_amount_win_rate_excluding_ties": rate,
                "p_two_sided_exact": p_value,
            }
        )

    out = pd.DataFrame(rows)
    family_size = len(out)
    intervals = [
        exact_familywise_ci(
            int(row["amount_high_wins"]),
            int(row["n_excl_tie"]),
            family_size,
        )
        for _, row in out.iterrows()
    ]
    out["familywise_ci_method"] = (
        f"Bonferroni exact binomial 95% familywise CI across {family_size} actor-task cells"
    )
    out["familywise_ci_lo"] = [lo for lo, _ in intervals]
    out["familywise_ci_hi"] = [hi for _, hi in intervals]
    out["familywise_ci_positive"] = out["familywise_ci_lo"] > 0.50
    out["holm_p_two_sided"] = holm_adjust(out["p_two_sided_exact"].tolist())
    out["holm_positive"] = (
        (out["larger_amount_win_rate_excluding_ties"] > 0.5)
        & (out["holm_p_two_sided"] < 0.05)
    )
    out["_actor_order"] = out["actor_label"].map(
        {label: idx for idx, label in enumerate(ACTOR_ORDER)}
    )
    out["_task_order"] = out["task"].map(
        {task: idx for idx, (task, _) in enumerate(TASK_ORDER)}
    )
    out = out.sort_values(["_task_order", "_actor_order"]).drop(
        columns=["_actor_order", "_task_order"]
    )
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    out.to_csv(PLOT_DATA_PATH, index=False)
    return out


def _strip_spines(ax, keep=()):
    for s in ("top", "right", "left", "bottom"):
        if s in keep:
            ax.spines[s].set_color("#9CA3AF")
            ax.spines[s].set_linewidth(0.7)
        else:
            ax.spines[s].set_visible(False)


def render_panel(ax, df_task: pd.DataFrame, task: str, panel_letter: str,
                 *, show_xlabel: bool) -> None:
    ax.set_facecolor(PANEL_BG)

    actors = [a for a in ACTOR_ORDER if a in set(df_task["actor_label"])]
    df_task = df_task.set_index("actor_label").loc[actors].reset_index()
    n = len(actors)

    for i, row in df_task.iterrows():
        y = n - 1 - i
        if bool(row["familywise_ci_positive"]):
            ax.add_patch(mpatches.Rectangle(
                (0.005, y - 0.42), 0.99, 0.84,
                facecolor=CI_POS_BAND, edgecolor="none", zorder=0,
            ))

    ax.axvline(0.50, color=CHANCE_LINE, lw=1.2, ls=(0, (4, 3)), alpha=0.9, zorder=1)

    for i, row in df_task.iterrows():
        y = n - 1 - i
        actor = row["actor_label"]
        c = MODEL_COLORS[actor]
        rate = float(row["larger_amount_win_rate_excluding_ties"])
        lo = float(row["familywise_ci_lo"])
        hi = float(row["familywise_ci_hi"])
        ci_pos = bool(row["familywise_ci_positive"])

        ax.hlines(y, lo, hi, color=c, lw=4.0,
                  alpha=0.42 if ci_pos else 0.30,
                  capstyle="round", zorder=2)
        for x in (lo, hi):
            ax.vlines(x, y - 0.16, y + 0.16, color=c,
                      lw=1.3, alpha=0.55 if ci_pos else 0.40, zorder=2)
        ax.scatter(rate, y, s=145, color=c, edgecolor="white",
                   linewidth=1.6, zorder=4)
        label_x = min(hi + 0.025, 1.02)
        ax.text(label_x, y, f"{rate:.2f}",
                ha="left", va="center",
                fontsize=10.0, color=c, fontweight="semibold", zorder=5)

    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels(actors, fontsize=10.5)
    for tick, actor in zip(ax.get_yticklabels(), actors):
        tick.set_color(MODEL_COLORS[actor])
        tick.set_fontweight("semibold")

    ax.set_xlim(0.005, 1.06)
    ax.set_ylim(-0.55, n - 0.10)
    ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.tick_params(axis="x", labelsize=10)
    if show_xlabel:
        ax.set_xlabel("Larger-amount-side win rate (ties excluded)",
                      color=INK, labelpad=4, fontsize=11)
    else:
        ax.set_xlabel("")

    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    _strip_spines(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=SUBTLE, length=3, width=0.6)

    ax.text(-0.005, 1.20, task, transform=ax.transAxes,
            ha="left", va="top",
            fontsize=13, color=INK, fontweight="bold")

    n_pos = int(df_task["familywise_ci_positive"].sum())
    n_total = len(df_task)
    chip_text = f"{n_pos} / {n_total} CI-positive"
    chip_bg = CI_POS_PILL_BG if n_pos > 0 else NEUTRAL_PILL_BG
    chip_fg = CI_POS_PILL_INK if n_pos > 0 else NEUTRAL_PILL_INK
    ax.text(1.0, 1.20, chip_text, transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10.5, color=chip_fg, fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.30",
                      facecolor=chip_bg, edgecolor="none"))

    if "n_excl_tie" in df_task.columns and df_task["n_excl_tie"].notna().any():
        n_excl = df_task["n_excl_tie"].dropna().astype(int)
        if len(n_excl) > 0:
            n_lo, n_hi = int(n_excl.min()), int(n_excl.max())
            n_note = (f"n = {n_lo} pairs / actor"
                      if n_lo == n_hi else f"n = {n_lo}-{n_hi} pairs / actor")
            ax.text(-0.005, 1.06, n_note, transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=9.0, color=SUBTLE)

    ax.text(-0.30, 1.18, panel_letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=INK, va="top", ha="left")


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    plt.rcParams.update({
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
    })

    fig = plt.figure(figsize=(12.8, 7.0), facecolor=PAPER_BG)
    gs = fig.add_gridspec(
        2, 2,
        top=0.92, bottom=0.10,
        left=0.10, right=0.985,
        hspace=0.55, wspace=0.55,
    )

    panel_letters = ["A", "B", "C", "D"]
    for k, (task_key, task_label) in enumerate(TASK_ORDER):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        show_xlabel = (k // 2 == 1)
        df_task = df[df["task"] == task_key]
        if df_task.empty:
            ax.text(0.5, 0.5, f"no data: {task_label}",
                    transform=ax.transAxes, ha="center", va="center")
            continue
        render_panel(ax, df_task, task_label, panel_letters[k],
                     show_xlabel=show_xlabel)

    fig.savefig(PNG_PATH, dpi=360, bbox_inches="tight", facecolor=PAPER_BG)
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor=PAPER_BG)
    print(f"Wrote {PLOT_DATA_PATH}")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
