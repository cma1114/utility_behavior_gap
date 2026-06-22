#!/usr/bin/env python3
"""Plot the essay-only high-low reasoning-traces lollipop figure.

This uses the completed reasoning-trace model-task summary and writes a fixed
PNG layout. No model/API calls are made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utility_behavior_gap.constants import ACTOR_LABEL
from utility_behavior_gap.paths import ANALYSIS, ROOT


FIGURES = ANALYSIS / "figures"
DEFAULT_INPUT = ANALYSIS / "highlow_reasoning_traces_medium__tasks-essay__7-actors_model_task.csv"
DEFAULT_OUT = FIGURES / "highlow_reasoning_traces_medium_essay_lollipop.png"
DEFAULT_PAPER_OUT = ROOT / "CURRENT_PAPER" / "figures" / "highlow_reasoning_traces_medium_essay_lollipop.png"

MODEL_ORDER = [
    "deepseek-v3.2-or",
    "gpt-5.4-mini-or",
    "glm-5.1-or",
    "kimi-k2.5-or",
    "mimo-v25-pro-or",
    "qwen3.5-9b-or",
    "qwen3.6-plus-or",
]

MODEL_COLORS = {
    "deepseek-v3.2-or": "#2A8C9E",
    "gpt-5.4-mini-or": "#3C66CC",
    "glm-5.1-or": "#5B616B",
    "kimi-k2.5-or": "#D36B13",
    "mimo-v25-pro-or": "#2C8B57",
    "qwen3.5-9b-or": "#6F45C7",
    "qwen3.6-plus-or": "#C2304A",
}


def pct_text(value: float) -> str:
    return f"{value:.2f}"


def load_cells(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["task"].eq("essay")].copy()
    missing = set(MODEL_ORDER) - set(df["actor"].astype(str))
    if missing:
        raise ValueError(f"missing actor rows: {sorted(missing)}")
    df["actor"] = pd.Categorical(df["actor"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values("actor").reset_index(drop=True)
    return df


def plot(df: pd.DataFrame, out: Path, *, title: str) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(10.8, 6.2), facecolor="white")

    y_positions = list(range(len(df)))[::-1]
    ax.set_xlim(0, 1.06)
    ax.set_ylim(-0.55, len(df) - 0.45)
    ax.axvline(0.5, color="#9CA3AF", ls=(0, (4, 3)), lw=1.5, zorder=1)
    for x in (0.0, 0.25, 0.5, 0.75, 1.0):
        ax.axvline(x, color="#E6EAF2", lw=0.9, zorder=0)

    ci_positive = 0
    for y, row in zip(y_positions, df.itertuples(index=False), strict=True):
        color = MODEL_COLORS[str(row.actor)]
        x = float(row.high_win_rate_excluding_ties)
        lo = float(row.fwer95_ci_low)
        hi = float(row.fwer95_ci_high)
        if lo > 0.5:
            ci_positive += 1
        ax.hlines(y, lo, hi, color=color, alpha=0.32, lw=6.2, zorder=2)
        ax.vlines([lo, hi], y - 0.16, y + 0.16, color=color, alpha=0.42, lw=1.6, zorder=2)
        ax.scatter([x], [y], s=150, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(
            min(1.02, hi + 0.025),
            y,
            pct_text(x),
            color=color,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="center",
        )

    labels = [ACTOR_LABEL.get(str(actor), str(actor)) for actor in df["actor"].astype(str)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=14, fontweight="bold")
    for tick, actor in zip(ax.get_yticklabels(), df["actor"].astype(str), strict=True):
        tick.set_color(MODEL_COLORS[str(actor)])
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", labelsize=12, colors="#6B7280")
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([f"{x:.2f}" for x in [0.0, 0.25, 0.5, 0.75, 1.0]])
    ax.set_xlabel("High-utility-side win rate (ties excluded)", fontsize=14, color="#1F2329", labelpad=14)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#A7AFBD")
    ax.spines["bottom"].set_linewidth(1.0)

    n_min = int(df["pairs"].min())
    n_max = int(df["pairs"].max())
    n_note = f"n = {n_min} pairs / actor" if n_min == n_max else f"n = {n_min}-{n_max} pairs / actor"

    fig.text(0.5, 0.94, title, ha="center", va="top", fontsize=22, fontweight="bold", color="#1F2329")
    fig.text(0.18, 0.875, f"{n_note}; FWER 95% CIs", ha="left", va="top", fontsize=14, color="#6B7280")
    fig.text(
        0.92,
        0.875,
        f"{ci_positive} / {len(df)} CI-positive",
        ha="right",
        va="top",
        fontsize=12,
        color="#4B5563",
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#F1F3F7", edgecolor="none"),
    )

    fig.subplots_adjust(left=0.18, right=0.94, top=0.80, bottom=0.16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--title", default="High-Low Utility with Actor Reasoning: Essay Writing")
    parser.add_argument("--copy-current-paper", action="store_true")
    args = parser.parse_args()

    df = load_cells(args.input)
    plot(df, args.out, title=args.title)
    print(f"figure: {args.out}")
    if args.copy_current_paper:
        DEFAULT_PAPER_OUT.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_PAPER_OUT.write_bytes(args.out.read_bytes())
        print(f"paper figure: {DEFAULT_PAPER_OUT}")


if __name__ == "__main__":
    main()
