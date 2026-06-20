#!/usr/bin/env python3
"""Plot essay effort-dose lollipops for neutral, mid, and strong prompts."""

from __future__ import annotations

import glob
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utility_behavior_gap.constants import ACTOR_LABEL, ACTORS
from utility_behavior_gap.paths import ANALYSIS, ROOT


FIGURES = ANALYSIS / "figures"
SUMMARY_CSV = ANALYSIS / "framed_user_mid_effort_essay_dose_summary.csv"
OUT_DATA = ANALYSIS / "framed_user_effort_dose_lollipop_data.csv"
OUT_COMBINED = FIGURES / "framed_user_effort_dose_lollipops.png"
OUT_COMBINED_PDF = FIGURES / "framed_user_effort_dose_lollipops.pdf"
OUT_MID = FIGURES / "framed_user_mid_vs_neutral_lollipop.png"
OUT_STRONG_MID = FIGURES / "framed_user_strong_vs_mid_lollipop.png"

PAPER_BG = "#FFFFFF"
INK = "#1A1A1F"
SUBTLE = "#6B7280"
GRID = "#E9EDF5"
CHANCE_LINE = "#9CA3AF"
CI_POS_BAND = "#E8F4ED"

MODEL_COLORS = {
    "DeepSeek V3.2": "#2A8C9E",
    "GPT-5.4 mini": "#3A66C9",
    "GLM-5.1": "#5B6068",
    "Kimi K2.5": "#D4711B",
    "MiMo V2.5 Pro": "#2E8C5C",
    "Qwen3.5 9B": "#6E45BD",
    "Qwen3.6 Plus": "#C2304A",
}


def bootstrap_ci(values: pd.Series, *, seed: int, iterations: int = 10000) -> tuple[float, float]:
    vals = values.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        value = float(vals[0])
        return value, value
    rng = np.random.default_rng(seed)
    samples = rng.choice(vals, size=(iterations, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def actor_from_mid_path(path: str) -> str:
    return path.split("__essay__framed_user_mid_effort_essay__")[1].split("__")[0]


def actor_from_strong_path(path: str) -> str:
    return path.split("__essay__framed_user_strong_headroom__")[1].split("__")[0]


def load_pair_rows() -> pd.DataFrame:
    mid_rows: list[pd.DataFrame] = []
    for path in sorted(
        glob.glob(str(ANALYSIS / "framed_user_mid_effort_essay__essay__framed_user_mid_effort_essay__*__pair_outcomes.csv"))
    ):
        df = pd.read_csv(path)
        df["actor"] = actor_from_mid_path(path)
        mid_rows.append(df)
    strong_rows: list[pd.DataFrame] = []
    for path in sorted(
        glob.glob(str(ANALYSIS / "framed_user_strong__essay__framed_user_strong_headroom__*__pair_outcomes.csv"))
    ):
        if "repeat-block" in Path(path).name:
            continue
        df = pd.read_csv(path)
        df["actor"] = actor_from_strong_path(path)
        strong_rows.append(df)

    if not mid_rows or not strong_rows:
        raise FileNotFoundError("missing mid-effort or strong-effort pair outcome CSVs")

    mid = pd.concat(mid_rows, ignore_index=True)
    strong = pd.concat(strong_rows, ignore_index=True)
    merged = mid[
        ["actor", "actor_label", "source_headroom_pair_uid", "item_label", "strong_net_score"]
    ].rename(columns={"strong_net_score": "mid_net_score"})
    merged = merged.merge(
        strong[["actor", "source_headroom_pair_uid", "strong_net_score"]].rename(
            columns={"strong_net_score": "max_net_score"}
        ),
        on=["actor", "source_headroom_pair_uid"],
        how="inner",
        validate="one_to_one",
    )
    merged["strong_minus_mid_net"] = merged["max_net_score"] - merged["mid_net_score"]
    return merged


def build_plot_data() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_CSV)
    mid_summary = summary[summary["contrast"].eq("mid_effort_vs_framed_neutral")].copy()
    mid_summary["actor_label"] = mid_summary["actor"].map(ACTOR_LABEL).fillna(mid_summary["actor"])
    mid_summary = mid_summary.set_index("actor")

    paired = load_pair_rows()
    rows: list[dict[str, object]] = []
    for idx, actor in enumerate(ACTORS):
        if actor not in mid_summary.index:
            continue
        sub = paired[paired["actor"].eq(actor)]
        if len(sub) == 0:
            continue
        mid = mid_summary.loc[actor]
        lo, hi = bootstrap_ci(sub["strong_minus_mid_net"], seed=20260616 + idx)
        rows.append(
            {
                "actor": actor,
                "actor_label": ACTOR_LABEL.get(actor, actor),
                "n_pairs": int(mid["n_pairs"]),
                "n_excluding_ties": int(mid["n_excluding_ties"]),
                "mid_wins": int(mid["treatment_wins"]),
                "neutral_wins": int(mid["neutral_wins"]),
                "mid_ties": int(mid["ties"]),
                "mid_win_rate_excluding_ties": float(mid["win_rate_excluding_ties"]),
                "mid_fwer95_ci_low": float(mid["fwer95_ci_low"]),
                "mid_fwer95_ci_high": float(mid["fwer95_ci_high"]),
                "mid_holm_positive": bool(mid["holm_positive"]),
                "strong_minus_mid_pairs": int(len(sub)),
                "strong_minus_mid_net": float(sub["strong_minus_mid_net"].mean()),
                "strong_minus_mid_ci_low": lo,
                "strong_minus_mid_ci_high": hi,
                "strong_minus_mid_positive_pairs": int(sub["strong_minus_mid_net"].gt(0).sum()),
                "strong_minus_mid_negative_pairs": int(sub["strong_minus_mid_net"].lt(0).sum()),
                "strong_minus_mid_equal_pairs": int(sub["strong_minus_mid_net"].eq(0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DATA, index=False)
    return out


def strip_spines(ax) -> None:
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.spines["bottom"].set_linewidth(0.7)


def render_mid_panel(ax, df: pd.DataFrame) -> None:
    actor_labels = [ACTOR_LABEL[actor] for actor in ACTORS if actor in set(df["actor"])]
    plot_df = df.set_index("actor_label").reindex(actor_labels).reset_index()
    n = len(plot_df)
    ax.set_facecolor(PAPER_BG)
    ax.axvline(0.50, color=CHANCE_LINE, lw=1.2, ls=(0, (4, 3)), alpha=0.9, zorder=1)
    for i, row in plot_df.iterrows():
        y = n - 1 - i
        color = MODEL_COLORS[row["actor_label"]]
        if bool(row["mid_holm_positive"]):
            ax.add_patch(plt.Rectangle((0.005, y - 0.42), 0.99, 0.84, facecolor=CI_POS_BAND, edgecolor="none"))
        lo = float(row["mid_fwer95_ci_low"])
        hi = float(row["mid_fwer95_ci_high"])
        rate = float(row["mid_win_rate_excluding_ties"])
        ax.hlines(y, lo, hi, color=color, lw=4.0, alpha=0.34, capstyle="round", zorder=2)
        ax.vlines([lo, hi], y - 0.16, y + 0.16, color=color, lw=1.2, alpha=0.45, zorder=2)
        ax.scatter(rate, y, s=135, color=color, edgecolor="white", linewidth=1.5, zorder=4)
        ax.text(min(hi + 0.025, 1.02), y, f"{rate:.2f}", ha="left", va="center", fontsize=9.8, color=color, fontweight="semibold")
    ax.set_xlim(0.005, 1.06)
    ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.set_xlabel("Mid-effort win rate vs neutral (ties excluded)", color=INK, labelpad=5)
    ax.set_title("Essay Writing", loc="left", fontsize=16, fontweight="bold", color=INK, pad=26)
    n_pos = int(plot_df["mid_holm_positive"].sum())
    n_total = int(len(plot_df))
    chip_bg = "#DDF1E3" if n_pos else "#F1F2F5"
    chip_fg = "#2F7A4F" if n_pos else "#4B5563"
    ax.text(
        1.0,
        1.16,
        f"{n_pos}/{n_total} adj-CI > 0.5",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.0,
        color=chip_fg,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.30", facecolor=chip_bg, edgecolor="none"),
    )
    judged_n = plot_df["n_pairs"].astype(int)
    non_tie_n = plot_df["n_excluding_ties"].astype(int)
    judged_note = (
        f"{int(judged_n.iloc[0])} judged pairs / actor"
        if int(judged_n.min()) == int(judged_n.max())
        else f"{int(judged_n.min())}-{int(judged_n.max())} judged pairs / actor"
    )
    non_tie_note = (
        f"{int(non_tie_n.iloc[0])} non-tie pairs / actor"
        if int(non_tie_n.min()) == int(non_tie_n.max())
        else f"{int(non_tie_n.min())}-{int(non_tie_n.max())} non-tie pairs / actor"
    )
    ax.text(
        0.0,
        1.04,
        f"{judged_note}; {non_tie_note}; FWER 95% CIs",
        transform=ax.transAxes,
        fontsize=9.2,
        color=SUBTLE,
    )
    set_actor_axis(ax, actor_labels)


def render_strong_mid_panel(ax, df: pd.DataFrame) -> None:
    actor_labels = [ACTOR_LABEL[actor] for actor in ACTORS if actor in set(df["actor"])]
    plot_df = df.set_index("actor_label").reindex(actor_labels).reset_index()
    n = len(plot_df)
    ax.set_facecolor(PAPER_BG)
    ax.axvline(0.0, color=CHANCE_LINE, lw=1.2, ls=(0, (4, 3)), alpha=0.9, zorder=1)
    for i, row in plot_df.iterrows():
        y = n - 1 - i
        color = MODEL_COLORS[row["actor_label"]]
        lo = float(row["strong_minus_mid_ci_low"])
        hi = float(row["strong_minus_mid_ci_high"])
        value = float(row["strong_minus_mid_net"])
        if lo > 0:
            ax.add_patch(plt.Rectangle((-0.11, y - 0.42), 0.75, 0.84, facecolor=CI_POS_BAND, edgecolor="none"))
        ax.hlines(y, lo, hi, color=color, lw=4.0, alpha=0.34, capstyle="round", zorder=2)
        ax.vlines([lo, hi], y - 0.16, y + 0.16, color=color, lw=1.2, alpha=0.45, zorder=2)
        ax.scatter(value, y, s=135, color=color, edgecolor="white", linewidth=1.5, zorder=4)
        ax.text(min(hi + 0.018, 0.62), y, f"{value:+.2f}", ha="left", va="center", fontsize=9.8, color=color, fontweight="semibold")
    ax.set_xlim(-0.11, 0.64)
    ax.set_xticks([-0.10, 0.00, 0.20, 0.40, 0.60])
    ax.set_xlabel("Strong minus mid net panel score", color=INK, labelpad=5)
    ax.set_title("Strong vs Mid", loc="left", fontsize=16, fontweight="bold", color=INK, pad=18)
    ax.text(0.0, 1.02, "Paired difference through common neutral baseline", transform=ax.transAxes, fontsize=10.5, color=SUBTLE)
    set_actor_axis(ax, actor_labels)


def set_actor_axis(ax, actor_labels: list[str]) -> None:
    n = len(actor_labels)
    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels(actor_labels, fontsize=10.2)
    for tick, actor_label in zip(ax.get_yticklabels(), actor_labels):
        tick.set_color(MODEL_COLORS.get(actor_label, INK))
        tick.set_fontweight("semibold")
    ax.set_ylim(-0.55, n - 0.25)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=SUBTLE, length=3, width=0.6)
    strip_spines(ax)


def save_single(df: pd.DataFrame, kind: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 5.2), dpi=220)
    fig.patch.set_facecolor(PAPER_BG)
    if kind == "mid":
        render_mid_panel(ax, df)
    else:
        render_strong_mid_panel(ax, df)
    fig.tight_layout(pad=1.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = build_plot_data()
    save_single(df, "mid", OUT_MID)
    save_single(df, "strong_mid", OUT_STRONG_MID)

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.4), dpi=220)
    fig.patch.set_facecolor(PAPER_BG)
    render_mid_panel(axes[0], df)
    render_strong_mid_panel(axes[1], df)
    fig.suptitle("Essay Effort Dose", fontsize=18, fontweight="bold", color=INK, y=1.03)
    fig.tight_layout(pad=1.2, w_pad=2.2)
    fig.savefig(OUT_COMBINED, bbox_inches="tight")
    fig.savefig(OUT_COMBINED_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"plot data: {OUT_DATA}")
    print(f"combined: {OUT_COMBINED}")
    print(f"mid vs neutral: {OUT_MID}")
    print(f"strong vs mid: {OUT_STRONG_MID}")


if __name__ == "__main__":
    main()
