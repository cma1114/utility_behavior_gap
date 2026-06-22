#!/usr/bin/env python3
"""Plot utility trend checks as a two-panel win-rate figure.

Left panel: high-low utility gap predicts high-side win rate.
Right panel: absolute high-side utility predicts beating a baseline.

This script reuses the same loaders, binning, and marginal-logistic fits used by
the single-panel trend analyses. No model/API calls are made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utility_behavior_gap.paths import ANALYSIS, ROOT
from utility_behavior_gap.scripts.analyze_canonical_utility_gap_trend import (
    GAP_COLUMNS,
    build_bins as build_gap_bins,
    filter_trials as filter_gap_trials,
    fit_marginal_lines as fit_gap_marginal_lines,
    load_trials as load_gap_trials,
    parse_csv_arg,
)
from utility_behavior_gap.scripts.analyze_high_utility_neutral_trend import (
    BRIDGES,
    UTILITY_COLUMNS,
    build_bins as build_absolute_bins,
    filter_trials as filter_absolute_trials,
    fit_marginal_lines as fit_absolute_marginal_lines,
    load_trials as load_absolute_trials,
)


FIGURES = ANALYSIS / "figures"
DEFAULT_OUT = FIGURES / "utility_trend_two_panel_gap_and_absolute_framed_empty.png"
DEFAULT_PAPER_OUT = ROOT / "CURRENT_PAPER" / "figures" / "utility_trend_two_panel_gap_and_absolute_framed_empty.png"


def logistic_line(result, term: str, xs: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(result.params["Intercept"] + result.params[term] * xs)))


def display_baseline(value: str) -> str:
    return value.replace("-", " ")


def plot_win_rate_panel(
    ax,
    *,
    trials,
    bins,
    x_col: str,
    x_mean_col: str,
    x_label: str,
    title: str,
    panel_label: str,
    fit_result,
    fit_term: str,
) -> None:
    xs = np.linspace(float(trials[x_col].min()), float(trials[x_col].max()), 250)
    ys = logistic_line(fit_result, fit_term, xs)

    ax.axhline(0.5, color="#9CA3AF", ls=(0, (4, 3)), lw=1.1)
    ax.errorbar(
        bins[x_mean_col],
        bins["high_win_rate_excluding_ties"],
        yerr=[
            bins["high_win_rate_excluding_ties"] - bins["high_win_rate_ci_lo"],
            bins["high_win_rate_ci_hi"] - bins["high_win_rate_excluding_ties"],
        ],
        fmt="o",
        ms=5.0,
        color="#2563EB",
        ecolor="#93C5FD",
        elinewidth=1.3,
        capsize=3,
        label="equal-count bins",
    )
    ax.plot(xs, ys, color="#C2304A", lw=2.0, label="marginal logistic fit")
    ax.set_ylim(0.25, 0.75)
    ax.set_xlabel(x_label)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.text(
        -0.12,
        1.04,
        panel_label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", choices=sorted(BRIDGES), default="framed_empty")
    parser.add_argument("--gap-scale", choices=sorted(GAP_COLUMNS), default="actor_z")
    parser.add_argument("--utility-scale", choices=sorted(UTILITY_COLUMNS), default="actor_z")
    parser.add_argument("--actors", default="", help="Comma-separated actor ids to include in both panels.")
    parser.add_argument("--tasks", default="", help="Comma-separated task ids to include in both panels.")
    parser.add_argument("--domains", default="", help="Comma-separated domains to include in both panels.")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--copy-current-paper", action="store_true")
    args = parser.parse_args()

    actors = parse_csv_arg(args.actors)
    tasks = parse_csv_arg(args.tasks)
    domains = parse_csv_arg(args.domains)

    gap_col = GAP_COLUMNS[args.gap_scale]
    gap_trials = filter_gap_trials(load_gap_trials(), actors=actors, tasks=tasks, domains=domains)
    gap_bins = build_gap_bins(gap_trials, gap_col, args.bins, args.seed)
    gap_logit, _ = fit_gap_marginal_lines(gap_trials, gap_col)

    bridge = BRIDGES[args.bridge]
    utility_col = UTILITY_COLUMNS[args.utility_scale]
    absolute_trials = filter_absolute_trials(
        load_absolute_trials(
            bridge["pair_outcomes"],
            outcome_col=bridge["outcome_col"],
            loss_value=bridge["loss_value"],
        ),
        actors=actors,
        tasks=tasks,
        domains=domains,
    )
    absolute_bins = build_absolute_bins(absolute_trials, utility_col, args.bins, args.seed)
    absolute_logit, _ = fit_absolute_marginal_lines(absolute_trials, utility_col)

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 200, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), facecolor="white", sharey=True)
    fig.suptitle("Utility Magnitude Does Not Reliably Predict Output Quality", fontsize=13, fontweight="bold")

    plot_win_rate_panel(
        axes[0],
        trials=gap_trials,
        bins=gap_bins,
        x_col=gap_col,
        x_mean_col="gap_mean",
        x_label="High-low utility gap (within-actor z)",
        title="High utility vs low utility",
        panel_label="A",
        fit_result=gap_logit,
        fit_term="gap_value",
    )
    axes[0].set_ylabel("High-utility side win rate (ties excluded)")
    plot_win_rate_panel(
        axes[1],
        trials=absolute_trials,
        bins=absolute_bins,
        x_col=utility_col,
        x_mean_col="utility_mean",
        x_label="High-side utility (within-actor z)",
        title=f"High utility vs {display_baseline(bridge['baseline'])}",
        panel_label="B",
        fit_result=absolute_logit,
        fit_term="utility_value",
    )
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"figure: {args.out}")
    if args.copy_current_paper:
        DEFAULT_PAPER_OUT.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_PAPER_OUT.write_bytes(args.out.read_bytes())
        print(f"paper figure: {DEFAULT_PAPER_OUT}")


if __name__ == "__main__":
    main()
