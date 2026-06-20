#!/usr/bin/env python3
"""Paper-facing outcome figures and length controls for moral-low bridges.

This script is local-only: it uses already judged bridge outcomes and the
standard text-feature catalog. It never calls model APIs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import beta, binomtest

from utility_behavior_gap.constants import ACTOR_LABEL, ACTORS, TASK_LABEL
from utility_behavior_gap.paths import ANALYSIS


FIGURES = ANALYSIS / "figures"
BY_OUTPUT = ANALYSIS / "final_text_analysis_by_output.csv"
TASK_ORDER = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]

COMPARISONS = {
    "moral_low_vs_r0": {
        "label": "Moral Low Versus R0",
        "baseline_label": "R0",
        "pair_outcomes": ANALYSIS / "moral_r0_bridge__combined_7runs__pair_outcomes.csv",
        "by_actor_task": ANALYSIS / "moral_r0_bridge__combined_7runs__by_actor_task.csv",
        "baseline_win_column": "r0_wins",
        "baseline_output_id_column": "source_r0_output_id",
    },
    "moral_low_vs_framed_neutral": {
        "label": "Moral Low Versus Framed Neutral",
        "baseline_label": "Framed neutral",
        "pair_outcomes": ANALYSIS / "moral_neutral_bridge__combined_7runs__pair_outcomes.csv",
        "by_actor_task": ANALYSIS / "moral_neutral_bridge__combined_7runs__by_actor_task.csv",
        "baseline_win_column": "framed_neutral_wins",
        "baseline_output_id_column": "source_neutral_output_id",
    },
    "moral_low_vs_framed_empty": {
        "label": "Moral Low Versus Framed Empty",
        "baseline_label": "Framed empty",
        "pair_outcomes": ANALYSIS / "moral_empty_bridge__combined_7runs__pair_outcomes.csv",
        "by_actor_task": ANALYSIS / "moral_empty_bridge__combined_7runs__by_actor_task.csv",
        "baseline_win_column": "framed_empty_wins",
        "baseline_output_id_column": "source_empty_output_id",
    },
}

PAPER_BG = "#FFFFFF"
INK = "#1A1A1F"
SUBTLE = "#6B7280"
GRID = "#E9EDF5"
CHANCE_LINE = "#9CA3AF"
CI_LOW_BAND = "#FDE8E8"
CI_HIGH_BAND = "#E8F4ED"
CI_LOW_PILL_BG = "#FDE2E2"
CI_LOW_PILL_INK = "#9F1D1D"
CI_HIGH_PILL_BG = "#DDF1E3"
CI_HIGH_PILL_INK = "#2F7A4F"
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


def exact_familywise_ci(wins: int, total: int, family_size: int = 28, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    tail_alpha = alpha / (2 * family_size)
    lo = 0.0 if wins == 0 else float(beta.ppf(tail_alpha, wins, total - wins + 1))
    hi = 1.0 if wins == total else float(beta.ppf(1 - tail_alpha, wins + 1, total - wins))
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


def build_cells(by_actor_task: pd.DataFrame, *, comparison: str) -> pd.DataFrame:
    cfg = COMPARISONS[comparison]
    baseline_col = cfg["baseline_win_column"]
    rows: list[dict[str, object]] = []
    for actor in ACTORS:
        for task in TASK_ORDER:
            sub = by_actor_task[by_actor_task["actor"].eq(actor) & by_actor_task["task"].eq(task)]
            if sub.empty:
                moral_wins = baseline_wins = ties = unresolved = n_pairs = 0
            else:
                row = sub.iloc[0]
                moral_wins = int(row.get("moral_bad_wins", 0))
                baseline_wins = int(row.get(baseline_col, 0))
                ties = int(row.get("ties", 0))
                unresolved = int(row.get("unresolved", 0))
                n_pairs = int(row.get("n_pairs", moral_wins + baseline_wins + ties + unresolved))
            non_tie = moral_wins + baseline_wins
            lo, hi = exact_familywise_ci(moral_wins, non_tie)
            p_value = binomtest(moral_wins, non_tie, 0.5, alternative="two-sided").pvalue if non_tie else math.nan
            rows.append(
                {
                    "comparison": comparison,
                    "actor": actor,
                    "actor_label": ACTOR_LABEL.get(actor, actor),
                    "task": task,
                    "task_label": TASK_LABEL.get(task, task),
                    "n_pairs": n_pairs,
                    "resolved_n_excluding_ties": non_tie,
                    "moral_low_wins": moral_wins,
                    "baseline_wins": baseline_wins,
                    "ties": ties,
                    "unresolved": unresolved,
                    "moral_low_win_rate_excluding_ties": moral_wins / non_tie if non_tie else math.nan,
                    "familywise_ci_lo": lo,
                    "familywise_ci_hi": hi,
                    "familywise_ci_above": bool(np.isfinite(lo) and lo > 0.5),
                    "familywise_ci_below": bool(np.isfinite(hi) and hi < 0.5),
                    "p_two_sided_exact": p_value,
                }
            )
    out = pd.DataFrame(rows)
    out["holm_p_two_sided"] = holm_adjust(out["p_two_sided_exact"].fillna(1.0).tolist())
    out["holm_above"] = out["moral_low_win_rate_excluding_ties"].gt(0.5) & out["holm_p_two_sided"].lt(0.05)
    out["holm_below"] = out["moral_low_win_rate_excluding_ties"].lt(0.5) & out["holm_p_two_sided"].lt(0.05)
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
        if bool(row.get("familywise_ci_below", False)):
            ax.add_patch(mpatches.Rectangle((0.005, y - 0.42), 0.99, 0.84, facecolor=CI_LOW_BAND, edgecolor="none", zorder=0))
        elif bool(row.get("familywise_ci_above", False)):
            ax.add_patch(mpatches.Rectangle((0.005, y - 0.42), 0.99, 0.84, facecolor=CI_HIGH_BAND, edgecolor="none", zorder=0))
    ax.axvline(0.50, color=CHANCE_LINE, lw=1.2, ls=(0, (4, 3)), alpha=0.9, zorder=1)

    for i, row in df_task.iterrows():
        y = n - 1 - i
        actor_label = row["actor_label"]
        color = MODEL_COLORS.get(actor_label, "#555555")
        rate = float(row["moral_low_win_rate_excluding_ties"]) if np.isfinite(row["moral_low_win_rate_excluding_ties"]) else math.nan
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
    ax.set_xlabel("Moral-low-side win rate (ties excluded)" if show_xlabel else "", color=INK, labelpad=4, fontsize=11)
    ax.text(-0.005, 1.20, task_label, transform=ax.transAxes, ha="left", va="top", fontsize=13, color=INK, fontweight="bold")
    n_below = int(df_task["familywise_ci_below"].sum())
    n_above = int(df_task["familywise_ci_above"].sum())
    if n_below:
        chip_bg, chip_fg, chip_text = CI_LOW_PILL_BG, CI_LOW_PILL_INK, f"{n_below} / {len(df_task)} CI-below"
    elif n_above:
        chip_bg, chip_fg, chip_text = CI_HIGH_PILL_BG, CI_HIGH_PILL_INK, f"{n_above} / {len(df_task)} CI-above"
    else:
        chip_bg, chip_fg, chip_text = NEUTRAL_PILL_BG, NEUTRAL_PILL_INK, f"0 / {len(df_task)} CI-different"
    ax.text(1.0, 1.20, chip_text, transform=ax.transAxes, ha="right", va="top", fontsize=10.2, color=chip_fg, fontweight="semibold", bbox=dict(boxstyle="round,pad=0.30", facecolor=chip_bg, edgecolor="none"))
    n_excl = df_task["resolved_n_excluding_ties"].dropna().astype(int)
    if len(n_excl):
        n_lo, n_hi = int(n_excl.min()), int(n_excl.max())
        note = f"n = {n_lo} pairs / actor" if n_lo == n_hi else f"n = {n_lo}-{n_hi} pairs / actor"
        ax.text(-0.005, 1.06, f"{note}; FWER 95% CIs", transform=ax.transAxes, ha="left", va="top", fontsize=9, color=SUBTLE)
    ax.text(-0.30, 1.18, panel_letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, va="top", ha="left")


def plot_lollipop(cells: pd.DataFrame, path: Path, *, title: str) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 200})
    fig = plt.figure(figsize=(12.8, 7.0), facecolor=PAPER_BG)
    gs = fig.add_gridspec(2, 2, top=0.88, bottom=0.10, left=0.10, right=0.985, hspace=0.55, wspace=0.55)
    fig.suptitle(title, fontsize=14, fontweight="bold", color=INK, y=0.975)
    for idx, task in enumerate(TASK_ORDER):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        render_panel(ax, cells[cells["task"].eq(task)].copy(), TASK_LABEL.get(task, task), "ABCD"[idx], show_xlabel=(idx // 2 == 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def outcomes_with_delta_words(comparison: str, outcomes: pd.DataFrame, catalog_path: Path) -> pd.DataFrame | None:
    cfg = COMPARISONS[comparison]
    baseline_col = str(cfg["baseline_output_id_column"])
    if not catalog_path.exists() or "source_moral_output_id" not in outcomes.columns or baseline_col not in outcomes.columns:
        return None
    catalog = pd.read_csv(catalog_path, usecols=["output_id", "words"], low_memory=False)
    word_by_output = dict(zip(catalog["output_id"].astype(str), pd.to_numeric(catalog["words"], errors="coerce")))
    out = outcomes.copy()
    out["moral_low_words"] = out["source_moral_output_id"].astype(str).map(word_by_output)
    out["baseline_words"] = out[baseline_col].astype(str).map(word_by_output)
    out["delta_words"] = out["moral_low_words"] - out["baseline_words"]
    out = out[pd.notna(out["moral_bad_win_excluding_ties"])].copy()
    out["moral_low_win"] = out["moral_bad_win_excluding_ties"].astype(float)
    out["delta_words"] = pd.to_numeric(out["delta_words"], errors="coerce")
    return out.dropna(subset=["moral_low_win", "delta_words"])


def fixed_effect_length_adjustment(
    df: pd.DataFrame,
    *,
    fixed_effect_cols: list[str],
) -> tuple[float, float, float, int]:
    """Estimate win probability at delta_words=0 with simple fixed effects.

    This is a linear probability model on non-tie panel outcomes. It keeps the
    observed actor/task mix and asks what the moral-low win probability would be
    if moral-low and baseline outputs had equal word counts. Intervals use HC3
    robust standard errors for the average adjusted prediction.
    """
    if df.empty:
        return math.nan, math.nan, math.nan, 0
    y = df["moral_low_win"].astype(float).to_numpy()
    x_parts = [pd.Series(1.0, index=df.index, name="const"), df["delta_words"].astype(float)]
    for col in fixed_effect_cols:
        if col in df.columns and df[col].nunique(dropna=False) > 1:
            x_parts.append(pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=float))
    x_df = pd.concat(x_parts, axis=1)
    x_df = x_df.loc[:, ~x_df.columns.duplicated()].astype(float)
    model = sm.OLS(y, x_df.to_numpy())
    result = model.fit().get_robustcov_results(cov_type="HC3")
    x_zero = x_df.copy()
    x_zero["delta_words"] = 0.0
    contrast = x_zero.to_numpy().mean(axis=0)
    point = float(np.dot(contrast, result.params))
    se = float(np.sqrt(np.dot(contrast, np.dot(result.cov_params(), contrast))))
    lo = point - 1.96 * se
    hi = point + 1.96 * se
    return point, lo, hi, int(len(df))


def length_control_rows(
    comparison: str,
    data: pd.DataFrame | None,
) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    specs: list[tuple[str, dict[str, str], list[str]]] = [("overall", {}, ["actor", "task"])]
    specs += [("task", {"task": task}, ["actor"]) for task in TASK_ORDER]
    specs += [("actor", {"actor": actor}, ["task"]) for actor in ACTORS]
    for idx, (scope, filters, fixed_effects) in enumerate(specs):
        sub = data.copy()
        for key, value in filters.items():
            sub = sub[sub[key].eq(value)]
        point, lo, hi, n = fixed_effect_length_adjustment(
            sub,
            fixed_effect_cols=fixed_effects,
        )
        raw = float(sub["moral_low_win"].mean()) if len(sub) else math.nan
        rows.append(
            {
                "comparison": comparison,
                "scope": scope,
                **filters,
                "n_non_tie_pairs": n,
                "raw_moral_low_win_rate": raw,
                "word_count_adjusted_moral_low_win_rate": point,
                "word_count_adjusted_ci_lo": lo,
                "word_count_adjusted_ci_hi": hi,
                "mean_delta_words_moral_low_minus_baseline": float(sub["delta_words"].mean()) if len(sub) else math.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["actor_label"] = out["actor"].map(ACTOR_LABEL).fillna(out.get("actor", ""))
    out["task_label"] = out["task"].map(TASK_LABEL).fillna(out.get("task", ""))
    return out


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        else:
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else str(value))
    columns = [str(col) for col in display.columns]
    rows = display.astype(str).values.tolist()
    widths = [
        max([len(columns[idx])] + [len(row[idx]) for row in rows])
        for idx in range(len(columns))
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    header = fmt_row(columns)
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([header, sep] + [fmt_row(row) for row in rows])


def summarize_outputs(comparison: str, cells: pd.DataFrame, adjusted: pd.DataFrame, missing_reason: str | None) -> str:
    cfg = COMPARISONS[comparison]
    lines = [f"## {cfg['label']}", ""]
    if missing_reason:
        lines.append(missing_reason)
        lines.append("")
        return "\n".join(lines)
    sig = cells[cells["familywise_ci_above"] | cells["familywise_ci_below"]].copy()
    sig["direction"] = np.where(sig["familywise_ci_below"], "moral-low worse", "moral-low better")
    display = sig[[
        "actor_label",
        "task_label",
        "resolved_n_excluding_ties",
        "moral_low_win_rate_excluding_ties",
        "familywise_ci_lo",
        "familywise_ci_hi",
        "direction",
    ]].rename(
        columns={
            "actor_label": "model",
            "task_label": "task",
            "resolved_n_excluding_ties": "n",
            "moral_low_win_rate_excluding_ties": "win_rate",
            "familywise_ci_lo": "FWER_CI_lo",
            "familywise_ci_hi": "FWER_CI_hi",
        }
    )
    lines.extend(["Significant model-task cells using FWER 95% CIs:", "", markdown_table(display), ""])
    if not adjusted.empty:
        adj = adjusted[adjusted["scope"].eq("overall")][[
            "n_non_tie_pairs",
            "raw_moral_low_win_rate",
            "word_count_adjusted_moral_low_win_rate",
            "word_count_adjusted_ci_lo",
            "word_count_adjusted_ci_hi",
            "mean_delta_words_moral_low_minus_baseline",
        ]]
        lines.extend(["Word-count controlled overall result:", "", markdown_table(adj), ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        action="append",
        choices=sorted(COMPARISONS),
        help="Comparison to analyze. Repeatable. Defaults to all known moral bridge comparisons.",
    )
    parser.add_argument("--feature-catalog", type=Path, default=BY_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparisons = args.comparison or list(COMPARISONS)
    summary_lines = [
        "# Moral-Low Baseline Bridge Figures and Length Controls",
        "",
        "This analysis uses judged bridge outcomes where available. Lollipop figures show moral-low-side win rates with ties excluded. Red-highlighted rows have FWER 95% CIs entirely below 0.5; green-highlighted rows are entirely above 0.5.",
        "",
        "The word-count control is a linear-probability adjustment of panel win/loss outcomes, excluding panel ties, using exact source output IDs to recover word counts. Intervals use HC3 robust standard errors for the average adjusted prediction.",
        "",
    ]
    for comparison in comparisons:
        cfg = COMPARISONS[comparison]
        pair_path = Path(cfg["pair_outcomes"])
        by_actor_task_path = Path(cfg["by_actor_task"])
        missing_reason: str | None = None
        cells = pd.DataFrame()
        adjusted = pd.DataFrame()
        if not pair_path.exists() or not by_actor_task_path.exists():
            missing_reason = f"No judged outcome bridge was found for `{comparison}`. Feature deltas may exist, but a standard win-rate figure and length-controlled outcome regression require pair judgments."
        else:
            outcomes = pd.read_csv(pair_path, low_memory=False)
            by_actor_task = pd.read_csv(by_actor_task_path)
            cells = build_cells(by_actor_task, comparison=comparison)
            cells_path = ANALYSIS / f"{comparison}_model_task_cells.csv"
            cells.to_csv(cells_path, index=False)
            figure_path = FIGURES / f"{comparison}_model_task_lollipop.png"
            plot_lollipop(cells, figure_path, title=cfg["label"])
            length_data = outcomes_with_delta_words(comparison, outcomes, args.feature_catalog)
            adjusted = length_control_rows(comparison, length_data)
            adjusted.to_csv(ANALYSIS / f"{comparison}_word_count_control.csv", index=False)
            sig = cells[cells["familywise_ci_above"] | cells["familywise_ci_below"]].copy()
            sig.to_csv(ANALYSIS / f"{comparison}_significant_model_task_cells.csv", index=False)
            print(f"{comparison}: figure {figure_path}")
            print(f"{comparison}: cells {cells_path}")
            print(f"{comparison}: length control {ANALYSIS / f'{comparison}_word_count_control.csv'}")
        summary_lines.append(summarize_outputs(comparison, cells, adjusted, missing_reason))
    summary_path = ANALYSIS / "moral_baseline_bridge_figures_and_length_controls.md"
    summary_path.write_text("\n".join(summary_lines).strip() + "\n", encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
