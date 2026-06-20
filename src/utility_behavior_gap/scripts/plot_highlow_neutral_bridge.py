#!/usr/bin/env python3
"""Paper-style figures for high-utility versus framed-neutral bridge judgments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utility_behavior_gap.paths import ROOT
from utility_behavior_gap.scripts.analyze_highlow_neutral_bridge_judging import default_run_dir, load_pair_rows
from utility_behavior_gap.scripts.plot_canonical_highn_highlow import (
    ANALYSIS,
    FIGURES,
    TASK_ORDER,
    DOMAIN_ORDER,
    build_aggregate_rows,
    build_cell_rows,
    pct,
    plot_lollipop,
)


def bridge_high_rows(run_dirs: list[Path]) -> pd.DataFrame:
    pairs = pd.concat([load_pair_rows(run_dir) for run_dir in run_dirs], ignore_index=True)
    high = pairs[pairs["side"].eq("high")].copy()
    if high.empty:
        raise SystemExit(f"No high-side bridge rows found in {run_dir}")
    high["target_win"] = high["outcome_vs_neutral"].eq("side")
    high["target_loss"] = high["outcome_vs_neutral"].eq("neutral")
    high["tie"] = high["outcome_vs_neutral"].eq("tie")
    high["resolved"] = high["target_win"] | high["target_loss"]
    return high


def write_summary_md(
    *,
    run_dirs: list[Path],
    stem: str,
    aggregate: pd.DataFrame,
    figure_paths: list[Path],
) -> Path:
    total = aggregate[aggregate["breakout"].eq("overall")]
    by_task = aggregate[aggregate["breakout"].eq("task")]
    by_domain = aggregate[aggregate["breakout"].eq("domain")]
    lines = [
        "# High Utility Versus Framed Neutral",
        "",
        "Source bridge runs:",
        "",
    ]
    lines.extend([f"- `{run_dir}`" for run_dir in run_dirs])
    lines.extend([
        "",
        "Primary estimate: equal-cell mean over actor x task x domain cells. Panel ties are excluded from the win-rate denominator and reported separately.",
        "",
    ])
    lines.extend([
        "## Overall",
        "",
        "| resolved | high wins | neutral wins | ties | pooled | equal-cell mean | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    row = total.iloc[0]
    lines.append(
        f"| {int(row['resolved_n'])} | {int(row['target_wins'])} | {int(row['target_losses'])} | "
        f"{int(row['ties'])} | {pct(row['pooled_win_rate'])} | {pct(row['equal_cell_mean'])} | "
        f"{pct(row['equal_cell_ci_lo'])}-{pct(row['equal_cell_ci_hi'])} |"
    )
    lines += [
        "",
        "## By Task",
        "",
        "| task | resolved | high wins | neutral wins | ties | equal-cell mean | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_task.sort_values("task").iterrows():
        lines.append(
            f"| {row['task_label']} | {int(row['resolved_n'])} | {int(row['target_wins'])} | "
            f"{int(row['target_losses'])} | {int(row['ties'])} | {pct(row['equal_cell_mean'])} | "
            f"{pct(row['equal_cell_ci_lo'])}-{pct(row['equal_cell_ci_hi'])} |"
        )
    lines += [
        "",
        "## By Domain",
        "",
        "| domain | resolved | high wins | neutral wins | ties | equal-cell mean | 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_domain.sort_values("domain").iterrows():
        lines.append(
            f"| {row['domain']} | {int(row['resolved_n'])} | {int(row['target_wins'])} | "
            f"{int(row['target_losses'])} | {int(row['ties'])} | {pct(row['equal_cell_mean'])} | "
            f"{pct(row['equal_cell_ci_lo'])}-{pct(row['equal_cell_ci_hi'])} |"
        )
    lines += ["", "## Figures", ""]
    for path in figure_paths:
        lines.append(f"- `{path}`")
    path = ANALYSIS / f"{stem}_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Bridge run directory. Repeat for multiple per-actor bridge runs. Default: latest bridge run.",
    )
    args = parser.parse_args()

    run_dirs = args.run_dir or [default_run_dir()]
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = (
        f"highlow_neutral_bridge_high__{run_dirs[0].name}"
        if len(run_dirs) == 1
        else f"highlow_neutral_bridge_high__combined_{len(run_dirs)}runs"
    )

    high = bridge_high_rows(run_dirs)
    high.to_csv(ANALYSIS / f"{stem}_pair_outcomes.csv", index=False)

    aggregate = build_aggregate_rows(high)
    aggregate.to_csv(ANALYSIS / f"{stem}_aggregate.csv", index=False)

    all_cells = build_cell_rows(high)
    all_cells.to_csv(ANALYSIS / f"{stem}_model_task_cells.csv", index=False)
    figure_paths: list[Path] = []
    overall_stem = FIGURES / f"{stem}_model_task_lollipop"
    plot_lollipop(all_cells, overall_stem, title="High Utility Versus Framed Neutral: All Domains")
    figure_paths += [overall_stem.with_suffix(".png"), overall_stem.with_suffix(".pdf")]

    domain_frames = []
    for domain in DOMAIN_ORDER:
        cells = build_cell_rows(high, domain=domain)
        domain_frames.append(cells)
        domain_stem = FIGURES / f"{stem}_{domain}_model_task_lollipop"
        plot_lollipop(cells, domain_stem, title=f"High Utility Versus Framed Neutral: {domain.capitalize()} Domain")
        figure_paths += [domain_stem.with_suffix(".png"), domain_stem.with_suffix(".pdf")]
    pd.concat(domain_frames, ignore_index=True).to_csv(ANALYSIS / f"{stem}_domain_model_task_cells.csv", index=False)

    summary_path = write_summary_md(
        run_dirs=run_dirs,
        stem=stem,
        aggregate=aggregate,
        figure_paths=figure_paths,
    )
    print(f"summary: {summary_path}")
    print(f"aggregate: {ANALYSIS / f'{stem}_aggregate.csv'}")
    print(f"model-task cells: {ANALYSIS / f'{stem}_model_task_cells.csv'}")
    print(f"figures: {overall_stem.with_suffix('.png')} and domain breakouts")


if __name__ == "__main__":
    main()
