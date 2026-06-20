#!/usr/bin/env python3
"""Paper-style figures for high-utility versus R0 bridge judgments."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from utility_behavior_gap.constants import ACTORS
from utility_behavior_gap.paths import OUTPUT_API, ROOT
from utility_behavior_gap.scripts.analyze_highlow_r0_bridge_judging import default_run_dir, load_pair_rows
from utility_behavior_gap.scripts.plot_canonical_highn_highlow import (
    ANALYSIS,
    DOMAIN_ORDER,
    FIGURES,
    build_aggregate_rows,
    build_cell_rows,
    pct,
    plot_lollipop,
)

PAPER_READY = ROOT / "outputs" / "paper_ready"
PAPER_READY_FIGURES = PAPER_READY / "figures"
PAPER_READY_RESULTS = PAPER_READY / "results"
CURRENT_PAPER = ROOT / "CURRENT_PAPER"


def high_utility_manifest_run_dirs() -> list[Path]:
    run_dirs: list[Path] = []
    for actor in ACTORS:
        manifest_list = OUTPUT_API / "runs" / f"high_utility_r0_bridge_manifests__{actor}.tsv"
        if not manifest_list.exists():
            raise FileNotFoundError(manifest_list)
        for line in manifest_list.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            _, manifest = line.split("\t")
            run_dirs.append(Path(manifest).parent)
    return run_dirs


def bridge_high_rows(run_dirs: list[Path]) -> pd.DataFrame:
    pairs = pd.concat([load_pair_rows(run_dir) for run_dir in run_dirs], ignore_index=True)
    high = pairs[pairs["side"].eq("high")].copy()
    if high.empty:
        raise SystemExit("No high-side bridge rows found")
    high["target_win"] = high["outcome_vs_r0"].eq("side")
    high["target_loss"] = high["outcome_vs_r0"].eq("r0")
    high["tie"] = high["outcome_vs_r0"].eq("tie")
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
        "# High Utility Versus R0",
        "",
        "Source bridge runs:",
        "",
    ]
    lines.extend([f"- `{run_dir}`" for run_dir in run_dirs])
    lines.extend(
        [
            "",
            "Primary estimate: equal-cell mean over actor x task x domain cells. Panel ties are excluded from the win-rate denominator and reported separately.",
            "",
            "## Overall",
            "",
            "| resolved | high wins | R0 wins | ties | pooled | equal-cell mean | 95% CI |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
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
        "| task | resolved | high wins | R0 wins | ties | equal-cell mean | 95% CI |",
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
        "| domain | resolved | high wins | R0 wins | ties | equal-cell mean | 95% CI |",
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


def copy_paper_ready_outputs(*, stem: str) -> list[Path]:
    """Copy canonical bridge artifacts into stable paper-ready filenames.

    The source files remain in outputs/analysis with run-specific stems. The
    paper-ready copies are stable names, so manuscript references do not drift
    when a rerun creates a new timestamped analysis stem.
    """
    PAPER_READY_FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_READY_RESULTS.mkdir(parents=True, exist_ok=True)

    summary_source = ANALYSIS / f"{stem}_summary.md"
    summary_dest = PAPER_READY_RESULTS / "high_utility_vs_r0_summary.md"
    copies = [
        (ANALYSIS / f"{stem}_aggregate.csv", PAPER_READY_RESULTS / "high_utility_vs_r0_aggregate.csv"),
        (ANALYSIS / f"{stem}_pair_outcomes.csv", PAPER_READY_RESULTS / "high_utility_vs_r0_pair_outcomes.csv"),
        (ANALYSIS / f"{stem}_model_task_cells.csv", PAPER_READY_RESULTS / "high_utility_vs_r0_model_task_cells.csv"),
        (
            ANALYSIS / f"{stem}_domain_model_task_cells.csv",
            PAPER_READY_RESULTS / "high_utility_vs_r0_domain_model_task_cells.csv",
        ),
        (
            FIGURES / f"{stem}_model_task_lollipop.png",
            PAPER_READY_FIGURES / "high_utility_vs_r0_model_task_lollipop.png",
        ),
        (
            FIGURES / f"{stem}_model_task_lollipop.pdf",
            PAPER_READY_FIGURES / "high_utility_vs_r0_model_task_lollipop.pdf",
        ),
    ]
    for domain in DOMAIN_ORDER:
        copies.extend(
            [
                (
                    FIGURES / f"{stem}_{domain}_model_task_lollipop.png",
                    PAPER_READY_FIGURES / f"high_utility_vs_r0_domain_{domain}_model_task_lollipop.png",
                ),
                (
                    FIGURES / f"{stem}_{domain}_model_task_lollipop.pdf",
                    PAPER_READY_FIGURES / f"high_utility_vs_r0_domain_{domain}_model_task_lollipop.pdf",
                ),
            ]
        )

    written: list[Path] = []
    for source, dest in copies:
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, dest)
        written.append(dest)
    if not summary_source.exists():
        raise FileNotFoundError(summary_source)
    figure_paths = [path for path in written if path.parent == PAPER_READY_FIGURES]
    summary = summary_source.read_text(encoding="utf-8")
    if "\n## Figures\n" in summary:
        summary = summary.split("\n## Figures\n", 1)[0].rstrip()
    summary += (
        "\n\n## Inference Convention\n\n"
        "Aggregate rows report equal-cell means with 95% bootstrap CIs. "
        "Model-by-task lollipop figures use the same convention as the canonical "
        "high-low and direct-instruction figures: tie-excluded win rates with "
        "Bonferroni exact binomial FWER 95% CIs across the 28 actor-task cells. "
        "Holm-adjusted p-values are retained in the model-task CSV.\n\n"
        "## Paper-Ready Figures\n\n"
    )
    summary += "\n".join(f"- `{path}`" for path in figure_paths)
    summary += "\n"
    summary_dest.write_text(summary, encoding="utf-8")
    written.insert(0, summary_dest)
    if CURRENT_PAPER.exists():
        for path in list(written):
            rel = path.relative_to(PAPER_READY)
            dest = CURRENT_PAPER / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            written.append(dest)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Bridge run directory. Repeat for multiple per-actor bridge runs. Default: latest bridge run.",
    )
    parser.add_argument(
        "--from-high-utility-manifests",
        action="store_true",
        help="Read the seven high_utility_r0_bridge_manifests__*.tsv files.",
    )
    parser.add_argument(
        "--paper-ready",
        action="store_true",
        help="Also copy stable paper-ready outputs to outputs/paper_ready/{figures,results}.",
    )
    args = parser.parse_args()

    if args.from_high_utility_manifests:
        run_dirs = high_utility_manifest_run_dirs()
    else:
        run_dirs = args.run_dir or [default_run_dir()]

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = (
        f"highlow_r0_bridge_high__{run_dirs[0].name}"
        if len(run_dirs) == 1
        else f"highlow_r0_bridge_high__combined_{len(run_dirs)}runs"
    )

    high = bridge_high_rows(run_dirs)
    high.to_csv(ANALYSIS / f"{stem}_pair_outcomes.csv", index=False)

    aggregate = build_aggregate_rows(high)
    aggregate.to_csv(ANALYSIS / f"{stem}_aggregate.csv", index=False)

    all_cells = build_cell_rows(high)
    all_cells.to_csv(ANALYSIS / f"{stem}_model_task_cells.csv", index=False)
    figure_paths: list[Path] = []
    overall_stem = FIGURES / f"{stem}_model_task_lollipop"
    plot_lollipop(all_cells, overall_stem, title="High Utility Versus R0: All Domains")
    figure_paths += [overall_stem.with_suffix(".png"), overall_stem.with_suffix(".pdf")]

    domain_frames = []
    for domain in DOMAIN_ORDER:
        cells = build_cell_rows(high, domain=domain)
        domain_frames.append(cells)
        domain_stem = FIGURES / f"{stem}_{domain}_model_task_lollipop"
        plot_lollipop(cells, domain_stem, title=f"High Utility Versus R0: {domain.capitalize()} Domain")
        figure_paths += [domain_stem.with_suffix(".png"), domain_stem.with_suffix(".pdf")]
    pd.concat(domain_frames, ignore_index=True).to_csv(
        ANALYSIS / f"{stem}_domain_model_task_cells.csv",
        index=False,
    )

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
    if args.paper_ready:
        written = copy_paper_ready_outputs(stem=stem)
        print("paper-ready outputs:")
        for path in written:
            print(path)


if __name__ == "__main__":
    main()
