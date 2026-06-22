#!/usr/bin/env python3
"""Paper-style figures for high-utility versus framed-neutral bridge judgments."""

from __future__ import annotations

import argparse
import shutil
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
    exact_familywise_ci,
    pct,
    plot_lollipop,
)

PAPER_READY = ROOT / "outputs" / "paper_ready"
PAPER_READY_FIGURES = PAPER_READY / "figures"
PAPER_READY_RESULTS = PAPER_READY / "results"
CURRENT_PAPER = ROOT / "CURRENT_PAPER"


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


def build_bonferroni_aggregate_rows(high: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs: list[tuple[str, dict[str, str]]] = [("overall", {})]
    specs += [("task", {"task": task}) for task in TASK_ORDER]
    family_size = len(specs)
    for breakout, filters in specs:
        sub = high.copy()
        for key, value in filters.items():
            sub = sub[sub[key].eq(value)]
        resolved = sub[sub["resolved"]].copy()
        wins = int(resolved["target_win"].sum())
        n = int(len(resolved))
        lo, hi = exact_familywise_ci(wins, n, family_size=family_size)
        row = {
            "breakout": breakout,
            **filters,
            "resolved_n": n,
            "high_wins": wins,
            "framed_neutral_wins": n - wins,
            "ties": int(sub["tie"].sum()),
            "high_win_rate_excluding_ties": wins / n if n else float("nan"),
            "bonferroni_ci_lo": lo,
            "bonferroni_ci_hi": hi,
            "bonferroni_family_size": family_size,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out["task_label"] = out["task"].map(TASK_ORDER_LABELS).fillna("Overall") if "task" in out else "Overall"
    return out


TASK_ORDER_LABELS = {
    "essay": "Essay writing",
    "grant_proposal_abstract": "Grant abstract",
    "incident_postmortem": "Incident postmortem",
    "translation": "Translation",
}


def write_bonferroni_aggregate_md(rows: pd.DataFrame, path: Path) -> None:
    lines = [
        "# High Utility Versus Framed Neutral: Aggregate Bonferroni CIs",
        "",
        "Rows report pooled tie-excluded high-utility-side win rates. Confidence intervals are exact binomial intervals with Bonferroni correction over the five aggregate rows shown here: overall plus four tasks.",
        "",
        "| row | resolved | high wins | framed-neutral wins | ties | win rate | Bonferroni 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        label = "Overall" if row["breakout"] == "overall" else str(row["task_label"])
        lines.append(
            f"| {label} | {int(row['resolved_n'])} | {int(row['high_wins'])} | "
            f"{int(row['framed_neutral_wins'])} | {int(row['ties'])} | "
            f"{pct(float(row['high_win_rate_excluding_ties']))} | "
            f"{pct(float(row['bonferroni_ci_lo']))}-{pct(float(row['bonferroni_ci_hi']))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_paper_ready_outputs(*, stem: str) -> list[Path]:
    PAPER_READY_FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_READY_RESULTS.mkdir(parents=True, exist_ok=True)
    copies = [
        (ANALYSIS / f"{stem}_summary.md", PAPER_READY_RESULTS / "high_utility_vs_framed_neutral_summary.md"),
        (ANALYSIS / f"{stem}_aggregate.csv", PAPER_READY_RESULTS / "high_utility_vs_framed_neutral_aggregate.csv"),
        (
            ANALYSIS / f"{stem}_bonferroni_aggregate.csv",
            PAPER_READY_RESULTS / "high_utility_vs_framed_neutral_bonferroni_aggregate.csv",
        ),
        (
            ANALYSIS / f"{stem}_bonferroni_aggregate.md",
            PAPER_READY_RESULTS / "high_utility_vs_framed_neutral_bonferroni_aggregate.md",
        ),
        (
            ANALYSIS / f"{stem}_pair_outcomes.csv",
            PAPER_READY_RESULTS / "high_utility_vs_framed_neutral_pair_outcomes.csv",
        ),
        (
            ANALYSIS / f"{stem}_model_task_cells.csv",
            PAPER_READY_RESULTS / "high_utility_vs_framed_neutral_model_task_cells.csv",
        ),
        (
            ANALYSIS / f"{stem}_domain_model_task_cells.csv",
            PAPER_READY_RESULTS / "high_utility_vs_framed_neutral_domain_model_task_cells.csv",
        ),
        (
            FIGURES / f"{stem}_model_task_lollipop.png",
            PAPER_READY_FIGURES / "high_utility_vs_framed_neutral_model_task_lollipop.png",
        ),
    ]
    for domain in DOMAIN_ORDER:
        copies.append(
            (
                FIGURES / f"{stem}_{domain}_model_task_lollipop.png",
                PAPER_READY_FIGURES / f"high_utility_vs_framed_neutral_domain_{domain}_model_task_lollipop.png",
            )
        )
    written: list[Path] = []
    for source, dest in copies:
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, dest)
        written.append(dest)
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
    parser.add_argument("--paper-ready", action="store_true", help="Copy stable paper-ready outputs to outputs/paper_ready and CURRENT_PAPER.")
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
    bonferroni_aggregate = build_bonferroni_aggregate_rows(high)
    bonferroni_aggregate.to_csv(ANALYSIS / f"{stem}_bonferroni_aggregate.csv", index=False)
    write_bonferroni_aggregate_md(
        bonferroni_aggregate,
        ANALYSIS / f"{stem}_bonferroni_aggregate.md",
    )

    all_cells = build_cell_rows(high)
    all_cells.to_csv(ANALYSIS / f"{stem}_model_task_cells.csv", index=False)
    figure_paths: list[Path] = []
    overall_stem = FIGURES / f"{stem}_model_task_lollipop"
    plot_lollipop(all_cells, overall_stem, title="", write_pdf=False)
    figure_paths.append(overall_stem.with_suffix(".png"))

    domain_frames = []
    for domain in DOMAIN_ORDER:
        cells = build_cell_rows(high, domain=domain)
        domain_frames.append(cells)
        domain_stem = FIGURES / f"{stem}_{domain}_model_task_lollipop"
        plot_lollipop(cells, domain_stem, title="", write_pdf=False)
        figure_paths.append(domain_stem.with_suffix(".png"))
    pd.concat(domain_frames, ignore_index=True).to_csv(ANALYSIS / f"{stem}_domain_model_task_cells.csv", index=False)

    summary_path = write_summary_md(
        run_dirs=run_dirs,
        stem=stem,
        aggregate=aggregate,
        figure_paths=figure_paths,
    )
    print(f"summary: {summary_path}")
    print(f"aggregate: {ANALYSIS / f'{stem}_aggregate.csv'}")
    print(f"bonferroni aggregate: {ANALYSIS / f'{stem}_bonferroni_aggregate.md'}")
    print(f"model-task cells: {ANALYSIS / f'{stem}_model_task_cells.csv'}")
    print(f"figures: {overall_stem.with_suffix('.png')} and domain breakouts")
    if args.paper_ready:
        written = copy_paper_ready_outputs(stem=stem)
        print("paper-ready outputs:")
        for path in written:
            print(path)


if __name__ == "__main__":
    main()
