#!/usr/bin/env python3
"""Run the live OpenRouter reproduction pipeline."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from utility_behavior_gap.constants import ACTORS
from utility_behavior_gap.paths import ROOT

ALL_COMPARISONS = "highlow_main,highlow_same_count,system_prompt,moral_nolabel,amount"

ANALYSIS_COMMANDS = [
    "utility_behavior_gap.scripts.aggregate_results",
    "utility_behavior_gap.scripts.summarize_paper_tables",
    "utility_behavior_gap.scripts.analyze_amount_pooled",
    "utility_behavior_gap.scripts.analyze_utility_gap_dose_response",
]

PLOT_COMMANDS = [
    "utility_behavior_gap.scripts.plot_highlow_main",
    "utility_behavior_gap.scripts.plot_highlow_within_count",
    "utility_behavior_gap.scripts.plot_sys_prompt_main",
    "utility_behavior_gap.scripts.plot_moral_nolabel_main",
    "utility_behavior_gap.scripts.plot_incentive_amount_main",
    "utility_behavior_gap.scripts.plot_utility_top_bottom",
]


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", default=ALL_COMPARISONS)
    parser.add_argument("--tasks", default="", help="Optional comma-separated task ids.")
    parser.add_argument("--actors", default="", help="Optional comma-separated actor ids.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic placeholder generations and votes.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny pipeline check instead of the full paper grid.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure rendering.")
    parser.add_argument("--generation-limit", type=int, default=None)
    parser.add_argument("--judging-limit", type=int, default=None)
    args = parser.parse_args()

    select_args = [sys.executable, "-m", "utility_behavior_gap.scripts.select_pairs"]
    prepare_args = [
        sys.executable,
        "-m",
        "utility_behavior_gap.scripts.prepare_generation_jobs",
        "--comparisons",
        args.comparisons,
    ]

    if args.tasks:
        prepare_args += ["--tasks", args.tasks]
    if args.actors:
        prepare_args += ["--actors", args.actors]

    if args.smoke:
        select_args += ["--default-pairs-per-cell", "1", "--same-count-pairs-per-cell", "1"]
        prepare_args += [
            "--actors",
            args.actors or ACTORS[0],
            "--tasks",
            args.tasks or "essay",
            "--pairs-per-actor-domain",
            "1",
            "--items-per-task",
            "1",
            "--moral-pairs",
            "1",
            "--system-repeats",
            "1",
            "--amount-repeats",
            "1",
            "--moral-causes-per-item",
            "1",
        ]

    run(select_args)
    run(prepare_args)

    generation_args = [sys.executable, "-m", "utility_behavior_gap.scripts.run_generation"]
    judging_args = [sys.executable, "-m", "utility_behavior_gap.scripts.run_judging"]
    if args.dry_run:
        generation_args.append("--dry-run")
        judging_args.append("--dry-run")
    if args.generation_limit is not None:
        generation_args += ["--limit", str(args.generation_limit)]
    if args.judging_limit is not None:
        judging_args += ["--limit", str(args.judging_limit)]

    run(generation_args)
    run(judging_args)
    run([sys.executable, "-m", "utility_behavior_gap.scripts.aggregate_judgments"])

    if args.smoke:
        run([sys.executable, "-m", "utility_behavior_gap.scripts.aggregate_results"])
        if args.dry_run:
            shutil.rmtree(ROOT / "outputs", ignore_errors=True)
            print(f"removed dry-run smoke outputs under {ROOT / 'outputs'}")
        return

    for module in ANALYSIS_COMMANDS:
        run([sys.executable, "-m", module])
    if not args.no_plots:
        for module in PLOT_COMMANDS:
            run([sys.executable, "-m", module])


if __name__ == "__main__":
    main()
