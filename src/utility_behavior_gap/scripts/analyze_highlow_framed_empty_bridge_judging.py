#!/usr/bin/env python3
"""Analyze high/low utility versus framed-empty bridge judgments."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t

from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API


LATEST_PATH = OUTPUT_API / "runs" / "highlow_framed_empty_bridge_latest.txt"
FRAMED_EMPTY_CONDITION = "framed_empty"


def default_run_dir() -> Path:
    if not LATEST_PATH.exists():
        raise FileNotFoundError(f"{LATEST_PATH} does not exist; pass --run-dir explicitly")
    return Path(LATEST_PATH.read_text(encoding="utf-8").strip())


def judge_verdicts_for_pair(votes: list[dict[str, Any]]) -> list[str]:
    by_judge: dict[str, list[str]] = defaultdict(list)
    for vote in votes:
        if vote.get("success") is False:
            continue
        by_judge[str(vote.get("judge_model", ""))].append(str(vote.get("winner_condition", "")))
    return [derive_judge_verdict(values) for _, values in sorted(by_judge.items())]


def panel_from_votes(job: dict[str, Any], votes: list[dict[str, Any]]) -> str:
    return derive_panel_winner_condition(job, judge_verdicts_for_pair(votes))


def outcome_from_panel(panel: str, side_condition: str) -> tuple[str, int | None]:
    if panel == side_condition:
        return "side", 1
    if panel == FRAMED_EMPTY_CONDITION:
        return "framed_empty", -1
    if panel == "tie":
        return "tie", 0
    return "unresolved", None


def load_pair_rows(run_dir: Path) -> pd.DataFrame:
    jobs = read_jsonl(run_dir / "generation_jobs.jsonl")
    votes_path = run_dir / "judge_votes.jsonl"
    if not votes_path.exists():
        raise FileNotFoundError(votes_path)
    votes_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote in read_jsonl(votes_path):
        votes_by_pair[str(vote.get("pair_uid", ""))].append(vote)

    rows: list[dict[str, Any]] = []
    for job in jobs:
        side = str(job.get("side", ""))
        side_condition = str(job.get("side_condition") or job.get("condition_a", ""))
        panel = panel_from_votes(job, votes_by_pair.get(str(job["pair_uid"]), []))
        outcome, signed_score = outcome_from_panel(panel, side_condition)
        rows.append(
            {
                "actor": job.get("actor", ""),
                "task": job.get("task", ""),
                "domain": job.get("domain", ""),
                "side": side,
                "side_condition": side_condition,
                "panel_winner_condition": panel,
                "outcome_vs_framed_empty": outcome,
                "side_net_score": signed_score,
                "high_vs_framed_empty_score": signed_score if side == "high" else "",
                "low_below_framed_empty_score": -signed_score if side == "low" and signed_score is not None else "",
                "high_utility": job.get("high_utility", ""),
                "low_utility": job.get("low_utility", ""),
                "delta_u": job.get("delta_u", ""),
                "high_description": job.get("high_description", ""),
                "low_description": job.get("low_description", ""),
                "item_id": job.get("item_id", ""),
                "item_label": job.get("item_label", ""),
                "pair_idx": job.get("pair_idx", ""),
                "repeat": job.get("repeat", ""),
                "bridge_pair_uid": job.get("pair_uid", ""),
                "source_highlow_output_id": job.get("source_highlow_output_id", ""),
                "source_framed_empty_output_id": job.get("source_framed_empty_output_id", ""),
                "source_highlow_pair_uid": job.get("source_highlow_pair_uid", ""),
                "source_framed_empty_pair_uid": job.get("source_framed_empty_pair_uid", ""),
                "n_vote_rows": len(votes_by_pair.get(str(job["pair_uid"]), [])),
            }
        )
    df = pd.DataFrame(rows)
    for col in (
        "high_utility",
        "low_utility",
        "delta_u",
        "side_net_score",
        "high_vs_framed_empty_score",
        "low_below_framed_empty_score",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def bootstrap_ci(values: pd.Series, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    vals = values.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan)
    if len(vals) == 1 or iterations <= 0:
        mean = float(vals.mean())
        return (mean, mean)
    samples = rng.choice(vals, size=(iterations, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return (float(lo), float(hi))


def summarize(df: pd.DataFrame, group_cols: list[str], *, iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False, sort=True)
    for key, sub in grouped:
        if group_cols:
            if len(group_cols) == 1 and not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(group_cols, key))
        else:
            row = {"group": "overall"}
        side_wins = int(sub["outcome_vs_framed_empty"].eq("side").sum())
        framed_empty_wins = int(sub["outcome_vs_framed_empty"].eq("framed_empty").sum())
        ties = int(sub["outcome_vs_framed_empty"].eq("tie").sum())
        unresolved = int(sub["outcome_vs_framed_empty"].eq("unresolved").sum())
        non_ties = side_wins + framed_empty_wins
        resolved = non_ties + ties
        side_net_score = float(sub["side_net_score"].dropna().mean()) if resolved else np.nan
        ci_lo, ci_hi = bootstrap_ci(sub["side_net_score"], iterations, rng)
        row.update(
            {
                "n_pairs": int(len(sub)),
                "resolved_pairs": resolved,
                "side_wins": side_wins,
                "framed_empty_wins": framed_empty_wins,
                "ties": ties,
                "unresolved": unresolved,
                "side_win_rate_excluding_ties": side_wins / non_ties if non_ties else np.nan,
                "side_net_score": side_net_score,
                "side_net_score_ci_lo": ci_lo,
                "side_net_score_ci_hi": ci_hi,
            }
        )
        if "side" in row and row["side"] == "low":
            row["low_below_framed_empty_score"] = -side_net_score if resolved else np.nan
            row["low_below_framed_empty_score_ci_lo"] = -ci_hi if resolved else np.nan
            row["low_below_framed_empty_score_ci_hi"] = -ci_lo if resolved else np.nan
        if "side" in row and row["side"] == "high":
            row["high_vs_framed_empty_score"] = side_net_score
            row["high_vs_framed_empty_score_ci_lo"] = ci_lo
            row["high_vs_framed_empty_score_ci_hi"] = ci_hi
        rows.append(row)
    return pd.DataFrame(rows)


def t_ci(values: pd.Series) -> tuple[float, float]:
    vals = values.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    if len(vals) == 1:
        value = float(vals.mean())
        return value, value
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    crit = float(t.ppf(0.975, len(vals) - 1))
    return mean - crit * se, mean + crit * se


def summarize_equal_cells(cell_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = [((), cell_df)] if not group_cols else cell_df.groupby(group_cols, dropna=False, sort=True)
    for key, sub in grouped:
        if group_cols:
            if len(group_cols) == 1 and not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(group_cols, key))
        else:
            row = {"group": "overall_equal_actor_task_domain"}
        win_lo, win_hi = t_ci(sub["side_win_rate_excluding_ties"])
        net_lo, net_hi = t_ci(sub["side_net_score"])
        row.update(
            {
                "n_actor_task_domain_cells": int(len(sub)),
                "n_pairs": int(sub["n_pairs"].sum()),
                "side_wins": int(sub["side_wins"].sum()),
                "framed_empty_wins": int(sub["framed_empty_wins"].sum()),
                "ties": int(sub["ties"].sum()),
                "unresolved": int(sub["unresolved"].sum()),
                "side_win_rate_equal_cell": float(sub["side_win_rate_excluding_ties"].mean()),
                "side_win_rate_equal_cell_ci_lo": win_lo,
                "side_win_rate_equal_cell_ci_hi": win_hi,
                "side_net_score_equal_cell": float(sub["side_net_score"].mean()),
                "side_net_score_equal_cell_ci_lo": net_lo,
                "side_net_score_equal_cell_ci_hi": net_hi,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary_markdown(
    run_dirs: list[Path],
    summaries: dict[str, pd.DataFrame],
    equal_summaries: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    side = summaries["by_side"]
    equal_side = equal_summaries["equal_by_side"]
    lines = [
        "# High/Low Utility Versus Framed Empty Bridge",
        "",
        "Run dirs:",
        "",
    ]
    lines.extend([f"- `{run_dir}`" for run_dir in run_dirs])
    lines.extend(
        [
            "",
            "Scores are from the utility side's perspective: utility-side win = +1, framed-empty win = -1, tie = 0. For low-utility rows, `low_below_framed_empty_score` is the sign-flipped score, so positive values mean framed-empty beat the low-utility output.",
            "",
            "Primary paper-facing summaries average actor x task x domain cells with equal weight and use t intervals over those cells. Pooled summaries are included as descriptive counts only.",
            "",
            "## Equal Actor-Task-Domain Summary",
            "",
            "| side | cells | pairs | side wins | framed-empty wins | ties | unresolved | equal-cell win rate | 95% CI | equal-cell net score | 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in equal_side.iterrows():
        lines.append(
            f"| {row['side']} | {row['n_actor_task_domain_cells']} | {row['n_pairs']} | "
            f"{row['side_wins']} | {row['framed_empty_wins']} | {row['ties']} | {row['unresolved']} | "
            f"{row['side_win_rate_equal_cell']:.3f} | "
            f"{row['side_win_rate_equal_cell_ci_lo']:.3f}-{row['side_win_rate_equal_cell_ci_hi']:.3f} | "
            f"{row['side_net_score_equal_cell']:.3f} | "
            f"{row['side_net_score_equal_cell_ci_lo']:.3f}-{row['side_net_score_equal_cell_ci_hi']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Pooled Side Summary",
            "",
            "| side | pairs | side wins | framed-empty wins | ties | unresolved | side net score | 95% bootstrap CI | tie-excluded side win rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in side.iterrows():
        win_rate = row["side_win_rate_excluding_ties"]
        lines.append(
            f"| {row['side']} | {row['n_pairs']} | {row['side_wins']} | {row['framed_empty_wins']} | "
            f"{row['ties']} | {row['unresolved']} | {row['side_net_score']:.3f} | "
            f"{row['side_net_score_ci_lo']:.3f}-{row['side_net_score_ci_hi']:.3f} | "
            f"{win_rate:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Bridge run directory. May be passed multiple times. Default: latest bridge run.",
    )
    parser.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap iterations for side-net-score CIs.")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    run_dirs = args.run_dir or [default_run_dir()]
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    stem = (
        f"highlow_framed_empty_bridge__{run_dirs[0].name}"
        if len(run_dirs) == 1
        else f"highlow_framed_empty_bridge__combined_{len(run_dirs)}runs"
    )

    pairs = pd.concat([load_pair_rows(run_dir) for run_dir in run_dirs], ignore_index=True)
    pair_path = ANALYSIS / f"{stem}__pair_outcomes.csv"
    pairs.to_csv(pair_path, index=False)

    summaries = {
        "overall": summarize(pairs, [], iterations=args.bootstrap, seed=args.seed),
        "by_side": summarize(pairs, ["side"], iterations=args.bootstrap, seed=args.seed),
        "by_side_actor": summarize(pairs, ["side", "actor"], iterations=args.bootstrap, seed=args.seed),
        "by_side_task": summarize(pairs, ["side", "task"], iterations=args.bootstrap, seed=args.seed),
        "by_side_domain": summarize(pairs, ["side", "domain"], iterations=args.bootstrap, seed=args.seed),
        "by_side_actor_task": summarize(pairs, ["side", "actor", "task"], iterations=args.bootstrap, seed=args.seed),
        "by_side_actor_domain": summarize(pairs, ["side", "actor", "domain"], iterations=args.bootstrap, seed=args.seed),
        "by_side_task_domain": summarize(pairs, ["side", "task", "domain"], iterations=args.bootstrap, seed=args.seed),
        "by_side_actor_task_domain": summarize(
            pairs, ["side", "actor", "task", "domain"], iterations=args.bootstrap, seed=args.seed
        ),
    }
    for name, summary in summaries.items():
        summary.to_csv(ANALYSIS / f"{stem}__{name}.csv", index=False)

    cell_summary = summaries["by_side_actor_task_domain"].copy()
    equal_summaries = {
        "equal_by_side": summarize_equal_cells(cell_summary, ["side"]),
        "equal_by_side_task": summarize_equal_cells(cell_summary, ["side", "task"]),
        "equal_by_side_domain": summarize_equal_cells(cell_summary, ["side", "domain"]),
        "equal_by_side_actor": summarize_equal_cells(cell_summary, ["side", "actor"]),
    }
    for name, summary in equal_summaries.items():
        summary.to_csv(ANALYSIS / f"{stem}__{name}.csv", index=False)

    md_path = ANALYSIS / f"{stem}__summary.md"
    write_summary_markdown(run_dirs, summaries, equal_summaries, md_path)

    print(f"pair outcomes: {pair_path}")
    for name in summaries:
        print(f"{name}: {ANALYSIS / f'{stem}__{name}.csv'}")
    print(f"summary: {md_path}")
    print(summaries["by_side"].to_string(index=False))


if __name__ == "__main__":
    main()
