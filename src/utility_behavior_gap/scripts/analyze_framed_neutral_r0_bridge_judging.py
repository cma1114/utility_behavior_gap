#!/usr/bin/env python3
"""Analyze framed-neutral versus R0 bridge judgments."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API


LATEST_PATH = OUTPUT_API / "runs" / "framed_neutral_r0_bridge_latest.txt"
FRAMED_NEUTRAL = "framed_neutral"
R0 = "r0"


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


def outcome_from_panel(panel: str) -> tuple[str, int | None]:
    if panel == FRAMED_NEUTRAL:
        return "framed_neutral", 1
    if panel == R0:
        return "r0", -1
    if panel == "tie":
        return "tie", 0
    return "unresolved", None


def load_pair_rows(run_dir: Path) -> pd.DataFrame:
    jobs = read_jsonl(run_dir / "generation_jobs.jsonl")
    votes_path = run_dir / "judge_votes.jsonl"
    votes_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if votes_path.exists():
        for vote in read_jsonl(votes_path):
            votes_by_pair[str(vote.get("pair_uid", ""))].append(vote)

    rows: list[dict[str, Any]] = []
    for job in jobs:
        panel = panel_from_votes(job, votes_by_pair.get(str(job["pair_uid"]), []))
        outcome, score = outcome_from_panel(panel)
        rows.append(
            {
                "actor": job.get("actor", ""),
                "task": job.get("task", ""),
                "item_label": job.get("item_label", ""),
                "repeat": job.get("repeat", ""),
                "pair_uid": job.get("pair_uid", ""),
                "panel_winner_condition": panel,
                "outcome": outcome,
                "framed_neutral_net_score": score,
                "source_framed_neutral_output_id": job.get("source_framed_neutral_output_id", ""),
                "source_r0_output_id": job.get("source_r0_output_id", ""),
                "n_vote_rows": len(votes_by_pair.get(str(job["pair_uid"]), [])),
            }
        )
    df = pd.DataFrame(rows)
    df["framed_neutral_net_score"] = pd.to_numeric(df["framed_neutral_net_score"], errors="coerce")
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
        framed_wins = int(sub["outcome"].eq("framed_neutral").sum())
        r0_wins = int(sub["outcome"].eq("r0").sum())
        ties = int(sub["outcome"].eq("tie").sum())
        unresolved = int(sub["outcome"].eq("unresolved").sum())
        non_ties = framed_wins + r0_wins
        resolved = non_ties + ties
        score = float(sub["framed_neutral_net_score"].dropna().mean()) if resolved else np.nan
        ci_lo, ci_hi = bootstrap_ci(sub["framed_neutral_net_score"], iterations, rng)
        row.update(
            {
                "n_pairs": int(len(sub)),
                "resolved_pairs": resolved,
                "framed_neutral_wins": framed_wins,
                "r0_wins": r0_wins,
                "ties": ties,
                "unresolved": unresolved,
                "framed_neutral_win_rate_excluding_ties": framed_wins / non_ties if non_ties else np.nan,
                "framed_neutral_net_score": score,
                "framed_neutral_net_score_ci_lo": ci_lo,
                "framed_neutral_net_score_ci_hi": ci_hi,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary_markdown(run_dir: Path, summaries: dict[str, pd.DataFrame], path: Path) -> None:
    overall = summaries["overall"]
    by_task = summaries["by_task"]
    lines = [
        "# Framed Neutral Versus R0 Judging Bridge",
        "",
        f"Run dir: `{run_dir}`",
        "",
        "Score definition: framed-neutral win = +1, R0 win = -1, tie = 0.",
        "Positive values mean judges prefer the neutral sponsor/evaluation wrapper output over the bare-task output.",
        "",
        "## Overall",
        "",
        "| pairs | framed neutral wins | R0 wins | ties | unresolved | framed-neutral net score | 95% bootstrap CI | tie-excluded framed-neutral win rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"| {row['n_pairs']} | {row['framed_neutral_wins']} | {row['r0_wins']} | "
            f"{row['ties']} | {row['unresolved']} | {row['framed_neutral_net_score']:.3f} | "
            f"{row['framed_neutral_net_score_ci_lo']:.3f}-{row['framed_neutral_net_score_ci_hi']:.3f} | "
            f"{row['framed_neutral_win_rate_excluding_ties']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## By Task",
            "",
            "| task | pairs | framed neutral wins | R0 wins | ties | unresolved | net score | 95% bootstrap CI | tie-excluded win rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in by_task.iterrows():
        lines.append(
            f"| {row['task']} | {row['n_pairs']} | {row['framed_neutral_wins']} | {row['r0_wins']} | "
            f"{row['ties']} | {row['unresolved']} | {row['framed_neutral_net_score']:.3f} | "
            f"{row['framed_neutral_net_score_ci_lo']:.3f}-{row['framed_neutral_net_score_ci_hi']:.3f} | "
            f"{row['framed_neutral_win_rate_excluding_ties']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="Bridge run directory. Default: latest bridge run.")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    run_dir = args.run_dir or default_run_dir()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    stem = f"framed_neutral_r0_bridge__{run_dir.name}"

    pairs = load_pair_rows(run_dir)
    pair_path = ANALYSIS / f"{stem}__pair_outcomes.csv"
    pairs.to_csv(pair_path, index=False)

    summaries = {
        "overall": summarize(pairs, [], iterations=args.bootstrap, seed=args.seed),
        "by_actor": summarize(pairs, ["actor"], iterations=args.bootstrap, seed=args.seed),
        "by_task": summarize(pairs, ["task"], iterations=args.bootstrap, seed=args.seed),
        "by_actor_task": summarize(pairs, ["actor", "task"], iterations=args.bootstrap, seed=args.seed),
    }
    for name, summary in summaries.items():
        summary.to_csv(ANALYSIS / f"{stem}__{name}.csv", index=False)

    md_path = ANALYSIS / f"{stem}__summary.md"
    write_summary_markdown(run_dir, summaries, md_path)

    print(f"pair outcomes: {pair_path}")
    for name in summaries:
        print(f"{name}: {ANALYSIS / f'{stem}__{name}.csv'}")
    print(f"summary: {md_path}")
    print(summaries["overall"].to_string(index=False))
    print(summaries["by_task"].to_string(index=False))


if __name__ == "__main__":
    main()
