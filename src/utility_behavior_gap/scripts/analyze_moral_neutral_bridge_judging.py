#!/usr/bin/env python3
"""Analyze moral-bad versus framed-neutral bridge judgments."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest, t

from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API


LATEST_PATH = OUTPUT_API / "runs" / "moral_neutral_bridge_latest.txt"
MORAL_BAD = "moral_bad"
FRAMED_NEUTRAL = "framed_neutral"


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


def outcome_from_panel(panel: str) -> tuple[str, int | None, int | None]:
    if panel == MORAL_BAD:
        return "moral_bad", 1, 1
    if panel == FRAMED_NEUTRAL:
        return "framed_neutral", 0, -1
    if panel == "tie":
        return "tie", None, 0
    return "unresolved", None, None


def matching_votes(
    *,
    job: dict[str, Any],
    generations: dict[str, dict[str, Any]],
    votes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out_a = generations.get(f"{job['pair_uid']}::a")
    out_b = generations.get(f"{job['pair_uid']}::b")
    if out_a is None or out_b is None:
        return []
    expected = (output_text_fingerprint(out_a), output_text_fingerprint(out_b))
    return [
        vote
        for vote in votes
        if (vote.get("source_output_a_hash"), vote.get("source_output_b_hash")) == expected
    ]


def load_pair_rows(run_dir: Path) -> pd.DataFrame:
    jobs = read_jsonl(run_dir / "generation_jobs.jsonl")
    generations = {str(row["output_id"]): row for row in read_jsonl(run_dir / "generations.jsonl")}
    votes_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    votes_path = run_dir / "judge_votes.jsonl"
    if votes_path.exists():
        for vote in read_jsonl(votes_path):
            votes_by_pair[str(vote.get("pair_uid", ""))].append(vote)

    rows: list[dict[str, Any]] = []
    for job in jobs:
        pair_votes = matching_votes(
            job=job,
            generations=generations,
            votes=votes_by_pair.get(str(job["pair_uid"]), []),
        )
        panel = panel_from_votes(job, pair_votes)
        outcome, win_value, net_score = outcome_from_panel(panel)
        rows.append(
            {
                "actor": job.get("actor", ""),
                "task": job.get("task", ""),
                "item_id": job.get("item_id", ""),
                "item_label": job.get("item_label", ""),
                "repeat": job.get("repeat", ""),
                "source_block": job.get("source_block", ""),
                "pair_uid": job.get("pair_uid", ""),
                "cause_pair_label": job.get("cause_pair_label", ""),
                "moral_bad_label": job.get("moral_bad_label", ""),
                "panel_winner_condition": panel,
                "outcome": outcome,
                "moral_bad_win_excluding_ties": win_value,
                "moral_bad_net_score": net_score,
                "source_moral_output_id": job.get("source_moral_output_id", ""),
                "source_neutral_output_id": job.get("source_neutral_output_id", ""),
                "source_moral_pair_uid": job.get("source_moral_pair_uid", ""),
                "source_neutral_pair_uid": job.get("source_neutral_pair_uid", ""),
                "n_matching_vote_rows": len(pair_votes),
            }
        )
    df = pd.DataFrame(rows)
    for col in ("moral_bad_win_excluding_ties", "moral_bad_net_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def bootstrap_ci(values: pd.Series, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    vals = values.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return (math.nan, math.nan)
    if len(vals) == 1 or iterations <= 0:
        value = float(vals.mean())
        return value, value
    samples = rng.choice(vals, size=(iterations, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(lo), float(hi)


def exact_ci(wins: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    lo = 0.0 if wins == 0 else float(beta.ppf(alpha / 2, wins, total - wins + 1))
    hi = 1.0 if wins == total else float(beta.ppf(1 - alpha / 2, wins + 1, total - wins))
    return lo, hi


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
        moral_wins = int(sub["outcome"].eq("moral_bad").sum())
        neutral_wins = int(sub["outcome"].eq("framed_neutral").sum())
        ties = int(sub["outcome"].eq("tie").sum())
        unresolved = int(sub["outcome"].eq("unresolved").sum())
        non_ties = moral_wins + neutral_wins
        resolved = non_ties + ties
        win_ci_lo, win_ci_hi = exact_ci(moral_wins, non_ties)
        net_score = float(sub["moral_bad_net_score"].dropna().mean()) if resolved else math.nan
        net_ci_lo, net_ci_hi = bootstrap_ci(sub["moral_bad_net_score"], iterations, rng)
        p_value = binomtest(moral_wins, non_ties, 0.5, alternative="two-sided").pvalue if non_ties else math.nan
        row.update(
            {
                "n_pairs": int(len(sub)),
                "resolved_pairs": resolved,
                "moral_bad_wins": moral_wins,
                "framed_neutral_wins": neutral_wins,
                "ties": ties,
                "unresolved": unresolved,
                "moral_bad_win_rate_excluding_ties": moral_wins / non_ties if non_ties else math.nan,
                "moral_bad_win_rate_ci_lo": win_ci_lo,
                "moral_bad_win_rate_ci_hi": win_ci_hi,
                "moral_bad_net_score": net_score,
                "moral_bad_net_score_ci_lo": net_ci_lo,
                "moral_bad_net_score_ci_hi": net_ci_hi,
                "p_two_sided_exact": p_value,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def t_ci(values: pd.Series) -> tuple[float, float]:
    vals = values.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        value = float(vals.mean())
        return value, value
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / math.sqrt(len(vals)))
    crit = float(t.ppf(0.975, len(vals) - 1))
    return mean - crit * se, mean + crit * se


def summarize_equal_actor_task_cells(cell_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = [((), cell_df)] if not group_cols else cell_df.groupby(group_cols, dropna=False, sort=True)
    for key, sub in grouped:
        if group_cols:
            if len(group_cols) == 1 and not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(group_cols, key))
        else:
            row = {"group": "overall_equal_actor_task"}

        win_lo, win_hi = t_ci(sub["moral_bad_win_rate_excluding_ties"])
        net_lo, net_hi = t_ci(sub["moral_bad_net_score"])
        row.update(
            {
                "n_actor_task_cells": int(len(sub)),
                "n_pairs": int(sub["n_pairs"].sum()),
                "moral_bad_wins": int(sub["moral_bad_wins"].sum()),
                "framed_neutral_wins": int(sub["framed_neutral_wins"].sum()),
                "ties": int(sub["ties"].sum()),
                "unresolved": int(sub["unresolved"].sum()),
                "moral_bad_win_rate_equal_actor_task": float(sub["moral_bad_win_rate_excluding_ties"].mean()),
                "moral_bad_win_rate_equal_actor_task_ci_lo": win_lo,
                "moral_bad_win_rate_equal_actor_task_ci_hi": win_hi,
                "moral_bad_net_score_equal_actor_task": float(sub["moral_bad_net_score"].mean()),
                "moral_bad_net_score_equal_actor_task_ci_lo": net_lo,
                "moral_bad_net_score_equal_actor_task_ci_hi": net_hi,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.3f}"


def write_summary_markdown(
    run_dirs: list[Path],
    summaries: dict[str, pd.DataFrame],
    equal_summaries: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    overall = summaries["overall"]
    by_task = summaries["by_task"]
    equal_overall = equal_summaries["equal_overall"]
    equal_by_task = equal_summaries["equal_by_task"]
    lines = [
        "# Moral-Bad Versus Framed-Neutral Bridge",
        "",
        "Run dirs:",
        "",
    ]
    lines.extend([f"- `{run_dir}`" for run_dir in run_dirs])
    lines.extend(
        [
            "",
            "Score definition: moral-bad win = +1, framed-neutral win = -1, tie = 0.",
            "The win-rate denominator excludes panel ties; the net score includes panel ties as 0.",
            "Only clean moral-bad outputs are included, using hash-checked LLM refusal/degenerate labels.",
            "Primary paper-facing summaries average actor-task cells with equal weight and use t intervals over those cells.",
            "Pooled summaries are included as descriptive counts only.",
            "",
            "## Equal Actor-Task Summary",
            "",
            "| cells | pairs | moral-bad wins | neutral wins | ties | unresolved | win rate | 95% CI | net score | 95% CI |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in equal_overall.iterrows():
        lines.append(
            f"| {row['n_actor_task_cells']} | {row['n_pairs']} | {row['moral_bad_wins']} | "
            f"{row['framed_neutral_wins']} | {row['ties']} | {row['unresolved']} | "
            f"{fmt(row['moral_bad_win_rate_equal_actor_task'])} | "
            f"{fmt(row['moral_bad_win_rate_equal_actor_task_ci_lo'])}-{fmt(row['moral_bad_win_rate_equal_actor_task_ci_hi'])} | "
            f"{fmt(row['moral_bad_net_score_equal_actor_task'])} | "
            f"{fmt(row['moral_bad_net_score_equal_actor_task_ci_lo'])}-{fmt(row['moral_bad_net_score_equal_actor_task_ci_hi'])} |"
        )
    lines.extend(
        [
            "",
            "## Equal Actor-Task Summary By Task",
            "",
            "| task | cells | pairs | moral-bad wins | neutral wins | ties | unresolved | win rate | 95% CI | net score | 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in equal_by_task.iterrows():
        lines.append(
            f"| {row['task']} | {row['n_actor_task_cells']} | {row['n_pairs']} | {row['moral_bad_wins']} | "
            f"{row['framed_neutral_wins']} | {row['ties']} | {row['unresolved']} | "
            f"{fmt(row['moral_bad_win_rate_equal_actor_task'])} | "
            f"{fmt(row['moral_bad_win_rate_equal_actor_task_ci_lo'])}-{fmt(row['moral_bad_win_rate_equal_actor_task_ci_hi'])} | "
            f"{fmt(row['moral_bad_net_score_equal_actor_task'])} | "
            f"{fmt(row['moral_bad_net_score_equal_actor_task_ci_lo'])}-{fmt(row['moral_bad_net_score_equal_actor_task_ci_hi'])} |"
        )
    lines.extend(
        [
            "",
            "## Pooled Overall",
            "",
            "| pairs | moral-bad wins | neutral wins | ties | unresolved | win rate | exact 95% CI | net score | 95% bootstrap CI | p(two-sided) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in overall.iterrows():
        lines.append(
            f"| {row['n_pairs']} | {row['moral_bad_wins']} | {row['framed_neutral_wins']} | "
            f"{row['ties']} | {row['unresolved']} | {fmt(row['moral_bad_win_rate_excluding_ties'])} | "
            f"{fmt(row['moral_bad_win_rate_ci_lo'])}-{fmt(row['moral_bad_win_rate_ci_hi'])} | "
            f"{fmt(row['moral_bad_net_score'])} | "
            f"{fmt(row['moral_bad_net_score_ci_lo'])}-{fmt(row['moral_bad_net_score_ci_hi'])} | "
            f"{fmt(row['p_two_sided_exact'])} |"
        )
    lines.extend(
        [
            "",
            "## Pooled By Task",
            "",
            "| task | pairs | moral-bad wins | neutral wins | ties | unresolved | win rate | exact 95% CI | net score | 95% bootstrap CI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in by_task.iterrows():
        lines.append(
            f"| {row['task']} | {row['n_pairs']} | {row['moral_bad_wins']} | {row['framed_neutral_wins']} | "
            f"{row['ties']} | {row['unresolved']} | {fmt(row['moral_bad_win_rate_excluding_ties'])} | "
            f"{fmt(row['moral_bad_win_rate_ci_lo'])}-{fmt(row['moral_bad_win_rate_ci_hi'])} | "
            f"{fmt(row['moral_bad_net_score'])} | "
            f"{fmt(row['moral_bad_net_score_ci_lo'])}-{fmt(row['moral_bad_net_score_ci_hi'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Bridge run directory. Repeat for multiple per-actor bridge runs. Default: latest bridge run.",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    run_dirs = args.run_dir or [default_run_dir()]
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    stem = (
        f"moral_neutral_bridge__{run_dirs[0].name}"
        if len(run_dirs) == 1
        else f"moral_neutral_bridge__combined_{len(run_dirs)}runs"
    )

    pairs = pd.concat([load_pair_rows(run_dir) for run_dir in run_dirs], ignore_index=True)
    pair_path = ANALYSIS / f"{stem}__pair_outcomes.csv"
    pairs.to_csv(pair_path, index=False)

    summaries = {
        "overall": summarize(pairs, [], iterations=args.bootstrap, seed=args.seed),
        "by_actor": summarize(pairs, ["actor"], iterations=args.bootstrap, seed=args.seed),
        "by_task": summarize(pairs, ["task"], iterations=args.bootstrap, seed=args.seed),
        "by_source_block": summarize(pairs, ["source_block"], iterations=args.bootstrap, seed=args.seed),
        "by_actor_task": summarize(pairs, ["actor", "task"], iterations=args.bootstrap, seed=args.seed),
    }
    equal_summaries = {
        "equal_overall": summarize_equal_actor_task_cells(summaries["by_actor_task"], []),
        "equal_by_task": summarize_equal_actor_task_cells(summaries["by_actor_task"], ["task"]),
        "equal_by_actor": summarize_equal_actor_task_cells(summaries["by_actor_task"], ["actor"]),
    }
    for name, summary in summaries.items():
        summary.to_csv(ANALYSIS / f"{stem}__{name}.csv", index=False)
    for name, summary in equal_summaries.items():
        summary.to_csv(ANALYSIS / f"{stem}__{name}.csv", index=False)

    md_path = ANALYSIS / f"{stem}__summary.md"
    write_summary_markdown(run_dirs, summaries, equal_summaries, md_path)

    print(f"pair outcomes: {pair_path}")
    for name in summaries:
        print(f"{name}: {ANALYSIS / f'{stem}__{name}.csv'}")
    for name in equal_summaries:
        print(f"{name}: {ANALYSIS / f'{stem}__{name}.csv'}")
    print(f"summary: {md_path}")
    print(equal_summaries["equal_overall"].to_string(index=False))
    print(equal_summaries["equal_by_task"].to_string(index=False))
    print(summaries["overall"].to_string(index=False))
    print(summaries["by_task"].to_string(index=False))


if __name__ == "__main__":
    main()
