#!/usr/bin/env python3
"""Analyze MiMo framed-user-strong versus framed-neutral judgments."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API


LATEST_PATH = OUTPUT_API / "runs" / "mimo_framed_user_strong_latest.txt"
STRONG_CONDITION = "framed_user_strong"
NEUTRAL_CONDITION = "framed_neutral"


def default_run_dir() -> Path:
    if not LATEST_PATH.exists():
        raise FileNotFoundError(f"{LATEST_PATH} does not exist; pass --run-dir explicitly")
    return Path(LATEST_PATH.read_text(encoding="utf-8").strip())


def word_count(text: str) -> int:
    return len(re.findall(r"\b\S+\b", text or ""))


def judge_verdicts_for_pair(votes: list[dict[str, Any]]) -> list[str]:
    by_judge: dict[str, list[str]] = defaultdict(list)
    for vote in votes:
        if vote.get("success") is False:
            continue
        by_judge[str(vote.get("judge_model", ""))].append(str(vote.get("winner_condition", "")))
    return [derive_judge_verdict(values) for _, values in sorted(by_judge.items())]


def panel_from_votes(job: dict[str, Any], votes: list[dict[str, Any]]) -> str:
    return derive_panel_winner_condition(job, judge_verdicts_for_pair(votes))


def load_pair_rows(run_dir: Path) -> pd.DataFrame:
    jobs = read_jsonl(run_dir / "generation_jobs.jsonl")
    generations = {row["output_id"]: row for row in read_jsonl(run_dir / "generations.jsonl")}
    votes_path = run_dir / "judge_votes.jsonl"
    if not votes_path.exists():
        raise FileNotFoundError(votes_path)
    votes_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote in read_jsonl(votes_path):
        votes_by_pair[str(vote.get("pair_uid", ""))].append(vote)

    rows: list[dict[str, Any]] = []
    for job in jobs:
        panel = panel_from_votes(job, votes_by_pair.get(str(job["pair_uid"]), []))
        if panel == STRONG_CONDITION:
            outcome = "strong"
            strong_net_score: int | None = 1
        elif panel == NEUTRAL_CONDITION:
            outcome = "neutral"
            strong_net_score = -1
        elif panel == "tie":
            outcome = "tie"
            strong_net_score = 0
        else:
            outcome = "unresolved"
            strong_net_score = None
        strong = generations.get(f"{job['pair_uid']}::a", {})
        neutral = generations.get(f"{job['pair_uid']}::b", {})
        rows.append(
            {
                "task": job.get("task", ""),
                "item_label": job.get("item_label", ""),
                "panel_winner_condition": panel,
                "outcome": outcome,
                "strong_net_score": strong_net_score,
                "strong_words": word_count(str(strong.get("output_text", ""))),
                "neutral_words": word_count(str(neutral.get("output_text", ""))),
                "word_diff_strong_minus_neutral": word_count(str(strong.get("output_text", "")))
                - word_count(str(neutral.get("output_text", ""))),
                "pair_uid": job["pair_uid"],
                "source_headroom_pair_uid": job.get("source_headroom_pair_uid", ""),
                "source_neutral_output_id": job.get("source_neutral_output_id", ""),
                "n_vote_rows": len(votes_by_pair.get(str(job["pair_uid"]), [])),
            }
        )
    df = pd.DataFrame(rows)
    for col in ("strong_net_score", "strong_words", "neutral_words", "word_diff_strong_minus_neutral"):
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
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False, sort=True)
    rows: list[dict[str, Any]] = []
    for key, sub in grouped:
        if group_cols:
            if len(group_cols) == 1 and not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(group_cols, key))
        else:
            row = {"group": "overall"}
        strong_wins = int(sub["outcome"].eq("strong").sum())
        neutral_wins = int(sub["outcome"].eq("neutral").sum())
        ties = int(sub["outcome"].eq("tie").sum())
        unresolved = int(sub["outcome"].eq("unresolved").sum())
        non_ties = strong_wins + neutral_wins
        resolved = non_ties + ties
        score = float(sub["strong_net_score"].dropna().mean()) if resolved else np.nan
        ci_lo, ci_hi = bootstrap_ci(sub["strong_net_score"], iterations, rng)
        row.update(
            {
                "n_pairs": int(len(sub)),
                "resolved_pairs": resolved,
                "strong_wins": strong_wins,
                "neutral_wins": neutral_wins,
                "ties": ties,
                "unresolved": unresolved,
                "strong_win_rate_excluding_ties": strong_wins / non_ties if non_ties else np.nan,
                "strong_net_score": score,
                "strong_net_score_ci_lo": ci_lo,
                "strong_net_score_ci_hi": ci_hi,
                "mean_word_diff_strong_minus_neutral": float(sub["word_diff_strong_minus_neutral"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    headers = [str(col) for col in show.columns]
    rows = [[str(row[col]) for col in show.columns] for _, row in show.iterrows()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_summary(run_dir: Path, summaries: dict[str, pd.DataFrame], path: Path) -> None:
    lines = [
        "# MiMo Framed User-Strong Headroom",
        "",
        f"Run dir: `{run_dir}`",
        "",
        "Score definition: user-prompt strong win = +1, framed-neutral win = -1, tie = 0.",
        "",
        "## Overall",
        "",
        markdown_table(summaries["overall"]),
        "",
        "## By Task",
        "",
        markdown_table(summaries["by_task"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory. Default: latest prepared run.")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    run_dir = args.run_dir or default_run_dir()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    stem = f"mimo_framed_user_strong__{run_dir.name}"

    pairs = load_pair_rows(run_dir)
    pair_path = ANALYSIS / f"{stem}__pair_outcomes.csv"
    pairs.to_csv(pair_path, index=False)

    summaries = {
        "overall": summarize(pairs, [], iterations=args.bootstrap, seed=args.seed),
        "by_task": summarize(pairs, ["task"], iterations=args.bootstrap, seed=args.seed),
    }
    for name, summary in summaries.items():
        summary.to_csv(ANALYSIS / f"{stem}__{name}.csv", index=False)
    summary_path = ANALYSIS / f"{stem}__summary.md"
    write_summary(run_dir, summaries, summary_path)

    print(f"pair outcomes: {pair_path}")
    for name in summaries:
        print(f"{name}: {ANALYSIS / f'{stem}__{name}.csv'}")
    print(f"summary: {summary_path}")
    print(summaries["by_task"].to_string(index=False))


if __name__ == "__main__":
    main()
