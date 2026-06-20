#!/usr/bin/env python3
"""Analyze blind task-rubric feature-coding runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utility_behavior_gap.constants import ACTOR_LABEL, TASK_LABEL
from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.output_exclusions import (
    filter_semantic_excluded_pair_rows,
    filter_valid_pair_rows,
)
from utility_behavior_gap.scripts.run_task_rubric_pilot import (
    DIMENSION_DESCRIPTIONS,
    TASK_DIMENSIONS,
)


RANDOM_SEED = 20260615


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in df.to_numpy()
    ]
    return "\n".join([header, sep] + rows)


def stable_seed(seed: int, values: tuple[Any, ...]) -> int:
    digest = hashlib.sha256(repr(values).encode("utf-8")).hexdigest()
    return seed + int(digest[:8], 16)


def bootstrap_actor_ci(values: np.ndarray, *, iterations: int, seed: int) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) <= 1 or iterations <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates = rng.choice(clean, size=(iterations, len(clean)), replace=True).mean(axis=1)
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def winner_side_to_left_right_score(winner_side: str) -> float:
    if winner_side == "left":
        return 1.0
    if winner_side == "right":
        return -1.0
    return 0.0


def load_codes(run_dir: Path, *, allowed_pair_uids: set[str] | None = None) -> pd.DataFrame:
    codes_path = run_dir / "rubric_feature_codes.jsonl"
    if not codes_path.exists():
        raise FileNotFoundError(codes_path)

    rows: list[dict[str, Any]] = []
    for raw in read_jsonl(codes_path):
        if raw.get("success") is False:
            continue
        if allowed_pair_uids is not None and str(raw.get("coding_pair_uid")) not in allowed_pair_uids:
            continue
        task = str(raw["task"])
        for dimension in TASK_DIMENSIONS[task]:
            code = raw.get("codes", {}).get(dimension, {})
            rows.append(
                {
                    "coding_pair_uid": raw["coding_pair_uid"],
                    "comparison_name": raw.get("comparison_name", ""),
                    "source_mode": raw.get("source_mode", ""),
                    "actor": raw["actor"],
                    "actor_label": ACTOR_LABEL.get(str(raw["actor"]), str(raw["actor"])),
                    "task": task,
                    "task_label": TASK_LABEL.get(task, task),
                    "left_condition": raw.get("left_condition", ""),
                    "right_condition": raw.get("right_condition", ""),
                    "dimension": dimension,
                    "description": DIMENSION_DESCRIPTIONS.get(dimension, ""),
                    "winner": code.get("winner", ""),
                    "winner_side": code.get("winner_side", ""),
                    "left_minus_right_score": winner_side_to_left_right_score(
                        str(code.get("winner_side", ""))
                    ),
                    "reason": code.get("reason", ""),
                }
            )
    if not rows:
        raise SystemExit(f"No successful rubric codes found in {codes_path}.")
    return pd.DataFrame(rows)


def clean_sample_pair_uids(run_dir: Path) -> tuple[set[str] | None, dict[str, Any], dict[str, Any]]:
    sample_path = run_dir / "rubric_feature_sample.csv"
    if not sample_path.exists():
        return (
            None,
            {"valid_output_filter_applied": False, "reason": "missing rubric sample"},
            {"semantic_exclusion_filter_applied": False, "reason": "missing rubric sample"},
        )
    sample = pd.read_csv(sample_path, low_memory=False)
    sample, valid_report = filter_valid_pair_rows(sample)
    sample, semantic_report = filter_semantic_excluded_pair_rows(sample)
    return set(sample["coding_pair_uid"].astype(str)), valid_report, semantic_report


def summarize(df: pd.DataFrame, *, iterations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    actor_rows: list[dict[str, Any]] = []
    for (task, dimension), sub in df.groupby(["task", "dimension"], sort=True):
        actor_means = sub.groupby("actor", sort=True)["left_minus_right_score"].mean()
        ci_low, ci_high = bootstrap_actor_ci(
            actor_means.to_numpy(dtype=float),
            iterations=iterations,
            seed=stable_seed(seed, (task, dimension)),
        )
        values = sub["left_minus_right_score"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "task": task,
                "task_label": TASK_LABEL.get(task, task),
                "dimension": dimension,
                "description": DIMENSION_DESCRIPTIONS.get(dimension, ""),
                "n_pairs": int(sub["coding_pair_uid"].nunique()),
                "n_actors": int(len(actor_means)),
                "mean_left_minus_right_equal_actor": float(actor_means.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_excludes_zero": bool(
                    np.isfinite(ci_low)
                    and np.isfinite(ci_high)
                    and ((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
                ),
                "raw_pair_mean": float(np.mean(values)),
                "raw_pair_sd": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
                "pct_left_better": float(np.mean(values > 0)),
                "pct_right_better": float(np.mean(values < 0)),
                "pct_tie_or_not_applicable": float(np.mean(values == 0)),
            }
        )
        for actor, actor_sub in sub.groupby("actor", sort=True):
            actor_values = actor_sub["left_minus_right_score"].to_numpy(dtype=float)
            actor_rows.append(
                {
                    "task": task,
                    "task_label": TASK_LABEL.get(task, task),
                    "actor": actor,
                    "actor_label": ACTOR_LABEL.get(actor, actor),
                    "dimension": dimension,
                    "n_pairs": int(actor_sub["coding_pair_uid"].nunique()),
                    "mean_left_minus_right": float(np.mean(actor_values)),
                    "pct_left_better": float(np.mean(actor_values > 0)),
                    "pct_right_better": float(np.mean(actor_values < 0)),
                    "pct_tie_or_not_applicable": float(np.mean(actor_values == 0)),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(actor_rows)


def compact_task_table(summary: pd.DataFrame, task: str, left: str, right: str) -> pd.DataFrame:
    sub = summary[summary["task"].eq(task)].copy()
    sub = sub.sort_values("mean_left_minus_right_equal_actor", ascending=False)
    return pd.DataFrame(
        {
            "marker": sub["dimension"],
            f"delta ({left} - {right})": sub["mean_left_minus_right_equal_actor"].map(lambda x: fmt(x, 3)),
            "95% CI": [
                f"[{fmt(lo, 3)}, {fmt(hi, 3)}]"
                for lo, hi in zip(sub["ci_low"], sub["ci_high"])
            ],
            f"% {left} better": sub["pct_left_better"].map(lambda x: f"{100*x:.1f}%"),
            f"% {right} better": sub["pct_right_better"].map(lambda x: f"{100*x:.1f}%"),
            "% tie/NA": sub["pct_tie_or_not_applicable"].map(lambda x: f"{100*x:.1f}%"),
            "description": sub["description"],
        }
    )


def write_summary(
    run_dir: Path,
    summary: pd.DataFrame,
    codes: pd.DataFrame,
    *,
    iterations: int,
    valid_output_filter_report: dict[str, Any],
    semantic_exclusion_filter_report: dict[str, Any],
) -> Path:
    meta_path = run_dir / "sample_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    left = str(metadata.get("left_condition") or codes["left_condition"].dropna().iloc[0])
    right = str(metadata.get("right_condition") or codes["right_condition"].dropna().iloc[0])
    comparison = str(metadata.get("comparison_name") or codes["comparison_name"].dropna().iloc[0])

    lines = [
        "# Task-Rubric Feature-Coding Analysis",
        "",
        f"Comparison: `{comparison}`",
        f"Direction: `{left} - {right}`.",
        "",
        "Positive values mean the left arm was coded better on that task-specific marker; negative values mean the right arm was coded better. Confidence intervals are bootstrapped over actors within each task.",
        "",
        f"- run directory: `{run_dir}`",
        f"- valid-output filter: `{valid_output_filter_report}`",
        f"- semantic exclusion filter: `{semantic_exclusion_filter_report}`",
        f"- successful coded pairs: `{codes['coding_pair_uid'].nunique()}`",
        f"- bootstrap iterations: `{iterations}`",
        "",
    ]
    for task in TASK_DIMENSIONS:
        if task not in set(summary["task"]):
            continue
        lines.extend(
            [
                f"## {TASK_LABEL.get(task, task)}",
                "",
                markdown_table(compact_task_table(summary, task, left, right)),
                "",
            ]
        )
    lines.extend(
        [
            "## Outputs",
            "",
            f"- by task/dimension: `{run_dir / 'rubric_feature_analysis_by_task_dimension.csv'}`",
            f"- by actor/task/dimension: `{run_dir / 'rubric_feature_analysis_by_actor_task_dimension.csv'}`",
            f"- flat codes: `{run_dir / 'rubric_feature_analysis_flat_codes.csv'}`",
        ]
    )
    path = run_dir / "rubric_feature_analysis_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_pair_uids, valid_output_filter_report, semantic_exclusion_filter_report = clean_sample_pair_uids(
        args.run_dir
    )
    codes = load_codes(args.run_dir, allowed_pair_uids=allowed_pair_uids)
    summary, by_actor = summarize(codes, iterations=args.bootstrap_iterations, seed=args.seed)
    codes.to_csv(args.run_dir / "rubric_feature_analysis_flat_codes.csv", index=False)
    summary.to_csv(args.run_dir / "rubric_feature_analysis_by_task_dimension.csv", index=False)
    by_actor.to_csv(args.run_dir / "rubric_feature_analysis_by_actor_task_dimension.csv", index=False)
    summary_path = write_summary(
        args.run_dir,
        summary,
        codes,
        iterations=args.bootstrap_iterations,
        valid_output_filter_report=valid_output_filter_report,
        semantic_exclusion_filter_report=semantic_exclusion_filter_report,
    )
    print(f"successful coded pairs: {codes['coding_pair_uid'].nunique()}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
