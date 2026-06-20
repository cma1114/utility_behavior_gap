#!/usr/bin/env python3
"""Task-specific LLM-rubric deltas for direct instruction.

This uses the existing rubric-coded sample, not the full generic text-feature
catalog. For direct instruction, positive values mean the rubric coder judged
the strong/exhortative output better than the framed-neutral output on that
dimension; negative values mean the framed-neutral output was better; zero
means tie, not applicable, or unresolved.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utility_behavior_gap.constants import ACTOR_LABEL, TASK_LABEL
from utility_behavior_gap.paths import ANALYSIS
from utility_behavior_gap.scripts.run_task_rubric_pilot import (
    DIMENSION_DESCRIPTIONS,
    TASK_DIMENSIONS,
)


DEFAULT_INPUT = ANALYSIS / "audit_task_rubric_feature_models_flat_pairs.csv"
OUT_PREFIX = "direct_instruction_rubric_deltas"
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


def actor_bootstrap_ci(
    actor_values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(actor_values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1 or iterations <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_dimension(
    df: pd.DataFrame,
    *,
    task: str,
    dimension: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    col = f"rubric_hl__{dimension}"
    if col not in df.columns:
        raise ValueError(f"Missing rubric column: {col}")

    sub = df[df["task"].eq(task)].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub[sub[col].notna()].copy()
    if sub.empty:
        return {}

    actor_means = sub.groupby("actor", sort=True)[col].mean()
    ci_low, ci_high = actor_bootstrap_ci(
        actor_means.to_numpy(dtype=float),
        iterations=iterations,
        seed=stable_seed(seed, (task, dimension)),
    )
    values = sub[col].to_numpy(dtype=float)
    return {
        "task": task,
        "task_label": TASK_LABEL.get(task, task),
        "dimension": dimension,
        "description": DIMENSION_DESCRIPTIONS.get(dimension, ""),
        "n_pairs": int(len(values)),
        "n_actors": int(len(actor_means)),
        "mean_strong_minus_neutral_equal_actor": float(actor_means.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_excludes_zero": bool(
            np.isfinite(ci_low)
            and np.isfinite(ci_high)
            and ((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
        ),
        "raw_pair_mean": float(np.mean(values)),
        "raw_pair_sd": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
        "pct_strong_better": float(np.mean(values > 0)),
        "pct_neutral_better": float(np.mean(values < 0)),
        "pct_tie_or_not_applicable": float(np.mean(values == 0)),
    }


def summarize_by_actor(df: pd.DataFrame, *, task: str, dimension: str) -> pd.DataFrame:
    col = f"rubric_hl__{dimension}"
    sub = df[df["task"].eq(task)].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub[sub[col].notna()]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for actor, actor_df in sub.groupby("actor", sort=True):
        values = actor_df[col].to_numpy(dtype=float)
        rows.append(
            {
                "task": task,
                "task_label": TASK_LABEL.get(task, task),
                "actor": actor,
                "actor_label": ACTOR_LABEL.get(actor, actor),
                "dimension": dimension,
                "n_pairs": int(len(values)),
                "mean_strong_minus_neutral": float(np.mean(values)),
                "pct_strong_better": float(np.mean(values > 0)),
                "pct_neutral_better": float(np.mean(values < 0)),
                "pct_tie_or_not_applicable": float(np.mean(values == 0)),
            }
        )
    return pd.DataFrame(rows)


def compact_task_table(summary: pd.DataFrame, task: str) -> pd.DataFrame:
    sub = summary[summary["task"].eq(task)].copy()
    sub = sub.sort_values("mean_strong_minus_neutral_equal_actor", ascending=False)
    return pd.DataFrame(
        {
            "marker": sub["dimension"],
            "delta": sub["mean_strong_minus_neutral_equal_actor"].map(lambda x: fmt(x, 3)),
            "95% CI": [
                f"[{fmt(lo, 3)}, {fmt(hi, 3)}]"
                for lo, hi in zip(sub["ci_low"], sub["ci_high"])
            ],
            "% strong better": sub["pct_strong_better"].map(lambda x: f"{100*x:.1f}%"),
            "% neutral better": sub["pct_neutral_better"].map(lambda x: f"{100*x:.1f}%"),
            "% tie/NA": sub["pct_tie_or_not_applicable"].map(lambda x: f"{100*x:.1f}%"),
            "description": sub["description"],
        }
    )


def write_summary(out_dir: Path, summary: pd.DataFrame, by_actor: pd.DataFrame, *, input_path: Path) -> Path:
    lines = [
        "# Direct-Instruction LLM-Rubric Deltas",
        "",
        "Comparison: strong/exhortative direct-instruction output minus framed-neutral output.",
        "",
        "These are task-specific qualitative markers coded by an LLM rubric judge. The analysis uses the existing rubric-coded sample, not the full generic text-feature catalog.",
        "",
        "Positive values mean the strong output was coded better on that marker; negative values mean the framed-neutral output was coded better. Confidence intervals are bootstrapped over actors within each task.",
        "",
        f"- input: `{input_path}`",
        f"- coded direct-instruction pairs: `{int(summary[['task', 'n_pairs']].drop_duplicates()['n_pairs'].sum())}`",
        f"- tasks: `{', '.join(TASK_LABEL.get(task, task) for task in TASK_DIMENSIONS)}`",
        "",
    ]
    for task in TASK_DIMENSIONS:
        if task not in set(summary["task"]):
            continue
        lines.extend(
            [
                f"## {TASK_LABEL.get(task, task)}",
                "",
                markdown_table(compact_task_table(summary, task)),
                "",
            ]
        )
    lines.extend(
        [
            "## Outputs",
            "",
            f"- by task/dimension: `{out_dir / (OUT_PREFIX + '_by_task_dimension.csv')}`",
            f"- by actor/task/dimension: `{out_dir / (OUT_PREFIX + '_by_actor_task_dimension.csv')}`",
            f"- summary markdown: `{out_dir / (OUT_PREFIX + '_summary.md')}`",
        ]
    )
    path = out_dir / f"{OUT_PREFIX}_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    direct = df[df["contrast"].eq("direct_instruction")].copy()
    if direct.empty:
        raise ValueError("No direct_instruction rows found in rubric flat-pair file.")

    rows: list[dict[str, Any]] = []
    actor_rows: list[pd.DataFrame] = []
    for task, dimensions in TASK_DIMENSIONS.items():
        task_df = direct[direct["task"].eq(task)]
        if task_df.empty:
            continue
        for dimension in dimensions:
            rows.append(
                summarize_dimension(
                    direct,
                    task=task,
                    dimension=dimension,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed,
                )
            )
            actor_rows.append(summarize_by_actor(direct, task=task, dimension=dimension))

    summary = pd.DataFrame([row for row in rows if row])
    by_actor = pd.concat([frame for frame in actor_rows if not frame.empty], ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / f"{OUT_PREFIX}_by_task_dimension.csv"
    actor_path = args.out_dir / f"{OUT_PREFIX}_by_actor_task_dimension.csv"
    summary.to_csv(summary_path, index=False)
    by_actor.to_csv(actor_path, index=False)
    md_path = write_summary(args.out_dir, summary, by_actor, input_path=args.input)

    print(f"coded direct pairs: {len(direct)}")
    print(f"task/dimension rows: {len(summary)}")
    print(f"summary: {md_path}")
    print(f"by task/dimension: {summary_path}")


if __name__ == "__main__":
    main()
