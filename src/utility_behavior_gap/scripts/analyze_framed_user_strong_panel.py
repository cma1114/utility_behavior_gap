#!/usr/bin/env python3
"""Panel analysis for framed user-prompt-strong headroom runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utility_behavior_gap.constants import ACTOR_LABEL_ORDER, TASK_LABEL
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API
from utility_behavior_gap.scripts.analyze_framed_user_strong_judging import load_pair_rows


MANIFEST_GLOB = "framed_user_strong_manifests__*.tsv"


def read_manifest_lists(paths: list[Path]) -> list[Path]:
    manifests: list[Path] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            _, _, manifest = line.split("\t")
            manifests.append(Path(manifest))
    return manifests


def load_panel(paths: list[Path]) -> pd.DataFrame:
    manifests = read_manifest_lists(paths)
    if not manifests:
        raise ValueError("no framed-user-strong manifests found")
    frames: list[pd.DataFrame] = []
    for manifest in manifests:
        run_dir = manifest.parent
        frame = load_pair_rows(run_dir)
        frame["run_dir"] = str(run_dir)
        frame["run_id"] = run_dir.name
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel["strong_net_score"] = pd.to_numeric(panel["strong_net_score"], errors="coerce")
    panel["word_diff_strong_minus_neutral"] = pd.to_numeric(
        panel["word_diff_strong_minus_neutral"], errors="coerce"
    )
    return panel


def cell_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (actor, actor_label, task, task_label), sub in panel.groupby(
        ["actor", "actor_label", "task", "task_label"], sort=True
    ):
        strong = int(sub["outcome"].eq("strong").sum())
        neutral = int(sub["outcome"].eq("neutral").sum())
        ties = int(sub["outcome"].eq("tie").sum())
        unresolved = int(sub["outcome"].eq("unresolved").sum())
        non_ties = strong + neutral
        resolved = non_ties + ties
        rows.append(
            {
                "actor": actor,
                "actor_label": actor_label,
                "task": task,
                "task_label": task_label,
                "n_pairs": int(len(sub)),
                "resolved_pairs": resolved,
                "strong_wins": strong,
                "neutral_wins": neutral,
                "ties": ties,
                "unresolved": unresolved,
                "strong_win_rate_excluding_ties": strong / non_ties if non_ties else np.nan,
                "strong_net_score": float(sub["strong_net_score"].dropna().mean()),
                "mean_word_diff_strong_minus_neutral": float(
                    sub["word_diff_strong_minus_neutral"].mean()
                ),
            }
        )
    out = pd.DataFrame(rows)
    actor_order = {label: i for i, label in enumerate(ACTOR_LABEL_ORDER)}
    task_order = {label: i for i, label in enumerate(TASK_LABEL.values())}
    out["_actor_order"] = out["actor_label"].map(actor_order).fillna(999)
    out["_task_order"] = out["task_label"].map(task_order).fillna(999)
    out = out.sort_values(["_actor_order", "_task_order"]).drop(columns=["_actor_order", "_task_order"])
    return out


def bootstrap_mean_ci(values: np.ndarray, iterations: int, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    if len(values) == 1 or iterations <= 0:
        mean = float(values.mean())
        return (mean, mean)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return (float(lo), float(hi))


def summarize_cells(
    cells: pd.DataFrame,
    group_cols: list[str],
    *,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    grouped = [((), cells)] if not group_cols else cells.groupby(group_cols, dropna=False, sort=True)
    rows: list[dict[str, Any]] = []
    for idx, (key, sub) in enumerate(grouped):
        if group_cols:
            if len(group_cols) == 1 and not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(group_cols, key))
        else:
            row = {"group": "overall"}
        values = sub["strong_net_score"].to_numpy(dtype=float)
        ci_lo, ci_hi = bootstrap_mean_ci(values, iterations, seed + idx)
        strong = int(sub["strong_wins"].sum())
        neutral = int(sub["neutral_wins"].sum())
        ties = int(sub["ties"].sum())
        unresolved = int(sub["unresolved"].sum())
        non_ties = strong + neutral
        row.update(
            {
                "n_cells": int(len(sub)),
                "n_pairs": int(sub["n_pairs"].sum()),
                "resolved_pairs": int(sub["resolved_pairs"].sum()),
                "strong_wins": strong,
                "neutral_wins": neutral,
                "ties": ties,
                "unresolved": unresolved,
                "pooled_strong_win_rate_excluding_ties": strong / non_ties if non_ties else np.nan,
                "mean_cell_strong_net_score": float(np.nanmean(values)),
                "cell_bootstrap_ci_lo": ci_lo,
                "cell_bootstrap_ci_hi": ci_hi,
                "mean_cell_word_diff_strong_minus_neutral": float(
                    sub["mean_word_diff_strong_minus_neutral"].mean()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
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


def write_summary(
    *,
    overall: pd.DataFrame,
    by_actor: pd.DataFrame,
    by_task: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Framed User-Prompt-Strong Panel",
        "",
        "Treatment: corrected framed-neutral user prompt plus task-specific max-effort cue appended to the user prompt.",
        "Control: corrected framed-neutral prompt. Both arms use blank system prompts.",
        "",
        "Primary score: strong win = +1, framed-neutral win = -1, tie = 0. Means are equal-cell means over actor-task cells.",
        "",
        "## Overall",
        "",
        markdown_table(overall),
        "",
        "## By Task",
        "",
        markdown_table(by_task),
        "",
        "## By Actor",
        "",
        markdown_table(by_actor),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-glob", default=MANIFEST_GLOB)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    manifest_lists = sorted((OUTPUT_API / "runs").glob(args.manifest_glob))
    panel = load_panel(manifest_lists)
    cells = cell_summary(panel)

    overall = summarize_cells(cells, [], iterations=args.bootstrap, seed=args.seed)
    by_actor = summarize_cells(cells, ["actor", "actor_label"], iterations=args.bootstrap, seed=args.seed)
    by_task = summarize_cells(cells, ["task", "task_label"], iterations=args.bootstrap, seed=args.seed)

    actor_order = {label: i for i, label in enumerate(ACTOR_LABEL_ORDER)}
    task_order = {label: i for i, label in enumerate(TASK_LABEL.values())}
    by_actor["_order"] = by_actor["actor_label"].map(actor_order).fillna(999)
    by_actor = by_actor.sort_values("_order").drop(columns=["_order"])
    by_task["_order"] = by_task["task_label"].map(task_order).fillna(999)
    by_task = by_task.sort_values("_order").drop(columns=["_order"])

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    stem = "framed_user_strong_panel"
    panel.to_csv(ANALYSIS / f"{stem}_pairs.csv", index=False)
    cells.to_csv(ANALYSIS / f"{stem}_cell_summary.csv", index=False)
    overall.to_csv(ANALYSIS / f"{stem}_overall.csv", index=False)
    by_actor.to_csv(ANALYSIS / f"{stem}_by_actor.csv", index=False)
    by_task.to_csv(ANALYSIS / f"{stem}_by_task.csv", index=False)
    summary_path = ANALYSIS / f"{stem}_summary.md"
    write_summary(overall=overall, by_actor=by_actor, by_task=by_task, path=summary_path)

    print(f"pairs: {ANALYSIS / f'{stem}_pairs.csv'}")
    print(f"cell summary: {ANALYSIS / f'{stem}_cell_summary.csv'}")
    print(f"overall: {ANALYSIS / f'{stem}_overall.csv'}")
    print(f"by actor: {ANALYSIS / f'{stem}_by_actor.csv'}")
    print(f"by task: {ANALYSIS / f'{stem}_by_task.csv'}")
    print(f"summary: {summary_path}")
    print(overall.to_string(index=False))
    print(by_task.to_string(index=False))
    print(by_actor.to_string(index=False))


if __name__ == "__main__":
    main()
