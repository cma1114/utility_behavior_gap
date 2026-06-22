#!/usr/bin/env python3
"""Consolidated aggregate win-rate CIs for every main-text contrast.

All aggregate (overall and by-task) confidence intervals in the paper are
computed here with a single method: the two-stage equal-cell bootstrap used by
``plot_canonical_highn_highlow.bootstrap_equal_cell``. Each bootstrap iteration
resamples the design cells with replacement and, within each resampled cell,
resamples the paired outcomes with replacement, then averages the per-cell win
rates (equal weight per cell). This propagates both cell-to-cell variance and
within-cell sampling noise, and reproduces the canonical high-low/ceiling CIs
bit-for-bit.

Previously the calibration contrasts mixed methods (t-intervals for the direct
by-task and overall numbers; a coarser task-level bootstrap for the role/harmful
overall numbers). Running everything through this module makes the numbers
consistent and matches the description in Methods 3.4.

Run: ``python -m utility_behavior_gap.scripts.consolidate_aggregate_cis``
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = REPO_ROOT / "outputs" / "analysis"
PAPER_RESULTS = REPO_ROOT / "CURRENT_PAPER" / "results"

TASK_ORDER = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]
BASE_SEED = 20260615
ITERATIONS = 5000

Row = dict[str, str]
ValueFn = Callable[[Row], float | None]


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0"}


def target_win_from_columns(row: Row) -> float | None:
    """For files that already carry ``resolved`` and ``target_win`` columns."""
    if not _truthy(row.get("resolved", "")):
        return None
    return 1.0 if _truthy(row["target_win"]) else 0.0


def winner_is(predicted: str, *, exclude: tuple[str, ...] = ("tie", "unresolved")) -> ValueFn:
    """For files that only carry ``panel_winner_condition``."""

    def value(row: Row) -> float | None:
        winner = row["panel_winner_condition"]
        if winner in exclude:
            return None
        return 1.0 if winner == predicted else 0.0

    return value


class Contrast:
    def __init__(
        self,
        key: str,
        label: str,
        section: str,
        path: Path,
        cell_cols: list[str],
        value_fn: ValueFn,
        row_filter: Callable[[Row], bool] | None = None,
    ) -> None:
        self.key = key
        self.label = label
        self.section = section
        self.path = path
        self.cell_cols = cell_cols
        self.value_fn = value_fn
        self.row_filter = row_filter

    def load(self) -> list[Row]:
        with self.path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if self.row_filter is not None:
            rows = [r for r in rows if self.row_filter(r)]
        return rows


CONTRASTS = [
    Contrast(
        "highlow", "High-low utility", "4.2",
        ANALYSIS / "canonical_highn_condition_results_pair_outcomes.csv",
        ["actor", "task", "domain"], target_win_from_columns,
        row_filter=lambda r: r["condition"] == "utility",
    ),
    Contrast(
        "direct", "Direct effort", "4.3",
        ANALYSIS / "framed_user_strong_panel_pairs.csv",
        ["actor", "task"], winner_is("framed_user_strong"),
    ),
    Contrast(
        "role", "Role (world-class)", "4.4",
        PAPER_RESULTS / "framed_user_prompt_role_pair_outcomes.csv",
        ["actor", "task"], winner_is("user_strong"),
    ),
    Contrast(
        "harmful", "Harmful vs framed-empty", "4.6",
        PAPER_RESULTS / "moral_low_vs_framed_empty_pair_outcomes.csv",
        ["actor", "task"], winner_is("moral_bad", exclude=("tie",)),
    ),
    Contrast(
        "ceiling", "High-utility vs framed-neutral", "4.7",
        PAPER_RESULTS / "high_utility_vs_framed_neutral_pair_outcomes.csv",
        ["actor", "task", "domain"], target_win_from_columns,
    ),
]


def cell_arrays(rows: Iterable[Row], cell_cols: list[str], value_fn: ValueFn) -> list[np.ndarray]:
    """Group resolved pairs into per-cell arrays of 0/1 wins, sorted by cell key.

    Sorting matches ``pandas.groupby(..., sort=True)`` so the bootstrap RNG draws
    line up with the canonical generator.
    """
    cells: dict[tuple[str, ...], list[float]] = {}
    for row in rows:
        value = value_fn(row)
        if value is None:
            continue
        cells.setdefault(tuple(row[c] for c in cell_cols), []).append(value)
    return [np.asarray(cells[key], dtype=float) for key in sorted(cells) if cells[key]]


def bootstrap_equal_cell(arrays: list[np.ndarray], seed: int, iterations: int = ITERATIONS):
    """Two-stage equal-cell bootstrap; mirrors plot_canonical_highn_highlow."""
    if not arrays:
        return float("nan"), float("nan"), float("nan"), 0
    point = float(np.mean([arr.mean() for arr in arrays]))
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for idx in range(iterations):
        sample = rng.integers(0, len(arrays), size=len(arrays))
        rates = [
            float(rng.choice(arrays[int(cell_idx)], size=len(arrays[int(cell_idx)]), replace=True).mean())
            for cell_idx in sample
        ]
        estimates[idx] = float(np.mean(rates))
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return point, float(lo), float(hi), len(arrays)


def compute(contrast: Contrast) -> list[dict[str, object]]:
    rows = contrast.load()
    out: list[dict[str, object]] = []
    # idx 0 = overall; idx 1..4 = by-task in TASK_ORDER, matching the canonical seeding.
    specs = [("overall", None)] + [("task", task) for task in TASK_ORDER]
    for idx, (scope, task) in enumerate(specs):
        subset = rows if task is None else [r for r in rows if r["task"] == task]
        arrays = cell_arrays(subset, contrast.cell_cols, contrast.value_fn)
        point, lo, hi, n_cells = bootstrap_equal_cell(arrays, seed=BASE_SEED + idx)
        out.append(
            {
                "contrast": contrast.key,
                "label": contrast.label,
                "section": contrast.section,
                "scope": scope,
                "task": task or "overall",
                "win_rate": round(point, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "n_cells": n_cells,
                "seed": BASE_SEED + idx,
                "iterations": ITERATIONS,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ANALYSIS / "aggregate_winrate_bootstrap_cis.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    all_rows: list[dict[str, object]] = []
    print(f"| {'Contrast':28s} | {'Scope':12s} | win% | 95% CI | cells |")
    print(f"|{'-'*30}|{'-'*14}|------|--------|-------|")
    for contrast in CONTRASTS:
        if not contrast.path.exists():
            print(f"| {contrast.label:28s} | MISSING FILE: {contrast.path}")
            continue
        rows = compute(contrast)
        all_rows.extend(rows)
        for r in rows:
            scope = r["task"] if r["scope"] == "task" else "overall"
            print(
                f"| {r['label']:28s} | {scope:12s} | "
                f"{r['win_rate']*100:4.1f} | "
                f"[{r['ci_lo']*100:.1f}, {r['ci_hi']*100:.1f}] | {r['n_cells']:>5} |"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
