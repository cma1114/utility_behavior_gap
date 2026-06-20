#!/usr/bin/env python3
"""Compute cluster-aware confidence intervals for essay trial-level data.

The paper's win-rate denominator excludes ties and judge-panel disagreements.
This script keeps that convention, but avoids treating repeated trials as
independent evidence about the aggregate effect.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from utility_behavior_gap.paths import ANALYSIS, ROOT
from utility_behavior_gap.stats import wilson_ci


ESSAY_DATA = ROOT / "essay_all_conditions"
SUMMARY_CSV = ANALYSIS / "essay_trial_level_ci_summary.csv"
ACTOR_CSV = ANALYSIS / "essay_trial_level_actor_rates.csv"
CELL_CSV = ANALYSIS / "essay_trial_level_primary_cell_rates.csv"
NOTE_MD = ROOT / "notes" / "essay_trial_level_ci_analysis.md"

CONDITION_ORDER = ["direct", "highlow", "moral", "amount"]
PREDICTED_SIDE = {
    "direct": "strong system prompt",
    "highlow": "high-utility outcome",
    "moral": "pro-social cause",
    "amount": "larger donation amount",
}

# Two-sided 95% t critical values for the small actor-level descriptive CI.
T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def read_trials(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*/*.json")):
        if path.parent.name not in CONDITION_ORDER:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition = str(payload["condition"])
        actor_id = path.stem
        for idx, trial in enumerate(payload["trials"]):
            row = dict(trial)
            row["condition"] = condition
            row["actor_id"] = actor_id
            row["trial_index"] = idx
            row["domain"] = row.get("domain") or ""
            row["note"] = row.get("note") or ""
            row["win_value"] = {"A": 1, "B": 0}.get(row.get("winner_arm"))
            row["item_key"] = item_key(row)
            rows.append(row)
    return rows


def item_key(row: dict[str, Any]) -> str:
    condition = row["condition"]
    if condition == "amount":
        return str(row["essay_topic"])
    if condition == "moral":
        return " | ".join([str(row["essay_topic"]), str(row["arm_A"]), str(row["arm_B"])])
    if condition == "direct":
        # Direct has domain/topic design cells plus repeated identical-outcome
        # trials; the note distinguishes those repeated outcome draws.
        return " | ".join(
            [
                str(row["domain"]),
                str(row["essay_topic"]),
                str(row["note"]),
            ]
        )
    if condition == "highlow":
        return " | ".join(
            [
                str(row["domain"]),
                str(row["essay_topic"]),
                str(row["arm_A"]),
                str(row["arm_B"]),
            ]
        )
    raise ValueError(f"unknown condition: {condition}")


def second_cluster(row: dict[str, Any]) -> str:
    """Return the non-actor cluster for the primary crossed bootstrap."""

    if row["condition"] in {"direct", "highlow"}:
        return str(row["domain"])
    return str(row["item_key"])


def primary_cell_label(condition: str) -> str:
    if condition in {"direct", "highlow"}:
        return "actor_x_domain"
    return "actor_x_item"


def condition_rows(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["condition"] == condition]


def valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("win_value") in {0, 1}]


def ratio(wins: int, losses: int) -> float:
    return wins / (wins + losses) if wins + losses else float("nan")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - idx) + sorted_values[hi] * (idx - lo)


def rounded(value: float, digits: int = 4) -> float:
    if math.isnan(value):
        return value
    return round(value, digits)


def cell_rates(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    valid = valid_rows(rows)
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in valid:
        grouped[(str(row["actor"]), second_cluster(row))].append(int(row["win_value"]))

    cells: list[dict[str, Any]] = []
    for (actor, cluster), values in sorted(grouped.items()):
        wins = sum(values)
        losses = len(values) - wins
        cells.append(
            {
                "condition": condition,
                "primary_cell": primary_cell_label(condition),
                "actor": actor,
                "cluster": cluster,
                "n_valid": len(values),
                "wins_A": wins,
                "wins_B": losses,
                "win_rate_A": wins / len(values),
            }
        )
    return cells


def crossed_cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    reps: int,
    seed: int,
) -> tuple[float, float, float]:
    valid = valid_rows(rows)
    actors = sorted({str(row["actor"]) for row in valid})
    clusters = sorted({second_cluster(row) for row in valid})

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in valid:
        grouped[(str(row["actor"]), second_cluster(row))].append(int(row["win_value"]))

    observed_rates = []
    for values in grouped.values():
        observed_rates.append(sum(values) / len(values))
    observed = mean(observed_rates)

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(reps):
        sampled_actors = [rng.choice(actors) for _ in actors]
        sampled_clusters = [rng.choice(clusters) for _ in clusters]
        rates = []
        for actor in sampled_actors:
            for cluster in sampled_clusters:
                values = grouped.get((actor, cluster))
                if not values:
                    continue
                wins = sum(rng.choice(values) for _ in values)
                rates.append(wins / len(values))
        if rates:
            estimates.append(mean(rates))

    return observed, percentile(estimates, 0.025), percentile(estimates, 0.975)


def actor_summaries(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    out = []
    for actor in sorted({str(row["actor"]) for row in rows}):
        sub = [row for row in rows if str(row["actor"]) == actor]
        counts = Counter(str(row["winner_arm"]) for row in sub)
        wins = counts["A"]
        losses = counts["B"]
        rate = ratio(wins, losses)
        out.append(
            {
                "condition": condition,
                "actor": actor,
                "n_rows": len(sub),
                "wins_A": wins,
                "wins_B": losses,
                "ties": counts["TIE"],
                "disagreements": counts["disagree"],
                "n_valid": wins + losses,
                "win_rate_A": rounded(rate),
            }
        )
    return out


def actor_t_ci(actor_rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    rates = [float(row["win_rate_A"]) for row in actor_rows if row["n_valid"]]
    if len(rates) < 2:
        return float("nan"), float("nan"), float("nan")
    estimate = mean(rates)
    df = len(rates) - 1
    tcrit = T_CRIT_95.get(df, 1.96)
    se = stdev(rates) / math.sqrt(len(rates))
    return estimate, estimate - tcrit * se, estimate + tcrit * se


def summarize_condition(
    rows: list[dict[str, Any]],
    condition: str,
    *,
    reps: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    sub = condition_rows(rows, condition)
    counts = Counter(str(row["winner_arm"]) for row in sub)
    wins = counts["A"]
    losses = counts["B"]
    pooled_rate, naive_lo, naive_hi = wilson_ci(wins, wins + losses)

    actors = actor_summaries(sub, condition)
    actor_mean, actor_lo, actor_hi = actor_t_ci(actors)
    primary_est, boot_lo, boot_hi = crossed_cluster_bootstrap(
        sub,
        reps=reps,
        seed=seed + CONDITION_ORDER.index(condition),
    )
    cells = cell_rates(sub, condition)

    summary = {
        "condition": condition,
        "predicted_A_side": PREDICTED_SIDE[condition],
        "primary_cell": primary_cell_label(condition),
        "bootstrap_reps": reps,
        "n_rows": len(sub),
        "n_valid": wins + losses,
        "wins_A": wins,
        "wins_B": losses,
        "ties": counts["TIE"],
        "disagreements": counts["disagree"],
        "pooled_win_rate_A": rounded(pooled_rate),
        "naive_wilson_lo": rounded(naive_lo),
        "naive_wilson_hi": rounded(naive_hi),
        "equal_actor_mean": rounded(actor_mean),
        "actor_t_lo": rounded(actor_lo),
        "actor_t_hi": rounded(actor_hi),
        "primary_cluster_mean": rounded(primary_est),
        "cluster_boot_lo": rounded(boot_lo),
        "cluster_boot_hi": rounded(boot_hi),
        "n_primary_cells": len(cells),
        "n_actors": len({row["actor"] for row in sub}),
        "n_second_clusters": len({second_cluster(row) for row in valid_rows(sub)}),
    }
    return summary, actors, cells


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value) * 100:.1f}%"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Condition | A side | A/B/TIE/disagree | Pooled | Primary cluster mean | Cluster-aware 95% CI | Naive Wilson 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        counts = f"{row['wins_A']} / {row['wins_B']} / {row['ties']} / {row['disagreements']}"
        lines.append(
            "| {condition} | {side} | {counts} | {pooled} | {primary} | {boot_lo}-{boot_hi} | {naive_lo}-{naive_hi} |".format(
                condition=row["condition"],
                side=row["predicted_A_side"],
                counts=counts,
                pooled=pct(row["pooled_win_rate_A"]),
                primary=pct(row["primary_cluster_mean"]),
                boot_lo=pct(row["cluster_boot_lo"]),
                boot_hi=pct(row["cluster_boot_hi"]),
                naive_lo=pct(row["naive_wilson_lo"]),
                naive_hi=pct(row["naive_wilson_hi"]),
            )
        )
    return "\n".join(lines)


def write_note(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    table = markdown_table(summary_rows)
    text = f"""# Essay Trial-Level Confidence Intervals

This note documents the confidence-interval calculation for the trial-level essay
data in `essay_all_conditions/`.

## Input

The analysis reads the per-model JSON files under `essay_all_conditions/<condition>/`.
Those files include the trial inputs, both generated essays, the three judge votes,
the majority vote, and the decoded `winner_arm`. The consolidated CSV is useful
for checking, but the script reads JSON so condition-specific item labels such as
direct-condition outcome notes remain available.

## Estimand

For each condition, arm A is the paper-predicted side:

- `direct`: strong system prompt.
- `highlow`: high-utility outcome.
- `moral`: pro-social cause.
- `amount`: larger donation amount.

The primary estimand is the A-side win probability conditional on a resolved A/B
panel result. `TIE` and `disagree` rows are reported, but excluded from the win-rate
denominator to match the paper's convention.

The primary aggregate gives equal weight to repeated design cells rather than to
raw rows:

- `direct` and `highlow`: mean over actor x domain cells.
- `moral` and `amount`: mean over actor x item cells. For `amount`, the item is
  the essay topic; for `moral`, the item is the topic plus the pro-social/harmful
  cause pair.

## Confidence Interval

The primary CI is a nonparametric crossed-cluster bootstrap with a fixed seed.
Actors are always resampled as one clustering dimension. The second clustering
dimension is domain for `direct` and `highlow`, and item for `moral` and `amount`.
Within each selected actor x cluster cell, resolved A/B outcomes are resampled
with replacement. Each bootstrap replicate averages the selected cell win rates.

This is deliberately wider than a row-level Wilson interval because the raw rows
are repeated measurements within actors, domains, topics, causes, and/or repeated
temperature samples. The row-level Wilson CI is included only as a diagnostic for
how much the independent-trial assumption would shrink uncertainty.

## Results

{table}

## Output Files

- `outputs/analysis/essay_trial_level_ci_summary.csv`
- `outputs/analysis/essay_trial_level_actor_rates.csv`
- `outputs/analysis/essay_trial_level_primary_cell_rates.csv`

## Command

```bash
python -m utility_behavior_gap.scripts.analyze_essay_trial_level_cis
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ESSAY_DATA)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    rows = read_trials(args.data_dir)
    summary_rows: list[dict[str, Any]] = []
    actor_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        summary, actors, cells = summarize_condition(
            rows,
            condition,
            reps=args.bootstrap_reps,
            seed=args.seed,
        )
        summary_rows.append(summary)
        actor_rows.extend(actors)
        cell_rows.extend(cells)

    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(ACTOR_CSV, actor_rows)
    write_csv(CELL_CSV, cell_rows)
    write_note(NOTE_MD, summary_rows)

    print(f"wrote {len(summary_rows)} summary rows to {SUMMARY_CSV}")
    print(f"wrote {len(actor_rows)} actor rows to {ACTOR_CSV}")
    print(f"wrote {len(cell_rows)} primary-cell rows to {CELL_CSV}")
    print(f"wrote methods note to {NOTE_MD}")
    for row in summary_rows:
        print(
            "{condition}: primary={primary} CI={lo}-{hi} pooled={pooled}".format(
                condition=row["condition"],
                primary=pct(row["primary_cluster_mean"]),
                lo=pct(row["cluster_boot_lo"]),
                hi=pct(row["cluster_boot_hi"]),
                pooled=pct(row["pooled_win_rate_A"]),
            )
        )


if __name__ == "__main__":
    main()
