"""Aggregate generated judge records into the tables used by paper scripts."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from utility_behavior_gap.constants import (
    ACTOR_LABEL,
    ACTOR_LABEL_ORDER,
    ACTORS,
    AMOUNT_TASK_ORDER,
    DOMAIN_LABEL,
    DOMAINS,
    HIGHLOW_TASK_ORDER,
    MORAL_TASK_ORDER,
    TASK_LABEL,
)
from utility_behavior_gap.paths import ANALYSIS, INPUTS, OUTPUT_RAW, PROCESSED
from utility_behavior_gap.stats import parse_entity_count, spearman, wilson_ci


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def as_int(value: str) -> int:
    return int(float(value))


def as_float(value: str) -> float:
    return float(value)


def positive_label(ci_lo: float) -> str:
    return "yes" if ci_lo > 0.50 else "no"


def bool_string(value: bool) -> str:
    return "True" if value else "False"


def comparison_counts(rows: list[dict[str, str]], predicted: str, other: str) -> tuple[int, int, int]:
    n_pred = sum(1 for row in rows if row["counted_winner_condition"] == predicted)
    n_other = sum(1 for row in rows if row["counted_winner_condition"] == other)
    n_tie = sum(1 for row in rows if row["counted_winner_condition"] == "tie")
    return n_pred, n_other, n_tie


def aggregate_highlow_main(judged: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = [row for row in judged if row["comparison"] == "highlow_main"]
    for task_label in HIGHLOW_TASK_ORDER:
        for actor_label in ACTOR_LABEL_ORDER:
            sub = [row for row in source if row["task_label"] == task_label and row["actor_label"] == actor_label]
            n_high, n_low, n_tie = comparison_counts(sub, "high", "low")
            n_excl = n_high + n_low
            rate, lo, hi = wilson_ci(n_high, n_excl)
            rows.append(
                {
                    "task": task_label,
                    "actor": actor_label,
                    "n_high": n_high,
                    "n_low": n_low,
                    "n_tie": n_tie,
                    "n_excl_tie": n_excl,
                    "high_win_rate": round(rate, 4),
                    "ci_lo": round(lo, 4),
                    "ci_hi": round(hi, 4),
                    "ci_positive": positive_label(lo),
                }
            )
    return rows


def aggregate_highlow_same_count(judged: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = [row for row in judged if row["comparison"] == "highlow_same_count"]
    for actor in ACTORS:
        for domain in ["religions", "animals", "countries"]:
            sub = [row for row in source if row["actor"] == actor and row["domain"] == domain]
            n_high, n_low, n_tie = comparison_counts(sub, "high", "low")
            n_excl = n_high + n_low
            rate, lo, hi = wilson_ci(n_high, n_excl)
            rows.append(
                {
                    "domain": DOMAIN_LABEL[domain],
                    "actor": ACTOR_LABEL[actor],
                    "n_high": n_high,
                    "n_low": n_low,
                    "n_tie": n_tie,
                    "n_excl_tie": n_excl,
                    "high_win_rate": round(rate, 4),
                    "ci_lo": round(lo, 4),
                    "ci_hi": round(hi, 4),
                    "ci_positive": positive_label(lo),
                }
            )
    return rows


def aggregate_system_prompt(judged: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = [row for row in judged if row["comparison"] == "system_prompt"]
    for task_label in HIGHLOW_TASK_ORDER:
        for actor_label in ACTOR_LABEL_ORDER:
            sub = [row for row in source if row["task_label"] == task_label and row["actor_label"] == actor_label]
            n_strong, n_normal, n_tie = comparison_counts(sub, "sys_strong", "sys_normal")
            n_excl = n_strong + n_normal
            rate, lo, hi = wilson_ci(n_strong, n_excl)
            rows.append(
                {
                    "task": task_label,
                    "actor": actor_label,
                    "strong_wins": n_strong,
                    "normal_wins": n_normal,
                    "ties": n_tie,
                    "strong_win_rate": round(rate, 4),
                    "ci_low": round(lo, 4),
                    "ci_high": round(hi, 4),
                    "ci_positive": positive_label(lo),
                    "utility_high_side_win_rate": "",
                    "utility_ci_low": "",
                    "utility_ci_high": "",
                    "utility_ci_positive": "",
                }
            )
    return rows


def aggregate_moral(judged: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = [row for row in judged if row["comparison"] == "moral_nolabel"]
    for task in MORAL_TASK_ORDER:
        for actor in ACTORS:
            sub = [row for row in source if row["task"] == task and row["actor"] == actor]
            n_good, n_bad, n_tie = comparison_counts(sub, "moral_good", "moral_bad")
            n_excl = n_good + n_bad
            rate, lo, hi = wilson_ci(n_good, n_excl)
            rows.append(
                {
                    "task": task,
                    "actor": actor,
                    "actor_label": ACTOR_LABEL[actor],
                    "task_label": TASK_LABEL[task],
                    "n_good": n_good,
                    "n_bad": n_bad,
                    "n_tie": n_tie,
                    "n_excl_tie": n_excl,
                    "win_rate": round(rate, 4),
                    "ci_lo": round(lo, 4),
                    "ci_hi": round(hi, 4),
                    "ci_positive": bool_string(lo > 0.50),
                }
            )
    return rows


def aggregate_amount(judged: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = [row for row in judged if row["comparison"] == "amount"]
    for task in AMOUNT_TASK_ORDER:
        for actor in ACTORS:
            sub = [row for row in source if row["task"] == task and row["actor"] == actor]
            n_high, n_low, n_tie = comparison_counts(sub, "amount_high", "amount_low")
            n_excl = n_high + n_low
            rate, lo, hi = wilson_ci(n_high, n_excl)
            rows.append(
                {
                    "task": task,
                    "actor": actor,
                    "condition": "amount",
                    "n_left_wins": n_high,
                    "n_right_wins": n_low,
                    "n_ties": n_tie,
                    "n_excl_tie": n_excl,
                    "left_winrate_excl_tie": round(rate, 4),
                    "ci_lo": round(lo, 4),
                    "ci_hi": round(hi, 4),
                    "positive_ci_excludes_50": bool_string(lo > 0.50),
                    "actor_label": ACTOR_LABEL[actor],
                    "task_label": "Essay" if task == "essay" else TASK_LABEL[task],
                    "condition_label": "Larger amount",
                    "panel_note": "$1M side wins over $100",
                    "win_rate": round(rate, 4),
                }
            )
    return rows


def build_dose_response_trials(judged: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in judged:
        if row["comparison"] != "highlow_main":
            continue
        winner = row["counted_winner_condition"]
        if winner not in {"high", "low", "tie"}:
            continue
        rows.append(
            {
                "actor": row["actor"],
                "task": row["task"],
                "domain": row["domain"],
                "item_id": row["item_id"],
                "framing": row["framing"],
                "high_text": row["high_description"],
                "low_text": row["low_description"],
                "u_high": row["high_utility"],
                "u_low": row["low_utility"],
                "delta_u": row["delta_u"],
                "winner": winner,
                "high_win": "" if winner == "tie" else (1 if winner == "high" else 0),
            }
        )
    return rows


def build_utility_replication(utility_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_actor_domain: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in utility_rows:
        by_actor_domain[(row["actor"], row["domain"])].append(row)

    rows: list[dict[str, Any]] = []
    for actor in ACTORS:
        out: dict[str, Any] = {"actor": actor}
        for domain in DOMAINS:
            options = by_actor_domain[(actor, domain)]
            out[f"{domain}_holdout_accuracy"] = round(as_float(options[0]["holdout_accuracy"]), 4)
            if domain == "political":
                continue
            grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
            parsed = 0
            for option in options:
                parsed_entity = parse_entity_count(option["description"], domain)
                if parsed_entity is None:
                    continue
                parsed += 1
                entity, count = parsed_entity
                grouped[entity].append((count, as_float(option["utility_mean"])))
            per_entity = []
            for pairs in grouped.values():
                if len(pairs) < 3:
                    continue
                rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
                if rho is not None:
                    per_entity.append(round(rho, 4))
            out[f"{domain}_entity_mean_spearman"] = round(sum(per_entity) / len(per_entity), 4)
            out[f"{domain}_entities"] = len(per_entity)
        rows.append(out)
    return rows


def build_utility_top_bottom(utility_rows: list[dict[str, str]], k: int = 10) -> list[dict[str, Any]]:
    by_actor_domain: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in utility_rows:
        by_actor_domain[(row["actor"], row["domain"])].append(row)

    rows: list[dict[str, Any]] = []
    for actor in ACTORS:
        for domain in DOMAINS:
            options = sorted(
                by_actor_domain[(actor, domain)],
                key=lambda row: as_float(row["utility_mean"]),
                reverse=True,
            )
            n = len(options)
            for idx, option in enumerate(options[:k], start=1):
                rows.append(
                    {
                        "actor": actor,
                        "actor_label": ACTOR_LABEL[actor],
                        "domain": domain,
                        "domain_label": DOMAIN_LABEL[domain],
                        "side": "top",
                        "rank_within_side": idx,
                        "utility_rank_overall": idx,
                        "utility": option["utility_mean"],
                        "description": option["description"],
                    }
                )
            for idx, option in enumerate(reversed(options[-k:]), start=1):
                rows.append(
                    {
                        "actor": actor,
                        "actor_label": ACTOR_LABEL[actor],
                        "domain": domain,
                        "domain_label": DOMAIN_LABEL[domain],
                        "side": "bottom",
                        "rank_within_side": idx,
                        "utility_rank_overall": n - idx + 1,
                        "utility": option["utility_mean"],
                        "description": option["description"],
                    }
                )
    return rows


def aggregate_all(root: Path) -> None:
    processed = PROCESSED
    analysis = ANALYSIS
    judged_path = OUTPUT_RAW / "judged_pairs.csv"
    if not judged_path.exists():
        raise FileNotFoundError(
            "Run live generation, live judging, and "
            "`python -m utility_behavior_gap.scripts.aggregate_judgments` first."
        )
    judged = read_rows(judged_path)
    utility = read_rows(INPUTS / "utility_options.csv")

    write_rows(processed / "highlow_main_data.csv", aggregate_highlow_main(judged))
    write_rows(processed / "highlow_within_count_data.csv", aggregate_highlow_same_count(judged))
    write_rows(processed / "system_prompt_calibration_data.csv", aggregate_system_prompt(judged))
    write_rows(processed / "moral_nolabel_main_data.csv", aggregate_moral(judged))
    write_rows(processed / "incentive_channel_data.csv", aggregate_amount(judged))
    write_rows(analysis / "utility_gap_dose_response_trials.csv", build_dose_response_trials(judged))
    write_rows(analysis / "utility_replication_diagnostics.csv", build_utility_replication(utility))
    write_rows(analysis / "utility_top_bottom_10_by_actor_domain.csv", build_utility_top_bottom(utility))
