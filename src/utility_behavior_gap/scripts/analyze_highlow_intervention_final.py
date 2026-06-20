#!/usr/bin/env python3
"""Final reproducible analysis for the cleaned high-low intervention batch."""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utility_behavior_gap.constants import ACTOR_LABEL, ACTORS, DOMAIN_LABEL, TASK_LABEL
from utility_behavior_gap.io_utils import read_csv_rows, read_jsonl, write_csv_rows
from utility_behavior_gap.paths import ANALYSIS, FIGURES, OUTPUT_API, OUTPUT_RAW


COMPARISON = "highlow_intervention"
TASK_ORDER = ["essay", "translation", "incident_postmortem", "grant_proposal_abstract"]
PLOT_TASK_ORDER = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]
DOMAIN_ORDER = ["religions", "animals", "countries", "political"]
DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_SEED = 20260608
RUNS_DIR = OUTPUT_API / "runs"
FAMILY_ALPHA = 0.05

INK = "#1A1A1F"
SUBTLE = "#6B7280"
GRID = "#E9EDF5"
CHANCE_LINE = "#9CA3AF"
CI_POS_BAND = "#E8F4ED"
CI_NEG_BAND = "#F8E7E7"

MODEL_COLORS = {
    "deepseek-v3.2-or": "#2A8C9E",
    "gpt-5.4-mini-or": "#3A66C9",
    "glm-5.1-or": "#5B6068",
    "kimi-k2.5-or": "#D4711B",
    "mimo-v25-pro-or": "#2E8C5C",
    "qwen3.5-9b-or": "#6E45BD",
    "qwen3.6-plus-or": "#C2304A",
}

TASK_COLORS = {
    "essay": "#3A66C9",
    "translation": "#2A8C9E",
    "incident_postmortem": "#D4711B",
    "grant_proposal_abstract": "#2E8C5C",
}


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()


def wilson(wins: int, total: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if total == 0:
        return (float("nan"),) * 3
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def bonferroni_exact_ci(wins: int, total: int, family_size: int, alpha: float = FAMILY_ALPHA) -> tuple[float, float]:
    """Exact binomial simultaneous CI using Bonferroni over the plotted family."""
    if total == 0:
        return float("nan"), float("nan")
    tail_alpha = alpha / (2 * family_size)
    lo = 0.0 if wins == 0 else float(beta.ppf(tail_alpha, wins, total - wins + 1))
    hi = 1.0 if wins == total else float(beta.ppf(1.0 - tail_alpha, wins + 1, total - wins))
    return lo, hi


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def highlow_pair_paths() -> list[Path]:
    return sorted(OUTPUT_RAW.glob(f"{COMPARISON}__*__*__judged_pairs.csv"))


def highlow_vote_paths() -> list[Path]:
    return sorted(OUTPUT_RAW.glob(f"{COMPARISON}__*__*__judge_votes.csv"))


def parse_pair_row(row: dict[str, str], source_file: Path) -> dict[str, Any] | None:
    if row.get("comparison") != COMPARISON:
        return None
    winner = (row.get("counted_winner_condition") or row.get("panel_winner_condition") or "").strip()
    if winner not in {"high", "low", "tie"}:
        return None
    try:
        high_utility = float(row["high_utility"])
        low_utility = float(row["low_utility"])
        delta_u = float(row["delta_u"])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "source_file": str(source_file),
        "pair_uid": row["pair_uid"],
        "actor": row["actor"],
        "actor_label": row.get("actor_label") or ACTOR_LABEL.get(row["actor"], row["actor"]),
        "task": row["task"],
        "task_label": row.get("task_label") or TASK_LABEL.get(row["task"], row["task"]),
        "domain": row["domain"],
        "domain_label": row.get("domain_label") or DOMAIN_LABEL.get(row["domain"], row["domain"]),
        "pair_idx": int(row["pair_idx"]),
        "item_id": row["item_id"],
        "item_label": row["item_label"],
        "winner": winner,
        "high_win_excluding_ties": 1 if winner == "high" else (0 if winner == "low" else np.nan),
        "high_win_ties_half": 1.0 if winner == "high" else (0.0 if winner == "low" else 0.5),
        "high_utility": high_utility,
        "low_utility": low_utility,
        "delta_u": delta_u,
        "high_description": row.get("high_description", ""),
        "low_description": row.get("low_description", ""),
        "high_consequence": row.get("high_consequence", ""),
        "low_consequence": row.get("low_consequence", ""),
        "utility_pair_cluster": "|".join(
            [
                row["actor"],
                row["domain"],
                row["pair_idx"],
                row.get("high_description", ""),
                row.get("low_description", ""),
            ]
        ),
    }


def load_pairs() -> tuple[pd.DataFrame, list[Path]]:
    rows: list[dict[str, Any]] = []
    used_paths: list[Path] = []
    for path in highlow_pair_paths():
        file_rows = read_csv_rows(path)
        parsed = [parsed for row in file_rows if (parsed := parse_pair_row(row, path)) is not None]
        if parsed:
            rows.extend(parsed)
            used_paths.append(path)
    if not rows:
        raise ValueError("No cleaned high-low judged-pair snapshots found.")
    df = pd.DataFrame(rows)
    df["pair_idx"] = df["pair_idx"].astype(int)
    df["high_win_excluding_ties"] = pd.to_numeric(df["high_win_excluding_ties"], errors="coerce")
    df["high_win_ties_half"] = df["high_win_ties_half"].astype(float)
    return df, used_paths


def load_votes() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in highlow_vote_paths():
        for row in read_csv_rows(path):
            winner = row.get("winner_condition", "")
            if winner not in {"high", "low", "tie", "unresolved"}:
                continue
            rows.append(
                {
                    "source_file": str(path),
                    "pair_uid": row["pair_uid"],
                    "judge_index": int(row["judge_index"]),
                    "judge_model": row["judge_model"],
                    "winner_condition": winner,
                    "high_vote": 1 if winner == "high" else (0 if winner == "low" else np.nan),
                    "high_vote_ties_half": 1.0 if winner == "high" else (0.0 if winner == "low" else 0.5),
                    "success": row.get("success", ""),
                }
            )
    return pd.DataFrame(rows)


def native_generation_finish_reason(generation: dict[str, Any]) -> str:
    try:
        return str(generation["raw_response"]["choices"][0].get("native_finish_reason") or "")
    except (KeyError, IndexError, TypeError):
        return ""


def generation_completion_tokens(generation: dict[str, Any]) -> int | None:
    usage = generation.get("usage") or generation.get("raw_response", {}).get("usage") or {}
    value = usage.get("completion_tokens")
    return int(value) if isinstance(value, int | float) else None


def generation_max_tokens(generation: dict[str, Any]) -> int | None:
    value = generation.get("max_tokens") or generation.get("request", {}).get("max_tokens")
    return int(value) if isinstance(value, int | float) else None


def run_dir_jobs(run_dir: Path) -> list[dict[str, Any]]:
    jobs_path = run_dir / "generation_jobs.jsonl"
    if not jobs_path.exists():
        return []
    try:
        return read_jsonl(jobs_path)
    except (OSError, ValueError):
        return []


def matching_generation_run(actor: str, task: str) -> Path | None:
    matches: list[Path] = []
    if not RUNS_DIR.exists():
        return None
    for run_dir in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
        jobs = run_dir_jobs(run_dir)
        if not jobs:
            continue
        if {str(job.get("comparison")) for job in jobs} != {COMPARISON}:
            continue
        if {str(job.get("actor")) for job in jobs} != {actor}:
            continue
        if {str(job.get("task")) for job in jobs} != {task}:
            continue
        matches.append(run_dir)
    if not matches:
        return None
    return max(matches, key=lambda path: ((path / "generation_jobs.jsonl").stat().st_mtime, path.name))


def generation_quality_scan(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    run_dirs: list[Path] = []
    expected_cells = len(ACTORS) * len(TASK_ORDER)
    expected_generations = expected_cells * 320 * 2

    for actor, task in sorted(df[["actor", "task"]].drop_duplicates().itertuples(index=False, name=None)):
        run_dir = matching_generation_run(actor, task)
        if run_dir is None:
            rows.append(
                {
                    "check": f"generation_run_dir:{actor}:{task}",
                    "observed": "missing",
                    "expected": "present",
                }
            )
            continue
        run_dirs.append(run_dir)
        generations_path = run_dir / "generations.jsonl"
        failures_path = run_dir / "generation_failures.jsonl"
        generations = read_jsonl(generations_path) if generations_path.exists() else []
        failures = read_jsonl(failures_path) if failures_path.exists() else []
        rows.append(
            {
                "check": f"cell_generations:{actor}:{task}",
                "observed": len(generations),
                "expected": 640,
            }
        )
        rows.append(
            {
                "check": f"cell_generation_retry_failures_logged:{actor}:{task}",
                "observed": len(failures),
                "expected": len(failures),
            }
        )

    generations_all: list[dict[str, Any]] = []
    failures_all: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        generations_path = run_dir / "generations.jsonl"
        failures_path = run_dir / "generation_failures.jsonl"
        if generations_path.exists():
            generations_all.extend(read_jsonl(generations_path))
        if failures_path.exists():
            failures_all.extend(read_jsonl(failures_path))

    raw_explicit_truncation = 0
    raw_non_stop = 0
    raw_degenerate_tiny = 0
    near_95 = 0
    near_98 = 0
    for generation in generations_all:
        finish_reason = str(generation.get("finish_reason") or "")
        native_finish_reason = native_generation_finish_reason(generation)
        output_text = str(generation.get("output_text") or "")
        output_tiny = len(output_text.strip()) < 20
        output_non_stop = finish_reason != "stop"
        output_truncated = finish_reason.lower() in {"length", "max_tokens", "max_tokens_exceeded"} or native_finish_reason.lower() in {
            "length",
            "max_tokens",
            "max_tokens_exceeded",
        }
        if finish_reason != "stop":
            raw_non_stop += 1
        if output_truncated:
            raw_explicit_truncation += 1
        completion_tokens = generation_completion_tokens(generation)
        max_tokens = generation_max_tokens(generation)
        if completion_tokens is not None and max_tokens is not None and max_tokens > 0:
            ratio = completion_tokens / max_tokens
            near_95 += int(ratio >= 0.95)
            near_98 += int(ratio >= 0.98)
        if output_tiny:
            raw_degenerate_tiny += 1
        if output_non_stop or output_truncated or output_tiny:
            invalid_rows.append(
                {
                    "pair_uid": generation.get("pair_uid", ""),
                    "output_id": generation.get("output_id", ""),
                    "actor": generation.get("actor", ""),
                    "task": generation.get("job", {}).get("task", ""),
                    "domain": generation.get("job", {}).get("domain", ""),
                    "condition": generation.get("condition", ""),
                    "run_id": generation.get("run_id", ""),
                    "finish_reason": finish_reason,
                    "native_finish_reason": native_finish_reason,
                    "completion_tokens": completion_tokens if completion_tokens is not None else "",
                    "max_tokens": max_tokens if max_tokens is not None else "",
                    "chars": len(output_text),
                    "non_stop_finish_reason": int(output_non_stop),
                    "explicit_length_truncation": int(output_truncated),
                    "degenerate_chars_lt_20": int(output_tiny),
                    "output_preview": output_text[:200],
                }
            )

    rows.extend(
        [
            {"check": "generation_run_dirs", "observed": len(run_dirs), "expected": expected_cells},
            {"check": "generation_rows", "observed": len(generations_all), "expected": expected_generations},
            {
                "check": "unique_generation_output_id",
                "observed": len({str(row.get("output_id")) for row in generations_all}),
                "expected": expected_generations,
            },
            {
                "check": "generation_retry_failures_logged",
                "observed": len(failures_all),
                "expected": len(failures_all),
            },
            {"check": "raw_non_stop_generation_finish_reason_reported", "observed": raw_non_stop, "expected": raw_non_stop},
            {"check": "raw_explicit_length_truncation", "observed": raw_explicit_truncation, "expected": 0},
            {"check": "raw_degenerate_generation_chars_lt_20_reported", "observed": raw_degenerate_tiny, "expected": raw_degenerate_tiny},
            {
                "check": "invalid_generation_outputs_excluded",
                "observed": len(invalid_rows),
                "expected": len(invalid_rows),
            },
            {"check": "near_token_cap_95_reported", "observed": near_95, "expected": near_95},
            {"check": "near_token_cap_98_reported", "observed": near_98, "expected": near_98},
        ]
    )
    return rows, invalid_rows


def audit_rows(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    votes: pd.DataFrame,
    used_paths: list[Path],
    generation_audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_cells = len(ACTORS) * len(TASK_ORDER)
    expected_pairs = expected_cells * 320
    expected_analyzed_pairs = len(df)
    expected_votes = expected_analyzed_pairs * 3
    rows.append({"check": "judged_pair_snapshot_files", "observed": len(used_paths), "expected": expected_cells})
    rows.append({"check": "raw_judged_pairs", "observed": len(raw_df), "expected": expected_pairs})
    rows.append({"check": "raw_unique_pair_uid", "observed": raw_df["pair_uid"].nunique(), "expected": expected_pairs})
    rows.append(
        {
            "check": "excluded_pairs_with_invalid_generation",
            "observed": len(raw_df) - len(df),
            "expected": len(raw_df) - len(df),
        }
    )
    rows.append({"check": "analyzed_judged_pairs", "observed": len(df), "expected": expected_analyzed_pairs})
    rows.append({"check": "analyzed_unique_pair_uid", "observed": df["pair_uid"].nunique(), "expected": expected_analyzed_pairs})
    rows.append({"check": "analyzed_judge_votes", "observed": len(votes), "expected": expected_votes})
    if not votes.empty:
        rows.append(
            {
                "check": "analyzed_unique_pair_uid_x_judge_model",
                "observed": votes[["pair_uid", "judge_model"]].drop_duplicates().shape[0],
                "expected": expected_votes,
            }
        )
    invalid_pair_counts = (
        raw_df.loc[~raw_df["pair_uid"].isin(set(df["pair_uid"]))]
        .groupby(["actor", "task"])
        .size()
        .to_dict()
    )
    invalid_domain_counts = (
        raw_df.loc[~raw_df["pair_uid"].isin(set(df["pair_uid"]))]
        .groupby(["actor", "task", "domain"])
        .size()
        .to_dict()
    )
    for (actor, task), sub in raw_df.groupby(["actor", "task"], sort=True):
        rows.append(
            {
                "check": f"cell_pairs_raw:{actor}:{task}",
                "observed": len(sub),
                "expected": 320,
            }
        )
    for (actor, task), sub in df.groupby(["actor", "task"], sort=True):
        rows.append(
            {
                "check": f"cell_pairs_analyzed:{actor}:{task}",
                "observed": len(sub),
                "expected": 320 - int(invalid_pair_counts.get((actor, task), 0)),
            }
        )
    for (actor, task, domain), sub in raw_df.groupby(["actor", "task", "domain"], sort=True):
        rows.append(
            {
                "check": f"cell_domain_pairs_raw:{actor}:{task}:{domain}",
                "observed": len(sub),
                "expected": 80,
            }
        )
    for (actor, task, domain), sub in df.groupby(["actor", "task", "domain"], sort=True):
        rows.append(
            {
                "check": f"cell_domain_pairs_analyzed:{actor}:{task}:{domain}",
                "observed": len(sub),
                "expected": 80 - int(invalid_domain_counts.get((actor, task, domain), 0)),
            }
        )
    rows.extend(generation_audit)
    return rows


def count_winners(sub: pd.DataFrame) -> dict[str, int]:
    counts = sub["winner"].value_counts()
    return {
        "n_pairs": int(len(sub)),
        "high_wins": int(counts.get("high", 0)),
        "low_wins": int(counts.get("low", 0)),
        "ties": int(counts.get("tie", 0)),
    }


def summarize_group(df: pd.DataFrame, group_cols: list[str], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, sort=True)
    for key, sub in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        counts = count_winners(sub)
        non_tied = counts["high_wins"] + counts["low_wins"]
        pooled_rate, pooled_lo, pooled_hi = wilson(counts["high_wins"], non_tied)
        tie_half_rate = (
            (counts["high_wins"] + 0.5 * counts["ties"]) / counts["n_pairs"] if counts["n_pairs"] else float("nan")
        )
        row: dict[str, Any] = {
            "summary_level": label,
            **{col: value for col, value in zip(group_cols, key_tuple, strict=False)},
            **counts,
            "n_non_tied": non_tied,
            "pooled_high_win_rate_excluding_ties": pooled_rate,
            "pooled_wilson_ci_lo": pooled_lo,
            "pooled_wilson_ci_hi": pooled_hi,
            "pooled_high_win_rate_ties_half": tie_half_rate,
            "mean_delta_u": float(sub["delta_u"].mean()),
        }
        rows.append(row)
    return rows


def add_actor_task_multiplicity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    p_values: list[float] = []
    family_size = len(rows)
    for row in rows:
        wins = int(row["high_wins"])
        total = int(row["n_non_tied"])
        p_values.append(binomtest(wins, total, 0.5, alternative="two-sided").pvalue)
    adjusted = holm_adjust(p_values)
    for row, p_value, holm_p in zip(rows, p_values, adjusted, strict=True):
        wins = int(row["high_wins"])
        total = int(row["n_non_tied"])
        ci_lo, ci_hi = bonferroni_exact_ci(wins, total, family_size)
        rate = float(row["pooled_high_win_rate_excluding_ties"])
        row["p_two_sided_exact"] = p_value
        row["holm_p_two_sided"] = holm_p
        row["holm_positive"] = bool(rate > 0.5 and holm_p < 0.05)
        row["holm_negative"] = bool(rate < 0.5 and holm_p < 0.05)
        row["familywise_ci_method"] = (
            f"Bonferroni exact binomial 95% familywise CI across {family_size} actor-task cells"
        )
        row["familywise_ci_lo"] = ci_lo
        row["familywise_ci_hi"] = ci_hi
        row["familywise_ci_positive"] = bool(ci_lo > 0.5)
        row["familywise_ci_negative"] = bool(ci_hi < 0.5)
    return rows


def stratum_means(df: pd.DataFrame, outcome_col: str, *, drop_na: bool) -> pd.Series:
    values = df.dropna(subset=[outcome_col]) if drop_na else df
    return values.groupby(["actor", "task", "domain"], sort=True)[outcome_col].mean()


def balanced_point_estimate(df: pd.DataFrame, outcome_col: str, *, drop_na: bool) -> float:
    return float(stratum_means(df, outcome_col, drop_na=drop_na).mean())


def bootstrap_balanced_mean(
    df: pd.DataFrame,
    outcome_col: str,
    *,
    drop_na: bool,
    n_bootstrap: int,
    seed: int,
    resample_domains: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = df.dropna(subset=[outcome_col]) if drop_na else df
    actors = sorted(values["actor"].unique())
    tasks = [task for task in TASK_ORDER if task in set(values["task"])]
    domains = [domain for domain in DOMAIN_ORDER if domain in set(values["domain"])]
    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    for key, sub in values.groupby(["actor", "task", "domain"], sort=True):
        arrays[key] = sub[outcome_col].to_numpy(dtype=float)

    point = float(np.mean([array.mean() for array in arrays.values()]))
    estimates = np.empty(n_bootstrap)
    fixed_domains = np.array(domains, dtype=object)
    for i in range(n_bootstrap):
        sampled_actors = rng.choice(actors, size=len(actors), replace=True)
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        sampled_domains = (
            rng.choice(domains, size=len(domains), replace=True) if resample_domains else fixed_domains
        )
        cell_rates: list[float] = []
        for actor in sampled_actors:
            for task in sampled_tasks:
                for domain in sampled_domains:
                    array = arrays.get((str(actor), str(task), str(domain)))
                    if array is None or len(array) == 0:
                        continue
                    sample = rng.choice(array, size=len(array), replace=True)
                    cell_rates.append(float(sample.mean()))
        estimates[i] = float(np.mean(cell_rates))

    return {
        "point": point,
        "ci_lo": float(np.quantile(estimates, 0.025)),
        "ci_hi": float(np.quantile(estimates, 0.975)),
        "boot_sd": float(estimates.std(ddof=1)),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "resample_domains": resample_domains,
    }


def bootstrap_balanced_group_mean(
    df: pd.DataFrame,
    outcome_col: str,
    *,
    group_col: str,
    group_value: str,
    drop_na: bool,
    n_bootstrap: int,
    seed: int,
    resample_domains: bool,
) -> dict[str, Any]:
    subset = df[df[group_col] == group_value].copy()
    rng = np.random.default_rng(seed)
    values = subset.dropna(subset=[outcome_col]) if drop_na else subset
    tasks = [task for task in TASK_ORDER if task in set(values["task"])]
    domains = [domain for domain in DOMAIN_ORDER if domain in set(values["domain"])]
    actors = [actor for actor in ACTORS if actor in set(values["actor"])]
    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    for key, sub in values.groupby(["actor", "task", "domain"], sort=True):
        arrays[key] = sub[outcome_col].to_numpy(dtype=float)

    point = float(np.mean([array.mean() for array in arrays.values()]))
    estimates = np.empty(n_bootstrap)
    fixed_domains = np.array(domains, dtype=object)
    fixed_tasks = np.array(tasks, dtype=object)
    fixed_actors = np.array(actors, dtype=object)
    for i in range(n_bootstrap):
        sampled_actors = (
            fixed_actors
            if group_col == "actor"
            else rng.choice(actors, size=len(actors), replace=True)
        )
        sampled_tasks = (
            fixed_tasks
            if group_col == "task"
            else rng.choice(tasks, size=len(tasks), replace=True)
        )
        sampled_domains = (
            rng.choice(domains, size=len(domains), replace=True) if resample_domains else fixed_domains
        )
        cell_rates: list[float] = []
        for actor in sampled_actors:
            for task in sampled_tasks:
                for domain in sampled_domains:
                    array = arrays.get((str(actor), str(task), str(domain)))
                    if array is None or len(array) == 0:
                        continue
                    sample = rng.choice(array, size=len(array), replace=True)
                    cell_rates.append(float(sample.mean()))
        estimates[i] = float(np.mean(cell_rates))

    return {
        "balanced_bootstrap_point": point,
        "balanced_bootstrap_ci_lo": float(np.quantile(estimates, 0.025)),
        "balanced_bootstrap_ci_hi": float(np.quantile(estimates, 0.975)),
        "balanced_bootstrap_sd": float(estimates.std(ddof=1)),
        "balanced_bootstrap_n": n_bootstrap,
        "balanced_bootstrap_seed": seed,
        "balanced_bootstrap_resample_domains": resample_domains,
    }


def add_balanced_bootstrap_to_summary(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    *,
    group_col: str,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        group_value = str(row[group_col])
        row = dict(row)
        row.update(
            bootstrap_balanced_group_mean(
                df,
                "high_win_excluding_ties",
                group_col=group_col,
                group_value=group_value,
                drop_na=True,
                n_bootstrap=n_bootstrap,
                seed=seed + idx,
                resample_domains=False,
            )
        )
        row["ci_method"] = (
            "balanced bootstrap over remaining crossed factors and trials; domains fixed; "
            "pooled Wilson CI retained as descriptive"
        )
        out.append(row)
    return out


def overall_rows(df: pd.DataFrame, *, n_bootstrap: int, seed: int) -> list[dict[str, Any]]:
    non_tied = df.dropna(subset=["high_win_excluding_ties"])
    high = int(non_tied["high_win_excluding_ties"].sum())
    n = int(len(non_tied))
    pooled_rate, pooled_lo, pooled_hi = wilson(high, n)
    rows = [
        {
            "estimand": "primary_balanced_actor_task_domain_excluding_ties",
            "outcome": "panel high-side win, ties excluded",
            "ci_method": "crossed bootstrap: actors and tasks resampled; domains fixed; trials resampled within actor-task-domain stratum",
            **bootstrap_balanced_mean(
                df,
                "high_win_excluding_ties",
                drop_na=True,
                n_bootstrap=n_bootstrap,
                seed=seed,
                resample_domains=False,
            ),
            "n_pairs": int(len(df)),
            "n_non_tied": n,
            "high_wins": high,
            "low_wins": int(n - high),
            "ties": int((df["winner"] == "tie").sum()),
        },
        {
            "estimand": "sensitivity_balanced_actor_task_domain_ties_half",
            "outcome": "panel high-side win, ties counted as 0.5",
            "ci_method": "crossed bootstrap: actors and tasks resampled; domains fixed; trials resampled within actor-task-domain stratum",
            **bootstrap_balanced_mean(
                df,
                "high_win_ties_half",
                drop_na=False,
                n_bootstrap=n_bootstrap,
                seed=seed + 1,
                resample_domains=False,
            ),
            "n_pairs": int(len(df)),
            "n_non_tied": n,
            "high_wins": high,
            "low_wins": int(n - high),
            "ties": int((df["winner"] == "tie").sum()),
        },
        {
            "estimand": "sensitivity_resample_domains_excluding_ties",
            "outcome": "panel high-side win, ties excluded",
            "ci_method": "crossed bootstrap: actors, tasks, and domains resampled; trials resampled within actor-task-domain stratum",
            **bootstrap_balanced_mean(
                df,
                "high_win_excluding_ties",
                drop_na=True,
                n_bootstrap=n_bootstrap,
                seed=seed + 2,
                resample_domains=True,
            ),
            "n_pairs": int(len(df)),
            "n_non_tied": n,
            "high_wins": high,
            "low_wins": int(n - high),
            "ties": int((df["winner"] == "tie").sum()),
        },
        {
            "estimand": "descriptive_pooled_excluding_ties",
            "outcome": "panel high-side win, ties excluded",
            "ci_method": "Wilson binomial CI; descriptive only, not primary",
            "point": pooled_rate,
            "ci_lo": pooled_lo,
            "ci_hi": pooled_hi,
            "boot_sd": "",
            "n_bootstrap": "",
            "seed": "",
            "resample_domains": "",
            "n_pairs": int(len(df)),
            "n_non_tied": n,
            "high_wins": high,
            "low_wins": int(n - high),
            "ties": int((df["winner"] == "tie").sum()),
        },
    ]
    return rows


def regression_row(name: str, model_type: str, result: Any, term: str = "delta_u") -> dict[str, Any]:
    ci = result.conf_int()
    if hasattr(ci, "loc"):
        ci_lo, ci_hi = [float(x) for x in ci.loc[term]]
    else:
        idx = list(result.params.index).index(term)
        ci_lo, ci_hi = [float(x) for x in ci[idx]]
    coef = float(result.params[term])
    row: dict[str, Any] = {
        "model": name,
        "model_type": model_type,
        "n": int(result.nobs),
        "term": term,
        "coef": coef,
        "se": float(result.bse[term]),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": float(result.pvalues[term]),
    }
    if model_type == "logit":
        row.update({"odds_ratio": math.exp(coef), "or_ci_lo": math.exp(ci_lo), "or_ci_hi": math.exp(ci_hi)})
    else:
        row.update({"odds_ratio": "", "or_ci_lo": "", "or_ci_hi": ""})
    return row


def dose_response_regressions(df: pd.DataFrame) -> list[dict[str, Any]]:
    non_tied = df.dropna(subset=["high_win_excluding_ties"]).copy()
    non_tied["high_win"] = non_tied["high_win_excluding_ties"].astype(int)
    cluster = non_tied["utility_pair_cluster"]
    rows: list[dict[str, Any]] = []
    uncontrolled = smf.ols("high_win ~ delta_u", data=non_tied).fit(
        cov_type="cluster", cov_kwds={"groups": cluster}
    )
    rows.append(regression_row("uncontrolled_lpm", "linear_probability", uncontrolled))
    fixed = smf.ols("high_win ~ delta_u + C(actor) + C(task) + C(domain)", data=non_tied).fit(
        cov_type="cluster", cov_kwds={"groups": cluster}
    )
    rows.append(regression_row("fixed_effect_lpm_actor_task_domain", "linear_probability", fixed))
    logit = smf.glm(
        "high_win ~ delta_u + C(actor) + C(task) + C(domain)",
        data=non_tied,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": cluster})
    rows.append(regression_row("fixed_effect_logit_actor_task_domain", "logit", logit))
    for task in TASK_ORDER:
        sub = non_tied[non_tied["task"] == task]
        if len(sub) > 50 and sub["high_win"].nunique() > 1:
            result = smf.ols("high_win ~ delta_u + C(actor) + C(domain)", data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["utility_pair_cluster"]}
            )
            rows.append(regression_row(f"task_{task}_lpm", "linear_probability", result))
    for domain in DOMAIN_ORDER:
        sub = non_tied[non_tied["domain"] == domain]
        if len(sub) > 50 and sub["high_win"].nunique() > 1:
            result = smf.ols("high_win ~ delta_u + C(actor) + C(task)", data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["utility_pair_cluster"]}
            )
            rows.append(regression_row(f"domain_{domain}_lpm", "linear_probability", result))
    return rows


def dose_response_bins(df: pd.DataFrame, n_bins: int) -> list[dict[str, Any]]:
    non_tied = df.dropna(subset=["high_win_excluding_ties"]).copy().sort_values("delta_u")
    non_tied["bin"] = pd.qcut(np.arange(len(non_tied)), q=n_bins, labels=False) + 1
    rows: list[dict[str, Any]] = []
    for bin_id, sub in non_tied.groupby("bin", sort=True):
        high = int(sub["high_win_excluding_ties"].sum())
        n = int(len(sub))
        rate, ci_lo, ci_hi = wilson(high, n)
        rows.append(
            {
                "bin": int(bin_id),
                "n": n,
                "delta_u_min": float(sub["delta_u"].min()),
                "delta_u_mean": float(sub["delta_u"].mean()),
                "delta_u_max": float(sub["delta_u"].max()),
                "high_wins": high,
                "low_wins": n - high,
                "high_win_rate": rate,
                "wilson_ci_lo": ci_lo,
                "wilson_ci_hi": ci_hi,
            }
        )
    return rows


def judge_summary(votes: pd.DataFrame, pairs: pd.DataFrame) -> list[dict[str, Any]]:
    if votes.empty:
        return []
    merged = votes.merge(pairs[["pair_uid", "actor", "task", "domain"]], on="pair_uid", how="left")
    rows: list[dict[str, Any]] = []
    for cols, label in [(["judge_model"], "judge"), (["judge_model", "task"], "judge_task")]:
        for key, sub in merged.groupby(cols, sort=True):
            key_tuple = key if isinstance(key, tuple) else (key,)
            valid = sub[sub["winner_condition"].isin(["high", "low"])]
            high = int((valid["winner_condition"] == "high").sum())
            n = int(len(valid))
            rate, ci_lo, ci_hi = wilson(high, n)
            row = {
                "summary_level": label,
                **{col: value for col, value in zip(cols, key_tuple, strict=False)},
                "n_votes": int(len(sub)),
                "n_high_low_votes": n,
                "high_votes": high,
                "low_votes": int(n - high),
                "tie_votes": int((sub["winner_condition"] == "tie").sum()),
                "unresolved_votes": int((sub["winner_condition"] == "unresolved").sum()),
                "high_vote_rate_excluding_ties": rate,
                "wilson_ci_lo": ci_lo,
                "wilson_ci_hi": ci_hi,
            }
            rows.append(row)
    return rows


def rounded_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({key: (round(value, 6) if isinstance(value, float) else value) for key, value in row.items()})
    return out


def _strip_spines(ax: Any, keep: tuple[str, ...] = ("bottom",)) -> None:
    for spine in ("top", "right", "left", "bottom"):
        if spine in keep:
            ax.spines[spine].set_color("#9CA3AF")
            ax.spines[spine].set_linewidth(0.7)
        else:
            ax.spines[spine].set_visible(False)


def summary_rows_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "balanced_bootstrap_point" in df.columns:
        df["rate"] = df["balanced_bootstrap_point"].astype(float)
        df["ci_lo"] = df["balanced_bootstrap_ci_lo"].astype(float)
        df["ci_hi"] = df["balanced_bootstrap_ci_hi"].astype(float)
    elif "familywise_ci_lo" in df.columns and "familywise_ci_hi" in df.columns:
        df["rate"] = df["pooled_high_win_rate_excluding_ties"].astype(float)
        df["ci_lo"] = df["familywise_ci_lo"].astype(float)
        df["ci_hi"] = df["familywise_ci_hi"].astype(float)
    else:
        df["rate"] = df["pooled_high_win_rate_excluding_ties"].astype(float)
        df["ci_lo"] = df["pooled_wilson_ci_lo"].astype(float)
        df["ci_hi"] = df["pooled_wilson_ci_hi"].astype(float)
    return df


def render_lollipop_panel(
    ax: Any,
    df_panel: pd.DataFrame,
    *,
    item_col: str,
    item_order: list[str],
    item_labels: dict[str, str],
    item_colors: dict[str, str],
    title: str,
    panel_letter: str | None,
    show_xlabel: bool,
    xlabel: str = "High-utility-side win rate (ties excluded)",
) -> None:
    present = set(df_panel[item_col].astype(str))
    ordered = [item for item in item_order if item in present]
    if not ordered:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    df_ordered = df_panel.assign(_key=df_panel[item_col].astype(str)).set_index("_key").loc[ordered].reset_index()
    n = len(df_ordered)
    has_holm = "holm_positive" in df_ordered.columns
    has_familywise = "familywise_ci_positive" in df_ordered.columns

    for i, row in df_ordered.iterrows():
        y = n - 1 - i
        if has_familywise:
            positive = bool(row["familywise_ci_positive"])
            negative = bool(row["familywise_ci_negative"])
        elif has_holm:
            positive = bool(row["holm_positive"])
            negative = bool(row["holm_negative"]) if "holm_negative" in row else False
        else:
            positive = float(row["ci_lo"]) > 0.5
            negative = float(row["ci_hi"]) < 0.5
        if positive:
            ax.add_patch(Rectangle((0.005, y - 0.42), 0.99, 0.84, facecolor=CI_POS_BAND, edgecolor="none", zorder=0))
        elif negative:
            ax.add_patch(Rectangle((0.005, y - 0.42), 0.99, 0.84, facecolor=CI_NEG_BAND, edgecolor="none", zorder=0))

    ax.axvline(0.5, color=CHANCE_LINE, lw=1.2, ls=(0, (4, 3)), zorder=1)
    for i, row in df_ordered.iterrows():
        y = n - 1 - i
        item = str(row[item_col])
        color = item_colors.get(item, "#3A66C9")
        rate = float(row["rate"])
        lo = float(row["ci_lo"])
        hi = float(row["ci_hi"])
        if has_familywise:
            highlighted = bool(row["familywise_ci_positive"]) or bool(row["familywise_ci_negative"])
        elif has_holm:
            highlighted = bool(row["holm_positive"]) or bool(row["holm_negative"])
        else:
            highlighted = lo > 0.5 or hi < 0.5
        ax.hlines(y, lo, hi, color=color, lw=4.0, alpha=0.45 if highlighted else 0.30, capstyle="round", zorder=2)
        ax.vlines([lo, hi], y - 0.15, y + 0.15, color=color, lw=1.2, alpha=0.55, zorder=2)
        ax.scatter(rate, y, s=120, color=color, edgecolor="white", linewidth=1.4, zorder=3)
        ax.text(min(hi + 0.025, 1.02), y, f"{rate:.2f}", ha="left", va="center", fontsize=9.0, color=color)

    labels = [item_labels.get(str(row[item_col]), str(row[item_col])) for _, row in df_ordered.iterrows()]
    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels(labels, fontsize=9.5)
    for tick, item in zip(ax.get_yticklabels(), ordered, strict=False):
        tick.set_color(item_colors.get(item, INK))
        tick.set_fontweight("semibold")

    ax.set_xlim(0.005, 1.06)
    ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.55, n - 0.10)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=SUBTLE, length=3, width=0.6)
    _strip_spines(ax)
    ax.set_title(title, loc="left", fontsize=12.5, color=INK, fontweight="bold", pad=18)
    if show_xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10.5)
    if panel_letter:
        ax.text(-0.28, 1.11, panel_letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK)
    n_excl = df_ordered["n_non_tied"].astype(int)
    n_note = f"n = {int(n_excl.min())}" if int(n_excl.min()) == int(n_excl.max()) else f"n = {int(n_excl.min())}-{int(n_excl.max())}"
    if "familywise_ci_lo" in df_ordered.columns:
        n_note = f"{n_note} non-tied pairs; FWER 95% CIs"
    else:
        n_note = f"{n_note} non-tied pairs"
    ax.text(0.0, 1.04, n_note, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, color=SUBTLE)
    if has_familywise or has_holm:
        n_pos = int(df_ordered["familywise_ci_positive"].sum()) if has_familywise else int(df_ordered["holm_positive"].sum())
        n_total = len(df_ordered)
        chip_bg = "#DDF1E3" if n_pos > 0 else "#F1F2F5"
        chip_fg = "#2F7A4F" if n_pos > 0 else "#4B5563"
        chip_label = "adj-CI > 0.5" if has_familywise else "Holm-positive"
        ax.text(
            1.0,
            1.11,
            f"{n_pos}/{n_total} {chip_label}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            color=chip_fg,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.30", facecolor=chip_bg, edgecolor="none"),
        )


def write_actor_task_lollipop(rows: list[dict[str, Any]], path_base: Path, *, domain_label: str = "") -> None:
    df = summary_rows_frame(rows)
    fig = plt.figure(figsize=(12.8, 7.0), facecolor="white")
    grid = fig.add_gridspec(2, 2, top=0.91, bottom=0.10, left=0.10, right=0.985, hspace=0.55, wspace=0.50)
    for idx, task in enumerate(PLOT_TASK_ORDER):
        ax = fig.add_subplot(grid[idx // 2, idx % 2])
        title = TASK_LABEL[task] if not domain_label else f"{TASK_LABEL[task]} - {domain_label}"
        render_lollipop_panel(
            ax,
            df[df["task"] == task],
            item_col="actor",
            item_order=ACTORS,
            item_labels=ACTOR_LABEL,
            item_colors=MODEL_COLORS,
            title=title,
            panel_letter=chr(ord("A") + idx),
            show_xlabel=idx // 2 == 1,
        )
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(fig)


def write_overall_lollipop(
    rows: list[dict[str, Any]],
    path_base: Path,
    *,
    item_col: str,
    item_order: list[str],
    item_labels: dict[str, str],
    item_colors: dict[str, str],
    title: str,
) -> None:
    df = summary_rows_frame(rows)
    height = max(3.3, 0.48 * len(item_order) + 1.1)
    fig, ax = plt.subplots(figsize=(6.7, height), facecolor="white")
    render_lollipop_panel(
        ax,
        df,
        item_col=item_col,
        item_order=item_order,
        item_labels=item_labels,
        item_colors=item_colors,
        title=title,
        panel_letter=None,
        show_xlabel=True,
    )
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(fig)


def trend_regressions(df: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    non_tied = df.dropna(subset=["high_win_excluding_ties"]).copy()
    non_tied["high_win"] = non_tied["high_win_excluding_ties"].astype(int)
    rows: list[dict[str, Any]] = []
    uncontrolled = smf.ols("high_win ~ delta_u", data=non_tied).fit(
        cov_type="cluster", cov_kwds={"groups": non_tied["utility_pair_cluster"]}
    )
    rows.append(regression_row(f"{label}_uncontrolled_lpm", "linear_probability", uncontrolled))
    terms = ["delta_u"]
    for col in ["actor", "task", "domain"]:
        if non_tied[col].nunique() > 1:
            terms.append(f"C({col})")
    formula = "high_win ~ " + " + ".join(terms)
    fixed = smf.ols(formula, data=non_tied).fit(
        cov_type="cluster", cov_kwds={"groups": non_tied["utility_pair_cluster"]}
    )
    rows.append(regression_row(f"{label}_fixed_effect_lpm", "linear_probability", fixed))
    return rows


def write_linear_trend_figure(
    df: pd.DataFrame,
    bins: list[dict[str, Any]],
    regressions_for_subset: list[dict[str, Any]],
    path_base: Path,
    *,
    title: str,
) -> None:
    non_tied = df.dropna(subset=["high_win_excluding_ties"]).copy()
    non_tied["high_win"] = non_tied["high_win_excluding_ties"].astype(int)
    marginal = next(row for row in regressions_for_subset if row["model"].endswith("_uncontrolled_lpm"))
    fixed = next(row for row in regressions_for_subset if row["model"].endswith("_fixed_effect_lpm"))
    marginal_fit = smf.ols("high_win ~ delta_u", data=non_tied).fit()

    fig, ax = plt.subplots(figsize=(7.0, 4.4), facecolor="white")
    xs = np.array([row["delta_u_mean"] for row in bins], dtype=float)
    ys = np.array([row["high_win_rate"] for row in bins], dtype=float)
    lo = np.array([row["wilson_ci_lo"] for row in bins], dtype=float)
    hi = np.array([row["wilson_ci_hi"] for row in bins], dtype=float)
    ax.axhline(0.5, color=CHANCE_LINE, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="o", color="#2563EB", ecolor="#93C5FD", capsize=3, ms=7, zorder=3)

    line_x = np.linspace(float(non_tied["delta_u"].min()), float(non_tied["delta_u"].max()), 100)
    line_y = marginal_fit.params["Intercept"] + marginal_fit.params["delta_u"] * line_x
    ax.plot(line_x, line_y, color="#C2304A", lw=2.0, zorder=2)
    ax.text(
        0.02,
        0.04,
        (
            f"Marginal slope: {100 * marginal['coef']:.1f} pp/unit, p={marginal['p_value']:.3f}\n"
            f"FE slope: {100 * fixed['coef']:.1f} pp/unit, p={fixed['p_value']:.3f}"
        ),
        transform=ax.transAxes,
        fontsize=9,
        color=INK,
    )
    ax.set_xlabel("Fitted utility gap: delta_u = u_high - u_low")
    ax.set_ylabel("High-utility-side win rate, ties excluded")
    ax.set_ylim(0.35, 0.65)
    ax.set_title(title, fontsize=13, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    _strip_spines(ax, keep=("left", "bottom"))
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(fig)


def write_actor_task_heatmap(df: pd.DataFrame, path_base: Path) -> None:
    rates = (
        df.dropna(subset=["high_win_excluding_ties"])
        .groupby(["actor", "task"], sort=True)["high_win_excluding_ties"]
        .mean()
        .unstack("task")
        .reindex(index=ACTORS, columns=TASK_ORDER)
    )
    labels = [ACTOR_LABEL.get(actor, actor) for actor in rates.index]
    matrix = rates.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    image = ax.imshow(matrix - 0.5, cmap="RdBu_r", vmin=-0.12, vmax=0.12, aspect="auto")
    ax.set_xticks(np.arange(len(TASK_ORDER)), [TASK_LABEL[task] for task in TASK_ORDER], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{100 * matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("High-low intervention: high-side win rate by actor and task")
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label("Win-rate difference from 50 percentage points")
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_task_domain_heatmap(df: pd.DataFrame, path_base: Path) -> None:
    rates = (
        df.dropna(subset=["high_win_excluding_ties"])
        .groupby(["task", "domain"], sort=True)["high_win_excluding_ties"]
        .mean()
        .unstack("domain")
        .reindex(index=TASK_ORDER, columns=DOMAIN_ORDER)
    )
    matrix = rates.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    image = ax.imshow(matrix - 0.5, cmap="RdBu_r", vmin=-0.12, vmax=0.12, aspect="auto")
    ax.set_xticks(np.arange(len(DOMAIN_ORDER)), [DOMAIN_LABEL[domain] for domain in DOMAIN_ORDER], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(TASK_ORDER)), [TASK_LABEL[task] for task in TASK_ORDER])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{100 * matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("High-low intervention: high-side win rate by task and utility domain")
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label("Win-rate difference from 50 percentage points")
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_dose_response_figure(bins: list[dict[str, Any]], regressions: list[dict[str, Any]], path_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = np.array([row["delta_u_mean"] for row in bins], dtype=float)
    ys = np.array([row["high_win_rate"] for row in bins], dtype=float)
    lo = np.array([row["wilson_ci_lo"] for row in bins], dtype=float)
    hi = np.array([row["wilson_ci_hi"] for row in bins], dtype=float)
    ax.axhline(0.5, color="#9CA3AF", lw=1.2, ls=(0, (4, 3)))
    ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="o", color="#2563EB", ecolor="#93C5FD", capsize=3)
    fixed = next(row for row in regressions if row["model"] == "fixed_effect_lpm_actor_task_domain")
    ax.text(
        0.02,
        0.04,
        f"FE LPM slope: {100 * fixed['coef']:.1f} pp/unit, p={fixed['p_value']:.3f}",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.set_xlabel("Fitted utility gap: delta_u = u_high - u_low")
    ax.set_ylabel("High-side win rate, excluding ties")
    ax.set_ylim(0.35, 0.65)
    ax.set_title("High-low intervention: utility-gap dose response")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(
    path: Path,
    *,
    df: pd.DataFrame,
    invalid_generation_rows: list[dict[str, Any]],
    used_paths: list[Path],
    audit: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    summaries: dict[str, list[dict[str, Any]]],
    regressions: list[dict[str, Any]],
    bins: list[dict[str, Any]],
    trend_rows: list[dict[str, Any]],
    policy_bins: list[dict[str, Any]],
    output_files: list[Path],
    command: str,
) -> None:
    primary = next(row for row in overall if row["estimand"] == "primary_balanced_actor_task_domain_excluding_ties")
    ties_half = next(row for row in overall if row["estimand"] == "sensitivity_balanced_actor_task_domain_ties_half")
    pooled = next(row for row in overall if row["estimand"] == "descriptive_pooled_excluding_ties")
    fixed_lpm = next(row for row in regressions if row["model"] == "fixed_effect_lpm_actor_task_domain")
    fixed_logit = next(row for row in regressions if row["model"] == "fixed_effect_logit_actor_task_domain")
    overall_marginal = next(row for row in trend_rows if row["model"] == "overall_uncontrolled_lpm")
    policy_marginal = next(row for row in trend_rows if row["model"] == "policy_uncontrolled_lpm")
    policy_fixed = next(row for row in trend_rows if row["model"] == "policy_fixed_effect_lpm")
    task_rows = summaries["task"]
    actor_rows = summaries["actor"]
    domain_rows = summaries["domain"]
    invalid_pair_count = len({str(row.get("pair_uid")) for row in invalid_generation_rows})
    explicit_truncation_count = sum(int(row.get("explicit_length_truncation") or 0) for row in invalid_generation_rows)
    degenerate_count = sum(int(row.get("degenerate_chars_lt_20") or 0) for row in invalid_generation_rows)

    lines = [
        "# Final High-Low Intervention Analysis",
        "",
        "## Reproducibility",
        "",
        "Run from the repository root:",
        "",
        f"```bash\n{command}\n```",
        "",
        "The script reads only frozen raw snapshots matching "
        "`outputs/raw/highlow_intervention__*__*__judged_pairs.csv` and the corresponding judge-vote snapshots.",
        "",
        "## Analysis Choice",
        "",
        "Primary estimand: the unweighted mean of high-side panel win rates across the balanced "
        "actor x task x utility-domain design cells, excluding panel ties. This gives each actor, "
        "task, and domain equal design weight rather than allowing cells with fewer ties to count more.",
        "",
        "Primary CI: nonparametric crossed bootstrap with actors and tasks resampled as crossed factors, "
        "the four utility domains kept as fixed balanced strata, and trials resampled within each "
        "actor-task-domain stratum. A sensitivity CI also resamples domains.",
        "",
        "Panel ties are excluded for the primary analysis because that matches the paper's denominator. "
        "A ties-as-half sensitivity is reported immediately below it.",
        "",
        "## Data Audit",
        "",
        f"- Snapshot files: {len(used_paths)}.",
        f"- Analyzed judged pair rows: {len(df):,}.",
        f"- Unique pair IDs: {df['pair_uid'].nunique():,}.",
        f"- Excluded pairs with invalid final generations: {invalid_pair_count:,}.",
        f"- Explicit length truncations excluded: {explicit_truncation_count:,}.",
        f"- Degenerate final generations excluded: {degenerate_count:,}.",
        f"- Panel ties: {int((df['winner'] == 'tie').sum()):,}.",
        f"- Audit checks failed: {sum(int(row['observed'] != row['expected']) for row in audit)}.",
        "",
        "## Primary Result",
        "",
        (
            f"- Balanced high-side win rate excluding ties: {pct(primary['point'])} "
            f"(95% bootstrap CI {pct(primary['ci_lo'])}-{pct(primary['ci_hi'])})."
        ),
        (
            f"- Ties-as-half sensitivity: {pct(ties_half['point'])} "
            f"(95% bootstrap CI {pct(ties_half['ci_lo'])}-{pct(ties_half['ci_hi'])})."
        ),
        (
            f"- Descriptive pooled rate excluding ties: {pooled['high_wins']}/{pooled['n_non_tied']} = "
            f"{pct(pooled['point'])} (Wilson 95% CI {pct(pooled['ci_lo'])}-{pct(pooled['ci_hi'])})."
        ),
        "",
        "## Model Summary",
        "",
        "Model-level CIs use the same design-aware principle as the primary analysis: for each fixed model, "
        "tasks are resampled as crossed factors, domains are fixed balanced strata, and trials are resampled "
        "within task-domain strata. Pooled Wilson CIs are retained in the CSV as descriptive checks.",
        "",
        "| Model | High | Low | Tie | Balanced High Win Rate | 95% Bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in actor_rows:
        lines.append(
            f"| {ACTOR_LABEL.get(row['actor'], row['actor'])} | {row['high_wins']} | {row['low_wins']} | "
            f"{row['ties']} | {pct(row['balanced_bootstrap_point'])} | "
            f"{pct(row['balanced_bootstrap_ci_lo'])}-{pct(row['balanced_bootstrap_ci_hi'])} |"
        )
    lines.extend(
        [
            "",
            "## Task Summary",
            "",
            "| Task | High | Low | Tie | Balanced High Win Rate | 95% Bootstrap CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in task_rows:
        lines.append(
            f"| {TASK_LABEL.get(row['task'], row['task'])} | {row['high_wins']} | {row['low_wins']} | "
            f"{row['ties']} | {pct(row['balanced_bootstrap_point'])} | "
            f"{pct(row['balanced_bootstrap_ci_lo'])}-{pct(row['balanced_bootstrap_ci_hi'])} |"
        )
    lines.extend(["", "## Domain Summary", "", "| Domain | High | Low | Tie | Pooled High Win Rate |", "|---|---:|---:|---:|---:|"])
    for row in domain_rows:
        lines.append(
            f"| {DOMAIN_LABEL.get(row['domain'], row['domain'])} | {row['high_wins']} | {row['low_wins']} | "
            f"{row['ties']} | {pct(row['pooled_high_win_rate_excluding_ties'])} |"
        )
    lines.extend(
        [
            "",
            "## Utility-Gap Dose Response",
            "",
            (
                f"- Fixed-effect LPM slope: {fixed_lpm['coef']:.4f} probability units per one utility unit "
                f"(clustered SE {fixed_lpm['se']:.4f}, 95% CI {fixed_lpm['ci_lo']:.4f} to "
                f"{fixed_lpm['ci_hi']:.4f}, p={fixed_lpm['p_value']:.4g})."
            ),
            (
                f"- Fixed-effect logit odds ratio: {fixed_logit['odds_ratio']:.3f} "
                f"(95% CI {fixed_logit['or_ci_lo']:.3f}-{fixed_logit['or_ci_hi']:.3f}, "
                f"p={fixed_logit['p_value']:.4g})."
            ),
            "",
            "| Utility Gap Bin | Mean delta_u | N | High Win Rate |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in bins:
        lines.append(f"| {row['bin']} | {row['delta_u_mean']:.3f} | {row['n']} | {pct(row['high_win_rate'])} |")
    lines.extend(
        [
            "",
            "## Linear-Trend Figures",
            "",
            (
                f"- Overall marginal linear trend: {100 * overall_marginal['coef']:.2f} percentage points per "
                f"utility unit (clustered 95% CI {100 * overall_marginal['ci_lo']:.2f} to "
                f"{100 * overall_marginal['ci_hi']:.2f}, p={overall_marginal['p_value']:.4g})."
            ),
            (
                f"- Policy-domain marginal linear trend: {100 * policy_marginal['coef']:.2f} percentage points per "
                f"utility unit (clustered 95% CI {100 * policy_marginal['ci_lo']:.2f} to "
                f"{100 * policy_marginal['ci_hi']:.2f}, p={policy_marginal['p_value']:.4g})."
            ),
            (
                f"- Policy-domain fixed-effect linear trend, controlling actor and task: "
                f"{100 * policy_fixed['coef']:.2f} percentage points per utility unit "
                f"(clustered 95% CI {100 * policy_fixed['ci_lo']:.2f} to "
                f"{100 * policy_fixed['ci_hi']:.2f}, p={policy_fixed['p_value']:.4g})."
            ),
            "",
            "| Policy Utility Gap Bin | Mean delta_u | N | High Win Rate |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in policy_bins:
        lines.append(f"| {row['bin']} | {row['delta_u_mean']:.3f} | {row['n']} | {pct(row['high_win_rate'])} |")
    lines.extend(["", "## Output Files", ""])
    for output_file in output_files:
        lines.append(f"- `{output_file}`")
    lines.extend(["", "## Input Files", ""])
    for input_file in used_paths:
        lines.append(f"- `{input_file}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "figure.dpi": 200})

    raw_pairs, used_paths = load_pairs()
    generation_audit, invalid_generation_rows = generation_quality_scan(raw_pairs)
    invalid_pair_uids = {str(row["pair_uid"]) for row in invalid_generation_rows if row.get("pair_uid")}
    pairs = raw_pairs.loc[~raw_pairs["pair_uid"].isin(invalid_pair_uids)].copy()
    votes = load_votes()
    votes = votes.loc[votes["pair_uid"].isin(set(pairs["pair_uid"]))].copy()
    audit = audit_rows(raw_pairs, pairs, votes, used_paths, generation_audit)
    overall = overall_rows(pairs, n_bootstrap=args.bootstraps, seed=args.seed)
    summaries = {
        "overall": summarize_group(pairs, [], "overall"),
        "actor": add_balanced_bootstrap_to_summary(
            summarize_group(pairs, ["actor"], "actor"),
            pairs,
            group_col="actor",
            n_bootstrap=args.bootstraps,
            seed=args.seed + 100,
        ),
        "task": add_balanced_bootstrap_to_summary(
            summarize_group(pairs, ["task"], "task"),
            pairs,
            group_col="task",
            n_bootstrap=args.bootstraps,
            seed=args.seed + 200,
        ),
        "domain": summarize_group(pairs, ["domain"], "domain"),
        "actor_task": summarize_group(pairs, ["actor", "task"], "actor_task"),
        "actor_task_domain": summarize_group(pairs, ["actor", "task", "domain"], "actor_task_domain"),
    }
    summaries["actor_task"] = add_actor_task_multiplicity(summaries["actor_task"])
    regressions = dose_response_regressions(pairs)
    bins = dose_response_bins(pairs, args.bins)
    judge_rows = judge_summary(votes, pairs)

    prefix = f"{COMPARISON}_final"
    paths = {
        "trial_rows": ANALYSIS / f"{prefix}_trials.csv",
        "excluded_generation_outputs": ANALYSIS / f"{prefix}_excluded_generation_outputs.csv",
        "audit": ANALYSIS / f"{prefix}_audit.csv",
        "overall": ANALYSIS / f"{prefix}_overall.csv",
        "actor_summary": ANALYSIS / f"{prefix}_actor_summary.csv",
        "task_summary": ANALYSIS / f"{prefix}_task_summary.csv",
        "domain_summary": ANALYSIS / f"{prefix}_domain_summary.csv",
        "actor_task_summary": ANALYSIS / f"{prefix}_actor_task_summary.csv",
        "actor_task_domain_summary": ANALYSIS / f"{prefix}_actor_task_domain_summary.csv",
        "dose_bins": ANALYSIS / f"{prefix}_dose_response_bins.csv",
        "dose_regressions": ANALYSIS / f"{prefix}_dose_response_regressions.csv",
        "linear_trend_regressions": ANALYSIS / f"{prefix}_linear_trend_regressions.csv",
        "policy_dose_bins": ANALYSIS / f"{prefix}_policy_dose_response_bins.csv",
        "judge_summary": ANALYSIS / f"{prefix}_judge_summary.csv",
        "summary_md": ANALYSIS / f"{prefix}_analysis.md",
        "actor_task_lollipop": FIGURES / f"{prefix}_actor_task_lollipop",
        "actor_overall_lollipop": FIGURES / f"{prefix}_actor_overall_lollipop",
        "task_overall_lollipop": FIGURES / f"{prefix}_task_overall_lollipop",
        "dose_overall_linear": FIGURES / f"{prefix}_dose_response_overall_linear",
        "dose_policy_linear": FIGURES / f"{prefix}_dose_response_policy_linear",
    }

    trial_columns = [
        "pair_uid",
        "actor",
        "task",
        "domain",
        "pair_idx",
        "item_id",
        "item_label",
        "winner",
        "high_win_excluding_ties",
        "high_win_ties_half",
        "high_utility",
        "low_utility",
        "delta_u",
        "high_description",
        "low_description",
        "high_consequence",
        "low_consequence",
        "utility_pair_cluster",
        "source_file",
    ]
    write_csv_rows(paths["trial_rows"], rounded_rows(pairs[trial_columns].to_dict("records")))
    if invalid_generation_rows:
        write_csv_rows(paths["excluded_generation_outputs"], rounded_rows(invalid_generation_rows))
    else:
        write_csv_rows(
            paths["excluded_generation_outputs"],
            [{"pair_uid": "", "note": "no invalid final generation outputs"}],
        )
    write_csv_rows(paths["audit"], audit)
    write_csv_rows(paths["overall"], rounded_rows(overall))
    write_csv_rows(paths["actor_summary"], rounded_rows(summaries["actor"]))
    write_csv_rows(paths["task_summary"], rounded_rows(summaries["task"]))
    write_csv_rows(paths["domain_summary"], rounded_rows(summaries["domain"]))
    write_csv_rows(paths["actor_task_summary"], rounded_rows(summaries["actor_task"]))
    write_csv_rows(paths["actor_task_domain_summary"], rounded_rows(summaries["actor_task_domain"]))
    write_csv_rows(paths["dose_bins"], rounded_rows(bins))
    write_csv_rows(paths["dose_regressions"], rounded_rows(regressions))
    write_csv_rows(paths["judge_summary"], rounded_rows(judge_rows))

    policy_pairs = pairs[pairs["domain"] == "political"].copy()
    policy_bins = dose_response_bins(policy_pairs, args.bins)
    trend_rows = trend_regressions(pairs, "overall") + trend_regressions(policy_pairs, "policy")
    write_csv_rows(paths["linear_trend_regressions"], rounded_rows(trend_rows))
    write_csv_rows(paths["policy_dose_bins"], rounded_rows(policy_bins))

    write_actor_task_lollipop(summaries["actor_task"], paths["actor_task_lollipop"])
    write_overall_lollipop(
        summaries["actor"],
        paths["actor_overall_lollipop"],
        item_col="actor",
        item_order=ACTORS,
        item_labels=ACTOR_LABEL,
        item_colors=MODEL_COLORS,
        title="High-low intervention: model means",
    )
    write_overall_lollipop(
        summaries["task"],
        paths["task_overall_lollipop"],
        item_col="task",
        item_order=PLOT_TASK_ORDER,
        item_labels=TASK_LABEL,
        item_colors=TASK_COLORS,
        title="High-low intervention: task means",
    )
    domain_figure_paths: list[Path] = []
    for domain in DOMAIN_ORDER:
        domain_base = FIGURES / f"{prefix}_actor_task_lollipop__domain-{slug(domain)}"
        domain_rows = [row for row in summaries["actor_task_domain"] if row["domain"] == domain]
        write_actor_task_lollipop(domain_rows, domain_base, domain_label=DOMAIN_LABEL[domain])
        domain_figure_paths.extend([domain_base.with_suffix(".pdf"), domain_base.with_suffix(".png")])
    write_linear_trend_figure(
        pairs,
        bins,
        [row for row in trend_rows if row["model"].startswith("overall_")],
        paths["dose_overall_linear"],
        title="High-low intervention: overall utility-gap linear trend",
    )
    write_linear_trend_figure(
        policy_pairs,
        policy_bins,
        [row for row in trend_rows if row["model"].startswith("policy_")],
        paths["dose_policy_linear"],
        title="High-low intervention: policy-domain utility-gap linear trend",
    )

    output_files = [
        paths["trial_rows"],
        paths["excluded_generation_outputs"],
        paths["audit"],
        paths["overall"],
        paths["actor_summary"],
        paths["task_summary"],
        paths["domain_summary"],
        paths["actor_task_summary"],
        paths["actor_task_domain_summary"],
        paths["dose_bins"],
        paths["dose_regressions"],
        paths["linear_trend_regressions"],
        paths["policy_dose_bins"],
        paths["judge_summary"],
        paths["summary_md"],
        paths["actor_task_lollipop"].with_suffix(".pdf"),
        paths["actor_task_lollipop"].with_suffix(".png"),
        paths["actor_overall_lollipop"].with_suffix(".pdf"),
        paths["actor_overall_lollipop"].with_suffix(".png"),
        paths["task_overall_lollipop"].with_suffix(".pdf"),
        paths["task_overall_lollipop"].with_suffix(".png"),
        *domain_figure_paths,
        paths["dose_overall_linear"].with_suffix(".pdf"),
        paths["dose_overall_linear"].with_suffix(".png"),
        paths["dose_policy_linear"].with_suffix(".pdf"),
        paths["dose_policy_linear"].with_suffix(".png"),
    ]
    command = (
        "python -m utility_behavior_gap.scripts.analyze_highlow_intervention_final "
        f"--bootstraps {args.bootstraps} --seed {args.seed} --bins {args.bins}"
    )
    write_markdown_summary(
        paths["summary_md"],
        df=pairs,
        invalid_generation_rows=invalid_generation_rows,
        used_paths=used_paths,
        audit=audit,
        overall=overall,
        summaries=summaries,
        regressions=regressions,
        bins=bins,
        trend_rows=trend_rows,
        policy_bins=policy_bins,
        output_files=output_files,
        command=command,
    )

    primary = next(row for row in overall if row["estimand"] == "primary_balanced_actor_task_domain_excluding_ties")
    ties_half = next(row for row in overall if row["estimand"] == "sensitivity_balanced_actor_task_domain_ties_half")
    fixed = next(row for row in regressions if row["model"] == "fixed_effect_lpm_actor_task_domain")
    print("high-low final analysis complete")
    print(f"judged pairs: {len(pairs):,}; judge votes: {len(votes):,}")
    print(f"primary high-side win rate: {primary['point']:.4f} [{primary['ci_lo']:.4f}, {primary['ci_hi']:.4f}]")
    print(f"ties-as-half sensitivity: {ties_half['point']:.4f} [{ties_half['ci_lo']:.4f}, {ties_half['ci_hi']:.4f}]")
    print(
        "dose-response FE LPM slope: "
        f"{fixed['coef']:.4f} [{fixed['ci_lo']:.4f}, {fixed['ci_hi']:.4f}], p={fixed['p_value']:.4g}"
    )
    print(f"summary: {paths['summary_md']}")


if __name__ == "__main__":
    main()
