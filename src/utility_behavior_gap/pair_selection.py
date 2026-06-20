"""Deterministic high-low pair selection from fitted utility inputs."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any

from utility_behavior_gap.constants import ACTOR_LABEL, ACTORS, DOMAIN_LABEL, DOMAINS
from utility_behavior_gap.io_utils import read_csv_rows, write_csv_rows
from utility_behavior_gap.paths import INPUTS, OUTPUT_INPUTS


def as_float(value: str) -> float:
    return float(value)


def utility_options_with_actor_overlays() -> list[dict[str, str]]:
    rows = read_csv_rows(INPUTS / "utility_options.csv")
    overlays: dict[str, list[dict[str, str]]] = {}
    for path in sorted(OUTPUT_INPUTS.glob("utility_options__*.csv")):
        overlay_rows = read_csv_rows(path)
        actors = {row["actor"] for row in overlay_rows}
        if len(actors) != 1:
            raise ValueError(f"expected exactly one actor in {path}, found {sorted(actors)}")
        overlays[next(iter(actors))] = overlay_rows
    if not overlays:
        return rows
    overlay_actors = set(overlays)
    merged = [row for row in rows if row["actor"] not in overlay_actors]
    for actor in sorted(overlays):
        merged.extend(overlays[actor])
    return merged


def option_key(option: dict[str, str]) -> str:
    return option["option_id"]


def count_category(description: str) -> str:
    match = re.match(r"(\d+)\s+", description)
    return f"count_{match.group(1)}" if match else "__no_count__"


def category_map(options: list[dict[str, str]], pair_set: str) -> dict[str, str]:
    if pair_set == "default":
        return {option_key(option): "__all__" for option in options}
    if pair_set == "same_count":
        return {option_key(option): count_category(option["description"]) for option in options}
    raise ValueError(f"unknown pair_set: {pair_set}")


def sample_tercile_pairs(
    options: list[dict[str, str]],
    *,
    pair_set: str,
    pairs_per_cell: int,
    seed: int,
) -> list[tuple[dict[str, str], dict[str, str], str]]:
    """Match the experiment rule: sample high/low options from top/bottom thirds.

    If a category map is supplied, the thirds are computed inside each category
    and each sampled pair stays within one category.
    """

    rng = random.Random(seed)
    categories = category_map(options, pair_set)
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for option in options:
        by_category[categories[option_key(option)]].append(option)

    high_by_category: dict[str, list[dict[str, str]]] = {}
    low_by_category: dict[str, list[dict[str, str]]] = {}
    for category, group in by_category.items():
        if category.startswith("__no_") or len(group) < 3:
            continue
        ordered = sorted(group, key=lambda row: as_float(row["utility_mean"]), reverse=True)
        cutoff = max(1, len(ordered) // 3)
        high_pool = ordered[:cutoff]
        high_ids = {option_key(row) for row in high_pool}
        low_pool = [row for row in ordered[-cutoff:] if option_key(row) not in high_ids]
        if high_pool and low_pool:
            high_by_category[category] = high_pool
            low_by_category[category] = low_pool

    valid_categories = [category for category in high_by_category if category in low_by_category]
    if not valid_categories:
        return []
    rows: list[tuple[dict[str, str], dict[str, str], str]] = []
    attempts = 0
    max_attempts = pairs_per_cell * 20
    while len(rows) < pairs_per_cell and attempts < max_attempts:
        attempts += 1
        category = rng.choice(valid_categories)
        high = rng.choice(high_by_category[category])
        low = rng.choice(low_by_category[category])
        if option_key(high) == option_key(low):
            continue
        rows.append((high, low, category))
    return rows


def default_pairs(options: list[dict[str, str]], pairs_per_cell: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_cell: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for option in options:
        by_cell[(option["actor"], option["domain"])].append(option)

    for actor in ACTORS:
        for domain in DOMAINS:
            pairs = sample_tercile_pairs(
                by_cell[(actor, domain)],
                pair_set="default",
                pairs_per_cell=pairs_per_cell,
                seed=seed,
            )
            for idx, (high, low, category) in enumerate(pairs):
                rows.append(pair_row("default", actor, domain, idx, category, high, low))
    return rows


def same_count_pairs(options: list[dict[str, str]], pairs_per_cell: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_cell: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for option in options:
        by_cell[(option["actor"], option["domain"])].append(option)

    for actor in ACTORS:
        for domain in ["religions", "animals", "countries"]:
            pairs = sample_tercile_pairs(
                by_cell[(actor, domain)],
                pair_set="same_count",
                pairs_per_cell=pairs_per_cell,
                seed=seed,
            )
            for idx, (high, low, category) in enumerate(pairs):
                rows.append(pair_row("same_count", actor, domain, idx, category, high, low))
    return rows


def gap_stratified_pairs_for_actor(
    *,
    actor: str,
    domains: list[str],
    pairs_per_domain_bin: int,
    bins: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample within-domain high-low pairs evenly across utility-gap rank bins."""

    rng = random.Random(seed)
    options = utility_options_with_actor_overlays()
    by_cell: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for option in options:
        if option["actor"] == actor:
            by_cell[(option["actor"], option["domain"])].append(option)

    rows: list[dict[str, Any]] = []
    for domain in domains:
        domain_options = by_cell[(actor, domain)]
        all_pairs: list[tuple[float, dict[str, str], dict[str, str]]] = []
        for high in domain_options:
            high_u = as_float(high["utility_mean"])
            for low in domain_options:
                low_u = as_float(low["utility_mean"])
                if high_u > low_u:
                    all_pairs.append((high_u - low_u, high, low))
        if len(all_pairs) < pairs_per_domain_bin * bins:
            raise ValueError(
                f"not enough high-low pairs for actor={actor}, domain={domain}: "
                f"need {pairs_per_domain_bin * bins}, found {len(all_pairs)}"
            )
        all_pairs.sort(key=lambda item: item[0])
        for bin_index in range(bins):
            start = (bin_index * len(all_pairs)) // bins
            end = ((bin_index + 1) * len(all_pairs)) // bins
            bin_pairs = all_pairs[start:end]
            if len(bin_pairs) < pairs_per_domain_bin:
                raise ValueError(
                    f"gap bin {bin_index + 1} for actor={actor}, domain={domain} "
                    f"has only {len(bin_pairs)} pairs"
                )
            sampled = rng.sample(bin_pairs, pairs_per_domain_bin)
            bin_min = min(delta for delta, _high, _low in bin_pairs)
            bin_max = max(delta for delta, _high, _low in bin_pairs)
            for sample_index, (_delta, high, low) in enumerate(sampled):
                pair_idx = len(rows)
                row = pair_row(
                    "gap_stratified",
                    actor,
                    domain,
                    pair_idx,
                    f"gap_bin_{bin_index + 1:02d}",
                    high,
                    low,
                )
                row.update(
                    {
                        "gap_bin": str(bin_index + 1),
                        "gap_bin_count": str(bins),
                        "gap_bin_min_delta_u": f"{bin_min:.9f}",
                        "gap_bin_max_delta_u": f"{bin_max:.9f}",
                        "gap_bin_sample_index": str(sample_index),
                        "gap_sampling_seed": str(seed),
                    }
                )
                rows.append(row)
    return rows


def pair_row(
    pair_set: str,
    actor: str,
    domain: str,
    pair_idx: int,
    category: str,
    high: dict[str, str],
    low: dict[str, str],
) -> dict[str, Any]:
    high_u = as_float(high["utility_mean"])
    low_u = as_float(low["utility_mean"])
    return {
        "pair_set": pair_set,
        "actor": actor,
        "actor_label": ACTOR_LABEL[actor],
        "domain": domain,
        "domain_label": DOMAIN_LABEL[domain],
        "pair_idx": pair_idx,
        "category": category,
        "high_option_id": high["option_id"],
        "high_description": high["description"],
        "high_utility": high["utility_mean"],
        "high_variance": high["utility_variance"],
        "low_option_id": low["option_id"],
        "low_description": low["description"],
        "low_utility": low["utility_mean"],
        "low_variance": low["utility_variance"],
        "delta_u": high_u - low_u,
    }


def select_pairs(
    *,
    default_pairs_per_cell: int = 80,
    same_count_pairs_per_cell: int = 80,
    seed: int = 42,
) -> list[dict[str, Any]]:
    options = utility_options_with_actor_overlays()
    rows = default_pairs(options, default_pairs_per_cell, seed)
    rows.extend(same_count_pairs(options, same_count_pairs_per_cell, seed))
    return rows


def write_selected_pairs(
    *,
    default_pairs_per_cell: int = 80,
    same_count_pairs_per_cell: int = 80,
    seed: int = 42,
) -> None:
    rows = select_pairs(
        default_pairs_per_cell=default_pairs_per_cell,
        same_count_pairs_per_cell=same_count_pairs_per_cell,
        seed=seed,
    )
    write_csv_rows(OUTPUT_INPUTS / "selected_pairs.csv", rows)
