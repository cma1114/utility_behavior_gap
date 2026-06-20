#!/usr/bin/env python3
"""Full modgrid condition summaries with crossed bootstrap confidence intervals.

The corrected fund-wording rerun currently contains high-low, moral, direct
instruction/headroom, and essay framing rows. It does not contain amount rows.
By default this script therefore reports amount from the legacy modgrid
analysis and marks those rows explicitly in ``source_dataset``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utility_behavior_gap.paths import ROOT

ANALYSIS = ROOT / "outputs" / "analysis"
FUND_PAIRS = ANALYSIS / "fund_wording_judged_pairs.csv"
LEGACY_PAIRS = ANALYSIS / "modgrid_judged_pairs.csv"
FUND_MORAL_REFUSALS = ANALYSIS / "fund_wording_moral_refusal_classifications.jsonl"

REQUESTED_CONDITIONS = {
    "amount",
    "direct_instruction",
    "high_low",
    "moral",
}


def condition_from_comparison(comparison: str) -> str | None:
    if comparison.endswith("_amount"):
        return "amount"
    if comparison.endswith("_headroom"):
        return "direct_instruction"
    if comparison.endswith("_highlow"):
        return "high_low"
    if comparison.endswith("_moral"):
        return "moral"
    return None


def task_family(comparison: str) -> str:
    if comparison.startswith("modgrid_essay_"):
        return "essay"
    if comparison.startswith("modgrid_translation_"):
        return "translation"
    if comparison.startswith("modgrid_grant_"):
        return "grant_proposal_abstract"
    if comparison.startswith("modgrid_incident_"):
        return "incident_postmortem"
    return ""


def wilson(wins: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def load_pairs(include_legacy_amount: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if FUND_PAIRS.exists():
        fund = pd.read_csv(FUND_PAIRS, keep_default_na=False)
        fund["condition"] = fund["comparison"].map(condition_from_comparison)
        fund = fund[fund["condition"].isin(REQUESTED_CONDITIONS)].copy()
        fund["source_dataset"] = "fund_wording_rerun"
        frames.append(fund)

    if include_legacy_amount:
        has_fund_amount = bool(frames) and any(frame["condition"].eq("amount").any() for frame in frames)
        if not has_fund_amount:
            legacy = pd.read_csv(LEGACY_PAIRS, keep_default_na=False)
            legacy["condition"] = legacy["comparison"].map(condition_from_comparison)
            legacy = legacy[legacy["condition"].eq("amount")].copy()
            legacy["source_dataset"] = "legacy_modgrid_amount_unrerun"
            frames.append(legacy)

    if not frames:
        raise SystemExit("No requested modgrid pair files found.")

    df = pd.concat(frames, ignore_index=True)
    df["condition"] = df["condition"].fillna("")
    df = df[df["condition"].isin(REQUESTED_CONDITIONS)].copy()
    df["task"] = df["task"].replace("", np.nan)
    df["task"] = df["task"].fillna(df["comparison"].map(task_family))
    df["domain"] = df["domain"].replace("", "not_applicable")
    df["is_tie"] = df["panel_winner_condition"].eq("tie")
    df["resolved"] = ~df["panel_winner_condition"].isin(["tie", "unresolved", ""])
    df["target_win"] = np.where(
        df["resolved"],
        df["panel_winner_condition"].eq(df["predicted_condition"]).astype(float),
        np.nan,
    )
    return df


def factor_values(df: pd.DataFrame, factor: str, filter_values: dict[str, str]) -> list[str]:
    if factor in filter_values:
        return [filter_values[factor]]
    values = sorted(str(v) for v in df[factor].dropna().unique())
    if factor == "domain":
        values = [v for v in values if v != "not_applicable"] or ["not_applicable"]
    return values


def ci_method_label(filter_values: dict[str, str], factors: list[str], n_bootstrap: int) -> str:
    resampled = [factor for factor in factors if factor not in filter_values and factor != "domain"]
    if resampled:
        return (
            f"balanced crossed bootstrap; resampled factors={'+'.join(resampled)}; "
            f"domains fixed when present; raw trials resampled within remaining cells; B={n_bootstrap}"
        )
    return f"within-cell trial bootstrap; domains fixed when present; B={n_bootstrap}"


def summarize(df: pd.DataFrame, filter_values: dict[str, str], *, n_bootstrap: int, seed: int) -> dict[str, Any]:
    sub = df.copy()
    for col, val in filter_values.items():
        sub = sub[sub[col].astype(str).eq(str(val))]

    factors = ["actor", "task"]
    if sub["domain"].nunique(dropna=True) > 1 or (
        "domain" in filter_values and filter_values["domain"] != "not_applicable"
    ):
        factors.append("domain")

    resolved = sub[sub["resolved"]].copy()
    wins = int(resolved["target_win"].sum()) if not resolved.empty else 0
    n = int(len(resolved))
    ties = int(sub["is_tie"].sum())
    pooled_lo, pooled_hi = wilson(wins, n)

    if n == 0:
        return {
            **filter_values,
            "resolved_n": 0,
            "target_wins": 0,
            "losses": 0,
            "ties": ties,
            "mean_win_rate": math.nan,
            "ci_lo": math.nan,
            "ci_hi": math.nan,
            "ci_method": "no resolved pairs",
            "n_bootstrap": 0,
            "source_dataset": ",".join(sorted(sub["source_dataset"].unique())),
        }

    arrays: dict[tuple[str, ...], np.ndarray] = {}
    for key, group in resolved.groupby(factors, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        arrays[tuple(str(part) for part in key)] = group["target_win"].to_numpy(dtype=float)

    # Equal-cell estimand: each observed actor x task x domain cell gets equal
    # weight, preventing cells with fewer ties from dominating the mean.
    point = float(np.mean([array.mean() for array in arrays.values()]))

    rng = np.random.default_rng(seed)
    factor_levels = [factor_values(resolved, factor, filter_values) for factor in factors]
    estimates = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sampled_levels = []
        for factor, levels in zip(factors, factor_levels):
            levels_array = np.array(levels, dtype=object)
            if factor in filter_values or factor == "domain" or len(levels_array) <= 1:
                sampled_levels.append(levels_array)
            else:
                sampled_levels.append(rng.choice(levels_array, size=len(levels_array), replace=True))

        cell_rates: list[float] = []
        for actor in sampled_levels[0]:
            for task in sampled_levels[1]:
                if len(factors) == 2:
                    array = arrays.get((str(actor), str(task)))
                    if array is None:
                        continue
                    cell_rates.append(float(rng.choice(array, size=len(array), replace=True).mean()))
                    continue
                for domain in sampled_levels[2]:
                    array = arrays.get((str(actor), str(task), str(domain)))
                    if array is None:
                        continue
                    cell_rates.append(float(rng.choice(array, size=len(array), replace=True).mean()))
        estimates[idx] = float(np.mean(cell_rates)) if cell_rates else math.nan

    estimates = estimates[~np.isnan(estimates)]
    return {
        **filter_values,
        "resolved_n": n,
        "target_wins": wins,
        "losses": n - wins,
        "ties": ties,
        "mean_win_rate": point,
            "pooled_win_rate": wins / n,
            "pooled_wilson_ci_lo": pooled_lo,
            "pooled_wilson_ci_hi": pooled_hi,
            "ci_lo": float(np.quantile(estimates, 0.025)) if len(estimates) else math.nan,
            "ci_hi": float(np.quantile(estimates, 0.975)) if len(estimates) else math.nan,
        "ci_method": ci_method_label(filter_values, factors, n_bootstrap),
        "n_bootstrap": n_bootstrap,
        "source_dataset": ",".join(sorted(sub["source_dataset"].unique())),
    }


def add_breakout_rows(
    rows: list[dict[str, Any]],
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    condition: str,
    n_bootstrap: int,
    seed: int,
) -> None:
    label = "total" if not group_cols else "_x_".join(
        "model" if col == "actor" else col for col in group_cols
    )
    if not group_cols:
        groups = [()]
    elif len(group_cols) == 1:
        groups = df.groupby(group_cols[0], sort=True).groups.keys()
    else:
        groups = df.groupby(group_cols, sort=True).groups.keys()
    for idx, key in enumerate(groups):
        if group_cols:
            key_tuple = key if isinstance(key, tuple) else (key,)
            filters = {col: str(val) for col, val in zip(group_cols, key_tuple)}
        else:
            filters = {}
        row = summarize(df, filters, n_bootstrap=n_bootstrap, seed=seed + len(rows) + idx)
        row["condition"] = condition
        row["breakout"] = label
        for col in ["actor", "task", "domain"]:
            row.setdefault(col, "all")
        rows.append(row)


def write_markdown(results: pd.DataFrame, path: Path, *, include_legacy_amount: bool) -> None:
    def pct(x: float) -> str:
        if pd.isna(x):
            return ""
        return f"{100 * x:.1f}%"

    lines = [
        "# Full modgrid results",
        "",
        "Win rate is the target-side panel win rate among resolved pairs; panel ties are excluded from the denominator.",
        "The reported mean is an equal-cell mean over the remaining crossed design cells in that row.",
        "CIs are nonparametric crossed bootstraps over the unfixed design factors, with raw trials resampled inside cells.",
        f"Legacy amount included: {include_legacy_amount}",
        f"Fund-wording moral refusal screen present: {FUND_MORAL_REFUSALS.exists()}",
        "",
        "## Total",
        "",
        "| condition | source | resolved | wins | ties | pooled | pooled Wilson CI | equal-cell mean | equal-cell bootstrap CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    total = results[results["breakout"].eq("total")].sort_values("condition")
    for _, row in total.iterrows():
        lines.append(
            f"| {row['condition']} | {row['source_dataset']} | {int(row['resolved_n'])} | "
            f"{int(row['target_wins'])} | {int(row['ties'])} | {pct(row['pooled_win_rate'])} | "
            f"{pct(row['pooled_wilson_ci_lo'])}-{pct(row['pooled_wilson_ci_hi'])} | "
            f"{pct(row['mean_win_rate'])} | {pct(row['ci_lo'])}-{pct(row['ci_hi'])} |"
        )

    for breakout, title in [
        ("task", "By Task"),
        ("model", "By Model"),
        ("domain", "By Domain"),
        ("model_x_task", "By Model x Task"),
        ("task_x_domain", "By Task x Domain"),
    ]:
        sub = results[results["breakout"].eq(breakout)].copy()
        if sub.empty:
            continue
        lines += ["", f"## {title}", ""]
        if breakout in {"model_x_task", "task_x_domain"}:
            lines += [f"See CSV for full `{breakout}` table; first 20 rows shown.", ""]
            sub = sub.head(20)
        columns = [col for col in ["condition", "actor", "task", "domain"] if col in sub.columns]
        header = columns + [
            "source_dataset", "resolved_n", "target_wins", "ties",
            "pooled", "pooled_wilson_ci", "equal_cell_mean", "equal_cell_bootstrap_ci",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for _, row in sub.sort_values(columns).iterrows():
            values = [str(row.get(col, "")) for col in columns]
            values += [
                str(row["source_dataset"]),
                str(int(row["resolved_n"])),
                str(int(row["target_wins"])),
                str(int(row["ties"])),
                pct(row["pooled_win_rate"]),
                f"{pct(row['pooled_wilson_ci_lo'])}-{pct(row['pooled_wilson_ci_hi'])}",
                pct(row["mean_win_rate"]),
                f"{pct(row['ci_lo'])}-{pct(row['ci_hi'])}",
            ]
            lines.append("| " + " | ".join(values) + " |")

    lines += [
        "",
        "## Data Note",
        "",
        "The corrected fund-wording rerun does not currently contain amount rows.",
        "Rows with `source_dataset=legacy_modgrid_amount_unrerun` are amount results from the prior modgrid run.",
        "Domain breakouts are only emitted for high-low because the other three conditions do not contain a utility-domain factor.",
        "Fund-wording moral rows are not refusal-screened unless `fund_wording_moral_refusal_classifications.jsonl` exists.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument(
        "--no-legacy-amount",
        action="store_true",
        help="Do not include legacy amount rows when corrected fund-wording amount rows are absent.",
    )
    args = parser.parse_args()

    include_legacy_amount = not args.no_legacy_amount
    pairs = load_pairs(include_legacy_amount=include_legacy_amount)

    rows: list[dict[str, Any]] = []
    for condition in ["direct_instruction", "amount", "moral", "high_low"]:
        cond_df = pairs[pairs["condition"].eq(condition)].copy()
        if cond_df.empty:
            continue
        for group_cols in [
            [],
            ["actor"],
            ["task"],
            ["domain"],
            ["actor", "task"],
            ["actor", "domain"],
            ["task", "domain"],
            ["actor", "task", "domain"],
        ]:
            # Domain breakouts are only meaningful for high-low. The other
            # conditions have no utility-domain factor in the design.
            if "domain" in group_cols and condition != "high_low":
                continue
            add_breakout_rows(
                rows,
                cond_df,
                group_cols,
                condition=condition,
                n_bootstrap=args.bootstraps,
                seed=args.seed,
            )

    results = pd.DataFrame(rows)
    ordered_cols = [
        "condition",
        "breakout",
        "actor",
        "task",
        "domain",
        "source_dataset",
        "resolved_n",
        "target_wins",
        "losses",
        "ties",
        "pooled_win_rate",
        "pooled_wilson_ci_lo",
        "pooled_wilson_ci_hi",
        "mean_win_rate",
        "ci_lo",
        "ci_hi",
        "ci_method",
        "n_bootstrap",
    ]
    for col in ordered_cols:
        if col not in results.columns:
            results[col] = ""
    results = results[ordered_cols].sort_values(["condition", "breakout", "actor", "task", "domain"])

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS / "modgrid_full_breakout_results.csv"
    md_path = ANALYSIS / "modgrid_full_breakout_results.md"
    results.to_csv(csv_path, index=False)
    write_markdown(results, md_path, include_legacy_amount=include_legacy_amount)
    print(f"wrote {len(results)} rows to {csv_path}")
    print(f"wrote summary to {md_path}")


if __name__ == "__main__":
    main()
