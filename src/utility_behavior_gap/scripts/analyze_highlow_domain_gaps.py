#!/usr/bin/env python3
"""Summarize utility gaps by high-low domain and export example pairs."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from utility_behavior_gap.paths import ROOT

RUNS = ROOT / "outputs" / "api" / "runs"
ANALYSIS = ROOT / "outputs" / "analysis"
MANIFEST_GLOB = "fund_wording_rerun_manifests__*.tsv"
DOMAIN_ORDER = ["animals", "countries", "political", "religions"]


def manifest_paths() -> list[Path]:
    paths: list[Path] = []
    for tsv in sorted(RUNS.glob(MANIFEST_GLOB)):
        with tsv.open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    paths.append(Path(parts[2]))
    return paths


def unique_highlow_pairs() -> pd.DataFrame:
    pairs: dict[tuple[str, ...], dict[str, Any]] = {}
    for manifest in manifest_paths():
        with manifest.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                job = json.loads(line)
                if not str(job.get("comparison", "")).endswith("_highlow"):
                    continue
                key = (
                    job["actor"],
                    job["domain"],
                    job["pair_set"],
                    str(job["pair_idx"]),
                    job["high_description"],
                    job["low_description"],
                )
                pairs[key] = {
                    "actor": job["actor"],
                    "domain": job["domain"],
                    "pair_set": job["pair_set"],
                    "pair_idx": job["pair_idx"],
                    "delta_u": float(job["delta_u"]),
                    "high_utility": float(job["high_utility"]),
                    "low_utility": float(job["low_utility"]),
                    "high_description": job["high_description"],
                    "low_description": job["low_description"],
                    "high_consequence": job["high_consequence"],
                    "low_consequence": job["low_consequence"],
                }
    return pd.DataFrame(pairs.values())


def political_pair_judgment_table(judged_path: Path) -> pd.DataFrame:
    jobs_by_uid: dict[str, dict[str, Any]] = {}
    unique_pairs: dict[tuple[str, ...], dict[str, Any]] = {}
    for manifest in manifest_paths():
        with manifest.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                job = json.loads(line)
                if not str(job.get("comparison", "")).endswith("_highlow"):
                    continue
                if job.get("domain") != "political":
                    continue
                jobs_by_uid[job["pair_uid"]] = job
                key = (
                    job["actor"],
                    job["domain"],
                    job["pair_set"],
                    str(job["pair_idx"]),
                    job["high_description"],
                    job["low_description"],
                )
                unique_pairs.setdefault(key, job)

    judged = pd.read_csv(judged_path, keep_default_na=False)
    judged = judged[judged["pair_uid"].isin(jobs_by_uid)].copy()
    judged_by_uid = {row["pair_uid"]: row for _, row in judged.iterrows()}

    task_order = ["essay", "translation", "incident_postmortem", "grant_proposal_abstract"]
    rows: list[dict[str, Any]] = []
    for key, seed_job in sorted(
        unique_pairs.items(), key=lambda item: (item[1]["actor"], float(item[1]["delta_u"]))
    ):
        actor, domain, _pair_set, pair_idx, high_description, low_description = key
        row: dict[str, Any] = {
            "actor": actor,
            "high_utility": float(seed_job["high_utility"]),
            "low_utility": float(seed_job["low_utility"]),
            "delta_u": float(seed_job["delta_u"]),
            "high_description": high_description,
            "low_description": low_description,
        }
        high_wins = 0
        low_wins = 0
        ties = 0
        missing = 0
        for task in task_order:
            task_jobs = [
                job for job in jobs_by_uid.values()
                if job["actor"] == actor
                and job["domain"] == domain
                and str(job["pair_idx"]) == pair_idx
                and job["high_description"] == high_description
                and job["low_description"] == low_description
                and job["task"] == task
            ]
            if len(task_jobs) != 1:
                missing += 1
                row[f"{task}_outcome"] = "missing"
                continue
            job = task_jobs[0]
            judged_row = judged_by_uid.get(job["pair_uid"])
            panel = "missing" if judged_row is None else judged_row["panel_winner_condition"]
            if panel == "hl_high":
                outcome = "high"
                high_wins += 1
            elif panel == "hl_low":
                outcome = "low"
                low_wins += 1
            elif panel == "tie":
                outcome = "tie"
                ties += 1
            else:
                outcome = panel or "missing"
                missing += 1
            row[f"{task}_outcome"] = outcome

        resolved = high_wins + low_wins
        row["net_high_wins_across_tasks"] = high_wins - low_wins
        row["net_high_win_score"] = (high_wins - low_wins) / len(task_order)
        row["task_high_win_rate_excluding_ties"] = high_wins / resolved if resolved else ""
        rows.append(row)

    columns = [
        "actor",
        "high_utility",
        "low_utility",
        "delta_u",
        "net_high_wins_across_tasks",
        "net_high_win_score",
        "task_high_win_rate_excluding_ties",
        "high_description",
        "low_description",
        "essay_outcome",
        "translation_outcome",
        "incident_postmortem_outcome",
        "grant_proposal_abstract_outcome",
    ]
    return pd.DataFrame(rows)[columns]


def quantile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    idx = (len(values) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def add_actor_normalized_columns(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.copy()
    z_scores: list[float] = []
    percentiles: list[float] = []
    for _, row in pairs.iterrows():
        vals = pairs.loc[pairs["actor"].eq(row["actor"]), "delta_u"].sort_values().to_list()
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((val - mean) ** 2 for val in vals) / (len(vals) - 1))
        less = sum(val < row["delta_u"] for val in vals)
        equal = sum(val == row["delta_u"] for val in vals)
        z_scores.append((row["delta_u"] - mean) / sd if sd else 0.0)
        percentiles.append((less + 0.5 * equal) / len(vals))
    pairs["actor_z_delta_u"] = z_scores
    pairs["actor_percentile_delta_u"] = percentiles
    return pairs


def domain_gap_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain in DOMAIN_ORDER:
        sub = pairs[pairs["domain"].eq(domain)]
        vals = sub["delta_u"].to_list()
        actor_means = sub.groupby("actor")["delta_u"].mean().to_list()
        rows.append({
            "domain": domain,
            "unique_pairs": len(sub),
            "raw_mean_delta_u": sum(vals) / len(vals),
            "median_delta_u": quantile(vals, 0.5),
            "q25_delta_u": quantile(vals, 0.25),
            "q75_delta_u": quantile(vals, 0.75),
            "actor_equal_mean_delta_u": sum(actor_means) / len(actor_means),
            "mean_actor_z_delta_u": sub["actor_z_delta_u"].mean(),
            "mean_actor_percentile_delta_u": sub["actor_percentile_delta_u"].mean(),
        })
    return pd.DataFrame(rows)


def actor_domain_gap_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    return (
        pairs.groupby(["actor", "domain"], as_index=False)
        .agg(unique_pairs=("delta_u", "size"), mean_delta_u=("delta_u", "mean"),
             median_delta_u=("delta_u", "median"))
        .sort_values(["actor", "domain"])
    )


def regression_rows(judged_path: Path) -> list[dict[str, Any]]:
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return []
    judged = pd.read_csv(judged_path, keep_default_na=False)
    judged = judged[judged["comparison"].str.endswith("_highlow")].copy()
    judged = judged[judged["panel_winner_condition"].isin(["hl_high", "hl_low"])].copy()
    judged["high_win"] = judged["panel_winner_condition"].eq("hl_high").astype(int)
    judged["delta_u"] = pd.to_numeric(judged["delta_u"])
    judged["domain"] = pd.Categorical(
        judged["domain"], categories=["countries", "animals", "religions", "political"]
    )
    rows: list[dict[str, Any]] = []
    formulas = [
        "high_win ~ delta_u",
        "high_win ~ delta_u + C(domain)",
        "high_win ~ delta_u + C(domain) + C(actor) + C(task)",
    ]
    for formula in formulas:
        model = smf.ols(formula, data=judged).fit(
            cov_type="cluster", cov_kwds={"groups": judged["actor"]}
        )
        conf = model.conf_int()
        for term in ["delta_u", "C(domain)[T.animals]", "C(domain)[T.religions]", "C(domain)[T.political]"]:
            if term not in model.params:
                continue
            rows.append({
                "formula": formula,
                "term": term,
                "estimate": model.params[term],
                "ci_lo": conf.loc[term, 0],
                "ci_hi": conf.loc[term, 1],
                "note": "linear probability model; descriptive gap-adjusted check; countries reference domain",
            })
    return rows


def write_markdown(
    domain_summary: pd.DataFrame,
    actor_summary: pd.DataFrame,
    regression: list[dict[str, Any]],
    examples: pd.DataFrame,
    path: Path,
) -> None:
    def f(x: float) -> str:
        return f"{x:.3f}"

    lines = [
        "# High-Low Domain Utility Gaps",
        "",
        "Unique utility pairs are deduplicated across tasks before summarizing utility gaps.",
        "The utility scale is most defensible within actor because each actor has its own fitted utility model.",
        "",
        "## Domain Gap Summary",
        "",
        "| domain | pairs | mean delta_u | median | q25-q75 | actor-z mean | actor-percentile mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in domain_summary.iterrows():
        lines.append(
            f"| {row['domain']} | {int(row['unique_pairs'])} | {f(row['raw_mean_delta_u'])} | "
            f"{f(row['median_delta_u'])} | {f(row['q25_delta_u'])}-{f(row['q75_delta_u'])} | "
            f"{f(row['mean_actor_z_delta_u'])} | {f(row['mean_actor_percentile_delta_u'])} |"
        )

    lines += [
        "",
        "## Actor Means",
        "",
        "| actor | animals | countries | political | religions | largest |",
        "|---|---:|---:|---:|---:|---|",
    ]
    pivot = actor_summary.pivot(index="actor", columns="domain", values="mean_delta_u")
    for actor, row in pivot.sort_index().iterrows():
        vals = {domain: float(row[domain]) for domain in DOMAIN_ORDER}
        largest = max(vals, key=vals.get)
        lines.append(
            f"| {actor} | {f(vals['animals'])} | {f(vals['countries'])} | "
            f"{f(vals['political'])} | {f(vals['religions'])} | {largest} |"
        )

    if regression:
        lines += [
            "",
            "## Gap-Adjusted Check",
            "",
            "Linear probability models use resolved judged rows and cluster SEs by actor. Countries are the reference domain.",
            "",
            "| formula | term | estimate | 95% CI |",
            "|---|---|---:|---:|",
        ]
        for row in regression:
            lines.append(
                f"| `{row['formula']}` | {row['term']} | {f(row['estimate'])} | "
                f"{f(row['ci_lo'])}-{f(row['ci_hi'])} |"
            )

    lines += [
        "",
        "## Largest-Gap Examples",
        "",
    ]
    for domain in DOMAIN_ORDER:
        lines += [f"### {domain.title()}", ""]
        sub = examples[examples["domain"].eq(domain)].head(5)
        for _, row in sub.iterrows():
            lines += [
                f"- `{row['actor']}`, delta_u={f(row['delta_u'])}",
                f"  - High: {row['high_consequence']}",
                f"  - Low: {row['low_consequence']}",
            ]
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    pairs = add_actor_normalized_columns(unique_highlow_pairs())
    domain_summary = domain_gap_summary(pairs)
    actor_summary = actor_domain_gap_summary(pairs)
    examples = pairs.sort_values(["domain", "delta_u"], ascending=[True, False]).copy()
    regression = regression_rows(ANALYSIS / "fund_wording_judged_pairs.csv")

    paths = {
        "domain": ANALYSIS / "highlow_domain_gap_summary.csv",
        "actor_domain": ANALYSIS / "highlow_actor_domain_gap_summary.csv",
        "examples": ANALYSIS / "highlow_domain_gap_examples.csv",
        "regression": ANALYSIS / "highlow_domain_gap_adjusted_lpm.csv",
        "political_pairs": ANALYSIS / "highlow_political_pairs_with_judged_outcomes.csv",
        "markdown": ANALYSIS / "highlow_domain_gap_summary.md",
    }
    domain_summary.to_csv(paths["domain"], index=False)
    actor_summary.to_csv(paths["actor_domain"], index=False)
    examples.to_csv(paths["examples"], index=False)
    political_pair_judgment_table(ANALYSIS / "fund_wording_judged_pairs.csv").to_csv(
        paths["political_pairs"], index=False
    )
    if regression:
        with paths["regression"].open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(regression[0]))
            writer.writeheader()
            writer.writerows(regression)
    write_markdown(domain_summary, actor_summary, regression, examples, paths["markdown"])
    for label, path in paths.items():
        if path.exists():
            print(f"{label}: {path}")


if __name__ == "__main__":
    main()
