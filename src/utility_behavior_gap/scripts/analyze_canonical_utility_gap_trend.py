#!/usr/bin/env python3
"""Analyze whether larger high-low utility gaps predict stronger high-side performance.

This script uses the current canonical high-N high-low results:

* base repeats 0-4 from ``fund_wording_rerun_manifests__*.tsv``
* extension repeats 5-9 from ``canonical_highn10_manifests__*.tsv``
* panel outcomes from ``canonical_highn_condition_results_pair_outcomes.csv``

The primary gap predictor is a within-actor standardized utility gap, because
the fitted utility scale is most defensible within a model. Raw gap and
within-actor percentile are also exported for sensitivity checks.
"""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import beta
from scipy.stats import pearsonr, spearmanr

from utility_behavior_gap.constants import ACTOR_LABEL, TASK_LABEL
from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API


RUNS = OUTPUT_API / "runs"
FIGURES = ANALYSIS / "figures"
OUTCOMES_PATH = ANALYSIS / "canonical_highn_condition_results_pair_outcomes.csv"
BASE_MANIFEST_GLOB = "fund_wording_rerun_manifests__*.tsv"
HIGHN_MANIFEST_GLOB = "canonical_highn10_manifests__*.tsv"
MANIFEST_GLOBS = (BASE_MANIFEST_GLOB, HIGHN_MANIFEST_GLOB)
TASK_ORDER = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]
DOMAIN_ORDER = ["animals", "countries", "political", "religions"]
GAP_COLUMNS = {
    "raw": "delta_u",
    "actor_z": "delta_u_actor_z",
    "actor_percentile": "delta_u_actor_percentile",
}


def parse_csv_arg(value: str) -> set[str] | None:
    values = {part.strip() for part in value.split(",") if part.strip()}
    return values or None


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()


def suffix(*, gap_scale: str, actors: set[str] | None, tasks: set[str] | None, domains: set[str] | None) -> str:
    parts = [f"gap-{slug(gap_scale)}"]
    for label, values in (("actor", actors), ("task", tasks), ("domain", domains)):
        if values:
            parts.append(f"{label}-" + "+".join(slug(value) for value in sorted(values)))
    return "__" + "__".join(parts)


def read_manifest_paths(glob: str) -> list[Path]:
    paths: list[Path] = []
    for tsv in sorted(RUNS.glob(glob)):
        for line in tsv.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"unexpected manifest row in {tsv}: {line!r}")
            paths.append(Path(parts[2]))
    if not paths:
        raise FileNotFoundError(f"no manifest paths found for {glob}")
    return paths


def load_current_highlow_jobs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for glob in MANIFEST_GLOBS:
        for manifest_path in read_manifest_paths(glob):
            for job in read_jsonl(manifest_path):
                if job.get("condition_a") != "hl_high" or job.get("condition_b") != "hl_low":
                    continue
                if not str(job.get("comparison", "")).endswith("_highlow"):
                    continue
                pair_uid = str(job["pair_uid"])
                if pair_uid in seen:
                    raise ValueError(f"duplicate current high-low pair_uid: {pair_uid}")
                seen.add(pair_uid)
                rows.append(
                    {
                        "pair_uid": pair_uid,
                        "actor": job.get("actor", ""),
                        "actor_label": job.get("actor_label", ACTOR_LABEL.get(str(job.get("actor", "")), "")),
                        "task": job.get("task", ""),
                        "task_label": job.get("task_label", TASK_LABEL.get(str(job.get("task", "")), "")),
                        "domain": job.get("domain", ""),
                        "domain_label": job.get("domain_label", ""),
                        "item_label": job.get("item_label", ""),
                        "repeat": job.get("repeat", ""),
                        "comparison": job.get("comparison", ""),
                        "pair_idx": job.get("pair_idx", ""),
                        "pair_set": job.get("pair_set", ""),
                        "high_utility": float(job.get("high_utility", np.nan)),
                        "low_utility": float(job.get("low_utility", np.nan)),
                        "delta_u": float(job.get("delta_u", np.nan)),
                        "high_description": job.get("high_description", ""),
                        "low_description": job.get("low_description", ""),
                        "high_consequence": job.get("high_consequence", ""),
                        "low_consequence": job.get("low_consequence", ""),
                        "source_manifest": str(manifest_path),
                        "source_run_dir": str(manifest_path.parent),
                    }
                )
    return pd.DataFrame(rows)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def add_actor_gap_scales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cluster_cols = ["actor", "domain", "pair_idx", "high_description", "low_description"]
    unique = df[cluster_cols + ["delta_u"]].drop_duplicates().copy()
    means = unique.groupby("actor")["delta_u"].mean()
    sds = unique.groupby("actor")["delta_u"].std(ddof=1).replace(0, np.nan)
    df["delta_u_actor_z"] = [
        (row.delta_u - means[row.actor]) / sds[row.actor] if pd.notna(sds[row.actor]) else 0.0
        for row in df[["actor", "delta_u"]].itertuples(index=False)
    ]
    percentiles: dict[tuple[str, float], float] = {}
    for actor, sub in unique.groupby("actor", sort=True):
        vals = sub["delta_u"].to_numpy(dtype=float)
        for value in np.unique(vals):
            less = float((vals < value).sum())
            equal = float((vals == value).sum())
            percentiles[(actor, float(value))] = (less + 0.5 * equal) / len(vals)
    df["delta_u_actor_percentile"] = [
        percentiles[(row.actor, float(row.delta_u))]
        for row in df[["actor", "delta_u"]].itertuples(index=False)
    ]
    df["utility_pair_cluster"] = df[cluster_cols].astype(str).agg("|".join, axis=1)
    df["task_item_cluster"] = df[["actor", "task", "domain", "item_label"]].astype(str).agg("|".join, axis=1)
    return df


def load_trials() -> pd.DataFrame:
    jobs = load_current_highlow_jobs()
    outcomes = pd.read_csv(OUTCOMES_PATH)
    outcomes = outcomes[outcomes["condition"].eq("utility")].copy()
    outcomes["target_win"] = as_bool(outcomes["target_win"])
    outcomes["target_loss"] = as_bool(outcomes["target_loss"])
    outcomes["tie"] = as_bool(outcomes["tie"])
    outcomes["resolved"] = as_bool(outcomes["resolved"])
    keep_cols = [
        "pair_uid",
        "source_block",
        "target_win",
        "target_loss",
        "tie",
        "resolved",
        "panel_winner_condition",
    ]
    merged = jobs.merge(outcomes[keep_cols], on="pair_uid", how="left", validate="one_to_one")
    missing = merged["panel_winner_condition"].isna().sum()
    if missing:
        raise ValueError(f"{missing} current high-low manifest rows are missing canonical outcomes")
    merged["high_win"] = np.where(merged["resolved"], merged["target_win"].astype(int), np.nan)
    merged["net_score"] = np.select(
        [merged["target_win"], merged["target_loss"], merged["tie"]],
        [1.0, -1.0, 0.0],
        default=np.nan,
    )
    return add_actor_gap_scales(merged)


def filter_trials(
    df: pd.DataFrame,
    *,
    actors: set[str] | None,
    tasks: set[str] | None,
    domains: set[str] | None,
) -> pd.DataFrame:
    out = df.copy()
    if actors:
        out = out[out["actor"].isin(actors)]
    if tasks:
        out = out[out["task"].isin(tasks)]
    if domains:
        out = out[out["domain"].isin(domains)]
    if out.empty:
        raise ValueError("filters left no high-low utility rows")
    return out


def exact_ci(wins: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return (np.nan, np.nan)
    lo = 0.0 if wins == 0 else float(beta.ppf(alpha / 2, wins, total - wins + 1))
    hi = 1.0 if wins == total else float(beta.ppf(1 - alpha / 2, wins + 1, total - wins))
    return lo, hi


def bootstrap_mean_ci(values: Iterable[float], *, seed: int, iterations: int = 5000) -> tuple[float, float]:
    vals = np.array(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan)
    if len(vals) == 1:
        return (float(vals[0]), float(vals[0]))
    rng = np.random.default_rng(seed)
    boot = rng.choice(vals, size=(iterations, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def build_bins(df: pd.DataFrame, gap_col: str, n_bins: int, seed: int) -> pd.DataFrame:
    ordered = df.sort_values([gap_col, "actor", "task", "domain", "pair_uid"]).reset_index(drop=True)
    ordered["gap_bin"] = pd.qcut(ordered.index + 1, q=min(n_bins, len(ordered)), labels=False) + 1
    rows: list[dict[str, Any]] = []
    for bin_id, sub in ordered.groupby("gap_bin", sort=True):
        resolved = sub[sub["resolved"]].copy()
        wins = int(resolved["target_win"].sum())
        losses = int(resolved["target_loss"].sum())
        n_resolved = wins + losses
        win_lo, win_hi = exact_ci(wins, n_resolved)
        net_lo, net_hi = bootstrap_mean_ci(sub["net_score"], seed=seed + int(bin_id))
        rows.append(
            {
                "gap_bin": int(bin_id),
                "n_pairs": int(len(sub)),
                "n_resolved": int(n_resolved),
                "high_wins": wins,
                "low_wins": losses,
                "ties": int(sub["tie"].sum()),
                "gap_min": float(sub[gap_col].min()),
                "gap_max": float(sub[gap_col].max()),
                "gap_mean": float(sub[gap_col].mean()),
                "raw_delta_u_mean": float(sub["delta_u"].mean()),
                "high_win_rate_excluding_ties": wins / n_resolved if n_resolved else np.nan,
                "high_win_rate_ci_lo": win_lo,
                "high_win_rate_ci_hi": win_hi,
                "net_score_mean": float(sub["net_score"].mean()),
                "net_score_ci_lo": net_lo,
                "net_score_ci_hi": net_hi,
            }
        )
    return pd.DataFrame(rows)


def fixed_effect_terms(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [f"C({col})" for col in cols if col in df.columns and df[col].nunique() > 1]


def cluster_fit_ols(formula: str, data: pd.DataFrame, cluster_col: str):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})


def cluster_fit_logit(formula: str, data: pd.DataFrame, cluster_col: str):
    return smf.glm(formula, data=data, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster_col]}
    )


def regression_row(
    *,
    scope: str,
    outcome: str,
    model: str,
    result: Any,
    term: str,
) -> dict[str, Any]:
    coef = float(result.params[term])
    ci_lo, ci_hi = [float(x) for x in result.conf_int().loc[term]]
    row = {
        "scope": scope,
        "outcome": outcome,
        "model": model,
        "term": term,
        "n": int(result.nobs),
        "coef": coef,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": float(result.pvalues[term]),
    }
    if outcome == "high_win_logit":
        row["odds_ratio"] = math.exp(coef)
        row["or_ci_lo"] = math.exp(ci_lo)
        row["or_ci_hi"] = math.exp(ci_hi)
    else:
        row["odds_ratio"] = np.nan
        row["or_ci_lo"] = np.nan
        row["or_ci_hi"] = np.nan
    return row


def fit_scope(scope: str, df: pd.DataFrame, gap_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resolved = df[df["resolved"]].copy()
    if len(resolved) >= 30 and resolved["high_win"].nunique() > 1:
        fe = fixed_effect_terms(resolved, ["actor", "task", "domain"])
        formula = "high_win ~ gap_value" + (" + " + " + ".join(fe) if fe else "")
        resolved["gap_value"] = resolved[gap_col]
        lpm = cluster_fit_ols(formula, resolved, "utility_pair_cluster")
        rows.append(
            regression_row(scope=scope, outcome="high_win_lpm", model=formula, result=lpm, term="gap_value")
        )
        logit = cluster_fit_logit(formula, resolved, "utility_pair_cluster")
        rows.append(
            regression_row(scope=scope, outcome="high_win_logit", model=formula, result=logit, term="gap_value")
        )
    if len(df) >= 30 and df["net_score"].nunique() > 1:
        fe = fixed_effect_terms(df, ["actor", "task", "domain"])
        formula = "net_score ~ gap_value" + (" + " + " + ".join(fe) if fe else "")
        data = df.copy()
        data["gap_value"] = data[gap_col]
        net = cluster_fit_ols(formula, data, "utility_pair_cluster")
        rows.append(regression_row(scope=scope, outcome="net_score_ols", model=formula, result=net, term="gap_value"))
    return rows


def build_regressions(df: pd.DataFrame, gap_col: str) -> pd.DataFrame:
    rows = fit_scope("overall", df, gap_col)
    for task in TASK_ORDER:
        sub = df[df["task"].eq(task)].copy()
        if not sub.empty:
            rows.extend(fit_scope(f"task={task}", sub, gap_col))
    for domain in DOMAIN_ORDER:
        sub = df[df["domain"].eq(domain)].copy()
        if not sub.empty:
            rows.extend(fit_scope(f"domain={domain}", sub, gap_col))
    for actor in sorted(df["actor"].unique()):
        sub = df[df["actor"].eq(actor)].copy()
        if not sub.empty:
            rows.extend(fit_scope(f"actor={actor}", sub, gap_col))
    return pd.DataFrame(rows)


def correlation_row(scope: str, outcome: str, method: str, x: pd.Series, y: pd.Series) -> dict[str, Any]:
    mask = x.notna() & y.notna()
    x_clean = x[mask].astype(float)
    y_clean = y[mask].astype(float)
    if len(x_clean) < 3 or x_clean.nunique() < 2 or y_clean.nunique() < 2:
        return {
            "scope": scope,
            "outcome": outcome,
            "method": method,
            "n": int(len(x_clean)),
            "estimate": np.nan,
            "p_value": np.nan,
        }
    if method == "pearson":
        estimate, p_value = pearsonr(x_clean, y_clean)
    elif method == "spearman":
        estimate, p_value = spearmanr(x_clean, y_clean)
    else:
        raise ValueError(f"unknown correlation method: {method}")
    return {
        "scope": scope,
        "outcome": outcome,
        "method": method,
        "n": int(len(x_clean)),
        "estimate": float(estimate),
        "p_value": float(p_value),
    }


def build_correlations(df: pd.DataFrame, gap_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("overall", df)]
    scopes.extend((f"task={task}", df[df["task"].eq(task)]) for task in TASK_ORDER)
    scopes.extend((f"domain={domain}", df[df["domain"].eq(domain)]) for domain in DOMAIN_ORDER)
    scopes.extend((f"actor={actor}", df[df["actor"].eq(actor)]) for actor in sorted(df["actor"].unique()))
    for scope, sub in scopes:
        if sub.empty:
            continue
        resolved = sub[sub["resolved"]]
        for method in ("pearson", "spearman"):
            rows.append(correlation_row(scope, "high_win_excluding_ties", method, resolved[gap_col], resolved["high_win"]))
            rows.append(correlation_row(scope, "net_score", method, sub[gap_col], sub["net_score"]))
    return pd.DataFrame(rows)


def fit_marginal_lines(df: pd.DataFrame, gap_col: str) -> tuple[Any, Any]:
    resolved = df[df["resolved"]].copy()
    resolved["gap_value"] = resolved[gap_col]
    logit = smf.glm("high_win ~ gap_value", data=resolved, family=sm.families.Binomial()).fit()
    net_data = df.copy()
    net_data["gap_value"] = net_data[gap_col]
    net = smf.ols("net_score ~ gap_value", data=net_data).fit()
    return logit, net


def plot_bins(df: pd.DataFrame, bins: pd.DataFrame, gap_col: str, png: Path, pdf: Path, title: str) -> None:
    logit, net = fit_marginal_lines(df, gap_col)
    x_min = float(df[gap_col].min())
    x_max = float(df[gap_col].max())
    xs = np.linspace(x_min, x_max, 250)
    win_ys = 1 / (1 + np.exp(-(logit.params["Intercept"] + logit.params["gap_value"] * xs)))
    net_ys = net.params["Intercept"] + net.params["gap_value"] * xs

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 200, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), facecolor="white")
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.axhline(0.5, color="#9CA3AF", ls=(0, (4, 3)), lw=1.1)
    ax.errorbar(
        bins["gap_mean"],
        bins["high_win_rate_excluding_ties"],
        yerr=[
            bins["high_win_rate_excluding_ties"] - bins["high_win_rate_ci_lo"],
            bins["high_win_rate_ci_hi"] - bins["high_win_rate_excluding_ties"],
        ],
        fmt="o",
        color="#2563EB",
        ecolor="#93C5FD",
        capsize=3,
        label="equal-count bins",
    )
    ax.plot(xs, win_ys, color="#C2304A", lw=2.0, label="marginal logistic fit")
    ax.set_ylim(0.25, 0.75)
    ax.set_xlabel("Utility gap")
    ax.set_ylabel("High-side win rate, ties excluded")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.axhline(0.0, color="#9CA3AF", ls=(0, (4, 3)), lw=1.1)
    ax.errorbar(
        bins["gap_mean"],
        bins["net_score_mean"],
        yerr=[
            bins["net_score_mean"] - bins["net_score_ci_lo"],
            bins["net_score_ci_hi"] - bins["net_score_mean"],
        ],
        fmt="o",
        color="#2A8C9E",
        ecolor="#9BC7D0",
        capsize=3,
        label="equal-count bins",
    )
    ax.plot(xs, net_ys, color="#C2304A", lw=2.0, label="marginal linear fit")
    ax.set_ylim(-0.30, 0.30)
    ax.set_xlabel("Utility gap")
    ax.set_ylabel("Net score: high win=1, tie=0, low win=-1")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(axis="y", color="#E5E7EB", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_summary(
    path: Path,
    *,
    df: pd.DataFrame,
    bins: pd.DataFrame,
    regressions: pd.DataFrame,
    correlations: pd.DataFrame,
    gap_scale: str,
    filters_label: str,
    figure_png: Path,
) -> None:
    resolved = df[df["resolved"]]
    high_wins = int(resolved["target_win"].sum())
    low_wins = int(resolved["target_loss"].sum())
    ties = int(df["tie"].sum())
    win_lo, win_hi = exact_ci(high_wins, high_wins + low_wins)
    net_lo, net_hi = bootstrap_mean_ci(df["net_score"], seed=20260615)
    overall_rows = regressions[regressions["scope"].eq("overall")]
    overall_corr = correlations[correlations["scope"].eq("overall")]

    def reg_line(outcome: str) -> str:
        row = overall_rows[overall_rows["outcome"].eq(outcome)].iloc[0]
        if outcome == "high_win_logit":
            return (
                f"{row['odds_ratio']:.3f} odds ratio "
                f"(95% CI {row['or_ci_lo']:.3f}-{row['or_ci_hi']:.3f}, p={row['p_value']:.4g})"
            )
        unit = "win-rate points" if outcome == "high_win_lpm" else "net-score units"
        return (
            f"{row['coef']:.4f} {unit} "
            f"(95% CI {row['ci_lo']:.4f} to {row['ci_hi']:.4f}, p={row['p_value']:.4g})"
        )

    lines = [
        "# Canonical High-Low Utility Gap Trend",
        "",
        f"Filters: {filters_label}.",
        f"Gap predictor: `{gap_scale}`.",
        "",
        "Input: current canonical high-N high-low rows only, merged back to the current base and high-N manifests by `pair_uid`.",
        "",
        "Outcome definitions:",
        "",
        "- Tie-excluded win rate: high-side wins divided by high-side plus low-side wins.",
        "- Net score: high-side win = 1, panel tie = 0, low-side win = -1.",
        "- Primary cross-model gap scale: within-actor standardized utility gap, so one unit is one within-model standard deviation of utility gaps.",
        "",
        "## Overall",
        "",
        f"- Pairs: {len(df)} total; {len(resolved)} non-tied; {ties} ties.",
        f"- High-side tie-excluded win rate: {high_wins}/{high_wins + low_wins} = {pct(high_wins / (high_wins + low_wins))} (95% exact CI {pct(win_lo)}-{pct(win_hi)}).",
        f"- Mean net score: {df['net_score'].mean():.3f} (bootstrap 95% CI {net_lo:.3f} to {net_hi:.3f}).",
        f"- Adjusted linear win-rate trend: {reg_line('high_win_lpm')}.",
        f"- Adjusted logistic win-rate trend: {reg_line('high_win_logit')}.",
        f"- Adjusted net-score trend: {reg_line('net_score_ols')}.",
        f"- Pearson correlation with tie-excluded high wins: {overall_corr[(overall_corr['outcome'].eq('high_win_excluding_ties')) & (overall_corr['method'].eq('pearson'))]['estimate'].iloc[0]:.3f}.",
        f"- Pearson correlation with net score: {overall_corr[(overall_corr['outcome'].eq('net_score')) & (overall_corr['method'].eq('pearson'))]['estimate'].iloc[0]:.3f}.",
        "",
        "The adjusted models include available actor, task, and domain fixed effects and cluster standard errors by utility-pair cluster.",
        "",
        "## Binned Summary",
        "",
        "| bin | mean gap | mean raw delta_u | pairs | non-tied | high win rate | net score |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in bins.iterrows():
        lines.append(
            f"| {int(row['gap_bin'])} | {row['gap_mean']:.3f} | {row['raw_delta_u_mean']:.3f} | "
            f"{int(row['n_pairs'])} | {int(row['n_resolved'])} | "
            f"{pct(row['high_win_rate_excluding_ties'])} | {row['net_score_mean']:.3f} |"
        )
    lines.extend(["", "## Figure", "", f"- `{figure_png}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", default="", help="Comma-separated actor ids to include.")
    parser.add_argument("--tasks", default="", help="Comma-separated task ids to include.")
    parser.add_argument("--domains", default="", help="Comma-separated domains to include.")
    parser.add_argument("--gap-scale", choices=sorted(GAP_COLUMNS), default="actor_z")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()

    actors = parse_csv_arg(args.actors)
    tasks = parse_csv_arg(args.tasks)
    domains = parse_csv_arg(args.domains)
    gap_col = GAP_COLUMNS[args.gap_scale]
    filters_label = "; ".join(
        part
        for part in [
            "actors=" + ",".join(sorted(actors)) if actors else "",
            "tasks=" + ",".join(sorted(tasks)) if tasks else "",
            "domains=" + ",".join(sorted(domains)) if domains else "",
        ]
        if part
    ) or "all actors, tasks, and domains"

    trials = filter_trials(load_trials(), actors=actors, tasks=tasks, domains=domains)
    out_suffix = suffix(gap_scale=args.gap_scale, actors=actors, tasks=tasks, domains=domains)
    prefix = f"canonical_highn_utility_gap_trend{out_suffix}"
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    trials_path = ANALYSIS / f"{prefix}_trials.csv"
    bins_path = ANALYSIS / f"{prefix}_bins.csv"
    regressions_path = ANALYSIS / f"{prefix}_regressions.csv"
    correlations_path = ANALYSIS / f"{prefix}_correlations.csv"
    summary_path = ANALYSIS / f"{prefix}_summary.md"
    figure_png = FIGURES / f"{prefix}.png"
    figure_pdf = FIGURES / f"{prefix}.pdf"

    bins = build_bins(trials, gap_col, args.bins, args.seed)
    regressions = build_regressions(trials, gap_col)
    correlations = build_correlations(trials, gap_col)
    plot_bins(
        trials,
        bins,
        gap_col,
        figure_png,
        figure_pdf,
        title=f"Utility-gap trend ({filters_label}; gap scale={args.gap_scale})",
    )
    write_summary(
        summary_path,
        df=trials,
        bins=bins,
        regressions=regressions,
        correlations=correlations,
        gap_scale=args.gap_scale,
        filters_label=filters_label,
        figure_png=figure_png,
    )
    trials.to_csv(trials_path, index=False)
    bins.to_csv(bins_path, index=False)
    regressions.to_csv(regressions_path, index=False)
    correlations.to_csv(correlations_path, index=False)

    print(f"summary: {summary_path}")
    print(f"trials: {trials_path}")
    print(f"bins: {bins_path}")
    print(f"regressions: {regressions_path}")
    print(f"correlations: {correlations_path}")
    print(f"figure: {figure_png}")
    overall = regressions[regressions['scope'].eq('overall')][['outcome', 'coef', 'ci_lo', 'ci_hi', 'p_value']]
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
