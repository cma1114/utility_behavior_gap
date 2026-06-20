#!/usr/bin/env python3
"""Analyze whether absolute high-side utility predicts beating framed neutral."""

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
from scipy.stats import beta, pearsonr, spearmanr

from utility_behavior_gap.paths import ANALYSIS


DEFAULT_PAIR_OUTCOMES = ANALYSIS / "highlow_neutral_bridge__combined_7runs__pair_outcomes.csv"
FIGURES = ANALYSIS / "figures"
TASK_ORDER = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]
DOMAIN_ORDER = ["animals", "countries", "political", "religions"]
UTILITY_COLUMNS = {
    "raw": "high_utility",
    "actor_z": "high_utility_actor_z",
    "actor_percentile": "high_utility_actor_percentile",
}


def parse_csv_arg(value: str) -> set[str] | None:
    values = {part.strip() for part in value.split(",") if part.strip()}
    return values or None


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()


def output_suffix(
    *,
    utility_scale: str,
    actors: set[str] | None,
    tasks: set[str] | None,
    domains: set[str] | None,
) -> str:
    parts = [f"utility-{slug(utility_scale)}"]
    for label, values in (("actor", actors), ("task", tasks), ("domain", domains)):
        if values:
            parts.append(f"{label}-" + "+".join(slug(value) for value in sorted(values)))
    return "__" + "__".join(parts)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_trials(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"no rows in {path}")
    df = df[df["side"].eq("high")].copy()
    df["high_utility"] = pd.to_numeric(df["high_utility"], errors="coerce")
    df["side_net_score"] = pd.to_numeric(df["side_net_score"], errors="coerce")
    df["target_win"] = df["outcome_vs_neutral"].eq("side")
    df["target_loss"] = df["outcome_vs_neutral"].eq("neutral")
    df["tie"] = df["outcome_vs_neutral"].eq("tie")
    df["resolved"] = df["target_win"] | df["target_loss"]
    df["high_win"] = np.where(df["resolved"], df["target_win"].astype(int), np.nan)
    df["net_score"] = df["side_net_score"]
    cluster_cols = ["actor", "domain", "pair_idx", "high_description"]
    df["utility_target_cluster"] = df[cluster_cols].astype(str).agg("|".join, axis=1)
    return add_actor_utility_scales(df)


def add_actor_utility_scales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    unique = df[["actor", "domain", "pair_idx", "high_description", "high_utility"]].drop_duplicates().copy()
    means = unique.groupby("actor")["high_utility"].mean()
    sds = unique.groupby("actor")["high_utility"].std(ddof=1).replace(0, np.nan)
    df["high_utility_actor_z"] = [
        (row.high_utility - means[row.actor]) / sds[row.actor] if pd.notna(sds[row.actor]) else 0.0
        for row in df[["actor", "high_utility"]].itertuples(index=False)
    ]
    percentiles: dict[tuple[str, float], float] = {}
    for actor, sub in unique.groupby("actor", sort=True):
        vals = sub["high_utility"].to_numpy(dtype=float)
        for value in np.unique(vals):
            less = float((vals < value).sum())
            equal = float((vals == value).sum())
            percentiles[(actor, float(value))] = (less + 0.5 * equal) / len(vals)
    df["high_utility_actor_percentile"] = [
        percentiles[(row.actor, float(row.high_utility))]
        for row in df[["actor", "high_utility"]].itertuples(index=False)
    ]
    return df


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
        raise ValueError("filters left no bridge rows")
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


def build_bins(df: pd.DataFrame, utility_col: str, n_bins: int, seed: int) -> pd.DataFrame:
    ordered = df.sort_values([utility_col, "actor", "task", "domain", "bridge_pair_uid"]).reset_index(drop=True)
    ordered["utility_bin"] = pd.qcut(ordered.index + 1, q=min(n_bins, len(ordered)), labels=False) + 1
    rows: list[dict[str, Any]] = []
    for bin_id, sub in ordered.groupby("utility_bin", sort=True):
        resolved = sub[sub["resolved"]]
        wins = int(resolved["target_win"].sum())
        losses = int(resolved["target_loss"].sum())
        n_resolved = wins + losses
        win_lo, win_hi = exact_ci(wins, n_resolved)
        net_lo, net_hi = bootstrap_mean_ci(sub["net_score"], seed=seed + int(bin_id))
        rows.append(
            {
                "utility_bin": int(bin_id),
                "n_pairs": int(len(sub)),
                "n_resolved": int(n_resolved),
                "high_wins": wins,
                "neutral_wins": losses,
                "ties": int(sub["tie"].sum()),
                "utility_min": float(sub[utility_col].min()),
                "utility_max": float(sub[utility_col].max()),
                "utility_mean": float(sub[utility_col].mean()),
                "raw_high_utility_mean": float(sub["high_utility"].mean()),
                "high_win_rate_excluding_ties": wins / n_resolved if n_resolved else np.nan,
                "high_win_rate_ci_lo": win_lo,
                "high_win_rate_ci_hi": win_hi,
                "net_score_mean": float(sub["net_score"].mean()),
                "net_score_ci_lo": net_lo,
                "net_score_ci_hi": net_hi,
            }
        )
    return pd.DataFrame(rows)


def fixed_effect_terms(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [f"C({column})" for column in columns if column in df.columns and df[column].nunique() > 1]


def regression_row(scope: str, outcome: str, model: str, result: Any, term: str) -> dict[str, Any]:
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


def fit_scope(scope: str, df: pd.DataFrame, utility_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resolved = df[df["resolved"]].copy()
    if len(resolved) >= 30 and resolved["high_win"].nunique() > 1:
        resolved["utility_value"] = resolved[utility_col]
        fe = fixed_effect_terms(resolved, ["actor", "task", "domain"])
        formula = "high_win ~ utility_value" + (" + " + " + ".join(fe) if fe else "")
        lpm = smf.ols(formula, data=resolved).fit(
            cov_type="cluster", cov_kwds={"groups": resolved["utility_target_cluster"]}
        )
        rows.append(regression_row(scope, "high_win_lpm", formula, lpm, "utility_value"))
        logit = smf.glm(formula, data=resolved, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": resolved["utility_target_cluster"]}
        )
        rows.append(regression_row(scope, "high_win_logit", formula, logit, "utility_value"))
    if len(df) >= 30 and df["net_score"].nunique() > 1:
        data = df.copy()
        data["utility_value"] = data[utility_col]
        fe = fixed_effect_terms(data, ["actor", "task", "domain"])
        formula = "net_score ~ utility_value" + (" + " + " + ".join(fe) if fe else "")
        net = smf.ols(formula, data=data).fit(
            cov_type="cluster", cov_kwds={"groups": data["utility_target_cluster"]}
        )
        rows.append(regression_row(scope, "net_score_ols", formula, net, "utility_value"))
    return rows


def build_regressions(df: pd.DataFrame, utility_col: str) -> pd.DataFrame:
    rows = fit_scope("overall", df, utility_col)
    for task in TASK_ORDER:
        sub = df[df["task"].eq(task)]
        if not sub.empty:
            rows.extend(fit_scope(f"task={task}", sub, utility_col))
    for domain in DOMAIN_ORDER:
        sub = df[df["domain"].eq(domain)]
        if not sub.empty:
            rows.extend(fit_scope(f"domain={domain}", sub, utility_col))
    for actor in sorted(df["actor"].unique()):
        sub = df[df["actor"].eq(actor)]
        if not sub.empty:
            rows.extend(fit_scope(f"actor={actor}", sub, utility_col))
    return pd.DataFrame(rows)


def correlation_row(scope: str, outcome: str, method: str, x: pd.Series, y: pd.Series) -> dict[str, Any]:
    mask = x.notna() & y.notna()
    x_clean = x[mask].astype(float)
    y_clean = y[mask].astype(float)
    if len(x_clean) < 3 or x_clean.nunique() < 2 or y_clean.nunique() < 2:
        return {"scope": scope, "outcome": outcome, "method": method, "n": int(len(x_clean)), "estimate": np.nan, "p_value": np.nan}
    if method == "pearson":
        estimate, p_value = pearsonr(x_clean, y_clean)
    elif method == "spearman":
        estimate, p_value = spearmanr(x_clean, y_clean)
    else:
        raise ValueError(f"unknown method: {method}")
    return {"scope": scope, "outcome": outcome, "method": method, "n": int(len(x_clean)), "estimate": float(estimate), "p_value": float(p_value)}


def build_correlations(df: pd.DataFrame, utility_col: str) -> pd.DataFrame:
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
            rows.append(correlation_row(scope, "high_win_excluding_ties", method, resolved[utility_col], resolved["high_win"]))
            rows.append(correlation_row(scope, "net_score", method, sub[utility_col], sub["net_score"]))
    return pd.DataFrame(rows)


def fit_marginal_lines(df: pd.DataFrame, utility_col: str) -> tuple[Any, Any]:
    resolved = df[df["resolved"]].copy()
    resolved["utility_value"] = resolved[utility_col]
    logit = smf.glm("high_win ~ utility_value", data=resolved, family=sm.families.Binomial()).fit()
    data = df.copy()
    data["utility_value"] = data[utility_col]
    net = smf.ols("net_score ~ utility_value", data=data).fit()
    return logit, net


def plot_bins(df: pd.DataFrame, bins: pd.DataFrame, utility_col: str, png: Path, pdf: Path, title: str) -> None:
    logit, net = fit_marginal_lines(df, utility_col)
    xs = np.linspace(float(df[utility_col].min()), float(df[utility_col].max()), 250)
    win_ys = 1 / (1 + np.exp(-(logit.params["Intercept"] + logit.params["utility_value"] * xs)))
    net_ys = net.params["Intercept"] + net.params["utility_value"] * xs

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 200, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), facecolor="white")
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.axhline(0.5, color="#9CA3AF", ls=(0, (4, 3)), lw=1.1)
    ax.errorbar(
        bins["utility_mean"],
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
    ax.set_xlabel("High-side fitted utility")
    ax.set_ylabel("High-utility side win rate, ties excluded")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.axhline(0.0, color="#9CA3AF", ls=(0, (4, 3)), lw=1.1)
    ax.errorbar(
        bins["utility_mean"],
        bins["net_score_mean"],
        yerr=[bins["net_score_mean"] - bins["net_score_ci_lo"], bins["net_score_ci_hi"] - bins["net_score_mean"]],
        fmt="o",
        color="#2A8C9E",
        ecolor="#9BC7D0",
        capsize=3,
        label="equal-count bins",
    )
    ax.plot(xs, net_ys, color="#C2304A", lw=2.0, label="marginal linear fit")
    ax.set_ylim(-0.30, 0.30)
    ax.set_xlabel("High-side fitted utility")
    ax.set_ylabel("Net score: high win=1, tie=0, neutral win=-1")
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


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_summary(
    path: Path,
    *,
    df: pd.DataFrame,
    bins: pd.DataFrame,
    regressions: pd.DataFrame,
    correlations: pd.DataFrame,
    utility_scale: str,
    filters_label: str,
    figure_png: Path,
    input_path: Path,
) -> None:
    resolved = df[df["resolved"]]
    wins = int(resolved["target_win"].sum())
    neutral_wins = int(resolved["target_loss"].sum())
    ties = int(df["tie"].sum())
    win_lo, win_hi = exact_ci(wins, wins + neutral_wins)
    net_lo, net_hi = bootstrap_mean_ci(df["net_score"], seed=20260615)
    overall = regressions[regressions["scope"].eq("overall")]
    overall_corr = correlations[correlations["scope"].eq("overall")]

    def reg_line(outcome: str) -> str:
        row = overall[overall["outcome"].eq(outcome)].iloc[0]
        if outcome == "high_win_logit":
            return f"{row['odds_ratio']:.3f} odds ratio (95% CI {row['or_ci_lo']:.3f}-{row['or_ci_hi']:.3f}, p={row['p_value']:.4g})"
        unit = "win-rate points" if outcome == "high_win_lpm" else "net-score units"
        return f"{row['coef']:.4f} {unit} (95% CI {row['ci_lo']:.4f} to {row['ci_hi']:.4f}, p={row['p_value']:.4g})"

    pearson_win = overall_corr[
        overall_corr["outcome"].eq("high_win_excluding_ties") & overall_corr["method"].eq("pearson")
    ]["estimate"].iloc[0]
    pearson_net = overall_corr[overall_corr["outcome"].eq("net_score") & overall_corr["method"].eq("pearson")][
        "estimate"
    ].iloc[0]

    lines = [
        "# High Utility Versus Framed Neutral: Absolute Utility Trend",
        "",
        f"Filters: {filters_label}.",
        f"Utility predictor: `{utility_scale}`.",
        f"Input pair outcomes: `{input_path}`.",
        "",
        "Outcome definitions:",
        "",
        "- Tie-excluded win rate: high-utility side wins divided by high-utility plus framed-neutral wins.",
        "- Net score: high-utility win = 1, panel tie = 0, framed-neutral win = -1.",
        "- Primary cross-model utility scale: within-actor standardized high-side utility, so one unit is one within-model standard deviation of high-side utility values.",
        "",
        "## Overall",
        "",
        f"- Pairs: {len(df)} total; {len(resolved)} non-tied; {ties} ties.",
        f"- High-utility side tie-excluded win rate: {wins}/{wins + neutral_wins} = {pct(wins / (wins + neutral_wins))} (95% exact CI {pct(win_lo)}-{pct(win_hi)}).",
        f"- Mean net score: {df['net_score'].mean():.3f} (bootstrap 95% CI {net_lo:.3f} to {net_hi:.3f}).",
        f"- Adjusted linear win-rate trend: {reg_line('high_win_lpm')}.",
        f"- Adjusted logistic win-rate trend: {reg_line('high_win_logit')}.",
        f"- Adjusted net-score trend: {reg_line('net_score_ols')}.",
        f"- Pearson correlation with tie-excluded high wins: {pearson_win:.3f}.",
        f"- Pearson correlation with net score: {pearson_net:.3f}.",
        "",
        "The adjusted models include available actor, task, and domain fixed effects and cluster standard errors by high-utility target cluster.",
        "",
        "## Binned Summary",
        "",
        "| bin | mean utility | mean raw high utility | pairs | non-tied | high win rate | net score |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in bins.iterrows():
        lines.append(
            f"| {int(row['utility_bin'])} | {row['utility_mean']:.3f} | {row['raw_high_utility_mean']:.3f} | "
            f"{int(row['n_pairs'])} | {int(row['n_resolved'])} | {pct(row['high_win_rate_excluding_ties'])} | {row['net_score_mean']:.3f} |"
        )
    lines.extend(["", "## Figure", "", f"- `{figure_png}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-outcomes", type=Path, default=DEFAULT_PAIR_OUTCOMES)
    parser.add_argument("--actors", default="", help="Comma-separated actor ids to include.")
    parser.add_argument("--tasks", default="", help="Comma-separated task ids to include.")
    parser.add_argument("--domains", default="", help="Comma-separated domains to include.")
    parser.add_argument("--utility-scale", choices=sorted(UTILITY_COLUMNS), default="actor_z")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()

    actors = parse_csv_arg(args.actors)
    tasks = parse_csv_arg(args.tasks)
    domains = parse_csv_arg(args.domains)
    utility_col = UTILITY_COLUMNS[args.utility_scale]
    filters_label = "; ".join(
        part
        for part in [
            "actors=" + ",".join(sorted(actors)) if actors else "",
            "tasks=" + ",".join(sorted(tasks)) if tasks else "",
            "domains=" + ",".join(sorted(domains)) if domains else "",
        ]
        if part
    ) or "all actors, tasks, and domains"

    trials = filter_trials(load_trials(args.pair_outcomes), actors=actors, tasks=tasks, domains=domains)
    out_suffix = output_suffix(utility_scale=args.utility_scale, actors=actors, tasks=tasks, domains=domains)
    prefix = f"high_utility_neutral_trend{out_suffix}"
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    trials_path = ANALYSIS / f"{prefix}_trials.csv"
    bins_path = ANALYSIS / f"{prefix}_bins.csv"
    regressions_path = ANALYSIS / f"{prefix}_regressions.csv"
    correlations_path = ANALYSIS / f"{prefix}_correlations.csv"
    summary_path = ANALYSIS / f"{prefix}_summary.md"
    figure_png = FIGURES / f"{prefix}.png"
    figure_pdf = FIGURES / f"{prefix}.pdf"

    bins = build_bins(trials, utility_col, args.bins, args.seed)
    regressions = build_regressions(trials, utility_col)
    correlations = build_correlations(trials, utility_col)
    plot_bins(
        trials,
        bins,
        utility_col,
        figure_png,
        figure_pdf,
        title=f"High utility versus framed neutral ({filters_label}; utility scale={args.utility_scale})",
    )
    write_summary(
        summary_path,
        df=trials,
        bins=bins,
        regressions=regressions,
        correlations=correlations,
        utility_scale=args.utility_scale,
        filters_label=filters_label,
        figure_png=figure_png,
        input_path=args.pair_outcomes,
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
    overall = regressions[regressions["scope"].eq("overall")][["outcome", "coef", "ci_lo", "ci_hi", "p_value"]]
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
