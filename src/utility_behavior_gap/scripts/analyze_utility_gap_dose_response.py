#!/usr/bin/env python3
"""Test whether larger fitted utility gaps predict high-side wins.

Primary estimand:
  Pair-level linear trend in high-side win probability per one-unit increase in
  delta_u = u_high - u_low, excluding tied panels.

Input is the cleaned live high-low intervention snapshots:
  outputs/raw/highlow_intervention__<actor>__<task>__judged_pairs.csv

Outputs:
  outputs/analysis/highlow_intervention_utility_gap_dose_response_trials.csv
  outputs/analysis/highlow_intervention_utility_gap_dose_response_bins.csv
  outputs/analysis/highlow_intervention_utility_gap_dose_response_regression.csv
  outputs/analysis/highlow_intervention_utility_gap_dose_response_summary.md
  outputs/figures/highlow_intervention_utility_gap_dose_response.{pdf,png}
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utility_behavior_gap.constants import ACTORS
from utility_behavior_gap.io_utils import read_csv_rows, write_csv_rows
from utility_behavior_gap.paths import ANALYSIS, FIGURES, OUTPUT_RAW


TASK_ORDER = ["essay", "translation", "incident_postmortem", "grant_proposal_abstract"]
DOMAIN_ORDER = ["religions", "animals", "countries", "political"]
COMPARISON = "highlow_intervention"


def parse_csv_arg(value: str) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()


def filter_suffix(
    *,
    actors: set[str] | None,
    tasks: set[str] | None,
    domains: set[str] | None,
) -> str:
    parts: list[str] = []
    for label, values in [("actor", actors), ("task", tasks), ("domain", domains)]:
        if not values:
            continue
        value_slug = "+".join(slug(value) for value in sorted(values))
        parts.append(f"{label}-{value_slug}")
    return "" if not parts else "__" + "__".join(parts)


def filter_label(
    *,
    actors: set[str] | None,
    tasks: set[str] | None,
    domains: set[str] | None,
) -> str:
    parts: list[str] = []
    if actors:
        parts.append("actors: " + ", ".join(sorted(actors)))
    if tasks:
        parts.append("tasks: " + ", ".join(sorted(tasks)))
    if domains:
        parts.append("domains: " + ", ".join(sorted(domains)))
    return "all completed snapshots" if not parts else "; ".join(parts)


def output_paths(
    *,
    actors: set[str] | None,
    tasks: set[str] | None,
    domains: set[str] | None,
) -> dict[str, Path]:
    prefix = f"{COMPARISON}_utility_gap_dose_response{filter_suffix(actors=actors, tasks=tasks, domains=domains)}"
    return {
        "trials": ANALYSIS / f"{prefix}_trials.csv",
        "bins": ANALYSIS / f"{prefix}_bins.csv",
        "regression": ANALYSIS / f"{prefix}_regression.csv",
        "summary": ANALYSIS / f"{prefix}_summary.md",
        "pdf": FIGURES / f"{prefix}.pdf",
        "png": FIGURES / f"{prefix}.png",
        "coef_pdf": FIGURES / f"{prefix}_slope_coefficients.pdf",
        "coef_png": FIGURES / f"{prefix}_slope_coefficients.png",
    }


def wilson(wins: int, total: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if total == 0:
        return (float("nan"),) * 3
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def judged_pair_paths() -> list[Path]:
    return sorted(OUTPUT_RAW.glob(f"{COMPARISON}__*__*__judged_pairs.csv"))


def trial_from_judged_row(row: dict[str, str]) -> dict[str, Any] | None:
    winner = (row.get("counted_winner_condition") or row.get("panel_winner_condition") or "").strip()
    if winner not in {"high", "low", "tie"}:
        return None
    try:
        delta_u = float(row["delta_u"])
        u_high = float(row["high_utility"])
        u_low = float(row["low_utility"])
    except (KeyError, TypeError, ValueError):
        return None
    pair_cluster = "|".join(
        [
            row.get("actor", ""),
            row.get("domain", ""),
            row.get("pair_idx", ""),
            row.get("high_description", ""),
            row.get("low_description", ""),
        ]
    )
    return {
        "comparison": row.get("comparison", ""),
        "actor": row.get("actor", ""),
        "task": row.get("task", ""),
        "domain": row.get("domain", ""),
        "pair_idx": row.get("pair_idx", ""),
        "item_id": row.get("item_id", ""),
        "item_label": row.get("item_label", ""),
        "pair_uid": row.get("pair_uid", ""),
        "pair_cluster": pair_cluster,
        "high_text": row.get("high_description", ""),
        "low_text": row.get("low_description", ""),
        "u_high": u_high,
        "u_low": u_low,
        "delta_u": delta_u,
        "winner": winner,
        "high_win": "" if winner == "tie" else (1 if winner == "high" else 0),
    }


def build_trials_from_snapshots(
    actors: set[str] | None,
    tasks: set[str] | None,
    domains: set[str] | None,
) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = judged_pair_paths()
    rows: list[dict[str, Any]] = []
    used_paths: list[Path] = []
    for path in paths:
        file_rows = read_csv_rows(path)
        kept = 0
        for row in file_rows:
            if row.get("comparison") != COMPARISON:
                continue
            if actors is not None and row.get("actor") not in actors:
                continue
            if tasks is not None and row.get("task") not in tasks:
                continue
            if domains is not None and row.get("domain") not in domains:
                continue
            trial = trial_from_judged_row(row)
            if trial is None:
                continue
            rows.append(trial)
            kept += 1
        if kept:
            used_paths.append(path)
    return rows, used_paths


def read_trials_csv(
    path: Path,
    actors: set[str] | None,
    tasks: set[str] | None,
    domains: set[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv_rows(path):
        if row.get("comparison", COMPARISON) != COMPARISON:
            continue
        if actors is not None and row.get("actor") not in actors:
            continue
        if tasks is not None and row.get("task") not in tasks:
            continue
        if domains is not None and row.get("domain") not in domains:
            continue
        winner = row.get("winner", "")
        if winner not in {"high", "low", "tie"}:
            continue
        high_win_raw = row.get("high_win", row.get("high_win_excluding_ties", ""))
        high_win = "" if winner == "tie" else int(float(high_win_raw))
        rows.append(
            {
                **row,
                "comparison": row.get("comparison", COMPARISON),
                "delta_u": float(row["delta_u"]),
                "u_high": float(row.get("u_high", row.get("high_utility", ""))),
                "u_low": float(row.get("u_low", row.get("low_utility", ""))),
                "high_win": high_win,
                "pair_cluster": row.get("pair_cluster")
                or row.get("utility_pair_cluster")
                or "|".join([row.get("actor", ""), row.get("domain", ""), row.get("pair_idx", "")]),
            }
        )
    return rows


def trials_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("No high-low trial rows found.")
    df = pd.DataFrame(rows)
    df["delta_u"] = df["delta_u"].astype(float)
    df["high_win_numeric"] = pd.to_numeric(df["high_win"], errors="coerce")
    return df


def non_tied_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["high_win_numeric"].notna()].copy()
    out["high_win"] = out["high_win_numeric"].astype(int)
    return out


def formula_with_fixed_effects(df: pd.DataFrame) -> str:
    terms = ["delta_u"]
    for column in ["actor", "task", "domain"]:
        if column in df.columns and df[column].nunique() > 1:
            terms.append(f"C({column})")
    return "high_win ~ " + " + ".join(terms)


def regression_row(name: str, model_type: str, result: Any, term: str = "delta_u") -> dict[str, Any]:
    coef = float(result.params[term])
    se = float(result.bse[term])
    p_value = float(result.pvalues[term])
    ci_lo, ci_hi = [float(x) for x in result.conf_int().loc[term]]
    row: dict[str, Any] = {
        "model": name,
        "model_type": model_type,
        "n": int(result.nobs),
        "term": term,
        "coef": coef,
        "se_clustered": se,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
    }
    if model_type == "logit":
        row.update(
            {
                "odds_ratio": math.exp(coef),
                "or_ci_lo": math.exp(ci_lo),
                "or_ci_hi": math.exp(ci_hi),
            }
        )
    else:
        row.update({"odds_ratio": "", "or_ci_lo": "", "or_ci_hi": ""})
    return row


def write_bins(df: pd.DataFrame, path: Path, n_bins: int) -> list[dict[str, Any]]:
    ordered = df.sort_values("delta_u").reset_index(drop=True)
    ordered["bin"] = pd.qcut(ordered.index + 1, q=min(n_bins, len(ordered)), labels=False) + 1
    rows: list[dict[str, Any]] = []
    for bin_id, sub in ordered.groupby("bin", sort=True):
        wins = int(sub["high_win"].sum())
        n = int(len(sub))
        rate, ci_lo, ci_hi = wilson(wins, n)
        rows.append(
            {
                "bin": int(bin_id),
                "delta_u_min": float(sub["delta_u"].min()),
                "delta_u_max": float(sub["delta_u"].max()),
                "delta_u_mean": float(sub["delta_u"].mean()),
                "n": n,
                "high_wins": wins,
                "low_wins": n - wins,
                "high_win_rate": rate,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }
        )
    write_csv_rows(path, rows)
    return rows


def write_regressions(df: pd.DataFrame, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cluster_groups = df["pair_cluster"]

    lpm_uncontrolled = smf.ols("high_win ~ delta_u", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": cluster_groups}
    )
    rows.append(regression_row("uncontrolled: high_win ~ delta_u", "linear_probability", lpm_uncontrolled))

    fe_formula = formula_with_fixed_effects(df)
    lpm_fe = smf.ols(fe_formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})
    rows.append(regression_row(f"fixed_effects: {fe_formula}", "linear_probability", lpm_fe))

    logit_uncontrolled = smf.glm("high_win ~ delta_u", data=df, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": cluster_groups}
    )
    rows.append(regression_row("uncontrolled: logit high_win ~ delta_u", "logit", logit_uncontrolled))

    logit_fe = smf.glm(fe_formula, data=df, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": cluster_groups}
    )
    rows.append(regression_row(f"fixed_effects logit: {fe_formula}", "logit", logit_fe))

    for task in TASK_ORDER:
        sub = df[df["task"] == task]
        if len(sub) >= 30 and sub["high_win"].nunique() > 1:
            result = smf.ols("high_win ~ delta_u", data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["pair_cluster"]}
            )
            rows.append(regression_row(f"task={task}: high_win ~ delta_u", "linear_probability", result))

    for domain in DOMAIN_ORDER:
        sub = df[df["domain"] == domain]
        if len(sub) >= 30 and sub["high_win"].nunique() > 1:
            result = smf.ols("high_win ~ delta_u", data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["pair_cluster"]}
            )
            rows.append(regression_row(f"domain={domain}: high_win ~ delta_u", "linear_probability", result))

    write_csv_rows(path, rows)
    return rows


def write_figure(
    df: pd.DataFrame,
    bins: list[dict[str, Any]],
    pdf_path: Path,
    png_path: Path,
    subset_label: str,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    mids = [row["delta_u_mean"] for row in bins]
    rates = [row["high_win_rate"] for row in bins]
    lo_err = [row["high_win_rate"] - row["ci_lo"] for row in bins]
    hi_err = [row["ci_hi"] - row["high_win_rate"] for row in bins]

    ax.axhline(0.5, color="#9CA3AF", ls=(0, (4, 3)), lw=1.2, zorder=1)
    ax.errorbar(
        mids,
        rates,
        yerr=[lo_err, hi_err],
        fmt="o",
        color="#2A8C9E",
        ecolor="#9DBFC7",
        capsize=3,
        ms=7,
        zorder=3,
        label="equal-count bins, Wilson 95% CI",
    )

    model = smf.glm("high_win ~ delta_u", data=df, family=sm.families.Binomial()).fit()
    xs = np.linspace(float(df["delta_u"].min()), float(df["delta_u"].max()), 200)
    ys = 1 / (1 + np.exp(-(model.params["Intercept"] + model.params["delta_u"] * xs)))
    ax.plot(xs, ys, color="#C2304A", lw=2, zorder=2, label="logistic fit")

    ax.set_xlabel("Fitted utility gap: delta_u = u_high - u_low")
    ax.set_ylabel("High-utility-side win rate, excluding ties")
    ax.set_ylim(0.25, 0.75)
    ax.set_title(f"Utility-gap dose-response ({subset_label})", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def coefficient_label(model_name: str) -> str:
    if model_name == "uncontrolled: high_win ~ delta_u":
        return "Marginal linear trend"
    if model_name.startswith("fixed_effects:"):
        return "Domain-adjusted linear trend"
    if model_name.startswith("domain="):
        domain = model_name.split(":", 1)[0].replace("domain=", "")
        return f"Within {domain}"
    if model_name.startswith("task="):
        task = model_name.split(":", 1)[0].replace("task=", "")
        return f"Within {task}"
    return model_name


def write_coefficient_figure(regressions: list[dict[str, Any]], pdf_path: Path, png_path: Path) -> None:
    rows = [
        row
        for row in regressions
        if row["model_type"] == "linear_probability" and row["term"] == "delta_u"
    ]
    if not rows:
        return
    rows = list(reversed(rows))
    labels = [coefficient_label(str(row["model"])) for row in rows]
    coefs = np.array([100 * float(row["coef"]) for row in rows])
    lo = np.array([100 * float(row["ci_lo"]) for row in rows])
    hi = np.array([100 * float(row["ci_hi"]) for row in rows])
    y = np.arange(len(rows))

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "figure.dpi": 200})
    height = max(3.2, 0.42 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(7.3, height), facecolor="white")
    ax.axvline(0, color="#9CA3AF", lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.errorbar(
        coefs,
        y,
        xerr=[coefs - lo, hi - coefs],
        fmt="o",
        color="#2563EB",
        ecolor="#93C5FD",
        capsize=3,
        ms=6,
        zorder=2,
    )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Change in high-side win rate per one utility-gap unit, percentage points")
    ax.set_title("Utility-gap linear trend estimates with 95% confidence intervals", fontsize=12)
    ax.grid(axis="x", color="#E5E7EB", lw=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_summary(
    path: Path,
    trials: pd.DataFrame,
    df: pd.DataFrame,
    bins: list[dict[str, Any]],
    regressions: list[dict[str, Any]],
    used_paths: list[Path],
    subset_label: str,
) -> None:
    high = int(df["high_win"].sum())
    n = int(len(df))
    rate, ci_lo, ci_hi = wilson(high, n)
    ties = int((trials["winner"] == "tie").sum())
    primary = next(row for row in regressions if row["model"].startswith("fixed_effects:"))
    logistic = next(row for row in regressions if row["model"].startswith("fixed_effects logit:"))

    lines = [
        "# Utility-gap dose-response: cleaned high-low intervention",
        "",
        f"Subset: {subset_label}.",
        "",
        "Definitions:",
        "",
        "- High-side win rate: among non-tied judged pairs, the fraction where the output attached to the higher fitted utility won.",
        "- Utility gap, `delta_u`: fitted utility of the high-side consequence minus fitted utility of the low-side consequence.",
        "- Linear trend: the estimated change in high-side win rate as the utility gap gets larger.",
        "- Percentage points per utility unit: a slope of 0.05 means a five percentage-point higher win rate for each one-unit increase in `delta_u`.",
        "- Marginal model: a trend model with no extra grouping terms.",
        "- Domain-adjusted model: a trend model that allows religions, animals, countries, and political policies to have different baseline win rates.",
        "- Logistic sensitivity model: a second version of the trend test for a yes/no outcome; it is reported as an odds ratio.",
        "- Odds ratio: values above 1 mean higher odds that the high-side output wins as the utility gap increases; values below 1 mean lower odds.",
        "- Clustered confidence interval: the uncertainty interval allows repeated observations from the same actor/domain/utility-pair contrast to be correlated.",
        "- Binomial confidence interval: the uncertainty interval around a simple win fraction.",
        "",
        "Primary test: pair-level linear trend predicting high-side win from `delta_u`, excluding ties.",
        "",
        f"- Included judged pairs: {len(trials)} total; {n} non-tied; {ties} tied.",
        f"- Overall high-side win rate: {high}/{n} = {pct(rate)} (binomial 95% CI {pct(ci_lo)}-{pct(ci_hi)}).",
        (
            "- Domain-adjusted linear trend: "
            f"{100 * primary['coef']:.1f} percentage points per one utility unit "
            f"(95% CI {100 * primary['ci_lo']:.1f} to {100 * primary['ci_hi']:.1f} percentage points, "
            f"p={primary['p_value']:.4g})."
        ),
        (
            "- Domain-adjusted logistic sensitivity odds ratio: "
            f"{logistic['odds_ratio']:.3f} per one utility unit "
            f"(95% CI {logistic['or_ci_lo']:.3f}-{logistic['or_ci_hi']:.3f}, "
            f"p={logistic['p_value']:.4g})."
        ),
        "",
        "## Bin Summary",
        "",
        "| Bin | Mean delta_u | N | High win rate |",
        "|---:|---:|---:|---:|",
    ]
    for row in bins:
        lines.append(
            f"| {row['bin']} | {row['delta_u_mean']:.3f} | {row['n']} | {pct(row['high_win_rate'])} |"
        )
    lines.extend(["", "## Input Files", ""])
    for used_path in used_paths:
        lines.append(f"- `{used_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rounded_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({key: (round(value, 6) if isinstance(value, float) else value) for key, value in row.items()})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=None, help="Optional prebuilt trial-level CSV.")
    parser.add_argument("--actors", default="", help="Comma-separated actor ids to include.")
    parser.add_argument("--tasks", default="", help="Comma-separated task ids to include.")
    parser.add_argument("--domains", default="", help="Comma-separated utility domains to include.")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    actors = parse_csv_arg(args.actors)
    tasks = parse_csv_arg(args.tasks)
    domains = parse_csv_arg(args.domains)
    unknown_actors = sorted(actors - set(ACTORS)) if actors else []
    if unknown_actors:
        raise ValueError(f"unknown actor ids: {', '.join(unknown_actors)}")

    if args.input_csv:
        rows = read_trials_csv(args.input_csv, actors, tasks, domains)
        used_paths = [args.input_csv]
    else:
        rows, used_paths = build_trials_from_snapshots(actors, tasks, domains)

    paths = output_paths(actors=actors, tasks=tasks, domains=domains)
    subset_label = filter_label(actors=actors, tasks=tasks, domains=domains)
    trials = trials_frame(rows)
    df = non_tied_frame(trials)
    if df.empty:
        raise ValueError("No non-tied high-low trials found.")

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_csv_rows(paths["trials"], rounded_rows(rows))
    bins = write_bins(df, paths["bins"], args.bins)
    regressions = write_regressions(df, paths["regression"])
    write_figure(df, bins, paths["pdf"], paths["png"], subset_label)
    write_coefficient_figure(regressions, paths["coef_pdf"], paths["coef_png"])
    write_summary(paths["summary"], trials, df, bins, regressions, used_paths, subset_label)

    high = int(df["high_win"].sum())
    n = int(len(df))
    rate, lo, hi = wilson(high, n)
    primary = next(row for row in regressions if row["model"].startswith("fixed_effects:"))
    logistic = next(row for row in regressions if row["model"].startswith("fixed_effects logit:"))
    print(f"comparison: {COMPARISON}")
    print(f"input files: {len(used_paths)}")
    print(f"non-tied pairs: {n}; ties: {int((trials['winner'] == 'tie').sum())}")
    print(f"overall high win rate: {high}/{n} = {rate:.4f} [{lo:.4f}, {hi:.4f}]")
    print(
        "domain-adjusted linear slope per utility unit: "
        f"{100 * primary['coef']:.1f} percentage points "
        f"(95% CI {100 * primary['ci_lo']:.1f} to {100 * primary['ci_hi']:.1f}, p={primary['p_value']:.4g})"
    )
    print(
        "domain-adjusted logistic sensitivity odds ratio per utility unit: "
        f"{logistic['odds_ratio']:.3f} [{logistic['or_ci_lo']:.3f}, "
        f"{logistic['or_ci_hi']:.3f}], p={logistic['p_value']:.4g}"
    )
    for key in ["trials", "bins", "regression", "summary", "pdf", "png", "coef_pdf", "coef_png"]:
        print(f"wrote {paths[key]}")


if __name__ == "__main__":
    main()
