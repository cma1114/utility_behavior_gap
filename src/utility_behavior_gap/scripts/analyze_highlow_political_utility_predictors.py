#!/usr/bin/env python3
"""Model political high-low outcomes from utility endpoints and gaps.

This deliberately avoids the unidentified specification
``high_utility + low_utility + delta_u`` because ``delta_u = high_utility -
low_utility``. Instead it fits two interpretable specifications:

1. ``gap_only``: outcome ~ utility gap
2. ``endpoint``: outcome ~ high-side utility + low-side utility

Cross-model aggregate models standardize predictors within actor and summarize
actor-level coefficients with t intervals across actors. Per-model models use
raw utilities, since the scale is meaningful within a fitted actor utility
model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from utility_behavior_gap.paths import ROOT

ANALYSIS = ROOT / "outputs" / "analysis"
INPUT = ANALYSIS / "highlow_political_pairs_with_judged_outcomes.csv"

PAIR_DATA = ANALYSIS / "highlow_political_utility_predictor_pair_data.csv"
TASK_DATA = ANALYSIS / "highlow_political_utility_predictor_task_data.csv"
ACTOR_COEFFICIENTS = ANALYSIS / "highlow_political_utility_predictor_actor_coefficients.csv"
AGGREGATE_RESULTS = ANALYSIS / "highlow_political_utility_predictor_cross_model_results.csv"
PER_MODEL_RESULTS = ANALYSIS / "highlow_political_utility_predictor_per_model_results.csv"
SUMMARY = ANALYSIS / "highlow_political_utility_predictor_summary.md"

TASK_COLUMNS = {
    "essay": "essay_outcome",
    "translation": "translation_outcome",
    "incident_postmortem": "incident_postmortem_outcome",
    "grant_proposal_abstract": "grant_proposal_abstract_outcome",
}


@dataclass(frozen=True)
class Spec:
    name: str
    raw_terms: tuple[str, ...]
    z_terms: tuple[str, ...]
    description: str


SPECS = (
    Spec(
        name="gap_only",
        raw_terms=("delta_u",),
        z_terms=("delta_u_actor_z",),
        description="utility gap only",
    ),
    Spec(
        name="endpoint",
        raw_terms=("high_utility", "low_utility"),
        z_terms=("high_utility_actor_z", "low_utility_actor_z"),
        description="high-side and low-side utilities, no gap term",
    ),
)


def signed_outcome(value: str) -> int:
    if value == "high":
        return 1
    if value == "low":
        return -1
    if value == "tie":
        return 0
    raise ValueError(f"unexpected task outcome: {value!r}")


def actor_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    out: list[float] = []
    for _, row in df.iterrows():
        sub = df.loc[df["actor"].eq(row["actor"]), col]
        sd = sub.std(ddof=1)
        out.append(float((row[col] - sub.mean()) / sd) if sd else 0.0)
    return pd.Series(out, index=df.index)


def load_pair_data() -> pd.DataFrame:
    pairs = pd.read_csv(INPUT)
    for col in ["high_utility", "low_utility", "delta_u"]:
        pairs[col] = pd.to_numeric(pairs[col])
        pairs[f"{col}_actor_z"] = actor_zscore(pairs, col)

    for task, col in TASK_COLUMNS.items():
        pairs[f"{task}_signed_score"] = pairs[col].map(signed_outcome)

    signed_cols = [f"{task}_signed_score" for task in TASK_COLUMNS]
    pairs["net_high_wins_across_tasks_recomputed"] = pairs[signed_cols].sum(axis=1)
    pairs["net_high_win_score_recomputed"] = pairs[signed_cols].mean(axis=1)
    return pairs


def make_task_data(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    utility_cols = [
        "high_utility",
        "low_utility",
        "delta_u",
        "high_utility_actor_z",
        "low_utility_actor_z",
        "delta_u_actor_z",
        "high_description",
        "low_description",
    ]
    for _, row in pairs.iterrows():
        base = {"actor": row["actor"], **{col: row[col] for col in utility_cols}}
        for task, col in TASK_COLUMNS.items():
            outcome = row[col]
            rows.append({
                **base,
                "task": task,
                "task_outcome": outcome,
                "signed_score": signed_outcome(outcome),
            })
    return pd.DataFrame(rows)


def formula(outcome: str, terms: tuple[str, ...]) -> str:
    return f"{outcome} ~ {' + '.join(terms)}"


def fit_ols(data: pd.DataFrame, outcome: str, terms: tuple[str, ...], *, cov_type: str = "HC3"):
    return smf.ols(formula(outcome, terms), data=data).fit(cov_type=cov_type)


def actor_level_coefficients(pairs: pd.DataFrame, task_data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    datasets: list[tuple[str, str, str, pd.DataFrame, str]] = [
        (
            "pair_net_score",
            "all_tasks",
            "net_high_win_score_recomputed",
            pairs,
            "mean signed task score across four tasks",
        )
    ]
    for task, sub in task_data.groupby("task", sort=True):
        datasets.append(("task_signed_score", task, "signed_score", sub, "high=+1 tie=0 low=-1"))

    for family, task, outcome, data, outcome_description in datasets:
        for actor, actor_data in data.groupby("actor", sort=True):
            for spec in SPECS:
                fit = fit_ols(actor_data, outcome, spec.z_terms, cov_type="HC3")
                for term in spec.z_terms:
                    rows.append({
                        "family": family,
                        "task": task,
                        "actor": actor,
                        "spec": spec.name,
                        "term": term,
                        "coefficient": float(fit.params[term]),
                        "n": int(fit.nobs),
                        "r_squared": float(fit.rsquared),
                        "outcome": outcome,
                        "outcome_description": outcome_description,
                        "predictor_scale": "actor-standardized",
                        "formula": formula(outcome, spec.z_terms),
                    })
    return pd.DataFrame(rows)


def aggregate_coefficients(actor_coefs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, sub in actor_coefs.groupby(["family", "task", "spec", "term"], sort=True):
        family, task, spec, term = key
        vals = sub["coefficient"].to_numpy(dtype=float)
        n = len(vals)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1))
        se = sd / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, df=n - 1)
        tstat = mean / se if se else np.nan
        p = 2 * stats.t.sf(abs(tstat), df=n - 1) if se else np.nan
        rows.append({
            "family": family,
            "task": task,
            "spec": spec,
            "term": term,
            "mean_coefficient": mean,
            "ci_lo": mean - tcrit * se,
            "ci_hi": mean + tcrit * se,
            "t_stat": tstat,
            "p_value": p,
            "n_actors": n,
            "significant_p_lt_0_05": bool(p < 0.05) if not np.isnan(p) else False,
            "coefficient_scale": "signed-score units per 1 actor-SD predictor increase",
            "inference": "one coefficient per actor; mean tested against zero with t(df=6)",
        })
    return pd.DataFrame(rows)


def per_model_results(pairs: pd.DataFrame, task_data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    datasets: list[tuple[str, str, str, pd.DataFrame, str]] = [
        (
            "pair_net_score",
            "all_tasks",
            "net_high_win_score_recomputed",
            pairs,
            "mean signed task score across four tasks",
        )
    ]
    for task, sub in task_data.groupby("task", sort=True):
        datasets.append(("task_signed_score", task, "signed_score", sub, "high=+1 tie=0 low=-1"))

    for family, task, outcome, data, outcome_description in datasets:
        for actor, actor_data in data.groupby("actor", sort=True):
            for spec in SPECS:
                fit = fit_ols(actor_data, outcome, spec.raw_terms, cov_type="HC3")
                conf = fit.conf_int()
                for term in spec.raw_terms:
                    rows.append({
                        "family": family,
                        "task": task,
                        "actor": actor,
                        "spec": spec.name,
                        "term": term,
                        "coefficient": float(fit.params[term]),
                        "ci_lo": float(conf.loc[term, 0]),
                        "ci_hi": float(conf.loc[term, 1]),
                        "p_value": float(fit.pvalues[term]),
                        "significant_p_lt_0_05": bool(fit.pvalues[term] < 0.05),
                        "n": int(fit.nobs),
                        "r_squared": float(fit.rsquared),
                        "outcome": outcome,
                        "outcome_description": outcome_description,
                        "predictor_scale": "raw within-actor utility units",
                        "se_type": "HC3 robust",
                        "formula": formula(outcome, spec.raw_terms),
                    })
    return pd.DataFrame(rows)


def coefficient_table(df: pd.DataFrame, *, max_rows: int = 40) -> list[str]:
    if df.empty:
        return ["No terms reached p < 0.05."]
    lines = [
        "| family | task | actor | spec | term | estimate | 95% CI | p |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    show = df.head(max_rows)
    for _, row in show.iterrows():
        actor = row.get("actor", "aggregate")
        lines.append(
            f"| {row['family']} | {row['task']} | {actor} | {row['spec']} | {row['term']} | "
            f"{row['coefficient'] if 'coefficient' in row else row['mean_coefficient']:.3f} | "
            f"{row['ci_lo']:.3f}-{row['ci_hi']:.3f} | {row['p_value']:.4f} |"
        )
    if len(df) > max_rows:
        lines.append(f"| ... | ... | ... | ... | ... | {len(df) - max_rows} more rows in CSV | ... | ... |")
    return lines


def write_summary(
    pairs: pd.DataFrame,
    task_data: pd.DataFrame,
    aggregate: pd.DataFrame,
    per_model: pd.DataFrame,
) -> None:
    aggregate_sig = aggregate[aggregate["significant_p_lt_0_05"]].copy()
    per_model_sig = per_model[per_model["significant_p_lt_0_05"]].copy()
    aggregate_for_table = aggregate_sig.rename(columns={"mean_coefficient": "coefficient"})

    lines = [
        "# Political High-Low Utility Predictor Models",
        "",
        "This analysis uses only the political-domain high-low pairs.",
        "",
        "## Model Design",
        "",
        "- The all-three predictor model is not fit because `delta_u = high_utility - low_utility`.",
        "- Cross-model aggregate models standardize utilities within actor, fit one regression per actor, and test the mean coefficient across actors.",
        "- Per-model models use raw within-actor utilities and HC3 robust standard errors.",
        "- Outcomes are signed: high win = +1, tie = 0, low win = -1.",
        "",
        "## Outcome Counts",
        "",
        f"- Political pairs: {len(pairs)}",
        f"- Pair-level mean net score: {pairs['net_high_win_score_recomputed'].mean():.3f}",
        "",
        "| task | high | low | tie |",
        "|---|---:|---:|---:|",
    ]
    for task, sub in task_data.groupby("task", sort=True):
        lines.append(
            f"| {task} | {(sub['signed_score'] == 1).sum()} | "
            f"{(sub['signed_score'] == -1).sum()} | {(sub['signed_score'] == 0).sum()} |"
        )

    lines += [
        "",
        "## Significant Cross-Model Aggregate Terms",
        "",
        *coefficient_table(aggregate_for_table.sort_values(["family", "task", "spec", "term"])),
        "",
        "## Significant Per-Model Terms",
        "",
        *coefficient_table(per_model_sig.sort_values(["actor", "family", "task", "spec", "term"])),
        "",
        "## Output Files",
        "",
        f"- `{AGGREGATE_RESULTS.relative_to(ROOT)}`",
        f"- `{PER_MODEL_RESULTS.relative_to(ROOT)}`",
        f"- `{ACTOR_COEFFICIENTS.relative_to(ROOT)}`",
        f"- `{PAIR_DATA.relative_to(ROOT)}`",
        f"- `{TASK_DATA.relative_to(ROOT)}`",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    pairs = load_pair_data()
    task_data = make_task_data(pairs)
    actor_coefs = actor_level_coefficients(pairs, task_data)
    aggregate = aggregate_coefficients(actor_coefs)
    per_model = per_model_results(pairs, task_data)

    pairs.to_csv(PAIR_DATA, index=False)
    task_data.to_csv(TASK_DATA, index=False)
    actor_coefs.to_csv(ACTOR_COEFFICIENTS, index=False)
    aggregate.to_csv(AGGREGATE_RESULTS, index=False)
    per_model.to_csv(PER_MODEL_RESULTS, index=False)
    write_summary(pairs, task_data, aggregate, per_model)

    print(f"aggregate: {AGGREGATE_RESULTS}")
    print(f"per_model: {PER_MODEL_RESULTS}")
    print(f"actor_coefficients: {ACTOR_COEFFICIENTS}")
    print(f"summary: {SUMMARY}")


if __name__ == "__main__":
    main()
