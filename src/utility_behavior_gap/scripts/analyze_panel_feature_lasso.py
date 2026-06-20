#!/usr/bin/env python3
"""Elastic-net feature models for decisive panel preference.

The target is panel strength among decisive three-judge panels:

* LLL -> 0 high-side judge verdicts
* HLL -> 1 high-side judge verdict
* HHL -> 2 high-side judge verdicts
* HHH -> 3 high-side judge verdicts

Judge verdicts are first collapsed within judge across presentation orders
using the same `derive_judge_verdict` rule as the outcome analyses. Pairs with
judge ties or unresolved judge verdicts are audited but excluded from the
primary model.

The main models use text-feature deltas from `final_text_analysis_pair_deltas`.
Actor is deliberately not used as a control.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.judging import derive_judge_verdict
from utility_behavior_gap.paths import ANALYSIS


PAIR_DELTAS = ANALYSIS / "final_text_analysis_pair_deltas.csv"
FEATURE_DEFINITIONS = ANALYSIS / "final_text_analysis_feature_definitions.csv"

OUT_PREFIX = "panel_feature_lasso"
RANDOM_SEED = 20260614

MODEL_SPECS = [
    ("feature_only", []),
    ("task_adjusted", ["task"]),
    ("task_condition_adjusted", ["task", "contrast"]),
]

warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.linear_model\._logistic")


def log(message: str) -> None:
    print(message, flush=True)


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def text_feature_names() -> list[str]:
    defs = pd.read_csv(FEATURE_DEFINITIONS)
    return defs.loc[defs["type"].eq("feature"), "name"].tolist()


def load_run_cache(run_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for index, run_dir in enumerate(run_dirs, start=1):
        log(f"Loading run logs {index}/{len(run_dirs)}: {run_dir.name}")
        generations = read_jsonl_if_exists(run_dir / "generations.jsonl")
        votes = read_jsonl_if_exists(run_dir / "judge_votes.jsonl")
        gen_by_output_id = {str(row.get("output_id") or ""): row for row in generations}
        votes_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for vote in votes:
            votes_by_pair[str(vote.get("pair_uid") or "")].append(vote)
        cache[str(run_dir)] = {
            "generations": gen_by_output_id,
            "votes_by_pair": votes_by_pair,
            "n_generations": len(generations),
            "n_vote_rows": len(votes),
        }
    return cache


def valid_votes_for_pair(
    *,
    row: pd.Series,
    cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    run = cache[str(row["run_dir"])]
    high_gen = run["generations"].get(str(row["high_output_id"]))
    low_gen = run["generations"].get(str(row["low_output_id"]))
    if high_gen is None or low_gen is None:
        return []
    expected_hashes = (output_text_fingerprint(high_gen), output_text_fingerprint(low_gen))
    votes = []
    for vote in run["votes_by_pair"].get(str(row["pair_uid"]), []):
        if not vote.get("success"):
            continue
        vote_hashes = (vote.get("source_output_a_hash"), vote.get("source_output_b_hash"))
        if all(vote_hashes) and vote_hashes != expected_hashes:
            continue
        votes.append(vote)
    return votes


def judge_verdict_signature(row: pd.Series, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    votes = valid_votes_for_pair(row=row, cache=cache)
    by_judge: dict[str, list[str]] = defaultdict(list)
    for vote in votes:
        by_judge[str(vote.get("judge_model", ""))].append(str(vote.get("winner_condition", "")))

    high_raw = str(row["high_raw_condition"])
    low_raw = str(row["low_raw_condition"])
    judge_rows: list[tuple[str, str, str]] = []
    counts = {"H": 0, "L": 0, "T": 0, "U": 0, "O": 0}
    for judge_model, conditions in sorted(by_judge.items()):
        verdict = derive_judge_verdict(conditions)
        if verdict == high_raw:
            code = "H"
        elif verdict == low_raw:
            code = "L"
        elif verdict == "tie":
            code = "T"
        elif verdict == "unresolved" or not verdict:
            code = "U"
        else:
            code = "O"
        counts[code] += 1
        judge_rows.append((judge_model, verdict, code))

    judge_order_signature = "".join(code for _judge, _verdict, code in judge_rows)
    canonical_signature = "".join(code * counts[code] for code in ["H", "L", "T", "U", "O"])
    decisive = counts["H"] + counts["L"] == 3 and counts["T"] == 0 and counts["U"] == 0 and counts["O"] == 0
    return {
        "n_judges_with_votes": len(judge_rows),
        "n_high": counts["H"],
        "n_low": counts["L"],
        "n_tie_judges": counts["T"],
        "n_unresolved_judges": counts["U"],
        "n_other_judges": counts["O"],
        "panel_signature": canonical_signature or "none",
        "judge_order_signature": judge_order_signature or "none",
        "decisive_signature": decisive,
        "judge_verdicts_json": json.dumps(
            [
                {"judge_model": judge, "verdict_raw_condition": verdict, "verdict_code": code}
                for judge, verdict, code in judge_rows
            ],
            sort_keys=True,
        ),
    }


def add_panel_signatures(pair_df: pd.DataFrame) -> pd.DataFrame:
    run_dirs = sorted(Path(str(path)) for path in pair_df["run_dir"].dropna().unique())
    cache = load_run_cache(run_dirs)
    rows = []
    for idx, row in pair_df.iterrows():
        if idx and idx % 1000 == 0:
            log(f"Reconstructed panel signatures for {idx}/{len(pair_df)} pairs")
        rows.append(judge_verdict_signature(row, cache))
    sig_df = pd.DataFrame(rows)
    out = pd.concat([pair_df.reset_index(drop=True), sig_df], axis=1)
    return out


def finite_feature_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    feature_cols = [f"delta_{name}" for name in feature_names if f"delta_{name}" in df.columns]
    out = df.copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    before = len(out)
    out = out.dropna(subset=feature_cols)
    dropped = before - len(out)
    if dropped:
        log(f"Dropped {dropped} decisive rows with missing feature deltas.")
    return out


def task_standardize(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        standardized = pd.Series(index=out.index, dtype=float)
        for _task, sub in out.groupby("task", dropna=False):
            values = pd.to_numeric(sub[col], errors="coerce")
            mean = values.mean()
            sd = values.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:
                standardized.loc[sub.index] = 0.0
            else:
                standardized.loc[sub.index] = (values - mean) / sd
        out[col] = standardized
    return out


def expanded_binomial_rows(df: pd.DataFrame, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    y = []
    weights = []
    groups = []
    matrix = x.to_numpy(dtype=float)
    for i, (_idx, row) in enumerate(df.iterrows()):
        high = int(row["n_high"])
        low = int(row["n_low"])
        if high:
            rows.append(matrix[i])
            y.append(1)
            weights.append(high)
            groups.append(i)
        if low:
            rows.append(matrix[i])
            y.append(0)
            weights.append(low)
            groups.append(i)
    return np.vstack(rows), np.asarray(y), np.asarray(weights, dtype=float), np.asarray(groups)


def intercept_log_loss(y_train: np.ndarray, w_train: np.ndarray, y_test: np.ndarray, w_test: np.ndarray) -> float:
    p = float(np.average(y_train, weights=w_train))
    p = min(max(p, 1e-6), 1 - 1e-6)
    probs = np.column_stack([np.full_like(y_test, 1 - p, dtype=float), np.full_like(y_test, p, dtype=float)])
    return float(log_loss(y_test, probs, sample_weight=w_test, labels=[0, 1]))


def fit_unpenalized_logit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
) -> LogisticRegression | None:
    if x_train.shape[1] == 0:
        return None
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(x_train, y_train, sample_weight=w_train)
    return model


def cv_baseline_loss(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    losses: list[float] = []
    for train_idx, test_idx in splits:
        if x.shape[1] == 0:
            losses.append(intercept_log_loss(y[train_idx], weights[train_idx], y[test_idx], weights[test_idx]))
            continue
        model = fit_unpenalized_logit(x[train_idx], y[train_idx], weights[train_idx])
        if model is None:
            losses.append(intercept_log_loss(y[train_idx], weights[train_idx], y[test_idx], weights[test_idx]))
        else:
            probs = model.predict_proba(x[test_idx])
            losses.append(float(log_loss(y[test_idx], probs, sample_weight=weights[test_idx], labels=[0, 1])))
    return float(np.mean(losses))


def fit_elastic_net_cv(
    *,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    feature_count: int,
    l1_ratios: list[float],
    c_values: list[float],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[LogisticRegression, dict[str, Any], pd.DataFrame]:
    score_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for l1_ratio in l1_ratios:
        for c_value in c_values:
            losses: list[float] = []
            for train_idx, test_idx in splits:
                model = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=l1_ratio,
                    C=c_value,
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", FutureWarning)
                    model.fit(x[train_idx], y[train_idx], sample_weight=weights[train_idx])
                probs = model.predict_proba(x[test_idx])
                losses.append(float(log_loss(y[test_idx], probs, sample_weight=weights[test_idx], labels=[0, 1])))
            row = {
                "l1_ratio": l1_ratio,
                "C": c_value,
                "mean_cv_log_loss": float(np.mean(losses)),
                "sd_cv_log_loss": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
            }
            score_rows.append(row)
            if best is None or row["mean_cv_log_loss"] < best["mean_cv_log_loss"]:
                best = row
    assert best is not None
    final_model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=float(best["l1_ratio"]),
        C=float(best["C"]),
        max_iter=8000,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        final_model.fit(x, y, sample_weight=weights)
    return final_model, best, pd.DataFrame(score_rows)


def stability_selection(
    *,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    feature_names: list[str],
    feature_count: int,
    best_l1_ratio: float,
    best_c: float,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    selected_counts = Counter()
    for train_idx, _test_idx in splits:
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=best_l1_ratio,
            C=best_c,
            max_iter=8000,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", FutureWarning)
            model.fit(x[train_idx], y[train_idx], sample_weight=weights[train_idx])
        coefs = model.coef_[0][:feature_count]
        for name, coef in zip(feature_names[:feature_count], coefs):
            if abs(coef) > 1e-7:
                selected_counts[name] += 1
    folds = len(splits)
    return {name: selected_counts[name] / folds for name in feature_names[:feature_count]}


def design_matrix(df: pd.DataFrame, feature_cols: list[str], control_cols: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    x_features = df[feature_cols].copy()
    x_features.columns = [col.removeprefix("delta_") for col in feature_cols]
    feature_names = x_features.columns.tolist()
    if not control_cols:
        return x_features, feature_names, []
    controls = pd.get_dummies(df[control_cols].astype(str), columns=control_cols, drop_first=True, dtype=float)
    x = pd.concat([x_features.reset_index(drop=True), controls.reset_index(drop=True)], axis=1)
    return x, feature_names, controls.columns.tolist()


def control_matrix(df: pd.DataFrame, control_cols: list[str]) -> pd.DataFrame:
    if not control_cols:
        return pd.DataFrame(index=df.index)
    return pd.get_dummies(df[control_cols].astype(str), columns=control_cols, drop_first=True, dtype=float)


def run_model(
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    spec_name: str,
    control_cols: list[str],
    l1_ratios: list[float],
    c_values: list[float],
    n_splits: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    x_df, feature_names, control_names = design_matrix(df, feature_cols, control_cols)
    x, y, weights, groups = expanded_binomial_rows(df, x_df)
    control_df = control_matrix(df, control_cols)
    x_baseline, y_baseline, weights_baseline, groups_baseline = expanded_binomial_rows(df, control_df)
    assert np.array_equal(y, y_baseline)
    assert np.array_equal(weights, weights_baseline)
    assert np.array_equal(groups, groups_baseline)

    group_kfold = GroupKFold(n_splits=n_splits)
    splits = list(group_kfold.split(x, y, groups=groups))
    baseline_loss = cv_baseline_loss(x_baseline, y, weights, groups, splits)
    model, best, cv_grid = fit_elastic_net_cv(
        x=x,
        y=y,
        weights=weights,
        groups=groups,
        feature_count=len(feature_names),
        l1_ratios=l1_ratios,
        c_values=c_values,
        splits=splits,
    )
    stability = stability_selection(
        x=x,
        y=y,
        weights=weights,
        feature_names=feature_names,
        feature_count=len(feature_names),
        best_l1_ratio=float(best["l1_ratio"]),
        best_c=float(best["C"]),
        splits=splits,
    )
    coef_rows: list[dict[str, Any]] = []
    for name, coef in zip(feature_names + control_names, model.coef_[0]):
        coef_rows.append(
            {
                "model": spec_name,
                "term": name,
                "term_type": "feature" if name in feature_names else "control",
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
                "selected": bool(abs(coef) > 1e-7),
                "selection_frequency_across_cv_folds": stability.get(name, math.nan) if name in feature_names else math.nan,
                "best_l1_ratio": float(best["l1_ratio"]),
                "best_C": float(best["C"]),
            }
        )
    coef_df = pd.DataFrame(coef_rows)
    perf_row = {
        "model": spec_name,
        "n_pairs": int(len(df)),
        "n_expanded_rows": int(len(y)),
        "controls": ",".join(control_cols) if control_cols else "none",
        "features": len(feature_names),
        "best_l1_ratio": float(best["l1_ratio"]),
        "best_C": float(best["C"]),
        "baseline_cv_log_loss": baseline_loss,
        "elastic_net_cv_log_loss": float(best["mean_cv_log_loss"]),
        "log_loss_improvement": baseline_loss - float(best["mean_cv_log_loss"]),
        "pseudo_r2_vs_baseline": 1 - float(best["mean_cv_log_loss"]) / baseline_loss if baseline_loss else math.nan,
        "selected_feature_count": int(
            coef_df[(coef_df["term_type"].eq("feature")) & (coef_df["selected"])].shape[0]
        ),
    }
    cv_grid.insert(0, "model", spec_name)
    return [perf_row], coef_df, cv_grid


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    show = df.loc[:, columns].head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join("---" for _ in show.columns) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in show.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"| ... | {len(df) - max_rows} more rows omitted |" + " |" * (len(show.columns) - 2))
    return "\n".join(lines)


def write_summary(
    *,
    path: Path,
    signature_counts: pd.DataFrame,
    decisive_counts: pd.DataFrame,
    performance: pd.DataFrame,
    coefficients: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    selected = (
        coefficients[(coefficients["term_type"].eq("feature")) & (coefficients["selected"])]
        .sort_values(["model", "abs_coefficient"], ascending=[True, False])
        .copy()
    )
    lines = [
        "# Panel Feature Elastic-Net Analysis",
        "",
        "Primary outcome: decisive panel strength, encoded as high-side wins out of three judge verdicts.",
        "The decisive signatures are `LLL`, `HLL`, `HHL`, and `HHH`. Judge verdicts are first collapsed across presentation orders.",
        "",
        "Actor is not included as a control. Feature deltas are high side minus low side and are standardized within task before modeling.",
        "",
        "Models:",
        "",
        "- `feature_only`: standardized text-feature deltas only.",
        "- `task_adjusted`: standardized text-feature deltas plus task controls.",
        "- `task_condition_adjusted`: standardized text-feature deltas plus task and contrast controls.",
        "",
        "All models are weighted binomial logistic elastic-net models. A pair with `HHL` contributes two high-side successes and one low-side success; `LLL` contributes zero high-side successes and three low-side successes.",
        "",
        "## Panel Signature Counts",
        "",
        markdown_table(signature_counts, ["panel_signature", "pairs"]),
        "",
        "## Decisive Counts By Contrast",
        "",
        markdown_table(decisive_counts, ["contrast", "pairs", "HHH", "HHL", "HLL", "LLL"]),
        "",
        "## Cross-Validated Performance",
        "",
        markdown_table(
            performance,
            [
                "model",
                "n_pairs",
                "controls",
                "best_l1_ratio",
                "best_C",
                "baseline_cv_log_loss",
                "elastic_net_cv_log_loss",
                "log_loss_improvement",
                "pseudo_r2_vs_baseline",
                "selected_feature_count",
            ],
        ),
        "",
        "## Selected Features",
        "",
        markdown_table(
            selected,
            [
                "model",
                "term",
                "coefficient",
                "selection_frequency_across_cv_folds",
                "best_l1_ratio",
                "best_C",
            ],
            max_rows=80,
        ),
        "",
        "## Feature Set",
        "",
        "The candidate features were the raw final text-analysis feature deltas:",
        "",
    ]
    lines.extend(f"- `{col.removeprefix('delta_')}`" for col in feature_cols)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--l1-ratios", default="0.1,0.5,0.9,1.0")
    parser.add_argument("--c-values", default="0.01,0.03,0.1,0.3,1,3,10")
    args = parser.parse_args()

    l1_ratios = [float(value) for value in args.l1_ratios.split(",") if value.strip()]
    c_values = [float(value) for value in args.c_values.split(",") if value.strip()]

    log(f"Loading pair deltas: {PAIR_DELTAS}")
    pair_df = pd.read_csv(PAIR_DELTAS, low_memory=False)
    feature_names = text_feature_names()
    feature_cols = [f"delta_{name}" for name in feature_names if f"delta_{name}" in pair_df.columns]
    log(f"Candidate text features: {len(feature_cols)}")

    with_signatures = add_panel_signatures(pair_df)
    signature_path = ANALYSIS / f"{OUT_PREFIX}_panel_signatures.csv"
    with_signatures.to_csv(signature_path, index=False)
    log(f"Panel signatures: {signature_path}")

    signature_counts = (
        with_signatures.groupby("panel_signature", dropna=False)
        .size()
        .reset_index(name="pairs")
        .sort_values("pairs", ascending=False)
    )
    decisive = with_signatures[with_signatures["decisive_signature"].astype(bool)].copy()
    decisive = finite_feature_frame(decisive, feature_names)
    decisive = task_standardize(decisive, feature_cols)
    decisive_path = ANALYSIS / f"{OUT_PREFIX}_model_data.csv"
    decisive.to_csv(decisive_path, index=False)
    log(f"Decisive model data: {decisive_path} ({len(decisive)} pairs)")

    decisive_counts = (
        decisive.pivot_table(
            index="contrast",
            columns="panel_signature",
            values="pair_uid",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for sig in ["HHH", "HHL", "HLL", "LLL"]:
        if sig not in decisive_counts.columns:
            decisive_counts[sig] = 0
    decisive_counts["pairs"] = decisive_counts[["HHH", "HHL", "HLL", "LLL"]].sum(axis=1)
    decisive_counts = decisive_counts[["contrast", "pairs", "HHH", "HHL", "HLL", "LLL"]]

    performance_rows: list[dict[str, Any]] = []
    coefficient_frames: list[pd.DataFrame] = []
    cv_frames: list[pd.DataFrame] = []
    for spec_name, control_cols in MODEL_SPECS:
        log(f"Fitting {spec_name}")
        perf, coef_df, cv_grid = run_model(
            df=decisive,
            feature_cols=feature_cols,
            spec_name=spec_name,
            control_cols=control_cols,
            l1_ratios=l1_ratios,
            c_values=c_values,
            n_splits=args.n_splits,
        )
        performance_rows.extend(perf)
        coefficient_frames.append(coef_df)
        cv_frames.append(cv_grid)

    performance = pd.DataFrame(performance_rows)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    cv_grid = pd.concat(cv_frames, ignore_index=True)

    performance_path = ANALYSIS / f"{OUT_PREFIX}_cv_performance.csv"
    coefficient_path = ANALYSIS / f"{OUT_PREFIX}_coefficients.csv"
    selected_path = ANALYSIS / f"{OUT_PREFIX}_selected_features.csv"
    cv_grid_path = ANALYSIS / f"{OUT_PREFIX}_cv_grid.csv"
    decisive_counts_path = ANALYSIS / f"{OUT_PREFIX}_decisive_counts.csv"
    summary_path = ANALYSIS / f"{OUT_PREFIX}_summary.md"

    performance.to_csv(performance_path, index=False)
    coefficients.to_csv(coefficient_path, index=False)
    coefficients[(coefficients["term_type"].eq("feature")) & (coefficients["selected"])].to_csv(
        selected_path,
        index=False,
    )
    cv_grid.to_csv(cv_grid_path, index=False)
    decisive_counts.to_csv(decisive_counts_path, index=False)
    write_summary(
        path=summary_path,
        signature_counts=signature_counts,
        decisive_counts=decisive_counts,
        performance=performance,
        coefficients=coefficients,
        feature_cols=feature_cols,
    )

    print(f"summary: {summary_path}")
    print(f"performance: {performance_path}")
    print(f"coefficients: {coefficient_path}")
    print(f"selected features: {selected_path}")
    print(performance.to_string(index=False))


if __name__ == "__main__":
    main()
