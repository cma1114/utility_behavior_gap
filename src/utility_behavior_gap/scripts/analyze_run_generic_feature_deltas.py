#!/usr/bin/env python3
"""Standard generic-feature deltas for one two-arm generation run.

This is a local analysis over stored run artifacts. It computes the standard
paper-facing generic feature set from ``analysis_specs/feature_definitions.yaml``
for a named left condition minus a named right condition.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

from utility_behavior_gap.feature_specs import generic_feature_info, standard_generic_feature_ids
from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition
from utility_behavior_gap.paths import ANALYSIS, ANALYSIS_SPECS
from utility_behavior_gap.scripts.analyze_final_text_features import (
    load_textstat_module,
    load_vader_lexicon,
    load_zipf_frequency,
    scalar_features,
)


DEFAULT_RUN_DIR = Path(
    "/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/"
    "essay__user_prompt_role__gpt-5.4-mini-or__2026-06-20_16-58-28Z__hash-958a1d578531"
)
FEATURES = standard_generic_feature_ids()
FEATURE_INFO = generic_feature_info()


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    return "\n".join(lines)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def fmt_ci(lo: float, hi: float, digits: int = 3) -> str:
    return f"[{fmt(lo, digits)}, {fmt(hi, digits)}]"


def bonferroni_exact_ci(
    wins: int,
    total: int,
    *,
    family_size: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    tail_alpha = alpha / (2 * family_size)
    lo = 0.0 if wins == 0 else float(beta.ppf(tail_alpha, wins, total - wins + 1))
    hi = 1.0 if wins == total else float(beta.ppf(1.0 - tail_alpha, wins + 1, total - wins))
    return lo, hi


def output_is_valid(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    if row.get("success") is False:
        return False
    finish = str(row.get("finish_reason") or "")
    if finish and finish != "stop":
        return False
    return bool(str(row.get("output_text") or "").strip())


def generations_by_pair_condition(generations: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for generation in generations:
        pair_uid = str(generation.get("pair_uid") or "")
        condition = str(generation.get("condition") or "")
        if pair_uid and condition:
            out[pair_uid][condition] = generation
    return out


def generation_by_output_id(generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("output_id") or ""): row for row in generations if row.get("output_id")}


def valid_votes_for_pair(
    *,
    pair_uid: str,
    votes: list[dict[str, Any]],
    output_a: dict[str, Any] | None,
    output_b: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if output_a is None or output_b is None:
        return []
    expected = (output_text_fingerprint(output_a), output_text_fingerprint(output_b))
    valid: list[dict[str, Any]] = []
    for vote in votes:
        if vote.get("success") is False:
            continue
        if str(vote.get("pair_uid") or "") != pair_uid:
            continue
        vote_hashes = (vote.get("source_output_a_hash"), vote.get("source_output_b_hash"))
        if vote_hashes != expected:
            continue
        valid.append(vote)
    return valid


def judge_verdicts(votes: list[dict[str, Any]]) -> list[str]:
    by_judge: dict[str, list[str]] = defaultdict(list)
    for vote in votes:
        by_judge[str(vote.get("judge_model") or "")].append(str(vote.get("winner_condition") or ""))
    return [derive_judge_verdict(values) for _, values in sorted(by_judge.items())]


def panel_winner(job: dict[str, Any], votes: list[dict[str, Any]]) -> str:
    return derive_panel_winner_condition(job, judge_verdicts(votes))


def add_quantitative_detail(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["left_numeric_count_for_composite"] = out["left_numbers"] + out["left_percentages"]
    out["right_numeric_count_for_composite"] = out["right_numbers"] + out["right_percentages"]
    out["left_quantitative_detail"] = np.nan
    out["right_quantitative_detail"] = np.nan
    for task, idx in out.groupby("task", dropna=False).groups.items():
        left = out.loc[idx, "left_numeric_count_for_composite"].to_numpy(dtype=float)
        right = out.loc[idx, "right_numeric_count_for_composite"].to_numpy(dtype=float)
        pooled = np.concatenate([left[np.isfinite(left)], right[np.isfinite(right)]])
        if len(pooled) == 0:
            continue
        mean = float(np.mean(pooled))
        sd = float(np.std(pooled, ddof=0))
        if not math.isfinite(sd) or sd <= 0:
            out.loc[idx, "left_quantitative_detail"] = 0.0
            out.loc[idx, "right_quantitative_detail"] = 0.0
        else:
            out.loc[idx, "left_quantitative_detail"] = (
                out.loc[idx, "left_numeric_count_for_composite"] - mean
            ) / sd
            out.loc[idx, "right_quantitative_detail"] = (
                out.loc[idx, "right_numeric_count_for_composite"] - mean
            ) / sd
    out["delta_quantitative_detail"] = out["left_quantitative_detail"] - out["right_quantitative_detail"]
    return out


def bootstrap_ci(values: np.ndarray, *, iterations: int, seed: int) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return math.nan, math.nan
    if len(clean) == 1 or iterations <= 0:
        point = float(np.mean(clean))
        return point, point
    rng = np.random.default_rng(seed)
    estimates = rng.choice(clean, size=(iterations, len(clean)), replace=True).mean(axis=1)
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def panel_correlation(delta: pd.Series, score: pd.Series) -> tuple[float, int]:
    work = pd.DataFrame(
        {
            "delta": pd.to_numeric(delta, errors="coerce"),
            "score": pd.to_numeric(score, errors="coerce"),
        }
    ).dropna()
    if len(work) < 3 or work["delta"].nunique() < 2 or work["score"].nunique() < 2:
        return math.nan, int(len(work))
    return float(work["delta"].corr(work["score"])), int(len(work))


def summarize_features(pairs: pd.DataFrame, *, iterations: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(FEATURES):
        delta_col = f"delta_{feature}"
        left_col = f"left_{feature}"
        right_col = f"right_{feature}"
        deltas = pd.to_numeric(pairs[delta_col], errors="coerce").to_numpy(dtype=float)
        valid = deltas[np.isfinite(deltas)]
        if len(valid) == 0:
            continue
        ci_lo, ci_hi = bootstrap_ci(valid, iterations=iterations, seed=seed + index)
        panel_r, panel_n = panel_correlation(pairs[delta_col], pairs["left_panel_score"])
        sd = float(np.std(valid, ddof=1)) if len(valid) > 1 else math.nan
        info = FEATURE_INFO[feature]
        rows.append(
            {
                "feature": feature,
                "label": info.get("label", feature),
                "definition": info.get("definition", ""),
                "n_pairs": int(len(valid)),
                "left_mean": float(pd.to_numeric(pairs[left_col], errors="coerce").mean()),
                "right_mean": float(pd.to_numeric(pairs[right_col], errors="coerce").mean()),
                "mean_delta_left_minus_right": float(np.mean(valid)),
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "ci_excludes_zero": bool(
                    math.isfinite(ci_lo)
                    and math.isfinite(ci_hi)
                    and ((ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0))
                ),
                "standardized_delta_pair_sd": float(np.mean(valid) / sd) if sd and sd > 0 else math.nan,
                "pct_pairs_left_greater": float(np.mean(valid > 0)),
                "pct_pairs_right_greater": float(np.mean(valid < 0)),
                "pct_pairs_equal": float(np.mean(valid == 0)),
                "panel_preference_r": panel_r,
                "panel_preference_n": panel_n,
            }
        )
    return pd.DataFrame(rows)


def compact_feature_table(summary: pd.DataFrame, *, left_label: str, right_label: str) -> pd.DataFrame:
    out = summary.copy()
    out["abs_std_delta"] = out["standardized_delta_pair_sd"].abs()
    out = out.sort_values("abs_std_delta", ascending=False)
    return pd.DataFrame(
        {
            "Feature": out["label"],
            "Delta": out["mean_delta_left_minus_right"].map(lambda value: fmt(value, 3)),
            "95% CI": [fmt_ci(lo, hi, 3) for lo, hi in zip(out["ci_low"], out["ci_high"])],
            "Std delta": out["standardized_delta_pair_sd"].map(lambda value: fmt(value, 3)),
            "Panel preference (r)": [
                "" if not math.isfinite(float(r)) else f"{float(r):+.2f} (n={int(n)})"
                for r, n in zip(out["panel_preference_r"], out["panel_preference_n"])
            ],
            f"% {left_label} higher": out["pct_pairs_left_greater"].map(lambda value: f"{100 * value:.1f}%"),
            f"% {right_label} higher": out["pct_pairs_right_greater"].map(lambda value: f"{100 * value:.1f}%"),
            "Definition": out["definition"],
        }
    )


def load_pairs(
    *,
    run_dir: Path,
    left_condition: str,
    right_condition: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    jobs = read_jsonl(run_dir / "generation_jobs.jsonl")
    generations = read_jsonl(run_dir / "generations.jsonl")
    votes = read_jsonl(run_dir / "judge_votes.jsonl")
    by_output_id = generation_by_output_id(generations)
    by_pair_condition = generations_by_pair_condition(generations)
    votes_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote in votes:
        votes_by_pair[str(vote.get("pair_uid") or "")].append(vote)

    textstat_module = load_textstat_module()
    positive_words, negative_words = load_vader_lexicon()
    zipf_frequency_fn, wordfreq_status = load_zipf_frequency()

    rows: list[dict[str, Any]] = []
    invalid_pairs = 0
    stale_or_failed_vote_rows = 0
    for job in jobs:
        pair_uid = str(job.get("pair_uid") or "")
        left = by_pair_condition.get(pair_uid, {}).get(left_condition)
        right = by_pair_condition.get(pair_uid, {}).get(right_condition)
        if not output_is_valid(left) or not output_is_valid(right):
            invalid_pairs += 1
            continue

        source_a = by_output_id.get(f"{pair_uid}::a")
        source_b = by_output_id.get(f"{pair_uid}::b")
        valid_votes = valid_votes_for_pair(
            pair_uid=pair_uid,
            votes=votes_by_pair.get(pair_uid, []),
            output_a=source_a,
            output_b=source_b,
        )
        stale_or_failed_vote_rows += len(votes_by_pair.get(pair_uid, [])) - len(valid_votes)
        panel = panel_winner(job, valid_votes)
        if panel == left_condition:
            panel_score: float | None = 1.0
        elif panel == right_condition:
            panel_score = -1.0
        elif panel == "tie":
            panel_score = 0.0
        else:
            panel_score = math.nan

        left_text = str(left.get("output_text") or "")
        right_text = str(right.get("output_text") or "")
        left_features = scalar_features(
            left_text,
            textstat_module=textstat_module,
            positive_words=positive_words,
            negative_words=negative_words,
            zipf_frequency_fn=zipf_frequency_fn,
        )
        right_features = scalar_features(
            right_text,
            textstat_module=textstat_module,
            positive_words=positive_words,
            negative_words=negative_words,
            zipf_frequency_fn=zipf_frequency_fn,
        )
        row: dict[str, Any] = {
            "run_id": run_dir.name,
            "pair_uid": pair_uid,
            "actor": job.get("actor", ""),
            "actor_label": job.get("actor_label", ""),
            "task": job.get("task", ""),
            "task_label": job.get("task_label", ""),
            "item_label": job.get("item_label", ""),
            "left_condition": left_condition,
            "right_condition": right_condition,
            "left_output_id": left.get("output_id", ""),
            "right_output_id": right.get("output_id", ""),
            "panel_winner_condition": panel,
            "left_panel_score": panel_score,
            "n_vote_rows_valid_current_outputs": len(valid_votes),
            "left_finish_reason": left.get("finish_reason", ""),
            "right_finish_reason": right.get("finish_reason", ""),
        }
        for feature, value in left_features.items():
            row[f"left_{feature}"] = value
        for feature, value in right_features.items():
            row[f"right_{feature}"] = value
        for feature in set(left_features) & set(right_features):
            row[f"delta_{feature}"] = left_features[feature] - right_features[feature]
        rows.append(row)

    pairs = add_quantitative_detail(pd.DataFrame(rows))
    report = {
        "jobs": len(jobs),
        "valid_pairs": len(pairs),
        "invalid_or_missing_output_pairs": invalid_pairs,
        "vote_rows": len(votes),
        "stale_or_failed_vote_rows_ignored": stale_or_failed_vote_rows,
        "textstat_available": textstat_module is not None,
        "vader_available": bool(positive_words or negative_words),
        "wordfreq_status": wordfreq_status,
    }
    return pairs, report


def outcome_summary(
    pairs: pd.DataFrame,
    *,
    left_condition: str,
    right_condition: str,
    family_size: int,
) -> dict[str, Any]:
    wins = int(pairs["panel_winner_condition"].eq(left_condition).sum())
    losses = int(pairs["panel_winner_condition"].eq(right_condition).sum())
    ties = int(pairs["panel_winner_condition"].eq("tie").sum())
    unresolved = int((~pairs["panel_winner_condition"].isin([left_condition, right_condition, "tie"])).sum())
    non_ties = wins + losses
    rate = wins / non_ties if non_ties else math.nan
    ci_lo, ci_hi = bonferroni_exact_ci(wins, non_ties, family_size=family_size)
    p_value = binomtest(wins, non_ties, 0.5, alternative="two-sided").pvalue if non_ties else math.nan
    return {
        "pairs": int(len(pairs)),
        "left_wins": wins,
        "right_wins": losses,
        "ties": ties,
        "unresolved": unresolved,
        "non_tied_pairs": non_ties,
        "left_win_rate_excluding_ties": rate,
        "familywise_ci_lo": ci_lo,
        "familywise_ci_hi": ci_hi,
        "familywise_ci_positive": bool(math.isfinite(ci_lo) and ci_lo > 0.5),
        "p_two_sided_exact": p_value,
        "family_size": family_size,
        "familywise_ci_method": f"Bonferroni exact binomial 95% CI across {family_size} plotted cells",
    }


def write_summary(
    *,
    path: Path,
    run_dir: Path,
    left_label: str,
    right_label: str,
    left_condition: str,
    right_condition: str,
    pairs_path: Path,
    feature_path: Path,
    compact: pd.DataFrame,
    outcome: dict[str, Any],
    report: dict[str, Any],
    bootstrap_iterations: int,
) -> None:
    outcome_table = pd.DataFrame(
        [
            {
                "pairs": outcome["pairs"],
                f"{left_label} wins": outcome["left_wins"],
                f"{right_label} wins": outcome["right_wins"],
                "ties": outcome["ties"],
                "non-tied": outcome["non_tied_pairs"],
                f"{left_label} win rate excl. ties": fmt(outcome["left_win_rate_excluding_ties"], 3),
                "FWER 95% CI": fmt_ci(outcome["familywise_ci_lo"], outcome["familywise_ci_hi"], 3),
                "adj-CI > 0.5": str(outcome["familywise_ci_positive"]),
            }
        ]
    )
    lines = [
        f"# Standard Generic Text Features: {left_label} vs {right_label}",
        "",
        f"Run dir: `{run_dir}`",
        "",
        f"Comparison: `{left_condition} - {right_condition}`.",
        "",
        "## Standard Outcome Check",
        "",
        "This is the paper-figure style outcome for this model cell: tie-excluded left-side win rate, with the CI adjusted over the plotted family.",
        "",
        markdown_table(outcome_table),
        "",
        f"- exact two-sided binomial p-value before multiplicity correction: `{fmt(outcome['p_two_sided_exact'], 4)}`",
        f"- CI method: `{outcome['familywise_ci_method']}`",
        "",
        "## Generic Feature Deltas",
        "",
        "Deltas are left minus right. CIs are paired bootstraps over matched pairs. These are text-feature CIs, not win-rate CIs.",
        "",
        markdown_table(compact),
        "",
        "## Inputs And Filters",
        "",
        f"- editable feature definitions: `{ANALYSIS_SPECS / 'feature_definitions.yaml'}`",
        f"- standard generic features: `{', '.join(FEATURES)}`",
        f"- bootstrap iterations: `{bootstrap_iterations}`",
        f"- run-artifact report: `{report}`",
        "",
        "## Outputs",
        "",
        f"- pair-level feature deltas: `{pairs_path}`",
        f"- feature summary: `{feature_path}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--left-condition", default="user_strong")
    parser.add_argument("--right-condition", default="user_normal")
    parser.add_argument("--left-label", default="role strong")
    parser.add_argument("--right-label", default="role weak")
    parser.add_argument("--out-prefix")
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument(
        "--family-size",
        type=int,
        default=7,
        help="Number of plotted cells for Bonferroni-adjusted outcome CI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs, report = load_pairs(
        run_dir=args.run_dir,
        left_condition=args.left_condition,
        right_condition=args.right_condition,
    )
    if pairs.empty:
        raise ValueError("No valid matched pairs found.")
    features = summarize_features(pairs, iterations=args.bootstrap_iterations, seed=args.seed)
    compact = compact_feature_table(features, left_label=args.left_label, right_label=args.right_label)
    outcome = outcome_summary(
        pairs,
        left_condition=args.left_condition,
        right_condition=args.right_condition,
        family_size=args.family_size,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or f"generic_features__{args.left_condition}_vs_{args.right_condition}__{args.run_dir.name}"
    pairs_path = args.out_dir / f"{prefix}__pair_deltas.csv"
    feature_path = args.out_dir / f"{prefix}__summary.csv"
    summary_path = args.out_dir / f"{prefix}__summary.md"
    pairs.to_csv(pairs_path, index=False)
    features.to_csv(feature_path, index=False)
    write_summary(
        path=summary_path,
        run_dir=args.run_dir,
        left_label=args.left_label,
        right_label=args.right_label,
        left_condition=args.left_condition,
        right_condition=args.right_condition,
        pairs_path=pairs_path,
        feature_path=feature_path,
        compact=compact,
        outcome=outcome,
        report=report,
        bootstrap_iterations=args.bootstrap_iterations,
    )

    print(f"summary: {summary_path}")
    print(
        f"outcome: {outcome['left_wins']}/{outcome['left_wins'] + outcome['right_wins']} "
        f"{args.left_label} wins excluding ties, FWER CI "
        f"{fmt_ci(outcome['familywise_ci_lo'], outcome['familywise_ci_hi'], 3)}"
    )
    print(f"features: {feature_path}")
    print(f"pairs: {pairs_path}")


if __name__ == "__main__":
    main()
