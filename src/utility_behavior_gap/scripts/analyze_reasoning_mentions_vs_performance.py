#!/usr/bin/env python3
"""Relate incentive mentions in reasoning traces to judged high-low performance."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import binomtest, pearsonr, spearmanr

from utility_behavior_gap.paths import ROOT


ANALYSIS = ROOT / "outputs" / "analysis"
DEFAULT_PAIR_OUTCOMES = ANALYSIS / "highlow_reasoning_traces_medium__tasks-essay__7-actors_pair_outcomes.csv"
DEFAULT_MENTION_ROWS = ANALYSIS / "reasoning_incentive_mentions__success_only_per_output.csv"
OUT_PREFIX = "reasoning_mentions_vs_performance"

FEATURES = [
    "mentions_any_incentive",
    "mentions_scaffold",
    "mentions_target_terms",
    "mentions_exact_assigned_target",
    "mentions_meta_recognition",
]


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def mention_pattern(high: bool, low: bool) -> str:
    if high and low:
        return "both"
    if high:
        return "high_only"
    if low:
        return "low_only"
    return "neither"


def load_pair_rows(pair_path: Path, mention_path: Path, *, require_both_traces: bool) -> pd.DataFrame:
    pairs = pd.read_csv(pair_path)
    mentions = pd.read_csv(mention_path)
    for feature in FEATURES:
        mentions[feature] = bool_series(mentions[feature])
    mentions["has_trace"] = pd.to_numeric(mentions["trace_chars"], errors="coerce").fillna(0).gt(0)

    keep_cols = [
        "output_id",
        "has_trace",
        "trace_chars",
        "reasoning_tokens",
        *FEATURES,
        "scaffold_hits",
        "target_hits",
        "meta_hits",
        "exact_target_fields",
        "trace_snippet",
    ]
    high_mentions = mentions[keep_cols].add_prefix("high_")
    low_mentions = mentions[keep_cols].add_prefix("low_")
    out = pairs.merge(high_mentions, left_on="high_output_id", right_on="high_output_id", how="left")
    out = out.merge(low_mentions, left_on="low_output_id", right_on="low_output_id", how="left")
    for side in ["high", "low"]:
        out[f"{side}_has_trace"] = out[f"{side}_has_trace"].fillna(False).astype(bool)
        for feature in FEATURES:
            out[f"{side}_{feature}"] = out[f"{side}_{feature}"].fillna(False).astype(bool)

    if require_both_traces:
        out = out[out["high_has_trace"] & out["low_has_trace"]].copy()

    out["panel_score"] = out["high_win"].astype(int) - out["low_win"].astype(int)
    out["resolved"] = out["resolved_excluding_ties"].astype(int).eq(1)
    return out.reset_index(drop=True)


def summarize_pattern(df: pd.DataFrame, feature: str, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = df.copy()
    work["high_mention"] = work[f"high_{feature}"]
    work["low_mention"] = work[f"low_{feature}"]
    work["mention_pattern"] = [
        mention_pattern(high, low)
        for high, low in zip(work["high_mention"], work["low_mention"], strict=True)
    ]
    for key, sub in work.groupby([*group_cols, "mention_pattern"], dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row = {col: value for col, value in zip([*group_cols, "mention_pattern"], key_values, strict=True)}
        high_wins = int(sub["high_win"].sum())
        low_wins = int(sub["low_win"].sum())
        ties = int(sub["tie"].sum())
        resolved = high_wins + low_wins
        row.update(
            {
                "feature": feature,
                "pairs": int(len(sub)),
                "high_wins": high_wins,
                "low_wins": low_wins,
                "ties": ties,
                "n_excluding_ties": resolved,
                "high_win_rate_excluding_ties": high_wins / resolved if resolved else math.nan,
                "net_score_all_pairs": (high_wins - low_wins) / len(sub) if len(sub) else math.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def directional_tests(df: pd.DataFrame, feature: str, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = df.copy()
    high = work[f"high_{feature}"]
    low = work[f"low_{feature}"]
    work = work[high.ne(low)].copy()
    if work.empty:
        return pd.DataFrame()
    work["mention_side"] = ["high" if value else "low" for value in work[f"high_{feature}"]]
    work["mention_side_wins"] = (
        (work["mention_side"].eq("high") & work["high_win"].eq(1))
        | (work["mention_side"].eq("low") & work["low_win"].eq(1))
    )
    work["mention_side_loses"] = (
        (work["mention_side"].eq("high") & work["low_win"].eq(1))
        | (work["mention_side"].eq("low") & work["high_win"].eq(1))
    )
    work["mention_alignment_score"] = (
        work["mention_side_wins"].astype(int) - work["mention_side_loses"].astype(int)
    )
    grouped = [((), work)] if not group_cols else list(work.groupby(group_cols, dropna=False, sort=True))
    for key, sub in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        row = {col: value for col, value in zip(group_cols, key_values, strict=True)}
        wins = int(sub["mention_side_wins"].sum())
        losses = int(sub["mention_side_loses"].sum())
        ties = int(sub["tie"].sum())
        resolved = wins + losses
        p = float(binomtest(wins, resolved, 0.5, alternative="two-sided").pvalue) if resolved else math.nan
        row.update(
            {
                "feature": feature,
                "asymmetric_pairs": int(len(sub)),
                "mention_side_wins": wins,
                "mention_side_loses": losses,
                "ties": ties,
                "n_excluding_ties": resolved,
                "mention_side_win_rate_excluding_ties": wins / resolved if resolved else math.nan,
                "net_alignment_score_all_pairs": (wins - losses) / len(sub) if len(sub) else math.nan,
                "two_sided_binomial_p": p,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def correlations(df: pd.DataFrame, feature: str, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = df.copy()
    work["mention_diff"] = work[f"high_{feature}"].astype(int) - work[f"low_{feature}"].astype(int)
    grouped = [((), work)] if not group_cols else list(work.groupby(group_cols, dropna=False, sort=True))
    for key, sub in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        row = {col: value for col, value in zip(group_cols, key_values, strict=True)}
        row["feature"] = feature
        row["pairs"] = int(len(sub))
        row["n_asymmetric"] = int(sub["mention_diff"].ne(0).sum())
        if sub["mention_diff"].nunique(dropna=True) > 1 and sub["panel_score"].nunique(dropna=True) > 1:
            pearson = pearsonr(sub["mention_diff"], sub["panel_score"])
            spearman = spearmanr(sub["mention_diff"], sub["panel_score"])
            row["pearson_r_mention_diff_vs_panel_score"] = float(pearson.statistic)
            row["pearson_p"] = float(pearson.pvalue)
            row["spearman_r_mention_diff_vs_panel_score"] = float(spearman.statistic)
            row["spearman_p"] = float(spearman.pvalue)
        else:
            row["pearson_r_mention_diff_vs_panel_score"] = math.nan
            row["pearson_p"] = math.nan
            row["spearman_r_mention_diff_vs_panel_score"] = math.nan
            row["spearman_p"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, *, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else format(float(value), floatfmt))
        elif pd.api.types.is_bool_dtype(show[col]):
            show[col] = show[col].map(lambda value: "TRUE" if bool(value) else "FALSE")
        else:
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(col) for col in show.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in show.columns) + " |")
    return "\n".join(lines)


def write_examples(pair_df: pd.DataFrame, path: Path, *, feature: str) -> None:
    work = pair_df.copy()
    work["mention_pattern"] = [
        mention_pattern(high, low)
        for high, low in zip(work[f"high_{feature}"], work[f"low_{feature}"], strict=True)
    ]
    specs = [
        ("high_only_high_wins", "high_only", "hl_high"),
        ("high_only_low_wins", "high_only", "hl_low"),
        ("low_only_low_wins", "low_only", "hl_low"),
        ("low_only_high_wins", "low_only", "hl_high"),
    ]
    lines = [
        f"# Reasoning Mention vs Performance Examples: `{feature}`",
        "",
        "Snippets are from the lexical incentive-mention audit, not the final essays.",
        "",
    ]
    for label, pattern, winner in specs:
        sub = work[work["mention_pattern"].eq(pattern) & work["panel_winner_condition"].eq(winner)].copy()
        lines.extend([f"## {label}", ""])
        if sub.empty:
            lines.extend(["_No examples._", ""])
            continue
        for _, row in sub.head(8).iterrows():
            mention_side = "high" if pattern == "high_only" else "low"
            lines.extend(
                [
                    f"- `{row['actor']}` / `{row['domain']}` / winner `{row['panel_winner_condition']}` / `{row['pair_uid']}`",
                    f"  - high snippet: {str(row.get('high_trace_snippet', ''))[:360]}",
                    f"  - low snippet: {str(row.get('low_trace_snippet', ''))[:360]}",
                    f"  - mentioning side: `{mention_side}`",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    *,
    pair_df: pd.DataFrame,
    pattern_summary: pd.DataFrame,
    directional: pd.DataFrame,
    corr: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Reasoning Mentions vs Performance",
        "",
        "Pair-level analysis joining returned readable reasoning-trace incentive mentions to panel outcomes.",
        "",
        "Primary filter: both high and low outputs must have readable trace text.",
        "",
        f"- included pairs: {len(pair_df)}",
        f"- high wins: {int(pair_df['high_win'].sum())}",
        f"- low wins: {int(pair_df['low_win'].sum())}",
        f"- ties: {int(pair_df['tie'].sum())}",
        "",
        "## Directional Test",
        "",
        "Among pairs where exactly one side mentions the feature, this asks whether the mentioning side wins more often than the non-mentioning side, excluding ties.",
        "",
    ]
    cols = [
        "feature",
        "asymmetric_pairs",
        "mention_side_wins",
        "mention_side_loses",
        "ties",
        "n_excluding_ties",
        "mention_side_win_rate_excluding_ties",
        "net_alignment_score_all_pairs",
        "two_sided_binomial_p",
    ]
    lines.append(markdown_table(directional[cols], floatfmt=".3f"))
    lines.extend(["", "## Pattern Summary", ""])
    pcols = [
        "feature",
        "mention_pattern",
        "pairs",
        "high_wins",
        "low_wins",
        "ties",
        "n_excluding_ties",
        "high_win_rate_excluding_ties",
        "net_score_all_pairs",
    ]
    lines.append(markdown_table(pattern_summary[pcols], floatfmt=".3f"))
    lines.extend(["", "## Correlations", ""])
    ccols = [
        "feature",
        "pairs",
        "n_asymmetric",
        "pearson_r_mention_diff_vs_panel_score",
        "pearson_p",
        "spearman_r_mention_diff_vs_panel_score",
        "spearman_p",
    ]
    lines.append(markdown_table(corr[ccols], floatfmt=".3f"))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-outcomes", type=Path, default=DEFAULT_PAIR_OUTCOMES)
    parser.add_argument("--mention-rows", type=Path, default=DEFAULT_MENTION_ROWS)
    parser.add_argument("--allow-missing-traces", action="store_true")
    parser.add_argument("--out-prefix", default=OUT_PREFIX)
    args = parser.parse_args()

    pair_df = load_pair_rows(
        args.pair_outcomes,
        args.mention_rows,
        require_both_traces=not args.allow_missing_traces,
    )
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    pattern = pd.concat(
        [summarize_pattern(pair_df, feature, []) for feature in FEATURES],
        ignore_index=True,
    )
    pattern_by_actor = pd.concat(
        [summarize_pattern(pair_df, feature, ["actor", "actor_label"]) for feature in FEATURES],
        ignore_index=True,
    )
    directional = pd.concat(
        [directional_tests(pair_df, feature, []) for feature in FEATURES],
        ignore_index=True,
    )
    directional_by_actor = pd.concat(
        [directional_tests(pair_df, feature, ["actor", "actor_label"]) for feature in FEATURES],
        ignore_index=True,
    )
    corr = pd.concat([correlations(pair_df, feature, []) for feature in FEATURES], ignore_index=True)
    corr_by_actor = pd.concat(
        [correlations(pair_df, feature, ["actor", "actor_label"]) for feature in FEATURES],
        ignore_index=True,
    )

    suffix = "__allow_missing_traces" if args.allow_missing_traces else "__both_traces_required"
    prefix = f"{args.out_prefix}{suffix}"
    pair_path = ANALYSIS / f"{prefix}_pair_level.csv"
    pattern_path = ANALYSIS / f"{prefix}_pattern_summary.csv"
    pattern_actor_path = ANALYSIS / f"{prefix}_pattern_by_actor.csv"
    directional_path = ANALYSIS / f"{prefix}_directional_tests.csv"
    directional_actor_path = ANALYSIS / f"{prefix}_directional_by_actor.csv"
    corr_path = ANALYSIS / f"{prefix}_correlations.csv"
    corr_actor_path = ANALYSIS / f"{prefix}_correlations_by_actor.csv"
    examples_paths = {
        feature: ANALYSIS / f"{prefix}_examples_{feature}.md"
        for feature in FEATURES
    }
    summary_path = ANALYSIS / f"{prefix}_summary.md"

    pair_df.to_csv(pair_path, index=False)
    pattern.to_csv(pattern_path, index=False)
    pattern_by_actor.to_csv(pattern_actor_path, index=False)
    directional.to_csv(directional_path, index=False)
    directional_by_actor.to_csv(directional_actor_path, index=False)
    corr.to_csv(corr_path, index=False)
    corr_by_actor.to_csv(corr_actor_path, index=False)
    for feature, examples_path in examples_paths.items():
        write_examples(pair_df, examples_path, feature=feature)
    write_summary(pair_df=pair_df, pattern_summary=pattern, directional=directional, corr=corr, path=summary_path)

    print(f"pair level: {pair_path}")
    print(f"pattern summary: {pattern_path}")
    print(f"directional tests: {directional_path}")
    print(f"correlations: {corr_path}")
    print(f"examples: {examples_paths['mentions_any_incentive']}")
    print(f"summary: {summary_path}")
    print()
    print(directional.to_string(index=False))


if __name__ == "__main__":
    main()
