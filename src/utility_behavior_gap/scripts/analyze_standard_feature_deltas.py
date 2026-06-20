#!/usr/bin/env python3
"""Paired standard generic-feature deltas for finalized high/low contrasts.

This local analysis uses ``outputs/analysis/final_text_analysis_pair_deltas.csv``.
It is the contrast-generic version of the direct-instruction feature-delta
summary. It compares the row's high arm against its matched low arm:

    high_condition - low_condition

The primary overall estimate is an equal actor-task-cell mean. Confidence
intervals use a nonparametric bootstrap over actor and task cells, so larger
task samples do not dominate the aggregate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utility_behavior_gap.constants import TASK_LABEL
from utility_behavior_gap.output_exclusions import (
    filter_semantic_excluded_pair_rows,
    filter_valid_pair_rows,
)
from utility_behavior_gap.paths import ANALYSIS, ANALYSIS_SPECS
from utility_behavior_gap.scripts.analyze_direct_instruction_feature_deltas import (
    FEATURE_DEFINITIONS,
    PAIR_DELTAS,
    RANDOM_SEED,
    add_composite_features,
    available_features,
    fmt,
    markdown_table,
    summarize_scope,
)


CONTRASTS: dict[str, dict[str, str]] = {
    "direct_instruction": {
        "high_condition": "direct_high",
        "low_condition": "direct_low",
        "high_raw_condition": "framed_user_strong",
        "low_raw_condition": "framed_neutral",
        "description": "finalized exhortative user-prompt arm minus framed neutral",
    },
    "utility": {
        "high_condition": "utility_high",
        "low_condition": "utility_low",
        "high_raw_condition": "hl_high",
        "low_raw_condition": "hl_low",
        "description": "high-utility arm minus low-utility arm",
    },
    "amount": {
        "high_condition": "amount_high",
        "low_condition": "amount_low",
        "high_raw_condition": "amount_high",
        "low_raw_condition": "amount_low",
        "description": "larger-amount arm minus smaller-amount arm",
    },
    "moral": {
        "high_condition": "moral_high",
        "low_condition": "moral_low",
        "high_raw_condition": "moral_good",
        "low_raw_condition": "moral_bad",
        "description": "morally good arm minus morally bad arm",
    },
}


RENAMES = {
    "strong_mean_equal_cell": "high_mean_equal_cell",
    "neutral_mean_equal_cell": "low_mean_equal_cell",
    "mean_delta_strong_minus_neutral": "mean_delta_high_minus_low",
    "raw_strong_mean": "raw_high_mean",
    "raw_neutral_mean": "raw_low_mean",
    "pct_pairs_strong_greater": "pct_pairs_high_greater",
    "pct_pairs_neutral_greater": "pct_pairs_low_greater",
}


def validate_rows(df: pd.DataFrame, *, contrast: str) -> None:
    spec = CONTRASTS[contrast]
    mask = (
        df["high_condition"].eq(spec["high_condition"])
        & df["low_condition"].eq(spec["low_condition"])
        & df["high_raw_condition"].eq(spec["high_raw_condition"])
        & df["low_raw_condition"].eq(spec["low_raw_condition"])
    )
    if mask.all():
        return
    bad = df.loc[
        ~mask,
        [
            "actor",
            "task",
            "source_dataset",
            "high_condition",
            "low_condition",
            "high_raw_condition",
            "low_raw_condition",
            "run_id",
        ],
    ].head(20)
    raise ValueError(
        f"{contrast} feature rows do not match the expected finalized high/low arms.\n"
        + bad.to_string(index=False)
    )


def relabel_summary(df: pd.DataFrame, *, contrast: str) -> pd.DataFrame:
    out = df.rename(columns=RENAMES).copy()
    spec = CONTRASTS[contrast]
    out.insert(0, "contrast", contrast)
    out.insert(1, "high_condition", spec["high_condition"])
    out.insert(2, "low_condition", spec["low_condition"])
    return out


def compact_table(overall: pd.DataFrame) -> pd.DataFrame:
    selected = overall.copy()
    selected["abs_std_delta"] = selected["standardized_delta_pair_sd"].abs()
    selected = selected.sort_values("abs_std_delta", ascending=False)
    return pd.DataFrame(
        {
            "feature": selected["feature"],
            "high mean": selected["high_mean_equal_cell"].map(lambda x: fmt(x, 3)),
            "low mean": selected["low_mean_equal_cell"].map(lambda x: fmt(x, 3)),
            "delta": selected["mean_delta_high_minus_low"].map(lambda x: fmt(x, 3)),
            "95% CI": [
                f"[{fmt(lo, 3)}, {fmt(hi, 3)}]"
                for lo, hi in zip(selected["ci_low"], selected["ci_high"])
            ],
            "std delta": selected["standardized_delta_pair_sd"].map(lambda x: fmt(x, 3)),
        }
    )


def display_effects(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": df["feature"],
            "delta": df["mean_delta_high_minus_low"].map(lambda x: fmt(x, 3)),
            "95% CI": [
                f"[{fmt(lo, 3)}, {fmt(hi, 3)}]"
                for lo, hi in zip(df["ci_low"], df["ci_high"])
            ],
            "std delta": df["standardized_delta_pair_sd"].map(lambda x: fmt(x, 3)),
        }
    )


def task_summary(by_task: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, sub in by_task.groupby("task", sort=True):
        clear = sub[sub["ci_excludes_zero"]].copy()
        positive = clear[clear["mean_delta_high_minus_low"].gt(0)]
        negative = clear[clear["mean_delta_high_minus_low"].lt(0)]
        rows.append(
            {
                "task": TASK_LABEL.get(str(task), str(task)),
                "clear high > low features": str(len(positive)),
                "clear low > high features": str(len(negative)),
                "largest high > low": (
                    positive.sort_values("standardized_delta_pair_sd", ascending=False)[
                        "feature"
                    ].iloc[0]
                    if not positive.empty
                    else ""
                ),
                "largest low > high": (
                    negative.sort_values("standardized_delta_pair_sd", ascending=True)[
                        "feature"
                    ].iloc[0]
                    if not negative.empty
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    *,
    out_dir: Path,
    out_prefix: str,
    contrast: str,
    overall: pd.DataFrame,
    by_task: pd.DataFrame,
    skipped_features: list[str],
    input_path: Path,
    definitions_path: Path,
    iterations: int,
    valid_output_filter_report: dict[str, object],
    semantic_exclusion_filter_report: dict[str, object],
) -> Path:
    spec = CONTRASTS[contrast]
    positive = overall.sort_values("standardized_delta_pair_sd", ascending=False)
    negative = overall.sort_values("standardized_delta_pair_sd", ascending=True)
    clear = overall[overall["ci_excludes_zero"]].copy()

    lines = [
        f"# {contrast.replace('_', ' ').title()} Standard Generic-Feature Deltas",
        "",
        f"Comparison: `{spec['high_condition']} - {spec['low_condition']}` ({spec['description']}).",
        "",
        "The primary overall estimate is the equal actor-task-cell mean. Its confidence interval uses a nonparametric bootstrap over actor and task cells.",
        "",
        "Generic features use the standard paper-facing set from `analysis_specs/feature_definitions.yaml`: words, paragraphs, unique-word ratio, quantitative detail, Flesch-Kincaid grade, positive-word rate, and negative-word rate. Quantitative detail is `z(numbers + percentages)` standardized within task before paired differencing.",
        "",
        f"- input pair catalog: `{input_path}`",
        f"- feature definitions CSV: `{definitions_path}`",
        f"- editable feature spec: `{ANALYSIS_SPECS / 'feature_definitions.yaml'}`",
        f"- bootstrap iterations: `{iterations}`",
        f"- valid-output filter: `{valid_output_filter_report}`",
        f"- semantic exclusion filter: `{semantic_exclusion_filter_report}`",
        f"- pairs: `{int(overall['n_pairs'].max()) if not overall.empty else 0}`",
        f"- actor-task cells: `{int(overall['n_cells'].max()) if not overall.empty else 0}`",
        f"- standard generic features analyzed: `{len(overall)}`",
        f"- clear overall differences: `{len(clear)}`",
        f"- skipped feature columns: `{', '.join(skipped_features) if skipped_features else 'none'}`",
        "",
        "## Compact Overall Table",
        "",
        markdown_table(compact_table(overall)),
        "",
        "## Largest High-Minus-Low Deltas",
        "",
        markdown_table(display_effects(positive)),
        "",
        "## Largest Low-Minus-High Deltas",
        "",
        markdown_table(display_effects(negative)),
        "",
        "## Task Summary",
        "",
        markdown_table(task_summary(by_task)),
        "",
        "## Outputs",
        "",
        f"- overall: `{out_dir / (out_prefix + '_overall.csv')}`",
        f"- by task: `{out_dir / (out_prefix + '_by_task.csv')}`",
        f"- by actor: `{out_dir / (out_prefix + '_by_actor.csv')}`",
        f"- by actor-task: `{out_dir / (out_prefix + '_by_actor_task.csv')}`",
        f"- significant overall features: `{out_dir / (out_prefix + '_overall_significant.csv')}`",
    ]

    summary_path = out_dir / f"{out_prefix}_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    clear.to_csv(out_dir / f"{out_prefix}_overall_significant.csv", index=False)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast", choices=sorted(CONTRASTS), default="utility")
    parser.add_argument("--input", type=Path, default=PAIR_DELTAS)
    parser.add_argument("--definitions", type=Path, default=FEATURE_DEFINITIONS)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS)
    parser.add_argument("--out-prefix")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_prefix = args.out_prefix or f"{args.contrast}_feature_deltas"

    pairs = pd.read_csv(args.input, low_memory=False)
    contrast_df = pairs[pairs["contrast"].eq(args.contrast)].copy()
    if contrast_df.empty:
        raise ValueError(f"No {args.contrast!r} rows found in pair-delta catalog.")
    validate_rows(contrast_df, contrast=args.contrast)
    contrast_df, valid_output_filter_report = filter_valid_pair_rows(contrast_df)
    if contrast_df.empty:
        raise ValueError(
            f"No {args.contrast!r} rows remain after valid-output filtering."
        )
    contrast_df, semantic_exclusion_filter_report = filter_semantic_excluded_pair_rows(
        contrast_df
    )
    if contrast_df.empty:
        raise ValueError(
            f"No {args.contrast!r} rows remain after semantic exclusion filtering."
        )
    contrast_df = add_composite_features(contrast_df)

    features, definition_by_name, skipped = available_features(contrast_df, args.definitions)

    overall = relabel_summary(
        summarize_scope(
            contrast_df,
            features=features,
            definition_by_name=definition_by_name,
            group_cols=[],
            cell_cols=["actor", "task"],
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
        contrast=args.contrast,
    )
    by_task = relabel_summary(
        summarize_scope(
            contrast_df,
            features=features,
            definition_by_name=definition_by_name,
            group_cols=["task"],
            cell_cols=["actor"],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 100,
        ),
        contrast=args.contrast,
    )
    by_actor = relabel_summary(
        summarize_scope(
            contrast_df,
            features=features,
            definition_by_name=definition_by_name,
            group_cols=["actor"],
            cell_cols=["task"],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 200,
        ),
        contrast=args.contrast,
    )
    by_actor_task = relabel_summary(
        summarize_scope(
            contrast_df,
            features=features,
            definition_by_name=definition_by_name,
            group_cols=["actor", "task"],
            cell_cols=[],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 300,
        ),
        contrast=args.contrast,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(args.out_dir / f"{out_prefix}_overall.csv", index=False)
    by_task.to_csv(args.out_dir / f"{out_prefix}_by_task.csv", index=False)
    by_actor.to_csv(args.out_dir / f"{out_prefix}_by_actor.csv", index=False)
    by_actor_task.to_csv(args.out_dir / f"{out_prefix}_by_actor_task.csv", index=False)

    summary_path = write_summary(
        out_dir=args.out_dir,
        out_prefix=out_prefix,
        contrast=args.contrast,
        overall=overall,
        by_task=by_task,
        skipped_features=skipped,
        input_path=args.input,
        definitions_path=args.definitions,
        iterations=args.bootstrap_iterations,
        valid_output_filter_report=valid_output_filter_report,
        semantic_exclusion_filter_report=semantic_exclusion_filter_report,
    )

    print(f"contrast: {args.contrast}")
    print(f"valid-output filter: {valid_output_filter_report}")
    print(f"semantic exclusion filter: {semantic_exclusion_filter_report}")
    print(f"pairs: {len(contrast_df)}")
    print(f"features analyzed: {len(features)}")
    print(f"summary: {summary_path}")
    print(f"overall: {args.out_dir / (out_prefix + '_overall.csv')}")


if __name__ == "__main__":
    main()
