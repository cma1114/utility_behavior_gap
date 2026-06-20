#!/usr/bin/env python3
"""Generic feature deltas for arbitrary matched output arms.

This local, no-API analysis reads ``final_text_analysis_by_output.csv`` and
matches two conditions on actor, task, item label, and repeat. It produces the
same generic-feature summary schema used by ``make_feature_appendix_table.py``,
so bridge comparisons can be combined with task-specific LLM rubric coding in
one paper-facing table.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from utility_behavior_gap.output_exclusions import (
    filter_semantic_excluded_output_catalog,
    filter_valid_output_catalog,
)
from utility_behavior_gap.paths import ANALYSIS
from utility_behavior_gap.scripts.analyze_direct_instruction_feature_deltas import (
    FEATURE_DEFINITIONS,
    RANDOM_SEED,
    add_composite_features,
    available_features,
    fmt,
    markdown_table,
    summarize_scope,
)


BY_OUTPUT = ANALYSIS / "final_text_analysis_by_output.csv"
PAIR_KEYS = ["actor", "task", "item_label", "repeat"]


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "none"


def dedupe_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    sub = df[df["condition"].eq(condition)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for condition={condition!r}.")
    duplicates = sub[sub.duplicated(PAIR_KEYS, keep=False)]
    if not duplicates.empty:
        sample = duplicates[PAIR_KEYS + ["output_id", "run_dir"]].head(20).to_string(index=False)
        raise ValueError(f"{condition!r} has duplicate match keys; refusing ambiguous match.\n{sample}")
    return sub


def build_pairs(df: pd.DataFrame, left_condition: str, right_condition: str) -> pd.DataFrame:
    left = dedupe_condition(df, left_condition)
    right = dedupe_condition(df, right_condition)
    merged = left.merge(
        right,
        on=PAIR_KEYS,
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(f"No matched rows for {left_condition!r} vs {right_condition!r}.")

    rows: list[dict[str, object]] = []
    feature_like_columns = [
        col
        for col in df.columns
        if col
        not in {
            "actor",
            "actor_label",
            "task",
            "task_label",
            "domain",
            "domain_label",
            "item_label",
            "item_index",
            "repeat",
            "pair_idx",
            "pair_set",
            "run_dir",
            "run_id",
            "source_dataset",
            "source_family",
            "contrast",
            "pair_uid",
            "condition",
            "raw_condition",
            "side",
            "output_id",
            "output_fingerprint",
            "finish_reason",
            "native_finish_reason",
        }
    ]
    for row in merged.to_dict(orient="records"):
        out: dict[str, object] = {
            "actor": row["actor"],
            "task": row["task"],
            "task_label": row.get("task_label_left") or row.get("task_label_right") or row["task"],
            "item_label": row["item_label"],
            "item_index": row.get("item_index_left", row.get("item_index_right", "")),
            "repeat": row["repeat"],
            "pair_uid": (
                f"{left_condition}_vs_{right_condition}:"
                f"{row['actor']}:{row['task']}:i{slug(str(row['item_label']))}:r{row['repeat']}"
            ),
            "high_condition": left_condition,
            "low_condition": right_condition,
            "high_raw_condition": row.get("raw_condition_left", left_condition),
            "low_raw_condition": row.get("raw_condition_right", right_condition),
            "high_output_id": row.get("output_id_left", ""),
            "low_output_id": row.get("output_id_right", ""),
            "source_dataset": "arm_match",
            "run_id": "",
        }
        for feature in feature_like_columns:
            left_col = f"{feature}_left"
            right_col = f"{feature}_right"
            if left_col not in row or right_col not in row:
                continue
            left_value = pd.to_numeric(pd.Series([row[left_col]]), errors="coerce").iloc[0]
            right_value = pd.to_numeric(pd.Series([row[right_col]]), errors="coerce").iloc[0]
            out[f"high_{feature}"] = left_value
            out[f"low_{feature}"] = right_value
            out[f"delta_{feature}"] = left_value - right_value
        rows.append(out)

    return pd.DataFrame(rows)


def infer_bridge_output_cols(df: pd.DataFrame) -> tuple[str, str]:
    left_candidates = [
        "source_highlow_output_id",
        "source_amount_output_id",
        "source_moral_output_id",
        "source_side_output_id",
    ]
    right_candidates = [
        "source_neutral_output_id",
        "source_r0_output_id",
        "source_framed_empty_output_id",
        "source_empty_output_id",
    ]
    left = next((name for name in left_candidates if name in df.columns), None)
    right = next((name for name in right_candidates if name in df.columns), None)
    if left is None or right is None:
        raise ValueError(
            "Could not infer bridge output id columns; pass --bridge-left-output-col "
            "and --bridge-right-output-col."
        )
    return left, right


def build_pairs_from_bridge(
    catalog: pd.DataFrame,
    *,
    bridge_path: Path,
    left_condition: str,
    right_condition: str,
    left_output_col: str | None,
    right_output_col: str | None,
) -> pd.DataFrame:
    bridge = pd.read_csv(bridge_path, low_memory=False)
    if "actor" in bridge.columns:
        bridge = bridge[~bridge["actor"].fillna("").astype(str).eq("actor")].copy()
    if left_output_col is None or right_output_col is None:
        inferred_left, inferred_right = infer_bridge_output_cols(bridge)
        left_output_col = left_output_col or inferred_left
        right_output_col = right_output_col or inferred_right

    needed = [left_output_col, right_output_col]
    missing = [name for name in needed if name not in bridge.columns]
    if missing:
        raise ValueError(f"Bridge file missing output id column(s): {missing}")

    catalog_by_id = catalog.drop_duplicates("output_id", keep="first").set_index("output_id")
    feature_like_columns = [
        col
        for col in catalog.columns
        if col
        not in {
            "actor",
            "actor_label",
            "task",
            "task_label",
            "domain",
            "domain_label",
            "item_label",
            "item_index",
            "repeat",
            "pair_idx",
            "pair_set",
            "run_dir",
            "run_id",
            "source_dataset",
            "source_family",
            "contrast",
            "pair_uid",
            "condition",
            "raw_condition",
            "side",
            "output_id",
            "output_fingerprint",
            "finish_reason",
            "native_finish_reason",
        }
    ]

    rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for row in bridge.to_dict(orient="records"):
        left_output_id = str(row.get(left_output_col, "") or "")
        right_output_id = str(row.get(right_output_col, "") or "")
        if left_output_id not in catalog_by_id.index or right_output_id not in catalog_by_id.index:
            missing_rows.append(
                {
                    "left_output_id": left_output_id,
                    "right_output_id": right_output_id,
                    "bridge_pair_uid": row.get("pair_uid") or row.get("bridge_pair_uid", ""),
                }
            )
            continue
        left = catalog_by_id.loc[left_output_id]
        right = catalog_by_id.loc[right_output_id]
        out: dict[str, object] = {
            "actor": row.get("actor", left.get("actor", "")),
            "task": row.get("task", left.get("task", "")),
            "task_label": row.get("task_label", left.get("task_label", left.get("task", ""))),
            "item_label": row.get("item_label") or row.get("item_id") or left.get("item_label", ""),
            "item_index": left.get("item_index", ""),
            "repeat": row.get("repeat", left.get("repeat", "")),
            "pair_uid": row.get("pair_uid") or row.get("bridge_pair_uid", ""),
            "high_condition": left_condition,
            "low_condition": right_condition,
            "high_raw_condition": left.get("raw_condition", left_condition),
            "low_raw_condition": right.get("raw_condition", right_condition),
            "high_output_id": left_output_id,
            "low_output_id": right_output_id,
            "source_dataset": "bridge_outcomes",
            "run_id": "",
        }
        if "panel_winner_condition" in row:
            out["panel_winner_condition"] = row["panel_winner_condition"]
        for feature in feature_like_columns:
            left_value = pd.to_numeric(pd.Series([left.get(feature)]), errors="coerce").iloc[0]
            right_value = pd.to_numeric(pd.Series([right.get(feature)]), errors="coerce").iloc[0]
            out[f"high_{feature}"] = left_value
            out[f"low_{feature}"] = right_value
            out[f"delta_{feature}"] = left_value - right_value
        rows.append(out)

    if not rows:
        sample = missing_rows[:5]
        raise ValueError(f"No bridge rows had both outputs in the feature catalog. Sample: {sample}")
    return pd.DataFrame(rows)


def relabel_summary(
    df: pd.DataFrame,
    *,
    left_condition: str,
    right_condition: str,
    left_key: str,
    right_key: str,
) -> pd.DataFrame:
    out = df.rename(
        columns={
            "strong_mean_equal_cell": f"{left_key}_mean_equal_cell",
            "neutral_mean_equal_cell": f"{right_key}_mean_equal_cell",
            "mean_delta_strong_minus_neutral": f"mean_delta_{left_key}_minus_{right_key}",
            "raw_strong_mean": f"raw_{left_key}_mean",
            "raw_neutral_mean": f"raw_{right_key}_mean",
            "pct_pairs_strong_greater": f"pct_pairs_{left_key}_greater",
            "pct_pairs_neutral_greater": f"pct_pairs_{right_key}_greater",
        }
    ).copy()
    out.insert(0, "comparison", f"{left_condition}_vs_{right_condition}")
    out.insert(1, "left_condition", left_condition)
    out.insert(2, "right_condition", right_condition)
    return out


def write_summary(
    *,
    path: Path,
    left_condition: str,
    right_condition: str,
    left_key: str,
    right_key: str,
    pairs: pd.DataFrame,
    overall: pd.DataFrame,
    by_task: pd.DataFrame,
    skipped_features: list[str],
    input_path: Path,
    definitions_path: Path,
    iterations: int,
    out_prefix: Path,
    valid_output_filter_report: dict[str, object],
    semantic_exclusion_filter_report: dict[str, object],
) -> None:
    delta_col = f"mean_delta_{left_key}_minus_{right_key}"
    clear = overall[overall["ci_excludes_zero"]].copy()

    compact = overall.copy()
    compact["abs_std_delta"] = compact["standardized_delta_pair_sd"].abs()
    compact = compact.sort_values("abs_std_delta", ascending=False)
    compact_display = pd.DataFrame(
        {
            "feature": compact["feature"],
            "delta": compact[delta_col].map(lambda x: fmt(x, 3)),
            "95% CI": [
                f"[{fmt(lo, 3)}, {fmt(hi, 3)}]"
                for lo, hi in zip(compact["ci_low"], compact["ci_high"])
            ],
            "std delta": compact["standardized_delta_pair_sd"].map(lambda x: fmt(x, 3)),
        }
    )

    task_rows: list[dict[str, str]] = []
    for task, sub in by_task.groupby("task", sort=True):
        task_clear = sub[sub["ci_excludes_zero"]].copy()
        positive = task_clear[task_clear[delta_col].gt(0)]
        negative = task_clear[task_clear[delta_col].lt(0)]
        task_rows.append(
            {
                "task": str(sub["task_label"].iloc[0]) if "task_label" in sub else str(task),
                f"clear {left_condition} > {right_condition} features": str(len(positive)),
                f"clear {right_condition} > {left_condition} features": str(len(negative)),
            }
        )

    lines = [
        f"# {left_condition} Versus {right_condition} Standard Generic-Feature Deltas",
        "",
        f"Comparison: `{left_condition} - {right_condition}` on matched actor/task/item/repeat outputs.",
        "",
        "The primary overall estimate is the equal actor-task-cell mean. Confidence intervals use the same bootstrap convention as the other standard generic-feature tables.",
        "",
        f"- input output catalog: `{input_path}`",
        f"- feature definitions CSV: `{definitions_path}`",
        f"- bootstrap iterations: `{iterations}`",
        f"- valid-output filter: `{valid_output_filter_report}`",
        f"- semantic exclusion filter: `{semantic_exclusion_filter_report}`",
        f"- matched pairs: `{len(pairs)}`",
        f"- standard generic features analyzed: `{len(overall)}`",
        f"- clear overall differences: `{len(clear)}`",
        f"- skipped feature columns: `{', '.join(skipped_features) if skipped_features else 'none'}`",
        "",
        "## Compact Overall Table",
        "",
        markdown_table(compact_display),
        "",
        "## Task Summary",
        "",
        markdown_table(pd.DataFrame(task_rows)),
        "",
        "## Outputs",
        "",
        f"- overall: `{out_prefix}_overall.csv`",
        f"- by task: `{out_prefix}_by_task.csv`",
        f"- by actor: `{out_prefix}_by_actor.csv`",
        f"- by actor-task: `{out_prefix}_by_actor_task.csv`",
        f"- matched pairs: `{out_prefix}_pairs.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-condition", required=True)
    parser.add_argument("--right-condition", required=True)
    parser.add_argument("--left-key")
    parser.add_argument("--right-key")
    parser.add_argument("--input", type=Path, default=BY_OUTPUT)
    parser.add_argument("--definitions", type=Path, default=FEATURE_DEFINITIONS)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS)
    parser.add_argument("--out-prefix")
    parser.add_argument(
        "--bridge-outcomes",
        type=Path,
        help="Optional judged bridge pair-outcome CSV. When provided, pairs are built from exact output ids in this file instead of condition key matching.",
    )
    parser.add_argument("--bridge-left-output-col")
    parser.add_argument("--bridge-right-output-col")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_key = args.left_key or slug(args.left_condition)
    right_key = args.right_key or slug(args.right_condition)
    out_prefix = args.out_prefix or f"{left_key}_vs_{right_key}_feature_deltas"
    out_stem = args.out_dir / out_prefix

    catalog = pd.read_csv(args.input, low_memory=False)
    catalog, valid_output_filter_report = filter_valid_output_catalog(catalog)
    catalog, semantic_exclusion_filter_report = filter_semantic_excluded_output_catalog(
        catalog
    )
    if args.bridge_outcomes:
        pairs = build_pairs_from_bridge(
            catalog,
            bridge_path=args.bridge_outcomes,
            left_condition=args.left_condition,
            right_condition=args.right_condition,
            left_output_col=args.bridge_left_output_col,
            right_output_col=args.bridge_right_output_col,
        )
    else:
        pairs = build_pairs(catalog, args.left_condition, args.right_condition)
    pairs = add_composite_features(pairs)
    features, definition_by_name, skipped = available_features(pairs, args.definitions)

    overall = relabel_summary(
        summarize_scope(
            pairs,
            features=features,
            definition_by_name=definition_by_name,
            group_cols=[],
            cell_cols=["actor", "task"],
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
        left_condition=args.left_condition,
        right_condition=args.right_condition,
        left_key=left_key,
        right_key=right_key,
    )
    by_task = relabel_summary(
        summarize_scope(
            pairs,
            features=features,
            definition_by_name=definition_by_name,
            group_cols=["task"],
            cell_cols=["actor"],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 100,
        ),
        left_condition=args.left_condition,
        right_condition=args.right_condition,
        left_key=left_key,
        right_key=right_key,
    )
    by_actor = relabel_summary(
        summarize_scope(
            pairs,
            features=features,
            definition_by_name=definition_by_name,
            group_cols=["actor"],
            cell_cols=["task"],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 200,
        ),
        left_condition=args.left_condition,
        right_condition=args.right_condition,
        left_key=left_key,
        right_key=right_key,
    )
    by_actor_task = relabel_summary(
        summarize_scope(
            pairs,
            features=features,
            definition_by_name=definition_by_name,
            group_cols=["actor", "task"],
            cell_cols=[],
            iterations=args.bootstrap_iterations,
            seed=args.seed + 300,
        ),
        left_condition=args.left_condition,
        right_condition=args.right_condition,
        left_key=left_key,
        right_key=right_key,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_stem.with_name(out_stem.name + "_pairs.csv"), index=False)
    overall.to_csv(out_stem.with_name(out_stem.name + "_overall.csv"), index=False)
    by_task.to_csv(out_stem.with_name(out_stem.name + "_by_task.csv"), index=False)
    by_actor.to_csv(out_stem.with_name(out_stem.name + "_by_actor.csv"), index=False)
    by_actor_task.to_csv(out_stem.with_name(out_stem.name + "_by_actor_task.csv"), index=False)
    overall[overall["ci_excludes_zero"]].to_csv(
        out_stem.with_name(out_stem.name + "_overall_significant.csv"),
        index=False,
    )
    summary_path = out_stem.with_name(out_stem.name + "_summary.md")
    write_summary(
        path=summary_path,
        left_condition=args.left_condition,
        right_condition=args.right_condition,
        left_key=left_key,
        right_key=right_key,
        pairs=pairs,
        overall=overall,
        by_task=by_task,
        skipped_features=skipped,
        input_path=args.input,
        definitions_path=args.definitions,
        iterations=args.bootstrap_iterations,
        out_prefix=out_stem,
        valid_output_filter_report=valid_output_filter_report,
        semantic_exclusion_filter_report=semantic_exclusion_filter_report,
    )

    print(f"comparison: {args.left_condition} - {args.right_condition}")
    print(f"valid-output filter: {valid_output_filter_report}")
    print(f"semantic exclusion filter: {semantic_exclusion_filter_report}")
    print(f"matched pairs: {len(pairs)}")
    print(f"features analyzed: {len(features)}")
    print(f"summary: {summary_path}")
    print(f"by task: {out_stem.with_name(out_stem.name + '_by_task.csv')}")


if __name__ == "__main__":
    main()
