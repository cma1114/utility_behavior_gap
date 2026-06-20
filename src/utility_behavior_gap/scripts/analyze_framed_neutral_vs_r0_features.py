#!/usr/bin/env python3
"""Paired text-feature comparison of framed neutral versus bare-task R0.

This is a local, no-API scaffold-control analysis. It uses the manifest-driven
text-feature catalog produced by ``analyze_final_text_features`` and compares
the direct-instruction neutral arm (``direct_low`` / framed neutral) against
the bare-task baseline (``r0``) on exactly matched actor, task, item, and repeat
keys.

The comparison answers a narrow question: how much does the neutral sponsor /
evaluation wrapper itself change outputs relative to a bare request?
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utility_behavior_gap.constants import ACTOR_LABEL, TASK_LABEL
from utility_behavior_gap.paths import ANALYSIS


DEFAULT_INPUT = ANALYSIS / "final_text_analysis_by_output.csv"
DEFAULT_DEFINITIONS = ANALYSIS / "final_text_analysis_feature_definitions.csv"
OUT_PREFIX = "framed_neutral_vs_r0_features"

PAIR_KEYS = ["actor", "task", "item_label", "repeat"]
FRAME_CONDITION = "direct_low"
R0_CONDITION = "r0"
FRAME_SOURCE_MARKER = "framed_user_strong_headroom"
REPEAT_BLOCK_MARKER = "__repeat-block"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{100 * value:.1f}%"


def fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in df.to_numpy()
    ]
    return "\n".join([header, sep] + rows)


def resolve_path(path: str | None, default: Path) -> Path:
    if path is None:
        return default
    path_obj = Path(path)
    if path_obj.is_absolute() or path_obj.exists():
        return path_obj
    return ANALYSIS / path_obj


def load_definitions(path: str | None) -> tuple[list[str], list[str], dict[str, str]]:
    path_obj = resolve_path(path, DEFAULT_DEFINITIONS)
    definitions = pd.read_csv(path_obj)
    feature_names = definitions.loc[definitions["type"].eq("feature"), "name"].tolist()
    artifact_names = definitions.loc[definitions["type"].eq("artifact"), "name"].tolist()
    definition_by_name = dict(zip(definitions["name"], definitions["definition"]))
    return feature_names, artifact_names, definition_by_name


def check_unique_keys(df: pd.DataFrame, condition: str) -> None:
    sub = df[df["condition"].eq(condition)]
    duplicates = sub[sub.duplicated(PAIR_KEYS, keep=False)]
    if not duplicates.empty:
        sample = duplicates[PAIR_KEYS + ["output_id"]].head(20).to_string(index=False)
        raise ValueError(
            f"{condition} has duplicate paired keys; refusing to build ambiguous pairs.\n{sample}"
        )


def select_framed_neutral(catalog: pd.DataFrame) -> pd.DataFrame:
    """Select the direct-neutral block that is item-matched to the R0 baseline.

    The final text catalog can include extra direct-instruction repeat blocks.
    Those are valid for the main direct-instruction analysis, but R0 was only
    generated for the original framed-neutral item block. Excluding repeat
    blocks keeps this scaffold-control comparison one-to-one.
    """
    framed = catalog[catalog["condition"].eq(FRAME_CONDITION)].copy()
    if "run_dir" not in framed.columns:
        return framed
    run_dir = framed["run_dir"].fillna("").astype(str)
    selected = framed[
        run_dir.str.contains(FRAME_SOURCE_MARKER, regex=False)
        & ~run_dir.str.contains(REPEAT_BLOCK_MARKER, regex=False)
    ].copy()
    if selected.empty:
        raise ValueError(
            "No framed-neutral rows matched the original non-repeat framed_user_strong_headroom block."
        )
    return selected


def build_pairs(
    catalog: pd.DataFrame,
    *,
    feature_names: list[str],
    artifact_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_base = [
        "actor",
        "task",
        "task_label",
        "item_label",
        "item_index",
        "repeat",
        "output_id",
        "output_fingerprint",
        "run_id",
        "run_dir",
        "finish_reason",
        "native_finish_reason",
        "completion_tokens",
        "max_tokens",
        "token_cap_ratio",
    ]
    value_names = [name for name in feature_names + artifact_names if name in catalog.columns]
    keep = [col for col in keep_base if col in catalog.columns] + value_names

    framed_source = select_framed_neutral(catalog)
    check_unique_keys(framed_source, FRAME_CONDITION)
    check_unique_keys(catalog, R0_CONDITION)

    framed = framed_source[keep].copy()
    r0 = catalog[catalog["condition"].eq(R0_CONDITION)][keep].copy()
    pairs = framed.merge(
        r0,
        on=PAIR_KEYS,
        how="inner",
        suffixes=("_framed_neutral", "_r0"),
        validate="one_to_one",
    )

    missing_rows: list[dict[str, Any]] = []
    frame_keys = set(map(tuple, framed[PAIR_KEYS].to_numpy()))
    r0_keys = set(map(tuple, r0[PAIR_KEYS].to_numpy()))
    for key in sorted(frame_keys - r0_keys):
        missing_rows.append(dict(zip(PAIR_KEYS, key), missing_side="r0"))
    for key in sorted(r0_keys - frame_keys):
        missing_rows.append(dict(zip(PAIR_KEYS, key), missing_side="framed_neutral"))
    missing = pd.DataFrame(missing_rows)

    delta_columns: dict[str, pd.Series] = {}
    for feature in value_names:
        framed_col = f"{feature}_framed_neutral"
        r0_col = f"{feature}_r0"
        pairs[framed_col] = pd.to_numeric(pairs[framed_col], errors="coerce")
        pairs[r0_col] = pd.to_numeric(pairs[r0_col], errors="coerce")
        delta_columns[f"delta_{feature}_framed_neutral_minus_r0"] = pairs[framed_col] - pairs[r0_col]

    pairs = pd.concat([pairs, pd.DataFrame(delta_columns, index=pairs.index)], axis=1)
    pairs = pairs.copy()
    pairs["actor_label"] = pairs["actor"].map(ACTOR_LABEL).fillna(pairs["actor"])
    pairs["task_label"] = pairs["task"].map(TASK_LABEL).fillna(pairs["task"])
    return pairs, missing


def equal_cell_mean(df: pd.DataFrame, value_col: str, cell_cols: list[str]) -> tuple[float, int, int]:
    values = pd.to_numeric(df[value_col], errors="coerce")
    clean = df.loc[values.notna()].copy()
    clean[value_col] = values.loc[values.notna()].to_numpy(dtype=float)
    if clean.empty:
        return math.nan, 0, 0
    if not cell_cols:
        return float(clean[value_col].mean()), int(len(clean)), 1
    cell_means = clean.groupby(cell_cols, dropna=False, sort=True)[value_col].mean()
    return float(cell_means.mean()), int(len(clean)), int(len(cell_means))


def bootstrap_equal_cell_mean(
    df: pd.DataFrame,
    value_col: str,
    *,
    cell_cols: list[str],
    iterations: int,
    seed: int,
) -> tuple[float, float, float, int, int]:
    observed, n_rows, n_cells = equal_cell_mean(df, value_col, cell_cols)
    if n_rows == 0 or not np.isfinite(observed):
        return observed, math.nan, math.nan, n_rows, n_cells
    if iterations <= 0:
        return observed, math.nan, math.nan, n_rows, n_cells

    rng = np.random.default_rng(seed)
    if not cell_cols:
        vals = pd.to_numeric(df[value_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) <= 1:
            return observed, observed, observed, n_rows, n_cells
        estimates = rng.choice(vals, size=(iterations, len(vals)), replace=True).mean(axis=1)
        lo, hi = np.quantile(estimates, [0.025, 0.975])
        return observed, float(lo), float(hi), n_rows, n_cells

    clean = df.copy()
    clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
    clean = clean[clean[value_col].notna()]
    cell_frames = [
        sub[value_col].to_numpy(dtype=float)
        for _, sub in clean.groupby(cell_cols, dropna=False, sort=True)
        if not sub.empty
    ]
    if len(cell_frames) <= 1:
        return observed, observed, observed, n_rows, n_cells

    cell_boot_means = np.column_stack(
        [
            rng.choice(vals, size=(iterations, len(vals)), replace=True).mean(axis=1)
            for vals in cell_frames
        ]
    )
    sampled_indices = rng.integers(0, len(cell_frames), size=(iterations, len(cell_frames)))
    estimates = np.take_along_axis(cell_boot_means, sampled_indices, axis=1).mean(axis=1)
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return observed, float(lo), float(hi), n_rows, n_cells


def group_specs() -> list[tuple[str, list[str], list[str]]]:
    return [
        ("overall", [], ["actor", "task"]),
        ("by_task", ["task"], ["actor"]),
        ("by_actor", ["actor"], ["task"]),
        ("by_actor_task", ["actor", "task"], []),
    ]


def summarize_values(
    pairs: pd.DataFrame,
    *,
    names: list[str],
    definition_by_name: dict[str, str],
    kind: str,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seed_offset = 0
    for scope, group_cols, cell_cols in group_specs():
        grouped = [((), pairs)] if not group_cols else pairs.groupby(group_cols, dropna=False, sort=True)
        for key, sub in grouped:
            base: dict[str, Any] = {"scope": scope}
            if group_cols:
                if len(group_cols) == 1 and not isinstance(key, tuple):
                    key = (key,)
                base.update(dict(zip(group_cols, key)))
            for name in names:
                framed_col = f"{name}_framed_neutral"
                r0_col = f"{name}_r0"
                delta_col = f"delta_{name}_framed_neutral_minus_r0"
                if framed_col not in sub.columns or r0_col not in sub.columns or delta_col not in sub.columns:
                    continue
                delta, lo, hi, n_pairs, n_cells = bootstrap_equal_cell_mean(
                    sub,
                    delta_col,
                    cell_cols=cell_cols,
                    iterations=iterations,
                    seed=seed + seed_offset,
                )
                framed_mean, _, _ = equal_cell_mean(sub, framed_col, cell_cols)
                r0_mean, _, _ = equal_cell_mean(sub, r0_col, cell_cols)
                raw_delta = pd.to_numeric(sub[delta_col], errors="coerce").dropna()
                if raw_delta.empty:
                    paired_sd = math.nan
                    standardized_delta = math.nan
                    framed_greater = r0_greater = equal = 0
                else:
                    paired_sd = float(raw_delta.std(ddof=1)) if len(raw_delta) > 1 else math.nan
                    standardized_delta = float(delta / paired_sd) if paired_sd and np.isfinite(paired_sd) else math.nan
                    framed_greater = int((raw_delta > 0).sum())
                    r0_greater = int((raw_delta < 0).sum())
                    equal = int((raw_delta == 0).sum())
                rows.append(
                    {
                        **base,
                        "kind": kind,
                        "name": name,
                        "definition": definition_by_name.get(name, ""),
                        "n_pairs": n_pairs,
                        "n_cells": n_cells,
                        "framed_neutral_mean": framed_mean,
                        "r0_mean": r0_mean,
                        "mean_delta_framed_neutral_minus_r0": delta,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "paired_delta_sd": paired_sd,
                        "standardized_mean_delta": standardized_delta,
                        "framed_neutral_greater_pairs": framed_greater,
                        "r0_greater_pairs": r0_greater,
                        "equal_pairs": equal,
                    }
                )
                seed_offset += 1
    return pd.DataFrame(rows)


def write_summary(
    *,
    path,
    input_path: str,
    definitions_path: str,
    pairs: pd.DataFrame,
    missing: pd.DataFrame,
    feature_summary: pd.DataFrame,
    artifact_summary: pd.DataFrame,
    iterations: int,
) -> None:
    counts = pairs.groupby("task", dropna=False).size().reset_index(name="matched_pairs")
    counts["task_label"] = counts["task"].map(TASK_LABEL).fillna(counts["task"])
    counts = counts[["task_label", "matched_pairs"]]

    overall = feature_summary[
        feature_summary["scope"].eq("overall")
        & feature_summary["kind"].eq("feature")
        & feature_summary["name"].isin(
            [
                "words",
                "sentences",
                "paragraphs",
                "avg_sentence_words",
                "numeric_specificity_per_1k",
                "transition_markers_per_1k",
                "counterargument_markers_per_1k",
                "positive_words_per_1k",
                "negative_words_per_1k",
                "textstat_flesch_reading_ease",
                "textstat_flesch_kincaid_grade",
                "spacy_modifier_rate",
            ]
        )
    ].copy()
    overall_display = overall[
        [
            "name",
            "framed_neutral_mean",
            "r0_mean",
            "mean_delta_framed_neutral_minus_r0",
            "ci_lo",
            "ci_hi",
            "standardized_mean_delta",
        ]
    ].copy()
    for col in ["framed_neutral_mean", "r0_mean", "mean_delta_framed_neutral_minus_r0", "ci_lo", "ci_hi"]:
        overall_display[col] = overall_display[col].map(lambda x: fmt(float(x), 3))
    overall_display["standardized_mean_delta"] = overall_display["standardized_mean_delta"].map(
        lambda x: fmt(float(x), 3)
    )

    words_task = feature_summary[
        feature_summary["scope"].eq("by_task")
        & feature_summary["kind"].eq("feature")
        & feature_summary["name"].eq("words")
    ].copy()
    words_task["task_label"] = words_task["task"].map(TASK_LABEL).fillna(words_task["task"])
    words_task_display = words_task[
        ["task_label", "n_pairs", "framed_neutral_mean", "r0_mean", "mean_delta_framed_neutral_minus_r0", "ci_lo", "ci_hi"]
    ].copy()
    for col in ["framed_neutral_mean", "r0_mean", "mean_delta_framed_neutral_minus_r0", "ci_lo", "ci_hi"]:
        words_task_display[col] = words_task_display[col].map(lambda x: fmt(float(x), 2))

    trunc = artifact_summary[
        artifact_summary["scope"].eq("overall")
        & artifact_summary["kind"].eq("artifact")
        & artifact_summary["name"].isin(["explicit_length_truncation", "near_token_cap_95", "refusal_or_meta"])
    ][["name", "framed_neutral_mean", "r0_mean", "mean_delta_framed_neutral_minus_r0", "ci_lo", "ci_hi"]].copy()
    for col in ["framed_neutral_mean", "r0_mean", "mean_delta_framed_neutral_minus_r0", "ci_lo", "ci_hi"]:
        trunc[col] = trunc[col].map(lambda x: fmt(float(x), 4))

    lines = [
        "# Framed Neutral Versus R0 Feature Analysis",
        "",
        "This is a local, no-API scaffold-control analysis.",
        "",
        "Comparison:",
        "",
        "- `framed_neutral`: condition `direct_low`, the neutral sponsor/evaluation wrapper.",
        "- `R0`: condition `r0`, the bare task request.",
        "- Pairing key: `actor`, `task`, `item_label`, `repeat`.",
        "",
        "Interpretation: positive deltas mean framed neutral has more of the feature than R0.",
        "This does not use panel judgments; it describes text-feature changes caused by the neutral wrapper itself.",
        "",
        "Inference:",
        "",
        f"- Bootstrap iterations: {iterations}.",
        "- Overall means and CIs use equal-weight actor x task cells.",
        "- By-task summaries use equal-weight actor cells.",
        "- By-actor summaries use equal-weight task cells.",
        "- By-actor-task summaries use row bootstrap within that cell.",
        "",
        f"Input catalog: `{input_path}`",
        f"Feature definitions: `{definitions_path}`",
        "",
        "## Matched Coverage",
        "",
        markdown_table(counts),
        "",
        f"Unmatched rows: {len(missing)}",
        "",
        "## Overall Generic Feature Deltas",
        "",
        markdown_table(overall_display),
        "",
        "## Word Count By Task",
        "",
        markdown_table(words_task_display),
        "",
        "## Key Artifact Checks",
        "",
        markdown_table(trunc),
        "",
        "## Output Files",
        "",
        f"- `{ANALYSIS / f'{OUT_PREFIX}_pairs.csv'}`",
        f"- `{ANALYSIS / f'{OUT_PREFIX}_feature_summary.csv'}`",
        f"- `{ANALYSIS / f'{OUT_PREFIX}_artifact_summary.csv'}`",
        f"- `{ANALYSIS / f'{OUT_PREFIX}_missing_pairs.csv'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="final_text_analysis_by_output.csv path")
    parser.add_argument("--definitions", default=str(DEFAULT_DEFINITIONS), help="feature definitions CSV path")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    input_path = str(resolve_path(args.input, DEFAULT_INPUT))
    definitions_path = str(resolve_path(args.definitions, DEFAULT_DEFINITIONS))
    feature_names, artifact_names, definition_by_name = load_definitions(definitions_path)
    catalog = pd.read_csv(input_path, low_memory=False)

    missing_conditions = {FRAME_CONDITION, R0_CONDITION} - set(catalog["condition"].dropna().unique())
    if missing_conditions:
        raise ValueError(f"missing expected conditions from text catalog: {sorted(missing_conditions)}")

    pairs, missing = build_pairs(catalog, feature_names=feature_names, artifact_names=artifact_names)
    if pairs.empty:
        raise ValueError("no matched framed-neutral/R0 pairs found")

    feature_summary = summarize_values(
        pairs,
        names=[name for name in feature_names if name in catalog.columns],
        definition_by_name=definition_by_name,
        kind="feature",
        iterations=args.bootstrap,
        seed=args.seed,
    )
    artifact_summary = summarize_values(
        pairs,
        names=[name for name in artifact_names if name in catalog.columns],
        definition_by_name=definition_by_name,
        kind="artifact",
        iterations=args.bootstrap,
        seed=args.seed + 100000,
    )

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    pair_path = ANALYSIS / f"{OUT_PREFIX}_pairs.csv"
    feature_path = ANALYSIS / f"{OUT_PREFIX}_feature_summary.csv"
    artifact_path = ANALYSIS / f"{OUT_PREFIX}_artifact_summary.csv"
    missing_path = ANALYSIS / f"{OUT_PREFIX}_missing_pairs.csv"
    summary_path = ANALYSIS / f"{OUT_PREFIX}_summary.md"

    pairs.to_csv(pair_path, index=False)
    feature_summary.to_csv(feature_path, index=False)
    artifact_summary.to_csv(artifact_path, index=False)
    missing.to_csv(missing_path, index=False)
    write_summary(
        path=summary_path,
        input_path=input_path,
        definitions_path=definitions_path,
        pairs=pairs,
        missing=missing,
        feature_summary=feature_summary,
        artifact_summary=artifact_summary,
        iterations=args.bootstrap,
    )

    print(f"matched pairs: {len(pairs)}")
    print(f"missing pairs: {len(missing)}")
    print(f"pairs: {pair_path}")
    print(f"feature summary: {feature_path}")
    print(f"artifact summary: {artifact_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
