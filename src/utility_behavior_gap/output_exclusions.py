"""Shared output-exclusion filters for analysis.

Two exclusions are intentionally separate:

1. Mechanical invalid-output checks, which use local metadata such as missing
   output text, non-stop finish reasons, and explicit truncation flags.
2. Semantic exclusion labels, which come from a classifier and are applied by
   output_id wherever those labels exist. The current classifier file is
   historically named for moral refusals because that is where the issue first
   showed up, but this module does not special-case moral conditions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utility_behavior_gap.paths import ANALYSIS


SEMANTIC_CLASSIFICATIONS_PATH = ANALYSIS / "canonical_highn_moral_refusal_classifications.jsonl"
CLEAN_LABEL = "clean"
FEATURE_OUTPUT_CATALOG = ANALYSIS / "final_text_analysis_by_output.csv"
MECHANICAL_INVALID_FLAGS = (
    "missing_output",
    "generation_success_false",
    "finish_reason_missing",
    "non_stop_finish",
    "explicit_length_truncation",
    "empty_output",
)


def _as_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def mechanical_invalid_output_mask(
    df: pd.DataFrame,
    *,
    flags: tuple[str, ...] = MECHANICAL_INVALID_FLAGS,
) -> pd.Series:
    """Return rows that fail mechanical output-validity checks."""

    mask = pd.Series(False, index=df.index)
    for col in flags:
        if col in df.columns:
            mask |= pd.to_numeric(df[col], errors="coerce").fillna(0).ne(0)
    return mask


def filter_valid_output_catalog(
    df: pd.DataFrame,
    *,
    flags: tuple[str, ...] = MECHANICAL_INVALID_FLAGS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop mechanically invalid generated outputs before feature analyses."""

    invalid = mechanical_invalid_output_mask(df, flags=flags)
    report = {
        "valid_output_filter_applied": True,
        "invalid_flags": list(flags),
        "rows_before_filter": int(len(df)),
        "rows_after_filter": int((~invalid).sum()),
        "rows_dropped": int(invalid.sum()),
    }
    return df[~invalid].copy(), report


def invalid_output_ids_from_catalog(
    path: Path = FEATURE_OUTPUT_CATALOG,
    *,
    flags: tuple[str, ...] = MECHANICAL_INVALID_FLAGS,
) -> set[str]:
    if not path.exists():
        return set()
    cols = {"output_id", *flags}
    catalog = pd.read_csv(path, usecols=lambda col: col in cols, low_memory=False)
    if "output_id" not in catalog.columns:
        return set()
    invalid = mechanical_invalid_output_mask(catalog, flags=flags)
    return set(catalog.loc[invalid, "output_id"].dropna().astype(str))


def filter_valid_pair_rows(
    df: pd.DataFrame,
    *,
    side_prefixes: tuple[str, ...] = ("high", "low", "left", "right"),
    flags: tuple[str, ...] = MECHANICAL_INVALID_FLAGS,
    output_catalog_path: Path = FEATURE_OUTPUT_CATALOG,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop paired rows if any side is mechanically invalid."""

    keep = pd.Series(True, index=df.index)
    side_rows_checked = 0
    invalid_side_rows = 0
    used_inline_flags = False
    invalid_ids: set[str] | None = None

    for prefix in side_prefixes:
        output_col = f"{prefix}_output_id"
        inline_cols = [f"{prefix}_{flag}" for flag in flags if f"{prefix}_{flag}" in df.columns]
        if inline_cols:
            used_inline_flags = True
            side_invalid = pd.Series(False, index=df.index)
            for col in inline_cols:
                side_invalid |= pd.to_numeric(df[col], errors="coerce").fillna(0).ne(0)
            side_rows_checked += int(len(df))
            invalid_side_rows += int(side_invalid.sum())
            keep &= ~side_invalid
            continue
        if output_col in df.columns:
            if invalid_ids is None:
                invalid_ids = invalid_output_ids_from_catalog(
                    output_catalog_path,
                    flags=flags,
                )
            side_invalid = _as_text(df[output_col]).isin(invalid_ids)
            side_rows_checked += int(len(df))
            invalid_side_rows += int(side_invalid.sum())
            keep &= ~side_invalid

    report = {
        "valid_output_filter_applied": True,
        "invalid_flags": list(flags),
        "used_inline_flags": used_inline_flags,
        "side_rows_checked": side_rows_checked,
        "invalid_side_rows": invalid_side_rows,
        "pairs_before_filter": int(len(df)),
        "pairs_after_filter": int(keep.sum()),
        "pairs_dropped": int(len(df) - keep.sum()),
    }
    return df[keep].copy(), report


def load_semantic_classifications(
    path: Path = SEMANTIC_CLASSIFICATIONS_PATH,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["output_id", "label"])
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=["output_id", "label"])
    df = pd.DataFrame(rows)
    required = {"output_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Semantic classification file is missing required column(s): {sorted(missing)}"
        )
    conflicts = (
        df.groupby("output_id")["label"]
        .nunique(dropna=False)
        .loc[lambda values: values.gt(1)]
    )
    if not conflicts.empty:
        sample = ", ".join(conflicts.head(10).index.astype(str).tolist())
        raise ValueError(f"Conflicting semantic labels for output_id(s): {sample}")
    return df.drop_duplicates("output_id", keep="last")


def semantic_label_sets(
    path: Path = SEMANTIC_CLASSIFICATIONS_PATH,
) -> tuple[set[str], set[str], int]:
    classifications = load_semantic_classifications(path)
    if classifications.empty:
        return set(), set(), 0
    output_ids = classifications["output_id"].astype(str)
    clean = set(output_ids[classifications["label"].eq(CLEAN_LABEL)])
    nonclean = set(output_ids[~classifications["label"].eq(CLEAN_LABEL)])
    return clean, nonclean, int(len(classifications))


def filter_semantic_excluded_output_catalog(
    df: pd.DataFrame,
    *,
    output_col: str = "output_id",
    classification_path: Path = SEMANTIC_CLASSIFICATIONS_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop rows whose output_id has a non-clean semantic exclusion label.

    Unclassified rows are kept and reported as unclassified, not treated as
    classifier-clean.
    """

    if output_col not in df.columns:
        return df.copy(), {
            "semantic_exclusion_filter_applied": False,
            "reason": "missing output_id",
        }

    _clean_ids, nonclean_ids, classified_count = semantic_label_sets(classification_path)
    output_ids = _as_text(df[output_col])
    nonclean = output_ids.isin(nonclean_ids)
    classified = output_ids.isin(_clean_ids | nonclean_ids)
    filtered = df[~nonclean].copy()
    report = {
        "semantic_exclusion_filter_applied": True,
        "classification_path": str(classification_path),
        "classified_outputs_in_file": classified_count,
        "rows_before_filter": int(len(df)),
        "classified_rows_seen": int(classified.sum()),
        "nonclean_classified_rows_dropped": int(nonclean.sum()),
        "unclassified_rows_seen": int((~classified).sum()),
        "rows_after_filter": int(len(filtered)),
    }
    return filtered, report


def filter_semantic_excluded_pair_rows(
    df: pd.DataFrame,
    *,
    side_prefixes: tuple[str, ...] = ("high", "low", "left", "right"),
    classification_path: Path = SEMANTIC_CLASSIFICATIONS_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop paired rows if any side has a non-clean semantic exclusion label.

    The filter is condition-agnostic: it only looks at side output_id columns.
    """

    clean_ids, nonclean_ids, classified_count = semantic_label_sets(classification_path)
    classified_ids = clean_ids | nonclean_ids
    keep = pd.Series(True, index=df.index)
    side_rows_checked = 0
    classified_side_rows = 0
    nonclean_side_rows = 0

    for prefix in side_prefixes:
        output_col = f"{prefix}_output_id"
        if output_col not in df.columns:
            continue
        output_ids = _as_text(df[output_col])
        classified = output_ids.isin(classified_ids)
        nonclean = output_ids.isin(nonclean_ids)
        side_rows_checked += int(len(df))
        classified_side_rows += int(classified.sum())
        nonclean_side_rows += int(nonclean.sum())
        keep &= ~nonclean

    filtered = df[keep].copy()
    report = {
        "semantic_exclusion_filter_applied": True,
        "classification_path": str(classification_path),
        "classified_outputs_in_file": classified_count,
        "side_rows_checked": side_rows_checked,
        "classified_side_rows_seen": classified_side_rows,
        "nonclean_classified_side_rows_dropped": nonclean_side_rows,
        "unclassified_side_rows_seen": side_rows_checked - classified_side_rows,
        "pairs_before_filter": int(len(df)),
        "pairs_after_filter": int(len(filtered)),
        "pairs_dropped": int(len(df) - len(filtered)),
    }
    return filtered, report
