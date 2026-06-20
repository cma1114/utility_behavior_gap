#!/usr/bin/env python3
"""Combine generic and LLM-rubric feature deltas into a paper table.

This is the reusable version of the direct-instruction feature appendix table.
It assumes the generic-feature file was produced by
``analyze_standard_feature_deltas.py`` and the rubric directory was produced by
``run_task_rubric_feature_coding.py`` followed by
``analyze_task_rubric_feature_coding.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utility_behavior_gap.constants import PLOT_TASK_ORDER, TASK_BY_LABEL
from utility_behavior_gap.feature_specs import (
    FEATURE_SPEC,
    generic_feature_info,
    rubric_dimension_info,
    rubric_dimension_labels,
    standard_generic_feature_ids,
    task_rubric_display_digits,
)
from utility_behavior_gap.output_exclusions import (
    filter_semantic_excluded_pair_rows,
    filter_valid_pair_rows,
)


GENERIC_FEATURE_INFO = generic_feature_info()
STANDARD_GENERIC_FEATURES = standard_generic_feature_ids()
RUBRIC_DIMENSION_INFO = rubric_dimension_info()
RUBRIC_LABELS = rubric_dimension_labels()
RUBRIC_DISPLAY_DIGITS = task_rubric_display_digits()
TASK_ORDER = [TASK_BY_LABEL[label] for label in PLOT_TASK_ORDER]
TASK_LABEL_BY_KEY = {value: key for key, value in TASK_BY_LABEL.items()}
PANEL_ASSOCIATION_COLUMN = "Panel preference (r)"


def fmt(value: float, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    rounded = round(float(value), digits)
    if rounded == 0:
        rounded = 0.0
    if digits == 0:
        return f"{rounded:,.0f}"
    return f"{rounded:.{digits}f}"


def fmt_pct(value: float) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{100 * float(value):.1f}%"


def fmt_ci(lo: float, hi: float, digits: int = 3) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"[{fmt(float(lo), digits)}, {fmt(float(hi), digits)}]"


def fmt_r(value: float, n: int) -> str:
    if value is None or pd.isna(value) or n < 3:
        return ""
    rounded = round(float(value), 2)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:+.2f} (n={int(n):,})"


def display_digits(feature_id: str, feature_family: str, values: list[float] | None = None) -> int:
    info = GENERIC_FEATURE_INFO.get(feature_id, {})
    small_threshold = info.get("small_abs_threshold")
    if small_threshold is not None:
        vals = [
            abs(float(value))
            for value in (values or [])
            if value is not None and not pd.isna(value)
        ]
        if vals and max(vals) < float(small_threshold):
            return int(info.get("small_abs_display_digits", info.get("display_digits", 2)))
    if feature_family == "Generic text feature":
        return int(info.get("display_digits", 2))
    if feature_family == "LLM rubric marker":
        return RUBRIC_DISPLAY_DIGITS
    return 2


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


def latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def source_sort(value: str) -> int:
    return {"Generic text feature": 0, "LLM rubric marker": 1}.get(value, 9)


def task_sort(value: str) -> int:
    task = TASK_BY_LABEL.get(value, value)
    return TASK_ORDER.index(task) if task in TASK_ORDER else 99


def direction_label(delta: float, clear: bool, left_label: str, right_label: str) -> str:
    if not clear:
        return "No clear difference"
    if delta > 0:
        return f"Higher for {left_label}"
    if delta < 0:
        return f"Higher for {right_label}"
    return "No clear difference"


def col(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"missing expected column; tried {candidates}")


def clean_repeated_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    repeated = pd.Series(False, index=df.index)
    for name in ("actor", "task", "pair_uid", "source_moral_output_id"):
        if name in df.columns:
            repeated |= df[name].fillna("").astype(str).eq(name)
    return df[~repeated].copy()


def task_label(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return TASK_LABEL_BY_KEY.get(text, text)


def infer_bridge_output_cols(df: pd.DataFrame) -> tuple[str, str]:
    left_candidates = [
        "source_highlow_output_id",
        "source_moral_output_id",
        "source_amount_output_id",
        "source_side_output_id",
        "source_old_output_id",
    ]
    right_candidates = [
        "source_r0_output_id",
        "source_framed_empty_output_id",
        "source_empty_output_id",
        "source_neutral_output_id",
        "source_baseline_output_id",
    ]
    left = next((name for name in left_candidates if name in df.columns), None)
    right = next((name for name in right_candidates if name in df.columns), None)
    if left is None or right is None:
        raise KeyError(
            "Could not infer bridge output id columns; pass --bridge-left-output-col "
            "and --bridge-right-output-col."
        )
    return left, right


def infer_bridge_score(df: pd.DataFrame, score_col: str | None) -> pd.Series:
    if score_col:
        if score_col not in df.columns:
            raise KeyError(f"bridge score column not found: {score_col}")
        return pd.to_numeric(df[score_col], errors="coerce")

    score_candidates = [
        "high_vs_r0_score",
        "high_vs_framed_empty_score",
        "effort_boost_score",
        "amount_high_net_score",
        "moral_bad_net_score",
        "side_net_score",
    ]
    for name in score_candidates:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    if "r0_minus_moral_bad_net_score" in df.columns:
        return -pd.to_numeric(df["r0_minus_moral_bad_net_score"], errors="coerce")

    if "panel_winner_condition" not in df.columns:
        raise KeyError("No bridge score column or panel_winner_condition found.")

    winners = df["panel_winner_condition"].fillna("").astype(str)
    left_winners = {
        "hl_high",
        "utility_high",
        "amount_high",
        "moral_bad",
        "moral_low",
        "direct_high",
        "framed_strong",
    }
    right_winners = {
        "r0",
        "framed_empty",
        "framed_neutral",
        "amount_low",
        "moral_good",
        "moral_high",
        "direct_low",
    }
    score = pd.Series(pd.NA, index=df.index, dtype="Float64")
    score.loc[winners.isin(left_winners)] = 1
    score.loc[winners.isin(right_winners)] = -1
    score.loc[winners.eq("tie")] = 0
    return score.astype(float)


def load_bridge_scores(
    path: Path | None,
    *,
    score_col: str | None = None,
    left_output_col: str | None = None,
    right_output_col: str | None = None,
) -> pd.DataFrame | None:
    if path is None:
        return None
    df = clean_repeated_header_rows(pd.read_csv(path, low_memory=False))
    left_col, right_col = (
        (left_output_col, right_output_col)
        if left_output_col and right_output_col
        else infer_bridge_output_cols(df)
    )
    score = infer_bridge_score(df, score_col)
    out = pd.DataFrame(
        {
            "left_output_id": df[left_col].fillna("").astype(str),
            "right_output_id": df[right_col].fillna("").astype(str),
            "panel_score_left_minus_right": score,
        }
    )
    if "task_label" in df.columns:
        out["task_label"] = df["task_label"].map(task_label)
    elif "task" in df.columns:
        out["task_label"] = df["task"].map(task_label)
    out = out[
        out["left_output_id"].ne("")
        & out["right_output_id"].ne("")
        & out["panel_score_left_minus_right"].notna()
    ].copy()
    group_cols = ["left_output_id", "right_output_id"]
    agg = {"panel_score_left_minus_right": "mean"}
    if "task_label" in out.columns:
        agg["task_label"] = "first"
    return out.groupby(group_cols, as_index=False).agg(agg)


def panel_correlation(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    work = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()
    if len(work) < 3 or work["x"].nunique() < 2 or work["y"].nunique() < 2:
        return float("nan"), int(len(work))
    return float(work["x"].corr(work["y"])), int(len(work))


def generic_delta_col(feature_id: str, pairs: pd.DataFrame) -> str | None:
    candidates = [
        f"delta_{feature_id}",
        f"left_minus_right_{feature_id}",
        f"{feature_id}_left_minus_right",
    ]
    return next((name for name in candidates if name in pairs.columns), None)


def load_generic_panel_associations(
    path: Path | None,
    *,
    comparison: str | None,
    bridge_scores: pd.DataFrame | None,
) -> dict[tuple[str, str], tuple[float, int]]:
    if path is None:
        return {}
    pairs = clean_repeated_header_rows(pd.read_csv(path, low_memory=False))
    if comparison and "contrast" in pairs.columns:
        pairs = pairs[pairs["contrast"].fillna("").astype(str).eq(comparison)].copy()
    pairs, _valid_report = filter_valid_pair_rows(pairs, side_prefixes=("high", "low"))
    pairs, _semantic_report = filter_semantic_excluded_pair_rows(
        pairs, side_prefixes=("high", "low")
    )

    if bridge_scores is not None:
        if "high_output_id" not in pairs.columns or "low_output_id" not in pairs.columns:
            return {}
        pairs = pairs.merge(
            bridge_scores,
            left_on=["high_output_id", "low_output_id"],
            right_on=["left_output_id", "right_output_id"],
            how="inner",
        )
        score_col = "panel_score_left_minus_right"
    elif "effect_score_high_minus_low" in pairs.columns:
        score_col = "effect_score_high_minus_low"
    else:
        return {}

    if "task_label" not in pairs.columns and "task" in pairs.columns:
        pairs["task_label"] = pairs["task"].map(task_label)
    out: dict[tuple[str, str], tuple[float, int]] = {}
    for feature_id in STANDARD_GENERIC_FEATURES:
        delta_col = generic_delta_col(feature_id, pairs)
        if delta_col is None:
            continue
        for task, sub in pairs.groupby("task_label", dropna=True):
            out[(str(task), feature_id)] = panel_correlation(sub[delta_col], sub[score_col])
    return out


def load_rubric_panel_associations(
    run_dir: Path,
    *,
    bridge_scores: pd.DataFrame | None,
) -> dict[tuple[str, str], tuple[float, int]]:
    flat_path = run_dir / "rubric_feature_analysis_flat_codes.csv"
    sample_path = run_dir / "rubric_feature_sample.csv"
    if not flat_path.exists() or not sample_path.exists():
        return {}

    flat = clean_repeated_header_rows(pd.read_csv(flat_path, low_memory=False))
    sample = clean_repeated_header_rows(pd.read_csv(sample_path, low_memory=False))
    sample, _valid_report = filter_valid_pair_rows(sample, side_prefixes=("left", "right"))
    sample, _semantic_report = filter_semantic_excluded_pair_rows(
        sample, side_prefixes=("left", "right")
    )

    if "effect_score_left_minus_right" in sample.columns:
        sample["panel_score_left_minus_right"] = pd.to_numeric(
            sample["effect_score_left_minus_right"], errors="coerce"
        )
    if (
        sample.get("panel_score_left_minus_right", pd.Series(dtype=float)).notna().sum() == 0
        and bridge_scores is not None
    ):
        sample = sample.drop(columns=["panel_score_left_minus_right"], errors="ignore").merge(
            bridge_scores,
            on=["left_output_id", "right_output_id"],
            how="inner",
        )

    if "panel_score_left_minus_right" not in sample.columns:
        return {}

    score = sample[
        ["coding_pair_uid", "panel_score_left_minus_right"]
    ].dropna(subset=["panel_score_left_minus_right"])
    merged = flat.merge(score, on="coding_pair_uid", how="inner")
    out: dict[tuple[str, str], tuple[float, int]] = {}
    for (task, dimension), sub in merged.groupby(["task_label", "dimension"], dropna=True):
        out[(str(task), str(dimension))] = panel_correlation(
            sub["left_minus_right_score"], sub["panel_score_left_minus_right"]
        )
    return out


def load_generic(
    path: Path,
    *,
    left_key: str,
    right_key: str,
    left_label: str,
    right_label: str,
    sample_label: str,
    panel_associations: dict[tuple[str, str], tuple[float, int]] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["feature"].isin(STANDARD_GENERIC_FEATURES)].copy()
    delta_col = col(
        df,
        [
            f"mean_delta_{left_key}_minus_{right_key}",
            "mean_delta_high_minus_low",
            "mean_delta_strong_minus_neutral",
        ],
    )
    left_pct_col = col(
        df,
        [f"pct_pairs_{left_key}_greater", "pct_pairs_high_greater", "pct_pairs_strong_greater"],
    )
    right_pct_col = col(
        df,
        [f"pct_pairs_{right_key}_greater", "pct_pairs_low_greater", "pct_pairs_neutral_greater"],
    )
    rows = []
    for row in df.to_dict(orient="records"):
        delta = float(row[delta_col])
        clear = bool(row["ci_excludes_zero"])
        association_r, association_n = (panel_associations or {}).get(
            (str(row["task_label"]), str(row["feature"])),
            (float("nan"), 0),
        )
        digits = display_digits(row["feature"], "Generic text feature", [delta])
        ci_digits = display_digits(
            row["feature"], "Generic text feature", [row["ci_low"], row["ci_high"]]
        )
        scale = f"Raw feature units; positive means {left_label} output has more/higher feature value."
        if row["feature"] == "quantitative_detail":
            scale = f"Within-task z-score units; positive means {left_label} output has more quantitative detail."
        rows.append(
            {
                "Task": row["task_label"],
                "Feature family": "Generic text feature",
                "Feature": str(
                    GENERIC_FEATURE_INFO.get(row["feature"], {}).get("label", row["feature"])
                ),
                "Feature id": row["feature"],
                "Sample": sample_label,
                "N pairs": int(row["n_pairs"]),
                "N clusters": int(row["n_cells"]),
                "Estimate": delta,
                "95% CI low": float(row["ci_low"]),
                "95% CI high": float(row["ci_high"]),
                "Estimate display": fmt(delta, digits),
                "95% CI": fmt_ci(row["ci_low"], row["ci_high"], ci_digits),
                "Clear difference": clear,
                "Direction": direction_label(delta, clear, left_label, right_label),
                "Panel association r": association_r,
                "Panel association n": association_n,
                PANEL_ASSOCIATION_COLUMN: fmt_r(association_r, association_n),
                f"% {left_label} higher/better": fmt_pct(row[left_pct_col]),
                f"% {right_label} higher/better": fmt_pct(row[right_pct_col]),
                "% equal/tie/NA": fmt_pct(row["pct_pairs_equal"]),
                "Scale": scale,
                "Definition": str(
                    GENERIC_FEATURE_INFO.get(row["feature"], {}).get("definition", row["definition"])
                ),
                "Source file": str(path),
            }
        )
    return pd.DataFrame(rows)


def load_rubric(
    run_dir: Path,
    *,
    left_label: str,
    right_label: str,
    sample_label: str,
    panel_associations: dict[tuple[str, str], tuple[float, int]] | None = None,
) -> pd.DataFrame:
    path = run_dir / "rubric_feature_analysis_by_task_dimension.csv"
    df = pd.read_csv(path)
    rows = []
    for row in df.to_dict(orient="records"):
        delta = float(row["mean_left_minus_right_equal_actor"])
        clear = bool(row["ci_excludes_zero"])
        association_r, association_n = (panel_associations or {}).get(
            (str(row["task_label"]), str(row["dimension"])),
            (float("nan"), 0),
        )
        digits = display_digits(row["dimension"], "LLM rubric marker", [delta])
        ci_digits = display_digits(
            row["dimension"], "LLM rubric marker", [row["ci_low"], row["ci_high"]]
        )
        rows.append(
            {
                "Task": row["task_label"],
                "Feature family": "LLM rubric marker",
                "Feature": RUBRIC_LABELS.get(row["dimension"], row["dimension"]),
                "Feature id": row["dimension"],
                "Sample": sample_label,
                "N pairs": int(row["n_pairs"]),
                "N clusters": int(row["n_actors"]),
                "Estimate": delta,
                "95% CI low": float(row["ci_low"]),
                "95% CI high": float(row["ci_high"]),
                "Estimate display": fmt(delta, digits),
                "95% CI": fmt_ci(row["ci_low"], row["ci_high"], ci_digits),
                "Clear difference": clear,
                "Direction": direction_label(delta, clear, left_label, right_label),
                "Panel association r": association_r,
                "Panel association n": association_n,
                PANEL_ASSOCIATION_COLUMN: fmt_r(association_r, association_n),
                f"% {left_label} higher/better": fmt_pct(row["pct_left_better"]),
                f"% {right_label} higher/better": fmt_pct(row["pct_right_better"]),
                "% equal/tie/NA": fmt_pct(row["pct_tie_or_not_applicable"]),
                "Scale": (
                    "Comparative rubric score in [-1, 1]; positive means rubric coder "
                    f"favored {left_label} output."
                ),
                "Definition": str(
                    RUBRIC_DIMENSION_INFO.get(row["dimension"], {}).get(
                        "definition", row["description"]
                    )
                ),
                "Source file": str(path),
            }
        )
    return pd.DataFrame(rows)


def display_table(df: pd.DataFrame, left_label: str, right_label: str) -> pd.DataFrame:
    return df[
        [
            "Feature",
            "Estimate display",
            "95% CI",
            PANEL_ASSOCIATION_COLUMN,
            f"% {left_label} higher/better",
            f"% {right_label} higher/better",
            "Definition",
        ]
    ].rename(columns={"Estimate display": "Delta"})


def markdown_task_sections(df: pd.DataFrame, left_label: str, right_label: str) -> str:
    sections: list[str] = []
    for task in PLOT_TASK_ORDER:
        task_rows = df[df["Task"].eq(task)]
        if task_rows.empty:
            continue
        sections.extend(
            [f"## {task}", "", markdown_table(display_table(task_rows, left_label, right_label)), ""]
        )
    return "\n".join(sections).rstrip() + "\n"


def write_latex_task_table(
    df: pd.DataFrame, path: Path, *, left_label: str, right_label: str
) -> None:
    columns = [
        "Feature",
        "Delta",
        "95% CI",
        PANEL_ASSOCIATION_COLUMN,
        f"% {left_label} higher/better",
        f"% {right_label} higher/better",
        "Definition",
    ]
    lines = [
        r"\begin{longtable}{lllllll}",
        r"\toprule",
        " & ".join(latex_escape(col) for col in columns) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(latex_escape(col) for col in columns) + r" \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{7}{r}{Continued on next page} \\",
        r"\midrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for task in PLOT_TASK_ORDER:
        task_rows = df[df["Task"].eq(task)]
        if task_rows.empty:
            continue
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textbf{{{latex_escape(task)}}}}} \\")
        for row in display_table(task_rows, left_label, right_label).to_dict(orient="records"):
            lines.append(" & ".join(latex_escape(row[col]) for col in columns) + r" \\")
    lines.append(r"\end{longtable}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    all_rows: pd.DataFrame,
    clear_rows: pd.DataFrame,
    path: Path,
    *,
    title: str,
    comparison: str,
    left_label: str,
    right_label: str,
    out_prefix: Path,
) -> None:
    generic_labels = [
        str(GENERIC_FEATURE_INFO[feature_id].get("label", feature_id))
        for feature_id in STANDARD_GENERIC_FEATURES
    ]
    lines = [
        f"# {title}",
        "",
        f"Comparison: {comparison}.",
        "",
        "The table combines the available feature families:",
        "",
        "- Generic text features are computed on the full paired dataset for this contrast.",
        "- LLM rubric markers are task-specific comparative codes from the random actor-balanced A/B-coded sample when that coding exists.",
        "",
        f"Editable feature labels, definitions, and display rounding come from `{FEATURE_SPEC}`.",
        "",
        "The generic text rows use the standard paper-facing feature set: "
        + ", ".join(generic_labels)
        + ". Quantitative detail is `z(numbers + percentages)`, standardized within task before paired differencing.",
        "",
        f"Positive deltas mean the {left_label} output had more of the generic feature or was favored by the rubric coder on the task-specific marker. Confidence intervals are over actor/task cells for the generic full-sample rows and over actors within task for the rubric rows.",
        "",
        f"`{PANEL_ASSOCIATION_COLUMN}` is the within-comparison correlation between left-minus-right feature score and left-minus-right panel score. Positive values mean the judging panel tended to prefer the side with more of that feature; negative values mean the panel tended to prefer the side with less.",
        "",
        f"- total rows: `{len(all_rows)}`",
        f"- clear-difference rows: `{len(clear_rows)}`",
        "",
        "## Clear Differences Only",
        "",
    ]
    for task in PLOT_TASK_ORDER:
        task_rows = clear_rows[clear_rows["Task"].eq(task)]
        if task_rows.empty:
            continue
        lines.extend(
            [f"### {task}", "", markdown_table(display_table(task_rows, left_label, right_label)), ""]
        )
    lines.extend(
        [
            "## Output Files",
            "",
            f"- full CSV: `{out_prefix}_all.csv`",
            f"- clear-differences CSV: `{out_prefix}_clear_differences.csv`",
            f"- clear-differences Markdown: `{out_prefix}_clear_differences.md`",
            f"- LaTeX longtable: `{out_prefix}_clear_differences.tex`",
            "",
            "The full CSV includes all rows, including features whose confidence interval includes zero.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic-by-task", type=Path, required=True)
    parser.add_argument("--rubric-run-dir", type=Path)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--left-key", default="high")
    parser.add_argument("--right-key", default="low")
    parser.add_argument("--left-label", default="high")
    parser.add_argument("--right-label", default="low")
    parser.add_argument("--generic-sample-label", default="Full paired dataset")
    parser.add_argument("--rubric-sample-label", default="Random actor-balanced A/B-coded sample")
    parser.add_argument(
        "--generic-pairs",
        type=Path,
        help="Pair-level generic feature CSV used to compute panel-preference correlations.",
    )
    parser.add_argument(
        "--generic-pairs-comparison",
        help="Optional contrast value to filter --generic-pairs before computing correlations.",
    )
    parser.add_argument(
        "--bridge-outcomes",
        type=Path,
        help="Bridge pair-outcome CSV for arm-match comparisons whose feature pair file has no panel score.",
    )
    parser.add_argument("--bridge-score-col")
    parser.add_argument("--bridge-left-output-col")
    parser.add_argument("--bridge-right-output-col")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bridge_scores = load_bridge_scores(
        args.bridge_outcomes,
        score_col=args.bridge_score_col,
        left_output_col=args.bridge_left_output_col,
        right_output_col=args.bridge_right_output_col,
    )
    generic_panel_associations = load_generic_panel_associations(
        args.generic_pairs,
        comparison=args.generic_pairs_comparison,
        bridge_scores=bridge_scores,
    )
    rubric_panel_associations = (
        load_rubric_panel_associations(
            args.rubric_run_dir,
            bridge_scores=bridge_scores,
        )
        if args.rubric_run_dir
        else {}
    )
    generic = load_generic(
        args.generic_by_task,
        left_key=args.left_key,
        right_key=args.right_key,
        left_label=args.left_label,
        right_label=args.right_label,
        sample_label=args.generic_sample_label,
        panel_associations=generic_panel_associations,
    )
    rows = [generic]
    if args.rubric_run_dir:
        rows.append(
            load_rubric(
                args.rubric_run_dir,
                left_label=args.left_label,
                right_label=args.right_label,
                sample_label=args.rubric_sample_label,
                panel_associations=rubric_panel_associations,
            )
        )
    all_rows = pd.concat(rows, ignore_index=True)
    all_rows["_task_sort"] = all_rows["Task"].map(task_sort)
    all_rows["_source_sort"] = all_rows["Feature family"].map(source_sort)
    all_rows["_abs_estimate"] = all_rows["Estimate"].abs()
    all_rows = all_rows.sort_values(
        ["_task_sort", "_source_sort", "Clear difference", "_abs_estimate", "Feature"],
        ascending=[True, True, False, False, True],
    ).drop(columns=["_task_sort", "_source_sort", "_abs_estimate"])

    clear_rows = all_rows[all_rows["Clear difference"]].copy()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    all_path = args.out_prefix.with_name(args.out_prefix.name + "_all.csv")
    clear_path = args.out_prefix.with_name(args.out_prefix.name + "_clear_differences.csv")
    md_path = args.out_prefix.with_name(args.out_prefix.name + "_clear_differences.md")
    tex_path = args.out_prefix.with_name(args.out_prefix.name + "_clear_differences.tex")
    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.md")

    all_rows.to_csv(all_path, index=False)
    clear_rows.to_csv(clear_path, index=False)
    md_path.write_text(
        markdown_task_sections(clear_rows, args.left_label, args.right_label),
        encoding="utf-8",
    )
    write_latex_task_table(clear_rows, tex_path, left_label=args.left_label, right_label=args.right_label)
    write_summary(
        all_rows,
        clear_rows,
        summary_path,
        title=args.title,
        comparison=args.comparison,
        left_label=args.left_label,
        right_label=args.right_label,
        out_prefix=args.out_prefix,
    )

    print(f"full rows: {len(all_rows)}")
    print(f"clear-difference rows: {len(clear_rows)}")
    print(f"summary: {summary_path}")
    print(f"full csv: {all_path}")
    print(f"clear csv: {clear_path}")
    print(f"markdown: {md_path}")
    print(f"tex: {tex_path}")


if __name__ == "__main__":
    main()
