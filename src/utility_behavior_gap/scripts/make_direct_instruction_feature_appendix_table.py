#!/usr/bin/env python3
"""Combine direct-instruction generic and LLM-rubric feature deltas.

The output is a paper-appendix-ready long table. It keeps the two feature
families separate because their estimates are on different scales:

- Generic text features: raw paired feature deltas from the full direct-
  instruction dataset.
- LLM task-rubric markers: comparative left-minus-right rubric scores from the
  random A/B-coded sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utility_behavior_gap.constants import PLOT_TASK_ORDER, TASK_BY_LABEL, TASK_LABEL
from utility_behavior_gap.feature_specs import (
    FEATURE_SPEC,
    generic_feature_info,
    rubric_dimension_info,
    rubric_dimension_labels,
    standard_generic_feature_ids,
    task_rubric_display_digits,
)
from utility_behavior_gap.paths import ANALYSIS


GENERIC_BY_TASK = ANALYSIS / "direct_instruction_feature_deltas_by_task.csv"
DEFAULT_RUBRIC_RUN_DIR = (
    ANALYSIS
    / "task_rubric_feature_coding"
    / "direct_instruction__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash"
)
OUT_PREFIX = ANALYSIS / "direct_instruction_combined_feature_appendix"

GENERIC_FEATURE_INFO = generic_feature_info()
STANDARD_GENERIC_FEATURES = standard_generic_feature_ids()
RUBRIC_DIMENSION_INFO = rubric_dimension_info()
RUBRIC_LABELS = rubric_dimension_labels()
RUBRIC_DISPLAY_DIGITS = task_rubric_display_digits()

TASK_ORDER = [TASK_BY_LABEL[label] for label in PLOT_TASK_ORDER]


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


def markdown_task_sections(df: pd.DataFrame) -> str:
    sections: list[str] = []
    for task in PLOT_TASK_ORDER:
        task_rows = df[df["Task"].eq(task)]
        if task_rows.empty:
            continue
        sections.extend([f"## {task}", "", markdown_table(display_table(task_rows)), ""])
    return "\n".join(sections).rstrip() + "\n"


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


def write_latex_task_table(df: pd.DataFrame, path: Path) -> None:
    columns = [
        "Feature",
        "Delta",
        "95% CI",
        "% strong higher/better",
        "% neutral higher/better",
        "Definition",
    ]
    lines = [
        r"\begin{longtable}{llllll}",
        r"\toprule",
        " & ".join(latex_escape(col) for col in columns) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(latex_escape(col) for col in columns) + r" \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{6}{r}{Continued on next page} \\",
        r"\midrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for task in PLOT_TASK_ORDER:
        task_rows = df[df["Task"].eq(task)]
        if task_rows.empty:
            continue
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{latex_escape(task)}}}}} \\")
        for row in display_table(task_rows).to_dict(orient="records"):
            lines.append(" & ".join(latex_escape(row[col]) for col in columns) + r" \\")
    lines.append(r"\end{longtable}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_sort(value: str) -> int:
    return {"Generic text feature": 0, "LLM rubric marker": 1}.get(value, 9)


def task_sort(value: str) -> int:
    task = TASK_BY_LABEL.get(value, value)
    return TASK_ORDER.index(task) if task in TASK_ORDER else 99


def direction_label(delta: float, clear: bool) -> str:
    if not clear:
        return "No clear difference"
    if delta > 0:
        return "Higher under strong prompt"
    if delta < 0:
        return "Higher under neutral prompt"
    return "No clear difference"


def load_generic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["feature"].isin(STANDARD_GENERIC_FEATURES)].copy()
    rows = []
    for row in df.to_dict(orient="records"):
        delta = float(row["mean_delta_strong_minus_neutral"])
        clear = bool(row["ci_excludes_zero"])
        digits = display_digits(row["feature"], "Generic text feature", [delta])
        ci_digits = display_digits(
            row["feature"], "Generic text feature", [row["ci_low"], row["ci_high"]]
        )
        scale = "Raw feature units; positive means strong prompt output has more/higher feature value."
        if row["feature"] == "quantitative_detail":
            scale = "Within-task z-score units; positive means strong prompt output has more quantitative detail."
        rows.append(
            {
                "Task": row["task_label"],
                "Feature family": "Generic text feature",
                "Feature": str(GENERIC_FEATURE_INFO.get(row["feature"], {}).get("label", row["feature"])),
                "Feature id": row["feature"],
                "Sample": "Full paired direct-instruction dataset",
                "N pairs": int(row["n_pairs"]),
                "N clusters": int(row["n_cells"]),
                "Estimate": delta,
                "95% CI low": float(row["ci_low"]),
                "95% CI high": float(row["ci_high"]),
                "Estimate display": fmt(delta, digits),
                "95% CI": fmt_ci(row["ci_low"], row["ci_high"], ci_digits),
                "Clear difference": clear,
                "Direction": direction_label(delta, clear),
                "% strong higher/better": fmt_pct(row["pct_pairs_strong_greater"]),
                "% neutral higher/better": fmt_pct(row["pct_pairs_neutral_greater"]),
                "% equal/tie/NA": fmt_pct(row["pct_pairs_equal"]),
                "Scale": scale,
                "Definition": str(GENERIC_FEATURE_INFO.get(row["feature"], {}).get("definition", row["definition"])),
                "Source file": str(path),
            }
        )
    return pd.DataFrame(rows)


def load_rubric(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "rubric_feature_analysis_by_task_dimension.csv"
    df = pd.read_csv(path)
    rows = []
    for row in df.to_dict(orient="records"):
        delta = float(row["mean_left_minus_right_equal_actor"])
        clear = bool(row["ci_excludes_zero"])
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
                "Sample": "Random actor-balanced A/B-coded sample",
                "N pairs": int(row["n_pairs"]),
                "N clusters": int(row["n_actors"]),
                "Estimate": delta,
                "95% CI low": float(row["ci_low"]),
                "95% CI high": float(row["ci_high"]),
                "Estimate display": fmt(delta, digits),
                "95% CI": fmt_ci(row["ci_low"], row["ci_high"], ci_digits),
                "Clear difference": clear,
                "Direction": direction_label(delta, clear),
                "% strong higher/better": fmt_pct(row["pct_left_better"]),
                "% neutral higher/better": fmt_pct(row["pct_right_better"]),
                "% equal/tie/NA": fmt_pct(row["pct_tie_or_not_applicable"]),
                "Scale": "Comparative rubric score in [-1, 1]; positive means rubric coder favored strong prompt output.",
                "Definition": str(RUBRIC_DIMENSION_INFO.get(row["dimension"], {}).get("definition", row["description"])),
                "Source file": str(path),
            }
        )
    return pd.DataFrame(rows)


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    return display[
        [
            "Feature",
            "Estimate display",
            "95% CI",
            "% strong higher/better",
            "% neutral higher/better",
            "Definition",
        ]
    ].rename(columns={"Estimate display": "Delta"})


def write_markdown(all_rows: pd.DataFrame, clear_rows: pd.DataFrame, path: Path) -> None:
    generic_labels = [
        str(GENERIC_FEATURE_INFO[feature_id].get("label", feature_id))
        for feature_id in STANDARD_GENERIC_FEATURES
    ]
    lines = [
        "# Direct-Instruction Feature Appendix Table",
        "",
        "Comparison: finalized strong/exhortative direct-instruction prompt versus framed neutral.",
        "",
        "The table combines two feature families:",
        "",
        "- Generic text features are computed on the full paired direct-instruction dataset.",
        "- LLM rubric markers are task-specific comparative codes from the random actor-balanced A/B-coded sample.",
        "",
        f"Editable feature labels, definitions, and display rounding come from `{FEATURE_SPEC}`.",
        "",
        "The generic text rows use the standard paper-facing feature set: "
        + ", ".join(generic_labels)
        + ". Quantitative detail is `z(numbers + percentages)`, standardized within task before paired differencing.",
        "",
        "Positive deltas mean the strong-prompt output had more of the generic feature or was favored by the rubric coder on the task-specific marker. Confidence intervals are over actor/task cells for the generic full-sample rows and over actors within task for the rubric rows.",
        "",
        "## Clear Differences Only",
        "",
    ]
    for task in PLOT_TASK_ORDER:
        task_rows = clear_rows[clear_rows["Task"].eq(task)]
        if task_rows.empty:
            continue
        lines.extend([f"### {task}", "", markdown_table(display_table(task_rows)), ""])
    lines.extend(
        [
            "## Output Files",
            "",
            f"- full CSV: `{OUT_PREFIX}_all.csv`",
            f"- clear-differences CSV: `{OUT_PREFIX}_clear_differences.csv`",
            f"- clear-differences Markdown: `{OUT_PREFIX}_clear_differences.md`",
            f"- LaTeX longtable: `{OUT_PREFIX}_clear_differences.tex`",
            "",
            "The full CSV includes all rows, including features whose confidence interval includes zero.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic-by-task", type=Path, default=GENERIC_BY_TASK)
    parser.add_argument("--rubric-run-dir", type=Path, default=DEFAULT_RUBRIC_RUN_DIR)
    parser.add_argument("--out-prefix", type=Path, default=OUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generic = load_generic(args.generic_by_task)
    rubric = load_rubric(args.rubric_run_dir)
    all_rows = pd.concat([generic, rubric], ignore_index=True)
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
    md_path.write_text(markdown_task_sections(clear_rows), encoding="utf-8")
    write_latex_task_table(clear_rows, tex_path)
    write_markdown(all_rows, clear_rows, summary_path)

    print(f"full rows: {len(all_rows)}")
    print(f"clear-difference rows: {len(clear_rows)}")
    print(f"summary: {summary_path}")
    print(f"full csv: {all_path}")
    print(f"clear csv: {clear_path}")
    print(f"tex: {tex_path}")


if __name__ == "__main__":
    main()
