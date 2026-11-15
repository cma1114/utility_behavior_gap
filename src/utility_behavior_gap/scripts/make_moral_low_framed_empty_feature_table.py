#!/usr/bin/env python3
"""Build the compact feature table for moral-low vs framed-empty.

Rows are selected symmetrically: the harmful-minus-framed-empty arm gap must
exclude zero, the panel association must exclude zero, and the standardized
arm gap must clear the display threshold. The table then labels whether each
shift favors harmful or framed-empty. This table combines current generic
text features with the existing task-specific LLM rubric coding.

No model/API calls are made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex

import pandas as pd

from utility_behavior_gap.constants import PLOT_TASK_ORDER
from utility_behavior_gap.paths import ANALYSIS
from utility_behavior_gap.scripts.make_direct_instruction_main_feature_table import (
    GENERIC_ORDER,
    RUBRIC_ORDER,
    ci_text,
    latex_escape,
    markdown_table,
    raw_digits_for_feature,
)


DEFAULT_SELECTION = ANALYSIS / "moral_low_vs_framed_empty_length_controlled_feature_selection.csv"
DEFAULT_OUT_PREFIX = ANALYSIS / "moral_low_vs_framed_empty_feature_table"
DEFAULT_THRESHOLD = 0.25

TABLE_COLUMNS = [
    "Task",
    "Dimension",
    "Harmful-minus-empty gap (SD units, 95% CI)",
    "Raw arm gap (95% CI)",
    "Panel association (95% CI)",
]


def bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"1", "true", "yes"})


def order_key(row: pd.Series) -> tuple[int, int, int]:
    task_label = str(row["task_label"])
    task_index = PLOT_TASK_ORDER.index(task_label) if task_label in PLOT_TASK_ORDER else 999
    feature = str(row["feature_id"])
    task = str(row["task"])
    if feature in GENERIC_ORDER:
        return (task_index, 0, GENERIC_ORDER.index(feature))
    rubric_order = RUBRIC_ORDER.get(task, [])
    feature_index = rubric_order.index(feature) if feature in rubric_order else 999
    return (task_index, 1, feature_index)


def build_table(selection_path: Path, *, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(selection_path)
    clear_gap = bool_series(selected["gap_ci_excludes_zero"])
    clear_panel = bool_series(selected["panel_ci_excludes_zero"])
    selected = selected[clear_gap & clear_panel].copy()
    selected = selected[pd.to_numeric(selected["gap_effect_sd"], errors="coerce").abs().ge(threshold)].copy()
    if "selected" in selected.columns:
        selected = selected.rename(columns={"selected": "source_sign_aligned_selected"})
    selected["included_in_table"] = True
    selected["sort_key"] = selected.apply(order_key, axis=1)
    selected = selected.sort_values("sort_key").drop(columns=["sort_key"])

    rows: list[dict[str, object]] = []
    last_task = None
    for row in selected.itertuples(index=False):
        task_label = str(row.task_label)
        rows.append(
            {
                "Task": task_label if task_label != last_task else "",
                "Dimension": row.feature_label,
                "Harmful-minus-empty gap (SD units, 95% CI)": ci_text(
                    row.gap_effect_sd,
                    row.gap_ci_low_sd,
                    row.gap_ci_high_sd,
                ),
                "Raw arm gap (95% CI)": ci_text(
                    row.gap_effect,
                    row.gap_ci_low,
                    row.gap_ci_high,
                    raw_digits=raw_digits_for_feature(str(row.feature_id)),
                ),
                "Panel association (95% CI)": ci_text(
                    row.panel_coef_per_sd,
                    row.panel_ci_low,
                    row.panel_ci_high,
                ),
            }
        )
        last_task = task_label
    return pd.DataFrame(rows, columns=TABLE_COLUMNS), selected


def latex_table(table: pd.DataFrame, *, threshold: float) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\caption{Feature shifts for moral-low versus framed-empty outputs.}",
        r"\label{tab:moral_low_framed_empty_features}",
        r"\begin{tabular}{@{}p{0.16\linewidth}p{0.25\linewidth}p{0.18\linewidth}p{0.18\linewidth}p{0.16\linewidth}@{}}",
        r"\toprule",
        r"Task & Dimension & Harmful--empty gap (SD) & Raw arm gap & Panel assoc. \\",
        r"\midrule",
    ]
    last_task = None
    for _, row in table.iterrows():
        task = str(row["Task"])
        if task and last_task is not None:
            lines.append(r"\specialrule{0.2pt}{1pt}{1pt}")
        shown_task = task if task != last_task else ""
        lines.append(
            " & ".join(
                [
                    latex_escape(shown_task),
                    latex_escape(row["Dimension"]),
                    latex_escape(row["Harmful-minus-empty gap (SD units, 95% CI)"]),
                    latex_escape(row["Raw arm gap (95% CI)"]),
                    latex_escape(row["Panel association (95% CI)"]),
                ]
            )
            + r" \\"
        )
        last_task = task
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize",
            (
                "Note. The table reports all dimensions, in either direction, with a "
                "harmful-minus-empty arm gap whose 95\\% CI excludes zero, a panel-association "
                "estimate whose 95\\% CI excludes zero, and an absolute standardized arm gap of "
                f"at least approximately {threshold:.2f} SD. Non-word dimensions are estimated "
                "in models that adjust for paired word-count difference and actor fixed effects; "
                "the Words row omits the word-count covariate because word count is the dimension "
                "being tested. The SD-scaled gap is the adjusted arm gap divided by the observed "
                "SD of paired differences for that dimension within task. Panel assoc. is the "
                "change in harmful-minus-empty panel score associated with a one-SD increase in "
                "the paired feature difference. Refusal and degenerate outputs were excluded "
                "before analysis; task-specific rubric rows use the existing successful LLM-coded "
                "sample."
            ),
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    table: pd.DataFrame,
    selected: pd.DataFrame,
    out_prefix: Path,
    *,
    threshold: float,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    tex_path = out_prefix.with_suffix(".tex")
    selected_detail = out_prefix.with_name(out_prefix.name + "_selected_detail.csv")

    table.to_csv(csv_path, index=False)
    selected.to_csv(selected_detail, index=False)
    tex_path.write_text(latex_table(table, threshold=threshold), encoding="utf-8")
    lines = [
        "# Moral Low Vs Framed Empty Feature Table",
        "",
        "Rows combine current generic text features with existing task-specific LLM rubric coding.",
        "Rows are selected symmetrically: clear arm gaps, clear panel associations, and absolute standardized arm gaps above the display threshold.",
        f"Main-text display threshold: standardized arm gap approximately `{threshold:.2f} SD` or larger.",
        "",
        markdown_table(table),
        "",
        "Outputs:",
        f"- CSV: `{csv_path}`",
        f"- LaTeX: `{tex_path}`",
        f"- selected detail: `{selected_detail}`",
    ]
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"csv: {csv_path}")
    print(f"summary: {md_path}")
    print(f"latex: {tex_path}")
    print(f"selected detail: {selected_detail}")


def parallel_rows(first: Path, second: Path, *, threshold: float) -> pd.DataFrame:
    """Union of paper-selected dimensions; never filter out the counterpart."""
    a, b = pd.read_csv(first), pd.read_csv(second)
    keys = ["task", "feature_id"]
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError("Duplicate task/feature rows")
    if set(map(tuple, a[keys].to_numpy())) != set(map(tuple, b[keys].to_numpy())):
        raise ValueError("Both inputs must contain the same full task/feature set")
    _, chosen_a = build_table(first, threshold=threshold)
    _, chosen_b = build_table(second, threshold=threshold)
    selected_a = set(map(tuple, chosen_a[keys].to_numpy()))
    selected_b = set(map(tuple, chosen_b[keys].to_numpy()))
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    for name in ("family", "controls_words_for_gap", "controls_words_for_panel"):
        if not merged[f"{name}_a"].eq(merged[f"{name}_b"]).all():
            raise ValueError(f"Non-parallel feature specifications: {name}")
    merged["selected_a"] = [(r.task, r.feature_id) in selected_a for r in merged.itertuples()]
    merged["selected_b"] = [(r.task, r.feature_id) in selected_b for r in merged.itertuples()]
    merged = merged[merged.selected_a | merged.selected_b].copy()
    merged["task_label"] = merged.task_label_a
    merged["feature_label"] = merged.feature_label_a
    merged["sort_key"] = merged.apply(order_key, axis=1)
    return merged.sort_values("sort_key").drop(columns="sort_key")


def write_parallel(first: Path, second: Path, out_prefix: Path, *, threshold: float, labels: tuple[str, str]) -> None:
    if labels[0] == labels[1]:
        raise ValueError("Comparison labels must be distinct")
    rows = parallel_rows(first, second, threshold=threshold)
    display, plain = [], []
    for row in rows.to_dict("records"):
        record = {"Task": row["task_label"], "Dimension": row["feature_label"]}
        for side, label in zip(("a", "b"), labels):
            for title, fields in (("gap", ("gap_effect_sd", "gap_ci_low_sd", "gap_ci_high_sd")),
                                  ("panel association", ("panel_coef_per_sd", "panel_ci_low", "panel_ci_high"))):
                value = ci_text(*(row[f"{field}_{side}"] for field in fields))
                record[f"{label}: {title}"] = value
        plain.append(record.copy())
        for side, label in zip(("a", "b"), labels):
            if row[f"selected_{side}"]:
                record[f"{label}: gap"] = "**" + record[f"{label}: gap"] + "**"
        display.append(record)
    shown, raw = pd.DataFrame(display), pd.DataFrame(plain)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_prefix.with_suffix(".csv"), index=False)
    note = (
        "Both contrasts are against framed empty. Gaps are outcome-minus-control differences in SD units, "
        "with nominal 95% confidence intervals. Negative gaps mean fewer words or weaker rubric-rated quality; "
        "for other generic features, they indicate less of that feature, not necessarily worse quality. "
        "A positive panel association means that more of the feature predicts greater panel preference for the outcome-side output. "
        "Non-word features adjust for paired word-count difference and actor in both models; Words adjusts for actor only. "
        "SDs are calculated within each task and contrast."
    )
    selection = (
        f"Rows meet the paper's criteria in at least one contrast: absolute gap >= {threshold:.2f} SD, "
        "and both gap and panel-association 95% intervals exclude zero. Both estimates are shown for every selected row, "
        "including uncertain and smaller counterparts. Bold gaps mark the contrasts meeting all three criteria. "
        "Selection uses unrounded values; rounding to 0.25 does not itself mean a row qualifies. "
        "These are the paper's feature intervals, not familywise-adjusted intervals. "
        "This is a descriptive comparison, not a test of the difference between contrasts."
    )
    absent = sorted(set(pd.read_csv(first).task_label) - set(rows.task_label))
    if absent:
        selection += " No dimensions qualify in either contrast for: " + ", ".join(absent) + "."
    command = shlex.join(["python", "-m", "utility_behavior_gap.scripts.make_moral_low_framed_empty_feature_table",
        "--selection", str(first), "--compare-selection", str(second), "--first-label", labels[0],
        "--second-label", labels[1], "--out-prefix", str(out_prefix), "--threshold", str(threshold)])
    md = [f"# {labels[0]} and {labels[1]} Versus Framed Empty", "", note, "", markdown_table(shown), "", selection,
          "", "## Reproduction", "", "No new calls or statistical fits: this table joins the existing full estimates by task and feature.",
          "", "```sh", command, "```", ""]
    out_prefix.with_suffix(".md").write_text("\n".join(md))
    tex = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
           r"\renewcommand{\arraystretch}{1.06}",
           r"\caption{Feature shifts relative to framed empty: " + latex_escape(" and ".join(labels)) + ".}",
           r"\label{tab:parallel_framed_empty_features}",
           r"\begin{tabular}{@{}p{0.13\linewidth}p{0.23\linewidth}p{0.15\linewidth}p{0.15\linewidth}p{0.15\linewidth}p{0.15\linewidth}@{}}",
           r"\toprule", r" & & \multicolumn{2}{c}{" + latex_escape(labels[0]) + r"} & \multicolumn{2}{c}{" + latex_escape(labels[1]) + r"} \\",
           r"\cmidrule(lr){3-4}\cmidrule(l){5-6}", r"Task & Dimension & Gap (SD) & Panel assoc. & Gap (SD) & Panel assoc. \\", r"\midrule"]
    previous = None
    for record in display:
        task = record["Task"]
        if previous is not None and previous != task:
            tex.append(r"\specialrule{0.2pt}{1pt}{1pt}")
        cells = [task if task != previous else "", record["Dimension"], *list(record.values())[2:]]
        formatted = []
        for value in cells:
            if value.startswith("**"):
                formatted.append(r"\textbf{" + latex_escape(value[2:-2]) + "}")
            else:
                formatted.append(latex_escape(value))
        tex.append(" & ".join(formatted) + r" \\")
        previous = task
    tex += [r"\bottomrule", r"\end{tabular}", r"\begin{minipage}{\linewidth}", r"\footnotesize",
            "Note. " + latex_escape(note + " " + selection), r"\end{minipage}", r"\end{table*}", ""]
    out_prefix.with_suffix(".tex").write_text("\n".join(tex))
    provenance = dict(inputs={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (first, second)},
                      threshold=threshold, selection="union; full counterpart retained", rows=len(rows),
                      command=command, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out_prefix.with_name(out_prefix.name + "_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(out_prefix.with_suffix(".md"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--compare-selection", type=Path)
    parser.add_argument("--first-label", default="Harmful")
    parser.add_argument("--second-label", default="Low policy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare_selection is not None:
        write_parallel(args.selection, args.compare_selection, args.out_prefix,
                       threshold=args.threshold, labels=(args.first_label, args.second_label))
        return
    table, selected = build_table(args.selection, threshold=args.threshold)
    write_outputs(table, selected, args.out_prefix, threshold=args.threshold)


if __name__ == "__main__":
    main()
