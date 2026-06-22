#!/usr/bin/env python3
"""Build the compact main-text feature table for high-low utility.

The table reports high-minus-low utility effects for the same dimensions shown
in the direct-instruction main feature table. That gives a direct comparison:
the features that move under an explicit effort cue can be inspected under the
high-low utility manipulation, without promoting an empty selected-feature table
to a paper artifact.

No model/API calls are made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utility_behavior_gap.constants import PLOT_TASK_ORDER
from utility_behavior_gap.paths import ANALYSIS
from utility_behavior_gap.scripts.analyze_length_controlled_feature_selection import (
    DEFAULT_PAIR_DELTAS,
    fit_adjusted_gap,
    fit_panel_association,
    load_pair_deltas,
)
from utility_behavior_gap.scripts.make_direct_instruction_main_feature_table import (
    GENERIC_ORDER,
    GRANT_RUBRIC_DIMENSIONS,
    RUBRIC_ORDER,
    ci_text,
    latex_escape,
    load_rubric_composite_data,
    markdown_table,
    raw_digits_for_feature,
)


DEFAULT_SELECTION = ANALYSIS / "utility_length_controlled_feature_selection.csv"
DEFAULT_OUT_PREFIX = ANALYSIS / "highlow_main_feature_table"
DEFAULT_UTILITY_RUBRIC_DIR = (
    ANALYSIS
    / "task_rubric_feature_coding"
    / "utility__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash"
)

DIRECT_INSTRUCTION_FEATURE_SET = [
    ("essay", "words"),
    ("essay", "rare_word_rate_per_1k"),
    ("essay", "argument_depth"),
    ("essay", "rhetorical_coherence_closure"),
    ("grant_proposal_abstract", "words"),
    ("grant_proposal_abstract", "mattr_50"),
    ("grant_proposal_abstract", "rare_word_rate_per_1k"),
    ("grant_proposal_abstract", "textstat_flesch_kincaid_grade"),
    ("grant_proposal_abstract", "grant_abstract_quality_composite"),
    ("incident_postmortem", "words"),
    ("incident_postmortem", "mattr_50"),
    ("incident_postmortem", "rare_word_rate_per_1k"),
    ("incident_postmortem", "impact_specificity"),
    ("incident_postmortem", "detection_observability_analysis"),
    ("incident_postmortem", "action_item_concreteness"),
    ("translation", "fluency_idiomaticity"),
    ("translation", "structural_clarity"),
]

TABLE_COLUMNS = [
    "Task",
    "Dimension",
    "Arm gap (SD units, 95% CI)",
    "Raw arm gap (95% CI)",
    "Panel association (95% CI)",
]


def bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"1", "true", "yes"})


def order_key(row: pd.Series) -> tuple[int, int, int]:
    task_label = str(row["task_label"])
    task_index = PLOT_TASK_ORDER.index(task_label) if task_label in PLOT_TASK_ORDER else 999
    if row["family"] == "Generic text feature":
        feature_order = (
            GENERIC_ORDER.index(str(row["feature_id"]))
            if str(row["feature_id"]) in GENERIC_ORDER
            else 999
        )
        family_order = 0
    else:
        task_order = RUBRIC_ORDER.get(str(row["task"]), [])
        feature_order = (
            task_order.index(str(row["feature_id"])) if str(row["feature_id"]) in task_order else 999
        )
        family_order = 1
    return (task_index, family_order, feature_order)


def formatted_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    rows = rows.copy()
    rows["sort_key"] = rows.apply(order_key, axis=1)
    rows = rows.sort_values("sort_key").drop(columns=["sort_key"])

    formatted: list[dict[str, object]] = []
    last_task = None
    for row in rows.itertuples(index=False):
        task_label = str(row.task_label)
        formatted.append(
            {
                "Task": task_label if task_label != last_task else "",
                "Dimension": row.feature_label,
                "Arm gap (SD units, 95% CI)": ci_text(
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
    return pd.DataFrame(formatted, columns=TABLE_COLUMNS)


def latex_table(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\caption{High-low utility shifts on direct-instruction feature dimensions.}",
        r"\label{tab:highlow_features}",
        r"\begin{tabular}{@{}p{0.16\linewidth}p{0.25\linewidth}p{0.18\linewidth}p{0.18\linewidth}p{0.16\linewidth}@{}}",
        r"\toprule",
        r"Task & Dimension & Arm gap (SD) & Raw arm gap & Panel assoc. \\",
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
                    latex_escape(row["Arm gap (SD units, 95% CI)"]),
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
                "Note. Rows are the dimensions reported in the direct-instruction feature table, "
                "with the same ordering and definitions. Here the arm gap is high-utility minus "
                "low-utility. Non-word dimensions adjust for paired word-count difference and actor "
                "fixed effects; the Words row omits the word-count covariate because word count is "
                "the dimension being tested. Arm gap (SD) is standardized by the observed SD of "
                "paired differences for that dimension within task. Panel assoc. is the change in "
                "panel score associated with a one-SD increase in the paired feature difference. "
                "None of these high-low arm gaps reaches the approximately 0.20 SD main-text "
                "effect-size threshold used for the direct-instruction feature table."
            ),
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    *,
    table: pd.DataFrame,
    nominal_table: pd.DataFrame,
    out_prefix: Path,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    tex_path = out_prefix.with_suffix(".tex")
    nominal_prefix = out_prefix.with_name(out_prefix.name + "_nominal_ci_rows")
    nominal_csv = nominal_prefix.with_suffix(".csv")
    nominal_md = nominal_prefix.with_suffix(".md")

    table.to_csv(csv_path, index=False)
    tex_path.write_text(latex_table(table), encoding="utf-8")
    nominal_table.to_csv(nominal_csv, index=False)

    lines = [
        "# High-Low Utility Main Feature Table",
        "",
        "Rows are the dimensions reported in the direct-instruction main feature table, with high-minus-low utility estimates shown for the same dimensions.",
        "Non-word features control for word-count difference.",
        "",
        markdown_table(table),
        "",
    ]
    if table.empty and not nominal_table.empty:
        lines.extend(
            [
                "## Nominal CI-Selected Rows Below Display Threshold",
                "",
                "These rows pass the CI/sign-alignment rule but are below the main-text effect-size threshold.",
                "",
                markdown_table(nominal_table),
                "",
            ]
        )
    lines.extend(
        [
            "Outputs:",
            f"- CSV: `{csv_path}`",
            f"- LaTeX: `{tex_path}`",
            f"- Nominal CI rows: `{nominal_csv}`",
        ]
    )
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    nominal_md.write_text(markdown_table(nominal_table) + "\n", encoding="utf-8")

    print(f"csv: {csv_path}")
    print(f"summary: {md_path}")
    print(f"latex: {tex_path}")
    print(f"nominal ci rows: {nominal_csv}")


def grant_composite_row(pair_deltas_path: Path, rubric_dir: Path) -> dict[str, object]:
    pair_deltas, _, _ = load_pair_deltas(pair_deltas_path, "utility")
    data = load_rubric_composite_data(pair_deltas, rubric_dir)
    gap = fit_adjusted_gap(data, feature_col="rubric_score", is_words=False)
    panel = fit_panel_association(data, feature_col="rubric_score", is_words=False)
    return {
        "task": "grant_proposal_abstract",
        "task_label": "Grant abstract",
        "family": "Task-specific quality marker",
        "feature_id": "grant_abstract_quality_composite",
        "feature_label": "Grant Abstract Quality Composite",
        "n_gap": gap["n_gap"],
        "gap_effect": gap["gap_effect"],
        "gap_ci_low": gap["gap_ci_low"],
        "gap_ci_high": gap["gap_ci_high"],
        "gap_p": gap["gap_p"],
        "gap_pair_delta_sd": gap["gap_pair_delta_sd"],
        "gap_effect_sd": gap["gap_effect_sd"],
        "gap_ci_low_sd": gap["gap_ci_low_sd"],
        "gap_ci_high_sd": gap["gap_ci_high_sd"],
        "n_panel": panel["n_panel"],
        "panel_coef_per_sd": panel["panel_coef_per_sd"],
        "panel_ci_low": panel["panel_ci_low"],
        "panel_ci_high": panel["panel_ci_high"],
        "panel_p": panel["panel_p"],
        "definition": f"Mean of the {len(GRANT_RUBRIC_DIMENSIONS)} grant proposal rubric dimensions.",
        "selected": False,
    }


def build_direct_feature_set_rows(
    *,
    selection_path: Path,
    pair_deltas_path: Path,
    rubric_dir: Path,
) -> pd.DataFrame:
    stats = pd.read_csv(selection_path)
    composite = pd.DataFrame([grant_composite_row(pair_deltas_path, rubric_dir)])
    stats = pd.concat([stats, composite], ignore_index=True)

    rows = []
    for task, feature_id in DIRECT_INSTRUCTION_FEATURE_SET:
        matches = stats[stats["task"].eq(task) & stats["feature_id"].eq(feature_id)]
        if matches.empty:
            raise ValueError(f"Missing high-low stats for {task}:{feature_id}")
        rows.append(matches.iloc[0].to_dict())
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--pair-deltas", type=Path, default=DEFAULT_PAIR_DELTAS)
    parser.add_argument("--rubric-dir", type=Path, default=DEFAULT_UTILITY_RUBRIC_DIR)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = pd.read_csv(args.selection)
    selected = selected[bool_series(selected["selected"])].copy()
    nominal_table = formatted_rows(selected)
    direct_feature_rows = build_direct_feature_set_rows(
        selection_path=args.selection,
        pair_deltas_path=args.pair_deltas,
        rubric_dir=args.rubric_dir,
    )
    table = formatted_rows(direct_feature_rows)
    write_outputs(
        table=table,
        nominal_table=nominal_table,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()
