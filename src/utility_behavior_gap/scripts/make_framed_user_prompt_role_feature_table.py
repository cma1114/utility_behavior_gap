#!/usr/bin/env python3
"""Build a compact feature table for the framed user-prompt role contrast.

Rows are selected symmetrically: the role-strong-minus-role-weak arm gap must
exclude zero, the panel association must exclude zero, and the standardized
arm gap must clear the display threshold. The selection input combines current
generic text features with task-specific LLM rubric coding.

No model/API calls are made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


DEFAULT_SELECTION = ANALYSIS / "framed_user_prompt_role_length_controlled_feature_selection.csv"
DEFAULT_OUT_PREFIX = ANALYSIS / "framed_user_prompt_role_feature_table"
DEFAULT_THRESHOLD = 0.25

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
    return pd.DataFrame(rows, columns=TABLE_COLUMNS), selected


def latex_table(table: pd.DataFrame, *, threshold: float) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\caption{Feature shifts in the user-prompt role contrast.}",
        r"\label{tab:role_features}",
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
                "Note. Rows are dimensions, in either direction, with a "
                "role-strong-minus-role-weak arm gap whose 95\\% CI excludes zero, a "
                "panel-association estimate whose 95\\% CI excludes zero, and an absolute "
                f"standardized arm gap of at least approximately {threshold:.2f} SD. Non-word "
                "dimensions adjust for paired word-count difference and actor fixed effects; "
                "the Words row omits the word-count covariate because word count is the dimension "
                "being tested. Arm gap (SD) is standardized by the observed SD of paired "
                "differences for that dimension within task. Panel assoc. is the change in "
                "role-strong-minus-role-weak panel score associated with a one-SD increase in "
                "the paired feature difference. The selection considered both generic text "
                "features and task-specific LLM rubric dimensions."
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
        "# Framed User-Prompt Role Feature Table",
        "",
        "Rows are selected symmetrically: clear arm gaps, clear panel associations, and absolute standardized arm gaps above the display threshold.",
        "The selection considered both generic text features and task-specific LLM rubric dimensions.",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table, selected = build_table(args.selection, threshold=args.threshold)
    write_outputs(table, selected, args.out_prefix, threshold=args.threshold)


if __name__ == "__main__":
    main()
