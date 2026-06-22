#!/usr/bin/env python3
"""Build the compact main-text feature table for direct instruction.

This table is intentionally more selective than the appendix table. It shows
features that:

1. passed the length-controlled feature-selection rule;
2. have a standardized arm gap of approximately 0.20 SD or larger; and
3. are not redundant with a broader composite shown in the same task.

Grant proposal abstracts are the special case: the seven task-specific grant
rubric markers are highly redundant, so the main table reports their mean as a
Grant Abstract Quality Composite. The appendix retains the individual markers.

No model/API calls are made.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from utility_behavior_gap.constants import PLOT_TASK_ORDER
from utility_behavior_gap.paths import ANALYSIS
from utility_behavior_gap.scripts.analyze_length_controlled_feature_selection import (
    DEFAULT_DIRECT_RUBRIC_DIR,
    DEFAULT_PAIR_DELTAS,
    fit_adjusted_gap,
    fit_panel_association,
    load_pair_deltas,
)


DEFAULT_SELECTION = ANALYSIS / "direct_instruction_length_controlled_feature_selection.csv"
DEFAULT_OUT_PREFIX = ANALYSIS / "direct_instruction_main_feature_table"

GENERIC_ORDER = [
    "words",
    "paragraphs",
    "mattr_50",
    "rare_word_rate_per_1k",
    "quantitative_detail_rate",
    "textstat_flesch_kincaid_grade",
    "positive_words_per_1k",
    "negative_words_per_1k",
]

RUBRIC_ORDER = {
    "essay": [
        "thesis_stakes_framing",
        "argument_depth",
        "concrete_example_quality",
        "counterargument_qualification",
        "rhetorical_coherence_closure",
        "plausibility_overreach",
    ],
    "grant_proposal_abstract": ["grant_abstract_quality_composite"],
    "incident_postmortem": [
        "impact_specificity",
        "timeline_precision",
        "root_cause_specificity",
        "contributing_factor_analysis",
        "detection_observability_analysis",
        "action_item_concreteness",
        "blameless_systems_framing",
        "operational_realism",
    ],
    "translation": [
        "fluency_idiomaticity",
        "terminology_precision",
        "named_entity_fidelity",
        "numeric_factual_fidelity",
        "additions_omissions",
        "structural_clarity",
    ],
}

GRANT_RUBRIC_DIMENSIONS = [
    "problem_significance",
    "intervention_specificity",
    "evaluation_rigor",
    "feasibility_implementation_readiness",
    "risk_mitigation",
    "measurable_impact",
    "stakeholder_context_fit",
]


def fmt_signed(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    rounded = round(float(value), digits)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:+.{digits}f}"


def fmt_raw(value: float, digits: int) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return fmt_signed(float(value), digits)


def ci_text(
    point: float,
    lo: float,
    hi: float,
    *,
    digits: int = 2,
    raw_digits: int | None = None,
) -> str:
    if raw_digits is None:
        formatter = lambda value: fmt_signed(value, digits)
    else:
        formatter = lambda value: fmt_raw(value, raw_digits)
    return f"{formatter(point)} [{formatter(lo)}, {formatter(hi)}]"


def raw_digits_for_feature(feature_id: str) -> int:
    if feature_id == "words":
        return 1
    if feature_id == "mattr_50":
        return 3
    if feature_id == "rare_word_rate_per_1k":
        return 1
    if feature_id in {"paragraphs", "quantitative_detail_rate", "textstat_flesch_kincaid_grade"}:
        return 2
    return 2


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    return "\n".join(lines)


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


def latex_table(df: pd.DataFrame, *, display_threshold: float) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\caption{Direct-instruction feature shifts favored by the judging panel.}",
        r"\label{tab:direct_instruction_features}",
        r"\begin{tabular}{@{}p{0.16\linewidth}p{0.25\linewidth}p{0.18\linewidth}p{0.18\linewidth}p{0.16\linewidth}@{}}",
        r"\toprule",
        r"Task & Dimension & Arm gap (SD) & Raw arm gap & Panel assoc. \\",
        r"\midrule",
    ]
    last_task = None
    for _, row in df.iterrows():
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
                "Note. The table reports dimensions that both changed under the strong prompt and "
                "tracked the judging panel's preferences. Specifically, each row has a "
                "strong-minus-neutral arm gap whose 95\\% CI excludes zero and a panel-association "
                "estimate whose 95\\% CI excludes zero, with aligned signs. Non-word dimensions are "
                "estimated in models that adjust for paired word-count difference and actor fixed "
                "effects; the Words row omits the word-count covariate because word count is the "
                "dimension being tested. Arm gap (SD) is the adjusted strong-minus-neutral gap "
                "standardized by the observed SD of paired differences for that dimension within "
                "task. Panel assoc. is the change in panel score associated with a one-SD increase "
                "in the paired feature difference. Main-text rows are retained when the standardized "
                f"arm gap is at least approximately {display_threshold:.2f} SD. MATTR-50 is a "
                "50-word moving-average type-token ratio, used as a length-robust lexical-variety "
                "measure. Grant Abstract Quality Composite is the mean of the seven grant-specific "
                "rubric dimensions, which were highly intercorrelated; individual grant dimensions "
                "are reported in the appendix."
            ),
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def load_rubric_composite_data(pair_deltas: pd.DataFrame, rubric_dir: Path) -> pd.DataFrame:
    flat = pd.read_csv(rubric_dir / "rubric_feature_analysis_flat_codes.csv")
    sample = pd.read_csv(rubric_dir / "rubric_feature_sample.csv")

    grant = flat[
        flat["task"].eq("grant_proposal_abstract") & flat["dimension"].isin(GRANT_RUBRIC_DIMENSIONS)
    ].copy()
    wide = grant.pivot_table(
        index="coding_pair_uid",
        columns="dimension",
        values="left_minus_right_score",
        aggfunc="first",
    )
    composite = wide[GRANT_RUBRIC_DIMENSIONS].mean(axis=1).rename("grant_abstract_quality_composite")

    sample_cols = [
        "coding_pair_uid",
        "source_pair_uid",
        "effect_score_left_minus_right",
    ]
    sample = sample[sample_cols].drop_duplicates()
    pair_cols = ["pair_uid", "actor", "task", "delta_words"]
    merged = (
        sample.merge(composite, left_on="coding_pair_uid", right_index=True, how="inner")
        .merge(
            pair_deltas[pair_cols].drop_duplicates(),
            left_on="source_pair_uid",
            right_on="pair_uid",
            how="left",
            validate="many_to_one",
        )
    )
    merged = merged[merged["task"].eq("grant_proposal_abstract")].copy()
    merged["rubric_score"] = pd.to_numeric(
        merged["grant_abstract_quality_composite"], errors="coerce"
    )
    merged["panel_score"] = pd.to_numeric(merged["effect_score_left_minus_right"], errors="coerce")
    merged["delta_words"] = pd.to_numeric(merged["delta_words"], errors="coerce")
    return merged


def grant_composite_row(pair_deltas_path: Path, rubric_dir: Path) -> dict[str, object]:
    pair_deltas, _, _ = load_pair_deltas(pair_deltas_path, "direct_instruction")
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
        "gap_pair_delta_sd": gap["gap_pair_delta_sd"],
        "gap_effect_sd": gap["gap_effect_sd"],
        "gap_ci_low_sd": gap["gap_ci_low_sd"],
        "gap_ci_high_sd": gap["gap_ci_high_sd"],
        "n_panel": panel["n_panel"],
        "panel_coef_per_sd": panel["panel_coef_per_sd"],
        "panel_ci_low": panel["panel_ci_low"],
        "panel_ci_high": panel["panel_ci_high"],
        "definition": "Mean of the seven grant proposal rubric dimensions.",
        "selected": True,
    }


def order_key(row: pd.Series) -> tuple[int, int, int]:
    task_index = PLOT_TASK_ORDER.index(str(row["task_label"]))
    if row["family"] == "Generic text feature":
        feature_order = GENERIC_ORDER.index(str(row["feature_id"]))
        family_order = 0
    else:
        feature_order = RUBRIC_ORDER[str(row["task"])].index(str(row["feature_id"]))
        family_order = 1
    return (task_index, family_order, feature_order)


def build_rows(
    *,
    selection_path: Path,
    pair_deltas_path: Path,
    rubric_dir: Path,
    threshold: float,
) -> pd.DataFrame:
    selected = pd.read_csv(selection_path)
    selected = selected[selected["selected"].astype(bool)].copy()
    selected = selected[selected["gap_effect_sd"].abs().ge(threshold)].copy()

    # Grant task-specific markers are too redundant for the main table.
    selected = selected[
        ~(
            selected["task"].eq("grant_proposal_abstract")
            & selected["family"].eq("LLM rubric marker")
        )
    ].copy()

    composite = pd.DataFrame([grant_composite_row(pair_deltas_path, rubric_dir)])
    selected = pd.concat([selected, composite], ignore_index=True)
    selected = selected[selected["gap_effect_sd"].abs().ge(threshold)].copy()
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
    return pd.DataFrame(rows)


def write_outputs(
    table: pd.DataFrame,
    out_prefix: Path,
    *,
    threshold: float,
    display_threshold: float,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    tex_path = out_prefix.with_suffix(".tex")

    table.to_csv(csv_path, index=False)
    lines = [
        "# Direct Instruction Main Feature Table",
        "",
        "Rows are selected features whose strong-minus-neutral arm gap and panel association both exclude zero with aligned signs. Non-word features control for word-count difference.",
        f"Main-text display threshold: standardized arm gap approximately `{display_threshold:.2f} SD` or larger.",
        "Generic text features are listed first in the standard feature-spec order; task-specific quality markers follow in task-rubric order.",
        "Grant task-specific markers are collapsed into the Grant Abstract Quality Composite because they are highly redundant.",
        "",
        markdown_table(table),
        "",
        "Outputs:",
        f"- CSV: `{csv_path}`",
        f"- LaTeX: `{tex_path}`",
    ]
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    tex_path.write_text(latex_table(table, display_threshold=display_threshold), encoding="utf-8")

    print(f"csv: {csv_path}")
    print(f"summary: {md_path}")
    print(f"latex: {tex_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--pair-deltas", type=Path, default=DEFAULT_PAIR_DELTAS)
    parser.add_argument("--rubric-dir", type=Path, default=DEFAULT_DIRECT_RUBRIC_DIR)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--display-threshold", type=float, default=0.20)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = build_rows(
        selection_path=args.selection,
        pair_deltas_path=args.pair_deltas,
        rubric_dir=args.rubric_dir,
        threshold=args.threshold,
    )
    write_outputs(
        table,
        args.out_prefix,
        threshold=args.threshold,
        display_threshold=args.display_threshold,
    )


if __name__ == "__main__":
    main()
