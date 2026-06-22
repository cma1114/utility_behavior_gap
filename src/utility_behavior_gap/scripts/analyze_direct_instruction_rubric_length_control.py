#!/usr/bin/env python3
"""Length-adjusted direct-instruction rubric effects.

This local-only analysis tests whether the strong direct-instruction arm scores
higher than the framed-neutral arm on each task-specific LLM rubric dimension
after controlling for word-count differences.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from utility_behavior_gap.constants import TASK_LABEL
from utility_behavior_gap.feature_specs import rubric_dimension_labels
from utility_behavior_gap.paths import ANALYSIS


DEFAULT_RUBRIC_DIR = (
    ANALYSIS
    / "task_rubric_feature_coding"
    / "direct_instruction__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash"
)
DEFAULT_PAIR_DELTAS = ANALYSIS / "final_text_analysis_pair_deltas.csv"
OUT_CSV = ANALYSIS / "direct_instruction_rubric_length_control.csv"
OUT_MD = ANALYSIS / "direct_instruction_rubric_length_control.md"


def fmt(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    rounded = round(float(value), digits)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:+.{digits}f}"


def normal_p_two_sided(z_value: float) -> float:
    if not math.isfinite(z_value):
        return math.nan
    return math.erfc(abs(z_value) / math.sqrt(2.0))


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


def length_adjusted_mean(df: pd.DataFrame) -> dict[str, float | int]:
    work = df.dropna(subset=["rubric_score", "delta_words", "actor"]).copy()
    if len(work) < 8 or work["rubric_score"].nunique(dropna=True) <= 1:
        return {
            "n": int(len(work)),
            "n_actors": int(work["actor"].nunique()) if "actor" in work else 0,
            "unadjusted_mean": math.nan,
            "adjusted_mean_at_equal_words": math.nan,
            "adjusted_ci_low": math.nan,
            "adjusted_ci_high": math.nan,
            "adjusted_p_two_sided": math.nan,
            "word_delta_coef": math.nan,
            "word_delta_p": math.nan,
        }

    y = work["rubric_score"].astype(float)
    parts = [
        pd.Series(1.0, index=work.index, name="const"),
        work["delta_words"].astype(float),
    ]
    if work["actor"].nunique() > 1:
        parts.append(pd.get_dummies(work["actor"].astype(str), prefix="actor", drop_first=True, dtype=float))
    x = pd.concat(parts, axis=1).astype(float)
    model = sm.OLS(y.to_numpy(), x.to_numpy())
    result = model.fit(cov_type="HC3")

    x_equal = x.copy()
    x_equal["delta_words"] = 0.0
    contrast = x_equal.to_numpy().mean(axis=0)
    point = float(np.dot(contrast, result.params))
    se = float(np.sqrt(np.dot(contrast, np.dot(result.cov_params(), contrast))))
    ci_low = point - 1.96 * se
    ci_high = point + 1.96 * se
    p_value = normal_p_two_sided(point / se) if se > 0 else math.nan

    word_idx = list(x.columns).index("delta_words")
    word_coef = float(result.params[word_idx])
    word_p = float(result.pvalues[word_idx])

    return {
        "n": int(len(work)),
        "n_actors": int(work["actor"].nunique()),
        "unadjusted_mean": float(y.mean()),
        "adjusted_mean_at_equal_words": point,
        "adjusted_ci_low": ci_low,
        "adjusted_ci_high": ci_high,
        "adjusted_p_two_sided": p_value,
        "word_delta_coef": word_coef,
        "word_delta_p": word_p,
    }


def load_data(rubric_dir: Path, pair_deltas_path: Path) -> pd.DataFrame:
    flat = pd.read_csv(rubric_dir / "rubric_feature_analysis_flat_codes.csv")
    sample = pd.read_csv(rubric_dir / "rubric_feature_sample.csv")
    pair_deltas = pd.read_csv(
        pair_deltas_path,
        usecols=["pair_uid", "contrast", "delta_words"],
        low_memory=False,
    )
    pair_deltas = pair_deltas[pair_deltas["contrast"].eq("direct_instruction")].copy()

    sample_cols = [
        "coding_pair_uid",
        "source_pair_uid",
        "effect_score_left_minus_right",
        "panel_winner_condition",
    ]
    sample = sample[sample_cols].drop_duplicates()
    sample = sample.merge(
        pair_deltas[["pair_uid", "delta_words"]],
        left_on="source_pair_uid",
        right_on="pair_uid",
        how="left",
        validate="many_to_one",
    )
    merged = flat.merge(
        sample[
            [
                "coding_pair_uid",
                "source_pair_uid",
                "effect_score_left_minus_right",
                "panel_winner_condition",
                "delta_words",
            ]
        ],
        on="coding_pair_uid",
        how="left",
        validate="many_to_one",
    )
    merged["rubric_score"] = pd.to_numeric(merged["left_minus_right_score"], errors="coerce")
    merged["delta_words"] = pd.to_numeric(merged["delta_words"], errors="coerce")
    return merged


def analyze(data: pd.DataFrame) -> pd.DataFrame:
    labels = rubric_dimension_labels()
    rows: list[dict[str, object]] = []
    for (task, dimension), group in data.groupby(["task", "dimension"], sort=True):
        result = length_adjusted_mean(group)
        adjusted = float(result["adjusted_mean_at_equal_words"])
        ci_low = float(result["adjusted_ci_low"])
        ci_high = float(result["adjusted_ci_high"])
        rows.append(
            {
                "task": task,
                "task_label": TASK_LABEL.get(str(task), str(task)),
                "dimension": dimension,
                "dimension_label": labels.get(str(dimension), str(dimension)),
                **result,
                "adjusted_ci_excludes_zero": bool(
                    math.isfinite(ci_low)
                    and math.isfinite(ci_high)
                    and (ci_low > 0 or ci_high < 0)
                ),
                "adjusted_direction": (
                    "strong > neutral"
                    if math.isfinite(adjusted) and adjusted > 0
                    else "neutral > strong"
                    if math.isfinite(adjusted) and adjusted < 0
                    else "no estimate"
                ),
            }
        )
    out = pd.DataFrame(rows)
    task_order = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]
    out["task_order"] = out["task"].map({task: index for index, task in enumerate(task_order)}).fillna(99)
    out = out.sort_values(["task_order", "adjusted_mean_at_equal_words"], ascending=[True, False]).drop(columns=["task_order"])
    return out


def write_markdown(results: pd.DataFrame, path: Path, *, rubric_dir: Path, pair_deltas_path: Path) -> None:
    lines = [
        "# Direct Instruction Rubric Effects Controlling for Length",
        "",
        "Question: does the strong direct-instruction arm score higher on each task-specific rubric dimension after controlling for word-count difference?",
        "",
        "Model fit separately for each task and rubric dimension:",
        "",
        "`rubric_score_strong_minus_neutral ~ delta_words + actor fixed effects`",
        "",
        "`rubric_score` is +1 when the rubric coder preferred the strong output, 0 for tie/not-applicable, and -1 when it preferred the neutral output. The reported adjusted mean is the model-implied average rubric score at `delta_words = 0`, averaged over the observed actor mix.",
        "",
        f"Rubric input: `{rubric_dir}`",
        f"Word deltas: `{pair_deltas_path}`",
        "",
    ]

    for task_label, group in results.groupby("task_label", sort=False):
        show = group.copy()
        show["adjusted effect"] = [
            f"{fmt(row.adjusted_mean_at_equal_words)} [{fmt(row.adjusted_ci_low)}, {fmt(row.adjusted_ci_high)}]"
            for row in show.itertuples()
        ]
        show["unadjusted"] = show["unadjusted_mean"].map(lambda value: fmt(value))
        show["p"] = show["adjusted_p_two_sided"].map(lambda value: "" if pd.isna(value) else f"{float(value):.3g}")
        show = show[
            [
                "dimension_label",
                "n",
                "unadjusted",
                "adjusted effect",
                "p",
                "adjusted_ci_excludes_zero",
            ]
        ].rename(
            columns={
                "dimension_label": "dimension",
                "adjusted_ci_excludes_zero": "CI excludes 0",
            }
        )
        lines.extend([f"## {task_label}", "", markdown_table(show), ""])
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric-dir", type=Path, default=DEFAULT_RUBRIC_DIR)
    parser.add_argument("--pair-deltas", type=Path, default=DEFAULT_PAIR_DELTAS)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.rubric_dir, args.pair_deltas)
    results = analyze(data)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_csv, index=False)
    write_markdown(results, args.out_md, rubric_dir=args.rubric_dir, pair_deltas_path=args.pair_deltas)
    print(f"rows: {len(results)}")
    print(f"csv: {args.out_csv}")
    print(f"summary: {args.out_md}")


if __name__ == "__main__":
    main()
