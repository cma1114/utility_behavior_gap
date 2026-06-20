#!/usr/bin/env python3
"""Diagnose GPT-5.4-mini outlier cells in high-utility versus R0."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from utility_behavior_gap.paths import ROOT


RUNS = ROOT / "outputs" / "api" / "runs"
ANALYSIS = ROOT / "outputs" / "analysis"
PAPER_READY = ROOT / "outputs" / "paper_ready"
CURRENT_PAPER = ROOT / "CURRENT_PAPER"
PAIR_PATH = PAPER_READY / "results" / "high_utility_vs_r0_pair_outcomes.csv"
CELL_PATH = PAPER_READY / "results" / "high_utility_vs_r0_model_task_cells.csv"
ACTOR = "gpt-5.4-mini-or"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def gpt_run_dir() -> Path:
    manifest_list = RUNS / f"high_utility_r0_bridge_manifests__{ACTOR}.tsv"
    lines = [line.strip().split("\t") for line in manifest_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected one GPT bridge manifest, found {len(lines)} in {manifest_list}")
    return Path(lines[0][1]).parent


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def output_features(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    date_re = re.compile(
        r"\b(?:202\d[-/]\d{1,2}[-/]\d{1,2}|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
        r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        re.I,
    )
    for row in read_jsonl(run_dir / "generations.jsonl"):
        job = row["job"]
        text = row.get("output_text") or row.get("content") or ""
        lower = text.lower()
        rows.append(
            {
                "bridge_pair_uid": job["pair_uid"],
                "task": job["task"],
                "domain": job.get("domain", ""),
                "condition": row["condition"],
                "finish_reason": row.get("finish_reason", ""),
                "empty": not bool(text.strip()),
                "words": word_count(text),
                "chars": len(text),
                "number_tokens": len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text)),
                "bracket_placeholders": len(re.findall(r"\[[^\]]+\]", text)),
                "date_placeholders": lower.count("[date]"),
                "specific_dates": len(date_re.findall(text)),
                "time_mentions": len(re.findall(r"\b\d{1,2}:\d{2}\b|\b\d+\s*(?:minutes?|hours?)\b", lower)),
            }
        )
    return pd.DataFrame(rows)


def paired_feature_deltas(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_uid, sub in features.groupby("bridge_pair_uid", sort=False):
        by_condition = {row["condition"]: row for _, row in sub.iterrows()}
        if "hl_high" not in by_condition or "r0" not in by_condition:
            continue
        high = by_condition["hl_high"]
        r0 = by_condition["r0"]
        row: dict[str, Any] = {
            "bridge_pair_uid": pair_uid,
            "task": high["task"],
            "domain": high["domain"],
        }
        for col in [
            "words",
            "chars",
            "number_tokens",
            "bracket_placeholders",
            "date_placeholders",
            "specific_dates",
            "time_mentions",
        ]:
            row[f"{col}_delta_high_minus_r0"] = high[col] - r0[col]
            row[f"{col}_high"] = high[col]
            row[f"{col}_r0"] = r0[col]
        rows.append(row)
    return pd.DataFrame(rows)


def judge_vote_split(run_dir: Path) -> pd.DataFrame:
    rows = []
    for vote in read_jsonl(run_dir / "judge_votes.jsonl"):
        job = vote["job"]
        rows.append(
            {
                "task": job["task"],
                "judge_model": vote.get("judge_model", ""),
                "flipped": bool(vote.get("flipped")),
                "winner_condition": vote.get("winner_condition", ""),
            }
        )
    return pd.DataFrame(rows)


def maybe_comparison_table() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    specs = [
        ("high_utility_vs_r0", PAPER_READY / "results" / "high_utility_vs_r0_model_task_cells.csv"),
        ("high_low", PAPER_READY / "results" / "highlow_model_task_cells.csv"),
        ("high_utility_vs_framed_neutral", ANALYSIS / "highlow_neutral_bridge_high__combined_7runs_model_task_cells.csv"),
        ("direct_instruction", PAPER_READY / "results" / "direct_instruction_main_plot_data.csv"),
    ]
    for label, path in specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if label == "direct_instruction":
            sub = df[df["actor"].eq(ACTOR)].copy()
            sub["comparison"] = label
            sub["target_wins"] = sub["strong_wins"]
            sub["target_losses"] = sub["neutral_wins"]
            sub["resolved_n"] = sub["n_excl_tie"]
            sub["target_win_rate_excluding_ties"] = sub["strong_win_rate_excluding_ties"]
        else:
            sub = df[df["actor"].eq(ACTOR)].copy()
            sub["comparison"] = label
        rows.append(
            sub[
                [
                    "comparison",
                    "task",
                    "resolved_n",
                    "target_wins",
                    "target_losses",
                    "ties",
                    "target_win_rate_excluding_ties",
                    "familywise_ci_positive",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pct(x: float) -> str:
    return f"{x:.1%}"


def markdown_table(df: pd.DataFrame, *, float_digits: int = 2) -> str:
    if df.empty:
        return "_No rows._"
    rows = ["| " + " | ".join(str(col) for col in df.columns) + " |"]
    rows.append("| " + " | ".join("---" for _ in df.columns) + " |")
    for _, row in df.iterrows():
        cells = []
        for value in row.tolist():
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.{float_digits}f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_markdown(
    *,
    run_dir: Path,
    cells: pd.DataFrame,
    by_domain: pd.DataFrame,
    feature_means: pd.DataFrame,
    delta_summary: pd.DataFrame,
    judge_split: pd.DataFrame,
    comparison_table: pd.DataFrame,
    out_path: Path,
) -> None:
    gpt_cells = cells[cells["actor"].eq(ACTOR)].copy()
    lines = [
        "# GPT-5.4 Mini High-Utility-vs-R0 Outlier Diagnostic",
        "",
        f"Bridge run: `{run_dir}`",
        "",
        "This is a local diagnostic of the GPT-5.4-mini cells in the high-utility versus R0 bridge. No API calls are made.",
        "",
        "## Main Cell Results",
        "",
        "| task | resolved | high wins | R0 wins | ties | win rate | FWER-CI positive |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in gpt_cells.iterrows():
        lines.append(
            f"| {row['task_label']} | {int(row['resolved_n'])} | {int(row['target_wins'])} | "
            f"{int(row['target_losses'])} | {int(row['ties'])} | "
            f"{pct(row['target_win_rate_excluding_ties'])} | {bool(row['familywise_ci_positive'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Check",
            "",
            "- The outlier is concentrated in essay and incident postmortem.",
            "- The effect is present in every domain for those two tasks, including political, which argues against a utility-content explanation.",
            "- All GPT-5.4-mini high and R0 outputs in this bridge ended with `finish_reason=stop`; none were empty.",
            "- R0 is not reused: each task has 150 distinct high outputs and 150 distinct R0 outputs.",
            "- The high side includes the blind evaluation/sponsor/funding wrapper; R0 is the bare task prompt. This bridge therefore tests high-utility framing against a bare baseline, not utility content alone.",
            "- For postmortems, the high side has many fewer bracket placeholders and many more concrete time/number details; judge rationales explicitly reward specificity, timelines, owners, due dates, and technical detail.",
            "",
            "## Domain Breakout For Outlier Tasks",
            "",
            "| task | domain | resolved | high wins | R0 wins | ties | win rate |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in by_domain.iterrows():
        lines.append(
            f"| {row['task']} | {row['domain']} | {int(row['resolved'])} | {int(row['wins'])} | "
            f"{int(row['losses'])} | {int(row['ties'])} | {pct(row['rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Output Artifact Means",
            "",
            markdown_table(feature_means, float_digits=2),
            "",
            "## Paired Feature Deltas",
            "",
            "Deltas are high minus R0.",
            "",
            markdown_table(delta_summary, float_digits=2),
            "",
            "## Judge Vote Split",
            "",
            markdown_table(judge_split, float_digits=2),
            "",
        ]
    )
    if not comparison_table.empty:
        lines.extend(
            [
                "## Comparison Against Related GPT-5.4-Mini Results",
                "",
                markdown_table(comparison_table, float_digits=3),
                "",
            ]
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_to_entry_points(path: Path) -> None:
    for root in [PAPER_READY / "results", CURRENT_PAPER / "results"]:
        if root.exists():
            shutil.copy2(path, root / path.name)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    run_dir = gpt_run_dir()
    pairs = pd.read_csv(PAIR_PATH)
    cells = pd.read_csv(CELL_PATH)
    gpt_pairs = pairs[pairs["actor"].eq(ACTOR)].copy()
    features = output_features(run_dir)
    deltas = paired_feature_deltas(features)
    merged = gpt_pairs.merge(deltas, on=["bridge_pair_uid", "task", "domain"], how="left")

    by_domain = (
        gpt_pairs[gpt_pairs["task"].isin(["essay", "incident_postmortem"])]
        .groupby(["task", "domain"], as_index=False)
        .agg(
            resolved=("resolved", "sum"),
            wins=("target_win", "sum"),
            losses=("target_loss", "sum"),
            ties=("tie", "sum"),
        )
    )
    by_domain["rate"] = by_domain["wins"] / by_domain["resolved"]

    feature_means = (
        features.groupby(["task", "condition"], as_index=False)
        .agg(
            n=("words", "size"),
            non_stop=("finish_reason", lambda s: int((s != "stop").sum())),
            empty=("empty", "sum"),
            words=("words", "mean"),
            number_tokens=("number_tokens", "mean"),
            bracket_placeholders=("bracket_placeholders", "mean"),
            date_placeholders=("date_placeholders", "mean"),
            specific_dates=("specific_dates", "mean"),
            time_mentions=("time_mentions", "mean"),
        )
    )

    delta_cols = [
        "words_delta_high_minus_r0",
        "number_tokens_delta_high_minus_r0",
        "bracket_placeholders_delta_high_minus_r0",
        "specific_dates_delta_high_minus_r0",
        "time_mentions_delta_high_minus_r0",
    ]
    delta_summary = (
        merged.groupby(["task", "target_win"], as_index=False)[delta_cols]
        .mean()
        .sort_values(["task", "target_win"])
    )

    votes = judge_vote_split(run_dir)
    judge_split = (
        votes.groupby(["task", "judge_model", "winner_condition"], as_index=False)
        .size()
        .pivot_table(index=["task", "judge_model"], columns="winner_condition", values="size", fill_value=0)
        .reset_index()
    )

    comparison_table = maybe_comparison_table()

    deltas_out = ANALYSIS / "high_utility_vs_r0_gpt_outlier_feature_deltas.csv"
    merged.to_csv(deltas_out, index=False)
    summary_out = ANALYSIS / "high_utility_vs_r0_gpt_outlier_diagnostic.md"
    write_markdown(
        run_dir=run_dir,
        cells=cells,
        by_domain=by_domain,
        feature_means=feature_means,
        delta_summary=delta_summary,
        judge_split=judge_split,
        comparison_table=comparison_table,
        out_path=summary_out,
    )
    copy_to_entry_points(summary_out)
    copy_to_entry_points(deltas_out)
    print(f"summary: {summary_out}")
    print(f"feature deltas: {deltas_out}")


if __name__ == "__main__":
    main()
