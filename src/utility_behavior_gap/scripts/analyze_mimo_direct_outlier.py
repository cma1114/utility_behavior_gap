#!/usr/bin/env python3
"""Diagnose MiMo direct-instruction responsiveness across runs.

This is a local analysis only: it reads existing trial-level files and API
logs, then writes compact reproducibility artifacts under outputs/analysis.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API, ROOT


MIMO_V25 = "mimo-v25-pro-or"
MIMO_V2 = "mimo-v2-pro-or"
FUND_MANIFEST = OUTPUT_API / "runs" / f"fund_wording_rerun_manifests__{MIMO_V25}.tsv"
FUND_CELLS = ANALYSIS / "fund_wording_results_cells.csv"
FUND_PAIRS = ANALYSIS / "fund_wording_judged_pairs.csv"
PAPER_ESSAY_DIRECT = ROOT / "essay_all_conditions" / "direct" / f"{MIMO_V2}.json"
PAPER_SYS_DIRS = {
    "grant_proposal_abstract": ROOT / "trial_level_data" / "sys_scaleup_grant_proposal_abstract_v1",
    "incident_postmortem": ROOT / "trial_level_data" / "sys_scaleup_incident_postmortem_v1_maxtok3000",
    "translation": ROOT / "trial_level_data" / "sys_scaleup_translation_v1",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def word_count(text: str) -> int:
    return len(re.findall(r"\b\S+\b", text or ""))


def condition_output_id(job: dict[str, Any], condition: str) -> str:
    if job.get("condition_a") == condition:
        return f"{job['pair_uid']}::a"
    if job.get("condition_b") == condition:
        return f"{job['pair_uid']}::b"
    raise ValueError(f"{job['pair_uid']} does not contain {condition}")


def rate_row(
    *,
    source: str,
    actor: str,
    task: str,
    wins: int,
    losses: int,
    ties: int,
    strong_condition: str,
    normal_condition: str,
    notes: str,
) -> dict[str, Any]:
    resolved = wins + losses
    return {
        "source": source,
        "actor": actor,
        "task": task,
        "strong_condition": strong_condition,
        "normal_condition": normal_condition,
        "strong_wins": wins,
        "normal_wins": losses,
        "ties": ties,
        "resolved": resolved,
        "strong_win_rate_excluding_ties": wins / resolved if resolved else "",
        "notes": notes,
    }


def paper_mimo_v2_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    data = json.loads(PAPER_ESSAY_DIRECT.read_text(encoding="utf-8"))["trials"]
    counts: Counter[str] = Counter()
    length_diffs: list[int] = []
    for trial in data:
        winner = trial.get("winner_arm")
        if winner == "A":
            counts["strong"] += 1
        elif winner == "B":
            counts["normal"] += 1
        elif str(trial.get("majority_vote", "")).upper() == "TIE":
            counts["tie"] += 1
        else:
            counts["other"] += 1
        length_diffs.append(word_count(trial.get("essay_A", "")) - word_count(trial.get("essay_B", "")))
    rows.append(
        rate_row(
            source="paper_essay_all_conditions/direct",
            actor=MIMO_V2,
            task="essay",
            wins=counts["strong"],
            losses=counts["normal"],
            ties=counts["tie"],
            strong_condition="strong system prompt",
            normal_condition="normal system prompt",
            notes=(
                f"paper trial file; {counts['other']} non-tie/non-A/B rows; "
                f"mean strong-minus-normal word count {sum(length_diffs) / len(length_diffs):.1f}"
            ),
        )
    )

    for task, directory in PAPER_SYS_DIRS.items():
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        cell = summary["by_task_actor"][task][MIMO_V2]
        rows.append(
            rate_row(
                source=str(directory.relative_to(ROOT)),
                actor=MIMO_V2,
                task=task,
                wins=int(cell["n_sys_strong"]),
                losses=int(cell["n_sys_normal"]),
                ties=int(cell["n_tie"]),
                strong_condition=str(cell["left_condition"]),
                normal_condition=str(cell["right_condition"]),
                notes="paper sys_scaleup summary",
            )
        )
    return rows


def current_headroom_rows() -> list[dict[str, Any]]:
    cells = pd.read_csv(FUND_CELLS)
    sub = cells[(cells["actor"].eq(MIMO_V25)) & (cells["comparison"].str.endswith("_headroom"))]
    rows: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        rows.append(
            rate_row(
                source="fund_wording_rerun_headroom",
                actor=MIMO_V25,
                task=row["task"],
                wins=int(row["predicted_wins"]),
                losses=int(row["resolved"] - row["predicted_wins"]),
                ties=int(row["ties"]),
                strong_condition="framed_strong",
                normal_condition="framed_neutral",
                notes="latest corrected fund-wording rerun; system max-effort on top of framed neutral intervention",
            )
        )
    return rows


def earlier_user_prompt_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "outputs" / "raw").glob(f"*direct*max_effort*{MIMO_V25}*judged_pairs.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            data = list(csv.DictReader(fh))
        if not data:
            continue
        pred = data[0]["predicted_condition"]
        other = data[0]["other_condition"]
        wins = sum(row["panel_winner_condition"] == pred for row in data)
        losses = sum(row["panel_winner_condition"] == other for row in data)
        ties = sum(row["panel_winner_condition"] == "tie" for row in data)
        task = data[0]["task"]
        rows.append(
            rate_row(
                source=path.name,
                actor=MIMO_V25,
                task=task,
                wins=wins,
                losses=losses,
                ties=ties,
                strong_condition=pred,
                normal_condition=other,
                notes="earlier clean no-outcome user-prompt max-effort rerun",
            )
        )
    return rows


def current_headroom_pair_diagnostics() -> pd.DataFrame:
    judged = pd.read_csv(FUND_PAIRS)
    judged = judged[(judged["actor"].eq(MIMO_V25)) & (judged["comparison"].str.endswith("_headroom"))]
    outcomes = {row["pair_uid"]: row["panel_winner_condition"] for _, row in judged.iterrows()}
    rows: list[dict[str, Any]] = []
    for line in FUND_MANIFEST.read_text(encoding="utf-8").splitlines():
        _actor, task, manifest = line.split("\t")
        manifest_path = Path(manifest)
        run_dir = manifest_path.parent
        jobs = read_jsonl(manifest_path)
        generations = {row["output_id"]: row for row in read_jsonl(run_dir / "generations.jsonl")}
        for job in jobs:
            if not str(job.get("comparison", "")).endswith("_headroom"):
                continue
            strong = generations.get(condition_output_id(job, "framed_strong"))
            neutral = generations.get(condition_output_id(job, "framed_neutral"))
            if not strong or not neutral:
                continue
            strong_words = word_count(strong.get("output_text", ""))
            neutral_words = word_count(neutral.get("output_text", ""))
            rows.append(
                {
                    "task": task,
                    "pair_uid": job["pair_uid"],
                    "item_label": job.get("item_label", ""),
                    "panel_winner_condition": outcomes.get(job["pair_uid"], "missing"),
                    "strong_words": strong_words,
                    "neutral_words": neutral_words,
                    "word_diff_strong_minus_neutral": strong_words - neutral_words,
                    "strong_finish_reason": strong.get("finish_reason", ""),
                    "neutral_finish_reason": neutral.get("finish_reason", ""),
                    "strong_starts_with_heading": str(strong.get("output_text", "")).lstrip().startswith(("#", "**")),
                    "neutral_starts_with_heading": str(neutral.get("output_text", "")).lstrip().startswith(("#", "**")),
                    "source_run_dir": str(run_dir),
                    "strong_output_id": strong["output_id"],
                    "neutral_output_id": neutral["output_id"],
                }
            )
    return pd.DataFrame(rows)


def write_examples(pair_diag: pd.DataFrame, path: Path) -> None:
    lines = [
        "# MiMo V2.5 Headroom Examples",
        "",
        "Examples are from the latest corrected fund-wording headroom rerun. Strong means `framed_strong`; neutral means `framed_neutral`.",
        "",
    ]
    generations_cache: dict[Path, dict[str, dict[str, Any]]] = {}

    def generation(row: pd.Series, output_id_col: str) -> dict[str, Any]:
        run_dir = Path(row["source_run_dir"])
        if run_dir not in generations_cache:
            generations_cache[run_dir] = {g["output_id"]: g for g in read_jsonl(run_dir / "generations.jsonl")}
        return generations_cache[run_dir][row[output_id_col]]

    for task in sorted(pair_diag["task"].unique()):
        task_rows = pair_diag[pair_diag["task"].eq(task)]
        for outcome in ("framed_neutral", "framed_strong", "tie"):
            sub = task_rows[task_rows["panel_winner_condition"].eq(outcome)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            strong = generation(row, "strong_output_id")
            neutral = generation(row, "neutral_output_id")
            lines += [
                f"## {task}: {outcome}",
                "",
                f"- item: {row['item_label']}",
                f"- word counts: strong {row['strong_words']}, neutral {row['neutral_words']}",
                "",
                "Strong excerpt:",
                "",
                str(strong.get("output_text", ""))[:900].strip(),
                "",
                "Neutral excerpt:",
                "",
                str(neutral.get("output_text", ""))[:900].strip(),
                "",
            ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    headers = [str(col) for col in formatted.columns]
    rows = [[str(row[col]) for col in formatted.columns] for _, row in formatted.iterrows()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    rate_rows = paper_mimo_v2_rows() + current_headroom_rows() + earlier_user_prompt_rows()
    rate_df = pd.DataFrame(rate_rows)
    rate_path = ANALYSIS / "mimo_direct_outlier_rate_comparison.csv"
    rate_df.to_csv(rate_path, index=False)

    pair_diag = current_headroom_pair_diagnostics()
    pair_path = ANALYSIS / "mimo_direct_outlier_current_headroom_pairs.csv"
    pair_diag.to_csv(pair_path, index=False)

    length_summary = (
        pair_diag.groupby("task", as_index=False)
        .agg(
            pairs=("pair_uid", "count"),
            strong_wins=("panel_winner_condition", lambda s: int((s == "framed_strong").sum())),
            neutral_wins=("panel_winner_condition", lambda s: int((s == "framed_neutral").sum())),
            ties=("panel_winner_condition", lambda s: int((s == "tie").sum())),
            mean_strong_words=("strong_words", "mean"),
            mean_neutral_words=("neutral_words", "mean"),
            mean_word_diff=("word_diff_strong_minus_neutral", "mean"),
            strong_longer_pairs=("word_diff_strong_minus_neutral", lambda s: int((s > 0).sum())),
            neutral_longer_pairs=("word_diff_strong_minus_neutral", lambda s: int((s < 0).sum())),
            non_stop_outputs=("strong_finish_reason", lambda s: int((s != "stop").sum())),
        )
    )
    length_path = ANALYSIS / "mimo_direct_outlier_current_headroom_length_summary.csv"
    length_summary.to_csv(length_path, index=False)

    examples_path = ANALYSIS / "mimo_direct_outlier_current_headroom_examples.md"
    write_examples(pair_diag, examples_path)

    summary_path = ANALYSIS / "mimo_direct_outlier_summary.md"
    lines = [
        "# MiMo Direct-Instruction Outlier Diagnostic",
        "",
        "Key comparison: paper MiMo V2 system-prompt direct condition versus current MiMo V2.5 corrected headroom/direct condition and earlier clean user-prompt max-effort reruns.",
        "",
        "## Rate Comparison",
        "",
        markdown_table(rate_df),
        "",
        "## Current MiMo V2.5 Headroom Length/Truncation Check",
        "",
        markdown_table(length_summary),
        "",
        "## Provisional Interpretation",
        "",
        "- MiMo V2.5 is not generally unresponsive to direct instruction: earlier clean user-prompt max-effort runs are positive, especially essay and incident postmortem.",
        "- The latest corrected headroom outlier is specific to the framed-neutral/system-prompt setup. The neutral side already has competition, expert judging, and funded-intervention framing, and MiMo V2.5 often produces polished formatted outputs in both arms.",
        "- The paper MiMo V2 result is mixed across tasks: positive for essay and grant abstract, but near null for translation and incident postmortem in the supplied trial-level sys-scaleup files.",
        "- Mechanical failure does not explain the current MiMo V2.5 headroom result: all checked strong/neutral outputs ended with `finish_reason=stop`.",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"rate comparison: {rate_path}")
    print(f"pair diagnostics: {pair_path}")
    print(f"length summary: {length_path}")
    print(f"examples: {examples_path}")
    print(f"summary: {summary_path}")
    print(rate_df.to_string(index=False))
    print()
    print(length_summary.to_string(index=False))


if __name__ == "__main__":
    main()
