#!/usr/bin/env python3
"""Analyze exploratory high-low utility runs with actor reasoning enabled.

This is intentionally separate from the canonical high-low analysis. It reads
only manifest lists named ``highlow_reasoning_medium_manifests__*.tsv`` unless
specific ``--manifest-list`` paths are supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import beta, binomtest

from utility_behavior_gap.constants import ACTOR_LABEL, TASK_LABEL
from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition
from utility_behavior_gap.paths import ROOT


RUNS = ROOT / "outputs" / "api" / "runs"
ANALYSIS = ROOT / "outputs" / "analysis"
DEFAULT_MANIFEST_GLOB = "highlow_reasoning_medium_manifests__*.tsv"
OUT_PREFIX = "highlow_reasoning_medium"
HIGH_CONDITION = "hl_high"
LOW_CONDITION = "hl_low"
EXPECTED_VOTES_PER_PAIR = 6
TASKS = {"essay", "translation", "grant_proposal_abstract", "incident_postmortem"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_manifest_list(path: Path) -> list[Path]:
    manifests: list[Path] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                manifests.append(Path(parts[2]))
    return manifests


def run_parts_from_manifest(manifest: Path) -> tuple[str, str, str]:
    parts = manifest.parent.name.split("__")
    task = parts[0] if len(parts) > 0 else "unknown-task"
    comparison = parts[1] if len(parts) > 1 else "unknown-comparison"
    actor = parts[2] if len(parts) > 2 else "unknown-actor"
    return task, comparison, actor


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)


def output_prefix(manifests: list[Path], args: argparse.Namespace) -> str:
    tasks = sorted({run_parts_from_manifest(path)[0] for path in manifests})
    actors = sorted({run_parts_from_manifest(path)[2] for path in manifests})
    task_part = "all" if str(args.tasks or "all") == "all" else "+".join(tasks)
    actor_part = actors[0] if len(actors) == 1 else f"{len(actors)}-actors"
    return f"{slug(args.out_prefix)}__tasks-{slug(task_part)}__{slug(actor_part)}"


def selected_tasks(args: argparse.Namespace) -> set[str] | None:
    value = str(args.tasks or "").strip()
    if not value or value == "all":
        return None
    out = {part.strip() for part in value.split(",") if part.strip()}
    unknown = out - TASKS
    if unknown:
        raise ValueError(f"unknown task(s): {', '.join(sorted(unknown))}")
    return out


def manifest_paths(args: argparse.Namespace) -> list[Path]:
    task_filter = selected_tasks(args)
    paths: list[Path] = []
    if args.manifest_list:
        for manifest_list in args.manifest_list:
            for path in read_manifest_list(manifest_list):
                if task_filter is None or path.parent.name.split("__", 1)[0] in task_filter:
                    paths.append(path)
    else:
        for manifest_list in sorted(RUNS.glob(DEFAULT_MANIFEST_GLOB)):
            for path in read_manifest_list(manifest_list):
                if task_filter is None or path.parent.name.split("__", 1)[0] in task_filter:
                    paths.append(path)
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def side_output_ids(job: dict[str, Any]) -> tuple[str, str]:
    pair_uid = str(job["pair_uid"])
    return f"{pair_uid}::a", f"{pair_uid}::b"


def condition_output_rows(
    job: dict[str, Any],
    out_a: dict[str, Any],
    out_b: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cond_a = str(job.get("condition_a", ""))
    cond_b = str(job.get("condition_b", ""))
    if cond_a == HIGH_CONDITION and cond_b == LOW_CONDITION:
        return out_a, out_b
    if cond_b == HIGH_CONDITION and cond_a == LOW_CONDITION:
        return out_b, out_a
    raise ValueError(f"unexpected conditions for {job.get('pair_uid')}: {cond_a!r}, {cond_b!r}")


def generation_valid(row: dict[str, Any]) -> bool:
    return (
        row.get("success") is not False
        and row.get("finish_reason") in {"stop", "dry_run", ""}
        and bool(str(row.get("output_text", "")).strip())
    )


def usage_int(row: dict[str, Any], key: str) -> int:
    value = row.get("usage", {}).get(key)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def reasoning_tokens(row: dict[str, Any]) -> int:
    details = row.get("usage", {}).get("completion_tokens_details", {})
    try:
        return int(details.get("reasoning_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def response_message(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_response", {})
    choices = raw.get("choices", []) if isinstance(raw, dict) else []
    if not choices or not isinstance(choices[0], dict):
        return {}
    message = choices[0].get("message", {})
    return message if isinstance(message, dict) else {}


def reasoning_trace_char_count(row: dict[str, Any]) -> int:
    message = response_message(row)
    total = 0
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str):
            total += len(value)
    details = message.get("reasoning_details")
    if details:
        total += len(json.dumps(details, ensure_ascii=False, sort_keys=True))
    return total


WORD_RE = re.compile(r"\b[\w'-]+\b")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def load_generation_rows(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {str(row["output_id"]): row for row in read_jsonl(run_dir / "generations.jsonl")}


def load_vote_rows(run_dir: Path) -> dict[str, dict[tuple[str, bool], dict[str, Any]]]:
    by_pair: dict[str, dict[tuple[str, bool], dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(run_dir / "judge_votes.jsonl"):
        if row.get("success") is not True:
            continue
        pair_uid = str(row.get("pair_uid", ""))
        judge_model = str(row.get("judge_model", ""))
        if not pair_uid or not judge_model:
            continue
        by_pair[pair_uid][(judge_model, bool(row.get("flipped")))] = row
    return by_pair


def exact_familywise_ci(wins: int, total: int, family_size: int, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    tail_alpha = alpha / (2 * max(1, family_size))
    lo = 0.0 if wins == 0 else float(beta.ppf(tail_alpha, wins, total - wins + 1))
    hi = 1.0 if wins == total else float(beta.ppf(1.0 - tail_alpha, wins + 1, total - wins))
    return lo, hi


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def panel_from_votes(
    job: dict[str, Any],
    votes: dict[tuple[str, bool], dict[str, Any]],
    output_a_hash: str,
    output_b_hash: str,
) -> tuple[str, int, int]:
    matching = [
        row
        for row in votes.values()
        if row.get("source_output_a_hash") == output_a_hash
        and row.get("source_output_b_hash") == output_b_hash
    ]
    if len(matching) < EXPECTED_VOTES_PER_PAIR:
        return "incomplete_judging", len(matching), 0

    by_judge: dict[str, list[str]] = defaultdict(list)
    for row in matching:
        by_judge[str(row["judge_model"])].append(str(row.get("winner_condition", "")))

    verdicts = [derive_judge_verdict(values) for values in by_judge.values()]
    resolved_verdicts = [value for value in verdicts if value != "unresolved"]
    if len(resolved_verdicts) < 3:
        return "incomplete_judging", len(matching), len(resolved_verdicts)
    return derive_panel_winner_condition(job, resolved_verdicts), len(matching), len(resolved_verdicts)


def build_pair_rows(manifests: list[Path], *, comparison_suffix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for manifest in manifests:
        run_dir = manifest.parent
        jobs = read_jsonl(manifest)
        generations = load_generation_rows(run_dir)
        votes_by_pair = load_vote_rows(run_dir)
        for job in jobs:
            comparison = str(job.get("comparison", ""))
            if not comparison.endswith(comparison_suffix):
                continue

            output_a_id, output_b_id = side_output_ids(job)
            out_a = generations.get(output_a_id)
            out_b = generations.get(output_b_id)
            audit_base = {
                "run_id": run_dir.name,
                "manifest": str(manifest),
                "actor": job.get("actor", ""),
                "task": job.get("task", ""),
                "comparison": comparison,
                "pair_uid": job.get("pair_uid", ""),
            }
            if out_a is None or out_b is None:
                audit_rows.append({**audit_base, "status": "missing_generation"})
                continue
            if not generation_valid(out_a) or not generation_valid(out_b):
                audit_rows.append({**audit_base, "status": "invalid_generation"})
                continue

            output_a_hash = output_text_fingerprint(out_a)
            output_b_hash = output_text_fingerprint(out_b)
            panel, n_matching_votes, n_judge_verdicts = panel_from_votes(
                job,
                votes_by_pair.get(str(job["pair_uid"]), {}),
                output_a_hash,
                output_b_hash,
            )
            if panel == "incomplete_judging":
                audit_rows.append(
                    {
                        **audit_base,
                        "status": "incomplete_judging",
                        "n_matching_votes": n_matching_votes,
                        "n_judge_verdicts": n_judge_verdicts,
                    }
                )
                continue

            high_out, low_out = condition_output_rows(job, out_a, out_b)
            high_text = str(high_out.get("output_text", ""))
            low_text = str(low_out.get("output_text", ""))
            high_wins = int(panel == HIGH_CONDITION)
            low_wins = int(panel == LOW_CONDITION)
            tie = int(panel == "tie")
            rows.append(
                {
                    "run_id": run_dir.name,
                    "manifest": str(manifest),
                    "actor": job.get("actor", ""),
                    "actor_label": ACTOR_LABEL.get(str(job.get("actor", "")), str(job.get("actor", ""))),
                    "task": job.get("task", ""),
                    "task_label": TASK_LABEL.get(str(job.get("task", "")), str(job.get("task", ""))),
                    "domain": job.get("domain", ""),
                    "comparison": comparison,
                    "pair_uid": job.get("pair_uid", ""),
                    "item_label": job.get("item_label", ""),
                    "panel_winner_condition": panel,
                    "high_win": high_wins,
                    "low_win": low_wins,
                    "tie": tie,
                    "resolved_excluding_ties": high_wins + low_wins,
                    "n_matching_votes": n_matching_votes,
                    "n_judge_verdicts": n_judge_verdicts,
                    "high_output_id": high_out.get("output_id", ""),
                    "low_output_id": low_out.get("output_id", ""),
                    "high_reasoning_tokens": reasoning_tokens(high_out),
                    "low_reasoning_tokens": reasoning_tokens(low_out),
                    "reasoning_tokens_high_minus_low": reasoning_tokens(high_out) - reasoning_tokens(low_out),
                    "high_reasoning_trace_chars": reasoning_trace_char_count(high_out),
                    "low_reasoning_trace_chars": reasoning_trace_char_count(low_out),
                    "reasoning_trace_chars_high_minus_low": reasoning_trace_char_count(high_out)
                    - reasoning_trace_char_count(low_out),
                    "high_completion_tokens": usage_int(high_out, "completion_tokens"),
                    "low_completion_tokens": usage_int(low_out, "completion_tokens"),
                    "completion_tokens_high_minus_low": usage_int(high_out, "completion_tokens")
                    - usage_int(low_out, "completion_tokens"),
                    "high_words": word_count(high_text),
                    "low_words": word_count(low_text),
                    "words_high_minus_low": word_count(high_text) - word_count(low_text),
                    "high_finish_reason": high_out.get("finish_reason", ""),
                    "low_finish_reason": low_out.get("finish_reason", ""),
                }
            )
            audit_rows.append({**audit_base, "status": "included", "n_matching_votes": n_matching_votes})

    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def summarize_model_task(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = pairs.groupby(["actor", "actor_label", "task", "task_label"], dropna=False, sort=True)
    family_size = grouped.ngroups
    p_values: list[float] = []
    for (actor, actor_label, task, task_label), sub in grouped:
        wins = int(sub["high_win"].sum())
        losses = int(sub["low_win"].sum())
        ties = int(sub["tie"].sum())
        n_excl_ties = wins + losses
        win_rate = wins / n_excl_ties if n_excl_ties else math.nan
        ci_lo, ci_hi = exact_familywise_ci(wins, n_excl_ties, family_size)
        p_value = float(binomtest(wins, n_excl_ties, 0.5, alternative="greater").pvalue) if n_excl_ties else math.nan
        p_values.append(1.0 if math.isnan(p_value) else p_value)
        rows.append(
            {
                "actor": actor,
                "actor_label": actor_label,
                "task": task,
                "task_label": task_label,
                "pairs": int(len(sub)),
                "high_wins": wins,
                "low_wins": losses,
                "ties": ties,
                "n_excluding_ties": n_excl_ties,
                "high_win_rate_excluding_ties": win_rate,
                "fwer95_ci_low": ci_lo,
                "fwer95_ci_high": ci_hi,
                "one_sided_binomial_p": p_value,
                "mean_high_reasoning_tokens": float(sub["high_reasoning_tokens"].mean()),
                "mean_low_reasoning_tokens": float(sub["low_reasoning_tokens"].mean()),
                "mean_reasoning_tokens_high_minus_low": float(sub["reasoning_tokens_high_minus_low"].mean()),
                "mean_high_reasoning_trace_chars": float(sub["high_reasoning_trace_chars"].mean()),
                "mean_low_reasoning_trace_chars": float(sub["low_reasoning_trace_chars"].mean()),
                "mean_reasoning_trace_chars_high_minus_low": float(
                    sub["reasoning_trace_chars_high_minus_low"].mean()
                ),
                "mean_words_high_minus_low": float(sub["words_high_minus_low"].mean()),
            }
        )
    adjusted = holm_adjust(p_values)
    for row, p_adjusted in zip(rows, adjusted, strict=True):
        row["holm_p"] = p_adjusted
        row["holm_positive"] = bool(row["holm_p"] < 0.05)
    return pd.DataFrame(rows).sort_values(["task", "actor"]).reset_index(drop=True)


def summarize_overall(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    specs = [
        ("overall", []),
        ("by_task", ["task", "task_label"]),
        ("by_actor", ["actor", "actor_label"]),
        ("by_domain", ["domain"]),
    ]
    for scope, cols in specs:
        grouped = [((), pairs)] if not cols else list(pairs.groupby(cols, dropna=False, sort=True))
        for key, sub in grouped:
            key_values = key if isinstance(key, tuple) else (key,)
            row = {"scope": scope}
            for col, value in zip(cols, key_values, strict=True):
                row[col] = value
            wins = int(sub["high_win"].sum())
            losses = int(sub["low_win"].sum())
            ties = int(sub["tie"].sum())
            n_excl_ties = wins + losses
            ci_lo, ci_hi = exact_familywise_ci(wins, n_excl_ties, 1)
            row.update(
                {
                    "pairs": int(len(sub)),
                    "high_wins": wins,
                    "low_wins": losses,
                    "ties": ties,
                    "n_excluding_ties": n_excl_ties,
                    "high_win_rate_excluding_ties": wins / n_excl_ties if n_excl_ties else math.nan,
                    "exact95_ci_low": ci_lo,
                    "exact95_ci_high": ci_hi,
                    "net_score_all_pairs": (wins - losses) / len(sub) if len(sub) else math.nan,
                    "mean_high_reasoning_tokens": float(sub["high_reasoning_tokens"].mean()),
                    "mean_low_reasoning_tokens": float(sub["low_reasoning_tokens"].mean()),
                    "mean_reasoning_tokens_high_minus_low": float(sub["reasoning_tokens_high_minus_low"].mean()),
                    "mean_high_reasoning_trace_chars": float(sub["high_reasoning_trace_chars"].mean()),
                    "mean_low_reasoning_trace_chars": float(sub["low_reasoning_trace_chars"].mean()),
                    "mean_reasoning_trace_chars_high_minus_low": float(
                        sub["reasoning_trace_chars_high_minus_low"].mean()
                    ),
                    "mean_words_high_minus_low": float(sub["words_high_minus_low"].mean()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, *, floatfmt: str = ".3f") -> str:
    """Render a small Markdown table without pandas' optional tabulate dependency."""

    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else format(float(value), floatfmt))
        elif pd.api.types.is_bool_dtype(show[col]):
            show[col] = show[col].map(lambda value: "TRUE" if bool(value) else "FALSE")
        else:
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(col) for col in show.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in show.columns) + " |")
    return "\n".join(lines)


def write_summary_md(
    *,
    pairs: pd.DataFrame,
    audit: pd.DataFrame,
    model_task: pd.DataFrame,
    overall: pd.DataFrame,
    path: Path,
) -> None:
    included = int(len(pairs))
    audit_counts = audit["status"].value_counts().to_dict() if not audit.empty else {}
    complete_cells = int(len(model_task)) if not model_task.empty else 0
    holm_positive = int(model_task["holm_positive"].sum()) if not model_task.empty else 0
    lines = [
        "# High-Low Reasoning-Medium Analysis",
        "",
        "Exploratory analysis of high-low utility runs with actor reasoning enabled.",
        "These outputs are not part of the canonical reasoning-off analysis.",
        "",
        "Counting rule: all six order-specific judge votes are required for a pair; each judge's two order votes are collapsed to one judge verdict, then the three judge verdicts determine the panel outcome.",
        "",
        f"- included pairs: {included}",
        f"- model-task cells: {complete_cells}",
        f"- Holm-positive model-task cells: {holm_positive}",
        f"- audit statuses: `{audit_counts}`",
        "",
        "## Overall",
        "",
    ]
    if not overall.empty:
        show = overall[overall["scope"].eq("overall")].copy()
        lines.append(markdown_table(show, floatfmt=".3f"))
    else:
        lines.append("_No included pairs yet._")
    lines.extend(["", "## Model x Task", ""])
    if not model_task.empty:
        cols = [
            "actor_label",
            "task_label",
            "pairs",
            "high_wins",
            "low_wins",
            "ties",
            "high_win_rate_excluding_ties",
            "fwer95_ci_low",
            "fwer95_ci_high",
            "holm_p",
            "holm_positive",
            "mean_reasoning_tokens_high_minus_low",
            "mean_reasoning_trace_chars_high_minus_low",
            "mean_words_high_minus_low",
        ]
        lines.append(markdown_table(model_task[cols], floatfmt=".3f"))
    else:
        lines.append("_No complete model-task cells yet._")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-list", type=Path, action="append", help="Manifest-list TSV to analyze.")
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated task ids to include, or all. Example: essay",
    )
    parser.add_argument(
        "--comparison-suffix",
        default="_highlow_reasoning_medium",
        help="Only analyze jobs whose comparison key ends with this suffix.",
    )
    parser.add_argument(
        "--out-prefix",
        default=OUT_PREFIX,
        help="Prefix for output files in outputs/analysis.",
    )
    args = parser.parse_args()

    manifests = manifest_paths(args)
    if not manifests:
        raise FileNotFoundError(f"No manifest lists found matching {RUNS / DEFAULT_MANIFEST_GLOB}")

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    pairs, audit = build_pair_rows(manifests, comparison_suffix=args.comparison_suffix)
    model_task = summarize_model_task(pairs)
    overall = summarize_overall(pairs)

    prefix = output_prefix(manifests, args)
    pair_path = ANALYSIS / f"{prefix}_pair_outcomes.csv"
    audit_path = ANALYSIS / f"{prefix}_audit.csv"
    model_task_path = ANALYSIS / f"{prefix}_model_task.csv"
    overall_path = ANALYSIS / f"{prefix}_summary.csv"
    summary_path = ANALYSIS / f"{prefix}_summary.md"

    pairs.to_csv(pair_path, index=False)
    audit.to_csv(audit_path, index=False)
    model_task.to_csv(model_task_path, index=False)
    overall.to_csv(overall_path, index=False)
    write_summary_md(pairs=pairs, audit=audit, model_task=model_task, overall=overall, path=summary_path)

    print(f"pairs: {pair_path}")
    print(f"audit: {audit_path}")
    print(f"model_task: {model_task_path}")
    print(f"summary: {summary_path}")
    if not model_task.empty:
        print(
            f"Holm-positive model-task cells: "
            f"{int(model_task['holm_positive'].sum())}/{len(model_task)}"
        )


if __name__ == "__main__":
    main()
