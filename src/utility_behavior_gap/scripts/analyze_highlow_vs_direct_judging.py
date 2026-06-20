#!/usr/bin/env python3
"""Summarize bridge judgments comparing high-low essays with direct controls."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl, write_csv_rows
from utility_behavior_gap.judging import derive_counted_winner_condition, derive_panel_winner_condition
from utility_behavior_gap.openrouter import judge_model_ids
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API
from utility_behavior_gap.stats import wilson_ci


CURRENT_JOBS = OUTPUT_API / "generation_jobs.jsonl"
CURRENT_GENERATIONS = OUTPUT_API / "generations.jsonl"
CURRENT_JUDGE_VOTES = OUTPUT_API / "judge_votes.jsonl"
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def path_from_jobs(jobs: list[dict[str, Any]], field: str, fallback: Path) -> Path:
    values = {str(job.get(field) or "") for job in jobs}
    values.discard("")
    if len(values) == 1:
        return Path(values.pop())
    return fallback


def load_run(run_dir: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if run_dir is None:
        jobs_path = CURRENT_JOBS
        jobs = read_jsonl(jobs_path)
        generations_path = path_from_jobs(jobs, "run_generations_path", CURRENT_GENERATIONS)
        votes_path = path_from_jobs(jobs, "run_judge_votes_path", CURRENT_JUDGE_VOTES)
    else:
        jobs_path = run_dir / "generation_jobs.jsonl"
        generations_path = run_dir / "generations.jsonl"
        votes_path = run_dir / "judge_votes.jsonl"
        jobs = read_jsonl(jobs_path)
    return jobs, read_jsonl_if_exists(generations_path), read_jsonl_if_exists(votes_path)


def latest_generation_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        output_id = str(row.get("output_id", ""))
        if not output_id:
            continue
        if row.get("success") is False or not str(row.get("output_text", "")).strip():
            continue
        out[output_id] = row
    return out


def pair_hashes(
    jobs: list[dict[str, Any]],
    generations: dict[str, dict[str, Any]],
) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for job in jobs:
        pair_uid = str(job["pair_uid"])
        output_a = generations.get(f"{pair_uid}::a")
        output_b = generations.get(f"{pair_uid}::b")
        if output_a and output_b:
            out[pair_uid] = (output_text_fingerprint(output_a), output_text_fingerprint(output_b))
    return out


def current_votes(
    votes: list[dict[str, Any]],
    hashes: dict[str, tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    judges = set(judge_model_ids())
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for vote in votes:
        if vote.get("success") is False:
            continue
        pair_uid = str(vote.get("pair_uid", ""))
        judge_model = str(vote.get("judge_model", ""))
        if judge_model not in judges:
            continue
        expected = hashes.get(pair_uid)
        if expected is None:
            continue
        if vote.get("source_output_a_hash") != expected[0] or vote.get("source_output_b_hash") != expected[1]:
            continue
        latest[(pair_uid, judge_model)] = vote
    return latest


def build_pair_rows(
    jobs: list[dict[str, Any]],
    generations: dict[str, dict[str, Any]],
    votes: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    judges = judge_model_ids()
    rows: list[dict[str, Any]] = []
    for job in jobs:
        pair_uid = str(job["pair_uid"])
        output_a = generations.get(f"{pair_uid}::a")
        output_b = generations.get(f"{pair_uid}::b")
        if not output_a or not output_b:
            continue
        pair_votes = [votes[(pair_uid, judge)] for judge in judges if (pair_uid, judge) in votes]
        vote_conditions = [str(vote.get("winner_condition", "")) for vote in pair_votes]
        panel_winner = derive_panel_winner_condition(job, vote_conditions)
        counted_winner = derive_counted_winner_condition(job, vote_conditions)
        rows.append(
            {
                "run_id": job.get("run_id", ""),
                "actor": job.get("actor", ""),
                "comparison": job.get("comparison", ""),
                "pair_uid": pair_uid,
                "topic": job.get("item_label", ""),
                "domain": job.get("domain", ""),
                "votes": len(pair_votes),
                "panel_winner": panel_winner,
                "counted_winner": counted_winner,
                "predicted_condition": job.get("predicted_condition", ""),
                "other_condition": job.get("other_condition", ""),
                "condition_a": job.get("condition_a", ""),
                "condition_b": job.get("condition_b", ""),
                "a_words": word_count(str(output_a.get("output_text", ""))),
                "b_words": word_count(str(output_b.get("output_text", ""))),
                "word_diff_a_minus_b": word_count(str(output_a.get("output_text", "")))
                - word_count(str(output_b.get("output_text", ""))),
                "source_highlow_side": job.get("source_highlow_side", ""),
                "source_highlow_trial_index": job.get("source_highlow_trial_index", ""),
                "source_highlow_arm": job.get("source_highlow_arm", ""),
                "source_direct_output_id": job.get("source_direct_output_id", ""),
            }
        )
    return rows


def summary_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        groups[(str(row["run_id"]), "overall")].append(row)
        groups[(str(row["run_id"]), str(row["topic"]))].append(row)
    for (run_id, topic), rows_for_group in sorted(groups.items()):
        predicted = str(rows_for_group[0]["predicted_condition"])
        other = str(rows_for_group[0]["other_condition"])
        counts = Counter(str(row["counted_winner"]) for row in rows_for_group)
        complete = [row for row in rows_for_group if int(row["votes"]) == len(judge_model_ids())]
        resolved_n = counts[predicted] + counts[other]
        rate, ci_low, ci_high = wilson_ci(counts[predicted], resolved_n)
        rows.append(
            {
                "run_id": run_id,
                "actor": rows_for_group[0]["actor"],
                "comparison": rows_for_group[0]["comparison"],
                "topic": topic,
                "pairs_with_both_outputs": len(rows_for_group),
                "pairs_with_all_judge_votes": len(complete),
                "predicted_condition": predicted,
                "other_condition": other,
                "predicted_wins": counts[predicted],
                "other_wins": counts[other],
                "ties": counts["tie"],
                "unresolved_or_incomplete": len(rows_for_group) - counts[predicted] - counts[other] - counts["tie"],
                "rate_ties_excluded": rate,
                "ci_low_ties_excluded": ci_low,
                "ci_high_ties_excluded": ci_high,
                "mean_word_diff_a_minus_b": sum(float(row["word_diff_a_minus_b"]) for row in rows_for_group)
                / len(rows_for_group),
            }
        )
    return rows


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# High-low vs direct-control bridge",
        "",
        "| actor | topic | pairs | complete | predicted wins | other wins | ties | rate excl. ties | 95% CI | mean words A-B |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rate = 100 * float(row["rate_ties_excluded"]) if row["rate_ties_excluded"] == row["rate_ties_excluded"] else float("nan")
        ci_low = 100 * float(row["ci_low_ties_excluded"]) if row["ci_low_ties_excluded"] == row["ci_low_ties_excluded"] else float("nan")
        ci_high = 100 * float(row["ci_high_ties_excluded"]) if row["ci_high_ties_excluded"] == row["ci_high_ties_excluded"] else float("nan")
        lines.append(
            f"| {row['actor']} | {row['topic']} | {row['pairs_with_both_outputs']} | "
            f"{row['pairs_with_all_judge_votes']} | {row['predicted_wins']} | {row['other_wins']} | "
            f"{row['ties']} | {rate:.1f}% | [{ci_low:.1f}, {ci_high:.1f}] | "
            f"{float(row['mean_word_diff_a_minus_b']):.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="Bridge run dir. Defaults to the current manifest.")
    args = parser.parse_args()

    jobs, generation_rows, vote_rows = load_run(args.run_dir)
    generations = latest_generation_by_id(generation_rows)
    hashes = pair_hashes(jobs, generations)
    votes = current_votes(vote_rows, hashes)
    pair_rows = build_pair_rows(jobs, generations, votes)
    if not pair_rows:
        raise ValueError("no pair rows with both source outputs found")
    summaries = summary_rows(pair_rows)

    run_id = str(jobs[0].get("run_id") or (args.run_dir.name if args.run_dir else "current"))
    pair_csv = ANALYSIS / f"{run_id}__pair_results.csv"
    summary_csv = ANALYSIS / f"{run_id}__summary.csv"
    summary_md = ANALYSIS / f"{run_id}__summary.md"
    write_csv_rows(pair_csv, pair_rows)
    write_csv_rows(summary_csv, summaries)
    write_markdown(summary_md, summaries)

    overall = next(row for row in summaries if row["topic"] == "overall")
    print(f"wrote {pair_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_md}")
    print(
        "overall: "
        f"{overall['predicted_wins']} {overall['predicted_condition']} wins, "
        f"{overall['other_wins']} {overall['other_condition']} wins, "
        f"{overall['ties']} ties; "
        f"rate excluding ties={100 * float(overall['rate_ties_excluded']):.1f}% "
        f"[{100 * float(overall['ci_low_ties_excluded']):.1f}, "
        f"{100 * float(overall['ci_high_ties_excluded']):.1f}]"
    )


if __name__ == "__main__":
    main()
