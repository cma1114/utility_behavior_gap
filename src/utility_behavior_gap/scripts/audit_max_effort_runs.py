#!/usr/bin/env python3
"""Audit completed 200-pair max-effort essay runs for length and mapping bugs."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import statsmodels.api as sm

from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl, write_csv_rows
from utility_behavior_gap.job_builder import build_generation_jobs
from utility_behavior_gap.judging import derive_counted_winner_condition, derive_panel_winner_condition
from utility_behavior_gap.openrouter import judge_model_ids
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API
from utility_behavior_gap.scripts.run_judging import parse_winner


RUNS = [
    {
        "run": "qwen_system",
        "comparison": "essay_direct_system_prompt_max_effort_full_topics",
        "actor": "qwen3.5-9b-or",
        "strong": "sys_strong",
        "weak": "sys_normal",
    },
    {
        "run": "qwen_user",
        "comparison": "essay_direct_user_prompt_max_effort_full_topics",
        "actor": "qwen3.5-9b-or",
        "strong": "user_strong",
        "weak": "user_normal",
    },
    {
        "run": "glm_user",
        "comparison": "essay_direct_user_prompt_max_effort_full_topics",
        "actor": "glm-5.1-or",
        "strong": "user_strong",
        "weak": "user_normal",
    },
    {
        "run": "deepseek_user",
        "comparison": "essay_direct_user_prompt_max_effort_full_topics",
        "actor": "deepseek-v3.2-or",
        "strong": "user_strong",
        "weak": "user_normal",
    },
]

GENERATIONS = OUTPUT_API / "generations.jsonl"
JUDGE_VOTES = OUTPUT_API / "judge_votes.jsonl"

BUG_AUDIT_CSV = ANALYSIS / "max_effort_bug_audit.csv"
LENGTH_SUMMARY_CSV = ANALYSIS / "max_effort_length_summary.csv"
POSITION_SUMMARY_CSV = ANALYSIS / "max_effort_position_summary.csv"
LENGTH_BIN_CSV = ANALYSIS / "max_effort_length_bins.csv"
LENGTH_ADJUSTED_CSV = ANALYSIS / "max_effort_length_adjusted.csv"

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
INSTRUCTION_LEAK_RE = re.compile(
    r"do not produce a default|merely adequate essay|maximum care and effort|"
    r"maximize the final essay|complete the request|high-stakes quality task",
    flags=re.IGNORECASE,
)
BLINDING_LEAK_RE = re.compile(
    r"\$1,?000|donat(?:e|ed|ion)|sponsoring organization|sponsor|prize",
    flags=re.IGNORECASE,
)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def latest_generations(path: Path) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    latest: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for row in read_jsonl(path):
        output_id = row.get("output_id")
        if not output_id:
            continue
        if row.get("success") is False or not str(row.get("output_text", "")).strip():
            continue
        counts[str(output_id)] += 1
        latest[str(output_id)] = row
    return latest, counts


def reconstruct_run(
    run: dict[str, str],
    generations: dict[str, dict[str, Any]],
    generation_counts: Counter[str],
    vote_log: list[dict[str, Any]],
    judges: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = build_generation_jobs(
        comparisons={run["comparison"]},
        tasks={"essay"},
        actors={run["actor"]},
    )
    pair_ids = {job["pair_uid"] for job in jobs}
    expected_output_ids = {f"{pair_uid}::{suffix}" for pair_uid in pair_ids for suffix in ("a", "b")}
    run_generations = {output_id: generations.get(output_id) for output_id in expected_output_ids}

    condition_mismatches = 0
    pair_hashes: dict[str, tuple[str, str]] = {}
    for job in jobs:
        for suffix, condition_field in (("a", "condition_a"), ("b", "condition_b")):
            output = run_generations.get(f"{job['pair_uid']}::{suffix}")
            if output and output.get("condition") != job[condition_field]:
                condition_mismatches += 1
        out_a = run_generations.get(f"{job['pair_uid']}::a")
        out_b = run_generations.get(f"{job['pair_uid']}::b")
        if out_a and out_b:
            pair_hashes[job["pair_uid"]] = (
                output_text_fingerprint(out_a),
                output_text_fingerprint(out_b),
            )

    latest_vote: dict[tuple[str, str], dict[str, Any]] = {}
    stale_votes = 0
    failed_current_votes = 0
    current_attempts = 0
    parse_mismatches = 0
    display_mismatches = 0
    winner_mismatches = 0

    for vote in vote_log:
        pair_uid = str(vote.get("pair_uid", ""))
        judge_model = str(vote.get("judge_model", ""))
        if pair_uid not in pair_ids or judge_model not in judges:
            continue
        expected_hashes = pair_hashes.get(pair_uid)
        if expected_hashes is None:
            continue
        if (
            vote.get("source_output_a_hash") != expected_hashes[0]
            or vote.get("source_output_b_hash") != expected_hashes[1]
        ):
            stale_votes += 1
            continue
        current_attempts += 1
        if vote.get("success") is False:
            failed_current_votes += 1
            continue
        latest_vote[(pair_uid, judge_model)] = vote

        if parse_winner(str(vote.get("vote_raw", ""))) != vote.get("parsed_winner"):
            parse_mismatches += 1

        out_a = run_generations.get(f"{pair_uid}::a")
        out_b = run_generations.get(f"{pair_uid}::b")
        if not out_a or not out_b:
            continue
        displayed = (vote.get("displayed_output_a_id"), vote.get("displayed_output_b_id"))
        expected_displayed = (
            (out_b["output_id"], out_a["output_id"])
            if vote.get("flipped")
            else (out_a["output_id"], out_b["output_id"])
        )
        if displayed != expected_displayed:
            display_mismatches += 1
        displayed_a = out_b if vote.get("flipped") else out_a
        displayed_b = out_a if vote.get("flipped") else out_b
        parsed = vote.get("parsed_winner")
        if parsed == "a":
            expected_winner = displayed_a["condition"]
        elif parsed == "b":
            expected_winner = displayed_b["condition"]
        elif parsed == "tie":
            expected_winner = "tie"
        else:
            expected_winner = "unresolved"
        if vote.get("winner_condition") != expected_winner:
            winner_mismatches += 1

    pair_rows: list[dict[str, Any]] = []
    vote_rows: list[dict[str, Any]] = []
    vote_count_by_pair: Counter[int] = Counter()
    for job in jobs:
        pair_uid = job["pair_uid"]
        out_a = run_generations.get(f"{pair_uid}::a")
        out_b = run_generations.get(f"{pair_uid}::b")
        if not out_a or not out_b:
            continue
        votes = [latest_vote[(pair_uid, judge)] for judge in judges if (pair_uid, judge) in latest_vote]
        vote_count_by_pair[len(votes)] += 1
        for vote in votes:
            strong_displayed_x = not bool(vote.get("flipped"))
            vote_rows.append(
                {
                    "run": run["run"],
                    "actor": run["actor"],
                    "comparison": run["comparison"],
                    "judge_model": vote["judge_model"],
                    "strong_displayed_x": str(strong_displayed_x),
                    "winner_condition": vote["winner_condition"],
                    "strong_condition": run["strong"],
                    "weak_condition": run["weak"],
                }
            )
        if len(votes) < len(judges):
            continue
        vote_conditions = [str(vote["winner_condition"]) for vote in votes]
        winner = derive_counted_winner_condition(job, vote_conditions)
        strong_output = out_a if out_a["condition"] == run["strong"] else out_b
        weak_output = out_a if out_a["condition"] == run["weak"] else out_b
        strong_words = word_count(strong_output["output_text"])
        weak_words = word_count(weak_output["output_text"])
        if strong_words > weak_words:
            longer = run["strong"]
        elif weak_words > strong_words:
            longer = run["weak"]
        else:
            longer = "equal"
        pair_rows.append(
            {
                "run": run["run"],
                "actor": run["actor"],
                "comparison": run["comparison"],
                "pair_uid": pair_uid,
                "domain": job.get("domain", ""),
                "topic": job.get("item_label", ""),
                "winner": winner,
                "strong_condition": run["strong"],
                "weak_condition": run["weak"],
                "strong_words": strong_words,
                "weak_words": weak_words,
                "word_diff": strong_words - weak_words,
                "longer_condition": longer,
                "strong_word_cap_violation": not (250 <= strong_words <= 400),
                "weak_word_cap_violation": not (250 <= weak_words <= 400),
                "strong_instruction_leak": bool(INSTRUCTION_LEAK_RE.search(strong_output["output_text"])),
                "weak_instruction_leak": bool(INSTRUCTION_LEAK_RE.search(weak_output["output_text"])),
                "strong_blinding_leak": bool(BLINDING_LEAK_RE.search(strong_output["output_text"])),
                "weak_blinding_leak": bool(BLINDING_LEAK_RE.search(weak_output["output_text"])),
            }
        )

    bug_row = {
        "run": run["run"],
        "actor": run["actor"],
        "comparison": run["comparison"],
        "jobs": len(jobs),
        "missing_generations": sum(1 for output_id in expected_output_ids if run_generations.get(output_id) is None),
        "generation_condition_mismatches": condition_mismatches,
        "duplicate_generation_rows": sum(1 for output_id in expected_output_ids if generation_counts[output_id] > 1),
        "stale_votes_ignored": stale_votes,
        "current_vote_attempts": current_attempts,
        "failed_current_votes": failed_current_votes,
        "latest_success_votes": len(latest_vote),
        "pairs_with_3_votes": vote_count_by_pair[3],
        "pairs_with_2_votes": vote_count_by_pair[2],
        "pairs_with_1_vote": vote_count_by_pair[1],
        "parse_mismatches": parse_mismatches,
        "display_mismatches": display_mismatches,
        "winner_mismatches": winner_mismatches,
    }
    return bug_row, pair_rows, vote_rows


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else float("nan")


def summarize_lengths(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name in sorted({row["run"] for row in pair_rows}):
        run_rows = [row for row in pair_rows if row["run"] == run_name]
        strong = run_rows[0]["strong_condition"]
        weak = run_rows[0]["weak_condition"]
        counts = Counter(row["winner"] for row in run_rows)
        resolved = counts[strong] + counts[weak]
        rows.append(
            {
                "run": run_name,
                "actor": run_rows[0]["actor"],
                "comparison": run_rows[0]["comparison"],
                "pairs": len(run_rows),
                "strong_wins": counts[strong],
                "weak_wins": counts[weak],
                "ties": counts["tie"],
                "strong_win_rate_ties_excluded": pct(counts[strong], resolved),
                "mean_strong_words": mean(row["strong_words"] for row in run_rows),
                "mean_weak_words": mean(row["weak_words"] for row in run_rows),
                "mean_word_diff": mean(row["word_diff"] for row in run_rows),
                "median_word_diff": median(row["word_diff"] for row in run_rows),
                "strong_longer_pairs": sum(row["longer_condition"] == strong for row in run_rows),
                "weak_longer_pairs": sum(row["longer_condition"] == weak for row in run_rows),
                "equal_length_pairs": sum(row["longer_condition"] == "equal" for row in run_rows),
                "strong_cap_violations": sum(bool(row["strong_word_cap_violation"]) for row in run_rows),
                "weak_cap_violations": sum(bool(row["weak_word_cap_violation"]) for row in run_rows),
                "strong_instruction_leaks": sum(bool(row["strong_instruction_leak"]) for row in run_rows),
                "weak_instruction_leaks": sum(bool(row["weak_instruction_leak"]) for row in run_rows),
                "strong_blinding_leaks": sum(bool(row["strong_blinding_leak"]) for row in run_rows),
                "weak_blinding_leaks": sum(bool(row["weak_blinding_leak"]) for row in run_rows),
            }
        )
    return rows


def summarize_positions(vote_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name in sorted({row["run"] for row in vote_rows}):
        run_rows = [row for row in vote_rows if row["run"] == run_name]
        strong = run_rows[0]["strong_condition"]
        weak = run_rows[0]["weak_condition"]
        shares = []
        for strong_as_x in (True, False):
            sub = [row for row in run_rows if row["strong_displayed_x"] == str(strong_as_x)]
            counts = Counter(row["winner_condition"] for row in sub)
            resolved = counts[strong] + counts[weak]
            share = pct(counts[strong], resolved)
            shares.append(share)
            rows.append(
                {
                    "run": run_name,
                    "actor": run_rows[0]["actor"],
                    "strong_displayed_as": "X" if strong_as_x else "Y",
                    "votes": len(sub),
                    "strong_vote_count": counts[strong],
                    "weak_vote_count": counts[weak],
                    "tie_vote_count": counts["tie"],
                    "strong_vote_rate_ties_excluded": share,
                    "position_balanced_strong_vote_rate": "",
                }
            )
        rows.append(
            {
                "run": run_name,
                "actor": run_rows[0]["actor"],
                "strong_displayed_as": "balanced_average",
                "votes": len(run_rows),
                "strong_vote_count": "",
                "weak_vote_count": "",
                "tie_vote_count": "",
                "strong_vote_rate_ties_excluded": "",
                "position_balanced_strong_vote_rate": sum(shares) / 2.0,
            }
        )
    return rows


def summarize_length_bins(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = [(-999, -21, "<= -21"), (-20, -1, "-20..-1"), (0, 0, "0"), (1, 20, "1..20"), (21, 999, ">= 21")]
    rows: list[dict[str, Any]] = []
    for run_name in sorted({row["run"] for row in pair_rows if row["comparison"] == "essay_direct_user_prompt_max_effort_full_topics"}):
        run_rows = [row for row in pair_rows if row["run"] == run_name]
        strong = run_rows[0]["strong_condition"]
        weak = run_rows[0]["weak_condition"]
        for lo, hi, label in bins:
            sub = [row for row in run_rows if lo <= int(row["word_diff"]) <= hi]
            counts = Counter(row["winner"] for row in sub)
            resolved = counts[strong] + counts[weak]
            rows.append(
                {
                    "run": run_name,
                    "word_diff_bin": label,
                    "pairs": len(sub),
                    "mean_word_diff": mean(row["word_diff"] for row in sub) if sub else "",
                    "strong_wins": counts[strong],
                    "weak_wins": counts[weak],
                    "ties": counts["tie"],
                    "strong_win_rate_ties_excluded": pct(counts[strong], resolved),
                }
            )
    return rows


def length_adjusted_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    specs = [
        ("user_full_topic_runs", [row for row in pair_rows if row["comparison"] == "essay_direct_user_prompt_max_effort_full_topics"]),
        ("all_200_pair_max_effort_runs", pair_rows),
    ]
    for label, rows in specs:
        resolved = [row for row in rows if row["winner"] in {row["strong_condition"], row["weak_condition"]}]
        run_names = sorted({row["run"] for row in resolved})
        y = [1 if row["winner"] == row["strong_condition"] else 0 for row in resolved]
        x = [
            [1, row["word_diff"]] + [1 if row["run"] == run_name else 0 for run_name in run_names[1:]]
            for row in resolved
        ]
        model = sm.Logit(y, x).fit(disp=False)
        conf = model.conf_int()
        length_coef = float(model.params[1])
        out.append(
            {
                "model": label,
                "term": "word_diff",
                "estimate": length_coef,
                "odds_ratio": math.exp(length_coef),
                "ci_low_or": math.exp(float(conf[1][0])),
                "ci_high_or": math.exp(float(conf[1][1])),
                "predicted_strong_win_at_equal_length": "",
            }
        )
        for run_name in run_names:
            row = [1, 0] + [1 if run_name == name else 0 for name in run_names[1:]]
            linear = sum(float(a) * float(b) for a, b in zip(row, model.params))
            probability = 1 / (1 + math.exp(-linear))
            out.append(
                {
                    "model": label,
                    "term": f"equal_length_prediction:{run_name}",
                    "estimate": "",
                    "odds_ratio": "",
                    "ci_low_or": "",
                    "ci_high_or": "",
                    "predicted_strong_win_at_equal_length": 100.0 * probability,
                }
            )
    return out


def main() -> None:
    generations, generation_counts = latest_generations(GENERATIONS)
    vote_log = read_jsonl(JUDGE_VOTES)
    judges = set(judge_model_ids())
    bug_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    vote_rows: list[dict[str, Any]] = []
    for run in RUNS:
        bug_row, run_pair_rows, run_vote_rows = reconstruct_run(
            run,
            generations,
            generation_counts,
            vote_log,
            judges,
        )
        bug_rows.append(bug_row)
        pair_rows.extend(run_pair_rows)
        vote_rows.extend(run_vote_rows)

    write_csv_rows(BUG_AUDIT_CSV, bug_rows)
    write_csv_rows(LENGTH_SUMMARY_CSV, summarize_lengths(pair_rows))
    write_csv_rows(POSITION_SUMMARY_CSV, summarize_positions(vote_rows))
    write_csv_rows(LENGTH_BIN_CSV, summarize_length_bins(pair_rows))
    write_csv_rows(LENGTH_ADJUSTED_CSV, length_adjusted_rows(pair_rows))

    print(f"wrote {BUG_AUDIT_CSV}")
    print(f"wrote {LENGTH_SUMMARY_CSV}")
    print(f"wrote {POSITION_SUMMARY_CSV}")
    print(f"wrote {LENGTH_BIN_CSV}")
    print(f"wrote {LENGTH_ADJUSTED_CSV}")


if __name__ == "__main__":
    main()
