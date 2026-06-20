#!/usr/bin/env python3
"""Export readable paired examples from completed generation/judging runs."""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from utility_behavior_gap.io_utils import read_jsonl
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API


RUNS_DIR = OUTPUT_API / "runs"


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "all"


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def condition_roles(job: dict[str, Any]) -> tuple[str, str]:
    """Return the strong/predicted condition and weak/comparison condition."""

    predicted = str(job.get("predicted_condition") or "")
    other = str(job.get("other_condition") or "")
    conditions = {str(job.get("condition_a") or ""), str(job.get("condition_b") or "")}
    conditions.discard("")

    if predicted and other:
        return predicted, other

    strong_like = [condition for condition in conditions if "strong" in condition or condition == "high"]
    weak_like = [
        condition
        for condition in conditions
        if "normal" in condition or "weak" in condition or condition == "low"
    ]
    if strong_like and weak_like:
        return sorted(strong_like)[0], sorted(weak_like)[0]

    condition_a = str(job.get("condition_a") or "")
    condition_b = str(job.get("condition_b") or "")
    return condition_a, condition_b


def panel_winner(votes: list[dict[str, Any]]) -> str:
    successful = [
        str(vote.get("winner_condition") or "")
        for vote in votes
        if vote.get("success") is not False and vote.get("winner_condition") != "unresolved"
    ]
    if not successful:
        return "no_majority"
    counts = Counter(successful)
    top_count = max(counts.values())
    top = [condition for condition, count in counts.items() if count == top_count]
    if len(top) == 1:
        return top[0]
    return "tie"


def winner_label(winner_condition: str, strong_condition: str, weak_condition: str) -> str:
    if winner_condition == strong_condition:
        return "strong"
    if winner_condition == weak_condition:
        return "weak"
    if winner_condition == "tie":
        return "tie"
    return "no-majority"


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)?|\d+(?:\.\d+)?%?", text))


def matching_run_dirs(
    *,
    run_dir: Path | None,
    actor: str | None,
    task: str | None,
    comparison: str | None,
    all_runs: bool,
) -> list[Path]:
    if run_dir is not None:
        return [run_dir]

    matches: list[Path] = []
    for candidate in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
        jobs_path = candidate / "generation_jobs.jsonl"
        if not jobs_path.exists():
            continue
        jobs = read_jsonl_if_exists(jobs_path)
        if not jobs:
            continue
        if actor and not any(str(job.get("actor")) == actor for job in jobs):
            continue
        if task and not any(str(job.get("task")) == task for job in jobs):
            continue
        if comparison and not any(str(job.get("comparison")) == comparison for job in jobs):
            continue
        matches.append(candidate)

    if all_runs or len(matches) <= 1:
        return matches
    return [max(matches, key=lambda path: ((path / "generation_jobs.jsonl").stat().st_mtime, path.name))]


def collect_examples(
    run_dirs: list[Path],
    *,
    actor: str | None,
    task: str | None,
    comparison: str | None,
    winner: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        jobs = read_jsonl_if_exists(run_dir / "generation_jobs.jsonl")
        generations = read_jsonl_if_exists(run_dir / "generations.jsonl")
        votes = read_jsonl_if_exists(run_dir / "judge_votes.jsonl")

        generation_by_output_id = {
            str(row.get("output_id")): row
            for row in generations
            if row.get("success") is not False and str(row.get("output_text") or "").strip()
        }
        votes_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for vote in votes:
            votes_by_pair[str(vote.get("pair_uid"))].append(vote)

        for job in jobs:
            if actor and str(job.get("actor")) != actor:
                continue
            if task and str(job.get("task")) != task:
                continue
            if comparison and str(job.get("comparison")) != comparison:
                continue
            strong_condition, weak_condition = condition_roles(job)
            out_a = generation_by_output_id.get(f"{job['pair_uid']}::a")
            out_b = generation_by_output_id.get(f"{job['pair_uid']}::b")
            if out_a is None or out_b is None:
                continue
            output_by_condition = {
                str(out_a.get("condition")): out_a,
                str(out_b.get("condition")): out_b,
            }
            if strong_condition not in output_by_condition or weak_condition not in output_by_condition:
                continue
            win_condition = panel_winner(votes_by_pair.get(str(job["pair_uid"]), []))
            label = winner_label(win_condition, strong_condition, weak_condition)
            if winner != "any" and label != winner:
                continue
            examples.append(
                {
                    "run_dir": run_dir,
                    "job": job,
                    "strong_condition": strong_condition,
                    "weak_condition": weak_condition,
                    "strong_output": output_by_condition[strong_condition],
                    "weak_output": output_by_condition[weak_condition],
                    "winner_condition": win_condition,
                    "winner_label": label,
                    "votes": votes_by_pair.get(str(job["pair_uid"]), []),
                }
            )
    return examples


def sorted_examples(examples: list[dict[str, Any]], *, order: str, seed: int) -> list[dict[str, Any]]:
    rows = list(examples)
    if order == "random":
        rng = random.Random(seed)
        rng.shuffle(rows)
        return rows
    if order == "shorter-strong-first":
        return sorted(
            rows,
            key=lambda row: word_count(row["strong_output"]["output_text"])
            - word_count(row["weak_output"]["output_text"]),
        )
    if order == "longer-strong-first":
        return sorted(
            rows,
            key=lambda row: word_count(row["strong_output"]["output_text"])
            - word_count(row["weak_output"]["output_text"]),
            reverse=True,
        )
    return sorted(rows, key=lambda row: str(row["job"].get("pair_uid")))


def prompt_block(label: str, system_prompt: str, user_prompt: str) -> str:
    parts = [f"#### {label} Prompt"]
    if system_prompt:
        parts.extend(["", "**system**", "", "```text", system_prompt, "```"])
    parts.extend(["", "**user**", "", "```text", user_prompt, "```"])
    return "\n".join(parts)


def vote_summary(votes: list[dict[str, Any]], strong_condition: str, weak_condition: str) -> str:
    if not votes:
        return "_No judge votes found._"
    lines = []
    for vote in sorted(votes, key=lambda row: (int(row.get("judge_index") or 0), str(row.get("judge_model")))):
        condition = str(vote.get("winner_condition") or "")
        label = winner_label(condition, strong_condition, weak_condition)
        lines.append(
            f"- judge {vote.get('judge_index', '')}: `{vote.get('judge_model', '')}` -> "
            f"`{label}` (`{condition}`; raw `{vote.get('vote_raw', '')}`)"
        )
    return "\n".join(lines)


def markdown_report(examples: list[dict[str, Any]], *, args: argparse.Namespace) -> str:
    lines = [
        "# Pair Examples",
        "",
        f"- actor filter: `{args.actor or 'any'}`",
        f"- task filter: `{args.task or 'any'}`",
        f"- comparison filter: `{args.comparison or 'any'}`",
        f"- winner filter: `{args.winner}`",
        f"- examples shown: `{len(examples)}`",
        "",
    ]
    for idx, example in enumerate(examples, start=1):
        job = example["job"]
        strong_output = example["strong_output"]
        weak_output = example["weak_output"]
        strong_words = word_count(strong_output["output_text"])
        weak_words = word_count(weak_output["output_text"])
        lines.extend(
            [
                f"## Example {idx}",
                "",
                f"- run: `{example['run_dir'].name}`",
                f"- pair_uid: `{job['pair_uid']}`",
                f"- actor: `{job.get('actor', '')}`",
                f"- task: `{job.get('task', '')}`",
                f"- comparison: `{job.get('comparison', '')}`",
                f"- item/topic: `{job.get('item_label') or job.get('item_id') or ''}`",
                f"- panel winner: `{example['winner_label']}` (`{example['winner_condition']}`)",
                f"- word counts: strong `{strong_words}`, weak `{weak_words}`, delta `{strong_words - weak_words}`",
                "",
                "### Judge Votes",
                "",
                vote_summary(example["votes"], example["strong_condition"], example["weak_condition"]),
                "",
                prompt_block(
                    "Strong",
                    str(job.get("system_prompt_a" if job.get("condition_a") == example["strong_condition"] else "system_prompt_b", "")),
                    str(job.get("prompt_a" if job.get("condition_a") == example["strong_condition"] else "prompt_b", "")),
                ),
                "",
                "#### Strong Output",
                "",
                "```text",
                str(strong_output["output_text"]),
                "```",
                "",
                prompt_block(
                    "Weak",
                    str(job.get("system_prompt_a" if job.get("condition_a") == example["weak_condition"] else "system_prompt_b", "")),
                    str(job.get("prompt_a" if job.get("condition_a") == example["weak_condition"] else "prompt_b", "")),
                ),
                "",
                "#### Weak Output",
                "",
                "```text",
                str(weak_output["output_text"]),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def default_out_path(args: argparse.Namespace) -> Path:
    name = "__".join(
        [
            "pair_examples",
            slug(args.actor or "any_actor"),
            slug(args.task or "any_task"),
            slug(args.comparison or "any_comparison"),
            f"winner-{slug(args.winner)}",
        ]
    )
    return ANALYSIS / f"{name}.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", help="Actor/model key, e.g. glm-5.1-or.")
    parser.add_argument("--task", help="Task key, e.g. essay.")
    parser.add_argument("--comparison", help="Comparison key, e.g. essay_direct_user_prompt_max_effort_full_topics.")
    parser.add_argument("--winner", choices=["strong", "weak", "tie", "no-majority", "any"], default="strong")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--order",
        choices=["pair_uid", "random", "shorter-strong-first", "longer-strong-first"],
        default="pair_uid",
    )
    parser.add_argument("--run-dir", type=Path, help="Use one specific run directory instead of discovering runs.")
    parser.add_argument("--all-runs", action="store_true", help="Include all matching runs instead of latest match.")
    parser.add_argument("--out", type=Path, help="Markdown output path. Defaults to outputs/analysis.")
    args = parser.parse_args()

    run_dirs = matching_run_dirs(
        run_dir=args.run_dir,
        actor=args.actor,
        task=args.task,
        comparison=args.comparison,
        all_runs=args.all_runs,
    )
    if not run_dirs:
        raise SystemExit("No matching run directories found.")

    examples = collect_examples(
        run_dirs,
        actor=args.actor,
        task=args.task,
        comparison=args.comparison,
        winner=args.winner,
    )
    examples = sorted_examples(examples, order=args.order, seed=args.seed)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit("No matching examples found.")

    out_path = args.out or default_out_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown_report(examples, args=args), encoding="utf-8")

    print(f"wrote {len(examples)} examples to {out_path}")
    print("run_dirs:")
    for run_dir in run_dirs:
        print(f"- {run_dir}")


if __name__ == "__main__":
    main()
