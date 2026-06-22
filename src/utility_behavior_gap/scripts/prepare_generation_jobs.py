#!/usr/bin/env python3
"""Prepare actor-generation jobs for live OpenRouter runs."""

from __future__ import annotations

import argparse

from utility_behavior_gap.job_builder import build_generation_jobs, write_generation_jobs
from utility_behavior_gap.paths import OUTPUT_API


def parse_set(value: str) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparisons",
        default="highlow_main",
        help=(
            "Comma-separated: highlow_main,highlow_same_count,system_prompt,"
            "highlow_intervention,highlow_intervention_gap_stratified,"
            "system_prompt_role,system_prompt_incentive,system_prompt_reputation,"
            "system_prompt_competition,system_prompt_effort,user_prompt_effort,"
            "user_prompt_role,framed_user_prompt_role,user_prompt_status,essay_direct_system_prompt,"
            "essay_direct_system_prompt_effort,essay_direct_system_prompt_max_effort,"
            "essay_direct_system_prompt_full_topics,"
            "essay_direct_system_prompt_effort_full_topics,"
            "essay_direct_system_prompt_max_effort_full_topics,direct_user_prompt_max_effort,"
            "essay_direct_user_prompt_max_effort_full_topics,moral_nolabel,amount"
            ",modgrid_{essay,translation,grant,incident}_highlow_reasoning_medium"
            ",modgrid_{essay,translation,grant,incident}_highlow_reasoning_traces_medium"
            ",modgrid_{essay,translation,grant,incident}_framed_empty"
        ),
    )
    parser.add_argument("--tasks", default="", help="Optional comma-separated task ids.")
    parser.add_argument("--actors", default="", help="Optional comma-separated actor ids.")
    parser.add_argument("--pairs-per-actor-domain", type=int, default=None)
    parser.add_argument("--items-per-task", type=int, default=None)
    parser.add_argument("--moral-pairs", type=int, default=None)
    parser.add_argument("--system-repeats", type=int, default=5)
    parser.add_argument("--amount-repeats", type=int, default=5)
    parser.add_argument("--moral-causes-per-item", type=int, default=5)
    parser.add_argument("--gap-bins", type=int, default=8)
    parser.add_argument("--gap-seed", type=int, default=20260609)
    parser.add_argument(
        "--modgrid-repeat-start",
        type=int,
        default=0,
        help=(
            "For modgrid conditions, emit only repeats with index >= this value "
            "while still assigning utility pairs and moral causes from the full "
            "--system-repeats design. Use 5 with --system-repeats 10 to add the "
            "second half of a planned 10-repeat design."
        ),
    )
    args = parser.parse_args()

    jobs = build_generation_jobs(
        comparisons=parse_set(args.comparisons) or set(),
        tasks=parse_set(args.tasks),
        actors=parse_set(args.actors),
        pairs_per_actor_domain=args.pairs_per_actor_domain,
        items_per_task=args.items_per_task,
        moral_pairs=args.moral_pairs,
        system_repeats=args.system_repeats,
        amount_repeats=args.amount_repeats,
        moral_causes_per_item=args.moral_causes_per_item,
        gap_bins=args.gap_bins,
        gap_seed=args.gap_seed,
        modgrid_repeat_start=args.modgrid_repeat_start,
    )
    manifest_path = write_generation_jobs(jobs)
    print(f"wrote {len(jobs)} generation jobs to {OUTPUT_API / 'generation_jobs.jsonl'}")
    print(f"wrote immutable generation job manifest to {manifest_path}")


if __name__ == "__main__":
    main()
