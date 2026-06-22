"""Build live API generation jobs from fixed inputs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from utility_behavior_gap.constants import ACTOR_LABEL, ACTORS, DOMAIN_LABEL, TASK_LABEL
from utility_behavior_gap.io_utils import read_csv_rows, read_jsonl, write_jsonl
from utility_behavior_gap.paths import INPUTS, OUTPUT_API, OUTPUT_INPUTS, ROOT
from utility_behavior_gap.pair_selection import gap_stratified_pairs_for_actor
from utility_behavior_gap.prompts import (
    BLINDING_NON_ESSAY,
    COMPETITION_STRONG_SYSTEM_PROMPTS,
    EFFORT_STRONG_SYSTEM_PROMPTS,
    EFFORT_WEAK_SYSTEM_PROMPTS,
    INCENTIVE_STRONG_SYSTEM_PROMPTS,
    MAX_EFFORT_STRONG_SYSTEM_PROMPTS,
    NORMAL_SYSTEM_PROMPTS,
    MODGRID_NEUTRAL_TARGET,
    REPUTATION_STRONG_SYSTEM_PROMPTS,
    ROLE_ONLY_STRONG_SYSTEM_PROMPTS,
    STRONG_SYSTEM_PROMPTS,
    USER_BARE_INSTRUCTIONS,
    USER_EFFORT_STRONG_INSTRUCTIONS,
    USER_EFFORT_WEAK_INSTRUCTIONS,
    USER_MAX_EFFORT_STRONG_INSTRUCTIONS,
    USER_ROLE_STRONG_INSTRUCTIONS,
    USER_ROLE_WEAK_INSTRUCTIONS,
    USER_STATUS_STRONG_INSTRUCTIONS,
    USER_STATUS_WEAK_INSTRUCTIONS,
    build_amount_prompt,
    build_essay_amount_prompt,
    build_essay_intervention_prompt,
    build_essay_modgrid_prompt,
    build_essay_moral_no_label_prompt,
    build_modgrid_empty_prompt,
    build_modgrid_fund_prompt,
    build_modgrid_prompt,
    build_essay_prompt,
    build_essay_system_prompt,
    build_grant_prompt,
    build_incident_prompt,
    build_intervention_prompt,
    build_moral_no_label_prompt,
    build_translation_prompt,
)


def base_prompt_for_task(task_row: dict[str, str]) -> str:
    if task_row["task"] == "essay":
        return f'Write a compelling, well-structured essay on this topic: "{task_row["item_label"]}"'
    return task_row["base_prompt"]


def normal_system_prompt(task_row: dict[str, str]) -> str:
    return NORMAL_SYSTEM_PROMPTS.get(task_row["task"], "")


def highlow_prompt(task_row: dict[str, str], outcome: str) -> str:
    task = task_row["task"]
    if task == "essay":
        return build_essay_prompt(task_row["item_label"], outcome)
    if task == "translation":
        return build_translation_prompt(task_row["base_prompt"], outcome)
    if task == "incident_postmortem":
        return build_incident_prompt(task_row["item_label"], outcome)
    if task == "grant_proposal_abstract":
        return build_grant_prompt(task_row["item_label"], outcome)
    raise ValueError(f"unknown task: {task}")


@lru_cache(maxsize=1)
def consequence_rewrite_by_original() -> dict[str, dict[str, str]]:
    path = INPUTS / "success_contingent_consequence_rewrites.csv"
    rows = read_csv_rows(path)
    mapping = {row["original_description"]: row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"duplicate original_description in {path}")
    return mapping


def rewritten_consequence(original_description: str) -> str:
    mapping = consequence_rewrite_by_original()
    try:
        return mapping[original_description]["proposed_consequence"]
    except KeyError as exc:
        raise KeyError(f"no rewritten consequence for utility option: {original_description!r}") from exc


SMALL_NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}

TENS_NUMBER_WORDS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def integer_words(value: int) -> str:
    if value < 0 or value > 999_999:
        raise ValueError(f"integer_words only supports 0..999999, got {value}")
    if value < 20:
        return SMALL_NUMBER_WORDS[value]
    if value < 100:
        tens = (value // 10) * 10
        ones = value % 10
        return TENS_NUMBER_WORDS[tens] if ones == 0 else f"{TENS_NUMBER_WORDS[tens]}-{SMALL_NUMBER_WORDS[ones]}"
    if value < 1000:
        hundreds = value // 100
        rest = value % 100
        return (
            f"{SMALL_NUMBER_WORDS[hundreds]} hundred"
            if rest == 0
            else f"{SMALL_NUMBER_WORDS[hundreds]} hundred {integer_words(rest)}"
        )
    thousands = value // 1000
    rest = value % 1000
    return (
        f"{integer_words(thousands)} thousand"
        if rest == 0
        else f"{integer_words(thousands)} thousand {integer_words(rest)}"
    )


NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})*\b|\b\d+\b")


def spell_numbers_in_text(text: str) -> str:
    """Spell numeric counts in utility consequences without changing metadata."""

    def replace(match: re.Match[str]) -> str:
        return integer_words(int(match.group(0).replace(",", "")))

    return NUMBER_RE.sub(replace, text)


def highlow_intervention_prompt(task_row: dict[str, str], outcome: str) -> str:
    consequence = rewritten_consequence(outcome)
    if task_row["task"] == "essay":
        return build_essay_intervention_prompt(task_row["item_label"], consequence)
    return build_intervention_prompt(task_row["task"], task_row["base_prompt"], consequence)


def amount_prompt(task_row: dict[str, str], amount: int) -> str:
    if task_row["task"] == "essay":
        return build_essay_amount_prompt(task_row["item_label"], amount)
    return build_amount_prompt(task_row["base_prompt"], amount)


def moral_prompt(task_row: dict[str, str], cause: str) -> str:
    if task_row["task"] == "essay":
        return build_essay_moral_no_label_prompt(task_row["item_label"], cause)
    return build_moral_no_label_prompt(task_row["base_prompt"], cause)


def system_prompt_pair(
    task_row: dict[str, str],
    *,
    strong_system_prompts: dict[str, str] | None = None,
    normal_system_prompts: dict[str, str] | None = None,
) -> tuple[str, str, str, str]:
    task = task_row["task"]
    strong_system_prompts = strong_system_prompts or STRONG_SYSTEM_PROMPTS
    normal_system_prompts = normal_system_prompts or NORMAL_SYSTEM_PROMPTS
    prompt = (
        build_essay_system_prompt(task_row["item_label"]) + BLINDING_NON_ESSAY
        if task == "essay"
        else task_row["base_prompt"] + BLINDING_NON_ESSAY
    )
    return (
        strong_system_prompts.get(task, ""),
        normal_system_prompts.get(task, ""),
        prompt,
        prompt,
    )


def user_prompt_pair(
    task_row: dict[str, str],
    *,
    strong_user_instructions: dict[str, str],
    normal_user_instructions: dict[str, str],
) -> tuple[str, str, str, str, str, str]:
    task = task_row["task"]
    base_prompt = build_essay_system_prompt(task_row["item_label"]) if task == "essay" else task_row["base_prompt"]
    strong_instruction = strong_user_instructions.get(task, "")
    normal_instruction = normal_user_instructions.get(task, "")
    return (
        "",
        "",
        f"{base_prompt}\n\n{strong_instruction}" if strong_instruction else base_prompt,
        f"{base_prompt}\n\n{normal_instruction}" if normal_instruction else base_prompt,
        strong_instruction,
        normal_instruction,
    )


def prompt_variant_id(*, system_prompt_a: str, system_prompt_b: str, prompt_a: str, prompt_b: str) -> str:
    payload = (
        f"system_prompt_a={system_prompt_a}\n---\n"
        f"system_prompt_b={system_prompt_b}\n---\n"
        f"prompt_a={prompt_a}\n---\n"
        f"prompt_b={prompt_b}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def user_prompt_variant_id(*, prompt_a: str, prompt_b: str) -> str:
    payload = f"prompt_a={prompt_a}\n---\nprompt_b={prompt_b}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def slug_part(value: str, *, max_len: int = 80) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value.strip())
    slug = "-".join(part for part in slug.split("-") if part)
    return (slug or "none")[:max_len]


def run_id_part(jobs: list[dict[str, Any]], field: str) -> str:
    values = sorted({str(job.get(field, "")).strip() for job in jobs if str(job.get(field, "")).strip()})
    if not values:
        return f"no-{field}"
    if len(values) == 1:
        return slug_part(values[0])
    if len(values) <= 3:
        return "+".join(slug_part(value, max_len=35) for value in values)
    return f"{len(values)}-{field}s"


def comparison_run_label(comparison: str, task: str) -> str:
    task_prefix = f"{task}_"
    if comparison.startswith(task_prefix):
        return comparison.removeprefix(task_prefix)
    return comparison


def run_id_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")


def generation_run_id(*, jobs: list[dict[str, Any]], digest: str) -> str:
    raw_comparison = run_id_part(jobs, "comparison")
    actor = run_id_part(jobs, "actor")
    task = run_id_part(jobs, "task")
    comparison = comparison_run_label(raw_comparison, task)
    return "__".join([task, comparison, actor, run_id_timestamp(), f"hash-{digest}"])


def direct_outcome_text(note: str) -> str:
    prefix = "outcome (identical both sides): "
    return note.removeprefix(prefix).strip()


def read_essay_direct_trials(data_dir: Path = ROOT / "essay_all_conditions" / "direct") -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    if not data_dir.exists():
        raise FileNotFoundError(f"missing essay direct trial data: {data_dir}")
    paths = [*sorted(data_dir.glob("*.json")), *sorted(OUTPUT_INPUTS.glob("essay_direct_trials__*.json"))]
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        actor = payload.get("actor_id") or path.stem.removeprefix("essay_direct_trials__")
        actor_label = payload.get("actor_label") or payload.get("actor") or ACTOR_LABEL.get(actor, actor)
        for trial_index, trial in enumerate(payload["trials"]):
            row = dict(trial)
            row["actor"] = actor
            row["actor_label"] = actor_label
            row["trial_index"] = str(trial_index)
            row["outcome"] = direct_outcome_text(str(row.get("note", "")))
            trials.append(row)
    return trials


def read_selected_pairs() -> list[dict[str, str]]:
    path = OUTPUT_INPUTS / "selected_pairs.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python -m utility_behavior_gap.scripts.select_pairs` first.")
    return read_csv_rows(path)


def limited_by_group(rows: list[dict[str, str]], key_fields: tuple[str, ...], limit: int | None) -> list[dict[str, str]]:
    if limit is None:
        return rows
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    kept = []
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if counts[key] >= limit:
            continue
        counts[key] += 1
        kept.append(row)
    return kept


def task_groups(task_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in task_rows:
        copy = dict(row)
        copy["item_index"] = str(len(grouped[row["task"]]))
        grouped[row["task"]].append(copy)
    return grouped


def task_by_label(task_rows: list[dict[str, str]], task: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for idx, row in enumerate(row for row in task_rows if row["task"] == task):
        copy = dict(row)
        copy["item_index"] = str(idx)
        out[copy["item_label"]] = copy
    return out


def task_rows_for_task(task_rows: list[dict[str, str]], task: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for idx, row in enumerate(row for row in task_rows if row["task"] == task):
        copy = dict(row)
        copy["item_index"] = str(idx)
        out.append(copy)
    return out


def unique_essay_direct_outcomes(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for trial in trials:
        key = (
            str(trial.get("actor") or ""),
            str(trial.get("domain") or ""),
            str(trial.get("outcome") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        row = dict(trial)
        row["outcome_index"] = str(len(outcomes))
        outcomes.append(row)
    return outcomes


def clean_essay_direct_prompt(topic: str) -> str:
    """Direct-instruction essay prompt with no utility/outcome scaffold."""

    return build_essay_system_prompt(topic)


def build_generation_jobs(
    *,
    comparisons: set[str],
    tasks: set[str] | None = None,
    actors: set[str] | None = None,
    pairs_per_actor_domain: int | None = None,
    items_per_task: int | None = None,
    moral_pairs: int | None = None,
    system_repeats: int = 5,
    amount_repeats: int = 5,
    moral_causes_per_item: int = 5,
    gap_bins: int = 8,
    gap_seed: int = 20260609,
    modgrid_repeat_start: int = 0,
) -> list[dict[str, Any]]:
    if modgrid_repeat_start < 0:
        raise ValueError("modgrid_repeat_start must be nonnegative")
    if modgrid_repeat_start > system_repeats:
        raise ValueError("modgrid_repeat_start cannot exceed system_repeats")

    all_task_rows = read_csv_rows(INPUTS / "task_items.csv")
    task_rows = all_task_rows
    if tasks is not None:
        task_rows = [row for row in task_rows if row["task"] in tasks]
    task_rows = limited_by_group(task_rows, ("task",), items_per_task)
    by_task = task_groups(task_rows)

    cause_rows = read_csv_rows(INPUTS / "moral_cause_pairs.csv")
    if moral_pairs is not None:
        cause_rows = cause_rows[:moral_pairs]

    jobs: list[dict[str, Any]] = []
    if "highlow_main" in comparisons or "highlow_same_count" in comparisons or "highlow_intervention" in comparisons:
        selected_pairs = read_selected_pairs()
        if actors is not None:
            selected_pairs = [row for row in selected_pairs if row["actor"] in actors]
        selected_pairs = limited_by_group(selected_pairs, ("actor", "domain", "pair_set"), pairs_per_actor_domain)
        for pair in selected_pairs:
            if "highlow_intervention" in comparisons and pair["pair_set"] == "default":
                comparison = "highlow_intervention"
            else:
                comparison = "highlow_same_count" if pair["pair_set"] == "same_count" else "highlow_main"
            if comparison not in comparisons:
                continue

            task_names = ["essay"] if comparison == "highlow_same_count" else sorted(by_task)
            for task_name in task_names:
                task_pool = by_task.get(task_name, [])
                if not task_pool:
                    continue
                task = task_pool[int(pair["pair_idx"]) % len(task_pool)]
                uid = f"{comparison}:{pair['actor']}:{pair['domain']}:{pair['pair_idx']}:{task['task']}:{task['item_index']}"
                system_prompt = normal_system_prompt(task)
                prompt_fn = highlow_intervention_prompt if comparison == "highlow_intervention" else highlow_prompt
                high_consequence = rewritten_consequence(pair["high_description"]) if comparison == "highlow_intervention" else ""
                low_consequence = rewritten_consequence(pair["low_description"]) if comparison == "highlow_intervention" else ""
                jobs.append(
                    common_job(
                        pair_uid=uid,
                        comparison=comparison,
                        actor=pair["actor"],
                        task_row=task,
                        domain=pair["domain"],
                        domain_label=DOMAIN_LABEL[pair["domain"]],
                        pair_idx=pair["pair_idx"],
                        pair_set=pair["pair_set"],
                        category=pair["category"],
                        condition_a="high",
                        condition_b="low",
                        prompt_a=prompt_fn(task, pair["high_description"]),
                        prompt_b=prompt_fn(task, pair["low_description"]),
                        system_prompt_a=system_prompt,
                        system_prompt_b=system_prompt,
                        predicted_condition="high",
                        other_condition="low",
                        high_description=pair["high_description"],
                        low_description=pair["low_description"],
                        high_consequence=high_consequence,
                        low_consequence=low_consequence,
                        high_utility=pair["high_utility"],
                        low_utility=pair["low_utility"],
                        delta_u=pair["delta_u"],
                        framing="funded_intervention" if comparison == "highlow_intervention" else "",
                    )
                )

    if "highlow_intervention_gap_stratified" in comparisons:
        if actors is None or len(actors) != 1:
            raise ValueError("highlow_intervention_gap_stratified requires exactly one actor via --actors")
        if tasks is not None and tasks != {"grant_proposal_abstract"}:
            raise ValueError("highlow_intervention_gap_stratified is currently defined for grant_proposal_abstract only")
        actor = next(iter(actors))
        task_pool = by_task.get("grant_proposal_abstract", [])
        if not task_pool:
            raise ValueError("no grant_proposal_abstract task rows available")
        pairs = gap_stratified_pairs_for_actor(
            actor=actor,
            domains=["religions", "animals", "countries", "political"],
            pairs_per_domain_bin=len(task_pool),
            bins=gap_bins,
            seed=gap_seed,
        )
        for pair in pairs:
            task = task_pool[int(pair["gap_bin_sample_index"]) % len(task_pool)]
            uid = (
                "highlow_intervention_gap_stratified:"
                f"{pair['actor']}:{pair['domain']}:bin{int(pair['gap_bin']):02d}:"
                f"s{pair['gap_bin_sample_index']}:{task['task']}:{task['item_index']}"
            )
            system_prompt = normal_system_prompt(task)
            jobs.append(
                common_job(
                    pair_uid=uid,
                    comparison="highlow_intervention_gap_stratified",
                    actor=pair["actor"],
                    task_row=task,
                    domain=pair["domain"],
                    domain_label=DOMAIN_LABEL[pair["domain"]],
                    pair_idx=pair["pair_idx"],
                    pair_set=pair["pair_set"],
                    category=pair["category"],
                    condition_a="high",
                    condition_b="low",
                    prompt_a=highlow_intervention_prompt(task, pair["high_description"]),
                    prompt_b=highlow_intervention_prompt(task, pair["low_description"]),
                    system_prompt_a=system_prompt,
                    system_prompt_b=system_prompt,
                    predicted_condition="high",
                    other_condition="low",
                    high_description=pair["high_description"],
                    low_description=pair["low_description"],
                    high_consequence=rewritten_consequence(pair["high_description"]),
                    low_consequence=rewritten_consequence(pair["low_description"]),
                    high_utility=pair["high_utility"],
                    low_utility=pair["low_utility"],
                    delta_u=pair["delta_u"],
                    framing="funded_intervention",
                    source_topic_index=str(task["item_index"]),
                    topic_design="all_grant_topics_x_domain_gap_bins",
                    gap_bin=pair["gap_bin"],
                    gap_bin_count=pair["gap_bin_count"],
                    gap_bin_min_delta_u=pair["gap_bin_min_delta_u"],
                    gap_bin_max_delta_u=pair["gap_bin_max_delta_u"],
                    gap_bin_sample_index=pair["gap_bin_sample_index"],
                    gap_sampling_seed=pair["gap_sampling_seed"],
                )
            )

    if "amount" in comparisons:
        for actor in sorted(actors or set(ACTORS)):
            for task_name in sorted(by_task):
                for task in by_task[task_name]:
                    for repeat in range(amount_repeats):
                        uid = f"amount:{actor}:{task['task']}:{task['item_index']}:r{repeat}"
                        system_prompt = normal_system_prompt(task)
                        jobs.append(
                            common_job(
                                pair_uid=uid,
                                comparison="amount",
                                actor=actor,
                                task_row=task,
                                domain="",
                                condition_a="amount_high",
                                condition_b="amount_low",
                                prompt_a=amount_prompt(task, 1000000),
                                prompt_b=amount_prompt(task, 100),
                                system_prompt_a=system_prompt,
                                system_prompt_b=system_prompt,
                                predicted_condition="amount_high",
                                other_condition="amount_low",
                                repeat=str(repeat),
                                amount_high="1000000",
                                amount_low="100",
                            )
                        )

    if "moral_nolabel" in comparisons:
        for actor in sorted(actors or set(ACTORS)):
            for task_name in sorted(by_task):
                for task in by_task[task_name]:
                    for sample_k in range(moral_causes_per_item):
                        cause_idx = (int(task["item_index"]) * moral_causes_per_item + sample_k) % len(cause_rows)
                        cause = cause_rows[cause_idx]
                        uid = f"moral_nolabel:{actor}:{task['task']}:{task['item_index']}:s{sample_k}"
                        system_prompt = normal_system_prompt(task)
                        jobs.append(
                            common_job(
                                pair_uid=uid,
                                comparison="moral_nolabel",
                                actor=actor,
                                task_row=task,
                                domain="",
                                condition_a="moral_good",
                                condition_b="moral_bad",
                                prompt_a=moral_prompt(task, cause["good_text"]),
                                prompt_b=moral_prompt(task, cause["bad_text"]),
                                system_prompt_a=system_prompt,
                                system_prompt_b=system_prompt,
                                predicted_condition="moral_good",
                                other_condition="moral_bad",
                                sample_k=str(sample_k),
                                cause_pair_label=cause["cause_pair_label"],
                                good_text=cause["good_text"],
                                bad_text=cause["bad_text"],
                            )
                        )

    if "direct_user_prompt_max_effort" in comparisons:
        selected_pairs = read_selected_pairs()
        if actors is not None:
            selected_pairs = [row for row in selected_pairs if row["actor"] in actors]
        selected_pairs = [row for row in selected_pairs if row["pair_set"] == "default"]
        selected_pairs = limited_by_group(selected_pairs, ("actor", "domain", "pair_set"), pairs_per_actor_domain)
        for pair in selected_pairs:
            for task_name in sorted(by_task):
                task_pool = by_task.get(task_name, [])
                if not task_pool:
                    continue
                task = task_pool[int(pair["pair_idx"]) % len(task_pool)]
                sys_a, sys_b, prompt_a, prompt_b, _, _ = user_prompt_pair(
                    task,
                    strong_user_instructions=USER_MAX_EFFORT_STRONG_INSTRUCTIONS,
                    normal_user_instructions=USER_EFFORT_WEAK_INSTRUCTIONS,
                )
                variant_id = prompt_variant_id(
                    system_prompt_a=sys_a,
                    system_prompt_b=sys_b,
                    prompt_a=prompt_a,
                    prompt_b=prompt_b,
                )
                uid = (
                    f"direct_user_prompt_max_effort:{pair['actor']}:{pair['domain']}:"
                    f"{pair['pair_idx']}:{task['task']}:{task['item_index']}:v{variant_id}"
                )
                jobs.append(
                    common_job(
                        pair_uid=uid,
                        comparison="direct_user_prompt_max_effort",
                        actor=pair["actor"],
                        task_row=task,
                        domain=pair["domain"],
                        domain_label=DOMAIN_LABEL[pair["domain"]],
                        pair_idx=pair["pair_idx"],
                        pair_set=pair["pair_set"],
                        category=pair["category"],
                        condition_a="user_strong",
                        condition_b="user_normal",
                        prompt_a=prompt_a,
                        prompt_b=prompt_b,
                        system_prompt_a=sys_a,
                        system_prompt_b=sys_b,
                        prompt_variant_id=variant_id,
                        predicted_condition="user_strong",
                        other_condition="user_normal",
                        framing="clean_direct_no_outcome",
                    )
                )

    essay_direct_system_prompt_comparisons: dict[str, tuple[dict[str, str], dict[str, str], bool]] = {
        "essay_direct_system_prompt": (STRONG_SYSTEM_PROMPTS, NORMAL_SYSTEM_PROMPTS, False),
        "essay_direct_system_prompt_effort": (EFFORT_STRONG_SYSTEM_PROMPTS, EFFORT_WEAK_SYSTEM_PROMPTS, False),
        "essay_direct_system_prompt_max_effort": (
            MAX_EFFORT_STRONG_SYSTEM_PROMPTS,
            EFFORT_WEAK_SYSTEM_PROMPTS,
            False,
        ),
        "essay_direct_system_prompt_full_topics": (
            STRONG_SYSTEM_PROMPTS,
            NORMAL_SYSTEM_PROMPTS,
            True,
        ),
        "essay_direct_system_prompt_effort_full_topics": (
            EFFORT_STRONG_SYSTEM_PROMPTS,
            EFFORT_WEAK_SYSTEM_PROMPTS,
            True,
        ),
        "essay_direct_system_prompt_max_effort_full_topics": (
            MAX_EFFORT_STRONG_SYSTEM_PROMPTS,
            EFFORT_WEAK_SYSTEM_PROMPTS,
            True,
        ),
    }
    for comparison, (strong_prompts, normal_prompts, use_full_topic_crossing) in (
        essay_direct_system_prompt_comparisons.items()
    ):
        if comparison not in comparisons:
            continue
        if tasks is not None and "essay" not in tasks:
            continue
        essay_tasks_by_label = task_by_label(all_task_rows, "essay")
        direct_trials = read_essay_direct_trials()
        if actors is not None:
            direct_trials = [row for row in direct_trials if row["actor"] in actors]
        direct_topic_labels = {str(trial["essay_topic"]) for trial in direct_trials}
        direct_essay_tasks: list[dict[str, str]] = []
        for task in task_rows_for_task(all_task_rows, "essay"):
            if task["item_label"] not in direct_topic_labels:
                continue
            copy = dict(task)
            copy["item_index"] = str(len(direct_essay_tasks))
            direct_essay_tasks.append(copy)
        trial_task_pairs: list[tuple[dict[str, Any], dict[str, str], str]] = []
        if use_full_topic_crossing:
            for outcome_trial in unique_essay_direct_outcomes(direct_trials):
                for task in direct_essay_tasks:
                    trial_task_pairs.append((outcome_trial, task, str(task["item_index"])))
        else:
            for trial in direct_trials:
                topic = str(trial["essay_topic"])
                task = essay_tasks_by_label.get(topic)
                if task is None:
                    raise ValueError(f"essay direct trial topic is missing from task_items.csv: {topic}")
                trial_task_pairs.append((trial, task, str(task["item_index"])))

        for trial, task, source_topic_index in trial_task_pairs:
            topic = str(task["item_label"])
            prompt = clean_essay_direct_prompt(topic)
            sys_a = strong_prompts["essay"]
            sys_b = normal_prompts["essay"]
            variant_id = prompt_variant_id(
                system_prompt_a=sys_a,
                system_prompt_b=sys_b,
                prompt_a=prompt,
                prompt_b=prompt,
            )
            if use_full_topic_crossing:
                uid = (
                    f"{comparison}:{trial['actor']}:essay:rep{trial['outcome_index']}:"
                    f"t{source_topic_index}:v{variant_id}"
                )
            else:
                uid = f"{comparison}:{trial['actor']}:essay:trial{trial['trial_index']}:v{variant_id}"
            jobs.append(
                common_job(
                    pair_uid=uid,
                    comparison=comparison,
                    actor=str(trial["actor"]),
                    task_row=task,
                    domain="",
                    domain_label="",
                    condition_a="sys_strong",
                    condition_b="sys_normal",
                    prompt_a=prompt,
                    prompt_b=prompt,
                    system_prompt_a=sys_a,
                    system_prompt_b=sys_b,
                    prompt_variant_id=variant_id,
                    predicted_condition="sys_strong",
                    other_condition="sys_normal",
                    source_trial_index=str(trial["trial_index"]),
                    framing="clean_direct_no_outcome",
                    repeat=str(trial.get("outcome_index", trial["trial_index"])),
                    source_topic_index=source_topic_index,
                    topic_design="full_topics" if use_full_topic_crossing else "source_topics",
                )
            )

    # R0 pilot: does the minimal "Do a good job" suffix itself lift judged
    # quality over the completely bare task prompt? Crosses the FULL essay
    # topic pool (30 topics) with `system_repeats` repeats per topic so that
    # topic — the clustering unit — carries the replication, rather than the
    # 5-topic essay-direct design.
    if "essay_user_prompt_goodjob_vs_bare" in comparisons and (tasks is None or "essay" in tasks):
        for actor in sorted(actors or set(ACTORS)):
            for task in task_rows_for_task(all_task_rows, "essay"):
                topic = str(task["item_label"])
                base_prompt = clean_essay_direct_prompt(topic)
                inst_a = USER_EFFORT_WEAK_INSTRUCTIONS["essay"]
                prompt_a = f"{base_prompt}\n\n{inst_a}"
                prompt_b = base_prompt
                variant_id = prompt_variant_id(
                    system_prompt_a="",
                    system_prompt_b="",
                    prompt_a=prompt_a,
                    prompt_b=prompt_b,
                )
                for rep in range(system_repeats):
                    jobs.append(
                        common_job(
                            pair_uid=(
                                f"essay_user_prompt_goodjob_vs_bare:{actor}:essay:"
                                f"i{task['item_index']}:r{rep}:v{variant_id}"
                            ),
                            comparison="essay_user_prompt_goodjob_vs_bare",
                            actor=actor,
                            task_row=task,
                            domain="",
                            domain_label="",
                            condition_a="user_goodjob",
                            condition_b="user_bare",
                            prompt_a=prompt_a,
                            prompt_b=prompt_b,
                            system_prompt_a="",
                            system_prompt_b="",
                            prompt_variant_id=variant_id,
                            predicted_condition="user_goodjob",
                            other_condition="user_bare",
                            framing="clean_direct_no_outcome",
                            repeat=str(rep),
                            topic_design="task_pool",
                        )
                    )

    # ------------------------------------------------------------------
    # Designed modulation experiment (protocol notes file). Contrasts on the
    # unified two-slot wrapper, named modgrid_{task_key}_{contrast}; 30 items
    # x `system_repeats` repeats per actor; blank system prompts except the
    # headroom contrast, whose intervention rides the system channel.
    # framing (vs bare R0) is built only for essay; non-essay framing pairs
    # are assembled later from stored outputs.
    modgrid_task_keys = {
        "essay": "essay",
        "translation": "translation",
        "grant": "grant_proposal_abstract",
        "incident": "incident_postmortem",
    }
    modgrid_contrasts = [
        "highlow",
        "highlow_fund",
        "highlow_spelled_numbers",
        "highlow_reasoning_medium",
        "highlow_reasoning_traces_medium",
        "moral",
        "amount",
        "framing",
        "framed_empty",
        "headroom",
    ]
    for task_key, task_name in modgrid_task_keys.items():
        wanted = {c for c in modgrid_contrasts
                  if f"modgrid_{task_key}_{c}" in comparisons}
        if task_key != "essay":
            wanted.discard("framing")
        if not wanted or (tasks is not None and task_name not in tasks):
            continue
        task_pool = task_rows_for_task(all_task_rows, task_name)
        slots = [(task, rep) for rep in range(system_repeats) for task in task_pool]
        emitted_slots = [
            (slot_idx, task, rep)
            for slot_idx, (task, rep) in enumerate(slots)
            if rep >= modgrid_repeat_start
        ]
        selected = read_selected_pairs() if {
            "highlow",
            "highlow_fund",
            "highlow_spelled_numbers",
            "highlow_reasoning_medium",
            "highlow_reasoning_traces_medium",
        } & wanted else []
        for actor in sorted(actors or set(ACTORS)):
            hl_pairs: list[dict[str, str]] = []
            if {
                "highlow",
                "highlow_fund",
                "highlow_spelled_numbers",
                "highlow_reasoning_medium",
                "highlow_reasoning_traces_medium",
            } & wanted:
                by_domain: dict[str, list[dict[str, str]]] = {}
                for row in selected:
                    if row["actor"] == actor and row["pair_set"] == "default":
                        by_domain.setdefault(row["domain"], []).append(row)
                domains = sorted(by_domain)
                quota = {d: (len(slots) // len(domains)) + (1 if i < len(slots) % len(domains) else 0)
                         for i, d in enumerate(domains)}
                pools = {d: sorted(by_domain[d], key=lambda r: int(r["pair_idx"]))[: quota[d]]
                         for d in domains}
                cursor = {d: 0 for d in domains}
                for i in range(len(slots)):
                    d = domains[i % len(domains)]
                    if cursor[d] >= len(pools[d]):
                        d = max(domains, key=lambda x: len(pools[x]) - cursor[x])
                    hl_pairs.append(pools[d][cursor[d]])
                    cursor[d] += 1
            for slot_idx, task, rep in emitted_slots:
                task_block = (clean_essay_direct_prompt(str(task["item_label"]))
                              if task_name == "essay" else str(task["base_prompt"]))
                neutral = build_modgrid_prompt(
                    task_name, task_block, amount=1000, target=MODGRID_NEUTRAL_TARGET)
                framed_empty = build_modgrid_empty_prompt(task_name, task_block)
                base = dict(actor=actor, task_row=task, domain="", domain_label="",
                            repeat=str(rep), framing="modgrid_unified_wrapper",
                            topic_design="task_pool",
                            source_note=(
                                f"canonical_highn_total_repeats_{system_repeats}"
                                if modgrid_repeat_start
                                else ""
                            ))
                uid_stem = f"{actor}:{task_name}:i{task['item_id']}:r{rep}"
                for hl_variant, prompt_builder in (
                    ("highlow", lambda tgt: build_modgrid_prompt(
                        task_name, task_block, amount=1000, target=tgt)),
                    # wording ablation: identical except the consequence
                    # sentence uses the no-amount "fund" phrasing
                    ("highlow_fund", lambda tgt: build_modgrid_fund_prompt(
                        task_name, task_block, target=tgt)),
                    ("highlow_spelled_numbers", lambda tgt: build_modgrid_prompt(
                        task_name, task_block, amount=1000, target=tgt)),
                    ("highlow_reasoning_medium", lambda tgt: build_modgrid_fund_prompt(
                        task_name, task_block, target=tgt)),
                    ("highlow_reasoning_traces_medium", lambda tgt: build_modgrid_fund_prompt(
                        task_name, task_block, target=tgt)),
                ):
                    if hl_variant not in wanted:
                        continue
                    pair = hl_pairs[slot_idx]
                    high_c = rewritten_consequence(pair["high_description"])
                    low_c = rewritten_consequence(pair["low_description"])
                    if hl_variant == "highlow_spelled_numbers":
                        high_c = spell_numbers_in_text(high_c)
                        low_c = spell_numbers_in_text(low_c)
                    jobs.append(common_job(
                        pair_uid=f"modgrid_{task_key}_{hl_variant}:{uid_stem}",
                        comparison=f"modgrid_{task_key}_{hl_variant}",
                        condition_a="hl_high", condition_b="hl_low",
                        prompt_a=prompt_builder(high_c),
                        prompt_b=prompt_builder(low_c),
                        predicted_condition="hl_high", other_condition="hl_low",
                        pair_idx=pair["pair_idx"], pair_set=pair["pair_set"],
                        prompt_variant_id=("spelled_numbers" if hl_variant == "highlow_spelled_numbers" else ""),
                        high_description=pair["high_description"],
                        low_description=pair["low_description"],
                        high_consequence=high_c, low_consequence=low_c,
                        high_utility=pair["high_utility"], low_utility=pair["low_utility"],
                        delta_u=pair["delta_u"],
                        **{**base, "domain": pair["domain"],
                           "domain_label": DOMAIN_LABEL[pair["domain"]]}))
                if "moral" in wanted:
                    cause = cause_rows[slot_idx % len(cause_rows)]
                    jobs.append(common_job(
                        pair_uid=f"modgrid_{task_key}_moral:{uid_stem}",
                        comparison=f"modgrid_{task_key}_moral",
                        condition_a="moral_good", condition_b="moral_bad",
                        prompt_a=build_modgrid_prompt(task_name, task_block, amount=1000, target=cause["good_text"]),
                        prompt_b=build_modgrid_prompt(task_name, task_block, amount=1000, target=cause["bad_text"]),
                        predicted_condition="moral_good", other_condition="moral_bad",
                        cause_pair_label=cause["cause_pair_label"],
                        good_text=cause["good_text"], bad_text=cause["bad_text"],
                        **base))
                if "amount" in wanted:
                    jobs.append(common_job(
                        pair_uid=f"modgrid_{task_key}_amount:{uid_stem}",
                        comparison=f"modgrid_{task_key}_amount",
                        condition_a="amount_high", condition_b="amount_low",
                        prompt_a=build_modgrid_prompt(task_name, task_block, amount=1000000, target=MODGRID_NEUTRAL_TARGET),
                        prompt_b=build_modgrid_prompt(task_name, task_block, amount=100, target=MODGRID_NEUTRAL_TARGET),
                        predicted_condition="amount_high", other_condition="amount_low",
                        amount_high="1000000", amount_low="100",
                        **base))
                if "framing" in wanted:
                    jobs.append(common_job(
                        pair_uid=f"modgrid_{task_key}_framing:{uid_stem}",
                        comparison=f"modgrid_{task_key}_framing",
                        condition_a="framed_neutral", condition_b="r0",
                        prompt_a=neutral, prompt_b=task_block,
                        predicted_condition="framed_neutral", other_condition="r0",
                        **base))
                if "framed_empty" in wanted:
                    jobs.append(common_job(
                        pair_uid=f"modgrid_{task_key}_framed_empty:{uid_stem}",
                        comparison=f"modgrid_{task_key}_framed_empty",
                        condition_a="framed_empty", condition_b="r0",
                        prompt_a=framed_empty, prompt_b=task_block,
                        predicted_condition="framed_empty", other_condition="r0",
                        **{**base, "framing": "modgrid_empty_competition_wrapper"}))
                if "headroom" in wanted:
                    jobs.append(common_job(
                        pair_uid=f"modgrid_{task_key}_headroom:{uid_stem}",
                        comparison=f"modgrid_{task_key}_headroom",
                        condition_a="framed_strong", condition_b="framed_neutral",
                        prompt_a=neutral, prompt_b=neutral,
                        system_prompt_a=MAX_EFFORT_STRONG_SYSTEM_PROMPTS[task_name],
                        system_prompt_b="",
                        predicted_condition="framed_strong", other_condition="framed_neutral",
                        **base))

    # comparison -> (instructions_a, instructions_b, full_topic_crossing,
    #                condition_a, condition_b); condition_a is the predicted side.
    essay_direct_user_prompt_comparisons: dict[
        str, tuple[dict[str, str], dict[str, str], bool, str, str]
    ] = {
        "essay_direct_user_prompt_max_effort_full_topics": (
            USER_MAX_EFFORT_STRONG_INSTRUCTIONS,
            USER_EFFORT_WEAK_INSTRUCTIONS,
            True,
            "user_strong",
            "user_normal",
        ),
    }
    for comparison, (
        strong_instructions,
        normal_instructions,
        use_full_topic_crossing,
        condition_a_name,
        condition_b_name,
    ) in essay_direct_user_prompt_comparisons.items():
        if comparison not in comparisons:
            continue
        if tasks is not None and "essay" not in tasks:
            continue
        direct_trials = read_essay_direct_trials()
        if actors is not None:
            direct_trials = [row for row in direct_trials if row["actor"] in actors]
        direct_topic_labels = {str(trial["essay_topic"]) for trial in direct_trials}
        direct_essay_tasks: list[dict[str, str]] = []
        for task in task_rows_for_task(all_task_rows, "essay"):
            if task["item_label"] not in direct_topic_labels:
                continue
            copy = dict(task)
            copy["item_index"] = str(len(direct_essay_tasks))
            direct_essay_tasks.append(copy)
        trial_task_pairs: list[tuple[dict[str, Any], dict[str, str], str]] = []
        if use_full_topic_crossing:
            for outcome_trial in unique_essay_direct_outcomes(direct_trials):
                for task in direct_essay_tasks:
                    trial_task_pairs.append((outcome_trial, task, str(task["item_index"])))
        else:
            essay_tasks_by_label = task_by_label(all_task_rows, "essay")
            for trial in direct_trials:
                topic = str(trial["essay_topic"])
                task = essay_tasks_by_label.get(topic)
                if task is None:
                    raise ValueError(f"essay direct trial topic is missing from task_items.csv: {topic}")
                trial_task_pairs.append((trial, task, str(task["item_index"])))

        for trial, task, source_topic_index in trial_task_pairs:
            topic = str(task["item_label"])
            base_prompt = clean_essay_direct_prompt(topic)
            inst_a = strong_instructions["essay"]
            inst_b = normal_instructions["essay"]
            prompt_a = f"{base_prompt}\n\n{inst_a}" if inst_a else base_prompt
            prompt_b = f"{base_prompt}\n\n{inst_b}" if inst_b else base_prompt
            variant_id = prompt_variant_id(
                system_prompt_a="",
                system_prompt_b="",
                prompt_a=prompt_a,
                prompt_b=prompt_b,
            )
            uid = (
                f"{comparison}:{trial['actor']}:essay:rep{trial['outcome_index']}:"
                f"t{source_topic_index}:v{variant_id}"
            )
            jobs.append(
                common_job(
                    pair_uid=uid,
                    comparison=comparison,
                    actor=str(trial["actor"]),
                    task_row=task,
                    domain="",
                    domain_label="",
                    condition_a=condition_a_name,
                    condition_b=condition_b_name,
                    prompt_a=prompt_a,
                    prompt_b=prompt_b,
                    system_prompt_a="",
                    system_prompt_b="",
                    prompt_variant_id=variant_id,
                    predicted_condition=condition_a_name,
                    other_condition=condition_b_name,
                    source_trial_index=str(trial["trial_index"]),
                    framing="clean_direct_no_outcome",
                    repeat=str(trial["outcome_index"]),
                    source_topic_index=source_topic_index,
                    topic_design="full_topics",
                )
            )

    system_prompt_comparisons: dict[str, tuple[dict[str, str], dict[str, str]]] = {
        "system_prompt": (STRONG_SYSTEM_PROMPTS, NORMAL_SYSTEM_PROMPTS),
        "system_prompt_role": (ROLE_ONLY_STRONG_SYSTEM_PROMPTS, NORMAL_SYSTEM_PROMPTS),
        "system_prompt_incentive": (INCENTIVE_STRONG_SYSTEM_PROMPTS, EFFORT_WEAK_SYSTEM_PROMPTS),
        "system_prompt_reputation": (REPUTATION_STRONG_SYSTEM_PROMPTS, EFFORT_WEAK_SYSTEM_PROMPTS),
        "system_prompt_competition": (COMPETITION_STRONG_SYSTEM_PROMPTS, EFFORT_WEAK_SYSTEM_PROMPTS),
        "system_prompt_effort": (EFFORT_STRONG_SYSTEM_PROMPTS, EFFORT_WEAK_SYSTEM_PROMPTS),
    }
    for comparison, (strong_prompts, normal_prompts) in system_prompt_comparisons.items():
        if comparison not in comparisons:
            continue
        for actor in sorted(actors or set(ACTORS)):
            for task_name in sorted(by_task):
                for task in by_task[task_name]:
                    for repeat in range(system_repeats):
                        sys_a, sys_b, prompt_a, prompt_b = system_prompt_pair(
                            task,
                            strong_system_prompts=strong_prompts,
                            normal_system_prompts=normal_prompts,
                        )
                        variant_id = prompt_variant_id(
                            system_prompt_a=sys_a,
                            system_prompt_b=sys_b,
                            prompt_a=prompt_a,
                            prompt_b=prompt_b,
                        )
                        uid = f"{comparison}:{actor}:{task['task']}:{task['item_index']}:v{variant_id}:r{repeat}"
                        jobs.append(
                            common_job(
                                pair_uid=uid,
                                comparison=comparison,
                                actor=actor,
                                task_row=task,
                                domain="",
                                condition_a="sys_strong",
                                condition_b="sys_normal",
                                prompt_a=prompt_a,
                                prompt_b=prompt_b,
                                system_prompt_a=sys_a,
                                system_prompt_b=sys_b,
                                prompt_variant_id=variant_id,
                                predicted_condition="sys_strong",
                                other_condition="sys_normal",
                                repeat=str(repeat),
                            )
                        )

    user_prompt_comparisons: dict[str, tuple[dict[str, str], dict[str, str]]] = {
        "user_prompt_effort": (USER_EFFORT_STRONG_INSTRUCTIONS, USER_EFFORT_WEAK_INSTRUCTIONS),
        "user_prompt_role": (USER_ROLE_STRONG_INSTRUCTIONS, USER_ROLE_WEAK_INSTRUCTIONS),
        "user_prompt_status": (USER_STATUS_STRONG_INSTRUCTIONS, USER_STATUS_WEAK_INSTRUCTIONS),
    }
    for comparison, (strong_instructions, normal_instructions) in user_prompt_comparisons.items():
        if comparison not in comparisons:
            continue
        for actor in sorted(actors or set(ACTORS)):
            for task_name in sorted(by_task):
                for task in by_task[task_name]:
                    for repeat in range(system_repeats):
                        sys_a, sys_b, prompt_a, prompt_b, inst_a, inst_b = user_prompt_pair(
                            task,
                            strong_user_instructions=strong_instructions,
                            normal_user_instructions=normal_instructions,
                        )
                        variant_id = user_prompt_variant_id(prompt_a=prompt_a, prompt_b=prompt_b)
                        uid = f"{comparison}:{actor}:{task['task']}:{task['item_index']}:v{variant_id}:r{repeat}"
                        jobs.append(
                            common_job(
                                pair_uid=uid,
                                comparison=comparison,
                                actor=actor,
                                task_row=task,
                                domain="",
                                condition_a="user_strong",
                                condition_b="user_normal",
                                prompt_a=prompt_a,
                                prompt_b=prompt_b,
                                system_prompt_a=sys_a,
                                system_prompt_b=sys_b,
                                prompt_variant_id=variant_id,
                                predicted_condition="user_strong",
                                other_condition="user_normal",
                                repeat=str(repeat),
                            )
                        )

    framed_user_prompt_comparisons: dict[str, tuple[dict[str, str], dict[str, str]]] = {
        "framed_user_prompt_role": (USER_ROLE_STRONG_INSTRUCTIONS, USER_ROLE_WEAK_INSTRUCTIONS),
    }
    for comparison, (strong_instructions, normal_instructions) in framed_user_prompt_comparisons.items():
        if comparison not in comparisons:
            continue
        for actor in sorted(actors or set(ACTORS)):
            for task_name in sorted(by_task):
                for task in by_task[task_name]:
                    task_block = (
                        build_essay_system_prompt(task["item_label"])
                        if task_name == "essay"
                        else task["base_prompt"]
                    )
                    neutral = build_modgrid_fund_prompt(
                        task_name,
                        task_block,
                        target=MODGRID_NEUTRAL_TARGET,
                    )
                    inst_a = strong_instructions.get(task_name, "")
                    inst_b = normal_instructions.get(task_name, "")
                    prompt_a = f"{neutral}\n\n{inst_a}" if inst_a else neutral
                    prompt_b = f"{neutral}\n\n{inst_b}" if inst_b else neutral
                    for repeat in range(modgrid_repeat_start, system_repeats):
                        variant_id = user_prompt_variant_id(prompt_a=prompt_a, prompt_b=prompt_b)
                        uid = f"{comparison}:{actor}:{task['task']}:{task['item_index']}:v{variant_id}:r{repeat}"
                        jobs.append(
                            common_job(
                                pair_uid=uid,
                                comparison=comparison,
                                actor=actor,
                                task_row=task,
                                domain="",
                                condition_a="user_strong",
                                condition_b="user_normal",
                                prompt_a=prompt_a,
                                prompt_b=prompt_b,
                                system_prompt_a="",
                                system_prompt_b="",
                                prompt_variant_id=variant_id,
                                predicted_condition="user_strong",
                                other_condition="user_normal",
                                repeat=str(repeat),
                                framing="modgrid_neutral_competition_wrapper",
                                source_note=(
                                    f"canonical_role_total_repeats_{system_repeats}"
                                    if modgrid_repeat_start
                                    else ""
                                ),
                            )
                        )

    return jobs


def common_job(
    *,
    pair_uid: str,
    comparison: str,
    actor: str,
    task_row: dict[str, str],
    domain: str,
    condition_a: str,
    condition_b: str,
    prompt_a: str,
    prompt_b: str,
    predicted_condition: str,
    other_condition: str,
    domain_label: str = "",
    pair_idx: str = "",
    pair_set: str = "",
    category: str = "",
    system_prompt_a: str = "",
    system_prompt_b: str = "",
    prompt_variant_id: str = "",
    high_description: str = "",
    low_description: str = "",
    high_consequence: str = "",
    low_consequence: str = "",
    high_utility: str = "",
    low_utility: str = "",
    delta_u: str = "",
    cause_pair_label: str = "",
    good_text: str = "",
    bad_text: str = "",
    amount_high: str = "",
    amount_low: str = "",
    repeat: str = "",
    sample_k: str = "",
    outcome: str = "",
    source_trial_index: str = "",
    source_note: str = "",
    framing: str = "",
    source_outcome_index: str = "",
    source_topic_index: str = "",
    topic_design: str = "",
    gap_bin: str = "",
    gap_bin_count: str = "",
    gap_bin_min_delta_u: str = "",
    gap_bin_max_delta_u: str = "",
    gap_bin_sample_index: str = "",
    gap_sampling_seed: str = "",
) -> dict[str, Any]:
    return {
        "pair_uid": pair_uid,
        "comparison": comparison,
        "actor": actor,
        "actor_label": ACTOR_LABEL[actor],
        "task": task_row["task"],
        "task_label": TASK_LABEL[task_row["task"]],
        "domain": domain,
        "domain_label": domain_label,
        "pair_idx": pair_idx,
        "pair_set": pair_set,
        "category": category,
        "item_id": task_row["item_id"],
        "item_index": task_row.get("item_index", ""),
        "item_label": task_row["item_label"],
        "axis": task_row["axis"],
        "axis_definition": task_row["axis_definition"],
        "base_prompt": base_prompt_for_task(task_row),
        "condition_a": condition_a,
        "condition_b": condition_b,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "system_prompt_a": system_prompt_a,
        "system_prompt_b": system_prompt_b,
        "prompt_variant_id": prompt_variant_id,
        "predicted_condition": predicted_condition,
        "other_condition": other_condition,
        "high_description": high_description,
        "low_description": low_description,
        "high_consequence": high_consequence,
        "low_consequence": low_consequence,
        "high_utility": high_utility,
        "low_utility": low_utility,
        "delta_u": delta_u,
        "cause_pair_label": cause_pair_label,
        "good_text": good_text,
        "bad_text": bad_text,
        "amount_high": amount_high,
        "amount_low": amount_low,
        "repeat": repeat,
        "sample_k": sample_k,
        "outcome": outcome,
        "source_trial_index": source_trial_index,
        "source_note": source_note,
        "framing": framing,
        "source_outcome_index": source_outcome_index,
        "source_topic_index": source_topic_index,
        "topic_design": topic_design,
        "gap_bin": gap_bin,
        "gap_bin_count": gap_bin_count,
        "gap_bin_min_delta_u": gap_bin_min_delta_u,
        "gap_bin_max_delta_u": gap_bin_max_delta_u,
        "gap_bin_sample_index": gap_bin_sample_index,
        "gap_sampling_seed": gap_sampling_seed,
    }


def write_generation_jobs(jobs: list[dict[str, Any]]) -> Path:
    current_path = OUTPUT_API / "generation_jobs.jsonl"
    payload = "\n".join(json.dumps(job, ensure_ascii=False, sort_keys=True) for job in jobs)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    run_id = generation_run_id(jobs=jobs, digest=digest)
    run_dir = OUTPUT_API / "runs" / run_id
    manifest_path = run_dir / "generation_jobs.jsonl"
    jobs = [
        {
            **job,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "run_manifest_path": str(manifest_path),
            "run_generations_path": str(run_dir / "generations.jsonl"),
            "run_generation_failures_path": str(run_dir / "generation_failures.jsonl"),
            "run_judge_votes_path": str(run_dir / "judge_votes.jsonl"),
        }
        for job in jobs
    ]
    write_jsonl(current_path, jobs)
    write_jsonl(manifest_path, jobs)
    return manifest_path


def read_generation_jobs() -> list[dict[str, Any]]:
    return read_jsonl(OUTPUT_API / "generation_jobs.jsonl")
