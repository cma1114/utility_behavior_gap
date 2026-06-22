"""Judge-panel counting rules for generated pairwise comparisons."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping


LEGACY_ESSAY_HIGHLOW_RUNS = {"bg_fixed_topic_default", "bg_fixed_topic_same_count"}
CALIBRATION_COMPARISONS = {
    "system_prompt",
    "system_prompt_role",
    "system_prompt_incentive",
    "system_prompt_reputation",
    "system_prompt_competition",
    "system_prompt_effort",
    "essay_direct_system_prompt",
    "essay_direct_system_prompt_effort",
    "essay_direct_system_prompt_max_effort",
    "essay_direct_system_prompt_full_topics",
    "essay_direct_system_prompt_effort_full_topics",
    "essay_direct_system_prompt_max_effort_full_topics",
    "essay_direct_user_prompt_max_effort_full_topics",
    "essay_user_prompt_goodjob_vs_bare",
    "modgrid_essay_moral",
    "modgrid_essay_amount",
    "modgrid_essay_framing",
    "modgrid_essay_headroom",
    "modgrid_translation_moral",
    "modgrid_translation_amount",
    "modgrid_translation_headroom",
    "modgrid_grant_moral",
    "modgrid_grant_amount",
    "modgrid_grant_headroom",
    "modgrid_incident_moral",
    "modgrid_incident_amount",
    "modgrid_incident_headroom",
    "direct_user_prompt_max_effort",
    "user_prompt_effort",
    "user_prompt_role",
    "framed_user_prompt_role",
    "user_prompt_status",
    "moral_nolabel",
    "amount",
}


def derive_judge_verdict(vote_conditions: Iterable[str]) -> str:
    """Collapse one judge's votes on a pair into a single verdict.

    With both-orders judging a judge casts two votes (one per presentation
    order). The judge's verdict is a condition only if every resolved vote
    names that same condition — an orientation flip, or any tie vote, is a
    position-driven or genuinely tied judgment and collapses to ``tie``.
    A single-vote history passes through unchanged, so single-order runs
    aggregate exactly as before.
    """
    valid = [value for value in vote_conditions if value and value != "unresolved"]
    if not valid:
        return "unresolved"
    distinct = set(valid)
    if len(distinct) == 1:
        return valid[0]
    return "tie"


def derive_panel_winner_condition(row: Mapping[str, str], vote_conditions: Iterable[str]) -> str:
    """Derive a panel winner from sanitized individual vote conditions.

    Older essay high-low runs represented a three-way X/Y/TIE split as a
    non-counted disagreement. New OpenRouter reruns use the later rule: split
    panels are counted as ties.
    """

    valid = [value for value in vote_conditions if value and value != "unresolved"]
    if not valid:
        return "unresolved"

    counts = Counter(valid)
    top_count = max(counts.values())
    top = [condition for condition, count in counts.items() if count == top_count]
    if len(top) == 1:
        return top[0]

    if row.get("source_run") in LEGACY_ESSAY_HIGHLOW_RUNS:
        return "unresolved"
    return "tie"


def derive_counted_winner_condition(row: Mapping[str, str], vote_conditions: Iterable[str]) -> str:
    """Return the winner condition used by the paper's win-rate denominators."""

    panel = derive_panel_winner_condition(row, vote_conditions)
    comparison = row.get("comparison", "")
    predicted = row.get("predicted_condition", "")
    other = row.get("other_condition", "")

    if comparison in {"highlow_main", "highlow_same_count"}:
        if row.get("source_run") in LEGACY_ESSAY_HIGHLOW_RUNS:
            return panel if panel in {"high", "low"} else ""
        return panel if panel in {"high", "low", "tie"} else ""

    if comparison in CALIBRATION_COMPARISONS:
        return panel if panel in {predicted, other, "tie"} else "tie"

    return panel if panel in {predicted, other, "tie"} else ""
