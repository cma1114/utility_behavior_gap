from collections import Counter

import pytest

from utility_behavior_gap.constants import ACTOR_MODEL_ID, ACTORS, JUDGE_MODEL_IDS
from utility_behavior_gap.job_builder import build_generation_jobs, direct_outcome_text
from utility_behavior_gap.openrouter import actor_model_id, judge_model_ids, require_openrouter_key
from utility_behavior_gap.pair_selection import select_pairs
from utility_behavior_gap.prompts import (
    EFFORT_STRONG_SYSTEM_PROMPTS,
    EFFORT_WEAK_SYSTEM_PROMPTS,
    MAX_EFFORT_STRONG_SYSTEM_PROMPTS,
    USER_MAX_EFFORT_STRONG_INSTRUCTIONS,
)


def assert_no_direct_outcome_scaffold(jobs):
    leak_markers = (
        "If your essay wins",
        "$1,000 will be donated",
        "following cause:",
        "Do not mention the prize",
        "donation",
        "reward",
        "sponsoring organization",
    )
    for job in jobs:
        for field in ("prompt_a", "prompt_b"):
            prompt = job[field]
            assert all(marker not in prompt for marker in leak_markers)
            if job["outcome"]:
                assert job["outcome"] not in prompt
        assert job["outcome"] == ""
        assert job["source_note"] == ""
        assert job["source_outcome_index"] == ""
        assert job["framing"] == "clean_direct_no_outcome"


def test_pair_selection_rebuilds_full_default_and_same_count_sets():
    rows = select_pairs(default_pairs_per_cell=2, same_count_pairs_per_cell=3)
    default = [row for row in rows if row["pair_set"] == "default"]
    same_count = [row for row in rows if row["pair_set"] == "same_count"]

    assert len(default) == len(ACTORS) * 4 * 2
    assert len(same_count) == len(ACTORS) * 3 * 3
    assert all(float(row["delta_u"]) > 0 for row in rows)
    assert {row["category"] for row in same_count} <= {row["category"] for row in rows if row["category"].startswith("count_")}


def test_openrouter_model_ids_are_versioned_public_constants():
    assert set(ACTORS) <= set(ACTOR_MODEL_ID)
    assert all(actor_model_id(actor) == ACTOR_MODEL_ID[actor] for actor in ACTORS)
    assert actor_model_id("mimo-v25-pro-or") == "xiaomi/mimo-v2.5-pro"
    assert judge_model_ids() == JUDGE_MODEL_IDS


def test_highlow_intervention_essay_jobs_use_rewritten_consequences():
    jobs = build_generation_jobs(
        comparisons={"highlow_intervention"},
        tasks={"essay"},
        actors={"gpt-5.4-mini-or"},
        pairs_per_actor_domain=1,
        items_per_task=1,
    )

    assert len(jobs) == 4
    assert {job["comparison"] for job in jobs} == {"highlow_intervention"}
    assert {job["task"] for job in jobs} == {"essay"}
    assert {job["framing"] for job in jobs} == {"funded_intervention"}
    for job in jobs:
        assert job["high_description"]
        assert job["low_description"]
        assert job["high_consequence"]
        assert job["low_consequence"]
        assert job["high_consequence"] in job["prompt_a"]
        assert job["low_consequence"] in job["prompt_b"]
        assert "select the best essay" in job["prompt_a"]
        assert "sponsor will fund the following intervention" in job["prompt_a"]
        assert "donated to" not in job["prompt_a"]
        assert "donated to" not in job["prompt_b"]


def test_highlow_intervention_can_run_all_tasks_with_rewritten_consequences():
    jobs = build_generation_jobs(
        comparisons={"highlow_intervention"},
        tasks={"essay", "translation", "incident_postmortem", "grant_proposal_abstract"},
        actors={"gpt-5.4-mini-or"},
        pairs_per_actor_domain=1,
        items_per_task=1,
    )

    assert len(jobs) == 16
    assert {job["comparison"] for job in jobs} == {"highlow_intervention"}
    assert {job["task"] for job in jobs} == {
        "essay",
        "translation",
        "incident_postmortem",
        "grant_proposal_abstract",
    }
    assert {job["framing"] for job in jobs} == {"funded_intervention"}
    for job in jobs:
        assert job["high_consequence"]
        assert job["low_consequence"]
        assert job["high_consequence"] in job["prompt_a"]
        assert job["low_consequence"] in job["prompt_b"]
        assert "sponsor will fund the following intervention" in job["prompt_a"]
        assert "donated to" not in job["prompt_a"]
        assert "donated to" not in job["prompt_b"]
        if job["task"] != "essay":
            assert "outputs produced for the same task" not in job["prompt_a"]
            assert "Do not mention the evaluation, sponsor, funding, or intervention" in job["prompt_a"]
        if job["task"] == "translation":
            assert "blind translation evaluation" in job["prompt_a"]
            assert "select the best translation" in job["prompt_a"]
            assert "Output only the translation" in job["prompt_a"]
        if job["task"] == "incident_postmortem":
            assert "blind postmortem-writing evaluation" in job["prompt_a"]
            assert "select the best postmortem" in job["prompt_a"]
            assert "Focus solely on the incident analysis" in job["prompt_a"]
        if job["task"] == "grant_proposal_abstract":
            assert "blind grant-abstract writing evaluation" in job["prompt_a"]
            assert "select the best abstract" in job["prompt_a"]
            assert "Focus solely on the proposed project" in job["prompt_a"]


def test_pair_selection_uses_mimo_v25_overlay():
    jobs = build_generation_jobs(
        comparisons={"highlow_intervention"},
        tasks={"essay"},
        actors={"mimo-v25-pro-or"},
        pairs_per_actor_domain=1,
        items_per_task=1,
    )

    assert len(jobs) == 4
    assert {job["actor"] for job in jobs} == {"mimo-v25-pro-or"}


def test_live_api_requires_non_placeholder_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "xxx")

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        require_openrouter_key()


def test_essay_direct_system_prompt_jobs_use_trial_level_outcome_design():
    jobs = build_generation_jobs(
        comparisons={"essay_direct_system_prompt"},
        tasks={"essay"},
        actors={"glm-5.1-or"},
    )

    assert len(jobs) == 80
    assert {job["comparison"] for job in jobs} == {"essay_direct_system_prompt"}
    assert {job["actor"] for job in jobs} == {"glm-5.1-or"}
    assert {job["framing"] for job in jobs} == {"clean_direct_no_outcome"}
    assert {job["domain"] for job in jobs} == {""}
    assert {job["condition_a"] for job in jobs} == {"sys_strong"}
    assert {job["condition_b"] for job in jobs} == {"sys_normal"}
    assert {job["predicted_condition"] for job in jobs} == {"sys_strong"}
    assert all(job["prompt_a"] == job["prompt_b"] for job in jobs)
    assert_no_direct_outcome_scaffold(jobs)
    assert all(job["system_prompt_a"] != job["system_prompt_b"] for job in jobs)

    assert len({job["source_trial_index"] for job in jobs}) == 80


def test_essay_direct_effort_jobs_use_same_design_with_effort_system_prompts():
    jobs = build_generation_jobs(
        comparisons={"essay_direct_system_prompt_effort"},
        tasks={"essay"},
        actors={"glm-5.1-or"},
    )

    assert len(jobs) == 80
    assert {job["comparison"] for job in jobs} == {"essay_direct_system_prompt_effort"}
    assert {job["domain"] for job in jobs} == {""}
    assert {job["framing"] for job in jobs} == {"clean_direct_no_outcome"}
    assert all(job["prompt_a"] == job["prompt_b"] for job in jobs)
    assert_no_direct_outcome_scaffold(jobs)
    assert {job["system_prompt_a"] for job in jobs} == {EFFORT_STRONG_SYSTEM_PROMPTS["essay"]}
    assert {job["system_prompt_b"] for job in jobs} == {EFFORT_WEAK_SYSTEM_PROMPTS["essay"]}

    assert len({job["source_trial_index"] for job in jobs}) == 80


def test_essay_direct_max_effort_jobs_use_same_design_with_max_effort_system_prompts():
    jobs = build_generation_jobs(
        comparisons={"essay_direct_system_prompt_max_effort"},
        tasks={"essay"},
        actors={"glm-5.1-or"},
    )

    assert len(jobs) == 80
    assert {job["comparison"] for job in jobs} == {"essay_direct_system_prompt_max_effort"}
    assert {job["domain"] for job in jobs} == {""}
    assert {job["framing"] for job in jobs} == {"clean_direct_no_outcome"}
    assert all(job["prompt_a"] == job["prompt_b"] for job in jobs)
    assert_no_direct_outcome_scaffold(jobs)
    assert {job["system_prompt_a"] for job in jobs} == {MAX_EFFORT_STRONG_SYSTEM_PROMPTS["essay"]}
    assert {job["system_prompt_b"] for job in jobs} == {EFFORT_WEAK_SYSTEM_PROMPTS["essay"]}

    assert len({job["source_trial_index"] for job in jobs}) == 80


def test_essay_direct_max_effort_full_topics_crosses_all_outcomes_with_all_essay_topics():
    jobs = build_generation_jobs(
        comparisons={"essay_direct_system_prompt_max_effort_full_topics"},
        tasks={"essay"},
        actors={"glm-5.1-or"},
    )

    assert len(jobs) == 200
    assert {job["comparison"] for job in jobs} == {"essay_direct_system_prompt_max_effort_full_topics"}
    assert {job["framing"] for job in jobs} == {"clean_direct_no_outcome"}
    assert {job["topic_design"] for job in jobs} == {"full_topics"}
    assert {job["domain"] for job in jobs} == {""}
    assert len({job["pair_uid"] for job in jobs}) == 200
    assert all(job["prompt_a"] == job["prompt_b"] for job in jobs)
    assert_no_direct_outcome_scaffold(jobs)
    assert {job["system_prompt_a"] for job in jobs} == {MAX_EFFORT_STRONG_SYSTEM_PROMPTS["essay"]}
    assert {job["system_prompt_b"] for job in jobs} == {EFFORT_WEAK_SYSTEM_PROMPTS["essay"]}
    assert {job["source_topic_index"] for job in jobs} == {"0", "1", "2", "3", "4"}
    assert len({job["repeat"] for job in jobs}) == 40

    assert Counter(job["item_label"] for job in jobs).most_common(1)[0][1] == 40


def test_modgrid_essay_contrasts_share_wrapper_and_slots():
    comparisons = {
        "modgrid_essay_highlow", "modgrid_essay_moral", "modgrid_essay_amount",
        "modgrid_essay_framing", "modgrid_essay_framed_empty", "modgrid_essay_headroom",
    }
    jobs = build_generation_jobs(
        comparisons=comparisons, tasks={"essay"}, actors={"gpt-5.4-mini-or"},
        system_repeats=5,
    )
    by_comp = {}
    for job in jobs:
        by_comp.setdefault(job["comparison"], []).append(job)
    assert set(by_comp) == comparisons
    for comp, rows in by_comp.items():
        assert len(rows) == 150, comp  # 30 topics x 5 repeats
        assert len({job["pair_uid"] for job in rows}) == 150
        assert len({job["item_label"] for job in rows}) == 30
        if comp == "modgrid_essay_framed_empty":
            assert {job["framing"] for job in rows} == {"modgrid_empty_competition_wrapper"}
        else:
            assert {job["framing"] for job in rows} == {"modgrid_unified_wrapper"}

    # every framed prompt embeds the bare task block verbatim
    framing = by_comp["modgrid_essay_framing"]
    for job in framing:
        assert job["prompt_b"] in job["prompt_a"]  # r0 block inside wrapper
        assert job["system_prompt_a"] == "" and job["system_prompt_b"] == ""
        assert "sponsor will fund the following intervention:" in job["prompt_a"]
        assert "A healthcare intervention at a children's hospital." in job["prompt_a"]
        assert "$1,000 toward the following:" not in job["prompt_a"]

    framed_empty = by_comp["modgrid_essay_framed_empty"]
    for job in framed_empty:
        assert job["condition_a"] == "framed_empty"
        assert job["condition_b"] == "r0"
        assert job["prompt_b"] in job["prompt_a"]  # r0 block inside empty competition wrapper
        assert job["system_prompt_a"] == "" and job["system_prompt_b"] == ""
        assert "select the best essay" in job["prompt_a"]
        assert "Do not mention the competition or judging process" in job["prompt_a"]
        assert "sponsor" not in job["prompt_a"].lower()
        assert "sponsor will fund" not in job["prompt_a"].lower()
        assert "sponsorship description" not in job["prompt_a"].lower()
        assert "following intervention" not in job["prompt_a"].lower()
        assert "children's hospital" not in job["prompt_a"]

    # headroom: identical user prompts, intervention only in the system channel
    for job in by_comp["modgrid_essay_headroom"]:
        assert job["prompt_a"] == job["prompt_b"]
        assert "maximum care and effort" in job["system_prompt_a"]
        assert job["system_prompt_b"] == ""

    # amount: $1,000,000 vs $100, same neutral target, comma formatting
    for job in by_comp["modgrid_essay_amount"]:
        assert "$1,000,000" in job["prompt_a"] and "$100 " in job["prompt_b"]
        assert "children's hospital" in job["prompt_a"]
        assert job["prompt_a"].replace("$1,000,000", "$100") == job["prompt_b"]

    # moral: same wrapper, cause texts in slots, 40 causes cycled
    moral = by_comp["modgrid_essay_moral"]
    assert len({job["cause_pair_label"] for job in moral}) == 40
    for job in moral:
        assert job["good_text"] in job["prompt_a"] and job["bad_text"] in job["prompt_b"]

    # highlow: actor's own pairs, all four domains near-balanced, rewrites in slots
    highlow = by_comp["modgrid_essay_highlow"]
    domain_counts = Counter(job["domain"] for job in highlow)
    assert set(domain_counts) == {"religions", "animals", "countries", "political"}
    assert max(domain_counts.values()) - min(domain_counts.values()) <= 1
    for job in highlow:
        assert job["high_consequence"] in job["prompt_a"]
        assert job["low_consequence"] in job["prompt_b"]
        assert float(job["delta_u"]) > 0


def test_modgrid_nonessay_contrasts_wrap_base_prompt():
    comparisons = {
        "modgrid_grant_highlow", "modgrid_grant_moral",
        "modgrid_grant_amount", "modgrid_grant_framed_empty", "modgrid_grant_headroom",
    }
    jobs = build_generation_jobs(
        comparisons=comparisons, tasks={"grant_proposal_abstract"},
        actors={"kimi-k2.5-or"}, system_repeats=5,
    )
    by_comp = {}
    for job in jobs:
        by_comp.setdefault(job["comparison"], []).append(job)
    assert set(by_comp) == comparisons
    for comp, rows in by_comp.items():
        assert len(rows) == 150, comp
        assert {job["task"] for job in rows} == {"grant_proposal_abstract"}
    for job in by_comp["modgrid_grant_moral"]:
        # base prompt embedded verbatim; wrapper, slot, blinding present
        assert job["task_row" if "task_row" in job else "item_label"]  # smoke
        assert "blind grant-abstract writing evaluation" in job["prompt_a"]
        assert "sponsor will fund the following intervention:" in job["prompt_a"]
        assert "$1,000 toward the following:" not in job["prompt_a"]
        assert job["good_text"] in job["prompt_a"]
        assert "Do not mention the evaluation, sponsor, funding, intervention" in job["prompt_a"]
    for job in by_comp["modgrid_grant_headroom"]:
        assert job["prompt_a"] == job["prompt_b"]
        assert "maximum care and effort" in job["system_prompt_a"]
        assert "grant proposal" in job["system_prompt_a"].lower()
        assert job["system_prompt_b"] == ""
    for job in by_comp["modgrid_grant_framed_empty"]:
        assert job["condition_a"] == "framed_empty"
        assert job["condition_b"] == "r0"
        assert job["prompt_b"] in job["prompt_a"]
        assert "blind grant-abstract writing evaluation" in job["prompt_a"]
        assert "select the best abstract" in job["prompt_a"]
        assert "Do not mention the evaluation or judging process" in job["prompt_a"]
        assert "sponsor" not in job["prompt_a"].lower()
        assert "sponsor will fund" not in job["prompt_a"].lower()
        assert "sponsorship description" not in job["prompt_a"].lower()
        assert "following intervention" not in job["prompt_a"].lower()
        assert "children's hospital" not in job["prompt_a"]
        assert job["framing"] == "modgrid_empty_competition_wrapper"
    # framing must NOT be buildable for non-essay
    framing_jobs = build_generation_jobs(
        comparisons={"modgrid_grant_framing"}, tasks={"grant_proposal_abstract"},
        actors={"kimi-k2.5-or"}, system_repeats=5,
    )
    assert framing_jobs == []


def test_modgrid_incident_highlow_fund_alias_matches_canonical_highlow():
    canonical = build_generation_jobs(
        comparisons={"modgrid_incident_highlow"},
        tasks={"incident_postmortem"}, actors={"glm-5.1-or"}, system_repeats=5)
    fund = build_generation_jobs(
        comparisons={"modgrid_incident_highlow_fund"},
        tasks={"incident_postmortem"}, actors={"glm-5.1-or"}, system_repeats=5)
    assert len(fund) == len(canonical) == 150
    canonical_by_slot = {j["pair_uid"].split(":", 1)[1]: j for j in canonical}
    for jf in fund:
        jc = canonical_by_slot[jf["pair_uid"].split(":", 1)[1]]
        # identical outcome pair in the identical slot
        assert jf["high_description"] == jc["high_description"]
        assert jf["low_description"] == jc["low_description"]
        assert jf["delta_u"] == jc["delta_u"]
        assert jf["system_prompt_a"] == jf["system_prompt_b"] == ""
        # highlow_fund is now a backward-compatible alias: canonical $1,000
        # modgrid high-low prompts already use the fund-intervention wording.
        for side in ("prompt_a", "prompt_b"):
            assert "the sponsor will fund the following intervention:" in jf[side]
            assert "$1,000" not in jf[side]
            assert jf[side] == jc[side]


def test_essay_user_goodjob_vs_bare_crosses_full_topic_pool_with_repeats():
    jobs = build_generation_jobs(
        comparisons={"essay_user_prompt_goodjob_vs_bare"},
        tasks={"essay"},
        actors={"gpt-5.4-mini-or"},
        system_repeats=7,
    )

    assert len(jobs) == 210  # 30 topics x 7 repeats
    assert {job["comparison"] for job in jobs} == {"essay_user_prompt_goodjob_vs_bare"}
    assert {job["framing"] for job in jobs} == {"clean_direct_no_outcome"}
    assert {job["topic_design"] for job in jobs} == {"task_pool"}
    assert {job["condition_a"] for job in jobs} == {"user_goodjob"}
    assert {job["condition_b"] for job in jobs} == {"user_bare"}
    assert {job["predicted_condition"] for job in jobs} == {"user_goodjob"}
    assert {job["other_condition"] for job in jobs} == {"user_bare"}
    assert {job["system_prompt_a"] for job in jobs} == {""}
    assert {job["system_prompt_b"] for job in jobs} == {""}
    assert len({job["pair_uid"] for job in jobs}) == 210
    assert len({job["item_label"] for job in jobs}) == 30
    assert {job["repeat"] for job in jobs} == {str(r) for r in range(7)}
    assert_no_direct_outcome_scaffold(jobs)
    assert all(job["prompt_a"].endswith("\n\nDo a good job") for job in jobs)
    # the bare arm is exactly the task prompt: no suffix, no trailing whitespace
    assert all(job["prompt_a"] == f"{job['prompt_b']}\n\nDo a good job" for job in jobs)
    assert all(job["prompt_b"] == job["prompt_b"].strip() for job in jobs)


def test_essay_direct_user_max_effort_full_topics_uses_blank_system_and_user_instruction_contrast():
    jobs = build_generation_jobs(
        comparisons={"essay_direct_user_prompt_max_effort_full_topics"},
        tasks={"essay"},
        actors={"qwen3.5-9b-or"},
    )

    assert len(jobs) == 200
    assert {job["comparison"] for job in jobs} == {"essay_direct_user_prompt_max_effort_full_topics"}
    assert {job["framing"] for job in jobs} == {"clean_direct_no_outcome"}
    assert {job["topic_design"] for job in jobs} == {"full_topics"}
    assert {job["condition_a"] for job in jobs} == {"user_strong"}
    assert {job["condition_b"] for job in jobs} == {"user_normal"}
    assert {job["predicted_condition"] for job in jobs} == {"user_strong"}
    assert {job["other_condition"] for job in jobs} == {"user_normal"}
    assert {job["system_prompt_a"] for job in jobs} == {""}
    assert {job["system_prompt_b"] for job in jobs} == {""}
    assert {job["domain"] for job in jobs} == {""}
    assert len({job["pair_uid"] for job in jobs}) == 200
    assert {job["source_topic_index"] for job in jobs} == {"0", "1", "2", "3", "4"}
    assert len({job["repeat"] for job in jobs}) == 40
    assert_no_direct_outcome_scaffold(jobs)
    assert all(job["prompt_a"].endswith(USER_MAX_EFFORT_STRONG_INSTRUCTIONS["essay"]) for job in jobs)
    assert all(job["prompt_b"].endswith("Do a good job") for job in jobs)
    assert all(
        job["prompt_a"].removesuffix(USER_MAX_EFFORT_STRONG_INSTRUCTIONS["essay"])
        == job["prompt_b"].removesuffix("Do a good job")
        for job in jobs
    )

    assert Counter(job["item_label"] for job in jobs).most_common(1)[0][1] == 40


def test_direct_user_max_effort_jobs_match_highlow_sampling_grid_without_outcome_scaffold():
    jobs = build_generation_jobs(
        comparisons={"direct_user_prompt_max_effort"},
        tasks={"translation"},
        actors={"qwen3.6-plus-or"},
    )

    assert len(jobs) == 320
    assert {job["comparison"] for job in jobs} == {"direct_user_prompt_max_effort"}
    assert {job["task"] for job in jobs} == {"translation"}
    assert {job["framing"] for job in jobs} == {"clean_direct_no_outcome"}
    assert {job["condition_a"] for job in jobs} == {"user_strong"}
    assert {job["condition_b"] for job in jobs} == {"user_normal"}
    assert {job["predicted_condition"] for job in jobs} == {"user_strong"}
    assert {job["other_condition"] for job in jobs} == {"user_normal"}
    assert {job["system_prompt_a"] for job in jobs} == {""}
    assert {job["system_prompt_b"] for job in jobs} == {""}
    assert Counter(job["domain"] for job in jobs) == {
        "religions": 80,
        "animals": 80,
        "countries": 80,
        "political": 80,
    }
    assert len({job["item_id"] for job in jobs}) == 30
    assert sorted(set(Counter(job["item_id"] for job in jobs).values())) == [8, 12]
    assert all(job["pair_set"] == "default" for job in jobs)
    assert_no_direct_outcome_scaffold(jobs)
    assert all(job["prompt_a"].endswith(USER_MAX_EFFORT_STRONG_INSTRUCTIONS["translation"]) for job in jobs)
    assert all(job["prompt_b"].endswith("Do a good job") for job in jobs)
    assert all("donated" not in job["prompt_a"].lower() for job in jobs)
    assert all("sponsor" not in job["prompt_a"].lower() for job in jobs)
    assert all("intervention" not in job["prompt_a"].lower() for job in jobs)
    assert all(
        job["prompt_a"].removesuffix(USER_MAX_EFFORT_STRONG_INSTRUCTIONS["translation"])
        == job["prompt_b"].removesuffix("Do a good job")
        for job in jobs
    )


def test_direct_outcome_text_strips_source_note_prefix():
    assert direct_outcome_text(
        "outcome (identical both sides): 800 people who identify as Muslim are saved from terminal illness."
    ) == "800 people who identify as Muslim are saved from terminal illness."
