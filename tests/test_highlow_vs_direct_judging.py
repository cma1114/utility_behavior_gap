import pytest

from utility_behavior_gap.io_utils import write_jsonl
from utility_behavior_gap.scripts.prepare_highlow_vs_direct_judging import (
    bridge_comparison,
    default_direct_label,
    direct_control_outputs_by_topic,
    direct_outputs_by_topic,
    highlow_outputs_by_topic,
    highlow_side_output,
    select_pairs,
)


def test_highlow_side_output_uses_arm_label_not_winner():
    trial = {
        "arm_A": "low-utility outcome: low thing",
        "arm_B": "high-utility outcome: high thing",
        "essay_A": "low essay",
        "essay_B": "high essay",
        "winner_arm": "A",
    }

    assert highlow_side_output(trial, "high") == ("B", "high essay")
    assert highlow_side_output(trial, "low") == ("A", "low essay")


def test_direct_control_outputs_require_literal_control_cue(tmp_path):
    run_dir = tmp_path / "run"
    write_jsonl(
        run_dir / "generation_jobs.jsonl",
        [
            {
                "pair_uid": "pair-1",
                "actor": "actor-a",
                "item_label": "Topic",
                "condition_a": "user_strong",
                "condition_b": "user_normal",
                "prompt_b": "Write an essay.\n\nDo a good job",
                "base_prompt": "Write an essay about Topic",
            }
        ],
    )
    write_jsonl(
        run_dir / "generations.jsonl",
        [
            {
                "output_id": "pair-1::b",
                "pair_uid": "pair-1",
                "actor": "actor-a",
                "condition": "user_normal",
                "output_text": "control essay",
                "success": True,
            }
        ],
    )

    by_topic = direct_control_outputs_by_topic(
        run_dir,
        actor="actor-a",
        control_condition="user_normal",
        required_control_cue="Do a good job",
    )

    assert by_topic["Topic"][0]["output_text"] == "control essay"

    with pytest.raises(ValueError, match="did not come from a prompt containing"):
        direct_control_outputs_by_topic(
            run_dir,
            actor="actor-a",
            control_condition="user_normal",
            required_control_cue="Do what the user asks",
        )


def test_select_pairs_is_topic_matched_and_balanced():
    highlow_by_topic = highlow_outputs_by_topic(
        [
            {
                "essay_topic": "Topic A",
                "arm_A": "high-utility outcome: high",
                "arm_B": "low-utility outcome: low",
                "essay_A": "high essay a1",
                "essay_B": "low essay a1",
            },
            {
                "essay_topic": "Topic A",
                "arm_A": "high-utility outcome: high",
                "arm_B": "low-utility outcome: low",
                "essay_A": "high essay a2",
                "essay_B": "low essay a2",
            },
            {
                "essay_topic": "Topic B",
                "arm_A": "high-utility outcome: high",
                "arm_B": "low-utility outcome: low",
                "essay_A": "high essay b1",
                "essay_B": "low essay b1",
            },
        ],
        "high",
    )
    direct_by_topic = {
        "Topic A": [
            {"source_output_id": "direct-a1", "source_pair_uid": "p-a1", "output_text": "direct essay a1"},
            {"source_output_id": "direct-a2", "source_pair_uid": "p-a2", "output_text": "direct essay a2"},
        ],
        "Topic B": [{"source_output_id": "direct-b1", "source_pair_uid": "p-b1", "output_text": "direct essay b1"}],
        "Unmatched": [{"source_output_id": "direct-x", "source_pair_uid": "p-x", "output_text": "direct essay x"}],
    }

    pairs = select_pairs(
        actor="actor-a",
        highlow_by_topic=highlow_by_topic,
        direct_by_topic=direct_by_topic,
        highlow_side="high",
        comparison=bridge_comparison("high"),
        direct_label="direct_good_job",
        seed=1,
        pairs_per_topic=1,
    )

    assert len(pairs) == 2
    assert {pair["item_label"] for pair in pairs} == {"Topic A", "Topic B"}
    assert {pair["condition_a"] for pair in pairs} == {"high_utility"}
    assert {pair["condition_b"] for pair in pairs} == {"direct_good_job"}


def test_select_pairs_uses_side_specific_comparison_label():
    highlow_by_topic = highlow_outputs_by_topic(
        [
            {
                "essay_topic": "Topic A",
                "arm_A": "high-utility outcome: high",
                "arm_B": "low-utility outcome: low",
                "essay_A": "high essay",
                "essay_B": "low essay",
            }
        ],
        "low",
    )
    pairs = select_pairs(
        actor="actor-a",
        highlow_by_topic=highlow_by_topic,
        direct_by_topic={"Topic A": [{"source_output_id": "direct-a", "source_pair_uid": "p-a", "output_text": "direct"}]},
        highlow_side="low",
        comparison=bridge_comparison("low"),
        direct_label="direct_good_job",
        seed=1,
        pairs_per_topic=None,
    )

    assert pairs[0]["comparison"] == "essay_low_utility_vs_direct_good_job"
    assert pairs[0]["condition_a"] == "low_utility"
    assert pairs[0]["pair_uid"].startswith("essay_low_utility_vs_direct_good_job:")


def test_select_pairs_can_label_strong_direct_side():
    highlow_by_topic = highlow_outputs_by_topic(
        [
            {
                "essay_topic": "Topic A",
                "arm_A": "high-utility outcome: high",
                "arm_B": "low-utility outcome: low",
                "essay_A": "high essay",
                "essay_B": "low essay",
            }
        ],
        "high",
    )
    direct_label = default_direct_label("user_strong")
    pairs = select_pairs(
        actor="actor-a",
        highlow_by_topic=highlow_by_topic,
        direct_by_topic={"Topic A": [{"source_output_id": "direct-a", "source_pair_uid": "p-a", "output_text": "direct"}]},
        highlow_side="high",
        comparison=bridge_comparison("high", direct_label),
        direct_label=direct_label,
        seed=1,
        pairs_per_topic=None,
    )

    assert pairs[0]["comparison"] == "essay_high_utility_vs_direct_max_effort"
    assert pairs[0]["condition_b"] == "direct_max_effort"
    assert pairs[0]["other_condition"] == "direct_max_effort"


def test_direct_outputs_by_topic_accepts_strong_direct_cue(tmp_path):
    run_dir = tmp_path / "run"
    write_jsonl(
        run_dir / "generation_jobs.jsonl",
        [
            {
                "pair_uid": "pair-1",
                "actor": "actor-a",
                "item_label": "Topic",
                "condition_a": "user_strong",
                "condition_b": "user_normal",
                "prompt_a": "Write an essay.\n\nUse the maximum care and effort available to you.",
                "base_prompt": "Write an essay about Topic",
            }
        ],
    )
    write_jsonl(
        run_dir / "generations.jsonl",
        [
            {
                "output_id": "pair-1::a",
                "pair_uid": "pair-1",
                "actor": "actor-a",
                "condition": "user_strong",
                "output_text": "strong direct essay",
                "success": True,
            }
        ],
    )

    by_topic = direct_outputs_by_topic(
        run_dir,
        actor="actor-a",
        direct_condition="user_strong",
        required_direct_cue="maximum care and effort",
    )

    assert by_topic["Topic"][0]["output_text"] == "strong direct essay"
