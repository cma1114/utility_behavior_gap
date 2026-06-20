import pytest

from utility_behavior_gap.analysis import comparison_counts
from utility_behavior_gap.io_utils import write_jsonl
from utility_behavior_gap.judging import derive_counted_winner_condition
from utility_behavior_gap.scripts.aggregate_judgments import votes_by_pair
from utility_behavior_gap.stats import parse_entity_count, spearman, wilson_ci


def test_comparison_counts_uses_counted_winner_only():
    rows = [
        {"counted_winner_condition": "high"},
        {"counted_winner_condition": "low"},
        {"counted_winner_condition": "tie"},
        {"counted_winner_condition": ""},
    ]

    assert comparison_counts(rows, "high", "low") == (1, 1, 1)


def test_wilson_ci_has_expected_shape():
    rate, lo, hi = wilson_ci(5, 10)

    assert rate == 0.5
    assert 0.2 < lo < rate
    assert rate < hi < 0.8


def test_parse_entity_count_and_spearman():
    assert parse_entity_count(
        "100 people who identify as Buddhist are saved from terminal illness.",
        "religions",
    ) == ("buddhist", 100)

    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)


def test_legacy_essay_highlow_disagreement_is_dropped():
    row = {
        "comparison": "highlow_main",
        "source_run": "bg_fixed_topic_default",
        "predicted_condition": "high",
        "other_condition": "low",
    }
    votes = ["high", "low", "tie"]

    assert derive_counted_winner_condition(row, votes) == ""


def test_nonessay_highlow_split_panel_counts_as_tie():
    row = {
        "comparison": "highlow_main",
        "source_run": "highlow_scaleup_translation_v1",
        "predicted_condition": "high",
        "other_condition": "low",
    }
    votes = ["high", "low", "tie"]

    assert derive_counted_winner_condition(row, votes) == "tie"


def test_calibration_unresolved_panel_counts_as_tie():
    row = {
        "comparison": "system_prompt",
        "source_run": "essay_system_prompt",
        "predicted_condition": "sys_strong",
        "other_condition": "sys_normal",
    }
    votes = ["unresolved", "unresolved", "tie"]

    assert derive_counted_winner_condition(row, votes) == "tie"


def test_votes_by_pair_prefers_current_hashed_votes_over_legacy_rows(tmp_path):
    path = tmp_path / "judge_votes.jsonl"
    write_jsonl(
        path,
        [
            {
                "pair_uid": "pair-1",
                "judge_model": "judge",
                "winner_condition": "tie",
            },
            {
                "pair_uid": "pair-1",
                "judge_model": "judge",
                "winner_condition": "sys_strong",
                "source_output_a_hash": "a",
                "source_output_b_hash": "b",
            },
        ],
    )

    votes = votes_by_pair(path, {"pair-1": ("a", "b")})

    assert [row["winner_condition"] for row in votes["pair-1"]] == ["sys_strong"]
