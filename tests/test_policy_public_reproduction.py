"""Public evidence checks; no private checkout or API credentials required."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("policy_followup", ROOT / "paper_reproduction/policy_framed_empty.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def test_duplicate_success_is_not_silently_reused():
    with pytest.raises(ValueError, match="Duplicate"):
        policy.unique([{"job_id":"a"}, {"job_id":"a"}], "job_id")


@pytest.mark.parametrize("flipped,expected", [(False,"left"),(True,"right")])
def test_panel_orientation(flipped, expected):
    response = {"choices":[{"message":{"content":"Answer: X"}}]}
    assert policy.judge_side({"flipped":flipped}, response) == expected


def test_public_parallel_table_retains_uncertain_counterpart():
    rows = policy.table.parallel_rows(
        ROOT / "CURRENT_PAPER/features/moral_low_vs_framed_empty_length_controlled_feature_selection.csv",
        ROOT / "docs/reviewer_response/policy_framed_empty/features/feature_results.csv", threshold=.25)
    assert len(rows) == 10 and rows.selected_a.sum() == 9 and rows.selected_b.sum() == 4
    words = rows[rows.task.eq("grant_proposal_abstract") & rows.feature_id.eq("words")].iloc[0]
    assert words.selected_a and not words.selected_b
    assert words.gap_ci_low_b < 0 < words.gap_ci_high_b


def test_unknown_panel_verdict_is_rejected():
    with pytest.raises(ValueError, match="Unresolved"):
        policy.judge_side({"flipped":False}, {"choices":[{"message":{"content":"No decision"}}]})
