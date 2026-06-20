import pytest

from utility_behavior_gap.io_utils import write_jsonl
from utility_behavior_gap.judging import derive_judge_verdict
from utility_behavior_gap.scripts.run_judging import (
    JUDGE_REASONING,
    judging_request_snapshot,
    pending_judge_requests,
    read_jobs_from_args,
    vote_matches_current_outputs,
)


def _job(uid: str) -> dict:
    return {"pair_uid": uid}


def _gens(uid: str) -> dict:
    return {f"{uid}::a": {"output_id": f"{uid}:a", "condition": "x", "output_text": "aa"},
            f"{uid}::b": {"output_id": f"{uid}:b", "condition": "y", "output_text": "bb"}}


def test_pending_both_orders_emits_one_request_per_orientation():
    pending = pending_judge_requests(
        jobs=[_job("p1")], generations=_gens("p1"), done=set(),
        judges=["j1", "j2"], seed=1, limit=None, orders="both",
    )
    keys = [(job["pair_uid"], judge, flip) for job, _, _, _, judge, flip in pending]
    assert sorted(keys) == [("p1", "j1", False), ("p1", "j1", True),
                            ("p1", "j2", False), ("p1", "j2", True)]


def test_pending_both_orders_tops_up_missing_orientation_only():
    done = {("p1", "j1", False)}
    pending = pending_judge_requests(
        jobs=[_job("p1")], generations=_gens("p1"), done=done,
        judges=["j1"], seed=1, limit=None, orders="both",
    )
    assert [(judge, flip) for _, _, _, _, judge, flip in pending] == [("j1", True)]


def test_pending_single_order_skips_pairs_judged_in_any_orientation():
    done = {("p1", "j1", True)}
    pending = pending_judge_requests(
        jobs=[_job("p1")], generations=_gens("p1"), done=done,
        judges=["j1"], seed=1, limit=None, orders="single",
    )
    assert pending == []


def test_derive_judge_verdict_collapses_orientation_flips_to_tie():
    assert derive_judge_verdict(["high", "high"]) == "high"
    assert derive_judge_verdict(["high", "low"]) == "tie"
    assert derive_judge_verdict(["high", "tie"]) == "tie"
    assert derive_judge_verdict(["tie", "tie"]) == "tie"
    assert derive_judge_verdict(["high"]) == "high"  # single-order passthrough
    assert derive_judge_verdict(["unresolved", "high"]) == "high"
    assert derive_judge_verdict(["unresolved"]) == "unresolved"


def test_vote_matches_current_outputs_requires_both_hashes():
    assert vote_matches_current_outputs(
        {"source_output_a_hash": "a", "source_output_b_hash": "b"},
        ("a", "b"),
    )
    assert not vote_matches_current_outputs({}, ("a", "b"))
    assert not vote_matches_current_outputs(
        {"source_output_a_hash": "old-a", "source_output_b_hash": "b"},
        ("a", "b"),
    )


def test_judging_request_snapshot_contains_literal_prompt(monkeypatch):
    monkeypatch.setattr("sys.argv", ["python", "-m", "utility_behavior_gap.scripts.run_judging"])
    snapshot = judging_request_snapshot(
        judge_model="openai/gpt-5-chat",
        prompt="Which essay is better?",
        temperature=0.0,
        max_tokens=120,
        seed=12345,
        flipped=True,
        source_output_a_id="pair::a",
        source_output_b_id="pair::b",
        displayed_output_a_id="pair::b",
        displayed_output_b_id="pair::a",
    )

    assert snapshot["messages"] == [{"role": "user", "content": "Which essay is better?"}]
    assert snapshot["reasoning"] == JUDGE_REASONING
    assert snapshot["flipped"] is True
    assert snapshot["source_output_a_id"] == "pair::a"
    assert snapshot["displayed_output_a_id"] == "pair::b"


def test_read_jobs_from_args_uses_explicit_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    write_jsonl(run_dir / "generation_jobs.jsonl", [{"pair_uid": "pair-1"}])

    assert read_jobs_from_args(run_dir, None) == [{"pair_uid": "pair-1"}]


def test_read_jobs_from_args_rejects_two_explicit_sources(tmp_path):
    with pytest.raises(ValueError, match="either --run-dir or --manifest"):
        read_jobs_from_args(tmp_path / "run", tmp_path / "manifest.jsonl")
