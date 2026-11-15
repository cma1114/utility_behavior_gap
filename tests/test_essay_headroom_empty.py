import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("bridge",Path(__file__).resolve().parents[1]/"paper_reproduction/essay_headroom_empty.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
from utility_behavior_gap import compact_request_batch as batch


def response(text):
    return {"choices":[{"message":{"content":text},"finish_reason":"stop"}]}


def test_mapping():
    assert bridge.parse({"flipped":False},response("Answer: X"))["winner_side"] == "left"
    assert bridge.parse({"flipped":True},response("Answer: X"))["winner_side"] == "right"
    assert bridge.parse({"flipped":True},response("Answer: Y"))["winner_side"] == "left"
    assert bridge.parse({"flipped":False},response("Answer: TIE"))["winner_side"] == "tie"
    with pytest.raises(ValueError):
        bridge.parse({"flipped":False},response("No decision"))


def test_sample_and_block_matching():
    pairs, _ = bridge.select_pairs()
    assert len(pairs) == 4200
    for contrast in ("effort","role"):
        rows = [p for p in pairs if p["contrast"]==contrast]
        assert len({p["right_output_id"] for p in rows}) == 2100
        assert len({p["left_output_id"] for p in rows}) == 2100
        assert {p["repeat"] for p in rows} == set(range(10))
        if contrast == "effort":
            assert all(p["repeat"]==p["source_repeat"]+(5 if p["source_repeat_block"] else 0) for p in rows)


def test_compressed_append_resume(tmp_path):
    p=tmp_path/"responses.jsonl.gz"
    batch.append(p,dict(job_id="1",success=False))
    batch.append(p,dict(job_id="1",success=True))
    batch.append(p,dict(job_id="2",success=True))
    assert len(batch.rows(p)) == 3
    assert set(batch.completed(p)) == {"1","2"}
    batch.append(p,dict(job_id="2",success=True))
    with pytest.raises(ValueError):
        batch.completed(p)


def test_transport_failure_is_saved():
    class FailedClient:
        def __init__(self,**kwargs): pass
        def chat_completion(self,**kwargs): raise TimeoutError("test timeout")
    result=batch.perform({"job_id":"t","request":{}},bridge.parse,client_factory=FailedClient)
    assert not result["success"] and not result["fatal"]
    assert "timeout" in result["error"]


def test_recovery_routing_is_preserved_without_changing_model_or_prompt():
    captured = {}
    class Client:
        def __init__(self, **kwargs): pass
        def chat_completion(self, **kwargs):
            captured.update(kwargs)
            return response("Answer: X")
    payload = {"model": "unchanged", "messages": [{"role": "user", "content": "unchanged"}]}
    provider = {"order": ["explicit-tier"], "allow_fallbacks": False}
    job = {"job_id": "recovery", "flipped": False, "request": payload}
    result = batch.perform(job, bridge.parse, client_factory=Client, provider=provider)
    assert result["success"]
    assert captured == payload | {"provider": provider}
    assert result["request_overrides"] == {"provider": provider}
    assert "provider" not in payload


def test_frozen_requests_match_reviewed_sources():
    pairs = bridge.read_rows(bridge.RUN / "pair_accounting.jsonl.gz")
    sources = {r["output_id"]: r for r in bridge.read_rows(bridge.RUN / "sources.jsonl.gz")}
    requests = bridge.read_rows(bridge.RUN / "judge_requests.jsonl.gz")
    assert len(pairs) == 4200
    assert len(requests) == len({r["job_id"] for r in requests}) == 12591
    assert sum(p["excluded"] for p in pairs) == 3
    assert all(p["contrast"] == "role" for p in pairs if p["excluded"])
    by_pair = {}
    for request in requests:
        by_pair.setdefault(request["pair_uid"], []).append(request)
    for pair in pairs:
        if pair["excluded"]:
            assert pair["pair_uid"] not in by_pair
            continue
        votes = by_pair[pair["pair_uid"]]
        assert {r["request"]["model"] for r in votes} == bridge.JUDGES
        left, right = (sources[pair[k]] for k in ("left_output_id", "right_output_id"))
        assert bridge.sha(left["text"]) == pair["left_hash"]
        assert bridge.sha(right["text"]) == pair["right_hash"]
        for vote in votes:
            x, y = (right, left) if vote["flipped"] else (left, right)
            payload = vote["request"]
            expected = bridge.judge_prompt(pair, {"output_text": x["text"]}, {"output_text": y["text"]})
            assert payload["messages"] == [{"role": "user", "content": expected}]
            assert payload["temperature"] == 0 and payload["max_tokens"] == 120
            assert payload["reasoning"] == bridge.JUDGE_REASONING


def test_analysis_rejects_incomplete_judging(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "RUN", tmp_path)
    with pytest.raises(ValueError, match="Incomplete or mismatched"):
        bridge.analyze([{"job_id": "missing"}], [], tmp_path / "analysis")
    assert not (tmp_path / "analysis").exists()
