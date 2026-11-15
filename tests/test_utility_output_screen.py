import importlib.util
from pathlib import Path

import pytest
import pandas as pd

spec=importlib.util.spec_from_file_location("screen",Path(__file__).resolve().parents[1]/"paper_reproduction/utility_output_screen.py")
screen=importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def test_classifier_must_return_complete_valid_json():
    good={"choices":[{"finish_reason":"stop","message":{"content":'{"label":"clean","evidence":""}'}}]}
    assert screen.parse({},good)=={"label":"clean","evidence":""}
    good["choices"][0]["finish_reason"]="length"
    with pytest.raises(ValueError,match="not complete"):screen.parse({},good)
    with pytest.raises(ValueError):screen.parse({}, {"choices":[{"finish_reason":"stop","message":{"content":"No"}}]})


def test_source_context_and_full_output_are_present_without_condition_identity():
    source={"task":"translation","item_label":"test source fragment","text":"translated text "*2000,"actor":"hidden actor","arm":"hidden low arm"}
    prompt=screen.screen_prompt(source)
    assert source["text"] in prompt and source["item_label"] in prompt
    assert source["actor"] not in prompt and source["arm"] not in prompt
    assert "$1,000 donation" not in prompt
    assert "source fragment is not truncated" in prompt


def test_analysis_rejects_incomplete_labels(tmp_path,monkeypatch):
    monkeypatch.setattr(screen,"RUN",tmp_path)
    with pytest.raises(ValueError,match="incomplete"):
        screen.initial_labels([{"job_id":"missing"}])


def test_manual_review_is_required_and_hash_bound(tmp_path,monkeypatch):
    monkeypatch.setattr(screen,"RUN",tmp_path)
    source={"output_id":"x","output_hash":"hash","actor":"a","task":"essay","domain":"political","arm":"low","pair_uid":"p","mechanical_flags":[]}
    label={"output_id":"x","output_hash":"hash","label":"full_refusal","evidence":"declined","provenance":"test"}
    monkeypatch.setattr(screen,"initial_labels",lambda jobs:{"x":label})
    with pytest.raises(ValueError,match="Manual review incomplete"):
        screen.adjudicated_labels([], {"x":source})
    monkeypatch.setattr(screen,"read",lambda path:[{"output_id":"x","output_hash":"wrong","label":"clean","reason":"not a refusal"}])
    with pytest.raises(AssertionError):screen.adjudicated_labels([], {"x":source})


def test_either_output_excludes_pair_without_changing_verdict():
    frame=pd.DataFrame([dict(contrast="highlow",pair_uid="p",left_output_id="h",right_output_id="l",winner="hl_high")])
    for labels in ({"h":"clean","l":"full_refusal"},{"h":"degenerate","l":"clean"}):
        result=screen.attach_pair_labels(frame,labels)
        assert result.excluded.tolist()==[True]
        assert result.winner.tolist()==["hl_high"]
    assert screen.attach_pair_labels(frame,{"h":"clean","l":"clean"}).excluded.tolist()==[False]
    with pytest.raises(ValueError,match="Missing output"):
        screen.attach_pair_labels(frame,{"h":"clean"})
    frame.loc[0,"winner"]=None
    with pytest.raises(ValueError,match="Unrecognized panel"):
        screen.attach_pair_labels(frame,{"h":"clean","l":"clean"})


def test_equal_cell_mean_and_ties_follow_paper(monkeypatch):
    from utility_behavior_gap.scripts import consolidate_aggregate_cis as ci
    captured=[]
    def bootstrap(arrays,seed,iterations):
        captured.extend(arrays)
        return sum(x.mean() for x in arrays)/len(arrays),.1,.9,len(arrays)
    monkeypatch.setattr(ci,"bootstrap_equal_cell",bootstrap)
    frame=pd.DataFrame([dict(actor="a",winner="hl_high",target_condition="hl_high")]*3
                       +[dict(actor="b",winner="hl_low",target_condition="hl_high")]
                       +[dict(actor="b",winner="tie",target_condition="hl_high")])
    result=screen.summarize_scores(frame,["actor"],seed=1)
    assert result["mean"]==.5 and result["non_tied"]==4 and result["ties"]==1
    assert [len(x) for x in captured]==[3,1]


def test_screen_stage_resumes_without_network(tmp_path,monkeypatch):
    batch=screen.batch
    batch.append(tmp_path/"screen_responses.jsonl.gz",dict(job_id="s",success=True))
    monkeypatch.setattr(batch,"account_usage",lambda:pytest.fail("Resume made a network request"))
    batch.run(tmp_path,[dict(job_id="s")],screen.parse,stage="screen")
    assert not (tmp_path/"judge_responses.jsonl.gz").exists()
