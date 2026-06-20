import json

import pytest

from utility_behavior_gap.scripts.run_generation import (
    GENERATION_PROVIDER,
    GENERATION_REASONING,
    append_generation_failure,
    finish_reason,
    generation_request_snapshot,
    is_valid_generation_row,
    reasoning_tokens,
    validate_generation_response,
)


def response_with(*, finish_reason="stop", reasoning=0):
    return {
        "choices": [{"finish_reason": finish_reason, "message": {"content": "ok"}}],
        "usage": {"completion_tokens_details": {"reasoning_tokens": reasoning}},
    }


def test_generation_reasoning_is_explicitly_disabled():
    assert GENERATION_REASONING == {"effort": "none"}


def test_generation_requires_provider_parameter_support():
    assert GENERATION_PROVIDER == {"require_parameters": True}


def test_generation_request_snapshot_contains_literal_messages(monkeypatch):
    monkeypatch.setattr("sys.argv", ["python", "-m", "utility_behavior_gap.scripts.run_generation"])
    snapshot = generation_request_snapshot(
        model="openai/gpt-5.4-mini",
        request={
            "output_id": "pair::a",
            "pair_uid": "pair",
            "condition": "sys_strong",
            "system_prompt": "system text",
            "prompt": "user text",
        },
        temperature=None,
        max_tokens=900,
    )

    assert snapshot["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]
    assert snapshot["reasoning"] == GENERATION_REASONING
    assert snapshot["provider"] == GENERATION_PROVIDER


def test_reasoning_tokens_reads_missing_as_zero():
    assert reasoning_tokens({}) == 0


def test_finish_reason_treats_provider_null_as_missing():
    assert finish_reason({"choices": [{"finish_reason": None}]}) == ""


def test_generation_validation_rejects_reasoning_tokens():
    with pytest.raises(RuntimeError, match="reasoning tokens"):
        validate_generation_response(
            output_id="out",
            text="Essay text.",
            response=response_with(reasoning=12),
            reason="stop",
        )


def test_generation_validation_rejects_length_finish():
    with pytest.raises(RuntimeError, match="finish_reason='length'"):
        validate_generation_response(
            output_id="out",
            text="Essay text.",
            response=response_with(finish_reason="length"),
            reason="length",
        )


def test_generation_validation_rejects_empty_text():
    with pytest.raises(RuntimeError, match="empty output_text"):
        validate_generation_response(
            output_id="out",
            text="",
            response=response_with(),
            reason="stop",
        )


def test_existing_generation_row_must_be_nonempty_stop_and_nonreasoning():
    assert is_valid_generation_row(
        {
            "output_text": "Essay text.",
            "finish_reason": "stop",
            "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
        }
    )
    assert not is_valid_generation_row(
        {
            "output_text": "",
            "finish_reason": "stop",
            "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
        }
    )
    assert not is_valid_generation_row(
        {
            "output_text": "Essay text.",
            "finish_reason": "length",
            "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
        }
    )
    assert is_valid_generation_row(
        {
            "output_text": "Essay text.",
            "finish_reason": "",
            "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
        }
    )
    assert not is_valid_generation_row(
        {
            "output_text": "Essay text.",
            "finish_reason": "stop",
            "usage": {"completion_tokens_details": {"reasoning_tokens": 1}},
        }
    )


def test_append_generation_failure_records_malformed_response(monkeypatch, tmp_path):
    failure_path = tmp_path / "generation_failures.jsonl"
    monkeypatch.setattr("utility_behavior_gap.scripts.run_generation.GENERATION_FAILURES", failure_path)

    append_generation_failure(
        job={"actor": "gpt-5.4-mini-or"},
        request={"output_id": "pair::a", "pair_uid": "pair", "condition": "user_strong"},
        model="openai/gpt-5.4-mini",
        text="",
        raw_response={"error": {"code": 504, "message": "The operation was aborted"}},
        reason="malformed_response",
        error="OpenRouter response did not contain a chat choice",
        latency_s=1.2,
        temperature=None,
        max_tokens=900,
    )

    text = failure_path.read_text(encoding="utf-8")
    assert '"finish_reason": "malformed_response"' in text
    assert '"output_id": "pair::a"' in text


def test_append_generation_failure_omits_duplicate_raw_response_content(monkeypatch, tmp_path):
    failure_path = tmp_path / "generation_failures.jsonl"
    monkeypatch.setattr("utility_behavior_gap.scripts.run_generation.GENERATION_FAILURES", failure_path)

    append_generation_failure(
        job={"actor": "gpt-5.4-mini-or"},
        request={"output_id": "pair::a", "pair_uid": "pair", "condition": "user_strong"},
        model="openai/gpt-5.4-mini",
        text="Essay text.",
        raw_response={
            "choices": [{"finish_reason": "length", "message": {"content": "Essay text."}}],
            "usage": {"completion_tokens": 2},
        },
        reason="length",
        error="partial output",
        latency_s=1.2,
        temperature=None,
        max_tokens=900,
    )

    row = json.loads(failure_path.read_text(encoding="utf-8"))
    assert row["output_text"] == "Essay text."
    message = row["raw_response"]["choices"][0]["message"]
    assert "content" not in message
    assert message["content_omitted"] is True
    assert message["content_char_count"] == len("Essay text.")


def test_append_generation_failure_records_transient_api_error(monkeypatch, tmp_path):
    failure_path = tmp_path / "generation_failures.jsonl"
    monkeypatch.setattr("utility_behavior_gap.scripts.run_generation.GENERATION_FAILURES", failure_path)

    append_generation_failure(
        job={"actor": "gpt-5.4-mini-or"},
        request={"output_id": "pair::b", "pair_uid": "pair", "condition": "user_normal"},
        model="openai/gpt-5.4-mini",
        text="",
        raw_response={},
        reason="api_error",
        error="OpenRouter request failed (504): The operation was aborted",
        latency_s=2.4,
        temperature=None,
        max_tokens=900,
    )

    text = failure_path.read_text(encoding="utf-8")
    assert '"finish_reason": "api_error"' in text
    assert '"output_id": "pair::b"' in text
    assert "504" in text
