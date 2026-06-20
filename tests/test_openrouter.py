from utility_behavior_gap.openrouter import response_text, response_without_message_content


def test_response_without_message_content_omits_only_nested_content():
    response = {
        "id": "gen-123",
        "model": "test/model",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Essay text.",
                    "refusal": None,
                },
            }
        ],
        "usage": {"completion_tokens": 3},
    }

    sanitized = response_without_message_content(response)

    assert response_text(response) == "Essay text."
    assert response["choices"][0]["message"]["content"] == "Essay text."
    assert sanitized["id"] == "gen-123"
    assert sanitized["model"] == "test/model"
    assert sanitized["usage"] == {"completion_tokens": 3}
    assert sanitized["choices"][0]["finish_reason"] == "stop"
    message = sanitized["choices"][0]["message"]
    assert "content" not in message
    assert message["content_omitted"] is True
    assert message["content_char_count"] == len("Essay text.")
    assert message["content_was_null"] is False


def test_response_without_message_content_records_null_content():
    response = {"choices": [{"message": {"role": "assistant", "content": None}}]}

    sanitized = response_without_message_content(response)

    message = sanitized["choices"][0]["message"]
    assert "content" not in message
    assert message["content_omitted"] is True
    assert message["content_char_count"] == 0
    assert message["content_was_null"] is True
