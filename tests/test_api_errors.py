from utility_behavior_gap.api_errors import is_rate_limit_api_error, is_transient_api_error


def test_transient_api_errors_are_retryable():
    assert is_transient_api_error(RuntimeError("OpenRouter request failed (504): The operation was aborted"))
    assert is_transient_api_error(RuntimeError("OpenRouter request failed (429): Rate limit exceeded"))
    assert is_transient_api_error(RuntimeError("OpenRouter request failed: timed out"))
    assert is_transient_api_error(RuntimeError("OpenRouter request failed: incomplete response body"))


def test_configuration_api_errors_remain_fatal():
    assert not is_transient_api_error(
        RuntimeError("OpenRouter request failed (400): Reasoning is mandatory and cannot be disabled.")
    )
    assert not is_transient_api_error(RuntimeError("OpenRouter request failed (400): invalid request"))


def test_rate_limit_errors_are_detected_separately():
    assert is_rate_limit_api_error(RuntimeError("OpenRouter request failed (429): Rate limit exceeded"))
    assert is_rate_limit_api_error(RuntimeError("too many requests"))
    assert not is_rate_limit_api_error(RuntimeError("OpenRouter request failed (504): The operation was aborted"))
