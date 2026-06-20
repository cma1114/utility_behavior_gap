"""Helpers for deciding whether an API failure is safe to retry later."""

from __future__ import annotations

import re


_TRANSIENT_API_ERROR_RE = re.compile(
    r"\b(?:429|500|502|503|504)\b|too many requests|rate limit|timeout|timed out|aborted|"
    r"temporar|network|urlopen|connection|dns|resolve|nodename|servname|unreachable|reset|refused|"
    r"incomplete response|incompleteread|incomplete read|chunk",
    flags=re.IGNORECASE,
)

_FATAL_API_ERROR_RE = re.compile(
    r"\b400\b|mandatory|cannot be disabled|unsupported|invalid request|require_parameters",
    flags=re.IGNORECASE,
)

_RATE_LIMIT_API_ERROR_RE = re.compile(r"\b429\b|too many requests|rate limit", flags=re.IGNORECASE)
_BACKOFF_API_ERROR_RE = re.compile(
    r"\b(?:429|500|502|503|504)\b|too many requests|rate limit|timeout|timed out|aborted|"
    r"temporar|network|urlopen|connection|dns|resolve|nodename|servname|unreachable|reset|refused|"
    r"incomplete response|incompleteread|incomplete read|chunk",
    flags=re.IGNORECASE,
)


def is_transient_api_error(error: BaseException) -> bool:
    """Return whether an API error should be logged and retried later.

    Fatal configuration errors must still stop the script immediately. Transient
    provider failures should not kill an unattended batch because every request
    has a stable output id and can be retried without duplicating completed work.
    """

    text = str(error)
    if _FATAL_API_ERROR_RE.search(text):
        return False
    return bool(_TRANSIENT_API_ERROR_RE.search(text))


def is_rate_limit_api_error(error: BaseException) -> bool:
    """Return whether an API error should pause the current request pass."""

    return bool(_RATE_LIMIT_API_ERROR_RE.search(str(error)))


def is_backoff_api_error(error: BaseException) -> bool:
    """Return whether a transient error should stop this pass and resume later."""

    text = str(error)
    if _FATAL_API_ERROR_RE.search(text):
        return False
    return bool(_BACKOFF_API_ERROR_RE.search(text))
