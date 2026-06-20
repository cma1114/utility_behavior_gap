"""Minimal OpenRouter chat-completion client."""

from __future__ import annotations

import json
import http.client
import os
import socket
import time
import urllib.error
import urllib.request
from copy import deepcopy
from typing import Any

from utility_behavior_gap.constants import ACTOR_MODEL_ID, ACTORS, JUDGE_MODEL_IDS
from utility_behavior_gap.io_utils import load_env_file
from utility_behavior_gap.paths import ROOT


class MalformedOpenRouterResponse(RuntimeError):
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        snippet = json.dumps(response, sort_keys=True)[:1000]
        super().__init__(f"OpenRouter response did not contain a chat choice: {snippet}")


def load_runtime_env() -> None:
    load_env_file(ROOT / ".env")


def require_openrouter_key() -> str:
    load_runtime_env()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or key == "xxx":
        raise RuntimeError(
            "Set OPENROUTER_API_KEY in .env or the shell environment before running live API calls."
        )
    return key


def actor_model_id(actor: str) -> str:
    try:
        return ACTOR_MODEL_ID[actor]
    except KeyError as exc:
        raise RuntimeError(f"No OpenRouter model id configured for actor {actor!r}.") from exc


def judge_model_ids() -> list[str]:
    return list(JUDGE_MODEL_IDS)


def configured_actor_models() -> dict[str, str]:
    return {actor: actor_model_id(actor) for actor in ACTORS}


class OpenRouterClient:
    def __init__(self, *, timeout_s: float = 120.0, max_retries: int = 3) -> None:
        load_runtime_env()
        self.api_key = require_openrouter_key()
        self.base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.site_url = os.environ.get("OPENROUTER_SITE_URL", "").strip()
        self.app_name = os.environ.get("OPENROUTER_APP_NAME", "Utility-Behavior Gap Reproduction").strip()

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int,
        reasoning: dict[str, Any] | None = None,
        provider: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if provider is not None:
            payload["provider"] = provider
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url and self.site_url != "xxx":
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    raw_body = response.read().decode("utf-8", errors="replace")
                return json.loads(raw_body)
            except json.JSONDecodeError as exc:
                # Non-JSON / truncated body (provider hiccup) — retry, then surface as transient.
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"OpenRouter request failed: malformed non-JSON response (temporary): {raw_body[:300]!r}"
                    ) from exc
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if attempt == self.max_retries or exc.code < 500:
                    raise RuntimeError(f"OpenRouter request failed ({exc.code}): {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
            except http.client.IncompleteRead as exc:
                if attempt == self.max_retries:
                    raise RuntimeError("OpenRouter request failed: incomplete response body") from exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"OpenRouter request timed out after {self.timeout_s}s") from exc
            time.sleep(min(2**attempt, 10))
        raise RuntimeError("OpenRouter request failed after retries")


def response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not choices:
        raise MalformedOpenRouterResponse(response)
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise MalformedOpenRouterResponse(response)
    content = message.get("content", "")
    return "" if content is None else str(content)


def response_without_message_content(response: dict[str, Any]) -> dict[str, Any]:
    """Return an audit copy of a response without duplicating assistant text.

    The scripts store extracted text in canonical fields such as ``output_text``
    or ``vote_raw``. Keeping the same text again inside
    ``raw_response.choices[*].message.content`` makes JSONL logs hard to browse
    without adding information, so the logging path omits only that nested
    content while preserving ids, model names, usage, finish reasons, and other
    provider metadata.
    """

    sanitized = deepcopy(response)
    choices = sanitized.get("choices")
    if not isinstance(choices, list):
        return sanitized
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict) or "content" not in message:
            continue
        content = message.pop("content")
        message["content_omitted"] = True
        message["content_char_count"] = 0 if content is None else len(str(content))
        message["content_was_null"] = content is None
    return sanitized
