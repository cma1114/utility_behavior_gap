"""Stable fingerprints for generated artifacts."""

from __future__ import annotations

import hashlib
from typing import Any


def text_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def output_text_fingerprint(row: dict[str, Any]) -> str:
    return text_fingerprint(str(row.get("output_text", "")))
