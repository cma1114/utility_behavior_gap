"""Small statistical helpers used by the reproduction pipeline."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable


RELIGIONS_RE = re.compile(
    r"^\s*(\d+)\s+(?:person|people)\s+who\s+(?:identify|identifies)\s+as\s+(\w+)",
    re.IGNORECASE,
)

COUNTRIES_RE = re.compile(
    r"^\s*(\d+)\s+(?:person|people)\s+from\s+([A-Z][A-Za-z\s]+?)\s+(?:is|are)\s+saved",
    re.IGNORECASE,
)

ANIMALS_RE = re.compile(
    r"^\s*(\d+)\s+([A-Za-z][A-Za-z\s\-]+?)\s+(?:is|are)\s+saved\s+from\s+dying",
    re.IGNORECASE,
)


def wilson_ci(wins: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    p = wins / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals)


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx = rank(xs)
    ry = rank(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denom_x = sum((a - mx) ** 2 for a in rx) ** 0.5
    denom_y = sum((b - my) ** 2 for b in ry) ** 0.5
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def parse_entity_count(description: str, domain: str) -> tuple[str, int] | None:
    if domain == "religions":
        match = RELIGIONS_RE.match(description)
        if match:
            return match.group(2).lower(), int(match.group(1))
    if domain == "countries":
        match = COUNTRIES_RE.match(description)
        if match:
            return match.group(2).strip().lower(), int(match.group(1))
    if domain == "animals":
        match = ANIMALS_RE.match(description)
        if match:
            return match.group(2).strip().lower(), int(match.group(1))
    return None
