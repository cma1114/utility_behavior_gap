#!/usr/bin/env python3
"""Build derived paper tables from generated judged-pair records."""

from __future__ import annotations

from utility_behavior_gap.analysis import aggregate_all
from utility_behavior_gap.paths import ROOT


def main() -> None:
    aggregate_all(ROOT)
    print(f"wrote derived tables under {ROOT / 'outputs'}")


if __name__ == "__main__":
    main()
