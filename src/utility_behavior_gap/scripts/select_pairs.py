#!/usr/bin/env python3
"""Select high-low pairs from fixed fitted utility inputs."""

from __future__ import annotations

import argparse

from utility_behavior_gap.pair_selection import write_selected_pairs
from utility_behavior_gap.paths import OUTPUT_INPUTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-pairs-per-cell", type=int, default=80)
    parser.add_argument("--same-count-pairs-per-cell", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    write_selected_pairs(
        default_pairs_per_cell=args.default_pairs_per_cell,
        same_count_pairs_per_cell=args.same_count_pairs_per_cell,
        seed=args.seed,
    )
    print(f"wrote {OUTPUT_INPUTS / 'selected_pairs.csv'}")


if __name__ == "__main__":
    main()
