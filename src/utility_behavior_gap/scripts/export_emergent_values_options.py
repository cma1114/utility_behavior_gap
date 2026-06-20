#!/usr/bin/env python3
"""Export this repo's utility outcomes for the CAIS emergent-values pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utility_behavior_gap.io_utils import read_csv_rows, write_csv_rows
from utility_behavior_gap.paths import INPUTS, OUTPUT_INPUTS


DEFAULT_OPTIONS_JSON = OUTPUT_INPUTS / "emergent_values_options.json"
DEFAULT_MANIFEST_CSV = OUTPUT_INPUTS / "emergent_values_options_manifest.csv"


def rows_for_actor(rows: list[dict[str, str]], actor: str) -> list[dict[str, str]]:
    out = [row for row in rows if row["actor"] == actor]
    if not out:
        raise ValueError(f"actor {actor!r} not found in utility_options.csv")
    return out


def validate_shared_option_layout(rows: list[dict[str, str]], source_rows: list[dict[str, str]]) -> None:
    source_layout = [(row["domain"], row["option_id"], row["description"]) for row in source_rows]
    by_actor: dict[str, list[tuple[str, str, str]]] = {}
    for row in rows:
        by_actor.setdefault(row["actor"], []).append((row["domain"], row["option_id"], row["description"]))
    mismatched = [actor for actor, layout in by_actor.items() if layout != source_layout]
    if mismatched:
        raise ValueError(
            "utility option layout differs across actors; cannot safely export a shared option list. "
            f"Mismatched actors: {', '.join(sorted(mismatched))}"
        )


def build_manifest(source_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    manifest = []
    seen_descriptions: set[str] = set()
    for idx, row in enumerate(source_rows):
        description = row["description"]
        if description in seen_descriptions:
            raise ValueError(f"duplicate outcome description: {description!r}")
        seen_descriptions.add(description)
        manifest.append(
            {
                "utility_option_index": idx,
                "domain": row["domain"],
                "domain_label": row["domain_label"],
                "option_id": row["option_id"],
                "description": description,
            }
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-actor", default="deepseek-v3.2-or")
    parser.add_argument("--options-json", default=str(DEFAULT_OPTIONS_JSON))
    parser.add_argument("--manifest-csv", default=str(DEFAULT_MANIFEST_CSV))
    args = parser.parse_args()

    rows = read_csv_rows(INPUTS / "utility_options.csv")
    source_rows = rows_for_actor(rows, args.source_actor)
    validate_shared_option_layout(rows, source_rows)
    manifest = build_manifest(source_rows)
    options = [row["description"] for row in manifest]

    options_json = Path(args.options_json)
    manifest_csv = Path(args.manifest_csv)

    options_json.parent.mkdir(parents=True, exist_ok=True)
    with open(options_json, "w", encoding="utf-8") as f:
        json.dump(options, f, indent=2)
        f.write("\n")
    write_csv_rows(manifest_csv, manifest)
    print(f"wrote {len(options)} options to {options_json}")
    print(f"wrote option manifest to {manifest_csv}")


if __name__ == "__main__":
    main()
