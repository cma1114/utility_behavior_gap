#!/usr/bin/env python3
"""Convert CAIS emergent-values utility output into this repo's CSV schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utility_behavior_gap.io_utils import read_csv_rows, write_csv_rows
from utility_behavior_gap.paths import INPUTS, OUTPUT_INPUTS


UTILITY_COLUMNS = [
    "actor",
    "actor_label",
    "domain",
    "domain_label",
    "option_id",
    "description",
    "utility_mean",
    "utility_variance",
    "train_accuracy",
    "train_log_loss",
    "holdout_accuracy",
    "holdout_log_loss",
]


def metric(metrics: dict[str, Any] | None, name: str) -> str:
    if not metrics or metrics.get(name) is None:
        return ""
    return str(metrics[name])


def base_metadata_by_description() -> dict[str, dict[str, str]]:
    rows = read_csv_rows(INPUTS / "utility_options.csv")
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        description = row["description"]
        if description not in out:
            out[description] = {
                "domain": row["domain"],
                "domain_label": row["domain_label"],
                "option_id": row["option_id"],
                "description": description,
            }
    return out


def load_results(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if "options" not in data or "utilities" not in data:
        raise ValueError("results JSON must contain 'options' and 'utilities'")
    return data


def utility_for_option(utilities: dict[str, Any], option_id: Any) -> dict[str, Any]:
    key = str(option_id)
    if key in utilities:
        return utilities[key]
    if option_id in utilities:
        return utilities[option_id]
    raise ValueError(f"missing utility for option id {option_id!r}")


def converted_rows(results: dict[str, Any], actor: str, actor_label: str) -> list[dict[str, str]]:
    metadata = base_metadata_by_description()
    metrics = results.get("metrics") or {}
    holdout = results.get("holdout_metrics") or {}
    rows = []
    missing_descriptions = []

    for option in results["options"]:
        description = option["description"]
        meta = metadata.get(description)
        if meta is None:
            missing_descriptions.append(description)
            continue
        util = utility_for_option(results["utilities"], option["id"])
        rows.append(
            {
                "actor": actor,
                "actor_label": actor_label,
                "domain": meta["domain"],
                "domain_label": meta["domain_label"],
                "option_id": meta["option_id"],
                "description": description,
                "utility_mean": str(util["mean"]),
                "utility_variance": str(util["variance"]),
                "train_accuracy": metric(metrics, "accuracy"),
                "train_log_loss": metric(metrics, "log_loss"),
                "holdout_accuracy": metric(holdout, "accuracy"),
                "holdout_log_loss": metric(holdout, "log_loss"),
            }
        )

    if missing_descriptions:
        sample = "; ".join(missing_descriptions[:5])
        raise ValueError(
            f"{len(missing_descriptions)} result options are not in data/inputs/utility_options.csv. "
            f"First missing descriptions: {sample}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--actor-label", required=True)
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. Default: outputs/inputs/utility_options__<actor>.csv",
    )
    args = parser.parse_args()

    results = load_results(Path(args.results_json))
    rows = converted_rows(results, actor=args.actor, actor_label=args.actor_label)
    output = Path(args.output) if args.output else OUTPUT_INPUTS / f"utility_options__{args.actor}.csv"
    write_csv_rows(output, rows, fields=UTILITY_COLUMNS)
    print(f"wrote {len(rows)} utility rows to {output}")


if __name__ == "__main__":
    main()
