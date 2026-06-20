#!/usr/bin/env python3
"""Text-feature diagnostics for high/low versus R0 bridge outputs.

Default target is the political-domain low-utility versus R0 grant-abstract
bridge. Deltas are always utility side minus R0, so negative values mean the
utility-side output has less of the feature than the bare-task R0 output.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from utility_behavior_gap.io_utils import read_jsonl, write_csv_rows
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API
from utility_behavior_gap.scripts.analyze_highlow_cell_mechanisms import (
    ARTIFACT_DEFINITIONS,
    FEATURE_DEFINITIONS,
    artifact_flags,
    ci_mean,
    generation_completion_tokens,
    generation_max_tokens,
    load_textstat_module,
    native_generation_finish_reason,
    scalar_features,
    spacy_features,
    text_similarity,
    write_feature_definitions,
)
from utility_behavior_gap.scripts.analyze_highlow_r0_bridge_judging import load_pair_rows


LATEST_PATH = OUTPUT_API / "runs" / "highlow_r0_bridge_latest.txt"
SIDE_CONDITIONS = {"high": "hl_high", "low": "hl_low"}
R0_CONDITION = "r0"


def default_run_dir() -> Path:
    if not LATEST_PATH.exists():
        raise FileNotFoundError(f"{LATEST_PATH} does not exist; pass --run-dir explicitly")
    return Path(LATEST_PATH.read_text(encoding="utf-8").strip())


def slug(value: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "all"


def generation_lookup(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {str(row["output_id"]): row for row in read_jsonl(run_dir / "generations.jsonl")}


def outcome_group(row: dict[str, Any]) -> str:
    outcome = str(row.get("outcome_vs_r0", ""))
    if outcome == "r0":
        return "r0_wins"
    if outcome == "side":
        return "side_wins"
    if outcome == "tie":
        return "ties"
    return "unresolved"


def collect_rows(
    *,
    run_dir: Path,
    task: str,
    side: str,
    domain: str,
    spacy_model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_df = load_pair_rows(run_dir)
    if domain != "all":
        pair_df = pair_df[pair_df["domain"].eq(domain)]
    pair_df = pair_df[pair_df["task"].eq(task) & pair_df["side"].eq(side)]
    if pair_df.empty:
        raise ValueError(f"no pair rows for task={task}, side={side}, domain={domain}")

    generations = generation_lookup(run_dir)
    textstat_module = load_textstat_module()
    side_condition = SIDE_CONDITIONS[side]

    texts_by_key: dict[tuple[str, str], str] = {}
    base_rows: list[dict[str, Any]] = []
    for row in pair_df.to_dict(orient="records"):
        pair_uid = str(row["bridge_pair_uid"])
        side_generation = generations.get(f"{pair_uid}::a")
        r0_generation = generations.get(f"{pair_uid}::b")
        side_text = str(side_generation.get("output_text") if side_generation else "" or "")
        r0_text = str(r0_generation.get("output_text") if r0_generation else "" or "")
        texts_by_key[(pair_uid, side_condition)] = side_text
        texts_by_key[(pair_uid, R0_CONDITION)] = r0_text
        base_rows.append({**row, "pair_uid": pair_uid})

    spacy_by_key = spacy_features(texts_by_key, spacy_model)

    pair_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    for row in base_rows:
        pair_uid = row["pair_uid"]
        side_generation = generations.get(f"{pair_uid}::a")
        r0_generation = generations.get(f"{pair_uid}::b")
        side_text = texts_by_key[(pair_uid, side_condition)]
        r0_text = texts_by_key[(pair_uid, R0_CONDITION)]

        features_by_condition: dict[str, dict[str, float]] = {}
        artifacts_by_condition: dict[str, dict[str, bool]] = {}
        for condition, generation, text in (
            (side, side_generation, side_text),
            (R0_CONDITION, r0_generation, r0_text),
        ):
            features = scalar_features(text, textstat_module)
            features.update(spacy_by_key[(pair_uid, side_condition if condition == side else R0_CONDITION)])
            artifacts = artifact_flags(generation, text)
            features_by_condition[condition] = features
            artifacts_by_condition[condition] = artifacts
            response_rows.append(
                {
                    "pair_uid": pair_uid,
                    "actor": row.get("actor", ""),
                    "task": row.get("task", ""),
                    "domain": row.get("domain", ""),
                    "item_label": row.get("item_label", ""),
                    "condition": condition,
                    "outcome_group": outcome_group(row),
                    "output_id": "" if generation is None else generation.get("output_id", ""),
                    "finish_reason": "" if generation is None else generation.get("finish_reason", ""),
                    "native_finish_reason": native_generation_finish_reason(generation),
                    "completion_tokens": generation_completion_tokens(generation) or "",
                    "max_tokens": generation_max_tokens(generation) or "",
                    "token_cap_ratio": (
                        (generation_completion_tokens(generation) or 0)
                        / (generation_max_tokens(generation) or 1)
                    )
                    if generation is not None
                    else "",
                    "words": features["words"],
                    "characters": features["characters"],
                    **{name: int(value) for name, value in artifacts.items()},
                }
            )

        out: dict[str, Any] = {
            "pair_uid": pair_uid,
            "actor": row.get("actor", ""),
            "task": row.get("task", ""),
            "domain": row.get("domain", ""),
            "side": side,
            "side_condition": side_condition,
            "item_label": row.get("item_label", ""),
            "pair_idx": row.get("pair_idx", ""),
            "repeat": row.get("repeat", ""),
            "outcome_vs_r0": row.get("outcome_vs_r0", ""),
            "outcome_group": outcome_group(row),
            "side_net_score": row.get("side_net_score", ""),
            "high_utility": row.get("high_utility", ""),
            "low_utility": row.get("low_utility", ""),
            "delta_u": row.get("delta_u", ""),
            "high_description": row.get("high_description", ""),
            "low_description": row.get("low_description", ""),
            "text_jaccard_similarity": text_similarity(side_text, r0_text),
        }
        for name in features_by_condition[side]:
            side_value = features_by_condition[side][name]
            r0_value = features_by_condition[R0_CONDITION][name]
            out[f"{side}_{name}"] = side_value
            out[f"r0_{name}"] = r0_value
            out[f"delta_{name}_{side}_minus_r0"] = side_value - r0_value
        for name in artifacts_by_condition[side]:
            side_value = int(artifacts_by_condition[side][name])
            r0_value = int(artifacts_by_condition[R0_CONDITION][name])
            out[f"{side}_{name}"] = side_value
            out[f"r0_{name}"] = r0_value
            out[f"delta_{name}_{side}_minus_r0"] = side_value - r0_value
        pair_rows.append(out)

    return pair_rows, response_rows


def summarize_features(pair_rows: list[dict[str, Any]], side: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for group in ("all_pairs", "r0_wins", "side_wins", "ties"):
        subset = pair_rows if group == "all_pairs" else [row for row in pair_rows if row["outcome_group"] == group]
        if not subset:
            continue
        for definition in FEATURE_DEFINITIONS:
            feature = definition.name
            delta_name = f"delta_{feature}_{side}_minus_r0"
            side_name = f"{side}_{feature}"
            r0_name = f"r0_{feature}"
            deltas = [float(row[delta_name]) for row in subset]
            side_values = [float(row[side_name]) for row in subset]
            r0_values = [float(row[r0_name]) for row in subset]
            ci_lo, ci_hi = ci_mean(deltas, rng)
            rows.append(
                {
                    "group": group,
                    "feature": feature,
                    "definition": definition.definition,
                    "n_pairs": len(subset),
                    f"{side}_mean": mean(side_values),
                    "r0_mean": mean(r0_values),
                    f"mean_delta_{side}_minus_r0": mean(deltas),
                    "bootstrap_ci_lo": ci_lo,
                    "bootstrap_ci_hi": ci_hi,
                    f"median_delta_{side}_minus_r0": median(deltas),
                    f"{side}_greater_pairs": sum(delta > 0 for delta in deltas),
                    "r0_greater_pairs": sum(delta < 0 for delta in deltas),
                    "equal_pairs": sum(delta == 0 for delta in deltas),
                }
            )
    return rows


def summarize_artifacts(pair_rows: list[dict[str, Any]], side: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in ("all_pairs", "r0_wins", "side_wins", "ties"):
        subset = pair_rows if group == "all_pairs" else [row for row in pair_rows if row["outcome_group"] == group]
        if not subset:
            continue
        for definition in ARTIFACT_DEFINITIONS:
            name = definition.name
            side_count = sum(int(row[f"{side}_{name}"]) for row in subset)
            r0_count = sum(int(row[f"r0_{name}"]) for row in subset)
            side_only = [row for row in subset if int(row[f"{side}_{name}"]) and not int(row[f"r0_{name}"])]
            r0_only = [row for row in subset if int(row[f"r0_{name}"]) and not int(row[f"{side}_{name}"])]
            rows.append(
                {
                    "group": group,
                    "artifact": name,
                    "definition": definition.definition,
                    "n_pairs": len(subset),
                    f"{side}_count": side_count,
                    "r0_count": r0_count,
                    f"{side}_rate": side_count / len(subset),
                    "r0_rate": r0_count / len(subset),
                    f"{side}_only_pairs": len(side_only),
                    "r0_only_pairs": len(r0_only),
                    "discordant_pairs": len(side_only) + len(r0_only),
                }
            )
    return rows


def format_float(value: Any, digits: int = 3) -> str:
    try:
        if value == "" or value is None or math.isnan(float(value)):
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def feature_table(feature_rows: list[dict[str, Any]], side: str, group: str) -> list[str]:
    wanted = [definition.name for definition in FEATURE_DEFINITIONS]
    lookup = {(row["group"], row["feature"]): row for row in feature_rows}
    lines = [
        f"### {group}",
        "",
        f"| feature | {side} mean | R0 mean | {side}-R0 delta | 95% bootstrap CI | {side} greater | R0 greater |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for feature in wanted:
        row = lookup.get((group, feature))
        if row is None:
            continue
        lines.append(
            f"| `{feature}` | {format_float(row[f'{side}_mean'])} | {format_float(row['r0_mean'])} | "
            f"{format_float(row[f'mean_delta_{side}_minus_r0'])} | "
            f"[{format_float(row['bootstrap_ci_lo'])}, {format_float(row['bootstrap_ci_hi'])}] | "
            f"{row[f'{side}_greater_pairs']} | {row['r0_greater_pairs']} |"
        )
    return lines


def artifact_table(artifact_rows: list[dict[str, Any]], side: str, group: str) -> list[str]:
    wanted = [
        "explicit_length_truncation",
        "near_token_cap_95",
        "near_token_cap_98",
        "below_requested_words",
        "above_requested_words",
        "outside_requested_words",
        "forbidden_incentive_leak",
        "refusal_or_meta",
        "generic_funding_word",
        "heading_or_title",
        "bold_markup",
        "bullet_or_numbered_list",
    ]
    lookup = {(row["group"], row["artifact"]): row for row in artifact_rows}
    lines = [
        f"### {group}",
        "",
        f"| check | {side} count | R0 count | {side} only | R0 only |",
        "|---|---:|---:|---:|---:|",
    ]
    for artifact in wanted:
        row = lookup.get((group, artifact))
        if row is None:
            continue
        lines.append(
            f"| `{artifact}` | {row[f'{side}_count']} | {row['r0_count']} | "
            f"{row[f'{side}_only_pairs']} | {row['r0_only_pairs']} |"
        )
    return lines


def markdown_summary(
    *,
    run_dir: Path,
    side: str,
    task: str,
    domain: str,
    pair_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
    outputs: list[Path],
) -> str:
    counts = Counter(row["outcome_group"] for row in pair_rows)
    lines = [
        "# High/Low Versus R0 Text Features",
        "",
        f"- run: `{run_dir}`",
        f"- task: `{task}`",
        f"- domain: `{domain}`",
        f"- side: `{side}` versus `R0`",
        f"- pairs: `{len(pair_rows)}`",
        f"- panel outcomes: `{dict(counts)}`",
        "",
        f"Deltas are `{side} minus R0`. Negative values mean the `{side}` output has less of that feature than the bare-task R0 output.",
        "",
        "## Feature Deltas",
        "",
    ]
    for group in ("all_pairs", "r0_wins", "side_wins", "ties"):
        if group == "all_pairs" or counts.get(group, 0):
            lines.extend(feature_table(feature_rows, side, group))
            lines.append("")
    lines.extend(["## Artifact Checks", ""])
    for group in ("all_pairs", "r0_wins"):
        if group == "all_pairs" or counts.get(group, 0):
            lines.extend(artifact_table(artifact_rows, side, group))
            lines.append("")
    lines.extend(["## Output Files", ""])
    lines.extend(f"- `{path}`" for path in outputs)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--task", default="grant_proposal_abstract")
    parser.add_argument("--side", choices=["high", "low"], default="low")
    parser.add_argument("--domain", default="political")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or default_run_dir()
    prefix = args.out_prefix or (
        f"highlow_r0_text_features__{slug(args.side)}_vs_r0__"
        f"{slug(args.task)}__{slug(args.domain)}__{slug(run_dir.name)}"
    )

    pair_rows, response_rows = collect_rows(
        run_dir=run_dir,
        task=args.task,
        side=args.side,
        domain=args.domain,
        spacy_model=args.spacy_model,
    )
    feature_rows = summarize_features(pair_rows, args.side, args.seed)
    artifact_rows = summarize_artifacts(pair_rows, args.side)

    pair_path = ANALYSIS / f"{prefix}__pair_features.csv"
    response_path = ANALYSIS / f"{prefix}__response_artifacts.csv"
    feature_path = ANALYSIS / f"{prefix}__feature_summary.csv"
    artifact_path = ANALYSIS / f"{prefix}__artifact_summary.csv"
    definition_path = ANALYSIS / f"{prefix}__feature_definitions.csv"
    summary_path = ANALYSIS / f"{prefix}__summary.md"

    write_csv_rows(pair_path, pair_rows)
    write_csv_rows(response_path, response_rows)
    write_csv_rows(feature_path, feature_rows)
    write_csv_rows(artifact_path, artifact_rows)
    write_feature_definitions(definition_path)
    outputs = [pair_path, response_path, feature_path, artifact_path, definition_path, summary_path]
    summary_path.write_text(
        markdown_summary(
            run_dir=run_dir,
            side=args.side,
            task=args.task,
            domain=args.domain,
            pair_rows=pair_rows,
            feature_rows=feature_rows,
            artifact_rows=artifact_rows,
            outputs=outputs,
        ),
        encoding="utf-8",
    )

    print(f"run_dir: {run_dir}")
    print(f"pairs: {len(pair_rows)}")
    print(f"responses: {len(response_rows)}")
    print(f"summary: {summary_path}")
    print(f"pair_features: {pair_path}")
    print(f"feature_summary: {feature_path}")


if __name__ == "__main__":
    main()
