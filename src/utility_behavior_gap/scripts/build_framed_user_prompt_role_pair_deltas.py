#!/usr/bin/env python3
"""Build pair-level feature deltas for the framed user-prompt role contrast.

The finalized role contrast compares a world-class role cue against a skilled
role cue inside the same competition-framed user prompt scaffold. This script
combines all current `framed_user_prompt_role` run manifests, computes the
standard text features, and writes a pair-delta CSV compatible with
`analyze_length_controlled_feature_selection`.

No model/API calls are made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API
from utility_behavior_gap.scripts.analyze_run_generic_feature_deltas import load_pairs


DEFAULT_OUT = ANALYSIS / "framed_user_prompt_role_pair_deltas.csv"
DEFAULT_MANIFEST_GLOB = "framed_user_prompt_role_manifests__*.tsv"
CONTRAST = "framed_user_prompt_role"
HIGH_CONDITION = "role_strong"
LOW_CONDITION = "role_weak"
HIGH_RAW_CONDITION = "user_strong"
LOW_RAW_CONDITION = "user_normal"


def manifest_paths(glob_pattern: str) -> list[Path]:
    return sorted((OUTPUT_API / "runs").glob(glob_pattern))


def run_dirs_from_manifest(path: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        manifest = Path(parts[-1])
        run_dirs.append(manifest.parent)
    return run_dirs


def all_run_dirs(glob_pattern: str) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for manifest in manifest_paths(glob_pattern):
        for run_dir in run_dirs_from_manifest(manifest):
            resolved = run_dir.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(run_dir)
    return sorted(out, key=lambda path: path.name)


def role_pair_rows(run_dir: Path) -> pd.DataFrame:
    pairs, report = load_pairs(
        run_dir=run_dir,
        left_condition=HIGH_RAW_CONDITION,
        right_condition=LOW_RAW_CONDITION,
    )
    if pairs.empty:
        return pairs

    rows = pairs.copy()
    rows["run_dir"] = str(run_dir)
    rows["source_dataset"] = CONTRAST
    rows["source_family"] = CONTRAST
    rows["contrast"] = CONTRAST
    rows["comparison"] = CONTRAST
    rows["high_condition"] = HIGH_CONDITION
    rows["low_condition"] = LOW_CONDITION
    rows["high_raw_condition"] = HIGH_RAW_CONDITION
    rows["low_raw_condition"] = LOW_RAW_CONDITION
    rows["high_output_id"] = rows["left_output_id"]
    rows["low_output_id"] = rows["right_output_id"]
    rows["left_run_dir"] = str(run_dir)
    rows["right_run_dir"] = str(run_dir)
    rows["effect_score_high_minus_low"] = rows["left_panel_score"]
    rows["panel_winner_raw_condition"] = rows["panel_winner_condition"]
    rows["panel_winner_condition"] = rows["panel_winner_condition"].replace(
        {HIGH_RAW_CONDITION: HIGH_CONDITION, LOW_RAW_CONDITION: LOW_CONDITION}
    )

    for col in list(rows.columns):
        if col.startswith("left_"):
            target = "high_" + col[len("left_") :]
            if target not in rows.columns:
                rows[target] = rows[col]
        elif col.startswith("right_"):
            target = "low_" + col[len("right_") :]
            if target not in rows.columns:
                rows[target] = rows[col]

    rows["run_report_valid_pairs"] = report.get("valid_pairs")
    rows["run_report_invalid_or_missing_output_pairs"] = report.get("invalid_or_missing_output_pairs")
    return rows


def build(glob_pattern: str) -> pd.DataFrame:
    run_dirs = all_run_dirs(glob_pattern)
    if not run_dirs:
        raise FileNotFoundError(f"No role manifest files matched {glob_pattern!r}")
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        frames.append(role_pair_rows(run_dir))
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["actor", "task", "run_id", "pair_uid"]).reset_index(drop=True)
    return out


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary(df: pd.DataFrame, path: Path, *, glob_pattern: str) -> None:
    by_task = df.groupby("task", dropna=False).size().reset_index(name="pairs")
    by_actor_task = df.groupby(["actor", "task"], dropna=False).size().reset_index(name="pairs")
    lines = [
        "# Framed User-Prompt Role Pair-Deltas",
        "",
        "Comparison: `role_strong - role_weak`, where the raw conditions are `user_strong - user_normal`.",
        "",
        f"- manifest glob: `{glob_pattern}`",
        f"- rows: `{len(df)}`",
        f"- actors: `{df['actor'].nunique()}`",
        f"- tasks: `{df['task'].nunique()}`",
        "",
        "## By Task",
        "",
        markdown_table(by_task),
        "",
        "## By Actor And Task",
        "",
        markdown_table(by_actor_task),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-glob", default=DEFAULT_MANIFEST_GLOB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = build(args.manifest_glob)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out, index=False)
    summary_path = args.out.with_suffix(".md")
    write_summary(pairs, summary_path, glob_pattern=args.manifest_glob)
    print(f"pair deltas: {args.out}")
    print(f"summary: {summary_path}")
    print(f"rows: {len(pairs)}")
    print(pairs.groupby('task').size().to_string())


if __name__ == "__main__":
    main()
