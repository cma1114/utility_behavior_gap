#!/usr/bin/env python3
"""Config-driven feature-table workflow.

This is the companion to ``pairwise.py``. It replaces repeated hand-written
command blocks for generic feature deltas and combined generic-plus-rubric
appendix tables with one config-driven entry point.

It does not call any model API. LLM rubric coding must already exist on disk.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "consolidated_workflow" / "configs" / "feature_tables.yaml"
PAIR_DELTAS = ROOT / "CURRENT_PAPER" / "features" / "final_text_analysis_pair_deltas.csv"
BY_OUTPUT = ROOT / "CURRENT_PAPER" / "features" / "final_text_analysis_by_output.csv"


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or not isinstance(data.get("tables"), dict):
        raise ValueError(f"config must contain a `tables` mapping: {path}")
    return data["tables"]


def table_config(configs: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in configs:
        raise ValueError(f"unknown table `{name}`; available: {', '.join(sorted(configs))}")
    cfg = dict(configs[name])
    cfg["id"] = name
    return cfg


def split_prefix(prefix: str | Path) -> tuple[Path, str]:
    path = repo_path(prefix)
    return path.parent, path.name


def path_with_suffix(prefix: str | Path, suffix: str) -> Path:
    path = repo_path(prefix)
    return path.with_name(path.name + suffix)


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("$ " + command_text(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def generic_delta_command(cfg: dict[str, Any], *, bootstrap_iterations: int | None) -> list[str]:
    out_dir, out_prefix = split_prefix(cfg["generic_out_prefix"])
    command = [sys.executable]

    if cfg["kind"] == "pair_delta":
        command.extend(
            [
                "-m",
                "utility_behavior_gap.scripts.analyze_standard_feature_deltas",
                "--contrast",
                str(cfg["contrast"]),
                "--out-dir",
                str(out_dir),
                "--out-prefix",
                out_prefix,
            ]
        )
    elif cfg["kind"] == "arm_match":
        command.extend(
            [
                "-m",
                "utility_behavior_gap.scripts.analyze_arm_match_feature_deltas",
                "--left-condition",
                str(cfg["left_condition"]),
                "--right-condition",
                str(cfg["right_condition"]),
                "--left-key",
                str(cfg["left_key"]),
                "--right-key",
                str(cfg["right_key"]),
                "--out-dir",
                str(out_dir),
                "--out-prefix",
                out_prefix,
            ]
        )
        if cfg.get("bridge_outcomes"):
            command.extend(["--bridge-outcomes", str(repo_path(cfg["bridge_outcomes"]))])
        if cfg.get("bridge_left_output_col"):
            command.extend(["--bridge-left-output-col", str(cfg["bridge_left_output_col"])])
        if cfg.get("bridge_right_output_col"):
            command.extend(["--bridge-right-output-col", str(cfg["bridge_right_output_col"])])
    else:
        raise ValueError(f"{cfg['id']}: unknown feature table kind `{cfg['kind']}`")

    if bootstrap_iterations is not None:
        command.extend(["--bootstrap-iterations", str(bootstrap_iterations)])
    return command


def appendix_command(cfg: dict[str, Any]) -> list[str]:
    generic_prefix = repo_path(cfg["generic_out_prefix"])
    generic_by_task = generic_prefix.with_name(generic_prefix.name + "_by_task.csv")
    appendix_prefix = repo_path(cfg["appendix_out_prefix"])
    command = [
        sys.executable,
        "-m",
        "utility_behavior_gap.scripts.make_feature_appendix_table",
        "--generic-by-task",
        str(generic_by_task),
        "--out-prefix",
        str(appendix_prefix),
        "--title",
        str(cfg["title"]),
        "--comparison",
        str(cfg.get("comparison") or cfg.get("contrast") or cfg["id"]),
        "--left-key",
        str(cfg["left_key"]),
        "--right-key",
        str(cfg["right_key"]),
        "--left-label",
        str(cfg["left_label"]),
        "--right-label",
        str(cfg["right_label"]),
    ]

    rubric_run_dir = cfg.get("rubric_run_dir")
    if rubric_run_dir:
        command.extend(["--rubric-run-dir", str(repo_path(rubric_run_dir))])

    if cfg["kind"] == "pair_delta":
        command.extend(
            [
                "--generic-pairs",
                str(PAIR_DELTAS),
                "--generic-pairs-comparison",
                str(cfg["contrast"]),
            ]
        )
    elif cfg["kind"] == "arm_match":
        generic_pairs = generic_prefix.with_name(generic_prefix.name + "_pairs.csv")
        command.extend(["--generic-pairs", str(generic_pairs)])
        if cfg.get("bridge_outcomes"):
            command.extend(["--bridge-outcomes", str(repo_path(cfg["bridge_outcomes"]))])
        if cfg.get("bridge_left_output_col"):
            command.extend(["--bridge-left-output-col", str(cfg["bridge_left_output_col"])])
        if cfg.get("bridge_right_output_col"):
            command.extend(["--bridge-right-output-col", str(cfg["bridge_right_output_col"])])
    return command


def table_outputs(cfg: dict[str, Any]) -> dict[str, Path]:
    generic_prefix = repo_path(cfg["generic_out_prefix"])
    appendix_prefix = repo_path(cfg["appendix_out_prefix"])
    return {
        "generic_by_task": generic_prefix.with_name(generic_prefix.name + "_by_task.csv"),
        "generic_pairs": generic_prefix.with_name(generic_prefix.name + "_pairs.csv"),
        "generic_summary": generic_prefix.with_name(generic_prefix.name + "_summary.md"),
        "appendix_summary": appendix_prefix.with_name(appendix_prefix.name + "_summary.md"),
        "appendix_clear_md": appendix_prefix.with_name(
            appendix_prefix.name + "_clear_differences.md"
        ),
        "appendix_clear_csv": appendix_prefix.with_name(
            appendix_prefix.name + "_clear_differences.csv"
        ),
        "appendix_all_csv": appendix_prefix.with_name(appendix_prefix.name + "_all.csv"),
    }


def audit_row(name: str, cfg: dict[str, Any]) -> dict[str, str | int]:
    inputs: list[Path] = [PAIR_DELTAS if cfg["kind"] == "pair_delta" else BY_OUTPUT]
    if cfg.get("bridge_outcomes"):
        inputs.append(repo_path(cfg["bridge_outcomes"]))
    if cfg.get("rubric_run_dir"):
        inputs.append(repo_path(cfg["rubric_run_dir"]))

    missing_inputs = [str(path) for path in inputs if not path.exists()]
    outputs = table_outputs(cfg)
    existing_outputs = [label for label, path in outputs.items() if path.exists()]
    return {
        "table": name,
        "kind": str(cfg["kind"]),
        "status": "PASS" if not missing_inputs else "MISSING_INPUTS",
        "missing_inputs": "; ".join(missing_inputs),
        "existing_outputs": ", ".join(existing_outputs),
        "generic_out_prefix": str(repo_path(cfg["generic_out_prefix"])),
        "appendix_out_prefix": str(repo_path(cfg["appendix_out_prefix"])),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        print("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")


def selected_tables(configs: dict[str, Any], names: list[str]) -> list[tuple[str, dict[str, Any]]]:
    if names == ["all"]:
        return [(name, dict(cfg, id=name)) for name, cfg in sorted(configs.items())]
    return [(name, table_config(configs, name)) for name in names]


def cmd_list(args: argparse.Namespace) -> None:
    configs = load_config(args.config)
    rows = [
        {
            "table": name,
            "kind": cfg.get("kind", ""),
            "title": cfg.get("title", name),
            "rubric": "yes" if cfg.get("rubric_run_dir") else "no",
        }
        for name, cfg in sorted(configs.items())
    ]
    print_table(rows)


def cmd_audit(args: argparse.Namespace) -> None:
    configs = load_config(args.config)
    rows = [audit_row(name, dict(cfg, id=name)) for name, cfg in sorted(configs.items())]
    print_table(rows)


def cmd_run(args: argparse.Namespace) -> None:
    configs = load_config(args.config)
    for name, cfg in selected_tables(configs, args.table):
        print(f"=== {name} ===", flush=True)
        if args.stage in ("deltas", "all"):
            run_command(
                generic_delta_command(cfg, bootstrap_iterations=args.bootstrap_iterations),
                dry_run=args.dry_run,
            )
        if args.stage in ("appendix", "all"):
            generic_by_task = table_outputs(cfg)["generic_by_task"]
            if args.stage == "appendix" and not generic_by_task.exists():
                raise FileNotFoundError(
                    f"{name}: generic by-task file is missing; run stage `deltas` first: "
                    f"{generic_by_task}"
                )
            run_command(appendix_command(cfg), dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list")
    list_parser.set_defaults(func=cmd_list)

    audit_parser = sub.add_parser("audit")
    audit_parser.set_defaults(func=cmd_audit)

    run_parser = sub.add_parser("run")
    run_parser.add_argument(
        "--table",
        nargs="+",
        required=True,
        help="One or more table ids, or `all`.",
    )
    run_parser.add_argument(
        "--stage",
        choices=["deltas", "appendix", "all"],
        default="all",
        help="Which local stage to run.",
    )
    run_parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        help="Override bootstrap iterations for the generic feature-delta stage.",
    )
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(func=cmd_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
