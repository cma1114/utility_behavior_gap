#!/usr/bin/env python3
"""Plan or run the MiMo V2.5 utility refit through the local CAIS pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from utility_behavior_gap.io_utils import load_env_file
from utility_behavior_gap.paths import OUTPUT_INPUTS, ROOT


EMERGENT_VALUES_ROOT = Path("/Users/christopherackerman/repos/emergent-values")
UTILITY_ANALYSIS = EMERGENT_VALUES_ROOT / "utility_analysis"
OPTIMIZE_SCRIPT_DIR = UTILITY_ANALYSIS / "experiments" / "compute_utilities"
OPTIMIZE_SCRIPT = OPTIMIZE_SCRIPT_DIR / "optimize_utility_model.py"
EV_CREATE_AGENT_CONFIG = UTILITY_ANALYSIS / "compute_utilities" / "create_agent.yaml"
FULL_OPTIONS_JSON = OUTPUT_INPUTS / "emergent_values_options.json"
CONFIG_KEY = "mimo_v25_hard_sampling_checkpointed"


ECHO_PREFIXES = (
    "Computing utilities",
    "Total possible edges:",
    "Training pool:",
    "Target total edges:",
    "Initial edges:",
    "Number of iterations:",
    "Iteration ",
    "Number of additional pairs",
    "Initial model",
    "Updated model",
    "Final model",
    "Generating pseudolabels",
    "Number of pseudolabels",
    "Evaluating utility model",
    "Wrote checkpoint:",
    "Number of timeouts:",
    "Number of generic errors:",
    "Number of null-content responses:",
    "Number of longer_than_expected responses:",
    "Number of unparseable responses:",
    "Total time taken:",
)
ECHO_TOKENS = (
    "Traceback",
    "RuntimeError",
    "[Error]",
    "[Timeout]",
    "HTTP Error",
    "Too many unparseable",
)


MODE_SETTINGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "option_count": 12,
        "num_epochs": 20,
        "edge_multiplier": 1,
        "degree": 2,
        "num_edges_per_iteration": 10,
        "K": 1,
        "use_pseudolabels": False,
        "holdout_fraction": 0.0,
        "default_concurrency_limit": 5,
        "default_base_timeout": 30,
        "default_max_tokens": 64,
    },
    "full": {
        "option_count": None,
        "num_epochs": 1000,
        "edge_multiplier": 2,
        "degree": 2,
        "num_edges_per_iteration": 500,
        "K": 10,
        "use_pseudolabels": True,
        "holdout_fraction": 0.05,
        "default_concurrency_limit": 10,
        "default_base_timeout": 30,
        "default_max_tokens": 64,
    },
}


def load_full_options() -> list[str]:
    if not FULL_OPTIONS_JSON.exists():
        raise FileNotFoundError(
            f"{FULL_OPTIONS_JSON} does not exist. Run "
            "`python -m utility_behavior_gap.scripts.export_emergent_values_options` first."
        )
    with FULL_OPTIONS_JSON.open(encoding="utf-8") as f:
        return json.load(f)


def write_options(mode_dir: Path, mode: str, option_count: int | None) -> Path:
    options = load_full_options()
    if option_count is not None:
        options = options[:option_count]
    path = mode_dir / f"mimo_v25_{mode}_options.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(options, f, indent=2)
        f.write("\n")
    return path


def write_config(mode_dir: Path, mode: str, settings: dict[str, Any]) -> Path:
    config = {
        CONFIG_KEY: {
            "utility_model_class": "ThurstonianActiveLearningUtilityModel",
            "utility_model_arguments": {
                "unparseable_mode": "distribution",
                "system_message": "You are a helpful assistant.",
                "num_epochs": settings["num_epochs"],
                "learning_rate": 0.01,
                "edge_multiplier": settings["edge_multiplier"],
                "degree": settings["degree"],
                "num_edges_per_iteration": settings["num_edges_per_iteration"],
                "P": 10.0,
                "Q": 20.0,
                "K": settings["K"],
                "use_logprobs": False,
                "use_pseudolabels": settings["use_pseudolabels"],
                "pseudolabel_confidence_threshold": 0.95,
                "checkpoint_dir": str(mode_dir / "checkpoints"),
                "checkpoint_prefix": f"mimo_v25_{mode}",
                "max_unparseable_fraction": 0.1,
            },
            "preference_graph_arguments": {
                "holdout_fraction": settings["holdout_fraction"],
                "holdout_seed": 42,
            },
        }
    }
    path = mode_dir / "compute_utilities_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


def write_agent_config(mode_dir: Path, *, concurrency_limit: int, base_timeout: int, max_tokens: int) -> Path:
    config = {
        "default": {
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "concurrency_limit": concurrency_limit,
            "base_timeout": base_timeout,
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 10.0,
            "completion_kwargs": {
                "reasoning_effort": "none",
            },
        }
    }
    path = mode_dir / "create_agent_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


def call_estimate(n_options: int, settings: dict[str, Any]) -> dict[str, int]:
    target_edges = int(settings["edge_multiplier"] * n_options * math.log2(n_options))
    initial_edges = (n_options * settings["degree"]) // 2
    remainder = max(target_edges - initial_edges, 0)
    iterations = math.ceil(remainder / settings["num_edges_per_iteration"]) if remainder else 0
    max_training_edges = n_options * (n_options - 1) // 2
    sampled_fit_edges = min(initial_edges + iterations * settings["num_edges_per_iteration"], max_training_edges)
    fit_prompts = sampled_fit_edges * 2
    holdout_edges = min(int((n_options * (n_options - 1) / 2) * settings["holdout_fraction"]), 1000)
    holdout_prompts = holdout_edges * 2
    fit_calls = fit_prompts * settings["K"]
    holdout_calls = holdout_prompts * settings["K"]
    return {
        "options": n_options,
        "target_edges": target_edges,
        "initial_edges": initial_edges,
        "iterations": iterations,
        "sampled_fit_edges": sampled_fit_edges,
        "fit_prompts": fit_prompts,
        "fit_calls": fit_calls,
        "holdout_prompts": holdout_prompts,
        "holdout_calls": holdout_calls,
        "k": settings["K"],
        "estimated_calls": fit_calls + holdout_calls,
    }


def build_command(mode_dir: Path, options_path: Path, config_path: Path, agent_config_path: Path, mode: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(OPTIMIZE_SCRIPT),
        "--model_key",
        "mimo-v25-pro-openrouter",
        "--save_dir",
        str(mode_dir),
        "--save_suffix",
        f"mimo_v25_{mode}",
        "--options_path",
        str(options_path),
        "--compute_utilities_config_path",
        str(config_path),
        "--compute_utilities_config_key",
        CONFIG_KEY,
        "--create_agent_config_path",
        str(agent_config_path),
        "--create_agent_config_key",
        "default",
    ]


def should_echo_output_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(stripped.startswith(prefix) for prefix in ECHO_PREFIXES) or any(token in stripped for token in ECHO_TOKENS)


def run_filtered(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing full CAIS output to {log_path}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    current_line: list[str] = []
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n\n=== run start ===\n")
        log.write("$ " + " ".join(command) + "\n")
        while True:
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            log.write(text)
            log.flush()
            for char in text:
                if char in "\r\n":
                    line = "".join(current_line)
                    current_line = []
                    if should_echo_output_line(line):
                        print(line.strip(), flush=True)
                else:
                    current_line.append(char)
        if current_line:
            line = "".join(current_line)
            if should_echo_output_line(line):
                print(line.strip(), flush=True)
        return_code = process.wait()
        log.write(f"\n=== run exit {return_code} ===\n")
    if return_code:
        print(f"utility refit failed with exit code {return_code}; full log: {log_path}", file=sys.stderr)
        raise subprocess.CalledProcessError(return_code, command)


def print_status(mode_dir: Path, mode: str) -> None:
    checkpoint_path = mode_dir / "checkpoints" / f"mimo_v25_{mode}_latest.json"
    if not checkpoint_path.exists():
        print(f"no checkpoint found at {checkpoint_path}")
        return
    with checkpoint_path.open(encoding="utf-8") as f:
        checkpoint = json.load(f)
    graph_data = checkpoint.get("graph_data", {})
    metrics = checkpoint.get("metrics", {})
    print(f"mode: {mode}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"stage: {checkpoint.get('stage', '')}")
    print(f"next_iteration: {checkpoint.get('next_iteration', '')}/{checkpoint.get('num_iterations', '')}")
    print(f"graph_edges: {len(graph_data.get('edges', {}))}")
    if metrics:
        print(f"log_loss: {metrics.get('log_loss', '')}")
        print(f"accuracy: {metrics.get('accuracy', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODE_SETTINGS), default="smoke")
    parser.add_argument("--run", action="store_true", help="Actually run the paid OpenRouter utility fit.")
    parser.add_argument(
        "--concurrency-limit",
        type=int,
        default=None,
        help="OpenRouter request concurrency for the CAIS agent config generated for this run.",
    )
    parser.add_argument(
        "--base-timeout",
        type=int,
        default=None,
        help="Per-call timeout seconds for the CAIS agent config generated for this run.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max completion tokens for each forced-choice utility call.",
    )
    parser.add_argument("--status", action="store_true", help="Print the latest checkpoint status without API calls.")
    parser.add_argument("--verbose", action="store_true", help="Stream full CAIS output instead of filtering it to a log.")
    parser.add_argument("--log-file", default=None, help="Path for the full CAIS output log when running non-verbosely.")
    args = parser.parse_args()

    if not OPTIMIZE_SCRIPT.exists():
        raise FileNotFoundError(f"CAIS utility script not found at {OPTIMIZE_SCRIPT}")

    settings = MODE_SETTINGS[args.mode]
    concurrency_limit = args.concurrency_limit or settings["default_concurrency_limit"]
    base_timeout = args.base_timeout or settings["default_base_timeout"]
    max_tokens = args.max_tokens or settings["default_max_tokens"]
    mode_dir = ROOT / "outputs" / "utility_refits" / f"mimo_v25_{args.mode}"
    if args.status:
        print_status(mode_dir, args.mode)
        return

    options_path = write_options(mode_dir, args.mode, settings["option_count"])
    config_path = write_config(mode_dir, args.mode, settings)
    agent_config_path = write_agent_config(
        mode_dir,
        concurrency_limit=concurrency_limit,
        base_timeout=base_timeout,
        max_tokens=max_tokens,
    )
    n_options = len(json.load(options_path.open(encoding="utf-8")))
    estimate = call_estimate(n_options, settings)
    command = build_command(mode_dir, options_path, config_path, agent_config_path, args.mode)

    print(f"mode: {args.mode}")
    for key, value in estimate.items():
        print(f"{key}: {value}")
    print(f"options_path: {options_path}")
    print(f"config_path: {config_path}")
    print(f"agent_config_path: {agent_config_path}")
    print(f"concurrency_limit: {concurrency_limit}")
    print(f"base_timeout: {base_timeout}")
    print(f"max_tokens: {max_tokens}")
    print(f"save_dir: {mode_dir}")
    log_path = Path(args.log_file) if args.log_file else mode_dir / f"run_{args.mode}.log"
    print(f"log_path: {log_path}")
    print("$ " + " ".join(command))

    if not args.run:
        print("plan only; add --run to spend API calls")
        return

    load_env_file(ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise RuntimeError("OPENROUTER_API_KEY is required in .env or the shell environment")
    env = os.environ.copy()
    try:
        if args.verbose:
            subprocess.run(command, cwd=OPTIMIZE_SCRIPT_DIR, env=env, check=True)
        else:
            run_filtered(command, cwd=OPTIMIZE_SCRIPT_DIR, env=env, log_path=log_path)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
