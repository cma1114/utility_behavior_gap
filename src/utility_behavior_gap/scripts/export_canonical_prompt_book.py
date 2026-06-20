#!/usr/bin/env python3
"""Export the current paper-facing prompt book from actual run manifests.

This script is intentionally separate from ``export_modgrid_prompts.py``.
The older exporter targets the obsolete ``4-comparisons`` run family. This
one records the current paper line:

- direct instruction: framed neutral versus user-prompt max effort
- high-low utility: high-utility versus low-utility intervention
- moral: good-cause versus bad-cause intervention
- amount: larger versus smaller funding amount
- R0: bare task reference prompt

The prompt text is copied from current ``generation_jobs.jsonl`` manifests,
not reconstructed from templates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utility_behavior_gap.constants import PLOT_TASK_ORDER, TASK_BY_LABEL
from utility_behavior_gap.paths import ROOT


OUT_PATH = ROOT / "outputs" / "analysis" / "canonical_prompt_book.md"
TASK_KEY = {
    "essay": "essay",
    "translation": "translation",
    "incident_postmortem": "incident",
    "grant_proposal_abstract": "grant",
}
DOMAINS = ["religions", "animals", "countries", "political"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def code_block(text: str) -> list[str]:
    return ["```", text if text else "(blank)", "```", ""]


def first_path(pattern: str) -> Path | None:
    paths = sorted((ROOT / "outputs" / "api" / "runs").glob(pattern))
    return paths[0] if paths else None


def first_job(path: Path | None, *, comparison: str | None = None, **match: str) -> dict[str, Any] | None:
    if path is None:
        return None
    for row in read_jsonl(path / "generation_jobs.jsonl"):
        if comparison is not None and row.get("comparison") != comparison:
            continue
        if all(str(row.get(key, "")) == str(value) for key, value in match.items()):
            return row
    return None


def r0_prompt(actor: str, task: str) -> str:
    path = ROOT / "outputs" / "api" / "r0_generations.jsonl"
    for row in read_jsonl(path):
        if row.get("actor") == actor and row.get("task") == task:
            return str(row.get("user_prompt", ""))
    return "(R0 prompt not found for this actor/task.)"


def latest_matching_dir(pattern: str) -> Path | None:
    paths = sorted((ROOT / "outputs" / "api" / "runs").glob(pattern))
    return paths[-1] if paths else None


def direct_run_dir(actor: str, task: str) -> Path | None:
    return first_path(f"{task}__framed_user_strong_headroom__{actor}__*/")


def modgrid_run_dir(actor: str, task: str) -> Path | None:
    key = TASK_KEY[task]
    combined = latest_matching_dir(
        f"{task}__modgrid_{key}_amount+modgrid_{key}_highlow+modgrid_{key}_moral__{actor}__*/"
    )
    if combined is not None:
        return combined
    return latest_matching_dir(f"{task}__modgrid_{key}_*__{actor}__*/")


def elided_judge_prompt(run_dir: Path | None) -> str:
    if run_dir is None:
        return "(no run directory found)"
    generations = {
        row["output_id"]: str(row.get("output_text") or "")
        for row in read_jsonl(run_dir / "generations.jsonl")
        if row.get("output_id")
    }
    for vote in read_jsonl(run_dir / "judge_votes.jsonl"):
        if not vote.get("success"):
            continue
        prompt = str(vote.get("request", {}).get("messages", [{}])[0].get("content", ""))
        for label, oid in (
            ("DISPLAYED OUTPUT A", vote.get("displayed_output_a_id")),
            ("DISPLAYED OUTPUT B", vote.get("displayed_output_b_id")),
        ):
            text = generations.get(str(oid), "")
            if text and text in prompt:
                prompt = prompt.replace(text, f"[{label} -- full text elided]")
        return prompt
    return "(no successful judge prompt found)"


def prompt_pair_section(title: str, job: dict[str, Any] | None) -> list[str]:
    if job is None:
        return [f"## {title}", "", "(no matching job found)", ""]
    manifest = str(job.get("run_manifest_path") or "")
    if not manifest:
        run_dir = str(job.get("run_dir") or "")
        manifest = str(Path(run_dir) / "generation_jobs.jsonl") if run_dir else ""
    lines = [f"## {title}", ""]
    lines += ["Source manifest:", "", f"`{manifest}`", ""]
    lines += ["### Side A system prompt", ""] + code_block(str(job.get("system_prompt_a", "")))
    lines += ["### Side A user prompt", ""] + code_block(str(job.get("prompt_a", "")))
    lines += ["### Side B system prompt", ""] + code_block(str(job.get("system_prompt_b", "")))
    lines += ["### Side B user prompt", ""] + code_block(str(job.get("prompt_b", "")))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", default="gpt-5.4-mini-or")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    actor = args.actor
    lines: list[str] = [
        "# Canonical Prompt Book",
        "",
        f"Source actor for example manifests: `{actor}`.",
        "",
        "This is the current paper-facing prompt book. It is generated from actual current run manifests, not from the older `outputs/analysis/modgrid_prompt_book.md` file.",
        "",
        "Key design points:",
        "",
        "- All current paper-facing generation system prompts shown here are blank.",
        "- Direct instruction uses the max-effort exhortation appended at the end of the user prompt.",
        "- High-low utility and moral arms use the no-amount funding wording: `the sponsor will fund the following intervention:`.",
        "- Amount arms use the amount wording because amount is the manipulated variable.",
        "- Current neutral target: `A healthcare intervention at a children's hospital.`",
        "",
    ]

    for task_label in PLOT_TASK_ORDER:
        task = TASK_BY_LABEL[task_label]
        key = TASK_KEY[task]
        direct_dir = direct_run_dir(actor, task)
        modgrid_dir = modgrid_run_dir(actor, task)
        direct_job = first_job(direct_dir, comparison="framed_user_strong_headroom")
        amount_job = first_job(modgrid_dir, comparison=f"modgrid_{key}_amount")
        moral_job = first_job(modgrid_dir, comparison=f"modgrid_{key}_moral")

        lines += [f"# Task: {task_label}", ""]
        lines += ["## R0 bare task", ""] + code_block(r0_prompt(actor, task))
        lines += prompt_pair_section(
            "Direct instruction: max-effort user prompt versus framed neutral",
            direct_job,
        )
        lines += prompt_pair_section("Amount: larger amount versus smaller amount", amount_job)
        lines += prompt_pair_section("Moral: good cause versus bad cause", moral_job)

        for domain in DOMAINS:
            job = first_job(modgrid_dir, comparison=f"modgrid_{key}_highlow", domain=domain)
            lines += prompt_pair_section(f"High-low utility: domain `{domain}`", job)

        lines += ["## Judge prompt example (outputs elided)", ""]
        lines += code_block(elided_judge_prompt(direct_dir or modgrid_dir))
        lines += ["---", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
