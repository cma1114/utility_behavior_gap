"""Export every modgrid prompt — all conditions x tasks x domains — verbatim.

Builds a single reviewable file from the ACTUAL run manifests and judge-vote
request snapshots (not re-rendered templates), so what appears is byte-what-ran:

- per task: the R0 bare prompt, then each arm's full system + user prompt
  (item 0 / repeat 0 slot; high-low shown once per outcome domain since its
  consequence content varies by domain and actor; moral shown for two cause
  pairs)
- per task: the judge prompt verbatim with the two embedded outputs elided
- appendix: decoding/budget parameters and the essay pilot's goodjob arm

Output: outputs/analysis/modgrid_prompt_book.md

Usage:
  python -m utility_behavior_gap.scripts.export_modgrid_prompts [--actor deepseek-v3.2-or]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from utility_behavior_gap.paths import ROOT

OUT_PATH = ROOT / "outputs" / "analysis" / "modgrid_prompt_book.md"
TASKS = ["essay", "translation", "grant_proposal_abstract", "incident_postmortem"]
MAX_TOKENS = {"essay": 900, "translation": 600,
              "grant_proposal_abstract": 1000, "incident_postmortem": 3000}


def block(text: str) -> list[str]:
    return ["```", text if text else "(blank)", "```", ""]


def load_jobs(actor: str) -> dict[str, list[dict]]:
    by_task = defaultdict(list)
    for path in sorted((ROOT / "outputs" / "api" / "runs").glob(f"*__4-comparisons__{actor}__*/generation_jobs.jsonl")):
        for line in open(path):
            j = json.loads(line)
            by_task[j["task"]].append(j)
    return by_task


def job_for(jobs: list[dict], comparison_suffix: str, **match) -> dict | None:
    for j in jobs:
        if not j["comparison"].endswith(comparison_suffix):
            continue
        if all(str(j.get(k, "")) == str(v) for k, v in match.items()):
            return j
    return None


def r0_prompt(task: str, actor: str) -> str | None:
    for line in open(ROOT / "outputs" / "api" / "r0_generations.jsonl"):
        r = json.loads(line)
        if r["task"] == task and r["actor"] == actor:
            return r["user_prompt"]
    return None


def judge_prompt_elided(task: str, actor: str) -> str | None:
    run_dirs = sorted((ROOT / "outputs" / "api" / "runs").glob(f"{task}__4-comparisons__{actor}__*"))
    if not run_dirs:
        return None
    run_dir = run_dirs[0]
    gens = {}
    for line in open(run_dir / "generations.jsonl"):
        g = json.loads(line)
        gens[g["output_id"]] = g.get("output_text") or ""
    for line in open(run_dir / "judge_votes.jsonl"):
        v = json.loads(line)
        if not v.get("success"):
            continue
        prompt = v["request"]["messages"][0]["content"]
        for which, oid in [("DISPLAYED OUTPUT A", v["displayed_output_a_id"]),
                           ("DISPLAYED OUTPUT B", v["displayed_output_b_id"])]:
            text = gens.get(oid, "")
            if text and text in prompt:
                prompt = prompt.replace(text, f"[{which} — full text elided]")
        return prompt
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actor", default="deepseek-v3.2-or")
    args = ap.parse_args()
    actor = args.actor

    by_task = load_jobs(actor)
    out = ["# Modgrid prompt book — every condition, task, and domain (verbatim)", "",
           f"Source: the run manifests and judge request snapshots for `{actor}`.",
           "All prompts are exactly what was sent. System prompts are blank",
           "except where shown. High-low consequence content is actor-specific",
           "(this actor's fitted utilities); other arms are identical across actors.", ""]

    for task in TASKS:
        jobs = by_task.get(task, [])
        out += [f"# TASK: {task}  (generation max_tokens {MAX_TOKENS[task]}, temperature 1.0, reasoning off)", ""]

        out += ["## R0 — bare task (no system prompt, no wrapper)", ""]
        rp = r0_prompt(task, actor)
        out += block(rp or "(R0 store missing for this task)")

        j = job_for(jobs, "_framing") or job_for(jobs, "_headroom")
        if j:
            neutral = j["prompt_a"] if j["comparison"].endswith("_headroom") else j["prompt_a"]
            out += ["## framed_neutral — wrapper, $1,000 / children's hospital (system prompt blank)", ""]
            out += block(neutral)
        j = job_for(jobs, "_headroom")
        if j:
            out += ["## framed_strong — SAME user prompt as framed_neutral; intervention in system prompt:", ""]
            out += ["**System prompt:**"] + block(j["system_prompt_a"])
        j = job_for(jobs, "_amount")
        if j:
            out += ["## amount_high — $1,000,000 / children's hospital", ""] + block(j["prompt_a"])
            out += ["## amount_low — $100 / children's hospital", ""] + block(j["prompt_b"])
        moral_jobs = [x for x in jobs if x["comparison"].endswith("_moral")][:2]
        for i, j in enumerate(moral_jobs):
            out += [f"## moral_good — cause pair `{j['cause_pair_label']}` (example {i+1})", ""]
            out += block(j["prompt_a"])
            out += [f"## moral_bad — cause pair `{j['cause_pair_label']}` (example {i+1})", ""]
            out += block(j["prompt_b"])
        for domain in ["religions", "animals", "countries", "political"]:
            j = job_for(jobs, "_highlow", domain=domain)
            if not j:
                continue
            out += [f"## hl_high — domain {domain} (delta_u {j['delta_u']})", ""] + block(j["prompt_a"])
            out += [f"## hl_low — domain {domain} (same pair)", ""] + block(j["prompt_b"])

        out += ["## JUDGE PROMPT (verbatim from a real request; outputs elided)", ""]
        jp = judge_prompt_elided(task, actor)
        out += block(jp or "(no judge votes found)")
        out += ["---", ""]

    out += ["# Appendix", "",
            "- Judge panel: anthropic/claude-haiku-4.5, google/gemini-3.1-flash-lite-preview, openai/gpt-5-mini;",
            "  both presentation orders per pair per judge; judge temperature 0.0, max_tokens 120.",
            "- Essay pilot (pre-modgrid) goodjob arm: bare task + suffix `Do a good job`,",
            "  blank system prompt (comparison essay_user_prompt_goodjob_vs_bare).", ""]
    OUT_PATH.write_text("\n".join(out))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
