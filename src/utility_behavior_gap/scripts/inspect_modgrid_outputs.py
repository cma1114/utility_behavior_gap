"""Pre-judging inspection of modgrid generation outputs.

Reads every ``outputs/api/runs/*__4-comparisons__*/generations.jsonl`` and
produces, WITHOUT any API calls:

- ``outputs/analysis/modgrid_inspection_summary.md`` — per task x condition:
  counts, word stats, finish reasons, near-token-cap rate, and counts of
  outputs containing forbidden-vocabulary words. The vocabulary match is a
  READING AID ONLY: human review (2026-06-10) found its flags to be
  overwhelmingly topical false positives ("employer-sponsored", "evaluation
  plan"). It must NEVER be used to exclude trials or as evidence of blinding
  violations; exclusions come only from the LLM refusal classifier.
- ``outputs/analysis/modgrid_samples_<task>.md`` — per condition: N
  deterministic samples (first items, repeat 0) with full text, plus every
  leak-flagged output's matching line and the shortest outputs per condition,
  for human reading before judging is approved.

Usage:
  python -m utility_behavior_gap.scripts.inspect_modgrid_outputs [--samples 2]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from utility_behavior_gap.paths import ROOT

OUT_DIR = ROOT / "outputs" / "analysis"
RUNS_GLOB = "*__4-comparisons__*/generations.jsonl"

LEAK_RE = re.compile(
    r"sponsor|donat|prize|competition|judge|evaluat|\$1,000|\$100\b|\$1,000,000",
    re.IGNORECASE,
)
# Rough token estimate to flag outputs near their cap (words * 1.4).
CAPS = {"essay": 900, "translation": 600, "grant_proposal_abstract": 1000,
        "incident_postmortem": 3000}


def load_outputs():
    rows = []
    for path in sorted((ROOT / "outputs" / "api" / "runs").glob(RUNS_GLOB)):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(r)
    return rows


def leak_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if LEAK_RE.search(ln)][:3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=2, help="reading samples per task x condition")
    args = ap.parse_args()

    rows = load_outputs()
    by_cell = defaultdict(list)  # (task, condition) -> rows
    for r in rows:
        by_cell[(r["pair_uid"].split(":")[2] if r["pair_uid"].count(":") >= 2 else "?",
                 r["condition"])].append(r)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Modgrid output inspection", "",
             f"- outputs read: {len(rows)}",
             "- leak regex is OVER-BROAD by design ('judge', 'competition',",
             "  'evaluat' appear topically); flagged lines are a reading queue.",
             "", "| task | condition | n | mean words | p5 | near-cap | finish!=stop | leak-flagged |",
             "|---|---|---|---|---|---|---|---|"]
    flagged = defaultdict(list)
    for (task, cond), cell in sorted(by_cell.items()):
        words = sorted(len((r.get("output_text") or "").split()) for r in cell)
        near_cap = sum(1 for r in cell
                       if len((r.get("output_text") or "").split()) * 1.4
                       >= 0.95 * CAPS.get(task, 10**9))
        nonstop = sum(1 for r in cell if r.get("finish_reason") != "stop")
        leaks = [r for r in cell if LEAK_RE.search(r.get("output_text") or "")]
        flagged[(task, cond)] = leaks
        lines.append(
            f"| {task} | {cond} | {len(cell)} | {statistics.mean(words):.0f} | "
            f"{words[int(0.05 * len(words))]} | {near_cap} | {nonstop} | {len(leaks)} |")
    (OUT_DIR / "modgrid_inspection_summary.md").write_text("\n".join(lines) + "\n")

    tasks = sorted({t for t, _ in by_cell})
    for task in tasks:
        out = [f"# Modgrid reading samples — {task}", ""]
        for (t, cond), cell in sorted(by_cell.items()):
            if t != task:
                continue
            out += [f"## {cond}", ""]
            sample = sorted(cell, key=lambda r: r["pair_uid"])[: args.samples]
            for r in sample:
                out += [f"### sample — {r['pair_uid']} ({r['actor']})", "",
                        r.get("output_text") or "(empty)", "", "---", ""]
            shortest = min(cell, key=lambda r: len((r.get("output_text") or "").split()))
            out += [f"### shortest — {shortest['pair_uid']} ({shortest['actor']}, "
                    f"{len((shortest.get('output_text') or '').split())} words)", "",
                    shortest.get("output_text") or "(empty)", "", "---", ""]
            if flagged[(t, cond)]:
                out += [f"### leak-flagged lines ({len(flagged[(t, cond)])} outputs)", ""]
                for r in flagged[(t, cond)][:20]:
                    for ln in leak_lines(r.get("output_text") or ""):
                        out.append(f"- `{r['pair_uid']}` ({r['actor']}): {ln}")
                out += [""]
        (OUT_DIR / f"modgrid_samples_{task}.md").write_text("\n".join(out) + "\n")
    print(f"wrote modgrid_inspection_summary.md and per-task sample files to {OUT_DIR}")


if __name__ == "__main__":
    main()
