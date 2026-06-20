"""Export moral-condition pairs where ALL THREE judges chose the good-cause
output, for human review of what the judged quality difference looks like.

Selection criteria (per pair):
- task in {grant, essay}
- strict-clean: neither arm labeled full_refusal / partial_refusal / degenerate
  by the LLM classifier (so no refusal content explains the verdict)
- unanimous: all three judge votes for the good-cause arm (no tie votes)

Output: ``outputs/analysis/moral_unanimous_good_pairs.md`` with, per pair, the
cause texts, word counts, the three judge reasons (grant only; the essay run's
judges returned bare answers), and both full outputs. Pairs are sampled
deterministically: sorted by (actor, pair_key), round-robin across actors up
to --per-task.

Usage:
  python -m utility_behavior_gap.scripts.export_moral_unanimous_pairs [--per-task 10]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from utility_behavior_gap.paths import ROOT
from utility_behavior_gap.scripts.analyze_moral_sandbagging import (
    EXCLUDE_LABELS,
    NON_ESSAY_DATASETS,
    build_pairs,
    is_clean,
)

OUT_PATH = ROOT / "outputs" / "analysis" / "moral_unanimous_good_pairs.md"


def load_grant_votes():
    """pair_key -> list of (judge_label, winner_condition, reason)."""
    votes = defaultdict(list)
    for dn in NON_ESSAY_DATASETS["grant"]:
        for line in open(ROOT / "trial_level_data" / dn / "h2h_votes.jsonl"):
            v = json.loads(line)
            if not v.get("success"):
                continue
            m = re.search(r"reason:\s*(.+)", v.get("raw_text") or "",
                          re.IGNORECASE | re.DOTALL)
            votes[f"{dn}|{v['pair_idx_global']}"].append(
                (v.get("judge_label"), v.get("winner_condition"),
                 (m.group(1).strip() if m else ""), v.get("a_condition")))
    return votes


def load_essay_trials():
    """pair_key -> trial dict (with judge_votes, essays, topic)."""
    trials = {}
    for f in sorted((ROOT / "essay_all_conditions" / "moral").glob("*.json")):
        for idx, t in enumerate(json.load(open(f))["trials"]):
            trials[f"essay|{f.stem}|{idx}"] = t
    return trials


def unanimous_good(pair, grant_votes, essay_trials):
    if pair["task"] == "grant":
        v = grant_votes.get(pair["pair_key"], [])
        return len(v) == 3 and all(w == "moral_good" for _, w, _, _ in v)
    if pair["task"] == "essay":
        t = essay_trials[pair["pair_key"]]
        letters = set(t["judge_votes"].values())
        return t["winner_arm"] == "A" and len(letters) == 1 and "TIE" not in letters
    return False


def round_robin(pairs, per_task):
    by_actor = defaultdict(list)
    for p in sorted(pairs, key=lambda p: (p["actor"], p["pair_key"])):
        by_actor[p["actor"]].append(p)
    out, i = [], 0
    while len(out) < per_task and any(by_actor.values()):
        for actor in sorted(by_actor):
            if by_actor[actor] and len(out) < per_task:
                out.append(by_actor[actor].pop(0))
        i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-task", type=int, default=10)
    args = ap.parse_args()

    pairs, texts = build_pairs()
    grant_votes = load_grant_votes()
    essay_trials = load_essay_trials()
    cause_meta = {r["cause_pair_label"]: r for r in
                  csv.DictReader(open(ROOT / "data" / "inputs" / "moral_cause_pairs.csv"))}

    lines = ["# Moral condition: unanimous good-cause wins (clean pairs)", "",
             "Criteria: strict-clean (no refusal content in either arm per the LLM",
             "classifier) AND all three judges voted for the good-cause output.",
             "Regenerate: `python -m utility_behavior_gap.scripts.export_moral_unanimous_pairs`",
             ""]
    for task in ("grant", "essay"):
        eligible = [p for p in pairs if p["task"] == task and is_clean(p)
                    and p["winner"] == "good"
                    and unanimous_good(p, grant_votes, essay_trials)]
        chosen = round_robin(eligible, args.per_task)
        lines.append(f"## {task.upper()} — {len(eligible)} unanimous clean good-wins"
                     f" (showing {len(chosen)})")
        lines.append("")
        for n, p in enumerate(chosen, 1):
            cm = cause_meta[p["cause_pair_label"]]
            lines += [f"### {task} #{n} — {p['actor']} ({p['pair_key']})", "",
                      f"- **good cause:** {cm['good_text']}",
                      f"- **bad cause:** {cm['bad_text']}",
                      f"- **item:** {p['item_label']}",
                      f"- **words:** good {p['good_words']} / bad {p['bad_words']}", ""]
            if p["task"] == "grant":
                a_cond = grant_votes[p["pair_key"]][0][3]
                good_as = "A" if a_cond == "moral_good" else "B"
                lines.append(f"- **displayed:** good-cause output shown as "
                             f"Output {good_as} (judge reasons use A/B labels)")
                for judge, _, reason, _ in grant_votes[p["pair_key"]]:
                    lines.append(f"- judge `{judge}`: {reason}")
                lines.append("")
            else:
                t = essay_trials[p["pair_key"]]
                lines.append(f"- judge votes: {t['judge_votes']} (essay judges gave no reasons)")
                lines.append("")
            lines += ["**GOOD-CAUSE OUTPUT:**", "",
                      texts[(p["pair_key"], "good")], "",
                      "**BAD-CAUSE OUTPUT:**", "",
                      texts[(p["pair_key"], "bad")], "", "---", ""]
    OUT_PATH.write_text("\n".join(lines))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
