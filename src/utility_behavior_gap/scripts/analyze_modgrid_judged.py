"""Judged results for the modgrid slice: win rates per contrast x task.

Reads the per-run manifests, generations, and both-orders judge votes from
``outputs/api/runs/*__4-comparisons__*/``, collapses votes to per-judge
verdicts (orientation flips -> tie), derives panel winners, applies the
pre-specified screens, and reports win rates with the agreed estimators.

Screens: pairs dropped if either output is missing/non-stop/empty
(mechanical, all conditions) or — moral pairs only — if either arm carries
refusal content per the LLM classifier. Panel ties excluded from win-rate
denominators. Refusal/tie counts are reported, not hidden.

Estimators per contrast x task: pooled win rate with naive Wilson
(descriptive), actor-clustered CI on the pooled rate (primary), equal-weight
actor mean with t(6) CI (sensitivity). Per-actor cells: Wilson (descriptive).

Also: judge diagnostics from both-orders judging — per judge: first-position
vote share and orientation-flip rate (position-determined fraction).

Outputs (outputs/analysis/):
- modgrid_judged_pairs.csv     one row per screened judged pair
- modgrid_results_cells.csv    per contrast x task x actor
- modgrid_results_summary.md   headline tables
- modgrid_judge_diagnostics.csv

Usage:
  python -m utility_behavior_gap.scripts.analyze_modgrid_judged
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition
from utility_behavior_gap.paths import ROOT

OUT_DIR = ROOT / "outputs" / "analysis"
CLASSIFICATIONS = OUT_DIR / "modgrid_moral_refusal_classifications.jsonl"
REFUSAL_LABELS = {"partial_refusal", "full_refusal"}
T_CRIT = {6: 2.447}


def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def t_ci(vals):
    k = len(vals)
    m = sum(vals) / k
    if k < 2 or k - 1 not in T_CRIT:
        return m, float("nan"), float("nan")
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (k - 1))
    h = T_CRIT[k - 1] * sd / math.sqrt(k)
    return m, m - h, m + h


def clustered_pooled_ci(cells: dict[str, tuple[int, int]]):
    """Pooled rate with actor-clustered SE (linearization, t(k-1))."""
    k = len(cells)
    wins = sum(w for w, n in cells.values())
    total = sum(n for w, n in cells.values())
    if total == 0 or k < 2 or k - 1 not in T_CRIT:
        return float("nan"), float("nan"), float("nan")
    p = wins / total
    se = math.sqrt(sum((w - p * n) ** 2 for w, n in cells.values())) / total * math.sqrt(k / (k - 1))
    h = T_CRIT[k - 1] * se
    return p, p - h, p + h


def moral_refusal_ids() -> set[str]:
    flagged = set()
    if CLASSIFICATIONS.exists():
        for line in open(CLASSIFICATIONS):
            r = json.loads(line)
            if r["label"] in REFUSAL_LABELS:
                flagged.add(r["output_id"])
    return flagged


def load_runs():
    """Yield (job, out_a, out_b, votes) per pair across all modgrid runs."""
    for run_dir in sorted((ROOT / "outputs" / "api" / "runs").glob("*__4-comparisons__*")):
        jobs_path = run_dir / "generation_jobs.jsonl"
        gens_path = run_dir / "generations.jsonl"
        votes_path = run_dir / "judge_votes.jsonl"
        if not (jobs_path.exists() and gens_path.exists() and votes_path.exists()):
            continue
        gens = {}
        for line in open(gens_path):
            r = json.loads(line)
            gens[r["output_id"]] = r
        votes = defaultdict(list)
        for line in open(votes_path):
            v = json.loads(line)
            if v.get("success"):
                votes[v["pair_uid"]].append(v)
        for line in open(jobs_path):
            job = json.loads(line)
            out_a = gens.get(f"{job['pair_uid']}::a")
            out_b = gens.get(f"{job['pair_uid']}::b")
            if out_a is None or out_b is None:
                continue
            yield job, out_a, out_b, votes.get(job["pair_uid"], [])


def main() -> None:
    flagged = moral_refusal_ids()
    judged_rows = []
    judge_stats = defaultdict(lambda: {"first": 0, "ab_votes": 0, "flips": 0,
                                       "consistent": 0, "two_vote_sets": 0})
    screen_counts = defaultdict(lambda: defaultdict(int))

    for job, out_a, out_b, votes in load_runs():
        comparison, task = job["comparison"], job["task"]
        key = (comparison, task)
        screen_counts[key]["pairs"] += 1
        # mechanical screen
        if any(r.get("finish_reason") != "stop" or not (r.get("output_text") or "").strip()
               for r in (out_a, out_b)):
            screen_counts[key]["mechanical_excluded"] += 1
            continue
        # moral refusal screen
        if "_moral" in comparison and (out_a["output_id"] in flagged
                                       or out_b["output_id"] in flagged):
            screen_counts[key]["refusal_excluded"] += 1
            continue
        hashes = (output_text_fingerprint(out_a), output_text_fingerprint(out_b))
        valid = [v for v in votes
                 if (v.get("source_output_a_hash"), v.get("source_output_b_hash")) == hashes]
        by_judge = defaultdict(list)
        for v in valid:
            by_judge[v["judge_model"]].append(v)
            if v.get("parsed_winner") in ("a", "b"):
                judge_stats[v["judge_model"]]["ab_votes"] += 1
                if v["parsed_winner"] == "a":
                    judge_stats[v["judge_model"]]["first"] += 1
        verdicts = {}
        for judge, vs in by_judge.items():
            verdicts[judge] = derive_judge_verdict([v["winner_condition"] for v in vs])
            if len(vs) == 2:
                judge_stats[judge]["two_vote_sets"] += 1
                conds = {v["winner_condition"] for v in vs}
                if len(conds) == 1 and "tie" not in conds:
                    judge_stats[judge]["consistent"] += 1
                elif len(conds) == 2 and "tie" not in conds:
                    judge_stats[judge]["flips"] += 1
        if len(verdicts) < 3:
            screen_counts[key]["insufficient_votes"] += 1
            continue
        panel = derive_panel_winner_condition(job, list(verdicts.values()))
        judged_rows.append(dict(
            pair_uid=job["pair_uid"], comparison=comparison, task=task,
            actor=job["actor"], item_label=job["item_label"],
            repeat=job.get("repeat", ""), domain=job.get("domain", ""),
            delta_u=job.get("delta_u", ""),
            cause_pair_label=job.get("cause_pair_label", ""),
            condition_a=job["condition_a"], condition_b=job["condition_b"],
            predicted_condition=job["predicted_condition"],
            panel_winner_condition=panel,
        ))

    with open(OUT_DIR / "modgrid_judged_pairs.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(judged_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(judged_rows)

    # results per contrast x task
    cell_rows = []
    lines = ["# Modgrid judged results", "",
             "Win rate = predicted-side panel wins / resolved pairs (ties excluded).",
             "Primary CI = actor-clustered on the pooled rate; equal-weight t(6) as",
             "sensitivity. Moral pairs are refusal-screened (counts shown).", ""]
    for key in sorted({(r["comparison"], r["task"]) for r in judged_rows}):
        comparison, task = key
        rows = [r for r in judged_rows if r["comparison"] == comparison and r["task"] == task]
        cells = {}
        for actor in sorted({r["actor"] for r in rows}):
            sub = [r for r in rows if r["actor"] == actor]
            wins = sum(1 for r in sub if r["panel_winner_condition"] == r["predicted_condition"])
            losses = sum(1 for r in sub if r["panel_winner_condition"] == r["condition_b"]
                         or (r["panel_winner_condition"] not in
                             (r["predicted_condition"], "tie", "unresolved")))
            cells[actor] = (wins, wins + losses)
            p, lo, hi = wilson(wins, wins + losses)
            cell_rows.append(dict(comparison=comparison, task=task, actor=actor,
                                  resolved=wins + losses, predicted_wins=wins,
                                  win_rate=round(p, 4) if wins + losses else "",
                                  wilson_lo=round(lo, 4) if wins + losses else "",
                                  wilson_hi=round(hi, 4) if wins + losses else "",
                                  ties=sum(1 for r in sub if r["panel_winner_condition"] == "tie")))
        pooled_w = sum(w for w, n in cells.values())
        pooled_n = sum(n for w, n in cells.values())
        p, lo_c, hi_c = clustered_pooled_ci(cells)
        rates = [w / n for w, n in cells.values() if n]
        m_eq, lo_eq, hi_eq = t_ci(rates)
        ties = sum(1 for r in rows if r["panel_winner_condition"] == "tie")
        sc = screen_counts[key]
        sig = "*" if (not math.isnan(lo_c) and lo_c > 0.5) else (
            "(neg)" if (not math.isnan(hi_c) and hi_c < 0.5) else "")
        lines.append(
            f"- **{comparison} / {task}**: {p:.1%} ({pooled_w}/{pooled_n}, ties {ties}) "
            f"clustered [{lo_c:.1%}, {hi_c:.1%}]{sig}; equal-weight {m_eq:.1%} "
            f"[{lo_eq:.1%}, {hi_eq:.1%}]"
            + (f"; refusal-excluded {sc['refusal_excluded']}" if sc["refusal_excluded"] else ""))
    with open(OUT_DIR / "modgrid_results_cells.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cell_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(cell_rows)

    lines += ["", "## Judge diagnostics (both-orders)", "",
              "| judge | first-position share | flip rate (position-determined) |",
              "|---|---|---|"]
    diag_rows = []
    for judge, s in sorted(judge_stats.items()):
        first = s["first"] / s["ab_votes"] if s["ab_votes"] else float("nan")
        flip = s["flips"] / s["two_vote_sets"] if s["two_vote_sets"] else float("nan")
        lines.append(f"| {judge} | {first:.1%} | {flip:.1%} |")
        diag_rows.append(dict(judge=judge, ab_votes=s["ab_votes"],
                              first_position_share=round(first, 4),
                              two_vote_sets=s["two_vote_sets"],
                              flip_rate=round(flip, 4),
                              consistent_rate=round(s["consistent"] / s["two_vote_sets"], 4)
                              if s["two_vote_sets"] else ""))
    with open(OUT_DIR / "modgrid_judge_diagnostics.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(diag_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(diag_rows)
    (OUT_DIR / "modgrid_results_summary.md").write_text("\n".join(lines) + "\n")
    print(f"{len(judged_rows)} judged pairs aggregated")
    print(f"wrote modgrid_judged_pairs.csv, modgrid_results_cells.csv, "
          f"modgrid_judge_diagnostics.csv, modgrid_results_summary.md")


if __name__ == "__main__":
    main()
