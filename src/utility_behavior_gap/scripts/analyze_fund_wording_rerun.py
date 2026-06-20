#!/usr/bin/env python3
"""Analyze the fund-wording modgrid rerun only.

Reads the actor-scoped manifest lists produced by
``run_fund_wording_actor.sh`` and aggregates those run directories, avoiding
the older donate-wording ``*__4-comparisons__*`` runs.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition
from utility_behavior_gap.paths import ROOT

RUNS_DIR = ROOT / "outputs" / "api" / "runs"
OUT_DIR = ROOT / "outputs" / "analysis"
MANIFEST_GLOB = "fund_wording_rerun_manifests__*.tsv"
CLASSIFICATIONS = OUT_DIR / "fund_wording_moral_refusal_classifications.jsonl"
REFUSAL_LABELS = {"partial_refusal", "full_refusal"}
T_CRIT = {6: 2.447}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def manifest_paths() -> list[Path]:
    paths: list[Path] = []
    for tsv in sorted(RUNS_DIR.glob(MANIFEST_GLOB)):
        with tsv.open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    paths.append(Path(parts[2]))
    return paths


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def t_ci(vals: list[float]) -> tuple[float, float, float]:
    k = len(vals)
    m = sum(vals) / k
    if k < 2 or k - 1 not in T_CRIT:
        return m, float("nan"), float("nan")
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (k - 1))
    h = T_CRIT[k - 1] * sd / math.sqrt(k)
    return m, m - h, m + h


def clustered_pooled_ci(cells: dict[str, tuple[int, int]]) -> tuple[float, float, float]:
    k = len(cells)
    wins = sum(w for w, _ in cells.values())
    total = sum(n for _, n in cells.values())
    if total == 0 or k < 2 or k - 1 not in T_CRIT:
        return float("nan"), float("nan"), float("nan")
    p = wins / total
    se = math.sqrt(sum((w - p * n) ** 2 for w, n in cells.values())) / total * math.sqrt(k / (k - 1))
    h = T_CRIT[k - 1] * se
    return p, p - h, p + h


def moral_refusal_ids() -> set[str]:
    flagged: set[str] = set()
    for row in read_jsonl(CLASSIFICATIONS):
        if row.get("label") in REFUSAL_LABELS:
            flagged.add(str(row["output_id"]))
    return flagged


def load_run(manifest: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    run_dir = manifest.parent
    jobs = read_jsonl(manifest)
    gens = {row["output_id"]: row for row in read_jsonl(run_dir / "generations.jsonl")}
    votes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(run_dir / "judge_votes.jsonl"):
        if row.get("success"):
            votes[row["pair_uid"]].append(row)
    return jobs, gens, votes


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flagged = moral_refusal_ids()
    judged_rows: list[dict[str, Any]] = []
    screen_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    judge_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"first": 0, "ab_votes": 0, "flips": 0, "consistent": 0, "two_vote_sets": 0,
                 "length_success": 0, "nonstop_success": 0}
    )

    for manifest in manifest_paths():
        jobs, gens, votes_by_pair = load_run(manifest)
        for job in jobs:
            comparison, task = job["comparison"], job["task"]
            key = (comparison, task)
            screen_counts[key]["pairs"] += 1
            out_a = gens.get(f"{job['pair_uid']}::a")
            out_b = gens.get(f"{job['pair_uid']}::b")
            if out_a is None or out_b is None:
                screen_counts[key]["missing_generation"] += 1
                continue
            if any(row.get("finish_reason") != "stop" or not (row.get("output_text") or "").strip()
                   for row in (out_a, out_b)):
                screen_counts[key]["mechanical_excluded"] += 1
                continue
            if flagged and "_moral" in comparison and (
                out_a["output_id"] in flagged or out_b["output_id"] in flagged
            ):
                screen_counts[key]["refusal_excluded"] += 1
                continue

            hashes = (output_text_fingerprint(out_a), output_text_fingerprint(out_b))
            valid_votes = [
                vote for vote in votes_by_pair.get(job["pair_uid"], [])
                if (vote.get("source_output_a_hash"), vote.get("source_output_b_hash")) == hashes
            ]
            by_judge: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for vote in valid_votes:
                by_judge[vote["judge_model"]].append(vote)
                if vote.get("success") and vote.get("finish_reason") != "stop":
                    judge_stats[vote["judge_model"]]["nonstop_success"] += 1
                    if vote.get("finish_reason") == "length":
                        judge_stats[vote["judge_model"]]["length_success"] += 1
                if vote.get("parsed_winner") in ("a", "b"):
                    judge_stats[vote["judge_model"]]["ab_votes"] += 1
                    if vote["parsed_winner"] == "a":
                        judge_stats[vote["judge_model"]]["first"] += 1

            verdicts: dict[str, str] = {}
            for judge, judge_votes in by_judge.items():
                verdicts[judge] = derive_judge_verdict([vote["winner_condition"] for vote in judge_votes])
                if len(judge_votes) == 2:
                    judge_stats[judge]["two_vote_sets"] += 1
                    conditions = {vote["winner_condition"] for vote in judge_votes}
                    if len(conditions) == 1 and "tie" not in conditions:
                        judge_stats[judge]["consistent"] += 1
                    elif len(conditions) == 2 and "tie" not in conditions:
                        judge_stats[judge]["flips"] += 1
            if len(verdicts) < 3:
                screen_counts[key]["insufficient_votes"] += 1
                continue

            panel = derive_panel_winner_condition(job, list(verdicts.values()))
            judged_rows.append({
                "pair_uid": job["pair_uid"],
                "comparison": comparison,
                "task": task,
                "actor": job["actor"],
                "item_label": job["item_label"],
                "repeat": job.get("repeat", ""),
                "domain": job.get("domain", ""),
                "delta_u": job.get("delta_u", ""),
                "cause_pair_label": job.get("cause_pair_label", ""),
                "condition_a": job["condition_a"],
                "condition_b": job["condition_b"],
                "predicted_condition": job["predicted_condition"],
                "panel_winner_condition": panel,
            })

    if not judged_rows:
        raise SystemExit("no judged rows found")

    with (OUT_DIR / "fund_wording_judged_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(judged_rows[0]))
        writer.writeheader()
        writer.writerows(judged_rows)

    cell_rows: list[dict[str, Any]] = []
    lines = [
        "# Fund-wording rerun judged results",
        "",
        "Win rate = predicted-side panel wins / resolved pairs (panel ties excluded).",
        "Primary CI = actor-clustered on the pooled rate; equal-weight t(6) is sensitivity.",
        "Moral refusal screen is applied only if fund-wording classifications exist.",
        f"Refusal classification file present: {CLASSIFICATIONS.exists()}",
        "",
    ]
    for key in sorted({(row["comparison"], row["task"]) for row in judged_rows}):
        comparison, task = key
        rows = [row for row in judged_rows if row["comparison"] == comparison and row["task"] == task]
        cells: dict[str, tuple[int, int]] = {}
        for actor in sorted({row["actor"] for row in rows}):
            sub = [row for row in rows if row["actor"] == actor]
            wins = sum(1 for row in sub if row["panel_winner_condition"] == row["predicted_condition"])
            losses = sum(1 for row in sub if row["panel_winner_condition"] == row["condition_b"]
                         or (row["panel_winner_condition"] not in
                             (row["predicted_condition"], "tie", "unresolved")))
            n = wins + losses
            cells[actor] = (wins, n)
            p, lo, hi = wilson(wins, n)
            cell_rows.append({
                "comparison": comparison,
                "task": task,
                "actor": actor,
                "resolved": n,
                "predicted_wins": wins,
                "win_rate": round(p, 4) if n else "",
                "wilson_lo": round(lo, 4) if n else "",
                "wilson_hi": round(hi, 4) if n else "",
                "ties": sum(1 for row in sub if row["panel_winner_condition"] == "tie"),
            })
        pooled_wins = sum(wins for wins, _ in cells.values())
        pooled_n = sum(n for _, n in cells.values())
        p, lo_c, hi_c = clustered_pooled_ci(cells)
        actor_rates = [wins / n for wins, n in cells.values() if n]
        m_eq, lo_eq, hi_eq = t_ci(actor_rates)
        ties = sum(1 for row in rows if row["panel_winner_condition"] == "tie")
        sc = screen_counts[key]
        sig = "*" if not math.isnan(lo_c) and lo_c > 0.5 else (
            "(neg)" if not math.isnan(hi_c) and hi_c < 0.5 else ""
        )
        lines.append(
            f"- **{comparison} / {task}**: {p:.1%} ({pooled_wins}/{pooled_n}, ties {ties}) "
            f"clustered [{lo_c:.1%}, {hi_c:.1%}]{sig}; equal-weight {m_eq:.1%} "
            f"[{lo_eq:.1%}, {hi_eq:.1%}]"
            + (f"; missing generations {sc['missing_generation']}" if sc["missing_generation"] else "")
            + (f"; mechanical-excluded {sc['mechanical_excluded']}" if sc["mechanical_excluded"] else "")
            + (f"; refusal-excluded {sc['refusal_excluded']}" if sc["refusal_excluded"] else "")
            + (f"; insufficient votes {sc['insufficient_votes']}" if sc["insufficient_votes"] else "")
        )

    with (OUT_DIR / "fund_wording_results_cells.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)

    diag_rows = []
    lines += ["", "## Judge diagnostics", "",
              "| judge | first-position share | flip rate | successful non-stop rows | successful length rows |",
              "|---|---|---|---|---|"]
    for judge, stats in sorted(judge_stats.items()):
        first = stats["first"] / stats["ab_votes"] if stats["ab_votes"] else float("nan")
        flip = stats["flips"] / stats["two_vote_sets"] if stats["two_vote_sets"] else float("nan")
        lines.append(
            f"| {judge} | {first:.1%} | {flip:.1%} | "
            f"{stats['nonstop_success']} | {stats['length_success']} |"
        )
        diag_rows.append({
            "judge": judge,
            "ab_votes": stats["ab_votes"],
            "first_position_share": round(first, 4),
            "two_vote_sets": stats["two_vote_sets"],
            "flip_rate": round(flip, 4),
            "consistent_rate": round(stats["consistent"] / stats["two_vote_sets"], 4)
            if stats["two_vote_sets"] else "",
            "successful_nonstop_rows": stats["nonstop_success"],
            "successful_length_rows": stats["length_success"],
        })
    with (OUT_DIR / "fund_wording_judge_diagnostics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(diag_rows[0]))
        writer.writeheader()
        writer.writerows(diag_rows)

    (OUT_DIR / "fund_wording_results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(judged_rows)} judged pairs aggregated")
    print("wrote fund_wording_judged_pairs.csv, fund_wording_results_cells.csv, "
          "fund_wording_judge_diagnostics.csv, fund_wording_results_summary.md")


if __name__ == "__main__":
    main()
