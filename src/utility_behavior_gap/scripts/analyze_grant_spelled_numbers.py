#!/usr/bin/env python3
"""Analyze the grant high-low spelled-number ablation.

The ablation uses the same grant topics and selected utility pairs as
``modgrid_grant_highlow`` but spells counts inside the utility consequence
text, e.g. "one thousand" instead of "1,000".
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from utility_behavior_gap.constants import ACTOR_LABEL, ACTORS, DOMAIN_LABEL
from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition
from utility_behavior_gap.paths import ROOT

RUNS_DIR = ROOT / "outputs" / "api" / "runs"
OUT_DIR = ROOT / "outputs" / "analysis"
MANIFEST_GLOB = "grant_spelled_numbers_manifests__*.tsv"
T_CRIT = {6: 2.447}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, center - half, center + half


def clustered_pooled_ci(cells: dict[str, tuple[int, int]]) -> tuple[float, float, float]:
    k = len(cells)
    wins = sum(w for w, _ in cells.values())
    total = sum(n for _, n in cells.values())
    if total == 0 or k < 2 or k - 1 not in T_CRIT:
        return float("nan"), float("nan"), float("nan")
    p = wins / total
    se = math.sqrt(sum((w - p * n) ** 2 for w, n in cells.values())) / total * math.sqrt(k / (k - 1))
    half = T_CRIT[k - 1] * se
    return p, p - half, p + half


def t_ci(values: list[float]) -> tuple[float, float, float]:
    k = len(values)
    if k == 0:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / k
    if k < 2 or k - 1 not in T_CRIT:
        return mean, float("nan"), float("nan")
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (k - 1))
    half = T_CRIT[k - 1] * sd / math.sqrt(k)
    return mean, mean - half, mean + half


def load_run(manifest: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    run_dir = manifest.parent
    jobs = read_jsonl(manifest)
    generations = {row["output_id"]: row for row in read_jsonl(run_dir / "generations.jsonl")}
    votes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(run_dir / "judge_votes.jsonl"):
        if row.get("success"):
            votes[row["pair_uid"]].append(row)
    return jobs, generations, votes


def judged_rows() -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    screens: dict[str, int] = defaultdict(int)
    for manifest in manifest_paths():
        jobs, generations, votes_by_pair = load_run(manifest)
        for job in jobs:
            screens["pairs"] += 1
            out_a = generations.get(f"{job['pair_uid']}::a")
            out_b = generations.get(f"{job['pair_uid']}::b")
            if out_a is None or out_b is None:
                screens["missing_generation"] += 1
                continue
            if any(
                output.get("finish_reason") != "stop" or not (output.get("output_text") or "").strip()
                for output in (out_a, out_b)
            ):
                screens["mechanical_excluded"] += 1
                continue

            hashes = (output_text_fingerprint(out_a), output_text_fingerprint(out_b))
            valid_votes = [
                vote
                for vote in votes_by_pair.get(job["pair_uid"], [])
                if (vote.get("source_output_a_hash"), vote.get("source_output_b_hash")) == hashes
            ]
            by_judge: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for vote in valid_votes:
                by_judge[vote["judge_model"]].append(vote)
            verdicts = [
                derive_judge_verdict([vote["winner_condition"] for vote in judge_votes])
                for judge_votes in by_judge.values()
            ]
            if len(verdicts) < 3:
                screens["insufficient_votes"] += 1
                continue

            panel = derive_panel_winner_condition(job, verdicts)
            rows.append(
                {
                    "pair_uid": job["pair_uid"],
                    "comparison": job["comparison"],
                    "actor": job["actor"],
                    "actor_label": ACTOR_LABEL.get(job["actor"], job["actor"]),
                    "task": job["task"],
                    "domain": job.get("domain", ""),
                    "domain_label": DOMAIN_LABEL.get(job.get("domain", ""), job.get("domain", "")),
                    "item_label": job["item_label"],
                    "repeat": job.get("repeat", ""),
                    "pair_idx": job.get("pair_idx", ""),
                    "pair_set": job.get("pair_set", ""),
                    "prompt_variant_id": job.get("prompt_variant_id", ""),
                    "high_description": job.get("high_description", ""),
                    "low_description": job.get("low_description", ""),
                    "high_consequence": job.get("high_consequence", ""),
                    "low_consequence": job.get("low_consequence", ""),
                    "high_utility": job.get("high_utility", ""),
                    "low_utility": job.get("low_utility", ""),
                    "delta_u": job.get("delta_u", ""),
                    "predicted_condition": job["predicted_condition"],
                    "panel_winner_condition": panel,
                }
            )
    return rows, screens


def summarize(rows: list[dict[str, Any]], group_cols: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    if group_cols:
        for row in rows:
            groups[tuple(str(row.get(col, "")) for col in group_cols)].append(row)
    else:
        groups[()].extend(rows)

    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        high = sum(row["panel_winner_condition"] == "hl_high" for row in group)
        low = sum(row["panel_winner_condition"] == "hl_low" for row in group)
        ties = sum(row["panel_winner_condition"] == "tie" for row in group)
        n = high + low
        p, lo, hi = wilson(high, n)
        record = {
            "breakout": "total" if not group_cols else "_x_".join(group_cols),
            "high_wins": high,
            "low_wins": low,
            "ties": ties,
            "n_excl_tie": n,
            "high_win_rate": p,
            "wilson_ci_lo": lo,
            "wilson_ci_hi": hi,
        }
        for col, value in zip(group_cols, key):
            record[col] = value
        out.append(record)
    return out


def write_outputs(rows: list[dict[str, Any]], screens: dict[str, int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    judged_path = OUT_DIR / "grant_spelled_numbers_judged_pairs.csv"
    cells_path = OUT_DIR / "grant_spelled_numbers_results_cells.csv"
    summary_path = OUT_DIR / "grant_spelled_numbers_results_summary.md"

    fieldnames = list(rows[0]) if rows else []
    with judged_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    cell_rows: list[dict[str, Any]] = []
    for group_cols in [[], ["actor", "actor_label"], ["domain", "domain_label"], ["actor", "actor_label", "domain", "domain_label"]]:
        cell_rows.extend(summarize(rows, group_cols))
    with cells_path.open("w", newline="", encoding="utf-8") as fh:
        preferred = [
            "breakout",
            "actor",
            "actor_label",
            "domain",
            "domain_label",
            "high_wins",
            "low_wins",
            "ties",
            "n_excl_tie",
            "high_win_rate",
            "wilson_ci_lo",
            "wilson_ci_hi",
        ]
        extras = sorted({key for row in cell_rows for key in row} - set(preferred))
        writer = csv.DictWriter(fh, fieldnames=preferred + extras, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cell_rows)

    actor_cells = {
        row["actor"]: (int(row["high_wins"]), int(row["n_excl_tie"]))
        for row in cell_rows
        if row["breakout"] == "actor_x_actor_label"
    }
    pooled, clustered_lo, clustered_hi = clustered_pooled_ci(actor_cells)
    actor_rates = [wins / n for wins, n in actor_cells.values() if n]
    equal_mean, equal_lo, equal_hi = t_ci(actor_rates)
    total = [row for row in cell_rows if row["breakout"] == "total"][0]

    def pct(value: float) -> str:
        return "" if math.isnan(value) else f"{100 * value:.1f}%"

    lines = [
        "# Grant high-low spelled-number ablation",
        "",
        "Counts inside utility consequences are spelled as words; selected utility pairs, grant topics, wrapper, and judging protocol are otherwise unchanged.",
        "",
        "## Overall",
        "",
        f"- Pooled high-side win rate: {pct(pooled)} ({total['high_wins']}/{total['n_excl_tie']}; ties {total['ties']})",
        f"- Actor-clustered CI: {pct(clustered_lo)}-{pct(clustered_hi)}",
        f"- Equal-actor mean: {pct(equal_mean)} ({pct(equal_lo)}-{pct(equal_hi)})",
        "",
        "## Screens",
        "",
    ]
    for key in ["pairs", "missing_generation", "mechanical_excluded", "insufficient_votes"]:
        lines.append(f"- {key}: {int(screens.get(key, 0))}")

    lines += [
        "",
        "## By actor",
        "",
        "| actor | high | low | ties | high win rate | Wilson CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        [row for row in cell_rows if row["breakout"] == "actor_x_actor_label"],
        key=lambda row: ACTORS.index(row["actor"]) if row["actor"] in ACTORS else 999,
    ):
        lines.append(
            f"| {row['actor_label']} | {row['high_wins']} | {row['low_wins']} | {row['ties']} | "
            f"{pct(float(row['high_win_rate']))} | {pct(float(row['wilson_ci_lo']))}-{pct(float(row['wilson_ci_hi']))} |"
        )

    lines += [
        "",
        "## By domain",
        "",
        "| domain | high | low | ties | high win rate | Wilson CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted([row for row in cell_rows if row["breakout"] == "domain_x_domain_label"], key=lambda row: row["domain"]):
        lines.append(
            f"| {row['domain_label']} | {row['high_wins']} | {row['low_wins']} | {row['ties']} | "
            f"{pct(float(row['high_win_rate']))} | {pct(float(row['wilson_ci_lo']))}-{pct(float(row['wilson_ci_hi']))} |"
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {judged_path}")
    print(f"wrote {cells_path}")
    print(f"wrote {summary_path}")


def main() -> None:
    rows, screens = judged_rows()
    if not rows:
        raise SystemExit("no judged spelled-number grant rows found")
    write_outputs(rows, screens)


if __name__ == "__main__":
    main()
