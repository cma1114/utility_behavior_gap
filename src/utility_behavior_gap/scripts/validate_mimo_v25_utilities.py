"""Validate the MiMo V2.5 fitted utilities used in the behavioral experiments.

The MiMo V2.5 refit (outputs/utility_refits/mimo_v25_full/) was a single POOLED
Thurstonian fit over all 610 outcomes, so it has one pooled held-out accuracy
(0.95) rather than the per-domain values the other six actors have. The pooled
utilities are the ones actually in selected_pairs.csv, so the correct
validation is to evaluate THAT fit per domain (not to re-fit per domain, which
would produce different utilities than the experiments used).

This script, entirely from stored refit data (no API):
  1. per-domain held-out accuracy + log-loss of the pooled fit (within-domain
     hold-out edges), matching the original Thurstonian eval formula;
  2. per-domain train accuracy + log-loss (within-domain training edges);
  3. global transitivity (fraction of intransitive triads) from the observed
     pairwise choices;
  4. rewrites the placeholder hold-out/train columns in
     outputs/inputs/utility_options__mimo-v25-pro-or.csv with the real
     per-domain values.

Usage:
  python -m utility_behavior_gap.scripts.validate_mimo_v25_utilities
  python -m utility_behavior_gap.scripts.validate_mimo_v25_utilities --no-write
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict

from utility_behavior_gap.paths import ROOT

REFIT = ROOT / "outputs" / "utility_refits" / "mimo_v25_full" / "results_mimo_v25_full.json"
OVERLAY = ROOT / "outputs" / "inputs" / "utility_options__mimo-v25-pro-or.csv"
SUMMARY = ROOT / "outputs" / "analysis" / "mimo_v25_utility_validation_summary.md"
DOMAINS = ["religions", "animals", "countries", "political"]
EPS = 1e-6


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-write", action="store_true", help="report only; do not rewrite the overlay")
    args = ap.parse_args()

    d = json.load(open(REFIT))
    ut = {int(k): v for k, v in d["utilities"].items()}
    id2desc = {o["id"]: o["description"].strip() for o in d["options"]}
    overlay_rows = list(csv.DictReader(open(OVERLAY)))
    desc2dom = {r["description"].strip(): r["domain"] for r in overlay_rows}
    edges = d["graph_data"]["edges"]

    def lookup(a, b):
        return edges.get(f"({a}, {b})") or edges.get(f"({b}, {a})")

    def pair_domain(a, b):
        da, db = desc2dom.get(id2desc.get(a)), desc2dom.get(id2desc.get(b))
        return da if da == db else None

    def score(pairs):
        """Per-domain (accuracy, log_loss, n) over within-domain edges in `pairs`."""
        acc = defaultdict(list)
        ll = defaultdict(list)
        overall_acc, overall_ll = [], []
        for pair in pairs:
            a, b = pair
            e = lookup(a, b)
            if e is None:
                continue
            A, B = e["option_A"]["id"], e["option_B"]["id"]
            muA, vA = ut[A]["mean"], ut[A]["variance"]
            muB, vB = ut[B]["mean"], ut[B]["variance"]
            p = min(max(phi((muA - muB) / math.sqrt(vA + vB + 1e-9)), EPS), 1 - EPS)
            y = e["probability_A"]
            correct = int((p >= 0.5) == (y >= 0.5))
            logloss = -(y * math.log(p) + (1 - y) * math.log(1 - p))
            overall_acc.append(correct)
            overall_ll.append(logloss)
            dom = pair_domain(A, B)
            if dom:
                acc[dom].append(correct)
                ll[dom].append(logloss)
        per = {dom: (sum(acc[dom]) / len(acc[dom]), sum(ll[dom]) / len(ll[dom]), len(acc[dom]))
               for dom in DOMAINS if acc[dom]}
        ov = (sum(overall_acc) / len(overall_acc), sum(overall_ll) / len(overall_ll), len(overall_acc))
        return per, ov

    holdout_per, holdout_ov = score(d["graph_data"]["holdout_edge_indices"])
    train_per, train_ov = score(d["graph_data"]["training_edges"])

    # transitivity over the observed preference tournament
    n = len(d["options"])
    import numpy as np
    W = np.zeros((n, n), dtype=np.int32)
    A = np.zeros((n, n), dtype=np.int32)
    for e in edges.values():
        a, b, pa = e["option_A"]["id"], e["option_B"]["id"], e["probability_A"]
        A[a, b] = A[b, a] = 1
        if pa > 0.5:
            W[a, b] = 1
        elif pa < 0.5:
            W[b, a] = 1
    cyclic = int(np.trace(np.linalg.matrix_power(W, 3)) // 3)
    complete = int(np.trace(np.linalg.matrix_power(A, 3)) // 6)
    intransitive_rate = cyclic / complete

    # report
    lines = ["# MiMo V2.5 utility validation (pooled fit, evaluated per domain)", "",
             f"Reported pooled hold-out accuracy: 0.95; recomputed overall: "
             f"{holdout_ov[0]:.4f} (sanity check).", "",
             "## Per-domain held-out accuracy / log-loss (within-domain hold-out edges)", "",
             "| domain | holdout_accuracy | holdout_log_loss | n | train_accuracy | train_n |",
             "|---|---|---|---|---|---|"]
    for dom in DOMAINS:
        ha, hl, hn = holdout_per.get(dom, (float("nan"),) * 3)
        ta, _, tn = train_per.get(dom, (float("nan"),) * 3)
        lines.append(f"| {dom} | {ha:.4f} | {hl:.4f} | {hn} | {ta:.4f} | {tn} |")
    mean_acc = sum(holdout_per[dom][0] for dom in DOMAINS) / len(DOMAINS)
    lines += ["",
              f"Equal-weight mean per-domain held-out accuracy: **{mean_acc:.4f}**",
              "",
              "## Transitivity", "",
              f"- complete triads (all three pairs observed): {complete:,}",
              f"- intransitive (cyclic) triads: {cyclic:,}",
              f"- intransitive-triad rate: **{intransitive_rate:.5f}** "
              f"(random-tournament baseline 0.25); transitivity {1 - intransitive_rate:.5f}",
              "",
              "Per-domain hold-out edge counts are far smaller than the per-domain-fit",
              "actors (the pooled hold-out is mostly cross-domain); see n above.", ""]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines))
    print("\n".join(lines))

    if not args.no_write:
        ha_by = {dom: holdout_per[dom][0] for dom in DOMAINS}
        hl_by = {dom: holdout_per[dom][1] for dom in DOMAINS}
        ta_by = {dom: train_per[dom][0] for dom in DOMAINS}
        tl_by = {dom: train_per[dom][1] for dom in DOMAINS}
        for r in overlay_rows:
            dom = r["domain"]
            r["holdout_accuracy"] = f"{ha_by[dom]:.10g}"
            r["holdout_log_loss"] = f"{hl_by[dom]:.10g}"
            r["train_accuracy"] = f"{ta_by[dom]:.10g}"
            r["train_log_loss"] = f"{tl_by[dom]:.10g}"
        with open(OVERLAY, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(overlay_rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(overlay_rows)
        print(f"\nrewrote per-domain hold-out/train values into {OVERLAY}")


if __name__ == "__main__":
    main()
