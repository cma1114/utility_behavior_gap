#!/usr/bin/env python3
"""Task B: pooled stats for the amount condition ($1,000,000 vs $100).

Predicted side = larger amount ($1M). Ties excluded from the win-rate
denominator but reported. Data: outputs/processed/incentive_channel_data.csv
(condition == "amount"): essay = 30 topics x 5 repeats; the three non-essay
tasks = 30 items x 5 repeats. n_left_wins = larger-amount wins, n_right_wins =
smaller-amount wins, n_ties = panel ties.

Outputs:
  outputs/analysis/amount_condition_pooled_by_task.csv
  outputs/analysis/amount_condition_per_cell.csv
"""
from __future__ import annotations
import csv
import math

from utility_behavior_gap.paths import ANALYSIS, PROCESSED

CSV = PROCESSED / "incentive_channel_data.csv"
ANA = ANALYSIS
ANA.mkdir(parents=True, exist_ok=True)

TASK_ORDER = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]
TASK_LABEL = {"essay": "Essay", "grant_proposal_abstract": "Grant abstract",
              "incident_postmortem": "Incident postmortem", "translation": "Translation"}


def wilson(w, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    rows = [r for r in csv.DictReader(open(CSV)) if r.get("condition") == "amount"]

    def num(x):
        try:
            return int(float(x))
        except Exception:
            return None

    # ---- per-cell ----
    cells = []
    for r in rows:
        lw, sw, ti = num(r["n_left_wins"]), num(r["n_right_wins"]), num(r["n_ties"])
        if lw is None or sw is None:
            print(f"[warn] missing counts {r['task']}/{r['actor']}"); continue
        ti = ti or 0
        n = lw + sw
        rate, lo, hi = wilson(lw, n)
        cells.append(dict(task=r["task"], actor=r["actor"],
                          n_large_amount_wins=lw, n_small_amount_wins=sw, n_ties=ti,
                          total_incl_ties=lw + sw + ti, non_tied_n=n,
                          larger_amount_win_rate=round(rate, 4),
                          ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                          ci_positive=("yes" if lo > 0.50 else "no")))
    with open(ANA / "amount_condition_per_cell.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(cells)

    # ---- per-task pooled ----
    bytask = []
    for tk in TASK_ORDER:
        sub = [c for c in cells if c["task"] == tk]
        lw = sum(c["n_large_amount_wins"] for c in sub)
        sw = sum(c["n_small_amount_wins"] for c in sub)
        ti = sum(c["n_ties"] for c in sub)
        n = lw + sw
        rate, lo, hi = wilson(lw, n)
        pos = sum(1 for c in sub if c["ci_positive"] == "yes")
        bytask.append(dict(task=tk, n_large_amount_wins=lw, n_small_amount_wins=sw, n_ties=ti,
                           total_incl_ties=lw + sw + ti, non_tied_n=n,
                           larger_amount_win_rate=round(rate, 4),
                           ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                           ci_positive_cells_of_7=pos))
    with open(ANA / "amount_condition_pooled_by_task.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bytask[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(bytask)

    # ---- overall ----
    LW = sum(c["n_large_amount_wins"] for c in cells)
    SW = sum(c["n_small_amount_wins"] for c in cells)
    TI = sum(c["n_ties"] for c in cells)
    N = LW + SW
    rate, lo, hi = wilson(LW, N)
    POS = sum(1 for c in cells if c["ci_positive"] == "yes")

    print(f"=== OVERALL (28 cells) ===")
    print(f"  larger wins={LW} smaller wins={SW} ties={TI} total(incl ties)={LW+SW+TI} non-tied={N}")
    print(f"  pooled larger-amount win rate={rate*100:.2f}% Wilson95%=[{lo*100:.2f},{hi*100:.2f}]  CI-positive cells={POS}/28")
    print("=== BY TASK ===")
    for b in bytask:
        print(f"  {TASK_LABEL[b['task']]:20s} large={b['n_large_amount_wins']:4d} small={b['n_small_amount_wins']:4d} "
              f"tie={b['n_ties']:3d} non-tied={b['non_tied_n']:4d} rate={b['larger_amount_win_rate']*100:.2f}% "
              f"CI[{b['ci_lo']*100:.1f},{b['ci_hi']*100:.1f}] pos={b['ci_positive_cells_of_7']}/7")
    return dict(LW=LW, SW=SW, TI=TI, N=N, rate=rate, lo=lo, hi=hi, POS=POS,
                bytask=bytask, cells=cells)


if __name__ == "__main__":
    main()
