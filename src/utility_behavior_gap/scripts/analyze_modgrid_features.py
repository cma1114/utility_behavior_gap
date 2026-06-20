"""Unified textual feature analysis of modgrid arms vs the R0 reference.

Runs entirely locally on stored generations (no API):
- all modgrid arm outputs (``outputs/api/runs/*__4-comparisons__*/generations.jsonl``)
- the R0 store (``outputs/api/r0_generations.jsonl``)

R0 shares the item x repeat grid with every arm, so each arm output gets a
slot-matched bare-task counterpart and deltas are within-slot pairs.

Screens (per protocol): outputs with finish_reason != stop or empty text are
dropped everywhere (mechanical). Moral arms additionally use the LLM refusal
classifier: refusal-content outputs are excluded from summaries/deltas but kept
in the per-output CSV with their label.

Features: length/structure (words, sentences, paragraphs, avg sentence words,
unique-word rate, mean word length); per-1k rhetoric markers (contrast framing,
counterargument, examples, transitions, qualification/hedging, semicolons+
colons, dashes); quantitative specificity per-1k (numbers, percentages,
currency); readability (textstat FK grade, reading ease, Gunning fog); VADER
sentiment (pos/neg proportions, compound); spaCy adjective/adverb rates
(tagger only; skip with --no-pos).

Outputs (outputs/analysis/):
- modgrid_features_by_output.csv
- modgrid_features_arm_means.csv      (task x condition x actor means)
- modgrid_features_vs_r0.csv          (slot-paired arm - R0 deltas; per-actor
                                       and actor-clustered t CI per task x condition)
- modgrid_features_summary.md

Usage:
  python -m utility_behavior_gap.scripts.analyze_modgrid_features [--no-pos] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utility_behavior_gap.paths import ROOT

OUT_DIR = ROOT / "outputs" / "analysis"
R0_PATH = ROOT / "outputs" / "api" / "r0_generations.jsonl"
CLASSIFICATIONS = OUT_DIR / "modgrid_moral_refusal_classifications.jsonl"
T_CRIT = {4: 2.776, 5: 2.571, 6: 2.447}

RHETORIC = {
    "contrast_per_1k": re.compile(
        r"\b(rather than|instead of|by contrast|on the other hand|not merely|not only|not just)\b", re.I),
    "counterargument_per_1k": re.compile(
        r"\b(critics|opponents|some (?:may )?argue|although|though|objections?)\b", re.I),
    "example_per_1k": re.compile(
        r"\b(for example|for instance|such as|including|consider)\b", re.I),
    "transition_per_1k": re.compile(
        r"\b(therefore|consequently|as a result|ultimately|moreover|furthermore|however)\b", re.I),
    "qualification_per_1k": re.compile(
        r"\b(may|might|could|perhaps|likely|often|sometimes|in some cases|to some extent)\b", re.I),
}
# One combined numeric-specificity pattern: plain numbers, percentages,
# currency figures (owner ruling 2026-06-10: a single score, not three).
NUMERIC_RE = re.compile(r"[$€£]\s?\d[\d,.]*|\d[\d,]*(?:\.\d+)?\s?(?:%|percent)?", re.I)

# VADER used as a sentiment-word LEXICON: separate counts of positive and
# negative words per 1k (owner ruling: two counts, no compound score).
_VADER_LEXICON = SentimentIntensityAnalyzer().lexicon
POSITIVE_WORDS = {w for w, v in _VADER_LEXICON.items() if v > 0}
NEGATIVE_WORDS = {w for w, v in _VADER_LEXICON.items() if v < 0}
TOKEN_RE = re.compile(r"[a-z']+")


def base_features(text: str) -> dict[str, float]:
    words = text.split()
    n = len(words) or 1
    sentences = max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)))
    paragraphs = len([p for p in text.split("\n\n") if p.strip()]) or 1
    tokens = TOKEN_RE.findall(text.lower())
    feats = {
        "words": float(len(words)),
        "sentences": float(sentences),
        "paragraphs": float(paragraphs),
        "avg_sentence_words": len(words) / sentences,
        "unique_word_rate": len({w.lower() for w in words}) / n,
        "semicolon_colon_per_1k": 1000 * (text.count(";") + text.count(":")) / n,
        "dash_per_1k": 1000 * (text.count("—") + text.count("–") + text.count(" - ")) / n,
        "numeric_specificity_per_1k": 1000 * len(NUMERIC_RE.findall(text)) / n,
        "fk_grade": float(textstat.flesch_kincaid_grade(text)),
        "pos_words_per_1k": 1000 * sum(1 for t in tokens if t in POSITIVE_WORDS) / n,
        "neg_words_per_1k": 1000 * sum(1 for t in tokens if t in NEGATIVE_WORDS) / n,
    }
    for name, pattern in RHETORIC.items():
        feats[name] = 1000 * len(pattern.findall(text)) / n
    return feats


def slot_key(uid: str, task: str, actor: str) -> tuple | None:
    """Parse ...:i{item_id}:r{rep} from the right (task names can contain ':i')."""
    stem, _, rep = uid.rpartition(":r")
    _, _, item = stem.rpartition(":i")
    if not rep.isdigit() or not item:
        return None
    return (task, actor, item, int(rep))


def load_rows(limit: int | None) -> list[dict]:
    """limit applies per source (arms / R0) so smoke tests still form pairs."""
    rows = []
    for path in sorted((ROOT / "outputs" / "api" / "runs").glob("*__4-comparisons__*/generations.jsonl")):
        task = path.parent.name.split("__")[0]
        for line in open(path):
            r = json.loads(line)
            if r.get("finish_reason") != "stop" or not (r.get("output_text") or "").strip():
                continue
            rows.append(dict(task=task, actor=r["actor"], condition=r["condition"],
                             output_id=r["output_id"],
                             slot=slot_key(r["pair_uid"], task, r["actor"]),
                             text=r["output_text"]))
    if limit:
        rows = rows[:limit]
    r0_rows = []
    for line in open(R0_PATH):
        r = json.loads(line)
        if not r.get("ok"):
            continue
        r0_rows.append(dict(task=r["task"], actor=r["actor"], condition="r0",
                            output_id=r["output_id"],
                            slot=(r["task"], r["actor"], str(r["item_id"]), int(r["repeat"])),
                            text=r["output_text"]))
    if limit:
        wanted = {r["slot"] for r in rows}
        r0_rows = [r for r in r0_rows if r["slot"] in wanted]
    return rows + r0_rows


def moral_labels() -> dict[str, str]:
    labels = {}
    if CLASSIFICATIONS.exists():
        for line in open(CLASSIFICATIONS):
            r = json.loads(line)
            labels[r["output_id"]] = r["label"]
    return labels


def add_pos_rates(rows: list[dict]) -> None:
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "attribute_ruler"])
    texts = (r["text"] for r in rows)
    for r, doc in zip(rows, nlp.pipe(texts, batch_size=64)):
        tokens = [t for t in doc if t.is_alpha]
        n = len(tokens) or 1
        # single combined modifier rate (owner ruling: one POS score)
        r["features"]["modifier_rate"] = sum(t.pos_ in ("ADJ", "ADV") for t in tokens) / n


def t_ci(vals: list[float]) -> tuple[float, float, float]:
    k = len(vals)
    m = sum(vals) / k
    if k < 2 or k - 1 not in T_CRIT:
        return m, float("nan"), float("nan")
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (k - 1))
    h = T_CRIT[k - 1] * sd / math.sqrt(k)
    return m, m - h, m + h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-pos", action="store_true", help="skip spaCy POS rates (much faster)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = load_rows(args.limit)
    labels = moral_labels()
    for r in rows:
        r["moral_label"] = labels.get(r["output_id"], "")
        r["features"] = base_features(r["text"])
    if not args.no_pos:
        add_pos_rates(rows)
    print(f"featurized {len(rows)} outputs")

    feature_names = sorted(rows[0]["features"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "modgrid_features_by_output.csv", "w") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["output_id", "task", "actor", "condition", "moral_label"] + feature_names)
        for r in rows:
            w.writerow([r["output_id"], r["task"], r["actor"], r["condition"],
                        r["moral_label"]] + [round(r["features"][f], 5) for f in feature_names])

    def screened(r: dict) -> bool:
        return r["moral_label"] in ("", "clean")

    # arm means per task x condition x actor
    cells = defaultdict(list)
    for r in rows:
        if screened(r):
            cells[(r["task"], r["condition"], r["actor"])].append(r["features"])
    mean_rows = []
    for (task, cond, actor), feats in sorted(cells.items()):
        row = dict(task=task, condition=cond, actor=actor, n=len(feats))
        for f in feature_names:
            row[f] = round(statistics.mean(x[f] for x in feats), 5)
        mean_rows.append(row)
    with open(OUT_DIR / "modgrid_features_arm_means.csv", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mean_rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(mean_rows)

    # slot-paired deltas: every arm vs R0, plus the within-condition contrasts
    by_slot: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if r["slot"] and screened(r):
            by_slot[(r["task"],) + r["slot"][1:]][r["condition"]] = r["features"]

    headline = ["words", "fk_grade", "numeric_specificity_per_1k",
                "pos_words_per_1k", "neg_words_per_1k", "qualification_per_1k"]
    arm_conditions = sorted({r["condition"] for r in rows if r["condition"] != "r0"})
    contrasts = ([(c, "r0", f"{c} vs R0") for c in arm_conditions]
                 + [("moral_good", "moral_bad", "moral: good - bad"),
                    ("amount_high", "amount_low", "amount: $1M - $100"),
                    ("hl_high", "hl_low", "highlow: high - low"),
                    ("framed_strong", "framed_neutral", "headroom: strong - neutral")])

    delta_rows = []
    summary_lines = ["# Modgrid feature deltas (slot-paired)", "",
                     "CI = t over the 7 per-actor mean deltas; * = CI excludes 0.",
                     "Moral arms are refusal-screened (clean outputs only).", ""]
    for task in sorted({r["task"] for r in rows}):
        summary_lines += [f"## {task}", "",
                          "| contrast | " + " | ".join(headline) + " |",
                          "|" + "---|" * (len(headline) + 1)]
        for cond_a, cond_b, label in contrasts:
            per_actor = defaultdict(lambda: defaultdict(list))
            for (t, actor, item, rep), conds in by_slot.items():
                if t != task or cond_a not in conds or cond_b not in conds:
                    continue
                for f in feature_names:
                    per_actor[actor][f].append(conds[cond_a][f] - conds[cond_b][f])
            if not per_actor:
                continue
            cells_out = {}
            for f in feature_names:
                actor_means = [statistics.mean(v[f]) for v in per_actor.values() if v[f]]
                m, lo, hi = t_ci(actor_means)
                sig = "*" if (not math.isnan(lo) and (lo > 0 or hi < 0)) else ""
                delta_rows.append(dict(task=task, contrast=label, condition_a=cond_a,
                                       condition_b=cond_b, feature=f,
                                       mean_delta=round(m, 5), ci_lo=round(lo, 5),
                                       ci_hi=round(hi, 5),
                                       n_pairs=sum(len(v[f]) for v in per_actor.values())))
                cells_out[f] = f"{m:+.2f}{sig}"
            summary_lines.append(
                f"| {label} | " + " | ".join(cells_out[f] for f in headline) + " |")
        summary_lines.append("")
    if delta_rows:
        with open(OUT_DIR / "modgrid_features_deltas.csv", "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(delta_rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(delta_rows)
    else:
        print("warning: no slot-paired deltas computed (missing R0 matches)")
    (OUT_DIR / "modgrid_features_summary.md").write_text("\n".join(summary_lines) + "\n")
    print(f"wrote modgrid_features_{{by_output,arm_means,deltas}}.csv and modgrid_features_summary.md")


if __name__ == "__main__":
    main()
