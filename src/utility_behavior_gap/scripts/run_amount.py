#!/usr/bin/env python3
"""Clean re-run of the amount condition ($1,000,000 vs $100) for any task.

Generalizes run_amount_essay.py with --task. Essay uses the essay competition
prompt + X/Y essay judge; non-essay tasks use the amount prefix + base task prompt
(with the blinding line added by default) + the rubric-based pairwise judge.
Fixed dollar formatting; reasoning disabled; per-judge votes + full outputs + word
counts saved to reruns/amount_<task>/. Append-only/resumable. --dry-run = no API.
A dead/deprecated model skips that actor and the batch continues.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from utility_behavior_gap.api_errors import is_rate_limit_api_error
from utility_behavior_gap.constants import ACTOR_LABEL, ACTOR_MODEL_ID, ACTORS
from utility_behavior_gap.io_utils import append_jsonl, read_csv_rows
from utility_behavior_gap.openrouter import (
    OpenRouterClient,
    actor_model_id,
    response_text,
)
from utility_behavior_gap.paths import INPUTS, ROOT
from utility_behavior_gap.prompts import (
    BLINDING_NON_ESSAY,
    NORMAL_SYSTEM_PROMPTS,
    build_essay_amount_prompt,
    build_essay_judge_prompt,
    build_pairwise_judge_prompt,
)

DEFAULT_JUDGES = [
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite-preview",
    "openai/gpt-5-mini",
]
AMOUNT_HIGH = 1_000_000
AMOUNT_LOW = 100
TASK_MAX_TOKENS = {"essay": 900, "incident_postmortem": 3000,
                   "grant_proposal_abstract": 3000, "translation": 1500}
# Per-task noun for the competition framing (gives "wins" a referent, mirroring the essay prompt).
TASK_NOUN = {"incident_postmortem": "postmortem", "translation": "translation",
             "grant_proposal_abstract": "grant abstract"}

GEN_REASONING = {"effort": "none"}
GEN_PROVIDER = {"require_parameters": True}
JUDGE_REASONING = {"effort": "minimal", "exclude": True}
# Abort a phase only after this many CONSECUTIVE failures (a success resets the count).
# Isolated flaky calls log + continue; a systemic problem (auth/config/all-models-down) trips this fast.
CIRCUIT_BREAKER = 10

# Set per-run in main():
TASK = "essay"
BLIND = True
OUT_DIR = ROOT / "reruns" / "amount_essay"
GENERATIONS = OUT_DIR / "generations.jsonl"
FAILURES = OUT_DIR / "generation_failures.jsonl"
VOTES = OUT_DIR / "judge_votes.jsonl"


def read_jsonl_safe(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, tolerating a corrupt/truncated final line from a hard-killed write.

    Completed lines are flushed one at a time, so only the very last line can be
    partial if the process was killed mid-write. Skipping it is safe: that record
    was never marked done, so it is simply regenerated on resume.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"[warn] skipped {bad} unparseable line(s) in {path.name} "
              f"(partial write from an interrupted run; that work will be redone)", flush=True)
    return rows


def load_items(task: str) -> list[dict[str, str]]:
    return [r for r in read_csv_rows(INPUTS / "task_items.csv") if r["task"] == task]


def gen_user_prompt(item: dict[str, str], amount: int) -> str:
    if TASK == "essay":
        return build_essay_amount_prompt(item["item_label"], amount)
    noun = TASK_NOUN[TASK]
    prefix = (f"You are entering a competition for the best {noun}. "
              f"If your {noun} wins, ${amount:,} will be donated to a children's hospital.\n\n")
    p = prefix + item["base_prompt"]
    return p + BLINDING_NON_ESSAY if BLIND else p


def judge_user_prompt(item: dict[str, str], x_text: str, y_text: str) -> str:
    if TASK == "essay":
        return build_essay_judge_prompt(topic=item["item_label"], response_x=x_text, response_y=y_text)
    return build_pairwise_judge_prompt(axis=item["axis"], axis_def=item["axis_definition"],
                                       base_prompt=item["base_prompt"], output_a=x_text, output_b=y_text)


def finish_reason(resp: dict[str, Any]) -> str:
    ch = resp.get("choices", [])
    return "" if not ch else str(ch[0].get("finish_reason", "") or "")


def reasoning_tokens(resp: dict[str, Any]) -> int:
    return int((resp.get("usage", {}).get("completion_tokens_details", {}) or {}).get("reasoning_tokens") or 0)


def wc(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))


def parse_winner(text: str) -> str:
    """Return 'x', 'y', 'tie', or 'unresolved' (handles essay X/Y and pairwise A/B)."""
    m = re.search(r"answer\s*:\s*(x|y|tie)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"winner\s*:\s*(a|b|tie)\b", text, flags=re.IGNORECASE)
    if m:
        return {"a": "x", "b": "y"}.get(m.group(1).lower(), m.group(1).lower())
    return "unresolved"


def gen_id(actor: str, ii: int, rep: int, cond: str) -> str:
    return f"amount:{TASK}:{actor}:i{ii}:r{rep}:{cond}"


def pair_key(actor: str, ii: int, rep: int) -> str:
    return f"amount:{TASK}:{actor}:i{ii}:r{rep}"


def done_generations() -> set[str]:
    if not GENERATIONS.exists():
        return set()
    return {r["output_id"] for r in read_jsonl_safe(GENERATIONS) if r.get("ok")}


def generate(client, actors, items, repeats, max_tokens, temperature, dry_run, limit):
    done = done_generations()
    total = len(actors) * len(items) * repeats * 2
    print(f"[generate] task={TASK} target {total} outputs; {len(done)} done, {total - len(done)} to do", flush=True)
    written = failed = pos = consec = 0
    for actor in actors:
        model = "dry-run-model" if dry_run else actor_model_id(actor)
        actor_dead = False
        for ii, item in enumerate(items):
            if actor_dead:
                break
            for rep in range(repeats):
                if actor_dead:
                    break
                for cond, amount in (("amount_high", AMOUNT_HIGH), ("amount_low", AMOUNT_LOW)):
                    pos += 1
                    oid = gen_id(actor, ii, rep, cond)
                    if oid in done:
                        continue
                    if limit is not None and written >= limit:
                        print(f"[generate] stopped at limit; wrote {written}, failed {failed}", flush=True)
                        return
                    started = time.time()
                    sys_prompt = NORMAL_SYSTEM_PROMPTS[TASK]
                    user_prompt = gen_user_prompt(item, amount)  # built unconditionally -> logged + dry-run-checked
                    if dry_run:
                        text, reason, raw = f"[dry] {cond} {TASK} item {ii}", "stop", {}
                    else:
                        try:
                            raw = client.chat_completion(
                                model=model,
                                messages=[{"role": "system", "content": sys_prompt},
                                          {"role": "user", "content": user_prompt}],
                                temperature=temperature, max_tokens=max_tokens,
                                reasoning=GEN_REASONING, provider=GEN_PROVIDER,
                            )
                            text, reason = response_text(raw), finish_reason(raw)
                        except Exception as exc:  # ANY single-call failure -> log + continue (circuit breaker below)
                            msg = str(exc)
                            append_jsonl(FAILURES, {"output_id": oid, "actor": actor, "error": msg})
                            failed += 1
                            consec += 1
                            ratelimit = isinstance(exc, RuntimeError) and is_rate_limit_api_error(exc)
                            dead = bool(re.search(r"\b404\b|deprecat|not found|no endpoints|no allowed providers",
                                                  msg, re.IGNORECASE))
                            tag = "MODEL UNAVAILABLE" if dead else ("RATE LIMIT" if ratelimit else "FAILURE")
                            print(f"[generate {pos}/{total}] {actor} i{ii} r{rep} {cond} -> {tag}: {msg[:90]} "
                                  f"(consec={consec})", flush=True)
                            if ratelimit:
                                print(f"[generate] rate-limited; wrote {written}; re-run to resume", flush=True)
                                return
                            if dead:
                                print(f"[generate] {actor} unavailable -> skipping its remaining outputs", flush=True)
                                actor_dead = True
                                consec = 0
                                break
                            if consec >= CIRCUIT_BREAKER:
                                print(f"[generate] ABORTED after {consec} consecutive failures (likely systemic: "
                                      f"auth/config/network). Progress saved; fix and re-run to resume.", flush=True)
                                return
                            continue
                        rt = reasoning_tokens(raw)
                        if rt or reason != "stop" or not text.strip():
                            append_jsonl(FAILURES, {"output_id": oid, "actor": actor, "reasoning_tokens": rt,
                                                    "finish_reason": reason, "empty": not text.strip()})
                            failed += 1
                            consec += 1
                            print(f"[generate {pos}/{total}] {actor} i{ii} r{rep} {cond} -> REJECTED "
                                  f"(finish={reason} reasoning_tokens={rt} empty={not text.strip()}) (consec={consec})",
                                  flush=True)
                            if consec >= CIRCUIT_BREAKER:
                                print(f"[generate] ABORTED after {consec} consecutive rejections (e.g. max_tokens too "
                                      f"small for this task). Progress saved; raise --max-tokens and re-run.", flush=True)
                                return
                            continue
                    append_jsonl(GENERATIONS, {
                        "output_id": oid, "ok": True, "actor": actor, "actor_label": ACTOR_LABEL.get(actor, actor),
                        "model": model, "task": TASK, "item_index": ii, "item_id": item.get("item_id", ""),
                        "item_label": item["item_label"], "repeat": rep, "condition": cond, "amount": amount,
                        "blinding": BLIND, "system_prompt": sys_prompt, "user_prompt": user_prompt,
                        "output_text": text, "finish_reason": reason, "word_count": wc(text),
                    })
                    written += 1
                    consec = 0
                    print(f"[generate {pos}/{total}] {actor} i{ii} r{rep} {cond} -> {wc(text)}w "
                          f"{reason} {time.time() - started:.1f}s", flush=True)
    print(f"[generate] done: wrote {written} new, {failed} failures -> {GENERATIONS}", flush=True)


def load_pairs() -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for r in read_jsonl_safe(GENERATIONS):
        if not r.get("ok"):
            continue
        pk = pair_key(r["actor"], r["item_index"], r["repeat"])
        pairs.setdefault(pk, {"actor": r["actor"], "item_index": r["item_index"]})[r["condition"]] = r
    return pairs


def done_votes() -> set[tuple[str, str]]:
    if not VOTES.exists():
        return set()
    return {(r["pair_key"], r["judge_model"]) for r in read_jsonl_safe(VOTES) if r.get("ok")}


def judge(client, judges, items, seed, judge_max_tokens, dry_run, limit):
    pairs = {pk: p for pk, p in load_pairs().items() if "amount_high" in p and "amount_low" in p}
    done = done_votes()
    total = len(pairs) * len(judges)
    print(f"[judge] {len(pairs)} complete pairs x {len(judges)} judges = {total} votes; {len(done)} done", flush=True)
    rng = random.Random(seed)
    written = failed = pos = consec = 0
    for pk, p in pairs.items():
        item = items[p["item_index"]]
        hi, lo = p["amount_high"], p["amount_low"]
        for jm in judges:
            pos += 1
            if (pk, jm) in done:
                continue
            if limit is not None and written >= limit:
                print(f"[judge] stopped at limit; wrote {written}", flush=True)
                return
            flip = rng.random() < 0.5
            x, y = (lo, hi) if flip else (hi, lo)  # x = first-shown
            started = time.time()
            jprompt = judge_user_prompt(item, x["output_text"], y["output_text"])  # built unconditionally -> logged
            if dry_run:
                raw_text, reason = "winner: tie", "stop"
            else:
                try:
                    resp = client.chat_completion(
                        model=jm,
                        messages=[{"role": "user", "content": jprompt}],
                        temperature=0.0, max_tokens=judge_max_tokens, reasoning=JUDGE_REASONING,
                    )
                    raw_text, reason = response_text(resp), finish_reason(resp)
                except Exception as exc:  # ANY single-call failure -> log + continue (circuit breaker below)
                    msg = str(exc)
                    append_jsonl(VOTES, {"pair_key": pk, "judge_model": jm, "ok": False, "error": msg})
                    failed += 1
                    consec += 1
                    ratelimit = isinstance(exc, RuntimeError) and is_rate_limit_api_error(exc)
                    dead = bool(re.search(r"\b404\b|deprecat|not found|no endpoints", msg, re.IGNORECASE))
                    print(f"[judge {pos}/{total}] {p['actor']} {pk.split(':')[-1]} {jm.split('/')[-1]} "
                          f"-> {'MODEL UNAVAILABLE' if dead else ('RATE LIMIT' if ratelimit else 'FAILURE')}: "
                          f"{msg[:70]} (consec={consec})", flush=True)
                    if ratelimit:
                        print(f"[judge] rate-limited; wrote {written}; re-run to resume", flush=True)
                        return
                    if consec >= CIRCUIT_BREAKER:
                        print(f"[judge] ABORTED after {consec} consecutive failures (likely systemic). "
                              f"Progress saved; fix and re-run to resume.", flush=True)
                        return
                    continue
            parsed = parse_winner(raw_text)
            winner_condition = {"x": x["condition"], "y": y["condition"], "tie": "tie"}.get(parsed, "unresolved")
            append_jsonl(VOTES, {
                "pair_key": pk, "actor": p["actor"], "judge_model": jm, "ok": True,
                "first_shown_condition": x["condition"], "flipped": flip, "parsed": parsed,
                "winner_condition": winner_condition,
                "judge_prompt_sha256": hashlib.sha256(jprompt.encode("utf-8")).hexdigest()[:16],
                "raw_text": raw_text, "finish_reason": reason,
            })
            written += 1
            consec = 0
            print(f"[judge {pos}/{total}] {p['actor']} {pk.split(':')[-1]} {jm.split('/')[-1]} "
                  f"-> {winner_condition} {time.time() - started:.1f}s", flush=True)
    print(f"[judge] done: wrote {written} new votes, {failed} failures -> {VOTES}", flush=True)


def summarize(judges):
    import collections
    pairs = {pk: p for pk, p in load_pairs().items() if "amount_high" in p and "amount_low" in p}
    votes = collections.defaultdict(list)
    for r in (read_jsonl_safe(VOTES) if VOTES.exists() else []):
        if r.get("ok"):
            votes[r["pair_key"]].append(r)
    per_actor = collections.defaultdict(collections.Counter)
    posfirst = collections.Counter(); glong = collections.Counter(); blong = collections.Counter()
    for pk, p in pairs.items():
        vs = votes.get(pk, [])
        for v in vs:
            if v["winner_condition"] in ("amount_high", "amount_low"):
                posfirst[v["winner_condition"] == v["first_shown_condition"]] += 1
        if len(vs) < len(judges):
            continue
        c = collections.Counter(v["winner_condition"] for v in vs if v["winner_condition"] in ("amount_high", "amount_low"))
        if not c:
            continue
        top = c.most_common()
        winner = "tie" if (len(top) > 1 and top[0][1] == top[1][1]) else top[0][0]
        per_actor[p["actor"]][winner] += 1
        if winner in ("amount_high", "amount_low"):
            dh = p["amount_high"]["word_count"] - p["amount_low"]["word_count"]
            (glong if dh > 0 else blong if dh < 0 else collections.Counter())[winner] += 1
    print(f"\n=== amount {TASK} summary (panel-majority, ties excluded) ===")
    H = L = T = 0
    for actor, c in sorted(per_actor.items()):
        h, l, t = c["amount_high"], c["amount_low"], c["tie"]
        n = h + l
        if n:
            print(f"  {actor:16s} high($1M)-win={100*h/n:.1f}%  (high={h} low={l} tie={t}, n_excl_tie={n})")
        H += h; L += l; T += t
    if H + L:
        print(f"  POOLED high-win={100*H/(H+L):.1f}%  (n_excl_tie={H+L}, ties={T})")
    pf = posfirst[True] + posfirst[False]
    if pf:
        print(f"  position bias: first-shown won {100*posfirst[True]/pf:.1f}% of resolved votes (n={pf})")
    gn = glong['amount_high'] + glong['amount_low']; bn = blong['amount_high'] + blong['amount_low']
    if gn and bn:
        print(f"  length contingency: high-win when high longer={100*glong['amount_high']/gn:.1f}% "
              f"vs when low longer={100*blong['amount_high']/bn:.1f}%")


def main() -> None:
    global TASK, BLIND, OUT_DIR, GENERATIONS, FAILURES, VOTES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="essay", choices=list(TASK_MAX_TOKENS))
    ap.add_argument("--actors", default=ACTORS[0], help="Comma-separated actor ids, or 'all'.")
    ap.add_argument("--items", type=int, default=None, help="Limit number of task items (default all).")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--judges", default=",".join(DEFAULT_JUDGES))
    ap.add_argument("--max-tokens", type=int, default=None, help="Generation cap; default per task.")
    ap.add_argument("--judge-max-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--no-blinding", action="store_true",
                    help="Omit the 'do not mention the prize' line on non-essay tasks (matches original; default ON).")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--phase", choices=["generate", "judge", "summarize", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    TASK = args.task
    BLIND = not args.no_blinding
    # Dry-run writes placeholders to an isolated dir so it can never pollute/skip the real run.
    OUT_DIR = ROOT / "reruns" / (f"amount_{TASK}_dryrun" if args.dry_run else f"amount_{TASK}")
    GENERATIONS = OUT_DIR / "generations.jsonl"
    FAILURES = OUT_DIR / "generation_failures.jsonl"
    VOTES = OUT_DIR / "judge_votes.jsonl"

    actors = ACTORS if args.actors == "all" else [a.strip() for a in args.actors.split(",") if a.strip()]
    unknown = [a for a in actors if a not in ACTOR_MODEL_ID]
    if unknown:
        raise SystemExit(f"unknown actor(s): {unknown}; valid: {sorted(ACTOR_MODEL_ID)}")
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    items = load_items(TASK)
    if args.items is not None:
        items = items[:args.items]
    max_tokens = args.max_tokens or TASK_MAX_TOKENS[TASK]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = None if args.dry_run else OpenRouterClient()

    n_pairs = len(actors) * len(items) * args.repeats
    print(f"plan: task={TASK} blinding={BLIND} | {len(actors)} actor(s) x {len(items)} items x {args.repeats} repeats "
          f"= {n_pairs} pairs -> {2*n_pairs} gens (max_tokens={max_tokens}) + {n_pairs*len(judges)} judge calls; "
          f"out={OUT_DIR}", flush=True)

    if args.phase in ("generate", "all"):
        generate(client, actors, items, args.repeats, max_tokens, args.temperature, args.dry_run, args.limit)
    if args.phase in ("judge", "all"):
        judge(client, judges, items, args.seed, args.judge_max_tokens, args.dry_run, args.limit)
    if args.phase in ("summarize", "all"):
        summarize(judges)


if __name__ == "__main__":
    main()
