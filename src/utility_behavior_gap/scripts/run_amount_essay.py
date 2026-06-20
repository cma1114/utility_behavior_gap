#!/usr/bin/env python3
"""Clean re-run of the ESSAY amount condition ($1,000,000 vs $100), self-contained.

Regenerates both arms with fixed dollar formatting, judges with a configurable
panel (default = gpt-5-mini panel, matching the non-essay scaleup), and writes
per-judge votes + full outputs + lengths to a fresh dir for auditing. Append-only
and resumable. Use --dry-run to validate the flow with zero API calls.

Phases (all run by default): generate -> judge -> summarize.
"""

from __future__ import annotations

import argparse
import random
import re
import time
from typing import Any

from utility_behavior_gap.api_errors import is_rate_limit_api_error, is_transient_api_error
from utility_behavior_gap.constants import ACTOR_LABEL, ACTOR_MODEL_ID, ACTORS
from utility_behavior_gap.io_utils import append_jsonl, read_csv_rows, read_jsonl
from utility_behavior_gap.openrouter import (
    MalformedOpenRouterResponse,
    OpenRouterClient,
    actor_model_id,
    response_text,
)
from utility_behavior_gap.paths import INPUTS, ROOT
from utility_behavior_gap.prompts import (
    NORMAL_SYSTEM_PROMPTS,
    build_essay_amount_prompt,
    build_essay_judge_prompt,
)

# Default judge panel = gpt-5-mini panel (constants.JUDGE_MODEL_IDS matches as of 2026-06-10).
DEFAULT_JUDGES = [
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite-preview",
    "openai/gpt-5-mini",
]
AMOUNT_HIGH = 1_000_000
AMOUNT_LOW = 100
ESSAY_SYSTEM = NORMAL_SYSTEM_PROMPTS["essay"]  # "You are a skilled essay writer."

OUT_DIR = ROOT / "reruns" / "amount_essay"
GENERATIONS = OUT_DIR / "generations.jsonl"
FAILURES = OUT_DIR / "generation_failures.jsonl"
VOTES = OUT_DIR / "judge_votes.jsonl"

GEN_REASONING = {"effort": "none"}
GEN_PROVIDER = {"require_parameters": True}
JUDGE_REASONING = {"effort": "minimal", "exclude": True}


def essay_topics(limit: int | None) -> list[str]:
    rows = [r["item_label"] for r in read_csv_rows(INPUTS / "task_items.csv") if r["task"] == "essay"]
    return rows if limit is None else rows[:limit]


def finish_reason(resp: dict[str, Any]) -> str:
    ch = resp.get("choices", [])
    return "" if not ch else str(ch[0].get("finish_reason", "") or "")


def reasoning_tokens(resp: dict[str, Any]) -> int:
    return int((resp.get("usage", {}).get("completion_tokens_details", {}) or {}).get("reasoning_tokens") or 0)


def wc(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))


def parse_winner(text: str) -> str:
    """Return 'x', 'y', 'tie', or 'unresolved' for the essay judge format."""
    m = re.search(r"answer\s*:\s*(x|y|tie)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"winner\s*:\s*(a|b|tie)\b", text, flags=re.IGNORECASE)
    if m:
        return {"a": "x", "b": "y"}.get(m.group(1).lower(), m.group(1).lower())
    return "unresolved"


# ---------- generation ----------

def gen_id(actor: str, ti: int, rep: int, cond: str) -> str:
    return f"amount_essay:{actor}:t{ti}:r{rep}:{cond}"


def done_generations() -> set[str]:
    if not GENERATIONS.exists():
        return set()
    return {r["output_id"] for r in read_jsonl(GENERATIONS) if r.get("ok")}


def generate(client, actors, topics, repeats, max_tokens, temperature, dry_run, limit):
    done = done_generations()
    total = len(actors) * len(topics) * repeats * 2
    print(f"[generate] target {total} outputs; {len(done)} already done, {total - len(done)} to do", flush=True)
    written = failed = pos = 0
    for actor in actors:
        model = "dry-run-model" if dry_run else actor_model_id(actor)
        actor_dead = False
        for ti, topic in enumerate(topics):
            if actor_dead:
                break
            for rep in range(repeats):
                if actor_dead:
                    break
                for cond, amount in (("amount_high", AMOUNT_HIGH), ("amount_low", AMOUNT_LOW)):
                    pos += 1
                    oid = gen_id(actor, ti, rep, cond)
                    if oid in done:
                        continue
                    if limit is not None and written >= limit:
                        print(f"[generate] stopped at limit; wrote {written}, failed {failed}", flush=True)
                        return
                    user_prompt = build_essay_amount_prompt(topic, amount)
                    started = time.time()
                    if dry_run:
                        text, reason, raw = f"[dry] {cond} essay on {topic[:30]}", "stop", {}
                    else:
                        try:
                            raw = client.chat_completion(
                                model=model,
                                messages=[{"role": "system", "content": ESSAY_SYSTEM},
                                          {"role": "user", "content": user_prompt}],
                                temperature=temperature,
                                max_tokens=max_tokens,
                                reasoning=GEN_REASONING,
                                provider=GEN_PROVIDER,
                            )
                            text, reason = response_text(raw), finish_reason(raw)
                        except (MalformedOpenRouterResponse, RuntimeError) as exc:
                            msg = str(exc)
                            transient = isinstance(exc, RuntimeError) and is_transient_api_error(exc)
                            ratelimit = isinstance(exc, RuntimeError) and is_rate_limit_api_error(exc)
                            dead = bool(re.search(r"\b404\b|deprecat|not found|no endpoints|no allowed providers",
                                                  msg, re.IGNORECASE))
                            if isinstance(exc, RuntimeError) and not transient and not ratelimit and not dead:
                                raise  # genuine config bug (400/require_parameters/etc.) -> fail loud
                            append_jsonl(FAILURES, {"output_id": oid, "actor": actor, "error": msg})
                            failed += 1
                            tag = "MODEL UNAVAILABLE" if dead else ("RATE LIMIT" if ratelimit else "API ERROR")
                            print(f"[generate {pos}/{total}] {actor} t{ti} r{rep} {cond} -> {tag}: {msg[:90]}", flush=True)
                            if ratelimit:
                                print(f"[generate] rate-limited; wrote {written}, failed {failed}; re-run to resume",
                                      flush=True)
                                return
                            if dead:
                                print(f"[generate] {actor} unavailable -> skipping its remaining outputs, continuing batch",
                                      flush=True)
                                actor_dead = True
                                break
                            continue
                        rt = reasoning_tokens(raw)
                        if rt or reason != "stop" or not text.strip():
                            append_jsonl(FAILURES, {"output_id": oid, "actor": actor,
                                                    "reasoning_tokens": rt, "finish_reason": reason,
                                                    "empty": not text.strip()})
                            failed += 1
                            print(f"[generate {pos}/{total}] {actor} t{ti} r{rep} {cond} -> REJECTED "
                                  f"(finish={reason} reasoning_tokens={rt} empty={not text.strip()})", flush=True)
                            continue
                    append_jsonl(GENERATIONS, {
                        "output_id": oid, "ok": True, "actor": actor, "actor_label": ACTOR_LABEL[actor],
                        "model": model, "topic_index": ti, "topic": topic, "repeat": rep,
                        "condition": cond, "amount": amount, "output_text": text,
                        "finish_reason": reason, "word_count": wc(text),
                    })
                    written += 1
                    print(f"[generate {pos}/{total}] {actor} t{ti} r{rep} {cond} -> {wc(text)}w "
                          f"{reason} {time.time() - started:.1f}s", flush=True)
    print(f"[generate] done: wrote {written} new, {failed} failures -> {GENERATIONS}", flush=True)


# ---------- judging ----------

def pair_key(actor: str, ti: int, rep: int) -> str:
    return f"amount_essay:{actor}:t{ti}:r{rep}"


def load_pairs() -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(GENERATIONS):
        if not r.get("ok"):
            continue
        pk = pair_key(r["actor"], r["topic_index"], r["repeat"])
        pairs.setdefault(pk, {"actor": r["actor"], "topic": r["topic"],
                              "topic_index": r["topic_index"], "repeat": r["repeat"]})[r["condition"]] = r
    return pairs


def done_votes() -> set[tuple[str, str]]:
    if not VOTES.exists():
        return set()
    return {(r["pair_key"], r["judge_model"]) for r in read_jsonl(VOTES) if r.get("ok")}


def judge(client, judges, seed, judge_max_tokens, dry_run, limit):
    pairs = load_pairs()
    complete = {pk: p for pk, p in pairs.items() if p.get("amount_high") and p.get("amount_low")}
    done = done_votes()
    total = len(complete) * len(judges)
    print(f"[judge] {len(complete)} complete pairs x {len(judges)} judges = {total} votes; "
          f"{len(done)} already done", flush=True)
    rng = random.Random(seed)
    written = failed = pos = 0
    for pk, p in complete.items():
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
            if dry_run:
                raw_text, reason, ok = "Answer: TIE", "stop", True
            else:
                try:
                    resp = client.chat_completion(
                        model=jm,
                        messages=[{"role": "user", "content": build_essay_judge_prompt(
                            topic=p["topic"], response_x=x["output_text"], response_y=y["output_text"])}],
                        temperature=0.0, max_tokens=judge_max_tokens, reasoning=JUDGE_REASONING,
                    )
                    raw_text, reason, ok = response_text(resp), finish_reason(resp), True
                except (MalformedOpenRouterResponse, RuntimeError) as exc:
                    if isinstance(exc, RuntimeError) and not is_transient_api_error(exc):
                        raise
                    append_jsonl(VOTES, {"pair_key": pk, "judge_model": jm, "ok": False, "error": str(exc)})
                    failed += 1
                    print(f"[judge {pos}/{total}] {p['actor']} {pk.split(':')[-1]} {jm.split('/')[-1]} "
                          f"-> API ERROR: {str(exc)[:70]}", flush=True)
                    if isinstance(exc, RuntimeError) and is_rate_limit_api_error(exc):
                        print(f"[judge] rate-limited; wrote {written}; re-run to resume", flush=True)
                        return
                    continue
            parsed = parse_winner(raw_text)
            winner_condition = {"x": x["condition"], "y": y["condition"], "tie": "tie"}.get(parsed, "unresolved")
            append_jsonl(VOTES, {
                "pair_key": pk, "actor": p["actor"], "judge_model": jm, "ok": ok,
                "first_shown_condition": x["condition"], "flipped": flip,
                "parsed": parsed, "winner_condition": winner_condition,
                "raw_text": raw_text, "finish_reason": reason,
            })
            written += 1
            print(f"[judge {pos}/{total}] {p['actor']} {pk.split(':')[-1]} {jm.split('/')[-1]} "
                  f"-> {winner_condition} {time.time() - started:.1f}s", flush=True)
    print(f"[judge] done: wrote {written} new votes, {failed} failures -> {VOTES}", flush=True)


# ---------- summarize ----------

def summarize(judges):
    import collections
    pairs = load_pairs()
    votes = collections.defaultdict(list)
    for r in read_jsonl(VOTES) if VOTES.exists() else []:
        if r.get("ok"):
            votes[r["pair_key"]].append(r)
    per_actor = collections.defaultdict(lambda: collections.Counter())
    posfirst = collections.Counter()
    glong = collections.Counter(); blong = collections.Counter()
    for pk, p in pairs.items():
        vs = votes.get(pk, [])
        for v in vs:
            if v["winner_condition"] in ("amount_high", "amount_low"):
                posfirst[v["winner_condition"] == v["first_shown_condition"]] += 1  # did first-shown win
        conds = [v["winner_condition"] for v in vs if v["winner_condition"] in ("amount_high", "amount_low")]
        if len(vs) < len(judges):
            continue
        c = collections.Counter(conds)
        if not c:
            continue
        top = c.most_common()
        winner = "tie" if (len(top) > 1 and top[0][1] == top[1][1]) else top[0][0]
        per_actor[p["actor"]][winner] += 1
        if winner in ("amount_high", "amount_low") and p.get("amount_high") and p.get("amount_low"):
            dh = p["amount_high"]["word_count"] - p["amount_low"]["word_count"]
            (glong if dh > 0 else blong if dh < 0 else collections.Counter())[winner] += 1
    print("\n=== amount-essay summary (panel-majority, ties excluded) ===")
    H = L = T = 0
    for actor, c in sorted(per_actor.items()):
        h, l, t = c["amount_high"], c["amount_low"], c["tie"]
        n = h + l
        print(f"  {actor:16s} high($1M)-win={100*h/n:.1f}%  (high={h} low={l} tie={t}, n_excl_tie={n})")
        H += h; L += l; T += t
    if H + L:
        print(f"  POOLED high-win={100*H/(H+L):.1f}%  (n_excl_tie={H+L}, ties={T})")
    pf = posfirst[True] + posfirst[False]
    if pf:
        print(f"  position bias: first-shown essay won {100*posfirst[True]/pf:.1f}% of votes (n={pf})")
    if glong['amount_high'] + glong['amount_low'] and blong['amount_high'] + blong['amount_low']:
        gw = glong['amount_high'] / (glong['amount_high'] + glong['amount_low'])
        bw = blong['amount_high'] / (blong['amount_high'] + blong['amount_low'])
        print(f"  length contingency: high-win when high longer={100*gw:.1f}% vs when low longer={100*bw:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actors", default=ACTORS[0],
                    help=f"Comma-separated actor ids (default 1-actor pilot: {ACTORS[0]}). 'all' = {len(ACTORS)} actors.")
    ap.add_argument("--topics", type=int, default=None, help="Limit number of essay topics (default all 30).")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--judges", default=",".join(DEFAULT_JUDGES))
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--judge-max-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=None, help="Generation temp; omit for provider default.")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--limit", type=int, default=None, help="Cap new API calls per phase (smoke testing).")
    ap.add_argument("--phase", choices=["generate", "judge", "summarize", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="No API calls; placeholder outputs/votes.")
    args = ap.parse_args()

    actors = ACTORS if args.actors == "all" else [a.strip() for a in args.actors.split(",") if a.strip()]
    unknown = [a for a in actors if a not in ACTOR_MODEL_ID]
    if unknown:
        raise SystemExit(f"unknown actor(s): {unknown}; valid: {sorted(ACTOR_MODEL_ID)}")
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    topics = essay_topics(args.topics)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = None if args.dry_run else OpenRouterClient()

    n_pairs = len(actors) * len(topics) * args.repeats
    print(f"plan: {len(actors)} actor(s) x {len(topics)} topics x {args.repeats} repeats = {n_pairs} pairs "
          f"-> {2*n_pairs} generations + {n_pairs*len(judges)} judge calls; judges={judges}; out={OUT_DIR}")

    if args.phase in ("generate", "all"):
        generate(client, actors, topics, args.repeats, args.max_tokens, args.temperature, args.dry_run, args.limit)
    if args.phase in ("judge", "all"):
        judge(client, judges, args.seed, args.judge_max_tokens, args.dry_run, args.limit)
    if args.phase in ("summarize", "all"):
        summarize(judges)


if __name__ == "__main__":
    main()
