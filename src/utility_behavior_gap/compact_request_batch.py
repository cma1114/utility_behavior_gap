"""Resumable bounded API requests; compact append-only evidence, no generation logic."""
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import gzip
import heapq
import json
from pathlib import Path
import shutil
import time
import urllib.request

from utility_behavior_gap.openrouter import OpenRouterClient


def rows(path):
    if not path.exists():
        return []
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed(path):
    result = {}
    for row in rows(path):
        if row["success"]:
            if row["job_id"] in result:
                raise ValueError("Duplicate successful request in immutable response history")
            result[row["job_id"]] = row
    return result


def append(path, row):
    # Each append is a complete gzip member, readable after an interrupted run.
    data = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    with path.open("ab") as handle:
        handle.write(gzip.compress(data, compresslevel=6, mtime=0))
        handle.flush()


def cost(row):
    return float((row.get("raw_response") or {}).get("usage", {}).get("cost") or 0.)


def account_usage():
    client = OpenRouterClient(max_retries=1)
    request = urllib.request.Request(client.base_url + "/credits", headers={"Authorization":"Bearer "+client.api_key})
    with urllib.request.urlopen(request, timeout=30) as handle:
        data = json.load(handle)["data"]
    return float(data["total_usage"]), float(data["total_credits"])-float(data["total_usage"])


def reservation(job):
    prices = {"anthropic/claude-haiku-4.5": (1.,5.), "openai/gpt-5-mini":(.25,2.),
              "google/gemini-3.1-flash-lite-preview":(.25,1.5)}
    payload = job["request"]
    inp, out = prices[payload["model"]]
    # Conservative English-text token estimate and 2x price margin.
    return 2*(sum(len(m["content"]) for m in payload["messages"])/2*inp + payload["max_tokens"]*out)/1e6


def perform(job, parse, client_factory=OpenRouterClient, provider=None):
    start, response = time.monotonic(), {}
    payload = dict(job["request"])
    if provider is not None:
        payload["provider"] = provider
    try:
        response = client_factory(timeout_s=90, max_retries=1).chat_completion(**payload)
        parsed = parse(job, response)
        success, error = True, ""
    except Exception as exc:
        response = getattr(exc, "response", response)
        parsed, success, error = {}, False, str(exc)
    if not isinstance(response, dict):
        response = {"unexpected_response":response}
    err = response.get("error") or {}
    code = err.get("code") if isinstance(err,dict) else None
    fatal = not success and (code in {400,401,402,403,404} or any(f"({c})" in error for c in (400,401,402,403,404)))
    return dict(job_id=job["job_id"], success=success, parsed=parsed, error=error, fatal=fatal,
                request_overrides={} if provider is None else {"provider": provider},
                raw_response=response, timestamp_unix=time.time(), latency_s=round(time.monotonic()-start,3))


def status(directory, *, stage="judge"):
    jobs = rows(directory / f"{stage}_requests.jsonl.gz")
    history = rows(directory / f"{stage}_responses.jsonl.gz")
    done = {r["job_id"] for r in history if r["success"]}
    return dict(complete=len(done), target=len(jobs), attempts=len(history),
                recorded_cost_usd=round(sum(cost(r) for r in history),4),
                by_contrast={c:dict(complete=sum(j["job_id"] in done for j in jobs if j["contrast"]==c),
                                   target=sum(j["contrast"]==c for j in jobs)) for c in sorted({j["contrast"] for j in jobs})})


def run(directory, jobs, parse, *, workers=24, budget=15., limit=None, provider=None, stage="judge"):
    if not 1 <= workers <= 32 or budget <= 0:
        raise ValueError("Invalid concurrency or budget")
    if stage not in {"judge", "screen"}:
        raise ValueError("Unsupported request stage")
    directory = Path(directory)
    with (directory / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(directory, jobs, parse, workers=workers, budget=budget, limit=limit, provider=provider, stage=stage)


def _run(directory, jobs, parse, *, workers, budget, limit, provider, stage):
    path = directory / f"{stage}_responses.jsonl.gz"
    done = completed(path)
    pending = [j for j in jobs if j["job_id"] not in done]
    if limit is not None:
        first = {}
        for job in pending:
            first.setdefault((job["contrast"],job["actor"],job.get("task"),job["request"]["model"]),job)
        ids = {j["job_id"] for j in first.values()}
        pending = (list(first.values())+[j for j in pending if j["job_id"] not in ids])[:limit]
    if not pending:
        print(f"All {stage} requests already complete", flush=True)
        return
    usage, credit = account_usage()
    guard_path = directory / "spend_guard.json"
    if not guard_path.exists():
        guard_path.write_text(json.dumps(dict(initial_usage=usage, budget_usd=budget),indent=2)+"\n")
    guard = json.loads(guard_path.read_text())
    budget = min(budget, guard["budget_usd"])
    history = rows(path)
    actual = sum(cost(r) for r in history)
    charged = max(actual, usage-guard["initial_usage"])
    account_at, actual_at = charged, actual
    if charged >= budget or credit < .5:
        raise RuntimeError("Budget or credit guard reached")
    attempts = Counter(r["job_id"] for r in history)
    queue = [(0.,i,j) for i,j in enumerate(pending)]
    heapq.heapify(queue)
    sequence, new = len(queue), 0
    inflight, stopped, exhausted = {}, "", []
    cooldown, last_print = 0., time.monotonic()
    print(f"{stage}: {len(pending)} pending calls, {workers} workers, ${budget:.2f} cap", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while queue or inflight:
            now = time.monotonic()
            if shutil.disk_usage(directory).free < 250*1024**2:
                stopped = "Disk guard reached; evidence retained"
            if new > 1.2*len(pending)+50:
                stopped = "Retry ceiling reached; resumable"
            charged = max(actual, account_at+actual-actual_at)
            if not stopped and now >= cooldown:
                while queue and queue[0][0] <= now and len(inflight)<workers:
                    job = queue[0][2]
                    reserved = sum(reservation(j) for j in inflight.values())
                    if charged+reserved+reservation(job) >= budget:
                        stopped = "Budget guard reached; unfinished requests retained"
                        break
                    heapq.heappop(queue)
                    attempts[job["job_id"]] += 1
                    inflight[pool.submit(perform, job, parse, provider=provider)] = job
            if stopped and not inflight:
                break
            if not inflight:
                time.sleep(.2)
                continue
            ready, _ = wait(inflight, timeout=1, return_when=FIRST_COMPLETED)
            for future in ready:
                job = inflight.pop(future)
                row = future.result()
                row["attempt"] = attempts[job["job_id"]]
                append(path,row)
                new += 1
                actual += cost(row)
                if row["success"]:
                    done[job["job_id"]] = row
                elif row["fatal"]:
                    stopped = row["error"]
                elif row["attempt"] >= 10:
                    exhausted.append(job["job_id"])
                else:
                    delay = min(30,2**min(5,row["attempt"]))
                    heapq.heappush(queue,(time.monotonic()+delay,sequence,job))
                    sequence += 1
                    if "429" in row["error"]:
                        cooldown = time.monotonic()+30
                if new % 250 == 0:
                    try:
                        usage, credit = account_usage()
                        account_at, actual_at = max(actual,usage-guard["initial_usage"]), actual
                        if credit < .5:
                            stopped = "Credit guard reached"
                    except Exception as exc:
                        stopped = "Cannot verify spending: "+str(exc)
            if time.monotonic()-last_print > 30 or not(queue or inflight):
                state = dict(complete=len(done),target=len(jobs),attempts=len(history)+new,recorded_cost_usd=round(actual,4),updated_unix=time.time())
                (directory / "status.json").write_text(json.dumps(state,indent=2)+"\n")
                print(f"{stage}: {len(done)}/{len(jobs)} complete; ${actual:.2f}",flush=True)
                last_print = time.monotonic()
    if stopped or exhausted:
        raise RuntimeError(stopped or f"{len(exhausted)} requests exhausted retries; rerun to resume")
