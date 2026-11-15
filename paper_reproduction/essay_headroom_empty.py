"""Essay-only bridges from saved effort/role treatments to framed empty.

Preparation reads the research archive once. Subsequent judging and analysis
use only the compact, immutable evidence in this public repository.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utility_behavior_gap.constants import ACTORS, ACTOR_LABEL
from utility_behavior_gap.io_utils import load_env_file
from utility_behavior_gap.judging import derive_panel_winner_condition
from utility_behavior_gap.openrouter import response_text
from utility_behavior_gap.scripts.run_judging import judge_prompt, parse_winner, JUDGE_REASONING

RUN = ROOT / "data/processed/essay_headroom_empty"
OUT = ROOT / "docs/reviewer_response/essay_headroom_empty"
JUDGES = {"anthropic/claude-haiku-4.5", "google/gemini-3.1-flash-lite-preview", "openai/gpt-5-mini"}
INPUTS = {
    "effort": "data/processed/framed_user_strong_panel_pairs.csv.gz",
    "effort_metadata": "data/processed/final_text_analysis_pair_deltas.csv.gz",
    "role": "CURRENT_PAPER/results/framed_user_prompt_role_pair_outcomes.csv",
    "role_metadata": "data/processed/framed_user_prompt_role_pair_deltas.csv.gz",
    "empty": "CURRENT_PAPER/results/high_utility_vs_framed_empty_pair_outcomes.csv",
}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_rows(path):
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def immutable(path, rows):
    if path.exists():
        if read_rows(path) != rows:
            raise ValueError(f"Immutable evidence changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def select_pairs():
    empty = pd.read_csv(ROOT / INPUTS["empty"])
    empty = empty[empty.task.eq("essay")].copy()
    keys = ["actor", "item_label", "repeat"]
    assert len(empty) == 2100 and not empty.duplicated(keys).any()
    direct = pd.read_csv(ROOT / INPUTS["effort"])
    direct = direct[direct.task.eq("essay")].copy()
    md = pd.read_csv(ROOT / INPUTS["effort_metadata"], usecols=["pair_uid", "high_output_id", "repeat", "contrast"])
    md = md[md.contrast.eq("direct_instruction")]
    direct = direct.merge(md, on="pair_uid", validate="one_to_one")
    direct["source_repeat"] = direct["repeat"]
    direct["repeat_block"] = direct.pair_uid.str.extract(r":block([^:]+):v", expand=False).fillna("")
    assert set(direct.repeat_block) == {"", "block01"}
    assert direct.repeat.between(0,4).all()
    direct["repeat"] = direct.repeat + 5*direct.repeat_block.eq("block01")
    role = pd.read_csv(ROOT / INPUTS["role"])
    role = role[role.task.eq("essay")].copy()
    md = pd.read_csv(ROOT / INPUTS["role_metadata"], usecols=["pair_uid", "item_label", "high_output_id"])
    role = role.merge(md, on="pair_uid", how="left", validate="one_to_one")
    role["item_index"] = role.pair_uid.str.extract(r":essay:(\d+):v", expand=False).astype(int)
    known = role.dropna(subset=["item_label"])
    assert known.groupby("item_index").item_label.nunique().eq(1).all()
    topic_map = known.drop_duplicates("item_index").set_index("item_index").item_label
    role["item_label"] = role.item_label.fillna(role.item_index.map(topic_map))
    role["high_output_id"] = role.high_output_id.fillna(role.pair_uid+"::a")
    role["repeat"] = role.pair_uid.str.extract(r":r(\d+)$", expand=False).astype(int)
    role["source_repeat"] = role["repeat"]
    role["repeat_block"] = ""
    result = []
    for contrast, frame in (("effort", direct), ("role", role)):
        assert len(frame) == 2100 and not frame.duplicated(keys).any()
        matched = frame.merge(empty[keys + ["source_framed_empty_output_id", "bridge_pair_uid"]],
                              on=keys, how="left", validate="one_to_one")
        assert matched.source_framed_empty_output_id.notna().all()
        assert matched.groupby("actor").size().to_dict() == {a: 300 for a in ACTORS}
        for row in matched.sort_values(keys).to_dict("records"):
            left, right = row["high_output_id"], row["source_framed_empty_output_id"]
            result.append(dict(contrast=contrast, actor=row["actor"], task="essay", item_label=row["item_label"],
                repeat=int(row["repeat"]), left_output_id=left, right_output_id=right,
                source_repeat=int(row["source_repeat"]), source_repeat_block=row["repeat_block"],
                source_pair_uid=row["pair_uid"], source_run=Path(row["run_dir"]).name,
                reference_pair_uid=row["bridge_pair_uid"],
                pair_uid="essay_"+contrast+"_empty:"+sha(left+"\n"+right)[:20]))
    return result, empty


def compact_source(row, output_id, *, prompt, member, category, pair):
    job = row["job"]
    return dict(output_id=output_id, output_hash=sha(row["output_text"]), text=row["output_text"],
        actor=pair["actor"], task="essay", item_label=pair["item_label"], repeat=int(pair["repeat"]),
        actor_prompt=prompt, system_prompt="", category=category,
        finish_reason=row.get("finish_reason"), success=row.get("success", True),
        actor_request=row.get("request"), actor_usage=row.get("usage"), model=row.get("model"),
        source_member=member, original_condition_a=job.get("condition_a"), original_condition_b=job.get("condition_b"))


def put_source(sources, source):
    old = sources.get(source["output_id"])
    if old and old["output_hash"] != source["output_hash"]:
        raise ValueError("Conflicting source outputs; cannot choose by old judgment")
    sources[source["output_id"]] = source


def prepare(archive):
    pairs, empty = select_pairs()
    snapshot = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in INPUTS.values()}
    snapshot_path = RUN / "input_snapshot.json"
    if snapshot_path.exists():
        if json.loads(snapshot_path.read_text())["inputs"] != snapshot:
            raise ValueError("Pinned source tables changed")
        return read_rows(RUN / "pairs.jsonl.gz"), {r["output_id"]:r for r in read_rows(RUN / "sources.jsonl.gz")}
    if archive is None:
        raise ValueError("Initial prepare requires --archive; analyze/judge do not")
    wanted = {(p["source_run"], p["left_output_id"]):p for p in pairs}
    run_names = {k[0] for k in wanted}
    reference = empty.set_index("bridge_pair_uid").to_dict("index")
    sources, high_texts, jobs, votes, audit, original_jobs = {}, {}, {}, defaultdict(dict), {}, {}
    treatment_rows = {}
    start = last = time.monotonic()
    print("Streaming archive once; retaining essay treatments and matched controls only", flush=True)
    with tarfile.open(archive, "r|gz") as tf:
        for member in tf:
            if time.monotonic()-last > 30:
                print(f"archive: {len(sources)}/6300 sources; {len(votes)}/2100 reference panels; {int(time.monotonic()-start)}s", flush=True)
                last = time.monotonic()
            parts = Path(member.name).parts
            if len(parts) < 3:
                continue
            folder, filename = parts[-2:]
            bridge = folder.startswith("highlow-framed-empty-bridge__")
            if not ((folder in run_names and filename in {"generations.jsonl", "generation_jobs.jsonl"}) or
                    (bridge and filename in {"generations.jsonl", "judge_votes.jsonl"})):
                continue
            digest, kept = hashlib.sha256(), 0
            for line in tf.extractfile(member):
                digest.update(line)
                row = json.loads(line)
                oid = row.get("output_id")
                if filename == "generation_jobs.jsonl" and (folder, row.get("pair_uid", "")+"::a") in wanted:
                    original_jobs[row["pair_uid"]+"::a"] = row
                if filename == "generations.jsonl" and (folder, oid) in wanted:
                    if oid in treatment_rows and treatment_rows[oid][0]["output_text"] != row["output_text"]:
                        raise ValueError("Conflicting treatment generations")
                    if "job" in row:
                        original_jobs.setdefault(oid,row["job"])
                    treatment_rows[oid] = ({k:row[k] for k in ("output_text","finish_reason","success","model","request","usage") if k in row},member.name)
                    kept += 1
                uid = row.get("pair_uid")
                if not bridge or uid not in reference:
                    continue
                ref = reference[uid]
                jobs[uid] = row["job"]
                if filename == "generations.jsonl":
                    if row["source_output_id"] == ref["source_framed_empty_output_id"]:
                        job = row["job"]
                        assert not job.get("system_prompt_b") and "sponsor" not in job["prompt_b"].lower()
                        put_source(sources, compact_source(row, row["source_output_id"], prompt=job["prompt_b"],
                            member=member.name, category="framed_empty", pair=ref))
                    elif row["source_output_id"] == ref["source_highlow_output_id"]:
                        high_texts[uid] = row["output_text"]
                    else:
                        raise ValueError("Unrecognized reference source")
                elif row.get("success") and row.get("winner_condition") != "unresolved":
                    model = row["judge_model"]
                    if model in votes[uid]:
                        raise ValueError("Multiple successful orders in supposedly single-order reference")
                    votes[uid][model] = row
                kept += 1
            audit[member.name] = dict(sha256=digest.hexdigest(), selected_rows=kept)
    for (_, oid), pair in wanted.items():
        if oid in treatment_rows:
            row, member = treatment_rows[oid]
            job = original_jobs[oid]
            expected = "framed_user_strong" if pair["contrast"] == "effort" else "user_strong"
            assert oid.endswith("::a") and job["condition_a"] == expected
            assert job["task"] == "essay" and job["actor"] == pair["actor"]
            assert job["item_label"] == pair["item_label"] and int(job["repeat"]) == pair["source_repeat"]
            assert str(job.get("repeat_block", "")) == pair["source_repeat_block"]
            assert not job.get("system_prompt_a")
            prompt = job["prompt_a"]
            assert "sponsor will fund the following intervention" in prompt
            assert "A healthcare intervention at a children's hospital." in prompt
            cue = "Your goal is to maximize the final essay's quality" if pair["contrast"] == "effort" else "You are a world-class essayist."
            assert cue in prompt
            row["job"] = job
            put_source(sources, compact_source(row,oid,prompt=prompt,member=member,category=pair["contrast"],pair=pair))
        if oid not in sources and oid in original_jobs:
            job = original_jobs[oid]
            assert job["item_label"] == pair["item_label"] and int(job["repeat"]) == pair["source_repeat"]
            assert not job.get("system_prompt_a")
            put_source(sources, compact_source(dict(job=job, output_text="", success=False), oid,
                prompt=job["prompt_a"], member=pair["source_run"]+"/generation_jobs.jsonl (no generation)",
                category=pair["contrast"], pair=pair))
    expected_ids = {p[k] for p in pairs for k in ("left_output_id", "right_output_id")}
    if set(sources) != expected_ids:
        raise ValueError(f"Missing {len(expected_ids-set(sources))} source outputs: {sorted(expected_ids-set(sources))[:5]}")
    for uid, ref in reference.items():
        assert set(votes[uid]) == JUDGES
        right = sources[ref["source_framed_empty_output_id"]]
        high = high_texts[uid]
        for model, vote in votes[uid].items():
            x, y = (right["text"], high) if vote["flipped"] else (high, right["text"])
            assert judge_prompt(jobs[uid], {"output_text":x}, {"output_text":y}) == vote["request"]["messages"][-1]["content"]
            assert vote["source_output_a_hash"] == sha(high) and vote["source_output_b_hash"] == right["output_hash"]
            assert vote["temperature"] == 0 and vote["max_tokens"] == 120 and vote["reasoning"] == JUDGE_REASONING
        assert derive_panel_winner_condition({}, [v["winner_condition"] for v in votes[uid].values()]) == ref["panel_winner_condition"]
    flags = []
    for source in sources.values():
        text = source["text"]
        source["mechanically_invalid"] = source["finish_reason"] != "stop" or source["success"] is False or not text.strip()
        words = re.findall(r"\b\w+\b", text)
        reasons = []
        if source["mechanically_invalid"]:
            reasons.append("completion")
        if len(words) < 100 or len(words) > 650:
            reasons.append("unusual length")
        if len(set(w.lower() for w in words))/max(1,len(words)) < .15:
            reasons.append("repetition")
        if re.search(r"(?i)(as an ai|i (?:cannot|can't|won't|must refuse)|unable to (?:write|comply)|cannot (?:write|assist|fulfill)|instead of (?:writing|an essay))", text):
            reasons.append("possible refusal/meta")
        if reasons:
            flags.append(dict(output_id=source["output_id"], output_hash=source["output_hash"], reasons=reasons, text=text, actor_prompt=source["actor_prompt"]))
    for pair in pairs:
        left, right = sources[pair["left_output_id"]], sources[pair["right_output_id"]]
        for key in ("actor", "task", "item_label", "repeat"):
            assert left[key] == right[key] == pair[key]
        pair["left_hash"], pair["right_hash"] = left["output_hash"], right["output_hash"]
        pair["judge_profiles"] = [{k:vote[k] for k in ("judge_model", "flipped", "temperature", "max_tokens", "reasoning")}
                                  for model, vote in sorted(votes[pair["reference_pair_uid"]].items())]
    immutable(RUN / "pairs.jsonl.gz", pairs)
    immutable(RUN / "sources.jsonl.gz", sorted(sources.values(), key=lambda x:x["output_id"]))
    immutable(RUN / "reference_audit.jsonl.gz", [dict(pair_uid=uid, profiles=sorted(votes[uid]),
        recorded_panel=ref["panel_winner_condition"], prompt_hashes={m:sha(v["request"]["messages"][-1]["content"]) for m,v in votes[uid].items()}) for uid,ref in reference.items()])
    (RUN / "flagged_review.json").write_text(json.dumps(flags, indent=2)+"\n")
    snapshot_path.write_text(json.dumps(dict(inputs=snapshot, archive_bytes=archive.stat().st_size,
        archive_mtime_ns=archive.stat().st_mtime_ns, members=audit), indent=2)+"\n")
    print(f"Prepared {len(pairs)} pairs, {len(sources)} unique outputs; {len(flags)} outputs flagged for manual inspection", flush=True)
    return pairs, sources


def build_requests(pairs, sources):
    review = json.loads((RUN / "manual_review.json").read_text())
    flags = json.loads((RUN / "flagged_review.json").read_text())
    assert review["reviewed"] == {f["output_id"]: f["output_hash"] for f in flags}
    exclusions = review.get("exclusions", {})
    assert set(exclusions).issubset(sources)
    requests, accounting = [], []
    for pair in pairs:
        left, right = sources[pair["left_output_id"]], sources[pair["right_output_id"]]
        blocks = [re.search(r"Write a compelling essay.*?(?=\n\nDo not mention)", s["actor_prompt"], re.S) for s in (left,right)]
        assert all(blocks) and blocks[0].group() == blocks[1].group(), "Underlying task instructions differ"
        reasons = [oid for oid in (left["output_id"], right["output_id"]) if sources[oid]["mechanically_invalid"] or oid in exclusions]
        accounting.append(pair | dict(excluded=bool(reasons), excluded_output_ids=reasons))
        if reasons:
            continue
        for profile in pair["judge_profiles"]:
            x,y = (right,left) if profile["flipped"] else (left,right)
            prompt = judge_prompt(pair, {"output_text":x["text"]}, {"output_text":y["text"]})
            payload = dict(model=profile["judge_model"], temperature=profile["temperature"], max_tokens=profile["max_tokens"],
                           reasoning=profile["reasoning"], messages=[dict(role="user", content=prompt)])
            requests.append(dict(job_id=sha(pair["pair_uid"]+json.dumps(payload,sort_keys=True)), pair_uid=pair["pair_uid"],
                contrast=pair["contrast"], actor=pair["actor"], task="essay", flipped=profile["flipped"], request=payload))
    immutable(RUN / "judge_requests.jsonl.gz", requests)
    immutable(RUN / "pair_accounting.jsonl.gz", accounting)
    return requests, accounting


def parse(job, response):
    winner = parse_winner(response_text(response))
    if winner == "unresolved":
        raise ValueError("Unparseable judge verdict")
    return dict(winner_side="tie" if winner == "tie" else ("right" if (winner == "a") == job["flipped"] else "left"))


def analyze(requests, accounting, out=OUT, *, allow_unresolved=False):
    from utility_behavior_gap import compact_request_batch as batch
    from utility_behavior_gap.scripts import plot_canonical_highn_highlow as plotting
    from utility_behavior_gap.scripts.consolidate_aggregate_cis import bootstrap_equal_cell
    import matplotlib.pyplot as plt
    done = batch.completed(RUN / "judge_responses.jsonl.gz")
    expected = {r["job_id"] for r in requests}
    if set(done)-expected or (expected-set(done) and not allow_unresolved):
        raise ValueError("Incomplete or mismatched judgments; no partial result publication")
    panels = defaultdict(dict)
    for request in requests:
        if request["job_id"] not in done:
            continue
        row = done[request["job_id"]]
        verdict = parse(request, row["raw_response"])
        assert verdict == row["parsed"]
        panel = panels[request["pair_uid"]]
        assert request["request"]["model"] not in panel
        panel[request["request"]["model"]] = verdict["winner_side"]
    records, unresolved, accounting_rows = [], [], []
    for pair in accounting:
        status = "source_excluded" if pair["excluded"] else "complete" if set(panels[pair["pair_uid"]]) == JUDGES else "unresolved"
        accounting_rows.append({k:v for k,v in pair.items() if k != "judge_profiles"} |
                               dict(panel_status=status, completed_judges=len(panels[pair["pair_uid"]])))
        if pair["excluded"]:
            continue
        if status == "unresolved":
            unresolved.append(pair["pair_uid"])
            continue
        assert set(panels[pair["pair_uid"]]) == JUDGES
        winner = derive_panel_winner_condition({}, panels[pair["pair_uid"]].values())
        records.append({k:v for k,v in pair.items() if k != "judge_profiles"} |
            dict(panel_winner=winner, target_win=winner=="left", target_loss=winner=="right", tie=winner=="tie"))
    frame = pd.DataFrame(records)
    aggregates, cells = [], []
    for contrast, group in frame.groupby("contrast", sort=True):
        for metric in ("win_rate", "half_win_score"):
            sub = group[~group.tie].copy() if metric == "win_rate" else group.copy()
            sub["score"] = sub.target_win.astype(float) + (.5*sub.tie if metric == "half_win_score" else 0)
            arrays = [g.score.to_numpy() for _,g in sub.groupby("actor", sort=True)]
            assert len(arrays) == 7
            point, lo, hi, n = bootstrap_equal_cell(arrays, seed=20260615, iterations=5000)
            aggregates.append(dict(contrast=contrast, metric=metric, mean=point, ci_low=lo, ci_high=hi,
                cells=n, pairs=len(group), wins=int(group.target_win.sum()), losses=int(group.target_loss.sum()), ties=int(group.tie.sum())))
        for actor, part in group.groupby("actor", sort=True):
            wins, losses, ties = (int(part[c].sum()) for c in ("target_win", "target_loss", "tie"))
            n = wins+losses
            lo, hi = plotting.exact_familywise_ci(wins,n,14)
            cells.append(dict(contrast=contrast, actor=actor, actor_label=ACTOR_LABEL[actor], task="essay",
                resolved_n=n, target_wins=wins, target_losses=losses, ties=ties,
                target_win_rate_excluding_ties=wins/n if n else float("nan"), familywise_ci_lo=lo,
                familywise_ci_hi=hi, familywise_ci_positive=bool(lo>.5), familywise_ci_negative=bool(hi<.5), family_size=14))
    agg, cell = pd.DataFrame(aggregates), pd.DataFrame(cells)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "pair_outcomes.csv",index=False)
    agg.to_csv(out / "aggregate_results.csv",index=False)
    cell.to_csv(out / "model_comparison_cells.csv",index=False)
    pd.DataFrame(accounting_rows).to_csv(out / "pair_accounting.csv",index=False)
    plt.rcParams.update({"font.family":"DejaVu Sans"})
    fig = plt.figure(figsize=(13.8,4.7),facecolor=plotting.PAPER_BG)
    grid = fig.add_gridspec(1,2,top=.76,bottom=.15,left=.12,right=.98,wspace=.60)
    labels = {"effort":"Strong effort", "role":"World-class role"}
    for i, contrast in enumerate(("effort","role")):
        ax = fig.add_subplot(grid[0,i])
        plotting.render_panel(ax,cell[cell.contrast.eq(contrast)],labels[contrast],"AB"[i],show_xlabel=True,
            x_label=labels[contrast]+" vs framed empty: win rate")
    fig.savefig(out / "essay_effort_role_vs_framed_empty.png", dpi=360, bbox_inches="tight",facecolor="white")
    plt.close(fig)
    history = batch.rows(RUN / "judge_responses.jsonl.gz")
    audit = dict(planned_pairs=len(accounting), analyzed_pairs=len(frame), excluded_pairs=sum(p["excluded"] for p in accounting),
        unresolved_pairs=unresolved, missing_judge_requests=sorted(expected-set(done)),
        unique_source_outputs=len(read_rows(RUN / "sources.jsonl.gz")), planned_judge_calls=len(requests),
        successful_judge_calls=len(done), attempts=len(history),
        routing_recovery_attempts=sum(bool(r.get("request_overrides")) for r in history),
        recorded_cost_usd=sum(batch.cost(r) for r in history), generation_calls=0, classification_calls=0,
        finish_reasons=dict(Counter((r["raw_response"].get("choices") or [{}])[0].get("finish_reason") for r in history if r["success"])),
        exclusions=json.loads((RUN / "manual_review.json").read_text()))
    (out / "run_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    interval = lambda r: f"{r['mean']:.1%} [{r['ci_low']:.1%}, {r['ci_high']:.1%}]"
    lines = ["# Essay Headroom Above Framed Empty", "", "Judging-only follow-up, all seven paper actors; no new generations. Incomplete panels are explicitly accounted for below, never counted as ties.", "",
        "| Comparison | Pairs | Wins / losses / ties | Equal-model win rate [95% CI] |", "| --- | ---: | ---: | ---: |"]
    for r in agg[agg.metric.eq("win_rate")].to_dict("records"):
        lines.append(f"| {labels[r['contrast']]} vs framed empty | {r['pairs']} | {r['wins']} / {r['losses']} / {r['ties']} | {interval(r)} |")
    lines += ["", "The aggregate is the equal-weight mean of seven actor win rates, excluding panel ties; confidence intervals use the paper's cell-then-pair bootstrap (5,000 draws, seed 20260615).", "",
        "![Essay headroom](essay_effort_role_vs_framed_empty.png)", "",
        "## By Model", "", "| Comparison | Model | Non-tied N | Win rate [adjusted 95% CI] |", "| --- | --- | ---: | ---: |"]
    for contrast in ("effort","role"):
        indexed = cell[cell.contrast.eq(contrast)].set_index("actor")
        for actor in ACTORS:
            r = indexed.loc[actor]
            lines.append(f"| {labels[contrast]} | {ACTOR_LABEL[actor]} | {r.resolved_n} | {r.target_win_rate_excluding_ties:.1%} [{r.familywise_ci_lo:.1%}, {r.familywise_ci_hi:.1%}] |")
    lines += ["", "The figure and model table use the paper's exact binomial Bonferroni intervals, with one family of the 14 planned model-comparison cells (seven models, two contrasts, essays only). N excludes ties, as in the paper figures.", "",
        "## Tie Sensitivity", "", "| Comparison | Equal-model half-win score [95% CI] |", "| --- | ---: |"]
    for r in agg[agg.metric.eq("half_win_score")].to_dict("records"):
        lines.append(f"| {labels[r['contrast']]} vs framed empty | {interval(r)} |")
    lines += ["", "## Exact Design", "",
        "Both treatment arms are the existing paper treatments: the sponsor-funded children's-hospital framing plus either the max-effort user suffix or the world-class essayist user suffix. The control retains the competition framing but has no sponsor, intervention, or effort/role suffix. All actor system prompts are blank. This tests headroom above framed empty, not the isolated causal effect of adding a cue to framed empty.", "",
        "All 30 essay topics and ten existing repetitions per actor are used. The effort dataset names its two five-repeat blocks 0-4 and block01/0-4; these are matched deterministically to empty repetitions 0-4 and 5-9. Original repeat/block metadata are retained. Role uses its existing repetitions 0-9. Each framed-empty output is reused once in each contrast. Selection and matching do not use prior panel winners.", "",
        "The panel is Claude Haiku 4.5, Gemini 3.1 Flash Lite Preview, and GPT-5-mini. Each supplies one vote in the same randomized orientation as the corresponding archived high-utility/framed-empty pair. Literal essay judge prompts and parameters were verified against all 6,300 archived reference votes before collection. Temperature=0, max_tokens=120, minimal/excluded judge reasoning. No actor condition or incentive is shown to judges. Two of three judges must favor a side; otherwise the panel is tied. A parseable verdict is accepted even if the judge reaches the cap, matching the reference procedure.", "",
        f"Planned pairs: {len(accounting)}; excluded for missing/invalid outputs or adjudicated unusability: {audit['excluded_pairs']}; unresolved panels: {len(unresolved)}; completed panels analyzed: {len(frame)}. Completion flags and simple text screens identify candidates for manual reading; those screens do not automatically exclude outputs. This run does not claim exhaustive LLM refusal classification. Flagged texts, manual decisions, and all excluded records remain preserved.", "",
        f"Recorded API cost: ${audit['recorded_cost_usd']:.2f}; {len(history)} attempts, {len(done)} successful calls of {len(requests)} planned. No actor-generation or classifier calls.", "",
        "One Gemini request for a GLM role essay repeatedly returned MALFORMED_RESPONSE with null content: ten default-routing attempts and one explicit Google AI Studio priority-tier attempt. The priority retry changed routing only, not the model, prompt, temperature, reasoning setting, or output cap; its exact override is saved with the response. " +
        ("This panel remains unresolved and excluded in full, not scored from two judges or imputed. " if unresolved else "The previously unresolved panel has now completed. ") +
        "The missing source is one MiMo role essay; two GLM role source essays were manually confirmed degenerate. All original evidence remains preserved.", "",
        "## Reproduce", "", "```sh", "python paper_reproduction/essay_headroom_empty.py analyze --allow-unresolved --out outputs/reproduced/essay_headroom_empty", "```", "",
        "Offline reproduction uses `data/processed/essay_headroom_empty/` only, plus this repo's standard analysis modules. It rebuilds panels from the saved raw responses. Exact requests, source prompts/texts, matching records, exclusions, and source hashes are preserved in that directory. Local account-credit and process-status files are not published or required. It does not need the private archive or credentials.", "",
        "Resume authorized collection with `python paper_reproduction/essay_headroom_empty.py judge --run`; check progress with the same script's `status` command. Initial extraction additionally requires `prepare --archive PATH`; do not rerun extraction for judging or analysis."]
    (out / "README.md").write_text("\n".join(lines)+"\n")
    inputs = list(RUN.glob("*.gz")) + [RUN/name for name in ("manual_review.json", "flagged_review.json", "input_snapshot.json")]
    inputs += [Path(__file__), Path(batch.__file__)] + [ROOT / "src/utility_behavior_gap" / name for name in (
        "constants.py", "judging.py", "prompts.py", "scripts/run_judging.py",
        "scripts/consolidate_aggregate_cis.py", "scripts/plot_canonical_highn_highlow.py")]
    (out / "provenance.json").write_text(json.dumps(dict(inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        source_experiment="essay treatments vs framed empty; existing paper outputs only", no_generation_calls=True,
        allow_unresolved=allow_unresolved, unresolved_panels=len(unresolved),
        model_interval_family=14, bootstrap_iterations=5000, seed=20260615),indent=2)+"\n")
    print(agg.to_string(index=False),flush=True)
    print("Report:",out / "README.md",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "judge", "analyze", "status"])
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--budget", type=float, default=15)
    parser.add_argument("--provider-tier", help="Explicit recovery routing; saved alongside each new response")
    parser.add_argument("--allow-unresolved", action="store_true", help="Analyze only complete panels and report every missing judgment")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.archive)
        return
    from utility_behavior_gap import compact_request_batch as batch
    if args.stage == "status":
        print(batch.status(RUN))
        return
    pairs = read_rows(RUN / "pairs.jsonl.gz")
    sources = {r["output_id"]:r for r in read_rows(RUN / "sources.jsonl.gz")}
    if not pairs or not sources:
        raise ValueError("Run prepare first")
    requests, accounting = build_requests(pairs, sources)
    if args.stage == "judge":
        print(f"Plan: {len(requests)} judge calls; no generation or classification calls", flush=True)
        if not args.run:
            return
        if args.env_file:
            load_env_file(args.env_file)
        provider = {"order": [args.provider_tier], "allow_fallbacks": False} if args.provider_tier else None
        batch.run(RUN, requests, parse, workers=args.workers, budget=args.budget, limit=args.limit, provider=provider)
    if args.stage == "analyze":
        analyze(requests, accounting,args.out,allow_unresolved=args.allow_unresolved)


if __name__ == "__main__":
    main()
