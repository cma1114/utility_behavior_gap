"""Offline policy follow-up: recorded votes/codes -> panels/fits -> verified tables.

No original run directory, research checkout, or live API is accessed. Called
by reproduce.py, which blocks network access before importing this module.
"""
from collections import defaultdict
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from utility_behavior_gap.judging import derive_panel_winner_condition
from utility_behavior_gap.openrouter import response_text
from utility_behavior_gap.scripts import analyze_length_controlled_feature_selection as features
from utility_behavior_gap.scripts import make_moral_low_framed_empty_feature_table as table
from utility_behavior_gap.scripts import plot_canonical_highn_highlow as plotting
from utility_behavior_gap.scripts.consolidate_aggregate_cis import bootstrap_equal_cell
from utility_behavior_gap.scripts.run_judging import judge_prompt, parse_winner
from utility_behavior_gap.scripts.run_task_rubric_feature_coding import actor_balanced_sample
from utility_behavior_gap.scripts.run_task_rubric_pilot import TASK_DIMENSIONS, task_context, rubric_prompt, extract_json_object, normalize_codes

JUDGES = {"anthropic/claude-haiku-4.5", "google/gemini-3.1-flash-lite-preview", "openai/gpt-5-mini"}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load(path):
    records = defaultdict(list)
    with gzip.open(path, "rt") as handle:
        for line in handle:
            item = json.loads(line)
            records[item["file"]].append(item["row"])
    return records


def unique(rows, key):
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError(f"Duplicate {key}")
        result[row[key]] = row
    return result


def successful(records, stage):
    return unique([r for r in records[stage] if r["success"]], "job_id")


def verify_sources(records):
    sources = unique(records["panel/sources.jsonl.gz"], "output_id")
    pairs = unique(records["panel/pair_accounting.jsonl.gz"], "pair_uid")
    if len(sources) != 4200 or len(pairs) != 2100:
        raise ValueError("Incomplete source cohort")
    for pair in pairs.values():
        excluded = False
        for side in ("left", "right"):
            source = sources[pair[f"{side}_output_id"]]
            if digest(source["text"]) != pair[f"{side}_hash"] or source["output_hash"] != pair[f"{side}_hash"]:
                raise ValueError("Source text hash mismatch")
            if source["actor"] != pair["actor"] or source["task"] != pair["task"] or source["item_label"] != pair["item_label"]:
                raise ValueError("Source/task metadata mismatch")
            if pair[f"{side}_label"] not in {"clean", "full_refusal", "partial_refusal", "degenerate"}:
                raise ValueError("Unclassified source")
            excluded |= source["mechanically_invalid"] or pair[f"{side}_label"] != "clean"
        if bool(excluded) != pair["excluded"]:
            raise ValueError("Exclusion mismatch")
    if sum(p["excluded"] for p in pairs.values()) != 10:
        raise ValueError("Exclusion accounting changed")
    return sources, pairs


def judge_side(job, response):
    choice = parse_winner(response_text(response))
    if choice == "unresolved":
        raise ValueError("Unresolved successful judgment")
    return "tie" if choice == "tie" else ("right" if (choice == "a") == job["flipped"] else "left")


def panels(records, sources, accounting):
    votes = defaultdict(dict)
    done = successful(records, "panel/judge_responses.jsonl")
    jobs = unique(records["panel/judge_requests.jsonl.gz"], "job_id")
    if set(jobs) != set(done) or len(jobs) != 6270:
        raise ValueError("Incomplete or unexpected judge requests")
    for job in jobs.values():
        pair = accounting[job["pair_uid"]]
        if pair["excluded"]:
            raise ValueError("Excluded pair was judged")
        request = job["request"]
        identity = digest(json.dumps(["judge", [pair["pair_uid"], request["model"]], request], sort_keys=True))
        if identity != job["job_id"]:
            raise ValueError("Judge request identity mismatch")
        profile = next(p for p in pair["judge_profiles"] if p["model"] == request["model"])
        if job["flipped"] != profile["flipped"] or any(request[k] != profile[k] for k in ("temperature", "max_tokens", "reasoning")):
            raise ValueError("Frozen judge protocol changed")
        a, b = sources[pair["left_output_id"]], sources[pair["right_output_id"]]
        if job["flipped"]:
            a, b = b, a
        prompt = judge_prompt(pair["judge_job"], {"output_text": a["text"]}, {"output_text": b["text"]})
        if request["messages"] != [{"role": "user", "content": prompt}]:
            raise ValueError("Literal judge request changed")
        side = judge_side(job, done[job["job_id"]]["raw_response"])
        if side != done[job["job_id"]]["parsed"]["winner_side"]:
            raise ValueError("Parsed judge side changed")
        if request["model"] in votes[pair["pair_uid"]]:
            raise ValueError("Duplicate judge in panel")
        votes[pair["pair_uid"]][request["model"]] = side
    rows = []
    for pair in accounting.values():
        if pair["excluded"]:
            continue
        if set(votes[pair["pair_uid"]]) != JUDGES:
            raise ValueError("Incomplete panel")
        winner = derive_panel_winner_condition({}, votes[pair["pair_uid"]].values())
        row = {k: v for k, v in pair.items() if k not in {"judge_job", "judge_profiles"}}
        rows.append(row | dict(panel_winner=winner, target_win=winner=="left", target_loss=winner=="right",
                               tie=winner=="tie", resolved=winner!="tie", panel_score={"left":1., "right":-1., "tie":0.}[winner]))
    return pd.DataFrame(rows)


def aggregate(data):
    rows = []
    for idx, task in enumerate(["overall", *plotting.TASK_ORDER]):
        part = data if task == "overall" else data[data.task.eq(task)]
        for metric in ("win_rate", "half_win_score"):
            sub = part[part.resolved].copy() if metric == "win_rate" else part.copy()
            sub["score"] = sub.target_win.astype(float) + (.5 * sub.tie if metric == "half_win_score" else 0)
            arrays = [g.score.to_numpy() for _, g in sub.groupby(["actor", "task"], sort=True)]
            mean, lo, hi, n = bootstrap_equal_cell(arrays, seed=20260615+idx, iterations=5000)
            rows.append(dict(task=task, metric=metric, mean=mean, ci_low=lo, ci_high=hi, cells=n,
                pairs=len(part), wins=int(part.target_win.sum()), losses=int(part.target_loss.sum()), ties=int(part.tie.sum())))
    return pd.DataFrame(rows)


def feature_inputs(records, pairs, sources):
    measured = pd.DataFrame(records["features/pair_measurements.csv"])
    joined = measured.merge(pairs, on="pair_uid", suffixes=("_saved", ""), validate="one_to_one")
    if len(joined) != 2090 or len(measured) != 2090:
        raise ValueError("Missing generic measurements")
    for name in ("actor", "task", "left_output_id", "right_output_id", "left_hash", "right_hash", "panel_score"):
        if not joined[name].eq(joined[name+"_saved"]).all():
            raise ValueError(f"Measurement/panel mismatch: {name}")
    jobs = unique(records["features/rubric_requests.jsonl.gz"], "job_id")
    done = successful(records, "features/rubric_responses.jsonl")
    if set(jobs) != set(done) or len(jobs) != 480:
        raise ValueError("Incomplete rubric coding")
    sample = actor_balanced_sample(pairs.sort_values("pair_uid").reset_index(drop=True), sample_size_per_task=120, seed=20261114)
    if set(sample.pair_uid) != {j["pair_uid"] for j in jobs.values()}:
        raise ValueError("Rubric sample differs from frozen random draw")
    lookup = joined.set_index("pair_uid")
    coded = []
    for job in jobs.values():
        pair = lookup.loc[job["pair_uid"]]
        request = job["request"]
        if (request["model"], request["temperature"], request["max_tokens"]) != ("google/gemini-2.5-flash", 0, 1400):
            raise ValueError("Rubric parameters changed")
        if {job["a_side"], job["b_side"]} != {"left", "right"}:
            raise ValueError("Invalid rubric orientation")
        if any(job[k] != pair[k] for k in ("left_output_id", "right_output_id", "left_hash", "right_hash", "actor", "task")):
            raise ValueError("Rubric source mapping changed")
        row = pd.Series(dict(task=pair.task, task_context=task_context(pair.task, pair.item_label),
            output_a=sources[pair[job["a_side"]+"_output_id"]]["text"].strip(),
            output_b=sources[pair[job["b_side"]+"_output_id"]]["text"].strip()))
        if request["messages"] != [{"role":"user", "content":rubric_prompt(row, TASK_DIMENSIONS[pair.task])}]:
            raise ValueError("Literal rubric request changed")
        identity = digest(json.dumps(["rubric", [job["pair_uid"], pair.left_hash, pair.right_hash], request], sort_keys=True))
        if identity != job["job_id"]:
            raise ValueError("Rubric request identity mismatch")
        response = done[job["job_id"]]["raw_response"]
        if response["choices"][0]["finish_reason"] != "stop":
            raise ValueError("Truncated rubric response")
        codes = normalize_codes(extract_json_object(response_text(response)), TASK_DIMENSIONS[pair.task])
        for dim, code in codes.items():
            side = job["a_side"] if code["winner"] == "A" else job["b_side"] if code["winner"] == "B" else code["winner"]
            if side != done[job["job_id"]]["parsed"]["codes"][dim]["winner_side"]:
                raise ValueError("Rubric side parsing mismatch")
            score = {"left":1., "right":-1., "tie":0., "not_applicable":np.nan}[side]
            coded.append(dict(pair_uid=job["pair_uid"], actor=pair.actor, task=pair.task, contrast="policy_low_vs_framed_empty",
                              dimension=dim, rubric_score=score, delta_words=pair.delta_words, panel_score=pair.panel_score))
    return joined, pd.DataFrame(coded)


def fit_features(pairs, rubric):
    generic = features.generic_rows(pairs)
    rows = []
    labels = features.rubric_dimension_labels()
    for (task, dim), group in rubric.groupby(["task", "dimension"]):
        rows.append(features.make_row(task=task, family="LLM rubric marker", feature_id=dim,
            feature_label=labels[dim], definition="See frozen task rubric", controls_words_for_gap=True, controls_words_for_panel=True,
            gap=features.fit_adjusted_gap(group, feature_col="rubric_score", is_words=False),
            panel=features.fit_panel_association(group, feature_col="rubric_score", is_words=False)))
    result = pd.concat([generic, pd.DataFrame(rows)], ignore_index=True)
    result["selected"] = result.gap_ci_excludes_zero & result.panel_ci_excludes_zero & result.gap_effect_sd.abs().ge(.25)
    for _, group in result.groupby(["task", "family"]):
        for term in ("gap", "panel"):
            valid = group[f"{term}_p"].dropna()
            if len(valid):
                result.loc[valid.index, f"{term}_p_holm"] = multipletests(valid, method="holm")[1]
    return result


def run(root, out, compare):
    reference = root / "docs/reviewer_response/policy_framed_empty"
    evidence = load(root / "data/processed/policy_framed_empty.jsonl.gz")
    sources, accounting = verify_sources(evidence)
    pairs = panels(evidence, sources, accounting)
    expected = pd.read_csv(reference / "pair_outcomes.csv")
    pd.testing.assert_series_equal(pairs.set_index("pair_uid").panel_winner.sort_index(),
                                   expected.set_index("pair_uid").panel_winner.sort_index())
    pairs.to_csv(out / "pair_outcomes.csv", index=False)
    aggregates = aggregate(pairs)
    compare(aggregates, reference / "aggregate_results.csv", ["task", "metric"])
    aggregates.to_csv(out / "aggregate_results.csv", index=False)
    cells = plotting.build_cell_rows(pairs, domain="political")
    compare(cells, reference / "model_task_cells.csv", ["actor", "task"])
    cells.to_csv(out / "model_task_cells.csv", index=False)
    plotting.plot_lollipop(cells, out / "low_policy_vs_framed_empty_lollipop", title="", x_label="Low-policy-side win rate (ties excluded)")
    measured, rubric = feature_inputs(evidence, pairs, sources)
    fits = fit_features(measured, rubric)
    compare(fits, reference / "features/feature_results.csv", ["task", "feature_id"],
            ["n_gap", "n_panel", "gap_effect", "gap_ci_low", "gap_ci_high", "gap_effect_sd", "gap_ci_low_sd", "gap_ci_high_sd",
             "gap_p", "gap_p_holm", "panel_coef_per_sd", "panel_ci_low", "panel_ci_high", "panel_p", "panel_p_holm", "selected"])
    fits.to_csv(out / "feature_results.csv", index=False)
    rubric.to_csv(out / "rubric_scores.csv", index=False)
    # Refit the harmful counterpart rather than treating its saved estimates as input.
    data = pd.read_csv(root / "data/processed/moral_low_vs_framed_empty_current_pair_deltas_for_selection.csv.gz")
    data = features.add_quantitative_detail_rate(data)
    data["panel_score"] = data.effect_score_high_minus_low
    harmful = pd.concat([features.generic_rows(data), features.rubric_rows(root / "CURRENT_PAPER/features/moral_low_vs_framed_empty_rubric_coding", data)])
    compare(harmful, root / "CURRENT_PAPER/features/moral_low_vs_framed_empty_length_controlled_feature_selection.csv", ["task", "feature_id"],
            ["n_gap", "n_panel", "gap_effect", "gap_ci_low", "gap_ci_high", "gap_effect_sd", "panel_coef_per_sd", "panel_ci_low", "panel_ci_high"])
    # Preserve the paper's display order only; all estimates above were refitted.
    order = pd.read_csv(root / "CURRENT_PAPER/features/moral_low_vs_framed_empty_length_controlled_feature_selection.csv").set_index(["task", "feature_id"]).index
    harmful = harmful.set_index(["task", "feature_id"]).loc[order].reset_index()
    harmful.to_csv(out / "harmful_feature_results.csv", index=False)
    table.write_parallel(out / "harmful_feature_results.csv", out / "feature_results.csv", out / "harmful_and_low_policy_feature_table", threshold=.25, labels=("Harmful", "Low policy"))
    pd.testing.assert_frame_equal(pd.read_csv(out / "harmful_and_low_policy_feature_table.csv"),
                                  pd.read_csv(reference / "features/harmful_and_low_policy_feature_table.csv"))
    (out / "audit.json").write_text(json.dumps(dict(sources=len(sources), planned_pairs=len(accounting), excluded_pairs=10,
        complete_panels=len(pairs), verified_literal_judge_requests=6270, verified_literal_rubric_requests=480,
        refitted_policy_features=len(fits), refitted_harmful_features=len(harmful), parallel_table_rows=10, api_calls=0), indent=2) + "\n")
    print("  policy: 2,090 panels reconstructed; 118 feature fits and 10 parallel rows verified", flush=True)
