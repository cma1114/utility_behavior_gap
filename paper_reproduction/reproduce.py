#!/usr/bin/env python3
"""Offline paper reproduction from versioned inputs, with numerical regression checks."""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
PAPER = ROOT / "CURRENT_PAPER"
sys.path.insert(0, str(ROOT / "src"))
TASKS = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]

# Frozen dataset choices, not searches for whichever run happens to be newest.
CONTRASTS = {
    "highlow": (PAPER / "results/highn_condition_results_pair_outcomes.csv", "", "actor task domain", "highlow"),
    "direct": (DATA / "framed_user_strong_panel_pairs.csv.gz", "framed_user_strong", "actor task", "direct_instruction_main"),
    "role": (PAPER / "results/framed_user_prompt_role_pair_outcomes.csv", "user_strong", "actor task", "framed_user_prompt_role"),
    "harmful": (PAPER / "results/moral_low_vs_framed_empty_pair_outcomes.csv", "moral_bad", "actor task", "moral_low_vs_framed_empty"),
    "ceiling": (PAPER / "results/high_utility_vs_framed_neutral_pair_outcomes.csv", "", "actor task domain", "high_utility_vs_framed_neutral"),
    "utility_empty": (PAPER / "results/high_utility_vs_framed_empty_pair_outcomes.csv", "", "actor task domain", "high_utility_vs_framed_empty"),
    "utility_r0": (PAPER / "results/high_utility_vs_r0_pair_outcomes.csv", "", "actor task domain", "high_utility_vs_r0"),
}
FEATURES = {
    "direct_instruction": ("final_text_analysis_pair_deltas.csv.gz", "direct_instruction_rubric_coding", "make_direct_instruction_main_feature_table", "direct_instruction_main_feature_table", .20),
    "utility": ("final_text_analysis_pair_deltas.csv.gz", "utility_rubric_coding", "make_highlow_main_feature_table", "highlow_main_feature_table", .20),
    "framed_user_prompt_role": ("framed_user_prompt_role_pair_deltas.csv.gz", "framed_user_prompt_role_rubric_coding", "make_framed_user_prompt_role_feature_table", "framed_user_prompt_role_feature_table", .25),
    "moral_low_vs_framed_empty": ("moral_low_vs_framed_empty_current_pair_deltas_for_selection.csv.gz", "moral_low_vs_framed_empty_rubric_coding", "make_moral_low_framed_empty_feature_table", "moral_low_vs_framed_empty_feature_table", .25),
}
CHECKS = []


def mod(name):
    return importlib.import_module("utility_behavior_gap.scripts." + name)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def markdown(frame):
    return mod("analyze_length_controlled_feature_selection").markdown_table(frame)


def audit():
    entries = json.loads((DATA / "provenance.json").read_text())
    for item in entries:
        path = ROOT / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"Input checksum mismatch: {path}")
    for path, *_ in CONTRASTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for _, folder, *_ in FEATURES.values():
        for name in ("rubric_feature_analysis_flat_codes.csv", "rubric_feature_sample.csv"):
            if not (PAPER / "features" / folder / name).is_file():
                raise FileNotFoundError(PAPER / "features" / folder / name)
    print(f"Verified {len(entries)} exported-input checksums and all required contrast/rubric inputs", flush=True)
    policy = json.loads((DATA / "policy_framed_empty_provenance.json").read_text())
    if sha(ROOT / policy["path"]) != policy["sha256"]:
        raise ValueError("Policy follow-up evidence checksum mismatch")


def compare(actual, expected, keys, columns=None):
    import numpy as np
    import pandas as pd
    reference = pd.read_csv(expected) if isinstance(expected, Path) else expected
    if actual.duplicated(keys).any() or reference.duplicated(keys).any():
        raise ValueError(f"Nonunique comparison keys for {expected}")
    a = actual.set_index(keys).sort_index()
    b = reference.set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index, b.index, exact=False)
    columns = columns or [c for c in b.columns if c in a and pd.api.types.is_numeric_dtype(b[c])]
    if not columns:
        raise ValueError(f"No numerical columns checked for {expected}")
    for column in columns:
        np.testing.assert_allclose(a[column].astype(float), b[column].astype(float),
                                   rtol=1e-7, atol=1e-9, equal_nan=True,
                                   err_msg=f"Mismatch: {expected}, column={column}")
    CHECKS.append({"reference": str(expected) if isinstance(expected, Path) else "in-memory reference",
                   "rows": len(a), "columns": columns})


def call(name, args, log):
    module = mod(name)
    old = sys.argv
    try:
        sys.argv = [name, *map(str, args)]
        with log.open("w") as handle, contextlib.redirect_stdout(handle):
            module.main()
    finally:
        sys.argv = old


def outcomes(out):
    import pandas as pd
    aggregate = mod("consolidate_aggregate_cis")
    plotting = mod("plot_canonical_highn_highlow")
    rows = []
    for key, (path, target, cells, stem) in CONTRASTS.items():
        data = pd.read_csv(path)
        if key == "highlow":
            data = data[data.condition.eq("utility")].copy()
        if target:
            winners = set(data.panel_winner_condition.dropna())
            known = {"direct": {"framed_user_strong", "framed_neutral", "tie", "unresolved"},
                     "role": {"user_strong", "user_normal", "tie", "unresolved"},
                     "harmful": {"moral_bad", "framed_empty", "tie", "unresolved"}}[key]
            if winners - known:
                raise ValueError(f"Unexpected winner labels in {key}: {winners - known}")
            data["target_win"] = data.panel_winner_condition.eq(target)
            data["tie"] = data.panel_winner_condition.eq("tie")
            data["resolved"] = ~data.panel_winner_condition.isin(["tie", "unresolved"])
            data["target_loss"] = data.resolved & ~data.target_win
        for col in ("target_win", "target_loss", "tie", "resolved"):
            data[col] = data[col].astype(str).str.lower().isin(["true", "1", "1.0"])
        if not (data.target_win.astype(int) + data.target_loss.astype(int) + data.tie.astype(int)).le(1).all():
            raise ValueError(f"Overlapping outcomes in {key}")
        plot_cells = plotting.build_cell_rows(data)
        refs = {"direct": "direct_instruction_main_plot_data.csv", "role": "framed_user_prompt_role_model_task_plot_data.csv"}
        reference = PAPER / "results" / refs.get(key, stem + "_model_task_cells.csv")
        compare(plot_cells, reference, ["actor", "task"], ["familywise_ci_lo", "familywise_ci_hi", "ties"])
        plot_cells.to_csv(out / (stem + "_cells.csv"), index=False)
        label = {"direct": "Strong-prompt", "role": "World-class-role", "harmful": "Harmful"}.get(key, "High-utility")
        plotting.plot_lollipop(plot_cells, out / (stem if key == "direct" else stem + "_model_task_lollipop"),
                              title="", x_label=f"{label}-side win rate (ties excluded)")
        if key == "highlow":
            for domain in sorted(data.domain.unique()):
                dc = plotting.build_cell_rows(data, domain=domain)
                dc.to_csv(out / f"highlow_{domain}_cells.csv", index=False)
                plotting.plot_lollipop(dc, out / f"highlow_domain_{domain}_model_task_lollipop", title=domain.title())
        spec = aggregate.Contrast(key, key, "", path, cells.split(),
            aggregate.winner_is(target) if target else aggregate.target_win_from_columns,
            row_filter=(lambda r: r["condition"] == "utility") if key == "highlow" else None)
        rows.extend(aggregate.compute(spec))
        print(f"  {key}: regenerated and checked {len(plot_cells)} model/task cells", flush=True)
    result = pd.DataFrame(rows)
    reference = ROOT / "paper_reproduction/reference/aggregate_winrate_bootstrap_cis.csv"
    compare(result[result.contrast.isin(["highlow", "direct", "role", "harmful", "ceiling"])],
            reference, ["contrast", "scope", "task"], ["win_rate", "ci_lo", "ci_hi", "n_cells"])
    result.to_csv(out / "aggregate_winrate_bootstrap_cis.csv", index=False)
    (out / "aggregate_winrate_bootstrap_cis.md").write_text(markdown(result[["contrast", "task", "win_rate", "ci_lo", "ci_hi", "n_cells"]]) + "\n")


def features(out):
    import pandas as pd
    analysis = Path(os.environ["UTILITY_BEHAVIOR_GAP_OUTPUTS"]) / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    for src, dst in [("output_validity.csv.gz", "final_text_analysis_by_output.csv"),
                     ("canonical_highn_moral_refusal_classifications.jsonl.gz", "canonical_highn_moral_refusal_classifications.jsonl")]:
        with gzip.open(DATA / src, "rb") as handle:
            (analysis / dst).write_bytes(handle.read())
    all_features = {}
    for contrast, (input_name, rubric, table_module, table_stem, threshold) in FEATURES.items():
        prefix = out / f"{contrast}_length_controlled_feature_selection"
        args = ["--contrast", contrast, "--pair-deltas", DATA / input_name,
                "--rubric-dir", PAPER / "features" / rubric, "--out-prefix", prefix]
        call("analyze_length_controlled_feature_selection", args, out / (contrast + ".log"))
        frame = pd.read_csv(prefix.with_suffix(".csv"))
        if len(frame) != 59:
            raise ValueError(f"Expected 59 feature/task rows, got {len(frame)} for {contrast}")
        compare(frame, PAPER / "features" / prefix.with_suffix(".csv").name,
                ["task", "feature_id"], ["n_gap", "n_panel", "gap_effect", "gap_ci_low", "gap_ci_high", "panel_coef_per_sd", "panel_p"])
        table_args = ["--selection", prefix.with_suffix(".csv"), "--threshold", threshold,
                      "--out-prefix", out / table_stem]
        if contrast in {"direct_instruction", "utility"}:
            table_args += ["--pair-deltas", DATA / input_name, "--rubric-dir", PAPER / "features" / rubric]
        call(table_module, table_args, out / (table_stem + ".log"))
        pd.testing.assert_frame_equal(pd.read_csv(out / (table_stem + ".csv")),
                                      pd.read_csv(PAPER / "features" / (table_stem + ".csv")))
        CHECKS.append({"reference": str(PAPER / "features" / (table_stem + ".csv")), "table_exact_match": True})
        all_features[contrast] = frame
        print(f"  {contrast}: refit and checked all 59 generic/rubric feature rows", flush=True)
    directionality(all_features, out)


def directionality(frames, out):
    import pandas as pd
    wilson = mod("plot_direct_instruction_main").wilson_ci
    rows = []
    for contrast, frame in frames.items():
        for task in ["all_tasks", *TASKS]:
            part = frame if task == "all_tasks" else frame[frame.task.eq(task)]
            part = part.dropna(subset=["gap_effect", "panel_coef_per_sd"])
            n = len(part)
            positive = int((part.gap_effect * part.panel_coef_per_sd > 0).sum())
            _, lo, hi = wilson(positive, n, z=1.96)
            rows.append(dict(contrast=contrast, task=task, n=n, positive=positive,
                             proportion=positive/n, ci_lo=lo, ci_hi=hi))
    result = pd.DataFrame(rows)
    result.to_csv(out / "feature_directionality.csv", index=False)
    (out / "feature_directionality.md").write_text(markdown(result) + "\n")
    reference = pd.read_csv(PAPER / "features/feature_directionality_by_contrast_with_cis.csv")
    reference = reference[reference.scope.eq("all_estimable")].copy()
    reference["contrast"] = reference.contrast.map({"Direct instruction": "direct_instruction", "Role": "framed_user_prompt_role",
        "High-low utility": "utility", "Moral low vs framed-empty": "moral_low_vs_framed_empty"})
    reference["task"] = reference.task.map({"All tasks": "all_tasks", "Essay writing": "essay", "Grant abstract": "grant_proposal_abstract",
        "Incident postmortem": "incident_postmortem", "Translation": "translation"})
    reference = reference.rename(columns={"n_feature_rows": "n", "directionally_positive": "positive", "mean_positive": "proportion", "ci_low": "ci_lo", "ci_high": "ci_hi"})
    compare(result, reference, ["contrast", "task"], ["n", "positive", "proportion", "ci_lo", "ci_hi"])
    labels = {"direct_instruction": "Direct instruction", "framed_user_prompt_role": "Role",
              "utility": "High-low utility", "moral_low_vs_framed_empty": "Moral low vs framed-empty"}
    # Retain the accepted table's formatting and note; replace only calculated rows.
    template = (PAPER / "features/feature_directionality_by_contrast_table.tex").read_text()
    lines = []
    for line in template.splitlines():
        label = line.split(" & ", 1)[0]
        matching = [key for key, value in labels.items() if value == label]
        if matching:
            sub = result[result.contrast.eq(matching[0])].set_index("task")
            values = []
            for task in ["all_tasks", *TASKS]:
                r = sub.loc[task]
                values.append(f"{100*r.proportion:.1f}\\% [{100*r.ci_lo:.1f}\\%, {100*r.ci_hi:.1f}\\%]")
            line = label + " & " + " & ".join(values) + " " + "\\\\"
        lines.append(line)
    rendered = "\n".join(lines) + "\n"
    if rendered != template:
        raise AssertionError("Directionality LaTeX differs from the accepted table")
    (out / "feature_directionality_by_contrast_table.tex").write_text(rendered)


def trends(out):
    import pandas as pd
    import matplotlib.pyplot as plt
    gap = mod("analyze_canonical_utility_gap_trend")
    absolute = mod("analyze_high_utility_neutral_trend")
    figure = mod("plot_utility_trend_two_panel")
    g = pd.read_csv(DATA / "canonical_highn_utility_gap_trend__gap-actor-z_trials.csv.gz")
    # Recompute scale and outcomes rather than use frozen fitted values.
    g = gap.add_actor_gap_scales(g)
    base = pd.read_csv(PAPER / "results/highn_condition_results_pair_outcomes.csv")
    base = base[base.condition.eq("utility")]
    compare(g, base, ["pair_uid"], ["target_win", "target_loss", "tie", "resolved"])
    a = absolute.load_trials(PAPER / "results/high_utility_vs_framed_empty_pair_outcomes.csv",
                            outcome_col="outcome_vs_framed_empty", loss_value="framed_empty")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), sharey=True)
    for index, (module, data, column, term, mean_col, title) in enumerate([
        (gap, g, "delta_u_actor_z", "gap_value", "gap_mean", "High utility vs low utility"),
        (absolute, a, "high_utility_actor_z", "utility_value", "utility_mean", "High utility vs framed empty"),
    ]):
        stem = "gap" if index == 0 else "absolute"
        bins = module.build_bins(data, column, 10, 20260615)
        regressions = module.build_regressions(data, column)
        reference_name = "canonical_highn_utility_gap_trend__gap-actor-z_regressions.csv" if index == 0 else "high_utility_framed_empty_trend__utility-actor-z_regressions.csv"
        reference = pd.read_csv(ROOT / "paper_reproduction/reference" / reference_name)
        checked = regressions.copy()
        checked["row"] = range(len(checked)); reference["row"] = range(len(reference))
        compare(checked, reference, ["row"])
        bins.to_csv(out / f"{stem}_bins.csv", index=False)
        regressions.to_csv(out / f"{stem}_regressions.csv", index=False)
        fit, _ = module.fit_marginal_lines(data, column)
        figure.plot_win_rate_panel(axes[index], trials=data, bins=bins, x_col=column, x_mean_col=mean_col,
            x_label="High-low utility gap (within-actor z)" if index == 0 else "High-side utility (within-actor z)",
            title=title, panel_label=chr(65+index), fit_result=fit, fit_term=term)
    axes[0].set_ylabel("High-utility side win rate (ties excluded)")
    fig.tight_layout()
    fig.savefig(out / "utility_trend_two_panel_gap_and_absolute_framed_empty.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def reasoning(out):
    import pandas as pd
    pairs = ROOT / "data/reasoning/highlow_reasoning_traces_medium__tasks-essay__7-actors_pair_outcomes.csv"
    labels = ROOT / "data/reasoning/reasoning_trace_saved_stance_labels.jsonl.gz"
    traces = ROOT / "data/reasoning/reasoning_trace_inputs.jsonl.gz"
    for name in ("analyze_highlow_reasoning", "analyze_reasoning_incentive_mentions",
                 "analyze_reasoning_mentions_vs_performance", "analyze_reasoning_trace_awareness",
                 "analyze_saved_reasoning_stance_labels"):
        mod(name).ANALYSIS = out
    call("analyze_highlow_reasoning", ["--pair-outcomes", pairs], out / "original_outcomes.log")
    call("analyze_reasoning_incentive_mentions", ["--trace-input", traces, "--tasks", "essay", "--expected-outputs", "4200"], out / "mentions.log")
    call("analyze_reasoning_mentions_vs_performance", ["--pair-outcomes", pairs, "--mention-rows",
         out / "reasoning_incentive_mentions__success_only_per_output.csv"], out / "mention_performance.log")
    call("analyze_reasoning_trace_awareness", ["--trace-input", traces, "--expected-outputs", "4200", "--pair-outcomes", pairs], out / "awareness.log")
    call("analyze_saved_reasoning_stance_labels", ["--labels", labels, "--pair-outcomes", pairs], out / "saved_labels.log")
    for name in ("highlow_reasoning_traces_medium__tasks-essay__7-actors_model_task.csv",
                 "highlow_reasoning_traces_medium__tasks-essay__7-actors_summary.csv",
                 "reasoning_incentive_mentions__success_only_side_summary.csv",
                 "reasoning_trace_saved_stance_labels_counts.csv"):
        frame = pd.read_csv(out / name)
        reference = pd.read_csv(PAPER / "results/reasoning_trace" / name)
        frame["row"] = range(len(frame)); reference["row"] = range(len(reference))
        compare(frame, reference, ["row"])
    call("plot_highlow_reasoning_essay_lollipop", ["--input",
         out / "highlow_reasoning_traces_medium__tasks-essay__7-actors_model_task.csv",
         "--out", out / "highlow_reasoning_traces_medium_essay_lollipop.png"], out / "reasoning_figure.log")
    call("analyze_qwen35_reasoning_stance_followup", ["--pair-inputs", DATA / "qwen35_followup_pair_inputs.csv.gz",
         "--archived-labels", labels, "--followup-labels", DATA / "qwen35_followup_local_labels.jsonl.gz",
         "--output-dir", out], out / "qwen_followup.log")
    for name in ("summary", "patterns", "pattern_contrasts", "flagged_difference"):
        filename = f"qwen35_reasoning_stance_{name}.csv"
        frame = pd.read_csv(out / filename)
        reference = pd.read_csv(PAPER / "results/reasoning_trace" / filename)
        frame["row"] = range(len(frame)); reference["row"] = range(len(reference))
        compare(frame, reference, ["row"])
    call("analyze_reasoning_suspicion_exclusion", ["--pair-outcomes", pairs, "--labels", labels,
         "--qwen-followup", out / "qwen35_reasoning_stance_summary.csv", "--output-dir", out], out / "exclusion.log")
    name = "reasoning_suspicion_disbelief_exclusion.csv"
    result = pd.read_csv(out / name)
    reference = pd.read_csv(PAPER / "results/reasoning_trace" / name)
    result["row"] = range(len(result)); reference["row"] = range(len(reference))
    compare(result, reference, ["row"])
    filtered = result[result["sample"].eq("exclude_suspicion_or_disbelief")].copy()
    plotter = mod("plot_highlow_reasoning_essay_lollipop")
    filtered["task"] = "essay"
    filtered.to_csv(out / "reasoning_suspicion_filtered_cells.csv", index=False)
    plotter.plot(plotter.load_cells(out / "reasoning_suspicion_filtered_cells.csv"),
        out / "highlow_reasoning_suspicion_filtered_essay_lollipop.png",
        title="High-Low Utility: Excluding Suspicion or Disbelief", title_fontsize=18,
        n_note_override="n = retained pairs; expanded Qwen3.5 sample")
    print("  Updated reasoning appendix checked, including the 1,620-pair Qwen follow-up", flush=True)


def policy(out):
    from policy_framed_empty import run
    run(ROOT, out, compare)


STEPS = {"outcomes": outcomes, "trends": trends, "features": features, "reasoning": reasoning, "policy": policy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "run"])
    parser.add_argument("--step", choices=STEPS, action="append")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/reproduced")
    args = parser.parse_args()
    out = args.out.resolve()
    if out == ROOT or PAPER == out or PAPER in out.parents:
        parser.error("Output must not overwrite the repository or CURRENT_PAPER")
    os.environ["UTILITY_BEHAVIOR_GAP_OUTPUTS"] = str(out / "work")
    os.environ.setdefault("MPLCONFIGDIR", str(out / "matplotlib_cache"))
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    def no_network(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise RuntimeError("Network access is forbidden during offline reproduction")
    sys.addaudithook(no_network)
    audit()
    if args.command == "audit":
        return
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    completed = []
    for name in args.step or list(STEPS):
        dest = out / name
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Reproducing {name}...", flush=True)
        STEPS[name](dest)
        completed.append(name)
        (out / "verification.json").write_text(json.dumps({"completed": completed,
            "checks": CHECKS, "elapsed_seconds": round(time.monotonic()-start, 2),
            "api_calls": 0}, indent=2) + "\n")
    print(f"Complete: {', '.join(completed)}; {len(CHECKS)} numerical checks; {out}", flush=True)


if __name__ == "__main__":
    main()
