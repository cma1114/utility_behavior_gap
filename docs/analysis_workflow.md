# Analysis Workflow

This document records the standard feature-analysis flow for the current paper
analyses. The editable source of truth for feature labels, definitions, display
rounding, and task-specific rubric dimensions is:

`analysis_specs/feature_definitions.yaml`

Generated CSVs and Markdown files in `outputs/analysis/` are analysis products,
not definition sources. If a feature definition or rubric dimension needs to
change, edit the YAML spec and rerun the relevant steps.

Feature analyses first exclude mechanically invalid outputs in every condition:
missing outputs, `success=false`, missing finish reason, non-`stop` finish,
explicit length truncation, or empty text. Tiny outputs and regex
`refusal_or_meta` flags are audited but not automatically excluded because they
can be valid for some tasks.

Feature analyses then apply a semantic exclusion filter by `output_id` wherever
classifier labels exist. Any output labeled as a full refusal, partial refusal,
or degenerate output is excluded from every analysis that would otherwise use
it. Outputs without a classifier label are retained but reported as
unclassified; they are not counted as classifier-clean. The current classifier
file is
`outputs/analysis/canonical_highn_moral_refusal_classifications.jsonl`, which
was created for canonical moral outputs because that is where refusals were
most common. The filtering code is condition-agnostic, so a future global
classifier file can be applied without changing the analysis logic.

## Standard Paper-Facing Comparison Figures

Model-by-task lollipop figures use one inference convention across paper-facing
comparisons: panel ties are excluded from the displayed win-rate denominator,
and each actor-task cell is plotted with a Bonferroni exact binomial FWER 95%
CI across the 28 actor-task cells in that comparison. Holm-adjusted p-values
are retained in the model-task CSVs as secondary multiplicity-adjusted tests,
but the plotted intervals and `CI-positive` counts are the FWER CIs.

Aggregate summaries use equal-cell means rather than pooled raw rates. For
high-low utility and utility bridge analyses, the aggregate cell is actor x
task x domain; for non-domain contrasts, it is actor x task unless a narrower
scope is explicitly stated.

Current paper-ready artifacts are copied to `outputs/paper_ready/` and mirrored
to `CURRENT_PAPER/`.

High utility versus R0 is rebuilt with:

```bash
bash src/utility_behavior_gap/scripts/analyze_high_utility_r0_bridge_all.sh
```

## Direct-Instruction Feature Table

Current contrast: finalized exhortative user prompt versus framed neutral.

Standard generic features:

- `words`
- `paragraphs`
- `unique_word_ratio`
- `quantitative_detail`
- `textstat_flesch_kincaid_grade`
- `positive_words_per_1k`
- `negative_words_per_1k`

`quantitative_detail` is defined as `z(numbers + percentages)`, standardized
within task before paired differencing.

Task-specific LLM rubric features are also defined in
`analysis_specs/feature_definitions.yaml`. The same definitions are used in the
rubric-coder prompt and in the final feature table.

Run the local generic-feature summary:

```bash
python -m utility_behavior_gap.scripts.analyze_direct_instruction_feature_deltas
```

Run the LLM rubric coding only when new coded data are needed. This command
makes paid API calls when `--run` is included:

```bash
python -m utility_behavior_gap.scripts.run_task_rubric_feature_coding \
  --contrast direct_instruction \
  --sample-size-per-task 120 \
  --run
```

Analyze an existing rubric-coding run:

```bash
python -m utility_behavior_gap.scripts.analyze_task_rubric_feature_coding \
  --run-dir outputs/analysis/task_rubric_feature_coding/direct_instruction__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash
```

Build the combined paper-facing table:

```bash
python -m utility_behavior_gap.scripts.make_feature_appendix_table \
  --generic-by-task outputs/analysis/direct_instruction_feature_deltas_by_task.csv \
  --generic-pairs outputs/analysis/final_text_analysis_pair_deltas.csv \
  --generic-pairs-comparison direct_instruction \
  --rubric-run-dir outputs/analysis/task_rubric_feature_coding/direct_instruction__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash \
  --out-prefix outputs/analysis/direct_instruction_combined_feature_appendix \
  --title "Direct Instruction Feature Appendix" \
  --comparison direct_instruction \
  --left-key strong \
  --right-key neutral \
  --left-label "direct high" \
  --right-label "direct low"
```

Primary outputs:

- `outputs/analysis/direct_instruction_combined_feature_appendix_summary.md`
- `outputs/analysis/direct_instruction_combined_feature_appendix_all.csv`
- `outputs/analysis/direct_instruction_combined_feature_appendix_clear_differences.csv`
- `outputs/analysis/direct_instruction_combined_feature_appendix_clear_differences.md`
- `outputs/analysis/direct_instruction_combined_feature_appendix_clear_differences.tex`

The full CSV keeps provenance columns. The Markdown and LaTeX outputs are
paper-facing views with compact columns and task-section headings. The compact
tables include `Panel preference (r)`, the within-comparison correlation between
left-minus-right feature score and left-minus-right panel score. Positive values
mean the judging panel tended to prefer the side with more of that feature;
negative values mean the panel tended to prefer the side with less.

## Standard Generic Feature Deltas

For any finalized high/low contrast in
`outputs/analysis/final_text_analysis_pair_deltas.csv`, run the same standard
generic-feature summary with:

```bash
python -m utility_behavior_gap.scripts.analyze_standard_feature_deltas \
  --contrast utility
```

Supported contrasts are `direct_instruction`, `utility`, `amount`, and `moral`.
The utility command compares `utility_high - utility_low`; the amount command
compares `amount_high - amount_low`; the moral command compares
`moral_high - moral_low`.

Primary utility outputs:

- `outputs/analysis/utility_feature_deltas_summary.md`
- `outputs/analysis/utility_feature_deltas_overall.csv`
- `outputs/analysis/utility_feature_deltas_by_task.csv`
- `outputs/analysis/utility_feature_deltas_by_actor.csv`
- `outputs/analysis/utility_feature_deltas_by_actor_task.csv`

Task-specific LLM rubric coding is separate from this local generic-feature
analysis and makes API calls when `--run` is included. For utility high versus
low, the command is:

```bash
python -m utility_behavior_gap.scripts.run_task_rubric_feature_coding \
  --contrast utility \
  --sample-size-per-task 120 \
  --run
```

Then analyze the resulting run directory with:

```bash
python -m utility_behavior_gap.scripts.analyze_task_rubric_feature_coding \
  --run-dir outputs/analysis/task_rubric_feature_coding/<utility-run-directory>
```

Build the combined table for a high/low contrast with:

```bash
python -m utility_behavior_gap.scripts.make_feature_appendix_table \
  --generic-by-task outputs/analysis/utility_feature_deltas_by_task.csv \
  --generic-pairs outputs/analysis/final_text_analysis_pair_deltas.csv \
  --generic-pairs-comparison utility \
  --rubric-run-dir outputs/analysis/task_rubric_feature_coding/utility__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash \
  --out-prefix outputs/analysis/utility_combined_feature_appendix \
  --title "Utility High-Low Feature Appendix" \
  --comparison utility \
  --left-key high \
  --right-key low \
  --left-label "utility high" \
  --right-label "utility low"
```

For bridge contrasts such as utility-high versus R0 or moral-low versus framed
empty, pass both the arm-match feature pair file and the bridge outcome file.
The bridge outcome join is what makes `Panel preference (r)` use the correct
panel judgment for that baseline comparison.
