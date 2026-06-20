# Public Repository Manifest

Status: draft, written before copying or staging any public repository files.

This manifest defines what should go into a clean public repository for the
paper. It is an allowlist. Files not covered here should stay out unless they
are explicitly promoted and documented.

The public repository should support two goals:

1. Reproduce the paper figures, tables, and reported statistics from frozen
   processed data without making paid API calls.
2. Document the prompts, exclusions, feature definitions, and analysis workflow
   well enough that the raw API archive can be audited separately.

Raw API generations, judge votes, and utility-refit logs should be treated as
an external audit archive by default, not as normal GitHub contents.

## Target Layout

Recommended public repo structure:

```text
README.md
LICENSE
CITATION.cff
pyproject.toml
requirements.txt
analysis_specs/
docs/
src/utility_behavior_gap/
tests/
paper/
  figures/
  tables/
  results/
  features/
data/
  inputs/
  processed/
  audit/
external_raw_archive_manifest.md
```

`paper/` is for human-facing paper artifacts. `data/` is for machine-readable
inputs and processed data used to regenerate those artifacts. The same file
should not be copied into both places unless there is a clear reason.

## Include In Git

### Project Metadata

Include:

- `README.md`
- `LICENSE`
- `CITATION.cff`
- `pyproject.toml`
- `requirements.txt`
- `.gitignore`

The public `README.md` should explain how to regenerate the paper-facing
figures and tables from committed processed data, and should clearly mark any
commands that make paid API calls.

### Core Package

Include the source package under `src/utility_behavior_gap/`, but only after
removing generated cache files.

Core modules expected to be needed:

- `src/utility_behavior_gap/__init__.py`
- `src/utility_behavior_gap/analysis.py`
- `src/utility_behavior_gap/api_errors.py`
- `src/utility_behavior_gap/constants.py`
- `src/utility_behavior_gap/feature_specs.py`
- `src/utility_behavior_gap/fingerprints.py`
- `src/utility_behavior_gap/io_utils.py`
- `src/utility_behavior_gap/job_builder.py`
- `src/utility_behavior_gap/judging.py`
- `src/utility_behavior_gap/moral_filtering.py`
- `src/utility_behavior_gap/openrouter.py`
- `src/utility_behavior_gap/output_exclusions.py`
- `src/utility_behavior_gap/pair_selection.py`
- `src/utility_behavior_gap/paths.py`
- `src/utility_behavior_gap/prompts.py`
- `src/utility_behavior_gap/stats.py`

Do not include `__pycache__/` or `.DS_Store`.

### Script Allowlist

The public repo should include scripts that are needed to regenerate the paper
results or audit the finalized workflow. Diagnostic scripts can be retained only
if moved into an explicitly labeled archival directory.

Generation and judging primitives:

- `src/utility_behavior_gap/scripts/prepare_generation_jobs.py`
- `src/utility_behavior_gap/scripts/run_generation.py`
- `src/utility_behavior_gap/scripts/run_judging.py`
- `src/utility_behavior_gap/scripts/status.py`

Finalized run wrappers:

- `src/utility_behavior_gap/scripts/run_framed_user_strong_actor.sh`
- `src/utility_behavior_gap/scripts/run_framed_user_strong_repeat_block_actor.sh`
- `src/utility_behavior_gap/scripts/run_fund_wording_actor.sh`
- `src/utility_behavior_gap/scripts/run_canonical_amount_base_actor.sh`
- `src/utility_behavior_gap/scripts/run_canonical_highn_actor.sh`
- `src/utility_behavior_gap/scripts/run_r0_generations.py`
- `src/utility_behavior_gap/scripts/run_framed_empty_actor.sh`
- `src/utility_behavior_gap/scripts/run_amount_neutral_bridge_actor.sh`
- `src/utility_behavior_gap/scripts/run_moral_neutral_bridge_actor.sh`
- `src/utility_behavior_gap/scripts/run_moral_r0_bridge_actor.sh`
- `src/utility_behavior_gap/scripts/run_moral_empty_bridge_actor.sh`
- `src/utility_behavior_gap/scripts/run_high_utility_r0_bridge_actor.sh`
- `src/utility_behavior_gap/scripts/run_high_utility_framed_empty_bridge_actor.sh`

Canonical analyses and figures:

- `src/utility_behavior_gap/scripts/export_canonical_prompt_book.py`
- `src/utility_behavior_gap/scripts/classify_canonical_highn_moral_refusals.py`
- `src/utility_behavior_gap/scripts/analyze_framed_user_strong_panel.py`
- `src/utility_behavior_gap/scripts/plot_direct_instruction_main.py`
- `src/utility_behavior_gap/scripts/analyze_canonical_highn_conditions.py`
- `src/utility_behavior_gap/scripts/plot_canonical_highn_highlow.py`
- `src/utility_behavior_gap/scripts/plot_canonical_highn_amount.py`
- `src/utility_behavior_gap/scripts/plot_canonical_highn_moral.py`
- `src/utility_behavior_gap/scripts/analyze_canonical_utility_gap_trend.py`
- `src/utility_behavior_gap/scripts/analyze_high_utility_neutral_trend.py`

Bridge analyses:

- `src/utility_behavior_gap/scripts/prepare_highlow_neutral_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_highlow_neutral_bridge_judging.py`
- `src/utility_behavior_gap/scripts/plot_highlow_neutral_bridge.py`
- `src/utility_behavior_gap/scripts/prepare_amount_neutral_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_amount_neutral_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_amount_neutral_bridge_all.sh`
- `src/utility_behavior_gap/scripts/prepare_moral_neutral_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_moral_neutral_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_moral_neutral_bridge_all.sh`
- `src/utility_behavior_gap/scripts/prepare_moral_r0_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_moral_r0_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_moral_r0_bridge_all.sh`
- `src/utility_behavior_gap/scripts/prepare_moral_empty_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_moral_empty_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_moral_empty_bridge_all.sh`
- `src/utility_behavior_gap/scripts/prepare_framed_neutral_r0_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_framed_neutral_r0_bridge_judging.py`
- `src/utility_behavior_gap/scripts/prepare_highlow_r0_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_highlow_r0_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_high_utility_r0_bridge_all.sh`
- `src/utility_behavior_gap/scripts/plot_highlow_r0_bridge.py`
- `src/utility_behavior_gap/scripts/prepare_highlow_framed_empty_bridge_judging.py`
- `src/utility_behavior_gap/scripts/analyze_highlow_framed_empty_bridge_judging.py`
- `src/utility_behavior_gap/scripts/plot_highlow_framed_empty_bridge.py`

Feature analyses:

- `src/utility_behavior_gap/scripts/analyze_final_text_features.py`
- `src/utility_behavior_gap/scripts/analyze_standard_feature_deltas.py`
- `src/utility_behavior_gap/scripts/analyze_arm_match_feature_deltas.py`
- `src/utility_behavior_gap/scripts/run_task_rubric_feature_coding.py`
- `src/utility_behavior_gap/scripts/analyze_task_rubric_feature_coding.py`
- `src/utility_behavior_gap/scripts/make_feature_appendix_table.py`
- `src/utility_behavior_gap/scripts/analyze_panel_feature_lasso.py`
- `src/utility_behavior_gap/scripts/analyze_task_rubric_feature_models.py`

Appendix analyses currently worth preserving:

- `src/utility_behavior_gap/scripts/prepare_framed_user_mid_effort_essay_jobs.py`
- `src/utility_behavior_gap/scripts/run_framed_user_mid_effort_essay_actor.sh`
- `src/utility_behavior_gap/scripts/plot_framed_user_effort_dose.py`
- `src/utility_behavior_gap/scripts/run_highlow_reasoning_actor.sh`
- `src/utility_behavior_gap/scripts/run_highlow_reasoning_traces_actor.sh`
- `src/utility_behavior_gap/scripts/analyze_highlow_reasoning.py`
- `src/utility_behavior_gap/scripts/inspect_reasoning_traces.py`
- `src/utility_behavior_gap/scripts/analyze_reasoning_incentive_mentions.py`
- `src/utility_behavior_gap/scripts/analyze_reasoning_mentions_vs_performance.py`

### Definitions And Documentation

Include:

- `analysis_specs/feature_definitions.yaml`
- `docs/canonical_prompt_book.md`
- `docs/canonical_artifact_index.md`
- `docs/analysis_workflow.md`
- `docs/appendix_findings_to_carry_forward.md`
- `docs/appendix_effort_dose_test.md`
- `docs/public_repo_manifest.md`
- `docs/public_repo_audit_checklist.md`

The canonical prompt book must be the current prompt book after the authorized
changes:

- direct-instruction strong text at the end of the user prompt, not in the
  system prompt;
- intervention wording for direct instruction, high-low utility, and moral:
  `fund the following intervention:`;
- moral neutral target:
  `A healthcare intervention at a children's hospital`;
- no hidden system prompt in the high-low, amount, moral, framed neutral,
  framed empty, or R0 baselines unless explicitly documented.

### Tests

Include tests that check the public workflow:

- prompt templates and prompt-book export;
- generation-job structure without making API calls;
- judge-job structure without making API calls;
- exclusion/filtering logic;
- statistics and confidence interval helpers;
- packaged entry points.

Do not include `tests/__pycache__/`, `.DS_Store`, or local clipping files.

## Paper Artifacts To Include

The current source locations are `CURRENT_PAPER/` and `outputs/paper_ready/`.
For the public repo, copy only the curated final artifacts into `paper/`.

### Figures

Include PNG figures needed for the paper and appendix. Include PDFs only if the
paper build requires vector versions.

Main figures:

- `direct_instruction_main.png`
- `highlow_model_task_lollipop.png`
- `amount_model_task_lollipop.png`
- `moral_model_task_lollipop.png`
- `high_utility_vs_framed_empty_model_task_lollipop.png`
- `high_utility_vs_r0_model_task_lollipop.png`

Domain appendix figures:

- `highlow_domain_animals_model_task_lollipop.png`
- `highlow_domain_countries_model_task_lollipop.png`
- `highlow_domain_political_model_task_lollipop.png`
- `highlow_domain_religions_model_task_lollipop.png`
- `high_utility_vs_framed_empty_domain_animals_model_task_lollipop.png`
- `high_utility_vs_framed_empty_domain_countries_model_task_lollipop.png`
- `high_utility_vs_framed_empty_domain_political_model_task_lollipop.png`
- `high_utility_vs_framed_empty_domain_religions_model_task_lollipop.png`
- `high_utility_vs_r0_domain_animals_model_task_lollipop.png`
- `high_utility_vs_r0_domain_countries_model_task_lollipop.png`
- `high_utility_vs_r0_domain_political_model_task_lollipop.png`
- `high_utility_vs_r0_domain_religions_model_task_lollipop.png`

Trend and appendix figures:

- `highlow_gap_trend_actor_z.png`
- `highlow_gap_trend_nonpolitical_actor_z.png`
- `highlow_gap_trend_political_actor_z.png`
- `high_utility_vs_neutral_trend_actor_z.png`
- `appendix_effort_dose_mid_vs_neutral.png`
- `appendix_effort_dose_combined.png`
- `highlow_reasoning_traces_medium_essay_lollipop.png`

### Result Tables And Summaries

Include compact model/task/domain result CSVs and Markdown summaries that back
the figures and paper claims:

- `direct_instruction_panel_summary.md`
- `direct_instruction_main_plot_data.csv`
- `highn_condition_results_summary.md`
- `highn_condition_results_summary.csv`
- `highn_condition_results_model_task.csv`
- `highn_condition_results_pair_outcomes.csv`
- `highlow_summary.md`
- `highlow_model_task_cells.csv`
- `highlow_domain_model_task_cells.csv`
- `amount_summary.md`
- `amount_model_task_cells.csv`
- `amount_high_vs_neutral_summary.md`
- `moral_summary.md` if present after final refresh
- `moral_model_task_cells.csv` if present after final refresh
- `moral_low_vs_neutral_summary.md`
- `high_utility_vs_neutral_summary.md`
- `high_utility_vs_framed_empty_summary.md`
- `high_utility_vs_framed_empty_aggregate.csv`
- `high_utility_vs_framed_empty_model_task_cells.csv`
- `high_utility_vs_framed_empty_domain_model_task_cells.csv`
- `high_utility_vs_framed_empty_pair_outcomes.csv`
- `high_utility_vs_r0_summary.md`
- `high_utility_vs_r0_aggregate.csv`
- `high_utility_vs_r0_model_task_cells.csv`
- `high_utility_vs_r0_domain_model_task_cells.csv`
- `high_utility_vs_r0_pair_outcomes.csv`
- `appendix_effort_dose_summary.md`
- `appendix_effort_dose_summary.csv`
- `appendix_effort_dose_lollipop_data.csv`

If a listed file is absent, either regenerate it or remove it from the paper
claim. Do not silently substitute an older similarly named file.

### Feature Tables

Include paper-facing feature tables and enough machine-readable support files
to regenerate them.

Full generic-plus-LLM-rubric feature tables currently exist for:

- direct instruction;
- utility high versus low;
- amount high versus low;
- moral high versus low;
- utility high versus R0;
- utility high versus framed empty;
- moral low versus R0;
- moral low versus framed empty.

Generic-only feature tables currently exist for:

- utility high versus framed neutral;
- amount high versus framed neutral;
- moral low versus framed neutral.

Do not label a generic-only table as a full feature table.

Include:

- `*_combined_feature_appendix_summary.md`
- `*_combined_feature_appendix_clear_differences.md`
- `*_combined_feature_appendix_clear_differences.csv`
- `*_combined_feature_appendix_clear_differences.tex`
- `*_combined_feature_appendix_all.csv`
- `*_generic_feature_appendix_summary.md`
- `*_generic_feature_appendix_clear_differences.md`
- `*_generic_feature_appendix_clear_differences.csv`
- `*_generic_feature_appendix_clear_differences.tex`
- `*_generic_feature_appendix_all.csv`
- `*_feature_deltas_summary.md`
- `*_feature_deltas_overall.csv`

### Processed Feature Catalogs

These files are large but useful for full processed-data reproducibility:

- `final_text_analysis_by_output.csv`
- `final_text_analysis_pair_deltas.csv`
- `final_text_analysis_summary.md`

If GitHub size becomes a problem, store these as compressed CSVs under
`data/processed/` and document the checksums. They are not figures or tables;
they are processed analysis data.

### Audit Data

Include compact audit data needed to justify exclusions and feature coding:

- `moral_refusal_classifications.jsonl`
- rubric analysis CSVs under each included rubric-coding directory;
- `first_rubric_prompt.txt`;
- `sample_metadata.json`.

The full raw rubric prompt and response JSONL files are useful audit artifacts
but can be large. Either include them under `data/audit/` or place them in the
external raw archive with checksums. Do not delete them from the private working
copy just because they are too large for GitHub.

## External Raw Archive

These should not be copied into normal GitHub contents by default:

- `outputs/api/runs/`;
- global mirrored API logs such as `outputs/api/generations.jsonl` and
  `outputs/api/judge_votes.jsonl`;
- generation and judging failure logs;
- reasoning-trace raw response logs;
- utility-refit checkpoints and active-learning logs;
- old or failed API runs;
- raw outputs from obsolete prompt variants.

Instead, create `external_raw_archive_manifest.md` containing:

- archive filename or storage URL;
- date created;
- checksum;
- byte size;
- list of included run directories;
- mapping from each paper comparison to the exact run directories used;
- warning that API reruns are nondeterministic and may cost money.

## Explicit Exclusions

Do not include:

- `.env` or any file containing API keys or provider credentials;
- `.venv/`;
- `.DS_Store`;
- `__pycache__/`;
- `.pytest_cache/`;
- `.claude/`;
- `CLAUDE.md`;
- local agent instruction or scratch files unless explicitly sanitized;
- `notes/` unless a specific note is promoted and edited for public release;
- `reruns/` unless a subdirectory is explicitly promoted;
- `outputs_bad_repo_prompt_run_20260531/`;
- obsolete `outputs/archive/` material;
- old `modgrid_*` prompt books, figures, or summaries unless placed in an
  explicitly labeled historical appendix;
- old global append-only API mirrors unless they are part of the external raw
  archive;
- one-off diagnostic scripts not in the script allowlist.

## Current Open Decisions

Before copying anything into a public repo, resolve these:

- whether large processed feature catalogs are committed directly, compressed,
  or moved to the external archive;
- whether PDF versions of figures are needed, or PNGs are enough;
- whether raw rubric prompt/response JSONL files are committed or archived
  externally;
- whether old exploratory analyses should be omitted entirely or preserved in a
  clearly labeled `archive/diagnostic/` directory;
- whether all framed-neutral bridge contrasts need full LLM-rubric coding or
  should remain explicitly generic-only.

