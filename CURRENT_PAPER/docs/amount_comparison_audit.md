# Amount Comparison Audit

Date: 2026-06-20

This is the full inventory of amount-related comparisons currently present in
the repository. It separates current paper-facing analyses from older,
diagnostic, or superseded amount runs. Do not treat a file as paper-ready merely
because it contains `amount` in its name.

## Current Paper-Facing Analyses

### 1. Amount high vs amount low

Primary files:

- Results summary: `CURRENT_PAPER/results/amount_summary.md`
- Model-task cells: `CURRENT_PAPER/results/amount_model_task_cells.csv`
- Figure: `CURRENT_PAPER/figures/amount_model_task_lollipop.png`
- Feature table: `CURRENT_PAPER/features/amount_combined_feature_appendix_clear_differences.md`
- Full feature rows: `CURRENT_PAPER/features/amount_combined_feature_appendix_all.csv`

Design:

- 8,400 pairs: 7 actors x 4 tasks x 30 items x 10 repeats.
- Source blocks: 4,200 corrected base pairs plus 4,200 high-N extension pairs.
- Repeats 0-9 are present for every task.
- `amount_target_mismatch_from_base` is false for all 8,400 rows.
- Blank system prompts.
- Manipulation: `$1,000,000` versus `$100` donated toward `A healthcare
  intervention at a children's hospital.`

Primary result, equal-cell mean over actor x task cells:

| scope | win rate | 95% CI |
| --- | ---: | --- |
| Overall | 51.6% | 48.0%-55.5% |
| Essay writing | 50.0% | 43.6%-56.6% |
| Grant abstract | 61.9% | 55.2%-68.8% |
| Incident postmortem | 49.6% | 45.0%-54.3% |
| Translation | 45.0% | 41.3%-48.7% |

Interpretation:

- The only clearly positive task-level amount effect is grant abstract.
- Translation is negative in the current high-N data.
- The all-task amount contrast is not a broad positive control.

Feature audit:

- The grant feature table shows `Quantitative detail` higher in amount high:
  delta 0.070, 95% CI [0.012, 0.130].
- The panel preference correlation for grant `Quantitative detail` is +0.36
  (n=2,100), computed from the source pair-delta file.
- This value was previously blank because the appendix-table builder computed
  panel associations before adding the composite `quantitative_detail` feature.
  `src/utility_behavior_gap/scripts/make_feature_appendix_table.py` now applies
  the composite-feature helper before computing those associations.
- Grant amount-high outputs are slightly shorter on average, so the grant effect
  is not a simple length artifact. In grant pairs, judges prefer the side with
  more words and more quantitative detail, but amount high is not simply longer
  overall.
- The strongest grant feature differences are the LLM-coded grant-quality
  dimensions: evaluation rigor, risk mitigation, intervention specificity,
  problem significance, feasibility/readiness, measurable impact, and
  stakeholder/context fit.

### 2. Amount high vs framed neutral

Primary files:

- Results summary: `CURRENT_PAPER/results/amount_high_vs_neutral_summary.md`
- Pair outcomes: `CURRENT_PAPER/results/amount_high_vs_framed_neutral_pair_outcomes.csv`
- Feature table: `CURRENT_PAPER/features/amount_high_vs_framed_neutral_generic_feature_appendix_clear_differences.md`
- Full feature rows: `CURRENT_PAPER/features/amount_high_vs_framed_neutral_generic_feature_appendix_all.csv`

Primary result, equal actor-task mean:

| scope | win rate | 95% CI |
| --- | ---: | --- |
| Overall | 53.9% | 50.8%-57.0% |
| Essay writing | 51.2% | 43.6%-58.9% |
| Grant abstract | 61.1% | 55.2%-67.0% |
| Incident postmortem | 54.9% | 49.3%-60.6% |
| Translation | 48.4% | 43.2%-53.6% |

Feature audit:

- Grant amount-high vs framed-neutral does not show a clear quantitative-detail
  increase. Its clear generic differences are negative-word rate,
  Flesch-Kincaid grade, paragraphs, and unique-word ratio.
- Incident amount-high vs framed-neutral does show a clear quantitative-detail
  increase, and the panel preference correlation for quantitative detail is
  positive.

## Clean Local Reruns, Superseded By Current High-N Analysis

These are in `reruns/amount_*`. They were useful while debugging amount, but
they are not the current paper-facing result because the current analysis uses
the canonical 10-repeat high-N dataset above.

Common setup:

- 7 actors x 30 items x 5 repeats = 1,050 pairs per task.
- Three-judge panel.
- `$1,000,000` versus `$100`.
- Non-essay tasks used blinding text and a competition/wins framing.
- Full generations and judge votes are in the `reruns/amount_*` folders.

Panel-majority summaries, ties excluded:

| task | pooled amount-high win rate | note |
| --- | ---: | --- |
| Essay writing | 57.5% | positive in this rerun |
| Grant abstract | 59.2% | positive in this rerun |
| Incident postmortem | 50.2% | heterogeneous/null |
| Translation | 52.4% | weak/null |

These clean reruns explain some earlier discussion that amount looked positive
for essay and grant. That is not the current high-N result: in the current
paper-facing 10-repeat dataset, essay is 50.0% and grant remains positive.

## Older/Stale Trial-Level Scaleups

These are in `trial_level_data/amount_scaleup_*`. They are not current
paper-facing sources.

Files:

- `trial_level_data/amount_scaleup_grant_proposal_abstract_v2`
- `trial_level_data/amount_scaleup_incident_postmortem_v2_maxtok3000`
- `trial_level_data/amount_scaleup_incident_postmortem_v2_maxtok3000_mimofix`
- `trial_level_data/amount_scaleup_translation_v2`

Summary of their `summary.json` files:

- Grant: only Qwen3.6 Plus was CI-positive; pooled pattern was weaker than the
  later clean rerun and different from the current high-N result.
- Incident: GPT-5.4 mini was CI-positive; MiMo appeared in a separate fix folder;
  overall pattern was heterogeneous.
- Translation: no actor was CI-positive.

These were part of the earlier confused/stale trail and should not be used for
paper-ready amount claims.

## Original Essay-Only Amount Data

Files:

- `essay_all_conditions/amount/*.json`

This is older essay-only amount data from the supplied trial-level data. It has
150 trials per actor and a different judge setup.

Tie-excluded pooled result:

| scope | high wins | low wins | ties | disagree | high win rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Essay only, pooled | 484 | 374 | 121 | 71 | 56.4% |

Per-actor high win rates:

| actor | high win rate |
| --- | ---: |
| DeepSeek V3.2 | 54.2% |
| GLM-5.1 | 53.7% |
| GPT-5.4 mini | 69.1% |
| Kimi K2.5 | 59.6% |
| MiMo V2 Pro | 56.3% |
| Qwen3.5 9B | 49.6% |
| Qwen3.6 Plus | 55.3% |

This result helps explain the earlier impression that amount moved essays, but
it is not the current paper-facing amount result.

## Diagnostic/Quality-Check Amount Artifacts

### Translation quality check

Files:

- `outputs/analysis/amount_translation_quality_check_summary.md`
- associated outputs/votes/examples in `outputs/analysis/amount_translation_quality_check_*`

Finding:

- Current canonical translation amount result remains negative after dropping
  outputs with untranslated Chinese characters or other mechanical flags.
- Duplicate hash-matched vote rows do not change any canonical panel outcome.

### Older figure-style amount artifact

File:

- `outputs/analysis/incentive_amount_main_plot_data.csv`

This is an older figure-style 150-pair-per-cell artifact with low resolved
counts and no Holm-positive cells. It should not be used as the current amount
result.

## Comparisons Not Found

I did not find current completed analyses for:

- amount low vs R0
- amount high vs R0
- amount low vs framed empty
- amount high vs framed empty

The current baseline bridge involving amount is `amount_high` vs
`framed_neutral`.

## Practical Rule

Use `CURRENT_PAPER/results` and `CURRENT_PAPER/features` for paper-facing amount
claims. Use `reruns/amount_*`, `trial_level_data/amount_scaleup_*`, and
`essay_all_conditions/amount` only as historical diagnostics unless a new
analysis explicitly promotes one of them into the paper-facing workflow.
