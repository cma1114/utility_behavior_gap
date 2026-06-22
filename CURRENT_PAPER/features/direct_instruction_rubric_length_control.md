# Direct Instruction Rubric Effects Controlling for Length

Question: does the strong direct-instruction arm score higher on each task-specific rubric dimension after controlling for word-count difference?

Model fit separately for each task and rubric dimension:

`rubric_score_strong_minus_neutral ~ delta_words + actor fixed effects`

`rubric_score` is +1 when the rubric coder preferred the strong output, 0 for tie/not-applicable, and -1 when it preferred the neutral output. The reported adjusted mean is the model-implied average rubric score at `delta_words = 0`, averaged over the observed actor mix.

Rubric input: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/task_rubric_feature_coding/direct_instruction__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash`
Word deltas: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_pair_deltas.csv`

## Essay writing

| dimension | n | unadjusted | adjusted effect | p | CI excludes 0 |
| --- | --- | --- | --- | --- | --- |
| Argument depth | 120 | +0.45 | +0.35 [+0.14, +0.55] | 0.000743 | True |
| Rhetorical coherence and closure | 120 | +0.34 | +0.24 [+0.05, +0.43] | 0.0123 | True |
| Thesis and stakes framing | 120 | +0.26 | +0.15 [-0.05, +0.35] | 0.139 | False |
| Concrete example quality | 120 | +0.25 | +0.13 [-0.08, +0.33] | 0.23 | False |
| Counterargument/qualification | 120 | +0.15 | +0.11 [-0.06, +0.29] | 0.205 | False |
| Avoids plausibility overreach | 120 | -0.01 | -0.01 [-0.03, +0.01] | 0.336 | False |

## Grant abstract

| dimension | n | unadjusted | adjusted effect | p | CI excludes 0 |
| --- | --- | --- | --- | --- | --- |
| Intervention specificity | 120 | +0.57 | +0.52 [+0.35, +0.69] | 3.44e-09 | True |
| Evaluation rigor | 120 | +0.55 | +0.51 [+0.34, +0.68] | 2.72e-09 | True |
| Risk mitigation | 120 | +0.53 | +0.49 [+0.31, +0.66] | 2.68e-08 | True |
| Stakeholder/context fit | 120 | +0.50 | +0.47 [+0.32, +0.62] | 5.74e-10 | True |
| Feasibility/readiness | 120 | +0.50 | +0.47 [+0.30, +0.63] | 3.3e-08 | True |
| Measurable impact | 120 | +0.48 | +0.45 [+0.28, +0.62] | 2.15e-07 | True |
| Problem significance | 120 | +0.47 | +0.42 [+0.27, +0.57] | 4.55e-08 | True |

## Incident postmortem

| dimension | n | unadjusted | adjusted effect | p | CI excludes 0 |
| --- | --- | --- | --- | --- | --- |
| Action-item concreteness | 119 | +0.51 | +0.36 [+0.12, +0.59] | 0.00314 | True |
| Impact specificity | 119 | +0.48 | +0.32 [+0.08, +0.56] | 0.00862 | True |
| Detection/observability analysis | 119 | +0.42 | +0.24 [+0.01, +0.48] | 0.0451 | True |
| Contributing-factor analysis | 119 | +0.46 | +0.23 [-0.01, +0.47] | 0.0626 | False |
| Root-cause specificity | 119 | +0.43 | +0.21 [-0.03, +0.46] | 0.0834 | False |
| Operational realism | 119 | +0.45 | +0.21 [-0.02, +0.45] | 0.0748 | False |
| Timeline precision | 119 | +0.35 | +0.20 [-0.05, +0.45] | 0.121 | False |
| Blameless systems framing | 119 | +0.13 | +0.03 [-0.06, +0.13] | 0.508 | False |

## Translation

| dimension | n | unadjusted | adjusted effect | p | CI excludes 0 |
| --- | --- | --- | --- | --- | --- |
| Fluency/idiomaticity | 120 | +0.21 | +0.24 [+0.10, +0.39] | 0.000725 | True |
| Structural clarity | 120 | +0.12 | +0.13 [+0.05, +0.22] | 0.00248 | True |
| Terminology precision | 120 | +0.10 | +0.11 [+0.00, +0.22] | 0.0465 | True |
| Named-entity fidelity | 120 | +0.03 | +0.05 [-0.07, +0.16] | 0.447 | False |
| Numeric/factual fidelity | 120 | +0.01 | +0.01 [-0.02, +0.04] | 0.606 | False |
| Avoids additions/omissions | 120 | -0.01 | -0.01 [-0.07, +0.05] | 0.726 | False |
