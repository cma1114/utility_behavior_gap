# Length-Controlled Feature Selection: direct_instruction

Selection rule: keep features whose left-minus-right arm gap excludes zero, whose panel association excludes zero, and whose signs align. Non-word features control for word-count difference in both tests. Word count itself is included when its raw arm gap and panel association are both clear.

Gap model for non-word features: `feature_delta ~ delta_words + actor fixed effects`.
Panel model for non-word features: `panel_score ~ standardized(feature_delta) + standardized(delta_words) + actor fixed effects`.
Word-count models omit the word-count control because word count is the feature.
The standardized arm gap divides the adjusted arm gap by the observed SD of the paired feature deltas within the same task. It is descriptive scale information, not a separate significance test.

Pair deltas: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/final_text_analysis_pair_deltas.csv`
Rubric run: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/task_rubric_feature_coding/direct_instruction__source-pair-deltas__n-per-task-120__seed-20260615__coder-google-gemini-2.5-flash`

## Essay writing

| family | feature | n gap | arm gap | arm gap, SD | n panel | panel assoc. |
| --- | --- | --- | --- | --- | --- | --- |
| Generic text feature | Words | 2100 | +17.03 [+15.51, +18.55] | +0.46 [+0.42, +0.50] | 2100 | +0.20 [+0.18, +0.23] |
| Generic text feature | Rare-word rate per 1k words | 2100 | +10.37 [+8.45, +12.29] | +0.47 [+0.38, +0.55] | 2100 | +0.10 [+0.07, +0.12] |
| Generic text feature | Positive-word rate per 1k words | 2100 | -2.64 [-3.41, -1.86] | -0.17 [-0.22, -0.12] | 2100 | -0.05 [-0.08, -0.03] |
| LLM rubric marker | Argument depth | 120 | +0.35 [+0.14, +0.55] | +0.39 [+0.16, +0.62] | 120 | +0.12 [+0.02, +0.22] |
| LLM rubric marker | Rhetorical coherence and closure | 120 | +0.24 [+0.05, +0.43] | +0.29 [+0.06, +0.52] | 120 | +0.13 [+0.02, +0.24] |

## Grant abstract

| family | feature | n gap | arm gap | arm gap, SD | n panel | panel assoc. |
| --- | --- | --- | --- | --- | --- | --- |
| Generic text feature | Words | 2095 | +14.43 [+12.00, +16.85] | +0.25 [+0.20, +0.29] | 2095 | +0.20 [+0.17, +0.23] |
| Generic text feature | Rare-word rate per 1k words | 2095 | +9.69 [+8.86, +10.51] | +0.51 [+0.47, +0.56] | 2095 | +0.15 [+0.12, +0.18] |
| Generic text feature | Quantitative detail rate | 2095 | +0.99 [+0.64, +1.33] | +0.13 [+0.08, +0.17] | 2095 | +0.22 [+0.19, +0.24] |
| Generic text feature | Flesch-Kincaid grade | 2095 | +0.33 [+0.27, +0.40] | +0.22 [+0.18, +0.27] | 2095 | +0.08 [+0.06, +0.11] |
| Generic text feature | MATTR-50 | 2095 | +0.01 [+0.00, +0.01] | +0.33 [+0.28, +0.37] | 2095 | +0.13 [+0.10, +0.16] |
| LLM rubric marker | Intervention specificity | 120 | +0.52 [+0.35, +0.69] | +0.63 [+0.42, +0.84] | 120 | +0.20 [+0.08, +0.32] |
| LLM rubric marker | Evaluation rigor | 120 | +0.51 [+0.34, +0.68] | +0.61 [+0.41, +0.81] | 120 | +0.20 [+0.07, +0.34] |
| LLM rubric marker | Risk mitigation | 120 | +0.49 [+0.31, +0.66] | +0.57 [+0.37, +0.77] | 120 | +0.25 [+0.13, +0.38] |
| LLM rubric marker | Stakeholder/context fit | 120 | +0.47 [+0.32, +0.62] | +0.63 [+0.43, +0.83] | 120 | +0.21 [+0.08, +0.33] |
| LLM rubric marker | Feasibility/readiness | 120 | +0.47 [+0.30, +0.63] | +0.56 [+0.36, +0.76] | 120 | +0.25 [+0.13, +0.38] |
| LLM rubric marker | Measurable impact | 120 | +0.45 [+0.28, +0.62] | +0.53 [+0.33, +0.74] | 120 | +0.24 [+0.11, +0.37] |
| LLM rubric marker | Problem significance | 120 | +0.42 [+0.27, +0.57] | +0.57 [+0.36, +0.77] | 120 | +0.18 [+0.04, +0.31] |

## Incident postmortem

| family | feature | n gap | arm gap | arm gap, SD | n panel | panel assoc. |
| --- | --- | --- | --- | --- | --- | --- |
| Generic text feature | Words | 2100 | +66.37 [+62.12, +70.62] | +0.58 [+0.54, +0.62] | 2100 | +0.26 [+0.22, +0.30] |
| Generic text feature | Rare-word rate per 1k words | 2100 | +5.67 [+4.32, +7.01] | +0.29 [+0.22, +0.36] | 2100 | +0.12 [+0.10, +0.15] |
| Generic text feature | Quantitative detail rate | 2100 | +1.76 [+0.84, +2.68] | +0.10 [+0.05, +0.15] | 2100 | +0.23 [+0.20, +0.25] |
| Generic text feature | Paragraphs | 2100 | +0.31 [+0.03, +0.60] | +0.05 [+0.01, +0.10] | 2100 | +0.10 [+0.07, +0.13] |
| Generic text feature | MATTR-50 | 2100 | +0.01 [+0.00, +0.01] | +0.28 [+0.22, +0.35] | 2100 | +0.06 [+0.03, +0.09] |
| LLM rubric marker | Action-item concreteness | 119 | +0.36 [+0.12, +0.59] | +0.41 [+0.14, +0.69] | 119 | +0.17 [+0.05, +0.30] |
| LLM rubric marker | Impact specificity | 119 | +0.32 [+0.08, +0.56] | +0.37 [+0.09, +0.64] | 119 | +0.16 [+0.03, +0.29] |
| LLM rubric marker | Detection/observability analysis | 119 | +0.24 [+0.01, +0.48] | +0.27 [+0.01, +0.52] | 119 | +0.15 [+0.03, +0.27] |

## Translation

| family | feature | n gap | arm gap | arm gap, SD | n panel | panel assoc. |
| --- | --- | --- | --- | --- | --- | --- |
| Generic text feature | Rare-word rate per 1k words | 3150 | +2.23 [+1.30, +3.16] | +0.08 [+0.04, +0.11] | 3150 | +0.04 [+0.00, +0.08] |
| Generic text feature | Words | 3150 | +0.45 [+0.28, +0.63] | +0.09 [+0.06, +0.13] | 3150 | +0.03 [+0.00, +0.06] |
| Generic text feature | MATTR-50 | 3150 | +0.00 [+0.00, +0.01] | +0.12 [+0.08, +0.15] | 3150 | +0.07 [+0.04, +0.09] |
| LLM rubric marker | Fluency/idiomaticity | 120 | +0.24 [+0.10, +0.39] | +0.31 [+0.13, +0.48] | 120 | +0.32 [+0.19, +0.44] |
| LLM rubric marker | Structural clarity | 120 | +0.13 [+0.05, +0.22] | +0.28 [+0.10, +0.46] | 120 | +0.16 [+0.02, +0.29] |
| LLM rubric marker | Terminology precision | 120 | +0.11 [+0.00, +0.22] | +0.19 [+0.00, +0.37] | 120 | +0.21 [+0.09, +0.33] |

## Output Files

- selected/full CSV: `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/analysis/direct_instruction_length_controlled_feature_selection.csv`

Selected rows: `31` of `59` tested feature-task rows.
