# Moral Low Vs Framed Empty Feature Sign Audit

This audit uses outputs/analysis/moral_low_vs_framed_empty_length_controlled_feature_selection.csv. The contrast is harmful/moral-low minus framed-empty; panel score is positive when harmful wins and negative when framed-empty wins.

The compact table showed only sign-aligned rows. For this contrast that is not the right explanatory filter, because harmful loses overall. Rows that explain the harmful-side loss are sign-misaligned: harmful has less of a feature the panel rewards, or more of a feature the panel penalizes.

## Counts
| task_label | harmful has feature difference in panel-disfavored direction | harmful has feature difference in panel-favored direction |
| --- | --- | --- |
| Essay writing | 2 | 3 |
| Grant abstract | 9 | 1 |
| Incident postmortem | 1 | 3 |
| Translation | 1 | 0 |

## Descriptive Product Sums
gap_effect_sd times panel_coef_per_sd; useful only as a sign/scale diagnostic because features are correlated.

| task_label | harmful has feature difference in panel-disfavored direction | harmful has feature difference in panel-favored direction |
| --- | --- | --- |
| Essay writing | -0.099 | 0.059 |
| Grant abstract | -0.535 | 0.02 |
| Incident postmortem | -0.09 | 0.118 |
| Translation | -0.003 | 0.0 |

## Clear Rows
| task_label | family | feature_id | feature_label | gap_effect | gap_ci_low | gap_ci_high | gap_effect_sd | gap_ci_low_sd | gap_ci_high_sd | panel_coef_per_sd | panel_ci_low | panel_ci_high | approx_contribution | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Essay writing | Generic text feature | words | Words | -14.996 | -16.381 | -13.611 | -0.408 | -0.445 | -0.37 | 0.233 | 0.187 | 0.279 | -0.095 | harmful has feature difference in panel-disfavored direction |
| Essay writing | Generic text feature | positive_words_per_1k | Positive-word rate per 1k words | 1.223 | 0.467 | 1.978 | 0.08 | 0.031 | 0.129 | -0.055 | -0.092 | -0.018 | -0.004 | harmful has feature difference in panel-disfavored direction |
| Grant abstract | Generic text feature | words | Words | -19.969 | -22.478 | -17.46 | -0.341 | -0.384 | -0.298 | 0.218 | 0.176 | 0.26 | -0.074 | harmful has feature difference in panel-disfavored direction |
| Grant abstract | LLM rubric marker | measurable_impact | Measurable impact | -0.262 | -0.438 | -0.086 | -0.289 | -0.484 | -0.095 | 0.347 | 0.156 | 0.537 | -0.1 | harmful has feature difference in panel-disfavored direction |
| Grant abstract | LLM rubric marker | risk_mitigation | Risk mitigation | -0.26 | -0.441 | -0.079 | -0.276 | -0.469 | -0.083 | 0.386 | 0.192 | 0.58 | -0.107 | harmful has feature difference in panel-disfavored direction |
| Grant abstract | LLM rubric marker | evaluation_rigor | Evaluation rigor | -0.254 | -0.442 | -0.065 | -0.262 | -0.457 | -0.067 | 0.242 | 0.06 | 0.423 | -0.063 | harmful has feature difference in panel-disfavored direction |
| Grant abstract | LLM rubric marker | stakeholder_context_fit | Stakeholder/context fit | -0.225 | -0.388 | -0.061 | -0.26 | -0.449 | -0.071 | 0.288 | 0.097 | 0.48 | -0.075 | harmful has feature difference in panel-disfavored direction |
| Grant abstract | LLM rubric marker | intervention_specificity | Intervention specificity | -0.217 | -0.401 | -0.033 | -0.228 | -0.42 | -0.035 | 0.284 | 0.092 | 0.476 | -0.065 | harmful has feature difference in panel-disfavored direction |
| Grant abstract | Generic text feature | mattr_50 | MATTR-50 | -0.003 | -0.004 | -0.002 | -0.168 | -0.215 | -0.121 | 0.142 | 0.102 | 0.181 | -0.024 | harmful has feature difference in panel-disfavored direction |
| Grant abstract | Generic text feature | rare_word_rate_per_1k | Rare-word rate per 1k words | -2.018 | -2.974 | -1.062 | -0.105 | -0.155 | -0.056 | 0.15 | 0.11 | 0.19 | -0.016 | harmful has feature difference in panel-disfavored direction |
| Grant abstract | Generic text feature | quantitative_detail_rate | Quantitative detail rate | -0.423 | -0.819 | -0.028 | -0.054 | -0.105 | -0.004 | 0.2 | 0.161 | 0.238 | -0.011 | harmful has feature difference in panel-disfavored direction |
| Incident postmortem | Generic text feature | words | Words | -24.616 | -28.314 | -20.918 | -0.29 | -0.333 | -0.246 | 0.31 | 0.271 | 0.349 | -0.09 | harmful has feature difference in panel-disfavored direction |
| Translation | Generic text feature | mattr_50 | MATTR-50 | -0.002 | -0.003 | -0.0 | -0.045 | -0.09 | -0.0 | 0.066 | 0.024 | 0.108 | -0.003 | harmful has feature difference in panel-disfavored direction |
| Essay writing | Generic text feature | paragraphs | Paragraphs | 0.343 | 0.294 | 0.392 | 0.317 | 0.272 | 0.363 | 0.083 | 0.04 | 0.125 | 0.026 | harmful has feature difference in panel-favored direction |
| Essay writing | Generic text feature | textstat_flesch_kincaid_grade | Flesch-Kincaid grade | 0.323 | 0.26 | 0.385 | 0.258 | 0.208 | 0.308 | 0.083 | 0.045 | 0.12 | 0.021 | harmful has feature difference in panel-favored direction |
| Essay writing | Generic text feature | mattr_50 | MATTR-50 | 0.001 | 0.0 | 0.003 | 0.068 | 0.018 | 0.118 | 0.169 | 0.131 | 0.206 | 0.011 | harmful has feature difference in panel-favored direction |
| Grant abstract | Generic text feature | textstat_flesch_kincaid_grade | Flesch-Kincaid grade | 0.304 | 0.234 | 0.374 | 0.212 | 0.163 | 0.261 | 0.094 | 0.054 | 0.133 | 0.02 | harmful has feature difference in panel-favored direction |
| Incident postmortem | LLM rubric marker | operational_realism | Operational realism | 0.2 | 0.031 | 0.369 | 0.211 | 0.033 | 0.389 | 0.438 | 0.259 | 0.617 | 0.093 | harmful has feature difference in panel-favored direction |
| Incident postmortem | Generic text feature | quantitative_detail_rate | Quantitative detail rate | 1.365 | 0.517 | 2.213 | 0.072 | 0.027 | 0.117 | 0.272 | 0.232 | 0.311 | 0.02 | harmful has feature difference in panel-favored direction |
| Incident postmortem | Generic text feature | rare_word_rate_per_1k | Rare-word rate per 1k words | 0.985 | 0.061 | 1.909 | 0.049 | 0.003 | 0.095 | 0.122 | 0.082 | 0.163 | 0.006 | harmful has feature difference in panel-favored direction |

CSV: outputs/analysis/moral_low_vs_framed_empty_feature_sign_audit.csv
