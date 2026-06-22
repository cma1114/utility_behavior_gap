# Framed User-Prompt Mid-Effort Essay Dose Summary

Exploratory essay-only dose test. Treatment prompt appends: `This essay is extremely important - give your absolute best effort!` Control is the same framed-neutral user prompt/output used in the current headroom runs. System prompts are blank on both sides.

Counts are panel outcomes: treatment wins / neutral wins / ties. Win rates exclude ties. Holm p-values are one-sided binomial tests across the seven actors within each contrast.

## Pooled

| contrast | pairs | treatment_wins | neutral_wins | ties | win_rate_excluding_ties | net_score_all_pairs | one_sided_binomial_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mid_effort_vs_framed_neutral | 1050 | 236 | 178 | 636 | 0.570 | 0.055 | 0.003 |
| max_effort_vs_framed_neutral | 1050 | 410 | 72 | 568 | 0.851 | 0.322 | 0.000 |

## Matched Actor-Level Comparison

| actor | mid W/L/T | mid win rate | mid Holm p | mid net | max W/L/T | max win rate | max Holm p | max net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-v3.2-or | 38/36/76 | 0.514 | 1.000 | 0.013 | 69/4/77 | 0.945 | 0.000 | 0.433 |
| glm-5.1-or | 31/17/102 | 0.646 | 0.149 | 0.093 | 64/7/79 | 0.901 | 0.000 | 0.380 |
| gpt-5.4-mini-or | 40/17/93 | 0.702 | 0.010 | 0.153 | 67/1/82 | 0.985 | 0.000 | 0.440 |
| kimi-k2.5-or | 45/18/87 | 0.714 | 0.003 | 0.180 | 84/2/64 | 0.977 | 0.000 | 0.547 |
| mimo-v25-pro-or | 38/45/67 | 0.458 | 1.000 | -0.047 | 55/31/64 | 0.640 | 0.013 | 0.160 |
| qwen3.5-9b-or | 25/27/98 | 0.481 | 1.000 | -0.013 | 37/18/95 | 0.673 | 0.013 | 0.127 |
| qwen3.6-plus-or | 19/18/113 | 0.514 | 1.000 | 0.007 | 34/9/107 | 0.791 | 0.000 | 0.167 |

CSV: `outputs/analysis/framed_user_mid_effort_essay_dose_summary.csv`