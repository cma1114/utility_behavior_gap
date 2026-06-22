# Framed User-Prompt-Strong Panel

Treatment: corrected framed-neutral user prompt plus task-specific max-effort cue appended to the user prompt.
Control: corrected framed-neutral prompt. Both arms use blank system prompts.

Primary score: strong win = +1, framed-neutral win = -1, tie = 0. Means are equal-cell means over actor-task cells.

## Overall

| group | n_cells | n_pairs | resolved_pairs | strong_wins | neutral_wins | ties | unresolved | pooled_strong_win_rate_excluding_ties | mean_cell_strong_net_score | cell_bootstrap_ci_lo | cell_bootstrap_ci_hi | mean_cell_word_diff_strong_minus_neutral |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 28 | 9450 | 9445 | 3650 | 1156 | 4639 | 5 | 0.759 | 0.283 | 0.225 | 0.343 | 22.493 |

## By Task

| task | task_label | n_cells | n_pairs | resolved_pairs | strong_wins | neutral_wins | ties | unresolved | pooled_strong_win_rate_excluding_ties | mean_cell_strong_net_score | cell_bootstrap_ci_lo | cell_bootstrap_ci_hi | mean_cell_word_diff_strong_minus_neutral |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| essay | Essay writing | 7 | 2100 | 2100 | 819 | 144 | 1137 | 0 | 0.850 | 0.321 | 0.210 | 0.434 | 16.516 |
| translation | Translation | 7 | 3150 | 3150 | 921 | 569 | 1660 | 0 | 0.618 | 0.112 | 0.054 | 0.166 | 0.334 |
| incident_postmortem | Incident postmortem | 7 | 2100 | 2100 | 970 | 224 | 906 | 0 | 0.812 | 0.355 | 0.260 | 0.467 | 61.194 |
| grant_proposal_abstract | Grant abstract | 7 | 2100 | 2095 | 940 | 219 | 936 | 5 | 0.811 | 0.344 | 0.259 | 0.424 | 11.929 |

## By Actor

| actor | actor_label | n_cells | n_pairs | resolved_pairs | strong_wins | neutral_wins | ties | unresolved | pooled_strong_win_rate_excluding_ties | mean_cell_strong_net_score | cell_bootstrap_ci_lo | cell_bootstrap_ci_hi | mean_cell_word_diff_strong_minus_neutral |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-v3.2-or | DeepSeek V3.2 | 4 | 1350 | 1345 | 564 | 129 | 652 | 5 | 0.814 | 0.351 | 0.191 | 0.460 | 48.542 |
| gpt-5.4-mini-or | GPT-5.4 mini | 4 | 1350 | 1350 | 442 | 155 | 753 | 0 | 0.740 | 0.242 | 0.057 | 0.383 | 8.985 |
| glm-5.1-or | GLM-5.1 | 4 | 1350 | 1350 | 606 | 124 | 620 | 0 | 0.830 | 0.376 | 0.273 | 0.478 | 14.241 |
| kimi-k2.5-or | Kimi K2.5 | 4 | 1350 | 1350 | 685 | 100 | 565 | 0 | 0.873 | 0.462 | 0.289 | 0.587 | 48.379 |
| mimo-v25-pro-or | MiMo V2.5 Pro | 4 | 1350 | 1350 | 514 | 266 | 570 | 0 | 0.659 | 0.196 | 0.125 | 0.263 | 26.069 |
| qwen3.5-9b-or | Qwen3.5 9B | 4 | 1350 | 1350 | 423 | 231 | 696 | 0 | 0.647 | 0.153 | 0.082 | 0.238 | 17.543 |
| qwen3.6-plus-or | Qwen3.6 Plus | 4 | 1350 | 1350 | 416 | 151 | 783 | 0 | 0.734 | 0.203 | 0.155 | 0.272 | -6.306 |
