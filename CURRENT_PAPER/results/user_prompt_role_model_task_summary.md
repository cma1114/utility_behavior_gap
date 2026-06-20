# World-class-role-side Versus Skilled-role-side

Comparison: `user_prompt_role`.
Left condition: `user_strong`. Right condition: `user_normal`.

Model-task CIs use exact binomial intervals Bonferroni-adjusted over 28 plotted/planned cells. Ties are excluded from displayed win rates.

Selected model-task run directories: 28 / 28.
Analyzable model-task cells with non-tied outcomes: 28 / 28.
Total completed pairs: 4192.
Total non-tied pairs: 1996.

Equal-cell mean win rate: 0.633.
FWER-positive cells: 11 / 28.

## Model-Task Cells

| task_label | actor_label | left_wins | right_wins | ties | unresolved | n_excluding_ties | left_win_rate_excluding_ties | familywise_ci_lo | familywise_ci_hi | familywise_ci_positive | holm_p_two_sided |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Essay writing | DeepSeek V3.2 | 39 | 9 | 102 | 0 | 48 | 0.812 | 0.589 | 0.947 | True | 0.000 |
| Essay writing | GLM-5.1 | 65 | 7 | 78 | 0 | 72 | 0.903 | 0.749 | 0.979 | True | 0.000 |
| Essay writing | GPT-5.4 mini | 116 | 0 | 34 | 0 | 116 | 1.000 | 0.941 | 1.000 | True | 0.000 |
| Essay writing | Kimi K2.5 | 43 | 11 | 96 | 0 | 54 | 0.796 | 0.585 | 0.932 | True | 0.000 |
| Essay writing | MiMo V2.5 Pro | 51 | 23 | 76 | 0 | 74 | 0.689 | 0.503 | 0.840 | True | 0.027 |
| Essay writing | Qwen3.5 9B | 23 | 27 | 100 | 0 | 50 | 0.460 | 0.248 | 0.683 | False | 1.000 |
| Essay writing | Qwen3.6 Plus | 45 | 9 | 96 | 0 | 54 | 0.833 | 0.629 | 0.953 | True | 0.000 |
| Grant abstract | DeepSeek V3.2 | 55 | 29 | 66 | 0 | 84 | 0.655 | 0.479 | 0.805 | False | 0.103 |
| Grant abstract | GLM-5.1 | 52 | 29 | 61 | 8 | 81 | 0.642 | 0.463 | 0.797 | False | 0.210 |
| Grant abstract | GPT-5.4 mini | 46 | 19 | 85 | 0 | 65 | 0.708 | 0.509 | 0.863 | True | 0.021 |
| Grant abstract | Kimi K2.5 | 61 | 26 | 63 | 0 | 87 | 0.701 | 0.531 | 0.840 | True | 0.005 |
| Grant abstract | MiMo V2.5 Pro | 53 | 42 | 55 | 0 | 95 | 0.558 | 0.394 | 0.713 | False | 1.000 |
| Grant abstract | Qwen3.5 9B | 27 | 34 | 89 | 0 | 61 | 0.443 | 0.251 | 0.647 | False | 1.000 |
| Grant abstract | Qwen3.6 Plus | 46 | 18 | 86 | 0 | 64 | 0.719 | 0.519 | 0.872 | True | 0.012 |
| Incident postmortem | DeepSeek V3.2 | 34 | 44 | 72 | 0 | 78 | 0.436 | 0.266 | 0.617 | False | 1.000 |
| Incident postmortem | GLM-5.1 | 53 | 20 | 77 | 0 | 73 | 0.726 | 0.540 | 0.869 | True | 0.003 |
| Incident postmortem | GPT-5.4 mini | 32 | 28 | 90 | 0 | 60 | 0.533 | 0.330 | 0.729 | False | 1.000 |
| Incident postmortem | Kimi K2.5 | 48 | 33 | 69 | 0 | 81 | 0.593 | 0.414 | 0.756 | False | 1.000 |
| Incident postmortem | MiMo V2.5 Pro | 70 | 28 | 52 | 0 | 98 | 0.714 | 0.555 | 0.843 | True | 0.001 |
| Incident postmortem | Qwen3.5 9B | 28 | 30 | 92 | 0 | 58 | 0.483 | 0.281 | 0.688 | False | 1.000 |
| Incident postmortem | Qwen3.6 Plus | 45 | 22 | 83 | 0 | 67 | 0.672 | 0.475 | 0.834 | False | 0.108 |
| Translation | DeepSeek V3.2 | 38 | 37 | 75 | 0 | 75 | 0.507 | 0.326 | 0.686 | False | 1.000 |
| Translation | GLM-5.1 | 31 | 27 | 92 | 0 | 58 | 0.534 | 0.327 | 0.734 | False | 1.000 |
| Translation | GPT-5.4 mini | 29 | 22 | 99 | 0 | 51 | 0.569 | 0.345 | 0.774 | False | 1.000 |
| Translation | Kimi K2.5 | 26 | 28 | 96 | 0 | 54 | 0.481 | 0.273 | 0.694 | False | 1.000 |
| Translation | MiMo V2.5 Pro | 42 | 34 | 74 | 0 | 76 | 0.553 | 0.370 | 0.726 | False | 1.000 |
| Translation | Qwen3.5 9B | 46 | 62 | 42 | 0 | 108 | 0.426 | 0.281 | 0.580 | False | 1.000 |
| Translation | Qwen3.6 Plus | 31 | 23 | 96 | 0 | 54 | 0.574 | 0.356 | 0.773 | False | 1.000 |

## Aggregate and Per-Task Means

| scope | task_label | left_wins | right_wins | ties | n_excluding_ties | pooled | equal_cell |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | All tasks | 1275 | 721 | 2196 | 1996 | 63.9% [61.7%, 66.0%] | 63.3% [53.0%, 75.7%] |
| task | Essay writing | 382 | 86 | 582 | 468 | 81.6% [77.9%, 84.9%] | 78.5% [65.0%, 90.0%] |
| task | Grant abstract | 340 | 197 | 505 | 537 | 63.3% [59.2%, 67.3%] | 63.2% [54.6%, 70.6%] |
| task | Incident postmortem | 310 | 205 | 535 | 515 | 60.2% [55.9%, 64.3%] | 59.4% [50.4%, 68.0%] |
| task | Translation | 243 | 233 | 574 | 476 | 51.1% [46.6%, 55.5%] | 52.1% [46.2%, 57.9%] |

## Run Audit

| actor | task | run_dir | expected_pairs | valid_pairs | complete_pairs | panel_user_strong | panel_user_normal | panel_tie | panel_unresolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-v3.2-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__user_prompt_role__deepseek-v3.2-or__2026-06-20_17-10-23Z__hash-e0eedfbe97df | 150 | 149 | 2 | 0 | 0 | 2 | 148 |
| deepseek-v3.2-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__user_prompt_role__deepseek-v3.2-or__2026-06-20_17-24-27Z__hash-e0eedfbe97df | 150 | 150 | 150 | 39 | 9 | 102 | 0 |
| glm-5.1-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__user_prompt_role__glm-5.1-or__2026-06-20_17-10-33Z__hash-5cc91aa949cf | 150 | 146 | 98 | 39 | 5 | 54 | 52 |
| glm-5.1-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__user_prompt_role__glm-5.1-or__2026-06-20_17-24-31Z__hash-5cc91aa949cf | 150 | 150 | 150 | 65 | 7 | 78 | 0 |
| gpt-5.4-mini-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__user_prompt_role__gpt-5.4-mini-or__2026-06-20_16-58-28Z__hash-958a1d578531 | 150 | 150 | 150 | 116 | 0 | 34 | 0 |
| kimi-k2.5-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__user_prompt_role__kimi-k2.5-or__2026-06-20_17-10-45Z__hash-9362ea34bf5a | 150 | 150 | 49 | 13 | 2 | 34 | 101 |
| kimi-k2.5-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__user_prompt_role__kimi-k2.5-or__2026-06-20_17-24-37Z__hash-9362ea34bf5a | 150 | 150 | 150 | 43 | 11 | 96 | 0 |
| mimo-v25-pro-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__user_prompt_role__mimo-v25-pro-or__2026-06-20_17-10-55Z__hash-e67181c7895f | 150 | 150 | 7 | 1 | 0 | 6 | 143 |
| mimo-v25-pro-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__user_prompt_role__mimo-v25-pro-or__2026-06-20_17-24-41Z__hash-e67181c7895f | 150 | 150 | 150 | 51 | 23 | 76 | 0 |
| qwen3.5-9b-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__user_prompt_role__qwen3.5-9b-or__2026-06-20_17-11-04Z__hash-aa44f3aec959 | 150 | 150 | 3 | 0 | 0 | 3 | 147 |
| qwen3.5-9b-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__user_prompt_role__qwen3.5-9b-or__2026-06-20_17-24-46Z__hash-aa44f3aec959 | 150 | 150 | 150 | 23 | 27 | 100 | 0 |
| qwen3.6-plus-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__user_prompt_role__qwen3.6-plus-or__2026-06-20_17-11-14Z__hash-9a0977ad62b1 | 150 | 150 | 110 | 29 | 5 | 76 | 40 |
| qwen3.6-plus-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__user_prompt_role__qwen3.6-plus-or__2026-06-20_17-24-50Z__hash-9a0977ad62b1 | 150 | 150 | 150 | 45 | 9 | 96 | 0 |
| deepseek-v3.2-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__user_prompt_role__deepseek-v3.2-or__2026-06-20_18-20-08Z__hash-3e0d51f38b3d | 150 | 150 | 150 | 55 | 29 | 66 | 0 |
| glm-5.1-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__user_prompt_role__glm-5.1-or__2026-06-20_18-36-12Z__hash-baf827bcd334 | 150 | 142 | 142 | 52 | 29 | 61 | 8 |
| gpt-5.4-mini-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__user_prompt_role__gpt-5.4-mini-or__2026-06-20_18-19-27Z__hash-829cd922dad6 | 150 | 150 | 150 | 46 | 19 | 85 | 0 |
| kimi-k2.5-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__user_prompt_role__kimi-k2.5-or__2026-06-20_18-20-28Z__hash-8d58261d2f60 | 150 | 150 | 150 | 61 | 26 | 63 | 0 |
| mimo-v25-pro-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__user_prompt_role__mimo-v25-pro-or__2026-06-20_18-20-39Z__hash-043a5a383970 | 150 | 150 | 150 | 53 | 42 | 55 | 0 |
| qwen3.5-9b-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__user_prompt_role__qwen3.5-9b-or__2026-06-20_18-20-50Z__hash-8353f913c92d | 150 | 150 | 150 | 27 | 34 | 89 | 0 |
| qwen3.6-plus-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__user_prompt_role__qwen3.6-plus-or__2026-06-20_18-21-00Z__hash-cbf14d2a0180 | 150 | 150 | 150 | 46 | 18 | 86 | 0 |
| deepseek-v3.2-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__user_prompt_role__deepseek-v3.2-or__2026-06-20_18-56-18Z__hash-c9d23ed8cc38 | 150 | 150 | 150 | 34 | 44 | 72 | 0 |
| glm-5.1-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__user_prompt_role__glm-5.1-or__2026-06-20_19-28-25Z__hash-170ef22f4ac4 | 150 | 150 | 150 | 53 | 20 | 77 | 0 |
| gpt-5.4-mini-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__user_prompt_role__gpt-5.4-mini-or__2026-06-20_18-55-34Z__hash-f4f0b03c17c8 | 150 | 150 | 150 | 32 | 28 | 90 | 0 |
| kimi-k2.5-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__user_prompt_role__kimi-k2.5-or__2026-06-20_18-56-26Z__hash-48bd6e207a01 | 150 | 150 | 150 | 48 | 33 | 69 | 0 |
| mimo-v25-pro-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__user_prompt_role__mimo-v25-pro-or__2026-06-20_18-56-56Z__hash-a7d287b836ea | 150 | 150 | 150 | 70 | 28 | 52 | 0 |
| qwen3.5-9b-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__user_prompt_role__qwen3.5-9b-or__2026-06-20_19-27-56Z__hash-7fd3e51efa51 | 150 | 150 | 150 | 28 | 30 | 92 | 0 |
| qwen3.6-plus-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__user_prompt_role__qwen3.6-plus-or__2026-06-20_18-56-55Z__hash-af1b1a892fe1 | 150 | 150 | 150 | 45 | 22 | 83 | 0 |
| deepseek-v3.2-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__user_prompt_role__deepseek-v3.2-or__2026-06-20_18-50-56Z__hash-132b1b231da4 | 150 | 150 | 150 | 38 | 37 | 75 | 0 |
| glm-5.1-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__user_prompt_role__glm-5.1-or__2026-06-20_18-51-14Z__hash-35f101e776da | 150 | 150 | 150 | 31 | 27 | 92 | 0 |
| gpt-5.4-mini-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__user_prompt_role__gpt-5.4-mini-or__2026-06-20_18-51-05Z__hash-15cfeeb1a17d | 150 | 150 | 150 | 29 | 22 | 99 | 0 |
| kimi-k2.5-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__user_prompt_role__kimi-k2.5-or__2026-06-20_18-51-25Z__hash-0494a0872a8a | 150 | 150 | 150 | 26 | 28 | 96 | 0 |
| mimo-v25-pro-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__user_prompt_role__mimo-v25-pro-or__2026-06-20_18-51-37Z__hash-d76af07cd970 | 150 | 150 | 150 | 42 | 34 | 74 | 0 |
| qwen3.5-9b-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__user_prompt_role__qwen3.5-9b-or__2026-06-20_18-51-47Z__hash-acbaef2c38c6 | 150 | 150 | 150 | 46 | 62 | 42 | 0 |
| qwen3.6-plus-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__user_prompt_role__qwen3.6-plus-or__2026-06-20_18-51-57Z__hash-b0256b716c80 | 150 | 150 | 150 | 31 | 23 | 96 | 0 |
