# World-class role cue Versus Skilled role cue

Comparison: `framed_user_prompt_role`.
Left condition: `user_strong`. Right condition: `user_normal`.

Model-task CIs use exact binomial intervals Bonferroni-adjusted over 28 plotted/planned cells. Ties are excluded from displayed win rates.

Selected model-task run directories: 28 / 28.
Analyzable model-task cells with non-tied outcomes: 28 / 28.
Total completed pairs: 8369.
Total non-tied pairs: 3755.

Equal-cell mean win rate: 0.612.
FWER-positive cells: 10 / 28.

## Model-Task Cells

| task_label | actor_label | left_wins | right_wins | ties | unresolved | n_excluding_ties | left_win_rate_excluding_ties | familywise_ci_lo | familywise_ci_hi | familywise_ci_positive | holm_p_two_sided |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Essay writing | DeepSeek V3.2 | 90 | 38 | 172 | 0 | 128 | 0.703 | 0.565 | 0.820 | True | 0.000 |
| Essay writing | GLM-5.1 | 98 | 23 | 177 | 2 | 121 | 0.810 | 0.678 | 0.906 | True | 0.000 |
| Essay writing | GPT-5.4 mini | 123 | 12 | 165 | 0 | 135 | 0.911 | 0.810 | 0.970 | True | 0.000 |
| Essay writing | Kimi K2.5 | 73 | 19 | 208 | 0 | 92 | 0.793 | 0.637 | 0.905 | True | 0.000 |
| Essay writing | MiMo V2.5 Pro | 102 | 63 | 134 | 1 | 165 | 0.618 | 0.494 | 0.732 | False | 0.054 |
| Essay writing | Qwen3.5 9B | 56 | 42 | 202 | 0 | 98 | 0.571 | 0.410 | 0.723 | False | 1.000 |
| Essay writing | Qwen3.6 Plus | 76 | 19 | 205 | 0 | 95 | 0.800 | 0.648 | 0.909 | True | 0.000 |
| Grant abstract | DeepSeek V3.2 | 75 | 51 | 174 | 0 | 126 | 0.595 | 0.452 | 0.728 | False | 0.601 |
| Grant abstract | GLM-5.1 | 115 | 71 | 106 | 8 | 186 | 0.618 | 0.502 | 0.726 | True | 0.029 |
| Grant abstract | GPT-5.4 mini | 58 | 42 | 200 | 0 | 100 | 0.580 | 0.420 | 0.729 | False | 1.000 |
| Grant abstract | Kimi K2.5 | 88 | 49 | 162 | 1 | 137 | 0.642 | 0.506 | 0.764 | True | 0.022 |
| Grant abstract | MiMo V2.5 Pro | 102 | 79 | 117 | 2 | 181 | 0.564 | 0.445 | 0.677 | False | 1.000 |
| Grant abstract | Qwen3.5 9B | 67 | 43 | 190 | 0 | 110 | 0.609 | 0.456 | 0.748 | False | 0.446 |
| Grant abstract | Qwen3.6 Plus | 93 | 51 | 156 | 0 | 144 | 0.646 | 0.513 | 0.764 | True | 0.012 |
| Incident postmortem | DeepSeek V3.2 | 79 | 78 | 141 | 2 | 157 | 0.503 | 0.377 | 0.629 | False | 1.000 |
| Incident postmortem | GLM-5.1 | 100 | 56 | 134 | 10 | 156 | 0.641 | 0.514 | 0.756 | True | 0.012 |
| Incident postmortem | GPT-5.4 mini | 59 | 72 | 169 | 0 | 131 | 0.450 | 0.316 | 0.590 | False | 1.000 |
| Incident postmortem | Kimi K2.5 | 71 | 62 | 167 | 0 | 133 | 0.534 | 0.396 | 0.668 | False | 1.000 |
| Incident postmortem | MiMo V2.5 Pro | 118 | 78 | 104 | 0 | 196 | 0.602 | 0.488 | 0.709 | False | 0.089 |
| Incident postmortem | Qwen3.5 9B | 64 | 51 | 185 | 0 | 115 | 0.557 | 0.408 | 0.699 | False | 1.000 |
| Incident postmortem | Qwen3.6 Plus | 70 | 53 | 177 | 0 | 123 | 0.569 | 0.425 | 0.706 | False | 1.000 |
| Translation | DeepSeek V3.2 | 64 | 58 | 176 | 2 | 122 | 0.525 | 0.381 | 0.665 | False | 1.000 |
| Translation | GLM-5.1 | 77 | 36 | 185 | 2 | 113 | 0.681 | 0.532 | 0.808 | True | 0.003 |
| Translation | GPT-5.4 mini | 51 | 58 | 191 | 0 | 109 | 0.468 | 0.320 | 0.620 | False | 1.000 |
| Translation | Kimi K2.5 | 60 | 65 | 175 | 0 | 125 | 0.480 | 0.341 | 0.622 | False | 1.000 |
| Translation | MiMo V2.5 Pro | 77 | 66 | 157 | 0 | 143 | 0.538 | 0.405 | 0.668 | False | 1.000 |
| Translation | Qwen3.5 9B | 110 | 93 | 96 | 1 | 203 | 0.542 | 0.430 | 0.651 | False | 1.000 |
| Translation | Qwen3.6 Plus | 64 | 47 | 189 | 0 | 111 | 0.577 | 0.425 | 0.719 | False | 1.000 |

## Aggregate and Per-Task Means

| scope | task_label | left_wins | right_wins | ties | n_excluding_ties | pooled | equal_cell |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | All tasks | 2280 | 1475 | 4614 | 3755 | 60.7% [59.1%, 62.3%] | 61.2% [53.2%, 70.9%] |
| task | Essay writing | 618 | 216 | 1263 | 834 | 74.1% [71.0%, 77.0%] | 74.4% [65.5%, 82.7%] |
| task | Grant abstract | 598 | 386 | 1105 | 984 | 60.8% [57.7%, 63.8%] | 60.8% [57.1%, 64.5%] |
| task | Incident postmortem | 561 | 450 | 1077 | 1011 | 55.5% [52.4%, 58.5%] | 55.1% [49.7%, 60.3%] |
| task | Translation | 503 | 423 | 1169 | 926 | 54.3% [51.1%, 57.5%] | 54.4% [48.8%, 60.7%] |

## Run Audit

| actor | task | run_dir | expected_pairs | valid_pairs | complete_pairs | panel_user_strong | panel_user_normal | panel_tie | panel_unresolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-v3.2-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__framed_user_prompt_role__deepseek-v3.2-or__2026-06-21_00-41-04Z__hash-3e1c3dcc3c45 | 150 | 150 | 150 | 45 | 16 | 89 | 0 |
| deepseek-v3.2-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__framed_user_prompt_role__deepseek-v3.2-or__2026-06-21_04-07-18Z__hash-c1a0f249ff74 | 150 | 150 | 150 | 45 | 22 | 83 | 0 |
| glm-5.1-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__framed_user_prompt_role__glm-5.1-or__2026-06-21_00-41-30Z__hash-c642903312de | 150 | 150 | 150 | 48 | 11 | 91 | 0 |
| glm-5.1-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__framed_user_prompt_role__glm-5.1-or__2026-06-21_04-07-49Z__hash-1c96df57442b | 150 | 148 | 148 | 50 | 12 | 86 | 2 |
| gpt-5.4-mini-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__framed_user_prompt_role__gpt-5.4-mini-or__2026-06-21_00-16-44Z__hash-b7363d777c17 | 150 | 150 | 150 | 55 | 9 | 86 | 0 |
| gpt-5.4-mini-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__framed_user_prompt_role__gpt-5.4-mini-or__2026-06-21_04-07-30Z__hash-5ae97cc759c5 | 150 | 150 | 150 | 68 | 3 | 79 | 0 |
| kimi-k2.5-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__framed_user_prompt_role__kimi-k2.5-or__2026-06-21_00-41-39Z__hash-f0b0df2d5052 | 150 | 150 | 150 | 47 | 9 | 94 | 0 |
| kimi-k2.5-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__framed_user_prompt_role__kimi-k2.5-or__2026-06-21_04-08-02Z__hash-b26912f9c9f0 | 150 | 150 | 150 | 26 | 10 | 114 | 0 |
| mimo-v25-pro-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__framed_user_prompt_role__mimo-v25-pro-or__2026-06-21_00-41-51Z__hash-9eee9dc1472f | 150 | 150 | 150 | 56 | 29 | 65 | 0 |
| mimo-v25-pro-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__framed_user_prompt_role__mimo-v25-pro-or__2026-06-21_04-08-14Z__hash-a8cb77161e6f | 150 | 149 | 149 | 46 | 34 | 69 | 1 |
| qwen3.5-9b-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__framed_user_prompt_role__qwen3.5-9b-or__2026-06-21_00-42-03Z__hash-f3f591fe08cf | 150 | 150 | 150 | 38 | 17 | 95 | 0 |
| qwen3.5-9b-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__framed_user_prompt_role__qwen3.5-9b-or__2026-06-21_04-08-25Z__hash-1a89e84a517c | 150 | 150 | 150 | 18 | 25 | 107 | 0 |
| qwen3.6-plus-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__framed_user_prompt_role__qwen3.6-plus-or__2026-06-21_00-42-13Z__hash-7fcf259cbef2 | 150 | 150 | 150 | 33 | 11 | 106 | 0 |
| qwen3.6-plus-or | essay | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/essay__framed_user_prompt_role__qwen3.6-plus-or__2026-06-21_04-08-34Z__hash-bc740c1487f6 | 150 | 150 | 150 | 43 | 8 | 99 | 0 |
| deepseek-v3.2-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__framed_user_prompt_role__deepseek-v3.2-or__2026-06-21_01-12-15Z__hash-fda380bf80ec | 150 | 150 | 150 | 41 | 26 | 83 | 0 |
| deepseek-v3.2-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__framed_user_prompt_role__deepseek-v3.2-or__2026-06-21_04-28-35Z__hash-f9ef5570c321 | 150 | 150 | 150 | 34 | 25 | 91 | 0 |
| glm-5.1-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__framed_user_prompt_role__glm-5.1-or__2026-06-21_01-11-31Z__hash-155bcb1e2fb4 | 150 | 145 | 145 | 57 | 33 | 55 | 5 |
| glm-5.1-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__framed_user_prompt_role__glm-5.1-or__2026-06-21_04-25-57Z__hash-b87630d97fe2 | 150 | 147 | 147 | 58 | 38 | 51 | 3 |
| gpt-5.4-mini-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__framed_user_prompt_role__gpt-5.4-mini-or__2026-06-21_00-41-18Z__hash-dd4659338406 | 150 | 150 | 150 | 33 | 21 | 96 | 0 |
| gpt-5.4-mini-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__framed_user_prompt_role__gpt-5.4-mini-or__2026-06-21_04-18-43Z__hash-512230f19fe3 | 150 | 150 | 150 | 25 | 21 | 104 | 0 |
| kimi-k2.5-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__framed_user_prompt_role__kimi-k2.5-or__2026-06-21_01-14-57Z__hash-c715def3cb5a | 150 | 150 | 150 | 43 | 28 | 79 | 0 |
| kimi-k2.5-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__framed_user_prompt_role__kimi-k2.5-or__2026-06-21_04-28-57Z__hash-0e10391fa849 | 150 | 149 | 149 | 45 | 21 | 83 | 1 |
| mimo-v25-pro-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__framed_user_prompt_role__mimo-v25-pro-or__2026-06-21_01-15-51Z__hash-3d27f85ee0cf | 150 | 148 | 148 | 50 | 35 | 63 | 2 |
| mimo-v25-pro-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__framed_user_prompt_role__mimo-v25-pro-or__2026-06-21_04-31-46Z__hash-d8ef4d8d2723 | 150 | 150 | 150 | 52 | 44 | 54 | 0 |
| qwen3.5-9b-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__framed_user_prompt_role__qwen3.5-9b-or__2026-06-21_01-14-09Z__hash-88d2532137cd | 150 | 150 | 150 | 32 | 20 | 98 | 0 |
| qwen3.5-9b-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__framed_user_prompt_role__qwen3.5-9b-or__2026-06-21_04-30-14Z__hash-75f98aee58c2 | 150 | 150 | 150 | 35 | 23 | 92 | 0 |
| qwen3.6-plus-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__framed_user_prompt_role__qwen3.6-plus-or__2026-06-21_01-13-53Z__hash-e747425f2b2c | 150 | 150 | 150 | 43 | 29 | 78 | 0 |
| qwen3.6-plus-or | grant_proposal_abstract | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/grant_proposal_abstract__framed_user_prompt_role__qwen3.6-plus-or__2026-06-21_04-27-47Z__hash-3c7f4afc0797 | 150 | 150 | 150 | 50 | 22 | 78 | 0 |
| deepseek-v3.2-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__framed_user_prompt_role__deepseek-v3.2-or__2026-06-21_01-48-28Z__hash-adc847d791d1 | 150 | 148 | 148 | 40 | 35 | 73 | 2 |
| deepseek-v3.2-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__framed_user_prompt_role__deepseek-v3.2-or__2026-06-21_05-08-51Z__hash-18a0d6e5dfe0 | 150 | 150 | 150 | 39 | 43 | 68 | 0 |
| glm-5.1-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__framed_user_prompt_role__glm-5.1-or__2026-06-21_01-43-04Z__hash-ff6042806f13 | 150 | 144 | 144 | 52 | 29 | 63 | 6 |
| glm-5.1-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__framed_user_prompt_role__glm-5.1-or__2026-06-21_04-55-49Z__hash-a287abf4fba4 | 150 | 146 | 146 | 48 | 27 | 71 | 4 |
| gpt-5.4-mini-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__framed_user_prompt_role__gpt-5.4-mini-or__2026-06-21_00-59-23Z__hash-3c99091da3fa | 150 | 150 | 150 | 31 | 40 | 79 | 0 |
| gpt-5.4-mini-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__framed_user_prompt_role__gpt-5.4-mini-or__2026-06-21_04-36-24Z__hash-93257a984d7a | 150 | 150 | 150 | 28 | 32 | 90 | 0 |
| kimi-k2.5-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__framed_user_prompt_role__kimi-k2.5-or__2026-06-21_01-50-52Z__hash-9294ab0166c6 | 150 | 150 | 150 | 35 | 31 | 84 | 0 |
| kimi-k2.5-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__framed_user_prompt_role__kimi-k2.5-or__2026-06-21_05-00-22Z__hash-d0e07b26f6e0 | 150 | 150 | 150 | 36 | 31 | 83 | 0 |
| mimo-v25-pro-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__framed_user_prompt_role__mimo-v25-pro-or__2026-06-21_01-51-15Z__hash-137a88250d8c | 150 | 150 | 150 | 66 | 41 | 43 | 0 |
| mimo-v25-pro-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__framed_user_prompt_role__mimo-v25-pro-or__2026-06-21_05-09-58Z__hash-074f9c8930b8 | 150 | 150 | 150 | 52 | 37 | 61 | 0 |
| qwen3.5-9b-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__framed_user_prompt_role__qwen3.5-9b-or__2026-06-21_01-39-58Z__hash-c1cb1fbc1c54 | 150 | 150 | 150 | 32 | 24 | 94 | 0 |
| qwen3.5-9b-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__framed_user_prompt_role__qwen3.5-9b-or__2026-06-21_05-07-55Z__hash-86abcf97a7f0 | 150 | 150 | 150 | 32 | 27 | 91 | 0 |
| qwen3.6-plus-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__framed_user_prompt_role__qwen3.6-plus-or__2026-06-21_01-45-08Z__hash-4e0f8f7f6e3f | 150 | 150 | 150 | 32 | 30 | 88 | 0 |
| qwen3.6-plus-or | incident_postmortem | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/incident_postmortem__framed_user_prompt_role__qwen3.6-plus-or__2026-06-21_04-58-46Z__hash-9cc0ff858c46 | 150 | 150 | 150 | 38 | 23 | 89 | 0 |
| deepseek-v3.2-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__framed_user_prompt_role__deepseek-v3.2-or__2026-06-21_02-54-08Z__hash-06073082262b | 150 | 148 | 148 | 31 | 29 | 88 | 2 |
| deepseek-v3.2-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__framed_user_prompt_role__deepseek-v3.2-or__2026-06-21_06-20-25Z__hash-384f340decd4 | 150 | 150 | 150 | 33 | 29 | 88 | 0 |
| glm-5.1-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__framed_user_prompt_role__glm-5.1-or__2026-06-21_02-20-25Z__hash-d475fb8cefa0 | 150 | 148 | 148 | 33 | 16 | 99 | 2 |
| glm-5.1-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__framed_user_prompt_role__glm-5.1-or__2026-06-21_05-35-16Z__hash-3bb9cbf86404 | 150 | 150 | 150 | 44 | 20 | 86 | 0 |
| gpt-5.4-mini-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__framed_user_prompt_role__gpt-5.4-mini-or__2026-06-21_01-24-51Z__hash-00ff46c00eb7 | 150 | 150 | 150 | 21 | 34 | 95 | 0 |
| gpt-5.4-mini-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__framed_user_prompt_role__gpt-5.4-mini-or__2026-06-21_04-56-32Z__hash-bb3d4a47118b | 150 | 150 | 150 | 30 | 24 | 96 | 0 |
| kimi-k2.5-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__framed_user_prompt_role__kimi-k2.5-or__2026-06-21_02-38-43Z__hash-3f81c47527c7 | 150 | 150 | 150 | 34 | 30 | 86 | 0 |
| kimi-k2.5-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__framed_user_prompt_role__kimi-k2.5-or__2026-06-21_05-56-04Z__hash-f4ae42e04d0a | 150 | 150 | 150 | 26 | 35 | 89 | 0 |
| mimo-v25-pro-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__framed_user_prompt_role__mimo-v25-pro-or__2026-06-21_02-38-28Z__hash-e570a8f605ed | 150 | 150 | 150 | 36 | 35 | 79 | 0 |
| mimo-v25-pro-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__framed_user_prompt_role__mimo-v25-pro-or__2026-06-21_05-57-39Z__hash-6a15a6a6fc2c | 150 | 150 | 150 | 41 | 31 | 78 | 0 |
| qwen3.5-9b-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__framed_user_prompt_role__qwen3.5-9b-or__2026-06-21_02-17-45Z__hash-039550f5396a | 150 | 149 | 149 | 58 | 47 | 44 | 1 |
| qwen3.5-9b-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__framed_user_prompt_role__qwen3.5-9b-or__2026-06-21_05-55-24Z__hash-aa0ad1c2fefb | 150 | 150 | 150 | 52 | 46 | 52 | 0 |
| qwen3.6-plus-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__framed_user_prompt_role__qwen3.6-plus-or__2026-06-21_02-24-53Z__hash-22a3fad454a1 | 150 | 150 | 150 | 33 | 25 | 92 | 0 |
| qwen3.6-plus-or | translation | /Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/translation__framed_user_prompt_role__qwen3.6-plus-or__2026-06-21_05-38-38Z__hash-faa050554750 | 150 | 150 | 150 | 31 | 22 | 97 | 0 |
