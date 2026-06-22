# High Utility Versus Framed Empty

Source bridge runs:

- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-framed-empty-bridge__2026-06-17_05-06-34Z__hash-b7900b37796a`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-framed-empty-bridge__2026-06-17_02-04-10Z__hash-1fd41e5a6d38`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-framed-empty-bridge__2026-06-17_05-01-30Z__hash-89a63352882b`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-framed-empty-bridge__2026-06-17_04-53-06Z__hash-c2a42490a03b`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-framed-empty-bridge__2026-06-17_05-03-45Z__hash-db20d31049c2`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-framed-empty-bridge__2026-06-17_02-19-15Z__hash-d6e5f556b8da`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-framed-empty-bridge__2026-06-17_02-21-29Z__hash-4281b28144a8`

Primary estimate: equal-cell mean over actor x task x domain cells. Panel ties are excluded from the win-rate denominator and reported separately.

## Overall

| resolved | high wins | framed-empty wins | ties | pooled | equal-cell mean | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 7780 | 3491 | 4289 | 620 | 44.9% | 45.0% | 42.4%-47.6% |

## By Task

| task | resolved | high wins | framed-empty wins | ties | equal-cell mean | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Essay writing | 1802 | 808 | 994 | 298 | 45.0% | 39.3%-50.4% |
| Grant abstract | 2098 | 744 | 1354 | 2 | 35.5% | 31.0%-40.2% |
| Incident postmortem | 2096 | 1102 | 994 | 4 | 52.6% | 47.7%-57.4% |
| Translation | 1784 | 837 | 947 | 316 | 47.0% | 43.4%-50.4% |

## By Domain

| domain | resolved | high wins | framed-empty wins | ties | equal-cell mean | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| animals | 1945 | 868 | 1077 | 155 | 44.7% | 39.9%-49.6% |
| countries | 1950 | 845 | 1105 | 150 | 43.5% | 38.1%-48.8% |
| political | 1952 | 899 | 1053 | 148 | 46.0% | 40.7%-51.0% |
| religions | 1933 | 879 | 1054 | 167 | 45.9% | 40.4%-51.3% |

## Inference Convention

Aggregate rows report equal-cell means with 95% bootstrap CIs. Model-by-task lollipop figures use the same convention as the canonical high-low and direct-instruction figures: tie-excluded win rates with Bonferroni exact binomial FWER 95% CIs across the 28 actor-task cells. Holm-adjusted p-values are retained in the model-task CSV.

## Paper-Ready Figures

- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/paper_ready/figures/high_utility_vs_framed_empty_model_task_lollipop.png`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/paper_ready/figures/high_utility_vs_framed_empty_domain_animals_model_task_lollipop.png`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/paper_ready/figures/high_utility_vs_framed_empty_domain_countries_model_task_lollipop.png`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/paper_ready/figures/high_utility_vs_framed_empty_domain_political_model_task_lollipop.png`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/paper_ready/figures/high_utility_vs_framed_empty_domain_religions_model_task_lollipop.png`
