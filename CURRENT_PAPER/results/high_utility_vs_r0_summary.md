# High Utility Versus R0

Source bridge runs:

- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-r0-bridge__2026-06-16_22-48-47Z__hash-d4b6c15666c3`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-r0-bridge__2026-06-16_22-48-56Z__hash-934fbfa8fc5d`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-r0-bridge__2026-06-16_22-49-06Z__hash-2f2876ee80b6`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-r0-bridge__2026-06-16_22-49-15Z__hash-99599ac7fd81`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-r0-bridge__2026-06-16_22-49-24Z__hash-cadce2236c7e`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-r0-bridge__2026-06-16_22-49-35Z__hash-79c7031fdc61`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/highlow-r0-bridge__2026-06-16_22-49-45Z__hash-f498fb25a15f`

Primary estimate: equal-cell mean over actor x task x domain cells. Panel ties are excluded from the win-rate denominator and reported separately.

## Overall

| resolved | high wins | R0 wins | ties | pooled | equal-cell mean | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 3914 | 2061 | 1853 | 286 | 52.7% | 52.8% | 49.4%-56.0% |

## By Task

| task | resolved | high wins | R0 wins | ties | equal-cell mean | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Essay writing | 909 | 476 | 433 | 141 | 52.3% | 44.0%-60.7% |
| Grant abstract | 1048 | 518 | 530 | 2 | 49.4% | 43.5%-55.3% |
| Incident postmortem | 1046 | 567 | 479 | 4 | 54.3% | 47.2%-61.4% |
| Translation | 911 | 500 | 411 | 139 | 54.9% | 50.2%-59.5% |

## By Domain

| domain | resolved | high wins | R0 wins | ties | equal-cell mean | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| animals | 985 | 506 | 479 | 79 | 51.6% | 45.6%-57.9% |
| countries | 989 | 512 | 477 | 75 | 51.7% | 44.5%-58.9% |
| political | 978 | 516 | 462 | 58 | 52.9% | 46.3%-59.3% |
| religions | 962 | 527 | 435 | 74 | 54.8% | 47.9%-61.4% |

## Inference Convention

Aggregate rows report equal-cell means with 95% bootstrap CIs. Model-by-task lollipop figures use the same convention as the canonical high-low and direct-instruction figures: tie-excluded win rates with Bonferroni exact binomial FWER 95% CIs across the 28 actor-task cells. Holm-adjusted p-values are retained in the model-task CSV.

## Paper-Ready Figures

- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/paper_ready/figures/high_utility_vs_r0_model_task_lollipop.png`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/paper_ready/figures/high_utility_vs_r0_domain_animals_model_task_lollipop.png`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/paper_ready/figures/high_utility_vs_r0_domain_countries_model_task_lollipop.png`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/paper_ready/figures/high_utility_vs_r0_domain_political_model_task_lollipop.png`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/paper_ready/figures/high_utility_vs_r0_domain_religions_model_task_lollipop.png`
