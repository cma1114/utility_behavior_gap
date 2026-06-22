# Amount High Versus Framed-Neutral Bridge

Run dirs:

- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/amount-neutral-bridge-current__2026-06-15_19-52-44Z__hash-67ea49cc1288`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/amount-neutral-bridge-current__2026-06-15_19-52-52Z__hash-3ea9b16c7d5a`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/amount-neutral-bridge-current__2026-06-15_19-53-02Z__hash-17207d323e85`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/amount-neutral-bridge-current__2026-06-15_19-53-31Z__hash-0145f2d43767`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/amount-neutral-bridge-current__2026-06-15_19-53-43Z__hash-0998d7207f1e`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/amount-neutral-bridge-current__2026-06-15_19-53-54Z__hash-45b8dc6d4ff5`
- `/Users/christopherackerman/repos/Utility-Behavior-Gap/outputs/api/runs/amount-neutral-bridge-current__2026-06-15_19-54-02Z__hash-a219c086bc27`

Score definition: amount-high win = +1, framed-neutral win = -1, tie = 0.
The win-rate denominator excludes panel ties; the net score includes panel ties as 0.
Primary paper-facing summaries average actor-task cells with equal weight and use t intervals over those cells.
Pooled summaries are included as descriptive counts only.

## Equal Actor-Task Summary

| cells | pairs | amount wins | neutral wins | ties | unresolved | win rate | 95% CI | net score | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 28 | 8400 | 2041 | 1717 | 4642 | 0 | 0.539 | 0.508-0.570 | 0.039 | 0.010-0.067 |

## Equal Actor-Task Summary By Task

| task | cells | pairs | amount wins | neutral wins | ties | unresolved | win rate | 95% CI | net score | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| essay | 7 | 2100 | 390 | 374 | 1336 | 0 | 0.512 | 0.436-0.589 | 0.008 | -0.040-0.055 |
| grant_proposal_abstract | 7 | 2100 | 628 | 392 | 1080 | 0 | 0.611 | 0.552-0.670 | 0.112 | 0.047-0.178 |
| incident_postmortem | 7 | 2100 | 566 | 462 | 1072 | 0 | 0.549 | 0.493-0.606 | 0.050 | -0.006-0.105 |
| translation | 7 | 2100 | 457 | 489 | 1154 | 0 | 0.484 | 0.432-0.536 | -0.015 | -0.063-0.032 |

## Pooled Overall

| pairs | amount wins | neutral wins | ties | unresolved | win rate | exact 95% CI | net score | 95% bootstrap CI | p(two-sided) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8400 | 2041 | 1717 | 4642 | 0 | 0.543 | 0.527-0.559 | 0.039 | 0.024-0.053 | 0.000 |

## Pooled By Task

| task | pairs | amount wins | neutral wins | ties | unresolved | win rate | exact 95% CI | net score | 95% bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| essay | 2100 | 390 | 374 | 1336 | 0 | 0.510 | 0.474-0.546 | 0.008 | -0.018-0.033 |
| grant_proposal_abstract | 2100 | 628 | 392 | 1080 | 0 | 0.616 | 0.585-0.646 | 0.112 | 0.083-0.141 |
| incident_postmortem | 2100 | 566 | 462 | 1072 | 0 | 0.551 | 0.520-0.581 | 0.050 | 0.021-0.079 |
| translation | 2100 | 457 | 489 | 1154 | 0 | 0.483 | 0.451-0.515 | -0.015 | -0.044-0.012 |
