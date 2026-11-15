# Complete Utility and Framed-Empty Refusal Screen

All 25,200 selected outputs were classified. Every proposed exclusion and completion-metadata flag was read against the original task before the final label was assigned. No outputs, prior labels, judgments, or original paper results were deleted or overwritten.

## Output Counts

| Arm | Outputs | Full refusal | Partial refusal | Degenerate | Clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| High utility | 8,400 | 0 (0.00%) | 1 (0.01%) | 3 (0.04%) | 8396 (99.95%) |
| Low utility | 8,400 | 8 (0.10%) | 1 (0.01%) | 1 (0.01%) | 8390 (99.88%) |
| Framed empty | 8,400 | 0 (0.00%) | 0 (0.00%) | 3 (0.04%) | 8397 (99.96%) |

Percentages describe all 8,400 outputs per arm, including outputs subsequently excluded from quality comparisons. The four labels are mutually exclusive. Detailed actor/task/domain counts are in `classification_counts.csv`; framed empty has no intrinsic outcome domain.

## High-Low Rate Differences

| Category | Low minus high, percentage points [95% CI] |
| --- | ---: |
| any refusal | 0.095 [-0.024, 0.262] |
| degenerate | -0.024 [-0.095, 0.036] |

These intervals use paired low-minus-high indicators, not independent-arm samples, and the paper's two-stage equal actor/task/domain-cell bootstrap (5,000 draws). They are descriptive aggregate intervals, not model-wise multiple-testing claims.

## Quality Sensitivity

| Comparison / task | Before: win rate [95% CI] | After: win rate [95% CI] | Pairs removed |
| --- | ---: | ---: | ---: |
| High vs low utility / Overall | 51.2% [48.7%, 53.6%] | 51.1% [48.7%, 53.6%] | 14 |
| High vs low utility / Essay writing | 50.2% [44.9%, 55.6%] | 50.2% [44.9%, 55.7%] | 3 |
| High vs low utility / Grant abstract | 53.8% [49.5%, 58.3%] | 53.6% [49.2%, 58.1%] | 8 |
| High vs low utility / Incident postmortem | 52.7% [48.2%, 57.3%] | 52.7% [48.2%, 57.3%] | 1 |
| High vs low utility / Translation | 47.9% [43.1%, 52.8%] | 47.9% [43.2%, 52.9%] | 2 |
| High utility vs framed empty / Overall | 45.0% [42.4%, 47.6%] | 45.0% [42.4%, 47.6%] | 7 |
| High utility vs framed empty / Essay writing | 45.0% [39.3%, 50.4%] | 45.0% [39.0%, 50.6%] | 2 |
| High utility vs framed empty / Grant abstract | 35.5% [31.0%, 40.2%] | 35.5% [31.2%, 40.1%] | 1 |
| High utility vs framed empty / Incident postmortem | 52.6% [47.7%, 57.4%] | 52.6% [47.6%, 57.3%] | 3 |
| High utility vs framed empty / Translation | 47.0% [43.4%, 50.4%] | 47.0% [43.6%, 50.5%] | 1 |
| Harmful vs framed empty / Overall | 40.5% [35.8%, 45.0%] | 40.5% [35.7%, 45.1%] | 3 |
| Harmful vs framed empty / Essay writing | 40.8% [28.5%, 51.2%] | 40.8% [28.6%, 51.0%] | 1 |
| Harmful vs framed empty / Grant abstract | 29.9% [23.3%, 38.3%] | 29.9% [23.3%, 38.3%] | 0 |
| Harmful vs framed empty / Incident postmortem | 46.9% [39.4%, 54.8%] | 46.8% [39.6%, 54.8%] | 2 |
| Harmful vs framed empty / Translation | 44.4% [41.1%, 47.8%] | 44.4% [41.1%, 47.8%] | 0 |

Primary quality means exclude panel ties and retain the paper's equal-cell weights and two-stage bootstrap: actor/task/domain for utility, actor/task for harmful. Before/after intervals use the same seed per scope. Existing panel verdicts are unchanged; an entire pair is excluded if either output is finally labelled non-clean. Domain and task/domain estimates are in `quality_sensitivity.csv`; the model/task table uses the original 28-cell exact Bonferroni interval convention.

The harmful rows already exclude harmful-side refusals/degeneracy under the paper's existing classification. Here, 'after' additionally screens every matched framed-empty control. This is not a new classification of the harmful responses.

## Methods and Evidence

The source cohort is pinned to the paper's selected pair tables and each archived output's exact ID and text hash. All actor system prompts and incentive/empty framing were checked during extraction. Classification uses openai/gpt-5-mini, temperature 0, max_tokens 150, minimal/excluded reasoning, and the existing four-category source-aware policy-screen prompt. The full output and original task are supplied; actor identity, utility side/value, and panel result are not. Every new literal request and raw response is retained in compressed files. Failed or truncated classifier responses cannot become valid labels.

Reused labels require the same output ID and hash. Prior high-policy screening used the same four definitions; its earlier source-context corrections and prior low-policy/empty adjudications remain preserved. Fresh proposed exclusions and prior non-clean labels were all re-read in full by the coding assistant for this audit; this was not a separate human adjudication. Missing finish metadata triggers review rather than an automatic truncation claim. Ordinary quality flaws are not classified as degeneracy. This is an LLM screen with review of positive flags, not an exhaustive manual reading of every output, so it cannot guarantee that no refusal was missed.

New API cost: $6.13; 19,964 new classifications and 5,236 reused labels. No generation, quality-judge, or rubric-feature calls.

## Reproduce

```sh
python paper_reproduction/utility_output_screen.py analyze --out outputs/reproduced/utility_output_screen
```

This is offline and uses the compact evidence under `data/processed/utility_output_screen/`, the existing public policy-screen evidence, and frozen paper labels. Preparation alone needs the private historical archive; analysis does not. Progress: `python paper_reproduction/utility_output_screen.py status`. Collection is resumable with `screen --run`; it requires explicit API authorization and credentials. The $15 guard and concurrency lock are local runtime state, not published account information.
