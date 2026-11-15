# Low policy Versus Framed empty: Paper Feature Analysis

[Side-by-side harmful and low-policy table](harmful_and_low_policy_feature_table.md).

[Interpretation and comparison with the harmful paper results](findings.md).

2,090 clean, already-judged pairs; 480 newly coded pairs (120 per task). 10 previously identified refusal/degenerate pairs were excluded before sampling. Recorded cost: $0.90; 481 attempts, 480 successful codings. No new actor generations, panel judgments, or refusal classifications.

## Paper-Style Selected Dimensions

| Task | Dimension | Low policy minus framed empty (SD units, 95% CI) | Raw arm gap (95% CI) | Panel association (95% CI) |
| --- | --- | --- | --- | --- |
| Essay writing | Words | -0.29 [-0.37, -0.21] | -10.4 [-13.2, -7.6] | +0.31 [+0.23, +0.38] |
| Grant abstract | Feasibility/readiness | -0.27 [-0.45, -0.08] | -0.25 [-0.42, -0.08] | +0.33 [+0.15, +0.51] |
|  | Risk mitigation | -0.41 [-0.58, -0.23] | -0.37 [-0.53, -0.21] | +0.31 [+0.13, +0.49] |
|  | Stakeholder/context fit | -0.27 [-0.45, -0.09] | -0.23 [-0.38, -0.08] | +0.25 [+0.06, +0.43] |

The complete per-task table, including non-selected and countervailing results, is in [all_features.md](all_features.md).

No dimensions meet the full display rule for: Incident postmortem, Translation.

## Methods

The eight generic features use all clean pairs. The task-specific rubric uses 120 random pairs per task, balanced across actors and sampled without looking at panel outcomes. One blinded A/B call rates all dimensions for a pair. Coder: google/gemini-2.5-flash, temperature 0, initial response cap 1,400 tokens. The literal rubric and these parameters were checked against the harmful paper's saved requests for all four tasks. Only truncated coding responses may be retried with a larger cap, up to 5,600 tokens; overrides are logged.

Positive gaps mean more of a feature under Low policy. Rubric scores are +1 for the left arm better, 0 for a tie, -1 for the right arm better; not-applicable scores are missing. Panel score is +1 for a left-arm panel win, 0 for a panel tie, -1 for a right-arm win. Non-word features adjust for paired word-count difference and actor in both gap and panel-association models; word count adjusts for actor only. The existing paper functions compute HC3 robust 95% confidence intervals. A standardized gap divides the adjusted raw gap by the observed SD of paired differences within task. Panel associations are changes in panel score per one-SD change in the paired feature difference.

The table applies exactly the harmful paper's symmetric display rule: both gap and panel-association 95% intervals exclude zero, and the absolute standardized gap is at least 0.25. Countervailing effects are retained. These are nominal feature intervals, not familywise intervals. Full estimates, including Holm-adjusted p-values within task/feature family, are in feature_results.csv. No causal mediation or equivalence of harmful and policy effects is claimed.

All eight generic dimensions and all task-specific dimensions are included in the full results: 14 essay, 15 grant, 16 postmortem, 14 translation rows. No partial generic-only table is substituted.

## Reproduction

```sh
python paper_reproduction/reproduce.py run --step policy
```

This command makes no API calls. It reparses the saved rubric responses and
recomputes all 59 feature rows for each contrast, including their intervals,
then verifies the parallel table. Output is separate under `outputs/reproduced/policy/`.
Exact requests and all response attempts are in `data/processed/policy_framed_empty.jsonl.gz`,
under the `features/rubric_requests.jsonl.gz` and `features/rubric_responses.jsonl` member names.
The old absolute paths in the preserved provenance describe the original run,
not files a reader must obtain. The historical collection source is also archived.
