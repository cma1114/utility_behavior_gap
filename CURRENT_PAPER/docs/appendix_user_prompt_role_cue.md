# User-Prompt Role Cue Finding

Status: paper-relevant result. This file records the framed role-cue analysis that should be carried forward into the paper and public repository.

## Contrast

The experiment compares two user-prompt role cues while holding the competition-framed task wrapper fixed:

- Strong role cue: world-class task-specific professional.
- Control role cue: skilled task-specific professional.

The cue is placed in the user prompt, not the system prompt. The system prompt is blank. The task wrapper is the same framed-neutral competition wrapper used in the current paradigm.

## Primary Figure

- `CURRENT_PAPER/figures/framed_user_prompt_role_model_task_lollipop.png`

The figure uses the same model-by-task lollipop format as the primary direct-instruction and high-low figures. Each plotted cell reports the world-class-role-side win rate with ties excluded. Model-task confidence intervals are exact binomial intervals Bonferroni-adjusted over the planned 28 cells.

## Aggregate Estimand

The primary aggregate estimate is the equal-cell mean win rate.

- Overall cells: actor x task.
- Task-specific cells: actors within that task.
- Ties are excluded from the win-rate denominator.
- Confidence intervals are nonparametric bootstraps over the relevant design cells, with raw non-tied trials resampled within cells.
- Pooled Wilson intervals are reported only as descriptive checks.

## Main Results

The world-class role cue improved judged output quality overall.

- Overall equal-cell mean: 61.2% [53.2%, 70.9%].
- Overall pooled win rate: 2280 / 3755 non-tied decisions = 60.7% [59.1%, 62.3%].
- FWER-positive model-task cells: 10 / 28.

By task:

- Essay writing: 74.4% [65.5%, 82.7%].
- Grant abstract: 60.8% [57.1%, 64.5%].
- Incident postmortem: 55.1% [49.7%, 60.3%].
- Translation: 54.4% [48.8%, 60.7%].

Interpretation: role cues are another explicit cue that can make models produce higher-quality outputs. The effect is strongest for essays and positive for grant abstracts; incident postmortems and translation are weaker under the equal-cell task-level CIs.

## Primary Result Files

- `CURRENT_PAPER/results/framed_user_prompt_role_aggregate.md`
- `CURRENT_PAPER/results/framed_user_prompt_role_aggregate.csv`
- `CURRENT_PAPER/results/framed_user_prompt_role_model_task_summary.md`
- `CURRENT_PAPER/results/framed_user_prompt_role_model_task_plot_data.csv`
- `CURRENT_PAPER/results/framed_user_prompt_role_pair_outcomes.csv`
- `CURRENT_PAPER/results/framed_user_prompt_role_run_audit.csv`

## Reproduction Notes

The extension to 300 planned pairs per actor-task used the existing 30 task items with repeats 5-9 added to the original repeats 0-4. The combined figure is generated with `plot_model_task_lollipop.py --combine-runs`, which merges the base and extension run blocks per actor-task and de-duplicates by pair id.
