# User-Prompt Role Cue Finding

Status: paper-relevant result. This file records the role-cue analysis that should be carried forward into the paper and public repository.

## Contrast

The experiment compares two user-prompt role cues while holding the task request fixed:

- Strong role cue: world-class task-specific professional.
- Control role cue: skilled task-specific professional.

The cue is placed in the user prompt, not the system prompt. The system prompt is blank.

## Primary Figure

- `CURRENT_PAPER/figures/user_prompt_role_model_task_lollipop.png`

The figure uses the same model-by-task lollipop format as the primary direct-instruction and high-low figures. Each plotted cell reports the strong-role-side win rate with ties excluded. Model-task confidence intervals are exact binomial intervals Bonferroni-adjusted over the planned 28 cells.

## Aggregate Estimand

The primary aggregate estimate is the equal-cell mean win rate.

- Overall cells: actor x task.
- Task-specific cells: actors within that task.
- Ties are excluded from the win-rate denominator.
- Confidence intervals are nonparametric bootstraps over the relevant design cells, with raw non-tied trials resampled within cells.
- Pooled Wilson intervals are reported only as descriptive checks.

## Main Results

The world-class role cue improved judged output quality overall.

- Overall equal-cell mean: 63.3% [53.0%, 75.7%].
- Overall pooled win rate: 1275 / 1996 non-tied decisions = 63.9%.
- FWER-positive model-task cells: 11 / 28.

By task:

- Essay writing: 78.5% [65.0%, 90.0%].
- Grant abstract: 63.2% [54.6%, 70.6%].
- Incident postmortem: 59.4% [50.4%, 68.0%].
- Translation: 52.1% [46.2%, 57.9%].

Interpretation: role cues are another explicit cue that can make models produce higher-quality outputs, especially in open-ended generation tasks. The effect is strongest for essays, positive for grants and incident postmortems, and near-null for translation.

## Primary Result Files

- `CURRENT_PAPER/results/user_prompt_role_aggregate.md`
- `CURRENT_PAPER/results/user_prompt_role_aggregate.csv`
- `CURRENT_PAPER/results/user_prompt_role_model_task_summary.md`
- `CURRENT_PAPER/results/user_prompt_role_model_task_plot_data.csv`
- `CURRENT_PAPER/results/user_prompt_role_pair_outcomes.csv`
- `CURRENT_PAPER/results/user_prompt_role_run_audit.csv`
