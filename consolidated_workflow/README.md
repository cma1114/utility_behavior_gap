# Consolidated Workflow

This directory is a new, separate workflow layer. It does not delete or modify
the existing scripts. Its purpose is to replace families of nearly identical
scripts with config-driven entry points.

## Pairwise Comparisons

Use `pairwise.py` for judged pairwise comparisons:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python consolidated_workflow/pairwise.py list
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python consolidated_workflow/pairwise.py audit
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python consolidated_workflow/pairwise.py summarize --comparison highlow
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python consolidated_workflow/pairwise.py plot --comparison highlow
```

This replaces the pattern of one prepare/analyze/plot script per bridge
comparison. The comparison-specific information lives in
`configs/pairwise_comparisons.yaml`.

For comparisons that have source output IDs, `pairwise.py prepare-manifest`
builds a run-local judging manifest that can be judged with the existing
`run_judging.py`:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python consolidated_workflow/pairwise.py prepare-manifest \
  --comparison high_utility_vs_r0 \
  --out-dir outputs/api/runs/example_high_utility_vs_r0

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m utility_behavior_gap.scripts.run_judging \
  --run-dir outputs/api/runs/example_high_utility_vs_r0 \
  --orders both \
  --workers 4
```

Do not run the judging command unless you intend to make API calls.

## Feature Tables

Use `feature_tables.py` for generic feature deltas and appendix tables:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python consolidated_workflow/feature_tables.py list
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python consolidated_workflow/feature_tables.py audit
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python consolidated_workflow/feature_tables.py run --table utility --stage all --dry-run
```

This replaces scattered hand-written command blocks for:

- high/low condition feature deltas;
- baseline bridge feature deltas;
- full generic-plus-rubric appendix tables;
- generic-only appendix tables when no LLM rubric coding exists.

The table-specific information lives in `configs/feature_tables.yaml`.

