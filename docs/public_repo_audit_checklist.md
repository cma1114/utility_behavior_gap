# Public Repository Audit Checklist

Use this checklist before copying files into a new public repository and again
before publishing the repository. The goal is to prevent stale prompts,
contaminated analyses, local junk files, secret leakage, and ambiguous artifact
provenance.

## 1. Scope And Canonicality

- [ ] The public repo is built from `docs/public_repo_manifest.md`, not by
  copying the whole working repo.
- [ ] Every copied script, data file, figure, table, and doc is either named in
  the manifest or explicitly added to it before copying.
- [ ] `docs/canonical_prompt_book.md` is the current prompt book and matches the
  prompt text used by the included data.
- [ ] No current paper claim relies on `outputs/analysis/modgrid_prompt_book.md`
  or any old `modgrid_*` prompt book.
- [ ] Direct instruction uses the finalized user-prompt max-effort treatment,
  not the older system-prompt treatment.
- [ ] Direct instruction, high-low utility, and moral conditions use the
  authorized `fund the following intervention:` wording where applicable.
- [ ] High-low, amount, moral, framed neutral, framed empty, and R0 baselines
  have no hidden system prompts unless a specific appendix says otherwise.
- [ ] Any appendix-only experiment is labeled as appendix/exploratory and cannot
  be confused with a main paper condition.

## 2. Secret And Local-File Check

- [ ] No `.env` file is present.
- [ ] No API keys, tokens, or provider credentials are present.
- [ ] No `.venv/` directory is present.
- [ ] No `.DS_Store` files are present.
- [ ] No `__pycache__/` directories or `.pyc` files are present.
- [ ] No `.pytest_cache/` directory is present.
- [ ] No `.claude/` directory is present.
- [ ] No `CLAUDE.md` file is present.
- [ ] No local screenshot, desktop, or scratch files are present.

Suggested local checks:

```bash
find . \( -name '.DS_Store' -o -name '__pycache__' -o -name '*.pyc' -o -name '.pytest_cache' -o -name '.venv' \) -print
find . \( -name '.env' -o -name 'CLAUDE.md' -o -name '.claude' \) -print
rg -n "OPENROUTER|API_KEY|Authorization|Bearer |sk-[A-Za-z0-9]|user_[A-Za-z0-9]" .
```

## 3. Code And Entry Points

- [ ] `pyproject.toml` and `requirements.txt` install a fresh environment
  without private paths.
- [ ] Public scripts use package-relative paths or documented environment
  variables, not absolute paths from the private working machine.
- [ ] Paid API scripts are marked clearly as optional rerun scripts.
- [ ] Local analysis scripts can run from frozen processed data without API
  keys.
- [ ] Diagnostic scripts not needed for the paper are omitted or moved under a
  clearly labeled archive directory.
- [ ] `src/utility_behavior_gap/scripts/README.md` matches the public script
  set and does not list missing or obsolete scripts as current.

Suggested checks:

```bash
python -m pytest
python -m utility_behavior_gap.scripts.export_canonical_prompt_book --help
python -m utility_behavior_gap.scripts.analyze_canonical_highn_conditions --help
python -m utility_behavior_gap.scripts.make_feature_appendix_table --help
```

## 4. Prompt And Manifest Integrity

- [ ] Each paper-facing run has an immutable manifest or processed-data record
  linking it to actor, task, condition, prompt hash, and run directory.
- [ ] Prompt hashes in processed results match the current prompt book.
- [ ] The public prompt book includes full system and user prompts, not only
  short labels.
- [ ] The source side labels in pair files are meaningful and match the paper
  terminology: high, low, strong, neutral, R0, framed empty, framed neutral.
- [ ] There are no obsolete labels such as `goodjob` in public filenames,
  script names, or condition labels unless they appear only in a historical
  archive.
- [ ] Every bridge analysis records which exact output IDs were compared.

## 5. Exclusions And Data Cleaning

- [ ] Mechanical invalid outputs are excluded from every analysis: missing
  output, failed API response, empty text, non-`stop` finish reason, and
  truncation.
- [ ] Refusal and degenerate-output classifications are applied by `output_id`,
  not by condition name.
- [ ] Moral refusal classifications are applied wherever those outputs appear,
  including bridge analyses.
- [ ] Analyses report how many outputs/pairs were excluded.
- [ ] Feature analyses and judging analyses use the same exclusion rules for a
  given comparison.
- [ ] Excluded outputs are not deleted from private raw logs; they are filtered
  out by analysis code.
- [ ] The public repo includes the classifier prompt, classifier output labels,
  and enough metadata to audit exclusions.

## 6. Statistics And Figures

- [ ] Main model-by-task lollipop figures use the standard convention:
  win rate excluding panel ties, with Bonferroni exact binomial family-wise
  95% CIs across the 28 actor-task cells.
- [ ] Holm-adjusted p-values are retained in CSV outputs where relevant, but
  not confused with the plotted CIs.
- [ ] Aggregate summaries use equal-cell means, not raw pooled rates.
- [ ] High-low utility and utility bridge aggregate cells are actor x task x
  domain unless a narrower scope is explicitly stated.
- [ ] Non-domain aggregate cells are actor x task unless otherwise stated.
- [ ] Every figure in `paper/figures/` has a matching data CSV or summary in
  `paper/results/` or `data/processed/`.
- [ ] Every paper-facing figure can be regenerated from committed processed
  data without paid API calls.
- [ ] Figure filenames identify the condition unambiguously.
- [ ] No stale figure with a misleadingly canonical name remains in the public
  paper artifact directory.

## 7. Feature Analysis

- [ ] The editable source of feature labels and definitions is
  `analysis_specs/feature_definitions.yaml`.
- [ ] The standard generic feature set is the documented set, including
  combined quantitative detail as `z(numbers + percentages)`.
- [ ] Full feature tables are labeled full only when they include both generic
  features and task-specific LLM-rubric coding.
- [ ] Generic-only tables are labeled generic-only.
- [ ] The table column `Panel preference (r)` is present only when the correct
  panel outcomes were joined for that comparison.
- [ ] Bridge feature tables use exact source output IDs from bridge outcome
  files, not approximate condition matching.
- [ ] Raw LLM rubric prompt/response JSONL files are either committed under
  `data/audit/` or listed in the external raw archive manifest.
- [ ] Feature tables were regenerated after any exclusion-rule change.

## 8. Raw Archive Boundary

- [ ] Raw API logs are not mixed into `paper/figures/`, `paper/results/`, or
  `paper/features/`.
- [ ] If raw logs are not committed, `external_raw_archive_manifest.md` exists.
- [ ] The external manifest lists every run directory needed to audit the paper
  analyses.
- [ ] The external manifest includes checksums and byte sizes.
- [ ] The external manifest states that rerunning API calls can change outputs
  and may cost money.
- [ ] Private failed runs and obsolete prompt variants are either omitted or
  clearly marked as historical failures, not paper evidence.

## 9. Size And Duplication

- [ ] Large CSV/JSONL files are included only once.
- [ ] Large processed files are compressed if necessary and documented.
- [ ] Figures and tables are not duplicated under multiple public paths.
- [ ] `paper/` contains human-facing artifacts; `data/` contains
  machine-readable reproducibility data.
- [ ] The same full output text is not duplicated in multiple public files
  unless required for audit and documented.

## 10. Final Reproducibility Smoke Test

Run from a fresh clone or temporary copy:

- [ ] Create a fresh virtual environment.
- [ ] Install package requirements.
- [ ] Run tests.
- [ ] Regenerate the canonical prompt book.
- [ ] Regenerate the main direct-instruction figure from processed data.
- [ ] Regenerate high-low, amount, and moral lollipop figures from processed
  data.
- [ ] Regenerate at least one bridge figure from processed data.
- [ ] Regenerate at least one combined feature appendix table from processed
  data.
- [ ] Confirm no command in the smoke test requires an API key.

## 11. Human Review Before Release

- [ ] Review `README.md` for a reader who has never seen the private working
  repo.
- [ ] Open every main PNG figure and confirm it is the intended current figure.
- [ ] Open every main Markdown/CSV table and confirm it matches the paper
  narrative.
- [ ] Check that generic-only feature analyses are not described as full rubric
  analyses.
- [ ] Check that null results are retained and not hidden by file selection.
- [ ] Check that appendix/exploratory results are not mixed into main results.
- [ ] Commit only after the manifest and this checklist are updated to match
  the actual copied public repo contents.

