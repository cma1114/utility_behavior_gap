# Public Reproduction Verification

Verified on 2026-11-14, using a fresh copy of the exact staged public-repository
files, not the research worktree. Unrelated uncommitted changes were excluded.
Python audit hooks blocked network connections and file access to both original
checkouts; only the existing Python environment's installed dependencies were
allowed. All analysis inputs and source modules came from the isolated copy.

## Commands and Results

```sh
python paper_reproduction/reproduce.py run
python -m pytest tests/test_paper_reproduction.py tests/test_policy_public_reproduction.py tests/test_reasoning_trace_analysis.py tests/test_analysis.py tests/test_packaged_entrypoints.py -q
```

- Completed all five sections: outcomes, trends, features, reasoning, policy.
- Passed all 33 numerical reference checks, in 81.68 seconds.
- Passed all 19 tests in the publication snapshot.
- Made no API calls. Created no PDFs. Reference estimates were not overwritten.
- Inspected the regenerated policy lollipop for clipping and labeling.

## Policy Evidence Checks

The new step verified 4,200 source texts and their hashes, 2,100 planned pairs,
and the exclusion of 10 refusal/degenerate pairs. It reconstructed all 2,090
remaining panels from raw responses to 6,270 exact judge requests. It reproduced
the aggregate intervals and all 28 model-task estimates and adjusted intervals.

The feature step reproduced the actor-balanced random sample, verified all 480
literal rubric requests, and reparsed the saved responses. It refitted all 59
low-policy and all 59 harmful feature rows with the paper's existing functions,
checked their estimates and intervals, and reproduced the complete 10-row
parallel table exactly. Saved reference values were used for checks and display
order only, never as fitted estimates.

This is reproduction from recorded judgments, rubric ratings, generic feature
measurements, and exclusion labels. It does not claim to independently recode
the texts or to regenerate identical stochastic actor responses.

The evidence archive is 23.69 MiB; the whole publication addition is about
29 MiB. No large run directories were copied into the repository. Credential
screening found no exposed credentials; three loose token-pattern matches were
inside opaque encrypted reasoning data, not authentication fields.

The input hashes and export inventory are in
`data/processed/policy_framed_empty_provenance.json`. Original source provenance
is retained alongside the reference results. The new command writes regenerated
files under `outputs/reproduced/`, including `verification.json` and
`policy/audit.json`, without replacing the accepted snapshots.
