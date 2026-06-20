"""Deprecated compatibility wrapper for output-exclusion filters.

New analysis code should import from :mod:`utility_behavior_gap.output_exclusions`.
This module remains only so older local scripts fail less abruptly while the
repository is being cleaned up.
"""

from utility_behavior_gap.output_exclusions import (  # noqa: F401
    CLEAN_LABEL,
    FEATURE_OUTPUT_CATALOG,
    MECHANICAL_INVALID_FLAGS,
    SEMANTIC_CLASSIFICATIONS_PATH,
    filter_semantic_excluded_output_catalog,
    filter_semantic_excluded_pair_rows,
    filter_valid_output_catalog,
    filter_valid_pair_rows,
    invalid_output_ids_from_catalog,
    load_semantic_classifications,
    mechanical_invalid_output_mask,
    semantic_label_sets,
)
