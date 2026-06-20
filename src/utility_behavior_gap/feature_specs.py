"""Load editable feature-analysis specifications."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from utility_behavior_gap.paths import ANALYSIS_SPECS


FEATURE_SPEC = ANALYSIS_SPECS / "feature_definitions.yaml"


@lru_cache(maxsize=None)
def load_feature_spec(path: str | Path = FEATURE_SPEC) -> dict[str, Any]:
    spec_path = Path(path)
    data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Feature spec did not parse to a mapping: {spec_path}")
    for key in ("generic_features", "task_rubric_features"):
        if key not in data:
            raise ValueError(f"Feature spec missing required key {key!r}: {spec_path}")
    return data


def normalize_text(value: Any) -> str:
    return " ".join(str(value).split())


def normalize_info_mapping(mapping: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, value in mapping.items():
        if not isinstance(value, dict):
            raise ValueError(f"Feature spec entry {key!r} must be a mapping")
        item = dict(value)
        for text_key in ("family", "label", "definition", "formula"):
            if text_key in item:
                item[text_key] = normalize_text(item[text_key])
        out[str(key)] = item
    return out


def generic_feature_info(path: str | Path = FEATURE_SPEC) -> dict[str, dict[str, Any]]:
    spec = load_feature_spec(path)
    features = spec["generic_features"].get("features", {})
    if not isinstance(features, dict):
        raise ValueError("generic_features.features must be a mapping")
    return normalize_info_mapping(features)


def standard_generic_feature_ids(path: str | Path = FEATURE_SPEC) -> list[str]:
    spec = load_feature_spec(path)
    feature_ids = spec["generic_features"].get("standard_set", [])
    if not isinstance(feature_ids, list):
        raise ValueError("generic_features.standard_set must be a list")
    features = generic_feature_info(path)
    missing = [feature_id for feature_id in feature_ids if feature_id not in features]
    if missing:
        raise ValueError(f"Standard generic feature(s) missing definitions: {missing}")
    return [str(feature_id) for feature_id in feature_ids]


def task_rubric_dimensions(path: str | Path = FEATURE_SPEC) -> dict[str, list[str]]:
    spec = load_feature_spec(path)
    tasks = spec["task_rubric_features"].get("tasks", {})
    dimensions = spec["task_rubric_features"].get("dimensions", {})
    if not isinstance(tasks, dict) or not isinstance(dimensions, dict):
        raise ValueError("task_rubric_features.tasks and .dimensions must be mappings")
    out: dict[str, list[str]] = {}
    for task, task_info in tasks.items():
        task_dims = task_info.get("dimensions", [])
        if not isinstance(task_dims, list):
            raise ValueError(f"dimensions for task {task!r} must be a list")
        missing = [dimension for dimension in task_dims if dimension not in dimensions]
        if missing:
            raise ValueError(f"Task {task!r} references undefined dimension(s): {missing}")
        out[str(task)] = [str(dimension) for dimension in task_dims]
    return out


def rubric_dimension_info(path: str | Path = FEATURE_SPEC) -> dict[str, dict[str, Any]]:
    spec = load_feature_spec(path)
    dimensions = spec["task_rubric_features"].get("dimensions", {})
    if not isinstance(dimensions, dict):
        raise ValueError("task_rubric_features.dimensions must be a mapping")
    return normalize_info_mapping(dimensions)


def rubric_dimension_descriptions(path: str | Path = FEATURE_SPEC) -> dict[str, str]:
    return {
        dimension: str(info.get("definition", ""))
        for dimension, info in rubric_dimension_info(path).items()
    }


def rubric_dimension_labels(path: str | Path = FEATURE_SPEC) -> dict[str, str]:
    return {
        dimension: str(info.get("label", dimension))
        for dimension, info in rubric_dimension_info(path).items()
    }


def task_rubric_display_digits(path: str | Path = FEATURE_SPEC) -> int:
    spec = load_feature_spec(path)
    return int(spec["task_rubric_features"].get("display_digits", 2))
