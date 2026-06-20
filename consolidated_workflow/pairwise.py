#!/usr/bin/env python3
"""Config-driven pairwise comparison workflow.

This script replaces families of near-identical prepare/analyze/plot scripts
with one implementation driven by `configs/pairwise_comparisons.yaml`.
It does not modify any legacy script.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import beta, binomtest, t


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utility_behavior_gap.constants import ACTOR_LABEL, ACTORS, TASK_LABEL  # noqa: E402
from utility_behavior_gap.fingerprints import text_fingerprint  # noqa: E402
from utility_behavior_gap.io_utils import read_jsonl, write_jsonl  # noqa: E402
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition  # noqa: E402


DEFAULT_CONFIG = ROOT / "consolidated_workflow" / "configs" / "pairwise_comparisons.yaml"
DEFAULT_TASK_ORDER = ["essay", "grant_proposal_abstract", "incident_postmortem", "translation"]


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or not isinstance(data.get("comparisons"), dict):
        raise ValueError(f"config must contain a `comparisons` mapping: {path}")
    return data["comparisons"]


def get_comparison(configs: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in configs:
        raise ValueError(f"unknown comparison `{name}`; available: {', '.join(sorted(configs))}")
    cfg = dict(configs[name])
    cfg["id"] = name
    return cfg


def digest(payload: Any, n: int = 12) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:n]


def read_pairs(cfg: dict[str, Any]) -> pd.DataFrame:
    path = repo_path(cfg["pair_outcomes"])
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    for col, expected in (cfg.get("filters") or {}).items():
        if col not in df.columns:
            raise ValueError(f"{cfg['id']}: filter column `{col}` missing from {path}")
        allowed = expected if isinstance(expected, list) else [expected]
        df = df[df[col].fillna("").astype(str).isin({str(x) for x in allowed})].copy()
    return df.reset_index(drop=True)


def normalize_pairs(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if "actor_label" not in out.columns and "actor" in out.columns:
        out["actor_label"] = out["actor"].astype(str).map(ACTOR_LABEL).fillna(out["actor"])
    if "task_label" not in out.columns and "task" in out.columns:
        out["task_label"] = out["task"].astype(str).map(TASK_LABEL).fillna(out["task"])
    if "domain" not in out.columns:
        out["domain"] = ""

    if {"target_win", "target_loss", "tie"}.issubset(out.columns):
        out["target_win"] = out["target_win"].fillna(False).astype(bool)
        out["target_loss"] = out["target_loss"].fillna(False).astype(bool)
        out["tie"] = out["tie"].fillna(False).astype(bool)
    else:
        panel_col = cfg.get("panel_winner_col", "panel_winner_condition")
        if panel_col not in out.columns:
            raise ValueError(f"{cfg['id']}: no target flags and missing `{panel_col}`")
        winners = out[panel_col].fillna("").astype(str)
        target_values = {str(x) for x in cfg["target_values"]}
        other_values = {str(x) for x in cfg["other_values"]}
        out["target_win"] = winners.isin(target_values)
        out["target_loss"] = winners.isin(other_values)
        out["tie"] = winners.eq("tie")
    out["resolved"] = out["target_win"] | out["target_loss"]
    out["effect_score_target_minus_other"] = np.where(
        out["target_win"],
        1.0,
        np.where(out["target_loss"], -1.0, np.where(out["tie"], 0.0, np.nan)),
    )
    out["comparison_id"] = cfg["id"]
    out["target_condition"] = cfg["target_condition"]
    out["other_condition"] = cfg["other_condition"]
    return out


def exact_familywise_ci(wins: int, total: int, family_size: int, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    tail_alpha = alpha / (2 * family_size)
    lo = 0.0 if wins == 0 else float(beta.ppf(tail_alpha, wins, total - wins + 1))
    hi = 1.0 if wins == total else float(beta.ppf(1.0 - tail_alpha, wins + 1, total - wins))
    return lo, hi


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda idx: p_values[idx])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def cell_rows(df: pd.DataFrame, group_cols: list[str], family_size: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, sub in df.groupby(group_cols, dropna=False, sort=True):
        if len(group_cols) == 1:
            key = (key,)
        row = dict(zip(group_cols, key))
        wins = int(sub["target_win"].sum())
        losses = int(sub["target_loss"].sum())
        ties = int(sub["tie"].sum())
        unresolved = int((~sub["target_win"] & ~sub["target_loss"] & ~sub["tie"]).sum())
        n = wins + losses
        lo, hi = exact_familywise_ci(wins, n, family_size)
        p = binomtest(wins, n, 0.5, alternative="two-sided").pvalue if n else math.nan
        row.update(
            {
                "n_pairs": int(len(sub)),
                "resolved_n": n,
                "target_wins": wins,
                "target_losses": losses,
                "ties": ties,
                "unresolved": unresolved,
                "target_win_rate_excluding_ties": wins / n if n else math.nan,
                "familywise_ci_lo": lo,
                "familywise_ci_hi": hi,
                "familywise_ci_positive": bool(np.isfinite(lo) and lo > 0.5),
                "familywise_ci_negative": bool(np.isfinite(hi) and hi < 0.5),
                "p_two_sided_exact": p,
                "mean_effect_score": float(pd.to_numeric(sub["effect_score_target_minus_other"], errors="coerce").mean()),
            }
        )
        if "actor" in row:
            row["actor_label"] = ACTOR_LABEL.get(str(row["actor"]), str(row["actor"]))
        if "task" in row:
            row["task_label"] = TASK_LABEL.get(str(row["task"]), str(row["task"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["holm_p_two_sided"] = holm_adjust(out["p_two_sided_exact"].fillna(1.0).tolist())
        out["holm_positive"] = out["target_win_rate_excluding_ties"].gt(0.5) & out["holm_p_two_sided"].lt(0.05)
        out["holm_negative"] = out["target_win_rate_excluding_ties"].lt(0.5) & out["holm_p_two_sided"].lt(0.05)
    return out


def t_ci(values: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        val = float(vals.mean())
        return val, val
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    crit = float(t.ppf(0.975, len(vals) - 1))
    return mean - crit * se, mean + crit * se


def aggregate_rows(cells: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    specs: list[tuple[str, list[str]]] = [("overall", [])]
    aggregate_cells = cfg.get("aggregate_cells", ["actor", "task"])
    if "task" in aggregate_cells:
        specs.append(("task", ["task"]))
    if "domain" in aggregate_cells and "domain" in cells.columns:
        specs.append(("domain", ["domain"]))
        specs.append(("task_domain", ["task", "domain"]))

    rows: list[dict[str, Any]] = []
    for breakout, group_cols in specs:
        groups = [((), cells)] if not group_cols else cells.groupby(group_cols, dropna=False, sort=True)
        for key, sub in groups:
            if group_cols:
                if len(group_cols) == 1:
                    key = (key,)
                row = dict(zip(group_cols, key))
            else:
                row = {}
            win_lo, win_hi = t_ci(sub["target_win_rate_excluding_ties"])
            score_lo, score_hi = t_ci(sub["mean_effect_score"])
            row.update(
                {
                    "breakout": breakout,
                    "n_cells": int(len(sub)),
                    "n_pairs": int(sub["n_pairs"].sum()),
                    "resolved_n": int(sub["resolved_n"].sum()),
                    "target_wins": int(sub["target_wins"].sum()),
                    "target_losses": int(sub["target_losses"].sum()),
                    "ties": int(sub["ties"].sum()),
                    "unresolved": int(sub["unresolved"].sum()),
                    "equal_cell_win_rate": float(sub["target_win_rate_excluding_ties"].mean()),
                    "equal_cell_win_rate_ci_lo": win_lo,
                    "equal_cell_win_rate_ci_hi": win_hi,
                    "equal_cell_score": float(sub["mean_effect_score"].mean()),
                    "equal_cell_score_ci_lo": score_lo,
                    "equal_cell_score_ci_hi": score_hi,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.3f}"
    if pd.isna(value):
        return ""
    return str(value)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_value(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_summary(path: Path, cfg: dict[str, Any], pairs: pd.DataFrame, model_task: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    overall = aggregate[aggregate["breakout"].eq("overall")]
    lines = [
        f"# {cfg['title']}",
        "",
        f"Comparison id: `{cfg['id']}`",
        f"Pair outcomes: `{cfg['pair_outcomes']}`",
        f"Target side: `{cfg['target_label']}`",
        f"Other side: `{cfg['other_label']}`",
        "",
    ]
    if not overall.empty:
        row = overall.iloc[0]
        lines.extend(
            [
                "## Overall Equal-Cell Summary",
                "",
                f"- Equal-cell win rate: `{row['equal_cell_win_rate']:.3f}` "
                f"[{row['equal_cell_win_rate_ci_lo']:.3f}, {row['equal_cell_win_rate_ci_hi']:.3f}]",
                f"- Equal-cell net score: `{row['equal_cell_score']:.3f}` "
                f"[{row['equal_cell_score_ci_lo']:.3f}, {row['equal_cell_score_ci_hi']:.3f}]",
                f"- Pairs: `{int(row['n_pairs'])}`; non-tie resolved: `{int(row['resolved_n'])}`; "
                f"ties: `{int(row['ties'])}`; unresolved: `{int(row['unresolved'])}`",
                "",
            ]
        )
    keep = [
        "actor_label",
        "task_label",
        "resolved_n",
        "target_wins",
        "target_losses",
        "ties",
        "target_win_rate_excluding_ties",
        "familywise_ci_lo",
        "familywise_ci_hi",
        "familywise_ci_positive",
        "familywise_ci_negative",
        "holm_p_two_sided",
    ]
    lines.extend(["## Model-Task Cells", "", markdown_table(model_task[[c for c in keep if c in model_task.columns]])])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(cfg: dict[str, Any], out_prefix: Path | None = None) -> dict[str, Path]:
    pairs = normalize_pairs(read_pairs(cfg), cfg)
    prefix = out_prefix or repo_path(cfg["output_prefix"])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    family_size = int(cfg.get("family_size", 28))
    model_task = cell_rows(pairs, ["actor", "task"], family_size)
    aggregate_cell_cols = list(cfg.get("aggregate_cells", ["actor", "task"]))
    aggregate_cells = cell_rows(pairs, aggregate_cell_cols, family_size)
    aggregate = aggregate_rows(aggregate_cells, cfg)

    paths = {
        "pairs": prefix.with_name(prefix.name + "_pair_outcomes.csv"),
        "model_task": prefix.with_name(prefix.name + "_model_task_cells.csv"),
        "aggregate": prefix.with_name(prefix.name + "_aggregate.csv"),
        "summary": prefix.with_name(prefix.name + "_summary.md"),
    }
    pairs.to_csv(paths["pairs"], index=False)
    model_task.to_csv(paths["model_task"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    if "domain" in aggregate_cell_cols and pairs["domain"].fillna("").astype(str).str.len().gt(0).any():
        domain_model_task = cell_rows(pairs, ["domain", "actor", "task"], family_size)
        paths["domain_model_task"] = prefix.with_name(prefix.name + "_domain_model_task_cells.csv")
        domain_model_task.to_csv(paths["domain_model_task"], index=False)
    write_summary(paths["summary"], cfg, pairs, model_task, aggregate)
    return paths


def resolve_catalogs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for match in sorted(glob.glob(str(repo_path(pattern)))):
            path = Path(match)
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def read_outputs(paths: list[Path], needed: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    remaining = set(needed)
    for path in paths:
        if not remaining:
            break
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                output_id = str(row.get("output_id", ""))
                if output_id in remaining:
                    found[output_id] = row
                    remaining.remove(output_id)
                    if not remaining:
                        break
    return found


def source_job(row: dict[str, Any]) -> dict[str, Any]:
    job = row.get("job")
    return job if isinstance(job, dict) else {}


def source_condition(row: dict[str, Any]) -> str:
    return str(row.get("condition") or "")


def prompt_from_source(row: dict[str, Any]) -> str:
    job = source_job(row)
    if row.get("user_prompt"):
        return str(row["user_prompt"])
    cond = source_condition(row)
    if job.get("condition_a") == cond and job.get("prompt_a"):
        return str(job["prompt_a"])
    if job.get("condition_b") == cond and job.get("prompt_b"):
        return str(job["prompt_b"])
    request = row.get("request") or {}
    for msg in request.get("messages", []):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return str(job.get("base_prompt", ""))


def system_prompt_from_source(row: dict[str, Any]) -> str:
    job = source_job(row)
    if row.get("system_prompt") is not None:
        return str(row.get("system_prompt") or "")
    cond = source_condition(row)
    if job.get("condition_a") == cond:
        return str(job.get("system_prompt_a", ""))
    if job.get("condition_b") == cond:
        return str(job.get("system_prompt_b", ""))
    return ""


def copied_generation(source: dict[str, Any], job: dict[str, Any], condition: str, suffix: str, run_id: str) -> dict[str, Any]:
    text = str(source.get("output_text", ""))
    return {
        "actor": job.get("actor", ""),
        "condition": condition,
        "finish_reason": source.get("finish_reason", ""),
        "job": job,
        "model": source.get("model", ""),
        "output_id": f"{job['pair_uid']}::{suffix}",
        "output_text": text,
        "pair_uid": job["pair_uid"],
        "run_id": run_id,
        "success": bool(text.strip()) and source.get("success", True) is not False and source.get("ok", True) is not False,
        "source_condition": source.get("condition", ""),
        "source_output_hash": text_fingerprint(text),
        "source_output_id": source.get("output_id", ""),
        "source_pair_uid": source.get("pair_uid", ""),
        "source_note": "Copied by consolidated_workflow/pairwise.py; no generation call was made.",
    }


def prepare_manifest(cfg: dict[str, Any], out_dir: Path, limit: int | None = None) -> None:
    left_col = cfg.get("source_left_output_col")
    right_col = cfg.get("source_right_output_col")
    if not left_col or not right_col:
        raise ValueError(f"{cfg['id']} does not define source output id columns")
    pairs = normalize_pairs(read_pairs(cfg), cfg)
    if limit:
        pairs = pairs.head(limit).copy()
    needed = set(pairs[left_col].dropna().astype(str)) | set(pairs[right_col].dropna().astype(str))
    catalogs = resolve_catalogs(cfg.get("output_catalogs") or [])
    outputs = read_outputs(catalogs, needed)
    missing = sorted(needed - set(outputs))
    if missing:
        raise ValueError(f"missing {len(missing)} source outputs; first examples: {missing[:10]}")

    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%SZ")
    run_id = out_dir.name if out_dir.name else f"consolidated__{cfg['id']}__{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    for idx, pair in pairs.iterrows():
        left = outputs[str(pair[left_col])]
        right = outputs[str(pair[right_col])]
        left_job = source_job(left)
        right_job = source_job(right)
        base_job = left_job or right_job
        actor = str(pair.get("actor") or base_job.get("actor") or left.get("actor") or "")
        task = str(pair.get("task") or base_job.get("task") or left.get("task") or "")
        pair_uid = f"consolidated:{cfg['id']}:{actor}:{task}:p{idx}:v{digest([left.get('output_id'), right.get('output_id')])}"
        job = {
            "actor": actor,
            "actor_label": ACTOR_LABEL.get(actor, actor),
            "axis": base_job.get("axis", ""),
            "axis_definition": base_job.get("axis_definition", ""),
            "base_prompt": base_job.get("base_prompt", ""),
            "comparison": cfg["id"],
            "condition_a": cfg["target_condition"],
            "condition_b": cfg["other_condition"],
            "item_label": str(pair.get("item_label") or base_job.get("item_label") or ""),
            "other_condition": cfg["other_condition"],
            "pair_uid": pair_uid,
            "predicted_condition": cfg["target_condition"],
            "prompt_a": prompt_from_source(left),
            "prompt_b": prompt_from_source(right),
            "run_dir": str(out_dir),
            "run_generation_failures_path": str(out_dir / "generation_failures.jsonl"),
            "run_generations_path": str(out_dir / "generations.jsonl"),
            "run_id": run_id,
            "run_judge_votes_path": str(out_dir / "judge_votes.jsonl"),
            "run_manifest_path": str(out_dir / "generation_jobs.jsonl"),
            "source_left_output_id": left.get("output_id", ""),
            "source_right_output_id": right.get("output_id", ""),
            "system_prompt_a": system_prompt_from_source(left),
            "system_prompt_b": system_prompt_from_source(right),
            "task": task,
            "task_label": TASK_LABEL.get(task, task),
        }
        for col in ("domain", "item_id", "repeat", "pair_idx", "delta_u", "high_utility", "low_utility", "cause_pair_label"):
            if col in pair.index and not pd.isna(pair[col]):
                job[col] = pair[col].item() if hasattr(pair[col], "item") else pair[col]
        if task != "essay" and not job["axis"]:
            raise ValueError(f"non-essay pair missing judge axis: {pair_uid}")
        jobs.append(job)
        generations.append(copied_generation(left, job, cfg["target_condition"], "a", run_id))
        generations.append(copied_generation(right, job, cfg["other_condition"], "b", run_id))

    write_jsonl(out_dir / "generation_jobs.jsonl", jobs)
    write_jsonl(out_dir / "generations.jsonl", generations)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "comparison": cfg["id"],
                "created_at_utc": stamp,
                "pair_outcomes": cfg["pair_outcomes"],
                "left_output_col": left_col,
                "right_output_col": right_col,
                "jobs": len(jobs),
                "generations": len(generations),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(jobs)} judging jobs to {out_dir / 'generation_jobs.jsonl'}")
    print(f"wrote {len(generations)} copied generations to {out_dir / 'generations.jsonl'}")
    print("judge with:")
    print(f"PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m utility_behavior_gap.scripts.run_judging --run-dir {out_dir} --orders both --workers 4")


def votes_by_pair(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return out
    for row in read_jsonl(path):
        out[str(row.get("pair_uid", ""))].append(row)
    return out


def status_run(run_dir: Path) -> None:
    jobs = read_jsonl(run_dir / "generation_jobs.jsonl")
    generations = read_jsonl(run_dir / "generations.jsonl") if (run_dir / "generations.jsonl").exists() else []
    votes = votes_by_pair(run_dir / "judge_votes.jsonl")
    complete = 0
    for job in jobs:
        by_judge: dict[str, list[str]] = defaultdict(list)
        for vote in votes.get(str(job["pair_uid"]), []):
            if vote.get("success") is not False:
                by_judge[str(vote.get("judge_model", ""))].append(str(vote.get("winner_condition", "")))
        verdicts = [derive_judge_verdict(values) for values in by_judge.values()]
        panel = derive_panel_winner_condition(job, verdicts)
        if panel != "unresolved":
            complete += 1
    print(f"run_dir: {run_dir}")
    print(f"jobs: {len(jobs)}")
    print(f"generations: {len(generations)}")
    print(f"judge_vote_rows: {sum(len(v) for v in votes.values())}")
    print(f"panel_resolved_pairs: {complete}/{len(jobs)}")


def plot_cells(cells: pd.DataFrame, cfg: dict[str, Any], out: Path, *, domain: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if domain is not None and "domain" in cells.columns:
        cells = cells[cells["domain"].fillna("").astype(str).eq(domain)].copy()
    task_order = [task for task in DEFAULT_TASK_ORDER if task in set(cells["task"].astype(str))]
    if not task_order:
        raise ValueError("no task rows available to plot")

    colors = {
        "DeepSeek V3.2": "#2A8C9E",
        "GPT-5.4 mini": "#3A66C9",
        "GLM-5.1": "#5B6068",
        "Kimi K2.5": "#D4711B",
        "MiMo V2.5 Pro": "#2E8C5C",
        "Qwen3.5 9B": "#6E45BD",
        "Qwen3.6 Plus": "#C2304A",
    }
    fig = plt.figure(figsize=(12.8, 7.0), facecolor="white")
    gs = fig.add_gridspec(2, 2, top=0.88, bottom=0.10, left=0.10, right=0.985, hspace=0.55, wspace=0.55)
    title = cfg["title"] if domain is None else f"{cfg['title']}: {domain}"
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#1A1A1F", y=0.975)
    actor_labels = [ACTOR_LABEL[actor] for actor in ACTORS]
    for idx, task in enumerate(task_order[:4]):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        sub = cells[cells["task"].astype(str).eq(task)].copy()
        sub["actor_label"] = sub["actor"].astype(str).map(ACTOR_LABEL).fillna(sub["actor"])
        sub = sub.set_index("actor_label").reindex(actor_labels).reset_index()
        n = len(actor_labels)
        ax.axvline(0.5, color="#9CA3AF", lw=1.2, ls=(0, (4, 3)), alpha=0.9)
        for row_idx, row in sub.iterrows():
            y = n - 1 - row_idx
            if bool(row.get("familywise_ci_positive", False)):
                ax.add_patch(mpatches.Rectangle((0.005, y - 0.42), 0.99, 0.84, facecolor="#E8F4ED", edgecolor="none", zorder=0))
            rate = row.get("target_win_rate_excluding_ties")
            if not np.isfinite(rate):
                continue
            label = row["actor_label"]
            color = colors.get(label, "#555555")
            lo = float(row.get("familywise_ci_lo", math.nan))
            hi = float(row.get("familywise_ci_hi", math.nan))
            ax.hlines(y, lo, hi, color=color, lw=4.0, alpha=0.34, capstyle="round")
            ax.vlines([lo, hi], y - 0.16, y + 0.16, color=color, lw=1.2, alpha=0.45)
            ax.scatter(rate, y, s=135, color=color, edgecolor="white", linewidth=1.5, zorder=4)
            ax.text(min(hi + 0.025, 1.02), y, f"{rate:.2f}", ha="left", va="center", fontsize=9.7, color=color, fontweight="semibold")
        ax.set_yticks([n - 1 - i for i in range(n)])
        ax.set_yticklabels(actor_labels, fontsize=10.2)
        for tick, label in zip(ax.get_yticklabels(), actor_labels):
            tick.set_color(colors.get(label, "#1A1A1F"))
            tick.set_fontweight("semibold")
        ax.set_xlim(0.005, 1.06)
        ax.set_ylim(-0.55, n - 0.10)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="x", color="#E9EDF5", lw=0.6)
        ax.tick_params(axis="y", length=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#9CA3AF")
        if idx // 2 == 1:
            ax.set_xlabel(cfg.get("x_label", "Target-side win rate (ties excluded)"), fontsize=11)
        ax.text(-0.005, 1.20, TASK_LABEL.get(task, task), transform=ax.transAxes, ha="left", va="top", fontsize=13, fontweight="bold")
        n_pos = int(sub["familywise_ci_positive"].fillna(False).sum())
        ax.text(1.0, 1.20, f"{n_pos} / {len(sub)} CI-positive", transform=ax.transAxes, ha="right", va="top", fontsize=10.2, fontweight="semibold", bbox=dict(boxstyle="round,pad=0.30", facecolor="#DDF1E3" if n_pos else "#F1F2F5", edgecolor="none"))
        ns = sub["resolved_n"].dropna().astype(int)
        if len(ns):
            note = f"n = {int(ns.min())}-{int(ns.max())} pairs / actor" if int(ns.min()) != int(ns.max()) else f"n = {int(ns.min())} pairs / actor"
            ax.text(-0.005, 1.06, f"{note}; FWER 95% CIs", transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#6B7280")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240)
    plt.close(fig)
    print(f"figure: {out}")


def cmd_list(args: argparse.Namespace) -> None:
    configs = load_config(args.config)
    for name, cfg in sorted(configs.items()):
        print(f"{name}\t{cfg.get('title', name)}\t{cfg.get('pair_outcomes', '')}")


def cmd_audit(args: argparse.Namespace) -> None:
    configs = load_config(args.config)
    rows = []
    for name, raw in sorted(configs.items()):
        cfg = dict(raw, id=name)
        path = repo_path(cfg["pair_outcomes"])
        exists = path.exists()
        row = {"comparison": name, "pair_outcomes": str(path), "exists": exists}
        if exists:
            df = read_pairs(cfg)
            missing = [col for col in ("actor", "task") if col not in df.columns]
            if "target_win" not in df.columns and cfg.get("panel_winner_col", "panel_winner_condition") not in df.columns:
                missing.append(cfg.get("panel_winner_col", "panel_winner_condition"))
            for col in (cfg.get("source_left_output_col"), cfg.get("source_right_output_col")):
                if col and col not in df.columns:
                    missing.append(col)
            row.update({"rows": len(df), "missing_columns": ",".join(missing), "status": "PASS" if not missing else "WARN"})
        else:
            row.update({"rows": 0, "missing_columns": "", "status": "MISSING"})
        rows.append(row)
    print(markdown_table(pd.DataFrame(rows)))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


def cmd_summarize(args: argparse.Namespace) -> None:
    cfg = get_comparison(load_config(args.config), args.comparison)
    paths = summarize(cfg, out_prefix=args.out_prefix)
    for key, path in paths.items():
        print(f"{key}: {path}")


def cmd_plot(args: argparse.Namespace) -> None:
    cfg = get_comparison(load_config(args.config), args.comparison)
    cells_path = args.cells or repo_path(cfg["output_prefix"]).with_name(Path(cfg["output_prefix"]).name + "_model_task_cells.csv")
    if not cells_path.exists():
        paths = summarize(cfg, out_prefix=args.out_prefix)
        cells_path = paths["model_task"]
    cells = pd.read_csv(cells_path)
    out = args.out or repo_path(cfg["output_prefix"]).with_name(Path(cfg["output_prefix"]).name + "_model_task_lollipop.png")
    plot_cells(cells, cfg, out, domain=args.domain)


def cmd_prepare_manifest(args: argparse.Namespace) -> None:
    cfg = get_comparison(load_config(args.config), args.comparison)
    prepare_manifest(cfg, repo_path(args.out_dir), limit=args.limit)


def cmd_status(args: argparse.Namespace) -> None:
    status_run(repo_path(args.run_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list")
    p.set_defaults(func=cmd_list)
    p = sub.add_parser("audit")
    p.add_argument("--json-out", type=Path)
    p.set_defaults(func=cmd_audit)
    p = sub.add_parser("summarize")
    p.add_argument("--comparison", required=True)
    p.add_argument("--out-prefix", type=Path)
    p.set_defaults(func=cmd_summarize)
    p = sub.add_parser("plot")
    p.add_argument("--comparison", required=True)
    p.add_argument("--cells", type=Path)
    p.add_argument("--out-prefix", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--domain")
    p.set_defaults(func=cmd_plot)
    p = sub.add_parser("prepare-manifest")
    p.add_argument("--comparison", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_prepare_manifest)
    p = sub.add_parser("status")
    p.add_argument("--run-dir", type=Path, required=True)
    p.set_defaults(func=cmd_status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

