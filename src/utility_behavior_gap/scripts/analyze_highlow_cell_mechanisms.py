#!/usr/bin/env python3
"""Inspect text mechanisms for one completed high-low intervention cell.

This script is descriptive.  It joins a single run directory's generation jobs,
generated outputs, and judge votes, then writes:

* pair-level feature deltas for high minus low outputs;
* summary tables for text features, degeneracy/artifacts, and judge position;
* a markdown file with randomly sampled high-winning examples.
"""

from __future__ import annotations

import argparse
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from utility_behavior_gap.io_utils import read_jsonl, write_csv_rows
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API


RUNS_DIR = OUTPUT_API / "runs"
WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)?|\d+(?:\.\d+)?%?")
ALPHA_RE = re.compile(r"[A-Za-z]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
VOWEL_RE = re.compile(r"[aeiouy]+")
HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s+|\*\*[^*\n]+:\*\*)", re.MULTILINE)
BOLD_RE = re.compile(r"\*\*")
BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+", re.MULTILINE)
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")
TITLE_FIRST_LINE_MIN_WORDS = 2
TITLE_FIRST_LINE_MAX_WORDS = 18


FORBIDDEN_INCENTIVE_RE = re.compile(
    r"\b("
    r"blind grant-abstract|blind evaluation|expert evaluator|sponsor|"
    r"selected as the best|selected as best|if (?:this|the) abstract|"
    r"medical intervention|terminal illness|people who identify as|"
    r"intervention description"
    r")\b",
    flags=re.IGNORECASE,
)
REFUSAL_META_RE = re.compile(
    r"\b("
    r"as an ai|i cannot|i can't|i am unable|i'm unable|"
    r"here is|here's|below is|word count"
    r")\b",
    flags=re.IGNORECASE,
)
GENERIC_FUNDING_RE = re.compile(r"\b(funding|funded|funder|grantmaker|sponsor)\b", re.IGNORECASE)


METHOD_MARKERS = (
    "randomized",
    "randomised",
    "controlled trial",
    "control group",
    "mixed-methods",
    "mixed methods",
    "pilot",
    "implementation",
    "evaluation",
    "feasibility",
    "risk",
    "mitigate",
    "scalable",
    "sustainability",
    "partnership",
    "advisory",
    "training",
    "survey",
    "interview",
    "focus group",
    "metric",
    "baseline",
    "outcome",
    "latency",
    "accuracy",
    "cost-effectiveness",
    "workflow",
)
SPECIFICITY_MARKERS = (
    "sample",
    "participants",
    "cohort",
    "sites",
    "regions",
    "prototype",
    "dashboard",
    "toolkit",
    "protocol",
    "rubric",
    "pre-",
    "post-",
    "quantitative",
    "qualitative",
    "primary outcome",
    "secondary outcome",
    "stakeholder",
)


@dataclass(frozen=True)
class TextFeature:
    name: str
    definition: str


FEATURE_DEFINITIONS = [
    TextFeature("words", "Regex word count."),
    TextFeature("characters", "Raw character count."),
    TextFeature("sentences", "Heuristic sentence count."),
    TextFeature("paragraphs", "Paragraph count split on blank lines."),
    TextFeature("avg_sentence_words", "Words divided by sentence count."),
    TextFeature("mean_word_length_chars", "Mean alphabetic token length."),
    TextFeature("unique_word_ratio", "Unique lowercased tokens divided by total tokens."),
    TextFeature("numbers", "Count of numeric tokens."),
    TextFeature("percentages", "Count of percentage expressions."),
    TextFeature("method_markers", "Count of predefined grant-method/evaluation markers."),
    TextFeature("specificity_markers", "Count of predefined concrete-specificity markers."),
    TextFeature("textstat_flesch_reading_ease", "textstat Flesch reading ease; higher is easier."),
    TextFeature("textstat_flesch_kincaid_grade", "textstat Flesch-Kincaid grade level."),
    TextFeature("textstat_gunning_fog", "textstat Gunning fog index."),
    TextFeature("spacy_adjective_rate", "spaCy ADJ tokens divided by alphabetic tokens."),
    TextFeature("spacy_adverb_rate", "spaCy ADV tokens divided by alphabetic tokens."),
    TextFeature("spacy_modifier_rate", "spaCy ADJ+ADV tokens divided by alphabetic tokens."),
]


ARTIFACT_DEFINITIONS = [
    TextFeature("missing_output", "No successful nonempty generation was found for this side."),
    TextFeature("non_stop_finish", "Generation finish_reason was not stop."),
    TextFeature("explicit_length_truncation", "OpenRouter or provider finish reason reported a length/token-cap stop."),
    TextFeature("near_token_cap_95", "Completion used at least 95% of the requested max_tokens."),
    TextFeature("near_token_cap_98", "Completion used at least 98% of the requested max_tokens."),
    TextFeature("empty_or_tiny", "Output has fewer than 100 words."),
    TextFeature("below_requested_words", "Output has fewer than 500 words."),
    TextFeature("above_requested_words", "Output has more than 700 words."),
    TextFeature("outside_requested_words", "Output is outside the requested 500-700 word range."),
    TextFeature("forbidden_incentive_leak", "Mentions the hidden evaluator/sponsor/intervention scaffold."),
    TextFeature("refusal_or_meta", "Contains refusal or meta-output language such as 'here is' or 'as an AI'."),
    TextFeature("generic_funding_word", "Mentions funding/funded/funder/grantmaker/sponsor."),
    TextFeature("heading_or_title", "Uses heading/title formatting in the output."),
    TextFeature("bold_markup", "Contains Markdown bold markup."),
    TextFeature("bullet_or_numbered_list", "Contains bullet or numbered-list lines."),
]


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "all"


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def latest_matching_run(actor: str, task: str, comparison: str) -> Path:
    matches: list[Path] = []
    for run_dir in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
        jobs_path = run_dir / "generation_jobs.jsonl"
        if not jobs_path.exists():
            continue
        jobs = read_jsonl_if_exists(jobs_path)
        if not jobs:
            continue
        if {str(job.get("actor")) for job in jobs} != {actor}:
            continue
        if {str(job.get("task")) for job in jobs} != {task}:
            continue
        if {str(job.get("comparison")) for job in jobs} != {comparison}:
            continue
        matches.append(run_dir)
    if not matches:
        raise SystemExit(f"No matching run found for actor={actor}, task={task}, comparison={comparison}.")
    return max(matches, key=lambda path: ((path / "generation_jobs.jsonl").stat().st_mtime, path.name))


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def alphabetic_words(token_list: list[str]) -> list[str]:
    return [token for token in token_list if ALPHA_RE.search(token)]


def sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(stripped) if sentence.strip()]
    return max(1, len(sentences))


def paragraph_count(text: str) -> int:
    return len([para for para in re.split(r"\n\s*\n+", text.strip()) if para.strip()])


def mean_word_length(token_list: list[str]) -> float:
    alpha_tokens = alphabetic_words(token_list)
    if not alpha_tokens:
        return 0.0
    return mean(len(re.sub(r"[^A-Za-z]", "", token)) for token in alpha_tokens)


def marker_count(text: str, markers: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(lower.count(marker.lower()) for marker in markers)


def title_like_first_line(text: str) -> bool:
    for line in text.strip().splitlines():
        stripped = line.strip().strip("#* ").strip()
        if not stripped:
            continue
        return (
            TITLE_FIRST_LINE_MIN_WORDS
            <= len(words(stripped))
            <= TITLE_FIRST_LINE_MAX_WORDS
            and not stripped.endswith(".")
        )
    return False


def load_textstat_module() -> Any:
    import textstat

    return textstat


def load_spacy_model(model_name: str) -> Any:
    import spacy

    return spacy.load(model_name, disable=["ner"])


def scalar_features(text: str, textstat_module: Any) -> dict[str, float]:
    token_list = words(text)
    alpha_tokens = alphabetic_words(token_list)
    lower_words = [token.lower() for token in token_list]
    sentences = sentence_count(text)
    out = {
        "words": float(len(token_list)),
        "characters": float(len(text)),
        "sentences": float(sentences),
        "paragraphs": float(paragraph_count(text)),
        "avg_sentence_words": float(len(token_list) / sentences) if sentences else 0.0,
        "mean_word_length_chars": float(mean_word_length(token_list)),
        "unique_word_ratio": float(len(set(lower_words)) / len(lower_words)) if lower_words else 0.0,
        "numbers": float(len(NUMBER_RE.findall(text))),
        "percentages": float(len(PERCENT_RE.findall(text))),
        "method_markers": float(marker_count(text, METHOD_MARKERS)),
        "specificity_markers": float(marker_count(text, SPECIFICITY_MARKERS)),
    }
    if text.strip():
        out.update(
            {
                "textstat_flesch_reading_ease": float(textstat_module.flesch_reading_ease(text)),
                "textstat_flesch_kincaid_grade": float(textstat_module.flesch_kincaid_grade(text)),
                "textstat_gunning_fog": float(textstat_module.gunning_fog(text)),
            }
        )
    else:
        out.update(
            {
                "textstat_flesch_reading_ease": 0.0,
                "textstat_flesch_kincaid_grade": 0.0,
                "textstat_gunning_fog": 0.0,
            }
        )
    return out


def spacy_features(texts_by_key: dict[tuple[str, str], str], model_name: str) -> dict[tuple[str, str], dict[str, float]]:
    nlp = load_spacy_model(model_name)
    keys = list(texts_by_key)
    docs = nlp.pipe((texts_by_key[key] for key in keys), batch_size=64)
    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, doc in zip(keys, docs):
        alpha_tokens = [token for token in doc if token.is_alpha]
        adjectives = [token for token in alpha_tokens if token.pos_ == "ADJ"]
        adverbs = [token for token in alpha_tokens if token.pos_ == "ADV"]
        denominator = len(alpha_tokens)
        out[key] = {
            "spacy_words": float(denominator),
            "spacy_adjectives": float(len(adjectives)),
            "spacy_adverbs": float(len(adverbs)),
            "spacy_adjective_rate": float(len(adjectives) / denominator) if denominator else 0.0,
            "spacy_adverb_rate": float(len(adverbs) / denominator) if denominator else 0.0,
            "spacy_modifier_rate": float((len(adjectives) + len(adverbs)) / denominator) if denominator else 0.0,
        }
    return out


def artifact_flags(generation: dict[str, Any] | None, text: str) -> dict[str, bool]:
    token_count = len(words(text))
    finish_reason = "" if generation is None else str(generation.get("finish_reason") or "")
    native_finish_reason = native_generation_finish_reason(generation)
    completion_tokens = generation_completion_tokens(generation)
    max_tokens = generation_max_tokens(generation)
    token_cap_ratio = (
        completion_tokens / max_tokens
        if completion_tokens is not None and max_tokens is not None and max_tokens > 0
        else 0.0
    )
    return {
        "missing_output": generation is None,
        "non_stop_finish": bool(generation is not None and finish_reason != "stop"),
        "explicit_length_truncation": bool(
            generation is not None
            and (
                finish_reason.lower() in {"length", "max_tokens", "max_tokens_exceeded"}
                or native_finish_reason.lower() in {"length", "max_tokens", "max_tokens_exceeded"}
            )
        ),
        "near_token_cap_95": token_cap_ratio >= 0.95,
        "near_token_cap_98": token_cap_ratio >= 0.98,
        "empty_or_tiny": token_count < 100,
        "below_requested_words": token_count < 500,
        "above_requested_words": token_count > 700,
        "outside_requested_words": token_count < 500 or token_count > 700,
        "forbidden_incentive_leak": bool(FORBIDDEN_INCENTIVE_RE.search(text)),
        "refusal_or_meta": bool(REFUSAL_META_RE.search(text)),
        "generic_funding_word": bool(GENERIC_FUNDING_RE.search(text)),
        "heading_or_title": bool(HEADING_RE.search(text) or title_like_first_line(text)),
        "bold_markup": bool(BOLD_RE.search(text)),
        "bullet_or_numbered_list": bool(BULLET_RE.search(text)),
    }


def native_generation_finish_reason(generation: dict[str, Any] | None) -> str:
    if generation is None:
        return ""
    try:
        return str(generation["raw_response"]["choices"][0].get("native_finish_reason") or "")
    except (KeyError, IndexError, TypeError):
        return ""


def generation_completion_tokens(generation: dict[str, Any] | None) -> int | None:
    if generation is None:
        return None
    usage = generation.get("usage") or generation.get("raw_response", {}).get("usage") or {}
    value = usage.get("completion_tokens")
    return int(value) if isinstance(value, int | float) else None


def generation_max_tokens(generation: dict[str, Any] | None) -> int | None:
    if generation is None:
        return None
    value = generation.get("max_tokens") or generation.get("request", {}).get("max_tokens")
    return int(value) if isinstance(value, int | float) else None


def latest_votes(votes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for vote in votes:
        if vote.get("success") is False:
            continue
        pair_uid = str(vote.get("pair_uid") or "")
        judge_index = str(vote.get("judge_index") or "")
        judge_model = str(vote.get("judge_model") or "")
        if pair_uid and judge_model:
            by_key[(pair_uid, judge_index, judge_model)] = vote
    return list(by_key.values())


def panel_winner(pair_votes: list[dict[str, Any]]) -> str:
    counts = Counter(str(vote.get("winner_condition") or "") for vote in pair_votes)
    if not counts:
        return "no_majority"
    top_count = max(counts.values())
    top = sorted(condition for condition, count in counts.items() if count == top_count)
    if len(top) == 1:
        return top[0]
    return "tie"


def output_position(vote: dict[str, Any], condition_by_output_id: dict[str, str], condition: str) -> str:
    output_a = str(vote.get("displayed_output_a_id") or "")
    output_b = str(vote.get("displayed_output_b_id") or "")
    if condition_by_output_id.get(output_a) == condition:
        return "A"
    if condition_by_output_id.get(output_b) == condition:
        return "B"
    return ""


def text_similarity(a: str, b: str) -> float:
    a_tokens = {token.lower() for token in words(a)}
    b_tokens = {token.lower() for token in words(b)}
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def ci_mean(values: list[float], rng: random.Random, draws: int = 5000) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    n = len(values)
    means = [mean(values[rng.randrange(n)] for _ in range(n)) for _draw in range(draws)]
    means.sort()
    return (means[int(0.025 * draws)], means[int(0.975 * draws)])


def collect_cell(run_dir: Path, spacy_model: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    jobs = read_jsonl(run_dir / "generation_jobs.jsonl")
    generations = read_jsonl(run_dir / "generations.jsonl")
    votes = latest_votes(read_jsonl(run_dir / "judge_votes.jsonl"))
    textstat_module = load_textstat_module()

    generations_by_pair_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    condition_by_output_id: dict[str, str] = {}
    for generation in generations:
        if generation.get("success") is False:
            continue
        output_id = str(generation.get("output_id") or "")
        pair_uid = str(generation.get("pair_uid") or "")
        condition = str(generation.get("condition") or "")
        text = str(generation.get("output_text") or "").strip()
        if pair_uid and condition and output_id and text:
            generations_by_pair_condition[pair_uid][condition] = generation
            condition_by_output_id[output_id] = condition

    votes_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote in votes:
        votes_by_pair[str(vote.get("pair_uid") or "")].append(vote)

    texts_by_key: dict[tuple[str, str], str] = {}
    for job in jobs:
        pair_uid = str(job["pair_uid"])
        for condition in ("high", "low"):
            generation = generations_by_pair_condition.get(pair_uid, {}).get(condition)
            texts_by_key[(pair_uid, condition)] = str(generation.get("output_text") if generation else "" or "")

    spacy_by_key = spacy_features(texts_by_key, spacy_model)

    pair_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    for job in jobs:
        pair_uid = str(job["pair_uid"])
        pair_votes = votes_by_pair.get(pair_uid, [])
        winner = panel_winner(pair_votes)
        output_by_condition = generations_by_pair_condition.get(pair_uid, {})
        features_by_condition: dict[str, dict[str, float]] = {}
        artifacts_by_condition: dict[str, dict[str, bool]] = {}
        for condition in ("high", "low"):
            generation = output_by_condition.get(condition)
            text = str(generation.get("output_text") if generation else "" or "")
            features = scalar_features(text, textstat_module)
            features.update(spacy_by_key[(pair_uid, condition)])
            artifacts = artifact_flags(generation, text)
            features_by_condition[condition] = features
            artifacts_by_condition[condition] = artifacts
            response_rows.append(
                {
                    "pair_uid": pair_uid,
                    "actor": job.get("actor", ""),
                    "task": job.get("task", ""),
                    "domain": job.get("domain", ""),
                    "item_label": job.get("item_label", ""),
                    "condition": condition,
                    "winner": winner,
                    "output_id": "" if generation is None else generation.get("output_id", ""),
                    "finish_reason": "" if generation is None else generation.get("finish_reason", ""),
                    "native_finish_reason": native_generation_finish_reason(generation),
                    "completion_tokens": generation_completion_tokens(generation) or "",
                    "max_tokens": generation_max_tokens(generation) or "",
                    "token_cap_ratio": (
                        (generation_completion_tokens(generation) or 0)
                        / (generation_max_tokens(generation) or 1)
                    )
                    if generation is not None
                    else "",
                    "words": features["words"],
                    "characters": features["characters"],
                    **{name: int(value) for name, value in artifacts.items()},
                }
            )

        high_text = texts_by_key[(pair_uid, "high")]
        low_text = texts_by_key[(pair_uid, "low")]
        counts = Counter(str(vote.get("winner_condition") or "") for vote in pair_votes)
        row: dict[str, Any] = {
            "pair_uid": pair_uid,
            "actor": job.get("actor", ""),
            "task": job.get("task", ""),
            "domain": job.get("domain", ""),
            "item_label": job.get("item_label", ""),
            "pair_idx": job.get("pair_idx", ""),
            "winner": winner,
            "high_votes": counts["high"],
            "low_votes": counts["low"],
            "tie_votes": counts["tie"],
            "delta_u": float(job.get("delta_u") or 0),
            "high_consequence": job.get("high_consequence", ""),
            "low_consequence": job.get("low_consequence", ""),
            "text_jaccard_similarity": text_similarity(high_text, low_text),
        }
        for name in features_by_condition["high"]:
            high_value = features_by_condition["high"][name]
            low_value = features_by_condition["low"][name]
            row[f"high_{name}"] = high_value
            row[f"low_{name}"] = low_value
            row[f"delta_{name}"] = high_value - low_value
        for name in artifacts_by_condition["high"]:
            high_value = int(artifacts_by_condition["high"][name])
            low_value = int(artifacts_by_condition["low"][name])
            row[f"high_{name}"] = high_value
            row[f"low_{name}"] = low_value
            row[f"delta_{name}"] = high_value - low_value
        pair_rows.append(row)

    vote_rows: list[dict[str, Any]] = []
    high_a_count_by_pair: Counter[str] = Counter()
    for vote in votes:
        pair_uid = str(vote.get("pair_uid") or "")
        high_position = output_position(vote, condition_by_output_id, "high")
        if high_position == "A":
            high_a_count_by_pair[pair_uid] += 1
        winner_condition = str(vote.get("winner_condition") or "")
        vote_rows.append(
            {
                "pair_uid": pair_uid,
                "judge_index": vote.get("judge_index", ""),
                "judge_model": vote.get("judge_model", ""),
                "high_displayed_position": high_position,
                "winner_condition": winner_condition,
                "high_vote": int(winner_condition == "high"),
                "low_vote": int(winner_condition == "low"),
                "tie_vote": int(winner_condition == "tie"),
            }
        )

    for row in pair_rows:
        row["high_displayed_as_a_votes"] = int(high_a_count_by_pair[str(row["pair_uid"])])

    metadata = {
        "run_dir": str(run_dir),
        "jobs": len(jobs),
        "generations": len(generations),
        "latest_votes": len(votes),
    }
    return pair_rows, response_rows, {"vote_rows": vote_rows, **metadata}


def summarize_feature_deltas(pair_rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    features = [definition.name for definition in FEATURE_DEFINITIONS]
    rows: list[dict[str, Any]] = []
    for feature in features:
        deltas = [float(row[f"delta_{feature}"]) for row in pair_rows]
        high_values = [float(row[f"high_{feature}"]) for row in pair_rows]
        low_values = [float(row[f"low_{feature}"]) for row in pair_rows]
        high_win_deltas = [float(row[f"delta_{feature}"]) for row in pair_rows if row["winner"] == "high"]
        low_win_deltas = [float(row[f"delta_{feature}"]) for row in pair_rows if row["winner"] == "low"]
        ci_lo, ci_hi = ci_mean(deltas, rng)
        rows.append(
            {
                "feature": feature,
                "high_mean": mean(high_values),
                "low_mean": mean(low_values),
                "mean_delta_high_minus_low": mean(deltas),
                "bootstrap_ci_lo": ci_lo,
                "bootstrap_ci_hi": ci_hi,
                "median_delta_high_minus_low": median(deltas),
                "high_greater_pairs": sum(delta > 0 for delta in deltas),
                "low_greater_pairs": sum(delta < 0 for delta in deltas),
                "equal_pairs": sum(delta == 0 for delta in deltas),
                "mean_delta_when_high_wins": mean(high_win_deltas) if high_win_deltas else "",
                "mean_delta_when_low_wins": mean(low_win_deltas) if low_win_deltas else "",
            }
        )
    return rows


def summarize_artifacts(pair_rows: list[dict[str, Any]], response_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_pairs = len(pair_rows)
    for definition in ARTIFACT_DEFINITIONS:
        name = definition.name
        high_count = sum(int(row[f"high_{name}"]) for row in pair_rows)
        low_count = sum(int(row[f"low_{name}"]) for row in pair_rows)
        high_only = [row for row in pair_rows if int(row[f"high_{name}"]) and not int(row[f"low_{name}"])]
        low_only = [row for row in pair_rows if int(row[f"low_{name}"]) and not int(row[f"high_{name}"])]
        discordant = high_only + low_only
        artifact_side_wins = 0
        artifact_side_losses = 0
        for row in discordant:
            artifact_side = "high" if int(row[f"high_{name}"]) else "low"
            if row["winner"] == artifact_side:
                artifact_side_wins += 1
            elif row["winner"] in {"high", "low"}:
                artifact_side_losses += 1
        rows.append(
            {
                "artifact": name,
                "definition": definition.definition,
                "high_count": high_count,
                "low_count": low_count,
                "high_rate": high_count / n_pairs if n_pairs else "",
                "low_rate": low_count / n_pairs if n_pairs else "",
                "high_only_pairs": len(high_only),
                "low_only_pairs": len(low_only),
                "discordant_pairs": len(discordant),
                "artifact_side_wins": artifact_side_wins,
                "artifact_side_losses": artifact_side_losses,
            }
        )
    missing_response_count = sum(int(row["missing_output"]) for row in response_rows)
    if missing_response_count:
        rows.append(
            {
                "artifact": "any_missing_response_rows",
                "definition": "Total missing response rows across both sides.",
                "high_count": "",
                "low_count": "",
                "high_rate": "",
                "low_rate": "",
                "high_only_pairs": "",
                "low_only_pairs": "",
                "discordant_pairs": missing_response_count,
                "artifact_side_wins": "",
                "artifact_side_losses": "",
            }
        )
    return rows


def summarize_winners(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, subset in [
        ("overall", pair_rows),
        ("religions", [row for row in pair_rows if row["domain"] == "religions"]),
        ("animals", [row for row in pair_rows if row["domain"] == "animals"]),
        ("countries", [row for row in pair_rows if row["domain"] == "countries"]),
        ("political", [row for row in pair_rows if row["domain"] == "political"]),
    ]:
        counts = Counter(row["winner"] for row in subset)
        resolved = counts["high"] + counts["low"]
        rows.append(
            {
                "group": key,
                "pairs": len(subset),
                "high_wins": counts["high"],
                "low_wins": counts["low"],
                "ties": counts["tie"],
                "no_majority": counts["no_majority"],
                "high_win_rate_excluding_ties": counts["high"] / resolved if resolved else "",
                "mean_delta_u": mean(float(row["delta_u"]) for row in subset) if subset else "",
            }
        )
    return rows


def summarize_vote_position(vote_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    groups.append(("all_judges", "overall", vote_rows))
    for position in ("A", "B"):
        groups.append(
            (
                "all_judges",
                position,
                [row for row in vote_rows if row["high_displayed_position"] == position],
            )
        )
    for judge in sorted({str(row["judge_model"]) for row in vote_rows}):
        judge_rows = [row for row in vote_rows if row["judge_model"] == judge]
        groups.append((judge, "overall", judge_rows))
        for position in ("A", "B"):
            groups.append(
                (
                    judge,
                    position,
                    [row for row in judge_rows if row["high_displayed_position"] == position],
                )
            )
    for judge, position, subset in groups:
        if not subset:
            continue
        high_votes = sum(int(row["high_vote"]) for row in subset)
        low_votes = sum(int(row["low_vote"]) for row in subset)
        rows.append(
            {
                "judge_model": judge,
                "high_displayed_position": position,
                "votes": len(subset),
                "high_votes": high_votes,
                "low_votes": low_votes,
                "tie_votes": sum(int(row["tie_vote"]) for row in subset),
                "high_vote_rate_excluding_ties": high_votes / max(1, high_votes + low_votes),
            }
        )
    return rows


def summarize_pair_position(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for high_a_count in range(4):
        subset = [row for row in pair_rows if int(row["high_displayed_as_a_votes"]) == high_a_count]
        counts = Counter(row["winner"] for row in subset)
        resolved = counts["high"] + counts["low"]
        rows.append(
            {
                "high_displayed_as_a_votes": high_a_count,
                "pairs": len(subset),
                "high_wins": counts["high"],
                "low_wins": counts["low"],
                "ties": counts["tie"],
                "high_win_rate_excluding_ties": counts["high"] / resolved if resolved else "",
            }
        )
    return rows


def panel_logit(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import statsmodels.api as sm

    feature_names = [
        "high_displayed_as_a_votes",
        "delta_words",
        "delta_numbers",
        "delta_percentages",
        "delta_method_markers",
        "delta_textstat_flesch_reading_ease",
        "delta_spacy_modifier_rate",
        "delta_outside_requested_words",
        "delta_forbidden_incentive_leak",
    ]
    rows = [row for row in pair_rows if row["winner"] in {"high", "low"}]
    if not rows:
        return []
    y = [1 if row["winner"] == "high" else 0 for row in rows]
    x = [[float(row[name]) for name in feature_names] for row in rows]
    x = sm.add_constant(x, has_constant="add")
    fit = sm.Logit(y, x).fit(disp=False)
    names = ["const"] + feature_names
    out: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        out.append(
            {
                "term": name,
                "coef": float(fit.params[index]),
                "se": float(fit.bse[index]),
                "z": float(fit.tvalues[index]),
                "p_value": float(fit.pvalues[index]),
                "ci_lo": float(fit.conf_int()[index][0]),
                "ci_hi": float(fit.conf_int()[index][1]),
                "odds_ratio": float(math.exp(fit.params[index])),
            }
        )
    return out


def markdown_examples(
    pair_rows: list[dict[str, Any]],
    run_dir: Path,
    *,
    limit: int,
    seed: int,
) -> str:
    jobs = {str(job["pair_uid"]): job for job in read_jsonl(run_dir / "generation_jobs.jsonl")}
    generations = read_jsonl(run_dir / "generations.jsonl")
    votes = latest_votes(read_jsonl(run_dir / "judge_votes.jsonl"))

    gen_by_pair_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for generation in generations:
        if generation.get("success") is False:
            continue
        pair_uid = str(generation.get("pair_uid") or "")
        condition = str(generation.get("condition") or "")
        if pair_uid and condition:
            gen_by_pair_condition[pair_uid][condition] = generation
    votes_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote in votes:
        votes_by_pair[str(vote.get("pair_uid") or "")].append(vote)

    high_win_rows = [row for row in pair_rows if row["winner"] == "high"]
    rng = random.Random(seed)
    rng.shuffle(high_win_rows)
    selected = high_win_rows[:limit]

    lines = [
        "# Qwen3.6 Plus Grant High-Low High-Win Examples",
        "",
        f"- run: `{run_dir.name}`",
        f"- selection: random sample among panel-level high wins, seed `{seed}`",
        f"- examples shown: `{len(selected)}`",
        "",
    ]
    for index, row in enumerate(selected, start=1):
        pair_uid = str(row["pair_uid"])
        job = jobs[pair_uid]
        high_text = str(gen_by_pair_condition[pair_uid]["high"].get("output_text") or "")
        low_text = str(gen_by_pair_condition[pair_uid]["low"].get("output_text") or "")
        pair_votes = sorted(
            votes_by_pair[pair_uid],
            key=lambda vote: (str(vote.get("judge_index") or ""), str(vote.get("judge_model") or "")),
        )
        lines.extend(
            [
                f"## Example {index}",
                "",
                f"- pair_uid: `{pair_uid}`",
                f"- domain: `{job.get('domain', '')}`",
                f"- project: `{job.get('item_label', '')}`",
                f"- vote split: high `{row['high_votes']}`, low `{row['low_votes']}`, tie `{row['tie_votes']}`",
                f"- word counts: high `{int(float(row['high_words']))}`, low `{int(float(row['low_words']))}`, delta `{int(float(row['delta_words']))}`",
                f"- high consequence: {job.get('high_consequence', '')}",
                f"- low consequence: {job.get('low_consequence', '')}",
                "",
                "### Judge Votes",
                "",
            ]
        )
        for vote in pair_votes:
            lines.append(
                f"- judge {vote.get('judge_index', '')}: `{vote.get('judge_model', '')}` -> "
                f"`{vote.get('winner_condition', '')}`; {str(vote.get('vote_raw', '')).replace(chr(10), ' / ')}"
            )
        lines.extend(
            [
                "",
                "### High Output",
                "",
                "```text",
                high_text,
                "```",
                "",
                "### Low Output",
                "",
                "```text",
                low_text,
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def markdown_summary(
    *,
    run_dir: Path,
    winner_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    pair_position_rows: list[dict[str, Any]],
    logit_rows: list[dict[str, Any]],
    outputs: list[Path],
) -> str:
    overall = winner_rows[0]
    top_features = sorted(
        feature_rows,
        key=lambda row: abs(float(row["mean_delta_high_minus_low"])),
        reverse=True,
    )
    selected_features = [
        "words",
        "textstat_flesch_reading_ease",
        "textstat_flesch_kincaid_grade",
        "mean_word_length_chars",
        "spacy_adjective_rate",
        "spacy_adverb_rate",
        "method_markers",
        "specificity_markers",
        "numbers",
    ]
    feature_lookup = {str(row["feature"]): row for row in feature_rows}
    lines = [
        "# High-Low Grant Cell Mechanism Analysis",
        "",
        f"- run: `{run_dir.name}`",
        f"- pairs: `{overall['pairs']}`",
        f"- result: high `{overall['high_wins']}`, low `{overall['low_wins']}`, ties `{overall['ties']}`",
        f"- high win rate excluding ties: `{100 * float(overall['high_win_rate_excluding_ties']):.1f}%`",
        "",
        "## Feature Deltas",
        "",
        "Deltas are high minus low within the same project/outcome pair.",
        "",
        "| feature | high mean | low mean | mean delta | 95% bootstrap CI | high greater | low greater |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for feature in selected_features:
        row = feature_lookup[feature]
        lines.append(
            f"| `{feature}` | {float(row['high_mean']):.3f} | {float(row['low_mean']):.3f} | "
            f"{float(row['mean_delta_high_minus_low']):.3f} | "
            f"[{float(row['bootstrap_ci_lo']):.3f}, {float(row['bootstrap_ci_hi']):.3f}] | "
            f"{row['high_greater_pairs']} | {row['low_greater_pairs']} |"
        )
    lines.extend(
        [
            "",
            "Largest absolute mean deltas:",
            "",
        ]
    )
    for row in top_features[:8]:
        lines.append(
            f"- `{row['feature']}`: high-low mean delta {float(row['mean_delta_high_minus_low']):.3f} "
            f"[{float(row['bootstrap_ci_lo']):.3f}, {float(row['bootstrap_ci_hi']):.3f}]"
        )
    lines.extend(
        [
            "",
            "## Degenerate Or Artifact Checks",
            "",
            "| check | high count | low count | high-only pairs | low-only pairs | artifact side wins/losses |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in artifact_rows:
        lines.append(
            f"| `{row['artifact']}` | {row['high_count']} | {row['low_count']} | "
            f"{row['high_only_pairs']} | {row['low_only_pairs']} | "
            f"{row['artifact_side_wins']}/{row['artifact_side_losses']} |"
        )
    lines.extend(
        [
            "",
            "## Judge Position",
            "",
            "| judge | high displayed as | votes | high votes | low votes | ties | high vote rate excluding ties |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in position_rows:
        lines.append(
            f"| `{row['judge_model']}` | `{row['high_displayed_position']}` | {row['votes']} | {row['high_votes']} | "
            f"{row['low_votes']} | {row['tie_votes']} | "
            f"{100 * float(row['high_vote_rate_excluding_ties']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Pair-Level Position Balance",
            "",
            "| high displayed as A among 3 judge votes | pairs | high wins | low wins | high win rate |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pair_position_rows:
        rate = row["high_win_rate_excluding_ties"]
        rate_text = "" if rate == "" else f"{100 * float(rate):.1f}%"
        lines.append(
            f"| {row['high_displayed_as_a_votes']} | {row['pairs']} | {row['high_wins']} | "
            f"{row['low_wins']} | {rate_text} |"
        )
    lines.extend(
        [
            "",
            "## Panel-Winner Logit",
            "",
            "Outcome is whether the panel-level winner was high. This is descriptive; it is not the primary confirmatory model.",
            "",
            "| term | coef | odds ratio | p |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in logit_rows:
        lines.append(
            f"| `{row['term']}` | {float(row['coef']):.3f} | {float(row['odds_ratio']):.3f} | "
            f"{float(row['p_value']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
        ]
    )
    for path in outputs:
        lines.append(f"- `{path}`")
    return "\n".join(lines) + "\n"


def write_feature_definitions(path: Path) -> None:
    rows = [
        {"kind": "feature", "name": definition.name, "definition": definition.definition}
        for definition in FEATURE_DEFINITIONS
    ] + [
        {"kind": "artifact", "name": definition.name, "definition": definition.definition}
        for definition in ARTIFACT_DEFINITIONS
    ]
    write_csv_rows(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", default="qwen3.6-plus-or")
    parser.add_argument("--task", default="grant_proposal_abstract")
    parser.add_argument("--comparison", default="highlow_intervention")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--out-prefix")
    args = parser.parse_args()

    run_dir = args.run_dir or latest_matching_run(args.actor, args.task, args.comparison)
    prefix = args.out_prefix or f"highlow_cell_mechanisms__{slug(args.actor)}__{slug(args.task)}"

    pair_rows, response_rows, metadata = collect_cell(run_dir, args.spacy_model)
    if not pair_rows:
        raise SystemExit(f"No pair rows found in {run_dir}.")

    feature_rows = summarize_feature_deltas(pair_rows, args.seed)
    artifact_rows = summarize_artifacts(pair_rows, response_rows)
    winner_rows = summarize_winners(pair_rows)
    position_rows = summarize_vote_position(metadata["vote_rows"])
    pair_position_rows = summarize_pair_position(pair_rows)
    logit_rows = panel_logit(pair_rows)

    pair_path = ANALYSIS / f"{prefix}__pair_features.csv"
    response_path = ANALYSIS / f"{prefix}__response_artifacts.csv"
    feature_path = ANALYSIS / f"{prefix}__feature_summary.csv"
    artifact_path = ANALYSIS / f"{prefix}__artifact_summary.csv"
    winner_path = ANALYSIS / f"{prefix}__winner_summary.csv"
    position_path = ANALYSIS / f"{prefix}__judge_position_summary.csv"
    pair_position_path = ANALYSIS / f"{prefix}__pair_position_summary.csv"
    logit_path = ANALYSIS / f"{prefix}__panel_logit.csv"
    definition_path = ANALYSIS / f"{prefix}__feature_definitions.csv"
    examples_path = ANALYSIS / f"{prefix}__high_win_examples.md"
    summary_path = ANALYSIS / f"{prefix}__summary.md"

    write_csv_rows(pair_path, pair_rows)
    write_csv_rows(response_path, response_rows)
    write_csv_rows(feature_path, feature_rows)
    write_csv_rows(artifact_path, artifact_rows)
    write_csv_rows(winner_path, winner_rows)
    write_csv_rows(position_path, position_rows)
    write_csv_rows(pair_position_path, pair_position_rows)
    write_csv_rows(logit_path, logit_rows)
    write_feature_definitions(definition_path)
    examples_path.write_text(
        markdown_examples(pair_rows, run_dir, limit=args.examples, seed=args.seed),
        encoding="utf-8",
    )
    outputs = [
        pair_path,
        response_path,
        feature_path,
        artifact_path,
        winner_path,
        position_path,
        pair_position_path,
        logit_path,
        definition_path,
        examples_path,
        summary_path,
    ]
    summary_path.write_text(
        markdown_summary(
            run_dir=run_dir,
            winner_rows=winner_rows,
            feature_rows=feature_rows,
            artifact_rows=artifact_rows,
            position_rows=position_rows,
            pair_position_rows=pair_position_rows,
            logit_rows=logit_rows,
            outputs=outputs,
        ),
        encoding="utf-8",
    )

    print(f"run_dir: {run_dir}")
    print(f"pairs: {len(pair_rows)}")
    print(f"responses: {len(response_rows)}")
    print(f"votes: {metadata['latest_votes']}")
    print(f"summary: {summary_path}")
    print(f"examples: {examples_path}")


if __name__ == "__main__":
    main()
