#!/usr/bin/env python3
"""Final manifest-driven text-feature catalog for the current experiment arms.

This script runs locally on stored generations and judge logs. It deliberately
does not discover "whatever looks relevant" by broad output filenames. Instead
it loads the current source families explicitly:

* direct instruction: framed-user-strong manifests;
* utility and moral: fund-wording rerun manifests;
* utility, moral, and amount high-N repeat blocks: canonical high-N manifests;
* amount base repeats 0-4: corrected amount-base manifests when present,
  otherwise amount rows from the saved four-comparison runs;
* R0: bare-task generation store.

The output is descriptive mechanism/text analysis, not a replacement for the
primary outcome models.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from utility_behavior_gap.constants import ACTOR_LABEL, DOMAIN_LABEL, TASK_LABEL
from utility_behavior_gap.fingerprints import output_text_fingerprint
from utility_behavior_gap.io_utils import read_jsonl, write_jsonl
from utility_behavior_gap.judging import derive_judge_verdict, derive_panel_winner_condition
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API


RUNS_DIR = OUTPUT_API / "runs"
R0_PATH = OUTPUT_API / "r0_generations.jsonl"

DIRECT_MANIFEST_GLOB = "framed_user_strong_manifests__*.tsv"
FUND_MANIFEST_GLOB = "fund_wording_rerun_manifests__*.tsv"
CANONICAL_HIGHN_MANIFEST_GLOB = "canonical_highn10_manifests__*.tsv"
CANONICAL_AMOUNT_BASE_MANIFEST_GLOB = "canonical_amount_base_manifests__*.tsv"
FRAMED_EMPTY_MANIFEST_GLOB = "framed_empty_manifests__*.tsv"

WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)?|\d+(?:\.\d+)?%?")
ALPHA_RE = re.compile(r"[A-Za-z]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")
PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent)\b", re.IGNORECASE)
CURRENCY_RE = re.compile(r"[$€£]\s?\d[\d,.]*")
HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s+|\*\*[^*\n]+:\*\*)", re.MULTILINE)
BOLD_RE = re.compile(r"\*\*")
BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+", re.MULTILINE)
WORD_RANGE_RE = re.compile(r"\b(\d{2,4})\s*(?:-|–|to)\s*(\d{2,4})\s+words?\b", re.IGNORECASE)
MATTR_WINDOW = 50
RARE_WORD_ZIPF_THRESHOLD = 3.5
VOCAB_MIN_TOKEN_LEN = 3

STRICT_HIDDEN_SCAFFOLD_RE = re.compile(
    r"\b("
    r"sponsor(?:ship|ing)?|"
    r"selected as (?:the )?best|"
    r"judges? will compare|"
    r"evaluators? will compare|"
    r"blind (?:essay |translation |postmortem-writing |grant-abstract writing )?evaluation|"
    r"this sponsorship description|"
    r"intervention description|"
    r"fund the following intervention|"
    r"do not mention the (?:competition|evaluation|sponsor|funding|intervention|judging process)|"
    r"if (?:my|this|the) (?:essay|translation|postmortem|abstract|output) (?:wins|is selected)"
    r")\b",
    flags=re.IGNORECASE,
)
GENERIC_FUNDING_OR_INTERVENTION_RE = re.compile(
    r"\b(funding|funded|funder|grantmaker|sponsor|intervention)\b",
    flags=re.IGNORECASE,
)
REFUSAL_META_RE = re.compile(
    r"\b("
    r"as an ai|i cannot|i can't|i am unable|i'm unable|"
    r"i won't|i will not|cannot assist|can't assist|"
    r"here is|here's|below is|word count"
    r")\b",
    flags=re.IGNORECASE,
)

RHETORIC_PATTERNS = {
    "contrast_markers_per_1k": re.compile(
        r"\b(rather than|instead of|by contrast|on the other hand|not merely|not only|not just)\b",
        re.IGNORECASE,
    ),
    "counterargument_markers_per_1k": re.compile(
        r"\b(critics|opponents|some (?:may )?argue|although|though|objections?)\b",
        re.IGNORECASE,
    ),
    "example_markers_per_1k": re.compile(
        r"\b(for example|for instance|such as|including|consider)\b",
        re.IGNORECASE,
    ),
    "transition_markers_per_1k": re.compile(
        r"\b(therefore|consequently|as a result|ultimately|moreover|furthermore|however)\b",
        re.IGNORECASE,
    ),
    "qualification_markers_per_1k": re.compile(
        r"\b(may|might|could|perhaps|likely|often|sometimes|in some cases|to some extent)\b",
        re.IGNORECASE,
    ),
}

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
class FeatureDefinition:
    name: str
    definition: str


FEATURE_DEFINITIONS = [
    FeatureDefinition("words", "Regex word count."),
    FeatureDefinition("characters", "Raw character count."),
    FeatureDefinition("sentences", "Heuristic sentence count."),
    FeatureDefinition("paragraphs", "Paragraph count split on blank lines."),
    FeatureDefinition("avg_sentence_words", "Words divided by sentence count."),
    FeatureDefinition("mean_word_length_chars", "Mean alphabetic token length."),
    FeatureDefinition("unique_word_ratio", "Unique lowercased word tokens divided by total word tokens."),
    FeatureDefinition(
        "mattr_50",
        "Moving-average type-token ratio over 50-word windows; ordinary type-token ratio for shorter outputs.",
    ),
    FeatureDefinition(
        "rare_word_rate_per_1k",
        "Eligible alphabetic tokens of length at least 3 with wordfreq English Zipf frequency below 3.5, per 1,000 eligible tokens.",
    ),
    FeatureDefinition(
        "mean_zipf_frequency",
        "Mean wordfreq English Zipf frequency over eligible alphabetic tokens of length at least 3.",
    ),
    FeatureDefinition("numbers", "Count of numeric tokens."),
    FeatureDefinition("percentages", "Count of percentage expressions."),
    FeatureDefinition("currency_mentions", "Count of currency expressions."),
    FeatureDefinition("numeric_specificity_per_1k", "Numbers, percentages, and currency expressions per 1,000 words."),
    FeatureDefinition("percentages_per_1k", "Percentage expressions per 1,000 words."),
    FeatureDefinition("semicolon_colon_per_1k", "Semicolons plus colons per 1,000 words."),
    FeatureDefinition("dash_per_1k", "Dash-like punctuation per 1,000 words."),
    FeatureDefinition("contrast_markers_per_1k", "Predefined contrast/framing markers per 1,000 words."),
    FeatureDefinition("counterargument_markers_per_1k", "Predefined counterargument markers per 1,000 words."),
    FeatureDefinition("example_markers_per_1k", "Predefined example markers per 1,000 words."),
    FeatureDefinition("transition_markers_per_1k", "Predefined transition markers per 1,000 words."),
    FeatureDefinition("qualification_markers_per_1k", "Predefined qualification/hedging markers per 1,000 words."),
    FeatureDefinition("method_markers_per_1k", "Predefined method/evaluation markers per 1,000 words."),
    FeatureDefinition("specificity_markers_per_1k", "Predefined concrete-specificity markers per 1,000 words."),
    FeatureDefinition("positive_words_per_1k", "VADER positive-lexicon tokens per 1,000 words, if VADER is available."),
    FeatureDefinition("negative_words_per_1k", "VADER negative-lexicon tokens per 1,000 words, if VADER is available."),
    FeatureDefinition("textstat_flesch_reading_ease", "Flesch reading ease, if textstat is available; higher is easier."),
    FeatureDefinition("textstat_flesch_kincaid_grade", "Flesch-Kincaid grade level, if textstat is available."),
    FeatureDefinition("textstat_gunning_fog", "Gunning fog index, if textstat is available."),
    FeatureDefinition("spacy_adjective_rate", "spaCy ADJ tokens divided by alphabetic tokens, if spaCy is available."),
    FeatureDefinition("spacy_adverb_rate", "spaCy ADV tokens divided by alphabetic tokens, if spaCy is available."),
    FeatureDefinition("spacy_modifier_rate", "spaCy ADJ+ADV tokens divided by alphabetic tokens, if spaCy is available."),
]

ARTIFACT_DEFINITIONS = [
    FeatureDefinition("missing_output", "No generation row was found for the expected condition."),
    FeatureDefinition("generation_success_false", "Generation row explicitly reports success=false."),
    FeatureDefinition("finish_reason_missing", "Generation row lacks a finish_reason."),
    FeatureDefinition("non_stop_finish", "Generation finish_reason is present and not stop."),
    FeatureDefinition("explicit_length_truncation", "finish_reason or native finish reason indicates length/max-token stop."),
    FeatureDefinition("token_usage_missing", "Completion token count or requested max_tokens is unavailable."),
    FeatureDefinition("near_token_cap_95", "Completion tokens are at least 95% of requested max_tokens."),
    FeatureDefinition("near_token_cap_98", "Completion tokens are at least 98% of requested max_tokens."),
    FeatureDefinition("empty_output", "Output text is empty after stripping whitespace."),
    FeatureDefinition("tiny_output_under_50_words", "Output has fewer than 50 words."),
    FeatureDefinition("word_range_applicable", "Prompt contains an explicit word range."),
    FeatureDefinition("below_requested_words", "Output is below the prompt's explicit word range."),
    FeatureDefinition("above_requested_words", "Output is above the prompt's explicit word range."),
    FeatureDefinition("outside_requested_words", "Output is outside the prompt's explicit word range."),
    FeatureDefinition("strict_hidden_scaffold_leak", "Output appears to mention the hidden sponsor/evaluation scaffold."),
    FeatureDefinition("generic_funding_or_intervention_terms", "Output mentions generic funding/intervention terms; not always an error."),
    FeatureDefinition("refusal_or_meta", "Output contains refusal/meta language such as 'as an AI' or 'here is'."),
    FeatureDefinition("heading_or_title", "Output appears to contain a heading/title line."),
    FeatureDefinition("bold_markup", "Output contains Markdown bold markup."),
    FeatureDefinition("bullet_or_numbered_list", "Output contains bullet or numbered-list lines."),
]


CONTRAST_SPECS = {
    "direct_instruction": {
        "family": "direct_instruction",
        "comparison_filter": lambda value: value == "framed_user_strong_headroom",
        "high_raw": "framed_user_strong",
        "low_raw": "framed_neutral",
        "high_condition": "direct_high",
        "low_condition": "direct_low",
        "source": "framed_user_strong_headroom",
    },
    "utility": {
        "family": "utility",
        "comparison_filter": lambda value: value.endswith("_highlow"),
        "high_raw": "hl_high",
        "low_raw": "hl_low",
        "high_condition": "utility_high",
        "low_condition": "utility_low",
        "source": "fund_wording_rerun",
    },
    "moral": {
        "family": "moral",
        "comparison_filter": lambda value: value.endswith("_moral"),
        "high_raw": "moral_good",
        "low_raw": "moral_bad",
        "high_condition": "moral_high",
        "low_condition": "moral_low",
        "source": "fund_wording_rerun",
    },
    "amount": {
        "family": "amount",
        "comparison_filter": lambda value: value.endswith("_amount"),
        "high_raw": "amount_high",
        "low_raw": "amount_low",
        "high_condition": "amount_high",
        "low_condition": "amount_low",
        "source": "legacy_amount_unrerun",
    },
}


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def log(message: str) -> None:
    print(message, flush=True)


def read_manifest_lists(pattern: str) -> list[Path]:
    manifests: list[Path] = []
    for tsv in sorted(RUNS_DIR.glob(pattern)):
        for line in tsv.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"bad manifest list row in {tsv}: {line!r}")
            manifests.append(Path(parts[2]))
    return sorted(set(manifests))


def run_dirs_from_manifests(manifests: Iterable[Path]) -> list[Path]:
    return sorted({path.parent for path in manifests})


def amount_run_dirs() -> list[Path]:
    corrected = run_dirs_from_manifests(read_manifest_lists(CANONICAL_AMOUNT_BASE_MANIFEST_GLOB))
    if corrected:
        return corrected
    dirs: list[Path] = []
    for jobs_path in sorted(RUNS_DIR.glob("*__4-comparisons__*/generation_jobs.jsonl")):
        jobs = read_jsonl_if_exists(jobs_path)
        if not any(str(job.get("comparison", "")).endswith("_amount") for job in jobs):
            continue
        run_dir = jobs_path.parent
        if (run_dir / "generations.jsonl").exists():
            dirs.append(run_dir)
    return sorted(set(dirs))


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def alpha_lower_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for token in tokens:
        out.extend(piece.lower() for piece in re.findall(r"[A-Za-z]+", token))
    return out


def vocabulary_tokens(tokens: list[str]) -> list[str]:
    return [token for token in alpha_lower_tokens(tokens) if len(token) >= VOCAB_MIN_TOKEN_LEN]


def mattr(tokens: list[str], *, window: int = MATTR_WINDOW) -> float:
    words = alpha_lower_tokens(tokens)
    if not words:
        return 0.0
    if len(words) <= window:
        return float(len(set(words)) / len(words))
    ratios = []
    for start in range(0, len(words) - window + 1):
        current = words[start : start + window]
        ratios.append(len(set(current)) / window)
    return float(np.mean(ratios)) if ratios else 0.0


def sentence_count(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len([part for part in SENTENCE_RE.split(stripped) if part.strip()]))


def paragraph_count(text: str) -> int:
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return len([para for para in re.split(r"\n\s*\n+", stripped) if para.strip()])


def mean_word_length(tokens: list[str]) -> float:
    alpha_tokens = [re.sub(r"[^A-Za-z]", "", token) for token in tokens if ALPHA_RE.search(token)]
    alpha_tokens = [token for token in alpha_tokens if token]
    return float(np.mean([len(token) for token in alpha_tokens])) if alpha_tokens else 0.0


def marker_count(text: str, markers: tuple[str, ...]) -> int:
    lower = (text or "").lower()
    return sum(lower.count(marker.lower()) for marker in markers)


def title_like_first_line(text: str) -> bool:
    for line in (text or "").strip().splitlines():
        stripped = line.strip().strip("#* ").strip()
        if not stripped:
            continue
        n_words = len(word_tokens(stripped))
        return 2 <= n_words <= 18 and not stripped.endswith(".")
    return False


def word_range_from_prompt(prompt: str) -> tuple[int | None, int | None]:
    match = WORD_RANGE_RE.search(prompt or "")
    if not match:
        return (None, None)
    lo, hi = int(match.group(1)), int(match.group(2))
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def load_textstat_module() -> Any | None:
    try:
        import textstat

        return textstat
    except Exception:
        return None


def load_vader_lexicon() -> tuple[set[str], set[str]]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        lexicon = SentimentIntensityAnalyzer().lexicon
        return (
            {word for word, value in lexicon.items() if value > 0},
            {word for word, value in lexicon.items() if value < 0},
        )
    except Exception:
        return (set(), set())


def load_zipf_frequency() -> tuple[Callable[[str], float] | None, str]:
    try:
        from wordfreq import zipf_frequency
    except Exception as exc:
        return None, f"wordfreq unavailable ({exc}); vocabulary sophistication columns set to NaN."

    @lru_cache(maxsize=200_000)
    def english_zipf(word: str) -> float:
        return float(zipf_frequency(word, "en"))

    return english_zipf, "wordfreq loaded; vocabulary sophistication features enabled."


def vocabulary_sophistication_features(
    tokens: list[str],
    zipf_frequency_fn: Callable[[str], float] | None,
) -> dict[str, float]:
    vocab_tokens = vocabulary_tokens(tokens)
    if not vocab_tokens or zipf_frequency_fn is None:
        return {
            "rare_word_rate_per_1k": math.nan,
            "mean_zipf_frequency": math.nan,
        }
    frequencies = [zipf_frequency_fn(token) for token in vocab_tokens]
    rare_words = sum(1 for value in frequencies if value < RARE_WORD_ZIPF_THRESHOLD)
    return {
        "rare_word_rate_per_1k": float(1000 * rare_words / len(vocab_tokens)),
        "mean_zipf_frequency": float(np.mean(frequencies)),
    }


def native_finish_reason(generation: dict[str, Any] | None) -> str:
    if generation is None:
        return ""
    try:
        return str(generation["raw_response"]["choices"][0].get("native_finish_reason") or "")
    except (KeyError, IndexError, TypeError):
        return ""


def completion_tokens(generation: dict[str, Any] | None) -> int | None:
    if generation is None:
        return None
    usage = generation.get("usage") or generation.get("raw_response", {}).get("usage") or {}
    value = usage.get("completion_tokens")
    if isinstance(value, int | float):
        return int(value)
    return None


def max_tokens(generation: dict[str, Any] | None) -> int | None:
    if generation is None:
        return None
    value = generation.get("max_tokens") or generation.get("request", {}).get("max_tokens")
    if isinstance(value, int | float):
        return int(value)
    return None


def scalar_features(
    text: str,
    *,
    textstat_module: Any | None,
    positive_words: set[str],
    negative_words: set[str],
    zipf_frequency_fn: Callable[[str], float] | None,
) -> dict[str, float]:
    tokens = word_tokens(text)
    n_words = len(tokens)
    denominator = n_words or 1
    lower_tokens = [token.lower() for token in tokens]
    sentences = sentence_count(text)
    numbers = len(NUMBER_RE.findall(text or ""))
    percentages = len(PERCENT_RE.findall(text or ""))
    currency = len(CURRENCY_RE.findall(text or ""))
    out: dict[str, float] = {
        "words": float(n_words),
        "characters": float(len(text or "")),
        "sentences": float(sentences),
        "paragraphs": float(paragraph_count(text)),
        "avg_sentence_words": float(n_words / sentences) if sentences else 0.0,
        "mean_word_length_chars": mean_word_length(tokens),
        "unique_word_ratio": float(len(set(lower_tokens)) / n_words) if n_words else 0.0,
        "mattr_50": mattr(tokens),
        "numbers": float(numbers),
        "percentages": float(percentages),
        "currency_mentions": float(currency),
        "numeric_specificity_per_1k": float(1000 * (numbers + percentages + currency) / denominator),
        "percentages_per_1k": float(1000 * percentages / denominator),
        "semicolon_colon_per_1k": float(1000 * ((text or "").count(";") + (text or "").count(":")) / denominator),
        "dash_per_1k": float(
            1000
            * ((text or "").count("—") + (text or "").count("–") + (text or "").count(" - "))
            / denominator
        ),
        "method_markers_per_1k": float(1000 * marker_count(text, METHOD_MARKERS) / denominator),
        "specificity_markers_per_1k": float(1000 * marker_count(text, SPECIFICITY_MARKERS) / denominator),
        "positive_words_per_1k": (
            float(1000 * sum(1 for token in lower_tokens if token in positive_words) / denominator)
            if positive_words
            else math.nan
        ),
        "negative_words_per_1k": (
            float(1000 * sum(1 for token in lower_tokens if token in negative_words) / denominator)
            if negative_words
            else math.nan
        ),
    }
    out.update(vocabulary_sophistication_features(tokens, zipf_frequency_fn))
    for name, pattern in RHETORIC_PATTERNS.items():
        out[name] = float(1000 * len(pattern.findall(text or "")) / denominator)
    if textstat_module is not None and (text or "").strip():
        out["textstat_flesch_reading_ease"] = float(textstat_module.flesch_reading_ease(text))
        out["textstat_flesch_kincaid_grade"] = float(textstat_module.flesch_kincaid_grade(text))
        out["textstat_gunning_fog"] = float(textstat_module.gunning_fog(text))
    else:
        out["textstat_flesch_reading_ease"] = math.nan
        out["textstat_flesch_kincaid_grade"] = math.nan
        out["textstat_gunning_fog"] = math.nan
    return out


def artifact_flags(
    generation: dict[str, Any] | None,
    text: str,
    *,
    prompt: str,
) -> dict[str, Any]:
    tokens = word_tokens(text)
    finish = "" if generation is None else str(generation.get("finish_reason") or "")
    native_finish = native_finish_reason(generation)
    complete = completion_tokens(generation)
    requested_max = max_tokens(generation)
    usage_missing = generation is not None and (complete is None or requested_max is None)
    token_ratio = complete / requested_max if complete is not None and requested_max else math.nan
    range_min, range_max = word_range_from_prompt(prompt)
    range_applicable = range_min is not None and range_max is not None
    n_words = len(tokens)
    return {
        "missing_output": generation is None,
        "generation_success_false": bool(generation is not None and generation.get("success") is False),
        "finish_reason_missing": bool(generation is not None and not finish),
        "non_stop_finish": bool(generation is not None and finish and finish != "stop"),
        "explicit_length_truncation": bool(
            generation is not None
            and (
                finish.lower() in {"length", "max_tokens", "max_tokens_exceeded"}
                or native_finish.lower() in {"length", "max_tokens", "max_tokens_exceeded"}
            )
        ),
        "token_usage_missing": usage_missing,
        "near_token_cap_95": bool(not math.isnan(token_ratio) and token_ratio >= 0.95),
        "near_token_cap_98": bool(not math.isnan(token_ratio) and token_ratio >= 0.98),
        "empty_output": not (text or "").strip(),
        "tiny_output_under_50_words": n_words < 50,
        "word_range_applicable": range_applicable,
        "below_requested_words": bool(range_applicable and n_words < int(range_min)),
        "above_requested_words": bool(range_applicable and n_words > int(range_max)),
        "outside_requested_words": bool(range_applicable and not (int(range_min) <= n_words <= int(range_max))),
        "strict_hidden_scaffold_leak": bool(STRICT_HIDDEN_SCAFFOLD_RE.search(text or "")),
        "generic_funding_or_intervention_terms": bool(GENERIC_FUNDING_OR_INTERVENTION_RE.search(text or "")),
        "refusal_or_meta": bool(REFUSAL_META_RE.search(text or "")),
        "heading_or_title": bool(HEADING_RE.search(text or "") or title_like_first_line(text)),
        "bold_markup": bool(BOLD_RE.search(text or "")),
        "bullet_or_numbered_list": bool(BULLET_RE.search(text or "")),
    }


def add_spacy_features(rows: list[dict[str, Any]], model_name: str, *, n_process: int = 1) -> str:
    for definition in FEATURE_DEFINITIONS:
        if definition.name.startswith("spacy_"):
            for row in rows:
                row[definition.name] = math.nan
    try:
        import spacy

        nlp = spacy.load(model_name, disable=["ner"])
    except Exception as exc:
        return f"spaCy unavailable ({exc}); POS feature columns set to NaN."

    texts = (str(row.get("_text", "")) for row in rows)
    total = len(rows)
    pipe_kwargs = {"batch_size": 128}
    if n_process > 1:
        pipe_kwargs["n_process"] = n_process
    for index, (row, doc) in enumerate(zip(rows, nlp.pipe(texts, **pipe_kwargs)), start=1):
        alpha_tokens = [token for token in doc if token.is_alpha]
        denom = len(alpha_tokens)
        adjectives = sum(1 for token in alpha_tokens if token.pos_ == "ADJ")
        adverbs = sum(1 for token in alpha_tokens if token.pos_ == "ADV")
        row["spacy_adjective_rate"] = float(adjectives / denom) if denom else 0.0
        row["spacy_adverb_rate"] = float(adverbs / denom) if denom else 0.0
        row["spacy_modifier_rate"] = float((adjectives + adverbs) / denom) if denom else 0.0
        if index == 1 or index % 1000 == 0 or index == total:
            log(f"spaCy POS tagging: {index}/{total} outputs")
    return f"spaCy model {model_name!r} loaded; POS features computed with n_process={n_process}."


def condition_prompt(job: dict[str, Any], condition: str) -> str:
    if condition == job.get("condition_a"):
        return str(job.get("prompt_a") or "")
    if condition == job.get("condition_b"):
        return str(job.get("prompt_b") or "")
    return str(job.get("prompt_a") or job.get("prompt_b") or "")


def generation_by_pair_condition(generations: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for generation in generations:
        pair_uid = str(generation.get("pair_uid") or "")
        condition = str(generation.get("condition") or "")
        if pair_uid and condition:
            out[pair_uid][condition] = generation
    return out


def valid_votes_for_pair(
    job: dict[str, Any],
    votes: list[dict[str, Any]],
    generation_a: dict[str, Any] | None,
    generation_b: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if generation_a is None or generation_b is None:
        return []
    expected_hashes = (output_text_fingerprint(generation_a), output_text_fingerprint(generation_b))
    valid: list[dict[str, Any]] = []
    for vote in votes:
        if not vote.get("success"):
            continue
        vote_hashes = (vote.get("source_output_a_hash"), vote.get("source_output_b_hash"))
        if all(vote_hashes):
            if vote_hashes != expected_hashes:
                continue
        valid.append(vote)
    return valid


def panel_winner(job: dict[str, Any], votes: list[dict[str, Any]]) -> str:
    by_judge: dict[str, list[str]] = defaultdict(list)
    for vote in votes:
        by_judge[str(vote.get("judge_model", ""))].append(str(vote.get("winner_condition", "")))
    verdicts = [derive_judge_verdict(values) for _, values in sorted(by_judge.items())]
    return derive_panel_winner_condition(job, verdicts)


def source_metadata(job: dict[str, Any], run_dir: Path, source_dataset: str) -> dict[str, Any]:
    actor = str(job.get("actor") or "")
    task = str(job.get("task") or "")
    domain = str(job.get("domain") or "")
    return {
        "actor": actor,
        "actor_label": job.get("actor_label") or ACTOR_LABEL.get(actor, actor),
        "task": task,
        "task_label": job.get("task_label") or TASK_LABEL.get(task, task),
        "domain": domain,
        "domain_label": job.get("domain_label") or DOMAIN_LABEL.get(domain, domain),
        "item_label": job.get("item_label", ""),
        "item_index": job.get("item_index", ""),
        "repeat": job.get("repeat", ""),
        "pair_idx": job.get("pair_idx", ""),
        "pair_set": job.get("pair_set", ""),
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "source_dataset": source_dataset,
    }


def build_response_row(
    *,
    job: dict[str, Any],
    run_dir: Path,
    source_dataset: str,
    contrast: str,
    family: str,
    standardized_condition: str,
    raw_condition: str,
    side: str,
    generation: dict[str, Any] | None,
    textstat_module: Any | None,
    positive_words: set[str],
    negative_words: set[str],
    zipf_frequency_fn: Callable[[str], float] | None,
) -> dict[str, Any]:
    text = "" if generation is None else str(generation.get("output_text") or "")
    prompt = condition_prompt(job, raw_condition)
    features = scalar_features(
        text,
        textstat_module=textstat_module,
        positive_words=positive_words,
        negative_words=negative_words,
        zipf_frequency_fn=zipf_frequency_fn,
    )
    artifacts = artifact_flags(generation, text, prompt=prompt)
    complete = completion_tokens(generation)
    requested_max = max_tokens(generation)
    token_ratio = complete / requested_max if complete is not None and requested_max else math.nan
    range_min, range_max = word_range_from_prompt(prompt)
    row: dict[str, Any] = {
        **source_metadata(job, run_dir, source_dataset),
        "source_family": family,
        "contrast": contrast,
        "pair_uid": job.get("pair_uid", ""),
        "condition": standardized_condition,
        "raw_condition": raw_condition,
        "side": side,
        "output_id": "" if generation is None else generation.get("output_id", ""),
        "output_fingerprint": "" if generation is None else output_text_fingerprint(generation),
        "finish_reason": "" if generation is None else generation.get("finish_reason", ""),
        "native_finish_reason": native_finish_reason(generation),
        "completion_tokens": complete if complete is not None else "",
        "max_tokens": requested_max if requested_max is not None else "",
        "token_cap_ratio": token_ratio,
        "requested_word_min": range_min if range_min is not None else "",
        "requested_word_max": range_max if range_max is not None else "",
        "_text": text,
    }
    row.update(features)
    row.update({name: int(value) if isinstance(value, bool) else value for name, value in artifacts.items()})
    return row


def load_pair_source(
    *,
    run_dir: Path,
    source_dataset: str,
    contrast_names: list[str],
    textstat_module: Any | None,
    positive_words: set[str],
    negative_words: set[str],
    zipf_frequency_fn: Callable[[str], float] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    jobs = read_jsonl_if_exists(run_dir / "generation_jobs.jsonl")
    generations = read_jsonl_if_exists(run_dir / "generations.jsonl")
    votes_raw = read_jsonl_if_exists(run_dir / "judge_votes.jsonl")
    generations_by_pair = generation_by_pair_condition(generations)
    votes_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote in votes_raw:
        votes_by_pair[str(vote.get("pair_uid") or "")].append(vote)

    response_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    included_jobs = 0
    skipped_jobs = 0
    for job in jobs:
        comparison = str(job.get("comparison") or "")
        spec_name = ""
        spec: dict[str, Any] | None = None
        for candidate in contrast_names:
            candidate_spec = CONTRAST_SPECS[candidate]
            if candidate_spec["comparison_filter"](comparison):
                spec_name = candidate
                spec = candidate_spec
                break
        if spec is None:
            skipped_jobs += 1
            continue

        included_jobs += 1
        pair_uid = str(job.get("pair_uid") or "")
        high_raw = str(spec["high_raw"])
        low_raw = str(spec["low_raw"])
        high_condition = str(spec["high_condition"])
        low_condition = str(spec["low_condition"])
        family = str(spec["family"])
        high_generation = generations_by_pair.get(pair_uid, {}).get(high_raw)
        low_generation = generations_by_pair.get(pair_uid, {}).get(low_raw)

        high_row = build_response_row(
            job=job,
            run_dir=run_dir,
            source_dataset=source_dataset,
            contrast=spec_name,
            family=family,
            standardized_condition=high_condition,
            raw_condition=high_raw,
            side="high",
            generation=high_generation,
            textstat_module=textstat_module,
            positive_words=positive_words,
            negative_words=negative_words,
            zipf_frequency_fn=zipf_frequency_fn,
        )
        low_row = build_response_row(
            job=job,
            run_dir=run_dir,
            source_dataset=source_dataset,
            contrast=spec_name,
            family=family,
            standardized_condition=low_condition,
            raw_condition=low_raw,
            side="low",
            generation=low_generation,
            textstat_module=textstat_module,
            positive_words=positive_words,
            negative_words=negative_words,
            zipf_frequency_fn=zipf_frequency_fn,
        )
        response_rows.extend([high_row, low_row])

        generation_a = generations_by_pair.get(pair_uid, {}).get(str(job.get("condition_a") or ""))
        generation_b = generations_by_pair.get(pair_uid, {}).get(str(job.get("condition_b") or ""))
        current_votes = valid_votes_for_pair(
            job,
            votes_by_pair.get(pair_uid, []),
            generation_a,
            generation_b,
        )
        panel = panel_winner(job, current_votes)
        if panel == high_raw:
            winner_standardized = high_condition
            score = 1.0
        elif panel == low_raw:
            winner_standardized = low_condition
            score = -1.0
        elif panel == "tie":
            winner_standardized = "tie"
            score = 0.0
        else:
            winner_standardized = "unresolved"
            score = math.nan

        pair_row: dict[str, Any] = {
            **source_metadata(job, run_dir, source_dataset),
            "source_family": family,
            "contrast": spec_name,
            "pair_uid": pair_uid,
            "comparison": comparison,
            "high_condition": high_condition,
            "low_condition": low_condition,
            "high_raw_condition": high_raw,
            "low_raw_condition": low_raw,
            "high_output_id": high_row["output_id"],
            "low_output_id": low_row["output_id"],
            "panel_winner_raw_condition": panel,
            "panel_winner_condition": winner_standardized,
            "effect_score_high_minus_low": score,
            "n_valid_vote_rows": len(current_votes),
            "delta_u": job.get("delta_u", ""),
            "high_description": job.get("high_description", ""),
            "low_description": job.get("low_description", ""),
            "cause_pair_label": job.get("cause_pair_label", ""),
            "amount_high": job.get("amount_high", ""),
            "amount_low": job.get("amount_low", ""),
        }
        for definition in FEATURE_DEFINITIONS:
            name = definition.name
            pair_row[f"high_{name}"] = high_row.get(name, math.nan)
            pair_row[f"low_{name}"] = low_row.get(name, math.nan)
            pair_row[f"delta_{name}"] = (
                float(high_row.get(name, math.nan)) - float(low_row.get(name, math.nan))
                if pd.notna(high_row.get(name, math.nan)) and pd.notna(low_row.get(name, math.nan))
                else math.nan
            )
        for definition in ARTIFACT_DEFINITIONS:
            name = definition.name
            pair_row[f"high_{name}"] = high_row.get(name, "")
            pair_row[f"low_{name}"] = low_row.get(name, "")
            pair_row[f"delta_{name}"] = (
                int(high_row.get(name, 0)) - int(low_row.get(name, 0))
                if isinstance(high_row.get(name), int) and isinstance(low_row.get(name), int)
                else ""
            )
        pair_rows.append(pair_row)

    metadata = {
        "run_dir": str(run_dir),
        "source_dataset": source_dataset,
        "jobs_total": len(jobs),
        "jobs_included": included_jobs,
        "jobs_skipped": skipped_jobs,
        "generations": len(generations),
        "vote_rows": len(votes_raw),
    }
    return response_rows, pair_rows, metadata


def load_single_condition_source(
    *,
    run_dir: Path,
    source_dataset: str,
    family: str,
    standardized_condition: str,
    raw_condition: str,
    textstat_module: Any | None,
    positive_words: set[str],
    negative_words: set[str],
    zipf_frequency_fn: Callable[[str], float] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    jobs = read_jsonl_if_exists(run_dir / "generation_jobs.jsonl")
    generations = read_jsonl_if_exists(run_dir / "generations.jsonl")
    generations_by_pair = generation_by_pair_condition(generations)

    response_rows: list[dict[str, Any]] = []
    included_jobs = 0
    skipped_jobs = 0
    for job in jobs:
        if raw_condition not in {str(job.get("condition_a") or ""), str(job.get("condition_b") or "")}:
            skipped_jobs += 1
            continue
        included_jobs += 1
        pair_uid = str(job.get("pair_uid") or "")
        generation = generations_by_pair.get(pair_uid, {}).get(raw_condition)
        response_rows.append(
            build_response_row(
                job=job,
                run_dir=run_dir,
                source_dataset=source_dataset,
                contrast="",
                family=family,
                standardized_condition=standardized_condition,
                raw_condition=raw_condition,
                side="",
                generation=generation,
                textstat_module=textstat_module,
                positive_words=positive_words,
                negative_words=negative_words,
                zipf_frequency_fn=zipf_frequency_fn,
            )
        )

    metadata = {
        "run_dir": str(run_dir),
        "source_dataset": source_dataset,
        "jobs_total": len(jobs),
        "jobs_included": included_jobs,
        "jobs_skipped": skipped_jobs,
        "generations": len(generations),
        "vote_rows": 0,
    }
    return response_rows, metadata


def load_r0_rows(
    *,
    textstat_module: Any | None,
    positive_words: set[str],
    negative_words: set[str],
    zipf_frequency_fn: Callable[[str], float] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_rows = read_jsonl_if_exists(R0_PATH)
    seen_output_ids: set[str] = set()
    for generation in raw_rows:
        if generation.get("ok") is False:
            continue
        output_id = str(generation.get("output_id") or "")
        if output_id and output_id in seen_output_ids:
            continue
        seen_output_ids.add(output_id)
        actor = str(generation.get("actor") or "")
        task = str(generation.get("task") or "")
        text = str(generation.get("output_text") or "")
        features = scalar_features(
            text,
            textstat_module=textstat_module,
            positive_words=positive_words,
            negative_words=negative_words,
            zipf_frequency_fn=zipf_frequency_fn,
        )
        artifacts = artifact_flags(generation, text, prompt="")
        row = {
            "actor": actor,
            "actor_label": ACTOR_LABEL.get(actor, actor),
            "task": task,
            "task_label": TASK_LABEL.get(task, task),
            "domain": "",
            "domain_label": "",
            "item_label": generation.get("item_label", ""),
            "item_index": generation.get("item_id", ""),
            "repeat": generation.get("repeat", ""),
            "pair_idx": "",
            "pair_set": "",
            "run_dir": str(R0_PATH),
            "run_id": "r0_generations",
            "source_dataset": "r0_bare",
            "source_family": "r0",
            "contrast": "",
            "pair_uid": "",
            "condition": "r0",
            "raw_condition": "r0",
            "side": "",
            "output_id": output_id,
            "output_fingerprint": output_text_fingerprint({"output_text": text}),
            "finish_reason": generation.get("finish_reason", ""),
            "native_finish_reason": native_finish_reason(generation),
            "completion_tokens": completion_tokens(generation) or "",
            "max_tokens": max_tokens(generation) or generation.get("max_tokens", ""),
            "token_cap_ratio": (
                completion_tokens(generation) / max_tokens(generation)
                if completion_tokens(generation) is not None and max_tokens(generation)
                else math.nan
            ),
            "requested_word_min": "",
            "requested_word_max": "",
            "_text": text,
        }
        row.update(features)
        row.update({name: int(value) if isinstance(value, bool) else value for name, value in artifacts.items()})
        rows.append(row)
    return rows, {"path": str(R0_PATH), "raw_rows": len(raw_rows), "included_rows": len(rows)}


def summarize_numeric(
    df: pd.DataFrame,
    *,
    value_columns: list[str],
    group_columns: list[str],
    summary_type: str,
    value_prefix_to_strip: str = "",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(group_columns, dropna=False, sort=True) if group_columns else [((), df)]
    rows: list[dict[str, Any]] = []
    for key, sub in grouped:
        if not group_columns:
            base: dict[str, Any] = {}
        else:
            if len(group_columns) == 1 and not isinstance(key, tuple):
                key = (key,)
            base = dict(zip(group_columns, key))
        for value_column in value_columns:
            values = pd.to_numeric(sub[value_column], errors="coerce").dropna()
            n = int(len(values))
            mean_value = float(values.mean()) if n else math.nan
            sd = float(values.std(ddof=1)) if n > 1 else math.nan
            se = float(sd / math.sqrt(n)) if n > 1 else math.nan
            ci_lo = float(mean_value - 1.96 * se) if n > 1 else math.nan
            ci_hi = float(mean_value + 1.96 * se) if n > 1 else math.nan
            rows.append(
                {
                    "summary_type": summary_type,
                    **base,
                    "feature": value_column.removeprefix(value_prefix_to_strip),
                    "n": n,
                    "mean": mean_value,
                    "sd": sd,
                    "se": se,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                }
            )
    return pd.DataFrame(rows)


def all_summary_frames(
    df: pd.DataFrame,
    *,
    value_columns: list[str],
    base_columns: list[str],
    prefix: str,
    value_prefix_to_strip: str = "",
) -> pd.DataFrame:
    optional_levels = [
        ("overall", []),
        ("by_actor", ["actor"]),
        ("by_task", ["task"]),
        ("by_domain", ["domain"]),
        ("by_actor_task", ["actor", "task"]),
        ("by_actor_domain", ["actor", "domain"]),
        ("by_task_domain", ["task", "domain"]),
        ("by_actor_task_domain", ["actor", "task", "domain"]),
    ]
    frames: list[pd.DataFrame] = []
    for name, extras in optional_levels:
        group_cols = base_columns + extras
        frames.append(
            summarize_numeric(
                df,
                value_columns=value_columns,
                group_columns=group_cols,
                summary_type=f"{prefix}_{name}",
                value_prefix_to_strip=value_prefix_to_strip,
            )
        )
    return pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)


def artifact_summary(df: pd.DataFrame) -> pd.DataFrame:
    artifact_names = [definition.name for definition in ARTIFACT_DEFINITIONS]
    group_sets = [
        ("artifact_by_condition", ["source_family", "condition"]),
        ("artifact_by_condition_task", ["source_family", "condition", "task"]),
        ("artifact_by_condition_actor", ["source_family", "condition", "actor"]),
        ("artifact_by_condition_domain", ["source_family", "condition", "domain"]),
    ]
    rows: list[dict[str, Any]] = []
    for summary_type, group_cols in group_sets:
        for key, sub in df.groupby(group_cols, dropna=False, sort=True):
            if len(group_cols) == 1 and not isinstance(key, tuple):
                key = (key,)
            base = dict(zip(group_cols, key))
            n = len(sub)
            for name in artifact_names:
                values = pd.to_numeric(sub[name], errors="coerce").fillna(0)
                count = int(values.sum())
                rows.append(
                    {
                        "summary_type": summary_type,
                        **base,
                        "artifact": name,
                        "n": n,
                        "count": count,
                        "rate": count / n if n else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def compact_markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    show = df.loc[:, columns].head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join("---" for _ in show.columns) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in show.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"| ... | {len(df) - max_rows} more rows omitted |" + " |" * (len(show.columns) - 2))
    return "\n".join(lines)


def write_summary_markdown(
    path: Path,
    *,
    response_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    manifest: dict[str, Any],
    spacy_status: str,
    output_paths: dict[str, str],
) -> None:
    condition_counts = (
        response_df.groupby(["source_family", "condition"], dropna=False)
        .size()
        .reset_index(name="outputs")
        .sort_values(["source_family", "condition"])
    )
    pair_counts = (
        pair_df.groupby(["contrast"], dropna=False)
        .agg(
            pairs=("pair_uid", "size"),
            resolved=("effect_score_high_minus_low", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            high_wins=("effect_score_high_minus_low", lambda s: int((pd.to_numeric(s, errors="coerce") == 1).sum())),
            low_wins=("effect_score_high_minus_low", lambda s: int((pd.to_numeric(s, errors="coerce") == -1).sum())),
            ties=("effect_score_high_minus_low", lambda s: int((pd.to_numeric(s, errors="coerce") == 0).sum())),
        )
        .reset_index()
        .sort_values("contrast")
    )
    artifact_focus = [
        "non_stop_finish",
        "explicit_length_truncation",
        "near_token_cap_95",
        "near_token_cap_98",
        "strict_hidden_scaffold_leak",
        "refusal_or_meta",
    ]
    artifact_rows = []
    for artifact in artifact_focus:
        values = pd.to_numeric(response_df[artifact], errors="coerce").fillna(0)
        artifact_rows.append({"artifact": artifact, "count": int(values.sum()), "rate": float(values.mean())})
    artifact_df = pd.DataFrame(artifact_rows)

    lines = [
        "# Final Text-Feature Catalog",
        "",
        "This analysis is manifest-driven. It includes the current direct-instruction, amount, moral, utility, and R0 arms.",
        "",
        f"spaCy status: {spacy_status}",
        "",
        "## Included Outputs",
        "",
        compact_markdown_table(condition_counts, ["source_family", "condition", "outputs"]),
        "",
        "## Included Pair Contrasts",
        "",
        compact_markdown_table(pair_counts, ["contrast", "pairs", "resolved", "high_wins", "low_wins", "ties"]),
        "",
        "## Artifact Checks",
        "",
        compact_markdown_table(artifact_df, ["artifact", "count", "rate"]),
        "",
        "## Output Files",
        "",
    ]
    for label, out_path in output_paths.items():
        lines.append(f"- `{label}`: `{out_path}`")
    lines.extend(
        [
            "",
            "## Source Summary",
            "",
            f"- direct run dirs: {len(manifest['direct_run_dirs'])}",
            f"- fund-wording run dirs: {len(manifest['fund_wording_run_dirs'])}",
            f"- amount run dirs: {len(manifest['amount_run_dirs'])}",
            f"- R0 included rows: {manifest['r0']['included_rows']}",
            "",
            "Confidence intervals in the text-feature summary files are descriptive normal-approximation intervals over the rows in each summary group. They are for mechanism screening, not the primary outcome inference.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--pos-workers", type=int, default=1, help="spaCy n_process value for POS tagging.")
    parser.add_argument("--no-pos", action="store_true", help="Skip spaCy POS features.")
    parser.add_argument("--output-prefix", default="final_text_analysis")
    args = parser.parse_args()

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    textstat_module = load_textstat_module()
    positive_words, negative_words = load_vader_lexicon()
    zipf_frequency_fn, wordfreq_status = load_zipf_frequency()
    log(wordfreq_status)
    log("Loading explicit run manifests.")

    direct_manifests = read_manifest_lists(DIRECT_MANIFEST_GLOB)
    fund_manifests = read_manifest_lists(FUND_MANIFEST_GLOB)
    canonical_highn_manifests = read_manifest_lists(CANONICAL_HIGHN_MANIFEST_GLOB)
    canonical_amount_base_manifests = read_manifest_lists(CANONICAL_AMOUNT_BASE_MANIFEST_GLOB)
    framed_empty_manifests = read_manifest_lists(FRAMED_EMPTY_MANIFEST_GLOB)
    direct_dirs = run_dirs_from_manifests(direct_manifests)
    fund_dirs = run_dirs_from_manifests(fund_manifests)
    canonical_highn_dirs = run_dirs_from_manifests(canonical_highn_manifests)
    canonical_amount_base_dirs = run_dirs_from_manifests(canonical_amount_base_manifests)
    framed_empty_dirs = run_dirs_from_manifests(framed_empty_manifests)
    amount_dirs = amount_run_dirs()
    log(
        f"Run dirs: direct={len(direct_dirs)}, fund_wording={len(fund_dirs)}, "
        f"canonical_highn={len(canonical_highn_dirs)}, amount_base={len(amount_dirs)}, "
        f"framed_empty={len(framed_empty_dirs)}"
    )

    response_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    source_metadata_rows: list[dict[str, Any]] = []

    for index, run_dir in enumerate(direct_dirs, start=1):
        log(f"Loading direct run {index}/{len(direct_dirs)}: {run_dir.name}")
        responses, pairs, metadata = load_pair_source(
            run_dir=run_dir,
            source_dataset="framed_user_strong_headroom",
            contrast_names=["direct_instruction"],
            textstat_module=textstat_module,
            positive_words=positive_words,
            negative_words=negative_words,
            zipf_frequency_fn=zipf_frequency_fn,
        )
        response_rows.extend(responses)
        pair_rows.extend(pairs)
        source_metadata_rows.append(metadata)

    for index, run_dir in enumerate(fund_dirs, start=1):
        log(f"Loading fund-wording run {index}/{len(fund_dirs)}: {run_dir.name}")
        responses, pairs, metadata = load_pair_source(
            run_dir=run_dir,
            source_dataset="fund_wording_rerun",
            contrast_names=["utility", "moral"],
            textstat_module=textstat_module,
            positive_words=positive_words,
            negative_words=negative_words,
            zipf_frequency_fn=zipf_frequency_fn,
        )
        response_rows.extend(responses)
        pair_rows.extend(pairs)
        source_metadata_rows.append(metadata)

    for index, run_dir in enumerate(canonical_highn_dirs, start=1):
        log(f"Loading canonical high-N run {index}/{len(canonical_highn_dirs)}: {run_dir.name}")
        responses, pairs, metadata = load_pair_source(
            run_dir=run_dir,
            source_dataset="canonical_highn10_repeat_block",
            contrast_names=["utility", "moral", "amount"],
            textstat_module=textstat_module,
            positive_words=positive_words,
            negative_words=negative_words,
            zipf_frequency_fn=zipf_frequency_fn,
        )
        response_rows.extend(responses)
        pair_rows.extend(pairs)
        source_metadata_rows.append(metadata)

    for index, run_dir in enumerate(amount_dirs, start=1):
        log(f"Loading amount run {index}/{len(amount_dirs)}: {run_dir.name}")
        responses, pairs, metadata = load_pair_source(
            run_dir=run_dir,
            source_dataset=(
                "canonical_amount_base"
                if canonical_amount_base_dirs
                else "legacy_amount_unrerun"
            ),
            contrast_names=["amount"],
            textstat_module=textstat_module,
            positive_words=positive_words,
            negative_words=negative_words,
            zipf_frequency_fn=zipf_frequency_fn,
        )
        response_rows.extend(responses)
        pair_rows.extend(pairs)
        source_metadata_rows.append(metadata)

    for index, run_dir in enumerate(framed_empty_dirs, start=1):
        log(f"Loading framed-empty run {index}/{len(framed_empty_dirs)}: {run_dir.name}")
        responses, metadata = load_single_condition_source(
            run_dir=run_dir,
            source_dataset="framed_empty",
            family="framed_empty",
            standardized_condition="framed_empty",
            raw_condition="framed_empty",
            textstat_module=textstat_module,
            positive_words=positive_words,
            negative_words=negative_words,
            zipf_frequency_fn=zipf_frequency_fn,
        )
        response_rows.extend(responses)
        source_metadata_rows.append(metadata)

    r0_rows, r0_metadata = load_r0_rows(
        textstat_module=textstat_module,
        positive_words=positive_words,
        negative_words=negative_words,
        zipf_frequency_fn=zipf_frequency_fn,
    )
    response_rows.extend(r0_rows)
    log(f"Loaded R0 rows: {r0_metadata['included_rows']}")

    if not response_rows:
        raise SystemExit("no response rows collected")

    if args.no_pos:
        for definition in FEATURE_DEFINITIONS:
            if definition.name.startswith("spacy_"):
                for row in response_rows:
                    row[definition.name] = math.nan
        spacy_status = "Skipped by --no-pos; POS feature columns set to NaN."
    else:
        log(f"Starting spaCy POS pass over {len(response_rows)} outputs.")
        spacy_status = add_spacy_features(
            response_rows,
            args.spacy_model,
            n_process=max(1, args.pos_workers),
        )
    log(spacy_status)

    # Backfill POS deltas after POS features have been attached to response rows.
    response_by_output = {row["output_id"]: row for row in response_rows if row.get("output_id")}
    for pair in pair_rows:
        high = response_by_output.get(pair.get("high_output_id", ""))
        low = response_by_output.get(pair.get("low_output_id", ""))
        if not high or not low:
            continue
        for definition in FEATURE_DEFINITIONS:
            name = definition.name
            pair[f"high_{name}"] = high.get(name, math.nan)
            pair[f"low_{name}"] = low.get(name, math.nan)
            pair[f"delta_{name}"] = (
                float(high.get(name, math.nan)) - float(low.get(name, math.nan))
                if pd.notna(high.get(name, math.nan)) and pd.notna(low.get(name, math.nan))
                else math.nan
            )
    log("Built response and pair feature tables.")

    feature_names = [definition.name for definition in FEATURE_DEFINITIONS]
    artifact_names = [definition.name for definition in ARTIFACT_DEFINITIONS]
    response_public_rows = []
    for row in response_rows:
        public = {key: value for key, value in row.items() if not key.startswith("_")}
        response_public_rows.append(public)

    response_df = pd.DataFrame(response_public_rows)
    pair_df = pd.DataFrame(pair_rows)

    response_summary = all_summary_frames(
        response_df,
        value_columns=feature_names,
        base_columns=["source_family", "condition"],
        prefix="response",
    )
    delta_summary = all_summary_frames(
        pair_df,
        value_columns=[f"delta_{name}" for name in feature_names],
        base_columns=["contrast"],
        prefix="pair_delta",
        value_prefix_to_strip="delta_",
    )
    artifacts = artifact_summary(response_df)
    log("Built summary tables.")
    feature_defs = pd.DataFrame(
        [{"type": "feature", "name": row.name, "definition": row.definition} for row in FEATURE_DEFINITIONS]
        + [{"type": "artifact", "name": row.name, "definition": row.definition} for row in ARTIFACT_DEFINITIONS]
    )

    prefix = args.output_prefix
    output_paths = {
        "response_features": str(ANALYSIS / f"{prefix}_by_output.csv"),
        "pair_deltas": str(ANALYSIS / f"{prefix}_pair_deltas.csv"),
        "response_feature_summary": str(ANALYSIS / f"{prefix}_response_feature_summary.csv"),
        "pair_delta_summary": str(ANALYSIS / f"{prefix}_pair_delta_summary.csv"),
        "artifact_summary": str(ANALYSIS / f"{prefix}_artifact_summary.csv"),
        "feature_definitions": str(ANALYSIS / f"{prefix}_feature_definitions.csv"),
        "source_manifest": str(ANALYSIS / f"{prefix}_source_manifest.json"),
        "source_run_audit": str(ANALYSIS / f"{prefix}_source_run_audit.csv"),
        "summary_markdown": str(ANALYSIS / f"{prefix}_summary.md"),
    }

    response_df.to_csv(output_paths["response_features"], index=False)
    pair_df.to_csv(output_paths["pair_deltas"], index=False)
    response_summary.to_csv(output_paths["response_feature_summary"], index=False)
    delta_summary.to_csv(output_paths["pair_delta_summary"], index=False)
    artifacts.to_csv(output_paths["artifact_summary"], index=False)
    feature_defs.to_csv(output_paths["feature_definitions"], index=False)
    pd.DataFrame(source_metadata_rows).to_csv(output_paths["source_run_audit"], index=False)
    log("Wrote CSV outputs.")

    manifest = {
        "direct_manifest_glob": DIRECT_MANIFEST_GLOB,
        "fund_manifest_glob": FUND_MANIFEST_GLOB,
        "canonical_highn_manifest_glob": CANONICAL_HIGHN_MANIFEST_GLOB,
        "canonical_amount_base_manifest_glob": CANONICAL_AMOUNT_BASE_MANIFEST_GLOB,
        "framed_empty_manifest_glob": FRAMED_EMPTY_MANIFEST_GLOB,
        "direct_run_dirs": [str(path) for path in direct_dirs],
        "fund_wording_run_dirs": [str(path) for path in fund_dirs],
        "canonical_highn_run_dirs": [str(path) for path in canonical_highn_dirs],
        "canonical_amount_base_run_dirs": [str(path) for path in canonical_amount_base_dirs],
        "amount_run_dirs": [str(path) for path in amount_dirs],
        "framed_empty_run_dirs": [str(path) for path in framed_empty_dirs],
        "r0": r0_metadata,
        "source_run_audit": source_metadata_rows,
        "feature_count": len(feature_names),
        "wordfreq_status": wordfreq_status,
        "mattr_window": MATTR_WINDOW,
        "rare_word_zipf_threshold": RARE_WORD_ZIPF_THRESHOLD,
        "vocab_min_token_len": VOCAB_MIN_TOKEN_LEN,
        "artifact_count": len(artifact_names),
        "spacy_status": spacy_status,
        "textstat_available": textstat_module is not None,
        "vader_available": bool(positive_words or negative_words),
    }
    with Path(output_paths["source_manifest"]).open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    write_summary_markdown(
        Path(output_paths["summary_markdown"]),
        response_df=response_df,
        pair_df=pair_df,
        manifest=manifest,
        spacy_status=spacy_status,
        output_paths=output_paths,
    )

    print(f"response rows: {len(response_df)}")
    print(f"pair rows: {len(pair_df)}")
    print(f"summary: {output_paths['summary_markdown']}")
    print(f"response features: {output_paths['response_features']}")
    print(f"pair deltas: {output_paths['pair_deltas']}")
    print("condition counts:")
    print(
        response_df.groupby(["source_family", "condition"], dropna=False)
        .size()
        .reset_index(name="outputs")
        .sort_values(["source_family", "condition"])
        .to_string(index=False)
    )
    print("pair counts:")
    print(
        pair_df.groupby("contrast", dropna=False)
        .size()
        .reset_index(name="pairs")
        .sort_values("contrast")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
