#!/usr/bin/env python3
"""Explore text features in clean max-effort essay direct-instruction runs.

The goal is descriptive, not confirmatory.  The feature definitions are simple
literal or regex counts intended to make qualitative inspection reproducible.
Only clean direct-instruction essay runs are included: generation jobs must use
``framing=clean_direct_no_outcome`` and the ``user_strong``/``user_normal``
conditions.

Feature families:

* ``delta_*`` columns are always strong minus normal within the same paired
  essay prompt.
* Bare feature names such as ``mean_word_length_chars`` and
  ``flesch_kincaid_grade`` are lightweight regex/heuristic metrics retained for
  continuity with earlier exploratory outputs.
* ``spacy_*`` features come from spaCy POS tagging with ``en_core_web_sm``.
* ``textstat_*`` features come from the textstat readability package.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from utility_behavior_gap.io_utils import read_jsonl, write_csv_rows
from utility_behavior_gap.paths import ANALYSIS, OUTPUT_API


RUNS_DIR = OUTPUT_API / "runs"
BY_PAIR_OUT = ANALYSIS / "max_effort_essay_mechanism_by_pair.csv"
SUMMARY_OUT = ANALYSIS / "max_effort_essay_mechanism_summary.csv"
ARTIFACT_OUT = ANALYSIS / "max_effort_essay_mechanism_artifacts.csv"
FEATURE_DEFINITIONS_OUT = ANALYSIS / "max_effort_essay_mechanism_feature_definitions.csv"

COMPARISON = "essay_direct_user_prompt_max_effort_full_topics"
STRONG = "user_strong"
NORMAL = "user_normal"
DEFAULT_SPACY_MODEL = "en_core_web_sm"

WORD_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)?|\d+(?:\.\d+)?%?")
ALPHA_RE = re.compile(r"[A-Za-z]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
VOWEL_RE = re.compile(r"[aeiouy]+")
ADJECTIVE_SUFFIXES = (
    "able",
    "ible",
    "al",
    "ant",
    "ary",
    "ent",
    "ful",
    "ic",
    "ical",
    "ive",
    "less",
    "ory",
    "ous",
)


def regex_count(pattern: str) -> Any:
    compiled = re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)
    return lambda text: len(compiled.findall(text))


def literal_count(*phrases: str) -> Any:
    lower_phrases = tuple(phrase.lower() for phrase in phrases)

    def count(text: str) -> int:
        lower = text.lower()
        return sum(lower.count(phrase) for phrase in lower_phrases)

    return count


@dataclass(frozen=True)
class CountFeature:
    name: str
    definition: str
    count: Any


@dataclass(frozen=True)
class FeatureDefinition:
    feature: str
    source: str
    definition: str


COUNT_FEATURES = [
    CountFeature(
        "contrast_framing",
        "Count of contrast phrases: rather than, instead of, by contrast, on the other hand, "
        "not merely, not only, not just, not simply, more than.",
        regex_count(
            r"\b(rather than|instead of|by contrast|on the other hand|"
            r"not merely|not only|not just|not simply|more than)\b"
        ),
    ),
    CountFeature(
        "counterargument",
        "Count of counterargument/concession markers: although, however, while, whereas, "
        "critics, opponents, skeptics, some argue, to be sure, admittedly, nevertheless.",
        regex_count(
            r"\b(although|however|while|whereas|critics|opponents|skeptics|"
            r"some argue|to be sure|admittedly|nevertheless)\b"
        ),
    ),
    CountFeature(
        "example_markers",
        "Count of concrete-example markers: for example, for instance, such as, including, "
        "consider, in cities like, libraries can, schools can, companies can.",
        regex_count(
            r"\b(for example|for instance|such as|including|consider|in cities like|"
            r"libraries can|schools can|companies can)\b"
        ),
    ),
    CountFeature(
        "structure_markers",
        "Count of discourse-structure markers: first, second, third, finally, moreover, "
        "furthermore, in addition, therefore, ultimately, in conclusion.",
        regex_count(
            r"\b(first|second|third|finally|moreover|furthermore|in addition|"
            r"therefore|ultimately|in conclusion)\b"
        ),
    ),
    CountFeature(
        "colon_semicolon",
        "Literal count of colon and semicolon characters.",
        lambda text: text.count(":") + text.count(";"),
    ),
    CountFeature(
        "imagistic_framing",
        "Count of recurring image/metaphor words observed before outcome coding: asphalt, "
        "frontier, crossroads, arteries, engine, cradle, pilots, passengers, pavement, "
        "bandage, disease, ladder.",
        literal_count(
            "asphalt",
            "frontier",
            "crossroads",
            "arteries",
            "engine",
            "cradle",
            "pilots",
            "passengers",
            "pavement",
            "bandage",
            "disease",
            "ladder",
        ),
    ),
]

ARTIFACT_PATTERNS = {
    "word_count_line": re.compile(r"word\s*count\s*:", flags=re.IGNORECASE),
    "markdown_heading": re.compile(r"^\s*#", flags=re.MULTILINE),
    "bold_markup": re.compile(r"\*\*"),
    "bullet_or_numbered_list": re.compile(r"^\s*(?:[-*]|\d+[.)])\s+", flags=re.MULTILINE),
    "explicit_thesis_word": re.compile(r"\bthesis\b", flags=re.IGNORECASE),
    "prompt_or_task_meta": re.compile(r"\b(prompt|essay question|word count)\b", flags=re.IGNORECASE),
    "instruction_leak": re.compile(
        r"default or merely adequate|maximum care|maximum effort|strongest essay|"
        r"maximize the final essay|complete the request",
        flags=re.IGNORECASE,
    ),
}


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def sentence_count(text: str) -> int:
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(text.strip()) if sentence.strip()]
    return len(sentences)


def alphabetic_words(token_list: list[str]) -> list[str]:
    return [token for token in token_list if ALPHA_RE.search(token)]


def syllable_count_word(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0
    if len(cleaned) <= 3:
        return 1

    count = len(VOWEL_RE.findall(cleaned))
    if cleaned.endswith("e") and not cleaned.endswith(("le", "ye")) and count > 1:
        count -= 1
    if cleaned.endswith("le") and len(cleaned) > 2 and cleaned[-3] not in "aeiouy":
        count += 1
    return max(1, count)


def flesch_kincaid_grade(token_list: list[str], sentences: int) -> float:
    alpha_tokens = alphabetic_words(token_list)
    if not alpha_tokens:
        return 0.0
    syllables = sum(syllable_count_word(token) for token in alpha_tokens)
    sentence_denominator = max(1, sentences)
    return 0.39 * (len(alpha_tokens) / sentence_denominator) + 11.8 * (
        syllables / len(alpha_tokens)
    ) - 15.59


def mean_word_length(token_list: list[str]) -> float:
    alpha_tokens = alphabetic_words(token_list)
    if not alpha_tokens:
        return 0.0
    return mean(len(re.sub(r"[^A-Za-z]", "", token)) for token in alpha_tokens)


def adjective_suffix_words(token_list: list[str]) -> int:
    alpha_tokens = alphabetic_words(token_list)
    return sum(
        len(token) >= 5 and token.lower().endswith(ADJECTIVE_SUFFIXES)
        for token in alpha_tokens
    )


def ly_adverb_words(token_list: list[str]) -> int:
    alpha_tokens = alphabetic_words(token_list)
    return sum(len(token) >= 5 and token.lower().endswith("ly") for token in alpha_tokens)


def title_like_first_line(text: str) -> bool:
    for line in text.strip().splitlines():
        stripped = line.strip().strip("#* ").strip()
        if not stripped:
            continue
        return 3 <= len(words(stripped)) <= 14 and not stripped.endswith(".")
    return False


def load_spacy_model(model_name: str) -> Any:
    """Load spaCy once, with a clear dependency error for reproducibility."""

    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is required for spacy_* mechanism features. Install it with: "
            ".venv/bin/python -m pip install spacy && "
            ".venv/bin/python -m spacy download en_core_web_sm"
        ) from exc

    try:
        return spacy.load(model_name, disable=["ner"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {model_name!r} is required. Install it with: "
            f".venv/bin/python -m spacy download {model_name}"
        ) from exc


def load_textstat_module() -> Any:
    try:
        import textstat
    except ImportError as exc:
        raise RuntimeError(
            "textstat is required for textstat_* mechanism features. Install it with: "
            ".venv/bin/python -m pip install textstat"
        ) from exc
    return textstat


def spacy_features_from_doc(doc: Any) -> dict[str, float]:
    word_tokens = [token for token in doc if token.is_alpha]
    sentences = list(doc.sents)
    adjectives = [token for token in word_tokens if token.pos_ == "ADJ"]
    adverbs = [token for token in word_tokens if token.pos_ == "ADV"]
    modifier_count = len(adjectives) + len(adverbs)
    token_count = len(word_tokens)
    return {
        "spacy_words": token_count,
        "spacy_sentences": len(sentences),
        "spacy_mean_word_length_chars": mean(len(token.text) for token in word_tokens)
        if word_tokens
        else 0.0,
        "spacy_adjective_words": len(adjectives),
        "spacy_adjective_rate": len(adjectives) / token_count if token_count else 0.0,
        "spacy_adverb_words": len(adverbs),
        "spacy_adverb_rate": len(adverbs) / token_count if token_count else 0.0,
        "spacy_modifier_words": modifier_count,
        "spacy_modifier_rate": modifier_count / token_count if token_count else 0.0,
    }


def textstat_features(text: str, textstat_module: Any) -> dict[str, float]:
    if not text.strip():
        return {
            "textstat_flesch_kincaid_grade": 0.0,
            "textstat_flesch_reading_ease": 0.0,
            "textstat_gunning_fog": 0.0,
        }
    return {
        "textstat_flesch_kincaid_grade": float(textstat_module.flesch_kincaid_grade(text)),
        "textstat_flesch_reading_ease": float(textstat_module.flesch_reading_ease(text)),
        "textstat_gunning_fog": float(textstat_module.gunning_fog(text)),
    }


def scalar_features(text: str, textstat_module: Any) -> dict[str, float]:
    token_list = words(text)
    alpha_tokens = alphabetic_words(token_list)
    lower_words = [token.lower() for token in token_list]
    sentences = sentence_count(text)
    paragraphs = [para.strip() for para in re.split(r"\n\s*\n+", text.strip()) if para.strip()]
    adjective_suffix_count = adjective_suffix_words(token_list)
    ly_adverb_count = ly_adverb_words(token_list)
    modifier_proxy_count = adjective_suffix_count + ly_adverb_count
    out: dict[str, float] = {
        "words": len(token_list),
        "alphabetic_words": len(alpha_tokens),
        "sentences": sentences,
        "paragraphs": len(paragraphs),
        "avg_sentence_words": len(token_list) / sentences if sentences else 0.0,
        "mean_word_length_chars": mean_word_length(token_list),
        "flesch_kincaid_grade": flesch_kincaid_grade(token_list, sentences),
        "unique_word_ratio": len(set(lower_words)) / len(lower_words) if lower_words else 0.0,
        "adjective_suffix_words": adjective_suffix_count,
        "adjective_suffix_rate": adjective_suffix_count / len(alpha_tokens) if alpha_tokens else 0.0,
        "ly_adverb_words": ly_adverb_count,
        "ly_adverb_rate": ly_adverb_count / len(alpha_tokens) if alpha_tokens else 0.0,
        "modifier_proxy_words": modifier_proxy_count,
        "modifier_proxy_rate": modifier_proxy_count / len(alpha_tokens) if alpha_tokens else 0.0,
        "title_like_first_line": float(title_like_first_line(text)),
    }
    for feature in COUNT_FEATURES:
        out[feature.name] = float(feature.count(text))
    for name, pattern in ARTIFACT_PATTERNS.items():
        out[name] = float(bool(pattern.search(text)))
    out.update(textstat_features(text, textstat_module))
    return out


def panel_winner(votes: list[dict[str, Any]]) -> str:
    counts = Counter(vote.get("winner_condition") for vote in votes if vote.get("success") is not False)
    if counts[STRONG] >= 2:
        return STRONG
    if counts[NORMAL] >= 2:
        return NORMAL
    if counts["tie"] >= 2:
        return "tie"
    return "no_majority"


def latest_successful_votes(votes: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for vote in votes:
        if vote.get("success") is False:
            continue
        pair_uid = str(vote.get("pair_uid", ""))
        judge_model = str(vote.get("judge_model", ""))
        if pair_uid and judge_model:
            latest[(pair_uid, judge_model)] = vote
    return latest


def clean_run_jobs(run_dir: Path) -> list[dict[str, Any]]:
    jobs_path = run_dir / "generation_jobs.jsonl"
    if not jobs_path.exists():
        return []
    jobs = read_jsonl(jobs_path)
    if not jobs:
        return []
    if {job.get("comparison") for job in jobs} != {COMPARISON}:
        return []
    if {job.get("framing") for job in jobs} != {"clean_direct_no_outcome"}:
        return []
    if {job.get("condition_a") for job in jobs} != {STRONG}:
        return []
    if {job.get("condition_b") for job in jobs} != {NORMAL}:
        return []
    if any(job.get("outcome") for job in jobs):
        return []
    return jobs


def run_pair_rows(run_dir: Path, nlp: Any, textstat_module: Any) -> list[dict[str, Any]]:
    jobs = clean_run_jobs(run_dir)
    if not jobs:
        return []
    generations_path = run_dir / "generations.jsonl"
    votes_path = run_dir / "judge_votes.jsonl"
    if not generations_path.exists() or not votes_path.exists():
        return []

    generations = read_jsonl(generations_path)
    votes = read_jsonl(votes_path)
    latest_votes = latest_successful_votes(votes)

    by_pair_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for generation in generations:
        if generation.get("success") is False:
            continue
        pair_uid = str(generation.get("pair_uid", ""))
        condition = str(generation.get("condition", ""))
        if pair_uid and condition in {STRONG, NORMAL}:
            by_pair_condition[pair_uid][condition] = generation

    feature_cache: dict[tuple[str, str], dict[str, float]] = {}
    spacy_inputs: list[tuple[tuple[str, str], str]] = []
    for job in jobs:
        pair_uid = str(job["pair_uid"])
        outputs = by_pair_condition.get(pair_uid, {})
        if STRONG not in outputs or NORMAL not in outputs:
            continue
        for condition in (STRONG, NORMAL):
            text = str(outputs[condition].get("output_text", ""))
            key = (pair_uid, condition)
            feature_cache[key] = scalar_features(text, textstat_module)
            spacy_inputs.append((key, text))

    for (key, _text), doc in zip(
        spacy_inputs,
        nlp.pipe((text for _key, text in spacy_inputs), batch_size=64),
    ):
        feature_cache[key].update(spacy_features_from_doc(doc))

    rows: list[dict[str, Any]] = []
    for job in jobs:
        pair_uid = str(job["pair_uid"])
        outputs = by_pair_condition.get(pair_uid, {})
        if STRONG not in outputs or NORMAL not in outputs:
            continue
        pair_votes = [vote for (uid, _), vote in latest_votes.items() if uid == pair_uid]
        strong = feature_cache[(pair_uid, STRONG)]
        normal = feature_cache[(pair_uid, NORMAL)]

        row: dict[str, Any] = {
            "run_id": run_dir.name,
            "actor": job.get("actor", ""),
            "pair_uid": pair_uid,
            "topic": job.get("item_id", ""),
            "winner": panel_winner(pair_votes),
            "successful_votes": len(pair_votes),
        }
        for name in strong:
            row[f"strong_{name}"] = strong[name]
            row[f"normal_{name}"] = normal[name]
            row[f"delta_{name}"] = strong[name] - normal[name]
        rows.append(row)
    return rows


def pct(num: int, den: int) -> float | str:
    return 100.0 * num / den if den else ""


def summarize_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    feature_names = [key.removeprefix("delta_") for key in rows[0] if key.startswith("delta_")]
    for actor in sorted({str(row["actor"]) for row in rows}):
        actor_rows = [row for row in rows if row["actor"] == actor]
        counts = Counter(row["winner"] for row in actor_rows)
        resolved = counts[STRONG] + counts[NORMAL]
        base = {
            "actor": actor,
            "pairs": len(actor_rows),
            "strong_wins": counts[STRONG],
            "normal_wins": counts[NORMAL],
            "ties": counts["tie"],
            "no_majority": counts["no_majority"],
            "strong_win_rate_ties_excluded": pct(counts[STRONG], resolved),
        }
        for feature in feature_names:
            deltas = [float(row[f"delta_{feature}"]) for row in actor_rows]
            base[f"mean_delta_{feature}"] = mean(deltas)
            base[f"median_delta_{feature}"] = median(deltas)
            base[f"strong_higher_{feature}"] = sum(delta > 0 for delta in deltas)
            base[f"normal_higher_{feature}"] = sum(delta < 0 for delta in deltas)
        out.append(base)
    return out


def summarize_artifacts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts = list(ARTIFACT_PATTERNS) + ["title_like_first_line"]
    out: list[dict[str, Any]] = []
    for actor in sorted({str(row["actor"]) for row in rows}):
        actor_rows = [row for row in rows if row["actor"] == actor]
        for artifact in artifacts:
            strong_count = sum(float(row[f"strong_{artifact}"]) > 0 for row in actor_rows)
            normal_count = sum(float(row[f"normal_{artifact}"]) > 0 for row in actor_rows)
            discordant = [
                row
                for row in actor_rows
                if bool(float(row[f"strong_{artifact}"])) != bool(float(row[f"normal_{artifact}"]))
                and row["winner"] in {STRONG, NORMAL}
            ]
            artifact_side_wins = 0
            artifact_side_losses = 0
            for row in discordant:
                artifact_side = STRONG if float(row[f"strong_{artifact}"]) else NORMAL
                if row["winner"] == artifact_side:
                    artifact_side_wins += 1
                else:
                    artifact_side_losses += 1
            out.append(
                {
                    "actor": actor,
                    "artifact": artifact,
                    "strong_count": strong_count,
                    "normal_count": normal_count,
                    "discordant_pairs": len(discordant),
                    "artifact_side_wins": artifact_side_wins,
                    "artifact_side_losses": artifact_side_losses,
                }
            )
    return out


def feature_definition_rows(spacy_model: str) -> list[dict[str, str]]:
    definitions = [
        FeatureDefinition("words", "regex", "Count of WORD_RE tokens, including alphabetic words and numbers."),
        FeatureDefinition("alphabetic_words", "regex", "Count of WORD_RE tokens containing at least one letter."),
        FeatureDefinition("sentences", "regex", "Sentence count from punctuation boundary regex (?<=[.!?])\\s+."),
        FeatureDefinition("paragraphs", "regex", "Paragraph count from blank-line splitting."),
        FeatureDefinition("avg_sentence_words", "regex", "WORD_RE token count divided by regex sentence count."),
        FeatureDefinition(
            "mean_word_length_chars",
            "regex",
            "Mean character length of alphabetic WORD_RE tokens; punctuation is stripped.",
        ),
        FeatureDefinition(
            "flesch_kincaid_grade",
            "heuristic",
            "Local Flesch-Kincaid grade using regex sentences and a simple vowel-group syllable counter.",
        ),
        FeatureDefinition(
            "unique_word_ratio",
            "regex",
            "Unique lowercase WORD_RE tokens divided by all WORD_RE tokens; length-sensitive.",
        ),
        FeatureDefinition(
            "adjective_suffix_words",
            "heuristic",
            "Count of alphabetic words ending in adjective-like suffixes: "
            + ", ".join(ADJECTIVE_SUFFIXES)
            + ".",
        ),
        FeatureDefinition(
            "adjective_suffix_rate",
            "heuristic",
            "adjective_suffix_words divided by alphabetic_words.",
        ),
        FeatureDefinition("ly_adverb_words", "heuristic", "Count of alphabetic words of length >=5 ending in -ly."),
        FeatureDefinition("ly_adverb_rate", "heuristic", "ly_adverb_words divided by alphabetic_words."),
        FeatureDefinition(
            "modifier_proxy_words",
            "heuristic",
            "adjective_suffix_words plus ly_adverb_words; retained for continuity with earlier outputs.",
        ),
        FeatureDefinition("modifier_proxy_rate", "heuristic", "modifier_proxy_words divided by alphabetic_words."),
        FeatureDefinition(
            "title_like_first_line",
            "regex",
            "First nonempty line has 3-14 WORD_RE tokens and does not end in a period.",
        ),
        FeatureDefinition("spacy_words", f"spaCy:{spacy_model}", "Count of spaCy tokens with token.is_alpha."),
        FeatureDefinition("spacy_sentences", f"spaCy:{spacy_model}", "Count of doc.sents using the spaCy parser."),
        FeatureDefinition(
            "spacy_mean_word_length_chars",
            f"spaCy:{spacy_model}",
            "Mean character length of spaCy alphabetic tokens.",
        ),
        FeatureDefinition(
            "spacy_adjective_words",
            f"spaCy:{spacy_model}",
            "Count of spaCy alphabetic tokens with universal POS tag ADJ.",
        ),
        FeatureDefinition(
            "spacy_adjective_rate",
            f"spaCy:{spacy_model}",
            "spacy_adjective_words divided by spacy_words.",
        ),
        FeatureDefinition(
            "spacy_adverb_words",
            f"spaCy:{spacy_model}",
            "Count of spaCy alphabetic tokens with universal POS tag ADV.",
        ),
        FeatureDefinition(
            "spacy_adverb_rate",
            f"spaCy:{spacy_model}",
            "spacy_adverb_words divided by spacy_words.",
        ),
        FeatureDefinition(
            "spacy_modifier_words",
            f"spaCy:{spacy_model}",
            "spacy_adjective_words plus spacy_adverb_words.",
        ),
        FeatureDefinition(
            "spacy_modifier_rate",
            f"spaCy:{spacy_model}",
            "spacy_modifier_words divided by spacy_words.",
        ),
        FeatureDefinition(
            "textstat_flesch_kincaid_grade",
            "textstat",
            "textstat.flesch_kincaid_grade(text); higher means a higher estimated grade level.",
        ),
        FeatureDefinition(
            "textstat_flesch_reading_ease",
            "textstat",
            "textstat.flesch_reading_ease(text); higher means easier reading.",
        ),
        FeatureDefinition(
            "textstat_gunning_fog",
            "textstat",
            "textstat.gunning_fog(text); higher means a higher estimated grade level.",
        ),
    ]
    definitions.extend(
        FeatureDefinition(feature.name, "regex/literal", feature.definition) for feature in COUNT_FEATURES
    )
    definitions.extend(
        FeatureDefinition(name, "regex", f"Boolean artifact flag from regex: {pattern.pattern}")
        for name, pattern in ARTIFACT_PATTERNS.items()
    )
    return [
        {"feature": definition.feature, "source": definition.source, "definition": definition.definition}
        for definition in definitions
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--by-pair-out", type=Path, default=BY_PAIR_OUT)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--artifact-out", type=Path, default=ARTIFACT_OUT)
    parser.add_argument("--feature-definitions-out", type=Path, default=FEATURE_DEFINITIONS_OUT)
    parser.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL)
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Include every matching clean run instead of selecting the latest run per actor.",
    )
    args = parser.parse_args()

    nlp = load_spacy_model(args.spacy_model)
    textstat_module = load_textstat_module()
    rows_by_run: dict[str, list[dict[str, Any]]] = {}
    run_mtime: dict[str, float] = {}
    skipped: list[str] = []
    for run_dir in sorted(path for path in args.runs_dir.iterdir() if path.is_dir()):
        rows = run_pair_rows(run_dir, nlp, textstat_module)
        if rows:
            rows_by_run[run_dir.name] = rows
            run_mtime[run_dir.name] = (run_dir / "generation_jobs.jsonl").stat().st_mtime
        elif (run_dir / "generation_jobs.jsonl").exists():
            skipped.append(run_dir.name)

    if not rows_by_run:
        raise ValueError("No clean max-effort essay runs found.")

    selected_runs = set(rows_by_run)
    if not args.all_runs:
        by_actor: dict[str, list[str]] = defaultdict(list)
        for run_id, rows in rows_by_run.items():
            by_actor[str(rows[0]["actor"])].append(run_id)
        selected_runs = {
            max(run_ids, key=lambda run_id: (run_mtime[run_id], run_id))
            for run_ids in by_actor.values()
        }

    all_rows: list[dict[str, Any]] = []
    for run_id in sorted(selected_runs):
        all_rows.extend(rows_by_run[run_id])

    write_csv_rows(args.by_pair_out, all_rows)
    write_csv_rows(args.summary_out, summarize_pairs(all_rows))
    write_csv_rows(args.artifact_out, summarize_artifacts(all_rows))
    write_csv_rows(args.feature_definitions_out, feature_definition_rows(args.spacy_model))

    print(f"wrote {args.by_pair_out}")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.artifact_out}")
    print(f"wrote {args.feature_definitions_out}")
    print(f"analyzed {len(all_rows)} pairs from {len({row['run_id'] for row in all_rows})} clean runs")
    if not args.all_runs:
        print("selected latest clean run per actor; pass --all-runs to include repeated actor runs")
    if skipped:
        print(f"skipped {len(skipped)} nonmatching or incomplete run dirs")


if __name__ == "__main__":
    main()
