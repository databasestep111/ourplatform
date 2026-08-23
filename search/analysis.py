"""
search/analysis.py

Advanced analysis and signal-generation layer for OurPlatform.

Responsibilities
----------------
- Analyse raw documents.
- Analyse structured queries.
- Generate document statistics.
- Generate query statistics.
- Calculate term-frequency signals.
- Calculate vocabulary statistics.
- Analyse fields.
- Analyse phrases.
- Detect duplicate-like content.
- Estimate document quality.
- Estimate query complexity.
- Generate freshness signals.
- Generate length-normalisation signals.
- Generate search-quality signals.
- Provide batch analysis.
- Provide configurable analyzers.
- Provide caching hooks.
- Provide diagnostic/explain functionality.

This module does NOT:
- retrieve documents
- rank final search results
- maintain the inverted index
- perform final filtering

Those responsibilities belong to their respective layers.

Architecture
------------

                RAW DOCUMENT
                     |
                     v
              +--------------+
              |   ANALYSIS   |
              +--------------+
                     |
          +----------+----------+
          |                     |
          v                     v
    DOCUMENT SIGNALS       INDEX / SEARCH


                RAW QUERY
                     |
                     v
              QueryParser
                     |
                     v
              +--------------+
              |   ANALYSIS   |
              +--------------+
                     |
                     v
                RETRIEVAL
                     |
                     v
                 RANKING

The goal is to make analysis a reusable signal-generation layer
rather than tightly coupling analysis to retrieval or ranking.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
import string
import unicodedata

from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_MAX_TERMS = 100_000
DEFAULT_MAX_QUERY_TERMS = 512

DEFAULT_MIN_TERM_LENGTH = 1
DEFAULT_MAX_TERM_LENGTH = 256

DEFAULT_CACHE_SIZE = 10_000

WORD_PATTERN = re.compile(
    r"[^\W_]+(?:['’-][^\W_]+)*",
    re.UNICODE,
)

SENTENCE_PATTERN = re.compile(
    r"(?<=[.!?])\s+"
)

WHITESPACE_PATTERN = re.compile(
    r"\s+"
)

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

NUMBER_PATTERN = re.compile(
    r"[-+]?\d+(?:[.,]\d+)?"
)

REPEATED_CHARACTER_PATTERN = re.compile(
    r"(.)\1{3,}"
)


# ============================================================
# ENUMERATIONS
# ============================================================


class AnalysisType(str, Enum):
    DOCUMENT = "document"
    QUERY = "query"
    FIELD = "field"
    TERM = "term"
    PHRASE = "phrase"
    BATCH = "batch"


class LanguageHint(str, Enum):
    UNKNOWN = "unknown"
    LATIN = "latin"
    CYRILLIC = "cyrillic"
    GREEK = "greek"
    ARABIC = "arabic"
    HEBREW = "hebrew"
    DEVANAGARI = "devanagari"
    CJK = "cjk"
    MIXED = "mixed"


class QualityBand(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SignalType(str, Enum):
    TERM = "term"
    LENGTH = "length"
    QUALITY = "quality"
    FRESHNESS = "freshness"
    STRUCTURE = "structure"
    LANGUAGE = "language"
    DUPLICATE = "duplicate"
    QUERY = "query"


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class AnalysisConfig:
    """
    Configuration controlling analysis behaviour.
    """

    min_term_length: int = DEFAULT_MIN_TERM_LENGTH
    max_term_length: int = DEFAULT_MAX_TERM_LENGTH

    max_document_terms: int = DEFAULT_MAX_TERMS
    max_query_terms: int = DEFAULT_MAX_QUERY_TERMS

    lowercase: bool = True
    normalize_unicode: bool = True

    remove_punctuation: bool = True
    ignore_numbers: bool = False

    include_term_positions: bool = True
    include_term_frequencies: bool = True
    include_vocabulary: bool = True

    detect_language: bool = True
    detect_duplicates: bool = True
    detect_quality: bool = True
    detect_freshness: bool = True

    calculate_readability: bool = True
    calculate_structure: bool = True

    cache_enabled: bool = True
    cache_size: int = DEFAULT_CACHE_SIZE

    quality_weight_length: float = 0.15
    quality_weight_vocabulary: float = 0.20
    quality_weight_structure: float = 0.20
    quality_weight_readability: float = 0.15
    quality_weight_noise: float = 0.15
    quality_weight_diversity: float = 0.15


DEFAULT_ANALYSIS_CONFIG = AnalysisConfig()


# ============================================================
# BASIC STATISTICS
# ============================================================


@dataclass
class LengthStatistics:
    """
    Length-related document statistics.
    """

    characters: int = 0
    characters_no_whitespace: int = 0

    words: int = 0
    unique_words: int = 0

    sentences: int = 0
    paragraphs: int = 0

    average_word_length: float = 0.0
    average_sentence_length: float = 0.0

    shortest_word_length: int = 0
    longest_word_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VocabularyStatistics:
    """
    Vocabulary and lexical-diversity statistics.
    """

    vocabulary_size: int = 0
    total_terms: int = 0

    type_token_ratio: float = 0.0
    hapax_count: int = 0
    hapax_ratio: float = 0.0

    repeated_term_count: int = 0
    lexical_density: float = 0.0

    top_terms: List[Tuple[str, int]] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["top_terms"] = [
            list(item)
            for item in self.top_terms
        ]
        return data


@dataclass
class TermStatistics:
    """
    Statistics for an individual term.
    """

    term: str

    frequency: int = 0
    positions: List[int] = field(
        default_factory=list
    )

    normalized_frequency: float = 0.0

    first_position: Optional[int] = None
    last_position: Optional[int] = None

    is_numeric: bool = False
    is_url: bool = False
    is_email: bool = False

    length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# FIELD STATISTICS
# ============================================================


@dataclass
class FieldStatistics:
    """
    Statistics describing one document field.
    """

    field_name: str

    characters: int = 0
    terms: int = 0
    unique_terms: int = 0

    average_term_length: float = 0.0

    top_terms: List[Tuple[str, int]] = field(
        default_factory=list
    )

    empty: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["top_terms"] = [
            list(item)
            for item in self.top_terms
        ]
        return data


# ============================================================
# STRUCTURE STATISTICS
# ============================================================


@dataclass
class StructureStatistics:
    """
    Structural characteristics of a document.
    """

    paragraph_count: int = 0
    sentence_count: int = 0

    heading_count: int = 0
    list_item_count: int = 0

    code_block_count: int = 0
    link_count: int = 0

    email_count: int = 0
    number_count: int = 0

    punctuation_count: int = 0

    uppercase_ratio: float = 0.0
    whitespace_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# LANGUAGE SIGNALS
# ============================================================


@dataclass
class LanguageStatistics:
    """
    Lightweight language/script analysis.

    This is intentionally heuristic rather than a full
    machine-learning language classifier.
    """

    hint: LanguageHint = LanguageHint.UNKNOWN

    latin_ratio: float = 0.0
    cyrillic_ratio: float = 0.0
    greek_ratio: float = 0.0
    arabic_ratio: float = 0.0
    hebrew_ratio: float = 0.0
    devanagari_ratio: float = 0.0
    cjk_ratio: float = 0.0

    mixed_script: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["hint"] = self.hint.value
        return data


# ============================================================
# READABILITY
# ============================================================


@dataclass
class ReadabilityStatistics:
    """
    Lightweight readability signals.

    These are signals rather than authoritative linguistic
    measurements.
    """

    average_sentence_length: float = 0.0
    average_word_length: float = 0.0

    syllable_estimate: int = 0

    flesch_like_score: float = 0.0

    complexity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# DUPLICATE SIGNALS
# ============================================================


@dataclass
class DuplicateStatistics:
    """
    Duplicate and near-duplicate signals.
    """

    content_hash: str = ""

    normalized_hash: str = ""

    shingle_count: int = 0

    fingerprint: str = ""

    repeated_sentence_ratio: float = 0.0

    repeated_term_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# FRESHNESS
# ============================================================


@dataclass
class FreshnessStatistics:
    """
    Freshness-related signals.
    """

    timestamp: Optional[str] = None

    age_seconds: Optional[float] = None

    freshness_score: float = 0.0

    valid_timestamp: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# QUALITY
# ============================================================


@dataclass
class QualityStatistics:
    """
    Aggregate document quality signals.
    """

    score: float = 0.0

    band: QualityBand = QualityBand.MEDIUM

    vocabulary_score: float = 0.0
    structure_score: float = 0.0
    readability_score: float = 0.0
    diversity_score: float = 0.0
    noise_score: float = 0.0
    length_score: float = 0.0

    reasons: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["band"] = self.band.value
        return data


# ============================================================
# QUERY STATISTICS
# ============================================================


@dataclass
class QueryStatistics:
    """
    Analysis signals generated from a structured query.
    """

    term_count: int = 0
    phrase_count: int = 0
    filter_count: int = 0

    operator_count: int = 0

    wildcard_count: int = 0
    fuzzy_count: int = 0
    boosted_count: int = 0
    field_count: int = 0

    average_term_length: float = 0.0

    complexity: float = 0.0

    ambiguity_score: float = 0.0

    has_question_shape: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# ANALYSIS RESULT
# ============================================================


@dataclass
class AnalysisResult:
    """
    Complete analysis result.

    This object is deliberately rich so downstream components
    can consume only the signals they need.
    """

    analysis_type: AnalysisType

    created_at: str = field(
        default_factory=lambda:
        datetime.now(timezone.utc).isoformat()
    )

    length: Optional[LengthStatistics] = None
    vocabulary: Optional[VocabularyStatistics] = None

    terms: Dict[str, TermStatistics] = field(
        default_factory=dict
    )

    fields: Dict[str, FieldStatistics] = field(
        default_factory=dict
    )

    structure: Optional[StructureStatistics] = None
    language: Optional[LanguageStatistics] = None

    readability: Optional[
        ReadabilityStatistics
    ] = None

    duplicate: Optional[
        DuplicateStatistics
    ] = None

    freshness: Optional[
        FreshnessStatistics
    ] = None

    quality: Optional[
        QualityStatistics
    ] = None

    query: Optional[
        QueryStatistics
    ] = None

    signals: Dict[str, float] = field(
        default_factory=dict
    )

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    valid: bool = True

    def signal(
        self,
        name: str,
        default: float = 0.0,
    ) -> float:
        return self.signals.get(
            name,
            default,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "analysis_type":
                self.analysis_type.value,
            "created_at":
                self.created_at,
            "length":
                self.length.to_dict()
                if self.length else None,
            "vocabulary":
                self.vocabulary.to_dict()
                if self.vocabulary else None,
            "terms": {
                key: value.to_dict()
                for key, value in self.terms.items()
            },
            "fields": {
                key: value.to_dict()
                for key, value in self.fields.items()
            },
            "structure":
                self.structure.to_dict()
                if self.structure else None,
            "language":
                self.language.to_dict()
                if self.language else None,
            "readability":
                self.readability.to_dict()
                if self.readability else None,
            "duplicate":
                self.duplicate.to_dict()
                if self.duplicate else None,
            "freshness":
                self.freshness.to_dict()
                if self.freshness else None,
            "quality":
                self.quality.to_dict()
                if self.quality else None,
            "query":
                self.query.to_dict()
                if self.query else None,
            "signals":
                dict(self.signals),
            "warnings":
                list(self.warnings),
            "errors":
                list(self.errors),
            "valid":
                self.valid,
        }

        return data


# ============================================================
# TEXT NORMALIZATION
# ============================================================


class TextNormalizer:
    """
    Shared text normalization utility.
    """

    def __init__(
        self,
        config: Optional[AnalysisConfig] = None,
    ):
        self.config = (
            config
            or DEFAULT_ANALYSIS_CONFIG
        )

    def normalize(
        self,
        text: Any,
    ) -> str:

        if text is None:
            return ""

        value = str(text)

        if self.config.normalize_unicode:
            value = unicodedata.normalize(
                "NFKC",
                value,
            )

        value = WHITESPACE_PATTERN.sub(
            " ",
            value,
        )

        value = value.strip()

        if self.config.lowercase:
            value = value.lower()

        return value

    def normalize_term(
        self,
        term: str,
    ) -> str:

        value = self.normalize(term)

        if self.config.remove_punctuation:
            value = value.strip(
                string.punctuation
            )

        return value


# ============================================================
# TERM EXTRACTION
# ============================================================


class TermExtractor:
    """
    Lightweight, deterministic term extraction.

    The tokenizer remains responsible for search-token syntax.
    This extractor is primarily concerned with analysis.
    """

    def __init__(
        self,
        config: Optional[AnalysisConfig] = None,
    ):
        self.config = (
            config
            or DEFAULT_ANALYSIS_CONFIG
        )

        self.normalizer = TextNormalizer(
            self.config
        )

    def extract(
        self,
        text: str,
        limit: Optional[int] = None,
    ) -> List[str]:

        normalized = self.normalizer.normalize(
            text
        )

        terms = []

        maximum = (
            limit
            or self.config.max_document_terms
        )

        for match in WORD_PATTERN.finditer(
            normalized
        ):

            term = match.group(0)

            term = self.normalizer.normalize_term(
                term
            )

            if not term:
                continue

            if (
                len(term)
                < self.config.min_term_length
            ):
                continue

            if (
                len(term)
                > self.config.max_term_length
            ):
                continue

            if (
                self.config.ignore_numbers
                and term.isdigit()
            ):
                continue

            terms.append(term)

            if len(terms) >= maximum:
                break

        return terms


# ============================================================
# TERM ANALYSER
# ============================================================


class TermAnalyzer:
    """
    Generates per-term statistics.
    """

    def analyze(
        self,
        terms: Sequence[str],
        include_positions: bool = True,
    ) -> Dict[str, TermStatistics]:

        frequencies = Counter(terms)

        positions = defaultdict(list)

        if include_positions:
            for index, term in enumerate(
                terms
            ):
                positions[term].append(
                    index
                )

        total = max(
            len(terms),
            1,
        )

        result = {}

        for term, frequency in frequencies.items():

            term_positions = positions.get(
                term,
                [],
            )

            result[term] = TermStatistics(
                term=term,
                frequency=frequency,
                positions=term_positions,
                normalized_frequency=(
                    frequency / total
                ),
                first_position=(
                    term_positions[0]
                    if term_positions
                    else None
                ),
                last_position=(
                    term_positions[-1]
                    if term_positions
                    else None
                ),
                is_numeric=term.isnumeric(),
                is_url=bool(
                    URL_PATTERN.fullmatch(term)
                ),
                is_email=bool(
                    EMAIL_PATTERN.fullmatch(term)
                ),
                length=len(term),
            )

        return result


# ============================================================
# VOCABULARY ANALYSER
# ============================================================


class VocabularyAnalyzer:
    """
    Computes lexical diversity and vocabulary statistics.
    """

    def analyze(
        self,
        terms: Sequence[str],
        top_n: int = 25,
    ) -> VocabularyStatistics:

        total = len(terms)

        if total == 0:
            return VocabularyStatistics()

        frequencies = Counter(terms)

        vocabulary_size = len(
            frequencies
        )

        hapax_count = sum(
            1
            for count in frequencies.values()
            if count == 1
        )

        repeated_count = sum(
            1
            for count in frequencies.values()
            if count > 1
        )

        type_token_ratio = (
            vocabulary_size / total
        )

        hapax_ratio = (
            hapax_count / vocabulary_size
            if vocabulary_size
            else 0.0
        )

        lexical_density = (
            vocabulary_size / total
        )

        return VocabularyStatistics(
            vocabulary_size=vocabulary_size,
            total_terms=total,
            type_token_ratio=round(
                type_token_ratio,
                6,
            ),
            hapax_count=hapax_count,
            hapax_ratio=round(
                hapax_ratio,
                6,
            ),
            repeated_term_count=repeated_count,
            lexical_density=round(
                lexical_density,
                6,
            ),
            top_terms=frequencies.most_common(
                top_n
            ),
        )


# ============================================================
# LENGTH ANALYSER
# ============================================================


class LengthAnalyzer:
    """
    Computes document length statistics.
    """

    def analyze(
        self,
        text: str,
        terms: Sequence[str],
    ) -> LengthStatistics:

        characters = len(text)

        characters_no_whitespace = len(
            re.sub(
                r"\s+",
                "",
                text,
            )
        )

        words = len(terms)

        unique_words = len(
            set(terms)
        )

        sentences = self._count_sentences(
            text
        )

        paragraphs = self._count_paragraphs(
            text
        )

        lengths = [
            len(term)
            for term in terms
        ]

        average_word_length = (
            statistics.mean(lengths)
            if lengths
            else 0.0
        )

        average_sentence_length = (
            words / sentences
            if sentences
            else 0.0
        )

        return LengthStatistics(
            characters=characters,
            characters_no_whitespace=(
                characters_no_whitespace
            ),
            words=words,
            unique_words=unique_words,
            sentences=sentences,
            paragraphs=paragraphs,
            average_word_length=round(
                average_word_length,
                4,
            ),
            average_sentence_length=round(
                average_sentence_length,
                4,
            ),
            shortest_word_length=(
                min(lengths)
                if lengths
                else 0
            ),
            longest_word_length=(
                max(lengths)
                if lengths
                else 0
            ),
        )

    @staticmethod
    def _count_sentences(
        text: str,
    ) -> int:

        if not text.strip():
            return 0

        sentences = [
            item
            for item in SENTENCE_PATTERN.split(
                text
            )
            if item.strip()
        ]

        return max(
            len(sentences),
            1,
        )

    @staticmethod
    def _count_paragraphs(
        text: str,
    ) -> int:

        paragraphs = [
            item
            for item in re.split(
                r"\n\s*\n",
                text,
            )
            if item.strip()
        ]

        return len(paragraphs)


# ============================================================
# STRUCTURE ANALYSER
# ============================================================


class StructureAnalyzer:
    """
    Detects coarse document structure.
    """

    def analyze(
        self,
        text: str,
    ) -> StructureStatistics:

        characters = len(text)

        if characters == 0:
            return StructureStatistics()

        paragraphs = len([
            item
            for item in re.split(
                r"\n\s*\n",
                text,
            )
            if item.strip()
        ])

        sentences = len([
            item
            for item in SENTENCE_PATTERN.split(
                text
            )
            if item.strip()
        ])

        headings = len(
            re.findall(
                r"(?m)^\s{0,3}#{1,6}\s+\S+",
                text,
            )
        )

        list_items = len(
            re.findall(
                r"(?m)^\s*(?:[-*+]|\d+[.)])\s+\S+",
                text,
            )
        )

        code_blocks = len(
            re.findall(
                r"```[\s\S]*?```",
                text,
            )
        )

        links = len(
            re.findall(
                r"https?://\S+",
                text,
                re.IGNORECASE,
            )
        )

        emails = len(
            EMAIL_PATTERN.findall(text)
        )

        numbers = len(
            NUMBER_PATTERN.findall(text)
        )

        punctuation = sum(
            1
            for character in text
            if character in string.punctuation
        )

        alphabetic = [
            character
            for character in text
            if character.isalpha()
        ]

        uppercase = sum(
            1
            for character in alphabetic
            if character.isupper()
        )

        uppercase_ratio = (
            uppercase / len(alphabetic)
            if alphabetic
            else 0.0
        )

        whitespace_ratio = (
            sum(
                1
                for character in text
                if character.isspace()
            )
            / characters
        )

        return StructureStatistics(
            paragraph_count=max(
                paragraphs,
                1,
            ),
            sentence_count=max(
                sentences,
                1,
            ),
            heading_count=headings,
            list_item_count=list_items,
            code_block_count=code_blocks,
            link_count=links,
            email_count=emails,
            number_count=numbers,
            punctuation_count=punctuation,
            uppercase_ratio=round(
                uppercase_ratio,
                6,
            ),
            whitespace_ratio=round(
                whitespace_ratio,
                6,
            ),
        )


# ============================================================
# LANGUAGE ANALYSER
# ============================================================


class LanguageAnalyzer:
    """
    Script-based language hint detector.
    """

    SCRIPT_RANGES = {
        LanguageHint.LATIN: (
            (0x0041, 0x024F),
        ),
        LanguageHint.CYRILLIC: (
            (0x0400, 0x052F),
        ),
        LanguageHint.GREEK: (
            (0x0370, 0x03FF),
        ),
        LanguageHint.ARABIC: (
            (0x0600, 0x06FF),
        ),
        LanguageHint.HEBREW: (
            (0x0590, 0x05FF),
        ),
        LanguageHint.DEVANAGARI: (
            (0x0900, 0x097F),
        ),
        LanguageHint.CJK: (
            (0x4E00, 0x9FFF),
        ),
    }

    def analyze(
        self,
        text: str,
    ) -> LanguageStatistics:

        counts = Counter()

        relevant = 0

        for character in text:

            if not character.isalpha():
                continue

            codepoint = ord(character)

            matched = False

            for script, ranges in (
                self.SCRIPT_RANGES.items()
            ):
                if any(
                    start
                    <= codepoint
                    <= end
                    for start, end in ranges
                ):
                    counts[script] += 1
                    matched = True
                    relevant += 1
                    break

            if not matched:
                continue

        if relevant == 0:
            return LanguageStatistics()

        ratios = {
            script: (
                count / relevant
            )
            for script, count
            in counts.items()
        }

        active = [
            script
            for script, ratio
            in ratios.items()
            if ratio >= 0.05
        ]

        if not active:
            hint = LanguageHint.UNKNOWN
        elif len(active) == 1:
            hint = active[0]
        else:
            hint = LanguageHint.MIXED

        return LanguageStatistics(
            hint=hint,
            latin_ratio=round(
                ratios.get(
                    LanguageHint.LATIN,
                    0.0,
                ),
                6,
            ),
            cyrillic_ratio=round(
                ratios.get(
                    LanguageHint.CYRILLIC,
                    0.0,
                ),
                6,
            ),
            greek_ratio=round(
                ratios.get(
                    LanguageHint.GREEK,
                    0.0,
                ),
                6,
            ),
            arabic_ratio=round(
                ratios.get(
                    LanguageHint.ARABIC,
                    0.0,
                ),
                6,
            ),
            hebrew_ratio=round(
                ratios.get(
                    LanguageHint.HEBREW,
                    0.0,
                ),
                6,
            ),
            devanagari_ratio=round(
                ratios.get(
                    LanguageHint.DEVANAGARI,
                    0.0,
                ),
                6,
            ),
            cjk_ratio=round(
                ratios.get(
                    LanguageHint.CJK,
                    0.0,
                ),
                6,
            ),
            mixed_script=(
                len(active) > 1
            ),
        )


# ============================================================
# READABILITY ANALYSER
# ============================================================


class ReadabilityAnalyzer:
    """
    Generates approximate readability signals.
    """

    VOWELS = set(
        "aeiouy"
    )

    def analyze(
        self,
        text: str,
        terms: Sequence[str],
        sentences: int,
    ) -> ReadabilityStatistics:

        words = len(terms)

        if words == 0:
            return ReadabilityStatistics()

        average_word_length = (
            sum(
                len(term)
                for term in terms
            )
            / words
        )

        average_sentence_length = (
            words / max(
                sentences,
                1,
            )
        )

        syllables = sum(
            self._estimate_syllables(term)
            for term in terms
        )

        syllables_per_word = (
            syllables / words
        )

        # A deliberately approximate
        # Flesch-like measure.
        score = (
            206.835
            - (
                1.015
                * average_sentence_length
            )
            - (
                84.6
                * syllables_per_word
            )
        )

        score = max(
            0.0,
            min(
                100.0,
                score,
            ),
        )

        complexity = (
            1.0
            - (
                score / 100.0
            )
        )

        return ReadabilityStatistics(
            average_sentence_length=round(
                average_sentence_length,
                4,
            ),
            average_word_length=round(
                average_word_length,
                4,
            ),
            syllable_estimate=syllables,
            flesch_like_score=round(
                score,
                4,
            ),
            complexity=round(
                complexity,
                6,
            ),
        )

    def _estimate_syllables(
        self,
        word: str,
    ) -> int:

        word = word.lower()

        if not word:
            return 0

        count = 0
        previous_vowel = False

        for character in word:

            is_vowel = (
                character in self.VOWELS
            )

            if (
                is_vowel
                and not previous_vowel
            ):
                count += 1

            previous_vowel = is_vowel

        if (
            word.endswith("e")
            and count > 1
        ):
            count -= 1

        return max(
            count,
            1,
        )


# ============================================================
# DUPLICATE ANALYSER
# ============================================================


class DuplicateAnalyzer:
    """
    Produces deterministic fingerprints useful for duplicate
    and near-duplicate detection.
    """

    def analyze(
        self,
        text: str,
        terms: Sequence[str],
    ) -> DuplicateStatistics:

        content_hash = hashlib.sha256(
            text.encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest()

        normalized = WHITESPACE_PATTERN.sub(
            " ",
            text.strip().lower(),
        )

        normalized_hash = hashlib.sha256(
            normalized.encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest()

        shingles = self._shingles(
            terms,
            size=5,
        )

        fingerprint = self._fingerprint(
            shingles
        )

        repeated_term_ratio = (
            self._repeated_term_ratio(
                terms
            )
        )

        repeated_sentence_ratio = (
            self._repeated_sentence_ratio(
                text
            )
        )

        return DuplicateStatistics(
            content_hash=content_hash,
            normalized_hash=normalized_hash,
            shingle_count=len(shingles),
            fingerprint=fingerprint,
            repeated_sentence_ratio=round(
                repeated_sentence_ratio,
                6,
            ),
            repeated_term_ratio=round(
                repeated_term_ratio,
                6,
            ),
        )

    @staticmethod
    def _shingles(
        terms: Sequence[str],
        size: int,
    ) -> Set[Tuple[str, ...]]:

        if len(terms) < size:
            return set()

        return {
            tuple(
                terms[index:index + size]
            )
            for index in range(
                len(terms) - size + 1
            )
        }

    @staticmethod
    def _fingerprint(
        shingles: Set[Tuple[str, ...]],
    ) -> str:

        if not shingles:
            return ""

        digest = hashlib.sha256()

        for shingle in sorted(
            shingles
        ):
            digest.update(
                " ".join(shingle).encode(
                    "utf-8"
                )
            )

        return digest.hexdigest()

    @staticmethod
    def _repeated_term_ratio(
        terms: Sequence[str],
    ) -> float:

        if not terms:
            return 0.0

        frequencies = Counter(terms)

        repeated = sum(
            count
            for count in frequencies.values()
            if count > 1
        )

        return repeated / len(terms)

    @staticmethod
    def _repeated_sentence_ratio(
        text: str,
    ) -> float:

        sentences = [
            sentence.strip().lower()
            for sentence
            in SENTENCE_PATTERN.split(text)
            if sentence.strip()
        ]

        if not sentences:
            return 0.0

        frequencies = Counter(
            sentences
        )

        repeated = sum(
            count
            for count in frequencies.values()
            if count > 1
        )

        return repeated / len(sentences)


# ============================================================
# FRESHNESS ANALYSER
# ============================================================


class FreshnessAnalyzer:
    """
    Converts timestamps into normalized freshness signals.
    """

    def analyze(
        self,
        timestamp: Any,
        half_life_seconds: float = (
            30 * 24 * 60 * 60
        ),
    ) -> FreshnessStatistics:

        if timestamp is None:
            return FreshnessStatistics()

        parsed = self._parse_timestamp(
            timestamp
        )

        if parsed is None:
            return FreshnessStatistics(
                timestamp=str(timestamp)
            )

        now = datetime.now(
            timezone.utc
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        age = max(
            0.0,
            (
                now - parsed
            ).total_seconds(),
        )

        if half_life_seconds <= 0:
            score = 0.0
        else:
            score = math.exp(
                -(
                    age
                    / half_life_seconds
                )
                * math.log(2)
            )

        return FreshnessStatistics(
            timestamp=parsed.isoformat(),
            age_seconds=age,
            freshness_score=round(
                score,
                8,
            ),
            valid_timestamp=True,
        )

    @staticmethod
    def _parse_timestamp(
        timestamp: Any,
    ) -> Optional[datetime]:

        if isinstance(
            timestamp,
            datetime,
        ):
            return timestamp

        value = str(
            timestamp
        ).strip()

        if not value:
            return None

        try:
            return datetime.fromisoformat(
                value.replace(
                    "Z",
                    "+00:00",
                )
            )
        except ValueError:
            return None


# ============================================================
# QUALITY ANALYSER
# ============================================================


class QualityAnalyzer:
    """
    Aggregates multiple signals into a document-quality estimate.

    The score is a heuristic and should be treated as a ranking
    signal rather than an absolute truth about content quality.
    """

    def __init__(
        self,
        config: Optional[AnalysisConfig] = None,
    ):
        self.config = (
            config
            or DEFAULT_ANALYSIS_CONFIG
        )

    def analyze(
        self,
        length: LengthStatistics,
        vocabulary: VocabularyStatistics,
        structure: StructureStatistics,
        readability: ReadabilityStatistics,
        duplicate: DuplicateStatistics,
    ) -> QualityStatistics:

        length_score = self._length_score(
            length
        )

        vocabulary_score = self._vocabulary_score(
            vocabulary
        )

        structure_score = self._structure_score(
            structure
        )

        readability_score = self._readability_score(
            readability
        )

        diversity_score = min(
            1.0,
            vocabulary.type_token_ratio
            * 1.5,
        )

        noise_score = self._noise_score(
            structure,
            duplicate,
        )

        score = (
            length_score
            * self.config.quality_weight_length
            + vocabulary_score
            * self.config.quality_weight_vocabulary
            + structure_score
            * self.config.quality_weight_structure
            + readability_score
            * self.config.quality_weight_readability
            + noise_score
            * self.config.quality_weight_noise
            + diversity_score
            * self.config.quality_weight_diversity
        )

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        reasons = self._build_reasons(
            length,
            vocabulary,
            structure,
            duplicate,
        )

        return QualityStatistics(
            score=round(
                score,
                6,
            ),
            band=self._band(score),
            vocabulary_score=round(
                vocabulary_score,
                6,
            ),
            structure_score=round(
                structure_score,
                6,
            ),
            readability_score=round(
                readability_score,
                6,
            ),
            diversity_score=round(
                diversity_score,
                6,
            ),
            noise_score=round(
                noise_score,
                6,
            ),
            length_score=round(
                length_score,
                6,
            ),
            reasons=reasons,
        )

    @staticmethod
    def _length_score(
        stats: LengthStatistics,
    ) -> float:

        if stats.words == 0:
            return 0.0

        if stats.words < 5:
            return 0.2

        if stats.words < 20:
            return 0.5

        if stats.words < 50:
            return 0.8

        return 1.0

    @staticmethod
    def _vocabulary_score(
        stats: VocabularyStatistics,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                stats.type_token_ratio
                * 1.5,
            ),
        )

    @staticmethod
    def _structure_score(
        stats: StructureStatistics,
    ) -> float:

        score = 0.0

        if stats.paragraph_count:
            score += 0.25

        if stats.sentence_count:
            score += 0.25

        if stats.heading_count:
            score += 0.2

        if stats.list_item_count:
            score += 0.1

        if (
            stats.whitespace_ratio
            < 0.8
        ):
            score += 0.2

        return min(
            score,
            1.0,
        )

    @staticmethod
    def _readability_score(
        stats: ReadabilityStatistics,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                stats.flesch_like_score
                / 100.0,
            ),
        )

    @staticmethod
    def _noise_score(
        structure: StructureStatistics,
        duplicate: DuplicateStatistics,
    ) -> float:

        score = 1.0

        score -= min(
            structure.uppercase_ratio,
            0.5,
        )

        score -= min(
            duplicate.repeated_term_ratio
            * 0.5,
            0.5,
        )

        score -= min(
            duplicate.repeated_sentence_ratio
            * 0.5,
            0.5,
        )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    @staticmethod
    def _band(
        score: float,
    ) -> QualityBand:

        if score < 0.2:
            return QualityBand.VERY_LOW

        if score < 0.4:
            return QualityBand.LOW

        if score < 0.6:
            return QualityBand.MEDIUM

        if score < 0.8:
            return QualityBand.HIGH

        return QualityBand.VERY_HIGH

    @staticmethod
    def _build_reasons(
        length: LengthStatistics,
        vocabulary: VocabularyStatistics,
        structure: StructureStatistics,
        duplicate: DuplicateStatistics,
    ) -> List[str]:

        reasons = []

        if length.words < 5:
            reasons.append(
                "Very short document."
            )

        if (
            vocabulary.type_token_ratio
            > 0.7
        ):
            reasons.append(
                "High lexical diversity."
            )

        if structure.heading_count:
            reasons.append(
                "Contains structural headings."
            )

        if (
            duplicate.repeated_term_ratio
            > 0.5
        ):
            reasons.append(
                "High term repetition."
            )

        if (
            duplicate.repeated_sentence_ratio
            > 0.2
        ):
            reasons.append(
                "Contains repeated sentences."
            )

        return reasons


# ============================================================
# QUERY ANALYSER
# ============================================================


class QueryAnalyzer:
    """
    Analyses the structured Query object produced by query.py.

    The import is intentionally local so this module can remain
    usable even when imported independently during development.
    """

    def analyze(
        self,
        query: Any,
    ) -> QueryStatistics:

        clauses = getattr(
            query,
            "clauses",
            [],
        )

        filters = getattr(
            query,
            "filters",
            [],
        )

        operators = getattr(
            query,
            "operators",
            [],
        )

        term_lengths = [
            len(
                getattr(
                    clause,
                    "value",
                    "",
                )
            )
            for clause in clauses
            if getattr(
                clause,
                "value",
                "",
            )
        ]

        wildcard_count = sum(
            1
            for clause in clauses
            if getattr(
                clause,
                "wildcard",
                False,
            )
        )

        fuzzy_count = sum(
            1
            for clause in clauses
            if getattr(
                clause,
                "fuzzy",
                False,
            )
        )

        boosted_count = sum(
            1
            for clause in clauses
            if getattr(
                clause,
                "boost",
                1.0,
            ) != 1.0
        )

        field_count = len({
            getattr(
                clause,
                "field",
                None,
            )
            for clause in clauses
            if getattr(
                clause,
                "field",
                None
            )
        })

        phrase_count = sum(
            1
            for clause in clauses
            if getattr(
                clause,
                "exact",
                False,
            )
        )

        average_length = (
            statistics.mean(
                term_lengths
            )
            if term_lengths
            else 0.0
        )

        complexity = (
            len(clauses)
            + len(filters) * 2
            + len(operators) * 1.5
            + wildcard_count * 1.5
            + fuzzy_count * 1.5
            + boosted_count
            + field_count
        )

        ambiguity = self._ambiguity_score(
            clauses
        )

        original = getattr(
            query,
            "original",
            "",
        )

        return QueryStatistics(
            term_count=len(clauses),
            phrase_count=phrase_count,
            filter_count=len(filters),
            operator_count=len(operators),
            wildcard_count=wildcard_count,
            fuzzy_count=fuzzy_count,
            boosted_count=boosted_count,
            field_count=field_count,
            average_term_length=round(
                average_length,
                4,
            ),
            complexity=round(
                complexity,
                4,
            ),
            ambiguity_score=round(
                ambiguity,
                6,
            ),
            has_question_shape=(
                str(original).strip().endswith("?")
            ),
        )

    @staticmethod
    def _ambiguity_score(
        clauses: Sequence[Any],
    ) -> float:

        if not clauses:
            return 1.0

        short_terms = sum(
            1
            for clause in clauses
            if len(
                getattr(
                    clause,
                    "value",
                    "",
                )
            ) <= 3
        )

        wildcard_terms = sum(
            1
            for clause in clauses
            if getattr(
                clause,
                "wildcard",
                False,
            )
        )

        score = (
            short_terms / len(clauses)
        )

        score += (
            wildcard_terms
            / len(clauses)
            * 0.5
        )

        return min(
            score,
            1.0,
        )


# ============================================================
# FIELD ANALYSER
# ============================================================


class FieldAnalyzer:
    """
    Analyse structured document fields.
    """

    def __init__(
        self,
        extractor: TermExtractor,
    ):
        self.extractor = extractor

    def analyze(
        self,
        fields: Mapping[str, Any],
        top_n: int = 15,
    ) -> Dict[str, FieldStatistics]:

        results = {}

        for field_name, value in fields.items():

            text = self._coerce_text(
                value
            )

            terms = self.extractor.extract(
                text
            )

            frequencies = Counter(
                terms
            )

            lengths = [
                len(term)
                for term in terms
            ]

            results[str(field_name)] = (
                FieldStatistics(
                    field_name=str(field_name),
                    characters=len(text),
                    terms=len(terms),
                    unique_terms=len(
                        frequencies
                    ),
                    average_term_length=(
                        statistics.mean(lengths)
                        if lengths
                        else 0.0
                    ),
                    top_terms=(
                        frequencies.most_common(
                            top_n
                        )
                    ),
                    empty=not bool(
                        text.strip()
                    ),
                )
            )

        return results

    @staticmethod
    def _coerce_text(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return " ".join(
                str(item)
                for item in value
            )

        if isinstance(
            value,
            Mapping,
        ):
            return " ".join(
                str(item)
                for item in value.values()
            )

        return str(value)


# ============================================================
# CACHE
# ============================================================


class AnalysisCache:
    """
    Small bounded in-memory analysis cache.

    This intentionally remains simple. A future deployment can
    replace it with Redis, SQLite, disk caching, or another
    storage implementation without changing the analyser API.
    """

    def __init__(
        self,
        maximum_size: int = DEFAULT_CACHE_SIZE,
    ):
        self.maximum_size = max(
            1,
            maximum_size,
        )

        self._data: Dict[
            str,
            AnalysisResult,
        ] = {}

        self._order: List[str] = []

    def get(
        self,
        key: str,
    ) -> Optional[AnalysisResult]:

        result = self._data.get(
            key
        )

        if result is None:
            return None

        if key in self._order:
            self._order.remove(
                key
            )

        self._order.append(
            key
        )

        return result

    def put(
        self,
        key: str,
        result: AnalysisResult,
    ) -> None:

        if key in self._order:
            self._order.remove(
                key
            )

        self._data[key] = result
        self._order.append(
            key
        )

        while len(
            self._order
        ) > self.maximum_size:

            oldest = self._order.pop(
                0
            )

            self._data.pop(
                oldest,
                None,
            )

    def clear(self) -> None:
        self._data.clear()
        self._order.clear()

    def size(self) -> int:
        return len(
            self._data
        )


# ============================================================
# ANALYSER PLUGIN API
# ============================================================


class AnalyzerPlugin:
    """
    Base interface for custom analysis plugins.
    """

    name = "base"

    def analyze(
        self,
        text: str,
        result: AnalysisResult,
    ) -> None:
        raise NotImplementedError


class NoiseAnalyzer(AnalyzerPlugin):
    """
    Detects common text-noise patterns.
    """

    name = "noise"

    def analyze(
        self,
        text: str,
        result: AnalysisResult,
    ) -> None:

        if not text:
            result.signals[
                "noise"
            ] = 1.0
            return

        score = 0.0

        if REPEATED_CHARACTER_PATTERN.search(
            text
        ):
            score += 0.25

        if (
            len(URL_PATTERN.findall(text))
            > 10
        ):
            score += 0.2

        if (
            len(
                EMAIL_PATTERN.findall(text)
            )
            > 10
        ):
            score += 0.2

        if (
            result.structure
            and result.structure.uppercase_ratio
            > 0.5
        ):
            score += 0.2

        result.signals[
            "noise"
        ] = min(
            score,
            1.0,
        )


class RepetitionAnalyzer(AnalyzerPlugin):
    """
    Generates repetition-related signals.
    """

    name = "repetition"

    def analyze(
        self,
        text: str,
        result: AnalysisResult,
    ) -> None:

        if not result.duplicate:
            return

        result.signals[
            "term_repetition"
        ] = result.duplicate.repeated_term_ratio

        result.signals[
            "sentence_repetition"
        ] = (
            result.duplicate
            .repeated_sentence_ratio
        )


# ============================================================
# MAIN ANALYSIS ENGINE
# ============================================================


class AnalysisEngine:
    """
    Main analysis orchestration engine.

    This class coordinates the individual analysers while
    keeping them independently replaceable.
    """

    def __init__(
        self,
        config: Optional[AnalysisConfig] = None,
    ):
        self.config = (
            config
            or DEFAULT_ANALYSIS_CONFIG
        )

        self.normalizer = TextNormalizer(
            self.config
        )

        self.extractor = TermExtractor(
            self.config
        )

        self.term_analyzer = TermAnalyzer()
        self.vocabulary_analyzer = (
            VocabularyAnalyzer()
        )
        self.length_analyzer = (
            LengthAnalyzer()
        )
        self.structure_analyzer = (
            StructureAnalyzer()
        )
        self.language_analyzer = (
            LanguageAnalyzer()
        )
        self.readability_analyzer = (
            ReadabilityAnalyzer()
        )
        self.duplicate_analyzer = (
            DuplicateAnalyzer()
        )
        self.freshness_analyzer = (
            FreshnessAnalyzer()
        )
        self.quality_analyzer = (
            QualityAnalyzer(
                self.config
            )
        )
        self.query_analyzer = (
            QueryAnalyzer()
        )
        self.field_analyzer = (
            FieldAnalyzer(
                self.extractor
            )
        )

        self.cache = AnalysisCache(
            self.config.cache_size
        )

        self.plugins: Dict[
            str,
            AnalyzerPlugin,
        ] = {}

        self.register_plugin(
            NoiseAnalyzer()
        )

        self.register_plugin(
            RepetitionAnalyzer()
        )

    # --------------------------------------------------------
    # PLUGINS
    # --------------------------------------------------------

    def register_plugin(
        self,
        plugin: AnalyzerPlugin,
    ) -> None:

        if not plugin.name:
            raise ValueError(
                "Analyzer plugin must have a name."
            )

        self.plugins[
            plugin.name
        ] = plugin

    def unregister_plugin(
        self,
        name: str,
    ) -> bool:

        if name not in self.plugins:
            return False

        del self.plugins[name]

        return True

    def list_plugins(self) -> List[str]:
        return sorted(
            self.plugins.keys()
        )

    # --------------------------------------------------------
    # DOCUMENT ANALYSIS
    # --------------------------------------------------------

    def analyze_document(
        self,
        text: Any,
        fields: Optional[
            Mapping[str, Any]
        ] = None,
        timestamp: Any = None,
        use_cache: bool = True,
    ) -> AnalysisResult:

        raw_text = (
            ""
            if text is None
            else str(text)
        )

        cache_key = self._cache_key(
            "document",
            raw_text,
            fields,
            timestamp,
        )

        if (
            use_cache
            and self.config.cache_enabled
        ):
            cached = self.cache.get(
                cache_key
            )

            if cached:
                return cached

        result = AnalysisResult(
            analysis_type=AnalysisType.DOCUMENT
        )

        try:
            terms = self.extractor.extract(
                raw_text
            )

            result.length = (
                self.length_analyzer.analyze(
                    raw_text,
                    terms,
                )
            )

            result.vocabulary = (
                self.vocabulary_analyzer.analyze(
                    terms
                )
            )

            result.terms = (
                self.term_analyzer.analyze(
                    terms,
                    self.config.include_term_positions,
                )
            )

            result.structure = (
                self.structure_analyzer.analyze(
                    raw_text
                )
            )

            if self.config.detect_language:
                result.language = (
                    self.language_analyzer.analyze(
                        raw_text
                    )
                )

            if self.config.calculate_readability:
                result.readability = (
                    self.readability_analyzer.analyze(
                        raw_text,
                        terms,
                        result.length.sentences,
                    )
                )

            if self.config.detect_duplicates:
                result.duplicate = (
                    self.duplicate_analyzer.analyze(
                        raw_text,
                        terms,
                    )
                )

            if self.config.detect_freshness:
                result.freshness = (
                    self.freshness_analyzer.analyze(
                        timestamp
                    )
                )

            if fields:
                result.fields = (
                    self.field_analyzer.analyze(
                        fields
                    )
                )

            if self.config.detect_quality:
                result.quality = (
                    self.quality_analyzer.analyze(
                        result.length,
                        result.vocabulary,
                        result.structure,
                        result.readability
                        or ReadabilityStatistics(),
                        result.duplicate
                        or DuplicateStatistics(),
                    )
                )

            self._build_document_signals(
                result
            )

            for plugin in self.plugins.values():
                try:
                    plugin.analyze(
                        raw_text,
                        result,
                    )
                except Exception as error:
                    result.warnings.append(
                        f"Plugin '{plugin.name}' failed: "
                        f"{error}"
                    )

        except Exception as error:

            result.valid = False

            result.errors.append(
                f"Document analysis failed: {error}"
            )

        if (
            use_cache
            and self.config.cache_enabled
            and result.valid
        ):
            self.cache.put(
                cache_key,
                result,
            )

        return result

    # --------------------------------------------------------
    # QUERY ANALYSIS
    # --------------------------------------------------------

    def analyze_query(
        self,
        query: Any,
        use_cache: bool = True,
    ) -> AnalysisResult:

        cache_key = self._cache_key(
            "query",
            repr(query),
        )

        if (
            use_cache
            and self.config.cache_enabled
        ):
            cached = self.cache.get(
                cache_key
            )

            if cached:
                return cached

        result = AnalysisResult(
            analysis_type=AnalysisType.QUERY
        )

        try:
            result.query = (
                self.query_analyzer.analyze(
                    query
                )
            )

            clauses = getattr(
                query,
                "clauses",
                [],
            )

            terms = [
                getattr(
                    clause,
                    "value",
                    "",
                )
                for clause in clauses
                if getattr(
                    clause,
                    "value",
                    "",
                )
            ]

            result.terms = (
                self.term_analyzer.analyze(
                    terms,
                    include_positions=False,
                )
            )

            result.signals[
                "query_complexity"
            ] = (
                result.query.complexity
            )

            result.signals[
                "query_ambiguity"
            ] = (
                result.query.ambiguity_score
            )

            result.signals[
                "query_specificity"
            ] = self._query_specificity(
                result.query
            )

        except Exception as error:

            result.valid = False

            result.errors.append(
                f"Query analysis failed: {error}"
            )

        if (
            use_cache
            and self.config.cache_enabled
            and result.valid
        ):
            self.cache.put(
                cache_key,
                result,
            )

        return result

    # --------------------------------------------------------
    # FIELD ANALYSIS
    # --------------------------------------------------------

    def analyze_fields(
        self,
        fields: Mapping[str, Any],
    ) -> AnalysisResult:

        result = AnalysisResult(
            analysis_type=AnalysisType.FIELD
        )

        try:
            result.fields = (
                self.field_analyzer.analyze(
                    fields
                )
            )

            total_terms = sum(
                stats.terms
                for stats
                in result.fields.values()
            )

            total_unique = sum(
                stats.unique_terms
                for stats
                in result.fields.values()
            )

            result.signals[
                "field_count"
            ] = float(
                len(result.fields)
            )

            result.signals[
                "field_terms"
            ] = float(
                total_terms
            )

            result.signals[
                "field_unique_terms"
            ] = float(
                total_unique
            )

        except Exception as error:

            result.valid = False

            result.errors.append(
                f"Field analysis failed: {error}"
            )

        return result

    # --------------------------------------------------------
    # BATCH ANALYSIS
    # --------------------------------------------------------

    def analyze_batch(
        self,
        documents: Iterable[Any],
        text_getter: Optional[
            Callable[[Any], str]
        ] = None,
    ) -> Iterator[AnalysisResult]:

        getter = (
            text_getter
            or self._default_text_getter
        )

        for document in documents:

            try:
                text = getter(
                    document
                )

                yield self.analyze_document(
                    text
                )

            except Exception as error:

                result = AnalysisResult(
                    analysis_type=(
                        AnalysisType.BATCH
                    ),
                    valid=False,
                )

                result.errors.append(
                    f"Batch analysis failed: {error}"
                )

                yield result

    # --------------------------------------------------------
    # SIGNAL GENERATION
    # --------------------------------------------------------

    def _build_document_signals(
        self,
        result: AnalysisResult,
    ) -> None:

        if result.length:
            result.signals[
                "document_length"
            ] = float(
                result.length.words
            )

            result.signals[
                "average_word_length"
            ] = (
                result.length.average_word_length
            )

            result.signals[
                "sentence_count"
            ] = float(
                result.length.sentences
            )

        if result.vocabulary:
            result.signals[
                "vocabulary_size"
            ] = float(
                result.vocabulary.vocabulary_size
            )

            result.signals[
                "type_token_ratio"
            ] = (
                result.vocabulary.type_token_ratio
            )

        if result.quality:
            result.signals[
                "quality"
            ] = result.quality.score

        if result.freshness:
            result.signals[
                "freshness"
            ] = (
                result.freshness.freshness_score
            )

        if result.language:
            result.signals[
                "language_confidence"
            ] = max(
                result.language.latin_ratio,
                result.language.cyrillic_ratio,
                result.language.greek_ratio,
                result.language.arabic_ratio,
                result.language.hebrew_ratio,
                result.language.devanagari_ratio,
                result.language.cjk_ratio,
            )

    @staticmethod
    def _query_specificity(
        query_stats: QueryStatistics,
    ) -> float:

        if query_stats.term_count == 0:
            return 0.0

        score = 0.0

        score += min(
            query_stats.term_count / 5.0,
            1.0,
        ) * 0.4

        score += min(
            query_stats.field_count / 3.0,
            1.0,
        ) * 0.2

        score += min(
            query_stats.phrase_count / 2.0,
            1.0,
        ) * 0.2

        score += (
            1.0
            - query_stats.ambiguity_score
        ) * 0.2

        return round(
            max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            ),
            6,
        )

    # --------------------------------------------------------
    # CACHE KEY
    # --------------------------------------------------------

    @staticmethod
    def _cache_key(
        *values: Any,
    ) -> str:

        payload = repr(
            values
        ).encode(
            "utf-8",
            errors="ignore",
        )

        return hashlib.sha256(
            payload
        ).hexdigest()

    # --------------------------------------------------------
    # DEFAULT GETTER
    # --------------------------------------------------------

    @staticmethod
    def _default_text_getter(
        document: Any,
    ) -> str:

        if document is None:
            return ""

        if isinstance(
            document,
            str,
        ):
            return document

        if isinstance(
            document,
            Mapping,
        ):
            for key in (
                "content",
                "text",
                "body",
                "document",
            ):
                if key in document:
                    return str(
                        document[key]
                    )

        return str(document)

    # --------------------------------------------------------
    # CACHE CONTROL
    # --------------------------------------------------------

    def clear_cache(self) -> None:
        self.cache.clear()

    def cache_size(self) -> int:
        return self.cache.size()

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    def explain_document(
        self,
        text: Any,
        fields: Optional[
            Mapping[str, Any]
        ] = None,
        timestamp: Any = None,
    ) -> Dict[str, Any]:

        result = self.analyze_document(
            text,
            fields=fields,
            timestamp=timestamp,
        )

        return {
            "analysis_type":
                result.analysis_type.value,
            "valid":
                result.valid,
            "signals":
                dict(result.signals),
            "length":
                result.length.to_dict()
                if result.length else None,
            "vocabulary":
                result.vocabulary.to_dict()
                if result.vocabulary else None,
            "quality":
                result.quality.to_dict()
                if result.quality else None,
            "language":
                result.language.to_dict()
                if result.language else None,
            "duplicate":
                result.duplicate.to_dict()
                if result.duplicate else None,
            "freshness":
                result.freshness.to_dict()
                if result.freshness else None,
            "warnings":
                list(result.warnings),
            "errors":
                list(result.errors),
        }

    def explain_query(
        self,
        query: Any,
    ) -> Dict[str, Any]:

        result = self.analyze_query(
            query
        )

        return {
            "analysis_type":
                result.analysis_type.value,
            "valid":
                result.valid,
            "query":
                result.query.to_dict()
                if result.query else None,
            "signals":
                dict(result.signals),
            "warnings":
                list(result.warnings),
            "errors":
                list(result.errors),
        }


# ============================================================
# DOCUMENT ANALYSIS HELPERS
# ============================================================


def analyze_document(
    text: Any,
    fields: Optional[
        Mapping[str, Any]
    ] = None,
    timestamp: Any = None,
    config: Optional[AnalysisConfig] = None,
) -> AnalysisResult:

    engine = AnalysisEngine(
        config
    )

    return engine.analyze_document(
        text,
        fields=fields,
        timestamp=timestamp,
    )


def analyze_query(
    query: Any,
    config: Optional[AnalysisConfig] = None,
) -> AnalysisResult:

    engine = AnalysisEngine(
        config
    )

    return engine.analyze_query(
        query
    )


# ============================================================
# GLOBAL ENGINE
# ============================================================


analysis = AnalysisEngine(
    DEFAULT_ANALYSIS_CONFIG
)


# ============================================================
# SELF TEST
# ============================================================


if __name__ == "__main__":

    sample = """
    OurPlatform is an advanced search and memory platform.

    It provides indexing, retrieval, ranking, filtering,
    query parsing, and analysis capabilities.

    Search quality depends on understanding both documents
    and queries.
    """

    print(
        "=" * 80
    )

    print(
        "DOCUMENT ANALYSIS"
    )

    print(
        "=" * 80
    )

    result = analysis.analyze_document(
        sample,
        fields={
            "title":
                "OurPlatform Search Engine",
            "category":
                "technology",
            "tags": [
                "search",
                "ai",
                "indexing",
            ],
        },
    )

    print(
        result.to_dict()
    )

    print(
        "\nCACHE SIZE:",
        analysis.cache_size(),
    )

    print(
        "\nPLUGINS:",
        analysis.list_plugins(),
    )

    print(
        "\nEXPLANATION:"
    )

    print(
        analysis.explain_document(
            sample
        )
    )