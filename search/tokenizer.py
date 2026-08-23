"""
OurPlatform Search Tokenizer
============================

Advanced text processing layer for the search subsystem.

Responsibilities
----------------
- Unicode normalization
- Case normalization
- Whitespace normalization
- Punctuation handling
- Token extraction
- Stop-word filtering
- Token positions
- Token spans
- Word frequencies
- N-grams
- Prefix generation
- Suffix generation
- Phrase extraction
- Query parsing
- Quoted phrase detection
- Field-aware query parsing
- Alias and synonym expansion
- Basic stemming
- Fuzzy matching
- Token statistics
- Text analysis
- Configurable processing pipelines

This module intentionally contains no search-ranking logic.
It converts text into useful representations that the
index and ranking layers can consume.
"""

from __future__ import annotations

import math
import re
import unicodedata

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class Token:
    """
    Represents one processed token.

    A token keeps both its normalized value and its
    original position inside the source text.
    """

    text: str

    original: str

    position: int

    start: int

    end: int

    sentence: int = 0

    paragraph: int = 0

    is_number: bool = False

    is_stop_word: bool = False

    field: Optional[str] = None

    metadata: Dict = field(
        default_factory=dict
    )

    @property
    def length(self) -> int:
        return len(self.text)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "original": self.original,
            "position": self.position,
            "start": self.start,
            "end": self.end,
            "sentence": self.sentence,
            "paragraph": self.paragraph,
            "is_number": self.is_number,
            "is_stop_word": self.is_stop_word,
            "field": self.field,
            "metadata": dict(self.metadata),
        }


@dataclass
class Phrase:
    """
    Represents a sequence of tokens that forms a phrase.
    """

    text: str

    tokens: List[str]

    start_position: int

    end_position: int

    quoted: bool = False

    field: Optional[str] = None

    score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "tokens": list(self.tokens),
            "start_position": self.start_position,
            "end_position": self.end_position,
            "quoted": self.quoted,
            "field": self.field,
            "score": self.score,
        }


@dataclass
class QueryTerm:
    """
    Represents a parsed search query term.
    """

    text: str

    field: Optional[str] = None

    phrase: bool = False

    required: bool = False

    excluded: bool = False

    fuzzy: bool = False

    prefix: bool = False

    boost: float = 1.0

    original: Optional[str] = None

    metadata: Dict = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "field": self.field,
            "phrase": self.phrase,
            "required": self.required,
            "excluded": self.excluded,
            "fuzzy": self.fuzzy,
            "prefix": self.prefix,
            "boost": self.boost,
            "original": self.original,
            "metadata": dict(self.metadata),
        }


@dataclass
class TokenizationResult:
    """
    Complete analysis result produced by the tokenizer.
    """

    original_text: str

    normalized_text: str

    tokens: List[Token] = field(
        default_factory=list
    )

    phrases: List[Phrase] = field(
        default_factory=list
    )

    frequencies: Dict[str, int] = field(
        default_factory=dict
    )

    unique_tokens: List[str] = field(
        default_factory=list
    )

    sentences: List[str] = field(
        default_factory=list
    )

    paragraphs: List[str] = field(
        default_factory=list
    )

    character_count: int = 0

    word_count: int = 0

    unique_count: int = 0

    processing_metadata: Dict = field(
        default_factory=dict
    )

    def token_texts(self) -> List[str]:
        return [
            token.text
            for token in self.tokens
        ]

    def to_dict(self) -> Dict:
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "tokens": [
                token.to_dict()
                for token in self.tokens
            ],
            "phrases": [
                phrase.to_dict()
                for phrase in self.phrases
            ],
            "frequencies": dict(
                self.frequencies
            ),
            "unique_tokens": list(
                self.unique_tokens
            ),
            "sentences": list(
                self.sentences
            ),
            "paragraphs": list(
                self.paragraphs
            ),
            "character_count": self.character_count,
            "word_count": self.word_count,
            "unique_count": self.unique_count,
            "processing_metadata": dict(
                self.processing_metadata
            ),
        }


# ============================================================
# TOKENIZER
# ============================================================


class Tokenizer:
    """
    Advanced configurable tokenizer.

    The tokenizer is designed to be used by:

        index.py
        ranking.py
        engine.py
        suggestions.py
        filters.py
        analytics.py
        research systems
        memory systems
    """

    DEFAULT_STOP_WORDS = {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }

    DEFAULT_SYNONYMS = {
        "ai": {
            "artificial",
            "intelligence",
            "machine",
            "learning",
        },
        "search": {
            "find",
            "lookup",
            "query",
            "retrieve",
        },
        "research": {
            "study",
            "investigate",
            "analysis",
        },
        "memory": {
            "remember",
            "recall",
            "storage",
        },
    }

    # Word-like tokens, including internal apostrophes
    # and hyphens.
    TOKEN_PATTERN = re.compile(
        r"""
        (?:
            [^\W_]+
            (?:['’-][^\W_]+)*
        )
        """,
        re.UNICODE | re.VERBOSE,
    )

    QUOTED_PATTERN = re.compile(
        r'"([^"]+)"'
    )

    FIELD_PATTERN = re.compile(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*:"
    )

    # --------------------------------------------------------
    # CONSTRUCTOR
    # --------------------------------------------------------

    def __init__(
        self,
        lowercase: bool = True,
        normalize_unicode: bool = True,
        remove_stop_words: bool = True,
        preserve_numbers: bool = True,
        preserve_decimals: bool = True,
        preserve_hyphens: bool = True,
        preserve_apostrophes: bool = True,
        minimum_token_length: int = 1,
        maximum_token_length: int = 128,
        enable_stemming: bool = False,
        enable_synonyms: bool = False,
        stop_words: Optional[
            Iterable[str]
        ] = None,
        synonyms: Optional[
            Dict[str, Iterable[str]]
        ] = None,
    ):

        self.lowercase = lowercase

        self.normalize_unicode = (
            normalize_unicode
        )

        self.remove_stop_words = (
            remove_stop_words
        )

        self.preserve_numbers = (
            preserve_numbers
        )

        self.preserve_decimals = (
            preserve_decimals
        )

        self.preserve_hyphens = (
            preserve_hyphens
        )

        self.preserve_apostrophes = (
            preserve_apostrophes
        )

        self.minimum_token_length = (
            minimum_token_length
        )

        self.maximum_token_length = (
            maximum_token_length
        )

        self.enable_stemming = (
            enable_stemming
        )

        self.enable_synonyms = (
            enable_synonyms
        )

        self.stop_words = set(
            stop_words
            if stop_words is not None
            else self.DEFAULT_STOP_WORDS
        )

        self.synonyms = defaultdict(
            set
        )

        source_synonyms = (
            synonyms
            if synonyms is not None
            else self.DEFAULT_SYNONYMS
        )

        for key, values in (
            source_synonyms.items()
        ):

            normalized_key = (
                self.normalize(key)
            )

            self.synonyms[
                normalized_key
            ].update(
                self.normalize(value)
                for value in values
            )

        self._compiled_token_pattern = (
            self.TOKEN_PATTERN
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def normalize(
        self,
        text: Optional[str],
    ) -> str:

        if text is None:
            return ""

        text = str(text)

        if self.normalize_unicode:

            text = unicodedata.normalize(
                "NFKC",
                text,
            )

        if self.lowercase:
            text = text.lower()

        # Normalize common typography.
        replacements = {
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "–": "-",
            "—": "-",
            "\u00a0": " ",
        }

        for old, new in (
            replacements.items()
        ):

            text = text.replace(
                old,
                new,
            )

        # Normalize whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ========================================================
    # ACCENT HANDLING
    # ========================================================

    def remove_accents(
        self,
        text: str,
    ) -> str:

        normalized = unicodedata.normalize(
            "NFD",
            text,
        )

        return "".join(
            character
            for character in normalized
            if unicodedata.category(
                character
            )
            != "Mn"
        )

    def normalize_search_text(
        self,
        text: str,
    ) -> str:

        text = self.normalize(
            text
        )

        return self.remove_accents(
            text
        )

    # ========================================================
    # SENTENCE / PARAGRAPH SPLITTING
    # ========================================================

    def split_sentences(
        self,
        text: str,
    ) -> List[str]:

        if not text:
            return []

        normalized = text.strip()

        if not normalized:
            return []

        parts = re.split(
            r"(?<=[.!?])\s+",
            normalized,
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    def split_paragraphs(
        self,
        text: str,
    ) -> List[str]:

        if not text:
            return []

        return [
            paragraph.strip()
            for paragraph in re.split(
                r"\n\s*\n",
                text,
            )
            if paragraph.strip()
        ]

    # ========================================================
    # RAW TOKEN EXTRACTION
    # ========================================================

    def extract_raw_tokens(
        self,
        text: str,
    ) -> List[str]:

        normalized = self.normalize(
            text
        )

        if not normalized:
            return []

        matches = (
            self._compiled_token_pattern
            .findall(normalized)
        )

        result = []

        for token in matches:

            if (
                not self.preserve_hyphens
            ):
                token = token.replace(
                    "-",
                    "",
                )

            if (
                not self.preserve_apostrophes
            ):
                token = token.replace(
                    "'",
                    "",
                )

            token = token.strip(
                "-'"
            )

            if token:
                result.append(
                    token
                )

        return result

    # ========================================================
    # TOKEN VALIDATION
    # ========================================================

    def is_number(
        self,
        token: str,
    ) -> bool:

        if not token:
            return False

        cleaned = token.replace(
            ",",
            "",
        )

        if self.preserve_decimals:
            cleaned = cleaned.replace(
                ".",
                "",
            )

        return cleaned.isdigit()

    def valid_token(
        self,
        token: str,
    ) -> bool:

        if not token:
            return False

        length = len(token)

        if (
            length
            < self.minimum_token_length
        ):
            return False

        if (
            length
            > self.maximum_token_length
        ):
            return False

        if (
            not self.preserve_numbers
            and self.is_number(token)
        ):
            return False

        return True

    # ========================================================
    # STOP WORDS
    # ========================================================

    def is_stop_word(
        self,
        token: str,
    ) -> bool:

        return (
            self.normalize(token)
            in self.stop_words
        )

    def add_stop_word(
        self,
        word: str,
    ):

        normalized = self.normalize(
            word
        )

        if normalized:
            self.stop_words.add(
                normalized
            )

    def remove_stop_word(
        self,
        word: str,
    ):

        normalized = self.normalize(
            word
        )

        self.stop_words.discard(
            normalized
        )

    def set_stop_words(
        self,
        words: Iterable[str],
    ):

        self.stop_words = {
            self.normalize(word)
            for word in words
            if self.normalize(word)
        }

    # ========================================================
    # STEMMING
    # ========================================================

    def stem(
        self,
        word: str,
    ) -> str:

        """
        Lightweight dependency-free stemmer.

        This is intentionally conservative.
        It is not intended to replace a linguistic
        stemmer or lemmatizer.
        """

        word = self.normalize(
            word
        )

        if len(word) <= 3:
            return word

        suffixes = [
            "ingly",
            "edly",
            "ation",
            "ments",
            "ment",
            "ness",
            "ingly",
            "ingly",
            "ing",
            "ers",
            "ies",
            "ied",
            "ed",
            "es",
            "s",
        ]

        for suffix in suffixes:

            if (
                word.endswith(suffix)
                and len(word)
                - len(suffix)
                >= 3
            ):

                stemmed = word[
                    : -len(suffix)
                ]

                # ies -> y
                if suffix == "ies":
                    return stemmed + "y"

                return stemmed

        return word

    # ========================================================
    # TOKEN OBJECT CREATION
    # ========================================================

    def _create_tokens(
        self,
        text: str,
    ) -> List[Token]:

        normalized = self.normalize(
            text
        )

        tokens = []

        sentence_index = 0
        paragraph_index = 0

        sentence_boundaries = []

        for match in re.finditer(
            r"[.!?]",
            normalized,
        ):

            sentence_boundaries.append(
                match.end()
            )

        paragraphs = (
            self.split_paragraphs(
                normalized
            )
        )

        paragraph_ranges = []

        cursor = 0

        for paragraph in paragraphs:

            position = normalized.find(
                paragraph,
                cursor,
            )

            if position >= 0:

                paragraph_ranges.append(
                    (
                        position,
                        position
                        + len(paragraph),
                    )
                )

                cursor = (
                    position
                    + len(paragraph)
                )

        for position, match in enumerate(
            self._compiled_token_pattern.finditer(
                normalized
            )
        ):

            original = match.group(
                0
            )

            token_text = original

            if not self.preserve_hyphens:
                token_text = (
                    token_text.replace(
                        "-",
                        "",
                    )
                )

            if not self.preserve_apostrophes:
                token_text = (
                    token_text.replace(
                        "'",
                        "",
                    )
                )

            token_text = token_text.strip(
                "-'"
            )

            if not self.valid_token(
                token_text
            ):
                continue

            while (
                sentence_index
                < len(sentence_boundaries)
                and match.start()
                >= sentence_boundaries[
                    sentence_index
                ]
            ):

                sentence_index += 1

            for index, (
                start,
                end,
            ) in enumerate(
                paragraph_ranges
            ):

                if (
                    start
                    <= match.start()
                    <= end
                ):

                    paragraph_index = index

                    break

            is_stop = (
                self.is_stop_word(
                    token_text
                )
            )

            tokens.append(
                Token(
                    text=token_text,
                    original=original,
                    position=position,
                    start=match.start(),
                    end=match.end(),
                    sentence=sentence_index,
                    paragraph=paragraph_index,
                    is_number=self.is_number(
                        token_text
                    ),
                    is_stop_word=is_stop,
                )
            )

        return tokens

    # ========================================================
    # TOKENIZATION
    # ========================================================

    def tokenize_objects(
        self,
        text: str,
        remove_stop_words: Optional[
            bool
        ] = None,
    ) -> List[Token]:

        if remove_stop_words is None:
            remove_stop_words = (
                self.remove_stop_words
            )

        tokens = self._create_tokens(
            text
        )

        result = []

        for token in tokens:

            if (
                remove_stop_words
                and token.is_stop_word
            ):
                continue

            if self.enable_stemming:

                token.metadata[
                    "stem"
                ] = self.stem(
                    token.text
                )

            result.append(
                token
            )

        # Re-number positions after filtering.
        for position, token in enumerate(
            result
        ):

            token.position = position

        return result

    def tokenize(
        self,
        text: str,
        remove_stop_words: Optional[
            bool
        ] = None,
    ) -> List[str]:

        return [
            token.text
            for token in self.tokenize_objects(
                text,
                remove_stop_words,
            )
        ]

    # ========================================================
    # FREQUENCIES
    # ========================================================

    def frequencies(
        self,
        text: str,
    ) -> Dict[str, int]:

        return dict(
            Counter(
                self.tokenize(text)
            )
        )

    def frequency_counter(
        self,
        text: str,
    ) -> Counter:

        return Counter(
            self.tokenize(text)
        )

    def unique_tokens(
        self,
        text: str,
    ) -> List[str]:

        return list(
            dict.fromkeys(
                self.tokenize(text)
            )
        )

    # ========================================================
    # TERM DENSITY
    # ========================================================

    def term_density(
        self,
        text: str,
    ) -> Dict[str, float]:

        frequencies = (
            self.frequency_counter(
                text
            )
        )

        total = sum(
            frequencies.values()
        )

        if total == 0:
            return {}

        return {
            token: count / total
            for token, count
            in frequencies.items()
        }

    # ========================================================
    # N-GRAMS
    # ========================================================

    def ngrams(
        self,
        tokens: Sequence[str],
        size: int = 2,
    ) -> List[Tuple[str, ...]]:

        if size <= 0:
            raise ValueError(
                "N-gram size must be positive."
            )

        if len(tokens) < size:
            return []

        return [
            tuple(
                tokens[index:index + size]
            )
            for index in range(
                len(tokens)
                - size
                + 1
            )
        ]

    def text_ngrams(
        self,
        text: str,
        size: int = 2,
    ) -> List[Tuple[str, ...]]:

        return self.ngrams(
            self.tokenize(
                text
            ),
            size,
        )

    def all_ngrams(
        self,
        text: str,
        minimum: int = 2,
        maximum: int = 3,
    ) -> Dict[int, List[Tuple[str, ...]]]:

        tokens = self.tokenize(
            text
        )

        result = {}

        for size in range(
            minimum,
            maximum + 1,
        ):

            result[size] = (
                self.ngrams(
                    tokens,
                    size,
                )
            )

        return result

    # ========================================================
    # PHRASES
    # ========================================================

    def extract_phrases(
        self,
        text: str,
        size: int = 2,
    ) -> List[Phrase]:

        tokens = self.tokenize_objects(
            text
        )

        phrases = []

        for index in range(
            len(tokens) - size + 1
        ):

            group = tokens[
                index:index + size
            ]

            phrase_tokens = [
                token.text
                for token in group
            ]

            phrase_text = " ".join(
                phrase_tokens
            )

            phrases.append(
                Phrase(
                    text=phrase_text,
                    tokens=phrase_tokens,
                    start_position=group[
                        0
                    ].position,
                    end_position=group[
                        -1
                    ].position,
                )
            )

        return phrases

    def extract_quoted_phrases(
        self,
        text: str,
    ) -> List[Phrase]:

        phrases = []

        for match in (
            self.QUOTED_PATTERN.finditer(
                text
            )
        ):

            phrase_text = (
                match.group(1)
            )

            tokens = self.tokenize(
                phrase_text
            )

            if not tokens:
                continue

            phrases.append(
                Phrase(
                    text=phrase_text,
                    tokens=tokens,
                    start_position=0,
                    end_position=max(
                        0,
                        len(tokens) - 1,
                    ),
                    quoted=True,
                )
            )

        return phrases

    # ========================================================
    # PREFIXES / SUFFIXES
    # ========================================================

    def prefixes(
        self,
        token: str,
        minimum_length: int = 2,
    ) -> List[str]:

        token = self.normalize(
            token
        )

        if len(token) < minimum_length:
            return []

        return [
            token[:length]
            for length in range(
                minimum_length,
                len(token) + 1,
            )
        ]

    def suffixes(
        self,
        token: str,
        minimum_length: int = 2,
    ) -> List[str]:

        token = self.normalize(
            token
        )

        if len(token) < minimum_length:
            return []

        return [
            token[-length:]
            for length in range(
                minimum_length,
                len(token) + 1,
            )
        ]

    # ========================================================
    # SYNONYMS
    # ========================================================

    def add_synonym(
        self,
        word: str,
        synonyms: Iterable[str],
    ):

        word = self.normalize(
            word
        )

        if not word:
            return

        self.synonyms[
            word
        ].update(
            self.normalize(value)
            for value in synonyms
            if self.normalize(value)
        )

    def remove_synonym(
        self,
        word: str,
        synonym: Optional[str] = None,
    ):

        word = self.normalize(
            word
        )

        if word not in self.synonyms:
            return

        if synonym is None:

            del self.synonyms[
                word
            ]

            return

        self.synonyms[
            word
        ].discard(
            self.normalize(
                synonym
            )
        )

    def get_synonyms(
        self,
        word: str,
    ) -> Set[str]:

        word = self.normalize(
            word
        )

        return set(
            self.synonyms.get(
                word,
                set(),
            )
        )

    def expand_synonyms(
        self,
        tokens: Iterable[str],
    ) -> List[str]:

        result = []

        for token in tokens:

            result.append(
                token
            )

            if self.enable_synonyms:

                result.extend(
                    self.get_synonyms(
                        token
                    )
                )

        return list(
            dict.fromkeys(result)
        )

    # ========================================================
    # FUZZY MATCHING
    # ========================================================

    @staticmethod
    def levenshtein(
        first: str,
        second: str,
    ) -> int:

        first = str(first)
        second = str(second)

        if first == second:
            return 0

        if not first:
            return len(second)

        if not second:
            return len(first)

        previous = list(
            range(
                len(second) + 1
            )
        )

        for index_a, char_a in enumerate(
            first,
            start=1,
        ):

            current = [index_a]

            for index_b, char_b in enumerate(
                second,
                start=1,
            ):

                insertion = (
                    current[index_b - 1]
                    + 1
                )

                deletion = (
                    previous[index_b]
                    + 1
                )

                substitution = (
                    previous[index_b - 1]
                    + (
                        0
                        if char_a
                        == char_b
                        else 1
                    )
                )

                current.append(
                    min(
                        insertion,
                        deletion,
                        substitution,
                    )
                )

            previous = current

        return previous[-1]

    @classmethod
    def similarity(
        cls,
        first: str,
        second: str,
    ) -> float:

        first = str(
            first
        ).lower()

        second = str(
            second
        ).lower()

        if first == second:
            return 1.0

        longest = max(
            len(first),
            len(second),
        )

        if longest == 0:
            return 1.0

        distance = cls.levenshtein(
            first,
            second,
        )

        return max(
            0.0,
            1.0
            - (
                distance
                / longest
            ),
        )

    def fuzzy_candidates(
        self,
        token: str,
        candidates: Iterable[str],
        threshold: float = 0.70,
        maximum: int = 10,
    ) -> List[Tuple[str, float]]:

        results = []

        for candidate in candidates:

            score = self.similarity(
                token,
                candidate,
            )

            if score >= threshold:

                results.append(
                    (
                        candidate,
                        score,
                    )
                )

        results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return results[:maximum]

    # ========================================================
    # QUERY PARSING
    # ========================================================

    def parse_query(
        self,
        query: str,
    ) -> List[QueryTerm]:

        query = str(
            query or ""
        ).strip()

        if not query:
            return []

        terms = []

        consumed = set()

        # -----------------------------------------------
        # Quoted phrases
        # -----------------------------------------------

        for match in (
            self.QUOTED_PATTERN.finditer(
                query
            )
        ):

            phrase_text = (
                match.group(1)
            )

            tokens = self.tokenize(
                phrase_text,
                remove_stop_words=False,
            )

            if tokens:

                terms.append(
                    QueryTerm(
                        text=" ".join(tokens),
                        phrase=True,
                        required=True,
                        original=match.group(0),
                    )
                )

            consumed.update(
                range(
                    match.start(),
                    match.end(),
                )
            )

        # -----------------------------------------------
        # Remaining query
        # -----------------------------------------------

        remaining = "".join(
            " "
            if index in consumed
            else character
            for index, character
            in enumerate(query)
        )

        raw_parts = remaining.split()

        for raw_part in raw_parts:

            if not raw_part:
                continue

            required = raw_part.startswith(
                "+"
            )

            excluded = raw_part.startswith(
                "-"
            )

            if (
                required
                or excluded
            ):

                raw_part = raw_part[
                    1:
                ]

            prefix = raw_part.endswith(
                "*"
            )

            if prefix:
                raw_part = raw_part[
                    :-1
                ]

            fuzzy = (
                raw_part.endswith(
                    "~"
                )
            )

            if fuzzy:
                raw_part = raw_part[
                    :-1
                ]

            field_name = None

            field_match = re.match(
                r"^([A-Za-z_][A-Za-z0-9_]*)\:(.+)$",
                raw_part,
            )

            if field_match:

                field_name = (
                    field_match.group(1)
                )

                raw_part = (
                    field_match.group(2)
                )

            tokens = self.tokenize(
                raw_part,
                remove_stop_words=False,
            )

            for token in tokens:

                terms.append(
                    QueryTerm(
                        text=token,
                        field=field_name,
                        required=required,
                        excluded=excluded,
                        fuzzy=fuzzy,
                        prefix=prefix,
                        original=raw_part,
                    )
                )

        return terms

    # ========================================================
    # QUERY ANALYSIS
    # ========================================================

    def analyze_query(
        self,
        query: str,
    ) -> Dict:

        terms = self.parse_query(
            query
        )

        return {
            "original": query,
            "normalized": self.normalize(
                query
            ),
            "terms": [
                term.to_dict()
                for term in terms
            ],
            "required": [
                term.text
                for term in terms
                if term.required
            ],
            "excluded": [
                term.text
                for term in terms
                if term.excluded
            ],
            "phrases": [
                term.text
                for term in terms
                if term.phrase
            ],
            "fields": sorted(
                {
                    term.field
                    for term in terms
                    if term.field
                }
            ),
        }

    # ========================================================
    # FULL TEXT ANALYSIS
    # ========================================================

    def analyze(
        self,
        text: str,
    ) -> TokenizationResult:

        normalized = self.normalize(
            text
        )

        token_objects = (
            self.tokenize_objects(
                text
            )
        )

        token_texts = [
            token.text
            for token in token_objects
        ]

        frequencies = Counter(
            token_texts
        )

        sentences = (
            self.split_sentences(
                text
            )
        )

        paragraphs = (
            self.split_paragraphs(
                text
            )
        )

        phrases = (
            self.extract_phrases(
                text,
                size=2,
            )
        )

        return TokenizationResult(
            original_text=text or "",
            normalized_text=normalized,
            tokens=token_objects,
            phrases=phrases,
            frequencies=dict(
                frequencies
            ),
            unique_tokens=list(
                frequencies.keys()
            ),
            sentences=sentences,
            paragraphs=paragraphs,
            character_count=len(
                text or ""
            ),
            word_count=len(
                token_objects
            ),
            unique_count=len(
                frequencies
            ),
            processing_metadata={
                "lowercase": self.lowercase,
                "unicode_normalization": (
                    self.normalize_unicode
                ),
                "stop_words_removed": (
                    self.remove_stop_words
                ),
                "stemming": (
                    self.enable_stemming
                ),
                "synonyms": (
                    self.enable_synonyms
                ),
            },
        )

    # ========================================================
    # DOCUMENT SIGNATURE
    # ========================================================

    def signature(
        self,
        text: str,
    ) -> str:

        """
        Produce a stable normalized representation
        useful for duplicate detection.
        """

        tokens = self.tokenize(
            text
        )

        return " ".join(
            tokens
        )

    # ========================================================
    # TEXT SIMILARITY
    # ========================================================

    def jaccard_similarity(
        self,
        first: str,
        second: str,
    ) -> float:

        first_set = set(
            self.tokenize(first)
        )

        second_set = set(
            self.tokenize(second)
        )

        if not first_set and not second_set:
            return 1.0

        if not first_set or not second_set:
            return 0.0

        intersection = (
            first_set
            & second_set
        )

        union = (
            first_set
            | second_set
        )

        return (
            len(intersection)
            / len(union)
        )

    def cosine_similarity(
        self,
        first: str,
        second: str,
    ) -> float:

        first_counts = (
            self.frequency_counter(
                first
            )
        )

        second_counts = (
            self.frequency_counter(
                second
            )
        )

        vocabulary = set(
            first_counts
        ) | set(
            second_counts
        )

        if not vocabulary:
            return 1.0

        dot_product = sum(
            first_counts.get(
                token,
                0,
            )
            * second_counts.get(
                token,
                0,
            )
            for token in vocabulary
        )

        first_magnitude = math.sqrt(
            sum(
                value * value
                for value
                in first_counts.values()
            )
        )

        second_magnitude = math.sqrt(
            sum(
                value * value
                for value
                in second_counts.values()
            )
        )

        if (
            first_magnitude == 0
            or second_magnitude == 0
        ):
            return 0.0

        return (
            dot_product
            / (
                first_magnitude
                * second_magnitude
            )
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    def statistics(
        self,
        text: str,
    ) -> Dict:

        result = self.analyze(
            text
        )

        frequencies = (
            result.frequencies
        )

        most_common = sorted(
            frequencies.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:20]

        return {
            "characters": (
                result.character_count
            ),
            "words": (
                result.word_count
            ),
            "unique_words": (
                result.unique_count
            ),
            "sentences": len(
                result.sentences
            ),
            "paragraphs": len(
                result.paragraphs
            ),
            "average_word_length": (
                sum(
                    len(token)
                    for token
                    in result.token_texts()
                )
                / result.word_count
                if result.word_count
                else 0.0
            ),
            "lexical_diversity": (
                result.unique_count
                / result.word_count
                if result.word_count
                else 0.0
            ),
            "most_common": most_common,
        }


# ============================================================
# DEFAULT TOKENIZER
# ============================================================


tokenizer = Tokenizer()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def tokenize(
    text: str,
) -> List[str]:

    return tokenizer.tokenize(
        text
    )


def normalize(
    text: str,
) -> str:

    return tokenizer.normalize(
        text
    )


def parse_query(
    query: str,
) -> List[QueryTerm]:

    return tokenizer.parse_query(
        query
    )


def analyze(
    text: str,
) -> TokenizationResult:

    return tokenizer.analyze(
        text
    )