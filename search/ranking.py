"""
OurPlatform Search Ranking Engine
=================================

Advanced relevance and ranking subsystem.

Responsibilities
----------------
- BM25-style lexical scoring
- TF-IDF scoring
- Term-frequency saturation
- Field-aware relevance
- Field boosting
- Exact-match scoring
- Phrase scoring
- Prefix scoring
- Fuzzy scoring
- Synonym scoring
- Title/content differentiation
- Document-quality scoring
- Popularity scoring
- Recency scoring
- Length normalization
- Score normalization
- Score combination
- Configurable ranking profiles
- Ranking explanations
- Candidate filtering
- Tie breaking
- Batch ranking
- Score statistics
- Query-term diagnostics

The ranking engine intentionally does not own the search index.
It consumes query/document information supplied by the search
engine and produces ordered relevance results.

This separation allows:

    tokenizer.py
        ↓
    index.py
        ↓
    ranking.py
        ↓
    engine.py

to evolve independently.
"""

from __future__ import annotations

import math
import statistics

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

try:
    from search.tokenizer import (
        Tokenizer,
        tokenizer as default_tokenizer,
    )
except ImportError:
    from tokenizer import (
        Tokenizer,
        tokenizer as default_tokenizer,
    )


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class RankingWeights:
    """
    Controls how individual relevance signals contribute
    to the final score.

    The values are intentionally configurable so the ranking
    engine can evolve without changing its core implementation.
    """

    lexical: float = 1.00

    title: float = 2.50

    heading: float = 1.80

    tags: float = 1.60

    keywords: float = 1.50

    exact: float = 3.00

    phrase: float = 2.40

    prefix: float = 0.80

    fuzzy: float = 0.55

    synonym: float = 0.45

    coverage: float = 1.30

    frequency: float = 0.35

    length: float = 0.25

    quality: float = 0.40

    popularity: float = 0.20

    recency: float = 0.20


@dataclass
class RankingConfig:
    """
    Global ranking configuration.
    """

    weights: RankingWeights = field(
        default_factory=RankingWeights
    )

    # BM25 parameters.
    bm25_k1: float = 1.20

    bm25_b: float = 0.75

    # Maximum contribution from extremely frequent terms.
    max_term_frequency_bonus: float = 1.50

    # Phrase matching.
    phrase_window: int = 5

    # Fuzzy matching.
    fuzzy_threshold: float = 0.72

    maximum_fuzzy_candidates: int = 5

    # Query coverage.
    minimum_coverage: float = 0.0

    # Length normalization.
    preferred_document_length: float = 300.0

    minimum_length_factor: float = 0.60

    maximum_length_factor: float = 1.20

    # Recency.
    recency_half_life_days: float = 30.0

    # Score normalization.
    normalize_scores: bool = True

    # Whether to apply boosts multiplicatively.
    multiplicative_boosts: bool = False

    # Ranking behaviour.
    remove_zero_score_results: bool = True

    # Default number of results.
    default_limit: int = 20

    # Tie-breaking precision.
    score_precision: int = 8


@dataclass
class DocumentProfile:
    """
    Normalized representation of a searchable document.

    The ranking engine accepts dictionaries too, but converting
    them into this structure gives the scorer a predictable
    interface.
    """

    document_id: Any

    text: str = ""

    title: str = ""

    heading: str = ""

    tags: List[str] = field(
        default_factory=list
    )

    keywords: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    quality: float = 0.0

    popularity: float = 0.0

    created_at: Optional[Any] = None

    updated_at: Optional[Any] = None

    length: int = 0

    term_frequencies: Dict[str, int] = field(
        default_factory=dict
    )

    fields: Dict[str, str] = field(
        default_factory=dict
    )

    def get_field(
        self,
        name: str,
    ) -> str:

        if name == "text":
            return self.text

        if name == "title":
            return self.title

        if name == "heading":
            return self.heading

        if name == "tags":
            return " ".join(self.tags)

        if name == "keywords":
            return " ".join(self.keywords)

        return str(
            self.fields.get(
                name,
                "",
            )
        )


@dataclass
class QueryProfile:
    """
    Normalized representation of a search query.
    """

    original: str

    terms: List[str] = field(
        default_factory=list
    )

    required_terms: List[str] = field(
        default_factory=list
    )

    excluded_terms: List[str] = field(
        default_factory=list
    )

    phrases: List[str] = field(
        default_factory=list
    )

    fields: Dict[str, List[str]] = field(
        default_factory=dict
    )

    fuzzy_terms: List[str] = field(
        default_factory=list
    )

    prefix_terms: List[str] = field(
        default_factory=list
    )

    synonyms: Dict[str, List[str]] = field(
        default_factory=dict
    )

    weights: Dict[str, float] = field(
        default_factory=dict
    )


@dataclass
class ScoreBreakdown:
    """
    Detailed explanation of how a document received its score.
    """

    lexical: float = 0.0

    title: float = 0.0

    heading: float = 0.0

    tags: float = 0.0

    keywords: float = 0.0

    exact: float = 0.0

    phrase: float = 0.0

    prefix: float = 0.0

    fuzzy: float = 0.0

    synonym: float = 0.0

    coverage: float = 0.0

    frequency: float = 0.0

    length: float = 0.0

    quality: float = 0.0

    popularity: float = 0.0

    recency: float = 0.0

    raw_total: float = 0.0

    normalized_total: float = 0.0

    matched_terms: List[str] = field(
        default_factory=list
    )

    unmatched_terms: List[str] = field(
        default_factory=list
    )

    excluded_matches: List[str] = field(
        default_factory=list
    )

    matched_phrases: List[str] = field(
        default_factory=list
    )

    fuzzy_matches: Dict[str, str] = field(
        default_factory=dict
    )

    field_matches: Dict[str, List[str]] = field(
        default_factory=dict
    )

    notes: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "lexical": self.lexical,
            "title": self.title,
            "heading": self.heading,
            "tags": self.tags,
            "keywords": self.keywords,
            "exact": self.exact,
            "phrase": self.phrase,
            "prefix": self.prefix,
            "fuzzy": self.fuzzy,
            "synonym": self.synonym,
            "coverage": self.coverage,
            "frequency": self.frequency,
            "length": self.length,
            "quality": self.quality,
            "popularity": self.popularity,
            "recency": self.recency,
            "raw_total": self.raw_total,
            "normalized_total": self.normalized_total,
            "matched_terms": list(
                self.matched_terms
            ),
            "unmatched_terms": list(
                self.unmatched_terms
            ),
            "excluded_matches": list(
                self.excluded_matches
            ),
            "matched_phrases": list(
                self.matched_phrases
            ),
            "fuzzy_matches": dict(
                self.fuzzy_matches
            ),
            "field_matches": {
                key: list(value)
                for key, value
                in self.field_matches.items()
            },
            "notes": list(
                self.notes
            ),
        }


@dataclass
class RankedResult:
    """
    Final ranked search result.
    """

    document_id: Any

    score: float

    document: Any

    rank: int = 0

    explanation: Optional[
        ScoreBreakdown
    ] = None

    matched_terms: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "document_id": self.document_id,
            "score": self.score,
            "rank": self.rank,
            "document": self.document,
            "matched_terms": list(
                self.matched_terms
            ),
            "explanation": (
                self.explanation.to_dict()
                if self.explanation
                else None
            ),
        }


# ============================================================
# RANKING ENGINE
# ============================================================


class RankingEngine:
    """
    Main search-ranking engine.

    The engine combines multiple independent signals rather
    than relying on a single score.

    Main pipeline:

        query
          ↓
        query profile
          ↓
        lexical matching
          ↓
        field scoring
          ↓
        phrase scoring
          ↓
        exact/prefix/fuzzy scoring
          ↓
        coverage
          ↓
        document quality
          ↓
        recency/popularity
          ↓
        normalization
          ↓
        final ranking
    """

    FIELD_WEIGHTS = {
        "title": 2.50,
        "heading": 1.80,
        "tags": 1.60,
        "keywords": 1.50,
        "text": 1.00,
    }

    def __init__(
        self,
        tokenizer_instance: Optional[
            Tokenizer
        ] = None,
        config: Optional[
            RankingConfig
        ] = None,
    ):

        self.tokenizer = (
            tokenizer_instance
            or default_tokenizer
        )

        self.config = (
            config
            or RankingConfig()
        )

        self._document_frequency: Dict[
            str,
            int
        ] = {}

        self._document_count = 0

        self._average_document_length = (
            self.config.preferred_document_length
        )

    # ========================================================
    # DOCUMENT PREPARATION
    # ========================================================

    def prepare_document(
        self,
        document: Any,
    ) -> DocumentProfile:

        if isinstance(
            document,
            DocumentProfile,
        ):

            return document

        if not isinstance(
            document,
            Mapping,
        ):

            document = {
                "id": getattr(
                    document,
                    "id",
                    getattr(
                        document,
                        "document_id",
                        None,
                    ),
                ),
                "text": str(
                    getattr(
                        document,
                        "text",
                        document,
                    )
                ),
            }

        document_id = (
            document.get(
                "id",
                document.get(
                    "document_id"
                ),
            )
        )

        text = str(
            document.get(
                "text",
                document.get(
                    "content",
                    "",
                ),
            )
            or ""
        )

        title = str(
            document.get(
                "title",
                "",
            )
            or ""
        )

        heading = str(
            document.get(
                "heading",
                document.get(
                    "headings",
                    "",
                ),
            )
            or ""
        )

        tags = self._as_string_list(
            document.get(
                "tags",
                [],
            )
        )

        keywords = self._as_string_list(
            document.get(
                "keywords",
                [],
            )
        )

        metadata = dict(
            document.get(
                "metadata",
                {},
            )
            or {}
        )

        quality = self._safe_float(
            document.get(
                "quality",
                metadata.get(
                    "quality",
                    0.0,
                ),
            )
        )

        popularity = self._safe_float(
            document.get(
                "popularity",
                metadata.get(
                    "popularity",
                    0.0,
                ),
            )
        )

        created_at = document.get(
            "created_at",
            metadata.get(
                "created_at"
            ),
        )

        updated_at = document.get(
            "updated_at",
            metadata.get(
                "updated_at"
            ),
        )

        fields = {}

        raw_fields = document.get(
            "fields",
            {},
        )

        if isinstance(
            raw_fields,
            Mapping,
        ):

            fields = {
                str(key): str(
                    value
                )
                for key, value
                in raw_fields.items()
            }

        # Build term frequencies from the main body.
        tokens = self.tokenizer.tokenize(
            text
        )

        term_frequencies = {}

        for token in tokens:

            term_frequencies[
                token
            ] = (
                term_frequencies.get(
                    token,
                    0,
                )
                + 1
            )

        return DocumentProfile(
            document_id=document_id,
            text=text,
            title=title,
            heading=heading,
            tags=tags,
            keywords=keywords,
            metadata=metadata,
            quality=quality,
            popularity=popularity,
            created_at=created_at,
            updated_at=updated_at,
            length=len(tokens),
            term_frequencies=term_frequencies,
            fields=fields,
        )

    @staticmethod
    def _as_string_list(
        value: Any,
    ) -> List[str]:

        if value is None:
            return []

        if isinstance(
            value,
            str,
        ):

            return [value]

        try:

            return [
                str(item)
                for item in value
                if str(item).strip()
            ]

        except TypeError:

            return [str(value)]

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:

            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return default

    # ========================================================
    # QUERY PREPARATION
    # ========================================================

    def prepare_query(
        self,
        query: Any,
    ) -> QueryProfile:

        if isinstance(
            query,
            QueryProfile,
        ):

            return query

        original = str(
            query or ""
        ).strip()

        parsed = (
            self.tokenizer.parse_query(
                original
            )
        )

        terms = []

        required = []

        excluded = []

        phrases = []

        fuzzy_terms = []

        prefix_terms = []

        fields = {}

        for item in parsed:

            if item.phrase:

                phrases.append(
                    item.text
                )

                continue

            normalized = (
                self.tokenizer.normalize(
                    item.text
                )
            )

            if not normalized:
                continue

            terms.append(
                normalized
            )

            if item.required:

                required.append(
                    normalized
                )

            if item.excluded:

                excluded.append(
                    normalized
                )

            if item.fuzzy:

                fuzzy_terms.append(
                    normalized
                )

            if item.prefix:

                prefix_terms.append(
                    normalized
                )

            if item.field:

                fields.setdefault(
                    item.field,
                    [],
                ).append(
                    normalized
                )

        # If the tokenizer is configured with synonym support,
        # preserve the relationship between original terms and
        # their expansions.
        synonyms = {}

        for term in terms:

            expanded = (
                self.tokenizer.get_synonyms(
                    term
                )
            )

            if expanded:

                synonyms[term] = list(
                    expanded
                )

        return QueryProfile(
            original=original,
            terms=list(
                dict.fromkeys(
                    terms
                )
            ),
            required_terms=list(
                dict.fromkeys(
                    required
                )
            ),
            excluded_terms=list(
                dict.fromkeys(
                    excluded
                )
            ),
            phrases=list(
                dict.fromkeys(
                    phrases
                )
            ),
            fields=fields,
            fuzzy_terms=list(
                dict.fromkeys(
                    fuzzy_terms
                )
            ),
            prefix_terms=list(
                dict.fromkeys(
                    prefix_terms
                )
            ),
            synonyms=synonyms,
        )

    # ========================================================
    # INDEX STATISTICS
    # ========================================================

    def set_index_statistics(
        self,
        document_count: int,
        document_frequency: Mapping[
            str,
            int
        ],
        average_document_length: float,
    ):

        self._document_count = max(
            0,
            int(document_count),
        )

        self._document_frequency = {
            str(term): max(
                0,
                int(frequency),
            )
            for term, frequency
            in document_frequency.items()
        }

        self._average_document_length = max(
            1.0,
            float(
                average_document_length
            ),
        )

    def document_frequency(
        self,
        term: str,
    ) -> int:

        return self._document_frequency.get(
            self.tokenizer.normalize(
                term
            ),
            0,
        )

    # ========================================================
    # IDF
    # ========================================================

    def idf(
        self,
        term: str,
    ) -> float:

        """
        BM25-compatible inverse document frequency.

        The +0.5 smoothing prevents division by zero and
        keeps rare terms strongly weighted without allowing
        infinite scores.
        """

        if self._document_count <= 0:
            return 1.0

        df = self.document_frequency(
            term
        )

        numerator = (
            self._document_count
            - df
            + 0.5
        )

        denominator = (
            df
            + 0.5
        )

        return max(
            0.0,
            math.log(
                1.0
                + (
                    numerator
                    / denominator
                )
            ),
        )

    # ========================================================
    # BM25
    # ========================================================

    def bm25(
        self,
        term_frequency: int,
        document_length: int,
        term: str,
    ) -> float:

        if term_frequency <= 0:
            return 0.0

        k1 = self.config.bm25_k1

        b = self.config.bm25_b

        average_length = max(
            1.0,
            self._average_document_length,
        )

        normalization = (
            1.0
            - b
            + (
                b
                * (
                    document_length
                    / average_length
                )
            )
        )

        denominator = (
            term_frequency
            + (
                k1
                * normalization
            )
        )

        if denominator <= 0:
            return 0.0

        score = (
            self.idf(term)
            * (
                (
                    term_frequency
                    * (
                        k1 + 1.0
                    )
                )
                / denominator
            )
        )

        return max(
            0.0,
            score,
        )

    # ========================================================
    # FIELD SCORING
    # ========================================================

    def field_tokens(
        self,
        document: DocumentProfile,
        field_name: str,
    ) -> List[str]:

        value = document.get_field(
            field_name
        )

        return self.tokenizer.tokenize(
            value
        )

    def field_contains(
        self,
        document: DocumentProfile,
        field_name: str,
        term: str,
    ) -> bool:

        term = self.tokenizer.normalize(
            term
        )

        return term in set(
            self.field_tokens(
                document,
                field_name,
            )
        )

    def score_field_term(
        self,
        document: DocumentProfile,
        term: str,
        field_name: str,
    ) -> float:

        tokens = self.field_tokens(
            document,
            field_name,
        )

        if not tokens:
            return 0.0

        frequency = tokens.count(
            term
        )

        if frequency <= 0:
            return 0.0

        base = self.bm25(
            frequency,
            len(tokens),
            term,
        )

        field_weight = (
            self.FIELD_WEIGHTS.get(
                field_name,
                1.0,
            )
        )

        return (
            base
            * field_weight
        )

    # ========================================================
    # LEXICAL SCORING
    # ========================================================

    def score_lexical(
        self,
        document: DocumentProfile,
        query: QueryProfile,
    ) -> Tuple[
        float,
        List[str],
    ]:

        total = 0.0

        matched = []

        body_tokens = set(
            self.tokenizer.tokenize(
                document.text
            )
        )

        for term in query.terms:

            frequency = (
                document.term_frequencies.get(
                    term,
                    0,
                )
            )

            if frequency > 0:

                score = self.bm25(
                    frequency,
                    document.length,
                    term,
                )

                total += score

                matched.append(
                    term
                )

            elif term in body_tokens:

                matched.append(
                    term
                )

        return (
            total,
            list(
                dict.fromkeys(
                    matched
                )
            ),
        )

    # ========================================================
    # EXACT MATCH
    # ========================================================

    def score_exact(
        self,
        document: DocumentProfile,
        query: QueryProfile,
    ) -> float:

        if not query.terms:
            return 0.0

        score = 0.0

        fields = [
            "title",
            "heading",
            "tags",
            "keywords",
            "text",
        ]

        for term in query.terms:

            for field_name in fields:

                if self.field_contains(
                    document,
                    field_name,
                    term,
                ):

                    score += (
                        self.FIELD_WEIGHTS.get(
                            field_name,
                            1.0,
                        )
                    )

        return score

    # ========================================================
    # PHRASE MATCHING
    # ========================================================

    def phrase_occurrences(
        self,
        document: DocumentProfile,
        phrase: str,
    ) -> int:

        phrase_tokens = (
            self.tokenizer.tokenize(
                phrase,
                remove_stop_words=False,
            )
        )

        if not phrase_tokens:
            return 0

        document_tokens = (
            self.tokenizer.tokenize(
                document.text,
                remove_stop_words=False,
            )
        )

        size = len(
            phrase_tokens
        )

        occurrences = 0

        for index in range(
            len(document_tokens)
            - size
            + 1
        ):

            window = document_tokens[
                index:index + size
            ]

            if window == phrase_tokens:

                occurrences += 1

        return occurrences

    def score_phrases(
        self,
        document: DocumentProfile,
        query: QueryProfile,
    ) -> Tuple[
        float,
        List[str],
    ]:

        total = 0.0

        matched = []

        for phrase in query.phrases:

            occurrences = (
                self.phrase_occurrences(
                    document,
                    phrase,
                )
            )

            if occurrences <= 0:
                continue

            # Diminishing returns for repeated phrases.
            contribution = (
                1.0
                + math.log1p(
                    occurrences
                )
            )

            total += contribution

            matched.append(
                phrase
            )

        return (
            total,
            matched,
        )

    # ========================================================
    # PREFIX MATCHING
    # ========================================================

    def score_prefix(
        self,
        document: DocumentProfile,
        query: QueryProfile,
    ) -> float:

        if not query.prefix_terms:
            return 0.0

        tokens = set(
            self.tokenizer.tokenize(
                document.text
            )
        )

        score = 0.0

        for prefix in query.prefix_terms:

            if not prefix:
                continue

            matches = [
                token
                for token in tokens
                if token.startswith(
                    prefix
                )
            ]

            for token in matches:

                length_ratio = (
                    len(prefix)
                    / max(
                        len(token),
                        1,
                    )
                )

                score += (
                    length_ratio
                    * 1.0
                )

        return score

    # ========================================================
    # FUZZY MATCHING
    # ========================================================

    def score_fuzzy(
        self,
        document: DocumentProfile,
        query: QueryProfile,
    ) -> Tuple[
        float,
        Dict[str, str],
    ]:

        if not query.fuzzy_terms:
            return (
                0.0,
                {},
            )

        document_tokens = set(
            self.tokenizer.tokenize(
                document.text
            )
        )

        total = 0.0

        matches = {}

        for term in query.fuzzy_terms:

            candidates = (
                self.tokenizer.fuzzy_candidates(
                    term,
                    document_tokens,
                    threshold=(
                        self.config
                        .fuzzy_threshold
                    ),
                    maximum=(
                        self.config
                        .maximum_fuzzy_candidates
                    ),
                )
            )

            if not candidates:
                continue

            best_token, similarity = (
                candidates[0]
            )

            # Exact matches should be handled by lexical
            # scoring, so fuzzy scoring rewards similarity
            # without overpowering exact matches.
            if (
                best_token == term
            ):
                continue

            total += similarity

            matches[term] = (
                best_token
            )

        return (
            total,
            matches,
        )

    # ========================================================
    # SYNONYM MATCHING
    # ========================================================

    def score_synonyms(
        self,
        document: DocumentProfile,
        query: QueryProfile,
    ) -> float:

        if not query.synonyms:
            return 0.0

        document_tokens = set(
            self.tokenizer.tokenize(
                document.text
            )
        )

        total = 0.0

        for term, synonyms in (
            query.synonyms.items()
        ):

            for synonym in synonyms:

                if synonym in document_tokens:

                    total += 1.0

        return total

    # ========================================================
    # TERM COVERAGE
    # ========================================================

    def coverage(
        self,
        document: DocumentProfile,
        query: QueryProfile,
    ) -> float:

        if not query.terms:
            return 0.0

        document_tokens = set(
            self.tokenizer.tokenize(
                document.text
            )
        )

        matched = sum(
            1
            for term in query.terms
            if term in document_tokens
        )

        return (
            matched
            / len(query.terms)
        )

    # ========================================================
    # TERM FREQUENCY BONUS
    # ========================================================

    def frequency_bonus(
        self,
        document: DocumentProfile,
        query: QueryProfile,
    ) -> float:

        if not query.terms:
            return 0.0

        bonus = 0.0

        for term in query.terms:

            frequency = (
                document.term_frequencies.get(
                    term,
                    0,
                )
            )

            if frequency <= 0:
                continue

            # Logarithmic growth prevents documents with
            # enormous repetition from dominating.
            contribution = (
                math.log1p(
                    frequency
                )
            )

            bonus += min(
                contribution,
                self.config
                .max_term_frequency_bonus,
            )

        return bonus

    # ========================================================
    # DOCUMENT LENGTH
    # ========================================================

    def length_factor(
        self,
        document: DocumentProfile,
    ) -> float:

        length = max(
            1,
            document.length,
        )

        preferred = max(
            1.0,
            self.config
            .preferred_document_length,
        )

        # Documents around the preferred length receive
        # a neutral factor. Extremely short or long documents
        # are gently penalized.
        ratio = (
            preferred
            / length
        )

        factor = math.sqrt(
            min(
                max(
                    ratio,
                    self.config
                    .minimum_length_factor,
                ),
                self.config
                .maximum_length_factor,
            )
        )

        return factor

    # ========================================================
    # QUALITY
    # ========================================================

    def quality_score(
        self,
        document: DocumentProfile,
    ) -> float:

        return self._bounded_score(
            document.quality
        )

    # ========================================================
    # POPULARITY
    # ========================================================

    def popularity_score(
        self,
        document: DocumentProfile,
    ) -> float:

        popularity = max(
            0.0,
            document.popularity,
        )

        if popularity == 0:
            return 0.0

        # Logarithmic compression prevents popularity from
        # overwhelming relevance.
        return min(
            1.0,
            math.log1p(
                popularity
            )
            / math.log(
                101.0
            ),
        )

    # ========================================================
    # RECENCY
    # ========================================================

    def _parse_datetime(
        self,
        value: Any,
    ) -> Optional[datetime]:

        if value is None:
            return None

        if isinstance(
            value,
            datetime,
        ):

            result = value

        elif isinstance(
            value,
            (int, float),
        ):

            try:

                result = datetime.fromtimestamp(
                    value,
                    tz=timezone.utc,
                )

            except (
                OverflowError,
                OSError,
                ValueError,
            ):

                return None

        else:

            text = str(
                value
            ).strip()

            if not text:
                return None

            try:

                result = datetime.fromisoformat(
                    text.replace(
                        "Z",
                        "+00:00",
                    )
                )

            except ValueError:

                return None

        if result.tzinfo is None:

            result = result.replace(
                tzinfo=timezone.utc
            )

        return result.astimezone(
            timezone.utc
        )

    def recency_score(
        self,
        document: DocumentProfile,
        now: Optional[
            datetime
        ] = None,
    ) -> float:

        timestamp = (
            document.updated_at
            or document.created_at
        )

        parsed = self._parse_datetime(
            timestamp
        )

        if parsed is None:
            return 0.0

        if now is None:

            now = datetime.now(
                timezone.utc
            )

        if now.tzinfo is None:

            now = now.replace(
                tzinfo=timezone.utc
            )

        age_seconds = max(
            0.0,
            (
                now
                - parsed
            ).total_seconds(),
        )

        age_days = (
            age_seconds
            / 86400.0
        )

        half_life = max(
            0.01,
            self.config
            .recency_half_life_days,
        )

        return math.pow(
            0.5,
            age_days / half_life,
        )

    # ========================================================
    # REQUIRED / EXCLUDED TERMS
    # ========================================================

    def check_exclusions(
        self,
        document: DocumentProfile,
        query: QueryProfile,
    ) -> List[str]:

        tokens = set(
            self.tokenizer.tokenize(
                document.text
            )
        )

        return [
            term
            for term in query.excluded_terms
            if term in tokens
        ]

    def check_required(
        self,
        document: DocumentProfile,
        query: QueryProfile,
    ) -> bool:

        if not query.required_terms:
            return True

        tokens = set(
            self.tokenizer.tokenize(
                document.text
            )
        )

        return all(
            term in tokens
            for term in query.required_terms
        )

    # ========================================================
    # FIELD-SPECIFIC QUERY MATCHING
    # ========================================================

    def score_requested_fields(
        self,
        document: DocumentProfile,
        query: QueryProfile,
    ) -> Tuple[
        Dict[str, float],
        Dict[str, List[str]],
    ]:

        scores = {}

        matches = {}

        for field_name, terms in (
            query.fields.items()
        ):

            field_score = 0.0

            field_matches = []

            for term in terms:

                score = (
                    self.score_field_term(
                        document,
                        term,
                        field_name,
                    )
                )

                if score > 0:

                    field_score += score

                    field_matches.append(
                        term
                    )

            scores[field_name] = (
                field_score
            )

            matches[field_name] = (
                field_matches
            )

        return (
            scores,
            matches,
        )

    # ========================================================
    # TOTAL SCORE
    # ========================================================

    def score(
        self,
        document: Any,
        query: Any,
        explain: bool = False,
    ) -> Tuple[
        float,
        Optional[ScoreBreakdown],
    ]:

        document_profile = (
            self.prepare_document(
                document
            )
        )

        query_profile = (
            self.prepare_query(
                query
            )
        )

        breakdown = (
            ScoreBreakdown()
        )

        # -----------------------------------------------
        # Hard filters
        # -----------------------------------------------

        excluded = (
            self.check_exclusions(
                document_profile,
                query_profile,
            )
        )

        if excluded:

            breakdown.excluded_matches = (
                excluded
            )

            breakdown.notes.append(
                "Document contains an excluded term."
            )

            return (
                0.0,
                breakdown if explain else None,
            )

        if not self.check_required(
            document_profile,
            query_profile,
        ):

            breakdown.notes.append(
                "Document is missing a required term."
            )

            return (
                0.0,
                breakdown if explain else None,
            )

        # -----------------------------------------------
        # Lexical
        # -----------------------------------------------

        (
            lexical,
            matched_terms,
        ) = self.score_lexical(
            document_profile,
            query_profile,
        )

        breakdown.lexical = lexical

        breakdown.matched_terms = (
            matched_terms
        )

        breakdown.unmatched_terms = [
            term
            for term in query_profile.terms
            if term not in matched_terms
        ]

        # -----------------------------------------------
        # Field scores
        # -----------------------------------------------

        title_score = 0.0
        heading_score = 0.0
        tags_score = 0.0
        keywords_score = 0.0

        for term in query_profile.terms:

            title_score += (
                self.score_field_term(
                    document_profile,
                    term,
                    "title",
                )
            )

            heading_score += (
                self.score_field_term(
                    document_profile,
                    term,
                    "heading",
                )
            )

            tags_score += (
                self.score_field_term(
                    document_profile,
                    term,
                    "tags",
                )
            )

            keywords_score += (
                self.score_field_term(
                    document_profile,
                    term,
                    "keywords",
                )
            )

        breakdown.title = (
            title_score
        )

        breakdown.heading = (
            heading_score
        )

        breakdown.tags = (
            tags_score
        )

        breakdown.keywords = (
            keywords_score
        )

        # -----------------------------------------------
        # Exact matching
        # -----------------------------------------------

        breakdown.exact = (
            self.score_exact(
                document_profile,
                query_profile,
            )
        )

        # -----------------------------------------------
        # Phrases
        # -----------------------------------------------

        (
            breakdown.phrase,
            breakdown.matched_phrases,
        ) = self.score_phrases(
            document_profile,
            query_profile,
        )

        # -----------------------------------------------
        # Prefix
        # -----------------------------------------------

        breakdown.prefix = (
            self.score_prefix(
                document_profile,
                query_profile,
            )
        )

        # -----------------------------------------------
        # Fuzzy
        # -----------------------------------------------

        (
            breakdown.fuzzy,
            breakdown.fuzzy_matches,
        ) = self.score_fuzzy(
            document_profile,
            query_profile,
        )

        # -----------------------------------------------
        # Synonyms
        # -----------------------------------------------

        breakdown.synonym = (
            self.score_synonyms(
                document_profile,
                query_profile,
            )
        )

        # -----------------------------------------------
        # Coverage
        # -----------------------------------------------

        coverage = self.coverage(
            document_profile,
            query_profile,
        )

        breakdown.coverage = coverage

        if (
            coverage
            < self.config.minimum_coverage
        ):

            breakdown.notes.append(
                "Document did not reach minimum query coverage."
            )

            return (
                0.0,
                breakdown if explain else None,
            )

        # -----------------------------------------------
        # Frequency
        # -----------------------------------------------

        breakdown.frequency = (
            self.frequency_bonus(
                document_profile,
                query_profile,
            )
        )

        # -----------------------------------------------
        # Length
        # -----------------------------------------------

        breakdown.length = (
            self.length_factor(
                document_profile
            )
        )

        # -----------------------------------------------
        # Quality
        # -----------------------------------------------

        breakdown.quality = (
            self.quality_score(
                document_profile
            )
        )

        # -----------------------------------------------
        # Popularity
        # -----------------------------------------------

        breakdown.popularity = (
            self.popularity_score(
                document_profile
            )
        )

        # -----------------------------------------------
        # Recency
        # -----------------------------------------------

        breakdown.recency = (
            self.recency_score(
                document_profile
            )
        )

        # -----------------------------------------------
        # Requested fields
        # -----------------------------------------------

        (
            requested_scores,
            requested_matches,
        ) = self.score_requested_fields(
            document_profile,
            query_profile,
        )

        for field_name, score in (
            requested_scores.items()
        ):

            if field_name == "title":

                breakdown.title += score

            elif field_name == "heading":

                breakdown.heading += score

            elif field_name == "tags":

                breakdown.tags += score

            elif field_name == "keywords":

                breakdown.keywords += score

        breakdown.field_matches = (
            requested_matches
        )

        # -----------------------------------------------
        # Combine signals
        # -----------------------------------------------

        weights = (
            self.config.weights
        )

        raw_total = 0.0

        raw_total += (
            breakdown.lexical
            * weights.lexical
        )

        raw_total += (
            breakdown.title
            * weights.title
        )

        raw_total += (
            breakdown.heading
            * weights.heading
        )

        raw_total += (
            breakdown.tags
            * weights.tags
        )

        raw_total += (
            breakdown.keywords
            * weights.keywords
        )

        raw_total += (
            breakdown.exact
            * weights.exact
        )

        raw_total += (
            breakdown.phrase
            * weights.phrase
        )

        raw_total += (
            breakdown.prefix
            * weights.prefix
        )

        raw_total += (
            breakdown.fuzzy
            * weights.fuzzy
        )

        raw_total += (
            breakdown.synonym
            * weights.synonym
        )

        raw_total += (
            breakdown.coverage
            * weights.coverage
        )

        raw_total += (
            breakdown.frequency
            * weights.frequency
        )

        raw_total += (
            breakdown.quality
            * weights.quality
        )

        raw_total += (
            breakdown.popularity
            * weights.popularity
        )

        raw_total += (
            breakdown.recency
            * weights.recency
        )

        # Length acts as a gentle modifier rather than
        # dominating the relevance score.
        raw_total *= (
            1.0
            + (
                (
                    breakdown.length
                    - 1.0
                )
                * weights.length
            )
        )

        raw_total = max(
            0.0,
            raw_total,
        )

        breakdown.raw_total = (
            raw_total
        )

        breakdown.normalized_total = (
            raw_total
        )

        return (
            raw_total,
            breakdown if explain else None,
        )

    # ========================================================
    # RESULT NORMALIZATION
    # ========================================================

    def normalize_result_scores(
        self,
        results: List[
            RankedResult
        ],
    ) -> List[
        RankedResult
    ]:

        if not results:
            return results

        maximum = max(
            result.score
            for result in results
        )

        minimum = min(
            result.score
            for result in results
        )

        if (
            maximum == minimum
        ):

            for result in results:

                result.score = 1.0

                if result.explanation:

                    result.explanation.normalized_total = (
                        1.0
                    )

            return results

        span = (
            maximum
            - minimum
        )

        for result in results:

            normalized = (
                result.score
                - minimum
            ) / span

            result.score = normalized

            if result.explanation:

                result.explanation.normalized_total = (
                    normalized
                )

        return results

    # ========================================================
    # RANK
    # ========================================================

    def rank(
        self,
        documents: Iterable[Any],
        query: Any,
        limit: Optional[int] = None,
        explain: bool = False,
    ) -> List[
        RankedResult
    ]:

        query_profile = (
            self.prepare_query(
                query
            )
        )

        if not query_profile.terms and not query_profile.phrases:

            return []

        ranked = []

        for document in documents:

            profile = (
                self.prepare_document(
                    document
                )
            )

            score, explanation = (
                self.score(
                    profile,
                    query_profile,
                    explain=explain,
                )
            )

            if (
                self.config
                .remove_zero_score_results
                and score <= 0
            ):

                continue

            ranked.append(
                RankedResult(
                    document_id=(
                        profile.document_id
                    ),
                    score=score,
                    document=document,
                    explanation=explanation,
                    matched_terms=(
                        explanation.matched_terms
                        if explanation
                        else []
                    ),
                )
            )

        # Highest score first.
        ranked.sort(
            key=lambda result: (
                -result.score,
                str(
                    result.document_id
                ),
            )
        )

        if self.config.normalize_scores:

            self.normalize_result_scores(
                ranked
            )

            ranked.sort(
                key=lambda result: (
                    -result.score,
                    str(
                        result.document_id
                    ),
                )
            )

        if limit is None:

            limit = (
                self.config.default_limit
            )

        limit = max(
            0,
            int(limit),
        )

        ranked = ranked[
            :limit
        ]

        for rank, result in enumerate(
            ranked,
            start=1,
        ):

            result.rank = rank

            result.score = round(
                result.score,
                self.config.score_precision,
            )

        return ranked

    # ========================================================
    # TOP RESULT
    # ========================================================

    def best_match(
        self,
        documents: Iterable[Any],
        query: Any,
        explain: bool = False,
    ) -> Optional[
        RankedResult
    ]:

        results = self.rank(
            documents,
            query,
            limit=1,
            explain=explain,
        )

        if not results:
            return None

        return results[0]

    # ========================================================
    # SCORE ONE DOCUMENT
    # ========================================================

    def score_document(
        self,
        document: Any,
        query: Any,
        explain: bool = True,
    ) -> Dict[str, Any]:

        score, breakdown = (
            self.score(
                document,
                query,
                explain=explain,
            )
        )

        return {
            "score": round(
                score,
                self.config.score_precision,
            ),
            "explanation": (
                breakdown.to_dict()
                if breakdown
                else None
            ),
        }

    # ========================================================
    # SCORE STATISTICS
    # ========================================================

    def score_statistics(
        self,
        results: Sequence[
            RankedResult
        ],
    ) -> Dict[str, float]:

        if not results:

            return {
                "count": 0,
                "minimum": 0.0,
                "maximum": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "total": 0.0,
            }

        scores = [
            result.score
            for result in results
        ]

        return {
            "count": len(
                scores
            ),
            "minimum": min(
                scores
            ),
            "maximum": max(
                scores
            ),
            "mean": statistics.mean(
                scores
            ),
            "median": statistics.median(
                scores
            ),
            "total": sum(
                scores
            ),
        }

    # ========================================================
    # RANKING PROFILE MANAGEMENT
    # ========================================================

    def update_weights(
        self,
        **weights: float,
    ):

        valid_fields = {
            name
            for name in vars(
                self.config.weights
            )
        }

        for name, value in (
            weights.items()
        ):

            if name not in valid_fields:

                raise ValueError(
                    f"Unknown ranking weight: {name}"
                )

            setattr(
                self.config.weights,
                name,
                float(value),
            )

    def get_weights(
        self,
    ) -> Dict[str, float]:

        return {
            name: float(value)
            for name, value
            in vars(
                self.config.weights
            ).items()
        }

    # ========================================================
    # INDEX PROFILE
    # ========================================================

    def describe_index(
        self,
    ) -> Dict[str, Any]:

        return {
            "document_count": (
                self._document_count
            ),
            "average_document_length": (
                self._average_document_length
            ),
            "terms_with_statistics": len(
                self._document_frequency
            ),
        }

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    def diagnose_query(
        self,
        query: str,
    ) -> Dict[str, Any]:

        profile = (
            self.prepare_query(
                query
            )
        )

        return {
            "original": profile.original,
            "terms": list(
                profile.terms
            ),
            "required_terms": list(
                profile.required_terms
            ),
            "excluded_terms": list(
                profile.excluded_terms
            ),
            "phrases": list(
                profile.phrases
            ),
            "fuzzy_terms": list(
                profile.fuzzy_terms
            ),
            "prefix_terms": list(
                profile.prefix_terms
            ),
            "fields": {
                key: list(value)
                for key, value
                in profile.fields.items()
            },
            "synonyms": {
                key: list(value)
                for key, value
                in profile.synonyms.items()
            },
        }

    def explain(
        self,
        document: Any,
        query: str,
    ) -> Dict[str, Any]:

        score, breakdown = (
            self.score(
                document,
                query,
                explain=True,
            )
        )

        return {
            "query": query,
            "score": score,
            "explanation": (
                breakdown.to_dict()
                if breakdown
                else None
            ),
        }

    # ========================================================
    # UTILITY
    # ========================================================

    @staticmethod
    def _bounded_score(
        value: float,
    ) -> float:

        if not math.isfinite(
            value
        ):

            return 0.0

        return min(
            1.0,
            max(
                0.0,
                value,
            ),
        )


# ============================================================
# DEFAULT ENGINE
# ============================================================


ranking = RankingEngine()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def rank(
    documents: Iterable[Any],
    query: str,
    limit: int = 20,
) -> List[
    RankedResult
]:

    return ranking.rank(
        documents,
        query,
        limit=limit,
    )


def score_document(
    document: Any,
    query: str,
) -> Dict[str, Any]:

    return ranking.score_document(
        document,
        query,
    )


def explain(
    document: Any,
    query: str,
) -> Dict[str, Any]:

    return ranking.explain(
        document,
        query,
    )


def diagnose_query(
    query: str,
) -> Dict[str, Any]:

    return ranking.diagnose_query(
        query
    )