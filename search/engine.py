"""
OurPlatform Search Engine
=========================

Unified search orchestration layer.

Pipeline:

    Query
      ↓
    Query preparation
      ↓
    Tokenization
      ↓
    Query expansion
      ↓
    Candidate retrieval
      ↓
    Filtering
      ↓
    Ranking
      ↓
    Post-processing
      ↓
    Final results

This module intentionally sits above:

    tokenizer.py
    index.py
    ranking.py

It coordinates them without making any one subsystem
responsible for everything.

Design goals
------------

- Fast candidate retrieval
- Top-K search
- Query caching
- Field-aware retrieval
- Prefix support
- Required/excluded terms
- Metadata filtering
- Phrase candidates
- Query expansion
- Ranking integration
- Result deduplication
- Search statistics
- Search history
- Debugging
- Configurable search strategies
- Graceful degradation when optional features
  are unavailable
"""

from __future__ import annotations

import hashlib
import math
import threading
import time

from collections import Counter, OrderedDict
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


# ============================================================
# IMPORTS
# ============================================================

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


try:
    from search.index import (
        SearchIndex,
        SearchCandidates,
        index as default_index,
    )
except ImportError:
    from index import (
        SearchIndex,
        SearchCandidates,
        index as default_index,
    )


try:
    from search.ranking import (
        RankingEngine,
        ranking as default_ranking,
    )
except ImportError:

    RankingEngine = Any

    default_ranking = None


# ============================================================
# CONSTANTS
# ============================================================

ENGINE_VERSION = "1.0"

DEFAULT_TOP_K = 10

DEFAULT_MAX_CANDIDATES = 1000

DEFAULT_CACHE_SIZE = 256

DEFAULT_MINIMUM_MATCH = 1

DEFAULT_MAX_QUERY_TERMS = 64

DEFAULT_MAX_EXPANSION_TERMS = 12


# ============================================================
# RESULT STRUCTURES
# ============================================================


@dataclass
class SearchResult:
    """
    Final user-facing search result.
    """

    document_id: Any

    score: float

    document: Dict[str, Any]

    matched_terms: List[str] = field(
        default_factory=list
    )

    highlights: List[str] = field(
        default_factory=list
    )

    rank: int = 0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "document_id": self.document_id,
            "score": self.score,
            "document": self.document,
            "matched_terms": list(
                self.matched_terms
            ),
            "highlights": list(
                self.highlights
            ),
            "rank": self.rank,
            "metadata": self.metadata,
        }


@dataclass
class SearchResponse:
    """
    Complete search response.

    Contains both results and diagnostics so callers can
    inspect how the engine reached its result set.
    """

    query: str

    results: List[SearchResult]

    total_candidates: int = 0

    returned_results: int = 0

    query_terms: List[str] = field(
        default_factory=list
    )

    expanded_terms: List[str] = field(
        default_factory=list
    )

    execution_time_ms: float = 0.0

    cache_hit: bool = False

    generation: int = 0

    strategy: str = "standard"

    diagnostics: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "query": self.query,
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "total_candidates": (
                self.total_candidates
            ),
            "returned_results": (
                self.returned_results
            ),
            "query_terms": list(
                self.query_terms
            ),
            "expanded_terms": list(
                self.expanded_terms
            ),
            "execution_time_ms": (
                self.execution_time_ms
            ),
            "cache_hit": self.cache_hit,
            "generation": self.generation,
            "strategy": self.strategy,
            "diagnostics": self.diagnostics,
        }


@dataclass
class QueryPlan:
    """
    Compiled representation of a search query.

    Separating query planning from execution makes it possible
    to optimize the expensive parts independently later.
    """

    original: str

    normalized: str

    terms: List[str]

    required_terms: List[str]

    excluded_terms: List[str]

    phrase_terms: List[str]

    prefix_terms: List[str]

    field_terms: Dict[
        str,
        List[str],
    ]

    filters: Dict[
        str,
        Any,
    ]

    minimum_match: int = 1

    mode: str = "or"

    top_k: int = DEFAULT_TOP_K

    strategy: str = "standard"

    def to_dict(self) -> Dict[str, Any]:

        return {
            "original": self.original,
            "normalized": self.normalized,
            "terms": list(
                self.terms
            ),
            "required_terms": list(
                self.required_terms
            ),
            "excluded_terms": list(
                self.excluded_terms
            ),
            "phrase_terms": list(
                self.phrase_terms
            ),
            "prefix_terms": list(
                self.prefix_terms
            ),
            "field_terms": {
                key: list(value)
                for key, value
                in self.field_terms.items()
            },
            "filters": dict(
                self.filters
            ),
            "minimum_match": (
                self.minimum_match
            ),
            "mode": self.mode,
            "top_k": self.top_k,
            "strategy": self.strategy,
        }


@dataclass
class EngineStatistics:
    """
    Runtime statistics for the search engine.
    """

    searches: int = 0

    successful_searches: int = 0

    empty_searches: int = 0

    cache_hits: int = 0

    cache_misses: int = 0

    total_candidates: int = 0

    total_results: int = 0

    total_execution_ms: float = 0.0

    fastest_search_ms: Optional[float] = None

    slowest_search_ms: Optional[float] = None

    average_search_ms: float = 0.0

    last_query: Optional[str] = None

    last_search_at: Optional[str] = None

    def record(
        self,
        execution_ms: float,
        candidate_count: int,
        result_count: int,
        query: str,
        cache_hit: bool,
    ):

        self.searches += 1

        self.successful_searches += 1

        self.total_candidates += (
            candidate_count
        )

        self.total_results += (
            result_count
        )

        self.total_execution_ms += (
            execution_ms
        )

        self.last_query = query

        self.last_search_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        if cache_hit:

            self.cache_hits += 1

        else:

            self.cache_misses += 1

        if (
            self.fastest_search_ms
            is None
            or execution_ms
            < self.fastest_search_ms
        ):

            self.fastest_search_ms = (
                execution_ms
            )

        if (
            self.slowest_search_ms
            is None
            or execution_ms
            > self.slowest_search_ms
        ):

            self.slowest_search_ms = (
                execution_ms
            )

        self.average_search_ms = (
            self.total_execution_ms
            / max(
                1,
                self.searches,
            )
        )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "searches": self.searches,
            "successful_searches": (
                self.successful_searches
            ),
            "empty_searches": (
                self.empty_searches
            ),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_candidates": (
                self.total_candidates
            ),
            "total_results": (
                self.total_results
            ),
            "total_execution_ms": (
                self.total_execution_ms
            ),
            "fastest_search_ms": (
                self.fastest_search_ms
            ),
            "slowest_search_ms": (
                self.slowest_search_ms
            ),
            "average_search_ms": (
                self.average_search_ms
            ),
            "last_query": self.last_query,
            "last_search_at": (
                self.last_search_at
            ),
        }


# ============================================================
# CACHE
# ============================================================


class SearchCache:
    """
    Small thread-safe LRU cache.

    Search results are cached only after ranking has completed.
    The cache is intentionally bounded so repeated searches do
    not grow memory forever.
    """

    def __init__(
        self,
        maximum_size: int = DEFAULT_CACHE_SIZE,
    ):

        self.maximum_size = max(
            1,
            int(
                maximum_size
            ),
        )

        self._data = OrderedDict()

        self._lock = threading.RLock()

    def get(
        self,
        key: str,
    ) -> Optional[
        SearchResponse
    ]:

        with self._lock:

            if key not in self._data:

                return None

            value = self._data.pop(
                key
            )

            self._data[
                key
            ] = value

            return value

    def set(
        self,
        key: str,
        value: SearchResponse,
    ):

        with self._lock:

            if key in self._data:

                self._data.pop(
                    key
                )

            self._data[
                key
            ] = value

            while (
                len(self._data)
                > self.maximum_size
            ):

                self._data.popitem(
                    last=False
                )

    def clear(self):

        with self._lock:

            self._data.clear()

    def size(self) -> int:

        with self._lock:

            return len(
                self._data
            )


# ============================================================
# SEARCH ENGINE
# ============================================================


class SearchEngine:
    """
    Main search orchestration engine.

    The engine does not own documents.

    It coordinates:

        tokenizer
        index
        ranking engine

    This separation allows each subsystem to evolve
    independently.
    """

    def __init__(
        self,
        tokenizer_instance: Optional[
            Tokenizer
        ] = None,
        search_index: Optional[
            SearchIndex
        ] = None,
        ranking_engine: Any = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        max_candidates: int = (
            DEFAULT_MAX_CANDIDATES
        ),
    ):

        self.tokenizer = (
            tokenizer_instance
            or default_tokenizer
        )

        self.index = (
            search_index
            or default_index
        )

        self.ranking = (
            ranking_engine
            or default_ranking
        )

        self.max_candidates = max(
            1,
            int(
                max_candidates
            ),
        )

        self.cache = SearchCache(
            cache_size
        )

        self.statistics = (
            EngineStatistics()
        )

        self.history = []

        self.max_history = 500

        self._lock = threading.RLock()

    # ========================================================
    # TIME
    # ========================================================

    @staticmethod
    def _now() -> str:

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def normalize_query(
        self,
        query: str,
    ) -> str:

        if query is None:

            return ""

        return " ".join(
            str(query)
            .strip()
            .split()
        )

    # ========================================================
    # TOKENIZATION
    # ========================================================

    def tokenize_query(
        self,
        query: str,
    ) -> List[str]:

        normalized = (
            self.normalize_query(
                query
            )
        )

        if not normalized:

            return []

        tokens = list(
            self.tokenizer.tokenize(
                normalized
            )
        )

        return tokens[
            :DEFAULT_MAX_QUERY_TERMS
        ]

    # ========================================================
    # QUERY PARSING
    # ========================================================

    def build_query_plan(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        mode: str = "or",
        minimum_match: int = (
            DEFAULT_MINIMUM_MATCH
        ),
        filters: Optional[
            Mapping[str, Any]
        ] = None,
        strategy: str = "standard",
    ) -> QueryPlan:

        normalized = (
            self.normalize_query(
                query
            )
        )

        terms = self.tokenize_query(
            normalized
        )

        required_terms = []

        excluded_terms = []

        phrase_terms = []

        prefix_terms = []

        field_terms = {}

        # ----------------------------------------------------
        # Basic operators
        # ----------------------------------------------------

        raw_parts = normalized.split()

        for part in raw_parts:

            if (
                part.startswith("+")
                and len(part) > 1
            ):

                required_terms.append(
                    part[1:]
                )

            elif (
                part.startswith("-")
                and len(part) > 1
            ):

                excluded_terms.append(
                    part[1:]
                )

            elif (
                part.endswith("*")
                and len(part) > 1
            ):

                prefix_terms.append(
                    part[:-1]
                )

        # ----------------------------------------------------
        # Quoted phrases
        # ----------------------------------------------------

        phrase_matches = []

        current = []

        inside_phrase = False

        for character in normalized:

            if character == '"':

                if inside_phrase:

                    phrase = "".join(
                        current
                    ).strip()

                    if phrase:

                        phrase_matches.append(
                            phrase
                        )

                    current = []

                    inside_phrase = False

                else:

                    inside_phrase = True

                continue

            if inside_phrase:

                current.append(
                    character
                )

        phrase_terms = phrase_matches

        # ----------------------------------------------------
        # Field syntax:
        #
        # title:python
        # author:john
        # category:science
        # ----------------------------------------------------

        for part in raw_parts:

            if ":" not in part:

                continue

            field_name, value = (
                part.split(
                    ":",
                    1,
                )
            )

            field_name = (
                field_name.strip()
            )

            value = value.strip()

            if (
                not field_name
                or not value
            ):

                continue

            field_terms.setdefault(
                field_name,
                [],
            ).extend(
                self.tokenize_query(
                    value
                )
            )

        return QueryPlan(
            original=query,
            normalized=normalized,
            terms=terms,
            required_terms=[
                token
                for term
                in required_terms
                for token
                in self.tokenize_query(
                    term
                )
            ],
            excluded_terms=[
                token
                for term
                in excluded_terms
                for token
                in self.tokenize_query(
                    term
                )
            ],
            phrase_terms=phrase_terms,
            prefix_terms=[
                self.normalize_query(
                    term
                )
                for term
                in prefix_terms
            ],
            field_terms=field_terms,
            filters=dict(
                filters or {}
            ),
            minimum_match=max(
                1,
                int(
                    minimum_match
                ),
            ),
            mode=(
                "and"
                if str(mode).lower()
                == "and"
                else "or"
            ),
            top_k=max(
                1,
                int(
                    top_k
                ),
            ),
            strategy=strategy,
        )

    # ========================================================
    # QUERY CACHE KEY
    # ========================================================

    def _cache_key(
        self,
        plan: QueryPlan,
    ) -> str:

        raw = repr(
            (
                plan.normalized,
                tuple(
                    plan.terms
                ),
                tuple(
                    plan.required_terms
                ),
                tuple(
                    plan.excluded_terms
                ),
                tuple(
                    plan.phrase_terms
                ),
                tuple(
                    plan.prefix_terms
                ),
                tuple(
                    sorted(
                        (
                            key,
                            tuple(
                                value
                            ),
                        )
                        for key, value
                        in plan.field_terms.items()
                    )
                ),
                tuple(
                    sorted(
                        plan.filters.items()
                    )
                ),
                plan.minimum_match,
                plan.mode,
                plan.top_k,
                plan.strategy,
                self.index.generation,
            )
        )

        return hashlib.sha256(
            raw.encode(
                "utf-8"
            )
        ).hexdigest()

    # ========================================================
    # QUERY EXPANSION
    # ========================================================

    def expand_query(
        self,
        terms: Sequence[str],
        maximum: int = (
            DEFAULT_MAX_EXPANSION_TERMS
        ),
    ) -> List[str]:

        expanded = []

        seen = set()

        for term in terms:

            normalized = (
                self.index._normalize_value(
                    term
                )
            )

            if (
                not normalized
                or normalized in seen
            ):

                continue

            seen.add(
                normalized
            )

            expanded.append(
                normalized
            )

            # ---------------------------------------------
            # Prefix-style related vocabulary
            # ---------------------------------------------

            related = (
                self.index.related_terms(
                    normalized,
                    limit=3,
                )
            )

            for related_term, _ in (
                related
            ):

                if (
                    related_term
                    in seen
                ):

                    continue

                seen.add(
                    related_term
                )

                expanded.append(
                    related_term
                )

                if len(
                    expanded
                ) >= maximum:

                    return expanded

        return expanded

    # ========================================================
    # CANDIDATE RETRIEVAL
    # ========================================================

    def retrieve_candidates(
        self,
        plan: QueryPlan,
        expanded_terms: Optional[
            Sequence[str]
        ] = None,
    ) -> SearchCandidates:

        terms = list(
            plan.terms
        )

        if expanded_terms:

            for term in expanded_terms:

                if term not in terms:

                    terms.append(
                        term
                    )

        candidates = (
            self.index.generate_candidates(
                terms=terms,
                required_terms=(
                    plan.required_terms
                ),
                excluded_terms=(
                    plan.excluded_terms
                ),
                mode=plan.mode,
                minimum_match=(
                    plan.minimum_match
                ),
            )
        )

        candidate_ids = set(
            candidates.document_ids
        )

        # ----------------------------------------------------
        # Prefix retrieval
        # ----------------------------------------------------

        for prefix in (
            plan.prefix_terms
        ):

            candidate_ids.update(
                self.index.candidates_for_prefix(
                    prefix
                )
            )

        # ----------------------------------------------------
        # Phrase retrieval
        # ----------------------------------------------------

        for phrase in (
            plan.phrase_terms
        ):

            phrase_candidates = (
                self.index.phrase_candidates(
                    phrase
                )
            )

            if plan.mode == "and":

                candidate_ids.intersection_update(
                    phrase_candidates
                )

            else:

                candidate_ids.update(
                    phrase_candidates
                )

        # ----------------------------------------------------
        # Field retrieval
        # ----------------------------------------------------

        if plan.field_terms:

            field_candidates = (
                self.index.candidates_for_fields(
                    plan.field_terms,
                    mode="or",
                )
            )

            if plan.mode == "and":

                candidate_ids.intersection_update(
                    field_candidates
                )

            else:

                candidate_ids.update(
                    field_candidates
                )

        # ----------------------------------------------------
        # Metadata filtering
        # ----------------------------------------------------

        candidate_ids = (
            self.index.filter_documents(
                candidate_ids,
                plan.filters,
            )
        )

        # ----------------------------------------------------
        # Hard candidate cap
        #
        # We keep a deterministic subset here.
        # Ranking can still reorder it.
        # ----------------------------------------------------

        if len(
            candidate_ids
        ) > self.max_candidates:

            candidate_ids = self._trim_candidates(
                candidate_ids,
                plan,
            )

        matched_terms = {}

        for document_id in (
            candidate_ids
        ):

            matched_terms[
                document_id
            ] = {
                term
                for term in terms
                if document_id
                in self.index.documents_for_term(
                    term
                )
            }

        return SearchCandidates(
            document_ids=list(
                candidate_ids
            ),
            matched_terms=matched_terms,
            required_terms=set(
                plan.required_terms
            ),
            excluded_terms=set(
                plan.excluded_terms
            ),
            total_candidates=len(
                candidate_ids
            ),
            generation=self.index.generation,
        )

    # ========================================================
    # CANDIDATE TRIMMING
    # ========================================================

    def _trim_candidates(
        self,
        candidates: Set[Any],
        plan: QueryPlan,
    ) -> Set[Any]:

        scored = []

        for document_id in candidates:

            record = self.index.get(
                document_id
            )

            if record is None:

                continue

            matched = 0

            for term in plan.terms:

                if term in (
                    record.term_frequencies
                ):

                    matched += (
                        record.term_frequencies[
                            term
                        ]
                    )

            # Cheap pre-ranking score.
            #
            # This avoids running the full ranking
            # algorithm over thousands of weak candidates.

            score = float(
                matched
            )

            for term in plan.terms:

                score += (
                    self.index.idf(
                        term
                    )
                    * 0.25
                )

            scored.append(
                (
                    score,
                    document_id,
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                str(
                    item[1]
                ),
            )
        )

        return {
            document_id
            for _, document_id
            in scored[
                :self.max_candidates
            ]
        }

    # ========================================================
    # RANKING
    # ========================================================

    def rank_candidates(
        self,
        query: str,
        plan: QueryPlan,
        candidates: SearchCandidates,
    ) -> List[
        Tuple[Any, float]
    ]:

        if not candidates.document_ids:

            return []

        # ----------------------------------------------------
        # Preferred ranking engine
        # ----------------------------------------------------

        if self.ranking is not None:

            try:

                ranked = (
                    self._rank_with_engine(
                        query,
                        plan,
                        candidates,
                    )
                )

                if ranked is not None:

                    return ranked

            except Exception:

                # Search should remain usable even when
                # an optional advanced ranking implementation
                # changes its interface.
                pass

        # ----------------------------------------------------
        # Built-in fallback ranker
        # ----------------------------------------------------

        return self._fallback_rank(
            plan,
            candidates,
        )

    # ========================================================
    # RANKING ENGINE ADAPTER
    # ========================================================

    def _rank_with_engine(
        self,
        query: str,
        plan: QueryPlan,
        candidates: SearchCandidates,
    ) -> Optional[
        List[
            Tuple[Any, float]
        ]
    ]:

        engine = self.ranking

        # -----------------------------------------------
        # Common ranking interface
        # -----------------------------------------------

        if hasattr(
            engine,
            "rank",
        ):

            try:

                ranked = engine.rank(
                    query=query,
                    documents=[
                        self.index.get(
                            document_id
                        )
                        for document_id
                        in candidates.document_ids
                    ],
                    index=self.index,
                    top_k=plan.top_k,
                )

                return self._normalize_ranked(
                    ranked
                )

            except TypeError:

                pass

        # -----------------------------------------------
        # Alternative interface
        # -----------------------------------------------

        if hasattr(
            engine,
            "score_documents",
        ):

            ranked = (
                engine.score_documents(
                    query,
                    candidates.document_ids,
                    index=self.index,
                )
            )

            return self._normalize_ranked(
                ranked
            )

        return None

    # ========================================================
    # NORMALIZE RANKING OUTPUT
    # ========================================================

    @staticmethod
    def _normalize_ranked(
        ranked: Any,
    ) -> List[
        Tuple[Any, float]
    ]:

        if ranked is None:

            return []

        normalized = []

        for item in ranked:

            if isinstance(
                item,
                Mapping,
            ):

                document_id = (
                    item.get(
                        "document_id"
                    )
                )

                score = float(
                    item.get(
                        "score",
                        0.0,
                    )
                )

                normalized.append(
                    (
                        document_id,
                        score,
                    )
                )

                continue

            if (
                isinstance(
                    item,
                    (tuple, list),
                )
                and len(item) >= 2
            ):

                normalized.append(
                    (
                        item[0],
                        float(
                            item[1]
                        ),
                    )
                )

        normalized.sort(
            key=lambda item: (
                -item[1],
                str(
                    item[0]
                ),
            )
        )

        return normalized

    # ========================================================
    # FALLBACK RANKER
    # ========================================================

    def _fallback_rank(
        self,
        plan: QueryPlan,
        candidates: SearchCandidates,
    ) -> List[
        Tuple[Any, float]
    ]:

        ranked = []

        average_length = (
            self.index.statistics
            .average_document_length
            or 1.0
        )

        for document_id in (
            candidates.document_ids
        ):

            record = self.index.get(
                document_id
            )

            if record is None:

                continue

            score = 0.0

            # ------------------------------------------------
            # Term relevance
            # ------------------------------------------------

            for term in plan.terms:

                frequency = (
                    record.term_frequencies.get(
                        term,
                        0,
                    )
                )

                if frequency <= 0:

                    continue

                idf = (
                    self.index.idf(
                        term
                    )
                )

                # Saturating term frequency.
                tf = (
                    1.0
                    + math.log(
                        frequency
                    )
                )

                score += (
                    tf
                    * idf
                )

            # ------------------------------------------------
            # Exact phrase bonus
            # ------------------------------------------------

            text = str(
                record.data.get(
                    "text",
                    "",
                )
            ).lower()

            for phrase in (
                plan.phrase_terms
            ):

                if (
                    phrase.lower()
                    in text
                ):

                    score += 5.0

            # ------------------------------------------------
            # Title bonus
            # ------------------------------------------------

            title = str(
                record.data.get(
                    "title",
                    "",
                )
            ).lower()

            for term in plan.terms:

                if term in title:

                    score += 2.5

            # ------------------------------------------------
            # Field bonus
            # ------------------------------------------------

            for field_name, terms in (
                plan.field_terms.items()
            ):

                field_value = str(
                    record.data.get(
                        field_name,
                        "",
                    )
                ).lower()

                for term in terms:

                    if term in field_value:

                        score += 2.0

            # ------------------------------------------------
            # Mild document-length normalization
            # ------------------------------------------------

            length_ratio = (
                record.length
                / average_length
            )

            if length_ratio > 1.0:

                score /= (
                    1.0
                    + (
                        math.log(
                            length_ratio
                        )
                        * 0.15
                    )
                )

            ranked.append(
                (
                    document_id,
                    score,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item[1],
                str(
                    item[0]
                ),
            )
        )

        return ranked[
            :plan.top_k
        ]

    # ========================================================
    # RESULT CONSTRUCTION
    # ========================================================

    def build_results(
        self,
        plan: QueryPlan,
        candidates: SearchCandidates,
        ranked: Sequence[
            Tuple[Any, float]
        ],
    ) -> List[
        SearchResult
    ]:

        results = []

        seen = set()

        for rank, (
            document_id,
            score,
        ) in enumerate(
            ranked,
            start=1,
        ):

            if document_id in seen:

                continue

            seen.add(
                document_id
            )

            record = self.index.get(
                document_id
            )

            if record is None:

                continue

            matched = sorted(
                candidates.matched_terms.get(
                    document_id,
                    set(),
                )
            )

            highlights = (
                self._generate_highlights(
                    record.data,
                    plan,
                )
            )

            metadata = {
                "document_length": (
                    record.length
                ),
                "document_version": (
                    record.version
                ),
                "index_generation": (
                    self.index.generation
                ),
            }

            results.append(
                SearchResult(
                    document_id=document_id,
                    score=float(
                        score
                    ),
                    document=dict(
                        record.data
                    ),
                    matched_terms=matched,
                    highlights=highlights,
                    rank=rank,
                    metadata=metadata,
                )
            )

        return results

    # ========================================================
    # HIGHLIGHTS
    # ========================================================

    def _generate_highlights(
        self,
        document: Mapping[str, Any],
        plan: QueryPlan,
    ) -> List[str]:

        text = str(
            document.get(
                "text",
                "",
            )
        )

        if not text:

            return []

        lowered = text.lower()

        positions = []

        for term in plan.terms:

            start = 0

            while True:

                position = (
                    lowered.find(
                        term.lower(),
                        start,
                    )
                )

                if position < 0:

                    break

                positions.append(
                    position
                )

                start = (
                    position
                    + max(
                        1,
                        len(term),
                    )
                )

                if len(
                    positions
                ) >= 5:

                    break

            if len(
                positions
            ) >= 5:

                break

        if not positions:

            return []

        snippets = []

        for position in positions[:5]:

            start = max(
                0,
                position - 70,
            )

            end = min(
                len(text),
                position + 140,
            )

            snippet = text[
                start:end
            ].strip()

            if start > 0:

                snippet = (
                    "..."
                    + snippet
                )

            if end < len(text):

                snippet = (
                    snippet
                    + "..."
                )

            snippets.append(
                snippet
            )

        return snippets

    # ========================================================
    # MAIN SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        mode: str = "or",
        minimum_match: int = (
            DEFAULT_MINIMUM_MATCH
        ),
        filters: Optional[
            Mapping[str, Any]
        ] = None,
        expand: bool = False,
        strategy: str = "standard",
        use_cache: bool = True,
    ) -> SearchResponse:

        started = time.perf_counter()

        normalized = (
            self.normalize_query(
                query
            )
        )

        if not normalized:

            self.statistics.empty_searches += 1

            return SearchResponse(
                query=query,
                results=[],
                strategy=strategy,
            )

        plan = self.build_query_plan(
            query=query,
            top_k=top_k,
            mode=mode,
            minimum_match=minimum_match,
            filters=filters,
            strategy=strategy,
        )

        cache_key = (
            self._cache_key(
                plan
            )
        )

        # ----------------------------------------------------
        # Cache
        # ----------------------------------------------------

        if use_cache:

            cached = self.cache.get(
                cache_key
            )

            if cached is not None:

                elapsed = (
                    time.perf_counter()
                    - started
                ) * 1000.0

                cached = SearchResponse(
                    query=cached.query,
                    results=cached.results,
                    total_candidates=(
                        cached.total_candidates
                    ),
                    returned_results=(
                        cached.returned_results
                    ),
                    query_terms=(
                        cached.query_terms
                    ),
                    expanded_terms=(
                        cached.expanded_terms
                    ),
                    execution_time_ms=elapsed,
                    cache_hit=True,
                    generation=cached.generation,
                    strategy=cached.strategy,
                    diagnostics=dict(
                        cached.diagnostics
                    ),
                )

                self.statistics.cache_hits += 1

                return cached

            self.statistics.cache_misses += 1

        # ----------------------------------------------------
        # Query expansion
        # ----------------------------------------------------

        expanded_terms = []

        if expand:

            expanded_terms = (
                self.expand_query(
                    plan.terms
                )
            )

        # ----------------------------------------------------
        # Candidate retrieval
        # ----------------------------------------------------

        candidates = (
            self.retrieve_candidates(
                plan,
                expanded_terms,
            )
        )

        # ----------------------------------------------------
        # Ranking
        # ----------------------------------------------------

        ranked = (
            self.rank_candidates(
                query,
                plan,
                candidates,
            )
        )

        # ----------------------------------------------------
        # Result construction
        # ----------------------------------------------------

        results = (
            self.build_results(
                plan,
                candidates,
                ranked,
            )
        )

        elapsed = (
            time.perf_counter()
            - started
        ) * 1000.0

        response = SearchResponse(
            query=query,
            results=results,
            total_candidates=(
                candidates.total_candidates
            ),
            returned_results=len(
                results
            ),
            query_terms=list(
                plan.terms
            ),
            expanded_terms=list(
                expanded_terms
            ),
            execution_time_ms=elapsed,
            cache_hit=False,
            generation=self.index.generation,
            strategy=strategy,
            diagnostics={
                "query_plan": (
                    plan.to_dict()
                ),
                "index_generation": (
                    self.index.generation
                ),
                "candidate_limit": (
                    self.max_candidates
                ),
                "ranking_engine": (
                    type(
                        self.ranking
                    ).__name__
                    if self.ranking is not None
                    else "fallback"
                ),
            },
        )

        # ----------------------------------------------------
        # Cache final response
        # ----------------------------------------------------

        if use_cache:

            self.cache.set(
                cache_key,
                response,
            )

        # ----------------------------------------------------
        # Runtime statistics
        # ----------------------------------------------------

        self.statistics.record(
            execution_ms=elapsed,
            candidate_count=(
                candidates.total_candidates
            ),
            result_count=len(
                results
            ),
            query=query,
            cache_hit=False,
        )

        self._record_history(
            response
        )

        return response

    # ========================================================
    # FAST SEARCH
    # ========================================================

    def fast_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> SearchResponse:

        """
        Optimized path for ordinary queries.

        Disables expensive query expansion and keeps the
        candidate set deliberately bounded.
        """

        return self.search(
            query=query,
            top_k=top_k,
            mode="or",
            minimum_match=1,
            expand=False,
            strategy="fast",
            use_cache=True,
        )

    # ========================================================
    # DEEP SEARCH
    # ========================================================

    def deep_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> SearchResponse:

        """
        Higher-quality search path.

        Uses query expansion and more flexible retrieval.
        """

        return self.search(
            query=query,
            top_k=top_k,
            mode="or",
            minimum_match=1,
            expand=True,
            strategy="deep",
            use_cache=True,
        )

    # ========================================================
    # EXACT SEARCH
    # ========================================================

    def exact_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> SearchResponse:

        """
        Strongly constrained search.

        Useful when the user expects all important query
        terms to occur.
        """

        return self.search(
            query=query,
            top_k=top_k,
            mode="and",
            minimum_match=1,
            expand=False,
            strategy="exact",
            use_cache=True,
        )

    # ========================================================
    # AUTOCOMPLETE
    # ========================================================

    def autocomplete(
        self,
        prefix: str,
        limit: int = 10,
    ) -> List[str]:

        if not prefix:

            return []

        return self.index.prefix_terms(
            prefix,
            limit=limit,
        )

    # ========================================================
    # SUGGESTIONS
    # ========================================================

    def suggestions(
        self,
        query: str,
        limit: int = 10,
    ) -> List[str]:

        terms = self.tokenize_query(
            query
        )

        if not terms:

            return []

        suggestions = []

        for term in terms:

            suggestions.extend(
                self.index.prefix_terms(
                    term,
                    limit=limit,
                )
            )

            suggestions.extend(
                related
                for related, _
                in self.index.related_terms(
                    term,
                    limit=limit,
                )
            )

        # Preserve order while deduplicating.

        result = []

        seen = set()

        for suggestion in suggestions:

            if suggestion in seen:

                continue

            seen.add(
                suggestion
            )

            result.append(
                suggestion
            )

            if len(
                result
            ) >= limit:

                break

        return result

    # ========================================================
    # SIMILAR DOCUMENTS
    # ========================================================

    def similar(
        self,
        document_id: Any,
        limit: int = 10,
    ) -> List[
        SearchResult
    ]:

        candidates = (
            self.index.similar_candidates(
                document_id,
                limit=limit,
            )
        )

        results = []

        for rank, (
            candidate_id,
            score,
        ) in enumerate(
            candidates,
            start=1,
        ):

            record = self.index.get(
                candidate_id
            )

            if record is None:

                continue

            results.append(
                SearchResult(
                    document_id=candidate_id,
                    score=float(
                        score
                    ),
                    document=dict(
                        record.data
                    ),
                    rank=rank,
                    metadata={
                        "similar_to": (
                            document_id
                        ),
                    },
                )
            )

        return results

    # ========================================================
    # SEARCH HISTORY
    # ========================================================

    def _record_history(
        self,
        response: SearchResponse,
    ):

        self.history.append(
            {
                "query": response.query,
                "timestamp": self._now(),
                "execution_time_ms": (
                    response.execution_time_ms
                ),
                "results": (
                    response.returned_results
                ),
                "candidates": (
                    response.total_candidates
                ),
                "cache_hit": (
                    response.cache_hit
                ),
            }
        )

        if len(
            self.history
        ) > self.max_history:

            del self.history[
                :len(
                    self.history
                )
                - self.max_history
            ]

    def get_history(
        self,
        limit: int = 50,
    ) -> List[
        Dict[str, Any]
    ]:

        return list(
            self.history[
                -max(
                    0,
                    int(limit),
                ):
            ]
        )

    def clear_history(
        self,
    ):

        self.history.clear()

    # ========================================================
    # CACHE MANAGEMENT
    # ========================================================

    def clear_cache(
        self,
    ):

        self.cache.clear()

    # ========================================================
    # INDEX INVALIDATION
    # ========================================================

    def invalidate(
        self,
    ):

        """
        Clear cached searches after external index changes.
        """

        self.cache.clear()

    # ========================================================
    # DOCUMENT MANAGEMENT
    # ========================================================

    def add_document(
        self,
        document: Any,
    ):

        result = self.index.add(
            document
        )

        self.invalidate()

        return result

    def update_document(
        self,
        document_id: Any,
        document: Any,
    ):

        result = self.index.update(
            document_id,
            document,
        )

        self.invalidate()

        return result

    def remove_document(
        self,
        document_id: Any,
    ):

        result = self.index.remove(
            document_id
        )

        self.invalidate()

        return result

    # ========================================================
    # BULK MANAGEMENT
    # ========================================================

    def index_documents(
        self,
        documents: Iterable[Any],
    ) -> int:

        count = self.index.add_many(
            documents
        )

        self.invalidate()

        return count

    # ========================================================
    # QUERY ANALYSIS
    # ========================================================

    def analyze_query(
        self,
        query: str,
    ) -> Dict[str, Any]:

        plan = self.build_query_plan(
            query
        )

        return {
            "plan": plan.to_dict(),
            "term_analysis": (
                self.index.analyze_terms(
                    plan.terms
                )
            ),
            "suggestions": (
                self.suggestions(
                    query
                )
            ),
        }

    # ========================================================
    # BENCHMARK
    # ========================================================

    def benchmark(
        self,
        queries: Sequence[str],
    ) -> Dict[str, Any]:

        timings = []

        for query in queries:

            started = (
                time.perf_counter()
            )

            self.search(
                query,
                use_cache=False,
            )

            elapsed = (
                time.perf_counter()
                - started
            ) * 1000.0

            timings.append(
                elapsed
            )

        if not timings:

            return {
                "queries": 0,
                "average_ms": 0.0,
                "minimum_ms": 0.0,
                "maximum_ms": 0.0,
            }

        return {
            "queries": len(
                timings
            ),
            "average_ms": (
                sum(timings)
                / len(timings)
            ),
            "minimum_ms": min(
                timings
            ),
            "maximum_ms": max(
                timings
            ),
            "timings_ms": timings,
        }

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:

        index_health = (
            self.index.validate()
        )

        ranking_available = (
            self.ranking is not None
        )

        tokenizer_available = (
            self.tokenizer is not None
        )

        healthy = (
            index_health["valid"]
            and tokenizer_available
        )

        return {
            "healthy": healthy,
            "engine_version": (
                ENGINE_VERSION
            ),
            "tokenizer": (
                tokenizer_available
            ),
            "index": index_health,
            "ranking_engine": (
                ranking_available
            ),
            "cache_size": (
                self.cache.size()
            ),
        }

    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    def debug_info(
        self,
    ) -> Dict[str, Any]:

        return {
            "engine_version": (
                ENGINE_VERSION
            ),
            "statistics": (
                self.statistics.to_dict()
            ),
            "cache_size": (
                self.cache.size()
            ),
            "history_size": (
                len(self.history)
            ),
            "max_candidates": (
                self.max_candidates
            ),
            "index_generation": (
                self.index.generation
            ),
            "index_statistics": (
                self.index.get_statistics()
            ),
            "health": (
                self.health_check()
            ),
        }


# ============================================================
# DEFAULT ENGINE
# ============================================================


engine = SearchEngine()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    **kwargs,
) -> SearchResponse:

    return engine.search(
        query,
        top_k=top_k,
        **kwargs,
    )


def fast_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> SearchResponse:

    return engine.fast_search(
        query,
        top_k=top_k,
    )


def deep_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> SearchResponse:

    return engine.deep_search(
        query,
        top_k=top_k,
    )


def exact_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> SearchResponse:

    return engine.exact_search(
        query,
        top_k=top_k,
    )


def autocomplete(
    prefix: str,
    limit: int = 10,
) -> List[str]:

    return engine.autocomplete(
        prefix,
        limit=limit,
    )


def suggestions(
    query: str,
    limit: int = 10,
) -> List[str]:

    return engine.suggestions(
        query,
        limit=limit,
    )


def similar(
    document_id: Any,
    limit: int = 10,
) -> List[SearchResult]:

    return engine.similar(
        document_id,
        limit=limit,
    )


def analyze_query(
    query: str,
) -> Dict[str, Any]:

    return engine.analyze_query(
        query
    )


def health_check() -> Dict[str, Any]:

    return engine.health_check()


def debug_info() -> Dict[str, Any]:

    return engine.debug_info()