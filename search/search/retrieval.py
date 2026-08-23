"""
search/retrieval.py

Advanced candidate retrieval layer.

Responsibilities
----------------
- Convert structured Query objects into candidate document IDs.
- Retrieve candidates from an index.
- Support multiple retrieval strategies.
- Handle terms, phrases, fields, wildcards and fuzzy terms.
- Apply required/prohibited clauses.
- Apply structured filters.
- Merge candidates from multiple strategies.
- Deduplicate results.
- Track retrieval statistics.
- Support configurable candidate limits.
- Provide retrieval explanations.
- Remain independent from final ranking.

Architecture
------------

    Query
      |
      v
    RetrievalEngine
      |
      +---- Exact retrieval
      +---- Phrase retrieval
      +---- Field retrieval
      +---- Prefix / wildcard retrieval
      +---- Fuzzy retrieval
      +---- Filter retrieval
      |
      v
    CandidateSet
      |
      v
    RankingEngine

The retrieval layer should answer:

    "Which documents are worth considering?"

It should NOT answer:

    "Which document is ultimately the best?"

That belongs to ranking.py.
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ============================================================
# ENUMERATIONS
# ============================================================


class RetrievalStrategy(str, Enum):
    """
    Available candidate-generation strategies.
    """

    EXACT = "exact"
    PHRASE = "phrase"
    FIELD = "field"
    PREFIX = "prefix"
    WILDCARD = "wildcard"
    FUZZY = "fuzzy"
    FILTER = "filter"
    FALLBACK = "fallback"


class MatchType(str, Enum):
    """
    Describes how a document entered the candidate set.
    """

    EXACT = "exact"
    PHRASE = "phrase"
    FIELD = "field"
    PREFIX = "prefix"
    WILDCARD = "wildcard"
    FUZZY = "fuzzy"
    FILTER = "filter"
    FALLBACK = "fallback"


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class RetrievalConfig:
    """
    Configuration for the retrieval engine.
    """

    maximum_candidates: int = 5000

    maximum_candidates_per_term: int = 2500

    minimum_fuzzy_term_length: int = 3

    maximum_fuzzy_distance: int = 2

    enable_exact: bool = True
    enable_phrases: bool = True
    enable_fields: bool = True
    enable_prefix: bool = True
    enable_wildcards: bool = True
    enable_fuzzy: bool = True
    enable_filters: bool = True

    fallback_to_all_documents: bool = False

    case_sensitive: bool = False

    merge_strategy_results: bool = True

    include_match_metadata: bool = True


DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()


# ============================================================
# CANDIDATE
# ============================================================


@dataclass
class Candidate:
    """
    A document returned by retrieval.

    Ranking.py can later use this object to calculate
    a much more sophisticated final relevance score.
    """

    document_id: Any

    retrieval_score: float = 0.0

    match_types: Set[str] = field(
        default_factory=set
    )

    matched_terms: Set[str] = field(
        default_factory=set
    )

    matched_fields: Set[str] = field(
        default_factory=set
    )

    matched_phrases: Set[str] = field(
        default_factory=set
    )

    matched_filters: Set[str] = field(
        default_factory=set
    )

    strategies: Set[str] = field(
        default_factory=set
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def add_match(
        self,
        match_type: MatchType,
        score: float = 1.0,
        term: Optional[str] = None,
        field_name: Optional[str] = None,
        phrase: Optional[str] = None,
        filter_name: Optional[str] = None,
    ) -> None:

        self.match_types.add(
            match_type.value
        )

        self.strategies.add(
            match_type.value
        )

        self.retrieval_score += score

        if term:
            self.matched_terms.add(term)

        if field_name:
            self.matched_fields.add(field_name)

        if phrase:
            self.matched_phrases.add(phrase)

        if filter_name:
            self.matched_filters.add(filter_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "retrieval_score": self.retrieval_score,
            "match_types": sorted(
                self.match_types
            ),
            "matched_terms": sorted(
                self.matched_terms
            ),
            "matched_fields": sorted(
                self.matched_fields
            ),
            "matched_phrases": sorted(
                self.matched_phrases
            ),
            "matched_filters": sorted(
                self.matched_filters
            ),
            "strategies": sorted(
                self.strategies
            ),
            "metadata": self.metadata,
        }


# ============================================================
# RETRIEVAL STATISTICS
# ============================================================


@dataclass
class RetrievalStats:
    """
    Diagnostics for a retrieval operation.
    """

    query_time_ms: float = 0.0

    candidates_before_deduplication: int = 0
    candidates_after_deduplication: int = 0

    exact_matches: int = 0
    phrase_matches: int = 0
    field_matches: int = 0
    prefix_matches: int = 0
    wildcard_matches: int = 0
    fuzzy_matches: int = 0
    filter_matches: int = 0

    required_removed: int = 0
    prohibited_removed: int = 0
    filtered_removed: int = 0

    strategies_used: List[str] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_time_ms": self.query_time_ms,
            "candidates_before_deduplication": (
                self.candidates_before_deduplication
            ),
            "candidates_after_deduplication": (
                self.candidates_after_deduplication
            ),
            "exact_matches": self.exact_matches,
            "phrase_matches": self.phrase_matches,
            "field_matches": self.field_matches,
            "prefix_matches": self.prefix_matches,
            "wildcard_matches": self.wildcard_matches,
            "fuzzy_matches": self.fuzzy_matches,
            "filter_matches": self.filter_matches,
            "required_removed": self.required_removed,
            "prohibited_removed": self.prohibited_removed,
            "filtered_removed": self.filtered_removed,
            "strategies_used": self.strategies_used,
            "warnings": self.warnings,
        }


# ============================================================
# RESULT CONTAINER
# ============================================================


@dataclass
class RetrievalResult:
    """
    Complete retrieval response.
    """

    candidates: List[Candidate] = field(
        default_factory=list
    )

    stats: RetrievalStats = field(
        default_factory=RetrievalStats
    )

    query: Any = None

    successful: bool = True

    errors: List[str] = field(
        default_factory=list
    )

    def document_ids(self) -> List[Any]:
        return [
            candidate.document_id
            for candidate in self.candidates
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "stats": self.stats.to_dict(),
            "successful": self.successful,
            "errors": self.errors,
        }


# ============================================================
# INDEX ADAPTER
# ============================================================


class IndexAdapter:
    """
    Compatibility layer between retrieval.py and whatever
    implementation exists inside index.py.

    The retrieval engine deliberately doesn't assume that the
    index has one exact API.

    It tries several conventional method names.

    This allows the index implementation to evolve without
    forcing the retrieval engine to be rewritten every time.
    """

    def __init__(self, index: Any):
        self.index = index

    # --------------------------------------------------------
    # GENERAL DOCUMENT ACCESS
    # --------------------------------------------------------

    def all_document_ids(self) -> List[Any]:

        methods = [
            "all_document_ids",
            "document_ids",
            "list_document_ids",
            "get_document_ids",
        ]

        for method_name in methods:

            method = getattr(
                self.index,
                method_name,
                None,
            )

            if callable(method):

                result = method()

                if result is None:
                    continue

                return list(result)

        documents = getattr(
            self.index,
            "documents",
            None,
        )

        if isinstance(
            documents,
            dict,
        ):
            return list(
                documents.keys()
            )

        return []

    # --------------------------------------------------------
    # TERM RETRIEVAL
    # --------------------------------------------------------

    def exact(
        self,
        term: str,
    ) -> Set[Any]:

        methods = [
            "search_term",
            "lookup_term",
            "get_postings",
            "postings",
            "exact_search",
            "find_term",
        ]

        for method_name in methods:

            method = getattr(
                self.index,
                method_name,
                None,
            )

            if callable(method):

                result = method(term)

                if result is None:
                    continue

                return self._extract_ids(
                    result
                )

        return set()

    # --------------------------------------------------------
    # FIELD RETRIEVAL
    # --------------------------------------------------------

    def field(
        self,
        field_name: str,
        term: str,
    ) -> Set[Any]:

        methods = [
            "search_field",
            "field_search",
            "lookup_field",
            "find_in_field",
        ]

        for method_name in methods:

            method = getattr(
                self.index,
                method_name,
                None,
            )

            if callable(method):

                result = method(
                    field_name,
                    term,
                )

                if result is None:
                    continue

                return self._extract_ids(
                    result
                )

        # Fallback to general exact lookup if
        # the index does not expose field search.
        return self.exact(term)

    # --------------------------------------------------------
    # PREFIX RETRIEVAL
    # --------------------------------------------------------

    def prefix(
        self,
        prefix: str,
    ) -> Set[Any]:

        methods = [
            "search_prefix",
            "prefix_search",
            "lookup_prefix",
            "find_prefix",
        ]

        for method_name in methods:

            method = getattr(
                self.index,
                method_name,
                None,
            )

            if callable(method):

                result = method(prefix)

                if result is None:
                    continue

                return self._extract_ids(
                    result
                )

        # Generic vocabulary fallback.
        vocabulary = self.vocabulary()

        matches = {
            term
            for term in vocabulary
            if term.startswith(prefix)
        }

        documents = set()

        for term in matches:
            documents.update(
                self.exact(term)
            )

        return documents

    # --------------------------------------------------------
    # WILDCARD RETRIEVAL
    # --------------------------------------------------------

    def wildcard(
        self,
        pattern: str,
    ) -> Set[Any]:

        methods = [
            "search_wildcard",
            "wildcard_search",
            "lookup_wildcard",
        ]

        for method_name in methods:

            method = getattr(
                self.index,
                method_name,
                None,
            )

            if callable(method):

                result = method(pattern)

                if result is None:
                    continue

                return self._extract_ids(
                    result
                )

        vocabulary = self.vocabulary()

        regex = self._wildcard_regex(
            pattern
        )

        documents = set()

        for term in vocabulary:

            if regex.fullmatch(term):
                documents.update(
                    self.exact(term)
                )

        return documents

    # --------------------------------------------------------
    # PHRASE RETRIEVAL
    # --------------------------------------------------------

    def phrase(
        self,
        phrase: str,
    ) -> Set[Any]:

        methods = [
            "search_phrase",
            "phrase_search",
            "lookup_phrase",
        ]

        for method_name in methods:

            method = getattr(
                self.index,
                method_name,
                None,
            )

            if callable(method):

                result = method(phrase)

                if result is None:
                    continue

                return self._extract_ids(
                    result
                )

        # Generic phrase fallback:
        # retrieve documents containing every term.
        terms = self._simple_terms(
            phrase
        )

        if not terms:
            return set()

        result = None

        for term in terms:

            current = self.exact(term)

            if result is None:
                result = current
            else:
                result &= current

        return result or set()

    # --------------------------------------------------------
    # FUZZY RETRIEVAL
    # --------------------------------------------------------

    def fuzzy(
        self,
        term: str,
        maximum_distance: int = 2,
    ) -> Set[Any]:

        methods = [
            "search_fuzzy",
            "fuzzy_search",
            "lookup_fuzzy",
        ]

        for method_name in methods:

            method = getattr(
                self.index,
                method_name,
                None,
            )

            if callable(method):

                result = method(
                    term,
                    maximum_distance,
                )

                if result is None:
                    continue

                return self._extract_ids(
                    result
                )

        vocabulary = self.vocabulary()

        documents = set()

        for candidate_term in vocabulary:

            distance = levenshtein_distance(
                term,
                candidate_term,
            )

            if distance <= maximum_distance:

                documents.update(
                    self.exact(
                        candidate_term
                    )
                )

        return documents

    # --------------------------------------------------------
    # DOCUMENT ACCESS
    # --------------------------------------------------------

    def get_document(
        self,
        document_id: Any,
    ) -> Any:

        methods = [
            "get_document",
            "document",
            "fetch",
            "lookup_document",
        ]

        for method_name in methods:

            method = getattr(
                self.index,
                method_name,
                None,
            )

            if callable(method):

                try:
                    return method(
                        document_id
                    )
                except Exception:
                    continue

        documents = getattr(
            self.index,
            "documents",
            None,
        )

        if isinstance(
            documents,
            dict,
        ):
            return documents.get(
                document_id
            )

        return None

    # --------------------------------------------------------
    # VOCABULARY
    # --------------------------------------------------------

    def vocabulary(self) -> Set[str]:

        methods = [
            "vocabulary",
            "terms",
            "get_terms",
            "list_terms",
        ]

        for method_name in methods:

            method = getattr(
                self.index,
                method_name,
                None,
            )

            if callable(method):

                result = method()

                if result is not None:
                    return set(result)

        for attribute_name in [
            "index",
            "inverted_index",
            "terms_index",
        ]:

            value = getattr(
                self.index,
                attribute_name,
                None,
            )

            if isinstance(
                value,
                dict,
            ):
                return set(
                    value.keys()
                )

        return set()

    # --------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------

    @staticmethod
    def _extract_ids(
        result: Any,
    ) -> Set[Any]:

        if result is None:
            return set()

        if isinstance(
            result,
            dict,
        ):
            return set(
                result.keys()
            )

        if isinstance(
            result,
            (list, tuple, set),
        ):

            output = set()

            for item in result:

                if isinstance(
                    item,
                    dict,
                ):

                    if "document_id" in item:
                        output.add(
                            item["document_id"]
                        )

                    elif "id" in item:
                        output.add(
                            item["id"]
                        )

                else:
                    output.add(item)

            return output

        return {result}

    @staticmethod
    def _simple_terms(
        phrase: str,
    ) -> List[str]:

        return re.findall(
            r"\b[\w'-]+\b",
            phrase.lower(),
        )

    @staticmethod
    def _wildcard_regex(
        pattern: str,
    ) -> re.Pattern:

        escaped = re.escape(
            pattern
        )

        escaped = escaped.replace(
            r"\*",
            ".*",
        )

        escaped = escaped.replace(
            r"\?",
            ".",
        )

        return re.compile(
            "^" + escaped + "$",
            re.IGNORECASE,
        )


# ============================================================
# RETRIEVAL ENGINE
# ============================================================


class RetrievalEngine:
    """
    Main candidate retrieval engine.
    """

    def __init__(
        self,
        index: Any,
        config: Optional[
            RetrievalConfig
        ] = None,
    ):

        self.config = (
            config
            or RetrievalConfig()
        )

        self.adapter = IndexAdapter(
            index
        )

    # --------------------------------------------------------
    # PUBLIC RETRIEVAL API
    # --------------------------------------------------------

    def retrieve(
        self,
        query: Any,
        limit: Optional[int] = None,
    ) -> RetrievalResult:

        start = time.perf_counter()

        result = RetrievalResult(
            query=query
        )

        stats = result.stats

        candidates: Dict[
            Any,
            Candidate
        ] = {}

        try:

            self._retrieve_clauses(
                query,
                candidates,
                stats,
            )

            stats.candidates_before_deduplication = (
                len(candidates)
            )

            self._apply_required_constraints(
                query,
                candidates,
                stats,
            )

            self._apply_prohibited_constraints(
                query,
                candidates,
                stats,
            )

            self._apply_filters(
                query,
                candidates,
                stats,
            )

            if (
                not candidates
                and self.config.fallback_to_all_documents
            ):
                self._fallback(
                    candidates,
                    stats,
                )

            ordered = sorted(
                candidates.values(),
                key=lambda item: (
                    -item.retrieval_score,
                    str(item.document_id),
                ),
            )

            maximum = (
                limit
                or self.config.maximum_candidates
            )

            maximum = max(
                1,
                min(
                    maximum,
                    self.config.maximum_candidates,
                ),
            )

            result.candidates = ordered[
                :maximum
            ]

            stats.candidates_after_deduplication = (
                len(result.candidates)
            )

        except Exception as error:

            result.successful = False

            result.errors.append(
                str(error)
            )

            stats.warnings.append(
                "Retrieval failed."
            )

        finally:

            stats.query_time_ms = (
                time.perf_counter()
                - start
            ) * 1000

        return result

    # --------------------------------------------------------
    # CLAUSE RETRIEVAL
    # --------------------------------------------------------

    def _retrieve_clauses(
        self,
        query: Any,
        candidates: Dict[Any, Candidate],
        stats: RetrievalStats,
    ) -> None:

        clauses = getattr(
            query,
            "clauses",
            [],
        )

        for clause in clauses:

            value = getattr(
                clause,
                "value",
                "",
            )

            if not value:
                continue

            field_name = getattr(
                clause,
                "field",
                None,
            )

            exact = getattr(
                clause,
                "exact",
                False,
            )

            wildcard = getattr(
                clause,
                "wildcard",
                False,
            )

            fuzzy = getattr(
                clause,
                "fuzzy",
                False,
            )

            if exact and self.config.enable_phrases:

                self._retrieve_phrase(
                    value,
                    candidates,
                    stats,
                )

                continue

            if field_name and self.config.enable_fields:

                self._retrieve_field(
                    field_name,
                    value,
                    candidates,
                    stats,
                )

                continue

            if wildcard and self.config.enable_wildcards:

                self._retrieve_wildcard(
                    value,
                    candidates,
                    stats,
                )

                continue

            if fuzzy and self.config.enable_fuzzy:

                self._retrieve_fuzzy(
                    value,
                    candidates,
                    stats,
                )

                continue

            if (
                self.config.enable_prefix
                and value.endswith("*")
                and not value.startswith("*")
            ):

                prefix = value[:-1]

                self._retrieve_prefix(
                    prefix,
                    candidates,
                    stats,
                )

                continue

            if self.config.enable_exact:

                self._retrieve_exact(
                    value,
                    candidates,
                    stats,
                )

    # --------------------------------------------------------
    # EXACT
    # --------------------------------------------------------

    def _retrieve_exact(
        self,
        term: str,
        candidates: Dict[Any, Candidate],
        stats: RetrievalStats,
    ) -> None:

        normalized = self._normalize(
            term
        )

        document_ids = self.adapter.exact(
            normalized
        )

        if document_ids:

            self._record_strategy(
                stats,
                RetrievalStrategy.EXACT,
            )

        for document_id in document_ids:

            candidate = self._get_candidate(
                candidates,
                document_id,
            )

            candidate.add_match(
                MatchType.EXACT,
                score=3.0,
                term=term,
            )

        stats.exact_matches += len(
            document_ids
        )

    # --------------------------------------------------------
    # PHRASES
    # --------------------------------------------------------

    def _retrieve_phrase(
        self,
        phrase: str,
        candidates: Dict[Any, Candidate],
        stats: RetrievalStats,
    ) -> None:

        document_ids = self.adapter.phrase(
            phrase
        )

        if document_ids:

            self._record_strategy(
                stats,
                RetrievalStrategy.PHRASE,
            )

        for document_id in document_ids:

            candidate = self._get_candidate(
                candidates,
                document_id,
            )

            candidate.add_match(
                MatchType.PHRASE,
                score=6.0,
                phrase=phrase,
            )

        stats.phrase_matches += len(
            document_ids
        )

    # --------------------------------------------------------
    # FIELD
    # --------------------------------------------------------

    def _retrieve_field(
        self,
        field_name: str,
        term: str,
        candidates: Dict[Any, Candidate],
        stats: RetrievalStats,
    ) -> None:

        document_ids = self.adapter.field(
            field_name,
            self._normalize(term),
        )

        if document_ids:

            self._record_strategy(
                stats,
                RetrievalStrategy.FIELD,
            )

        for document_id in document_ids:

            candidate = self._get_candidate(
                candidates,
                document_id,
            )

            candidate.add_match(
                MatchType.FIELD,
                score=5.0,
                term=term,
                field_name=field_name,
            )

        stats.field_matches += len(
            document_ids
        )

    # --------------------------------------------------------
    # PREFIX
    # --------------------------------------------------------

    def _retrieve_prefix(
        self,
        prefix: str,
        candidates: Dict[Any, Candidate],
        stats: RetrievalStats,
    ) -> None:

        document_ids = self.adapter.prefix(
            self._normalize(prefix)
        )

        if document_ids:

            self._record_strategy(
                stats,
                RetrievalStrategy.PREFIX,
            )

        for document_id in document_ids:

            candidate = self._get_candidate(
                candidates,
                document_id,
            )

            candidate.add_match(
                MatchType.PREFIX,
                score=2.0,
                term=prefix,
            )

        stats.prefix_matches += len(
            document_ids
        )

    # --------------------------------------------------------
    # WILDCARD
    # --------------------------------------------------------

    def _retrieve_wildcard(
        self,
        pattern: str,
        candidates: Dict[Any, Candidate],
        stats: RetrievalStats,
    ) -> None:

        document_ids = self.adapter.wildcard(
            self._normalize(pattern)
        )

        if document_ids:

            self._record_strategy(
                stats,
                RetrievalStrategy.WILDCARD,
            )

        for document_id in document_ids:

            candidate = self._get_candidate(
                candidates,
                document_id,
            )

            candidate.add_match(
                MatchType.WILDCARD,
                score=1.5,
                term=pattern,
            )

        stats.wildcard_matches += len(
            document_ids
        )

    # --------------------------------------------------------
    # FUZZY
    # --------------------------------------------------------

    def _retrieve_fuzzy(
        self,
        term: str,
        candidates: Dict[Any, Candidate],
        stats: RetrievalStats,
    ) -> None:

        normalized = self._normalize(
            term
        )

        if (
            len(normalized)
            < self.config.minimum_fuzzy_term_length
        ):
            stats.warnings.append(
                f"Fuzzy term '{term}' is too short."
            )
            return

        document_ids = self.adapter.fuzzy(
            normalized,
            self.config.maximum_fuzzy_distance,
        )

        if document_ids:

            self._record_strategy(
                stats,
                RetrievalStrategy.FUZZY,
            )

        for document_id in document_ids:

            candidate = self._get_candidate(
                candidates,
                document_id,
            )

            candidate.add_match(
                MatchType.FUZZY,
                score=1.0,
                term=term,
            )

        stats.fuzzy_matches += len(
            document_ids
        )

    # ========================================================
    # REQUIRED / PROHIBITED LOGIC
    # ========================================================

    def _apply_required_constraints(
        self,
        query: Any,
        candidates: Dict[Any, Candidate],
        stats: RetrievalStats,
    ) -> None:

        required = [
            clause
            for clause in getattr(
                query,
                "clauses",
                [],
            )
            if getattr(
                clause,
                "required",
                False,
            )
        ]

        if not required:
            return

        required_sets = []

        for clause in required:

            ids = self._retrieve_clause_ids(
                clause
            )

            required_sets.append(
                ids
            )

        if not required_sets:
            return

        valid_ids = set.intersection(
            *required_sets
        )

        for document_id in list(
            candidates.keys()
        ):

            if document_id not in valid_ids:

                del candidates[
                    document_id
                ]

                stats.required_removed += 1

    def _apply_prohibited_constraints(
        self,
        query: Any,
        candidates: Dict[Any, Candidate],
        stats: RetrievalStats,
    ) -> None:

        prohibited = [
            clause
            for clause in getattr(
                query,
                "clauses",
                [],
            )
            if getattr(
                clause,
                "prohibited",
                False,
            )
        ]

        if not prohibited:
            return

        prohibited_ids = set()

        for clause in prohibited:

            prohibited_ids.update(
                self._retrieve_clause_ids(
                    clause
                )
            )

        for document_id in list(
            candidates.keys()
        ):

            if document_id in prohibited_ids:

                del candidates[
                    document_id
                ]

                stats.prohibited_removed += 1

    # --------------------------------------------------------
    # CLAUSE ID RETRIEVAL
    # --------------------------------------------------------

    def _retrieve_clause_ids(
        self,
        clause: Any,
    ) -> Set[Any]:

        value = getattr(
            clause,
            "value",
            "",
        )

        if not value:
            return set()

        field_name = getattr(
            clause,
            "field",
            None,
        )

        exact = getattr(
            clause,
            "exact",
            False,
        )

        wildcard = getattr(
            clause,
            "wildcard",
            False,
        )

        fuzzy = getattr(
            clause,
            "fuzzy",
            False,
        )

        if exact:
            return self.adapter.phrase(
                value
            )

        if field_name:
            return self.adapter.field(
                field_name,
                value,
            )

        if wildcard:
            return self.adapter.wildcard(
                value
            )

        if fuzzy:
            return self.adapter.fuzzy(
                value,
                self.config.maximum_fuzzy_distance,
            )

        return self.adapter.exact(
            self._normalize(value)
        )

    # ========================================================
    # FILTERS
    # ========================================================

    def _apply_filters(
        self,
        query: Any,
        candidates: Dict[Any, Candidate],
        stats: RetrievalStats,
    ) -> None:

        if not self.config.enable_filters:
            return

        filters = getattr(
            query,
            "filters",
            [],
        )

        if not filters:
            return

        for document_id in list(
            candidates.keys()
        ):

            document = self.adapter.get_document(
                document_id
            )

            if document is None:
                continue

            valid = True

            for query_filter in filters:

                if not self._document_matches_filter(
                    document,
                    query_filter,
                ):

                    valid = False

                    stats.filtered_removed += 1

                    break

                candidate = candidates[
                    document_id
                ]

                candidate.add_match(
                    MatchType.FILTER,
                    score=0.5,
                    filter_name=(
                        getattr(
                            query_filter,
                            "field",
                            "unknown",
                        )
                    ),
                )

                stats.filter_matches += 1

            if not valid:

                del candidates[
                    document_id
                ]

    def _document_matches_filter(
        self,
        document: Any,
        query_filter: Any,
    ) -> bool:

        field_name = getattr(
            query_filter,
            "field",
            "",
        )

        operator = getattr(
            query_filter,
            "operator",
            "=",
        )

        expected = getattr(
            query_filter,
            "value",
            None,
        )

        actual = self._document_value(
            document,
            field_name,
        )

        if operator == "=":
            return self._compare_equal(
                actual,
                expected,
            )

        if operator == "!=":
            return not self._compare_equal(
                actual,
                expected,
            )

        if operator == ">":
            return self._safe_compare(
                actual,
                expected,
                lambda a, b: a > b,
            )

        if operator == ">=":
            return self._safe_compare(
                actual,
                expected,
                lambda a, b: a >= b,
            )

        if operator == "<":
            return self._safe_compare(
                actual,
                expected,
                lambda a, b: a < b,
            )

        if operator == "<=":
            return self._safe_compare(
                actual,
                expected,
                lambda a, b: a <= b,
            )

        return False

    @staticmethod
    def _document_value(
        document: Any,
        field_name: str,
    ) -> Any:

        if isinstance(
            document,
            dict,
        ):
            return document.get(
                field_name
            )

        return getattr(
            document,
            field_name,
            None,
        )

    @staticmethod
    def _compare_equal(
        actual: Any,
        expected: Any,
    ) -> bool:

        if isinstance(
            actual,
            (list, tuple, set),
        ):

            return any(
                RetrievalEngine._compare_equal(
                    value,
                    expected,
                )
                for value in actual
            )

        if isinstance(
            actual,
            str,
        ) and isinstance(
            expected,
            str,
        ):

            return (
                actual.lower()
                == expected.lower()
            )

        return actual == expected

    @staticmethod
    def _safe_compare(
        actual: Any,
        expected: Any,
        operation,
    ) -> bool:

        if actual is None:
            return False

        try:
            return operation(
                actual,
                expected,
            )
        except (
            TypeError,
            ValueError,
        ):
            return False

    # ========================================================
    # FALLBACK
    # ========================================================

    def _fallback(
        self,
        candidates: Dict[Any, Candidate],
        stats: RetrievalStats,
    ) -> None:

        document_ids = (
            self.adapter.all_document_ids()
        )

        self._record_strategy(
            stats,
            RetrievalStrategy.FALLBACK,
        )

        for document_id in document_ids:

            if len(candidates) >= (
                self.config.maximum_candidates
            ):
                break

            candidate = self._get_candidate(
                candidates,
                document_id,
            )

            candidate.add_match(
                MatchType.FALLBACK,
                score=0.01,
            )

    # ========================================================
    # CANDIDATE MANAGEMENT
    # ========================================================

    @staticmethod
    def _get_candidate(
        candidates: Dict[Any, Candidate],
        document_id: Any,
    ) -> Candidate:

        if document_id not in candidates:

            candidates[
                document_id
            ] = Candidate(
                document_id=document_id
            )

        return candidates[
            document_id
        ]

    @staticmethod
    def _record_strategy(
        stats: RetrievalStats,
        strategy: RetrievalStrategy,
    ) -> None:

        if strategy.value not in (
            stats.strategies_used
        ):

            stats.strategies_used.append(
                strategy.value
            )

    def _normalize(
        self,
        value: str,
    ) -> str:

        value = str(value).strip()

        if not self.config.case_sensitive:
            value = value.lower()

        return value


# ============================================================
# RETRIEVAL HELPERS
# ============================================================


def levenshtein_distance(
    left: str,
    right: str,
) -> int:
    """
    Calculate edit distance between two strings.

    Used as a fallback for fuzzy retrieval when the index
    doesn't provide native fuzzy search.
    """

    if left == right:
        return 0

    if not left:
        return len(right)

    if not right:
        return len(left)

    previous = list(
        range(
            len(right) + 1
        )
    )

    for i, left_char in enumerate(
        left,
        start=1,
    ):

        current = [
            i
        ]

        for j, right_char in enumerate(
            right,
            start=1,
        ):

            insertion = (
                current[j - 1] + 1
            )

            deletion = (
                previous[j] + 1
            )

            substitution = (
                previous[j - 1]
                + (
                    left_char != right_char
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


def retrieve(
    index: Any,
    query: Any,
    limit: Optional[int] = None,
    config: Optional[
        RetrievalConfig
    ] = None,
) -> RetrievalResult:
    """
    Convenience retrieval function.
    """

    engine = RetrievalEngine(
        index,
        config=config,
    )

    return engine.retrieve(
        query,
        limit=limit,
    )


def candidate_ids(
    result: RetrievalResult,
) -> List[Any]:
    """
    Extract document IDs from a retrieval result.
    """

    return result.document_ids()


def explain_candidate(
    candidate: Candidate,
) -> Dict[str, Any]:
    """
    Explain why a candidate entered the result set.
    """

    return {
        "document_id": candidate.document_id,
        "retrieval_score": candidate.retrieval_score,
        "match_types": sorted(
            candidate.match_types
        ),
        "matched_terms": sorted(
            candidate.matched_terms
        ),
        "matched_fields": sorted(
            candidate.matched_fields
        ),
        "matched_phrases": sorted(
            candidate.matched_phrases
        ),
        "matched_filters": sorted(
            candidate.matched_filters
        ),
        "strategies": sorted(
            candidate.strategies
        ),
    }


# ============================================================
# BATCH RETRIEVAL
# ============================================================


class BatchRetriever:
    """
    Execute multiple queries through the same retrieval engine.

    Reusing the engine avoids repeatedly constructing adapters
    and allows future caching layers to sit above this class.
    """

    def __init__(
        self,
        engine: RetrievalEngine,
    ):
        self.engine = engine

    def retrieve_many(
        self,
        queries: Iterable[Any],
        limit: Optional[int] = None,
    ) -> List[RetrievalResult]:

        results = []

        for query in queries:

            results.append(
                self.engine.retrieve(
                    query,
                    limit=limit,
                )
            )

        return results


# ============================================================
# RETRIEVAL MERGING
# ============================================================


def merge_retrieval_results(
    results: Sequence[
        RetrievalResult
    ],
    maximum_candidates: int = 5000,
) -> RetrievalResult:
    """
    Merge results from independent retrieval engines
    or retrieval passes.
    """

    merged = RetrievalResult()

    candidates: Dict[
        Any,
        Candidate
    ] = {}

    for result in results:

        if result.query is not None:
            merged.query = result.query

        for candidate in result.candidates:

            existing = candidates.get(
                candidate.document_id
            )

            if existing is None:

                existing = Candidate(
                    document_id=(
                        candidate.document_id
                    )
                )

                candidates[
                    candidate.document_id
                ] = existing

            existing.retrieval_score += (
                candidate.retrieval_score
            )

            existing.match_types.update(
                candidate.match_types
            )

            existing.matched_terms.update(
                candidate.matched_terms
            )

            existing.matched_fields.update(
                candidate.matched_fields
            )

            existing.matched_phrases.update(
                candidate.matched_phrases
            )

            existing.matched_filters.update(
                candidate.matched_filters
            )

            existing.strategies.update(
                candidate.strategies
            )

            existing.metadata.update(
                candidate.metadata
            )

    merged.candidates = sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.retrieval_score,
            str(candidate.document_id),
        ),
    )[
        :maximum_candidates
    ]

    merged.stats.candidates_after_deduplication = (
        len(merged.candidates)
    )

    for result in results:

        merged.stats.query_time_ms += (
            result.stats.query_time_ms
        )

        merged.stats.strategies_used.extend(
            result.stats.strategies_used
        )

    merged.stats.strategies_used = sorted(
        set(
            merged.stats.strategies_used
        )
    )

    return merged


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================


retrieval_config = RetrievalConfig()


# ============================================================
# MODULE EXPORTS
# ============================================================


__all__ = [
    "RetrievalStrategy",
    "MatchType",
    "RetrievalConfig",
    "Candidate",
    "RetrievalStats",
    "RetrievalResult",
    "IndexAdapter",
    "RetrievalEngine",
    "BatchRetriever",
    "levenshtein_distance",
    "retrieve",
    "candidate_ids",
    "explain_candidate",
    "merge_retrieval_results",
    "retrieval_config",
]