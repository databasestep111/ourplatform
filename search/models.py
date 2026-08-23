"""
OurPlatform Search Models

Core data structures used by the search subsystem.

This module intentionally contains data models only.
Search algorithms, indexing, ranking, caching, and storage
belong in their respective modules.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ============================================================
# DOCUMENT
# ============================================================

@dataclass
class SearchDocument:
    """
    A searchable document.

    A document can represent almost anything:

        - Memory
        - Research
        - Note
        - Web content
        - System information
        - Assistant-generated content
    """

    id: str

    content: str

    title: str = ""

    category: str = "general"

    tags: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=datetime.now
    )

    updated_at: datetime = field(
        default_factory=datetime.now
    )

    indexed_at: Optional[datetime] = None

    enabled: bool = True

    version: int = 1

    source: Optional[str] = None

    source_type: Optional[str] = None

    language: Optional[str] = None

    importance: float = 1.0

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    def normalized_content(self):
        return self.content.lower().strip()

    def normalized_title(self):
        return self.title.lower().strip()

    # --------------------------------------------------------
    # TAG MANAGEMENT
    # --------------------------------------------------------

    def add_tag(
        self,
        tag: str,
    ):

        tag = str(tag).strip()

        if (
            tag
            and tag not in self.tags
        ):

            self.tags.append(tag)

    def remove_tag(
        self,
        tag: str,
    ):

        if tag in self.tags:

            self.tags.remove(tag)

            return True

        return False

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    def set_metadata(
        self,
        key: str,
        value: Any,
    ):

        self.metadata[
            key
        ] = value

    def get_metadata(
        self,
        key: str,
        default=None,
    ):

        return self.metadata.get(
            key,
            default,
        )

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self):

        return {
            "id": self.id,
            "content": self.content,
            "title": self.title,
            "category": self.category,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "created_at": (
                self.created_at.isoformat()
            ),
            "updated_at": (
                self.updated_at.isoformat()
            ),
            "indexed_at": (
                self.indexed_at.isoformat()
                if self.indexed_at
                else None
            ),
            "enabled": self.enabled,
            "version": self.version,
            "source": self.source,
            "source_type": self.source_type,
            "language": self.language,
            "importance": self.importance,
        }


# ============================================================
# SEARCH QUERY
# ============================================================

@dataclass
class SearchQuery:
    """
    Structured search request.
    """

    text: str

    category: Optional[str] = None

    tags: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    limit: int = 10

    offset: int = 0

    min_score: float = 0.0

    include_disabled: bool = False

    include_archived: bool = False

    sort_by: str = "relevance"

    descending: bool = True

    fuzzy: bool = True

    exact: bool = False

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    def normalized(self):

        return self.text.lower().strip()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def validate(self):

        if not isinstance(
            self.text,
            str,
        ):

            raise TypeError(
                "Search query text "
                "must be a string."
            )

        if self.limit < 1:

            raise ValueError(
                "Search limit must "
                "be greater than zero."
            )

        if self.offset < 0:

            raise ValueError(
                "Search offset cannot "
                "be negative."
            )

        if self.min_score < 0:

            raise ValueError(
                "Minimum score cannot "
                "be negative."
            )

        return True


# ============================================================
# SEARCH RESULT
# ============================================================

@dataclass
class SearchResult:
    """
    A single ranked search result.
    """

    document: SearchDocument

    score: float = 0.0

    relevance: float = 0.0

    title_score: float = 0.0

    content_score: float = 0.0

    tag_score: float = 0.0

    category_score: float = 0.0

    metadata_score: float = 0.0

    importance_score: float = 0.0

    freshness_score: float = 0.0

    exact_match: bool = False

    matched_terms: List[str] = field(
        default_factory=list
    )

    highlights: List[str] = field(
        default_factory=list
    )

    rank: int = 0

    # --------------------------------------------------------
    # RESULT SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self):

        return {
            "id": self.document.id,
            "title": self.document.title,
            "content": self.document.content,
            "category": self.document.category,
            "tags": list(
                self.document.tags
            ),
            "score": self.score,
            "relevance": self.relevance,
            "title_score": self.title_score,
            "content_score": self.content_score,
            "tag_score": self.tag_score,
            "category_score": (
                self.category_score
            ),
            "metadata_score": (
                self.metadata_score
            ),
            "importance_score": (
                self.importance_score
            ),
            "freshness_score": (
                self.freshness_score
            ),
            "exact_match": self.exact_match,
            "matched_terms": list(
                self.matched_terms
            ),
            "highlights": list(
                self.highlights
            ),
            "rank": self.rank,
        }


# ============================================================
# SEARCH RESPONSE
# ============================================================

@dataclass
class SearchResponse:
    """
    Complete search operation response.
    """

    query: SearchQuery

    results: List[SearchResult] = field(
        default_factory=list
    )

    total: int = 0

    execution_time: float = 0.0

    from_cache: bool = False

    index_version: int = 1

    suggestions: List[str] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # RESULT MANAGEMENT
    # --------------------------------------------------------

    def add_result(
        self,
        result: SearchResult,
    ):

        self.results.append(
            result
        )

        self.total = len(
            self.results
        )

    def sort_results(
        self,
    ):

        self.results.sort(
            key=lambda result:
                result.score,
            reverse=True,
        )

        for index, result in enumerate(
            self.results,
            start=1,
        ):

            result.rank = index

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self):

        return {
            "query": self.query.text,
            "total": self.total,
            "execution_time": (
                self.execution_time
            ),
            "from_cache": self.from_cache,
            "index_version": (
                self.index_version
            ),
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "suggestions": list(
                self.suggestions
            ),
            "warnings": list(
                self.warnings
            ),
        }


# ============================================================
# INDEX ENTRY
# ============================================================

@dataclass
class IndexEntry:
    """
    Represents a term inside the search index.
    """

    term: str

    document_ids: List[str] = field(
        default_factory=list
    )

    frequencies: Dict[str, int] = field(
        default_factory=dict
    )

    positions: Dict[
        str,
        List[int]
    ] = field(
        default_factory=dict
    )

    document_frequency: int = 0

    # --------------------------------------------------------
    # DOCUMENT MANAGEMENT
    # --------------------------------------------------------

    def add_document(
        self,
        document_id: str,
        frequency: int = 1,
        position: Optional[int] = None,
    ):

        if document_id not in (
            self.document_ids
        ):

            self.document_ids.append(
                document_id
            )

        self.frequencies[
            document_id
        ] = frequency

        if position is not None:

            self.positions.setdefault(
                document_id,
                [],
            ).append(
                position
            )

        self.document_frequency = len(
            self.document_ids
        )

    def remove_document(
        self,
        document_id: str,
    ):

        if document_id in (
            self.document_ids
        ):

            self.document_ids.remove(
                document_id
            )

        self.frequencies.pop(
            document_id,
            None,
        )

        self.positions.pop(
            document_id,
            None,
        )

        self.document_frequency = len(
            self.document_ids
        )


# ============================================================
# SEARCH STATISTICS
# ============================================================

@dataclass
class SearchStatistics:
    """
    Runtime statistics for the search subsystem.
    """

    total_searches: int = 0

    successful_searches: int = 0

    empty_searches: int = 0

    failed_searches: int = 0

    cached_searches: int = 0

    indexed_documents: int = 0

    indexed_terms: int = 0

    total_results: int = 0

    average_results: float = 0.0

    average_execution_time: float = 0.0

    fastest_search: Optional[float] = None

    slowest_search: Optional[float] = None

    # --------------------------------------------------------
    # RECORD SEARCH
    # --------------------------------------------------------

    def record_search(
        self,
        result_count: int,
        execution_time: float,
        cached: bool = False,
    ):

        self.total_searches += 1

        self.total_results += (
            result_count
        )

        if result_count:

            self.successful_searches += 1

        else:

            self.empty_searches += 1

        if cached:

            self.cached_searches += 1

        self.average_results = (
            self.total_results
            / self.total_searches
        )

        previous_average = (
            self.average_execution_time
        )

        self.average_execution_time = (
            (
                previous_average
                * (
                    self.total_searches
                    - 1
                )
            )
            + execution_time
        ) / self.total_searches

        if (
            self.fastest_search
            is None
            or execution_time
            < self.fastest_search
        ):

            self.fastest_search = (
                execution_time
            )

        if (
            self.slowest_search
            is None
            or execution_time
            > self.slowest_search
        ):

            self.slowest_search = (
                execution_time
            )

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self):

        return {
            "total_searches": (
                self.total_searches
            ),
            "successful_searches": (
                self.successful_searches
            ),
            "empty_searches": (
                self.empty_searches
            ),
            "failed_searches": (
                self.failed_searches
            ),
            "cached_searches": (
                self.cached_searches
            ),
            "indexed_documents": (
                self.indexed_documents
            ),
            "indexed_terms": (
                self.indexed_terms
            ),
            "total_results": (
                self.total_results
            ),
            "average_results": (
                self.average_results
            ),
            "average_execution_time": (
                self.average_execution_time
            ),
            "fastest_search": (
                self.fastest_search
            ),
            "slowest_search": (
                self.slowest_search
            ),
        }


# ============================================================
# INDEX STATISTICS
# ============================================================

@dataclass
class IndexStatistics:
    """
    Statistics describing the current search index.
    """

    documents: int = 0

    terms: int = 0

    categories: int = 0

    tags: int = 0

    enabled_documents: int = 0

    disabled_documents: int = 0

    average_document_length: float = 0.0

    largest_document: int = 0

    smallest_document: int = 0

    version: int = 1

    last_rebuild: Optional[
        datetime
    ] = None

    last_update: Optional[
        datetime
    ] = None

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def to_dict(self):

        return {
            "documents": self.documents,
            "terms": self.terms,
            "categories": self.categories,
            "tags": self.tags,
            "enabled_documents": (
                self.enabled_documents
            ),
            "disabled_documents": (
                self.disabled_documents
            ),
            "average_document_length": (
                self.average_document_length
            ),
            "largest_document": (
                self.largest_document
            ),
            "smallest_document": (
                self.smallest_document
            ),
            "version": self.version,
            "last_rebuild": (
                self.last_rebuild.isoformat()
                if self.last_rebuild
                else None
            ),
            "last_update": (
                self.last_update.isoformat()
                if self.last_update
                else None
            ),
        }


# ============================================================
# SEARCH CONFIGURATION
# ============================================================

@dataclass
class SearchConfig:
    """
    Central configuration for the search engine.

    Keeping configuration here means the search engine
    doesn't need hard-coded behaviour everywhere.
    """

    max_results: int = 100

    default_results: int = 10

    minimum_score: float = 0.0

    title_weight: float = 3.0

    content_weight: float = 1.0

    tag_weight: float = 2.0

    category_weight: float = 1.5

    metadata_weight: float = 1.0

    importance_weight: float = 1.0

    freshness_weight: float = 0.5

    exact_match_bonus: float = 2.0

    fuzzy_matching: bool = True

    case_sensitive: bool = False

    enable_cache: bool = True

    enable_suggestions: bool = True

    enable_highlights: bool = True

    cache_size: int = 500

    cache_ttl_seconds: int = 300

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def validate(self):

        numeric_fields = [
            "max_results",
            "default_results",
            "cache_size",
            "cache_ttl_seconds",
        ]

        for field_name in numeric_fields:

            value = getattr(
                self,
                field_name,
            )

            if value < 1:

                raise ValueError(
                    f"{field_name} "
                    f"must be greater "
                    f"than zero."
                )

        if (
            self.default_results
            > self.max_results
        ):

            raise ValueError(
                "default_results cannot "
                "exceed max_results."
            )

        return True