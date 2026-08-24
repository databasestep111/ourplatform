"""
OurPlatform API Schemas
=======================

Version:
    1.0.0

Purpose:
    Defines the request/response contract used by the OurPlatform API.

Architecture:

    Web Frontend
        |
        v
    search_api.js
        |
        v
    HTTP / JSON
        |
        v
    Api.routes
        |
        v
    Api.search_api
        |
        v
    Api.schemas
        |
        v
    Search subsystem

This module is intentionally focused on DATA CONTRACTS.

It does not perform:
    - searching
    - indexing
    - ranking
    - retrieval
    - database operations
    - HTML rendering
    - frontend logic

Those responsibilities belong to their respective layers.

The goal is to make API communication predictable, validated,
serializable, and extensible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


# ============================================================================
# VERSION INFORMATION
# ============================================================================

SCHEMA_VERSION = "1.0.0"

API_VERSION = "v1"

SCHEMA_NAME = "OurPlatform API Schema"

SCHEMA_COMPATIBILITY = "stable-v1"


# ============================================================================
# GENERIC TYPE ALIASES
# ============================================================================

JSONPrimitive = Union[
    str,
    int,
    float,
    bool,
    None,
]

JSONValue = Union[
    JSONPrimitive,
    List["JSONValue"],
    Dict[str, "JSONValue"],
]

JSONDict = Dict[str, JSONValue]


# ============================================================================
# ENUMERATIONS
# ============================================================================

class SearchMode(str, Enum):
    """
    Supported search modes.
    """

    STANDARD = "standard"
    ADVANCED = "advanced"
    RESEARCH = "research"
    MEMORY = "memory"
    SEMANTIC = "semantic"
    AI = "ai"


class SearchSort(str, Enum):
    """
    Supported result sorting strategies.
    """

    RELEVANCE = "relevance"
    DATE = "date"
    OLDEST = "oldest"
    TITLE = "title"


class MatchType(str, Enum):
    """
    Describes how a result matched a query.
    """

    EXACT = "exact"
    PHRASE = "phrase"
    TITLE = "title"
    CONTENT = "content"
    TAG = "tag"
    CATEGORY = "category"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    MIXED = "mixed"


class ResultType(str, Enum):
    """
    Broad categories of searchable information.
    """

    GENERAL = "general"
    MEMORY = "memory"
    RESEARCH = "research"
    DOCUMENT = "document"
    NOTE = "note"
    KNOWLEDGE = "knowledge"


class APIErrorCode(str, Enum):
    """
    Standard API error identifiers.
    """

    INVALID_REQUEST = "invalid_request"
    INVALID_QUERY = "invalid_query"
    INVALID_FILTER = "invalid_filter"
    INVALID_LIMIT = "invalid_limit"
    INVALID_OFFSET = "invalid_offset"
    INVALID_ID = "invalid_id"
    MISSING_QUERY = "missing_query"
    NOT_FOUND = "not_found"
    INTERNAL_ERROR = "internal_error"
    SEARCH_ERROR = "search_error"
    TIMEOUT = "timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"


# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================

def normalize_string(
    value: Any,
    default: str = "",
) -> str:
    """
    Convert a value to a clean string.
    """

    if value is None:
        return default

    return str(value).strip()


def normalize_optional_string(
    value: Any,
) -> Optional[str]:
    """
    Convert a value into either a clean string or None.
    """

    result = normalize_string(value)

    return result if result else None


def normalize_string_list(
    values: Any,
) -> List[str]:
    """
    Normalize a string/list input into a clean unique list.
    """

    if values is None:
        return []

    if isinstance(values, str):
        values = values.split(",")

    if not isinstance(values, Iterable):
        return []

    result: List[str] = []

    for value in values:
        item = normalize_string(value)

        if item and item not in result:
            result.append(item)

    return result


def normalize_int(
    value: Any,
    default: int,
) -> int:
    """
    Convert a value to an integer where possible.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_bool(
    value: Any,
    default: bool = False,
) -> bool:
    """
    Normalize common boolean representations.
    """

    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()

        if lowered in {
            "true",
            "1",
            "yes",
            "on",
        }:
            return True

        if lowered in {
            "false",
            "0",
            "no",
            "off",
        }:
            return False

    return bool(value)


def normalize_enum(
    value: Any,
    enum_type: type[Enum],
    default: Optional[Enum] = None,
) -> Optional[Enum]:
    """
    Safely convert a value into an Enum member.
    """

    if value is None:
        return default

    if isinstance(value, enum_type):
        return value

    try:
        return enum_type(str(value).lower())
    except ValueError:
        return default


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_query(
    query: str,
    *,
    max_length: int = 1000,
) -> List[str]:
    """
    Validate a search query and return validation errors.
    """

    errors: List[str] = []

    query = normalize_string(query)

    if not query:
        errors.append("Search query cannot be empty.")

    if len(query) > max_length:
        errors.append(
            f"Search query cannot exceed {max_length} characters."
        )

    return errors


def validate_limit(
    limit: int,
    *,
    minimum: int = 1,
    maximum: int = 500,
) -> List[str]:
    """
    Validate pagination/result limits.
    """

    errors: List[str] = []

    if limit < minimum:
        errors.append(
            f"Limit must be at least {minimum}."
        )

    if limit > maximum:
        errors.append(
            f"Limit cannot exceed {maximum}."
        )

    return errors


def validate_offset(
    offset: int,
) -> List[str]:
    """
    Validate pagination offset.
    """

    if offset < 0:
        return ["Offset cannot be negative."]

    return []


def validate_identifier(
    identifier: Any,
) -> List[str]:
    """
    Validate an item identifier.
    """

    errors: List[str] = []

    try:
        value = int(identifier)
    except (TypeError, ValueError):
        errors.append("Identifier must be an integer.")
        return errors

    if value < 1:
        errors.append("Identifier must be greater than zero.")

    return errors


# ============================================================================
# PAGINATION
# ============================================================================

@dataclass
class PaginationRequest:
    """
    Incoming pagination configuration.
    """

    limit: int = 10
    offset: int = 0

    def __post_init__(self) -> None:
        self.limit = normalize_int(
            self.limit,
            10,
        )

        self.offset = normalize_int(
            self.offset,
            0,
        )

    def validate(self) -> List[str]:
        errors = []

        errors.extend(
            validate_limit(self.limit)
        )

        errors.extend(
            validate_offset(self.offset)
        )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "limit": self.limit,
            "offset": self.offset,
        }


@dataclass
class PaginationResponse:
    """
    Pagination information returned by the API.
    """

    page: int = 1
    limit: int = 10
    offset: int = 0
    total: int = 0
    total_pages: int = 0
    has_previous: bool = False
    has_next: bool = False
    previous_offset: Optional[int] = None
    next_offset: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# SEARCH FILTERS
# ============================================================================

@dataclass
class SearchFilters:
    """
    Structured search filtering configuration.
    """

    category: Optional[str] = None
    tag: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    result_type: Optional[str] = None
    match_type: Optional[str] = None

    def __post_init__(self) -> None:
        self.category = normalize_optional_string(
            self.category
        )

        self.tag = normalize_optional_string(
            self.tag
        )

        self.tags = normalize_string_list(
            self.tags
        )

        self.result_type = normalize_optional_string(
            self.result_type
        )

        self.match_type = normalize_optional_string(
            self.match_type
        )

    def validate(self) -> List[str]:
        errors = []

        if self.result_type:
            valid_types = {
                item.value
                for item in ResultType
            }

            if self.result_type not in valid_types:
                errors.append(
                    f"Unknown result type: {self.result_type}"
                )

        if self.match_type:
            valid_matches = {
                item.value
                for item in MatchType
            }

            if self.match_type not in valid_matches:
                errors.append(
                    f"Unknown match type: {self.match_type}"
                )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "tag": self.tag,
            "tags": list(self.tags),
            "type": self.result_type,
            "match_type": self.match_type,
        }


# ============================================================================
# SEARCH REQUEST
# ============================================================================

@dataclass
class SearchRequest:
    """
    Primary API search request.
    """

    query: str

    category: Optional[str] = None

    tags: List[str] = field(
        default_factory=list
    )

    result_type: Optional[str] = None

    mode: SearchMode = SearchMode.STANDARD

    sort: SearchSort = SearchSort.RELEVANCE

    limit: int = 10

    offset: int = 0

    fuzzy: bool = False

    semantic: bool = False

    include_metadata: bool = False

    include_explanations: bool = False

    def __post_init__(self) -> None:

        self.query = normalize_string(
            self.query
        )

        self.category = normalize_optional_string(
            self.category
        )

        self.tags = normalize_string_list(
            self.tags
        )

        self.result_type = normalize_optional_string(
            self.result_type
        )

        self.mode = (
            normalize_enum(
                self.mode,
                SearchMode,
                SearchMode.STANDARD,
            )
            or SearchMode.STANDARD
        )

        self.sort = (
            normalize_enum(
                self.sort,
                SearchSort,
                SearchSort.RELEVANCE,
            )
            or SearchSort.RELEVANCE
        )

        self.limit = normalize_int(
            self.limit,
            10,
        )

        self.offset = normalize_int(
            self.offset,
            0,
        )

        self.fuzzy = normalize_bool(
            self.fuzzy
        )

        self.semantic = normalize_bool(
            self.semantic
        )

        self.include_metadata = normalize_bool(
            self.include_metadata
        )

        self.include_explanations = normalize_bool(
            self.include_explanations
        )

    def validate(self) -> List[str]:

        errors = []

        errors.extend(
            validate_query(
                self.query
            )
        )

        errors.extend(
            validate_limit(
                self.limit
            )
        )

        errors.extend(
            validate_offset(
                self.offset
            )
        )

        if self.result_type:
            valid_types = {
                item.value
                for item in ResultType
            }

            if self.result_type not in valid_types:
                errors.append(
                    f"Unknown result type: {self.result_type}"
                )

        return errors

    @property
    def is_valid(self) -> bool:
        return not self.validate()

    def to_dict(self) -> Dict[str, Any]:

        return {
            "query": self.query,
            "category": self.category,
            "tags": list(self.tags),
            "type": self.result_type,
            "mode": self.mode.value,
            "sort": self.sort.value,
            "limit": self.limit,
            "offset": self.offset,
            "fuzzy": self.fuzzy,
            "semantic": self.semantic,
            "include_metadata": self.include_metadata,
            "include_explanations": self.include_explanations,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "SearchRequest":

        return cls(
            query=data.get("query", ""),
            category=data.get("category"),
            tags=data.get("tags", []),
            result_type=data.get("type"),
            mode=data.get(
                "mode",
                SearchMode.STANDARD,
            ),
            sort=data.get(
                "sort",
                SearchSort.RELEVANCE,
            ),
            limit=data.get(
                "limit",
                10,
            ),
            offset=data.get(
                "offset",
                0,
            ),
            fuzzy=data.get(
                "fuzzy",
                False,
            ),
            semantic=data.get(
                "semantic",
                False,
            ),
            include_metadata=data.get(
                "include_metadata",
                False,
            ),
            include_explanations=data.get(
                "include_explanations",
                False,
            ),
        )


# ============================================================================
# QUERY ANALYSIS
# ============================================================================

@dataclass
class QueryAnalysis:
    """
    Structured information about a parsed query.
    """

    raw_query: str = ""

    terms: List[str] = field(
        default_factory=list
    )

    phrases: List[str] = field(
        default_factory=list
    )

    fields: List[str] = field(
        default_factory=list
    )

    filters: List[str] = field(
        default_factory=list
    )

    operators: List[str] = field(
        default_factory=list
    )

    intent: Optional[str] = None

    complexity: Optional[str] = None

    fuzzy_requested: bool = False

    semantic_requested: bool = False

    def __post_init__(self) -> None:

        self.raw_query = normalize_string(
            self.raw_query
        )

        self.terms = normalize_string_list(
            self.terms
        )

        self.phrases = normalize_string_list(
            self.phrases
        )

        self.fields = normalize_string_list(
            self.fields
        )

        self.filters = normalize_string_list(
            self.filters
        )

        self.operators = normalize_string_list(
            self.operators
        )

        self.intent = normalize_optional_string(
            self.intent
        )

        self.complexity = normalize_optional_string(
            self.complexity
        )

        self.fuzzy_requested = normalize_bool(
            self.fuzzy_requested
        )

        self.semantic_requested = normalize_bool(
            self.semantic_requested
        )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "raw_query": self.raw_query,
            "terms": self.terms,
            "phrases": self.phrases,
            "fields": self.fields,
            "filters": self.filters,
            "operators": self.operators,
            "intent": self.intent,
            "complexity": self.complexity,
            "fuzzy_requested": self.fuzzy_requested,
            "semantic_requested": self.semantic_requested,
        }


# ============================================================================
# SEARCH RESULT
# ============================================================================

@dataclass
class SearchResult:
    """
    Standard searchable result returned to the frontend.
    """

    id: int

    title: str = "Untitled"

    content: str = ""

    snippet: str = ""

    score: float = 0.0

    category: str = "general"

    result_type: str = "general"

    tags: List[str] = field(
        default_factory=list
    )

    created_at: Optional[str] = None

    updated_at: Optional[str] = None

    url: Optional[str] = None

    match_type: Optional[str] = None

    explanation: Optional[str] = None

    highlights: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        self.id = normalize_int(
            self.id,
            0,
        )

        self.title = normalize_string(
            self.title,
            "Untitled",
        )

        self.content = normalize_string(
            self.content
        )

        self.snippet = normalize_string(
            self.snippet
        )

        try:
            self.score = float(
                self.score
            )
        except (TypeError, ValueError):
            self.score = 0.0

        self.category = normalize_string(
            self.category,
            "general",
        )

        self.result_type = normalize_string(
            self.result_type,
            "general",
        )

        self.tags = normalize_string_list(
            self.tags
        )

        self.created_at = normalize_optional_string(
            self.created_at
        )

        self.updated_at = normalize_optional_string(
            self.updated_at
        )

        self.url = normalize_optional_string(
            self.url
        )

        self.match_type = normalize_optional_string(
            self.match_type
        )

        self.explanation = normalize_optional_string(
            self.explanation
        )

        self.highlights = normalize_string_list(
            self.highlights
        )

        if not isinstance(
            self.metadata,
            dict,
        ):
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:

        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "snippet": self.snippet,
            "score": self.score,
            "category": self.category,
            "type": self.result_type,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "url": self.url,
            "match_type": self.match_type,
            "explanation": self.explanation,
            "highlights": self.highlights,
            "metadata": self.metadata,
        }


# ============================================================================
# SEARCH METADATA
# ============================================================================

@dataclass
class SearchMetadata:
    """
    Diagnostics and engine information.
    """

    engine: str = "OurPlatform Search"

    version: str = "1.0.0"

    documents_examined: int = 0

    candidates: int = 0

    ranking_enabled: bool = True

    index_enabled: bool = True

    search_time_ms: Optional[float] = None

    mode: str = SearchMode.STANDARD.value

    retrieval_strategy: Optional[str] = None

    ranking_strategy: Optional[str] = None

    index_strategy: Optional[str] = None

    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:

        return {
            "engine": self.engine,
            "version": self.version,
            "documents_examined": self.documents_examined,
            "candidates": self.candidates,
            "ranking_enabled": self.ranking_enabled,
            "index_enabled": self.index_enabled,
            "search_time_ms": self.search_time_ms,
            "mode": self.mode,
            "retrieval_strategy": self.retrieval_strategy,
            "ranking_strategy": self.ranking_strategy,
            "index_strategy": self.index_strategy,
            "cache_hit": self.cache_hit,
        }


# ============================================================================
# SEARCH STATISTICS
# ============================================================================

@dataclass
class SearchStatistics:
    """
    Aggregate search-system statistics.
    """

    total_items: int = 0

    total_categories: int = 0

    total_tags: int = 0

    total_memories: int = 0

    total_research_records: int = 0

    total_searches: int = 0

    indexed_items: int = 0

    index_enabled: bool = True

    ranking_enabled: bool = True

    semantic_enabled: bool = False

    fuzzy_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:

        return asdict(self)


# ============================================================================
# SUGGESTIONS
# ============================================================================

@dataclass
class SearchSuggestion:
    """
    Suggested alternative query.
    """

    text: str

    score: float = 0.0

    reason: Optional[str] = None

    category: Optional[str] = None

    def __post_init__(self) -> None:

        self.text = normalize_string(
            self.text
        )

        try:
            self.score = float(
                self.score
            )
        except (TypeError, ValueError):
            self.score = 0.0

        self.reason = normalize_optional_string(
            self.reason
        )

        self.category = normalize_optional_string(
            self.category
        )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "text": self.text,
            "score": self.score,
            "reason": self.reason,
            "category": self.category,
        }


# ============================================================================
# API ERROR
# ============================================================================

@dataclass
class APIError:
    """
    Standard API error payload.
    """

    code: Union[
        APIErrorCode,
        str,
    ]

    message: str

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    field: Optional[str] = None

    timestamp: str = field(
        default_factory=lambda:
        datetime.utcnow().isoformat()
    )

    def __post_init__(self) -> None:

        if isinstance(
            self.code,
            APIErrorCode,
        ):
            self.code = self.code.value

        self.message = normalize_string(
            self.message
        )

        if not isinstance(
            self.details,
            dict,
        ):
            self.details = {}

        self.field = normalize_optional_string(
            self.field
        )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "field": self.field,
            "timestamp": self.timestamp,
        }


# ============================================================================
# API RESPONSE
# ============================================================================

@dataclass
class APIResponse:
    """
    Generic successful or unsuccessful API response.
    """

    success: bool

    data: Any = None

    message: str = ""

    errors: List[APIError] = field(
        default_factory=list
    )

    status: int = 200

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        self.success = normalize_bool(
            self.success
        )

        self.message = normalize_string(
            self.message
        )

        self.status = normalize_int(
            self.status,
            200,
        )

        if not isinstance(
            self.errors,
            list,
        ):
            self.errors = []

        normalized_errors = []

        for error in self.errors:

            if isinstance(
                error,
                APIError,
            ):
                normalized_errors.append(
                    error
                )

            elif isinstance(
                error,
                dict,
            ):
                normalized_errors.append(
                    APIError(
                        code=error.get(
                            "code",
                            APIErrorCode.INTERNAL_ERROR.value,
                        ),
                        message=error.get(
                            "message",
                            "Unknown error.",
                        ),
                        details=error.get(
                            "details",
                            {},
                        ),
                        field=error.get(
                            "field"
                        ),
                        timestamp=error.get(
                            "timestamp",
                            datetime.utcnow().isoformat(),
                        ),
                    )
                )

        self.errors = normalized_errors

        if not isinstance(
            self.metadata,
            dict,
        ):
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:

        return {
            "success": self.success,
            "data": serialize_value(
                self.data
            ),
            "message": self.message,
            "errors": [
                error.to_dict()
                for error in self.errors
            ],
            "status": self.status,
            "metadata": self.metadata,
        }


# ============================================================================
# SEARCH RESPONSE DATA
# ============================================================================

@dataclass
class SearchResponseData:
    """
    Data payload returned from a search operation.
    """

    query: str

    results: List[SearchResult] = field(
        default_factory=list
    )

    total_results: int = 0

    pagination: Optional[
        PaginationResponse
    ] = None

    query_analysis: Optional[
        QueryAnalysis
    ] = None

    search_metadata: Optional[
        SearchMetadata
    ] = None

    suggestions: List[
        SearchSuggestion
    ] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:

        self.query = normalize_string(
            self.query
        )

        self.total_results = normalize_int(
            self.total_results,
            len(self.results),
        )

        normalized_results = []

        for result in self.results:

            if isinstance(
                result,
                SearchResult,
            ):
                normalized_results.append(
                    result
                )

            elif isinstance(
                result,
                dict,
            ):
                normalized_results.append(
                    SearchResult(
                        id=result.get(
                            "id",
                            0,
                        ),
                        title=result.get(
                            "title",
                            "Untitled",
                        ),
                        content=result.get(
                            "content",
                            "",
                        ),
                        snippet=result.get(
                            "snippet",
                            "",
                        ),
                        score=result.get(
                            "score",
                            0,
                        ),
                        category=result.get(
                            "category",
                            "general",
                        ),
                        result_type=result.get(
                            "type",
                            "general",
                        ),
                        tags=result.get(
                            "tags",
                            [],
                        ),
                        created_at=result.get(
                            "created_at"
                        ),
                        updated_at=result.get(
                            "updated_at"
                        ),
                        url=result.get(
                            "url"
                        ),
                        match_type=result.get(
                            "match_type"
                        ),
                        explanation=result.get(
                            "explanation"
                        ),
                        highlights=result.get(
                            "highlights",
                            [],
                        ),
                        metadata=result.get(
                            "metadata",
                            {},
                        ),
                    )
                )

        self.results = normalized_results

    def to_dict(self) -> Dict[str, Any]:

        return {
            "query": self.query,
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "total_results": self.total_results,
            "pagination": (
                self.pagination.to_dict()
                if self.pagination
                else None
            ),
            "query_analysis": (
                self.query_analysis.to_dict()
                if self.query_analysis
                else None
            ),
            "search_metadata": (
                self.search_metadata.to_dict()
                if self.search_metadata
                else None
            ),
            "suggestions": [
                suggestion.to_dict()
                for suggestion in self.suggestions
            ],
        }


# ============================================================================
# SERIALIZATION
# ============================================================================

def serialize_value(
    value: Any,
) -> Any:
    """
    Convert supported schema objects into JSON-compatible values.
    """

    if isinstance(
        value,
        Enum,
    ):
        return value.value

    if isinstance(
        value,
        APIError,
    ):
        return value.to_dict()

    if isinstance(
        value,
        SearchResult,
    ):
        return value.to_dict()

    if isinstance(
        value,
        SearchResponseData,
    ):
        return value.to_dict()

    if isinstance(
        value,
        SearchRequest,
    ):
        return value.to_dict()

    if isinstance(
        value,
        QueryAnalysis,
    ):
        return value.to_dict()

    if isinstance(
        value,
        SearchMetadata,
    ):
        return value.to_dict()

    if isinstance(
        value,
        PaginationResponse,
    ):
        return value.to_dict()

    if isinstance(
        value,
        SearchSuggestion,
    ):
        return value.to_dict()

    if is_dataclass(value):
        return {
            key: serialize_value(item)
            for key, item
            in asdict(value).items()
        }

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            serialize_value(item)
            for item in value
        ]

    return value


# ============================================================================
# REQUEST FACTORY
# ============================================================================

def build_search_request(
    data: Optional[
        Mapping[str, Any]
    ] = None,
    **kwargs: Any,
) -> SearchRequest:
    """
    Build a SearchRequest from incoming API data.
    """

    payload: Dict[str, Any] = {}

    if data:
        payload.update(data)

    payload.update(kwargs)

    return SearchRequest.from_dict(
        payload
    )


# ============================================================================
# RESPONSE FACTORIES
# ============================================================================

def success_response(
    data: Any = None,
    *,
    message: str = "",
    status: int = 200,
    metadata: Optional[
        Dict[str, Any]
    ] = None,
) -> APIResponse:
    """
    Build a standard successful API response.
    """

    return APIResponse(
        success=True,
        data=data,
        message=message,
        status=status,
        metadata=metadata or {},
    )


def error_response(
    code: Union[
        APIErrorCode,
        str,
    ],
    message: str,
    *,
    status: int = 400,
    details: Optional[
        Dict[str, Any]
    ] = None,
    field: Optional[str] = None,
) -> APIResponse:
    """
    Build a standard unsuccessful API response.
    """

    error = APIError(
        code=code,
        message=message,
        details=details or {},
        field=field,
    )

    return APIResponse(
        success=False,
        data=None,
        message=message,
        errors=[error],
        status=status,
    )


# ============================================================================
# SEARCH RESPONSE FACTORY
# ============================================================================

def build_search_response(
    request: SearchRequest,
    results: Sequence[
        Union[
            SearchResult,
            Mapping[str, Any],
        ]
    ],
    *,
    total_results: Optional[int] = None,
    pagination: Optional[
        PaginationResponse
    ] = None,
    query_analysis: Optional[
        QueryAnalysis
    ] = None,
    search_metadata: Optional[
        SearchMetadata
    ] = None,
    suggestions: Optional[
        Sequence[
            Union[
                SearchSuggestion,
                Mapping[str, Any],
                str,
            ]
        ]
    ] = None,
) -> APIResponse:
    """
    Build the canonical search response.
    """

    normalized_results: List[
        SearchResult
    ] = []

    for result in results:

        if isinstance(
            result,
            SearchResult,
        ):
            normalized_results.append(
                result
            )
            continue

        if isinstance(
            result,
            Mapping,
        ):

            normalized_results.append(
                SearchResult(
                    id=result.get(
                        "id",
                        0,
                    ),
                    title=result.get(
                        "title",
                        "Untitled",
                    ),
                    content=result.get(
                        "content",
                        "",
                    ),
                    snippet=result.get(
                        "snippet",
                        "",
                    ),
                    score=result.get(
                        "score",
                        0,
                    ),
                    category=result.get(
                        "category",
                        "general",
                    ),
                    result_type=result.get(
                        "type",
                        "general",
                    ),
                    tags=result.get(
                        "tags",
                        [],
                    ),
                    created_at=result.get(
                        "created_at"
                    ),
                    updated_at=result.get(
                        "updated_at"
                    ),
                    url=result.get(
                        "url"
                    ),
                    match_type=result.get(
                        "match_type"
                    ),
                    explanation=result.get(
                        "explanation"
                    ),
                    highlights=result.get(
                        "highlights",
                        [],
                    ),
                    metadata=result.get(
                        "metadata",
                        {},
                    ),
                )
            )


    normalized_suggestions: List[
        SearchSuggestion
    ] = []

    for suggestion in suggestions or []:

        if isinstance(
            suggestion,
            SearchSuggestion,
        ):
            normalized_suggestions.append(
                suggestion
            )

        elif isinstance(
            suggestion,
            Mapping,
        ):
            normalized_suggestions.append(
                SearchSuggestion(
                    text=suggestion.get(
                        "text",
                        "",
                    ),
                    score=suggestion.get(
                        "score",
                        0,
                    ),
                    reason=suggestion.get(
                        "reason"
                    ),
                    category=suggestion.get(
                        "category"
                    ),
                )
            )

        elif isinstance(
            suggestion,
            str,
        ):
            normalized_suggestions.append(
                SearchSuggestion(
                    text=suggestion
                )
            )


    total = (
        total_results
        if total_results is not None
        else len(normalized_results)
    )


    data = SearchResponseData(
        query=request.query,
        results=normalized_results,
        total_results=total,
        pagination=pagination,
        query_analysis=query_analysis,
        search_metadata=search_metadata,
        suggestions=normalized_suggestions,
    )


    return success_response(
        data=data,
        message="Search completed successfully.",
    )


# ============================================================================
# ERROR VALIDATION FACTORY
# ============================================================================

def validate_search_request(
    request: SearchRequest,
) -> Optional[APIResponse]:
    """
    Validate a SearchRequest and return an API error response
    when invalid.

    Returns:
        None when valid.
        APIResponse when invalid.
    """

    errors = request.validate()

    if not errors:
        return None


    api_errors = [
        APIError(
            code=(
                APIErrorCode.INVALID_REQUEST.value
            ),
            message=message,
        )
        for message in errors
    ]


    return APIResponse(
        success=False,
        data=None,
        message="Search request validation failed.",
        errors=api_errors,
        status=400,
    )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [

    # Versions

    "SCHEMA_VERSION",
    "API_VERSION",
    "SCHEMA_NAME",
    "SCHEMA_COMPATIBILITY",

    # Enums

    "SearchMode",
    "SearchSort",
    "MatchType",
    "ResultType",
    "APIErrorCode",

    # Helpers

    "normalize_string",
    "normalize_optional_string",
    "normalize_string_list",
    "normalize_int",
    "normalize_bool",
    "normalize_enum",

    # Validation

    "validate_query",
    "validate_limit",
    "validate_offset",
    "validate_identifier",

    # Pagination

    "PaginationRequest",
    "PaginationResponse",

    # Filters

    "SearchFilters",

    # Search

    "SearchRequest",
    "SearchResult",
    "SearchResponseData",

    # Analysis

    "QueryAnalysis",

    # Metadata

    "SearchMetadata",
    "SearchStatistics",

    # Suggestions

    "SearchSuggestion",

    # Errors

    "APIError",
    "APIResponse",

    # Serialization

    "serialize_value",

    # Factories

    "build_search_request",
    "success_response",
    "error_response",
    "build_search_response",
    "validate_search_request",
]