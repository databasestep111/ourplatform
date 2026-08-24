"""
OurPlatform Schema Adapter
==========================

Purpose
-------
Provides a stable translation layer between the existing Search subsystem
and the API schema layer.

Architecture
------------

    Search subsystem
          |
          v
    schema_adapter.py
          |
          v
    Api.schemas
          |
          v
    integration.py
          |
          v
    API routes / frontend

This module intentionally does NOT perform searching.

Responsibilities
----------------
- Convert backend result objects into SearchResult objects.
- Normalize dictionaries returned by older search components.
- Convert API requests into backend-friendly dictionaries.
- Convert search output into canonical API responses.
- Preserve unknown metadata instead of throwing it away.
- Provide compatibility with multiple backend result shapes.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence

from .schemas import (
    APIError,
    APIErrorCode,
    APIResponse,
    PaginationResponse,
    QueryAnalysis,
    SearchMetadata,
    SearchRequest,
    SearchResponseData,
    SearchResult,
    SearchSuggestion,
    build_search_request,
    build_search_response,
    error_response,
    serialize_value,
    success_response,
)


# ============================================================================
# CONSTANTS
# ============================================================================

ADAPTER_VERSION = "1.0.0"

DEFAULT_RESULT_TYPE = "general"
DEFAULT_CATEGORY = "general"
DEFAULT_TITLE = "Untitled"


# ============================================================================
# BASIC COMPATIBILITY HELPERS
# ============================================================================

def _get_value(
    source: Any,
    *names: str,
    default: Any = None,
) -> Any:
    """
    Retrieve a value from either a mapping or an object.

    This allows the adapter to work with:

        dict
        dataclass
        normal Python object
        search-engine result object
    """

    if source is None:
        return default

    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]

    for name in names:
        if hasattr(source, name):
            try:
                return getattr(source, name)
            except Exception:
                continue

    return default


def _as_mapping(
    value: Any,
) -> dict[str, Any]:
    """
    Convert a backend object into a dictionary where possible.
    """

    if value is None:
        return {}

    if isinstance(value, Mapping):
        return dict(value)

    if is_dataclass(value):
        try:
            return asdict(value)
        except Exception:
            pass

    if hasattr(value, "to_dict"):
        try:
            result = value.to_dict()

            if isinstance(result, Mapping):
                return dict(result)

        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return dict(vars(value))
        except Exception:
            pass

    return {}


def _string(
    value: Any,
    default: str = "",
) -> str:
    """
    Safely normalize a value into a string.
    """

    if value is None:
        return default

    return str(value).strip()


def _list(
    value: Any,
) -> list[Any]:
    """
    Normalize an arbitrary value into a list.
    """

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, Mapping):
        return [value]

    if isinstance(value, Iterable):
        try:
            return list(value)
        except TypeError:
            return []

    return [value]


def _float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Safely convert a value to float.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(
    value: Any,
    default: int = 0,
) -> int:
    """
    Safely convert a value to integer.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ============================================================================
# RESULT ADAPTER
# ============================================================================

def adapt_result(
    result: Any,
    *,
    preserve_unknown_fields: bool = True,
) -> SearchResult:
    """
    Convert one backend search result into SearchResult.

    Supported backend naming variations include:

        id / document_id / result_id
        title / name
        content / text / body
        snippet / summary / preview
        score / relevance / rank
        category / collection
        type / result_type / document_type
        tags / keywords
        created_at / created
        updated_at / updated
        url / link
        match_type / match
        explanation / reason
        highlights / matches
        metadata / meta
    """

    data = _as_mapping(result)

    identifier = _get_value(
        result,
        "id",
        "document_id",
        "result_id",
        "item_id",
        default=data.get("id", 0),
    )

    title = _get_value(
        result,
        "title",
        "name",
        "heading",
        default=data.get("title", DEFAULT_TITLE),
    )

    content = _get_value(
        result,
        "content",
        "text",
        "body",
        "document",
        default=data.get("content", ""),
    )

    snippet = _get_value(
        result,
        "snippet",
        "summary",
        "preview",
        "excerpt",
        default=data.get("snippet", ""),
    )

    score = _get_value(
        result,
        "score",
        "relevance",
        "rank",
        "similarity",
        default=data.get("score", 0.0),
    )

    category = _get_value(
        result,
        "category",
        "collection",
        "group",
        default=data.get(
            "category",
            DEFAULT_CATEGORY,
        ),
    )

    result_type = _get_value(
        result,
        "result_type",
        "type",
        "document_type",
        "kind",
        default=data.get(
            "type",
            DEFAULT_RESULT_TYPE,
        ),
    )

    tags = _get_value(
        result,
        "tags",
        "keywords",
        "labels",
        default=data.get("tags", []),
    )

    created_at = _get_value(
        result,
        "created_at",
        "created",
        "creation_date",
        default=data.get("created_at"),
    )

    updated_at = _get_value(
        result,
        "updated_at",
        "updated",
        "modified",
        "modification_date",
        default=data.get("updated_at"),
    )

    url = _get_value(
        result,
        "url",
        "link",
        "href",
        default=data.get("url"),
    )

    match_type = _get_value(
        result,
        "match_type",
        "match",
        "matching_strategy",
        default=data.get("match_type"),
    )

    explanation = _get_value(
        result,
        "explanation",
        "reason",
        "ranking_explanation",
        default=data.get("explanation"),
    )

    highlights = _get_value(
        result,
        "highlights",
        "matches",
        "highlighted",
        default=data.get("highlights", []),
    )

    metadata = _get_value(
        result,
        "metadata",
        "meta",
        "extra",
        default=data.get("metadata", {}),
    )

    if not isinstance(metadata, dict):
        metadata = {}

    if preserve_unknown_fields and data:
        known_fields = {
            "id",
            "document_id",
            "result_id",
            "item_id",
            "title",
            "name",
            "heading",
            "content",
            "text",
            "body",
            "document",
            "snippet",
            "summary",
            "preview",
            "excerpt",
            "score",
            "relevance",
            "rank",
            "similarity",
            "category",
            "collection",
            "group",
            "result_type",
            "type",
            "document_type",
            "kind",
            "tags",
            "keywords",
            "labels",
            "created_at",
            "created",
            "creation_date",
            "updated_at",
            "updated",
            "modified",
            "modification_date",
            "url",
            "link",
            "href",
            "match_type",
            "match",
            "matching_strategy",
            "explanation",
            "reason",
            "ranking_explanation",
            "highlights",
            "matches",
            "highlighted",
            "metadata",
            "meta",
            "extra",
        }

        for key, value in data.items():
            if key not in known_fields:
                metadata.setdefault(key, value)

    return SearchResult(
        id=_int(identifier),
        title=_string(title, DEFAULT_TITLE),
        content=_string(content),
        snippet=_string(snippet),
        score=_float(score),
        category=_string(
            category,
            DEFAULT_CATEGORY,
        ),
        result_type=_string(
            result_type,
            DEFAULT_RESULT_TYPE,
        ),
        tags=_list(tags),
        created_at=(
            _string(created_at)
            if created_at is not None
            else None
        ),
        updated_at=(
            _string(updated_at)
            if updated_at is not None
            else None
        ),
        url=(
            _string(url)
            if url is not None
            else None
        ),
        match_type=(
            _string(match_type)
            if match_type is not None
            else None
        ),
        explanation=(
            _string(explanation)
            if explanation is not None
            else None
        ),
        highlights=_list(highlights),
        metadata=metadata,
    )


# ============================================================================
# RESULT COLLECTION ADAPTER
# ============================================================================

def adapt_results(
    results: Optional[Sequence[Any]],
) -> list[SearchResult]:
    """
    Convert an entire backend result collection.
    """

    if results is None:
        return []

    adapted: list[SearchResult] = []

    for result in results:
        try:
            adapted.append(
                adapt_result(result)
            )
        except Exception:
            # A malformed individual result should not destroy
            # an otherwise valid search response.
            continue

    return adapted


# ============================================================================
# REQUEST ADAPTER
# ============================================================================

def adapt_request(
    data: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> SearchRequest:
    """
    Convert incoming API data into the canonical SearchRequest.
    """

    payload: dict[str, Any] = {}

    if data:
        payload.update(data)

    payload.update(kwargs)

    # Compatibility aliases.
    if "result_type" in payload and "type" not in payload:
        payload["type"] = payload["result_type"]

    if "search_mode" in payload and "mode" not in payload:
        payload["mode"] = payload["search_mode"]

    if "sort_by" in payload and "sort" not in payload:
        payload["sort"] = payload["sort_by"]

    if "page_size" in payload and "limit" not in payload:
        payload["limit"] = payload["page_size"]

    return build_search_request(payload)


# ============================================================================
# QUERY ANALYSIS ADAPTER
# ============================================================================

def adapt_query_analysis(
    analysis: Any,
) -> Optional[QueryAnalysis]:
    """
    Convert a parser/analysis result into QueryAnalysis.
    """

    if analysis is None:
        return None

    if isinstance(analysis, QueryAnalysis):
        return analysis

    data = _as_mapping(analysis)

    return QueryAnalysis(
        raw_query=_get_value(
            analysis,
            "raw_query",
            "query",
            "original_query",
            default=data.get("raw_query", ""),
        ),
        terms=_get_value(
            analysis,
            "terms",
            "tokens",
            default=data.get("terms", []),
        ),
        phrases=_get_value(
            analysis,
            "phrases",
            default=data.get("phrases", []),
        ),
        fields=_get_value(
            analysis,
            "fields",
            "field_queries",
            default=data.get("fields", []),
        ),
        filters=_get_value(
            analysis,
            "filters",
            default=data.get("filters", []),
        ),
        operators=_get_value(
            analysis,
            "operators",
            "boolean_operators",
            default=data.get("operators", []),
        ),
        intent=_get_value(
            analysis,
            "intent",
            default=data.get("intent"),
        ),
        complexity=_get_value(
            analysis,
            "complexity",
            default=data.get("complexity"),
        ),
        fuzzy_requested=_get_value(
            analysis,
            "fuzzy_requested",
            "fuzzy",
            default=data.get("fuzzy_requested", False),
        ),
        semantic_requested=_get_value(
            analysis,
            "semantic_requested",
            "semantic",
            default=data.get(
                "semantic_requested",
                False,
            ),
        ),
    )


# ============================================================================
# METADATA ADAPTER
# ============================================================================

def adapt_search_metadata(
    metadata: Any,
) -> Optional[SearchMetadata]:
    """
    Convert backend diagnostics into SearchMetadata.
    """

    if metadata is None:
        return None

    if isinstance(metadata, SearchMetadata):
        return metadata

    data = _as_mapping(metadata)

    return SearchMetadata(
        engine=_string(
            _get_value(
                metadata,
                "engine",
                default=data.get(
                    "engine",
                    "OurPlatform Search",
                ),
            )
        ),
        version=_string(
            _get_value(
                metadata,
                "version",
                default=data.get(
                    "version",
                    ADAPTER_VERSION,
                ),
            )
        ),
        documents_examined=_int(
            _get_value(
                metadata,
                "documents_examined",
                "documents",
                "examined",
                default=data.get(
                    "documents_examined",
                    0,
                ),
            )
        ),
        candidates=_int(
            _get_value(
                metadata,
                "candidates",
                "candidate_count",
                default=data.get(
                    "candidates",
                    0,
                ),
            )
        ),
        ranking_enabled=bool(
            _get_value(
                metadata,
                "ranking_enabled",
                default=data.get(
                    "ranking_enabled",
                    True,
                )
            )
        ),
        index_enabled=bool(
            _get_value(
                metadata,
                "index_enabled",
                default=data.get(
                    "index_enabled",
                    True,
                )
            )
        ),
        search_time_ms=_get_value(
            metadata,
            "search_time_ms",
            "duration_ms",
            "elapsed_ms",
            default=data.get(
                "search_time_ms"
            ),
        ),
        mode=_string(
            _get_value(
                metadata,
                "mode",
                "search_mode",
                default=data.get(
                    "mode",
                    "standard",
                ),
            )
        ),
        retrieval_strategy=_get_value(
            metadata,
            "retrieval_strategy",
            "retrieval",
            default=data.get(
                "retrieval_strategy"
            ),
        ),
        ranking_strategy=_get_value(
            metadata,
            "ranking_strategy",
            "ranking",
            default=data.get(
                "ranking_strategy"
            ),
        ),
        index_strategy=_get_value(
            metadata,
            "index_strategy",
            "index",
            default=data.get(
                "index_strategy"
            ),
        ),
        cache_hit=bool(
            _get_value(
                metadata,
                "cache_hit",
                default=data.get(
                    "cache_hit",
                    False,
                ),
            )
        ),
    )


# ============================================================================
# SUGGESTION ADAPTER
# ============================================================================

def adapt_suggestion(
    suggestion: Any,
) -> SearchSuggestion:
    """
    Convert a backend suggestion into SearchSuggestion.
    """

    if isinstance(
        suggestion,
        SearchSuggestion,
    ):
        return suggestion

    if isinstance(
        suggestion,
        str,
    ):
        return SearchSuggestion(
            text=suggestion
        )

    data = _as_mapping(suggestion)

    return SearchSuggestion(
        text=_get_value(
            suggestion,
            "text",
            "query",
            "suggestion",
            default=data.get("text", ""),
        ),
        score=_get_value(
            suggestion,
            "score",
            "relevance",
            default=data.get("score", 0.0),
        ),
        reason=_get_value(
            suggestion,
            "reason",
            "explanation",
            default=data.get("reason"),
        ),
        category=_get_value(
            suggestion,
            "category",
            default=data.get("category"),
        ),
    )


def adapt_suggestions(
    suggestions: Optional[Sequence[Any]],
) -> list[SearchSuggestion]:
    """
    Adapt all backend suggestions.
    """

    if not suggestions:
        return []

    return [
        adapt_suggestion(item)
        for item in suggestions
    ]


# ============================================================================
# PAGINATION ADAPTER
# ============================================================================

def adapt_pagination(
    pagination: Any,
) -> Optional[PaginationResponse]:
    """
    Convert backend pagination information.
    """

    if pagination is None:
        return None

    if isinstance(
        pagination,
        PaginationResponse,
    ):
        return pagination

    data = _as_mapping(pagination)

    page = _int(
        _get_value(
            pagination,
            "page",
            default=data.get("page", 1),
        ),
        1,
    )

    limit = _int(
        _get_value(
            pagination,
            "limit",
            "page_size",
            default=data.get("limit", 10),
        ),
        10,
    )

    offset = _int(
        _get_value(
            pagination,
            "offset",
            default=data.get("offset", 0),
        )
    )

    total = _int(
        _get_value(
            pagination,
            "total",
            "total_results",
            default=data.get("total", 0),
        )
    )

    total_pages = _int(
        _get_value(
            pagination,
            "total_pages",
            "pages",
            default=data.get("total_pages", 0),
        )
    )

    has_previous = bool(
        _get_value(
            pagination,
            "has_previous",
            "previous",
            default=data.get(
                "has_previous",
                False,
            ),
        )
    )

    has_next = bool(
        _get_value(
            pagination,
            "has_next",
            "next",
            default=data.get(
                "has_next",
                False,
            ),
        )
    )

    previous_offset = _get_value(
        pagination,
        "previous_offset",
        default=data.get(
            "previous_offset"
        ),
    )

    next_offset = _get_value(
        pagination,
        "next_offset",
        default=data.get(
            "next_offset"
        ),
    )

    return PaginationResponse(
        page=page,
        limit=limit,
        offset=offset,
        total=total,
        total_pages=total_pages,
        has_previous=has_previous,
        has_next=has_next,
        previous_offset=previous_offset,
        next_offset=next_offset,
    )


# ============================================================================
# BACKEND OUTPUT EXTRACTION
# ============================================================================

def extract_results(
    backend_output: Any,
) -> list[SearchResult]:
    """
    Extract results from common backend return shapes.

    Supported examples:

        [result, result]

        {"results": [...]}

        {"items": [...]}

        {"data": {"results": [...]}}

        object.results

        object.items
    """

    if backend_output is None:
        return []

    if isinstance(
        backend_output,
        SearchResponseData,
    ):
        return backend_output.results

    if isinstance(
        backend_output,
        APIResponse,
    ):
        return extract_results(
            backend_output.data
        )

    if isinstance(
        backend_output,
        (list, tuple),
    ):
        return adapt_results(
            backend_output
        )

    data = _as_mapping(
        backend_output
    )

    candidates = _get_value(
        backend_output,
        "results",
        "items",
        "documents",
        "matches",
        default=None,
    )

    if candidates is None and "data" in data:
        return extract_results(
            data["data"]
        )

    return adapt_results(
        _list(candidates)
    )


# ============================================================================
# TOTAL RESULT EXTRACTION
# ============================================================================

def extract_total(
    backend_output: Any,
    default: int = 0,
) -> int:
    """
    Extract total result count from backend output.
    """

    if backend_output is None:
        return default

    if isinstance(
        backend_output,
        SearchResponseData,
    ):
        return backend_output.total_results

    if isinstance(
        backend_output,
        APIResponse,
    ):
        return extract_total(
            backend_output.data,
            default,
        )

    value = _get_value(
        backend_output,
        "total_results",
        "total",
        "count",
        "result_count",
        default=None,
    )

    if value is not None:
        return _int(
            value,
            default,
        )

    data = _as_mapping(
        backend_output
    )

    if "data" in data:
        return extract_total(
            data["data"],
            default,
        )

    return default


# ============================================================================
# CANONICAL RESPONSE ADAPTER
# ============================================================================

def adapt_search_response(
    request: SearchRequest,
    backend_output: Any,
    *,
    query_analysis: Any = None,
    search_metadata: Any = None,
    pagination: Any = None,
    suggestions: Optional[Sequence[Any]] = None,
) -> APIResponse:
    """
    Convert arbitrary backend search output into the canonical API response.
    """

    if isinstance(
        backend_output,
        APIResponse,
    ):
        if backend_output.success:
            backend_results = extract_results(
                backend_output
            )

            total = extract_total(
                backend_output,
                len(backend_results),
            )

            return build_search_response(
                request,
                backend_results,
                total_results=total,
                pagination=adapt_pagination(
                    pagination
                ),
                query_analysis=adapt_query_analysis(
                    query_analysis
                ),
                search_metadata=adapt_search_metadata(
                    search_metadata
                ),
                suggestions=adapt_suggestions(
                    suggestions
                ),
            )

        return backend_output

    results = extract_results(
        backend_output
    )

    total = extract_total(
        backend_output,
        len(results),
    )

    return build_search_response(
        request,
        results,
        total_results=total,
        pagination=adapt_pagination(
            pagination
        ),
        query_analysis=adapt_query_analysis(
            query_analysis
        ),
        search_metadata=adapt_search_metadata(
            search_metadata
        ),
        suggestions=adapt_suggestions(
            suggestions
        ),
    )


# ============================================================================
# ERROR ADAPTER
# ============================================================================

def adapt_error(
    error: Any,
    *,
    default_code: str = APIErrorCode.INTERNAL_ERROR.value,
    default_status: int = 500,
) -> APIResponse:
    """
    Convert exceptions or backend error objects into APIResponse.
    """

    if isinstance(
        error,
        APIResponse,
    ):
        return error

    if isinstance(
        error,
        APIError,
    ):
        return APIResponse(
            success=False,
            data=None,
            message=error.message,
            errors=[error],
            status=default_status,
        )

    if isinstance(
        error,
        Exception,
    ):
        message = str(error) or "An internal error occurred."

        return error_response(
            default_code,
            message,
            status=default_status,
        )

    if isinstance(
        error,
        Mapping,
    ):
        code = error.get(
            "code",
            default_code,
        )

        message = error.get(
            "message",
            "An API error occurred.",
        )

        details = error.get(
            "details",
            {},
        )

        return error_response(
            code,
            str(message),
            status=_int(
                error.get(
                    "status",
                    default_status,
                ),
                default_status,
            ),
            details=(
                details
                if isinstance(details, dict)
                else {}
            ),
            field=error.get("field"),
        )

    return error_response(
        default_code,
        str(error) or "An internal error occurred.",
        status=default_status,
    )


# ============================================================================
# SERIALIZATION BRIDGE
# ============================================================================

def to_api_dict(
    value: Any,
) -> Any:
    """
    Public serialization entry point for integration.py.

    Keeps serialization rules centralized in schemas.py while providing
    integration.py with a stable adapter function.
    """

    return serialize_value(value)


# ============================================================================
# BACKEND REQUEST PAYLOAD
# ============================================================================

def to_backend_payload(
    request: SearchRequest,
) -> dict[str, Any]:
    """
    Convert SearchRequest into a backend-friendly dictionary.

    This is deliberately separate from SearchRequest.to_dict() so the
    backend can evolve without changing the public API contract.
    """

    return {
        "query": request.query,
        "category": request.category,
        "tags": list(request.tags),
        "result_type": request.result_type,
        "mode": request.mode.value,
        "sort": request.sort.value,
        "limit": request.limit,
        "offset": request.offset,
        "fuzzy": request.fuzzy,
        "semantic": request.semantic,
        "include_metadata": request.include_metadata,
        "include_explanations": request.include_explanations,
    }


# ============================================================================
# SAFE INTEGRATION EXECUTION
# ============================================================================

def execute_backend(
    backend_callable: Any,
    request: SearchRequest,
    **extra_kwargs: Any,
) -> Any:
    """
    Execute a backend search callable with a normalized request.

    The adapter tries the modern request-object interface first and falls
    back to a dictionary interface for compatibility with older components.
    """

    payload = to_backend_payload(
        request
    )

    try:
        return backend_callable(
            request=request,
            **extra_kwargs,
        )

    except TypeError:
        try:
            return backend_callable(
                payload,
                **extra_kwargs,
            )

        except TypeError:
            return backend_callable(
                request.query,
                **payload,
                **extra_kwargs,
            )


# ============================================================================
# HIGH-LEVEL SEARCH BRIDGE
# ============================================================================

def run_search_bridge(
    backend_callable: Any,
    request: SearchRequest,
    *,
    query_analysis: Any = None,
    search_metadata: Any = None,
    pagination: Any = None,
    suggestions: Optional[Sequence[Any]] = None,
    **extra_kwargs: Any,
) -> APIResponse:
    """
    Complete bridge between a backend search callable and the API layer.

    integration.py can call this instead of manually translating every
    backend result.
    """

    validation_error = None

    try:
        from .schemas import validate_search_request

        validation_error = validate_search_request(
            request
        )

    except Exception as exc:
        return adapt_error(
            exc
        )

    if validation_error is not None:
        return validation_error

    try:
        backend_output = execute_backend(
            backend_callable,
            request,
            **extra_kwargs,
        )

        return adapt_search_response(
            request,
            backend_output,
            query_analysis=query_analysis,
            search_metadata=search_metadata,
            pagination=pagination,
            suggestions=suggestions,
        )

    except Exception as exc:
        return adapt_error(
            exc,
            default_code=APIErrorCode.SEARCH_ERROR.value,
            default_status=500,
        )


# ============================================================================
# MODULE INFORMATION
# ============================================================================

def adapter_info() -> dict[str, Any]:
    """
    Return adapter diagnostics.
    """

    return {
        "name": "OurPlatform Schema Adapter",
        "version": ADAPTER_VERSION,
        "schema_version": "1.0.0",
        "purpose": "Search/API compatibility bridge",
        "supports_request_adaptation": True,
        "supports_result_adaptation": True,
        "supports_response_adaptation": True,
        "supports_error_adaptation": True,
        "supports_legacy_backend_shapes": True,
    }


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "ADAPTER_VERSION",
    "adapt_result",
    "adapt_results",
    "adapt_request",
    "adapt_query_analysis",
    "adapt_search_metadata",
    "adapt_suggestion",
    "adapt_suggestions",
    "adapt_pagination",
    "extract_results",
    "extract_total",
    "adapt_search_response",
    "adapt_error",
    "to_api_dict",
    "to_backend_payload",
    "execute_backend",
    "run_search_bridge",
    "adapter_info",
]