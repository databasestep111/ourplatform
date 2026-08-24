"""
OurPlatform Search API.

This module is the API adapter for the existing search engine.

Responsibilities
----------------
- Receive normalized API search requests.
- Extract and validate search parameters.
- Call the existing Search engine.
- Apply category/tag/limit constraints.
- Normalize search results.
- Return consistent APIResponse objects.
- Provide search statistics and item operations.
- Keep API concerns separate from search-engine logic.

Architecture
------------

    Frontend
        |
        v
    /api/search
        |
        v
    search_api.py
        |
        v
    search/search.py
        |
        v
    Search
        |
        v
    ranked results
        |
        v
    APIResponse
        |
        v
    Frontend

The search engine remains the source of truth for search behaviour.
This module is an adapter, not a second search engine.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from .routes import (
    APIRequest,
    APIResponse,
    error_response,
    success_response,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_LIMIT = 10

MAX_LIMIT = 500

MIN_LIMIT = 1


# ============================================================================
# BACKEND LOADING
# ============================================================================

def get_search_engine() -> Any:
    """
    Retrieve the existing global search engine.

    The current search implementation exposes:

        search = Search()

    from search/search.py.

    Importing it lazily prevents the API package from creating
    unnecessary circular-import problems during application startup.
    """

    try:

        from search.search import search

        return search

    except ImportError as exc:

        raise RuntimeError(
            "Unable to load the existing search engine "
            "from search.search."
        ) from exc


# ============================================================================
# REQUEST VALUES
# ============================================================================

def _first_value(
    value: Any,
    default: Any = None,
) -> Any:
    """
    Extract the first value from a list-like request parameter.

    Useful when a web framework represents repeated parameters
    as lists.
    """

    if isinstance(
        value,
        (list, tuple),
    ):

        if not value:
            return default

        return value[0]

    return value if value is not None else default


def _as_string(
    value: Any,
    default: str = "",
) -> str:
    """
    Safely convert a request value into a string.
    """

    if value is None:
        return default

    return str(value).strip()


def _as_list(
    value: Any,
) -> list[str]:
    """
    Normalize a request parameter into a list of strings.

    Supports:

        "one"

        ["one", "two"]

        ("one", "two")

        "one,two"

        "one;two"
    """

    if value is None:
        return []

    if isinstance(
        value,
        (list, tuple, set),
    ):

        values = value

    else:

        text = str(value)

        if "," in text:

            values = text.split(",")

        elif ";" in text:

            values = text.split(";")

        else:

            values = [text]

    result = []

    for item in values:

        normalized = str(
            item
        ).strip()

        if normalized:
            result.append(
                normalized
            )

    return result


def _as_integer(
    value: Any,
    default: int,
) -> int:
    """
    Safely parse an integer.
    """

    if value is None:
        return default

    try:

        return int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return default


# ============================================================================
# LIMIT NORMALIZATION
# ============================================================================

def normalize_limit(
    value: Any,
) -> int:
    """
    Normalize the requested result limit.

    Limits are bounded so a frontend request cannot accidentally
    ask the in-memory search engine for an unreasonable number
    of results.
    """

    limit = _as_integer(
        value,
        DEFAULT_LIMIT,
    )

    if limit < MIN_LIMIT:
        return MIN_LIMIT

    if limit > MAX_LIMIT:
        return MAX_LIMIT

    return limit


# ============================================================================
# RESULT NORMALIZATION
# ============================================================================

def normalize_result(
    result: Any,
) -> dict[str, Any]:
    """
    Convert a search-engine result into an API-safe dictionary.

    The current Search implementation returns dictionaries, but
    this function also tolerates simple objects and other mappings
    so the API remains adaptable as the backend evolves.
    """

    if isinstance(
        result,
        Mapping,
    ):

        normalized = dict(
            result
        )

    elif hasattr(
        result,
        "__dict__",
    ):

        normalized = dict(
            result.__dict__
        )

    else:

        normalized = {
            "content": str(
                result
            )
        }

    normalized.setdefault(
        "id",
        None,
    )

    normalized.setdefault(
        "title",
        "Untitled",
    )

    normalized.setdefault(
        "content",
        "",
    )

    normalized.setdefault(
        "category",
        "general",
    )

    normalized.setdefault(
        "tags",
        [],
    )

    normalized.setdefault(
        "score",
        0,
    )

    return normalized


def normalize_results(
    results: Iterable[Any],
) -> list[dict[str, Any]]:
    """
    Normalize an iterable of search results.
    """

    return [
        normalize_result(
            result
        )
        for result in results
    ]


# ============================================================================
# SEARCH REQUEST
# ============================================================================

def extract_search_parameters(
    request: APIRequest,
) -> dict[str, Any]:
    """
    Extract supported search parameters from an API request.

    Supported parameters currently include:

        query
        category
        tags
        limit

    Additional parameters can be introduced later without changing
    the basic API request structure.
    """

    query = _as_string(
        request.get(
            "query",
            "",
        )
    )

    category = _as_string(
        request.get(
            "category",
            "",
        )
    )

    tags = _as_list(
        request.get(
            "tags",
            [],
        )
    )

    limit = normalize_limit(
        request.get(
            "limit",
            DEFAULT_LIMIT,
        )
    )

    return {
        "query": query,
        "category": (
            category
            if category
            else None
        ),
        "tags": tags,
        "limit": limit,
    }


# ============================================================================
# VALIDATION
# ============================================================================

def validate_search_parameters(
    parameters: Mapping[str, Any],
) -> Optional[APIResponse]:
    """
    Validate search parameters.

    Returns an APIResponse when validation fails.
    Returns None when parameters are valid.
    """

    query = _as_string(
        parameters.get(
            "query",
            "",
        )
    )

    if not query:

        return error_response(
            message=(
                "A search query is required."
            ),
            status_code=400,
            code="missing_query",
        )

    if len(query) > 5000:

        return error_response(
            message=(
                "Search query is too long."
            ),
            status_code=400,
            code="query_too_long",
            maximum_length=5000,
        )

    return None


# ============================================================================
# CORE SEARCH
# ============================================================================

def execute_search(
    *,
    query: str,
    category: Optional[str] = None,
    tags: Optional[list[str]] = None,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """
    Execute a search against the existing search engine.

    This is the main integration point between the API and
    search/search.py.
    """

    engine = get_search_engine()

    normalized_limit = normalize_limit(
        limit
    )

    normalized_tags = (
        tags
        if tags is not None
        else []
    )

    results = engine.find(
        query=query,
        category=category,
        tags=normalized_tags,
        limit=normalized_limit,
    )

    return normalize_results(
        results
    )


# ============================================================================
# SEARCH ENDPOINT
# ============================================================================

def search_endpoint(
    request: APIRequest,
) -> APIResponse:
    """
    Main /api/search endpoint.

    Supported request fields:

        query
        category
        tags
        limit

    Example conceptual request:

        {
            "query": "machine learning",
            "category": "research",
            "tags": ["ai"],
            "limit": 20
        }
    """

    parameters = extract_search_parameters(
        request
    )

    validation_error = (
        validate_search_parameters(
            parameters
        )
    )

    if validation_error is not None:
        return validation_error

    try:

        results = execute_search(
            query=parameters["query"],
            category=parameters["category"],
            tags=parameters["tags"],
            limit=parameters["limit"],
        )

    except Exception as exc:

        return error_response(
            message=(
                "The search backend could not "
                "complete the request."
            ),
            status_code=500,
            code="search_backend_error",
            error=str(exc),
        )

    return success_response(
        data={
            "query": parameters["query"],
            "results": results,
            "count": len(results),
            "filters": {
                "category": parameters[
                    "category"
                ],
                "tags": parameters[
                    "tags"
                ],
            },
        },
        message="Search completed successfully.",
        query=parameters["query"],
        result_count=len(results),
    )


# ============================================================================
# SEARCH BY TITLE
# ============================================================================

def search_title_endpoint(
    request: APIRequest,
) -> APIResponse:
    """
    Search only indexed titles.
    """

    query = _as_string(
        request.get(
            "query",
            "",
        )
    )

    limit = normalize_limit(
        request.get(
            "limit",
            DEFAULT_LIMIT,
        )
    )

    if not query:

        return error_response(
            message="A title query is required.",
            status_code=400,
            code="missing_query",
        )

    try:

        engine = get_search_engine()

        results = engine.search_title(
            query,
            limit=limit,
        )

        normalized = normalize_results(
            results
        )

    except Exception as exc:

        return error_response(
            message=(
                "Title search failed."
            ),
            status_code=500,
            code="title_search_error",
            error=str(exc),
        )

    return success_response(
        data={
            "query": query,
            "results": normalized,
            "count": len(normalized),
        },
        message="Title search completed.",
    )


# ============================================================================
# SEARCH CONTENT
# ============================================================================

def search_content_endpoint(
    request: APIRequest,
) -> APIResponse:
    """
    Search indexed content directly.
    """

    query = _as_string(
        request.get(
            "query",
            "",
        )
    )

    limit = normalize_limit(
        request.get(
            "limit",
            DEFAULT_LIMIT,
        )
    )

    if not query:

        return error_response(
            message="A content query is required.",
            status_code=400,
            code="missing_query",
        )

    try:

        engine = get_search_engine()

        results = engine.search_content(
            query,
            limit=limit,
        )

        normalized = normalize_results(
            results
        )

    except Exception as exc:

        return error_response(
            message=(
                "Content search failed."
            ),
            status_code=500,
            code="content_search_error",
            error=str(exc),
        )

    return success_response(
        data={
            "query": query,
            "results": normalized,
            "count": len(normalized),
        },
        message="Content search completed.",
    )


# ============================================================================
# CATEGORY ENDPOINT
# ============================================================================

def category_endpoint(
    request: APIRequest,
) -> APIResponse:
    """
    Return all items belonging to a category.
    """

    category = _as_string(
        request.get(
            "category",
            "",
        )
    )

    if not category:

        return error_response(
            message="A category is required.",
            status_code=400,
            code="missing_category",
        )

    try:

        engine = get_search_engine()

        results = engine.by_category(
            category
        )

        normalized = normalize_results(
            results
        )

    except Exception as exc:

        return error_response(
            message=(
                "Category lookup failed."
            ),
            status_code=500,
            code="category_lookup_error",
            error=str(exc),
        )

    return success_response(
        data={
            "category": category,
            "results": normalized,
            "count": len(normalized),
        },
        message="Category lookup completed.",
    )


# ============================================================================
# TAG ENDPOINT
# ============================================================================

def tag_endpoint(
    request: APIRequest,
) -> APIResponse:
    """
    Return all items associated with a tag.
    """

    tag = _as_string(
        request.get(
            "tag",
            "",
        )
    )

    if not tag:

        return error_response(
            message="A tag is required.",
            status_code=400,
            code="missing_tag",
        )

    try:

        engine = get_search_engine()

        results = engine.by_tag(
            tag
        )

        normalized = normalize_results(
            results
        )

    except Exception as exc:

        return error_response(
            message=(
                "Tag lookup failed."
            ),
            status_code=500,
            code="tag_lookup_error",
            error=str(exc),
        )

    return success_response(
        data={
            "tag": tag,
            "results": normalized,
            "count": len(normalized),
        },
        message="Tag lookup completed.",
    )


# ============================================================================
# SEARCH STATISTICS
# ============================================================================

def statistics_endpoint(
    request: APIRequest,
) -> APIResponse:
    """
    Return search-index statistics.
    """

    try:

        engine = get_search_engine()

        statistics = engine.statistics()

    except Exception as exc:

        return error_response(
            message=(
                "Search statistics could not "
                "be retrieved."
            ),
            status_code=500,
            code="statistics_error",
            error=str(exc),
        )

    return success_response(
        data=statistics,
        message="Search statistics retrieved.",
    )


# ============================================================================
# SEARCH COUNT
# ============================================================================

def count_endpoint(
    request: APIRequest,
) -> APIResponse:
    """
    Return the number of indexed search items.
    """

    try:

        engine = get_search_engine()

        count = engine.count()

    except Exception as exc:

        return error_response(
            message=(
                "Search count could not "
                "be retrieved."
            ),
            status_code=500,
            code="count_error",
            error=str(exc),
        )

    return success_response(
        data={
            "count": count,
        },
        message="Search count retrieved.",
    )


# ============================================================================
# SEARCH CATALOGUE
# ============================================================================

def categories_endpoint(
    request: APIRequest,
) -> APIResponse:
    """
    Return available search categories.
    """

    try:

        engine = get_search_engine()

        categories = engine.categories()

    except Exception as exc:

        return error_response(
            message=(
                "Search categories could not "
                "be retrieved."
            ),
            status_code=500,
            code="categories_error",
            error=str(exc),
        )

    return success_response(
        data={
            "categories": categories,
            "count": len(
                categories
            ),
        },
        message="Search categories retrieved.",
    )


def tags_endpoint(
    request: APIRequest,
) -> APIResponse:
    """
    Return available search tags.
    """

    try:

        engine = get_search_engine()

        tags = engine.tags()

    except Exception as exc:

        return error_response(
            message=(
                "Search tags could not "
                "be retrieved."
            ),
            status_code=500,
            code="tags_error",
            error=str(exc),
        )

    return success_response(
        data={
            "tags": tags,
            "count": len(
                tags
            ),
        },
        message="Search tags retrieved.",
    )


# ============================================================================
# ITEM RETRIEVAL
# ============================================================================

def get_item_endpoint(
    request: APIRequest,
) -> APIResponse:
    """
    Retrieve one indexed item by ID.
    """

    raw_id = request.get(
        "id"
    )

    if raw_id is None:

        return error_response(
            message="An item ID is required.",
            status_code=400,
            code="missing_item_id",
        )

    try:

        item_id = int(
            raw_id
        )

    except (
        TypeError,
        ValueError,
    ):

        return error_response(
            message="Item ID must be an integer.",
            status_code=400,
            code="invalid_item_id",
        )

    try:

        engine = get_search_engine()

        item = engine.get(
            item_id
        )

    except Exception as exc:

        return error_response(
            message=(
                "Item lookup failed."
            ),
            status_code=500,
            code="item_lookup_error",
            error=str(exc),
        )

    if item is None:

        return error_response(
            message="Search item was not found.",
            status_code=404,
            code="item_not_found",
            item_id=item_id,
        )

    return success_response(
        data=normalize_result(
            item
        ),
        message="Search item retrieved.",
    )


# ============================================================================
# DUPLICATE CHECK
# ============================================================================

def duplicate_endpoint(
    request: APIRequest,
) -> APIResponse:
    """
    Check whether content already exists in the search index.
    """

    content = _as_string(
        request.get(
            "content",
            "",
        )
    )

    if not content:

        return error_response(
            message="Content is required.",
            status_code=400,
            code="missing_content",
        )

    try:

        engine = get_search_engine()

        duplicate = engine.has_duplicate(
            content
        )

    except Exception as exc:

        return error_response(
            message=(
                "Duplicate check failed."
            ),
            status_code=500,
            code="duplicate_check_error",
            error=str(exc),
        )

    return success_response(
        data={
            "duplicate": bool(
                duplicate
            ),
        },
        message="Duplicate check completed.",
    )


# ============================================================================
# ROUTER INTEGRATION
# ============================================================================

def register_search_routes(
    router: Any,
) -> Any:
    """
    Register search-related endpoints with the central API router.

    The router is passed in rather than imported globally, which
    keeps the API modules loosely coupled.
    """

    router.add_route(
        "/api/search",
        search_endpoint,
        methods=("GET", "POST"),
        name="search",
        description=(
            "Search indexed platform information."
        ),
        replace=True,
    )

    router.add_route(
        "/api/search/title",
        search_title_endpoint,
        methods=("GET", "POST"),
        name="search_title",
        description=(
            "Search indexed titles."
        ),
        replace=True,
    )

    router.add_route(
        "/api/search/content",
        search_content_endpoint,
        methods=("GET", "POST"),
        name="search_content",
        description=(
            "Search indexed content."
        ),
        replace=True,
    )

    router.add_route(
        "/api/search/category",
        category_endpoint,
        methods=("GET", "POST"),
        name="search_category",
        description=(
            "Retrieve items by category."
        ),
        replace=True,
    )

    router.add_route(
        "/api/search/tag",
        tag_endpoint,
        methods=("GET", "POST"),
        name="search_tag",
        description=(
            "Retrieve items by tag."
        ),
        replace=True,
    )

    router.add_route(
        "/api/search/statistics",
        statistics_endpoint,
        methods=("GET",),
        name="search_statistics",
        description=(
            "Return search-index statistics."
        ),
        replace=True,
    )

    router.add_route(
        "/api/search/count",
        count_endpoint,
        methods=("GET",),
        name="search_count",
        description=(
            "Return the number of indexed items."
        ),
        replace=True,
    )

    router.add_route(
        "/api/search/categories",
        categories_endpoint,
        methods=("GET",),
        name="search_categories",
        description=(
            "Return available search categories."
        ),
        replace=True,
    )

    router.add_route(
        "/api/search/tags",
        tags_endpoint,
        methods=("GET",),
        name="search_tags",
        description=(
            "Return available search tags."
        ),
        replace=True,
    )

    router.add_route(
        "/api/search/item",
        get_item_endpoint,
        methods=("GET", "POST"),
        name="search_item",
        description=(
            "Retrieve an indexed item by ID."
        ),
        replace=True,
    )

    router.add_route(
        "/api/search/duplicate",
        duplicate_endpoint,
        methods=("GET", "POST"),
        name="search_duplicate",
        description=(
            "Check whether content already exists."
        ),
        replace=True,
    )

    return router


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def search(
    query: str,
    *,
    category: Optional[str] = None,
    tags: Optional[list[str]] = None,
    limit: int = DEFAULT_LIMIT,
) -> APIResponse:
    """
    Perform an internal API-style search without constructing
    an APIRequest manually.

    Useful for tests and internal backend consumers.
    """

    request = APIRequest(
        method="POST",
        path="/api/search",
        body={
            "query": query,
            "category": category,
            "tags": tags or [],
            "limit": limit,
        },
    )

    return search_endpoint(
        request
    )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "MIN_LIMIT",
    "get_search_engine",
    "normalize_limit",
    "normalize_result",
    "normalize_results",
    "extract_search_parameters",
    "validate_search_parameters",
    "execute_search",
    "search_endpoint",
    "search_title_endpoint",
    "search_content_endpoint",
    "category_endpoint",
    "tag_endpoint",
    "statistics_endpoint",
    "count_endpoint",
    "categories_endpoint",
    "tags_endpoint",
    "get_item_endpoint",
    "duplicate_endpoint",
    "register_search_routes",
    "search",
]