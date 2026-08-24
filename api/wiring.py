"""
OurPlatform API integration layer.

Connects the central API router to the rest of the application.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .routes import (
    APIResponse,
    api_call,
    create_router,
    dispatch,
    router,
)


# ============================================================================
# ROUTER
# ============================================================================

def get_router():
    """
    Return the application's central API router.
    """

    return router


# ============================================================================
# DISPATCH
# ============================================================================

def handle_request(
    request: Any,
) -> APIResponse:
    """
    Send an incoming request through the API router.
    """

    return dispatch(request)


# ============================================================================
# INTERNAL API CALL
# ============================================================================

def call_api(
    path: str,
    *,
    method: str = "GET",
    query: Optional[Mapping[str, Any]] = None,
    body: Optional[Mapping[str, Any]] = None,
    headers: Optional[Mapping[str, str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> APIResponse:
    """
    Convenience wrapper for internal API calls.
    """

    return api_call(
        path,
        method=method,
        query=query,
        body=body,
        headers=headers,
        metadata=metadata,
    )


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_api():
    """
    Initialize and return the application's API router.

    routes.py is responsible for registering the specialized
    API modules, including search_api.py.
    """

    return create_router()


# ============================================================================
# SEARCH CHECK
# ============================================================================

def search(
    query: str,
    *,
    category: Optional[str] = None,
    tags: Optional[list[str]] = None,
    limit: int = 10,
) -> APIResponse:
    """
    Run a search through the connected API.

    This ultimately reaches search_api.py and then
    the existing search backend.
    """

    return call_api(
        "/api/search",
        method="POST",
        body={
            "query": query,
            "category": category,
            "tags": tags or [],
            "limit": limit,
        },
    )


__all__ = [
    "router",
    "get_router",
    "handle_request",
    "call_api",
    "initialize_api",
    "search",
]