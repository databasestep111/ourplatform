"""
OurPlatform API routing layer.

This module provides the central routing/dispatch layer for the API.

Responsibilities
----------------
- Register API endpoints.
- Dispatch incoming requests.
- Normalize request data.
- Normalize API responses.
- Handle common API errors.
- Provide health and information endpoints.
- Connect specialized API modules without duplicating their logic.

Architecture
------------

    Frontend
        |
        v
    HTTP/API request
        |
        v
    routes.py
        |
        +--> search_api
        +--> query_api
        +--> ranking_api
        +--> retrieval_api
        +--> filters_api
        +--> suggestions_api
        +--> research_api
        |
        v
    Existing backend systems

This module is deliberately an orchestration layer.
Search, ranking, research, and other backend logic should remain
inside their respective modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional


# ============================================================================
# TYPES
# ============================================================================

Handler = Callable[..., Any]


# ============================================================================
# REQUEST OBJECT
# ============================================================================

@dataclass
class APIRequest:
    """
    Normalized representation of an API request.

    The web framework can translate its native request object into
    this structure before dispatching it.

    Keeping this representation framework-independent makes the
    search API easier to test outside the web server.
    """

    method: str = "GET"

    path: str = "/"

    query: Dict[str, Any] = field(
        default_factory=dict
    )

    body: Dict[str, Any] = field(
        default_factory=dict
    )

    headers: Dict[str, str] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def normalized_method(self) -> str:
        """
        Return an uppercase HTTP method.
        """

        return (
            self.method
            or "GET"
        ).upper()

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve a value from the body first, then query data.
        """

        if key in self.body:
            return self.body[key]

        return self.query.get(
            key,
            default,
        )


# ============================================================================
# RESPONSE OBJECT
# ============================================================================

@dataclass
class APIResponse:
    """
    Standardized API response.

    Every endpoint can return this structure before the web framework
    converts it into its native JSON response.
    """

    data: Any = None

    status_code: int = 200

    message: str = "OK"

    success: bool = True

    errors: list[Dict[str, Any]] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the response into a JSON-friendly dictionary.
        """

        return {
            "success": self.success,
            "status": self.status_code,
            "message": self.message,
            "data": self.data,
            "errors": self.errors,
            "metadata": self.metadata,
        }


# ============================================================================
# ROUTE DEFINITION
# ============================================================================

@dataclass
class RouteDefinition:
    """
    Describes one API endpoint.
    """

    path: str

    handler: Handler

    methods: tuple[str, ...] = (
        "GET",
    )

    name: Optional[str] = None

    description: str = ""

    enabled: bool = True

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def supports(
        self,
        method: str,
    ) -> bool:
        """
        Determine whether the route accepts the given method.
        """

        return (
            method.upper()
            in {
                item.upper()
                for item in self.methods
            }
        )


# ============================================================================
# ROUTER
# ============================================================================

class APIRouter:
    """
    Lightweight framework-independent API router.

    It provides route registration and dispatch without forcing
    the search system to depend directly on Flask, FastAPI, Django,
    or another framework.
    """

    def __init__(self) -> None:

        self.routes: Dict[
            str,
            RouteDefinition
        ] = {}

        self.middleware: list[
            Callable[
                [APIRequest],
                APIRequest
            ]
        ] = []

        self.before_handlers: list[
            Callable[
                [APIRequest],
                Optional[APIResponse]
            ]
        ] = []

        self.after_handlers: list[
            Callable[
                [APIRequest, APIResponse],
                APIResponse
            ]
        ] = []

        self.error_handlers: Dict[
            type[Exception],
            Callable[
                [Exception, APIRequest],
                APIResponse
            ]
        ] = {}

    # ---------------------------------------------------------------------
    # ROUTE REGISTRATION
    # ---------------------------------------------------------------------

    def add_route(
        self,
        path: str,
        handler: Handler,
        *,
        methods: tuple[str, ...] = ("GET",),
        name: Optional[str] = None,
        description: str = "",
        enabled: bool = True,
        replace: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RouteDefinition:
        """
        Register an API route.
        """

        normalized_path = self._normalize_path(
            path
        )

        if (
            normalized_path in self.routes
            and not replace
        ):
            raise ValueError(
                f"Route already exists: "
                f"{normalized_path}"
            )

        route = RouteDefinition(
            path=normalized_path,
            handler=handler,
            methods=tuple(
                method.upper()
                for method in methods
            ),
            name=name,
            description=description,
            enabled=enabled,
            metadata=dict(
                metadata or {}
            ),
        )

        self.routes[
            normalized_path
        ] = route

        return route

    def remove_route(
        self,
        path: str,
    ) -> bool:
        """
        Remove a registered route.
        """

        normalized_path = self._normalize_path(
            path
        )

        if normalized_path not in self.routes:
            return False

        del self.routes[
            normalized_path
        ]

        return True

    def get_route(
        self,
        path: str,
    ) -> Optional[RouteDefinition]:
        """
        Retrieve a route definition.
        """

        return self.routes.get(
            self._normalize_path(path)
        )

    # ---------------------------------------------------------------------
    # MIDDLEWARE
    # ---------------------------------------------------------------------

    def add_middleware(
        self,
        middleware: Callable[
            [APIRequest],
            APIRequest
        ],
    ) -> None:
        """
        Add request middleware.
        """

        self.middleware.append(
            middleware
        )

    def add_before_handler(
        self,
        handler: Callable[
            [APIRequest],
            Optional[APIResponse]
        ],
    ) -> None:
        """
        Add a handler executed before route dispatch.
        """

        self.before_handlers.append(
            handler
        )

    def add_after_handler(
        self,
        handler: Callable[
            [APIRequest, APIResponse],
            APIResponse
        ],
    ) -> None:
        """
        Add a handler executed after route dispatch.
        """

        self.after_handlers.append(
            handler
        )

    # ---------------------------------------------------------------------
    # ERROR HANDLING
    # ---------------------------------------------------------------------

    def register_error_handler(
        self,
        exception_type: type[Exception],
        handler: Callable[
            [Exception, APIRequest],
            APIResponse
        ],
    ) -> None:
        """
        Register a custom exception handler.
        """

        self.error_handlers[
            exception_type
        ] = handler

    # ---------------------------------------------------------------------
    # DISPATCH
    # ---------------------------------------------------------------------

    def dispatch(
        self,
        request: APIRequest,
    ) -> APIResponse:
        """
        Dispatch an API request.

        Processing order:

            request
              ↓
            middleware
              ↓
            before handlers
              ↓
            route lookup
              ↓
            method validation
              ↓
            endpoint handler
              ↓
            after handlers
              ↓
            response
        """

        try:

            request = self._apply_middleware(
                request
            )

            early_response = (
                self._run_before_handlers(
                    request
                )
            )

            if early_response is not None:
                return early_response

            route = self.get_route(
                request.path
            )

            if route is None:

                return error_response(
                    message="API route not found.",
                    status_code=404,
                    code="route_not_found",
                    path=request.path,
                )

            if not route.enabled:

                return error_response(
                    message="API route is disabled.",
                    status_code=503,
                    code="route_disabled",
                    path=request.path,
                )

            method = (
                request.normalized_method()
            )

            if not route.supports(method):

                return error_response(
                    message=(
                        "HTTP method is not supported "
                        "by this endpoint."
                    ),
                    status_code=405,
                    code="method_not_allowed",
                    method=method,
                    allowed_methods=list(
                        route.methods
                    ),
                )

            result = route.handler(
                request
            )

            response = normalize_response(
                result
            )

            response.metadata.setdefault(
                "route",
                route.path,
            )

            response.metadata.setdefault(
                "method",
                method,
            )

            response.metadata.setdefault(
                "timestamp",
                utc_timestamp(),
            )

            return self._run_after_handlers(
                request,
                response,
            )

        except Exception as exc:

            return self._handle_exception(
                exc,
                request,
            )

    # ---------------------------------------------------------------------
    # INTERNAL ROUTING
    # ---------------------------------------------------------------------

    def _normalize_path(
        self,
        path: str,
    ) -> str:
        """
        Normalize API paths.
        """

        if not path:
            return "/"

        path = str(path).strip()

        if not path.startswith("/"):
            path = "/" + path

        if (
            len(path) > 1
            and path.endswith("/")
        ):
            path = path[:-1]

        return path

    def _apply_middleware(
        self,
        request: APIRequest,
    ) -> APIRequest:
        """
        Apply request middleware in registration order.
        """

        current = request

        for middleware in self.middleware:

            result = middleware(
                current
            )

            if result is None:
                continue

            current = result

        return current

    def _run_before_handlers(
        self,
        request: APIRequest,
    ) -> Optional[APIResponse]:
        """
        Execute pre-dispatch handlers.
        """

        for handler in self.before_handlers:

            response = handler(
                request
            )

            if response is not None:
                return normalize_response(
                    response
                )

        return None

    def _run_after_handlers(
        self,
        request: APIRequest,
        response: APIResponse,
    ) -> APIResponse:
        """
        Execute post-dispatch handlers.
        """

        current = response

        for handler in self.after_handlers:

            result = handler(
                request,
                current,
            )

            if result is not None:
                current = normalize_response(
                    result
                )

        return current

    def _handle_exception(
        self,
        exception: Exception,
        request: APIRequest,
    ) -> APIResponse:
        """
        Resolve an exception through the registered error handlers.
        """

        exception_type = type(
            exception
        )

        handler = self.error_handlers.get(
            exception_type
        )

        if handler is not None:

            try:

                return normalize_response(
                    handler(
                        exception,
                        request,
                    )
                )

            except Exception:
                pass

        return error_response(
            message="Internal API error.",
            status_code=500,
            code="internal_error",
        )

    # ---------------------------------------------------------------------
    # INTROSPECTION
    # ---------------------------------------------------------------------

    def describe(self) -> list[Dict[str, Any]]:
        """
        Return a machine-readable description of registered routes.
        """

        result = []

        for route in self.routes.values():

            result.append(
                {
                    "path": route.path,
                    "name": route.name,
                    "methods": list(
                        route.methods
                    ),
                    "description": (
                        route.description
                    ),
                    "enabled": route.enabled,
                    "metadata": dict(
                        route.metadata
                    ),
                }
            )

        return result


# ============================================================================
# RESPONSE HELPERS
# ============================================================================

def normalize_response(
    result: Any,
) -> APIResponse:
    """
    Convert common endpoint return values into APIResponse.
    """

    if isinstance(
        result,
        APIResponse,
    ):
        return result

    if isinstance(
        result,
        tuple,
    ):

        if len(result) == 2:

            data, status = result

            return APIResponse(
                data=data,
                status_code=int(status),
                success=(
                    200
                    <= int(status)
                    < 400
                ),
            )

        if len(result) == 3:

            data, status, message = result

            return APIResponse(
                data=data,
                status_code=int(status),
                message=str(message),
                success=(
                    200
                    <= int(status)
                    < 400
                ),
            )

    return APIResponse(
        data=result
    )


def success_response(
    data: Any = None,
    *,
    message: str = "OK",
    status_code: int = 200,
    **metadata: Any,
) -> APIResponse:
    """
    Create a successful API response.
    """

    return APIResponse(
        data=data,
        status_code=status_code,
        message=message,
        success=True,
        metadata=metadata,
    )


def error_response(
    *,
    message: str,
    status_code: int = 400,
    code: str = "api_error",
    **details: Any,
) -> APIResponse:
    """
    Create a standardized API error response.
    """

    return APIResponse(
        data=None,
        status_code=status_code,
        message=message,
        success=False,
        errors=[
            {
                "code": code,
                "message": message,
                "details": details,
            }
        ],
    )


# ============================================================================
# GENERAL REQUEST NORMALIZATION
# ============================================================================

def normalize_request(
    request: Any,
) -> APIRequest:
    """
    Convert common request representations into APIRequest.

    This makes it possible to adapt Flask/FastAPI/custom framework
    requests later without changing the router itself.
    """

    if isinstance(
        request,
        APIRequest,
    ):
        return request

    if isinstance(
        request,
        Mapping,
    ):

        return APIRequest(
            method=str(
                request.get(
                    "method",
                    "GET",
                )
            ),
            path=str(
                request.get(
                    "path",
                    "/",
                )
            ),
            query=dict(
                request.get(
                    "query",
                    {},
                )
                or {}
            ),
            body=dict(
                request.get(
                    "body",
                    {},
                )
                or {}
            ),
            headers=dict(
                request.get(
                    "headers",
                    {},
                )
                or {}
            ),
            metadata=dict(
                request.get(
                    "metadata",
                    {},
                )
                or {}
            ),
        )

    method = getattr(
        request,
        "method",
        "GET",
    )

    path = getattr(
        request,
        "path",
        "/",
    )

    query = getattr(
        request,
        "query",
        {},
    )

    body = getattr(
        request,
        "body",
        {},
    )

    headers = getattr(
        request,
        "headers",
        {},
    )

    return APIRequest(
        method=str(
            method
        ),
        path=str(
            path
        ),
        query=dict(
            query or {}
        ),
        body=dict(
            body or {}
        ),
        headers=dict(
            headers or {}
        ),
    )


# ============================================================================
# STANDARD API ENDPOINTS
# ============================================================================

def health_endpoint(
    request: APIRequest,
) -> APIResponse:
    """
    Basic API health endpoint.

    Does not perform a search.
    """

    try:

        from . import health_check

        return success_response(
            data=health_check(),
            message="API is healthy.",
        )

    except Exception as exc:

        return error_response(
            message="API health check failed.",
            status_code=500,
            code="health_check_failed",
            error=str(exc),
        )


def info_endpoint(
    request: APIRequest,
) -> APIResponse:
    """
    Return API package metadata.
    """

    try:

        from . import get_api_info

        return success_response(
            data=get_api_info(),
            message="API information retrieved.",
        )

    except Exception as exc:

        return error_response(
            message="Unable to retrieve API information.",
            status_code=500,
            code="api_info_failed",
            error=str(exc),
        )


def not_implemented_endpoint(
    request: APIRequest,
) -> APIResponse:
    """
    Temporary endpoint handler.

    Specialized modules will replace these handlers as they
    are connected to the real backend systems.
    """

    return error_response(
        message=(
            "This API capability has not yet been connected "
            "to its backend service."
        ),
        status_code=501,
        code="not_implemented",
    )


# ============================================================================
# ROUTER CONSTRUCTION
# ============================================================================

def create_router() -> APIRouter:
    """
    Create the central API router.

    Specialized API modules can register their handlers onto
    this router as they are implemented.
    """

    router = APIRouter()

    router.add_route(
        "/api/health",
        health_endpoint,
        methods=("GET",),
        name="health",
        description=(
            "Return API health information."
        ),
    )

    router.add_route(
        "/api/info",
        info_endpoint,
        methods=("GET",),
        name="info",
        description=(
            "Return API package information."
        ),
    )

    router.add_route(
        "/api/search",
        not_implemented_endpoint,
        methods=("GET", "POST"),
        name="search",
        description=(
            "Search indexed platform information."
        ),
    )

    router.add_route(
        "/api/query",
        not_implemented_endpoint,
        methods=("GET", "POST"),
        name="query",
        description=(
            "Analyse and normalize search queries."
        ),
    )

    router.add_route(
        "/api/ranking",
        not_implemented_endpoint,
        methods=("GET", "POST"),
        name="ranking",
        description=(
            "Rank search candidates."
        ),
    )

    router.add_route(
        "/api/retrieval",
        not_implemented_endpoint,
        methods=("GET", "POST"),
        name="retrieval",
        description=(
            "Retrieve search candidates."
        ),
    )

    router.add_route(
        "/api/filters",
        not_implemented_endpoint,
        methods=("GET", "POST"),
        name="filters",
        description=(
            "Apply structured search filters."
        ),
    )

    router.add_route(
        "/api/suggestions",
        not_implemented_endpoint,
        methods=("GET", "POST"),
        name="suggestions",
        description=(
            "Generate search suggestions."
        ),
    )

    router.add_route(
        "/api/research",
        not_implemented_endpoint,
        methods=("GET", "POST"),
        name="research",
        description=(
            "Connect search with research systems."
        ),
    )

    router.add_route(
        "/api/results",
        not_implemented_endpoint,
        methods=("GET", "POST"),
        name="results",
        description=(
            "Return structured search results."
        ),
    )

    return router


# ============================================================================
# DEFAULT ROUTER
# ============================================================================

router = create_router()


# ============================================================================
# PUBLIC DISPATCH FUNCTIONS
# ============================================================================

def dispatch(
    request: Any,
) -> APIResponse:
    """
    Dispatch a request through the default API router.
    """

    normalized = normalize_request(
        request
    )

    return router.dispatch(
        normalized
    )


def api_call(
    path: str,
    *,
    method: str = "GET",
    query: Optional[Mapping[str, Any]] = None,
    body: Optional[Mapping[str, Any]] = None,
    headers: Optional[Mapping[str, str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> APIResponse:
    """
    Convenience function for internal API calls and testing.
    """

    request = APIRequest(
        method=method,
        path=path,
        query=dict(
            query or {}
        ),
        body=dict(
            body or {}
        ),
        headers=dict(
            headers or {}
        ),
        metadata=dict(
            metadata or {}
        ),
    )

    return dispatch(
        request
    )


# ============================================================================
# ROUTE INSPECTION
# ============================================================================

def list_routes() -> list[Dict[str, Any]]:
    """
    Return all registered routes.
    """

    return router.describe()


def route_exists(
    path: str,
) -> bool:
    """
    Determine whether a route exists.
    """

    return (
        router.get_route(
            path
        )
        is not None
    )


# ============================================================================
# TIME
# ============================================================================

def utc_timestamp() -> str:
    """
    Return an ISO-8601 UTC timestamp.
    """

    return (
        datetime.now(
            timezone.utc
        ).isoformat()
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "APIRequest",
    "APIResponse",
    "RouteDefinition",
    "APIRouter",
    "normalize_response",
    "success_response",
    "error_response",
    "normalize_request",
    "health_endpoint",
    "info_endpoint",
    "create_router",
    "router",
    "dispatch",
    "api_call",
    "list_routes",
    "route_exists",
]