"""
OurPlatform API Integration Layer
=================================

Purpose
-------
Central integration/orchestration layer for the API subsystem.

This module connects:

    Frontend
        |
        v
    API Routes
        |
        v
    API Integration
        |
        +----> API Schemas
        |
        +----> Search API
        |
        +----> Search Service
        |
        +----> Existing Search Engine
        |
        v
    Normalized API Response

Design goals
------------
1. Keep existing modules stable.
2. Avoid duplicating search logic.
3. Keep schema objects and route responses compatible.
4. Provide one central entry point for API initialization.
5. Provide health/status checks.
6. Provide safe dispatching.
7. Provide service discovery.
8. Make future API modules easy to attach.
9. Prevent circular imports through lazy loading.
10. Give the application one integration boundary.

This file is an ORCHESTRATION layer.

It should NOT contain:
    - HTML
    - frontend JavaScript
    - database implementation
    - search ranking algorithms
    - tokenization algorithms
    - indexing algorithms
    - retrieval algorithms

Those belong to their respective subsystems.
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# VERSION
# ============================================================================

INTEGRATION_VERSION = "1.0.0"

API_VERSION = "v1"

INTEGRATION_NAME = "OurPlatform API Integration"


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_SEARCH_LIMIT = 10

MAX_SEARCH_LIMIT = 500

HEALTHY = "healthy"

DEGRADED = "degraded"

UNAVAILABLE = "unavailable"


# ============================================================================
# TYPE ALIASES
# ============================================================================

Handler = Callable[..., Any]


# ============================================================================
# INTEGRATION STATE
# ============================================================================

@dataclass
class IntegrationState:
    """
    Runtime state of the API integration layer.
    """

    initialized: bool = False

    routes_registered: bool = False

    search_api_available: bool = False

    search_service_available: bool = False

    schemas_available: bool = False

    search_engine_available: bool = False

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    started_at: Optional[float] = None

    initialization_time_ms: Optional[float] = None

    def add_error(
        self,
        message: str,
    ) -> None:

        if message not in self.errors:
            self.errors.append(message)

    def add_warning(
        self,
        message: str,
    ) -> None:

        if message not in self.warnings:
            self.warnings.append(message)

    @property
    def status(self) -> str:

        if self.errors:
            return DEGRADED

        if self.initialized:
            return HEALTHY

        return UNAVAILABLE

    def to_dict(self) -> dict[str, Any]:

        return {
            "initialized": self.initialized,
            "routes_registered": self.routes_registered,
            "search_api_available": self.search_api_available,
            "search_service_available": self.search_service_available,
            "schemas_available": self.schemas_available,
            "search_engine_available": self.search_engine_available,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "started_at": self.started_at,
            "initialization_time_ms": self.initialization_time_ms,
            "status": self.status,
        }


# ============================================================================
# CENTRAL INTEGRATION CLASS
# ============================================================================

class APIIntegration:
    """
    Central API integration coordinator.

    One instance should normally be used by the application.
    """

    def __init__(
        self,
        *,
        router: Any = None,
    ) -> None:

        self.router = router

        self.state = IntegrationState()

        self.modules: dict[str, Any] = {}

        self.handlers: dict[str, Handler] = {}

        self.services: dict[str, Any] = {}

        self.metadata: dict[str, Any] = {
            "name": INTEGRATION_NAME,
            "version": INTEGRATION_VERSION,
            "api_version": API_VERSION,
        }

    # ========================================================================
    # MODULE LOADING
    # ========================================================================

    def load_module(
        self,
        module_name: str,
        *,
        required: bool = False,
    ) -> Optional[Any]:
        """
        Lazily import a module.

        Keeping imports here reduces circular-import problems.
        """

        if module_name in self.modules:
            return self.modules[module_name]

        try:

            module = importlib.import_module(
                module_name
            )

            self.modules[module_name] = module

            return module

        except Exception as exc:

            message = (
                f"Unable to load module "
                f"'{module_name}': {exc}"
            )

            if required:
                self.state.add_error(
                    message
                )

                logger.exception(
                    message
                )

            else:
                self.state.add_warning(
                    message
                )

                logger.warning(
                    message
                )

            return None

    # ========================================================================
    # SCHEMA LOADING
    # ========================================================================

    def load_schemas(self) -> Optional[Any]:
        """
        Load Api.schema / Api.schemas.

        Supports either naming convention so the integration
        layer remains tolerant of the existing project structure.
        """

        candidates = (
            "Api.schema",
            "Api.schemas",
            "api.schema",
            "api.schemas",
        )

        for module_name in candidates:

            module = self.load_module(
                module_name
            )

            if module is not None:

                self.state.schemas_available = True

                return module

        self.state.add_warning(
            "API schema module could not be loaded."
        )

        return None

    # ========================================================================
    # SEARCH API LOADING
    # ========================================================================

    def load_search_api(self) -> Optional[Any]:
        """
        Load the search API adapter.
        """

        candidates = (
            "Api.search_api",
            "api.search_api",
        )

        for module_name in candidates:

            module = self.load_module(
                module_name
            )

            if module is not None:

                self.state.search_api_available = True

                return module

        self.state.add_error(
            "Search API module could not be loaded."
        )

        return None

    # ========================================================================
    # SEARCH SERVICE LOADING
    # ========================================================================

    def load_search_service(self) -> Optional[Any]:
        """
        Load the central search service if present.

        Search service is optional during early startup because
        Search_api.py can still operate directly against the
        existing search engine.
        """

        candidates = (
            "Api.search_service",
            "api.search_service",
        )

        for module_name in candidates:

            module = self.load_module(
                module_name
            )

            if module is not None:

                self.state.search_service_available = True

                return module

        self.state.add_warning(
            "Search service module is not available yet."
        )

        return None

    # ========================================================================
    # SEARCH ENGINE CHECK
    # ========================================================================

    def check_search_engine(self) -> bool:
        """
        Check whether the existing search engine can be imported.

        The integration layer does not replace the search engine.
        """

        candidates = (
            "search.search",
            "Search.search",
        )

        for module_name in candidates:

            module = self.load_module(
                module_name
            )

            if module is None:
                continue

            if hasattr(
                module,
                "search",
            ):

                self.state.search_engine_available = True

                self.services[
                    "search_engine"
                ] = getattr(
                    module,
                    "search",
                )

                return True

        self.state.add_error(
            "Existing search engine could not be located."
        )

        return False

    # ========================================================================
    # HANDLER REGISTRATION
    # ========================================================================

    def register_handler(
        self,
        name: str,
        handler: Handler,
    ) -> None:
        """
        Register an integration-level handler.
        """

        if not name:
            raise ValueError(
                "Handler name cannot be empty."
            )

        if not callable(handler):
            raise TypeError(
                "Handler must be callable."
            )

        self.handlers[name] = handler

    # ========================================================================
    # SEARCH HANDLERS
    # ========================================================================

    def register_search_handlers(self) -> int:
        """
        Discover and register search API handlers.

        Returns the number of handlers registered.
        """

        search_api = (
            self.load_search_api()
        )

        if search_api is None:
            return 0

        handler_names = (
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
        )

        registered = 0

        for name in handler_names:

            handler = getattr(
                search_api,
                name,
                None,
            )

            if callable(handler):

                self.register_handler(
                    name,
                    handler,
                )

                registered += 1

        return registered

    # ========================================================================
    # ROUTE REGISTRATION
    # ========================================================================

    def register_routes(self) -> bool:
        """
        Register API routes with the application's router.

        Search_api.py owns the actual route definitions.
        This layer only activates them.
        """

        if self.router is None:

            self.state.add_warning(
                "No router supplied; routes were not registered."
            )

            return False

        search_api = (
            self.load_search_api()
        )

        if search_api is None:
            return False

        register_function = getattr(
            search_api,
            "register_search_routes",
            None,
        )

        if not callable(
            register_function
        ):

            self.state.add_error(
                "Search API does not expose register_search_routes()."
            )

            return False

        try:

            register_function(
                self.router
            )

            self.state.routes_registered = True

            return True

        except Exception as exc:

            self.state.add_error(
                f"Route registration failed: {exc}"
            )

            logger.exception(
                "API route registration failed."
            )

            return False

    # ========================================================================
    # SERVICE REGISTRATION
    # ========================================================================

    def register_services(self) -> None:
        """
        Register available API services.

        This intentionally stores references instead of creating
        duplicate service instances.
        """

        search_service = (
            self.load_search_service()
        )

        if search_service is not None:

            self.services[
                "search_service"
            ] = search_service

        search_api = (
            self.load_search_api()
        )

        if search_api is not None:

            self.services[
                "search_api"
            ] = search_api

        schemas = (
            self.load_schemas()
        )

        if schemas is not None:

            self.services[
                "schemas"
            ] = schemas

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def initialize(self) -> IntegrationState:
        """
        Initialize the complete API integration layer.

        Initialization order:

            1. Schemas
            2. Search API
            3. Search service
            4. Search engine
            5. Search handlers
            6. Routes
        """

        if self.state.initialized:
            return self.state

        started = time.perf_counter()

        self.state.started_at = time.time()

        self.load_schemas()

        self.load_search_api()

        self.load_search_service()

        self.check_search_engine()

        self.register_services()

        self.register_search_handlers()

        self.register_routes()

        self.state.initialized = True

        elapsed = (
            time.perf_counter()
            - started
        ) * 1000

        self.state.initialization_time_ms = (
            round(
                elapsed,
                3,
            )
        )

        return self.state

    # ========================================================================
    # HEALTH
    # ========================================================================

    def health(self) -> dict[str, Any]:
        """
        Return integration health information.
        """

        return {
            "name": INTEGRATION_NAME,
            "version": INTEGRATION_VERSION,
            "api_version": API_VERSION,
            "status": self.state.status,
            "state": self.state.to_dict(),
            "handlers": sorted(
                self.handlers.keys()
            ),
            "services": sorted(
                self.services.keys()
            ),
        }

    # ========================================================================
    # CAPABILITIES
    # ========================================================================

    def capabilities(self) -> dict[str, Any]:
        """
        Describe what the current API layer can provide.
        """

        return {
            "search": (
                "search_endpoint"
                in self.handlers
            ),
            "title_search": (
                "search_title_endpoint"
                in self.handlers
            ),
            "content_search": (
                "search_content_endpoint"
                in self.handlers
            ),
            "category_search": (
                "category_endpoint"
                in self.handlers
            ),
            "tag_search": (
                "tag_endpoint"
                in self.handlers
            ),
            "statistics": (
                "statistics_endpoint"
                in self.handlers
            ),
            "count": (
                "count_endpoint"
                in self.handlers
            ),
            "categories": (
                "categories_endpoint"
                in self.handlers
            ),
            "tags": (
                "tags_endpoint"
                in self.handlers
            ),
            "item_lookup": (
                "get_item_endpoint"
                in self.handlers
            ),
            "duplicate_check": (
                "duplicate_endpoint"
                in self.handlers
            ),
            "schemas": self.state.schemas_available,
            "search_service": self.state.search_service_available,
            "search_engine": self.state.search_engine_available,
        }

    # ========================================================================
    # HANDLER DISPATCH
    # ========================================================================

    def dispatch(
        self,
        handler_name: str,
        request: Any,
    ) -> Any:
        """
        Dispatch an API request to a registered handler.
        """

        handler = self.handlers.get(
            handler_name
        )

        if handler is None:

            raise KeyError(
                f"Unknown API handler: {handler_name}"
            )

        return handler(
            request
        )

    # ========================================================================
    # SEARCH DISPATCH
    # ========================================================================

    def search(
        self,
        request: Any,
    ) -> Any:
        """
        Dispatch to the primary search endpoint.
        """

        return self.dispatch(
            "search_endpoint",
            request,
        )

    # ========================================================================
    # INTERNAL SEARCH
    # ========================================================================

    def search_internal(
        self,
        query: str,
        *,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> Any:
        """
        Perform an internal API search.

        Uses Search_api.search() when available.

        This is useful for backend consumers that want the same
        API contract without manually constructing an APIRequest.
        """

        search_api = (
            self.load_search_api()
        )

        if search_api is None:

            raise RuntimeError(
                "Search API is unavailable."
            )

        search_function = getattr(
            search_api,
            "search",
            None,
        )

        if not callable(
            search_function
        ):

            raise RuntimeError(
                "Search API does not expose search()."
            )

        return search_function(
            query,
            category=category,
            tags=tags or [],
            limit=limit,
        )

    # ========================================================================
    # SCHEMA FACTORIES
    # ========================================================================

    def build_search_request(
        self,
        data: Mapping[str, Any],
    ) -> Any:
        """
        Build a canonical SearchRequest when schemas are available.
        """

        schemas = (
            self.load_schemas()
        )

        if schemas is None:

            raise RuntimeError(
                "API schemas are unavailable."
            )

        factory = getattr(
            schemas,
            "build_search_request",
            None,
        )

        if callable(factory):

            return factory(
                data
            )

        search_request = getattr(
            schemas,
            "SearchRequest",
            None,
        )

        if search_request is None:

            raise RuntimeError(
                "SearchRequest is unavailable."
            )

        from_dict = getattr(
            search_request,
            "from_dict",
            None,
        )

        if callable(from_dict):

            return from_dict(
                data
            )

        return search_request(
            **dict(data)
        )

    # ========================================================================
    # RESPONSE NORMALIZATION
    # ========================================================================

    def normalize_response(
        self,
        response: Any,
    ) -> Any:
        """
        Convert supported response objects into dictionaries
        where possible.

        This prevents the frontend from needing to understand
        internal Python dataclasses.
        """

        if response is None:
            return None

        if isinstance(
            response,
            Mapping,
        ):

            return dict(
                response
            )

        to_dict = getattr(
            response,
            "to_dict",
            None,
        )

        if callable(to_dict):

            return to_dict()

        if hasattr(
            response,
            "__dict__",
        ):

            return dict(
                response.__dict__
            )

        return response

    # ========================================================================
    # DIAGNOSTICS
    # ========================================================================

    def diagnostics(self) -> dict[str, Any]:
        """
        Return detailed integration diagnostics.
        """

        return {
            "integration": {
                "name": INTEGRATION_NAME,
                "version": INTEGRATION_VERSION,
                "api_version": API_VERSION,
            },
            "state": self.state.to_dict(),
            "capabilities": self.capabilities(),
            "modules": sorted(
                self.modules.keys()
            ),
            "handlers": sorted(
                self.handlers.keys()
            ),
            "services": sorted(
                self.services.keys()
            ),
        }


# ============================================================================
# DEFAULT INTEGRATION INSTANCE
# ============================================================================

_integration: Optional[
    APIIntegration
] = None


# ============================================================================
# GLOBAL ACCESSOR
# ============================================================================

def get_integration(
    router: Any = None,
) -> APIIntegration:
    """
    Return the shared APIIntegration instance.

    If a router is supplied later, attach it to the existing
    integration object.
    """

    global _integration

    if _integration is None:

        _integration = APIIntegration(
            router=router
        )

    elif router is not None:

        _integration.router = router

    return _integration


# ============================================================================
# INITIALIZATION HELPER
# ============================================================================

def initialize_api(
    router: Any = None,
) -> IntegrationState:
    """
    Initialize the shared API integration layer.
    """

    integration = get_integration(
        router
    )

    return integration.initialize()


# ============================================================================
# HEALTH HELPER
# ============================================================================

def api_health(
    router: Any = None,
) -> dict[str, Any]:
    """
    Return API integration health.
    """

    integration = get_integration(
        router
    )

    if not integration.state.initialized:

        integration.initialize()

    return integration.health()


# ============================================================================
# CAPABILITY HELPER
# ============================================================================

def api_capabilities(
    router: Any = None,
) -> dict[str, Any]:
    """
    Return currently available API capabilities.
    """

    integration = get_integration(
        router
    )

    if not integration.state.initialized:

        integration.initialize()

    return integration.capabilities()


# ============================================================================
# SEARCH HELPER
# ============================================================================

def api_search(
    query: str,
    *,
    category: Optional[str] = None,
    tags: Optional[list[str]] = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> Any:
    """
    Perform an internal search through the integrated API layer.
    """

    integration = get_integration()

    if not integration.state.initialized:

        integration.initialize()

    return integration.search_internal(
        query,
        category=category,
        tags=tags,
        limit=limit,
    )


# ============================================================================
# RESET
# ============================================================================

def reset_integration() -> None:
    """
    Reset the shared integration instance.

    Primarily useful for tests and controlled application reloads.
    """

    global _integration

    _integration = None


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "INTEGRATION_VERSION",
    "API_VERSION",
    "INTEGRATION_NAME",
    "IntegrationState",
    "APIIntegration",
    "get_integration",
    "initialize_api",
    "api_health",
    "api_capabilities",
    "api_search",
    "reset_integration",
]