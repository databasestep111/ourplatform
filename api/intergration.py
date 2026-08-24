"""
OurPlatform API Integration Layer
=================================

Purpose
-------

This module is the composition and integration layer for the API package.

It is responsible for connecting the API components together without
duplicating their implementation.

Architecture
------------

    Web Application
          |
          v
       routes
          |
          v
     search_api
          |
          v
    search_service
          |
          v
      Search/*
          |
          v
       schemas
          |
          v
       Response


The important rule is:

    This file CONNECTS components.
    It does not replace them.

Responsibilities
----------------

- Discover API components.
- Build shared API dependencies.
- Configure SearchService.
- Register Search components.
- Expose application-level API state.
- Provide health and readiness information.
- Keep API initialization in one place.
- Prevent route modules from owning application wiring.
- Provide a stable integration boundary for future components.

This module should be imported by the application startup layer rather
than having every API module manually initialize every dependency.
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(
    "ourplatform.api.integration"
)


# ============================================================================
# VERSION INFORMATION
# ============================================================================

INTEGRATION_VERSION = "1.0.0"

API_LAYER_NAME = "OurPlatform API"

API_LAYER_STATUS = "initializing"


# ============================================================================
# OPTIONAL IMPORTS
# ============================================================================

#
# The integration layer should remain importable even when one optional
# component has not been connected yet.
#
# This is intentional.
#
# During development, different parts of the platform may become available
# at different times.
#

try:
    from .search_service import (
        SearchService,
        SearchComponentRegistry,
        SearchServiceConfig,
        get_search_service,
        configure_search_service,
        register_components,
        health_check as search_health_check,
    )

    SEARCH_SERVICE_AVAILABLE = True

except Exception as exc:

    SearchService = None
    SearchComponentRegistry = None
    SearchServiceConfig = None
    get_search_service = None
    configure_search_service = None
    register_components = None
    search_health_check = None

    SEARCH_SERVICE_AVAILABLE = False

    logger.warning(
        "Search service could not be imported: %s",
        exc,
    )


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class IntegrationComponent:
    """
    Describes one component registered with the API integration layer.
    """

    name: str

    component: Any = None

    required: bool = False

    initialized: bool = False

    healthy: bool = False

    error: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def status(self) -> Dict[str, Any]:

        return {
            "name": self.name,
            "required": self.required,
            "initialized": self.initialized,
            "healthy": self.healthy,
            "available": self.component is not None,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class IntegrationState:
    """
    Runtime state for the complete API integration layer.
    """

    initialized: bool = False

    healthy: bool = False

    ready: bool = False

    started_at: Optional[str] = None

    initialized_at: Optional[str] = None

    components: Dict[
        str,
        IntegrationComponent
    ] = field(
        default_factory=dict
    )

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    initialization_count: int = 0

    def add_component(
        self,
        name: str,
        component: Any = None,
        *,
        required: bool = False,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> IntegrationComponent:

        record = IntegrationComponent(
            name=name,
            component=component,
            required=required,
            metadata=dict(
                metadata or {}
            ),
        )

        self.components[name] = record

        return record

    def component(
        self,
        name: str,
    ) -> Optional[
        IntegrationComponent
    ]:

        return self.components.get(
            name
        )

    def status(self) -> Dict[str, Any]:

        return {
            "initialized": self.initialized,
            "healthy": self.healthy,
            "ready": self.ready,
            "started_at": self.started_at,
            "initialized_at": self.initialized_at,
            "initialization_count": (
                self.initialization_count
            ),
            "components": {
                name: record.status()
                for name, record
                in self.components.items()
            },
            "warnings": list(
                self.warnings
            ),
            "errors": list(
                self.errors
            ),
        }


# ============================================================================
# GLOBAL INTEGRATION STATE
# ============================================================================

_state = IntegrationState()

_lock = threading.RLock()

_initialized = False


# ============================================================================
# TIME HELPERS
# ============================================================================

def _utc_now() -> str:
    """
    Return a stable UTC timestamp for diagnostics.
    """

    return (
        datetime.now(
            timezone.utc
        ).isoformat()
    )


# ============================================================================
# COMPONENT REGISTRATION
# ============================================================================

def register_api_component(
    name: str,
    component: Any,
    *,
    required: bool = False,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> IntegrationComponent:
    """
    Register an API-layer component.

    Registration does not execute the component.
    It only makes the component available to the integration layer.
    """

    with _lock:

        record = _state.add_component(
            name,
            component,
            required=required,
            metadata=metadata,
        )

        record.initialized = (
            component is not None
        )

        record.healthy = (
            component is not None
        )

        return record


def unregister_api_component(
    name: str,
) -> bool:
    """
    Remove a component from the integration registry.
    """

    with _lock:

        if name not in _state.components:
            return False

        del _state.components[name]

        return True


def get_api_component(
    name: str,
) -> Any:
    """
    Retrieve a registered API component.
    """

    record = _state.component(
        name
    )

    if record is None:
        return None

    return record.component


def list_api_components() -> List[str]:
    """
    Return all registered component names.
    """

    return list(
        _state.components.keys()
    )


# ============================================================================
# MODULE DISCOVERY
# ============================================================================

def _safe_import(
    module_name: str,
) -> Optional[Any]:
    """
    Import an API module without allowing one optional module to prevent
    the entire integration layer from loading.
    """

    try:

        return importlib.import_module(
            module_name
        )

    except Exception as exc:

        logger.warning(
            "Could not import %s: %s",
            module_name,
            exc,
        )

        _state.warnings.append(
            f"{module_name}: {exc}"
        )

        return None


def discover_api_modules() -> Dict[
    str,
    Any,
]:
    """
    Discover the main modules in the API package.

    Existing files are imported rather than rewritten.
    """

    modules: Dict[
        str,
        Any,
    ] = {}

    candidates = {
        "routes": ".routes",
        "schemas": ".schemas",
        "search_api": ".search_api",
        "search_service": ".search_service",
    }

    for name, path in candidates.items():

        module = _safe_import(
            f"{__package__}{path}"
        )

        if module is not None:

            modules[name] = module

    return modules


# ============================================================================
# SCHEMA INTEGRATION
# ============================================================================

def initialize_schemas(
    modules: Mapping[
        str,
        Any,
    ],
) -> Optional[Any]:
    """
    Register the schema module.

    Schemas define the data contracts used by the API.
    """

    schemas = modules.get(
        "schemas"
    )

    if schemas is None:

        _state.warnings.append(
            "Schema module is unavailable."
        )

        return None

    register_api_component(
        "schemas",
        schemas,
        required=True,
        metadata={
            "role": (
                "request_response_contracts"
            ),
        },
    )

    return schemas


# ============================================================================
# SEARCH SERVICE INTEGRATION
# ============================================================================

def initialize_search_service(
    modules: Mapping[
        str,
        Any,
    ],
) -> Optional[Any]:
    """
    Initialize the SearchService integration.

    The existing SearchService remains responsible for orchestration.
    This layer simply obtains and registers it.
    """

    search_service_module = (
        modules.get(
            "search_service"
        )
    )

    if search_service_module is None:

        _state.errors.append(
            "Search service module is unavailable."
        )

        return None

    service = None

    getter = getattr(
        search_service_module,
        "get_search_service",
        None,
    )

    if callable(getter):

        try:

            service = getter()

        except Exception as exc:

            _state.errors.append(
                f"Search service initialization failed: {exc}"
            )

    if service is None:

        _state.warnings.append(
            "Search service exists but no service instance was obtained."
        )

    register_api_component(
        "search_service",
        service,
        required=True,
        metadata={
            "role": (
                "search_orchestration"
            ),
        },
    )

    return service


# ============================================================================
# SEARCH API INTEGRATION
# ============================================================================

def initialize_search_api(
    modules: Mapping[
        str,
        Any,
    ],
) -> Optional[Any]:
    """
    Register the API-facing search module.

    The search API remains responsible for handling API-level search
    operations. The integration layer does not copy its implementation.
    """

    search_api = modules.get(
        "search_api"
    )

    if search_api is None:

        _state.errors.append(
            "Search API module is unavailable."
        )

        return None

    register_api_component(
        "search_api",
        search_api,
        required=True,
        metadata={
            "role": (
                "api_search_interface"
            ),
        },
    )

    return search_api


# ============================================================================
# ROUTE INTEGRATION
# ============================================================================

def initialize_routes(
    modules: Mapping[
        str,
        Any,
    ],
) -> Optional[Any]:
    """
    Register the route module.

    Actual route registration remains inside the application's normal
    framework startup process.
    """

    routes = modules.get(
        "routes"
    )

    if routes is None:

        _state.errors.append(
            "Routes module is unavailable."
        )

        return None

    register_api_component(
        "routes",
        routes,
        required=True,
        metadata={
            "role": (
                "http_route_registration"
            ),
        },
    )

    return routes


# ============================================================================
# ROUTE REGISTRATION HELPERS
# ============================================================================

def find_route_registration_function(
    routes_module: Any,
) -> Optional[
    Callable[..., Any]
]:
    """
    Locate a conventional route registration function.

    We support several names so the integration layer can adapt to the
    existing project instead of forcing a rewrite.
    """

    if routes_module is None:
        return None

    candidates = (
        "register_routes",
        "register_api_routes",
        "setup_routes",
        "init_routes",
        "initialize_routes",
    )

    for name in candidates:

        candidate = getattr(
            routes_module,
            name,
            None,
        )

        if callable(candidate):

            return candidate

    return None


def register_routes_with_app(
    app: Any,
) -> bool:
    """
    Register API routes with a web application if the route module
    exposes a recognized registration function.

    Returns True when registration succeeds.
    """

    routes_module = (
        get_api_component(
            "routes"
        )
    )

    if routes_module is None:
        return False

    registration_function = (
        find_route_registration_function(
            routes_module
        )
    )

    if registration_function is None:

        _state.warnings.append(
            "No conventional route registration function was found."
        )

        return False

    try:

        result = registration_function(
            app
        )

        #
        # A route registration function may return None even when it
        # successfully modifies the application.
        #

        if result is False:
            return False

        return True

    except TypeError:

        #
        # Some projects expose a zero-argument route setup function.
        #

        try:

            result = registration_function()

            if result is False:
                return False

            return True

        except Exception as exc:

            _state.errors.append(
                f"Route registration failed: {exc}"
            )

            return False

    except Exception as exc:

        _state.errors.append(
            f"Route registration failed: {exc}"
        )

        return False


# ============================================================================
# SEARCH COMPONENT REGISTRATION
# ============================================================================

def register_search_components(
    *,
    query_parser: Any = None,
    analyzer: Any = None,
    tokenizer: Any = None,
    filter_engine: Any = None,
    index: Any = None,
    retriever: Any = None,
    ranker: Any = None,
    processor: Any = None,
    search_engine: Any = None,
    suggestion_engine: Any = None,
    statistics_provider: Any = None,
) -> Dict[str, bool]:
    """
    Connect actual Search subsystem implementations to SearchService.

    This function is intentionally dependency-injected.

    Later, when the real Search modules are ready, application startup
    can provide their actual functions/classes here.

    Nothing inside the Search subsystem is copied.
    """

    connected: Dict[
        str,
        bool,
    ] = {}

    if not SEARCH_SERVICE_AVAILABLE:

        _state.errors.append(
            "SearchService is unavailable."
        )

        return connected

    components = {
        "query_parser": query_parser,
        "analyzer": analyzer,
        "tokenizer": tokenizer,
        "filter_engine": filter_engine,
        "index": index,
        "retriever": retriever,
        "ranker": ranker,
        "processor": processor,
        "search_engine": search_engine,
        "suggestion_engine": suggestion_engine,
        "statistics_provider": statistics_provider,
    }

    provided = {
        name: component
        for name, component
        in components.items()
        if component is not None
    }

    if register_components is not None:

        try:

            register_components(
                **provided
            )

            for name in provided:

                connected[name] = True

                register_api_component(
                    f"search.{name}",
                    provided[name],
                    required=False,
                    metadata={
                        "role": (
                            "search_subsystem_dependency"
                        ),
                    },
                )

        except Exception as exc:

            _state.errors.append(
                f"Search component registration failed: {exc}"
            )

            for name in provided:
                connected[name] = False

    return connected


# ============================================================================
# CONVENTIONAL SEARCH MODULE DISCOVERY
# ============================================================================

def discover_search_components() -> Dict[
    str,
    Any,
]:
    """
    Attempt to discover the project's Search modules.

    This function does not assume that every module exposes a particular
    function. It simply imports the modules and records what is available.

    Actual adapters can be attached once their real interfaces are known.
    """

    discovered: Dict[
        str,
        Any,
    ] = {}

    search_modules = (
        "analysis",
        "engine",
        "filters",
        "index",
        "models",
        "processor",
        "query",
        "ranking",
        "retrieval",
        "search",
        "tokenizer",
    )

    for module_name in search_modules:

        full_name = (
            f"{__package__}.../Search/{module_name}"
        )

        #
        # Python package imports should normally use the actual package
        # name. The fallback below handles the project's conventional
        # top-level Search package.
        #

        candidates = (
            f"Search.{module_name}",
            f"search.{module_name}",
        )

        module = None

        for candidate in candidates:

            try:

                module = importlib.import_module(
                    candidate
                )

                break

            except Exception:
                continue

        if module is not None:

            discovered[
                module_name
            ] = module

            register_api_component(
                f"search.module.{module_name}",
                module,
                required=False,
                metadata={
                    "role": (
                        "search_subsystem_module"
                    ),
                },
            )

    return discovered


# ============================================================================
# CONVENTIONAL EXPORT DISCOVERY
# ============================================================================

def find_callable(
    module: Any,
    names: Iterable[str],
) -> Optional[
    Callable[..., Any]
]:
    """
    Find the first callable matching a list of conventional names.
    """

    if module is None:
        return None

    for name in names:

        candidate = getattr(
            module,
            name,
            None,
        )

        if callable(candidate):

            return candidate

    return None


def discover_callable_components(
    modules: Mapping[
        str,
        Any,
    ],
) -> Dict[
    str,
    Any,
]:
    """
    Discover conventional callable entry points from Search modules.

    This is intentionally conservative.

    It will only use functions/classes whose names match known
    conventions. It does not modify the Search modules.
    """

    mapping = {
        "query_parser": (
            "parse_query",
            "parse",
            "parse_search_query",
        ),

        "analyzer": (
            "analyze_query",
            "analyse_query",
            "analyze",
            "analyse",
        ),

        "tokenizer": (
            "tokenize",
            "tokenize_query",
        ),

        "filter_engine": (
            "apply_filters",
            "filter_results",
            "apply_filter",
        ),

        "retriever": (
            "retrieve",
            "retrieve_results",
            "retrieve_candidates",
        ),

        "ranker": (
            "rank",
            "rank_results",
            "rank_candidates",
        ),

        "processor": (
            "process_results",
            "process",
        ),

        "search_engine": (
            "search",
            "execute_search",
            "run_search",
        ),

        "suggestion_engine": (
            "suggest",
            "suggestions",
            "generate_suggestions",
        ),
    }

    discovered: Dict[
        str,
        Any,
    ] = {}

    module_order = (
        "query",
        "analysis",
        "filters",
        "index",
        "retrieval",
        "ranking",
        "processor",
        "search",
        "engine",
    )

    for component_name, names in mapping.items():

        for module_name in module_order:

            module = modules.get(
                module_name
            )

            component = find_callable(
                module,
                names,
            )

            if component is not None:

                discovered[
                    component_name
                ] = component

                break

    return discovered


# ============================================================================
# COMPLETE SEARCH DISCOVERY
# ============================================================================

def connect_discovered_search_components() -> Dict[
    str,
    bool,
]:
    """
    Discover Search modules and connect recognizable entry points.

    This is a convenience operation used during API initialization.
    """

    modules = (
        discover_search_components()
    )

    components = (
        discover_callable_components(
            modules
        )
    )

    return register_search_components(
        **components
    )


# ============================================================================
# API INITIALIZATION
# ============================================================================

def initialize_api(
    *,
    app: Any = None,
    connect_search: bool = True,
) -> IntegrationState:
    """
    Initialize the complete API integration layer.

    Order:

        1. Discover API modules.
        2. Register schemas.
        3. Register SearchService.
        4. Register search_api.
        5. Register routes.
        6. Discover Search components.
        7. Optionally attach routes to the application.
        8. Calculate readiness.
    """

    global _initialized
    global API_LAYER_STATUS

    with _lock:

        if _initialized:

            if app is not None:

                register_routes_with_app(
                    app
                )

            return _state

        API_LAYER_STATUS = (
            "initializing"
        )

        _state.started_at = (
            _utc_now()
        )

        _state.initialization_count += 1

        modules = (
            discover_api_modules()
        )

        initialize_schemas(
            modules
        )

        initialize_search_service(
            modules
        )

        initialize_search_api(
            modules
        )

        initialize_routes(
            modules
        )

        if connect_search:

            try:

                connect_discovered_search_components()

            except Exception as exc:

                _state.warnings.append(
                    "Automatic Search component "
                    f"discovery failed: {exc}"
                )

        if app is not None:

            register_routes_with_app(
                app
            )

        _state.initialized = True

        _state.initialized_at = (
            _utc_now()
        )

        _update_health()

        _initialized = True

        return _state


# ============================================================================
# HEALTH EVALUATION
# ============================================================================

def _update_health() -> None:
    """
    Calculate overall API integration health.
    """

    required_components = [
        component
        for component
        in _state.components.values()
        if component.required
    ]

    required_available = all(
        component.component is not None
        for component
        in required_components
    )

    required_healthy = all(
        component.healthy
        for component
        in required_components
    )

    _state.healthy = (
        required_available
        and required_healthy
        and not _state.errors
    )

    _state.ready = (
        _state.initialized
        and required_available
    )

    if _state.healthy:

        global API_LAYER_STATUS

        API_LAYER_STATUS = (
            "healthy"
        )

    elif _state.ready:

        API_LAYER_STATUS = (
            "ready_with_warnings"
        )

    else:

        API_LAYER_STATUS = (
            "degraded"
        )


# ============================================================================
# HEALTH CHECK
# ============================================================================

def health_check() -> Dict[str, Any]:
    """
    Return complete API integration health information.
    """

    with _lock:

        _update_health()

        result = {
            "service": API_LAYER_NAME,
            "version": INTEGRATION_VERSION,
            "status": API_LAYER_STATUS,
            "initialized": _state.initialized,
            "healthy": _state.healthy,
            "ready": _state.ready,
            "state": _state.status(),
        }

        if callable(
            search_health_check
        ):

            try:

                result[
                    "search_service"
                ] = search_health_check()

            except Exception as exc:

                result[
                    "search_service"
                ] = {
                    "healthy": False,
                    "error": str(
                        exc
                    ),
                }

        return result


# ============================================================================
# READINESS CHECK
# ============================================================================

def readiness_check() -> Dict[str, Any]:
    """
    Return whether the API layer is ready to accept requests.
    """

    with _lock:

        _update_health()

        return {
            "ready": _state.ready,
            "status": API_LAYER_STATUS,
            "required_components": {
                name: record.status()
                for name, record
                in _state.components.items()
                if record.required
            },
        }


# ============================================================================
# DIAGNOSTICS
# ============================================================================

def diagnostics() -> Dict[str, Any]:
    """
    Return detailed API integration diagnostics.
    """

    with _lock:

        _update_health()

        return {
            "integration": {
                "name": API_LAYER_NAME,
                "version": INTEGRATION_VERSION,
                "status": API_LAYER_STATUS,
            },

            "state": _state.status(),

            "components": {
                name: record.status()
                for name, record
                in _state.components.items()
            },

            "search_service_available": (
                SEARCH_SERVICE_AVAILABLE
            ),

            "search_service": (
                search_health_check()
                if callable(
                    search_health_check
                )
                else None
            ),
        }


# ============================================================================
# RESET
# ============================================================================

def reset_integration(
    *,
    clear_components: bool = True,
) -> None:
    """
    Reset integration state.

    Useful for tests and controlled application reinitialization.

    This does not modify any source files or Search components.
    """

    global _initialized
    global API_LAYER_STATUS

    with _lock:

        _initialized = False

        API_LAYER_STATUS = (
            "initializing"
        )

        _state.initialized = False

        _state.healthy = False

        _state.ready = False

        _state.started_at = None

        _state.initialized_at = None

        _state.warnings.clear()

        _state.errors.clear()

        _state.initialization_count = 0

        if clear_components:

            _state.components.clear()


# ============================================================================
# APPLICATION INTEGRATION
# ============================================================================

def integrate_application(
    app: Any,
    *,
    connect_search: bool = True,
) -> Any:
    """
    Integrate the API layer with the application's web framework.

    Returns the same application object so startup code can use:

        app = integrate_application(app)

    """

    initialize_api(
        app=app,
        connect_search=connect_search,
    )

    return app


# ============================================================================
# STARTUP / SHUTDOWN HOOKS
# ============================================================================

def startup(
    app: Any = None,
) -> IntegrationState:
    """
    Standard API startup entry point.
    """

    return initialize_api(
        app=app,
        connect_search=True,
    )


def shutdown() -> Dict[str, Any]:
    """
    Standard API shutdown hook.

    The current API layer does not own persistent resources, so shutdown
    currently marks the integration state as inactive without destroying
    the underlying Search objects.
    """

    global API_LAYER_STATUS

    with _lock:

        API_LAYER_STATUS = (
            "stopped"
        )

        return {
            "service": API_LAYER_NAME,
            "status": API_LAYER_STATUS,
            "timestamp": _utc_now(),
        }


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [

    # Constants

    "INTEGRATION_VERSION",
    "API_LAYER_NAME",
    "API_LAYER_STATUS",

    # Models

    "IntegrationComponent",
    "IntegrationState",

    # Registration

    "register_api_component",
    "unregister_api_component",
    "get_api_component",
    "list_api_components",

    # Discovery

    "discover_api_modules",
    "discover_search_components",
    "discover_callable_components",
    "find_callable",

    # Search integration

    "register_search_components",
    "connect_discovered_search_components",

    # Routes

    "register_routes_with_app",

    # Initialization

    "initialize_api",
    "integrate_application",
    "startup",
    "shutdown",

    # Diagnostics

    "health_check",
    "readiness_check",
    "diagnostics",

    # Testing

    "reset_integration",
]