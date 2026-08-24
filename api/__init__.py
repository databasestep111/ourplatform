"""
OurPlatform API package.

This module is the central initialization layer for the API package.

Responsibilities
----------------
- Define API identity and version information.
- Define supported API capabilities.
- Expose package metadata.
- Provide API lifecycle state.
- Provide feature registration.
- Provide health/status information.
- Provide shared API configuration.
- Provide compatibility information.
- Provide a controlled public package interface.

This module intentionally does NOT contain endpoint implementations.

Architecture:

    Frontend
        |
        v
    API package
        |
        +--> routes
        +--> search API
        +--> query API
        +--> ranking API
        +--> research API
        +--> suggestions API
        +--> filters API
        |
        v
    Existing backend systems

The API layer is an adapter/orchestration layer. Backend systems
remain responsible for their own domain logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional


# ============================================================================
# PACKAGE IDENTITY
# ============================================================================

API_NAME = "OurPlatform API"

API_VERSION = "0.1.0"

API_STATUS = "initializing"

API_DESCRIPTION = (
    "Application programming interface connecting the "
    "OurPlatform frontend with backend services."
)

API_PACKAGE = "api"


# ============================================================================
# COMPATIBILITY
# ============================================================================

MINIMUM_API_VERSION = "0.1.0"

CURRENT_API_VERSION = API_VERSION

SUPPORTED_API_VERSIONS = (
    "0.1.0",
)


# ============================================================================
# API CAPABILITIES
# ============================================================================

CAPABILITIES = {
    "search": True,
    "query_analysis": True,
    "ranking": True,
    "retrieval": True,
    "filters": True,
    "suggestions": True,
    "research": True,
    "results": True,
    "health": True,
    "metadata": True,
}


# ============================================================================
# ENDPOINT REGISTRY
# ============================================================================

DEFAULT_ENDPOINTS = {
    "health": "/api/health",
    "info": "/api/info",
    "search": "/api/search",
    "query": "/api/query",
    "ranking": "/api/ranking",
    "retrieval": "/api/retrieval",
    "filters": "/api/filters",
    "suggestions": "/api/suggestions",
    "research": "/api/research",
    "results": "/api/results",
}


# ============================================================================
# API STATE
# ============================================================================

@dataclass
class APIState:
    """
    Runtime state for the API package.

    The state object gives the API a central place to track whether
    the package has been initialized and which services are available.

    It does not contain the actual backend services themselves.
    """

    initialized: bool = False

    status: str = API_STATUS

    started_at: Optional[str] = None

    registered_services: list[str] = field(
        default_factory=list
    )

    registered_routes: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


_state = APIState()


# ============================================================================
# SERVICE REGISTRY
# ============================================================================

_services: Dict[str, Any] = {}


def register_service(
    name: str,
    service: Any,
    *,
    replace: bool = False,
) -> Any:
    """
    Register a backend service with the API layer.

    Parameters
    ----------
    name:
        Public service name.

    service:
        Backend object, callable, or service implementation.

    replace:
        Whether an existing registration may be replaced.

    Returns
    -------
    Any
        The registered service.

    Raises
    ------
    ValueError
        If the name is invalid.

    KeyError
        If a service already exists and replacement is disabled.
    """

    if not isinstance(name, str):
        raise ValueError(
            "Service name must be a string."
        )

    name = name.strip()

    if not name:
        raise ValueError(
            "Service name cannot be empty."
        )

    if (
        name in _services
        and not replace
    ):
        raise KeyError(
            f"API service already registered: {name}"
        )

    _services[name] = service

    if name not in _state.registered_services:
        _state.registered_services.append(name)

    return service


def unregister_service(
    name: str,
) -> bool:
    """
    Remove a registered service.

    Returns True when a service was removed and False when
    no matching service existed.
    """

    if name not in _services:
        return False

    del _services[name]

    if name in _state.registered_services:
        _state.registered_services.remove(name)

    return True


def get_service(
    name: str,
) -> Any:
    """
    Retrieve a registered service.

    Raises KeyError if the service is unavailable.
    """

    if name not in _services:
        raise KeyError(
            f"API service is not registered: {name}"
        )

    return _services[name]


def has_service(
    name: str,
) -> bool:
    """
    Check whether a service is registered.
    """

    return name in _services


def get_registered_services() -> list[str]:
    """
    Return a copy of the registered service names.
    """

    return list(
        _state.registered_services
    )


# ============================================================================
# ROUTE REGISTRY
# ============================================================================

_routes: Dict[str, str] = {}


def register_route(
    name: str,
    path: str,
    *,
    replace: bool = False,
) -> str:
    """
    Register an API route.

    This registry records the API contract. Actual route handling
    remains in the route modules/framework integration.
    """

    if not isinstance(name, str):
        raise ValueError(
            "Route name must be a string."
        )

    if not isinstance(path, str):
        raise ValueError(
            "Route path must be a string."
        )

    name = name.strip()
    path = path.strip()

    if not name:
        raise ValueError(
            "Route name cannot be empty."
        )

    if not path.startswith("/"):
        raise ValueError(
            "API route paths must begin with '/'."
        )

    if (
        name in _routes
        and not replace
    ):
        raise KeyError(
            f"API route already registered: {name}"
        )

    _routes[name] = path

    if path not in _state.registered_routes:
        _state.registered_routes.append(path)

    return path


def unregister_route(
    name: str,
) -> bool:
    """
    Remove a registered route.
    """

    if name not in _routes:
        return False

    path = _routes.pop(name)

    if path in _state.registered_routes:
        _state.registered_routes.remove(path)

    return True


def get_route(
    name: str,
) -> Optional[str]:
    """
    Return a registered route path.
    """

    return _routes.get(name)


def get_registered_routes() -> Dict[str, str]:
    """
    Return a copy of the API route registry.
    """

    return dict(_routes)


# ============================================================================
# CAPABILITY MANAGEMENT
# ============================================================================

def capability_enabled(
    name: str,
) -> bool:
    """
    Determine whether an API capability is enabled.
    """

    return bool(
        CAPABILITIES.get(
            name,
            False,
        )
    )


def enable_capability(
    name: str,
) -> None:
    """
    Enable an API capability.
    """

    CAPABILITIES[name] = True


def disable_capability(
    name: str,
) -> None:
    """
    Disable an API capability.
    """

    CAPABILITIES[name] = False


def get_capabilities() -> Dict[str, bool]:
    """
    Return a copy of the current capability map.
    """

    return dict(
        CAPABILITIES
    )


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_api(
    *,
    services: Optional[Mapping[str, Any]] = None,
    routes: Optional[Mapping[str, str]] = None,
) -> APIState:
    """
    Initialize the API package.

    This function is intentionally idempotent.

    Calling it multiple times does not require callers to manually
    reset the API state.
    """

    global API_STATUS

    API_STATUS = "initializing"

    _state.status = "initializing"

    if services:

        for name, service in services.items():

            register_service(
                name,
                service,
                replace=True,
            )

    if routes:

        for name, path in routes.items():

            register_route(
                name,
                path,
                replace=True,
            )

    _state.initialized = True

    _state.started_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    _state.status = "ready"

    API_STATUS = "ready"

    return _state


# ============================================================================
# SHUTDOWN
# ============================================================================

def shutdown_api(
    *,
    clear_services: bool = False,
    clear_routes: bool = False,
) -> APIState:
    """
    Transition the API into a stopped state.

    Services and routes are preserved by default so that a caller
    can inspect the API state after shutdown.
    """

    global API_STATUS

    _state.initialized = False

    _state.status = "stopped"

    API_STATUS = "stopped"

    if clear_services:

        _services.clear()

        _state.registered_services.clear()

    if clear_routes:

        _routes.clear()

        _state.registered_routes.clear()

    return _state


# ============================================================================
# API HEALTH
# ============================================================================

def is_ready() -> bool:
    """
    Return True when the API package has been initialized.
    """

    return (
        _state.initialized
        and _state.status == "ready"
    )


def health_check() -> Dict[str, Any]:
    """
    Return a lightweight health snapshot.

    This does not perform expensive backend searches.
    Endpoint-specific health checks can be implemented later.
    """

    return {
        "status": _state.status,
        "ready": is_ready(),
        "api": API_NAME,
        "version": API_VERSION,
        "services": len(
            _services
        ),
        "routes": len(
            _routes
        ),
        "timestamp": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }


# ============================================================================
# API INFORMATION
# ============================================================================

def get_api_info() -> Dict[str, Any]:
    """
    Return complete API package metadata.
    """

    return {
        "name": API_NAME,
        "package": API_PACKAGE,
        "version": API_VERSION,
        "minimum_version": MINIMUM_API_VERSION,
        "supported_versions": list(
            SUPPORTED_API_VERSIONS
        ),
        "description": API_DESCRIPTION,
        "status": _state.status,
        "ready": is_ready(),
        "capabilities": get_capabilities(),
        "routes": get_registered_routes(),
        "services": get_registered_services(),
        "started_at": _state.started_at,
    }


# ============================================================================
# ERROR REGISTRATION
# ============================================================================

def record_error(
    error: Any,
) -> None:
    """
    Record an API initialization/integration error.

    This is intentionally lightweight. Full API error handling
    belongs in the API error/response layer.
    """

    message = str(error).strip()

    if not message:
        message = "Unknown API error."

    _state.errors.append(
        message
    )


def get_errors() -> list[str]:
    """
    Return recorded API errors.
    """

    return list(
        _state.errors
    )


def clear_errors() -> None:
    """
    Clear recorded API errors.
    """

    _state.errors.clear()


# ============================================================================
# DEFAULT ROUTES
# ============================================================================

def register_default_routes() -> None:
    """
    Register the initial API contract.

    These are declarations only. The actual framework route
    implementations will be created in api/routes.py and the
    specialized API modules.
    """

    for name, path in DEFAULT_ENDPOINTS.items():

        register_route(
            name,
            path,
            replace=True,
        )


# ============================================================================
# PACKAGE RESET
# ============================================================================

def reset_api_state() -> None:
    """
    Completely reset package-level API state.

    Primarily useful for development and testing.
    """

    global API_STATUS

    _services.clear()

    _routes.clear()

    _state.initialized = False

    _state.status = "initializing"

    _state.started_at = None

    _state.registered_services.clear()

    _state.registered_routes.clear()

    _state.errors.clear()

    API_STATUS = "initializing"


# ============================================================================
# INITIAL PACKAGE SETUP
# ============================================================================

register_default_routes()


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "API_NAME",
    "API_VERSION",
    "API_STATUS",
    "API_DESCRIPTION",
    "API_PACKAGE",
    "MINIMUM_API_VERSION",
    "CURRENT_API_VERSION",
    "SUPPORTED_API_VERSIONS",
    "CAPABILITIES",
    "DEFAULT_ENDPOINTS",
    "APIState",
    "register_service",
    "unregister_service",
    "get_service",
    "has_service",
    "get_registered_services",
    "register_route",
    "unregister_route",
    "get_route",
    "get_registered_routes",
    "capability_enabled",
    "enable_capability",
    "disable_capability",
    "get_capabilities",
    "initialize_api",
    "shutdown_api",
    "is_ready",
    "health_check",
    "get_api_info",
    "record_error",
    "get_errors",
    "clear_errors",
    "register_default_routes",
    "reset_api_state",
]