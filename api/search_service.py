"""
OurPlatform Search Service
==========================

Version:
    1.0.0

Purpose:
    Central orchestration layer between the API and the existing
    OurPlatform Search subsystem.

Responsibilities:
    - Accept validated search requests.
    - Coordinate query analysis.
    - Coordinate tokenization.
    - Coordinate filtering.
    - Coordinate candidate retrieval.
    - Coordinate ranking.
    - Coordinate result processing.
    - Build API-compatible responses.
    - Expose diagnostics and execution metadata.
    - Provide one stable entry point for API routes.

Important architectural rule:

    This service DOES NOT replace the Search subsystem.

    It orchestrates it.

The actual search intelligence remains inside:

    Search/
        analysis.py
        engine.py
        filters.py
        index.py
        models.py
        processor.py
        query.py
        ranking.py
        retrieval.py
        search.py
        tokenizer.py

The service therefore acts as the integration boundary:

    API
      |
      v
    SearchService
      |
      +--> Query
      +--> Analysis
      +--> Tokenizer
      +--> Filters
      +--> Index
      +--> Retrieval
      +--> Ranking
      +--> Processor
      +--> Search Engine
      |
      v
    API Schema
      |
      v
    Frontend

This module intentionally avoids hard-coding assumptions about the
exact implementation of the Search subsystem. Adapters can be attached
to the real implementation as the project evolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .schemas import (
    APIError,
    APIErrorCode,
    APIResponse,
    QueryAnalysis,
    SearchMetadata,
    SearchRequest,
    SearchResult,
    SearchResponseData,
    SearchSuggestion,
    PaginationResponse,
    build_search_response,
    error_response,
    success_response,
    validate_search_request,
)


# ============================================================================
# VERSION
# ============================================================================

SERVICE_VERSION = "1.0.0"

SERVICE_NAME = "OurPlatform Search Service"


# ============================================================================
# INTERNAL TYPE ALIASES
# ============================================================================

SearchCallable = Callable[..., Any]

ResultLike = Any


# ============================================================================
# SERVICE CONFIGURATION
# ============================================================================

@dataclass
class SearchServiceConfig:
    """
    Configuration controlling orchestration behaviour.

    The configuration intentionally lives here rather than inside the
    individual Search modules so the API integration layer can control
    the overall workflow without changing the search algorithms.
    """

    default_limit: int = 10

    maximum_limit: int = 500

    maximum_query_length: int = 1000

    enable_analysis: bool = True

    enable_tokenization: bool = True

    enable_filters: bool = True

    enable_retrieval: bool = True

    enable_ranking: bool = True

    enable_processing: bool = True

    enable_metadata: bool = True

    enable_suggestions: bool = True

    fail_on_optional_component_error: bool = False

    include_debug_metadata: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_limit": self.default_limit,
            "maximum_limit": self.maximum_limit,
            "maximum_query_length": self.maximum_query_length,
            "enable_analysis": self.enable_analysis,
            "enable_tokenization": self.enable_tokenization,
            "enable_filters": self.enable_filters,
            "enable_retrieval": self.enable_retrieval,
            "enable_ranking": self.enable_ranking,
            "enable_processing": self.enable_processing,
            "enable_metadata": self.enable_metadata,
            "enable_suggestions": self.enable_suggestions,
            "fail_on_optional_component_error": (
                self.fail_on_optional_component_error
            ),
            "include_debug_metadata": self.include_debug_metadata,
        }


# ============================================================================
# COMPONENT REGISTRY
# ============================================================================

@dataclass
class SearchComponentRegistry:
    """
    Holds references to the real Search subsystem components.

    Nothing is copied here.

    The registry simply stores callable references so the service can
    orchestrate the existing implementation.

    Example later:

        registry.query_parser = real_query_parser
        registry.retriever = real_retriever
        registry.ranker = real_ranker
    """

    query_parser: Optional[SearchCallable] = None

    analyzer: Optional[SearchCallable] = None

    tokenizer: Optional[SearchCallable] = None

    filter_engine: Optional[SearchCallable] = None

    index: Optional[Any] = None

    retriever: Optional[SearchCallable] = None

    ranker: Optional[SearchCallable] = None

    processor: Optional[SearchCallable] = None

    search_engine: Optional[SearchCallable] = None

    suggestion_engine: Optional[SearchCallable] = None

    statistics_provider: Optional[SearchCallable] = None

    def registered_components(self) -> List[str]:
        """
        Return the names of components currently connected.
        """

        components: List[str] = []

        values = {
            "query_parser": self.query_parser,
            "analyzer": self.analyzer,
            "tokenizer": self.tokenizer,
            "filter_engine": self.filter_engine,
            "index": self.index,
            "retriever": self.retriever,
            "ranker": self.ranker,
            "processor": self.processor,
            "search_engine": self.search_engine,
            "suggestion_engine": self.suggestion_engine,
            "statistics_provider": self.statistics_provider,
        }

        for name, component in values.items():
            if component is not None:
                components.append(name)

        return components

    def is_connected(self) -> bool:
        """
        Return whether at least one actual search component is connected.
        """

        return bool(
            self.registered_components()
        )


# ============================================================================
# EXECUTION STATE
# ============================================================================

@dataclass
class SearchExecutionState:
    """
    Internal state accumulated during one search request.

    This object keeps intermediate data out of the API response until
    the service has finished orchestrating the request.
    """

    request: SearchRequest

    parsed_query: Any = None

    query_analysis: Optional[QueryAnalysis] = None

    tokens: List[str] = field(
        default_factory=list
    )

    candidates: List[Any] = field(
        default_factory=list
    )

    filtered_candidates: List[Any] = field(
        default_factory=list
    )

    ranked_results: List[Any] = field(
        default_factory=list
    )

    processed_results: List[Any] = field(
        default_factory=list
    )

    suggestions: List[SearchSuggestion] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    documents_examined: int = 0

    cache_hit: bool = False

    started_at: float = field(
        default_factory=perf_counter
    )

    completed_at: Optional[float] = None

    def finish(self) -> None:
        self.completed_at = perf_counter()

    @property
    def elapsed_ms(self) -> float:
        end = (
            self.completed_at
            if self.completed_at is not None
            else perf_counter()
        )

        return (
            end - self.started_at
        ) * 1000.0


# ============================================================================
# SERVICE
# ============================================================================

class SearchService:
    """
    Central search orchestration service.

    This class is deliberately dependency-injected.

    That means the service does not need to know how every Search
    component is internally implemented.

    Instead:

        SearchService
            receives
                SearchComponentRegistry
            and calls
                registered components

    This makes the service stable while the underlying search engine
    continues to evolve.
    """

    def __init__(
        self,
        *,
        registry: Optional[
            SearchComponentRegistry
        ] = None,
        config: Optional[
            SearchServiceConfig
        ] = None,
    ) -> None:

        self.registry = (
            registry
            if registry is not None
            else SearchComponentRegistry()
        )

        self.config = (
            config
            if config is not None
            else SearchServiceConfig()
        )

    # ========================================================================
    # PUBLIC ENTRY POINT
    # ========================================================================

    def search(
        self,
        request: SearchRequest,
    ) -> APIResponse:
        """
        Execute a complete search workflow.

        This is the primary method the API layer should call.

        Workflow:

            validate
                ↓
            query parsing
                ↓
            analysis
                ↓
            tokenization
                ↓
            candidate retrieval
                ↓
            filtering
                ↓
            ranking
                ↓
            processing
                ↓
            suggestions
                ↓
            response construction
        """

        validation_error = validate_search_request(
            request
        )

        if validation_error is not None:
            return validation_error

        state = SearchExecutionState(
            request=request
        )

        try:
            self._execute_workflow(
                state
            )

            state.finish()

            return self._build_response(
                state
            )

        except Exception as exc:
            state.finish()

            return self._build_exception_response(
                state,
                exc,
            )

    # ========================================================================
    # WORKFLOW
    # ========================================================================

    def _execute_workflow(
        self,
        state: SearchExecutionState,
    ) -> None:

        self._parse_query(
            state
        )

        self._analyse_query(
            state
        )

        self._tokenize_query(
            state
        )

        self._retrieve_candidates(
            state
        )

        self._apply_filters(
            state
        )

        self._rank_results(
            state
        )

        self._process_results(
            state
        )

        self._generate_suggestions(
            state
        )

    # ========================================================================
    # QUERY PARSING
    # ========================================================================

    def _parse_query(
        self,
        state: SearchExecutionState,
    ) -> None:

        component = (
            self.registry.query_parser
        )

        if component is None:
            state.parsed_query = (
                state.request.query
            )
            return

        try:
            state.parsed_query = self._invoke(
                component,
                state.request.query,
                request=state.request,
            )

        except Exception as exc:
            self._handle_optional_error(
                state,
                "Query parsing",
                exc,
            )

            state.parsed_query = (
                state.request.query
            )

    # ========================================================================
    # QUERY ANALYSIS
    # ========================================================================

    def _analyse_query(
        self,
        state: SearchExecutionState,
    ) -> None:

        if not self.config.enable_analysis:
            return

        component = (
            self.registry.analyzer
        )

        if component is None:
            return

        try:

            analysis = self._invoke(
                component,
                state.parsed_query,
                request=state.request,
            )

            state.query_analysis = (
                self._coerce_query_analysis(
                    analysis,
                    state.request.query,
                )
            )

        except Exception as exc:
            self._handle_optional_error(
                state,
                "Query analysis",
                exc,
            )

    # ========================================================================
    # TOKENIZATION
    # ========================================================================

    def _tokenize_query(
        self,
        state: SearchExecutionState,
    ) -> None:

        if not self.config.enable_tokenization:
            return

        component = (
            self.registry.tokenizer
        )

        if component is None:
            return

        try:

            tokens = self._invoke(
                component,
                state.parsed_query,
                request=state.request,
            )

            state.tokens = self._coerce_string_list(
                tokens
            )

        except Exception as exc:
            self._handle_optional_error(
                state,
                "Tokenization",
                exc,
            )

    # ========================================================================
    # RETRIEVAL
    # ========================================================================

    def _retrieve_candidates(
        self,
        state: SearchExecutionState,
    ) -> None:

        component = (
            self.registry.retriever
        )

        if component is None:

            component = (
                self.registry.search_engine
            )

        if component is None:
            return

        try:

            payload = {
                "query": state.parsed_query,
                "tokens": state.tokens,
                "request": state.request,
            }

            result = self._invoke(
                component,
                payload,
            )

            state.candidates = (
                self._coerce_results(
                    result
                )
            )

            state.documents_examined = (
                len(state.candidates)
            )

        except Exception as exc:
            raise RuntimeError(
                f"Candidate retrieval failed: {exc}"
            ) from exc

    # ========================================================================
    # FILTERING
    # ========================================================================

    def _apply_filters(
        self,
        state: SearchExecutionState,
    ) -> None:

        if not self.config.enable_filters:
            state.filtered_candidates = (
                list(state.candidates)
            )
            return

        component = (
            self.registry.filter_engine
        )

        if component is None:
            state.filtered_candidates = (
                list(state.candidates)
            )
            return

        try:

            result = self._invoke(
                component,
                state.candidates,
                request=state.request,
            )

            state.filtered_candidates = (
                self._coerce_results(
                    result
                )
            )

        except Exception as exc:

            self._handle_optional_error(
                state,
                "Filtering",
                exc,
            )

            state.filtered_candidates = (
                list(state.candidates)
            )

    # ========================================================================
    # RANKING
    # ========================================================================

    def _rank_results(
        self,
        state: SearchExecutionState,
    ) -> None:

        candidates = (
            state.filtered_candidates
        )

        if not self.config.enable_ranking:
            state.ranked_results = (
                list(candidates)
            )
            return

        component = (
            self.registry.ranker
        )

        if component is None:
            state.ranked_results = (
                list(candidates)
            )
            return

        try:

            result = self._invoke(
                component,
                candidates,
                request=state.request,
                query_analysis=state.query_analysis,
            )

            state.ranked_results = (
                self._coerce_results(
                    result
                )
            )

        except Exception as exc:

            self._handle_optional_error(
                state,
                "Ranking",
                exc,
            )

            state.ranked_results = (
                list(candidates)
            )

    # ========================================================================
    # RESULT PROCESSING
    # ========================================================================

    def _process_results(
        self,
        state: SearchExecutionState,
    ) -> None:

        results = (
            state.ranked_results
        )

        if not self.config.enable_processing:
            state.processed_results = (
                list(results)
            )
            return

        component = (
            self.registry.processor
        )

        if component is None:
            state.processed_results = (
                list(results)
            )
            return

        try:

            processed = self._invoke(
                component,
                results,
                request=state.request,
            )

            state.processed_results = (
                self._coerce_results(
                    processed
                )
            )

        except Exception as exc:

            self._handle_optional_error(
                state,
                "Result processing",
                exc,
            )

            state.processed_results = (
                list(results)
            )

    # ========================================================================
    # SUGGESTIONS
    # ========================================================================

    def _generate_suggestions(
        self,
        state: SearchExecutionState,
    ) -> None:

        if not self.config.enable_suggestions:
            return

        component = (
            self.registry.suggestion_engine
        )

        if component is None:
            return

        try:

            suggestions = self._invoke(
                component,
                state.request.query,
                request=state.request,
                results=state.processed_results,
            )

            state.suggestions = (
                self._coerce_suggestions(
                    suggestions
                )
            )

        except Exception as exc:

            self._handle_optional_error(
                state,
                "Suggestion generation",
                exc,
            )

    # ========================================================================
    # RESPONSE BUILDING
    # ========================================================================

    def _build_response(
        self,
        state: SearchExecutionState,
    ) -> APIResponse:

        results = [
            self._coerce_search_result(
                item
            )
            for item
            in state.processed_results
        ]

        results = self._apply_pagination(
            results,
            state.request,
        )

        pagination = (
            self._build_pagination(
                total=len(
                    state.processed_results
                ),
                request=state.request,
            )
        )

        metadata = (
            self._build_metadata(
                state
            )
        )

        response_data = SearchResponseData(
            query=state.request.query,
            results=results,
            total_results=len(
                state.processed_results
            ),
            pagination=pagination,
            query_analysis=state.query_analysis,
            search_metadata=metadata,
            suggestions=state.suggestions,
        )

        return success_response(
            data=response_data,
            message=(
                "Search completed successfully."
            ),
            metadata={
                "service": SERVICE_NAME,
                "service_version": SERVICE_VERSION,
                "schema_version": "1.0.0",
            },
        )

    # ========================================================================
    # PAGINATION
    # ========================================================================

    def _apply_pagination(
        self,
        results: Sequence[
            SearchResult
        ],
        request: SearchRequest,
    ) -> List[SearchResult]:

        start = max(
            request.offset,
            0,
        )

        end = (
            start
            + request.limit
        )

        return list(
            results[
                start:end
            ]
        )

    def _build_pagination(
        self,
        *,
        total: int,
        request: SearchRequest,
    ) -> PaginationResponse:

        limit = max(
            request.limit,
            1,
        )

        offset = max(
            request.offset,
            0,
        )

        page = (
            offset // limit
        ) + 1

        total_pages = (
            (total + limit - 1)
            // limit
            if total
            else 0
        )

        has_previous = (
            offset > 0
        )

        has_next = (
            offset + limit < total
        )

        previous_offset = (
            max(
                offset - limit,
                0,
            )
            if has_previous
            else None
        )

        next_offset = (
            offset + limit
            if has_next
            else None
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

    # ========================================================================
    # METADATA
    # ========================================================================

    def _build_metadata(
        self,
        state: SearchExecutionState,
    ) -> SearchMetadata:

        return SearchMetadata(
            engine=(
                "OurPlatform Search"
            ),
            version=SERVICE_VERSION,
            documents_examined=(
                state.documents_examined
            ),
            candidates=len(
                state.candidates
            ),
            ranking_enabled=(
                self.config.enable_ranking
            ),
            index_enabled=(
                self.registry.index is not None
            ),
            search_time_ms=(
                state.elapsed_ms
            ),
            mode=(
                state.request.mode.value
            ),
            retrieval_strategy=(
                "registered_retriever"
                if self.registry.retriever
                else "search_engine"
                if self.registry.search_engine
                else "unregistered"
            ),
            ranking_strategy=(
                "registered_ranker"
                if self.registry.ranker
                else "passthrough"
            ),
            index_strategy=(
                type(
                    self.registry.index
                ).__name__
                if self.registry.index
                else None
            ),
            cache_hit=(
                state.cache_hit
            ),
        )

    # ========================================================================
    # ERROR HANDLING
    # ========================================================================

    def _handle_optional_error(
        self,
        state: SearchExecutionState,
        component_name: str,
        exc: Exception,
    ) -> None:

        message = (
            f"{component_name} unavailable: "
            f"{exc}"
        )

        state.warnings.append(
            message
        )

        if (
            self.config
            .fail_on_optional_component_error
        ):
            raise RuntimeError(
                message
            ) from exc

    def _build_exception_response(
        self,
        state: SearchExecutionState,
        exc: Exception,
    ) -> APIResponse:

        return error_response(
            APIErrorCode.SEARCH_ERROR,
            "Search execution failed.",
            status=500,
            details={
                "service": SERVICE_NAME,
                "service_version": SERVICE_VERSION,
                "error_type": type(
                    exc
                ).__name__,
                "warnings": state.warnings,
            },
        )

    # ========================================================================
    # COMPONENT INVOCATION
    # ========================================================================

    @staticmethod
    def _invoke(
        component: SearchCallable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Invoke a registered component.

        Components are deliberately treated as callables so this
        service can work with functions, methods, or lightweight
        adapter objects.
        """

        return component(
            *args,
            **kwargs,
        )

    # ========================================================================
    # COERCION
    # ========================================================================

    @staticmethod
    def _coerce_results(
        value: Any,
    ) -> List[Any]:

        if value is None:
            return []

        if isinstance(
            value,
            Mapping,
        ):

            if "results" in value:
                value = value["results"]

            else:
                return [value]

        if isinstance(
            value,
            (str, bytes),
        ):
            return [value]

        try:
            return list(value)

        except TypeError:
            return [value]

    @staticmethod
    def _coerce_string_list(
        value: Any,
    ) -> List[str]:

        if value is None:
            return []

        if isinstance(
            value,
            str,
        ):
            return [
                item
                for item in value.split()
                if item
            ]

        try:
            return [
                str(item)
                for item in value
                if str(item).strip()
            ]

        except TypeError:
            return [str(value)]

    @staticmethod
    def _coerce_query_analysis(
        value: Any,
        query: str,
    ) -> Optional[
        QueryAnalysis
    ]:

        if value is None:
            return None

        if isinstance(
            value,
            QueryAnalysis,
        ):
            return value

        if isinstance(
            value,
            Mapping,
        ):

            return QueryAnalysis(
                raw_query=value.get(
                    "raw_query",
                    query,
                ),
                terms=value.get(
                    "terms",
                    [],
                ),
                phrases=value.get(
                    "phrases",
                    [],
                ),
                fields=value.get(
                    "fields",
                    [],
                ),
                filters=value.get(
                    "filters",
                    [],
                ),
                operators=value.get(
                    "operators",
                    [],
                ),
                intent=value.get(
                    "intent"
                ),
                complexity=value.get(
                    "complexity"
                ),
                fuzzy_requested=value.get(
                    "fuzzy_requested",
                    False,
                ),
                semantic_requested=value.get(
                    "semantic_requested",
                    False,
                ),
            )

        return QueryAnalysis(
            raw_query=query
        )

    @staticmethod
    def _coerce_search_result(
        value: Any,
    ) -> SearchResult:

        if isinstance(
            value,
            SearchResult,
        ):
            return value

        if isinstance(
            value,
            Mapping,
        ):

            return SearchResult(
                id=value.get(
                    "id",
                    0,
                ),
                title=value.get(
                    "title",
                    "Untitled",
                ),
                content=value.get(
                    "content",
                    "",
                ),
                snippet=value.get(
                    "snippet",
                    "",
                ),
                score=value.get(
                    "score",
                    0.0,
                ),
                category=value.get(
                    "category",
                    "general",
                ),
                result_type=value.get(
                    "type",
                    value.get(
                        "result_type",
                        "general",
                    ),
                ),
                tags=value.get(
                    "tags",
                    [],
                ),
                created_at=value.get(
                    "created_at"
                ),
                updated_at=value.get(
                    "updated_at"
                ),
                url=value.get(
                    "url"
                ),
                match_type=value.get(
                    "match_type"
                ),
                explanation=value.get(
                    "explanation"
                ),
                highlights=value.get(
                    "highlights",
                    [],
                ),
                metadata=value.get(
                    "metadata",
                    {},
                ),
            )

        return SearchResult(
            id=0,
            title=str(value),
            content=str(value),
        )

    @staticmethod
    def _coerce_suggestions(
        value: Any,
    ) -> List[
        SearchSuggestion
    ]:

        if value is None:
            return []

        if isinstance(
            value,
            str,
        ):
            value = [value]

        try:
            values = list(value)

        except TypeError:
            values = [value]

        results: List[
            SearchSuggestion
        ] = []

        for item in values:

            if isinstance(
                item,
                SearchSuggestion,
            ):
                results.append(
                    item
                )

            elif isinstance(
                item,
                Mapping,
            ):
                results.append(
                    SearchSuggestion(
                        text=item.get(
                            "text",
                            "",
                        ),
                        score=item.get(
                            "score",
                            0,
                        ),
                        reason=item.get(
                            "reason"
                        ),
                        category=item.get(
                            "category"
                        ),
                    )
                )

            else:
                results.append(
                    SearchSuggestion(
                        text=str(item)
                    )
                )

        return results

    # ========================================================================
    # STATUS / DIAGNOSTICS
    # ========================================================================

    def status(self) -> Dict[str, Any]:
        """
        Return the current integration status.

        Useful for an API health/debug endpoint.
        """

        components = (
            self.registry.registered_components()
        )

        return {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "schema_version": "1.0.0",
            "connected": bool(
                components
            ),
            "components": components,
            "component_count": len(
                components
            ),
            "configuration": (
                self.config.to_dict()
            ),
        }

    def component_available(
        self,
        name: str,
    ) -> bool:
        """
        Check whether a named component is connected.
        """

        if not hasattr(
            self.registry,
            name,
        ):
            return False

        return (
            getattr(
                self.registry,
                name,
            )
            is not None
        )

    def connected_components(
        self,
    ) -> List[str]:
        """
        Return connected component names.
        """

        return (
            self.registry
            .registered_components()
        )


# ============================================================================
# DEFAULT SERVICE
# ============================================================================

_default_registry = (
    SearchComponentRegistry()
)

_default_config = (
    SearchServiceConfig()
)

_default_service = SearchService(
    registry=_default_registry,
    config=_default_config,
)


# ============================================================================
# PUBLIC SERVICE FUNCTIONS
# ============================================================================

def get_search_service() -> SearchService:
    """
    Return the process-wide default SearchService.
    """

    return _default_service


def configure_search_service(
    *,
    registry: Optional[
        SearchComponentRegistry
    ] = None,
    config: Optional[
        SearchServiceConfig
    ] = None,
) -> SearchService:
    """
    Configure and replace the default orchestration service.

    Existing Search modules are not modified.

    This function only changes which implementations the service
    coordinates.
    """

    global _default_service

    _default_service = SearchService(
        registry=(
            registry
            if registry is not None
            else SearchComponentRegistry()
        ),
        config=(
            config
            if config is not None
            else SearchServiceConfig()
        ),
    )

    return _default_service


def search(
    request: SearchRequest,
) -> APIResponse:
    """
    Convenience wrapper around the default service.
    """

    return (
        get_search_service()
        .search(
            request
        )
    )


def search_from_dict(
    data: Mapping[str, Any],
) -> APIResponse:
    """
    Convenience wrapper for API request dictionaries.
    """

    request = SearchRequest.from_dict(
        data
    )

    return search(
        request
    )


def service_status() -> Dict[str, Any]:
    """
    Return default service status.
    """

    return (
        get_search_service()
        .status()
    )


# ============================================================================
# COMPONENT REGISTRATION HELPERS
# ============================================================================

def register_query_parser(
    component: SearchCallable,
) -> None:

    _default_registry.query_parser = (
        component
    )


def register_analyzer(
    component: SearchCallable,
) -> None:

    _default_registry.analyzer = (
        component
    )


def register_tokenizer(
    component: SearchCallable,
) -> None:

    _default_registry.tokenizer = (
        component
    )


def register_filter_engine(
    component: SearchCallable,
) -> None:

    _default_registry.filter_engine = (
        component
    )


def register_index(
    component: Any,
) -> None:

    _default_registry.index = (
        component
    )


def register_retriever(
    component: SearchCallable,
) -> None:

    _default_registry.retriever = (
        component
    )


def register_ranker(
    component: SearchCallable,
) -> None:

    _default_registry.ranker = (
        component
    )


def register_processor(
    component: SearchCallable,
) -> None:

    _default_registry.processor = (
        component
    )


def register_search_engine(
    component: SearchCallable,
) -> None:

    _default_registry.search_engine = (
        component
    )


def register_suggestion_engine(
    component: SearchCallable,
) -> None:

    _default_registry.suggestion_engine = (
        component
    )


def register_statistics_provider(
    component: SearchCallable,
) -> None:

    _default_registry.statistics_provider = (
        component
    )


# ============================================================================
# BULK REGISTRATION
# ============================================================================

def register_components(
    *,
    query_parser: Optional[
        SearchCallable
    ] = None,
    analyzer: Optional[
        SearchCallable
    ] = None,
    tokenizer: Optional[
        SearchCallable
    ] = None,
    filter_engine: Optional[
        SearchCallable
    ] = None,
    index: Optional[Any] = None,
    retriever: Optional[
        SearchCallable
    ] = None,
    ranker: Optional[
        SearchCallable
    ] = None,
    processor: Optional[
        SearchCallable
    ] = None,
    search_engine: Optional[
        SearchCallable
    ] = None,
    suggestion_engine: Optional[
        SearchCallable
    ] = None,
    statistics_provider: Optional[
        SearchCallable
    ] = None,
) -> SearchComponentRegistry:
    """
    Register any available Search components.

    Only supplied components are changed.

    This is important because we do not want to destroy existing
    registrations when a new subsystem is added later.
    """

    registry = _default_registry

    if query_parser is not None:
        registry.query_parser = (
            query_parser
        )

    if analyzer is not None:
        registry.analyzer = (
            analyzer
        )

    if tokenizer is not None:
        registry.tokenizer = (
            tokenizer
        )

    if filter_engine is not None:
        registry.filter_engine = (
            filter_engine
        )

    if index is not None:
        registry.index = (
            index
        )

    if retriever is not None:
        registry.retriever = (
            retriever
        )

    if ranker is not None:
        registry.ranker = (
            ranker
        )

    if processor is not None:
        registry.processor = (
            processor
        )

    if search_engine is not None:
        registry.search_engine = (
            search_engine
        )

    if suggestion_engine is not None:
        registry.suggestion_engine = (
            suggestion_engine
        )

    if statistics_provider is not None:
        registry.statistics_provider = (
            statistics_provider
        )

    return registry


# ============================================================================
# HEALTH CHECK
# ============================================================================

def health_check() -> Dict[str, Any]:
    """
    Lightweight integration health check.

    This does not execute a real search.

    It simply reports whether the orchestration layer has registered
    components.
    """

    service = (
        get_search_service()
    )

    status = service.status()

    return {
        "healthy": True,
        "service": status[
            "service"
        ],
        "version": status[
            "version"
        ],
        "connected": status[
            "connected"
        ],
        "component_count": status[
            "component_count"
        ],
        "components": status[
            "components"
        ],
    }


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [

    # Constants

    "SERVICE_VERSION",
    "SERVICE_NAME",

    # Configuration

    "SearchServiceConfig",

    # Registry

    "SearchComponentRegistry",

    # State

    "SearchExecutionState",

    # Service

    "SearchService",

    # Default service

    "get_search_service",
    "configure_search_service",

    # Search functions

    "search",
    "search_from_dict",

    # Diagnostics

    "service_status",
    "health_check",

    # Registration

    "register_query_parser",
    "register_analyzer",
    "register_tokenizer",
    "register_filter_engine",
    "register_index",
    "register_retriever",
    "register_ranker",
    "register_processor",
    "register_search_engine",
    "register_suggestion_engine",
    "register_statistics_provider",
    "register_components",
]