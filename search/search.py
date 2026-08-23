"""
search/search.py

High-level search orchestration engine.

This module is the main public search interface for OurPlatform.

Architecture
------------

                    +----------------+
                    |   Search API   |
                    +--------+-------+
                             |
                    +--------v-------+
                    | Query Parsing  |
                    +--------+-------+
                             |
                    +--------v-------+
                    | Query Analysis |
                    +--------+-------+
                             |
              +--------------+--------------+
              |                             |
      +-------v-------+             +-------v-------+
      |   Retrieval   |             |    Filters   |
      +-------+-------+             +-------+-------+
              |                             |
              +--------------+--------------+
                             |
                    +--------v-------+
                    |    Ranking     |
                    +--------+-------+
                             |
                    +--------v-------+
                    | Result Fusion  |
                    +--------+-------+
                             |
                    +--------v-------+
                    | Post Analysis  |
                    +--------+-------+
                             |
                    +--------v-------+
                    | Cache / History|
                    +----------------+

Design goals
------------

* Preserve the simple Search API from early versions.
* Provide a serious orchestration layer for the newer search modules.
* Avoid forcing every subsystem to exist.
* Support dependency injection.
* Support exact, phrase, fuzzy, wildcard and token matching.
* Support field-aware scoring.
* Support categories and tags.
* Support structured filters.
* Support pagination.
* Support sorting.
* Support result explanations.
* Support search suggestions.
* Support duplicate detection.
* Support document statistics.
* Support query history.
* Support caching.
* Support result diversification.
* Support configurable ranking weights.
* Support search diagnostics.
* Remain usable as a standalone module.

The engine intentionally does NOT perform web crawling.
That belongs to higher-level research/search-source infrastructure.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import threading
import time
import unicodedata

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# =====================================================================
# VERSION
# =====================================================================

SEARCH_ENGINE_VERSION = "2.0.0"


# =====================================================================
# CONSTANTS
# =====================================================================

DEFAULT_LIMIT = 10
DEFAULT_MAX_LIMIT = 100
DEFAULT_HISTORY_SIZE = 500
DEFAULT_CACHE_SIZE = 256

TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*"
)

WORD_PATTERN = re.compile(
    r"\b[\w'-]+\b",
    re.UNICODE,
)

WILDCARD_ESCAPE = re.compile(
    r"([.^$+{}\[\]\\|()])"
)

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================


def now_iso() -> str:
    """Return the current local timestamp as ISO-8601."""
    return datetime.now().isoformat()


def normalize_text(value: Any) -> str:
    """
    Normalize arbitrary values into searchable text.

    This deliberately keeps the original semantic characters while
    normalizing Unicode and whitespace.
    """

    if value is None:
        return ""

    text = str(value)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.replace(
        "\x00",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_key(value: Any) -> str:
    """Create a case-insensitive normalized key."""
    return normalize_text(value).casefold()


def tokenize(text: Any) -> List[str]:
    """
    Tokenize text into normalized terms.
    """

    text = normalize_text(text)

    return [
        token.casefold()
        for token in TOKEN_PATTERN.findall(text)
    ]


def unique_preserving_order(
    values: Iterable[Any],
) -> List[Any]:
    """Deduplicate while preserving insertion order."""

    seen = set()
    result = []

    for value in values:
        key = repr(value)

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Clamp a numeric value."""
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to float."""

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Safely convert a value to int."""

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def stable_hash(value: Any) -> str:
    """
    Create a stable SHA-256 identifier.

    Used for cache keys and content fingerprints.
    """

    if isinstance(value, (dict, list, tuple)):
        value = repr(value)

    return hashlib.sha256(
        str(value).encode(
            "utf-8",
            errors="replace",
        )
    ).hexdigest()


# =====================================================================
# CONFIGURATION
# =====================================================================


@dataclass
class SearchConfig:
    """
    Configuration for the search engine.
    """

    default_limit: int = DEFAULT_LIMIT

    maximum_limit: int = DEFAULT_MAX_LIMIT

    minimum_score: float = 0.0

    title_weight: float = 8.0

    content_weight: float = 1.0

    tag_weight: float = 5.0

    category_weight: float = 3.0

    exact_weight: float = 20.0

    phrase_weight: float = 15.0

    prefix_weight: float = 4.0

    fuzzy_weight: float = 2.0

    frequency_weight: float = 1.0

    recency_weight: float = 0.5

    length_penalty: float = 0.05

    enable_fuzzy: bool = True

    enable_wildcards: bool = True

    enable_phrase_matching: bool = True

    enable_prefix_matching: bool = True

    enable_query_expansion: bool = True

    enable_caching: bool = True

    enable_history: bool = True

    enable_deduplication: bool = True

    enable_diversification: bool = True

    enable_analysis: bool = True

    cache_size: int = DEFAULT_CACHE_SIZE

    history_size: int = DEFAULT_HISTORY_SIZE

    fuzzy_threshold: float = 0.72

    diversification_threshold: float = 0.92

    max_query_terms: int = 128

    max_document_tokens: int = 100_000


# =====================================================================
# RANKING WEIGHTS
# =====================================================================


@dataclass
class RankingWeights:
    """
    Fine-grained ranking controls.

    These can later be connected directly to ranking.py.
    """

    exact_title: float = 25.0

    exact_content: float = 12.0

    exact_tag: float = 18.0

    exact_category: float = 10.0

    phrase_title: float = 20.0

    phrase_content: float = 10.0

    term_title: float = 6.0

    term_content: float = 1.5

    term_tag: float = 4.0

    prefix_title: float = 3.0

    prefix_content: float = 0.8

    fuzzy_title: float = 3.0

    fuzzy_content: float = 1.0

    field_boost: float = 1.0

    popularity: float = 0.0

    recency: float = 0.25

    diversity: float = 0.25


# =====================================================================
# QUERY STRUCTURES
# =====================================================================


@dataclass
class QueryProfile:
    """
    Lightweight internal representation of a search query.

    This allows Search to work even when query.py is unavailable.
    """

    raw: str

    normalized: str

    terms: List[str] = field(
        default_factory=list
    )

    phrases: List[str] = field(
        default_factory=list
    )

    required_terms: List[str] = field(
        default_factory=list
    )

    prohibited_terms: List[str] = field(
        default_factory=list
    )

    filters: Dict[str, Any] = field(
        default_factory=dict
    )

    fields: Dict[str, List[str]] = field(
        default_factory=dict
    )

    wildcard_terms: List[str] = field(
        default_factory=list
    )

    fuzzy_terms: List[str] = field(
        default_factory=list
    )

    intent: str = "search"

    sort_field: str = "relevance"

    sort_direction: str = "desc"

    limit: int = DEFAULT_LIMIT

    offset: int = 0

    complexity: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# =====================================================================
# DOCUMENT REPRESENTATION
# =====================================================================


@dataclass
class SearchDocument:
    """
    Internal document representation.

    The public API continues returning dictionaries so existing code
    remains compatible.
    """

    id: int

    title: str

    content: str

    category: str

    tags: List[str]

    created_at: str

    updated_at: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    token_count: int = 0

    fingerprint: str = ""

    search_count: int = 0

    popularity: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """Return a public dictionary representation."""

        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": copy.deepcopy(
                self.metadata
            ),
        }


# =====================================================================
# SEARCH RESULT
# =====================================================================


@dataclass
class SearchResult:
    """
    Rich search result.

    Search.find() converts these back to dictionaries for compatibility.
    """

    document: SearchDocument

    score: float

    rank: int = 0

    matched_terms: List[str] = field(
        default_factory=list
    )

    matched_phrases: List[str] = field(
        default_factory=list
    )

    matched_fields: List[str] = field(
        default_factory=list
    )

    highlights: Dict[str, str] = field(
        default_factory=dict
    )

    explanation: Dict[str, Any] = field(
        default_factory=dict
    )

    def as_dict(
        self,
        include_explanation: bool = False,
    ) -> Dict[str, Any]:

        result = self.document.as_dict()

        result["score"] = round(
            self.score,
            6,
        )

        result["rank"] = self.rank

        result["matched_terms"] = list(
            self.matched_terms
        )

        result["matched_phrases"] = list(
            self.matched_phrases
        )

        result["matched_fields"] = list(
            self.matched_fields
        )

        if self.highlights:
            result["highlights"] = dict(
                self.highlights
            )

        if include_explanation:
            result["explanation"] = copy.deepcopy(
                self.explanation
            )

        return result


# =====================================================================
# SEARCH RESPONSE
# =====================================================================


@dataclass
class SearchResponse:
    """
    Full search response.

    Useful for advanced consumers while find() preserves the old API.
    """

    query: str

    results: List[SearchResult]

    total_candidates: int

    total_matches: int

    took_ms: float

    page: int

    limit: int

    offset: int

    query_profile: QueryProfile

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def as_dict(
        self,
        include_explanation: bool = False,
    ) -> Dict[str, Any]:

        return {
            "query": self.query,
            "results": [
                result.as_dict(
                    include_explanation
                )
                for result in self.results
            ],
            "total_candidates": (
                self.total_candidates
            ),
            "total_matches": (
                self.total_matches
            ),
            "took_ms": round(
                self.took_ms,
                4,
            ),
            "page": self.page,
            "limit": self.limit,
            "offset": self.offset,
            "query": self.query_profile.normalized,
            "intent": self.query_profile.intent,
            "metadata": copy.deepcopy(
                self.metadata
            ),
        }


# =====================================================================
# SEARCH CACHE
# =====================================================================


class SearchCache:
    """
    Small LRU-style cache.

    The cache stores immutable-ish result dictionaries rather than
    internal objects so callers cannot accidentally mutate engine state.
    """

    def __init__(
        self,
        maximum_size: int = DEFAULT_CACHE_SIZE,
    ):
        self.maximum_size = max(
            1,
            maximum_size,
        )

        self._data: Dict[
            str,
            Any,
        ] = {}

        self._order: Deque[str] = deque()

        self.hits = 0
        self.misses = 0

        self._lock = threading.RLock()

    def get(
        self,
        key: str,
    ) -> Any:

        with self._lock:

            if key not in self._data:
                self.misses += 1
                return None

            self.hits += 1

            value = copy.deepcopy(
                self._data[key]
            )

            try:
                self._order.remove(key)
            except ValueError:
                pass

            self._order.append(key)

            return value

    def set(
        self,
        key: str,
        value: Any,
    ) -> None:

        with self._lock:

            self._data[key] = copy.deepcopy(
                value
            )

            try:
                self._order.remove(key)
            except ValueError:
                pass

            self._order.append(key)

            while len(
                self._order
            ) > self.maximum_size:

                oldest = self._order.popleft()

                self._data.pop(
                    oldest,
                    None,
                )

    def clear(self) -> None:

        with self._lock:
            self._data.clear()
            self._order.clear()

    def stats(self) -> Dict[str, Any]:

        with self._lock:

            total = (
                self.hits
                + self.misses
            )

            hit_rate = (
                self.hits / total
                if total
                else 0.0
            )

            return {
                "size": len(
                    self._data
                ),
                "maximum_size": (
                    self.maximum_size
                ),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }


# =====================================================================
# SEARCH HISTORY
# =====================================================================


@dataclass
class SearchHistoryEntry:

    query: str

    normalized: str

    timestamp: str

    result_count: int

    took_ms: float

    intent: str


class SearchHistory:

    def __init__(
        self,
        maximum_size: int = DEFAULT_HISTORY_SIZE,
    ):

        self.maximum_size = max(
            1,
            maximum_size,
        )

        self.entries: Deque[
            SearchHistoryEntry
        ] = deque(
            maxlen=self.maximum_size
        )

        self._lock = threading.RLock()

    def add(
        self,
        query: str,
        normalized: str,
        result_count: int,
        took_ms: float,
        intent: str,
    ) -> None:

        with self._lock:

            self.entries.append(
                SearchHistoryEntry(
                    query=query,
                    normalized=normalized,
                    timestamp=now_iso(),
                    result_count=result_count,
                    took_ms=took_ms,
                    intent=intent,
                )
            )

    def recent(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:

        with self._lock:

            values = list(
                self.entries
            )[-max(0, limit):]

            values.reverse()

            return [
                entry.__dict__.copy()
                for entry in values
            ]

    def popular(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:

        with self._lock:

            counts = Counter(
                entry.normalized
                for entry in self.entries
                if entry.normalized
            )

            return [
                {
                    "query": query,
                    "count": count,
                }
                for query, count
                in counts.most_common(
                    limit
                )
            ]

    def clear(self) -> None:

        with self._lock:
            self.entries.clear()


# =====================================================================
# SEARCH ENGINE
# =====================================================================


class Search:
    """
    High-level search engine.

    This is the main public interface.

    It supports the original API:

        search.add(...)
        search.remove(...)
        search.get(...)
        search.update(...)
        search.find(...)

    while exposing advanced APIs:

        search.search(...)
        search.explain(...)
        search.suggest(...)
        search.analyze(...)
        search.search_title(...)
        search.search_content(...)
        search.by_category(...)
        search.by_tag(...)
        search.statistics(...)
        search.health(...)
    """

    # -----------------------------------------------------------------
    # CONSTRUCTION
    # -----------------------------------------------------------------

    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        ranking_weights: Optional[RankingWeights] = None,
        tokenizer: Any = None,
        index: Any = None,
        ranking: Any = None,
        query_parser: Any = None,
        analyzer: Any = None,
    ):

        self.config = config or SearchConfig()

        self.ranking_weights = (
            ranking_weights
            or RankingWeights()
        )

        self.tokenizer_engine = tokenizer
        self.index_engine = index
        self.ranking_engine = ranking
        self.query_parser_engine = query_parser
        self.analyzer_engine = analyzer

        self.items: List[
            Dict[str, Any]
        ] = []

        self._documents: Dict[
            int,
            SearchDocument,
        ] = {}

        self.next_id = 1

        self._lock = threading.RLock()

        self.cache = SearchCache(
            self.config.cache_size
        )

        self.history = SearchHistory(
            self.config.history_size
        )

        self._token_index: Dict[
            str,
            Set[int],
        ] = defaultdict(set)

        self._title_index: Dict[
            str,
            Set[int],
        ] = defaultdict(set)

        self._tag_index: Dict[
            str,
            Set[int],
        ] = defaultdict(set)

        self._category_index: Dict[
            str,
            Set[int],
        ] = defaultdict(set)

        self._prefix_index: Dict[
            str,
            Set[int],
        ] = defaultdict(set)

        self._document_tokens: Dict[
            int,
            Counter,
        ] = {}

        self._term_document_frequency: Counter = Counter()

        self._query_counter = 0

        self._mutation_counter = 0

        self._statistics = {
            "adds": 0,
            "updates": 0,
            "removes": 0,
            "searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        self._last_search: Optional[
            SearchResponse
        ] = None

    # =================================================================
    # DOCUMENT MANAGEMENT
    # =================================================================

    def add(
        self,
        content,
        title="Untitled",
        category="general",
        tags=None,
        metadata=None,
        item_id=None,
    ):
        """
        Add a searchable document.

        Backwards-compatible with the original API.
        """

        if tags is None:
            tags = []

        if metadata is None:
            metadata = {}

        content = normalize_text(content)
        title = normalize_text(title)
        category = normalize_text(category)

        tags = [
            normalize_text(tag)
            for tag in tags
            if normalize_text(tag)
        ]

        with self._lock:

            if item_id is None:
                item_id = self.next_id
                self.next_id += 1
            else:
                item_id = safe_int(
                    item_id,
                    self.next_id,
                )

                self.next_id = max(
                    self.next_id,
                    item_id + 1,
                )

            fingerprint = self._fingerprint(
                title,
                content,
                category,
                tags,
            )

            if (
                self.config.enable_deduplication
                and self._fingerprint_exists(
                    fingerprint
                )
            ):
                existing = self._document_by_fingerprint(
                    fingerprint
                )

                if existing is not None:
                    return existing.as_dict()

            timestamp = now_iso()

            document = SearchDocument(
                id=item_id,
                title=title,
                content=content,
                category=category,
                tags=tags,
                created_at=timestamp,
                updated_at=timestamp,
                metadata=copy.deepcopy(
                    metadata
                ),
                token_count=len(
                    tokenize(content)
                ),
                fingerprint=fingerprint,
            )

            self._documents[item_id] = document

            self.items.append(
                document.as_dict()
            )

            self._index_document(
                document
            )

            self._mutation_counter += 1

            self._statistics[
                "adds"
            ] += 1

            self.cache.clear()

            return document.as_dict()

    # -----------------------------------------------------------------
    # REMOVE
    # -----------------------------------------------------------------

    def remove(
        self,
        item_id,
    ):
        """Remove a document by ID."""

        with self._lock:

            document = self._documents.get(
                safe_int(item_id, -1)
            )

            if document is None:
                return False

            self._deindex_document(
                document
            )

            self._documents.pop(
                document.id,
                None,
            )

            self.items = [
                item
                for item in self.items
                if item.get("id")
                != document.id
            ]

            self._mutation_counter += 1

            self._statistics[
                "removes"
            ] += 1

            self.cache.clear()

            return True

    # -----------------------------------------------------------------
    # GET
    # -----------------------------------------------------------------

    def get(
        self,
        item_id,
    ):
        """Return a document by ID."""

        document = self._documents.get(
            safe_int(item_id, -1)
        )

        if document is None:
            return None

        return document.as_dict()

    # -----------------------------------------------------------------
    # UPDATE
    # -----------------------------------------------------------------

    def update(
        self,
        item_id,
        title=None,
        content=None,
        category=None,
        tags=None,
        metadata=None,
    ):
        """
        Update a document and rebuild its search representation.
        """

        item_id = safe_int(
            item_id,
            -1,
        )

        with self._lock:

            document = self._documents.get(
                item_id
            )

            if document is None:
                return None

            self._deindex_document(
                document
            )

            if title is not None:
                document.title = normalize_text(
                    title
                )

            if content is not None:
                document.content = normalize_text(
                    content
                )

            if category is not None:
                document.category = normalize_text(
                    category
                )

            if tags is not None:
                document.tags = [
                    normalize_text(tag)
                    for tag in tags
                    if normalize_text(tag)
                ]

            if metadata is not None:
                document.metadata = copy.deepcopy(
                    metadata
                )

            document.updated_at = now_iso()

            document.token_count = len(
                tokenize(
                    document.content
                )
            )

            document.fingerprint = self._fingerprint(
                document.title,
                document.content,
                document.category,
                document.tags,
            )

            self._index_document(
                document
            )

            self._sync_public_items()

            self._mutation_counter += 1

            self._statistics[
                "updates"
            ] += 1

            self.cache.clear()

            return document.as_dict()

    # =================================================================
    # QUERY PARSING
    # =================================================================

    def parse_query(
        self,
        query: str,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> QueryProfile:
        """
        Parse a query.

        If the advanced query parser exists, Search attempts to use it.
        Otherwise it uses its own robust fallback parser.
        """

        raw = normalize_text(query)

        if self.query_parser_engine is not None:

            try:
                parsed = self.query_parser_engine.parse(
                    raw
                )

                return self._adapt_external_query(
                    parsed,
                    raw,
                    limit,
                    offset,
                )

            except Exception:
                pass

        return self._fallback_query_parser(
            raw,
            limit,
            offset,
        )

    # -----------------------------------------------------------------
    # FALLBACK QUERY PARSER
    # -----------------------------------------------------------------

    def _fallback_query_parser(
        self,
        raw: str,
        limit: Optional[int],
        offset: int,
    ) -> QueryProfile:

        normalized = normalize_key(
            raw
        )

        profile = QueryProfile(
            raw=raw,
            normalized=normalized,
        )

        if not raw:
            profile.limit = (
                limit
                or self.config.default_limit
            )
            return profile

        # -------------------------------------------------------------
        # PHRASES
        # -------------------------------------------------------------

        phrase_matches = re.findall(
            r'"([^"]+)"',
            raw,
        )

        profile.phrases = [
            normalize_key(
                phrase
            )
            for phrase in phrase_matches
        ]

        # -------------------------------------------------------------
        # FIELDS / FILTERS
        # -------------------------------------------------------------

        remaining = raw

        field_pattern = re.compile(
            r"""
            (?P<field>
                [A-Za-z_][A-Za-z0-9_.-]*
            )
            :
            (?P<value>
                ".*?"
                |
                \S+
            )
            """,
            re.VERBOSE,
        )

        for match in field_pattern.finditer(
            raw
        ):

            field_name = normalize_key(
                match.group("field")
            )

            raw_value = match.group(
                "value"
            ).strip(
                '"'
            )

            value = normalize_key(
                raw_value
            )

            if field_name in {
                "category",
                "categories",
                "tag",
                "tags",
                "type",
                "author",
                "language",
                "status",
            }:

                profile.filters[
                    field_name
                ] = value

            elif field_name == "sort":

                parts = value.split(
                    ":"
                )

                profile.sort_field = (
                    parts[0]
                    or "relevance"
                )

                if len(parts) > 1:
                    profile.sort_direction = (
                        parts[1]
                        if parts[1] in {
                            "asc",
                            "desc",
                        }
                        else "desc"
                    )

            elif field_name in {
                "limit",
                "size",
            }:

                profile.limit = clamp(
                    safe_int(
                        value,
                        self.config.default_limit,
                    ),
                    1,
                    self.config.maximum_limit,
                )

            elif field_name == "offset":

                profile.offset = max(
                    0,
                    safe_int(
                        value,
                        0,
                    ),
                )

            else:

                profile.fields.setdefault(
                    field_name,
                    [],
                ).append(
                    value
                )

            remaining = remaining.replace(
                match.group(0),
                " ",
                1,
            )

        # -------------------------------------------------------------
        # BOOLEAN TERMS
        # -------------------------------------------------------------

        raw_tokens = re.findall(
            r'"[^"]+"|\S+',
            remaining,
        )

        for raw_token in raw_tokens:

            token = raw_token.strip()

            if not token:
                continue

            upper = token.upper()

            if upper in {
                "AND",
                "OR",
            }:
                continue

            if upper == "NOT":
                continue

            prohibited = (
                token.startswith("-")
            )

            required = (
                token.startswith("+")
            )

            if prohibited or required:
                token = token[1:]

            fuzzy = token.endswith(
                "~"
            )

            if fuzzy:
                token = token[:-1]
                profile.fuzzy_terms.append(
                    normalize_key(token)
                )

            wildcard = (
                "*" in token
                or "?" in token
            )

            if wildcard:
                profile.wildcard_terms.append(
                    normalize_key(token)
                )

            token = normalize_key(
                token
            )

            if not token:
                continue

            if token in STOP_WORDS:
                continue

            if prohibited:
                profile.prohibited_terms.append(
                    token
                )
                continue

            if required:
                profile.required_terms.append(
                    token
                )

            profile.terms.append(
                token
            )

        # -------------------------------------------------------------
        # INTENT
        # -------------------------------------------------------------

        profile.intent = (
            self._detect_intent(
                raw
            )
        )

        # -------------------------------------------------------------
        # COMPLEXITY
        # -------------------------------------------------------------

        profile.complexity = (
            len(profile.terms)
            + (
                len(profile.phrases)
                * 2
            )
            + (
                len(profile.filters)
                * 2
            )
            + (
                len(profile.fields)
                * 1.5
            )
            + (
                len(profile.wildcard_terms)
                * 2
            )
            + (
                len(profile.fuzzy_terms)
                * 1.5
            )
        )

        profile.limit = clamp(
            safe_int(
                limit,
                profile.limit
                or self.config.default_limit,
            ),
            1,
            self.config.maximum_limit,
        )

        profile.offset = max(
            0,
            safe_int(
                offset,
                profile.offset,
            ),
        )

        profile.metadata = {
            "term_count": len(
                profile.terms
            ),
            "phrase_count": len(
                profile.phrases
            ),
            "filter_count": len(
                profile.filters
            ),
            "wildcard_count": len(
                profile.wildcard_terms
            ),
            "fuzzy_count": len(
                profile.fuzzy_terms
            ),
        }

        return profile

    # -----------------------------------------------------------------
    # EXTERNAL QUERY ADAPTER
    # -----------------------------------------------------------------

    def _adapt_external_query(
        self,
        parsed: Any,
        raw: str,
        limit: Optional[int],
        offset: int,
    ) -> QueryProfile:

        profile = QueryProfile(
            raw=raw,
            normalized=normalize_key(
                raw
            ),
        )

        try:
            profile.terms = [
                normalize_key(value)
                for value in parsed.terms()
            ]
        except Exception:
            profile.terms = tokenize(
                raw
            )

        try:
            profile.phrases = [
                normalize_key(value)
                for value in parsed.phrases()
            ]
        except Exception:
            pass

        try:
            profile.filters = {
                item.field: item.value
                for item in parsed.filters
            }
        except Exception:
            pass

        try:
            profile.intent = (
                parsed.intent.value
            )
        except Exception:
            profile.intent = self._detect_intent(
                raw
            )

        try:
            profile.limit = safe_int(
                parsed.pagination.limit,
                self.config.default_limit,
            )

            profile.offset = safe_int(
                parsed.pagination.offset,
                0,
            )
        except Exception:
            profile.limit = self.config.default_limit
            profile.offset = 0

        if limit is not None:
            profile.limit = safe_int(
                limit,
                profile.limit,
            )

        profile.offset = offset

        profile.limit = int(
            clamp(
                profile.limit,
                1,
                self.config.maximum_limit,
            )
        )

        profile.offset = max(
            0,
            profile.offset,
        )

        return profile

    # =================================================================
    # SEARCH ENTRY POINT
    # =================================================================

    def find(
        self,
        query,
        category=None,
        tags=None,
        limit=10,
    ):
        """
        Backwards-compatible search method.

        Returns a list of dictionaries.
        """

        response = self.search(
            query=query,
            category=category,
            tags=tags,
            limit=limit,
        )

        return [
            result.as_dict()
            for result in response.results
        ]

    # -----------------------------------------------------------------
    # ADVANCED SEARCH
    # -----------------------------------------------------------------

    def search(
        self,
        query,
        category=None,
        tags=None,
        limit=None,
        offset=0,
        sort=None,
        include_explanation=False,
        highlight=False,
        diversify=None,
        use_cache=True,
    ) -> SearchResponse:
        """
        Execute the complete search pipeline.
        """

        start = time.perf_counter()

        self._statistics[
            "searches"
        ] += 1

        self._query_counter += 1

        profile = self.parse_query(
            query,
            limit=limit,
            offset=offset,
        )

        # -------------------------------------------------------------
        # EXTERNAL FILTERS
        # -------------------------------------------------------------

        if category is not None:

            profile.filters[
                "category"
            ] = normalize_key(
                category
            )

        if tags:

            profile.filters[
                "tags"
            ] = [
                normalize_key(tag)
                for tag in tags
            ]

        if sort:

            if isinstance(
                sort,
                str,
            ):

                sort_parts = sort.split(
                    ":"
                )

                profile.sort_field = (
                    sort_parts[0]
                    or "relevance"
                )

                if len(sort_parts) > 1:

                    profile.sort_direction = (
                        sort_parts[1]
                        if sort_parts[1]
                        in {
                            "asc",
                            "desc",
                        }
                        else "desc"
                    )

        if diversify is None:
            diversify = (
                self.config.enable_diversification
            )

        # -------------------------------------------------------------
        # CACHE
        # -------------------------------------------------------------

        cache_key = self._build_cache_key(
            profile,
            include_explanation,
            highlight,
            diversify,
        )

        if (
            use_cache
            and self.config.enable_caching
        ):

            cached = self.cache.get(
                cache_key
            )

            if cached is not None:

                self._statistics[
                    "cache_hits"
                ] += 1

                return self._response_from_cached(
                    cached
                )

            self._statistics[
                "cache_misses"
            ] += 1

        # -------------------------------------------------------------
        # CANDIDATE RETRIEVAL
        # -------------------------------------------------------------

        candidate_ids = (
            self._retrieve_candidates(
                profile
            )
        )

        total_candidates = len(
            candidate_ids
        )

        # -------------------------------------------------------------
        # FILTERING
        # -------------------------------------------------------------

        candidate_ids = [
            item_id
            for item_id in candidate_ids
            if self._passes_filters(
                self._documents[item_id],
                profile,
            )
        ]

        # -------------------------------------------------------------
        # SCORING
        # -------------------------------------------------------------

        results = []

        for item_id in candidate_ids:

            document = self._documents.get(
                item_id
            )

            if document is None:
                continue

            scored = self._score_document(
                document,
                profile,
            )

            if scored is None:
                continue

            if (
                scored.score
                < self.config.minimum_score
            ):
                continue

            results.append(
                scored
            )

        # -------------------------------------------------------------
        # SORTING / RANKING
        # -------------------------------------------------------------

        results = self._rank_results(
            results,
            profile,
        )

        # -------------------------------------------------------------
        # DIVERSIFICATION
        # -------------------------------------------------------------

        if diversify:

            results = self._diversify_results(
                results
            )

        total_matches = len(
            results
        )

        # -------------------------------------------------------------
        # PAGINATION
        # -------------------------------------------------------------

        paginated = results[
            profile.offset:
            profile.offset
            + profile.limit
        ]

        # -------------------------------------------------------------
        # RANK NUMBERS
        # -------------------------------------------------------------

        for index, result in enumerate(
            paginated,
            start=profile.offset + 1,
        ):

            result.rank = index

            if highlight:

                result.highlights = (
                    self._build_highlights(
                        result.document,
                        profile,
                    )
                )

            if include_explanation:

                result.explanation = (
                    self._build_explanation(
                        result,
                        profile,
                    )
                )

        # -------------------------------------------------------------
        # RESPONSE
        # -------------------------------------------------------------

        took_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        page = (
            profile.offset
            // profile.limit
        ) + 1

        response = SearchResponse(
            query=profile.raw,
            results=paginated,
            total_candidates=total_candidates,
            total_matches=total_matches,
            took_ms=took_ms,
            page=page,
            limit=profile.limit,
            offset=profile.offset,
            query_profile=profile,
            metadata={
                "engine_version": (
                    SEARCH_ENGINE_VERSION
                ),
                "query_id": self._query_counter,
                "cached": False,
                "mutation_version": (
                    self._mutation_counter
                ),
            },
        )

        self._last_search = response

        # -------------------------------------------------------------
        # HISTORY
        # -------------------------------------------------------------

        if self.config.enable_history:

            self.history.add(
                query=profile.raw,
                normalized=profile.normalized,
                result_count=total_matches,
                took_ms=took_ms,
                intent=profile.intent,
            )

        # -------------------------------------------------------------
        # CACHE
        # -------------------------------------------------------------

        if (
            use_cache
            and self.config.enable_caching
        ):

            self.cache.set(
                cache_key,
                self._cache_response(
                    response
                ),
            )

        return response

    # =================================================================
    # RETRIEVAL
    # =================================================================

    def _retrieve_candidates(
        self,
        profile: QueryProfile,
    ) -> Set[int]:
        """
        Retrieve candidates.

        The external index gets first opportunity.
        The internal inverted indexes provide a reliable fallback.
        """

        external = (
            self._retrieve_from_external_index(
                profile
            )
        )

        if external is not None:

            candidates = set(
                external
            )

        else:

            candidates = set()

            # ---------------------------------------------------------
            # TERM RETRIEVAL
            # ---------------------------------------------------------

            for term in profile.terms:

                term_ids = (
                    self._token_index.get(
                        term,
                        set(),
                    )
                )

                candidates.update(
                    term_ids
                )

            # ---------------------------------------------------------
            # PHRASE RETRIEVAL
            # ---------------------------------------------------------

            for phrase in profile.phrases:

                phrase_tokens = tokenize(
                    phrase
                )

                if not phrase_tokens:
                    continue

                first_token = (
                    phrase_tokens[0]
                )

                candidates.update(
                    self._token_index.get(
                        first_token,
                        set(),
                    )
                )

            # ---------------------------------------------------------
            # WILDCARDS
            # ---------------------------------------------------------

            for wildcard in (
                profile.wildcard_terms
            ):

                candidates.update(
                    self._retrieve_wildcard(
                        wildcard
                    )
                )

            # ---------------------------------------------------------
            # FUZZY
            # ---------------------------------------------------------

            if self.config.enable_fuzzy:

                for fuzzy in (
                    profile.fuzzy_terms
                ):

                    candidates.update(
                        self._retrieve_fuzzy(
                            fuzzy
                        )
                    )

            # ---------------------------------------------------------
            # FIELD-ONLY SEARCH
            # ---------------------------------------------------------

            if not candidates:

                for field_name, values in (
                    profile.fields.items()
                ):

                    candidates.update(
                        self._retrieve_field(
                            field_name,
                            values,
                        )
                    )

        # -------------------------------------------------------------
        # EMPTY / BROAD QUERY
        # -------------------------------------------------------------

        if not candidates:

            # For an empty query we don't fabricate matches.
            # A broad query can intentionally return everything,
            # but an actually empty query should remain empty.
            if profile.normalized:
                candidates = set(
                    self._documents.keys()
                )

        return candidates

    # -----------------------------------------------------------------
    # EXTERNAL INDEX
    # -----------------------------------------------------------------

    def _retrieve_from_external_index(
        self,
        profile: QueryProfile,
    ) -> Optional[Set[int]]:

        if self.index_engine is None:
            return None

        methods = (
            "search",
            "find",
            "retrieve",
            "lookup",
            "query",
        )

        for method_name in methods:

            method = getattr(
                self.index_engine,
                method_name,
                None,
            )

            if method is None:
                continue

            attempts = [
                lambda: method(profile),
                lambda: method(
                    profile.normalized
                ),
                lambda: method(
                    profile.terms
                ),
            ]

            for attempt in attempts:

                try:

                    raw = attempt()

                    if raw is None:
                        continue

                    return self._extract_ids(
                        raw
                    )

                except (
                    TypeError,
                    AttributeError,
                    ValueError,
                ):
                    continue

        return None

    # -----------------------------------------------------------------
    # EXTRACT IDS
    # -----------------------------------------------------------------

    def _extract_ids(
        self,
        values: Any,
    ) -> Set[int]:

        result = set()

        if isinstance(
            values,
            Mapping,
        ):

            values = values.values()

        if not isinstance(
            values,
            Iterable,
        ) or isinstance(
            values,
            (str, bytes),
        ):

            values = [values]

        for value in values:

            if isinstance(
                value,
                Mapping,
            ):

                candidate = value.get(
                    "id"
                )

                if candidate is not None:

                    result.add(
                        safe_int(
                            candidate,
                            -1,
                        )
                    )

            elif hasattr(
                value,
                "id",
            ):

                result.add(
                    safe_int(
                        getattr(
                            value,
                            "id",
                        ),
                        -1,
                    )
                )

            else:

                result.add(
                    safe_int(
                        value,
                        -1,
                    )
                )

        return {
            item_id
            for item_id in result
            if item_id >= 0
            and item_id in self._documents
        }

    # =================================================================
    # WILDCARD RETRIEVAL
    # =================================================================

    def _retrieve_wildcard(
        self,
        pattern: str,
    ) -> Set[int]:

        pattern = normalize_key(
            pattern
        )

        if not pattern:
            return set()

        regex = (
            "^"
            + WILDCARD_ESCAPE.sub(
                r"\\\1",
                pattern,
            )
            .replace(
                "*",
                ".*",
            )
            .replace(
                "?",
                ".",
            )
            + "$"
        )

        try:
            compiled = re.compile(
                regex
            )
        except re.error:
            return set()

        candidates = set()

        for token, ids in (
            self._token_index.items()
        ):

            if compiled.match(token):

                candidates.update(
                    ids
                )

        return candidates

    # =================================================================
    # FUZZY RETRIEVAL
    # =================================================================

    def _retrieve_fuzzy(
        self,
        term: str,
    ) -> Set[int]:

        term = normalize_key(
            term
        )

        candidates = set()

        if len(term) < 2:
            return candidates

        for indexed_term, ids in (
            self._token_index.items()
        ):

            similarity = (
                SequenceMatcher(
                    None,
                    term,
                    indexed_term,
                ).ratio()
            )

            if (
                similarity
                >= self.config.fuzzy_threshold
            ):

                candidates.update(
                    ids
                )

        return candidates

    # =================================================================
    # FIELD RETRIEVAL
    # =================================================================

    def _retrieve_field(
        self,
        field_name: str,
        values: Sequence[str],
    ) -> Set[int]:

        result = set()

        field_name = normalize_key(
            field_name
        )

        for document in (
            self._documents.values()
        ):

            value = self._field_value(
                document,
                field_name,
            )

            normalized = normalize_key(
                value
            )

            if any(
                normalize_key(item)
                in normalized
                for item in values
            ):

                result.add(
                    document.id
                )

        return result

    # =================================================================
    # FILTERING
    # =================================================================

    def _passes_filters(
        self,
        document: SearchDocument,
        profile: QueryProfile,
    ) -> bool:

        for field_name, expected in (
            profile.filters.items()
        ):

            if field_name in {
                "category",
                "categories",
            }:

                if normalize_key(
                    document.category
                ) != normalize_key(
                    expected
                ):

                    return False

            elif field_name in {
                "tag",
                "tags",
            }:

                actual_tags = {
                    normalize_key(tag)
                    for tag in document.tags
                }

                if isinstance(
                    expected,
                    list,
                ):

                    if not any(
                        normalize_key(tag)
                        in actual_tags
                        for tag in expected
                    ):

                        return False

                elif normalize_key(
                    expected
                ) not in actual_tags:

                    return False

            else:

                actual = self._field_value(
                    document,
                    field_name,
                )

                if not self._compare_filter(
                    actual,
                    expected,
                ):

                    return False

        # -------------------------------------------------------------
        # REQUIRED TERMS
        # -------------------------------------------------------------

        for term in (
            profile.required_terms
        ):

            if not self._document_contains_term(
                document,
                term,
            ):

                return False

        # -------------------------------------------------------------
        # PROHIBITED TERMS
        # -------------------------------------------------------------

        for term in (
            profile.prohibited_terms
        ):

            if self._document_contains_term(
                document,
                term,
            ):

                return False

        return True

    # -----------------------------------------------------------------
    # FILTER COMPARISON
    # -----------------------------------------------------------------

    def _compare_filter(
        self,
        actual: Any,
        expected: Any,
    ) -> bool:

        actual_key = normalize_key(
            actual
        )

        expected_key = normalize_key(
            expected
        )

        if actual_key == expected_key:
            return True

        return expected_key in actual_key

    # =================================================================
    # FIELD ACCESS
    # =================================================================

    def _field_value(
        self,
        document: SearchDocument,
        field_name: str,
    ) -> Any:

        if hasattr(
            document,
            field_name,
        ):

            return getattr(
                document,
                field_name,
            )

        return document.metadata.get(
            field_name
        )

    # =================================================================
    # SCORING
    # =================================================================

    def _score_document(
        self,
        document: SearchDocument,
        profile: QueryProfile,
    ) -> Optional[SearchResult]:

        title = normalize_key(
            document.title
        )

        content = normalize_key(
            document.content
        )

        tags = [
            normalize_key(tag)
            for tag in document.tags
        ]

        category = normalize_key(
            document.category
        )

        score = 0.0

        matched_terms = []
        matched_phrases = []
        matched_fields = []

        explanation = {
            "components": {},
            "matched_terms": [],
            "matched_phrases": [],
        }

        # -------------------------------------------------------------
        # EMPTY QUERY
        # -------------------------------------------------------------

        if not profile.terms and not profile.phrases:

            if not profile.filters:
                return None

            score = 0.01

        # -------------------------------------------------------------
        # TERM SCORING
        # -------------------------------------------------------------

        document_tokens = (
            self._document_tokens.get(
                document.id,
                Counter(),
            )
        )

        for term in profile.terms:

            term_score = 0.0

            title_count = (
                title.split().count(
                    term
                )
            )

            content_count = (
                document_tokens.get(
                    term,
                    0,
                )
            )

            tag_match = term in tags

            category_match = (
                term == category
            )

            if term in title:
                term_score += (
                    self.ranking_weights.term_title
                )

                matched_fields.append(
                    "title"
                )

            if title_count:
                term_score += (
                    title_count
                    * self.ranking_weights.term_title
                )

            if content_count:
                term_score += (
                    content_count
                    * self.ranking_weights.term_content
                )

                matched_fields.append(
                    "content"
                )

            if tag_match:

                term_score += (
                    self.ranking_weights.term_tag
                )

                matched_fields.append(
                    "tags"
                )

            if category_match:

                term_score += (
                    self.ranking_weights.exact_category
                )

                matched_fields.append(
                    "category"
                )

            if (
                normalize_key(
                    term
                ) in title
            ):

                term_score += (
                    self.ranking_weights.prefix_title
                )

            if term_score > 0:

                matched_terms.append(
                    term
                )

                explanation[
                    "matched_terms"
                ].append(
                    {
                        "term": term,
                        "score": term_score,
                    }
                )

            score += term_score

        # -------------------------------------------------------------
        # EXACT WHOLE QUERY
        # -------------------------------------------------------------

        query_text = profile.normalized

        if query_text:

            if query_text == title:

                score += (
                    self.ranking_weights.exact_title
                )

                explanation[
                    "components"
                ]["exact_title"] = (
                    self.ranking_weights.exact_title
                )

            if query_text in content:

                score += (
                    self.ranking_weights.exact_content
                )

                explanation[
                    "components"
                ]["exact_content"] = (
                    self.ranking_weights.exact_content
                )

            if query_text in tags:

                score += (
                    self.ranking_weights.exact_tag
                )

                explanation[
                    "components"
                ]["exact_tag"] = (
                    self.ranking_weights.exact_tag
                )

        # -------------------------------------------------------------
        # PHRASE MATCHING
        # -------------------------------------------------------------

        if self.config.enable_phrase_matching:

            for phrase in (
                profile.phrases
            ):

                if phrase in title:

                    score += (
                        self.ranking_weights.phrase_title
                    )

                    matched_phrases.append(
                        phrase
                    )

                    matched_fields.append(
                        "title"
                    )

                elif phrase in content:

                    score += (
                        self.ranking_weights.phrase_content
                    )

                    matched_phrases.append(
                        phrase
                    )

                    matched_fields.append(
                        "content"
                    )

        # -------------------------------------------------------------
        # WILDCARD SCORING
        # -------------------------------------------------------------

        for wildcard in (
            profile.wildcard_terms
        ):

            for indexed_term in (
                document_tokens.keys()
            ):

                if self._wildcard_match(
                    wildcard,
                    indexed_term,
                ):

                    score += (
                        self.config.prefix_weight
                    )

                    matched_terms.append(
                        indexed_term
                    )

        # -------------------------------------------------------------
        # FUZZY SCORING
        # -------------------------------------------------------------

        if self.config.enable_fuzzy:

            for fuzzy in (
                profile.fuzzy_terms
            ):

                best_similarity = 0.0

                for indexed_term in (
                    document_tokens.keys()
                ):

                    similarity = (
                        SequenceMatcher(
                            None,
                            fuzzy,
                            indexed_term,
                        ).ratio()
                    )

                    best_similarity = max(
                        best_similarity,
                        similarity,
                    )

                if (
                    best_similarity
                    >= self.config.fuzzy_threshold
                ):

                    score += (
                        best_similarity
                        * self.ranking_weights.fuzzy_content
                    )

                    matched_terms.append(
                        fuzzy
                    )

        # -------------------------------------------------------------
        # TERM COVERAGE
        # -------------------------------------------------------------

        if profile.terms:

            coverage = (
                len(
                    set(
                        matched_terms
                    )
                )
                / len(
                    set(
                        profile.terms
                    )
                )
            )

            score *= (
                0.75
                + (
                    0.25
                    * coverage
                )
            )

            explanation[
                "components"
            ]["coverage"] = coverage

        # -------------------------------------------------------------
        # RECENCY
        # -------------------------------------------------------------

        recency = self._recency_score(
            document
        )

        score += (
            recency
            * self.ranking_weights.recency
        )

        explanation[
            "components"
        ]["recency"] = recency

        # -------------------------------------------------------------
        # DOCUMENT LENGTH
        # -------------------------------------------------------------

        if document.token_count:

            penalty = min(
                1.0,
                math.log1p(
                    document.token_count
                )
                * self.config.length_penalty
                / 10.0,
            )

            score *= (
                1.0 - penalty
            )

            explanation[
                "components"
            ]["length_penalty"] = penalty

        # -------------------------------------------------------------
        # EXTERNAL RANKER
        # -------------------------------------------------------------

        score = self._apply_external_ranker(
            document,
            profile,
            score,
        )

        if score <= 0:
            return None

        return SearchResult(
            document=document,
            score=score,
            matched_terms=unique_preserving_order(
                matched_terms
            ),
            matched_phrases=unique_preserving_order(
                matched_phrases
            ),
            matched_fields=unique_preserving_order(
                matched_fields
            ),
            explanation=explanation,
        )

    # -----------------------------------------------------------------
    # EXTERNAL RANKER
    # -----------------------------------------------------------------

    def _apply_external_ranker(
        self,
        document: SearchDocument,
        profile: QueryProfile,
        score: float,
    ) -> float:

        if self.ranking_engine is None:
            return score

        methods = (
            "score",
            "rank",
            "calculate",
        )

        for method_name in methods:

            method = getattr(
                self.ranking_engine,
                method_name,
                None,
            )

            if method is None:
                continue

            attempts = [
                lambda: method(
                    document.as_dict(),
                    profile,
                ),
                lambda: method(
                    document.as_dict(),
                    profile.normalized,
                ),
                lambda: method(
                    document.as_dict(),
                    profile.terms,
                ),
            ]

            for attempt in attempts:

                try:

                    value = attempt()

                    if isinstance(
                        value,
                        Mapping,
                    ):

                        external_score = (
                            value.get(
                                "score"
                            )
                        )

                    else:

                        external_score = value

                    if external_score is not None:

                        return (
                            score
                            + safe_float(
                                external_score
                            )
                        )

                except Exception:
                    continue

        return score

    # =================================================================
    # RANKING
    # =================================================================

    def _rank_results(
        self,
        results: List[SearchResult],
        profile: QueryProfile,
    ) -> List[SearchResult]:

        if not results:
            return []

        # -------------------------------------------------------------
        # EXTERNAL RANKER
        # -------------------------------------------------------------

        results = self._external_rank_results(
            results,
            profile,
        )

        # -------------------------------------------------------------
        # SORT
        # -------------------------------------------------------------

        field_name = (
            profile.sort_field
            or "relevance"
        )

        reverse = (
            profile.sort_direction
            != "asc"
        )

        if field_name == "relevance":

            results.sort(
                key=lambda result: (
                    result.score,
                    result.document.id,
                ),
                reverse=True,
            )

        elif field_name == "title":

            results.sort(
                key=lambda result: normalize_key(
                    result.document.title
                ),
                reverse=reverse,
            )

        elif field_name == "created_at":

            results.sort(
                key=lambda result:
                    result.document.created_at,
                reverse=reverse,
            )

        elif field_name == "updated_at":

            results.sort(
                key=lambda result:
                    result.document.updated_at,
                reverse=reverse,
            )

        elif field_name == "popularity":

            results.sort(
                key=lambda result:
                    result.document.popularity,
                reverse=reverse,
            )

        else:

            results.sort(
                key=lambda result: safe_float(
                    result.document.metadata.get(
                        field_name,
                        0,
                    )
                ),
                reverse=reverse,
            )

        return results

    # -----------------------------------------------------------------
    # EXTERNAL RANK RESULTS
    # -----------------------------------------------------------------

    def _external_rank_results(
        self,
        results: List[SearchResult],
        profile: QueryProfile,
    ) -> List[SearchResult]:

        if self.ranking_engine is None:
            return results

        method = getattr(
            self.ranking_engine,
            "rank_results",
            None,
        )

        if method is None:
            return results

        try:

            raw = method(
                [
                    {
                        "document": result.document.as_dict(),
                        "score": result.score,
                    }
                    for result in results
                ],
                profile,
            )

            if not raw:
                return results

            return self._merge_external_ranking(
                results,
                raw,
            )

        except Exception:
            return results

    # -----------------------------------------------------------------
    # MERGE EXTERNAL RANKING
    # -----------------------------------------------------------------

    def _merge_external_ranking(
        self,
        results: List[SearchResult],
        external: Any,
    ) -> List[SearchResult]:

        score_map = {}

        if isinstance(
            external,
            Iterable,
        ):

            for position, item in enumerate(
                external
            ):

                if isinstance(
                    item,
                    Mapping,
                ):

                    item_id = item.get(
                        "id"
                    )

                    if item_id is None:
                        document = item.get(
                            "document"
                        )

                        if isinstance(
                            document,
                            Mapping,
                        ):
                            item_id = document.get(
                                "id"
                            )

                    if item_id is not None:

                        score_map[
                            safe_int(
                                item_id,
                                -1,
                            )
                        ] = safe_float(
                            item.get(
                                "score",
                                0,
                            )
                        )

                elif hasattr(
                    item,
                    "id",
                ):

                    score_map[
                        safe_int(
                            item.id,
                            -1,
                        )
                    ] = safe_float(
                        getattr(
                            item,
                            "score",
                            0,
                        )
                    )

        for result in results:

            external_score = score_map.get(
                result.document.id
            )

            if external_score is not None:

                result.score += (
                    external_score
                )

        return results

    # =================================================================
    # DIVERSIFICATION
    # =================================================================

    def _diversify_results(
        self,
        results: List[SearchResult],
    ) -> List[SearchResult]:

        if len(results) <= 2:
            return results

        selected = []

        for result in results:

            if not selected:

                selected.append(
                    result
                )
                continue

            too_similar = False

            for existing in selected[-5:]:

                similarity = (
                    self._document_similarity(
                        result.document,
                        existing.document,
                    )
                )

                if (
                    similarity
                    >= self.config.diversification_threshold
                ):

                    too_similar = True
                    break

            if not too_similar:

                selected.append(
                    result
                )

        # Don't accidentally destroy a small result set.
        if len(selected) < min(
            3,
            len(results),
        ):

            return results

        return selected

    # -----------------------------------------------------------------
    # DOCUMENT SIMILARITY
    # -----------------------------------------------------------------

    def _document_similarity(
        self,
        first: SearchDocument,
        second: SearchDocument,
    ) -> float:

        first_tokens = set(
            self._document_tokens.get(
                first.id,
                Counter(),
            ).keys()
        )

        second_tokens = set(
            self._document_tokens.get(
                second.id,
                Counter(),
            ).keys()
        )

        if not first_tokens or not second_tokens:
            return 0.0

        intersection = len(
            first_tokens
            & second_tokens
        )

        union = len(
            first_tokens
            | second_tokens
        )

        if not union:
            return 0.0

        return (
            intersection
            / union
        )

    # =================================================================
    # HIGHLIGHTING
    # =================================================================

    def _build_highlights(
        self,
        document: SearchDocument,
        profile: QueryProfile,
    ) -> Dict[str, str]:

        terms = unique_preserving_order(
            profile.terms
            + profile.fuzzy_terms
        )

        return {
            "title": self._highlight_text(
                document.title,
                terms,
            ),
            "content": self._highlight_text(
                document.content,
                terms,
            ),
        }

    # -----------------------------------------------------------------
    # HIGHLIGHT TEXT
    # -----------------------------------------------------------------

    def _highlight_text(
        self,
        text: str,
        terms: Sequence[str],
    ) -> str:

        result = text

        for term in sorted(
            terms,
            key=len,
            reverse=True,
        ):

            if not term:
                continue

            pattern = re.compile(
                re.escape(term),
                re.IGNORECASE,
            )

            result = pattern.sub(
                lambda match:
                    f"<mark>{match.group(0)}</mark>",
                result,
            )

        return result

    # =================================================================
    # EXPLANATIONS
    # =================================================================

    def _build_explanation(
        self,
        result: SearchResult,
        profile: QueryProfile,
    ) -> Dict[str, Any]:

        return {
            "score": result.score,
            "query": profile.normalized,
            "matched_terms": result.matched_terms,
            "matched_phrases": result.matched_phrases,
            "matched_fields": result.matched_fields,
            "components": result.explanation.get(
                "components",
                {},
            ),
            "ranking_weights": (
                self.ranking_weights.__dict__.copy()
            ),
        }

    # =================================================================
    # TITLE SEARCH
    # =================================================================

    def search_title(
        self,
        query,
        limit=10,
    ):

        query = normalize_key(
            query
        )

        results = []

        for document in (
            self._documents.values()
        ):

            if query in normalize_key(
                document.title
            ):

                results.append(
                    document.as_dict()
                )

        return results[:limit]

    # =================================================================
    # CONTENT SEARCH
    # =================================================================

    def search_content(
        self,
        query,
        limit=10,
    ):

        query = normalize_key(
            query
        )

        results = []

        for document in (
            self._documents.values()
        ):

            if query in normalize_key(
                document.content
            ):

                results.append(
                    document.as_dict()
                )

        return results[:limit]

    # =================================================================
    # CATEGORY
    # =================================================================

    def by_category(
        self,
        category,
    ):

        key = normalize_key(
            category
        )

        ids = self._category_index.get(
            key,
            set(),
        )

        return [
            self._documents[item_id].as_dict()
            for item_id in ids
            if item_id in self._documents
        ]

    # =================================================================
    # TAG
    # =================================================================

    def by_tag(
        self,
        tag,
    ):

        key = normalize_key(
            tag
        )

        ids = self._tag_index.get(
            key,
            set(),
        )

        return [
            self._documents[item_id].as_dict()
            for item_id in ids
            if item_id in self._documents
        ]

    # =================================================================
    # DUPLICATES
    # =================================================================

    def has_duplicate(
        self,
        content,
    ):

        fingerprint = self._fingerprint(
            "",
            normalize_text(content),
            "",
            [],
        )

        for document in (
            self._documents.values()
        ):

            if (
                normalize_key(
                    document.content
                )
                == normalize_key(
                    content
                )
            ):

                return True

            if (
                document.fingerprint
                == fingerprint
            ):

                return True

        return False

    # =================================================================
    # FIND SIMILAR
    # =================================================================

    def similar(
        self,
        item_id,
        limit=10,
    ) -> List[Dict[str, Any]]:

        document = self._documents.get(
            safe_int(
                item_id,
                -1,
            )
        )

        if document is None:
            return []

        scored = []

        for other in (
            self._documents.values()
        ):

            if other.id == document.id:
                continue

            similarity = (
                self._document_similarity(
                    document,
                    other,
                )
            )

            if similarity <= 0:
                continue

            item = other.as_dict()

            item["similarity"] = round(
                similarity,
                6,
            )

            scored.append(
                item
            )

        scored.sort(
            key=lambda item:
                item["similarity"],
            reverse=True,
        )

        return scored[:limit]

    # =================================================================
    # SUGGESTIONS
    # =================================================================

    def suggest(
        self,
        prefix,
        limit=10,
    ) -> List[str]:
        """
        Generate query suggestions from indexed vocabulary and history.
        """

        prefix = normalize_key(
            prefix
        )

        if not prefix:
            return []

        candidates = Counter()

        # -------------------------------------------------------------
        # HISTORY
        # -------------------------------------------------------------

        for entry in self.history.entries:

            if entry.normalized.startswith(
                prefix
            ):

                candidates[
                    entry.normalized
                ] += 10

        # -------------------------------------------------------------
        # TERMS
        # -------------------------------------------------------------

        for term in (
            self._token_index.keys()
        ):

            if term.startswith(
                prefix
            ):

                candidates[
                    term
                ] += 1

        return [
            value
            for value, _
            in candidates.most_common(
                limit
            )
        ]

    # =================================================================
    # ANALYSIS
    # =================================================================

    def analyze(
        self,
        query,
    ) -> Dict[str, Any]:
        """
        Analyze a query before executing it.
        """

        profile = self.parse_query(
            query
        )

        result = {
            "query": profile.raw,
            "normalized": profile.normalized,
            "intent": profile.intent,
            "terms": profile.terms,
            "phrases": profile.phrases,
            "required_terms": (
                profile.required_terms
            ),
            "prohibited_terms": (
                profile.prohibited_terms
            ),
            "filters": copy.deepcopy(
                profile.filters
            ),
            "fields": copy.deepcopy(
                profile.fields
            ),
            "wildcards": (
                profile.wildcard_terms
            ),
            "fuzzy_terms": (
                profile.fuzzy_terms
            ),
            "complexity": profile.complexity,
            "metadata": copy.deepcopy(
                profile.metadata
            ),
        }

        if self.analyzer_engine is not None:

            result = self._merge_external_analysis(
                result,
                profile,
            )

        return result

    # -----------------------------------------------------------------
    # EXTERNAL ANALYSIS
    # -----------------------------------------------------------------

    def _merge_external_analysis(
        self,
        current: Dict[str, Any],
        profile: QueryProfile,
    ) -> Dict[str, Any]:

        methods = (
            "analyze",
            "analyse",
        )

        for method_name in methods:

            method = getattr(
                self.analyzer_engine,
                method_name,
                None,
            )

            if method is None:
                continue

            for argument in (
                profile,
                profile.normalized,
                profile.raw,
            ):

                try:

                    external = method(
                        argument
                    )

                    if isinstance(
                        external,
                        Mapping,
                    ):

                        current.update(
                            external
                        )

                    return current

                except Exception:
                    continue

        return current

    # =================================================================
    # INDEXING
    # =================================================================

    def _index_document(
        self,
        document: SearchDocument,
    ) -> None:

        # -------------------------------------------------------------
        # TOKENS
        # -------------------------------------------------------------

        tokens = tokenize(
            document.content
        )

        if (
            len(tokens)
            > self.config.max_document_tokens
        ):

            tokens = tokens[
                :self.config.max_document_tokens
            ]

        counter = Counter(
            tokens
        )

        self._document_tokens[
            document.id
        ] = counter

        for token in counter:

            self._token_index[
                token
            ].add(
                document.id
            )

        # -------------------------------------------------------------
        # TITLE
        # -------------------------------------------------------------

        for token in tokenize(
            document.title
        ):

            self._title_index[
                token
            ].add(
                document.id
            )

        # -------------------------------------------------------------
        # TAGS
        # -------------------------------------------------------------

        for tag in document.tags:

            self._tag_index[
                normalize_key(tag)
            ].add(
                document.id
            )

        # -------------------------------------------------------------
        # CATEGORY
        # -------------------------------------------------------------

        self._category_index[
            normalize_key(
                document.category
            )
        ].add(
            document.id
        )

        # -------------------------------------------------------------
        # PREFIX INDEX
        # -------------------------------------------------------------

        for token in counter:

            maximum = min(
                len(token),
                20,
            )

            for size in range(
                1,
                maximum + 1,
            ):

                prefix = token[
                    :size
                ]

                self._prefix_index[
                    prefix
                ].add(
                    document.id
                )

        # -------------------------------------------------------------
        # DOCUMENT FREQUENCY
        # -------------------------------------------------------------

        for token in counter:

            self._term_document_frequency[
                token
            ] += 1

    # -----------------------------------------------------------------
    # DEINDEX
    # -----------------------------------------------------------------

    def _deindex_document(
        self,
        document: SearchDocument,
    ) -> None:

        document_id = document.id

        tokens = self._document_tokens.get(
            document_id,
            Counter(),
        )

        for token in tokens:

            ids = self._token_index.get(
                token
            )

            if ids:

                ids.discard(
                    document_id
                )

                if not ids:
                    self._token_index.pop(
                        token,
                        None,
                    )

            if (
                self._term_document_frequency.get(
                    token,
                    0,
                )
                > 0
            ):

                self._term_document_frequency[
                    token
                ] -= 1

        self._document_tokens.pop(
            document_id,
            None,
        )

        for index in (
            self._title_index,
            self._tag_index,
            self._category_index,
        ):

            for key in list(
                index.keys()
            ):

                index[key].discard(
                    document_id
                )

                if not index[key]:
                    index.pop(
                        key,
                        None,
                    )

        for prefix in list(
            self._prefix_index.keys()
        ):

            self._prefix_index[
                prefix
            ].discard(
                document_id
            )

            if not self._prefix_index[
                prefix
            ]:

                self._prefix_index.pop(
                    prefix,
                    None,
                )

    # =================================================================
    # FINGERPRINTING
    # =================================================================

    def _fingerprint(
        self,
        title: str,
        content: str,
        category: str,
        tags: Sequence[str],
    ) -> str:

        payload = "|".join(
            [
                normalize_key(title),
                normalize_key(content),
                normalize_key(category),
                ",".join(
                    sorted(
                        normalize_key(tag)
                        for tag in tags
                    )
                ),
            ]
        )

        return stable_hash(
            payload
        )

    # -----------------------------------------------------------------
    # FINGERPRINT EXISTS
    # -----------------------------------------------------------------

    def _fingerprint_exists(
        self,
        fingerprint: str,
    ) -> bool:

        return any(
            document.fingerprint
            == fingerprint
            for document
            in self._documents.values()
        )

    # -----------------------------------------------------------------
    # DOCUMENT BY FINGERPRINT
    # -----------------------------------------------------------------

    def _document_by_fingerprint(
        self,
        fingerprint: str,
    ) -> Optional[SearchDocument]:

        for document in (
            self._documents.values()
        ):

            if (
                document.fingerprint
                == fingerprint
            ):

                return document

        return None

    # =================================================================
    # TOKEN MATCHING
    # =================================================================

    def _document_contains_term(
        self,
        document: SearchDocument,
        term: str,
    ) -> bool:

        return term in self._document_tokens.get(
            document.id,
            Counter(),
        )

    # -----------------------------------------------------------------
    # WILDCARD MATCH
    # -----------------------------------------------------------------

    def _wildcard_match(
        self,
        pattern: str,
        value: str,
    ) -> bool:

        escaped = WILDCARD_ESCAPE.sub(
            r"\\\1",
            normalize_key(pattern),
        )

        escaped = escaped.replace(
            "*",
            ".*",
        )

        escaped = escaped.replace(
            "?",
            ".",
        )

        try:

            return bool(
                re.fullmatch(
                    escaped,
                    normalize_key(value),
                )
            )

        except re.error:

            return False

    # =================================================================
    # RECENCY
    # =================================================================

    def _recency_score(
        self,
        document: SearchDocument,
    ) -> float:

        try:

            created = datetime.fromisoformat(
                document.created_at
            )

            age_days = max(
                0.0,
                (
                    datetime.now()
                    - created
                ).total_seconds()
                / 86400.0,
            )

            return 1.0 / (
                1.0
                + (
                    age_days
                    / 30.0
                )
            )

        except Exception:

            return 0.0

    # =================================================================
    # INTENT
    # =================================================================

    def _detect_intent(
        self,
        query: str,
    ) -> str:

        normalized = normalize_key(
            query
        )

        if not normalized:
            return "unknown"

        if normalized.startswith(
            "/"
        ):
            return "command"

        question_words = {
            "who",
            "what",
            "when",
            "where",
            "why",
            "how",
            "which",
            "can",
            "could",
            "does",
            "is",
            "are",
        }

        first_word = (
            normalized.split()[0]
            if normalized.split()
            else ""
        )

        if (
            normalized.endswith("?")
            or first_word
            in question_words
        ):

            return "question"

        if any(
            word in normalized
            for word in (
                "official",
                "homepage",
                "login",
                "website",
                "download",
            )
        ):

            return "navigational"

        return "search"

    # =================================================================
    # CACHE
    # =================================================================

    def _build_cache_key(
        self,
        profile: QueryProfile,
        explanation: bool,
        highlight: bool,
        diversify: bool,
    ) -> str:

        payload = {
            "query": profile.normalized,
            "terms": profile.terms,
            "phrases": profile.phrases,
            "filters": profile.filters,
            "fields": profile.fields,
            "limit": profile.limit,
            "offset": profile.offset,
            "sort": profile.sort_field,
            "direction": profile.sort_direction,
            "explanation": explanation,
            "highlight": highlight,
            "diversify": diversify,
            "mutation": self._mutation_counter,
        }

        return stable_hash(
            repr(payload)
        )

    # -----------------------------------------------------------------
    # CACHE RESPONSE
    # -----------------------------------------------------------------

    def _cache_response(
        self,
        response: SearchResponse,
    ) -> Dict[str, Any]:

        return {
            "query": response.query,
            "results": [
                result.as_dict(
                    include_explanation=True
                )
                for result in response.results
            ],
            "total_candidates": (
                response.total_candidates
            ),
            "total_matches": (
                response.total_matches
            ),
            "took_ms": response.took_ms,
            "page": response.page,
            "limit": response.limit,
            "offset": response.offset,
            "query_profile": copy.deepcopy(
                response.query_profile.__dict__
            ),
            "metadata": copy.deepcopy(
                response.metadata
            ),
        }

    # -----------------------------------------------------------------
    # RESPONSE FROM CACHE
    # -----------------------------------------------------------------

    def _response_from_cached(
        self,
        cached: Dict[str, Any],
    ) -> SearchResponse:

        profile_data = cached.get(
            "query_profile",
            {},
        )

        profile = QueryProfile(
            raw=profile_data.get(
                "raw",
                cached.get(
                    "query",
                    "",
                ),
            ),
            normalized=profile_data.get(
                "normalized",
                normalize_key(
                    cached.get(
                        "query",
                        "",
                    )
                ),
            ),
            terms=profile_data.get(
                "terms",
                [],
            ),
            phrases=profile_data.get(
                "phrases",
                [],
            ),
            required_terms=profile_data.get(
                "required_terms",
                [],
            ),
            prohibited_terms=profile_data.get(
                "prohibited_terms",
                [],
            ),
            filters=profile_data.get(
                "filters",
                {},
            ),
            fields=profile_data.get(
                "fields",
                {},
            ),
            wildcard_terms=profile_data.get(
                "wildcard_terms",
                [],
            ),
            fuzzy_terms=profile_data.get(
                "fuzzy_terms",
                [],
            ),
            intent=profile_data.get(
                "intent",
                "search",
            ),
            sort_field=profile_data.get(
                "sort_field",
                "relevance",
            ),
            sort_direction=profile_data.get(
                "sort_direction",
                "desc",
            ),
            limit=profile_data.get(
                "limit",
                DEFAULT_LIMIT,
            ),
            offset=profile_data.get(
                "offset",
                0,
            ),
            complexity=profile_data.get(
                "complexity",
                0.0,
            ),
            metadata=profile_data.get(
                "metadata",
                {},
            ),
        )

        results = []

        for item in cached.get(
            "results",
            [],
        ):

            document = SearchDocument(
                id=safe_int(
                    item.get(
                        "id"
                    ),
                    -1,
                ),
                title=item.get(
                    "title",
                    "",
                ),
                content=item.get(
                    "content",
                    "",
                ),
                category=item.get(
                    "category",
                    "",
                ),
                tags=item.get(
                    "tags",
                    [],
                ),
                created_at=item.get(
                    "created_at",
                    "",
                ),
                updated_at=item.get(
                    "updated_at",
                    "",
                ),
                metadata=item.get(
                    "metadata",
                    {},
                ),
            )

            results.append(
                SearchResult(
                    document=document,
                    score=safe_float(
                        item.get(
                            "score",
                            0,
                        )
                    ),
                    rank=safe_int(
                        item.get(
                            "rank",
                            0,
                        )
                    ),
                    matched_terms=item.get(
                        "matched_terms",
                        [],
                    ),
                    matched_phrases=item.get(
                        "matched_phrases",
                        [],
                    ),
                    matched_fields=item.get(
                        "matched_fields",
                        [],
                    ),
                    highlights=item.get(
                        "highlights",
                        {},
                    ),
                    explanation=item.get(
                        "explanation",
                        {},
                    ),
                )
            )

        return SearchResponse(
            query=cached.get(
                "query",
                "",
            ),
            results=results,
            total_candidates=safe_int(
                cached.get(
                    "total_candidates",
                    0,
                )
            ),
            total_matches=safe_int(
                cached.get(
                    "total_matches",
                    0,
                )
            ),
            took_ms=safe_float(
                cached.get(
                    "took_ms",
                    0,
                )
            ),
            page=safe_int(
                cached.get(
                    "page",
                    1,
                )
            ),
            limit=safe_int(
                cached.get(
                    "limit",
                    DEFAULT_LIMIT,
                )
            ),
            offset=safe_int(
                cached.get(
                    "offset",
                    0,
                )
            ),
            query_profile=profile,
            metadata=cached.get(
                "metadata",
                {},
            ),
        )

    # =================================================================
    # EXPLAIN SEARCH
    # =================================================================

    def explain(
        self,
        query,
        limit=10,
    ) -> Dict[str, Any]:
        """
        Execute a search and return detailed diagnostic information.
        """

        response = self.search(
            query,
            limit=limit,
            include_explanation=True,
            highlight=True,
            use_cache=False,
        )

        return {
            "engine_version": (
                SEARCH_ENGINE_VERSION
            ),
            "query": query,
            "analysis": self.analyze(
                query
            ),
            "response": response.as_dict(
                include_explanation=True
            ),
        }

    # =================================================================
    # STATISTICS
    # =================================================================

    def count(self):
        """Return number of indexed documents."""
        return len(
            self._documents
        )

    # -----------------------------------------------------------------
    # CATEGORIES
    # -----------------------------------------------------------------

    def categories(self):

        return sorted(
            {
                document.category
                for document
                in self._documents.values()
                if document.category
            }
        )

    # -----------------------------------------------------------------
    # TAGS
    # -----------------------------------------------------------------

    def tags(self):

        return sorted(
            {
                tag
                for document
                in self._documents.values()
                for tag in document.tags
            }
        )

    # -----------------------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------------------

    def statistics(self):

        category_counts = Counter(
            document.category
            for document
            in self._documents.values()
        )

        tag_counts = Counter(
            tag
            for document
            in self._documents.values()
            for tag in document.tags
        )

        average_length = 0.0

        if self._documents:

            average_length = (
                sum(
                    document.token_count
                    for document
                    in self._documents.values()
                )
                / len(
                    self._documents
                )
            )

        return {
            "engine_version": (
                SEARCH_ENGINE_VERSION
            ),
            "total_items": self.count(),
            "unique_terms": len(
                self._token_index
            ),
            "unique_title_terms": len(
                self._title_index
            ),
            "unique_tags": len(
                self._tag_index
            ),
            "unique_categories": len(
                self._category_index
            ),
            "prefix_entries": len(
                self._prefix_index
            ),
            "average_document_tokens": (
                average_length
            ),
            "categories": dict(
                category_counts
            ),
            "tag_counts": dict(
                tag_counts
            ),
            "operations": dict(
                self._statistics
            ),
            "cache": self.cache.stats(),
            "history_size": len(
                self.history.entries
            ),
            "mutation_version": (
                self._mutation_counter
            ),
        }

    # =================================================================
    # HEALTH
    # =================================================================

    def health(self) -> Dict[str, Any]:
        """
        Run internal consistency checks.
        """

        errors = []
        warnings = []

        document_ids = set(
            self._documents.keys()
        )

        indexed_ids = set()

        for ids in (
            self._token_index.values()
        ):

            indexed_ids.update(
                ids
            )

        orphaned = (
            indexed_ids
            - document_ids
        )

        if orphaned:

            errors.append(
                "Token index contains orphaned IDs."
            )

        # -------------------------------------------------------------
        # DOCUMENT TOKEN CONSISTENCY
        # -------------------------------------------------------------

        for document_id in document_ids:

            if document_id not in (
                self._document_tokens
            ):

                errors.append(
                    f"Document {document_id} "
                    "has no token representation."
                )

        # -------------------------------------------------------------
        # PUBLIC ITEM CONSISTENCY
        # -------------------------------------------------------------

        public_ids = {
            item.get("id")
            for item in self.items
        }

        if public_ids != document_ids:

            warnings.append(
                "Public item list differs from internal document store."
            )

        return {
            "healthy": not errors,
            "errors": errors,
            "warnings": warnings,
            "documents": len(
                document_ids
            ),
            "indexed_terms": len(
                self._token_index
            ),
            "mutation_version": (
                self._mutation_counter
            ),
        }

    # =================================================================
    # REBUILD INDEX
    # =================================================================

    def rebuild_index(self) -> Dict[str, Any]:
        """
        Completely rebuild internal search structures.

        Useful after migrations or debugging.
        """

        with self._lock:

            self._token_index.clear()
            self._title_index.clear()
            self._tag_index.clear()
            self._category_index.clear()
            self._prefix_index.clear()

            self._document_tokens.clear()
            self._term_document_frequency.clear()

            for document in (
                self._documents.values()
            ):

                self._index_document(
                    document
                )

            self._mutation_counter += 1

            self.cache.clear()

            return self.health()

    # =================================================================
    # SNAPSHOT
    # =================================================================

    def snapshot(self) -> Dict[str, Any]:
        """
        Export the in-memory search state.

        This is intentionally a plain Python dictionary so another
        persistence layer can serialize it.
        """

        return {
            "version": SEARCH_ENGINE_VERSION,
            "next_id": self.next_id,
            "documents": [
                document.as_dict()
                for document
                in self._documents.values()
            ],
            "mutation_version": (
                self._mutation_counter
            ),
        }

    # =================================================================
    # RESTORE
    # =================================================================

    def restore(
        self,
        snapshot: Mapping[str, Any],
        clear_existing: bool = True,
    ) -> int:
        """
        Restore documents from a snapshot.
        """

        if clear_existing:
            self.clear()

        documents = snapshot.get(
            "documents",
            [],
        )

        restored = 0

        for item in documents:

            if not isinstance(
                item,
                Mapping,
            ):
                continue

            self.add(
                content=item.get(
                    "content",
                    "",
                ),
                title=item.get(
                    "title",
                    "Untitled",
                ),
                category=item.get(
                    "category",
                    "general",
                ),
                tags=item.get(
                    "tags",
                    [],
                ),
                metadata=item.get(
                    "metadata",
                    {},
                ),
                item_id=item.get(
                    "id"
                ),
            )

            restored += 1

        return restored

    # =================================================================
    # HISTORY API
    # =================================================================

    def recent_searches(
        self,
        limit=20,
    ):

        return self.history.recent(
            limit
        )

    def popular_searches(
        self,
        limit=20,
    ):

        return self.history.popular(
            limit
        )

    def clear_history(self):

        self.history.clear()

    # =================================================================
    # CACHE API
    # =================================================================

    def clear_cache(self):

        self.cache.clear()

    # =================================================================
    # LAST SEARCH
    # =================================================================

    def last_search(
        self,
    ) -> Optional[Dict[str, Any]]:

        if self._last_search is None:
            return None

        return self._last_search.as_dict()

    # =================================================================
    # PUBLIC ITEM SYNC
    # =================================================================

    def _sync_public_items(
        self,
    ) -> None:

        self.items = [
            document.as_dict()
            for document
            in self._documents.values()
        ]

    # =================================================================
    # CLEAR
    # =================================================================

    def clear(self):
        """
        Clear all documents and indexes.
        """

        with self._lock:

            self.items.clear()

            self._documents.clear()

            self._token_index.clear()
            self._title_index.clear()
            self._tag_index.clear()
            self._category_index.clear()
            self._prefix_index.clear()

            self._document_tokens.clear()
            self._term_document_frequency.clear()

            self.next_id = 1

            self._mutation_counter += 1

            self.cache.clear()

            self._last_search = None

    # =================================================================
    # BATCH ADD
    # =================================================================

    def add_many(
        self,
        documents: Iterable[
            Mapping[str, Any]
        ],
    ) -> List[Dict[str, Any]]:
        """
        Add many documents.

        This provides a cleaner ingestion API for research/import
        systems.
        """

        results = []

        for document in documents:

            if not isinstance(
                document,
                Mapping,
            ):
                continue

            results.append(
                self.add(
                    content=document.get(
                        "content",
                        "",
                    ),
                    title=document.get(
                        "title",
                        "Untitled",
                    ),
                    category=document.get(
                        "category",
                        "general",
                    ),
                    tags=document.get(
                        "tags",
                        [],
                    ),
                    metadata=document.get(
                        "metadata",
                        {},
                    ),
                    item_id=document.get(
                        "id"
                    ),
                )
            )

        return results

    # =================================================================
    # INTERNAL DOCUMENT LOOKUP
    # =================================================================

    def _document_tokens_for(
        self,
        item_id: int,
    ) -> Counter:

        return self._document_tokens.get(
            item_id,
            Counter(),
        )

    # =================================================================
    # FIELD SEARCH
    # =================================================================

    def search_field(
        self,
        field_name: str,
        value: str,
        limit=10,
    ) -> List[Dict[str, Any]]:

        field_name = normalize_key(
            field_name
        )

        value = normalize_key(
            value
        )

        results = []

        for document in (
            self._documents.values()
        ):

            actual = normalize_key(
                self._field_value(
                    document,
                    field_name,
                )
            )

            if value in actual:

                results.append(
                    document.as_dict()
                )

        return results[:limit]

    # =================================================================
    # EXACT SEARCH
    # =================================================================

    def exact(
        self,
        query,
        limit=10,
    ) -> List[Dict[str, Any]]:

        normalized = normalize_key(
            query
        )

        results = []

        for document in (
            self._documents.values()
        ):

            if (
                normalized
                == normalize_key(
                    document.title
                )
                or normalized
                == normalize_key(
                    document.content
                )
            ):

                results.append(
                    document.as_dict()
                )

        return results[:limit]

    # =================================================================
    # PREFIX SEARCH
    # =================================================================

    def prefix(
        self,
        prefix,
        limit=10,
    ) -> List[Dict[str, Any]]:

        prefix = normalize_key(
            prefix
        )

        ids = self._prefix_index.get(
            prefix,
            set(),
        )

        return [
            self._documents[item_id].as_dict()
            for item_id in list(ids)[:limit]
            if item_id in self._documents
        ]

    # =================================================================
    # TERM INFORMATION
    # =================================================================

    def term_info(
        self,
        term,
    ) -> Dict[str, Any]:

        term = normalize_key(
            term
        )

        ids = self._token_index.get(
            term,
            set(),
        )

        return {
            "term": term,
            "document_frequency": len(
                ids
            ),
            "documents": sorted(
                ids
            ),
            "exists": bool(
                ids
            ),
        }

    # =================================================================
    # VOCABULARY
    # =================================================================

    def vocabulary(
        self,
        limit=100,
    ) -> List[Dict[str, Any]]:

        values = self._term_document_frequency.most_common(
            limit
        )

        return [
            {
                "term": term,
                "document_frequency": count,
            }
            for term, count in values
        ]

    # =================================================================
    # DEBUG PIPELINE
    # =================================================================

    def debug_pipeline(
        self,
        query,
    ) -> Dict[str, Any]:

        profile = self.parse_query(
            query
        )

        candidates = self._retrieve_candidates(
            profile
        )

        filtered = [
            item_id
            for item_id in candidates
            if self._passes_filters(
                self._documents[item_id],
                profile,
            )
        ]

        scored = []

        for item_id in filtered:

            result = self._score_document(
                self._documents[item_id],
                profile,
            )

            if result is not None:

                scored.append(
                    result.as_dict(
                        include_explanation=True
                    )
                )

        return {
            "query": query,
            "profile": profile.__dict__,
            "candidate_count": len(
                candidates
            ),
            "candidates": sorted(
                candidates
            ),
            "filtered_count": len(
                filtered
            ),
            "scored_count": len(
                scored
            ),
            "scored": scored,
        }


# =====================================================================
# DEFAULT ENGINE
# =====================================================================

search = Search()


# =====================================================================
# CONVENIENCE FUNCTIONS
# =====================================================================


def add(
    content,
    title="Untitled",
    category="general",
    tags=None,
    metadata=None,
):
    return search.add(
        content=content,
        title=title,
        category=category,
        tags=tags,
        metadata=metadata,
    )


def find(
    query,
    category=None,
    tags=None,
    limit=10,
):
    return search.find(
        query=query,
        category=category,
        tags=tags,
        limit=limit,
    )


def search_query(
    query,
    limit=10,
    offset=0,
):
    return search.search(
        query=query,
        limit=limit,
        offset=offset,
    )


def suggest(
    prefix,
    limit=10,
):
    return search.suggest(
        prefix,
        limit,
    )


def statistics():
    return search.statistics()


# =====================================================================
# SELF TEST
# =====================================================================


def _self_test():
    """
    Basic internal test suite.

    This deliberately tests the public API rather than implementation
    details.
    """

    engine = Search()

    first = engine.add(
        title="Python Programming",
        content=(
            "Python is a programming language "
            "used for software development."
        ),
        category="technology",
        tags=[
            "python",
            "programming",
        ],
    )

    second = engine.add(
        title="Machine Learning",
        content=(
            "Machine learning uses algorithms "
            "to learn patterns from data."
        ),
        category="science",
        tags=[
            "ai",
            "machine-learning",
        ],
    )

    third = engine.add(
        title="Python Data Science",
        content=(
            "Python is widely used for data "
            "science and machine learning."
        ),
        category="technology",
        tags=[
            "python",
            "data",
            "science",
        ],
    )

    assert engine.count() == 3

    results = engine.find(
        "python"
    )

    assert results

    assert results[0]["score"] > 0

    category_results = engine.find(
        "python",
        category="technology",
    )

    assert category_results

    tag_results = engine.by_tag(
        "python"
    )

    assert len(
        tag_results
    ) == 2

    title_results = engine.search_title(
        "machine"
    )

    assert title_results

    similar = engine.similar(
        third["id"]
    )

    assert similar

    suggestions = engine.suggest(
        "py"
    )

    assert "python" in suggestions

    analysis = engine.analyze(
        "python programming"
    )

    assert analysis["terms"]

    health = engine.health()

    assert health["healthy"]

    snapshot = engine.snapshot()

    restored = Search()

    count = restored.restore(
        snapshot
    )

    assert count == 3

    assert restored.count() == 3

    print(
        "Search engine self-test passed."
    )


# =====================================================================
# MODULE EXECUTION
# =====================================================================

if __name__ == "__main__":
    _self_test()