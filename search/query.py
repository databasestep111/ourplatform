"""
search/query.py

Advanced query parsing and representation layer.

Responsibilities
----------------
- Normalize raw search queries.
- Tokenize query syntax.
- Detect phrases.
- Detect boolean operators.
- Parse field-specific expressions.
- Parse filters and ranges.
- Parse sorting and pagination hints.
- Detect query intent.
- Calculate query metadata.
- Build a structured Query object.
- Provide safe serialization/deserialization.
- Provide helper methods for downstream retrieval/ranking.

This module intentionally does NOT retrieve documents.
It transforms user input into a structured representation that
the retrieval and ranking layers can understand.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# ENUMERATIONS
# ============================================================


class QueryIntent(str, Enum):
    UNKNOWN = "unknown"
    SEARCH = "search"
    QUESTION = "question"
    NAVIGATIONAL = "navigational"
    INFORMATIONAL = "informational"
    FILTERED = "filtered"
    COMMAND = "command"


class TokenType(str, Enum):
    TERM = "term"
    PHRASE = "phrase"
    OPERATOR = "operator"
    FIELD = "field"
    FILTER = "filter"
    RANGE = "range"
    SORT = "sort"
    LIMIT = "limit"
    OFFSET = "offset"
    GROUP = "group"
    UNKNOWN = "unknown"


class BooleanOperator(str, Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class SortDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class QueryConfig:
    """
    Configuration controlling query parsing behaviour.
    """

    default_limit: int = 20
    maximum_limit: int = 1000

    default_operator: BooleanOperator = BooleanOperator.AND

    enable_boolean_operators: bool = True
    enable_field_queries: bool = True
    enable_ranges: bool = True
    enable_phrases: bool = True
    enable_sorting: bool = True
    enable_pagination: bool = True
    enable_intent_detection: bool = True

    minimum_term_length: int = 1
    maximum_query_length: int = 4096

    lowercase_terms: bool = True
    strip_punctuation: bool = False

    allow_wildcards: bool = True
    allow_fuzzy_queries: bool = True


DEFAULT_CONFIG = QueryConfig()


# ============================================================
# QUERY TOKENS
# ============================================================


@dataclass
class QueryToken:
    """
    Individual parsed component of a query.
    """

    token_type: TokenType
    value: str

    position: int = 0
    field: Optional[str] = None

    operator: Optional[str] = None

    quoted: bool = False
    wildcard: bool = False
    fuzzy: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# QUERY FILTERS
# ============================================================


@dataclass
class QueryFilter:
    """
    Represents a structured metadata filter.
    """

    field: str
    operator: str
    value: Any

    negated: bool = False

    minimum: Any = None
    maximum: Any = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# QUERY CLAUSES
# ============================================================


@dataclass
class QueryClause:
    """
    Represents a searchable clause.

    Example:

        title:"machine learning"

    becomes approximately:

        field = "title"
        value = "machine learning"
    """

    value: str

    field: Optional[str] = None

    token_type: TokenType = TokenType.TERM

    required: bool = False
    prohibited: bool = False

    exact: bool = False

    wildcard: bool = False
    fuzzy: bool = False

    boost: float = 1.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# SORTING
# ============================================================


@dataclass
class SortSpec:
    """
    Describes result ordering.
    """

    field: str = "relevance"
    direction: SortDirection = SortDirection.DESC

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# PAGINATION
# ============================================================


@dataclass
class Pagination:
    """
    Pagination information.
    """

    limit: int = 20
    offset: int = 0

    @property
    def page(self) -> int:
        if self.limit <= 0:
            return 1

        return (self.offset // self.limit) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "limit": self.limit,
            "offset": self.offset,
            "page": self.page,
        }


# ============================================================
# QUERY METADATA
# ============================================================


@dataclass
class QueryMetadata:
    """
    Information about how a query was interpreted.
    """

    original_length: int = 0
    term_count: int = 0
    phrase_count: int = 0
    filter_count: int = 0

    has_boolean_logic: bool = False
    has_wildcards: bool = False
    has_fuzzy_terms: bool = False
    has_field_queries: bool = False
    has_sorting: bool = False

    complexity: float = 0.0

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    warnings: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# QUERY OBJECT
# ============================================================


@dataclass
class Query:
    """
    Complete structured representation of a search query.
    """

    original: str
    normalized: str

    clauses: List[QueryClause] = field(
        default_factory=list
    )

    filters: List[QueryFilter] = field(
        default_factory=list
    )

    tokens: List[QueryToken] = field(
        default_factory=list
    )

    operators: List[str] = field(
        default_factory=list
    )

    sort: SortSpec = field(
        default_factory=SortSpec
    )

    pagination: Pagination = field(
        default_factory=Pagination
    )

    intent: QueryIntent = QueryIntent.UNKNOWN

    metadata: QueryMetadata = field(
        default_factory=QueryMetadata
    )

    valid: bool = True

    errors: List[str] = field(
        default_factory=list
    )

    def terms(self) -> List[str]:
        """
        Return searchable terms.
        """

        return [
            clause.value
            for clause in self.clauses
            if clause.token_type == TokenType.TERM
        ]

    def phrases(self) -> List[str]:
        """
        Return exact phrase clauses.
        """

        return [
            clause.value
            for clause in self.clauses
            if clause.token_type == TokenType.PHRASE
        ]

    def fields(self) -> List[str]:
        """
        Return fields referenced by the query.
        """

        return list(
            {
                clause.field
                for clause in self.clauses
                if clause.field
            }
        )

    def has_filter(self, field_name: str) -> bool:
        return any(
            item.field.lower() == field_name.lower()
            for item in self.filters
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "clauses": [
                clause.to_dict()
                for clause in self.clauses
            ],
            "filters": [
                item.to_dict()
                for item in self.filters
            ],
            "tokens": [
                token.to_dict()
                for token in self.tokens
            ],
            "operators": self.operators,
            "sort": self.sort.to_dict(),
            "pagination": self.pagination.to_dict(),
            "intent": self.intent.value,
            "metadata": self.metadata.to_dict(),
            "valid": self.valid,
            "errors": self.errors,
        }


# ============================================================
# QUERY PARSER
# ============================================================


class QueryParser:
    """
    Main query parser.

    The parser is deliberately independent from the search index.
    """

    BOOLEAN_OPERATORS = {
        "AND",
        "OR",
        "NOT",
    }

    SORT_PATTERN = re.compile(
        r"^sort:([A-Za-z0-9_.-]+)"
        r"(?:[:=](asc|desc))?$",
        re.IGNORECASE,
    )

    LIMIT_PATTERN = re.compile(
        r"^(?:limit|size):(\d+)$",
        re.IGNORECASE,
    )

    OFFSET_PATTERN = re.compile(
        r"^offset:(\d+)$",
        re.IGNORECASE,
    )

    FIELD_PATTERN = re.compile(
        r"^([A-Za-z_][A-Za-z0-9_.-]*):(.+)$"
    )

    RANGE_PATTERN = re.compile(
        r"^([A-Za-z_][A-Za-z0-9_.-]*)"
        r"(>=|<=|>|<|=)"
        r"(.+)$"
    )

    BOOST_PATTERN = re.compile(
        r"^(.+)\^([0-9]+(?:\.[0-9]+)?)$"
    )

    FUZZY_PATTERN = re.compile(
        r"^(.+?)~([0-9]*)$"
    )

    def __init__(
        self,
        config: Optional[QueryConfig] = None,
    ):
        self.config = config or QueryConfig()

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def parse(self, raw_query: str) -> Query:
        """
        Parse a raw query into a Query object.
        """

        if raw_query is None:
            raw_query = ""

        original = str(raw_query)

        if len(original) > self.config.maximum_query_length:
            original = original[
                :self.config.maximum_query_length
            ]

        normalized = self.normalize(original)

        query = Query(
            original=original,
            normalized=normalized,
        )

        if not normalized:
            query.valid = False
            query.errors.append(
                "Query is empty."
            )
            return query

        try:
            raw_tokens = self._lex(normalized)

            query.tokens = self._classify_tokens(
                raw_tokens
            )

            self._extract_special_tokens(query)

            query.clauses = self._build_clauses(
                query.tokens
            )

            query.filters = self._extract_filters(
                query.tokens
            )

            query.intent = self._detect_intent(
                query
            )

            query.metadata = self._build_metadata(
                query
            )

            self._validate(query)

        except Exception as error:
            query.valid = False
            query.errors.append(
                f"Query parsing failed: {error}"
            )

        return query

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    def normalize(self, query: str) -> str:
        """
        Normalize whitespace and optionally casing.

        Quoted phrases are preserved as much as possible.
        """

        query = query.strip()

        if not query:
            return ""

        query = re.sub(
            r"\s+",
            " ",
            query,
        )

        if self.config.strip_punctuation:
            query = re.sub(
                r"[^\w\s:\"<>=~*.^-]",
                " ",
                query,
            )

        if self.config.lowercase_terms:
            query = self._lowercase_outside_quotes(
                query
            )

        return re.sub(
            r"\s+",
            " ",
            query,
        ).strip()

    def _lowercase_outside_quotes(
        self,
        text: str,
    ) -> str:

        result = []
        quoted = False

        for char in text:
            if char == '"':
                quoted = not quoted
                result.append(char)
                continue

            if quoted:
                result.append(char)
            else:
                result.append(char.lower())

        return "".join(result)

    # --------------------------------------------------------
    # LEXING
    # --------------------------------------------------------

    def _lex(
        self,
        query: str,
    ) -> List[Tuple[str, bool]]:
        """
        Split the query while preserving phrase information.

        Returns:

            [
                ("hello", False),
                ("machine learning", True),
            ]
        """

        try:
            lexer = shlex.shlex(
                query,
                posix=True,
            )

            lexer.whitespace_split = True
            lexer.commenters = ""

            return [
                (token, self._was_quoted(query, token))
                for token in lexer
            ]

        except ValueError:
            return self._fallback_lex(query)

    def _fallback_lex(
        self,
        query: str,
    ) -> List[Tuple[str, bool]]:

        pattern = re.compile(
            r'"([^"]*)"|(\S+)'
        )

        results = []

        for match in pattern.finditer(query):
            if match.group(1) is not None:
                results.append(
                    (match.group(1), True)
                )
            else:
                results.append(
                    (match.group(2), False)
                )

        return results

    def _was_quoted(
        self,
        query: str,
        token: str,
    ) -> bool:

        return (
            f'"{token}"' in query
            or "'" + token + "'" in query
        )

    # --------------------------------------------------------
    # TOKEN CLASSIFICATION
    # --------------------------------------------------------

    def _classify_tokens(
        self,
        raw_tokens: List[Tuple[str, bool]],
    ) -> List[QueryToken]:

        tokens = []

        for position, (value, quoted) in enumerate(
            raw_tokens
        ):

            token_type = TokenType.TERM

            upper = value.upper()

            if (
                self.config.enable_boolean_operators
                and upper in self.BOOLEAN_OPERATORS
                and not quoted
            ):
                token_type = TokenType.OPERATOR

            elif quoted:
                token_type = TokenType.PHRASE

            elif self.SORT_PATTERN.match(value):
                token_type = TokenType.SORT

            elif self.LIMIT_PATTERN.match(value):
                token_type = TokenType.LIMIT

            elif self.OFFSET_PATTERN.match(value):
                token_type = TokenType.OFFSET

            elif (
                self.config.enable_ranges
                and self.RANGE_PATTERN.match(value)
            ):
                token_type = TokenType.RANGE

            elif (
                self.config.enable_field_queries
                and self.FIELD_PATTERN.match(value)
            ):
                token_type = TokenType.FIELD

            token = QueryToken(
                token_type=token_type,
                value=value,
                position=position,
                quoted=quoted,
                wildcard=(
                    "*" in value
                    or "?" in value
                ),
                fuzzy=(
                    "~" in value
                ),
            )

            if token_type == TokenType.OPERATOR:
                token.operator = upper

            tokens.append(token)

        return tokens

    # --------------------------------------------------------
    # SPECIAL TOKEN EXTRACTION
    # --------------------------------------------------------

    def _extract_special_tokens(
        self,
        query: Query,
    ) -> None:

        for token in query.tokens:

            if token.token_type == TokenType.OPERATOR:
                query.operators.append(
                    token.operator
                )

            elif token.token_type == TokenType.SORT:
                match = self.SORT_PATTERN.match(
                    token.value
                )

                if match:
                    field_name = match.group(1)
                    direction = (
                        match.group(2)
                        or "desc"
                    )

                    query.sort = SortSpec(
                        field=field_name,
                        direction=SortDirection(
                            direction.lower()
                        ),
                    )

            elif token.token_type == TokenType.LIMIT:
                match = self.LIMIT_PATTERN.match(
                    token.value
                )

                if match:
                    value = int(match.group(1))

                    query.pagination.limit = min(
                        value,
                        self.config.maximum_limit,
                    )

            elif token.token_type == TokenType.OFFSET:
                match = self.OFFSET_PATTERN.match(
                    token.value
                )

                if match:
                    query.pagination.offset = int(
                        match.group(1)
                    )

    # --------------------------------------------------------
    # CLAUSE BUILDING
    # --------------------------------------------------------

    def _build_clauses(
        self,
        tokens: List[QueryToken],
    ) -> List[QueryClause]:

        clauses = []

        pending_operator = (
            self.config.default_operator
        )

        for token in tokens:

            if token.token_type in {
                TokenType.SORT,
                TokenType.LIMIT,
                TokenType.OFFSET,
            }:
                continue

            if token.token_type == TokenType.OPERATOR:
                try:
                    pending_operator = (
                        BooleanOperator(
                            token.operator
                        )
                    )
                except ValueError:
                    pending_operator = (
                        self.config.default_operator
                    )

                continue

            if token.token_type == TokenType.RANGE:
                continue

            field_name = None
            value = token.value
            exact = (
                token.token_type
                == TokenType.PHRASE
            )

            if token.token_type == TokenType.FIELD:
                parsed = self._parse_field_token(
                    value
                )

                if parsed:
                    field_name, value, exact = parsed

            fuzzy = False
            wildcard = False
            boost = 1.0

            value, fuzzy = self._parse_fuzzy(
                value
            )

            value, boost = self._parse_boost(
                value
            )

            wildcard = (
                "*" in value
                or "?" in value
            )

            clause = QueryClause(
                value=value,
                field=field_name,
                token_type=(
                    TokenType.PHRASE
                    if exact
                    else TokenType.TERM
                ),
                required=(
                    pending_operator
                    == BooleanOperator.AND
                ),
                prohibited=(
                    pending_operator
                    == BooleanOperator.NOT
                ),
                exact=exact,
                wildcard=wildcard,
                fuzzy=fuzzy,
                boost=boost,
            )

            clauses.append(clause)

            pending_operator = (
                self.config.default_operator
            )

        return clauses

    # --------------------------------------------------------
    # FIELD PARSING
    # --------------------------------------------------------

    def _parse_field_token(
        self,
        token: str,
    ) -> Optional[Tuple[str, str, bool]]:

        match = self.FIELD_PATTERN.match(token)

        if not match:
            return None

        field_name = match.group(1)
        value = match.group(2)

        exact = (
            value.startswith('"')
            and value.endswith('"')
        )

        value = value.strip('"')

        return (
            field_name,
            value,
            exact,
        )

    # --------------------------------------------------------
    # FILTER PARSING
    # --------------------------------------------------------

    def _extract_filters(
        self,
        tokens: List[QueryToken],
    ) -> List[QueryFilter]:

        filters = []

        for token in tokens:

            if token.token_type == TokenType.RANGE:
                parsed = self._parse_range(
                    token.value
                )

                if parsed:
                    filters.append(parsed)

                continue

            if token.token_type != TokenType.FIELD:
                continue

            parsed = self._parse_field_token(
                token.value
            )

            if not parsed:
                continue

            field_name, value, _ = parsed

            if self._looks_like_filter(
                field_name,
                value,
            ):
                filters.append(
                    QueryFilter(
                        field=field_name,
                        operator="=",
                        value=value,
                    )
                )

        return filters

    def _parse_range(
        self,
        value: str,
    ) -> Optional[QueryFilter]:

        match = self.RANGE_PATTERN.match(
            value
        )

        if not match:
            return None

        field_name = match.group(1)
        operator = match.group(2)
        raw_value = match.group(3)

        converted = self._convert_value(
            raw_value
        )

        return QueryFilter(
            field=field_name,
            operator=operator,
            value=converted,
        )

    def _looks_like_filter(
        self,
        field_name: str,
        value: str,
    ) -> bool:

        filter_fields = {
            "category",
            "categories",
            "tag",
            "tags",
            "type",
            "author",
            "language",
            "date",
            "created",
            "updated",
            "status",
        }

        return field_name.lower() in filter_fields

    # --------------------------------------------------------
    # VALUE CONVERSION
    # --------------------------------------------------------

    def _convert_value(
        self,
        value: str,
    ) -> Any:

        value = value.strip(
            '"\''
        )

        if value.lower() == "true":
            return True

        if value.lower() == "false":
            return False

        if value.lower() in {
            "null",
            "none",
        }:
            return None

        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            pass

        return value

    # --------------------------------------------------------
    # FUZZY SEARCH
    # --------------------------------------------------------

    def _parse_fuzzy(
        self,
        value: str,
    ) -> Tuple[str, bool]:

        if not self.config.allow_fuzzy_queries:
            return value, False

        match = self.FUZZY_PATTERN.match(
            value
        )

        if not match:
            return value, False

        term = match.group(1)

        if not term:
            return value, False

        return term, True

    # --------------------------------------------------------
    # BOOSTING
    # --------------------------------------------------------

    def _parse_boost(
        self,
        value: str,
    ) -> Tuple[str, float]:

        match = self.BOOST_PATTERN.match(
            value
        )

        if not match:
            return value, 1.0

        term = match.group(1)

        try:
            boost = float(
                match.group(2)
            )
        except ValueError:
            boost = 1.0

        return term, boost

    # --------------------------------------------------------
    # INTENT DETECTION
    # --------------------------------------------------------

    def _detect_intent(
        self,
        query: Query,
    ) -> QueryIntent:

        if not self.config.enable_intent_detection:
            return QueryIntent.UNKNOWN

        original = query.original.strip()

        if not original:
            return QueryIntent.UNKNOWN

        if original.startswith("/"):
            return QueryIntent.COMMAND

        if query.filters:
            return QueryIntent.FILTERED

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
            "is",
            "are",
            "does",
            "do",
        }

        first_word = (
            original.lower()
            .split()[0]
            if original.split()
            else ""
        )

        if (
            original.endswith("?")
            or first_word in question_words
        ):
            return QueryIntent.QUESTION

        navigational_terms = {
            "login",
            "homepage",
            "website",
            "official",
            "download",
        }

        if any(
            term in original.lower()
            for term in navigational_terms
        ):
            return QueryIntent.NAVIGATIONAL

        return QueryIntent.SEARCH

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    def _build_metadata(
        self,
        query: Query,
    ) -> QueryMetadata:

        metadata = QueryMetadata()

        metadata.original_length = len(
            query.original
        )

        metadata.term_count = len(
            query.terms()
        )

        metadata.phrase_count = len(
            query.phrases()
        )

        metadata.filter_count = len(
            query.filters
        )

        metadata.has_boolean_logic = bool(
            query.operators
        )

        metadata.has_wildcards = any(
            clause.wildcard
            for clause in query.clauses
        )

        metadata.has_fuzzy_terms = any(
            clause.fuzzy
            for clause in query.clauses
        )

        metadata.has_field_queries = any(
            clause.field
            for clause in query.clauses
        )

        metadata.has_sorting = (
            query.sort.field != "relevance"
        )

        metadata.complexity = (
            self._calculate_complexity(query)
        )

        return metadata

    def _calculate_complexity(
        self,
        query: Query,
    ) -> float:

        score = 0.0

        score += len(query.clauses) * 1.0
        score += len(query.filters) * 2.0
        score += len(query.operators) * 1.5

        score += sum(
            2.0
            for clause in query.clauses
            if clause.exact
        )

        score += sum(
            1.5
            for clause in query.clauses
            if clause.fuzzy
        )

        score += sum(
            1.0
            for clause in query.clauses
            if clause.wildcard
        )

        return round(
            score,
            3,
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def _validate(
        self,
        query: Query,
    ) -> None:

        if not query.clauses and not query.filters:
            query.valid = False

            query.errors.append(
                "Query contains no searchable clauses."
            )

        if (
            query.pagination.limit
            <= 0
        ):
            query.pagination.limit = (
                self.config.default_limit
            )

        if (
            query.pagination.offset
            < 0
        ):
            query.pagination.offset = 0

        for clause in query.clauses:

            if len(
                clause.value
            ) < self.config.minimum_term_length:

                query.metadata.warnings.append(
                    "Query contains a very short term."
                )

            if (
                clause.wildcard
                and not self.config.allow_wildcards
            ):
                query.errors.append(
                    "Wildcards are disabled."
                )

        if query.errors:
            query.valid = False


# ============================================================
# QUERY BUILDER
# ============================================================


class QueryBuilder:
    """
    Programmatic query builder.

    Useful when another subsystem wants to create a Query
    without parsing raw text.
    """

    def __init__(
        self,
        config: Optional[QueryConfig] = None,
    ):
        self.config = config or QueryConfig()

        self._clauses = []
        self._filters = []
        self._operators = []

        self._sort = SortSpec()
        self._pagination = Pagination(
            limit=self.config.default_limit
        )

        self._intent = QueryIntent.SEARCH

    def term(
        self,
        value: str,
        field_name: Optional[str] = None,
        boost: float = 1.0,
    ) -> "QueryBuilder":

        self._clauses.append(
            QueryClause(
                value=value,
                field=field_name,
                boost=boost,
            )
        )

        return self

    def phrase(
        self,
        value: str,
        field_name: Optional[str] = None,
    ) -> "QueryBuilder":

        self._clauses.append(
            QueryClause(
                value=value,
                field=field_name,
                token_type=TokenType.PHRASE,
                exact=True,
            )
        )

        return self

    def require(
        self,
        value: str,
        field_name: Optional[str] = None,
    ) -> "QueryBuilder":

        self._clauses.append(
            QueryClause(
                value=value,
                field=field_name,
                required=True,
            )
        )

        return self

    def exclude(
        self,
        value: str,
        field_name: Optional[str] = None,
    ) -> "QueryBuilder":

        self._clauses.append(
            QueryClause(
                value=value,
                field=field_name,
                prohibited=True,
            )
        )

        return self

    def filter(
        self,
        field_name: str,
        operator: str,
        value: Any,
    ) -> "QueryBuilder":

        self._filters.append(
            QueryFilter(
                field=field_name,
                operator=operator,
                value=value,
            )
        )

        return self

    def sort(
        self,
        field_name: str,
        direction: str = "desc",
    ) -> "QueryBuilder":

        self._sort = SortSpec(
            field=field_name,
            direction=SortDirection(
                direction.lower()
            ),
        )

        return self

    def paginate(
        self,
        limit: int,
        offset: int = 0,
    ) -> "QueryBuilder":

        self._pagination = Pagination(
            limit=min(
                max(limit, 1),
                self.config.maximum_limit,
            ),
            offset=max(
                offset,
                0,
            ),
        )

        return self

    def intent(
        self,
        intent: QueryIntent,
    ) -> "QueryBuilder":

        self._intent = intent

        return self

    def build(
        self,
    ) -> Query:

        clauses = list(
            self._clauses
        )

        query = Query(
            original="",
            normalized="",
            clauses=clauses,
            filters=list(
                self._filters
            ),
            operators=list(
                self._operators
            ),
            sort=self._sort,
            pagination=self._pagination,
            intent=self._intent,
        )

        parser = QueryParser(
            self.config
        )

        query.metadata = (
            parser._build_metadata(
                query
            )
        )

        return query


# ============================================================
# QUERY UTILITIES
# ============================================================


def parse_query(
    query: str,
    config: Optional[QueryConfig] = None,
) -> Query:
    """
    Convenience function for parsing a query.
    """

    return QueryParser(
        config=config
    ).parse(query)


def build_query() -> QueryBuilder:
    """
    Convenience function for creating a QueryBuilder.
    """

    return QueryBuilder()


def query_terms(
    query: Query,
) -> List[str]:
    """
    Extract searchable terms.
    """

    return query.terms()


def query_phrases(
    query: Query,
) -> List[str]:
    """
    Extract searchable phrases.
    """

    return query.phrases()


def query_fields(
    query: Query,
) -> List[str]:
    """
    Extract referenced fields.
    """

    return query.fields()


# ============================================================
# QUERY SERIALIZATION
# ============================================================


def serialize_query(
    query: Query,
) -> Dict[str, Any]:
    """
    Convert a Query into a serializable dictionary.
    """

    return query.to_dict()


def deserialize_query(
    data: Dict[str, Any],
) -> Query:

    clauses = [
        QueryClause(
            **item
        )
        for item in data.get(
            "clauses",
            [],
        )
    ]

    filters = [
        QueryFilter(
            **item
        )
        for item in data.get(
            "filters",
            [],
        )
    ]

    sort_data = data.get(
        "sort",
        {},
    )

    sort = SortSpec(
        field=sort_data.get(
            "field",
            "relevance",
        ),
        direction=SortDirection(
            sort_data.get(
                "direction",
                "desc",
            )
        ),
    )

    pagination_data = data.get(
        "pagination",
        {},
    )

    pagination = Pagination(
        limit=pagination_data.get(
            "limit",
            20,
        ),
        offset=pagination_data.get(
            "offset",
            0,
        ),
    )

    intent_value = data.get(
        "intent",
        QueryIntent.UNKNOWN.value,
    )

    try:
        intent = QueryIntent(
            intent_value
        )
    except ValueError:
        intent = QueryIntent.UNKNOWN

    return Query(
        original=data.get(
            "original",
            "",
        ),
        normalized=data.get(
            "normalized",
            "",
        ),
        clauses=clauses,
        filters=filters,
        sort=sort,
        pagination=pagination,
        intent=intent,
        valid=data.get(
            "valid",
            True,
        ),
        errors=data.get(
            "errors",
            [],
        ),
    )


# ============================================================
# QUERY INSPECTION
# ============================================================


def explain_query(
    query: Query,
) -> Dict[str, Any]:
    """
    Produce a human-readable diagnostic representation.

    Useful for debugging the search pipeline.
    """

    return {
        "original": query.original,
        "normalized": query.normalized,
        "intent": query.intent.value,

        "terms": query.terms(),

        "phrases": query.phrases(),

        "fields": query.fields(),

        "operators": query.operators,

        "filters": [
            item.to_dict()
            for item in query.filters
        ],

        "sorting": query.sort.to_dict(),

        "pagination": query.pagination.to_dict(),

        "complexity": (
            query.metadata.complexity
        ),

        "valid": query.valid,

        "errors": query.errors,

        "warnings": (
            query.metadata.warnings
        ),
    }


# ============================================================
# DEFAULT ENGINE
# ============================================================


query_parser = QueryParser(
    DEFAULT_CONFIG
)


# ============================================================
# MODULE SELF-TEST
# ============================================================


if __name__ == "__main__":

    examples = [
        "python programming",
        '"machine learning"',
        "title:python category:technology",
        "python AND programming",
        "python NOT java",
        "python~",
        "python^2",
        "date>=2025",
        "category:science sort:date desc",
        "machine learning limit:50 offset:100",
        'title:"search engine" AND category:technology',
    ]

    for example in examples:

        print(
            "\n" + "=" * 70
        )

        print(
            "QUERY:",
            example,
        )

        parsed = parse_query(
            example
        )

        print(
            explain_query(
                parsed
            )
        )