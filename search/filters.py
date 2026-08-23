"""
search/filters.py

Advanced structured filtering engine.

This module is responsible for deciding whether a document
satisfies structured constraints.

It deliberately stays separate from retrieval and ranking.

Architecture
------------

    Query
      |
      v
    Retrieval
      |
      v
    FilterEngine
      |
      v
    Candidate Documents
      |
      v
    Ranking
      |
      v
    Results

Filtering answers:

    "Is this document allowed?"

Ranking answers:

    "How relevant is this document?"

Supported capabilities
----------------------

- Equality
- Inequality
- Greater-than / less-than comparisons
- Inclusive ranges
- Membership
- Contains
- Starts-with / ends-with
- Regex
- Existence checks
- Empty / non-empty checks
- Boolean AND / OR / NOT
- Nested field access
- List-valued fields
- Case-insensitive text matching
- Numeric coercion
- Date-like comparisons
- Custom predicates
- Filter explanations
- Filter statistics
- Batch evaluation
- Short-circuit evaluation
- Configurable strictness
"""

from __future__ import annotations

import re
import time

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ============================================================
# ENUMERATIONS
# ============================================================


class FilterOperator(str, Enum):
    """Supported comparison operators."""

    EQUALS = "="
    NOT_EQUALS = "!="

    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="

    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="

    IN = "in"
    NOT_IN = "not_in"

    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"

    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"

    MATCHES = "matches"

    EXISTS = "exists"
    NOT_EXISTS = "not_exists"

    EMPTY = "empty"
    NOT_EMPTY = "not_empty"

    BETWEEN = "between"
    NOT_BETWEEN = "not_between"

    IS_TRUE = "is_true"
    IS_FALSE = "is_false"


class FilterLogic(str, Enum):
    """Boolean combination modes."""

    AND = "and"
    OR = "or"
    NOT = "not"


class FilterValueType(str, Enum):
    """Optional explicit value typing."""

    AUTO = "auto"
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    LIST = "list"


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class FilterConfig:
    """
    Configuration controlling filter behavior.
    """

    case_sensitive: bool = False

    strict_missing_fields: bool = False

    strict_type_comparisons: bool = False

    allow_regex: bool = True

    allow_custom_predicates: bool = True

    coerce_numeric_strings: bool = True

    coerce_boolean_strings: bool = True

    parse_iso_dates: bool = True

    maximum_regex_length: int = 1000

    maximum_nested_depth: int = 32

    cache_field_resolution: bool = True

    short_circuit: bool = True


DEFAULT_FILTER_CONFIG = FilterConfig()


# ============================================================
# FILTER CONDITION
# ============================================================


@dataclass
class FilterCondition:
    """
    A single atomic filter.

    Example:

        FilterCondition(
            field="rating",
            operator=FilterOperator.GREATER_THAN,
            value=4
        )
    """

    field: str

    operator: FilterOperator | str

    value: Any = None

    value_type: FilterValueType | str = (
        FilterValueType.AUTO
    )

    case_sensitive: Optional[bool] = None

    name: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def normalized_operator(
        self,
    ) -> FilterOperator:

        if isinstance(
            self.operator,
            FilterOperator,
        ):
            return self.operator

        value = str(
            self.operator
        ).strip().lower()

        aliases = {
            "eq": "=",
            "==": "=",
            "equal": "=",
            "equals": "=",

            "ne": "!=",
            "<>": "!=",
            "not_equal": "!=",
            "not_equals": "!=",

            "gt": ">",
            "gte": ">=",
            "ge": ">=",

            "lt": "<",
            "lte": "<=",
            "le": "<=",

            "member_of": "in",

            "notin": "not_in",

            "has": "contains",

            "startswith": "starts_with",
            "endswith": "ends_with",

            "regex": "matches",

            "isnull": "not_exists",
            "is_not_null": "exists",

            "range": "between",
        }

        value = aliases.get(
            value,
            value,
        )

        return FilterOperator(
            value
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "field": self.field,
            "operator": self.normalized_operator().value,
            "value": self.value,
            "value_type": (
                self.value_type.value
                if isinstance(
                    self.value_type,
                    FilterValueType,
                )
                else self.value_type
            ),
            "case_sensitive": (
                self.case_sensitive
            ),
            "name": self.name,
            "metadata": self.metadata,
        }


# ============================================================
# FILTER GROUP
# ============================================================


@dataclass
class FilterGroup:
    """
    Boolean group containing conditions or nested groups.

    Examples:

        AND:
            rating >= 4
            category = "book"

        OR:
            author = "Alice"
            author = "Bob"

        NOT:
            category = "spam"
    """

    logic: FilterLogic | str = FilterLogic.AND

    conditions: List[
        FilterCondition
    ] = field(default_factory=list)

    groups: List[
        "FilterGroup"
    ] = field(default_factory=list)

    name: Optional[str] = None

    def normalized_logic(
        self,
    ) -> FilterLogic:

        if isinstance(
            self.logic,
            FilterLogic,
        ):
            return self.logic

        return FilterLogic(
            str(
                self.logic
            ).lower()
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "logic": (
                self.normalized_logic().value
            ),
            "conditions": [
                condition.to_dict()
                for condition in self.conditions
            ],
            "groups": [
                group.to_dict()
                for group in self.groups
            ],
            "name": self.name,
        }


# ============================================================
# EVALUATION RESULT
# ============================================================


@dataclass
class FilterEvaluation:
    """
    Detailed result of evaluating one document.
    """

    matched: bool

    document_id: Any = None

    conditions_evaluated: int = 0

    conditions_matched: int = 0

    conditions_failed: int = 0

    missing_fields: List[str] = field(
        default_factory=list
    )

    failed_conditions: List[str] = field(
        default_factory=list
    )

    matched_conditions: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    elapsed_ms: float = 0.0

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "matched": self.matched,
            "document_id": self.document_id,
            "conditions_evaluated": (
                self.conditions_evaluated
            ),
            "conditions_matched": (
                self.conditions_matched
            ),
            "conditions_failed": (
                self.conditions_failed
            ),
            "missing_fields": (
                self.missing_fields
            ),
            "failed_conditions": (
                self.failed_conditions
            ),
            "matched_conditions": (
                self.matched_conditions
            ),
            "errors": self.errors,
            "elapsed_ms": self.elapsed_ms,
        }


# ============================================================
# FILTER STATISTICS
# ============================================================


@dataclass
class FilterStats:
    """
    Aggregate statistics for filtering.
    """

    documents_seen: int = 0

    documents_matched: int = 0

    documents_rejected: int = 0

    conditions_evaluated: int = 0

    conditions_matched: int = 0

    conditions_failed: int = 0

    missing_fields: int = 0

    errors: int = 0

    elapsed_ms: float = 0.0

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "documents_seen": self.documents_seen,
            "documents_matched": self.documents_matched,
            "documents_rejected": self.documents_rejected,
            "conditions_evaluated": (
                self.conditions_evaluated
            ),
            "conditions_matched": (
                self.conditions_matched
            ),
            "conditions_failed": (
                self.conditions_failed
            ),
            "missing_fields": self.missing_fields,
            "errors": self.errors,
            "elapsed_ms": self.elapsed_ms,
        }


# ============================================================
# FILTER ENGINE
# ============================================================


class FilterEngine:
    """
    Main filtering engine.
    """

    def __init__(
        self,
        config: Optional[
            FilterConfig
        ] = None,
    ):

        self.config = (
            config
            or FilterConfig()
        )

        self._field_cache: Dict[
            Tuple[int, str],
            Tuple[bool, Any],
        ] = {}

        self.custom_operators: Dict[
            str,
            Callable[
                [Any, Any],
                bool,
            ],
        ] = {}

    # ========================================================
    # PUBLIC API
    # ========================================================

    def evaluate(
        self,
        document: Any,
        condition_or_group: (
            FilterCondition
            | FilterGroup
        ),
        document_id: Any = None,
    ) -> FilterEvaluation:

        start = time.perf_counter()

        evaluation = FilterEvaluation(
            matched=False,
            document_id=document_id,
        )

        try:

            if isinstance(
                condition_or_group,
                FilterCondition,
            ):

                matched = self._evaluate_condition(
                    document,
                    condition_or_group,
                    evaluation,
                )

            else:

                matched = self._evaluate_group(
                    document,
                    condition_or_group,
                    evaluation,
                )

            evaluation.matched = matched

        except Exception as error:

            evaluation.errors.append(
                str(error)
            )

            evaluation.matched = False

        finally:

            evaluation.elapsed_ms = (
                time.perf_counter()
                - start
            ) * 1000

        return evaluation

    def matches(
        self,
        document: Any,
        condition_or_group: (
            FilterCondition
            | FilterGroup
        ),
    ) -> bool:

        return self.evaluate(
            document,
            condition_or_group,
        ).matched

    def filter_documents(
        self,
        documents: Iterable[Any],
        condition_or_group: (
            FilterCondition
            | FilterGroup
        ),
    ) -> List[Any]:

        output = []

        for document in documents:

            if self.matches(
                document,
                condition_or_group,
            ):
                output.append(
                    document
                )

        return output

    def filter_with_stats(
        self,
        documents: Iterable[Any],
        condition_or_group: (
            FilterCondition
            | FilterGroup
        ),
    ) -> Tuple[
        List[Any],
        FilterStats,
    ]:

        start = time.perf_counter()

        results = []

        stats = FilterStats()

        for document in documents:

            stats.documents_seen += 1

            evaluation = self.evaluate(
                document,
                condition_or_group,
            )

            stats.conditions_evaluated += (
                evaluation.conditions_evaluated
            )

            stats.conditions_matched += (
                evaluation.conditions_matched
            )

            stats.conditions_failed += (
                evaluation.conditions_failed
            )

            stats.missing_fields += len(
                evaluation.missing_fields
            )

            stats.errors += len(
                evaluation.errors
            )

            if evaluation.matched:

                results.append(
                    document
                )

                stats.documents_matched += 1

            else:

                stats.documents_rejected += 1

        stats.elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return results, stats

    # ========================================================
    # GROUP EVALUATION
    # ========================================================

    def _evaluate_group(
        self,
        document: Any,
        group: FilterGroup,
        evaluation: FilterEvaluation,
    ) -> bool:

        logic = group.normalized_logic()

        if logic == FilterLogic.AND:

            return self._evaluate_and(
                document,
                group,
                evaluation,
            )

        if logic == FilterLogic.OR:

            return self._evaluate_or(
                document,
                group,
                evaluation,
            )

        if logic == FilterLogic.NOT:

            return not self._evaluate_not(
                document,
                group,
                evaluation,
            )

        return False

    def _evaluate_and(
        self,
        document: Any,
        group: FilterGroup,
        evaluation: FilterEvaluation,
    ) -> bool:

        items = (
            list(group.conditions)
            + list(group.groups)
        )

        if not items:
            return True

        for item in items:

            matched = (
                self._evaluate_item(
                    document,
                    item,
                    evaluation,
                )
            )

            if (
                not matched
                and self.config.short_circuit
            ):
                return False

        return all(
            self._item_result(
                document,
                item,
            )
            for item in items
        )

    def _evaluate_or(
        self,
        document: Any,
        group: FilterGroup,
        evaluation: FilterEvaluation,
    ) -> bool:

        items = (
            list(group.conditions)
            + list(group.groups)
        )

        if not items:
            return False

        for item in items:

            matched = (
                self._evaluate_item(
                    document,
                    item,
                    evaluation,
                )
            )

            if matched:
                if self.config.short_circuit:
                    return True

        return any(
            self._item_result(
                document,
                item,
            )
            for item in items
        )

    def _evaluate_not(
        self,
        document: Any,
        group: FilterGroup,
        evaluation: FilterEvaluation,
    ) -> bool:

        items = (
            list(group.conditions)
            + list(group.groups)
        )

        if not items:
            return False

        if len(items) == 1:

            return self._evaluate_item(
                document,
                items[0],
                evaluation,
            )

        nested = FilterGroup(
            logic=FilterLogic.OR,
            conditions=[
                item
                for item in items
                if isinstance(
                    item,
                    FilterCondition,
                )
            ],
            groups=[
                item
                for item in items
                if isinstance(
                    item,
                    FilterGroup,
                )
            ],
        )

        return self._evaluate_group(
            document,
            nested,
            evaluation,
        )

    def _evaluate_item(
        self,
        document: Any,
        item: Any,
        evaluation: FilterEvaluation,
    ) -> bool:

        if isinstance(
            item,
            FilterCondition,
        ):

            return self._evaluate_condition(
                document,
                item,
                evaluation,
            )

        if isinstance(
            item,
            FilterGroup,
        ):

            return self._evaluate_group(
                document,
                item,
                evaluation,
            )

        evaluation.errors.append(
            f"Unsupported filter item: "
            f"{type(item).__name__}"
        )

        return False

    def _item_result(
        self,
        document: Any,
        item: Any,
    ) -> bool:

        temporary = FilterEvaluation(
            matched=False
        )

        return self._evaluate_item(
            document,
            item,
            temporary,
        )

    # ========================================================
    # CONDITION EVALUATION
    # ========================================================

    def _evaluate_condition(
        self,
        document: Any,
        condition: FilterCondition,
        evaluation: FilterEvaluation,
    ) -> bool:

        evaluation.conditions_evaluated += 1

        field_name = condition.field

        exists, actual = self._resolve_field(
            document,
            field_name,
        )

        if not exists:

            evaluation.missing_fields.append(
                field_name
            )

            result = self._evaluate_missing(
                condition
            )

        else:

            expected = self._prepare_expected(
                condition.value,
                condition.value_type,
            )

            result = self._apply_operator(
                actual,
                condition,
                expected,
            )

        label = (
            condition.name
            or self._condition_label(
                condition
            )
        )

        if result:

            evaluation.conditions_matched += 1

            evaluation.matched_conditions.append(
                label
            )

        else:

            evaluation.conditions_failed += 1

            evaluation.failed_conditions.append(
                label
            )

        return result

    # ========================================================
    # MISSING VALUES
    # ========================================================

    def _evaluate_missing(
        self,
        condition: FilterCondition,
    ) -> bool:

        operator = (
            condition.normalized_operator()
        )

        if operator == FilterOperator.NOT_EXISTS:
            return True

        if operator == FilterOperator.EXISTS:
            return False

        if operator == FilterOperator.EMPTY:
            return True

        return False

    # ========================================================
    # OPERATOR DISPATCH
    # ========================================================

    def _apply_operator(
        self,
        actual: Any,
        condition: FilterCondition,
        expected: Any,
    ) -> bool:

        operator = (
            condition.normalized_operator()
        )

        case_sensitive = (
            condition.case_sensitive
            if condition.case_sensitive is not None
            else self.config.case_sensitive
        )

        if operator == FilterOperator.EQUALS:
            return self._equals(
                actual,
                expected,
                case_sensitive,
            )

        if operator == FilterOperator.NOT_EQUALS:
            return not self._equals(
                actual,
                expected,
                case_sensitive,
            )

        if operator == FilterOperator.GREATER_THAN:
            return self._compare(
                actual,
                expected,
                lambda a, b: a > b,
            )

        if operator == FilterOperator.GREATER_THAN_OR_EQUAL:
            return self._compare(
                actual,
                expected,
                lambda a, b: a >= b,
            )

        if operator == FilterOperator.LESS_THAN:
            return self._compare(
                actual,
                expected,
                lambda a, b: a < b,
            )

        if operator == FilterOperator.LESS_THAN_OR_EQUAL:
            return self._compare(
                actual,
                expected,
                lambda a, b: a <= b,
            )

        if operator == FilterOperator.IN:
            return self._in(
                actual,
                expected,
                case_sensitive,
            )

        if operator == FilterOperator.NOT_IN:
            return not self._in(
                actual,
                expected,
                case_sensitive,
            )

        if operator == FilterOperator.CONTAINS:
            return self._contains(
                actual,
                expected,
                case_sensitive,
            )

        if operator == FilterOperator.NOT_CONTAINS:
            return not self._contains(
                actual,
                expected,
                case_sensitive,
            )

        if operator == FilterOperator.STARTS_WITH:
            return self._starts_with(
                actual,
                expected,
                case_sensitive,
            )

        if operator == FilterOperator.ENDS_WITH:
            return self._ends_with(
                actual,
                expected,
                case_sensitive,
            )

        if operator == FilterOperator.MATCHES:

            if not self.config.allow_regex:
                return False

            return self._regex(
                actual,
                expected,
                case_sensitive,
            )

        if operator == FilterOperator.EXISTS:
            return True

        if operator == FilterOperator.NOT_EXISTS:
            return False

        if operator == FilterOperator.EMPTY:
            return self._empty(actual)

        if operator == FilterOperator.NOT_EMPTY:
            return not self._empty(actual)

        if operator == FilterOperator.BETWEEN:
            return self._between(
                actual,
                expected,
            )

        if operator == FilterOperator.NOT_BETWEEN:
            return not self._between(
                actual,
                expected,
            )

        if operator == FilterOperator.IS_TRUE:
            return actual is True

        if operator == FilterOperator.IS_FALSE:
            return actual is False

        custom = self.custom_operators.get(
            operator.value
        )

        if custom:

            if not self.config.allow_custom_predicates:
                return False

            try:
                return bool(
                    custom(
                        actual,
                        expected,
                    )
                )
            except Exception:
                return False

        return False

    # ========================================================
    # EQUALITY
    # ========================================================

    def _equals(
        self,
        actual: Any,
        expected: Any,
        case_sensitive: bool,
    ) -> bool:

        if isinstance(
            actual,
            (list, tuple, set),
        ):

            return any(
                self._equals(
                    item,
                    expected,
                    case_sensitive,
                )
                for item in actual
            )

        if isinstance(
            actual,
            str,
        ) and isinstance(
            expected,
            str,
        ):

            if case_sensitive:
                return actual == expected

            return (
                actual.casefold()
                == expected.casefold()
            )

        if self.config.coerce_numeric_strings:

            numeric_actual = self._numeric(
                actual
            )

            numeric_expected = self._numeric(
                expected
            )

            if (
                numeric_actual is not None
                and numeric_expected is not None
            ):

                return (
                    numeric_actual
                    == numeric_expected
                )

        return actual == expected

    # ========================================================
    # COMPARISON
    # ========================================================

    def _compare(
        self,
        actual: Any,
        expected: Any,
        operation: Callable[
            [Any, Any],
            bool,
        ],
    ) -> bool:

        actual = self._normalize_comparable(
            actual
        )

        expected = self._normalize_comparable(
            expected
        )

        try:

            return bool(
                operation(
                    actual,
                    expected,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            if self.config.strict_type_comparisons:
                return False

            return False

    # ========================================================
    # MEMBERSHIP
    # ========================================================

    def _in(
        self,
        actual: Any,
        expected: Any,
        case_sensitive: bool,
    ) -> bool:

        if expected is None:
            return False

        if isinstance(
            actual,
            (list, tuple, set),
        ):

            return any(
                self._in(
                    item,
                    expected,
                    case_sensitive,
                )
                for item in actual
            )

        if isinstance(
            expected,
            str,
        ):
            expected_values = [
                expected
            ]

        else:

            try:
                expected_values = list(
                    expected
                )
            except TypeError:
                expected_values = [
                    expected
                ]

        return any(
            self._equals(
                actual,
                item,
                case_sensitive,
            )
            for item in expected_values
        )

    # ========================================================
    # CONTAINS
    # ========================================================

    def _contains(
        self,
        actual: Any,
        expected: Any,
        case_sensitive: bool,
    ) -> bool:

        if actual is None:
            return False

        if isinstance(
            actual,
            (list, tuple, set),
        ):

            return any(
                self._equals(
                    item,
                    expected,
                    case_sensitive,
                )
                for item in actual
            )

        if isinstance(
            actual,
            Mapping,
        ):

            return expected in actual

        if isinstance(
            actual,
            str,
        ):

            left = actual
            right = str(
                expected
            )

            if not case_sensitive:

                left = left.casefold()
                right = right.casefold()

            return right in left

        try:
            return expected in actual
        except TypeError:
            return False

    # ========================================================
    # STRING PREFIX / SUFFIX
    # ========================================================

    def _starts_with(
        self,
        actual: Any,
        expected: Any,
        case_sensitive: bool,
    ) -> bool:

        if not isinstance(
            actual,
            str,
        ):
            return False

        actual = (
            actual
            if case_sensitive
            else actual.casefold()
        )

        expected = str(
            expected
        )

        expected = (
            expected
            if case_sensitive
            else expected.casefold()
        )

        return actual.startswith(
            expected
        )

    def _ends_with(
        self,
        actual: Any,
        expected: Any,
        case_sensitive: bool,
    ) -> bool:

        if not isinstance(
            actual,
            str,
        ):
            return False

        actual = (
            actual
            if case_sensitive
            else actual.casefold()
        )

        expected = str(
            expected
        )

        expected = (
            expected
            if case_sensitive
            else expected.casefold()
        )

        return actual.endswith(
            expected
        )

    # ========================================================
    # REGEX
    # ========================================================

    def _regex(
        self,
        actual: Any,
        expected: Any,
        case_sensitive: bool,
    ) -> bool:

        if not isinstance(
            actual,
            str,
        ):
            return False

        pattern = str(
            expected
        )

        if len(pattern) > (
            self.config.maximum_regex_length
        ):
            return False

        flags = 0

        if not case_sensitive:
            flags |= re.IGNORECASE

        try:

            return (
                re.search(
                    pattern,
                    actual,
                    flags,
                )
                is not None
            )

        except re.error:

            return False

    # ========================================================
    # EMPTY
    # ========================================================

    @staticmethod
    def _empty(
        value: Any,
    ) -> bool:

        if value is None:
            return True

        if isinstance(
            value,
            str,
        ):
            return not value.strip()

        try:
            return len(value) == 0
        except TypeError:
            return False

    # ========================================================
    # RANGE
    # ========================================================

    def _between(
        self,
        actual: Any,
        expected: Any,
    ) -> bool:

        if expected is None:
            return False

        try:

            values = list(
                expected
            )

        except TypeError:

            return False

        if len(values) != 2:
            return False

        lower, upper = values

        return (
            self._compare(
                actual,
                lower,
                lambda a, b: a >= b,
            )
            and self._compare(
                actual,
                upper,
                lambda a, b: a <= b,
            )
        )

    # ========================================================
    # FIELD RESOLUTION
    # ========================================================

    def _resolve_field(
        self,
        document: Any,
        field_name: str,
    ) -> Tuple[bool, Any]:

        if not field_name:
            return False, None

        cache_key = (
            id(document),
            field_name,
        )

        if self.config.cache_field_resolution:

            cached = self._field_cache.get(
                cache_key
            )

            if cached is not None:
                return cached

        parts = field_name.split(".")

        if len(parts) > (
            self.config.maximum_nested_depth
        ):

            return False, None

        current = document

        for part in parts:

            if isinstance(
                current,
                Mapping,
            ):

                if part not in current:

                    result = (
                        False,
                        None,
                    )

                    self._cache_field(
                        cache_key,
                        result,
                    )

                    return result

                current = current[
                    part
                ]

            else:

                if not hasattr(
                    current,
                    part,
                ):

                    result = (
                        False,
                        None,
                    )

                    self._cache_field(
                        cache_key,
                        result,
                    )

                    return result

                current = getattr(
                    current,
                    part,
                )

        result = (
            True,
            current,
        )

        self._cache_field(
            cache_key,
            result,
        )

        return result

    def _cache_field(
        self,
        key: Tuple[int, str],
        value: Tuple[bool, Any],
    ) -> None:

        if self.config.cache_field_resolution:

            self._field_cache[
                key
            ] = value

    def clear_cache(
        self,
    ) -> None:

        self._field_cache.clear()

    # ========================================================
    # VALUE NORMALIZATION
    # ========================================================

    def _prepare_expected(
        self,
        value: Any,
        value_type: (
            FilterValueType
            | str
        ),
    ) -> Any:

        if value_type == FilterValueType.AUTO:
            return value

        if isinstance(
            value_type,
            str,
        ):

            try:
                value_type = FilterValueType(
                    value_type
                )
            except ValueError:
                return value

        if value_type == FilterValueType.STRING:
            return str(value)

        if value_type == FilterValueType.INTEGER:

            try:
                return int(value)
            except (
                TypeError,
                ValueError,
            ):
                return value

        if value_type == FilterValueType.FLOAT:

            try:
                return float(value)
            except (
                TypeError,
                ValueError,
            ):
                return value

        if value_type == FilterValueType.BOOLEAN:
            return self._boolean(value)

        if value_type == FilterValueType.DATE:

            return self._date(value)

        if value_type == FilterValueType.DATETIME:

            return self._datetime(value)

        if value_type == FilterValueType.LIST:

            if isinstance(
                value,
                (list, tuple, set),
            ):
                return list(value)

            return [value]

        return value

    def _normalize_comparable(
        self,
        value: Any,
    ) -> Any:

        if isinstance(
            value,
            str,
        ):

            if self.config.coerce_numeric_strings:

                numeric = self._numeric(
                    value
                )

                if numeric is not None:
                    return numeric

            if self.config.parse_iso_dates:

                parsed_datetime = (
                    self._datetime(
                        value
                    )
                )

                if parsed_datetime is not None:
                    return parsed_datetime

        return value

    # ========================================================
    # TYPE CONVERSION
    # ========================================================

    @staticmethod
    def _numeric(
        value: Any,
    ) -> Optional[float]:

        if isinstance(
            value,
            bool,
        ):
            return None

        if isinstance(
            value,
            (int, float),
        ):
            return float(value)

        if not isinstance(
            value,
            str,
        ):
            return None

        value = value.strip()

        if not value:
            return None

        try:
            return float(value)
        except ValueError:
            return None

    def _boolean(
        self,
        value: Any,
    ) -> Any:

        if isinstance(
            value,
            bool,
        ):
            return value

        if not self.config.coerce_boolean_strings:
            return value

        if isinstance(
            value,
            str,
        ):

            normalized = (
                value.strip().lower()
            )

            if normalized in {
                "true",
                "1",
                "yes",
                "on",
            }:
                return True

            if normalized in {
                "false",
                "0",
                "no",
                "off",
            }:
                return False

        return value

    def _date(
        self,
        value: Any,
    ) -> Any:

        if isinstance(
            value,
            date,
        ) and not isinstance(
            value,
            datetime,
        ):
            return value

        if isinstance(
            value,
            str,
        ):

            try:
                return date.fromisoformat(
                    value
                )
            except ValueError:
                return value

        return value

    def _datetime(
        self,
        value: Any,
    ) -> Any:

        if isinstance(
            value,
            datetime,
        ):
            return value

        if isinstance(
            value,
            str,
        ):

            try:
                return datetime.fromisoformat(
                    value
                )
            except ValueError:
                return value

        return value

    # ========================================================
    # CUSTOM OPERATORS
    # ========================================================

    def register_operator(
        self,
        name: str,
        function: Callable[
            [Any, Any],
            bool,
        ],
    ) -> None:

        if not name:
            raise ValueError(
                "Operator name cannot be empty."
            )

        if not callable(function):
            raise TypeError(
                "Operator must be callable."
            )

        self.custom_operators[
            name
        ] = function

    def unregister_operator(
        self,
        name: str,
    ) -> bool:

        if name in self.custom_operators:

            del self.custom_operators[
                name
            ]

            return True

        return False

    # ========================================================
    # EXPLANATION
    # ========================================================

    def explain(
        self,
        document: Any,
        condition_or_group: (
            FilterCondition
            | FilterGroup
        ),
        document_id: Any = None,
    ) -> Dict[str, Any]:

        evaluation = self.evaluate(
            document,
            condition_or_group,
            document_id=document_id,
        )

        return evaluation.to_dict()

    # ========================================================
    # CONDITION LABELING
    # ========================================================

    @staticmethod
    def _condition_label(
        condition: FilterCondition,
    ) -> str:

        operator = (
            condition.normalized_operator().value
        )

        return (
            f"{condition.field} "
            f"{operator} "
            f"{condition.value!r}"
        )


# ============================================================
# FACTORY HELPERS
# ============================================================


def equals(
    field: str,
    value: Any,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=FilterOperator.EQUALS,
        value=value,
        **kwargs,
    )


def not_equals(
    field: str,
    value: Any,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=FilterOperator.NOT_EQUALS,
        value=value,
        **kwargs,
    )


def greater_than(
    field: str,
    value: Any,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=FilterOperator.GREATER_THAN,
        value=value,
        **kwargs,
    )


def greater_than_or_equal(
    field: str,
    value: Any,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=(
            FilterOperator.GREATER_THAN_OR_EQUAL
        ),
        value=value,
        **kwargs,
    )


def less_than(
    field: str,
    value: Any,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=FilterOperator.LESS_THAN,
        value=value,
        **kwargs,
    )


def less_than_or_equal(
    field: str,
    value: Any,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=(
            FilterOperator.LESS_THAN_OR_EQUAL
        ),
        value=value,
        **kwargs,
    )


def inside(
    field: str,
    values: Iterable[Any],
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=FilterOperator.IN,
        value=list(values),
        **kwargs,
    )


def contains(
    field: str,
    value: Any,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=FilterOperator.CONTAINS,
        value=value,
        **kwargs,
    )


def starts_with(
    field: str,
    value: str,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=FilterOperator.STARTS_WITH,
        value=value,
        **kwargs,
    )


def ends_with(
    field: str,
    value: str,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=FilterOperator.ENDS_WITH,
        value=value,
        **kwargs,
    )


def matches(
    field: str,
    pattern: str,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=FilterOperator.MATCHES,
        value=pattern,
        **kwargs,
    )


def exists(
    field: str,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=FilterOperator.EXISTS,
        **kwargs,
    )


def not_exists(
    field: str,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=FilterOperator.NOT_EXISTS,
        **kwargs,
    )


def empty(
    field: str,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=FilterOperator.EMPTY,
        **kwargs,
    )


def not_empty(
    field: str,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=FilterOperator.NOT_EMPTY,
        **kwargs,
    )


def between(
    field: str,
    minimum: Any,
    maximum: Any,
    **kwargs,
) -> FilterCondition:

    return FilterCondition(
        field=field,
        operator=FilterOperator.BETWEEN,
        value=[
            minimum,
            maximum,
        ],
        **kwargs,
    )


# ============================================================
# GROUP HELPERS
# ============================================================


def all_of(
    *items: FilterCondition | FilterGroup,
    name: Optional[str] = None,
) -> FilterGroup:

    conditions = [
        item
        for item in items
        if isinstance(
            item,
            FilterCondition,
        )
    ]

    groups = [
        item
        for item in items
        if isinstance(
            item,
            FilterGroup,
        )
    ]

    return FilterGroup(
        logic=FilterLogic.AND,
        conditions=conditions,
        groups=groups,
        name=name,
    )


def any_of(
    *items: FilterCondition | FilterGroup,
    name: Optional[str] = None,
) -> FilterGroup:

    conditions = [
        item
        for item in items
        if isinstance(
            item,
            FilterCondition,
        )
    ]

    groups = [
        item
        for item in items
        if isinstance(
            item,
            FilterGroup,
        )
    ]

    return FilterGroup(
        logic=FilterLogic.OR,
        conditions=conditions,
        groups=groups,
        name=name,
    )


def none_of(
    *items: FilterCondition | FilterGroup,
    name: Optional[str] = None,
) -> FilterGroup:

    return FilterGroup(
        logic=FilterLogic.NOT,
        conditions=[
            item
            for item in items
            if isinstance(
                item,
                FilterCondition,
            )
        ],
        groups=[
            item
            for item in items
            if isinstance(
                item,
                FilterGroup,
            )
        ],
        name=name,
    )


# ============================================================
# CONVENIENCE API
# ============================================================


def apply_filters(
    documents: Iterable[Any],
    condition_or_group: (
        FilterCondition
        | FilterGroup
    ),
    config: Optional[
        FilterConfig
    ] = None,
) -> List[Any]:

    engine = FilterEngine(
        config=config
    )

    return engine.filter_documents(
        documents,
        condition_or_group,
    )


def matches_filter(
    document: Any,
    condition_or_group: (
        FilterCondition
        | FilterGroup
    ),
    config: Optional[
        FilterConfig
    ] = None,
) -> bool:

    engine = FilterEngine(
        config=config
    )

    return engine.matches(
        document,
        condition_or_group,
    )


# ============================================================
# DEFAULT ENGINE
# ============================================================


filter_config = FilterConfig()

filter_engine = FilterEngine(
    filter_config
)


# ============================================================
# EXPORTS
# ============================================================


__all__ = [
    "FilterOperator",
    "FilterLogic",
    "FilterValueType",
    "FilterConfig",
    "FilterCondition",
    "FilterGroup",
    "FilterEvaluation",
    "FilterStats",
    "FilterEngine",
    "equals",
    "not_equals",
    "greater_than",
    "greater_than_or_equal",
    "less_than",
    "less_than_or_equal",
    "inside",
    "contains",
    "starts_with",
    "ends_with",
    "matches",
    "exists",
    "not_exists",
    "empty",
    "not_empty",
    "between",
    "all_of",
    "any_of",
    "none_of",
    "apply_filters",
    "matches_filter",
    "filter_config",
    "filter_engine",
]