"""
research/models.py

Research Engine - Core Domain Models
====================================

This module defines the shared domain model for the research engine.

IMPORTANT ARCHITECTURAL RULE
----------------------------
This file describes the objects that exist inside a research
investigation. It does not execute research.

Execution belongs to modules such as:

    planner.py
    task.py
    query_generator.py
    source_manager.py
    evidence.py
    evaluator.py
    coverage.py
    synthesis.py
    pipeline.py
    researcher.py

The goal is to give those modules a common, versioned language.

HIGH-LEVEL RESEARCH GRAPH
-------------------------

                    Research Question
                           |
                           v
                    Research Session
                           |
                           v
                    Research Objectives
                           |
                           v
                     Research Tasks
                           |
                           v
                     Search Queries
                           |
                           v
                    Search Results
                           |
                           v
                       Sources
                           |
                           v
                       Evidence
                           |
                           v
                        Claims
                       /     \\
                      v       v
                Hypotheses  Contradictions
                       \\     /
                        v   v
                     Evaluation
                          |
                          v
                    Coverage / Gaps
                          |
                    +-----+-----+
                    |           |
                  gaps       sufficient
                    |           |
                    v           v
                new tasks    Synthesis
                    |           |
                    +-----<-----+
                          |
                          v
                    Research Result


DESIGN GOALS
------------

1. Strong typing without excessive coupling.
2. Stable serialization.
3. Explicit provenance.
4. Explicit relationships.
5. Confidence represented numerically and qualitatively.
6. Support for iterative research.
7. Support for contradiction handling.
8. Support for research graphs.
9. Support for budgets and stopping conditions.
10. Support for partial/incomplete research.
11. Backwards-compatible extension through metadata.
12. No dependency on the search implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# =====================================================================
# MODULE VERSION
# =====================================================================

MODEL_VERSION = "2.0.0"


# =====================================================================
# TIME UTILITIES
# =====================================================================


def utc_now() -> str:
    """
    Return the current timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc).isoformat()


# =====================================================================
# ENUMERATIONS
# =====================================================================


class ResearchStatus(str, Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    PLANNING = "planning"
    READY = "ready"
    SEARCHING = "searching"
    COLLECTING = "collecting"
    ANALYZING = "analyzing"
    EVALUATING = "evaluating"
    VERIFYING = "verifying"
    SYNTHESIZING = "synthesizing"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    STALLED = "stalled"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    QUESTION_ANALYSIS = "question_analysis"
    OBJECTIVE_DECOMPOSITION = "objective_decomposition"
    SOURCE_DISCOVERY = "source_discovery"
    INFORMATION_GATHERING = "information_gathering"
    FACT_CHECK = "fact_check"
    VERIFICATION = "verification"
    CONTRADICTION_CHECK = "contradiction_check"
    COMPARISON = "comparison"
    CAUSAL_ANALYSIS = "causal_analysis"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    DEFINITION = "definition"
    CONTEXT = "context"
    GAP_ANALYSIS = "gap_analysis"
    FOLLOW_UP = "follow_up"
    SYNTHESIS = "synthesis"
    QUALITY_REVIEW = "quality_review"
    CUSTOM = "custom"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    OPTIONAL = "optional"


class ConfidenceLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SourceType(str, Enum):
    UNKNOWN = "unknown"
    WEB = "web"
    OFFICIAL = "official"
    GOVERNMENT = "government"
    ACADEMIC = "academic"
    PAPER = "paper"
    BOOK = "book"
    REPORT = "report"
    NEWS = "news"
    DATABASE = "database"
    DOCUMENT = "document"
    ARCHIVE = "archive"
    REPOSITORY = "repository"
    FORUM = "forum"
    OTHER = "other"


class SourceAuthority(str, Enum):
    UNKNOWN = "unknown"
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class EvidenceType(str, Enum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"
    CONTEXTUAL = "contextual"
    DIRECT = "direct"
    INDIRECT = "indirect"


class ClaimStatus(str, Enum):
    PROPOSED = "proposed"
    DEVELOPING = "developing"
    SUPPORTED = "supported"
    WEAKLY_SUPPORTED = "weakly_supported"
    CONTESTED = "contested"
    CONTRADICTED = "contradicted"
    VERIFIED = "verified"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


class QueryPurpose(str, Enum):
    DISCOVERY = "discovery"
    DIRECT_ANSWER = "direct_answer"
    CONTEXT = "context"
    VERIFICATION = "verification"
    CONTRADICTION_CHECK = "contradiction_check"
    GAP_FILLING = "gap_filling"
    FOLLOW_UP = "follow_up"
    SOURCE_DISCOVERY = "source_discovery"
    COMPARISON = "comparison"


class ResearchDepth(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    EXHAUSTIVE = "exhaustive"


class NodeType(str, Enum):
    SESSION = "session"
    OBJECTIVE = "objective"
    TASK = "task"
    QUERY = "query"
    SOURCE = "source"
    EVIDENCE = "evidence"
    CLAIM = "claim"
    HYPOTHESIS = "hypothesis"
    GAP = "gap"
    DECISION = "decision"
    RESULT = "result"


class EdgeType(str, Enum):
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    BLOCKS = "blocks"
    DERIVED_FROM = "derived_from"
    GENERATED_FROM = "generated_from"
    RETURNED_BY = "returned_by"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXTUALIZES = "contextualizes"
    RELATES_TO = "relates_to"
    TESTS = "tests"
    RESOLVES = "resolves"
    EXPANDS = "expands"
    REFINES = "refines"
    DUPLICATES = "duplicates"
    FOLLOWS = "follows"


class QueryExecutionStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContradictionType(str, Enum):
    DIRECT = "direct"
    PARTIAL = "partial"
    TEMPORAL = "temporal"
    CONTEXTUAL = "contextual"
    DEFINitional = "definitional"
    METHODOLOGICAL = "methodological"
    UNKNOWN = "unknown"


class GapType(str, Enum):
    MISSING_EVIDENCE = "missing_evidence"
    MISSING_SOURCE = "missing_source"
    WEAK_SUPPORT = "weak_support"
    CONFLICTING_INFORMATION = "conflicting_information"
    UNANSWERED_OBJECTIVE = "unanswered_objective"
    AMBIGUITY = "ambiguity"
    LOW_COVERAGE = "low_coverage"
    OUTDATED_INFORMATION = "outdated_information"
    UNKNOWN = "unknown"


class DecisionType(str, Enum):
    CONTINUE = "continue"
    SEARCH_AGAIN = "search_again"
    VERIFY = "verify"
    EXPAND = "expand"
    NARROW = "narrow"
    STOP = "stop"
    SYNTHESIZE = "synthesize"
    ESCALATE = "escalate"


class StoppingReason(str, Enum):
    NONE = "none"
    TARGET_REACHED = "target_reached"
    COVERAGE_REACHED = "coverage_reached"
    DIMINISHING_RETURNS = "diminishing_returns"
    NO_NEW_INFORMATION = "no_new_information"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_ITERATIONS = "max_iterations"
    USER_CANCELLED = "user_cancelled"
    FAILURE = "failure"
    UNRESOLVABLE = "unresolvable"


class RelationshipStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


# =====================================================================
# NUMERIC HELPERS
# =====================================================================


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """
    Clamp a numeric value to a range.
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return minimum

    return max(minimum, min(maximum, value))


def confidence_level(value: float) -> ConfidenceLevel:
    """
    Convert a [0, 1] confidence score into a qualitative level.
    """

    value = clamp(value)

    if value < 0.20:
        return ConfidenceLevel.VERY_LOW

    if value < 0.40:
        return ConfidenceLevel.LOW

    if value < 0.70:
        return ConfidenceLevel.MODERATE

    if value < 0.90:
        return ConfidenceLevel.HIGH

    return ConfidenceLevel.VERY_HIGH


def weighted_average(
    values: Sequence[Tuple[float, float]],
) -> float:
    """
    Calculate a weighted average.

    Each tuple is:

        (value, weight)
    """

    if not values:
        return 0.0

    total_weight = sum(
        max(0.0, float(weight))
        for _, weight in values
    )

    if total_weight <= 0:
        return 0.0

    return clamp(
        sum(
            clamp(value) * max(0.0, float(weight))
            for value, weight in values
        )
        / total_weight
    )


# =====================================================================
# SERIALIZATION
# =====================================================================


def _serialize(value: Any) -> Any:
    """
    Recursively serialize enums, dataclasses, collections,
    and mappings.
    """

    if isinstance(value, Enum):
        return value.value

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _serialize(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, Mapping):
        return {
            str(key): _serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _serialize(item)
            for item in value
        ]

    return value


def model_to_dict(model: Any) -> Dict[str, Any]:
    """
    Convert a research model into a JSON-compatible dictionary.
    """

    result = _serialize(model)

    if isinstance(result, dict):
        return result

    return {"value": result}


# =====================================================================
# BASE MODEL
# =====================================================================


@dataclass
class ResearchModel:
    """
    Base class for persistent research objects.
    """

    id: str

    created_at: str = field(
        default_factory=utc_now
    )

    updated_at: str = field(
        default_factory=utc_now
    )

    model_version: str = MODEL_VERSION

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    tags: Set[str] = field(
        default_factory=set
    )

    def touch(self) -> None:
        self.updated_at = utc_now()

    def add_tag(self, tag: str) -> None:
        tag = str(tag).strip()

        if tag:
            self.tags.add(tag)
            self.touch()

    def remove_tag(self, tag: str) -> None:
        self.tags.discard(tag)
        self.touch()

    def update_metadata(self, **values: Any) -> None:
        self.metadata.update(values)
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# PROVENANCE
# =====================================================================


@dataclass
class ProvenanceRecord:
    """
    Describes where a piece of research information came from.
    """

    source_id: Optional[str] = None

    evidence_id: Optional[str] = None

    query_id: Optional[str] = None

    task_id: Optional[str] = None

    parent_id: Optional[str] = None

    extraction_method: str = ""

    location: Optional[str] = None

    captured_at: str = field(
        default_factory=utc_now
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# GRAPH MODELS
# =====================================================================


@dataclass
class ResearchNode:
    """
    Generic node within the research knowledge graph.
    """

    id: str

    node_type: NodeType

    label: str = ""

    description: str = ""

    confidence: float = 0.0

    importance: float = 0.5

    status: str = ""

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        self.confidence = clamp(
            self.confidence
        )

        self.importance = clamp(
            self.importance
        )

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


@dataclass
class ResearchEdge:
    """
    Directed relationship between two research nodes.
    """

    id: str

    source_id: str

    target_id: str

    edge_type: EdgeType

    strength: float = 1.0

    confidence: float = 1.0

    bidirectional: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        self.strength = clamp(
            self.strength
        )

        self.confidence = clamp(
            self.confidence
        )

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


@dataclass
class ResearchGraph:
    """
    Explicit graph representation of a research investigation.
    """

    nodes: Dict[str, ResearchNode] = field(
        default_factory=dict
    )

    edges: Dict[str, ResearchEdge] = field(
        default_factory=dict
    )

    root_id: Optional[str] = None

    version: int = 1

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def add_node(self, node: ResearchNode) -> None:
        self.nodes[node.id] = node
        self.version += 1

    def add_edge(self, edge: ResearchEdge) -> None:
        if edge.source_id not in self.nodes:
            raise ValueError(
                f"Unknown source node: {edge.source_id}"
            )

        if edge.target_id not in self.nodes:
            raise ValueError(
                f"Unknown target node: {edge.target_id}"
            )

        self.edges[edge.id] = edge
        self.version += 1

    def get_node(
        self,
        node_id: str,
    ) -> Optional[ResearchNode]:
        return self.nodes.get(node_id)

    def outgoing(
        self,
        node_id: str,
    ) -> List[ResearchEdge]:
        return [
            edge
            for edge in self.edges.values()
            if edge.source_id == node_id
        ]

    def incoming(
        self,
        node_id: str,
    ) -> List[ResearchEdge]:
        return [
            edge
            for edge in self.edges.values()
            if edge.target_id == node_id
        ]

    def neighbors(
        self,
        node_id: str,
    ) -> List[str]:
        neighbors: Set[str] = set()

        for edge in self.edges.values():
            if edge.source_id == node_id:
                neighbors.add(edge.target_id)

            elif edge.target_id == node_id:
                neighbors.add(edge.source_id)

        return list(neighbors)

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# OBJECTIVES
# =====================================================================


@dataclass
class ResearchObjective(ResearchModel):
    """
    A measurable objective within an investigation.
    """

    title: str = ""

    description: str = ""

    priority: Priority = Priority.NORMAL

    status: TaskStatus = TaskStatus.PENDING

    required: bool = True

    parent_objective_id: Optional[str] = None

    dependency_ids: List[str] = field(
        default_factory=list
    )

    child_objective_ids: List[str] = field(
        default_factory=list
    )

    success_criteria: List[str] = field(
        default_factory=list
    )

    keywords: List[str] = field(
        default_factory=list
    )

    unanswered_questions: List[str] = field(
        default_factory=list
    )

    coverage: float = 0.0

    confidence: float = 0.0

    def __post_init__(self) -> None:
        self.coverage = clamp(
            self.coverage
        )

        self.confidence = clamp(
            self.confidence
        )

    @property
    def complete(self) -> bool:
        return (
            self.status == TaskStatus.COMPLETED
            and self.coverage >= 0.80
        )


# =====================================================================
# TASKS
# =====================================================================


@dataclass
class TaskDependency:
    """
    Dependency relationship between tasks.
    """

    task_id: str

    depends_on: str

    required: bool = True

    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


@dataclass
class ResearchTask(ResearchModel):
    """
    Executable research unit.

    The task model is intentionally independent of execution.
    """

    title: str = ""

    description: str = ""

    task_type: TaskType = TaskType.CUSTOM

    status: TaskStatus = TaskStatus.PENDING

    priority: Priority = Priority.NORMAL

    objective_id: Optional[str] = None

    parent_task_id: Optional[str] = None

    dependency_ids: List[str] = field(
        default_factory=list
    )

    child_task_ids: List[str] = field(
        default_factory=list
    )

    query_ids: List[str] = field(
        default_factory=list
    )

    source_ids: List[str] = field(
        default_factory=list
    )

    evidence_ids: List[str] = field(
        default_factory=list
    )

    claim_ids: List[str] = field(
        default_factory=list
    )

    attempt_count: int = 0

    max_attempts: int = 3

    progress: float = 0.0

    confidence: float = 0.0

    error: Optional[str] = None

    started_at: Optional[str] = None

    completed_at: Optional[str] = None

    def __post_init__(self) -> None:
        self.progress = clamp(
            self.progress
        )

        self.confidence = clamp(
            self.confidence
        )

        self.max_attempts = max(
            1,
            int(self.max_attempts),
        )

    def start(self) -> None:
        self.status = TaskStatus.RUNNING

        if self.started_at is None:
            self.started_at = utc_now()

        self.touch()

    def complete(self) -> None:
        self.status = TaskStatus.COMPLETED
        self.progress = 1.0
        self.completed_at = utc_now()
        self.touch()

    def fail(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error = error
        self.touch()

    def can_retry(self) -> bool:
        return (
            self.attempt_count
            < self.max_attempts
        )


# =====================================================================
# SEARCH PLAN
# =====================================================================


@dataclass
class SearchStrategy:
    """
    Strategy controlling how the research engine uses Search.
    """

    name: str = "adaptive"

    breadth: float = 0.5

    depth: float = 0.5

    freshness_weight: float = 0.5

    authority_weight: float = 0.5

    diversity_weight: float = 0.5

    verification_strength: float = 0.5

    allow_follow_up: bool = True

    allow_contradiction_search: bool = True

    allow_query_expansion: bool = True

    allow_query_refinement: bool = True

    max_queries_per_objective: int = 10

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.breadth = clamp(self.breadth)
        self.depth = clamp(self.depth)
        self.freshness_weight = clamp(
            self.freshness_weight
        )
        self.authority_weight = clamp(
            self.authority_weight
        )
        self.diversity_weight = clamp(
            self.diversity_weight
        )
        self.verification_strength = clamp(
            self.verification_strength
        )


@dataclass
class SearchPlan:
    """
    Collection of search strategy and planned queries.
    """

    strategy: SearchStrategy = field(
        default_factory=SearchStrategy
    )

    query_ids: List[str] = field(
        default_factory=list
    )

    objective_ids: List[str] = field(
        default_factory=list
    )

    generated_count: int = 0

    executed_count: int = 0

    successful_count: int = 0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def record_execution(
        self,
        successful: bool,
    ) -> None:

        self.executed_count += 1

        if successful:
            self.successful_count += 1


# =====================================================================
# RESEARCH QUERIES
# =====================================================================


@dataclass
class ResearchQuery(ResearchModel):
    """
    Search query generated for a research purpose.
    """

    text: str = ""

    purpose: QueryPurpose = QueryPurpose.DISCOVERY

    objective_id: Optional[str] = None

    task_id: Optional[str] = None

    parent_query_id: Optional[str] = None

    priority: Priority = Priority.NORMAL

    status: QueryExecutionStatus = (
        QueryExecutionStatus.CREATED
    )

    generated_reason: str = ""

    expected_information: str = ""

    executed: bool = False

    result_count: int = 0

    useful_result_count: int = 0

    usefulness: float = 0.0

    novelty: float = 0.0

    redundancy: float = 0.0

    duplicate: bool = False

    execution_time_ms: Optional[float] = None

    def __post_init__(self) -> None:
        self.usefulness = clamp(
            self.usefulness
        )

        self.novelty = clamp(
            self.novelty
        )

        self.redundancy = clamp(
            self.redundancy
        )

    @property
    def effectiveness(self) -> float:
        """
        Approximate query effectiveness.

        More sophisticated evaluation belongs in evaluator.py.
        """

        return clamp(
            (
                self.usefulness * 0.45
                + self.novelty * 0.35
                + (1.0 - self.redundancy) * 0.20
            )
        )


@dataclass
class QueryAttempt:
    """
    Individual execution attempt of a research query.
    """

    id: str

    query_id: str

    attempt_number: int = 1

    status: QueryExecutionStatus = (
        QueryExecutionStatus.CREATED
    )

    result_count: int = 0

    useful_result_count: int = 0

    started_at: Optional[str] = None

    completed_at: Optional[str] = None

    error: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# SOURCES
# =====================================================================


@dataclass
class SourceProvenance:
    """
    Detailed origin information for a source.
    """

    canonical_identifier: Optional[str] = None

    url: Optional[str] = None

    domain: Optional[str] = None

    publisher: Optional[str] = None

    author: Optional[str] = None

    publication_date: Optional[str] = None

    accessed_at: Optional[str] = None

    discovered_by_query_ids: List[str] = field(
        default_factory=list
    )

    discovered_by_task_ids: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


@dataclass
class SourceRecord(ResearchModel):
    """
    Full research source representation.
    """

    title: str = ""

    source_type: SourceType = SourceType.UNKNOWN

    authority: SourceAuthority = (
        SourceAuthority.UNKNOWN
    )

    provenance: SourceProvenance = field(
        default_factory=SourceProvenance
    )

    relevance: float = 0.0

    reliability: float = 0.0

    quality: float = 0.0

    freshness: float = 0.0

    diversity_value: float = 0.0

    duplicate_of: Optional[str] = None

    parent_source_id: Optional[str] = None

    related_source_ids: List[str] = field(
        default_factory=list
    )

    discovered_at: str = field(
        default_factory=utc_now
    )

    used: bool = False

    def __post_init__(self) -> None:
        self.relevance = clamp(self.relevance)
        self.reliability = clamp(self.reliability)
        self.quality = clamp(self.quality)
        self.freshness = clamp(self.freshness)
        self.diversity_value = clamp(
            self.diversity_value
        )

    @property
    def research_score(self) -> float:
        return weighted_average(
            [
                (self.relevance, 0.30),
                (self.reliability, 0.25),
                (self.quality, 0.25),
                (self.freshness, 0.10),
                (self.diversity_value, 0.10),
            ]
        )


@dataclass
class SourceRelationship:
    """
    Relationship between two sources.
    """

    source_id: str

    related_source_id: str

    relationship: EdgeType = EdgeType.RELATES_TO

    strength: float = 0.5

    confidence: float = 0.5

    reason: str = ""

    def __post_init__(self) -> None:
        self.strength = clamp(self.strength)
        self.confidence = clamp(
            self.confidence
        )

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# EVIDENCE
# =====================================================================


@dataclass
class EvidenceRecord(ResearchModel):
    """
    Atomic evidence item used to evaluate claims.
    """

    statement: str = ""

    evidence_type: EvidenceType = EvidenceType.NEUTRAL

    source_id: Optional[str] = None

    claim_id: Optional[str] = None

    task_id: Optional[str] = None

    objective_id: Optional[str] = None

    provenance: Optional[ProvenanceRecord] = None

    strength: float = 0.0

    relevance: float = 0.0

    directness: float = 0.0

    reliability: float = 0.0

    context: str = ""

    excerpt: str = ""

    location: Optional[str] = None

    independently_confirmed: bool = False

    confirmation_count: int = 0

    def __post_init__(self) -> None:
        self.strength = clamp(self.strength)
        self.relevance = clamp(self.relevance)
        self.directness = clamp(self.directness)
        self.reliability = clamp(
            self.reliability
        )

    @property
    def evidence_score(self) -> float:
        return weighted_average(
            [
                (self.strength, 0.30),
                (self.relevance, 0.25),
                (self.directness, 0.20),
                (self.reliability, 0.25),
            ]
        )


# =====================================================================
# CLAIMS
# =====================================================================


@dataclass
class ClaimRecord(ResearchModel):
    """
    Structured research claim.

    A claim is deliberately separated from evidence. A claim can
    have multiple supporting and contradicting evidence items.
    """

    statement: str = ""

    status: ClaimStatus = ClaimStatus.PROPOSED

    objective_id: Optional[str] = None

    parent_claim_id: Optional[str] = None

    supporting_evidence_ids: List[str] = field(
        default_factory=list
    )

    contradicting_evidence_ids: List[str] = field(
        default_factory=list
    )

    source_ids: List[str] = field(
        default_factory=list
    )

    provenance: List[ProvenanceRecord] = field(
        default_factory=list
    )

    confidence: float = 0.0

    importance: float = 0.5

    specificity: float = 0.5

    contested: bool = False

    verified: bool = False

    verification_count: int = 0

    alternative_formulations: List[str] = field(
        default_factory=list
    )

    related_claim_ids: List[str] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self.confidence = clamp(
            self.confidence
        )

        self.importance = clamp(
            self.importance
        )

        self.specificity = clamp(
            self.specificity
        )

    @property
    def support_count(self) -> int:
        return len(
            self.supporting_evidence_ids
        )

    @property
    def contradiction_count(self) -> int:
        return len(
            self.contradicting_evidence_ids
        )

    @property
    def confidence_level(self) -> ConfidenceLevel:
        return confidence_level(
            self.confidence
        )


# =====================================================================
# HYPOTHESES
# =====================================================================


@dataclass
class Hypothesis:
    """
    Candidate explanation or interpretation.

    Useful for investigations where several explanations compete.
    """

    id: str

    statement: str

    objective_id: Optional[str] = None

    supporting_claim_ids: List[str] = field(
        default_factory=list
    )

    contradicting_claim_ids: List[str] = field(
        default_factory=list
    )

    confidence: float = 0.0

    probability: float = 0.0

    status: ClaimStatus = ClaimStatus.PROPOSED

    assumptions: List[str] = field(
        default_factory=list
    )

    predictions: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.confidence = clamp(
            self.confidence
        )

        self.probability = clamp(
            self.probability
        )

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# CONTRADICTIONS
# =====================================================================


@dataclass
class Contradiction:
    """
    Represents conflicting claims or evidence.
    """

    id: str

    left_claim_id: Optional[str] = None

    right_claim_id: Optional[str] = None

    left_evidence_id: Optional[str] = None

    right_evidence_id: Optional[str] = None

    contradiction_type: ContradictionType = (
        ContradictionType.UNKNOWN
    )

    severity: float = 0.5

    confidence: float = 0.5

    resolved: bool = False

    resolution: str = ""

    resolution_evidence_ids: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.severity = clamp(
            self.severity
        )

        self.confidence = clamp(
            self.confidence
        )

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# KNOWLEDGE GAPS
# =====================================================================


@dataclass
class KnowledgeGap:
    """
    Represents something the investigation does not sufficiently know.
    """

    id: str

    description: str

    gap_type: GapType = GapType.UNKNOWN

    objective_id: Optional[str] = None

    task_id: Optional[str] = None

    importance: float = 0.5

    severity: float = 0.5

    coverage: float = 0.0

    confidence: float = 0.0

    suggested_queries: List[str] = field(
        default_factory=list
    )

    suggested_tasks: List[str] = field(
        default_factory=list
    )

    resolved: bool = False

    resolution: str = ""

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.importance = clamp(
            self.importance
        )

        self.severity = clamp(
            self.severity
        )

        self.coverage = clamp(
            self.coverage
        )

        self.confidence = clamp(
            self.confidence
        )

    @property
    def priority_score(self) -> float:
        return weighted_average(
            [
                (self.importance, 0.45),
                (self.severity, 0.35),
                (1.0 - self.coverage, 0.20),
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# COVERAGE
# =====================================================================


@dataclass
class CoverageRecord:
    """
    Measures research coverage for an objective.
    """

    objective_id: str

    coverage: float = 0.0

    evidence_coverage: float = 0.0

    source_coverage: float = 0.0

    claim_coverage: float = 0.0

    confidence: float = 0.0

    unresolved_gap_ids: List[str] = field(
        default_factory=list
    )

    required: bool = True

    def __post_init__(self) -> None:
        self.coverage = clamp(self.coverage)
        self.evidence_coverage = clamp(
            self.evidence_coverage
        )
        self.source_coverage = clamp(
            self.source_coverage
        )
        self.claim_coverage = clamp(
            self.claim_coverage
        )
        self.confidence = clamp(
            self.confidence
        )

    def calculate(self) -> float:
        self.coverage = weighted_average(
            [
                (self.evidence_coverage, 0.35),
                (self.source_coverage, 0.20),
                (self.claim_coverage, 0.45),
            ]
        )

        return self.coverage

    @property
    def sufficiently_covered(self) -> bool:
        return (
            self.coverage >= 0.80
            and self.confidence >= 0.70
            and not self.unresolved_gap_ids
        )

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# RESEARCH DECISIONS
# =====================================================================


@dataclass
class ResearchDecision:
    """
    Records an important decision made by the research pipeline.
    """

    id: str

    decision_type: DecisionType

    reason: str = ""

    confidence: float = 0.0

    related_task_ids: List[str] = field(
        default_factory=list
    )

    related_query_ids: List[str] = field(
        default_factory=list
    )

    related_gap_ids: List[str] = field(
        default_factory=list
    )

    created_at: str = field(
        default_factory=utc_now
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.confidence = clamp(
            self.confidence
        )

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# STOPPING CRITERIA
# =====================================================================


@dataclass
class StoppingCriteria:
    """
    Defines when a research investigation should stop.
    """

    target_confidence: float = 0.85

    target_coverage: float = 0.90

    max_iterations: int = 20

    max_tasks: int = 100

    max_queries: int = 500

    max_sources: int = 1000

    diminishing_returns_threshold: float = 0.05

    no_new_information_limit: int = 3

    stop_on_target_confidence: bool = True

    stop_on_target_coverage: bool = True

    stop_on_budget: bool = True

    stop_on_diminishing_returns: bool = True

    stop_on_no_new_information: bool = True

    def __post_init__(self) -> None:
        self.target_confidence = clamp(
            self.target_confidence
        )

        self.target_coverage = clamp(
            self.target_coverage
        )

        self.max_iterations = max(
            1,
            int(self.max_iterations),
        )

        self.max_tasks = max(
            1,
            int(self.max_tasks),
        )

        self.max_queries = max(
            1,
            int(self.max_queries),
        )

        self.max_sources = max(
            1,
            int(self.max_sources),
        )

        self.diminishing_returns_threshold = clamp(
            self.diminishing_returns_threshold
        )

        self.no_new_information_limit = max(
            1,
            int(self.no_new_information_limit),
        )

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# RESEARCH BUDGET
# =====================================================================


@dataclass
class ResearchBudget:
    """
    Resource limits for an investigation.

    These are logical budgets rather than financial requirements.
    """

    max_queries: int = 500

    max_sources: int = 1000

    max_tasks: int = 100

    max_iterations: int = 20

    max_evidence_items: int = 5000

    max_claims: int = 2000

    max_execution_time_seconds: Optional[float] = None

    queries_used: int = 0

    sources_used: int = 0

    tasks_used: int = 0

    iterations_used: int = 0

    evidence_used: int = 0

    claims_used: int = 0

    execution_time_seconds: float = 0.0

    def exhausted(self) -> bool:
        return (
            self.queries_used >= self.max_queries
            or self.sources_used >= self.max_sources
            or self.tasks_used >= self.max_tasks
            or self.iterations_used >= self.max_iterations
            or self.evidence_used
            >= self.max_evidence_items
            or self.claims_used >= self.max_claims
            or (
                self.max_execution_time_seconds
                is not None
                and self.execution_time_seconds
                >= self.max_execution_time_seconds
            )
        )

    def consume_query(self) -> bool:
        if self.queries_used >= self.max_queries:
            return False

        self.queries_used += 1
        return True

    def consume_source(self) -> bool:
        if self.sources_used >= self.max_sources:
            return False

        self.sources_used += 1
        return True

    def consume_task(self) -> bool:
        if self.tasks_used >= self.max_tasks:
            return False

        self.tasks_used += 1
        return True

    def consume_iteration(self) -> bool:
        if self.iterations_used >= self.max_iterations:
            return False

        self.iterations_used += 1
        return True

    def consume_evidence(self) -> bool:
        if self.evidence_used >= self.max_evidence_items:
            return False

        self.evidence_used += 1
        return True

    def consume_claim(self) -> bool:
        if self.claims_used >= self.max_claims:
            return False

        self.claims_used += 1
        return True

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# RESEARCH ITERATION
# =====================================================================


@dataclass
class ResearchIteration:
    """
    One complete cycle of the research loop.
    """

    number: int

    started_at: str = field(
        default_factory=utc_now
    )

    completed_at: Optional[str] = None

    task_ids: List[str] = field(
        default_factory=list
    )

    query_ids: List[str] = field(
        default_factory=list
    )

    source_ids: List[str] = field(
        default_factory=list
    )

    evidence_ids: List[str] = field(
        default_factory=list
    )

    claim_ids: List[str] = field(
        default_factory=list
    )

    gap_ids: List[str] = field(
        default_factory=list
    )

    confidence_before: float = 0.0

    confidence_after: float = 0.0

    coverage_before: float = 0.0

    coverage_after: float = 0.0

    information_gain: float = 0.0

    decisions: List[str] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self.confidence_before = clamp(
            self.confidence_before
        )

        self.confidence_after = clamp(
            self.confidence_after
        )

        self.coverage_before = clamp(
            self.coverage_before
        )

        self.coverage_after = clamp(
            self.coverage_after
        )

        self.information_gain = clamp(
            self.information_gain
        )

    def complete(self) -> None:
        self.completed_at = utc_now()

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# RESEARCH METRICS
# =====================================================================


@dataclass
class ResearchMetrics:
    """
    Aggregate measurements for a research investigation.
    """

    tasks_created: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0

    queries_generated: int = 0
    queries_executed: int = 0
    useful_queries: int = 0

    sources_discovered: int = 0
    sources_used: int = 0
    duplicate_sources: int = 0

    evidence_collected: int = 0

    claims_created: int = 0
    claims_supported: int = 0
    claims_contradicted: int = 0
    claims_verified: int = 0

    hypotheses_created: int = 0

    contradictions_detected: int = 0
    contradictions_resolved: int = 0

    gaps_detected: int = 0
    gaps_resolved: int = 0

    iterations: int = 0

    average_source_quality: float = 0.0
    average_evidence_strength: float = 0.0
    average_claim_confidence: float = 0.0

    overall_confidence: float = 0.0
    overall_coverage: float = 0.0

    information_gain: float = 0.0

    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def start(self) -> None:
        if self.started_at is None:
            self.started_at = utc_now()

    def complete(self) -> None:
        self.completed_at = utc_now()

    def calculate_claim_confidence(
        self,
        claims: Iterable[ClaimRecord],
    ) -> float:

        values = [
            claim.confidence
            for claim in claims
        ]

        if not values:
            self.average_claim_confidence = 0.0
            return 0.0

        self.average_claim_confidence = (
            sum(values) / len(values)
        )

        self.overall_confidence = clamp(
            self.average_claim_confidence
        )

        return self.overall_confidence

    def calculate_coverage(
        self,
        records: Iterable[CoverageRecord],
    ) -> float:

        values = [
            record.coverage
            for record in records
            if record.required
        ]

        if not values:
            self.overall_coverage = 0.0
            return 0.0

        self.overall_coverage = clamp(
            sum(values) / len(values)
        )

        return self.overall_coverage

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# RESEARCH CONFIGURATION
# =====================================================================


@dataclass
class ResearchConfig:
    """
    Global configuration for an investigation.
    """

    depth: ResearchDepth = ResearchDepth.STANDARD

    minimum_confidence: float = 0.55

    target_confidence: float = 0.85

    target_coverage: float = 0.90

    enable_iterative_research: bool = True

    enable_follow_up_searches: bool = True

    enable_query_expansion: bool = True

    enable_query_refinement: bool = True

    enable_contradiction_detection: bool = True

    enable_source_diversity: bool = True

    enable_gap_analysis: bool = True

    enable_hypothesis_tracking: bool = True

    enable_provenance: bool = True

    enable_adaptive_stopping: bool = True

    enable_research_graph: bool = True

    allow_partial_results: bool = True

    require_evidence_for_claims: bool = True

    require_multiple_sources_for_high_confidence: bool = False

    minimum_sources_for_high_confidence: int = 2

    budget: ResearchBudget = field(
        default_factory=ResearchBudget
    )

    stopping: StoppingCriteria = field(
        default_factory=StoppingCriteria
    )

    search_strategy: SearchStrategy = field(
        default_factory=SearchStrategy
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.minimum_confidence = clamp(
            self.minimum_confidence
        )

        self.target_confidence = clamp(
            self.target_confidence
        )

        self.target_coverage = clamp(
            self.target_coverage
        )

        self.minimum_sources_for_high_confidence = max(
            1,
            int(
                self.minimum_sources_for_high_confidence
            ),
        )


# =====================================================================
# RESEARCH STATE
# =====================================================================


@dataclass
class ResearchState:
    """
    Mutable runtime state for the pipeline.
    """

    session_id: str

    status: ResearchStatus = ResearchStatus.CREATED

    current_task_id: Optional[str] = None

    current_objective_id: Optional[str] = None

    current_iteration: int = 0

    completed_task_ids: List[str] = field(
        default_factory=list
    )

    pending_task_ids: List[str] = field(
        default_factory=list
    )

    blocked_task_ids: List[str] = field(
        default_factory=list
    )

    failed_task_ids: List[str] = field(
        default_factory=list
    )

    active_query_ids: List[str] = field(
        default_factory=list
    )

    completed_query_ids: List[str] = field(
        default_factory=list
    )

    source_ids: List[str] = field(
        default_factory=list
    )

    evidence_ids: List[str] = field(
        default_factory=list
    )

    claim_ids: List[str] = field(
        default_factory=list
    )

    gap_ids: List[str] = field(
        default_factory=list
    )

    contradiction_ids: List[str] = field(
        default_factory=list
    )

    decision_ids: List[str] = field(
        default_factory=list
    )

    stopping_reason: StoppingReason = (
        StoppingReason.NONE
    )

    started_at: Optional[str] = None

    updated_at: str = field(
        default_factory=utc_now
    )

    def touch(self) -> None:
        self.updated_at = utc_now()

    def begin(self) -> None:
        self.status = ResearchStatus.PLANNING

        if self.started_at is None:
            self.started_at = utc_now()

        self.touch()

    def next_iteration(self) -> int:
        self.current_iteration += 1
        self.touch()
        return self.current_iteration

    def mark_task_complete(
        self,
        task_id: str,
    ) -> None:

        if task_id not in self.completed_task_ids:
            self.completed_task_ids.append(
                task_id
            )

        if task_id in self.pending_task_ids:
            self.pending_task_ids.remove(
                task_id
            )

        if task_id in self.blocked_task_ids:
            self.blocked_task_ids.remove(
                task_id
            )

        if task_id == self.current_task_id:
            self.current_task_id = None

        self.touch()

    def mark_task_failed(
        self,
        task_id: str,
    ) -> None:

        if task_id not in self.failed_task_ids:
            self.failed_task_ids.append(
                task_id
            )

        if task_id in self.pending_task_ids:
            self.pending_task_ids.remove(
                task_id
            )

        self.touch()


# =====================================================================
# RESEARCH SESSION
# =====================================================================


@dataclass
class ResearchSession(ResearchModel):
    """
    Complete persistent investigation container.

    This is the central object tying the entire research system together.
    """

    question: str = ""

    title: str = ""

    status: ResearchStatus = ResearchStatus.CREATED

    depth: ResearchDepth = ResearchDepth.STANDARD

    config: ResearchConfig = field(
        default_factory=ResearchConfig
    )

    state: Optional[ResearchState] = None

    graph: ResearchGraph = field(
        default_factory=ResearchGraph
    )

    objectives: Dict[str, ResearchObjective] = field(
        default_factory=dict
    )

    tasks: Dict[str, ResearchTask] = field(
        default_factory=dict
    )

    queries: Dict[str, ResearchQuery] = field(
        default_factory=dict
    )

    query_attempts: Dict[str, QueryAttempt] = field(
        default_factory=dict
    )

    sources: Dict[str, SourceRecord] = field(
        default_factory=dict
    )

    evidence: Dict[str, EvidenceRecord] = field(
        default_factory=dict
    )

    claims: Dict[str, ClaimRecord] = field(
        default_factory=dict
    )

    hypotheses: Dict[str, Hypothesis] = field(
        default_factory=dict
    )

    contradictions: Dict[str, Contradiction] = field(
        default_factory=dict
    )

    gaps: Dict[str, KnowledgeGap] = field(
        default_factory=dict
    )

    coverage: Dict[str, CoverageRecord] = field(
        default_factory=dict
    )

    decisions: Dict[str, ResearchDecision] = field(
        default_factory=dict
    )

    iterations: List[ResearchIteration] = field(
        default_factory=list
    )

    metrics: ResearchMetrics = field(
        default_factory=ResearchMetrics
    )

    final_result_id: Optional[str] = None

    stopping_reason: StoppingReason = (
        StoppingReason.NONE
    )

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = ResearchState(
                session_id=self.id
            )

    # --------------------------------------------------------------
    # OBJECTIVE MANAGEMENT
    # --------------------------------------------------------------

    def add_objective(
        self,
        objective: ResearchObjective,
    ) -> None:

        self.objectives[objective.id] = objective

        self.graph.add_node(
            ResearchNode(
                id=objective.id,
                node_type=NodeType.OBJECTIVE,
                label=objective.title,
                description=objective.description,
                confidence=objective.confidence,
                importance=(
                    1.0
                    if objective.required
                    else 0.5
                ),
            )
        )

        self.touch()

    # --------------------------------------------------------------
    # TASK MANAGEMENT
    # --------------------------------------------------------------

    def add_task(
        self,
        task: ResearchTask,
    ) -> None:

        self.tasks[task.id] = task

        self.graph.add_node(
            ResearchNode(
                id=task.id,
                node_type=NodeType.TASK,
                label=task.title,
                description=task.description,
                confidence=task.confidence,
            )
        )

        if task.objective_id:
            if task.objective_id in self.graph.nodes:
                edge_id = (
                    f"{task.objective_id}:contains:{task.id}"
                )

                self.graph.add_edge(
                    ResearchEdge(
                        id=edge_id,
                        source_id=task.objective_id,
                        target_id=task.id,
                        edge_type=EdgeType.CONTAINS,
                    )
                )

        self.metrics.tasks_created += 1
        self.touch()

    # --------------------------------------------------------------
    # QUERY MANAGEMENT
    # --------------------------------------------------------------

    def add_query(
        self,
        query: ResearchQuery,
    ) -> None:

        self.queries[query.id] = query

        self.graph.add_node(
            ResearchNode(
                id=query.id,
                node_type=NodeType.QUERY,
                label=query.text,
                description=query.generated_reason,
            )
        )

        if query.task_id:
            edge_id = (
                f"{query.task_id}:query:{query.id}"
            )

            if query.task_id in self.graph.nodes:
                self.graph.add_edge(
                    ResearchEdge(
                        id=edge_id,
                        source_id=query.task_id,
                        target_id=query.id,
                        edge_type=EdgeType.GENERATED_FROM,
                    )
                )

        self.metrics.queries_generated += 1
        self.touch()

    # --------------------------------------------------------------
    # SOURCE MANAGEMENT
    # --------------------------------------------------------------

    def add_source(
        self,
        source: SourceRecord,
    ) -> None:

        self.sources[source.id] = source

        self.graph.add_node(
            ResearchNode(
                id=source.id,
                node_type=NodeType.SOURCE,
                label=source.title,
                description=(
                    source.provenance.url
                    or ""
                ),
                confidence=source.reliability,
                importance=source.relevance,
            )
        )

        self.metrics.sources_discovered += 1
        self.touch()

    # --------------------------------------------------------------
    # EVIDENCE MANAGEMENT
    # --------------------------------------------------------------

    def add_evidence(
        self,
        item: EvidenceRecord,
    ) -> None:

        self.evidence[item.id] = item

        self.graph.add_node(
            ResearchNode(
                id=item.id,
                node_type=NodeType.EVIDENCE,
                label=item.statement[:120],
                confidence=item.evidence_score,
            )
        )

        if item.source_id:
            if item.source_id in self.graph.nodes:
                self.graph.add_edge(
                    ResearchEdge(
                        id=(
                            f"{item.source_id}:evidence:"
                            f"{item.id}"
                        ),
                        source_id=item.source_id,
                        target_id=item.id,
                        edge_type=EdgeType.DERIVED_FROM,
                        strength=item.evidence_score,
                    )
                )

        self.metrics.evidence_collected += 1
        self.touch()

    # --------------------------------------------------------------
    # CLAIM MANAGEMENT
    # --------------------------------------------------------------

    def add_claim(
        self,
        claim: ClaimRecord,
    ) -> None:

        self.claims[claim.id] = claim

        self.graph.add_node(
            ResearchNode(
                id=claim.id,
                node_type=NodeType.CLAIM,
                label=claim.statement[:120],
                confidence=claim.confidence,
                importance=claim.importance,
            )
        )

        for evidence_id in (
            claim.supporting_evidence_ids
        ):
            if evidence_id in self.graph.nodes:
                self.graph.add_edge(
                    ResearchEdge(
                        id=(
                            f"{evidence_id}:supports:"
                            f"{claim.id}"
                        ),
                        source_id=evidence_id,
                        target_id=claim.id,
                        edge_type=EdgeType.SUPPORTS,
                        strength=claim.confidence,
                    )
                )

        self.metrics.claims_created += 1

        if claim.status in {
            ClaimStatus.SUPPORTED,
            ClaimStatus.VERIFIED,
        }:
            self.metrics.claims_supported += 1

        if claim.status == ClaimStatus.CONTRADICTED:
            self.metrics.claims_contradicted += 1

        if claim.verified:
            self.metrics.claims_verified += 1

        self.touch()

    # --------------------------------------------------------------
    # GAP MANAGEMENT
    # --------------------------------------------------------------

    def add_gap(
        self,
        gap: KnowledgeGap,
    ) -> None:

        self.gaps[gap.id] = gap

        self.graph.add_node(
            ResearchNode(
                id=gap.id,
                node_type=NodeType.GAP,
                label=gap.description[:120],
                confidence=gap.confidence,
                importance=gap.importance,
            )
        )

        self.metrics.gaps_detected += 1
        self.touch()

    # --------------------------------------------------------------
    # CONTRADICTION MANAGEMENT
    # --------------------------------------------------------------

    def add_contradiction(
        self,
        contradiction: Contradiction,
    ) -> None:

        self.contradictions[
            contradiction.id
        ] = contradiction

        self.metrics.contradictions_detected += 1

        if contradiction.resolved:
            self.metrics.contradictions_resolved += 1

        self.touch()

    # --------------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------------

    def update_coverage(
        self,
        record: CoverageRecord,
    ) -> None:

        record.calculate()

        self.coverage[
            record.objective_id
        ] = record

        objective = self.objectives.get(
            record.objective_id
        )

        if objective:
            objective.coverage = record.coverage
            objective.confidence = record.confidence
            objective.touch()

        self.recalculate_overall_metrics()
        self.touch()

    # --------------------------------------------------------------
    # METRICS
    # --------------------------------------------------------------

    def recalculate_overall_metrics(self) -> None:

        if self.claims:
            self.metrics.calculate_claim_confidence(
                self.claims.values()
            )

        if self.coverage:
            self.metrics.calculate_coverage(
                self.coverage.values()
            )

    # --------------------------------------------------------------
    # STATE
    # --------------------------------------------------------------

    def start(self) -> None:

        self.status = ResearchStatus.PLANNING

        if self.state:
            self.state.begin()

        self.metrics.start()
        self.touch()

    def pause(self) -> None:

        self.status = ResearchStatus.PAUSED

        if self.state:
            self.state.status = (
                ResearchStatus.PAUSED
            )
            self.state.touch()

        self.touch()

    def complete(
        self,
        stopping_reason: StoppingReason = (
            StoppingReason.TARGET_REACHED
        ),
    ) -> None:

        self.status = ResearchStatus.COMPLETED

        self.stopping_reason = stopping_reason

        if self.state:
            self.state.status = (
                ResearchStatus.COMPLETED
            )
            self.state.stopping_reason = (
                stopping_reason
            )
            self.state.touch()

        self.metrics.complete()
        self.touch()

    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------

    @property
    def overall_confidence(self) -> float:
        return self.metrics.overall_confidence

    @property
    def overall_coverage(self) -> float:
        return self.metrics.overall_coverage

    @property
    def unresolved_gaps(self) -> List[KnowledgeGap]:
        return [
            gap
            for gap in self.gaps.values()
            if not gap.resolved
        ]

    @property
    def unresolved_contradictions(
        self,
    ) -> List[Contradiction]:
        return [
            item
            for item in self.contradictions.values()
            if not item.resolved
        ]

    @property
    def completed_objectives(
        self,
    ) -> List[ResearchObjective]:
        return [
            objective
            for objective in self.objectives.values()
            if objective.complete
        ]

    def ready_for_synthesis(self) -> bool:
        """
        Conservative check for whether synthesis can begin.
        """

        required_objectives = [
            objective
            for objective in self.objectives.values()
            if objective.required
        ]

        if not required_objectives:
            return False

        objectives_ready = all(
            objective.coverage >= 0.80
            for objective in required_objectives
        )

        confidence_ready = (
            self.overall_confidence
            >= self.config.minimum_confidence
        )

        return (
            objectives_ready
            and confidence_ready
        )

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


# =====================================================================
# FINAL RESULT
# =====================================================================


@dataclass
class ResearchFinding:
    """
    Individual finding included in a final result.
    """

    id: str

    statement: str

    confidence: float = 0.0

    importance: float = 0.5

    claim_ids: List[str] = field(
        default_factory=list
    )

    source_ids: List[str] = field(
        default_factory=list
    )

    evidence_ids: List[str] = field(
        default_factory=list
    )

    caveats: List[str] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self.confidence = clamp(
            self.confidence
        )

        self.importance = clamp(
            self.importance
        )

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


@dataclass
class ResearchResult(ResearchModel):
    """
    Final structured output of the research engine.

    The result intentionally retains provenance and uncertainty
    instead of reducing everything to a single text response.
    """

    question: str = ""

    status: ResearchStatus = ResearchStatus.CREATED

    summary: str = ""

    findings: List[ResearchFinding] = field(
        default_factory=list
    )

    claim_ids: List[str] = field(
        default_factory=list
    )

    source_ids: List[str] = field(
        default_factory=list
    )

    evidence_ids: List[str] = field(
        default_factory=list
    )

    objective_ids: List[str] = field(
        default_factory=list
    )

    unresolved_gap_ids: List[str] = field(
        default_factory=list
    )

    unresolved_contradiction_ids: List[str] = field(
        default_factory=list
    )

    confidence: float = 0.0

    coverage: float = 0.0

    limitations: List[str] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    stopping_reason: StoppingReason = (
        StoppingReason.NONE
    )

    generated_at: str = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        self.confidence = clamp(
            self.confidence
        )

        self.coverage = clamp(
            self.coverage
        )

    @property
    def high_confidence(self) -> bool:
        return self.confidence >= 0.80

    @property
    def sufficiently_covered(self) -> bool:
        return self.coverage >= 0.90

    def add_finding(
        self,
        finding: ResearchFinding,
    ) -> None:

        self.findings.append(finding)

    def add_warning(
        self,
        warning: str,
    ) -> None:

        if (
            warning
            and warning not in self.warnings
        ):
            self.warnings.append(warning)

    def add_limitation(
        self,
        limitation: str,
    ) -> None:

        if (
            limitation
            and limitation not in self.limitations
        ):
            self.limitations.append(
                limitation
            )


# =====================================================================
# FACTORY HELPERS
# =====================================================================


def create_objective(
    objective_id: str,
    title: str,
    description: str = "",
    priority: Priority = Priority.NORMAL,
    required: bool = True,
) -> ResearchObjective:

    return ResearchObjective(
        id=objective_id,
        title=title.strip(),
        description=description.strip(),
        priority=priority,
        required=required,
    )


def create_task(
    task_id: str,
    title: str,
    task_type: TaskType = TaskType.CUSTOM,
    description: str = "",
    objective_id: Optional[str] = None,
    priority: Priority = Priority.NORMAL,
) -> ResearchTask:

    return ResearchTask(
        id=task_id,
        title=title.strip(),
        description=description.strip(),
        task_type=task_type,
        objective_id=objective_id,
        priority=priority,
    )


def create_query(
    query_id: str,
    text: str,
    purpose: QueryPurpose = QueryPurpose.DISCOVERY,
    objective_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> ResearchQuery:

    return ResearchQuery(
        id=query_id,
        text=text.strip(),
        purpose=purpose,
        objective_id=objective_id,
        task_id=task_id,
    )


def create_session(
    session_id: str,
    question: str,
    title: str = "",
    config: Optional[ResearchConfig] = None,
) -> ResearchSession:

    return ResearchSession(
        id=session_id,
        question=question.strip(),
        title=title.strip(),
        config=config or ResearchConfig(),
    )


def create_result(
    result_id: str,
    question: str,
) -> ResearchResult:

    return ResearchResult(
        id=result_id,
        question=question.strip(),
    )


# =====================================================================
# PUBLIC API
# =====================================================================


__all__ = [
    # Version
    "MODEL_VERSION",

    # Enums
    "ResearchStatus",
    "TaskStatus",
    "TaskType",
    "Priority",
    "ConfidenceLevel",
    "SourceType",
    "SourceAuthority",
    "EvidenceType",
    "ClaimStatus",
    "QueryPurpose",
    "ResearchDepth",
    "NodeType",
    "EdgeType",
    "QueryExecutionStatus",
    "ContradictionType",
    "GapType",
    "DecisionType",
    "StoppingReason",
    "RelationshipStrength",

    # Helpers
    "utc_now",
    "clamp",
    "confidence_level",
    "weighted_average",
    "model_to_dict",

    # Core models
    "ResearchModel",
    "ProvenanceRecord",

    # Graph
    "ResearchNode",
    "ResearchEdge",
    "ResearchGraph",

    # Planning
    "ResearchObjective",
    "TaskDependency",
    "ResearchTask",
    "SearchStrategy",
    "SearchPlan",

    # Queries
    "ResearchQuery",
    "QueryAttempt",

    # Sources
    "SourceProvenance",
    "SourceRecord",
    "SourceRelationship",

    # Knowledge
    "EvidenceRecord",
    "ClaimRecord",
    "Hypothesis",
    "Contradiction",
    "KnowledgeGap",

    # Coverage
    "CoverageRecord",

    # Decisions / control
    "ResearchDecision",
    "StoppingCriteria",
    "ResearchBudget",
    "ResearchIteration",

    # Runtime
    "ResearchMetrics",
    "ResearchConfig",
    "ResearchState",
    "ResearchSession",

    # Output
    "ResearchFinding",
    "ResearchResult",

    # Factories
    "create_objective",
    "create_task",
    "create_query",
    "create_session",
    "create_result",
]