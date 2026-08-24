"""
OurPlatform AI Assistant Core
=============================

The central orchestration layer for the OurPlatform AI system.

Design goals
------------
* Keep the assistant independent from any particular model provider.
* Treat Search, API, tools, memory, and application services as capabilities.
* Provide one stable entry point for the rest of the application.
* Fail gracefully when optional subsystems are not connected yet.
* Keep state, routing, execution, validation, diagnostics, and configuration
  in one coherent foundation while allowing specialized systems to remain
  in their own modules.

This file is intentionally self-contained. Existing project components can be
attached through adapters/registrations without requiring them to exist at
import time.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)


# ============================================================================
# VERSION / CONSTANTS
# ============================================================================

ASSISTANT_NAME = "OurPlatform Assistant"
ASSISTANT_VERSION = "2.0.0"
PROTOCOL_VERSION = "1.0"

DEFAULT_MAX_HISTORY = 100
DEFAULT_MAX_CONTEXT_MESSAGES = 24
DEFAULT_MAX_OUTPUT_LENGTH = 12000
DEFAULT_MAX_TOOL_ROUNDS = 8
DEFAULT_TOOL_TIMEOUT = 30.0
DEFAULT_MODEL_TIMEOUT = 60.0
DEFAULT_CACHE_SIZE = 256

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"

TEXT_FIELDS = ("text", "content", "message", "response", "output")


# ============================================================================
# ENUMERATIONS
# ============================================================================

class AssistantMode(str, Enum):
    NORMAL = "normal"
    SEARCH = "search"
    RESEARCH = "research"
    TOOL = "tool"
    CONVERSATION = "conversation"
    COMMAND = "command"
    DEBUG = "debug"


class IntentType(str, Enum):
    CHAT = "chat"
    SEARCH = "search"
    RESEARCH = "research"
    COMMAND = "command"
    TOOL = "tool"
    QUESTION = "question"
    FOLLOW_UP = "follow_up"
    UNKNOWN = "unknown"


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"


class ErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    EMPTY_MESSAGE = "empty_message"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_FAILED = "tool_failed"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_FAILED = "model_failed"
    TIMEOUT = "timeout"
    CONTEXT_ERROR = "context_error"
    SAFETY_BLOCKED = "safety_blocked"
    INTERNAL_ERROR = "internal_error"


# ============================================================================
# LOW-LEVEL UTILITIES
# ============================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def truncate_text(value: Any, maximum: int) -> str:
    text = clean_text(value)
    if len(text) <= maximum:
        return text
    if maximum <= 3:
        return text[:maximum]
    return text[: maximum - 3] + "..."


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return json_safe(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return json_safe(vars(value))
        except Exception:
            pass
    return str(value)


def extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in TEXT_FIELDS:
            if key in value:
                return clean_text(value[key])
    for key in TEXT_FIELDS:
        if hasattr(value, key):
            return clean_text(getattr(value, key))
    return clean_text(value)


def is_awaitable(value: Any) -> bool:
    return inspect.isawaitable(value)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Message:
    role: str
    content: str
    id: str = field(default_factory=lambda: new_id("msg"))
    timestamp: str = field(default_factory=utc_iso)
    name: Optional[str] = None
    tool_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_value(cls, value: Any, default_role: str = ROLE_USER) -> "Message":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                role=clean_text(value.get("role", default_role)) or default_role,
                content=extract_text(value),
                id=clean_text(value.get("id")) or new_id("msg"),
                timestamp=clean_text(value.get("timestamp")) or utc_iso(),
                name=value.get("name"),
                tool_name=value.get("tool_name"),
                metadata=dict(value.get("metadata") or {}),
            )
        return cls(role=default_role, content=clean_text(value))


@dataclass
class Context:
    session_id: str
    user_id: Optional[str] = None
    mode: AssistantMode = AssistantMode.NORMAL
    variables: Dict[str, Any] = field(default_factory=dict)
    facts: Dict[str, Any] = field(default_factory=dict)
    active_task: Optional[str] = None
    previous_intent: Optional[IntentType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def remember(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "mode": self.mode.value,
            "variables": json_safe(self.variables),
            "facts": json_safe(self.facts),
            "active_task": self.active_task,
            "previous_intent": (
                self.previous_intent.value
                if self.previous_intent
                else None
            ),
            "metadata": json_safe(self.metadata),
        }


@dataclass
class Intent:
    type: IntentType
    confidence: float
    query: str
    entities: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    requires_search: bool = False
    requires_tool: bool = False
    requires_model: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("call"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class ToolResult:
    call_id: str
    name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class AssistantRequest:
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    mode: Optional[AssistantMode] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "AssistantRequest":
        return AssistantRequest(
            message=clean_text(self.message),
            session_id=clean_text(self.session_id) or None,
            user_id=clean_text(self.user_id) or None,
            mode=(
                self.mode
                if isinstance(self.mode, AssistantMode)
                else (
                    AssistantMode(clean_text(self.mode))
                    if self.mode
                    and clean_text(self.mode) in {m.value for m in AssistantMode}
                    else None
                )
            ),
            metadata=dict(self.metadata or {}),
            options=dict(self.options or {}),
        )


@dataclass
class AssistantResponse:
    text: str
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    session_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: new_id("req"))
    intent: Optional[Intent] = None
    tool_results: List[ToolResult] = field(default_factory=list)
    sources: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status in {
            ExecutionStatus.SUCCESS,
            ExecutionStatus.PARTIAL,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "status": self.status.value,
            "success": self.success,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "intent": self.intent.to_dict() if self.intent else None,
            "tool_results": [x.to_dict() for x in self.tool_results],
            "sources": json_safe(self.sources),
            "metadata": json_safe(self.metadata),
            "errors": json_safe(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass
class AssistantConfig:
    max_history: int = DEFAULT_MAX_HISTORY
    max_context_messages: int = DEFAULT_MAX_CONTEXT_MESSAGES
    max_output_length: int = DEFAULT_MAX_OUTPUT_LENGTH
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT
    model_timeout: float = DEFAULT_MODEL_TIMEOUT
    cache_size: int = DEFAULT_CACHE_SIZE
    enable_search: bool = True
    enable_tools: bool = True
    enable_memory: bool = True
    enable_model: bool = True
    enable_caching: bool = True
    enable_safety: bool = True
    enable_intent_detection: bool = True
    allow_unknown_tools: bool = False
    include_debug_metadata: bool = False
    system_prompt: str = (
        "You are the OurPlatform Assistant. Be useful, accurate, clear, "
        "and honest about uncertainty. Use connected capabilities when "
        "they materially improve the answer."
    )

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class AssistantStats:
    requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    partial_requests: int = 0
    blocked_requests: int = 0
    model_calls: int = 0
    search_calls: int = 0
    tool_calls: int = 0
    cache_hits: int = 0
    total_latency_ms: float = 0.0
    total_input_chars: int = 0
    total_output_chars: int = 0

    def record(self, response: AssistantResponse, latency_ms: float) -> None:
        self.requests += 1
        self.total_latency_ms += latency_ms
        self.total_input_chars += safe_int(response.metadata.get("input_chars"))
        self.total_output_chars += len(response.text)
        if response.status == ExecutionStatus.SUCCESS:
            self.successful_requests += 1
        elif response.status == ExecutionStatus.PARTIAL:
            self.partial_requests += 1
        elif response.status == ExecutionStatus.BLOCKED:
            self.blocked_requests += 1
        else:
            self.failed_requests += 1

    def to_dict(self) -> Dict[str, Any]:
        average = self.total_latency_ms / self.requests if self.requests else 0.0
        return {
            **json_safe(asdict(self)),
            "average_latency_ms": average,
        }


@dataclass
class Capability:
    name: str
    handler: Callable[..., Any]
    description: str = ""
    aliases: Tuple[str, ...] = ()
    enabled: bool = True
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, name: str) -> bool:
        target = clean_text(name).lower()
        return target == self.name.lower() or target in {
            x.lower() for x in self.aliases
        }


@dataclass
class MiddlewareResult:
    request: AssistantRequest
    context: Context
    response: Optional[AssistantResponse] = None


# ============================================================================
# PROTOCOLS
# ============================================================================

class ModelProvider(Protocol):
    def generate(self, messages: Sequence[Message], **kwargs: Any) -> Any:
        ...


class SearchProvider(Protocol):
    def search(self, query: str, **kwargs: Any) -> Any:
        ...


class MemoryProvider(Protocol):
    def load(self, session_id: str, **kwargs: Any) -> Any:
        ...

    def save(self, session_id: str, data: Any, **kwargs: Any) -> Any:
        ...


# ============================================================================
# SESSION STORE
# ============================================================================

class SessionStore:
    """Thread-safe in-memory session store.

    It is intentionally simple but exposes a clean boundary for replacing
    it later with Redis, a database, a file store, or another backend.
    """

    def __init__(self, max_sessions: int = 1000) -> None:
        self.max_sessions = max(1, max_sessions)
        self._sessions: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.RLock()

    def create(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        sid = clean_text(session_id) or new_id("session")
        with self._lock:
            data = self._sessions.get(sid)
            if data is None:
                data = {
                    "id": sid,
                    "user_id": user_id,
                    "created_at": utc_iso(),
                    "updated_at": utc_iso(),
                    "messages": [],
                    "context": {},
                }
                self._sessions[sid] = data
            elif user_id and not data.get("user_id"):
                data["user_id"] = user_id
            data["updated_at"] = utc_iso()
            self._sessions.move_to_end(sid)
            self._evict()
            return data

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self._sessions.get(session_id)
            if data is not None:
                self._sessions.move_to_end(session_id)
            return data

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def append_message(self, session_id: str, message: Message) -> None:
        with self._lock:
            data = self.create(session_id)
            data["messages"].append(message.to_dict())
            data["updated_at"] = utc_iso()

    def messages(self, session_id: str) -> List[Message]:
        data = self.get(session_id)
        if not data:
            return []
        return [Message.from_value(x) for x in data.get("messages", [])]

    def set_context(self, session_id: str, context: Context) -> None:
        with self._lock:
            data = self.create(session_id, context.user_id)
            data["context"] = context.to_dict()
            data["updated_at"] = utc_iso()

    def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        data = self.get(session_id)
        return data.get("context") if data else None

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [json_safe(x) for x in self._sessions.values()]

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def _evict(self) -> None:
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)


# ============================================================================
# MEMORY LAYER
# ============================================================================

class MemoryManager:
    """Small memory facade.

    The assistant can use local session memory immediately and can later
    delegate durable memory to a registered provider.
    """

    def __init__(
        self,
        provider: Optional[MemoryProvider] = None,
        enabled: bool = True,
    ) -> None:
        self.provider = provider
        self.enabled = enabled

    def load(self, session_id: str) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        if self.provider is None:
            return {}
        try:
            value = self.provider.load(session_id)
            if isinstance(value, Mapping):
                return dict(value)
            return {"value": value}
        except Exception:
            return {}

    def save(self, session_id: str, data: Mapping[str, Any]) -> bool:
        if not self.enabled or self.provider is None:
            return False
        try:
            self.provider.save(session_id, dict(data))
            return True
        except Exception:
            return False


# ============================================================================
# CACHE
# ============================================================================

class ResponseCache:
    """Bounded TTL cache for deterministic/repeatable assistant operations."""

    def __init__(self, maximum: int = DEFAULT_CACHE_SIZE, ttl: float = 60.0) -> None:
        self.maximum = max(1, maximum)
        self.ttl = max(0.0, ttl)
        self._items: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.RLock()

    def _expired(self, timestamp: float) -> bool:
        return self.ttl > 0 and (time.monotonic() - timestamp) >= self.ttl

    def get(self, key: str) -> Any:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            timestamp, value = item
            if self._expired(timestamp):
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._items[key] = (time.monotonic(), value)
            self._items.move_to_end(key)
            while len(self._items) > self.maximum:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._items)


# ============================================================================
# INTENT DETECTION
# ============================================================================

class IntentDetector:
    """Lightweight deterministic intent detector.

    This is not intended to replace a future ML classifier. It provides a
    useful baseline immediately and gives the orchestration layer a stable
    contract for deciding which capabilities should be involved.
    """

    SEARCH_PATTERNS = (
        r"\bsearch\b",
        r"\bfind\b",
        r"\blook\s+up\b",
        r"\blook\s+for\b",
        r"\bwhere\s+can\s+i\s+find\b",
        r"\bresults?\b",
    )

    RESEARCH_PATTERNS = (
        r"\bresearch\b",
        r"\binvestigate\b",
        r"\bcompare\b",
        r"\banalyse\b",
        r"\banalyze\b",
        r"\bdeep\s*dive\b",
    )

    COMMAND_PATTERNS = (
        r"^/",
        r"^\s*(open|close|reset|clear|configure|enable|disable|run|execute)\b",
    )

    QUESTION_PATTERNS = (
        r"\?$",
        r"^\s*(what|why|how|when|where|who|which|can|could|is|are|do|does)\b",
    )

    TOOL_PATTERNS = (
        r"\bcalculate\b",
        r"\bconvert\b",
        r"\bsummarize\b",
        r"\bformat\b",
        r"\bgenerate\b",
        r"\bcreate\b",
    )

    def detect(
        self,
        message: str,
        context: Optional[Context] = None,
    ) -> Intent:
        text = clean_text(message)
        lowered = text.lower()

        if not text:
            return Intent(IntentType.UNKNOWN, 0.0, text)

        if re.search(self.COMMAND_PATTERNS[0], text):
            return Intent(
                IntentType.COMMAND,
                0.98,
                text,
                requires_tool=True,
                requires_model=False,
                reasons=["command_prefix"],
            )

        if any(re.search(p, lowered) for p in self.RESEARCH_PATTERNS):
            return Intent(
                IntentType.RESEARCH,
                0.90,
                text,
                requires_search=True,
                reasons=["research_language"],
            )

        if any(re.search(p, lowered) for p in self.SEARCH_PATTERNS):
            return Intent(
                IntentType.SEARCH,
                0.86,
                text,
                requires_search=True,
                reasons=["search_language"],
            )

        if any(re.search(p, lowered) for p in self.TOOL_PATTERNS):
            return Intent(
                IntentType.TOOL,
                0.72,
                text,
                requires_tool=True,
                reasons=["action_language"],
            )

        if any(re.search(p, lowered) for p in self.QUESTION_PATTERNS):
            intent_type = (
                IntentType.FOLLOW_UP
                if context and context.previous_intent
                else IntentType.QUESTION
            )
            return Intent(
                intent_type,
                0.70,
                text,
                reasons=["question_language"],
            )

        if context and context.previous_intent:
            return Intent(
                IntentType.FOLLOW_UP,
                0.55,
                text,
                reasons=["conversation_context"],
            )

        return Intent(
            IntentType.CHAT,
            0.60,
            text,
            reasons=["default_conversation"],
        )


# ============================================================================
# TOOL REGISTRY
# ============================================================================

class ToolRegistry:
    """Registry and executor for assistant capabilities."""

    def __init__(self) -> None:
        self._tools: Dict[str, Capability] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        *,
        description: str = "",
        aliases: Iterable[str] = (),
        enabled: bool = True,
        priority: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Capability:
        capability = Capability(
            name=clean_text(name),
            handler=handler,
            description=description,
            aliases=tuple(aliases),
            enabled=enabled,
            priority=priority,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._tools[capability.name] = capability
        return capability

    def unregister(self, name: str) -> bool:
        with self._lock:
            return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[Capability]:
        target = clean_text(name)
        with self._lock:
            direct = self._tools.get(target)
            if direct:
                return direct
            for tool in self._tools.values():
                if tool.matches(target):
                    return tool
        return None

    def list(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        with self._lock:
            values = sorted(
                self._tools.values(),
                key=lambda x: (-x.priority, x.name),
            )
            return [
                {
                    "name": x.name,
                    "description": x.description,
                    "aliases": list(x.aliases),
                    "enabled": x.enabled,
                    "priority": x.priority,
                    "metadata": json_safe(x.metadata),
                }
                for x in values
                if not enabled_only or x.enabled
            ]

    def enable(self, name: str, value: bool = True) -> bool:
        tool = self.get(name)
        if tool is None:
            return False
        tool.enabled = value
        return True

    def call(
        self,
        call: ToolCall,
        *,
        timeout: float = DEFAULT_TOOL_TIMEOUT,
    ) -> ToolResult:
        started = time.perf_counter()
        tool = self.get(call.name)

        if tool is None:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                success=False,
                error=f"Tool not found: {call.name}",
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        if not tool.enabled:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                success=False,
                error=f"Tool disabled: {call.name}",
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        try:
            result = tool.handler(**call.arguments)
            if inspect.isawaitable(result):
                result = _run_awaitable(result, timeout=timeout)
            return ToolResult(
                call_id=call.id,
                name=tool.name,
                success=True,
                output=json_safe(result),
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return ToolResult(
                call_id=call.id,
                name=tool.name,
                success=False,
                error=str(exc),
                duration_ms=(time.perf_counter() - started) * 1000,
            )


# ============================================================================
# SAFETY / VALIDATION
# ============================================================================

class SafetyGuard:
    """Conservative application-level safety gate.

    This is deliberately not presented as a replacement for platform-level
    safety systems. It simply prevents obviously invalid or dangerous
    internal execution requests from being treated as normal tool calls.
    """

    BLOCKED_TOOL_NAMES = {
        "delete_everything",
        "wipe_database",
        "destroy_system",
    }

    def check_request(self, request: AssistantRequest) -> Tuple[bool, Optional[str]]:
        if not request.message.strip():
            return False, "Message cannot be empty."
        return True, None

    def check_tool(self, call: ToolCall) -> Tuple[bool, Optional[str]]:
        if call.name.lower() in self.BLOCKED_TOOL_NAMES:
            return False, f"Tool '{call.name}' is not permitted."
        return True, None

    def sanitize_output(self, text: str, maximum: int) -> str:
        return truncate_text(text, maximum)


# ============================================================================
# PROMPT / CONTEXT ASSEMBLY
# ============================================================================

class ContextAssembler:
    """Builds the model-facing conversation context."""

    def __init__(self, config: AssistantConfig) -> None:
        self.config = config

    def build(
        self,
        history: Sequence[Message],
        context: Context,
        *,
        intent: Optional[Intent] = None,
        sources: Optional[Sequence[Any]] = None,
    ) -> List[Message]:
        result = [
            Message(
                role=ROLE_SYSTEM,
                content=self.config.system_prompt,
                metadata={"generated": True},
            )
        ]

        if context.facts:
            result.append(
                Message(
                    role=ROLE_SYSTEM,
                    content=(
                        "Relevant remembered context:\n"
                        + json.dumps(
                            json_safe(context.facts),
                            ensure_ascii=False,
                        )
                    ),
                    metadata={"context": True},
                )
            )

        if intent:
            result.append(
                Message(
                    role=ROLE_SYSTEM,
                    content=(
                        f"Detected intent: {intent.type.value}. "
                        f"Confidence: {intent.confidence:.2f}."
                    ),
                    metadata={"intent": True},
                )
            )

        if sources:
            source_text = json.dumps(
                json_safe(list(sources)),
                ensure_ascii=False,
            )
            result.append(
                Message(
                    role=ROLE_SYSTEM,
                    content=f"Available search/context sources:\n{source_text}",
                    metadata={"sources": True},
                )
            )

        result.extend(
            list(history)[-self.config.max_context_messages :]
        )
        return result


# ============================================================================
# RESPONSE NORMALIZATION
# ============================================================================

class ResponseNormalizer:
    def __init__(self, config: AssistantConfig) -> None:
        self.config = config

    def normalize(
        self,
        result: Any,
        *,
        session_id: Optional[str] = None,
        intent: Optional[Intent] = None,
    ) -> AssistantResponse:
        if isinstance(result, AssistantResponse):
            result.text = self._clean(result.text)
            return result

        if isinstance(result, Mapping):
            text = extract_text(result)
            sources = result.get("sources", [])
            metadata = dict(result.get("metadata") or {})
            status_value = result.get("status", ExecutionStatus.SUCCESS.value)
            try:
                status = ExecutionStatus(status_value)
            except ValueError:
                status = ExecutionStatus.SUCCESS
            return AssistantResponse(
                text=self._clean(text),
                status=status,
                session_id=session_id,
                intent=intent,
                sources=list(sources or []),
                metadata=metadata,
            )

        return AssistantResponse(
            text=self._clean(extract_text(result)),
            session_id=session_id,
            intent=intent,
        )

    def _clean(self, text: str) -> str:
        text = clean_text(text)
        if not text:
            text = "I wasn't able to produce a response."
        return truncate_text(text, self.config.max_output_length)


# ============================================================================
# COMMAND ROUTER
# ============================================================================

class CommandRouter:
    """Handles built-in assistant commands."""

    def __init__(self, assistant: "Assistant") -> None:
        self.assistant = assistant

    def is_command(self, text: str) -> bool:
        return clean_text(text).startswith("/")

    def execute(
        self,
        text: str,
        request: AssistantRequest,
        context: Context,
    ) -> Optional[AssistantResponse]:
        raw = clean_text(text)
        if not raw.startswith("/"):
            return None

        parts = raw[1:].split()
        if not parts:
            return self.help(request, context)

        command = parts[0].lower()
        args = parts[1:]

        handlers = {
            "help": self.help,
            "status": self.status,
            "tools": self.tools,
            "clear": self.clear,
            "reset": self.reset,
            "mode": lambda r, c: self.mode(r, c, args),
            "history": self.history,
            "config": self.config,
        }

        handler = handlers.get(command)
        if handler is None:
            return AssistantResponse(
                text=f"Unknown command: /{command}. Try /help.",
                session_id=context.session_id,
                status=ExecutionStatus.FAILED,
            )
        return handler(request, context)

    def help(self, request: AssistantRequest, context: Context) -> AssistantResponse:
        return AssistantResponse(
            text=(
                "Available commands: /help, /status, /tools, /clear, "
                "/reset, /mode <mode>, /history, /config"
            ),
            session_id=context.session_id,
        )

    def status(self, request: AssistantRequest, context: Context) -> AssistantResponse:
        status = self.assistant.status()
        return AssistantResponse(
            text=json.dumps(status, indent=2, ensure_ascii=False),
            session_id=context.session_id,
            metadata={"command": "status"},
        )

    def tools(self, request: AssistantRequest, context: Context) -> AssistantResponse:
        tools = self.assistant.tools.list()
        return AssistantResponse(
            text=json.dumps(tools, indent=2, ensure_ascii=False),
            session_id=context.session_id,
            metadata={"command": "tools"},
        )

    def clear(self, request: AssistantRequest, context: Context) -> AssistantResponse:
        self.assistant.clear_session(context.session_id)
        return AssistantResponse(
            text="Conversation history cleared.",
            session_id=context.session_id,
        )

    def reset(self, request: AssistantRequest, context: Context) -> AssistantResponse:
        self.assistant.reset_session(context.session_id)
        return AssistantResponse(
            text="Session reset.",
            session_id=context.session_id,
        )

    def mode(
        self,
        request: AssistantRequest,
        context: Context,
        args: Sequence[str],
    ) -> AssistantResponse:
        if not args:
            return AssistantResponse(
                text=f"Current mode: {context.mode.value}",
                session_id=context.session_id,
            )
        try:
            context.mode = AssistantMode(args[0].lower())
        except ValueError:
            return AssistantResponse(
                text="Unknown mode.",
                session_id=context.session_id,
                status=ExecutionStatus.FAILED,
            )
        self.assistant._save_context(context)
        return AssistantResponse(
            text=f"Mode changed to {context.mode.value}.",
            session_id=context.session_id,
        )

    def history(self, request: AssistantRequest, context: Context) -> AssistantResponse:
        messages = self.assistant.history(context.session_id)
        text = "\n".join(
            f"{m.role}: {m.content}" for m in messages[-20:]
        ) or "No history."
        return AssistantResponse(
            text=text,
            session_id=context.session_id,
        )

    def config(self, request: AssistantRequest, context: Context) -> AssistantResponse:
        return AssistantResponse(
            text=json.dumps(
                self.assistant.config.to_dict(),
                indent=2,
                ensure_ascii=False,
            ),
            session_id=context.session_id,
        )


# ============================================================================
# MAIN ASSISTANT
# ============================================================================

class Assistant:
    """Central OurPlatform AI orchestration engine.

    The assistant is deliberately a coordinator. It does not attempt to
    reimplement Search, the API, or a model provider. Those systems plug in
    through the registration methods below.
    """

    def __init__(
        self,
        *,
        name: str = ASSISTANT_NAME,
        config: Optional[AssistantConfig] = None,
        model: Optional[Any] = None,
        search: Optional[Any] = None,
        memory: Optional[Any] = None,
        session_store: Optional[SessionStore] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.name = name
        self.version = ASSISTANT_VERSION
        self.config = config or AssistantConfig()

        self.model = model
        self.search_provider = search
        self.memory = MemoryManager(
            provider=memory,
            enabled=self.config.enable_memory,
        )

        self.sessions = session_store or SessionStore()
        self.tools = ToolRegistry()
        self.intent_detector = IntentDetector()
        self.safety = SafetyGuard()
        self.context_assembler = ContextAssembler(self.config)
        self.normalizer = ResponseNormalizer(self.config)
        self.cache = ResponseCache(self.config.cache_size)

        self.logger = logger or logging.getLogger("ourplatform.ai.assistant")
        self.stats = AssistantStats()

        self._middleware: List[Callable[..., Any]] = []
        self._hooks: Dict[str, List[Callable[..., Any]]] = {}
        self._lock = threading.RLock()
        self._started = utc_iso()

        self.commands = CommandRouter(self)

        self._register_builtin_tools()

    # ----------------------------------------------------------------------
    # PUBLIC REQUEST API
    # ----------------------------------------------------------------------

    def respond(
        self,
        message: Any,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        mode: Optional[AssistantMode] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> AssistantResponse:
        request = AssistantRequest(
            message=extract_text(message),
            session_id=session_id,
            user_id=user_id,
            mode=mode,
            metadata=dict(metadata or {}),
            options=dict(options or {}),
        )
        return self.handle(request)

    def handle(self, request: AssistantRequest) -> AssistantResponse:
        started = time.perf_counter()
        request = request.normalized()
        request_id = new_id("req")

        allowed, reason = self.safety.check_request(request)
        if not allowed:
            response = AssistantResponse(
                text=reason or "Request rejected.",
                status=ExecutionStatus.BLOCKED,
                request_id=request_id,
            )
            self.stats.record(response, (time.perf_counter() - started) * 1000)
            return response

        session = self.sessions.create(
            request.session_id,
            request.user_id,
        )
        sid = session["id"]
        context = self._load_context(sid, request.user_id)

        if request.mode:
            context.mode = request.mode

        response: Optional[AssistantResponse] = None

        try:
            self._emit("request_started", request=request, context=context)

            middleware_response = self._run_before_middleware(request, context)
            if middleware_response is not None:
                response = middleware_response
            else:
                response = self._execute(request, context, request_id)

            response = self._run_after_middleware(
                request,
                context,
                response,
            )

            response.session_id = sid
            response.request_id = request_id
            response.metadata.setdefault("input_chars", len(request.message))
            response.metadata.setdefault("latency_ms", (time.perf_counter() - started) * 1000)

            self._record_exchange(sid, request, response)
            self._save_context(context)
            self._emit(
                "request_completed",
                request=request,
                context=context,
                response=response,
            )

        except Exception as exc:
            self.logger.exception("Assistant request failed")
            response = AssistantResponse(
                text="I couldn't complete that request.",
                status=ExecutionStatus.FAILED,
                session_id=sid,
                request_id=request_id,
                errors=[
                    {
                        "code": ErrorCode.INTERNAL_ERROR.value,
                        "message": str(exc),
                    }
                ],
            )

        latency = (time.perf_counter() - started) * 1000
        response.metadata.setdefault("latency_ms", latency)
        self.stats.record(response, latency)
        return response

    async def arespond(self, message: Any, **kwargs: Any) -> AssistantResponse:
        request = AssistantRequest(
            message=extract_text(message),
            session_id=kwargs.pop("session_id", None),
            user_id=kwargs.pop("user_id", None),
            mode=kwargs.pop("mode", None),
            metadata=kwargs.pop("metadata", {}) or {},
            options=kwargs.pop("options", {}) or {},
        )
        return await self.ahandle(request)

    async def ahandle(self, request: AssistantRequest) -> AssistantResponse:
        request = request.normalized()
        if not request.message:
            return AssistantResponse(
                text="Message cannot be empty.",
                status=ExecutionStatus.BLOCKED,
            )

        # Execute the synchronous orchestration in a worker so async callers
        # do not have to know which registered components are synchronous.
        return await asyncio.to_thread(self.handle, request)

    # ----------------------------------------------------------------------
    # CORE EXECUTION
    # ----------------------------------------------------------------------

    def _execute(
        self,
        request: AssistantRequest,
        context: Context,
        request_id: str,
    ) -> AssistantResponse:
        intent = (
            self.intent_detector.detect(request.message, context)
            if self.config.enable_intent_detection
            else Intent(IntentType.CHAT, 0.5, request.message)
        )
        context.previous_intent = intent.type

        command_response = self.commands.execute(
            request.message,
            request,
            context,
        )
        if command_response is not None:
            command_response.intent = intent
            return command_response

        if intent.type in {IntentType.SEARCH, IntentType.RESEARCH}:
            search_response = self._search(request, context, intent)
            if search_response is not None:
                # Search is allowed to feed the model rather than always
                # becoming the final response.
                if request.options.get("search_only", False):
                    search_response.intent = intent
                    return search_response

                if self.model is None:
                    search_response.intent = intent
                    return search_response

                model_response = self._generate(
                    request,
                    context,
                    intent,
                    sources=search_response.sources,
                )
                if model_response:
                    model_response.intent = intent
                    model_response.sources = search_response.sources
                    return model_response

                search_response.intent = intent
                return search_response

        if intent.requires_tool and self.config.enable_tools:
            tool_response = self._try_tool_from_request(
                request,
                context,
                intent,
            )
            if tool_response is not None:
                return tool_response

        generated = self._generate(
            request,
            context,
            intent,
        )
        if generated is not None:
            generated.intent = intent
            return generated

        fallback = self._fallback_response(request, intent)
        fallback.intent = intent
        return fallback

    # ----------------------------------------------------------------------
    # SEARCH INTEGRATION
    # ----------------------------------------------------------------------

    def _search(
        self,
        request: AssistantRequest,
        context: Context,
        intent: Intent,
    ) -> Optional[AssistantResponse]:
        if not self.config.enable_search:
            return None
        provider = self.search_provider
        if provider is None:
            return None

        self.stats.search_calls += 1
        query = self._resolve_search_query(request.message, context)
        cache_key = f"search:{query.lower()}"

        if self.config.enable_caching:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.stats.cache_hits += 1
                return self._search_response(cached, query, cached=True)

        try:
            result = self._call_provider(
                provider,
                ("search", "__call__"),
                query,
                timeout=self.config.model_timeout,
                request=request,
            )
            normalized = self._normalize_search_result(result)
            if self.config.enable_caching:
                self.cache.put(cache_key, normalized)
            return self._search_response(normalized, query)
        except Exception as exc:
            self.logger.warning("Search failed: %s", exc)
            return AssistantResponse(
                text="Search is currently unavailable.",
                status=ExecutionStatus.PARTIAL,
                warnings=[str(exc)],
                sources=[],
            )

    def _search_response(
        self,
        result: Mapping[str, Any],
        query: str,
        *,
        cached: bool = False,
    ) -> AssistantResponse:
        results = list(result.get("results", []))
        if not results:
            text = f"I couldn't find any results for “{query}”."
        else:
            lines = [f"Search results for “{query}”:"]
            for index, item in enumerate(results[:10], 1):
                title = extract_text(
                    item.get("title", "")
                    if isinstance(item, Mapping)
                    else item
                )
                snippet = (
                    extract_text(item.get("snippet", ""))
                    if isinstance(item, Mapping)
                    else ""
                )
                lines.append(
                    f"{index}. {title or 'Untitled'}"
                    + (f" — {snippet}" if snippet else "")
                )
            text = "\n".join(lines)

        return AssistantResponse(
            text=text,
            sources=results,
            metadata={
                "query": query,
                "result_count": len(results),
                "cached": cached,
            },
        )

    def _normalize_search_result(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, Mapping):
            if "data" in result and isinstance(result["data"], Mapping):
                result = result["data"]
            return {
                "results": list(result.get("results", [])),
                "metadata": dict(result.get("metadata") or {}),
            }
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            return {"results": list(result), "metadata": {}}
        return {"results": [result] if result is not None else [], "metadata": {}}

    def _resolve_search_query(self, message: str, context: Context) -> str:
        text = clean_text(message)
        replacements = (
            ("search for ", ""),
            ("search ", ""),
            ("look up ", ""),
            ("find ", ""),
            ("look for ", ""),
        )
        lowered = text.lower()
        for prefix, replacement in replacements:
            if lowered.startswith(prefix):
                return text[len(prefix):].strip()
        return text

    # ----------------------------------------------------------------------
    # MODEL INTEGRATION
    # ----------------------------------------------------------------------

    def _generate(
        self,
        request: AssistantRequest,
        context: Context,
        intent: Intent,
        *,
        sources: Optional[Sequence[Any]] = None,
    ) -> Optional[AssistantResponse]:
        if not self.config.enable_model or self.model is None:
            return None

        self.stats.model_calls += 1
        history = self.history(context.session_id)
        messages = self.context_assembler.build(
            history,
            context,
            intent=intent,
            sources=sources,
        )

        try:
            result = self._call_provider(
                self.model,
                ("generate", "respond", "chat", "__call__"),
                messages,
                timeout=self.config.model_timeout,
                request=request,
                context=context,
                intent=intent,
            )
            response = self.normalizer.normalize(
                result,
                session_id=context.session_id,
                intent=intent,
            )
            response.sources = list(sources or response.sources or [])
            return response
        except Exception as exc:
            self.logger.warning("Model generation failed: %s", exc)
            return AssistantResponse(
                text="The AI model is currently unavailable.",
                status=ExecutionStatus.PARTIAL,
                session_id=context.session_id,
                warnings=[str(exc)],
            )

    # ----------------------------------------------------------------------
    # TOOL ROUTING
    # ----------------------------------------------------------------------

    def _try_tool_from_request(
        self,
        request: AssistantRequest,
        context: Context,
        intent: Intent,
    ) -> Optional[AssistantResponse]:
        explicit = request.options.get("tool")
        if explicit:
            call = ToolCall(
                name=str(explicit),
                arguments=dict(request.options.get("arguments") or {}),
            )
            return self._execute_tool(call, context)

        tool = self._infer_builtin_tool(request.message)
        if tool is None:
            return None

        call = ToolCall(
            name=tool[0],
            arguments=tool[1],
        )
        return self._execute_tool(call, context)

    def _execute_tool(
        self,
        call: ToolCall,
        context: Context,
    ) -> AssistantResponse:
        allowed, reason = self.safety.check_tool(call)
        if not allowed:
            return AssistantResponse(
                text=reason or "Tool execution blocked.",
                status=ExecutionStatus.BLOCKED,
                session_id=context.session_id,
            )

        self.stats.tool_calls += 1
        result = self.tools.call(
            call,
            timeout=self.config.tool_timeout,
        )

        if result.success:
            text = extract_text(result.output)
            if not text:
                text = json.dumps(
                    json_safe(result.output),
                    ensure_ascii=False,
                    indent=2,
                )
            return AssistantResponse(
                text=truncate_text(text, self.config.max_output_length),
                session_id=context.session_id,
                tool_results=[result],
            )

        return AssistantResponse(
            text=f"Tool '{call.name}' failed.",
            status=ExecutionStatus.PARTIAL,
            session_id=context.session_id,
            tool_results=[result],
            errors=[
                {
                    "code": ErrorCode.TOOL_FAILED.value,
                    "message": result.error or "Unknown tool error",
                }
            ],
        )

    def _infer_builtin_tool(
        self,
        message: str,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        lowered = message.lower().strip()

        if lowered.startswith("calculate "):
            expression = message.strip()[10:].strip()
            return "calculate", {"expression": expression}

        if lowered.startswith("convert "):
            payload = message.strip()[8:].strip()
            return "convert", {"value": payload}

        return None

    def _register_builtin_tools(self) -> None:
        self.tools.register(
            "calculate",
            self._tool_calculate,
            description="Evaluate a basic arithmetic expression.",
            aliases=("calculator", "math"),
        )
        self.tools.register(
            "convert",
            self._tool_convert,
            description="Parse a simple unit conversion request.",
            aliases=("unit_convert",),
        )
        self.tools.register(
            "session_info",
            self._tool_session_info,
            description="Return current session information.",
        )

    @staticmethod
    def _tool_calculate(expression: str) -> Dict[str, Any]:
        """Very small arithmetic evaluator.

        Only arithmetic characters are permitted. Names, imports, attribute
        access, calls, and other Python syntax are rejected.
        """
        expression = clean_text(expression)
        if not expression or len(expression) > 200:
            raise ValueError("Invalid expression.")

        if not re.fullmatch(r"[0-9+\-*/(). %]+", expression):
            raise ValueError("Expression contains unsupported characters.")

        # Deliberately use a restricted AST rather than eval.
        import ast
        import operator

        allowed = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        tree = ast.parse(expression, mode="eval")

        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
                return allowed[type(node.op)](evaluate(node.operand))
            if isinstance(node, ast.BinOp) and type(node.op) in allowed:
                left = evaluate(node.left)
                right = evaluate(node.right)
                if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
                    raise ValueError("Division by zero.")
                return allowed[type(node.op)](left, right)
            raise ValueError("Unsupported expression.")

        value = evaluate(tree)
        return {"expression": expression, "result": value}

    @staticmethod
    def _tool_convert(value: str) -> Dict[str, Any]:
        text = clean_text(value)
        match = re.fullmatch(
            r"([-+]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s+(?:to|in)\s+([a-zA-Z]+)",
            text,
        )
        if not match:
            return {
                "input": text,
                "result": None,
                "message": "Conversion format not recognised.",
            }

        number = float(match.group(1))
        source = match.group(2).lower()
        target = match.group(3).lower()

        factors = {
            ("km", "m"): 1000,
            ("m", "km"): 0.001,
            ("m", "cm"): 100,
            ("cm", "m"): 0.01,
            ("kg", "g"): 1000,
            ("g", "kg"): 0.001,
            ("l", "ml"): 1000,
            ("ml", "l"): 0.001,
        }

        if source == target:
            result = number
        elif (source, target) in factors:
            result = number * factors[(source, target)]
        else:
            return {
                "input": text,
                "result": None,
                "message": f"No built-in conversion for {source} -> {target}.",
            }

        return {
            "input": text,
            "value": number,
            "source": source,
            "target": target,
            "result": result,
        }

    def _tool_session_info(self, session_id: str = "") -> Dict[str, Any]:
        data = self.sessions.get(session_id)
        return data or {"session_id": session_id, "exists": False}

    # ----------------------------------------------------------------------
    # FALLBACKS
    # ----------------------------------------------------------------------

    def _fallback_response(
        self,
        request: AssistantRequest,
        intent: Intent,
    ) -> AssistantResponse:
        if self.model is None and self.search_provider is None:
            return AssistantResponse(
                text=(
                    "I'm connected and ready, but no AI model or search "
                    "provider has been attached yet."
                ),
                status=ExecutionStatus.PARTIAL,
                metadata={
                    "reason": "no_provider",
                    "intent": intent.type.value,
                },
            )

        return AssistantResponse(
            text="I couldn't determine the best way to handle that request.",
            status=ExecutionStatus.PARTIAL,
        )

    # ----------------------------------------------------------------------
    # HISTORY / CONTEXT
    # ----------------------------------------------------------------------

    def history(self, session_id: str) -> List[Message]:
        messages = self.sessions.messages(session_id)
        return messages[-self.config.max_history :]

    def clear_session(self, session_id: str) -> bool:
        data = self.sessions.get(session_id)
        if not data:
            return False
        data["messages"] = []
        data["updated_at"] = utc_iso()
        return True

    def reset_session(self, session_id: str) -> bool:
        return self.sessions.delete(session_id)

    def get_context(self, session_id: str) -> Context:
        return self._load_context(session_id)

    def set_context_value(self, session_id: str, key: str, value: Any) -> Context:
        context = self._load_context(session_id)
        context.set(key, value)
        self._save_context(context)
        return context

    def remember(self, session_id: str, key: str, value: Any) -> Context:
        context = self._load_context(session_id)
        context.remember(key, value)
        self._save_context(context)
        return context

    def _load_context(
        self,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> Context:
        raw = self.sessions.get_context(session_id) or {}
        mode_value = raw.get("mode", AssistantMode.NORMAL.value)
        try:
            mode = AssistantMode(mode_value)
        except ValueError:
            mode = AssistantMode.NORMAL

        previous_value = raw.get("previous_intent")
        try:
            previous = IntentType(previous_value) if previous_value else None
        except ValueError:
            previous = None

        context = Context(
            session_id=session_id,
            user_id=user_id or raw.get("user_id"),
            mode=mode,
            variables=dict(raw.get("variables") or {}),
            facts=dict(raw.get("facts") or {}),
            active_task=raw.get("active_task"),
            previous_intent=previous,
            metadata=dict(raw.get("metadata") or {}),
        )

        durable = self.memory.load(session_id)
        if durable:
            context.facts.update(durable.get("facts", durable))

        return context

    def _save_context(self, context: Context) -> None:
        self.sessions.set_context(context.session_id, context)
        self.memory.save(
            context.session_id,
            {"facts": context.facts},
        )

    def _record_exchange(
        self,
        session_id: str,
        request: AssistantRequest,
        response: AssistantResponse,
    ) -> None:
        self.sessions.append_message(
            session_id,
            Message(
                role=ROLE_USER,
                content=request.message,
                metadata={"request_id": response.request_id},
            ),
        )
        self.sessions.append_message(
            session_id,
            Message(
                role=ROLE_ASSISTANT,
                content=response.text,
                metadata={
                    "request_id": response.request_id,
                    "status": response.status.value,
                },
            ),
        )

    # ----------------------------------------------------------------------
    # REGISTRATION / EXTENSIBILITY
    # ----------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        handler: Callable[..., Any],
        *,
        description: str = "",
        aliases: Iterable[str] = (),
        enabled: bool = True,
        priority: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Capability:
        return self.tools.register(
            name,
            handler,
            description=description,
            aliases=aliases,
            enabled=enabled,
            priority=priority,
            metadata=metadata,
        )

    def register_capability(
        self,
        name: str,
        handler: Callable[..., Any],
        **kwargs: Any,
    ) -> Capability:
        return self.register_tool(name, handler, **kwargs)

    def attach_model(self, provider: Any) -> None:
        self.model = provider

    def attach_search(self, provider: Any) -> None:
        self.search_provider = provider

    def attach_memory(self, provider: Any) -> None:
        self.memory.provider = provider

    def add_middleware(self, middleware: Callable[..., Any]) -> None:
        self._middleware.append(middleware)

    def add_hook(self, event: str, handler: Callable[..., Any]) -> None:
        self._hooks.setdefault(event, []).append(handler)

    def remove_hook(self, event: str, handler: Callable[..., Any]) -> bool:
        handlers = self._hooks.get(event, [])
        try:
            handlers.remove(handler)
            return True
        except ValueError:
            return False

    # ----------------------------------------------------------------------
    # MIDDLEWARE / EVENTS
    # ----------------------------------------------------------------------

    def _run_before_middleware(
        self,
        request: AssistantRequest,
        context: Context,
    ) -> Optional[AssistantResponse]:
        for middleware in list(self._middleware):
            try:
                result = middleware(request, context)
                if inspect.isawaitable(result):
                    result = _run_awaitable(result)
                if isinstance(result, AssistantResponse):
                    return result
                if isinstance(result, Mapping) and result.get("response") is not None:
                    return self.normalizer.normalize(result["response"])
            except Exception as exc:
                self.logger.warning("Middleware failed: %s", exc)
        return None

    def _run_after_middleware(
        self,
        request: AssistantRequest,
        context: Context,
        response: AssistantResponse,
    ) -> AssistantResponse:
        current = response
        for middleware in list(self._middleware):
            try:
                result = middleware(request, context, current)
                if inspect.isawaitable(result):
                    result = _run_awaitable(result)
                if isinstance(result, AssistantResponse):
                    current = result
            except TypeError:
                continue
            except Exception as exc:
                self.logger.warning("After middleware failed: %s", exc)
        return current

    def _emit(self, event: str, **payload: Any) -> None:
        for handler in list(self._hooks.get(event, [])):
            try:
                result = handler(**payload)
                if inspect.isawaitable(result):
                    _run_awaitable(result)
            except Exception as exc:
                self.logger.warning("Hook '%s' failed: %s", event, exc)

    # ----------------------------------------------------------------------
    # PROVIDER INVOCATION
    # ----------------------------------------------------------------------

    @staticmethod
    def _call_provider(
        provider: Any,
        method_names: Sequence[str],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        target = None
        for name in method_names:
            candidate = getattr(provider, name, None)
            if callable(candidate):
                target = candidate
                break

        if target is None and callable(provider):
            target = provider

        if target is None:
            raise TypeError("Provider has no supported callable interface.")

        result = target(*args, **kwargs)
        if inspect.isawaitable(result):
            return _run_awaitable(result, timeout=timeout)
        return result

    # ----------------------------------------------------------------------
    # STATUS / DIAGNOSTICS
    # ----------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "protocol_version": PROTOCOL_VERSION,
            "started_at": self._started,
            "uptime_seconds": max(
                0.0,
                (utc_now() - datetime.fromisoformat(self._started)).total_seconds(),
            ),
            "providers": {
                "model": self.model is not None,
                "search": self.search_provider is not None,
                "memory": self.memory.provider is not None,
            },
            "tools": len(self.tools.list()),
            "sessions": len(self.sessions.list_sessions()),
            "cache_size": self.cache.size(),
            "configuration": self.config.to_dict(),
            "statistics": self.stats.to_dict(),
        }

    def health_check(self) -> Dict[str, Any]:
        status = self.status()
        return {
            "healthy": True,
            "service": self.name,
            "version": self.version,
            "model_connected": status["providers"]["model"],
            "search_connected": status["providers"]["search"],
            "memory_connected": status["providers"]["memory"],
            "tool_count": status["tools"],
        }

    def metrics(self) -> Dict[str, Any]:
        return self.stats.to_dict()

    def capabilities(self) -> List[Dict[str, Any]]:
        return self.tools.list()

    def export_session(self, session_id: str) -> Dict[str, Any]:
        data = self.sessions.get(session_id)
        if not data:
            return {}
        return json_safe(data)

    def import_session(self, data: Mapping[str, Any]) -> str:
        session_id = clean_text(data.get("id")) or new_id("session")
        session = self.sessions.create(
            session_id,
            data.get("user_id"),
        )
        session["messages"] = list(data.get("messages") or [])
        session["context"] = dict(data.get("context") or {})
        session["updated_at"] = utc_iso()
        return session_id

    # ----------------------------------------------------------------------
    # CONFIGURATION
    # ----------------------------------------------------------------------

    def configure(self, **values: Any) -> AssistantConfig:
        for key, value in values.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.context_assembler.config = self.config
        self.normalizer.config = self.config
        return self.config

    def enable(self, feature: str) -> bool:
        mapping = {
            "search": "enable_search",
            "tools": "enable_tools",
            "memory": "enable_memory",
            "model": "enable_model",
            "cache": "enable_caching",
            "safety": "enable_safety",
            "intent": "enable_intent_detection",
        }
        key = mapping.get(feature.lower(), feature)
        if not hasattr(self.config, key):
            return False
        setattr(self.config, key, True)
        if key == "enable_memory":
            self.memory.enabled = True
        return True

    def disable(self, feature: str) -> bool:
        mapping = {
            "search": "enable_search",
            "tools": "enable_tools",
            "memory": "enable_memory",
            "model": "enable_model",
            "cache": "enable_caching",
            "safety": "enable_safety",
            "intent": "enable_intent_detection",
        }
        key = mapping.get(feature.lower(), feature)
        if not hasattr(self.config, key):
            return False
        setattr(self.config, key, False)
        if key == "enable_memory":
            self.memory.enabled = False
        return True


# ============================================================================
# DEFAULT ASSISTANT
# ============================================================================

_default_assistant = Assistant()


def get_assistant() -> Assistant:
    return _default_assistant


def configure_assistant(
    *,
    config: Optional[AssistantConfig] = None,
    model: Optional[Any] = None,
    search: Optional[Any] = None,
    memory: Optional[Any] = None,
) -> Assistant:
    global _default_assistant

    if config is not None:
        _default_assistant.config = config
        _default_assistant.context_assembler.config = config
        _default_assistant.normalizer.config = config

    if model is not None:
        _default_assistant.attach_model(model)

    if search is not None:
        _default_assistant.attach_search(search)

    if memory is not None:
        _default_assistant.attach_memory(memory)

    return _default_assistant


def respond(
    message: Any,
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    mode: Optional[AssistantMode] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> AssistantResponse:
    return get_assistant().respond(
        message,
        session_id=session_id,
        user_id=user_id,
        mode=mode,
        metadata=metadata,
        options=options,
    )


def search(
    query: str,
    *,
    session_id: Optional[str] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> AssistantResponse:
    return respond(
        f"search {query}",
        session_id=session_id,
        options={
            "search_only": True,
            **dict(options or {}),
        },
    )


def health_check() -> Dict[str, Any]:
    return get_assistant().health_check()


def status() -> Dict[str, Any]:
    return get_assistant().status()


# ============================================================================
# ASYNC HELPERS
# ============================================================================

def _run_awaitable(
    awaitable: Awaitable[Any],
    *,
    timeout: Optional[float] = None,
) -> Any:
    """Run an awaitable from synchronous orchestration code.

    If called from a thread without a running event loop, asyncio.run is
    straightforward. If called while a loop is already running, a dedicated
    helper thread owns a fresh loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if timeout is None:
            return asyncio.run(awaitable)
        return asyncio.run(asyncio.wait_for(awaitable, timeout))

    result: List[Any] = []
    error: List[BaseException] = []

    def runner() -> None:
        try:
            async def wrapped() -> Any:
                if timeout is None:
                    return await awaitable
                return await asyncio.wait_for(awaitable, timeout)

            result.append(asyncio.run(wrapped()))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join((timeout or 60.0) + 1.0)

    if thread.is_alive():
        raise TimeoutError("Asynchronous operation timed out.")

    if error:
        raise error[0]

    return result[0] if result else None


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "ASSISTANT_NAME",
    "ASSISTANT_VERSION",
    "AssistantMode",
    "IntentType",
    "ExecutionStatus",
    "ErrorCode",
    "Message",
    "Context",
    "Intent",
    "ToolCall",
    "ToolResult",
    "AssistantRequest",
    "AssistantResponse",
    "AssistantConfig",
    "AssistantStats",
    "Capability",
    "SessionStore",
    "MemoryManager",
    "ResponseCache",
    "IntentDetector",
    "ToolRegistry",
    "SafetyGuard",
    "ContextAssembler",
    "ResponseNormalizer",
    "CommandRouter",
    "Assistant",
    "get_assistant",
    "configure_assistant",
    "respond",
    "search",
    "health_check",
    "status",
]
