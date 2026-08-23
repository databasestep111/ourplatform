from datetime import datetime
from collections import Counter
import traceback
import json


class Logger:
    """
    Advanced logging subsystem for OurPlatform.

    Designed to provide a central logging layer for
    Core and every other platform component.

    Features:
        - Structured log entries
        - Multiple log levels
        - Component tracking
        - Event tracking
        - Metadata
        - Exception capture
        - Search
        - Filtering
        - Recent activity
        - Statistics
        - Component statistics
        - Session tracking
        - Retention limits
        - Export-ready data
        - Debug mode
        - Runtime configuration
    """

    # ==========================================
    # LOG LEVELS
    # ==========================================

    LEVEL_INFO = "INFO"
    LEVEL_WARNING = "WARNING"
    LEVEL_ERROR = "ERROR"
    LEVEL_DEBUG = "DEBUG"
    LEVEL_EVENT = "EVENT"
    LEVEL_CRITICAL = "CRITICAL"

    LEVELS = {
        LEVEL_DEBUG: 10,
        LEVEL_INFO: 20,
        LEVEL_EVENT: 25,
        LEVEL_WARNING: 30,
        LEVEL_ERROR: 40,
        LEVEL_CRITICAL: 50,
    }

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(
        self,
        debug=False,
        max_entries=10000,
        minimum_level="DEBUG",
    ):

        self.debug = bool(debug)

        self.max_entries = max(
            int(max_entries),
            100
        )

        self.minimum_level = (
            minimum_level.upper()
        )

        if self.minimum_level not in self.LEVELS:
            self.minimum_level = (
                self.LEVEL_DEBUG
            )

        self.entries = []

        self.started_at = datetime.now()

        self.session_id = (
            self.started_at.strftime(
                "%Y%m%d%H%M%S"
            )
        )

        self.statistics = {
            level: 0
            for level in self.LEVELS
        }

        self.components = Counter()
        self.events = Counter()

    # ==========================================
    # INTERNAL HELPERS
    # ==========================================

    def _timestamp(self):
        return datetime.now().isoformat()

    def _level_allowed(self, level):

        level = level.upper()

        current_value = self.LEVELS.get(
            level,
            20
        )

        minimum_value = self.LEVELS.get(
            self.minimum_level,
            10
        )

        return current_value >= minimum_value

    def _create_entry(
        self,
        level,
        message,
        component=None,
        event=None,
        metadata=None,
        exception=None,
    ):

        entry = {
            "id": len(self.entries) + 1,
            "time": self._timestamp(),
            "session": self.session_id,
            "level": level,
            "message": str(message),
            "component": component,
            "event": event,
            "metadata": (
                metadata.copy()
                if isinstance(metadata, dict)
                else {}
            ),
        }

        if exception is not None:

            entry["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": (
                    traceback.format_exc()
                ),
            }

        return entry

    def _store(self, entry):

        self.entries.append(entry)

        level = entry["level"]

        if level in self.statistics:
            self.statistics[level] += 1

        component = entry.get("component")

        if component:
            self.components[component] += 1

        event = entry.get("event")

        if event:
            self.events[event] += 1

        self._trim()

        return entry

    def _trim(self):

        if len(self.entries) <= self.max_entries:
            return

        excess = (
            len(self.entries)
            - self.max_entries
        )

        del self.entries[:excess]

    def _output(self, entry):

        if not self.debug:
            return

        level = entry["level"]
        message = entry["message"]
        component = entry.get("component")

        prefix = f"[{level}]"

        if component:
            prefix += f" [{component}]"

        print(
            f"{prefix} {message}"
        )

    def _write(
        self,
        level,
        message,
        component=None,
        event=None,
        metadata=None,
        exception=None,
    ):

        level = level.upper()

        if level not in self.LEVELS:
            level = self.LEVEL_INFO

        if not self._level_allowed(level):
            return None

        entry = self._create_entry(
            level=level,
            message=message,
            component=component,
            event=event,
            metadata=metadata,
            exception=exception,
        )

        self._store(entry)
        self._output(entry)

        return entry

    # ==========================================
    # STANDARD LOGGING
    # ==========================================

    def log(
        self,
        message,
        component=None,
        metadata=None,
    ):

        return self._write(
            self.LEVEL_INFO,
            message,
            component=component,
            metadata=metadata,
        )

    info = log

    # ==========================================
    # DEBUG
    # ==========================================

    def debug_log(
        self,
        message,
        component=None,
        metadata=None,
    ):

        return self._write(
            self.LEVEL_DEBUG,
            message,
            component=component,
            metadata=metadata,
        )

    # ==========================================
    # WARNING
    # ==========================================

    def warning(
        self,
        message,
        component=None,
        metadata=None,
    ):

        return self._write(
            self.LEVEL_WARNING,
            message,
            component=component,
            metadata=metadata,
        )

    warn = warning

    # ==========================================
    # ERROR
    # ==========================================

    def error(
        self,
        error,
        component=None,
        metadata=None,
    ):

        return self._write(
            self.LEVEL_ERROR,
            str(error),
            component=component,
            metadata=metadata,
            exception=error,
        )

    # ==========================================
    # CRITICAL
    # ==========================================

    def critical(
        self,
        message,
        component=None,
        metadata=None,
    ):

        return self._write(
            self.LEVEL_CRITICAL,
            message,
            component=component,
            metadata=metadata,
        )

    # ==========================================
    # EXCEPTION
    # ==========================================

    def exception(
        self,
        error,
        component=None,
        metadata=None,
    ):

        return self.error(
            error,
            component=component,
            metadata=metadata,
        )

    # ==========================================
    # EVENT SYSTEM
    # ==========================================

    def event(
        self,
        event_name,
        message=None,
        component=None,
        metadata=None,
    ):

        if message is None:

            message = (
                f"Event emitted: "
                f"{event_name}"
            )

        return self._write(
            self.LEVEL_EVENT,
            message,
            component=component,
            event=event_name,
            metadata=metadata,
        )

    def record_event(
        self,
        event_name,
        component=None,
        metadata=None,
    ):

        return self.event(
            event_name=event_name,
            component=component,
            metadata=metadata,
        )

    # ==========================================
    # COMPONENT LOGGING
    # ==========================================

    def component(
        self,
        component,
        message,
        level=LEVEL_INFO,
        metadata=None,
    ):

        return self._write(
            level,
            message,
            component=component,
            metadata=metadata,
        )

    def component_started(
        self,
        component,
        metadata=None,
    ):

        return self.event(
            "component_started",
            component=component,
            metadata=metadata,
        )

    def component_stopped(
        self,
        component,
        metadata=None,
    ):

        return self.event(
            "component_stopped",
            component=component,
            metadata=metadata,
        )

    def component_error(
        self,
        component,
        error,
        metadata=None,
    ):

        return self.error(
            error,
            component=component,
            metadata=metadata,
        )

    # ==========================================
    # RETRIEVAL
    # ==========================================

    def get_all(self):

        return self.entries.copy()

    def get_logs(self):

        return self.filter(
            level=self.LEVEL_INFO
        )

    def get_warnings(self):

        return self.filter(
            level=self.LEVEL_WARNING
        )

    def get_errors(self):

        return self.filter(
            level=self.LEVEL_ERROR
        )

    def get_critical(self):

        return self.filter(
            level=self.LEVEL_CRITICAL
        )

    def get_debug_logs(self):

        return self.filter(
            level=self.LEVEL_DEBUG
        )

    def get_events(self):

        return self.filter(
            level=self.LEVEL_EVENT
        )

    # ==========================================
    # RECENT ACTIVITY
    # ==========================================

    def recent(self, limit=25):

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 25

        if limit <= 0:
            return []

        return self.entries[-limit:]

    def latest(self):

        if not self.entries:
            return None

        return self.entries[-1]

    # ==========================================
    # FILTERING
    # ==========================================

    def filter(
        self,
        level=None,
        component=None,
        event=None,
        session=None,
    ):

        results = []

        if level:
            level = level.upper()

        for entry in self.entries:

            if (
                level
                and entry["level"] != level
            ):
                continue

            if (
                component
                and entry.get("component")
                != component
            ):
                continue

            if (
                event
                and entry.get("event")
                != event
            ):
                continue

            if (
                session
                and entry.get("session")
                != session
            ):
                continue

            results.append(entry)

        return results

    # ==========================================
    # SEARCH
    # ==========================================

    def search(
        self,
        query,
        level=None,
        component=None,
        event=None,
    ):

        if not query:
            return []

        query = str(query).lower()

        candidates = self.filter(
            level=level,
            component=component,
            event=event,
        )

        results = []

        for entry in candidates:

            message = (
                entry.get("message", "")
                .lower()
            )

            if query in message:
                results.append(entry)
                continue

            metadata = entry.get(
                "metadata",
                {}
            )

            metadata_text = (
                json.dumps(
                    metadata,
                    default=str
                ).lower()
            )

            if query in metadata_text:
                results.append(entry)

        return results

    # ==========================================
    # LEVEL COUNTS
    # ==========================================

    def count(self, level=None):

        if level is None:
            return len(self.entries)

        return self.statistics.get(
            level.upper(),
            0
        )

    def count_logs(self):
        return self.count(self.LEVEL_INFO)

    def count_warnings(self):
        return self.count(self.LEVEL_WARNING)

    def count_errors(self):
        return self.count(self.LEVEL_ERROR)

    def count_debug(self):
        return self.count(self.LEVEL_DEBUG)

    def count_events(self):
        return self.count(self.LEVEL_EVENT)

    def count_critical(self):
        return self.count(self.LEVEL_CRITICAL)

    # ==========================================
    # COMPONENT STATISTICS
    # ==========================================

    def component_statistics(self):

        result = {}

        for entry in self.entries:

            component = (
                entry.get("component")
                or "system"
            )

            if component not in result:

                result[component] = {
                    "total": 0,
                    "info": 0,
                    "warnings": 0,
                    "errors": 0,
                    "critical": 0,
                    "debug": 0,
                    "events": 0,
                }

            result[component]["total"] += 1

            level = entry["level"]

            if level == self.LEVEL_INFO:
                result[component]["info"] += 1

            elif level == self.LEVEL_WARNING:
                result[component]["warnings"] += 1

            elif level == self.LEVEL_ERROR:
                result[component]["errors"] += 1

            elif level == self.LEVEL_CRITICAL:
                result[component]["critical"] += 1

            elif level == self.LEVEL_DEBUG:
                result[component]["debug"] += 1

            elif level == self.LEVEL_EVENT:
                result[component]["events"] += 1

        return result

    # ==========================================
    # EVENT STATISTICS
    # ==========================================

    def event_statistics(self):

        return dict(self.events)

    # ==========================================
    # GLOBAL STATISTICS
    # ==========================================

    def statistics_report(self):

        return {
            "total": len(self.entries),
            "levels": (
                self.statistics.copy()
            ),
            "components": (
                self.component_statistics()
            ),
            "events": (
                self.event_statistics()
            ),
        }

    # ==========================================
    # SESSION INFORMATION
    # ==========================================

    def get_session(self):

        return {
            "id": self.session_id,
            "started_at": (
                self.started_at.isoformat()
            ),
            "entries": len(self.entries),
        }

    # ==========================================
    # RETENTION
    # ==========================================

    def set_max_entries(self, amount):

        try:
            amount = int(amount)
        except (TypeError, ValueError):
            raise ValueError(
                "Maximum entries must "
                "be an integer."
            )

        if amount < 100:

            raise ValueError(
                "Maximum entries must "
                "be at least 100."
            )

        self.max_entries = amount

        self._trim()

    # ==========================================
    # LEVEL CONFIGURATION
    # ==========================================

    def set_minimum_level(self, level):

        level = str(level).upper()

        if level not in self.LEVELS:

            raise ValueError(
                f"Unknown log level: {level}"
            )

        self.minimum_level = level

    def get_minimum_level(self):

        return self.minimum_level

    # ==========================================
    # DEBUG CONFIGURATION
    # ==========================================

    def set_debug(self, enabled):

        self.debug = bool(enabled)

        self.log(
            f"Debug mode changed to "
            f"{self.debug}"
        )

    # ==========================================
    # CLEARING
    # ==========================================

    def clear(self):

        self.entries.clear()
        self.statistics = {
            level: 0
            for level in self.LEVELS
        }

        self.components.clear()
        self.events.clear()

    def clear_component(self, component):

        remaining = [
            entry
            for entry in self.entries
            if entry.get("component")
            != component
        ]

        self.entries = remaining

        self._rebuild_statistics()

    def clear_level(self, level):

        level = str(level).upper()

        self.entries = [
            entry
            for entry in self.entries
            if entry["level"] != level
        ]

        self._rebuild_statistics()

    def _rebuild_statistics(self):

        self.statistics = {
            level: 0
            for level in self.LEVELS
        }

        self.components.clear()
        self.events.clear()

        for entry in self.entries:

            level = entry["level"]

            if level in self.statistics:
                self.statistics[level] += 1

            component = entry.get(
                "component"
            )

            if component:
                self.components[component] += 1

            event = entry.get("event")

            if event:
                self.events[event] += 1

    # ==========================================
    # EXPORT
    # ==========================================

    def export(self):

        return [
            entry.copy()
            for entry in self.entries
        ]

    def export_json(self, indent=2):

        return json.dumps(
            self.entries,
            indent=indent,
            default=str,
        )

    # ==========================================
    # DIAGNOSTICS
    # ==========================================

    def diagnostics(self):

        latest = self.latest()

        return {
            "session": self.get_session(),
            "configuration": {
                "debug": self.debug,
                "minimum_level": (
                    self.minimum_level
                ),
                "max_entries": (
                    self.max_entries
                ),
            },
            "statistics": (
                self.statistics_report()
            ),
            "latest": latest,
        }

    # ==========================================
    # STATUS
    # ==========================================

    def status(self):

        return {
            "active": True,
            "debug": self.debug,
            "session": self.session_id,
            "started_at": (
                self.started_at.isoformat()
            ),
            "minimum_level": (
                self.minimum_level
            ),
            "max_entries": (
                self.max_entries
            ),
            "entries": len(self.entries),
            "statistics": (
                self.statistics.copy()
            ),
            "components": (
                dict(self.components)
            ),
            "events": (
                dict(self.events)
            ),
        }


# ==========================================
# GLOBAL LOGGER
# ==========================================

logger = Logger(
    debug=False,
    max_entries=10000,
)