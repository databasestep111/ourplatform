from datetime import datetime
from collections import Counter
import traceback
import uuid


class EventSystem:
    """
    Advanced internal event system for OurPlatform.

    Provides a central way for components to communicate
    without needing to directly depend on each other.

    Features:
        - Event registration
        - Event emission
        - Multiple listeners
        - One-time listeners
        - Listener priorities
        - Listener filtering
        - Event history
        - Event metadata
        - Event statistics
        - Event namespaces
        - Event cancellation
        - Listener error handling
        - Event replay
        - Event search
        - Runtime configuration
        - Diagnostics
    """

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(
        self,
        max_history=10000,
        debug=False,
    ):

        self.max_history = max(
            int(max_history),
            100,
        )

        self.debug = bool(debug)

        self.listeners = {}

        self.history = []

        self.statistics = Counter()

        self.listener_errors = []

        self.started_at = datetime.now()

        self.emitted_count = 0
        self.handled_count = 0
        self.failed_count = 0

    # ==========================================
    # INTERNAL HELPERS
    # ==========================================

    def _timestamp(self):

        return datetime.now().isoformat()

    def _event_id(self):

        return str(
            uuid.uuid4()
        )

    def _normalise_name(self, name):

        if not name:
            raise ValueError(
                "Event name cannot be empty."
            )

        return str(name).strip()

    def _trim_history(self):

        if len(self.history) <= self.max_history:
            return

        excess = (
            len(self.history)
            - self.max_history
        )

        del self.history[:excess]

    # ==========================================
    # EVENT REGISTRATION
    # ==========================================

    def register(
        self,
        event_name,
    ):

        event_name = self._normalise_name(
            event_name
        )

        if event_name not in self.listeners:

            self.listeners[event_name] = []

        return event_name

    # ==========================================
    # LISTENER REGISTRATION
    # ==========================================

    def on(
        self,
        event_name,
        listener,
        priority=0,
        once=False,
        name=None,
        component=None,
    ):

        event_name = self.register(
            event_name
        )

        if not callable(listener):

            raise TypeError(
                "Listener must be callable."
            )

        listener_id = str(
            uuid.uuid4()
        )

        record = {
            "id": listener_id,
            "name": (
                name
                or getattr(
                    listener,
                    "__name__",
                    "anonymous",
                )
            ),
            "listener": listener,
            "priority": int(priority),
            "once": bool(once),
            "component": component,
            "registered_at": (
                self._timestamp()
            ),
            "calls": 0,
            "errors": 0,
            "active": True,
        }

        self.listeners[
            event_name
        ].append(record)

        self.listeners[
            event_name
        ].sort(
            key=lambda item: item["priority"],
            reverse=True,
        )

        return listener_id

    # ==========================================
    # ONE-TIME LISTENERS
    # ==========================================

    def once(
        self,
        event_name,
        listener,
        priority=0,
        name=None,
        component=None,
    ):

        return self.on(
            event_name=event_name,
            listener=listener,
            priority=priority,
            once=True,
            name=name,
            component=component,
        )

    # ==========================================
    # LISTENER REMOVAL
    # ==========================================

    def off(
        self,
        event_name,
        listener_id,
    ):

        event_name = self._normalise_name(
            event_name
        )

        if event_name not in self.listeners:
            return False

        original = self.listeners[
            event_name
        ]

        remaining = [
            listener
            for listener in original
            if listener["id"]
            != listener_id
        ]

        removed = (
            len(remaining)
            != len(original)
        )

        self.listeners[
            event_name
        ] = remaining

        return removed

    def remove_listener(
        self,
        event_name,
        listener_id,
    ):

        return self.off(
            event_name,
            listener_id,
        )

    # ==========================================
    # REMOVE COMPONENT LISTENERS
    # ==========================================

    def remove_component(
        self,
        component,
    ):

        removed = 0

        for event_name in list(
            self.listeners.keys()
        ):

            original = self.listeners[
                event_name
            ]

            remaining = []

            for listener in original:

                if (
                    listener.get("component")
                    == component
                ):

                    removed += 1

                else:

                    remaining.append(
                        listener
                    )

            self.listeners[
                event_name
            ] = remaining

        return removed

    # ==========================================
    # EVENT EMISSION
    # ==========================================

    def emit(
        self,
        event_name,
        data=None,
        source=None,
        metadata=None,
        stop_on_error=False,
    ):

        event_name = self._normalise_name(
            event_name
        )

        event = {
            "id": self._event_id(),
            "name": event_name,
            "time": self._timestamp(),
            "source": source,
            "data": data,
            "metadata": (
                metadata.copy()
                if isinstance(
                    metadata,
                    dict,
                )
                else {}
            ),
            "handled": 0,
            "errors": 0,
            "cancelled": False,
        }

        self.emitted_count += 1

        listeners = list(
            self.listeners.get(
                event_name,
                [],
            )
        )

        for record in listeners:

            if not record["active"]:
                continue

            if event["cancelled"]:
                break

            try:

                result = record[
                    "listener"
                ](
                    data
                )

                record["calls"] += 1

                event["handled"] += 1

                self.handled_count += 1

                if result is False:

                    event[
                        "cancelled"
                    ] = True

                if record["once"]:

                    self.off(
                        event_name,
                        record["id"],
                    )

            except Exception as error:

                record["errors"] += 1

                event["errors"] += 1

                self.failed_count += 1

                self._record_listener_error(
                    event,
                    record,
                    error,
                )

                if stop_on_error:
                    break

        self.history.append(event)

        self.statistics[
            event_name
        ] += 1

        self._trim_history()

        if self.debug:

            print(
                "[EVENT]",
                event_name,
                event["id"],
            )

        return event

    # ==========================================
    # ASYNC-READY EMISSION INTERFACE
    # ==========================================

    def emit_safe(
        self,
        event_name,
        data=None,
        source=None,
        metadata=None,
    ):

        try:

            return self.emit(
                event_name,
                data=data,
                source=source,
                metadata=metadata,
            )

        except Exception as error:

            self.failed_count += 1

            self._record_system_error(
                error
            )

            return {
                "success": False,
                "error": str(error),
            }

    # ==========================================
    # CANCELLATION
    # ==========================================

    def cancel(
        self,
        event_id,
    ):

        for event in self.history:

            if event["id"] == event_id:

                event["cancelled"] = True

                return True

        return False

    # ==========================================
    # ERROR TRACKING
    # ==========================================

    def _record_listener_error(
        self,
        event,
        listener,
        error,
    ):

        self.listener_errors.append(
            {
                "time": self._timestamp(),
                "type": "listener",
                "event_id": event["id"],
                "event": event["name"],
                "listener": listener["name"],
                "component": (
                    listener.get(
                        "component"
                    )
                ),
                "error": str(error),
                "exception": (
                    type(error).__name__
                ),
                "traceback": (
                    traceback.format_exc()
                ),
            }
        )

    def _record_system_error(
        self,
        error,
    ):

        self.listener_errors.append(
            {
                "time": self._timestamp(),
                "type": "system",
                "error": str(error),
                "exception": (
                    type(error).__name__
                ),
                "traceback": (
                    traceback.format_exc()
                ),
            }
        )

    # ==========================================
    # EVENT HISTORY
    # ==========================================

    def get_history(
        self,
        limit=None,
    ):

        if limit is None:

            return self.history.copy()

        try:
            limit = int(limit)
        except (
            TypeError,
            ValueError,
        ):
            limit = 50

        if limit <= 0:
            return []

        return self.history[-limit:]

    def latest(self):

        if not self.history:
            return None

        return self.history[-1]

    # ==========================================
    # EVENT SEARCH
    # ==========================================

    def search(
        self,
        query,
        source=None,
    ):

        if not query:
            return []

        query = str(
            query
        ).lower()

        results = []

        for event in self.history:

            name = event[
                "name"
            ].lower()

            source_name = str(
                event.get(
                    "source",
                    "",
                )
            ).lower()

            if (
                query in name
                or query in source_name
            ):

                if (
                    source
                    and event.get(
                        "source"
                    )
                    != source
                ):
                    continue

                results.append(event)

        return results

    # ==========================================
    # FILTERING
    # ==========================================

    def filter(
        self,
        event_name=None,
        source=None,
        cancelled=None,
    ):

        results = []

        for event in self.history:

            if (
                event_name
                and event["name"]
                != event_name
            ):
                continue

            if (
                source
                and event.get("source")
                != source
            ):
                continue

            if (
                cancelled is not None
                and event["cancelled"]
                != cancelled
            ):
                continue

            results.append(event)

        return results

    # ==========================================
    # EVENT STATISTICS
    # ==========================================

    def event_statistics(self):

        return dict(
            self.statistics
        )

    def most_common_events(
        self,
        limit=10,
    ):

        return (
            self.statistics
            .most_common(limit)
        )

    # ==========================================
    # LISTENER INFORMATION
    # ==========================================

    def listeners_for(
        self,
        event_name,
    ):

        event_name = self._normalise_name(
            event_name
        )

        records = self.listeners.get(
            event_name,
            [],
        )

        result = []

        for record in records:

            result.append(
                {
                    "id": record["id"],
                    "name": record["name"],
                    "priority": (
                        record["priority"]
                    ),
                    "once": record["once"],
                    "component": (
                        record.get(
                            "component"
                        )
                    ),
                    "calls": record["calls"],
                    "errors": record["errors"],
                    "active": record["active"],
                    "registered_at": (
                        record[
                            "registered_at"
                        ]
                    ),
                }
            )

        return result

    def list_events(self):

        return list(
            self.listeners.keys()
        )

    def list_listeners(self):

        result = {}

        for event_name in self.listeners:

            result[event_name] = (
                self.listeners_for(
                    event_name
                )
            )

        return result

    # ==========================================
    # ENABLE / DISABLE LISTENERS
    # ==========================================

    def enable_listener(
        self,
        event_name,
        listener_id,
    ):

        for listener in self.listeners.get(
            event_name,
            [],
        ):

            if listener["id"] == listener_id:

                listener["active"] = True

                return True

        return False

    def disable_listener(
        self,
        event_name,
        listener_id,
    ):

        for listener in self.listeners.get(
            event_name,
            [],
        ):

            if listener["id"] == listener_id:

                listener["active"] = False

                return True

        return False

    # ==========================================
    # REPLAY
    # ==========================================

    def replay(
        self,
        event_id,
        include_metadata=True,
    ):

        original = None

        for event in self.history:

            if event["id"] == event_id:

                original = event

                break

        if original is None:
            return None

        metadata = {}

        if include_metadata:

            metadata = (
                original.get(
                    "metadata",
                    {}
                ).copy()
            )

        return self.emit(
            original["name"],
            data=original.get(
                "data"
            ),
            source=(
                original.get(
                    "source"
                )
            ),
            metadata=metadata,
        )

    # ==========================================
    # RETENTION
    # ==========================================

    def set_max_history(
        self,
        amount,
    ):

        try:

            amount = int(amount)

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                "History limit must "
                "be an integer."
            )

        if amount < 100:

            raise ValueError(
                "History limit must "
                "be at least 100."
            )

        self.max_history = amount

        self._trim_history()

    # ==========================================
    # CLEARING
    # ==========================================

    def clear_history(self):

        self.history.clear()

        self.statistics.clear()

        self.emitted_count = 0
        self.handled_count = 0
        self.failed_count = 0

    def clear_errors(self):

        self.listener_errors.clear()

        for event_name in self.listeners:

            for listener in self.listeners[
                event_name
            ]:

                listener["errors"] = 0

    # ==========================================
    # DIAGNOSTICS
    # ==========================================

    def diagnostics(self):

        return {
            "started_at": (
                self.started_at.isoformat()
            ),
            "configuration": {
                "debug": self.debug,
                "max_history": (
                    self.max_history
                ),
            },
            "events": {
                "emitted": (
                    self.emitted_count
                ),
                "handled": (
                    self.handled_count
                ),
                "failed": (
                    self.failed_count
                ),
                "history": len(
                    self.history
                ),
            },
            "registered_events": (
                self.list_events()
            ),
            "listeners": (
                self.list_listeners()
            ),
            "statistics": (
                self.event_statistics()
            ),
            "listener_errors": len(
                self.listener_errors
            ),
        }

    # ==========================================
    # STATUS
    # ==========================================

    def status(self):

        total_listeners = 0

        for event_name in self.listeners:

            total_listeners += len(
                self.listeners[
                    event_name
                ]
            )

        return {
            "active": True,
            "debug": self.debug,
            "events": len(
                self.listeners
            ),
            "listeners": (
                total_listeners
            ),
            "history": len(
                self.history
            ),
            "emitted": (
                self.emitted_count
            ),
            "handled": (
                self.handled_count
            ),
            "failed": (
                self.failed_count
            ),
            "errors": len(
                self.listener_errors
            ),
        }


# ==========================================
# GLOBAL EVENT SYSTEM
# ==========================================

events = EventSystem(
    max_history=10000,
    debug=False,
)