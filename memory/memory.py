"""
OurPlatform Advanced Memory Engine

Public-facing memory layer.

Architecture:

    Application
        ↓
    Memory
        ↓
    MemoryService
        ↓
    Storage
        ↓
    Persistent data

The Memory class is intentionally richer than the storage layer.
Storage is responsible for persistence.
Memory is responsible for memory-oriented behaviour.
"""

from datetime import datetime, timedelta
from copy import deepcopy
import re


class Memory:
    """
    Advanced memory management engine.

    Features:

        - Persistent memory
        - Backwards-compatible API
        - Ranked recall
        - Relevance scoring
        - Importance scoring
        - Category filtering
        - Tag filtering
        - Metadata filtering
        - Time-based filtering
        - Bulk creation
        - Bulk updates
        - Bulk deletion
        - Archiving
        - Restoration
        - Pinning
        - Duplicate detection
        - Context generation
        - Memory summaries
        - Statistics
        - Maintenance
        - Backups
        - Import/export
        - Health monitoring
        - Memory lifecycle management
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        storage=None,
        service=None,
        events=None,
        logger=None,
    ):

        if service is None:

            if storage is None:

                from memory.storage import Storage

                storage = Storage(
                    path="data/memory.json",
                    autosave=True,
                )

            from memory.memory_service import (
                MemoryService
            )

            service = MemoryService(
                storage=storage,
                events=events,
                logger=logger,
            )

        self.service = service
        self.storage = service.storage

        self.events = events
        self.logger = logger

        self.created_at = datetime.now()

        self.metrics = {
            "remember_calls": 0,
            "recall_calls": 0,
            "search_calls": 0,
            "update_calls": 0,
            "delete_calls": 0,
            "archive_calls": 0,
            "restore_calls": 0,
            "pin_calls": 0,
            "bulk_operations": 0,
            "context_builds": 0,
        }

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def _now(self):
        return datetime.now()

    def _parse_time(self, value):

        if not value:
            return None

        if isinstance(value, datetime):
            return value

        try:

            return datetime.fromisoformat(
                str(value)
            )

        except (ValueError, TypeError):

            return None

    def _emit(
        self,
        event_name,
        data=None,
    ):

        if self.events is None:
            return

        try:

            self.events.emit(
                event_name,
                data,
            )

        except Exception as error:

            self._log(
                f"Memory event error: {error}"
            )

    def _log(
        self,
        message,
    ):

        if self.logger is None:
            return

        try:

            if hasattr(
                self.logger,
                "log",
            ):

                self.logger.log(
                    message
                )

        except Exception:
            pass

    def _public(
        self,
        record,
    ):

        if record is None:
            return None

        metadata = deepcopy(
            record.get(
                "metadata",
                {},
            )
        )

        return {
            "id": record.get("id"),

            "information": record.get(
                "content",
                "",
            ),

            "content": record.get(
                "content",
                "",
            ),

            "title": record.get(
                "title"
            ),

            "category": record.get(
                "category",
                "general",
            ),

            "importance": metadata.get(
                "importance",
                1,
            ),

            "tags": list(
                record.get(
                    "tags",
                    [],
                )
            ),

            "metadata": metadata,

            "created_at": record.get(
                "created_at"
            ),

            "updated_at": record.get(
                "updated_at"
            ),

            "accessed_at": record.get(
                "accessed_at"
            ),

            "access_count": record.get(
                "access_count",
                0,
            ),

            "version": record.get(
                "version",
                1,
            ),

            "archived": record.get(
                "archived",
                False,
            ),

            "pinned": record.get(
                "pinned",
                False,
            ),
        }

    def _public_many(
        self,
        records,
    ):

        return [
            self._public(record)
            for record in records
        ]

    def _words(
        self,
        text,
    ):

        if not text:
            return set()

        return set(
            re.findall(
                r"\b[a-zA-Z0-9_]+\b",
                str(text).lower(),
            )
        )

    def _score(
        self,
        record,
        query,
    ):

        if not query:
            return 0.0

        query_words = self._words(
            query
        )

        content_words = self._words(
            record.get(
                "content",
                "",
            )
        )

        if not query_words:
            return 0.0

        overlap = (
            query_words
            & content_words
        )

        relevance = (
            len(overlap)
            / len(query_words)
        )

        importance = record.get(
            "metadata",
            {},
        ).get(
            "importance",
            1,
        )

        importance_bonus = (
            min(
                float(importance),
                10.0,
            )
            / 100
        )

        pinned_bonus = (
            0.10
            if record.get(
                "pinned",
                False,
            )
            else 0
        )

        return (
            relevance
            + importance_bonus
            + pinned_bonus
        )

    # =========================================================
    # REMEMBER
    # =========================================================

    def remember(
        self,
        information,
        category="general",
        importance=1,
        title=None,
        tags=None,
        metadata=None,
        pinned=False,
    ):

        self.metrics[
            "remember_calls"
        ] += 1

        metadata = deepcopy(
            metadata or {}
        )

        metadata[
            "importance"
        ] = importance

        record = self.service.create(
            content=information,
            title=title,
            category=category,
            tags=tags,
            metadata=metadata,
            pinned=pinned,
        )

        result = self._public(
            record
        )

        self._emit(
            "memory_created",
            result,
        )

        return result

    # =========================================================
    # BULK REMEMBER
    # =========================================================

    def remember_many(
        self,
        memories,
        category="general",
        importance=1,
    ):

        self.metrics[
            "bulk_operations"
        ] += 1

        results = []

        for item in memories:

            if isinstance(
                item,
                str,
            ):

                results.append(
                    self.remember(
                        item,
                        category=category,
                        importance=importance,
                    )
                )

                continue

            if not isinstance(
                item,
                dict,
            ):

                continue

            data = dict(item)

            results.append(
                self.remember(
                    information=data.get(
                        "information",
                        data.get(
                            "content",
                            "",
                        ),
                    ),
                    category=data.get(
                        "category",
                        category,
                    ),
                    importance=data.get(
                        "importance",
                        importance,
                    ),
                    title=data.get(
                        "title"
                    ),
                    tags=data.get(
                        "tags"
                    ),
                    metadata=data.get(
                        "metadata"
                    ),
                    pinned=data.get(
                        "pinned",
                        False,
                    ),
                )
            )

        return results

    # =========================================================
    # RECALL
    # =========================================================

    def recall(
        self,
        include_archived=False,
        category=None,
        tags=None,
        pinned=None,
        limit=None,
        offset=0,
        sort_by="updated_at",
        descending=True,
    ):

        self.metrics[
            "recall_calls"
        ] += 1

        records = self.service.list(
            category=category,
            include_archived=(
                include_archived
            ),
            pinned=pinned,
            limit=None,
            offset=0,
            sort_by=sort_by,
            descending=descending,
        )

        if tags:

            required = set(
                tags
                if isinstance(
                    tags,
                    (list, tuple, set),
                )
                else [tags]
            )

            records = [
                record
                for record in records
                if required.issubset(
                    set(
                        record.get(
                            "tags",
                            [],
                        )
                    )
                )
            ]

        if offset:

            records = records[
                offset:
            ]

        if limit is not None:

            records = records[
                :limit
            ]

        return self._public_many(
            records
        )

    # =========================================================
    # GET
    # =========================================================

    def get(
        self,
        memory_id,
    ):

        record = (
            self.service.get_optional(
                memory_id
            )
        )

        return self._public(
            record
        )

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        memory_id,
        information=None,
        category=None,
        importance=None,
        title=None,
        tags=None,
        metadata=None,
        pinned=None,
        archived=None,
    ):

        self.metrics[
            "update_calls"
        ] += 1

        current = self.get(
            memory_id
        )

        if current is None:
            return None

        changes = {}

        if information is not None:

            changes[
                "content"
            ] = information

        if category is not None:

            changes[
                "category"
            ] = category

        if title is not None:

            changes[
                "title"
            ] = title

        if tags is not None:

            changes[
                "tags"
            ] = tags

        if metadata is not None:

            changes[
                "metadata"
            ] = deepcopy(
                metadata
            )

        if importance is not None:

            new_metadata = deepcopy(
                current.get(
                    "metadata",
                    {},
                )
            )

            new_metadata[
                "importance"
            ] = importance

            changes[
                "metadata"
            ] = new_metadata

        if pinned is not None:

            changes[
                "pinned"
            ] = pinned

        if archived is not None:

            changes[
                "archived"
            ] = archived

        if not changes:

            return current

        record = self.service.update(
            memory_id,
            **changes,
        )

        result = self._public(
            record
        )

        self._emit(
            "memory_updated",
            result,
        )

        return result

    # =========================================================
    # BULK UPDATE
    # =========================================================

    def update_many(
        self,
        memory_ids,
        **changes,
    ):

        self.metrics[
            "bulk_operations"
        ] += 1

        results = []

        for memory_id in memory_ids:

            result = self.update(
                memory_id,
                **changes,
            )

            if result is not None:

                results.append(
                    result
                )

        return results

    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        query,
        category=None,
        tags=None,
        include_archived=False,
        limit=None,
    ):

        self.metrics[
            "search_calls"
        ] += 1

        records = self.service.search(
            query=query,
            category=category,
            tags=tags,
            include_archived=(
                include_archived
            ),
        )

        return self._public_many(
            records[:limit]
            if limit is not None
            else records
        )

    # =========================================================
    # RANKED SEARCH
    # =========================================================

    def ranked_search(
        self,
        query,
        category=None,
        tags=None,
        include_archived=False,
        limit=10,
    ):

        self.metrics[
            "search_calls"
        ] += 1

        records = self.service.search(
            query=query,
            category=category,
            tags=tags,
            include_archived=(
                include_archived
            ),
        )

        ranked = []

        for record in records:

            score = self._score(
                record,
                query,
            )

            if score <= 0:
                continue

            item = self._public(
                record
            )

            item[
                "relevance_score"
            ] = round(
                score,
                4,
            )

            ranked.append(
                item
            )

        ranked.sort(
            key=lambda item: (
                item[
                    "relevance_score"
                ],
                item[
                    "importance"
                ],
                item.get(
                    "pinned",
                    False,
                ),
            ),
            reverse=True,
        )

        return ranked[
            :limit
        ]

    # =========================================================
    # CONTEXT SEARCH
    # =========================================================

    def context(
        self,
        query,
        limit=5,
        max_characters=8000,
    ):

        self.metrics[
            "context_builds"
        ] += 1

        memories = self.ranked_search(
            query,
            limit=limit,
        )

        selected = []

        size = 0

        for memory in memories:

            text = memory[
                "information"
            ]

            if (
                size + len(text)
                > max_characters
            ):

                break

            selected.append(
                memory
            )

            size += len(text)

        return {
            "query": query,
            "count": len(
                selected
            ),
            "memories": selected,
            "context": "\n\n".join(
                memory[
                    "information"
                ]
                for memory in selected
            ),
        }

    # =========================================================
    # CATEGORY
    # =========================================================

    def by_category(
        self,
        category,
    ):

        return self.recall(
            category=category
        )

    def categories(self):

        return self.service.categories()

    def count_category(
        self,
        category,
    ):

        return self.service.count_category(
            category
        )

    # =========================================================
    # TAGS
    # =========================================================

    def tags(self):

        return self.service.tags()

    def by_tag(
        self,
        tag,
    ):

        return self._public_many(
            self.service.records_by_tag(
                tag
            )
        )

    # =========================================================
    # IMPORTANCE
    # =========================================================

    def important(
        self,
        minimum=3,
        limit=None,
    ):

        records = self.recall()

        results = [
            memory
            for memory in records
            if memory[
                "importance"
            ] >= minimum
        ]

        results.sort(
            key=lambda item: (
                item[
                    "importance"
                ],
                item.get(
                    "pinned",
                    False,
                ),
            ),
            reverse=True,
        )

        if limit is not None:

            results = results[
                :limit
            ]

        return results

    # =========================================================
    # RECENT
    # =========================================================

    def recent(
        self,
        days=7,
        limit=None,
    ):

        cutoff = (
            self._now()
            - timedelta(
                days=days
            )
        )

        results = []

        for memory in self.recall():

            created = self._parse_time(
                memory[
                    "created_at"
                ]
            )

            if (
                created
                and created >= cutoff
            ):

                results.append(
                    memory
                )

        results.sort(
            key=lambda item: (
                item[
                    "created_at"
                ]
            ),
            reverse=True,
        )

        if limit is not None:

            results = results[
                :limit
            ]

        return results

    # =========================================================
    # PINNED
    # =========================================================

    def pinned(
        self,
        limit=None,
    ):

        return self.recall(
            pinned=True,
            limit=limit,
        )

    # =========================================================
    # ARCHIVED
    # =========================================================

    def archived(
        self,
        limit=None,
    ):

        return self.recall(
            include_archived=True,
            limit=limit,
        )

    # =========================================================
    # ARCHIVE
    # =========================================================

    def archive(
        self,
        memory_id,
    ):

        self.metrics[
            "archive_calls"
        ] += 1

        result = self.service.archive(
            memory_id
        )

        public = self._public(
            result
        )

        self._emit(
            "memory_archived",
            public,
        )

        return public

    # =========================================================
    # RESTORE
    # =========================================================

    def restore(
        self,
        memory_id,
    ):

        self.metrics[
            "restore_calls"
        ] += 1

        result = self.service.restore(
            memory_id
        )

        public = self._public(
            result
        )

        self._emit(
            "memory_restored",
            public,
        )

        return public

    # =========================================================
    # PIN / UNPIN
    # =========================================================

    def pin(
        self,
        memory_id,
    ):

        self.metrics[
            "pin_calls"
        ] += 1

        return self._public(
            self.service.pin(
                memory_id
            )
        )

    def unpin(
        self,
        memory_id,
    ):

        return self._public(
            self.service.unpin(
                memory_id
            )
        )

    # =========================================================
    # FORGET
    # =========================================================

    def forget(
        self,
        memory_id,
    ):

        self.metrics[
            "delete_calls"
        ] += 1

        existing = self.get(
            memory_id
        )

        if existing is None:
            return False

        self.service.delete(
            memory_id
        )

        self._emit(
            "memory_forgotten",
            existing,
        )

        return True

    # =========================================================
    # FORGET MANY
    # =========================================================

    def forget_many(
        self,
        memory_ids,
    ):

        self.metrics[
            "bulk_operations"
        ] += 1

        deleted = 0

        for memory_id in memory_ids:

            if self.forget(
                memory_id
            ):

                deleted += 1

        return deleted

    # =========================================================
    # CLEAR
    # =========================================================

    def clear(
        self,
        include_archived=True,
    ):

        memories = self.recall(
            include_archived=(
                include_archived
            )
        )

        return self.forget_many(
            [
                memory["id"]
                for memory in memories
            ]
        )

    # =========================================================
    # METADATA
    # =========================================================

    def by_metadata(
        self,
        key,
        value,
    ):

        records = (
            self.service.find_by_metadata(
                key,
                value,
            )
        )

        return self._public_many(
            records
        )

    # =========================================================
    # DUPLICATES
    # =========================================================

    def duplicates(
        self,
        information,
    ):

        records = (
            self.service.find_duplicates(
                information
            )
        )

        return self._public_many(
            records
        )

    def is_duplicate(
        self,
        information,
    ):

        return self.service.is_duplicate(
            information
        )

    # =========================================================
    # SNAPSHOT
    # =========================================================

    def snapshot(self):

        return {
            "created_at": (
                self._now().isoformat()
            ),
            "count": self.count(),
            "categories": self.categories(),
            "tags": self.tags(),
            "statistics": (
                self.statistics()
            ),
            "health": (
                self.health_check()
            ),
        }

    # =========================================================
    # BACKUP
    # =========================================================

    def backup(
        self,
        destination=None,
    ):

        return self.service.backup(
            destination
        )

    # =========================================================
    # EXPORT
    # =========================================================

    def export(
        self,
        destination,
        include_archived=True,
    ):

        return self.service.export(
            destination,
            include_archived=(
                include_archived
            ),
        )

    # =========================================================
    # IMPORT
    # =========================================================

    def import_data(
        self,
        source,
        merge=True,
    ):

        return self.service.import_data(
            source,
            merge=merge,
        )

    # =========================================================
    # MAINTENANCE
    # =========================================================

    def purge_archived(self):

        return (
            self.service.purge_archived()
        )

    def cleanup_duplicates(self):

        return (
            self.service.cleanup_duplicates()
        )

    # =========================================================
    # STATISTICS
    # =========================================================

    def count(
        self,
        include_archived=True,
    ):

        return self.service.count(
            include_archived=(
                include_archived
            )
        )

    def statistics(self):

        return {
            "memory": deepcopy(
                self.metrics
            ),
            "storage": (
                self.service.statistics()
            ),
            "total_memories": (
                self.count()
            ),
            "active_memories": (
                self.count(
                    include_archived=False
                )
            ),
            "categories": len(
                self.categories()
            ),
            "tags": len(
                self.tags()
            ),
            "uptime_seconds": (
                self._now()
                - self.created_at
            ).total_seconds(),
        }

    # =========================================================
    # HEALTH
    # =========================================================

    def health_check(self):

        service_health = (
            self.service.health_check()
        )

        return {
            "healthy": (
                service_health.get(
                    "healthy",
                    False,
                )
            ),
            "memory": {
                "initialized": True,
                "created_at": (
                    self.created_at.isoformat()
                ),
            },
            "service": service_health,
        }

    # =========================================================
    # STATUS
    # =========================================================

    def status(self):

        return {
            "active": True,
            "created_at": (
                self.created_at.isoformat()
            ),
            "metrics": deepcopy(
                self.metrics
            ),
            "statistics": (
                self.statistics()
            ),
            "health": (
                self.health_check()
            ),
        }


# =============================================================
# GLOBAL MEMORY INSTANCE
# =============================================================

memory = Memory()