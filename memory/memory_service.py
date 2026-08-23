"""
OurPlatform Memory Service

Integration layer between the public Memory API and the
underlying Storage engine.

The service is deliberately kept separate from both
memory.py and storage.py so that the architecture can
grow without tightly coupling the systems.
"""

from datetime import datetime
from copy import deepcopy


class MemoryService:
    """
    High-level memory management service.

    Responsibilities:

        - Coordinate Memory and Storage
        - Create memories
        - Retrieve memories
        - Update memories
        - Delete memories
        - Search memories
        - Manage tags
        - Manage categories
        - Manage metadata
        - Archive / restore
        - Pin / unpin
        - Track statistics
        - Provide health information
        - Emit optional events
        - Write optional logs

    The service does not directly manage files.
    That responsibility belongs to Storage.
    """

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(
        self,
        storage,
        events=None,
        logger=None,
    ):

        if storage is None:

            raise ValueError(
                "Storage instance is required."
            )

        self.storage = storage
        self.events = events
        self.logger = logger

        self.started_at = datetime.now()

        self.operations = {
            "created": 0,
            "retrieved": 0,
            "updated": 0,
            "deleted": 0,
            "searched": 0,
            "archived": 0,
            "restored": 0,
            "pinned": 0,
            "unpinned": 0,
        }

    # ==========================================
    # INTERNAL LOGGING
    # ==========================================

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

    # ==========================================
    # INTERNAL EVENTS
    # ==========================================

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
                data=data,
                source="memory_service",
            )

        except Exception as error:

            self._log(
                "Memory event failed: "
                f"{error}"
            )

    # ==========================================
    # CREATE
    # ==========================================

    def create(
        self,
        content,
        title=None,
        category=None,
        tags=None,
        metadata=None,
        pinned=False,
    ):

        record = self.storage.create(
            content=content,
            title=title,
            category=category,
            tags=tags,
            metadata=metadata,
            pinned=pinned,
        )

        self.operations[
            "created"
        ] += 1

        self._log(
            f"Memory created: "
            f"{record['id']}"
        )

        self._emit(
            "memory_created",
            record,
        )

        return record

    # ==========================================
    # RETRIEVE
    # ==========================================

    def get(
        self,
        memory_id,
    ):

        record = self.storage.get(
            memory_id
        )

        self.operations[
            "retrieved"
        ] += 1

        self._emit(
            "memory_retrieved",
            {
                "id": memory_id,
            },
        )

        return record

    # ==========================================
    # OPTIONAL RETRIEVAL
    # ==========================================

    def get_optional(
        self,
        memory_id,
    ):

        record = self.storage.get_optional(
            memory_id
        )

        if record is not None:

            self.operations[
                "retrieved"
            ] += 1

        return record

    # ==========================================
    # UPDATE
    # ==========================================

    def update(
        self,
        memory_id,
        **changes,
    ):

        record = self.storage.update(
            memory_id,
            **changes,
        )

        self.operations[
            "updated"
        ] += 1

        self._log(
            f"Memory updated: "
            f"{memory_id}"
        )

        self._emit(
            "memory_updated",
            record,
        )

        return record

    # ==========================================
    # DELETE
    # ==========================================

    def delete(
        self,
        memory_id,
    ):

        record = self.storage.delete(
            memory_id
        )

        self.operations[
            "deleted"
        ] += 1

        self._log(
            f"Memory deleted: "
            f"{memory_id}"
        )

        self._emit(
            "memory_deleted",
            record,
        )

        return record

    # ==========================================
    # SEARCH
    # ==========================================

    def search(
        self,
        query,
        category=None,
        tags=None,
        include_archived=False,
    ):

        self.operations[
            "searched"
        ] += 1

        results = self.storage.search(
            query=query,
            category=category,
            tags=tags,
            include_archived=(
                include_archived
            ),
        )

        self._emit(
            "memory_search",
            {
                "query": query,
                "results": len(
                    results
                ),
            },
        )

        return results

    # ==========================================
    # LIST
    # ==========================================

    def list(
        self,
        category=None,
        include_archived=False,
        pinned=None,
        limit=None,
        offset=0,
        sort_by="updated_at",
        descending=True,
    ):

        return self.storage.list(
            category=category,
            include_archived=(
                include_archived
            ),
            pinned=pinned,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            descending=descending,
        )

    # ==========================================
    # COUNT
    # ==========================================

    def count(
        self,
        include_archived=True,
    ):

        return self.storage.count(
            include_archived=(
                include_archived
            )
        )

    # ==========================================
    # CATEGORIES
    # ==========================================

    def categories(self):

        return self.storage.categories()

    def count_category(
        self,
        category,
    ):

        return self.storage.count_category(
            category
        )

    # ==========================================
    # TAGS
    # ==========================================

    def tags(self):

        return self.storage.tags()

    def records_by_tag(
        self,
        tag,
    ):

        return self.storage.records_by_tag(
            tag
        )

    # ==========================================
    # METADATA
    # ==========================================

    def find_by_metadata(
        self,
        key,
        value,
    ):

        return self.storage.find_by_metadata(
            key,
            value,
        )

    # ==========================================
    # DUPLICATES
    # ==========================================

    def find_duplicates(
        self,
        content,
    ):

        return self.storage.find_duplicates(
            content
        )

    def is_duplicate(
        self,
        content,
    ):

        return bool(
            self.find_duplicates(
                content
            )
        )

    # ==========================================
    # ARCHIVE
    # ==========================================

    def archive(
        self,
        memory_id,
    ):

        record = self.storage.archive(
            memory_id
        )

        self.operations[
            "archived"
        ] += 1

        self._log(
            f"Memory archived: "
            f"{memory_id}"
        )

        self._emit(
            "memory_archived",
            record,
        )

        return record

    # ==========================================
    # RESTORE
    # ==========================================

    def restore(
        self,
        memory_id,
    ):

        record = self.storage.unarchive(
            memory_id
        )

        self.operations[
            "restored"
        ] += 1

        self._log(
            f"Memory restored: "
            f"{memory_id}"
        )

        self._emit(
            "memory_restored",
            record,
        )

        return record

    # ==========================================
    # PIN
    # ==========================================

    def pin(
        self,
        memory_id,
    ):

        record = self.storage.pin(
            memory_id
        )

        self.operations[
            "pinned"
        ] += 1

        self._emit(
            "memory_pinned",
            record,
        )

        return record

    # ==========================================
    # UNPIN
    # ==========================================

    def unpin(
        self,
        memory_id,
    ):

        record = self.storage.unpin(
            memory_id
        )

        self.operations[
            "unpinned"
        ] += 1

        self._emit(
            "memory_unpinned",
            record,
        )

        return record

    # ==========================================
    # BACKUP
    # ==========================================

    def backup(
        self,
        destination=None,
    ):

        result = self.storage.backup(
            destination
        )

        self._log(
            f"Memory backup created: "
            f"{result}"
        )

        self._emit(
            "memory_backup_created",
            {
                "path": result,
            },
        )

        return result

    # ==========================================
    # EXPORT
    # ==========================================

    def export(
        self,
        destination,
        include_archived=True,
    ):

        result = self.storage.export(
            destination,
            include_archived=(
                include_archived
            ),
        )

        self._emit(
            "memory_exported",
            {
                "path": result,
            },
        )

        return result

    # ==========================================
    # IMPORT
    # ==========================================

    def import_data(
        self,
        source,
        merge=True,
    ):

        result = self.storage.import_data(
            source,
            merge=merge,
        )

        self._log(
            f"Memory import completed: "
            f"{result} records"
        )

        self._emit(
            "memory_imported",
            {
                "records": result,
            },
        )

        return result

    # ==========================================
    # MAINTENANCE
    # ==========================================

    def purge_archived(self):

        result = (
            self.storage.purge_archived()
        )

        self._emit(
            "memory_archive_purged",
            {
                "deleted": result,
            },
        )

        return result

    def cleanup_duplicates(self):

        result = (
            self.storage.cleanup_duplicates()
        )

        self._emit(
            "memory_duplicates_cleaned",
            {
                "deleted": result,
            },
        )

        return result

    # ==========================================
    # STATISTICS
    # ==========================================

    def statistics(self):

        storage_stats = (
            self.storage.statistics()
        )

        return {
            "service": deepcopy(
                self.operations
            ),
            "storage": storage_stats,
            "total_memories": (
                self.storage.count()
            ),
        }

    # ==========================================
    # HEALTH
    # ==========================================

    def health_check(self):

        storage_health = (
            self.storage.health_check()
        )

        return {
            "healthy": (
                storage_health.get(
                    "healthy",
                    False,
                )
            ),
            "storage": storage_health,
            "service_started": (
                self.started_at.isoformat()
            ),
        }

    # ==========================================
    # STATUS
    # ==========================================

    def status(self):

        return {
            "active": True,
            "started_at": (
                self.started_at.isoformat()
            ),
            "operations": deepcopy(
                self.operations
            ),
            "storage": (
                self.storage.status()
            ),
        }


# ==========================================
# FACTORY
# ==========================================

def create_memory_service(
    storage,
    events=None,
    logger=None,
):

    return MemoryService(
        storage=storage,
        events=events,
        logger=logger,
    )