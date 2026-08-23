"""
OurPlatform Memory Storage Engine

Responsible for storing, retrieving, updating, deleting,
searching, validating, importing, exporting, and maintaining
persistent memory records.

This module intentionally has no dependency on the web layer.
"""

from datetime import datetime
from pathlib import Path
from copy import deepcopy
import json
import uuid
import threading


class StorageError(Exception):
    """Base storage exception."""


class StorageValidationError(StorageError):
    """Raised when a record is invalid."""


class StorageNotFoundError(StorageError):
    """Raised when a requested record does not exist."""


class Storage:
    """
    Persistent JSON-backed storage engine.

    Records support:

        id
        content
        title
        category
        tags
        metadata
        created_at
        updated_at
        accessed_at
        access_count
        version
        archived
        pinned

    The class also provides:

        - CRUD operations
        - searching
        - filtering
        - pagination
        - sorting
        - tags
        - categories
        - metadata
        - archive/pin support
        - import/export
        - backups
        - statistics
        - validation
        - thread-safe access
        - automatic persistence
    """

    SCHEMA_VERSION = 1

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(
        self,
        path="data/memory.json",
        autosave=True,
        create_file=True,
    ):

        self.path = Path(path)

        self.autosave = bool(
            autosave
        )

        self.records = {}

        self.lock = threading.RLock()

        self.started_at = datetime.now()

        self.last_loaded = None
        self.last_saved = None

        self.total_created = 0
        self.total_updated = 0
        self.total_deleted = 0
        self.total_reads = 0

        if create_file:

            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if self.path.exists():

                self.load()

            else:

                self.save()

    # ==========================================
    # TIME / IDS
    # ==========================================

    def _now(self):

        return datetime.now().isoformat()

    def _generate_id(self):

        return str(
            uuid.uuid4()
        )

    # ==========================================
    # VALIDATION
    # ==========================================

    def _validate_content(
        self,
        content,
    ):

        if content is None:
            raise StorageValidationError(
                "Content cannot be None."
            )

        if not isinstance(
            content,
            str,
        ):

            raise StorageValidationError(
                "Content must be a string."
            )

        if not content.strip():

            raise StorageValidationError(
                "Content cannot be empty."
            )

    def _validate_record(
        self,
        record,
    ):

        if not isinstance(
            record,
            dict,
        ):

            raise StorageValidationError(
                "Record must be a dictionary."
            )

        required = [
            "id",
            "content",
            "created_at",
            "updated_at",
        ]

        for field in required:

            if field not in record:

                raise StorageValidationError(
                    f"Missing required field: "
                    f"{field}"
                )

        self._validate_content(
            record["content"]
        )

        if not isinstance(
            record.get("tags", []),
            list,
        ):

            raise StorageValidationError(
                "Tags must be a list."
            )

        if not isinstance(
            record.get(
                "metadata",
                {},
            ),
            dict,
        ):

            raise StorageValidationError(
                "Metadata must be a dictionary."
            )

        return True

    # ==========================================
    # NORMALIZATION
    # ==========================================

    def _normalize_tags(
        self,
        tags,
    ):

        if tags is None:
            return []

        if isinstance(
            tags,
            str,
        ):

            tags = [tags]

        if not isinstance(
            tags,
            (list, tuple, set),
        ):

            raise StorageValidationError(
                "Tags must be a list-like value."
            )

        result = []

        for tag in tags:

            tag = str(tag).strip()

            if tag and tag not in result:

                result.append(tag)

        return result

    def _normalize_metadata(
        self,
        metadata,
    ):

        if metadata is None:
            return {}

        if not isinstance(
            metadata,
            dict,
        ):

            raise StorageValidationError(
                "Metadata must be a dictionary."
            )

        return deepcopy(
            metadata
        )

    def _build_record(
        self,
        content,
        title=None,
        category=None,
        tags=None,
        metadata=None,
        pinned=False,
        archived=False,
    ):

        self._validate_content(
            content
        )

        timestamp = self._now()

        record = {
            "id": self._generate_id(),

            "content": content.strip(),

            "title": (
                str(title).strip()
                if title is not None
                else None
            ),

            "category": (
                str(category).strip()
                if category is not None
                else "general"
            ),

            "tags": self._normalize_tags(
                tags
            ),

            "metadata": (
                self._normalize_metadata(
                    metadata
                )
            ),

            "created_at": timestamp,
            "updated_at": timestamp,
            "accessed_at": None,

            "access_count": 0,

            "version": 1,

            "archived": bool(
                archived
            ),

            "pinned": bool(
                pinned
            ),
        }

        self._validate_record(
            record
        )

        return record

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
        archived=False,
    ):

        with self.lock:

            record = self._build_record(
                content=content,
                title=title,
                category=category,
                tags=tags,
                metadata=metadata,
                pinned=pinned,
                archived=archived,
            )

            self.records[
                record["id"]
            ] = record

            self.total_created += 1

            self._autosave()

            return deepcopy(
                record
            )

    # ==========================================
    # READ
    # ==========================================

    def get(
        self,
        record_id,
        track_access=True,
    ):

        with self.lock:

            record = self.records.get(
                record_id
            )

            if record is None:

                raise StorageNotFoundError(
                    f"Record not found: "
                    f"{record_id}"
                )

            self.total_reads += 1

            if track_access:

                record[
                    "accessed_at"
                ] = self._now()

                record[
                    "access_count"
                ] += 1

            return deepcopy(
                record
            )

    def get_optional(
        self,
        record_id,
    ):

        try:

            return self.get(
                record_id
            )

        except StorageNotFoundError:

            return None

    # ==========================================
    # UPDATE
    # ==========================================

    def update(
        self,
        record_id,
        **changes,
    ):

        with self.lock:

            if record_id not in self.records:

                raise StorageNotFoundError(
                    f"Record not found: "
                    f"{record_id}"
                )

            record = self.records[
                record_id
            ]

            allowed = {
                "content",
                "title",
                "category",
                "tags",
                "metadata",
                "archived",
                "pinned",
            }

            for field, value in changes.items():

                if field not in allowed:

                    raise StorageValidationError(
                        f"Cannot update field: "
                        f"{field}"
                    )

                if field == "content":

                    self._validate_content(
                        value
                    )

                    record[field] = (
                        value.strip()
                    )

                elif field == "tags":

                    record[field] = (
                        self._normalize_tags(
                            value
                        )
                    )

                elif field == "metadata":

                    record[field] = (
                        self._normalize_metadata(
                            value
                        )
                    )

                elif field == "title":

                    record[field] = (
                        str(value).strip()
                        if value is not None
                        else None
                    )

                elif field == "category":

                    record[field] = (
                        str(value).strip()
                        if value is not None
                        else "general"
                    )

                else:

                    record[field] = bool(
                        value
                    )

            record[
                "updated_at"
            ] = self._now()

            record[
                "version"
            ] += 1

            self._validate_record(
                record
            )

            self.total_updated += 1

            self._autosave()

            return deepcopy(
                record
            )

    # ==========================================
    # DELETE
    # ==========================================

    def delete(
        self,
        record_id,
    ):

        with self.lock:

            if record_id not in self.records:

                raise StorageNotFoundError(
                    f"Record not found: "
                    f"{record_id}"
                )

            deleted = self.records.pop(
                record_id
            )

            self.total_deleted += 1

            self._autosave()

            return deepcopy(
                deleted
            )

    # ==========================================
    # ARCHIVING
    # ==========================================

    def archive(
        self,
        record_id,
    ):

        return self.update(
            record_id,
            archived=True,
        )

    def unarchive(
        self,
        record_id,
    ):

        return self.update(
            record_id,
            archived=False,
        )

    # ==========================================
    # PINNING
    # ==========================================

    def pin(
        self,
        record_id,
    ):

        return self.update(
            record_id,
            pinned=True,
        )

    def unpin(
        self,
        record_id,
    ):

        return self.update(
            record_id,
            pinned=False,
        )

    # ==========================================
    # SEARCH
    # ==========================================

    def search(
        self,
        query,
        include_archived=False,
        category=None,
        tags=None,
    ):

        if not query:

            return []

        query = str(
            query
        ).lower()

        required_tags = (
            self._normalize_tags(
                tags
            )
        )

        results = []

        with self.lock:

            for record in self.records.values():

                if (
                    record["archived"]
                    and not include_archived
                ):

                    continue

                if (
                    category
                    and record["category"]
                    != category
                ):

                    continue

                if required_tags:

                    record_tags = set(
                        record["tags"]
                    )

                    if not all(
                        tag in record_tags
                        for tag in required_tags
                    ):

                        continue

                searchable = " ".join(
                    [
                        record["content"],
                        record.get(
                            "title"
                        ) or "",
                        record.get(
                            "category"
                        ) or "",
                        " ".join(
                            record.get(
                                "tags",
                                []
                            )
                        ),
                    ]
                ).lower()

                if query in searchable:

                    results.append(
                        deepcopy(
                            record
                        )
                    )

        return results

    # ==========================================
    # LIST / PAGINATION
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

        with self.lock:

            records = []

            for record in self.records.values():

                if (
                    record["archived"]
                    and not include_archived
                ):

                    continue

                if (
                    category
                    and record["category"]
                    != category
                ):

                    continue

                if (
                    pinned is not None
                    and record["pinned"]
                    != bool(pinned)
                ):

                    continue

                records.append(
                    deepcopy(
                        record
                    )
                )

            records.sort(
                key=lambda item: (
                    item.get(
                        sort_by,
                        ""
                    )
                ),
                reverse=descending,
            )

            offset = max(
                int(offset),
                0,
            )

            if limit is None:

                return records[offset:]

            limit = max(
                int(limit),
                0,
            )

            return records[
                offset:
                offset + limit
            ]

    # ==========================================
    # COUNTING
    # ==========================================

    def count(
        self,
        include_archived=True,
    ):

        if include_archived:

            return len(
                self.records
            )

        return sum(
            1
            for record
            in self.records.values()
            if not record["archived"]
        )

    def count_category(
        self,
        category,
    ):

        return sum(
            1
            for record
            in self.records.values()
            if record["category"]
            == category
        )

    # ==========================================
    # CATEGORIES
    # ==========================================

    def categories(self):

        values = {
            record["category"]
            for record
            in self.records.values()
        }

        return sorted(
            values
        )

    # ==========================================
    # TAGS
    # ==========================================

    def tags(self):

        values = set()

        for record in self.records.values():

            values.update(
                record.get(
                    "tags",
                    []
                )
            )

        return sorted(
            values
        )

    def records_by_tag(
        self,
        tag,
    ):

        tag = str(tag).strip()

        results = []

        for record in self.records.values():

            if tag in record.get(
                "tags",
                [],
            ):

                results.append(
                    deepcopy(
                        record
                    )
                )

        return results

    # ==========================================
    # METADATA SEARCH
    # ==========================================

    def find_by_metadata(
        self,
        key,
        value,
    ):

        results = []

        for record in self.records.values():

            metadata = record.get(
                "metadata",
                {}
            )

            if metadata.get(key) == value:

                results.append(
                    deepcopy(
                        record
                    )
                )

        return results

    # ==========================================
    # DUPLICATE DETECTION
    # ==========================================

    def find_duplicates(
        self,
        content,
    ):

        normalized = (
            str(content)
            .strip()
            .lower()
        )

        results = []

        for record in self.records.values():

            if (
                record["content"]
                .strip()
                .lower()
                == normalized
            ):

                results.append(
                    deepcopy(
                        record
                    )
                )

        return results

    # ==========================================
    # PERSISTENCE
    # ==========================================

    def _autosave(self):

        if self.autosave:

            self.save()

    def save(self):

        with self.lock:

            payload = {
                "schema_version": (
                    self.SCHEMA_VERSION
                ),
                "saved_at": self._now(),
                "records": list(
                    self.records.values()
                ),
            }

            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary = self.path.with_suffix(
                ".tmp"
            )

            with temporary.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    payload,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            temporary.replace(
                self.path
            )

            self.last_saved = (
                datetime.now()
            )

            return True

    # ==========================================
    # LOAD
    # ==========================================

    def load(self):

        with self.lock:

            if not self.path.exists():

                return False

            try:

                with self.path.open(
                    "r",
                    encoding="utf-8",
                ) as file:

                    payload = json.load(
                        file
                    )

            except (
                json.JSONDecodeError,
                OSError,
            ) as error:

                raise StorageError(
                    f"Unable to load storage: "
                    f"{error}"
                )

            records = payload.get(
                "records",
                []
            )

            loaded = {}

            for record in records:

                try:

                    self._validate_record(
                        record
                    )

                except StorageValidationError:

                    continue

                loaded[
                    record["id"]
                ] = record

            self.records = loaded

            self.last_loaded = (
                datetime.now()
            )

            return True

    # ==========================================
    # BACKUP
    # ==========================================

    def backup(
        self,
        destination=None,
    ):

        if destination is None:

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            destination = (
                self.path.parent
                / f"{self.path.stem}"
                f"_backup_{timestamp}"
                f"{self.path.suffix}"
            )

        destination = Path(
            destination
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.lock:

            payload = {
                "schema_version": (
                    self.SCHEMA_VERSION
                ),
                "backup_created_at": (
                    self._now()
                ),
                "records": list(
                    self.records.values()
                ),
            }

            with destination.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    payload,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

        return str(
            destination
        )

    # ==========================================
    # IMPORT
    # ==========================================

    def import_data(
        self,
        source,
        merge=True,
    ):

        source = Path(
            source
        )

        if not source.exists():

            raise StorageError(
                f"Import source does not exist: "
                f"{source}"
            )

        try:

            with source.open(
                "r",
                encoding="utf-8",
            ) as file:

                payload = json.load(
                    file
                )

        except (
            OSError,
            json.JSONDecodeError,
        ) as error:

            raise StorageError(
                f"Unable to import data: "
                f"{error}"
            )

        imported = payload.get(
            "records",
            []
        )

        if not merge:

            self.records.clear()

        count = 0

        for record in imported:

            try:

                self._validate_record(
                    record
                )

            except StorageValidationError:

                continue

            record_id = record["id"]

            self.records[
                record_id
            ] = record

            count += 1

        self._autosave()

        return count

    # ==========================================
    # EXPORT
    # ==========================================

    def export(
        self,
        destination,
        include_archived=True,
    ):

        destination = Path(
            destination
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        records = []

        for record in self.records.values():

            if (
                record["archived"]
                and not include_archived
            ):

                continue

            records.append(
                deepcopy(
                    record
                )
            )

        payload = {
            "schema_version": (
                self.SCHEMA_VERSION
            ),
            "exported_at": self._now(),
            "records": records,
        }

        with destination.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return str(
            destination
        )

    # ==========================================
    # MAINTENANCE
    # ==========================================

    def purge_archived(self):

        with self.lock:

            ids = [
                record_id
                for record_id, record
                in self.records.items()
                if record["archived"]
            ]

            for record_id in ids:

                del self.records[
                    record_id
                ]

            self.total_deleted += len(
                ids
            )

            self._autosave()

            return len(ids)

    def cleanup_duplicates(self):

        with self.lock:

            seen = set()
            duplicates = []

            for record_id, record in (
                self.records.items()
            ):

                key = (
                    record["content"]
                    .strip()
                    .lower()
                )

                if key in seen:

                    duplicates.append(
                        record_id
                    )

                else:

                    seen.add(key)

            for record_id in duplicates:

                del self.records[
                    record_id
                ]

            self.total_deleted += len(
                duplicates
            )

            self._autosave()

            return len(
                duplicates
            )

    # ==========================================
    # STATISTICS
    # ==========================================

    def statistics(self):

        active = 0
        archived = 0
        pinned = 0

        for record in self.records.values():

            if record["archived"]:
                archived += 1
            else:
                active += 1

            if record["pinned"]:
                pinned += 1

        return {
            "total": len(
                self.records
            ),
            "active": active,
            "archived": archived,
            "pinned": pinned,
            "categories": len(
                self.categories()
            ),
            "tags": len(
                self.tags()
            ),
            "created": (
                self.total_created
            ),
            "updated": (
                self.total_updated
            ),
            "deleted": (
                self.total_deleted
            ),
            "reads": (
                self.total_reads
            ),
            "started_at": (
                self.started_at.isoformat()
            ),
            "last_loaded": (
                self.last_loaded.isoformat()
                if self.last_loaded
                else None
            ),
            "last_saved": (
                self.last_saved.isoformat()
                if self.last_saved
                else None
            ),
        }

    # ==========================================
    # HEALTH CHECK
    # ==========================================

    def health_check(self):

        try:

            exists = self.path.exists()

            readable = (
                self.path.is_file()
                if exists
                else True
            )

            return {
                "healthy": readable,
                "path": str(
                    self.path
                ),
                "exists": exists,
                "readable": readable,
                "records": len(
                    self.records
                ),
                "autosave": self.autosave,
            }

        except OSError as error:

            return {
                "healthy": False,
                "path": str(
                    self.path
                ),
                "exists": False,
                "readable": False,
                "records": len(
                    self.records
                ),
                "autosave": self.autosave,
                "error": str(error),
            }

    # ==========================================
    # STATUS
    # ==========================================

    def status(self):

        return {
            "active": True,
            "path": str(
                self.path
            ),
            "autosave": self.autosave,
            "schema_version": (
                self.SCHEMA_VERSION
            ),
            "records": len(
                self.records
            ),
            "statistics": (
                self.statistics()
            ),
            "health": (
                self.health_check()
            ),
        }


# ==========================================
# GLOBAL STORAGE
# ==========================================

storage = Storage(
    path="data/memory.json",
    autosave=True,
)