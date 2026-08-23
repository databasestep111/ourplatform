"""
OurPlatform Search Index
========================

High-performance, extensible inverted-index subsystem.

Responsibilities
----------------
- Document registration
- Document replacement
- Document deletion
- Inverted term indexes
- Field-specific indexes
- Term frequency tracking
- Document frequency tracking
- Prefix lookup
- Phrase candidate generation
- Candidate intersection
- Candidate union
- Required/excluded term filtering
- Metadata filtering
- Tag filtering
- Category filtering
- Field filtering
- Document statistics
- Collection statistics
- Average document length
- Incremental updates
- Bulk indexing
- Bulk deletion
- Index rebuilding
- Index validation
- Index diagnostics
- Snapshots
- Serialization
- Import/export hooks
- Query candidate generation
- Ranking-engine integration
- Version tracking
- Change tracking

Architecture
------------

    documents
        |
        +--> inverted index
        |
        +--> field indexes
        |
        +--> metadata indexes
        |
        +--> term statistics
        |
        +--> document statistics
        |
        v
    candidate retrieval
        |
        v
    ranking.py

The index deliberately does NOT decide final relevance.
It retrieves plausible candidates efficiently and leaves
relevance scoring to ranking.py.
"""

from __future__ import annotations

import copy
import json
import math
import re

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


try:
    from search.tokenizer import (
        Tokenizer,
        tokenizer as default_tokenizer,
    )
except ImportError:
    from tokenizer import (
        Tokenizer,
        tokenizer as default_tokenizer,
    )


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_FIELDS = (
    "text",
    "title",
    "heading",
    "tags",
    "keywords",
)

INDEX_VERSION = "2.0"


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class Posting:
    """
    Represents one document's occurrence of a term.

    Keeping frequency and positions available gives the index
    enough information for future phrase/proximity algorithms.
    """

    document_id: Any

    frequency: int = 0

    positions: List[int] = field(
        default_factory=list
    )

    field_frequencies: Dict[
        str,
        int,
    ] = field(
        default_factory=dict
    )

    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:

        return {
            "document_id": self.document_id,
            "frequency": self.frequency,
            "positions": list(
                self.positions
            ),
            "field_frequencies": dict(
                self.field_frequencies
            ),
            "weight": self.weight,
        }


@dataclass
class DocumentRecord:
    """
    Canonical indexed representation of a document.
    """

    document_id: Any

    data: Dict[str, Any]

    tokens: List[str] = field(
        default_factory=list
    )

    field_tokens: Dict[
        str,
        List[str],
    ] = field(
        default_factory=dict
    )

    term_frequencies: Dict[
        str,
        int,
    ] = field(
        default_factory=dict
    )

    field_term_frequencies: Dict[
        str,
        Dict[str, int],
    ] = field(
        default_factory=dict
    )

    length: int = 0

    created_at: str = ""

    updated_at: str = ""

    version: int = 1

    active: bool = True

    def to_dict(self) -> Dict[str, Any]:

        return {
            "document_id": self.document_id,
            "data": copy.deepcopy(
                self.data
            ),
            "tokens": list(
                self.tokens
            ),
            "field_tokens": {
                key: list(value)
                for key, value
                in self.field_tokens.items()
            },
            "term_frequencies": dict(
                self.term_frequencies
            ),
            "field_term_frequencies": {
                field_name: dict(
                    frequencies
                )
                for field_name, frequencies
                in self.field_term_frequencies.items()
            },
            "length": self.length,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "active": self.active,
        }


@dataclass
class IndexStatistics:
    """
    Collection-level index statistics.

    These statistics are consumed by ranking.py.
    """

    document_count: int = 0

    active_document_count: int = 0

    total_terms: int = 0

    unique_terms: int = 0

    average_document_length: float = 0.0

    maximum_document_length: int = 0

    minimum_document_length: int = 0

    total_indexed_fields: int = 0

    generation: int = 0

    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:

        return {
            "document_count": self.document_count,
            "active_document_count": (
                self.active_document_count
            ),
            "total_terms": self.total_terms,
            "unique_terms": self.unique_terms,
            "average_document_length": (
                self.average_document_length
            ),
            "maximum_document_length": (
                self.maximum_document_length
            ),
            "minimum_document_length": (
                self.minimum_document_length
            ),
            "total_indexed_fields": (
                self.total_indexed_fields
            ),
            "generation": self.generation,
            "last_updated": self.last_updated,
        }


@dataclass
class IndexChange:
    """
    Describes a modification to the index.
    """

    operation: str

    document_id: Any

    timestamp: str

    version: int

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "operation": self.operation,
            "document_id": self.document_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "details": copy.deepcopy(
                self.details
            ),
        }


@dataclass
class SearchCandidates:
    """
    Candidate retrieval result.

    Ranking happens later.
    """

    document_ids: List[Any]

    matched_terms: Dict[
        Any,
        Set[str],
    ] = field(
        default_factory=dict
    )

    required_terms: Set[str] = field(
        default_factory=set
    )

    excluded_terms: Set[str] = field(
        default_factory=set
    )

    total_candidates: int = 0

    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:

        return {
            "document_ids": list(
                self.document_ids
            ),
            "matched_terms": {
                str(key): sorted(
                    value
                )
                for key, value
                in self.matched_terms.items()
            },
            "required_terms": sorted(
                self.required_terms
            ),
            "excluded_terms": sorted(
                self.excluded_terms
            ),
            "total_candidates": (
                self.total_candidates
            ),
            "generation": self.generation,
        }


# ============================================================
# MAIN INDEX
# ============================================================


class SearchIndex:
    """
    Advanced inverted search index.

    The index maintains several structures simultaneously:

        documents
            document_id -> DocumentRecord

        inverted
            term -> document IDs

        postings
            term -> document ID -> Posting

        field_indexes
            field -> term -> document IDs

        metadata_indexes
            field -> value -> document IDs

        prefix_index
            prefix -> terms

    This lets the search engine retrieve candidates without
    scanning every document.
    """

    def __init__(
        self,
        tokenizer_instance: Optional[
            Tokenizer
        ] = None,
        fields: Optional[
            Sequence[str]
        ] = None,
        keep_positions: bool = True,
        max_history: int = 1000,
    ):

        self.tokenizer = (
            tokenizer_instance
            or default_tokenizer
        )

        self.fields = tuple(
            fields
            or DEFAULT_FIELDS
        )

        self.keep_positions = (
            keep_positions
        )

        self.max_history = max(
            1,
            int(
                max_history
            ),
        )

        # -----------------------------------------------
        # Core document storage
        # -----------------------------------------------

        self.documents: Dict[
            Any,
            DocumentRecord,
        ] = {}

        # -----------------------------------------------
        # Inverted index
        # -----------------------------------------------

        self.inverted: Dict[
            str,
            Set[Any],
        ] = defaultdict(set)

        # -----------------------------------------------
        # Posting lists
        # -----------------------------------------------

        self.postings: Dict[
            str,
            Dict[Any, Posting],
        ] = defaultdict(dict)

        # -----------------------------------------------
        # Field indexes
        # -----------------------------------------------

        self.field_indexes: Dict[
            str,
            Dict[
                str,
                Set[Any],
            ],
        ] = {
            field_name: defaultdict(set)
            for field_name
            in self.fields
        }

        # -----------------------------------------------
        # Prefix index
        # -----------------------------------------------

        self.prefix_index: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

        # -----------------------------------------------
        # Metadata indexes
        # -----------------------------------------------

        self.metadata_indexes: Dict[
            str,
            Dict[
                str,
                Set[Any],
            ],
        ] = defaultdict(
            lambda: defaultdict(set)
        )

        # -----------------------------------------------
        # Category / tag convenience indexes
        # -----------------------------------------------

        self.category_index: Dict[
            str,
            Set[Any],
        ] = defaultdict(set)

        self.tag_index: Dict[
            str,
            Set[Any],
        ] = defaultdict(set)

        # -----------------------------------------------
        # Statistics
        # -----------------------------------------------

        self.document_frequency: Dict[
            str,
            int,
        ] = defaultdict(int)

        self.collection_frequency: Dict[
            str,
            int,
        ] = defaultdict(int)

        self.document_lengths: Dict[
            Any,
            int,
        ] = {}

        self.statistics = (
            IndexStatistics()
        )

        # -----------------------------------------------
        # Versioning
        # -----------------------------------------------

        self.generation = 0

        self.change_history: List[
            IndexChange
        ] = []

        self.created_at = (
            self._now()
        )

        self.updated_at = (
            self.created_at
        )

        self.dirty = False

    # ========================================================
    # TIME
    # ========================================================

    @staticmethod
    def _now() -> str:

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ========================================================
    # DOCUMENT IDS
    # ========================================================

    def _resolve_document_id(
        self,
        document: Any,
    ) -> Any:

        if isinstance(
            document,
            Mapping,
        ):

            document_id = (
                document.get(
                    "id",
                    document.get(
                        "document_id"
                    ),
                )
            )

            if document_id is not None:

                return document_id

        for attribute in (
            "id",
            "document_id",
        ):

            value = getattr(
                document,
                attribute,
                None,
            )

            if value is not None:

                return value

        raise ValueError(
            "Document must contain an 'id' "
            "or 'document_id'."
        )

    # ========================================================
    # DOCUMENT NORMALIZATION
    # ========================================================

    def _normalize_document(
        self,
        document: Any,
    ) -> Dict[str, Any]:

        if isinstance(
            document,
            Mapping,
        ):

            return copy.deepcopy(
                dict(document)
            )

        result = {}

        for field_name in (
            set(self.fields)
            | {
                "id",
                "document_id",
                "metadata",
                "category",
                "tags",
                "keywords",
                "quality",
                "popularity",
                "created_at",
                "updated_at",
            }
        ):

            value = getattr(
                document,
                field_name,
                None,
            )

            if value is not None:

                result[
                    field_name
                ] = copy.deepcopy(
                    value
                )

        if "text" not in result:

            result["text"] = str(
                document
            )

        return result

    # ========================================================
    # TEXT EXTRACTION
    # ========================================================

    def _field_value(
        self,
        document: Mapping[str, Any],
        field_name: str,
    ) -> str:

        value = document.get(
            field_name,
            "",
        )

        if value is None:

            return ""

        if isinstance(
            value,
            (list, tuple, set),
        ):

            return " ".join(
                str(item)
                for item in value
            )

        if isinstance(
            value,
            Mapping,
        ):

            return " ".join(
                str(item)
                for item in value.values()
            )

        return str(value)

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:

        if not text:
            return []

        return list(
            self.tokenizer.tokenize(
                text
            )
        )

    # ========================================================
    # TERM POSITIONS
    # ========================================================

    @staticmethod
    def _positions(
        tokens: Sequence[str],
    ) -> Dict[
        str,
        List[int],
    ]:

        positions = defaultdict(list)

        for position, token in enumerate(
            tokens
        ):

            positions[token].append(
                position
            )

        return dict(
            positions
        )

    # ========================================================
    # RECORD CREATION
    # ========================================================

    def _build_record(
        self,
        document_id: Any,
        document: Mapping[str, Any],
        previous: Optional[
            DocumentRecord
        ] = None,
    ) -> DocumentRecord:

        field_tokens = {}

        field_frequencies = {}

        for field_name in self.fields:

            value = self._field_value(
                document,
                field_name,
            )

            tokens = self._tokenize(
                value
            )

            field_tokens[
                field_name
            ] = tokens

            field_frequencies[
                field_name
            ] = dict(
                Counter(tokens)
            )

        text_tokens = field_tokens.get(
            "text",
            [],
        )

        term_frequencies = dict(
            Counter(text_tokens)
        )

        now = self._now()

        created_at = document.get(
            "created_at"
        )

        if created_at is None:

            if previous:

                created_at = (
                    previous.created_at
                )

            else:

                created_at = now

        version = 1

        if previous:

            version = (
                previous.version
                + 1
            )

        return DocumentRecord(
            document_id=document_id,
            data=copy.deepcopy(
                dict(document)
            ),
            tokens=text_tokens,
            field_tokens=field_tokens,
            term_frequencies=term_frequencies,
            field_term_frequencies=(
                field_frequencies
            ),
            length=len(
                text_tokens
            ),
            created_at=str(
                created_at
            ),
            updated_at=str(
                document.get(
                    "updated_at",
                    now,
                )
            ),
            version=version,
            active=True,
        )

    # ========================================================
    # ADD DOCUMENT
    # ========================================================

    def add(
        self,
        document: Any,
        document_id: Any = None,
    ) -> DocumentRecord:

        normalized = (
            self._normalize_document(
                document
            )
        )

        if document_id is None:

            document_id = (
                self._resolve_document_id(
                    normalized
                )
            )

        normalized.setdefault(
            "id",
            document_id,
        )

        existing = self.documents.get(
            document_id
        )

        if existing is not None:

            return self.update(
                document_id,
                normalized,
            )

        record = self._build_record(
            document_id,
            normalized,
        )

        self.documents[
            document_id
        ] = record

        self._add_record_to_indexes(
            record
        )

        self._record_change(
            "add",
            document_id,
            {
                "length": record.length,
            },
        )

        self._recalculate_statistics()

        return record

    # ========================================================
    # BULK ADD
    # ========================================================

    def add_many(
        self,
        documents: Iterable[Any],
    ) -> int:

        count = 0

        for document in documents:

            self.add(
                document
            )

            count += 1

        return count

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        document_id: Any,
        document: Any,
    ) -> DocumentRecord:

        if document_id not in self.documents:

            return self.add(
                document,
                document_id=document_id,
            )

        normalized = (
            self._normalize_document(
                document
            )
        )

        normalized.setdefault(
            "id",
            document_id,
        )

        previous = self.documents[
            document_id
        ]

        self._remove_record_from_indexes(
            previous
        )

        record = self._build_record(
            document_id,
            normalized,
            previous=previous,
        )

        self.documents[
            document_id
        ] = record

        self._add_record_to_indexes(
            record
        )

        self._record_change(
            "update",
            document_id,
            {
                "version": record.version,
            },
        )

        self._recalculate_statistics()

        return record

    # ========================================================
    # PARTIAL UPDATE
    # ========================================================

    def patch(
        self,
        document_id: Any,
        changes: Mapping[str, Any],
    ) -> Optional[
        DocumentRecord
    ]:

        existing = self.documents.get(
            document_id
        )

        if existing is None:

            return None

        merged = copy.deepcopy(
            existing.data
        )

        merged.update(
            copy.deepcopy(
                dict(changes)
            )
        )

        return self.update(
            document_id,
            merged,
        )

    # ========================================================
    # REMOVE
    # ========================================================

    def remove(
        self,
        document_id: Any,
    ) -> bool:

        record = self.documents.get(
            document_id
        )

        if record is None:

            return False

        self._remove_record_from_indexes(
            record
        )

        del self.documents[
            document_id
        ]

        self.document_lengths.pop(
            document_id,
            None,
        )

        self._record_change(
            "remove",
            document_id,
        )

        self._recalculate_statistics()

        return True

    # ========================================================
    # BULK REMOVE
    # ========================================================

    def remove_many(
        self,
        document_ids: Iterable[Any],
    ) -> int:

        count = 0

        for document_id in list(
            document_ids
        ):

            if self.remove(
                document_id
            ):

                count += 1

        return count

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(
        self,
        keep_history: bool = False,
    ):

        self.documents.clear()

        self.inverted.clear()

        self.postings.clear()

        self.prefix_index.clear()

        self.metadata_indexes.clear()

        self.category_index.clear()

        self.tag_index.clear()

        self.document_frequency.clear()

        self.collection_frequency.clear()

        self.document_lengths.clear()

        for field_index in (
            self.field_indexes.values()
        ):

            field_index.clear()

        if not keep_history:

            self.change_history.clear()

        self.generation += 1

        self.dirty = True

        self._recalculate_statistics()

    # ========================================================
    # ADD RECORD TO INDEX
    # ========================================================

    def _add_record_to_indexes(
        self,
        record: DocumentRecord,
    ):

        document_id = (
            record.document_id
        )

        self.document_lengths[
            document_id
        ] = record.length

        positions = self._positions(
            record.tokens
        )

        # -----------------------------------------------
        # Main inverted index
        # -----------------------------------------------

        for term, frequency in (
            record.term_frequencies.items()
        ):

            self.inverted[
                term
            ].add(
                document_id
            )

            posting = Posting(
                document_id=document_id,
                frequency=frequency,
                positions=(
                    positions.get(
                        term,
                        [],
                    )
                    if self.keep_positions
                    else []
                ),
                field_frequencies={
                    field_name: frequencies.get(
                        term,
                        0,
                    )
                    for field_name,
                    frequencies
                    in record
                    .field_term_frequencies
                    .items()
                    if term in frequencies
                },
            )

            self.postings[
                term
            ][
                document_id
            ] = posting

            self.document_frequency[
                term
            ] += 1

            self.collection_frequency[
                term
            ] += frequency

            self._add_prefixes(
                term
            )

        # -----------------------------------------------
        # Field indexes
        # -----------------------------------------------

        for field_name, frequencies in (
            record.field_term_frequencies.items()
        ):

            if field_name not in (
                self.field_indexes
            ):

                self.field_indexes[
                    field_name
                ] = defaultdict(set)

            for term in frequencies:

                self.field_indexes[
                    field_name
                ][
                    term
                ].add(
                    document_id
                )

        # -----------------------------------------------
        # Metadata
        # -----------------------------------------------

        self._index_metadata(
            record
        )

        # -----------------------------------------------
        # Category
        # -----------------------------------------------

        category = record.data.get(
            "category"
        )

        if category is not None:

            self.category_index[
                self._normalize_value(
                    category
                )
            ].add(
                document_id
            )

        # -----------------------------------------------
        # Tags
        # -----------------------------------------------

        tags = record.data.get(
            "tags",
            [],
        )

        if isinstance(
            tags,
            str,
        ):

            tags = [
                tags
            ]

        for tag in (
            tags or []
        ):

            self.tag_index[
                self._normalize_value(
                    tag
                )
            ].add(
                document_id
            )

        self.dirty = True

    # ========================================================
    # REMOVE RECORD FROM INDEX
    # ========================================================

    def _remove_record_from_indexes(
        self,
        record: DocumentRecord,
    ):

        document_id = (
            record.document_id
        )

        # -----------------------------------------------
        # Main inverted index
        # -----------------------------------------------

        for term, frequency in (
            record.term_frequencies.items()
        ):

            documents = self.inverted.get(
                term
            )

            if documents:

                documents.discard(
                    document_id
                )

                if not documents:

                    self.inverted.pop(
                        term,
                        None,
                    )

            postings = self.postings.get(
                term
            )

            if postings:

                postings.pop(
                    document_id,
                    None,
                )

                if not postings:

                    self.postings.pop(
                        term,
                        None,
                    )

            self.document_frequency[
                term
            ] = max(
                0,
                self.document_frequency[
                    term
                ] - 1,
            )

            self.collection_frequency[
                term
            ] = max(
                0,
                self.collection_frequency[
                    term
                ] - frequency,
            )

            if (
                self.document_frequency[
                    term
                ] == 0
            ):

                self.document_frequency.pop(
                    term,
                    None
                )

            if (
                self.collection_frequency[
                    term
                ] == 0
            ):

                self.collection_frequency.pop(
                    term,
                    None
                )

        # -----------------------------------------------
        # Field indexes
        # -----------------------------------------------

        for field_name, frequencies in (
            record.field_term_frequencies.items()
        ):

            field_index = (
                self.field_indexes.get(
                    field_name
                )
            )

            if not field_index:
                continue

            for term in frequencies:

                documents = (
                    field_index.get(
                        term
                    )
                )

                if documents:

                    documents.discard(
                        document_id
                    )

                    if not documents:

                        field_index.pop(
                            term,
                            None,
                        )

        # -----------------------------------------------
        # Metadata
        # -----------------------------------------------

        self._remove_metadata(
            record
        )

        # -----------------------------------------------
        # Category
        # -----------------------------------------------

        category = record.data.get(
            "category"
        )

        if category is not None:

            normalized = (
                self._normalize_value(
                    category
                )
            )

            documents = (
                self.category_index.get(
                    normalized
                )
            )

            if documents:

                documents.discard(
                    document_id
                )

                if not documents:

                    self.category_index.pop(
                        normalized,
                        None,
                    )

        # -----------------------------------------------
        # Tags
        # -----------------------------------------------

        tags = record.data.get(
            "tags",
            [],
        )

        if isinstance(
            tags,
            str,
        ):

            tags = [
                tags
            ]

        for tag in (
            tags or []
        ):

            normalized = (
                self._normalize_value(
                    tag
                )
            )

            documents = (
                self.tag_index.get(
                    normalized
                )
            )

            if documents:

                documents.discard(
                    document_id
                )

                if not documents:

                    self.tag_index.pop(
                        normalized,
                        None,
                    )

        self.dirty = True

    # ========================================================
    # PREFIX INDEX
    # ========================================================

    def _add_prefixes(
        self,
        term: str,
        maximum_length: int = 20,
    ):

        if not term:
            return

        maximum = min(
            len(term),
            maximum_length,
        )

        for length in range(
            1,
            maximum + 1,
        ):

            prefix = term[
                :length
            ]

            self.prefix_index[
                prefix
            ].add(
                term
            )

    def prefix_terms(
        self,
        prefix: str,
        limit: Optional[int] = None,
    ) -> List[str]:

        prefix = self._normalize_value(
            prefix
        )

        terms = sorted(
            self.prefix_index.get(
                prefix,
                set(),
            )
        )

        if limit is not None:

            return terms[
                :max(
                    0,
                    int(limit),
                )
            ]

        return terms

    def prefix_documents(
        self,
        prefix: str,
    ) -> Set[Any]:

        terms = self.prefix_terms(
            prefix
        )

        result = set()

        for term in terms:

            result.update(
                self.inverted.get(
                    term,
                    set(),
                )
            )

        return result

    # ========================================================
    # METADATA INDEXING
    # ========================================================

    @staticmethod
    def _normalize_value(
        value: Any,
    ) -> str:

        return str(
            value
        ).strip().lower()

    def _index_metadata(
        self,
        record: DocumentRecord,
    ):

        metadata = record.data.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            Mapping,
        ):

            metadata = {}

        # Include selected top-level fields.
        combined = dict(
            metadata
        )

        for key in (
            "author",
            "source",
            "type",
            "status",
            "language",
            "category",
        ):

            if key in record.data:

                combined[key] = (
                    record.data[key]
                )

        for field_name, value in (
            combined.items()
        ):

            if isinstance(
                value,
                (dict, list, tuple, set),
            ):

                continue

            normalized = (
                self._normalize_value(
                    value
                )
            )

            if not normalized:
                continue

            self.metadata_indexes[
                field_name
            ][
                normalized
            ].add(
                record.document_id
            )

    def _remove_metadata(
        self,
        record: DocumentRecord,
    ):

        metadata = record.data.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            Mapping,
        ):

            metadata = {}

        combined = dict(
            metadata
        )

        for key in (
            "author",
            "source",
            "type",
            "status",
            "language",
            "category",
        ):

            if key in record.data:

                combined[key] = (
                    record.data[key]
                )

        for field_name, value in (
            combined.items()
        ):

            if isinstance(
                value,
                (dict, list, tuple, set),
            ):

                continue

            normalized = (
                self._normalize_value(
                    value
                )
            )

            field_index = (
                self.metadata_indexes.get(
                    field_name
                )
            )

            if not field_index:
                continue

            documents = (
                field_index.get(
                    normalized
                )
            )

            if documents:

                documents.discard(
                    record.document_id
                )

                if not documents:

                    field_index.pop(
                        normalized,
                        None,
                    )

    # ========================================================
    # BASIC LOOKUPS
    # ========================================================

    def get(
        self,
        document_id: Any,
    ) -> Optional[
        DocumentRecord
    ]:

        return self.documents.get(
            document_id
        )

    def exists(
        self,
        document_id: Any,
    ) -> bool:

        return (
            document_id
            in self.documents
        )

    def contains_term(
        self,
        term: str,
    ) -> bool:

        term = self._normalize_value(
            term
        )

        return bool(
            self.inverted.get(
                term
            )
        )

    def documents_for_term(
        self,
        term: str,
    ) -> Set[Any]:

        term = self._normalize_value(
            term
        )

        return set(
            self.inverted.get(
                term,
                set(),
            )
        )

    def posting(
        self,
        term: str,
        document_id: Any,
    ) -> Optional[
        Posting
    ]:

        term = self._normalize_value(
            term
        )

        return self.postings.get(
            term,
            {},
        ).get(
            document_id
        )

    # ========================================================
    # FIELD LOOKUPS
    # ========================================================

    def documents_for_field_term(
        self,
        field_name: str,
        term: str,
    ) -> Set[Any]:

        term = self._normalize_value(
            term
        )

        field_index = (
            self.field_indexes.get(
                field_name,
                {},
            )
        )

        return set(
            field_index.get(
                term,
                set(),
            )
        )

    def terms_for_field(
        self,
        field_name: str,
    ) -> Set[str]:

        return set(
            self.field_indexes.get(
                field_name,
                {},
            ).keys()
        )

    # ========================================================
    # CATEGORY / TAG
    # ========================================================

    def documents_for_category(
        self,
        category: str,
    ) -> Set[Any]:

        return set(
            self.category_index.get(
                self._normalize_value(
                    category
                ),
                set(),
            )
        )

    def documents_for_tag(
        self,
        tag: str,
    ) -> Set[Any]:

        return set(
            self.tag_index.get(
                self._normalize_value(
                    tag
                ),
                set(),
            )
        )

    # ========================================================
    # METADATA FILTER
    # ========================================================

    def documents_for_metadata(
        self,
        field_name: str,
        value: Any,
    ) -> Set[Any]:

        normalized = (
            self._normalize_value(
                value
            )
        )

        return set(
            self.metadata_indexes.get(
                field_name,
                {},
            ).get(
                normalized,
                set(),
            )
        )

    # ========================================================
    # BOOLEAN OPERATIONS
    # ========================================================

    @staticmethod
    def intersect(
        sets: Iterable[
            Set[Any]
        ],
    ) -> Set[Any]:

        sets = list(
            sets
        )

        if not sets:

            return set()

        sets.sort(
            key=len
        )

        result = set(
            sets[0]
        )

        for current in sets[1:]:

            result.intersection_update(
                current
            )

            if not result:

                break

        return result

    @staticmethod
    def union(
        sets: Iterable[
            Set[Any]
        ],
    ) -> Set[Any]:

        result = set()

        for current in sets:

            result.update(
                current
            )

        return result

    @staticmethod
    def difference(
        source: Set[Any],
        excluded: Iterable[
            Set[Any]
        ],
    ) -> Set[Any]:

        result = set(
            source
        )

        for current in excluded:

            result.difference_update(
                current
            )

        return result

    # ========================================================
    # CANDIDATE RETRIEVAL
    # ========================================================

    def candidates_for_terms(
        self,
        terms: Sequence[str],
        mode: str = "or",
    ) -> Set[Any]:

        normalized = [
            self._normalize_value(
                term
            )
            for term in terms
            if str(term).strip()
        ]

        if not normalized:

            return set()

        posting_sets = [
            self.documents_for_term(
                term
            )
            for term in normalized
        ]

        if mode.lower() == "and":

            return self.intersect(
                posting_sets
            )

        return self.union(
            posting_sets
        )

    # ========================================================
    # ADVANCED CANDIDATE GENERATION
    # ========================================================

    def generate_candidates(
        self,
        terms: Sequence[str],
        required_terms: Optional[
            Sequence[str]
        ] = None,
        excluded_terms: Optional[
            Sequence[str]
        ] = None,
        mode: str = "or",
        minimum_match: int = 1,
    ) -> SearchCandidates:

        normalized_terms = [
            self._normalize_value(
                term
            )
            for term in terms
            if str(term).strip()
        ]

        required_terms = [
            self._normalize_value(
                term
            )
            for term in (
                required_terms
                or []
            )
        ]

        excluded_terms = [
            self._normalize_value(
                term
            )
            for term in (
                excluded_terms
                or []
            )
        ]

        candidate_sets = [
            self.documents_for_term(
                term
            )
            for term in normalized_terms
        ]

        if mode.lower() == "and":

            candidates = self.intersect(
                candidate_sets
            )

        else:

            candidates = self.union(
                candidate_sets
            )

        # -----------------------------------------------
        # Required terms
        # -----------------------------------------------

        if required_terms:

            required_sets = [
                self.documents_for_term(
                    term
                )
                for term in required_terms
            ]

            candidates = self.intersect(
                [
                    candidates,
                    self.intersect(
                        required_sets
                    ),
                ]
            )

        # -----------------------------------------------
        # Excluded terms
        # -----------------------------------------------

        for term in excluded_terms:

            candidates.difference_update(
                self.documents_for_term(
                    term
                )
            )

        # -----------------------------------------------
        # Minimum-match logic
        # -----------------------------------------------

        if (
            minimum_match > 1
            and normalized_terms
        ):

            match_counts = Counter()

            for term in normalized_terms:

                for document_id in (
                    self.documents_for_term(
                        term
                    )
                ):

                    if (
                        document_id
                        in candidates
                    ):

                        match_counts[
                            document_id
                        ] += 1

            candidates = {
                document_id
                for document_id in candidates
                if (
                    match_counts[
                        document_id
                    ]
                    >= minimum_match
                )
            }

        matched_terms = {}

        for document_id in candidates:

            matched_terms[
                document_id
            ] = {
                term
                for term in normalized_terms
                if document_id
                in self.documents_for_term(
                    term
                )
            }

        return SearchCandidates(
            document_ids=list(
                candidates
            ),
            matched_terms=matched_terms,
            required_terms=set(
                required_terms
            ),
            excluded_terms=set(
                excluded_terms
            ),
            total_candidates=len(
                candidates
            ),
            generation=self.generation,
        )

    # ========================================================
    # PREFIX CANDIDATES
    # ========================================================

    def candidates_for_prefix(
        self,
        prefix: str,
    ) -> Set[Any]:

        return self.prefix_documents(
            prefix
        )

    # ========================================================
    # FIELD CANDIDATES
    # ========================================================

    def candidates_for_fields(
        self,
        field_terms: Mapping[
            str,
            Sequence[str],
        ],
        mode: str = "or",
    ) -> Set[Any]:

        field_sets = []

        for field_name, terms in (
            field_terms.items()
        ):

            term_sets = [
                self.documents_for_field_term(
                    field_name,
                    term,
                )
                for term in terms
            ]

            if not term_sets:
                continue

            if mode.lower() == "and":

                field_sets.append(
                    self.intersect(
                        term_sets
                    )
                )

            else:

                field_sets.append(
                    self.union(
                        term_sets
                    )
                )

        if not field_sets:

            return set()

        if mode.lower() == "and":

            return self.intersect(
                field_sets
            )

        return self.union(
            field_sets
        )

    # ========================================================
    # PHRASE CANDIDATES
    # ========================================================

    def phrase_candidates(
        self,
        phrase: str,
    ) -> Set[Any]:

        tokens = self._tokenize(
            phrase
        )

        if not tokens:

            return set()

        candidate_sets = [
            self.documents_for_term(
                token
            )
            for token in tokens
        ]

        return self.intersect(
            candidate_sets
        )

    # ========================================================
    # DOCUMENT ITERATION
    # ========================================================

    def iter_documents(
        self,
        active_only: bool = True,
    ) -> Iterator[
        DocumentRecord
    ]:

        for record in (
            self.documents.values()
        ):

            if (
                active_only
                and not record.active
            ):

                continue

            yield record

    def all_document_ids(
        self,
        active_only: bool = True,
    ) -> List[Any]:

        return [
            record.document_id
            for record
            in self.iter_documents(
                active_only=active_only
            )
        ]

    # ========================================================
    # STATISTICS
    # ========================================================

    def _recalculate_statistics(
        self,
    ):

        lengths = list(
            self.document_lengths.values()
        )

        active_documents = sum(
            1
            for record
            in self.documents.values()
            if record.active
        )

        total_terms = sum(
            lengths
        )

        if lengths:

            average = (
                total_terms
                / len(lengths)
            )

            minimum = min(
                lengths
            )

            maximum = max(
                lengths
            )

        else:

            average = 0.0

            minimum = 0

            maximum = 0

        self.statistics = (
            IndexStatistics(
                document_count=len(
                    self.documents
                ),
                active_document_count=(
                    active_documents
                ),
                total_terms=total_terms,
                unique_terms=len(
                    self.inverted
                ),
                average_document_length=(
                    average
                ),
                maximum_document_length=(
                    maximum
                ),
                minimum_document_length=(
                    minimum
                ),
                total_indexed_fields=len(
                    self.field_indexes
                ),
                generation=self.generation,
                last_updated=self._now(),
            )
        )

        self.generation += 1

        self.statistics.generation = (
            self.generation
        )

        self.updated_at = (
            self.statistics.last_updated
        )

    def get_statistics(
        self,
    ) -> Dict[str, Any]:

        return self.statistics.to_dict()

    # ========================================================
    # TERM STATISTICS
    # ========================================================

    def term_statistics(
        self,
        term: str,
    ) -> Dict[str, Any]:

        term = self._normalize_value(
            term
        )

        documents = (
            self.inverted.get(
                term,
                set(),
            )
        )

        return {
            "term": term,
            "document_frequency": (
                self.document_frequency.get(
                    term,
                    0,
                )
            ),
            "collection_frequency": (
                self.collection_frequency.get(
                    term,
                    0,
                )
            ),
            "documents": len(
                documents
            ),
        }

    def top_terms(
        self,
        limit: int = 50,
        by: str = "document_frequency",
    ) -> List[
        Tuple[str, int]
    ]:

        if by == "collection_frequency":

            source = (
                self.collection_frequency
            )

        else:

            source = (
                self.document_frequency
            )

        return sorted(
            source.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )[
            :max(
                0,
                int(limit),
            )
        ]

    # ========================================================
    # INDEX VALIDATION
    # ========================================================

    def validate(
        self,
    ) -> Dict[str, Any]:

        errors = []

        warnings = []

        # -----------------------------------------------
        # Verify documents exist in inverted index.
        # -----------------------------------------------

        for document_id, record in (
            self.documents.items()
        ):

            for term in record.term_frequencies:

                if (
                    document_id
                    not in self.inverted.get(
                        term,
                        set(),
                    )
                ):

                    errors.append(
                        {
                            "type": (
                                "missing_inverted_posting"
                            ),
                            "document_id": (
                                document_id
                            ),
                            "term": term,
                        }
                    )

        # -----------------------------------------------
        # Verify inverted entries point to documents.
        # -----------------------------------------------

        for term, document_ids in (
            self.inverted.items()
        ):

            for document_id in document_ids:

                if document_id not in (
                    self.documents
                ):

                    errors.append(
                        {
                            "type": (
                                "dangling_document_reference"
                            ),
                            "term": term,
                            "document_id": (
                                document_id
                            ),
                        }
                    )

        # -----------------------------------------------
        # Verify posting lists.
        # -----------------------------------------------

        for term, postings in (
            self.postings.items()
        ):

            for document_id, posting in (
                postings.items()
            ):

                if document_id not in (
                    self.inverted.get(
                        term,
                        set(),
                    )
                ):

                    errors.append(
                        {
                            "type": (
                                "posting_inconsistency"
                            ),
                            "term": term,
                            "document_id": (
                                document_id
                            ),
                        }
                    )

                if posting.frequency < 0:

                    errors.append(
                        {
                            "type": (
                                "negative_frequency"
                            ),
                            "term": term,
                            "document_id": (
                                document_id
                            ),
                        }
                    )

        # -----------------------------------------------
        # Verify document lengths.
        # -----------------------------------------------

        for document_id, record in (
            self.documents.items()
        ):

            if (
                self.document_lengths.get(
                    document_id
                )
                != record.length
            ):

                errors.append(
                    {
                        "type": (
                            "length_mismatch"
                        ),
                        "document_id": (
                            document_id
                        ),
                    }
                )

        if (
            self.statistics.unique_terms
            != len(
                self.inverted
            )
        ):

            warnings.append(
                "Cached unique-term statistics "
                "do not match the inverted index."
            )

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "error_count": len(
                errors
            ),
            "warning_count": len(
                warnings
            ),
            "generation": (
                self.generation
            ),
        }

    # ========================================================
    # REBUILD
    # ========================================================

    def rebuild(
        self,
        documents: Optional[
            Iterable[Any]
        ] = None,
    ) -> Dict[str, Any]:

        if documents is None:

            documents = [
                record.data
                for record
                in self.documents.values()
            ]

        documents = list(
            documents
        )

        self.clear(
            keep_history=True
        )

        added = self.add_many(
            documents
        )

        validation = self.validate()

        return {
            "documents_processed": len(
                documents
            ),
            "documents_indexed": added,
            "generation": (
                self.generation
            ),
            "validation": validation,
        }

    # ========================================================
    # CHANGE HISTORY
    # ========================================================

    def _record_change(
        self,
        operation: str,
        document_id: Any,
        details: Optional[
            Mapping[str, Any]
        ] = None,
    ):

        change = IndexChange(
            operation=operation,
            document_id=document_id,
            timestamp=self._now(),
            version=self.generation,
            details=dict(
                details or {}
            ),
        )

        self.change_history.append(
            change
        )

        if len(
            self.change_history
        ) > self.max_history:

            del self.change_history[
                :len(
                    self.change_history
                )
                - self.max_history
            ]

    def history(
        self,
        limit: Optional[int] = None,
    ) -> List[
        Dict[str, Any]
    ]:

        changes = self.change_history

        if limit is not None:

            changes = changes[
                -max(
                    0,
                    int(limit),
                ):
            ]

        return [
            change.to_dict()
            for change in changes
        ]

    # ========================================================
    # SNAPSHOT
    # ========================================================

    def snapshot(
        self,
    ) -> Dict[str, Any]:

        return {
            "index_version": INDEX_VERSION,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "generation": self.generation,
            "fields": list(
                self.fields
            ),
            "documents": {
                str(document_id): (
                    record.to_dict()
                )
                for document_id, record
                in self.documents.items()
            },
            "statistics": (
                self.statistics.to_dict()
            ),
        }

    def export_json(
        self,
        pretty: bool = False,
    ) -> str:

        snapshot = self.snapshot()

        if pretty:

            return json.dumps(
                snapshot,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        return json.dumps(
            snapshot,
            ensure_ascii=False,
            default=str,
        )

    # ========================================================
    # SNAPSHOT RESTORE
    # ========================================================

    def restore_documents(
        self,
        documents: Iterable[Any],
    ) -> int:

        self.clear(
            keep_history=False
        )

        return self.add_many(
            documents
        )

    # ========================================================
    # INDEX MEMORY ESTIMATE
    # ========================================================

    def memory_estimate(
        self,
    ) -> Dict[str, Any]:

        document_count = len(
            self.documents
        )

        term_count = len(
            self.inverted
        )

        posting_count = sum(
            len(postings)
            for postings
            in self.postings.values()
        )

        field_entries = sum(
            len(term_map)
            for term_map
            in self.field_indexes.values()
        )

        return {
            "documents": document_count,
            "unique_terms": term_count,
            "postings": posting_count,
            "field_term_entries": (
                field_entries
            ),
            "prefix_entries": len(
                self.prefix_index
            ),
            "metadata_fields": len(
                self.metadata_indexes
            ),
            "categories": len(
                self.category_index
            ),
            "tags": len(
                self.tag_index
            ),
        }

    # ========================================================
    # DOCUMENT FREQUENCY HELPERS
    # ========================================================

    def rare_terms(
        self,
        terms: Sequence[str],
        maximum_frequency: int = 3,
    ) -> List[str]:

        return [
            term
            for term in terms
            if self.document_frequency.get(
                self._normalize_value(
                    term
                ),
                0,
            )
            <= maximum_frequency
        ]

    def common_terms(
        self,
        terms: Sequence[str],
        minimum_frequency: int = 10,
    ) -> List[str]:

        return [
            term
            for term in terms
            if self.document_frequency.get(
                self._normalize_value(
                    term
                ),
                0,
            )
            >= minimum_frequency
        ]

    # ========================================================
    # IDF
    # ========================================================

    def idf(
        self,
        term: str,
    ) -> float:

        document_count = (
            self.statistics.active_document_count
        )

        if document_count <= 0:

            return 0.0

        frequency = (
            self.document_frequency.get(
                self._normalize_value(
                    term
                ),
                0,
            )
        )

        numerator = (
            document_count
            - frequency
            + 0.5
        )

        denominator = (
            frequency
            + 0.5
        )

        if denominator <= 0:

            return 0.0

        return max(
            0.0,
            math.log(
                1.0
                + (
                    numerator
                    / denominator
                )
            ),
        )

    # ========================================================
    # RANKING ENGINE INTEGRATION
    # ========================================================

    def ranking_statistics(
        self,
    ) -> Dict[str, Any]:

        return {
            "document_count": (
                self.statistics.active_document_count
            ),
            "document_frequency": dict(
                self.document_frequency
            ),
            "average_document_length": (
                self.statistics
                .average_document_length
            ),
            "total_terms": (
                self.statistics.total_terms
            ),
            "unique_terms": (
                self.statistics.unique_terms
            ),
        }

    def configure_ranking_engine(
        self,
        ranking_engine: Any,
    ):

        if not hasattr(
            ranking_engine,
            "set_index_statistics",
        ):

            raise TypeError(
                "Ranking engine does not expose "
                "set_index_statistics()."
            )

        ranking_engine.set_index_statistics(
            document_count=(
                self.statistics.active_document_count
            ),
            document_frequency=(
                self.document_frequency
            ),
            average_document_length=(
                self.statistics
                .average_document_length
                or 1.0
            ),
        )

    # ========================================================
    # QUERY TERM ANALYSIS
    # ========================================================

    def analyze_terms(
        self,
        terms: Sequence[str],
    ) -> List[
        Dict[str, Any]
    ]:

        analysis = []

        for raw_term in terms:

            term = self._normalize_value(
                raw_term
            )

            frequency = (
                self.document_frequency.get(
                    term,
                    0,
                )
            )

            analysis.append(
                {
                    "term": term,
                    "document_frequency": (
                        frequency
                    ),
                    "collection_frequency": (
                        self.collection_frequency.get(
                            term,
                            0,
                        )
                    ),
                    "idf": self.idf(
                        term
                    ),
                    "exists": (
                        frequency > 0
                    ),
                    "rare": (
                        frequency <= 3
                    ),
                }
            )

        return analysis

    # ========================================================
    # RELATED TERMS
    # ========================================================

    def related_terms(
        self,
        term: str,
        limit: int = 20,
    ) -> List[
        Tuple[str, int]
    ]:

        term = self._normalize_value(
            term
        )

        documents = (
            self.documents_for_term(
                term
            )
        )

        if not documents:

            return []

        cooccurrence = Counter()

        for document_id in documents:

            record = self.documents.get(
                document_id
            )

            if record is None:
                continue

            for candidate in (
                record.term_frequencies
            ):

                if candidate == term:
                    continue

                cooccurrence[
                    candidate
                ] += 1

        return cooccurrence.most_common(
            max(
                0,
                int(limit),
            )
        )

    # ========================================================
    # SIMILAR DOCUMENT CANDIDATES
    # ========================================================

    def similar_candidates(
        self,
        document_id: Any,
        limit: int = 20,
    ) -> List[
        Tuple[Any, float]
    ]:

        record = self.documents.get(
            document_id
        )

        if record is None:

            return []

        scores = Counter()

        source_terms = set(
            record.term_frequencies
        )

        for term in source_terms:

            for candidate_id in (
                self.documents_for_term(
                    term
                )
            ):

                if (
                    candidate_id
                    == document_id
                ):

                    continue

                scores[
                    candidate_id
                ] += self.idf(
                    term
                )

        return scores.most_common(
            max(
                0,
                int(limit),
            )
        )

    # ========================================================
    # COLLECTION FILTERING
    # ========================================================

    def filter_documents(
        self,
        document_ids: Iterable[Any],
        filters: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> Set[Any]:

        result = set(
            document_ids
        )

        if not filters:

            return result

        for field_name, expected in (
            filters.items()
        ):

            if field_name == "category":

                result.intersection_update(
                    self.documents_for_category(
                        str(expected)
                    )
                )

                continue

            if field_name == "tag":

                result.intersection_update(
                    self.documents_for_tag(
                        str(expected)
                    )
                )

                continue

            values = expected

            if not isinstance(
                values,
                (list, tuple, set),
            ):

                values = [
                    values
                ]

            field_matches = self.union(
                self.documents_for_metadata(
                    field_name,
                    value,
                )
                for value in values
            )

            result.intersection_update(
                field_matches
            )

        return result

    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    def debug_info(
        self,
    ) -> Dict[str, Any]:

        return {
            "index_version": INDEX_VERSION,
            "generation": self.generation,
            "dirty": self.dirty,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "fields": list(
                self.fields
            ),
            "statistics": (
                self.get_statistics()
            ),
            "memory_estimate": (
                self.memory_estimate()
            ),
            "history_size": len(
                self.change_history
            ),
            "validation": self.validate(),
        }


# ============================================================
# DEFAULT INDEX
# ============================================================


index = SearchIndex()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def add(
    document: Any,
) -> DocumentRecord:

    return index.add(
        document
    )


def add_many(
    documents: Iterable[Any],
) -> int:

    return index.add_many(
        documents
    )


def update(
    document_id: Any,
    document: Any,
) -> DocumentRecord:

    return index.update(
        document_id,
        document,
    )


def patch(
    document_id: Any,
    changes: Mapping[str, Any],
) -> Optional[
    DocumentRecord
]:

    return index.patch(
        document_id,
        changes,
    )


def remove(
    document_id: Any,
) -> bool:

    return index.remove(
        document_id
    )


def get(
    document_id: Any,
) -> Optional[
    DocumentRecord
]:

    return index.get(
        document_id
    )


def search_candidates(
    terms: Sequence[str],
    required_terms: Optional[
        Sequence[str]
    ] = None,
    excluded_terms: Optional[
        Sequence[str]
    ] = None,
    mode: str = "or",
) -> SearchCandidates:

    return index.generate_candidates(
        terms=terms,
        required_terms=required_terms,
        excluded_terms=excluded_terms,
        mode=mode,
    )


def statistics() -> Dict[str, Any]:

    return index.get_statistics()


def validate() -> Dict[str, Any]:

    return index.validate()


def rebuild(
    documents: Optional[
        Iterable[Any]
    ] = None,
) -> Dict[str, Any]:

    return index.rebuild(
        documents
    )


def debug_info() -> Dict[str, Any]:

    return index.debug_info()