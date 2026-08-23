"""
OurPlatform Search Document Processor
=====================================

Document ingestion and preparation layer.

Pipeline:

    RAW DOCUMENT
          |
          v
    Validation
          |
          v
    Field normalization
          |
          v
    Text extraction
          |
          v
    Unicode normalization
          |
          v
    Whitespace cleanup
          |
          v
    Structural analysis
          |
          v
    Metadata extraction
          |
          v
    Duplicate detection
          |
          v
    Chunking
          |
          v
    Search-ready records
          |
          v
        INDEX

This module deliberately sits BEFORE index.py.

Its job is not to rank documents.
Its job is to make documents consistently searchable.

Design goals
------------

- Accept many document shapes
- Normalize inconsistent input
- Preserve original information
- Extract searchable text
- Extract metadata
- Generate stable document fingerprints
- Detect duplicate content
- Detect near-duplicate content
- Split large documents into chunks
- Preserve parent/child relationships
- Estimate document quality
- Detect language hints
- Detect links
- Detect headings
- Detect code-like content
- Detect structured fields
- Support incremental processing
- Support batch processing
- Support configurable pipelines
- Provide diagnostics
- Avoid destructive transformations
- Remain independent from the index implementation
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import unicodedata

from collections import Counter, defaultdict
from dataclasses import (
    dataclass,
    field,
    asdict,
)
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ============================================================
# VERSION
# ============================================================

PROCESSOR_VERSION = "1.0.0"


# ============================================================
# LIMITS
# ============================================================

DEFAULT_MAX_DOCUMENT_LENGTH = 2_000_000

DEFAULT_CHUNK_SIZE = 900

DEFAULT_CHUNK_OVERLAP = 120

DEFAULT_MIN_CHUNK_SIZE = 80

DEFAULT_MAX_CHUNKS = 10_000

DEFAULT_MAX_METADATA_FIELDS = 128

DEFAULT_MAX_TITLE_LENGTH = 500

DEFAULT_MAX_TAGS = 128

DEFAULT_MAX_TAG_LENGTH = 200

DEFAULT_MAX_LINKS = 1_000

DEFAULT_MAX_HEADINGS = 500

DEFAULT_MAX_SENTENCES = 10_000

DEFAULT_MAX_TERMS = 20_000


# ============================================================
# REGEX
# ============================================================

WHITESPACE_RE = re.compile(
    r"\s+",
    re.UNICODE,
)

MULTI_NEWLINE_RE = re.compile(
    r"\n{3,}",
)

URL_RE = re.compile(
    r"""
    (?:
        https?://
        |
        www\.
    )
    [^\s<>"']+
    """,
    re.IGNORECASE | re.VERBOSE,
)

EMAIL_RE = re.compile(
    r"""
    [A-Z0-9._%+-]+
    @
    [A-Z0-9.-]+
    \.
    [A-Z]{2,}
    """,
    re.IGNORECASE | re.VERBOSE,
)

WORD_RE = re.compile(
    r"\b[\w'-]+\b",
    re.UNICODE,
)

HEADING_RE = re.compile(
    r"(?m)^\s*(#{1,6})\s+(.+?)\s*$"
)

CODE_FENCE_RE = re.compile(
    r"```.*?```",
    re.DOTALL,
)

HTML_TAG_RE = re.compile(
    r"<[^>]+>"
)

CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)

DUPLICATE_SPACE_RE = re.compile(
    r"[ \t]{2,}"
)

SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+"
)

NUMBER_RE = re.compile(
    r"\b\d+(?:[.,]\d+)*\b"
)

DATE_LIKE_RE = re.compile(
    r"""
    \b
    (?:
        \d{1,4}
        [-/]
        \d{1,2}
        [-/]
        \d{1,4}
    )
    \b
    """,
    re.VERBOSE,
)


# ============================================================
# ENUMS
# ============================================================


class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    REJECTED = "rejected"
    TRUNCATED = "truncated"
    ERROR = "error"


class DocumentType(str, Enum):
    UNKNOWN = "unknown"
    TEXT = "text"
    ARTICLE = "article"
    CODE = "code"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    NOTE = "note"
    DOCUMENT = "document"


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class ProcessorConfig:
    """
    Controls document processing behavior.

    The processor is deliberately configurable rather than
    relying on hard-coded behavior.
    """

    max_document_length: int = (
        DEFAULT_MAX_DOCUMENT_LENGTH
    )

    chunk_size: int = (
        DEFAULT_CHUNK_SIZE
    )

    chunk_overlap: int = (
        DEFAULT_CHUNK_OVERLAP
    )

    min_chunk_size: int = (
        DEFAULT_MIN_CHUNK_SIZE
    )

    max_chunks: int = (
        DEFAULT_MAX_CHUNKS
    )

    remove_html: bool = True

    normalize_unicode: bool = True

    normalize_whitespace: bool = True

    preserve_line_breaks: bool = True

    extract_metadata: bool = True

    detect_language: bool = True

    detect_code: bool = True

    detect_links: bool = True

    detect_headings: bool = True

    detect_duplicates: bool = True

    detect_near_duplicates: bool = True

    generate_chunks: bool = True

    keep_original_text: bool = True

    generate_term_statistics: bool = True

    generate_quality_score: bool = True

    lowercase_fingerprint: bool = True

    strip_urls_from_fingerprint: bool = False

    max_metadata_fields: int = (
        DEFAULT_MAX_METADATA_FIELDS
    )

    max_title_length: int = (
        DEFAULT_MAX_TITLE_LENGTH
    )

    max_tags: int = DEFAULT_MAX_TAGS

    max_tag_length: int = (
        DEFAULT_MAX_TAG_LENGTH
    )

    max_links: int = DEFAULT_MAX_LINKS

    max_headings: int = (
        DEFAULT_MAX_HEADINGS
    )

    max_sentences: int = (
        DEFAULT_MAX_SENTENCES
    )

    max_terms: int = (
        DEFAULT_MAX_TERMS
    )


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class DocumentStatistics:
    """
    Search-oriented document statistics.
    """

    character_count: int = 0

    word_count: int = 0

    unique_word_count: int = 0

    sentence_count: int = 0

    paragraph_count: int = 0

    line_count: int = 0

    heading_count: int = 0

    link_count: int = 0

    email_count: int = 0

    number_count: int = 0

    code_block_count: int = 0

    average_word_length: float = 0.0

    lexical_diversity: float = 0.0

    estimated_reading_time_minutes: float = 0.0

    quality_score: float = 0.0

    language_hint: str = "unknown"

    document_type: str = (
        DocumentType.UNKNOWN.value
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentChunk:
    """
    A searchable section of a larger document.
    """

    chunk_id: str

    parent_id: str

    index: int

    text: str

    start_offset: int

    end_offset: int

    token_estimate: int

    heading: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:

        return asdict(self)


@dataclass
class ProcessedDocument:
    """
    Fully processed search-ready document.
    """

    document_id: str

    title: str

    text: str

    original_text: Optional[str]

    document_type: str

    metadata: Dict[str, Any]

    tags: List[str]

    links: List[str]

    headings: List[str]

    statistics: DocumentStatistics

    fingerprint: str

    normalized_fingerprint: str

    chunks: List[DocumentChunk]

    parent_id: Optional[str] = None

    status: str = (
        ProcessingStatus.SUCCESS.value
    )

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    processed_at: str = ""

    processor_version: str = (
        PROCESSOR_VERSION
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "document_id": self.document_id,
            "title": self.title,
            "text": self.text,
            "original_text": self.original_text,
            "document_type": self.document_type,
            "metadata": dict(
                self.metadata
            ),
            "tags": list(
                self.tags
            ),
            "links": list(
                self.links
            ),
            "headings": list(
                self.headings
            ),
            "statistics": (
                self.statistics.to_dict()
            ),
            "fingerprint": self.fingerprint,
            "normalized_fingerprint": (
                self.normalized_fingerprint
            ),
            "chunks": [
                chunk.to_dict()
                for chunk in self.chunks
            ],
            "parent_id": self.parent_id,
            "status": self.status,
            "warnings": list(
                self.warnings
            ),
            "errors": list(
                self.errors
            ),
            "processed_at": (
                self.processed_at
            ),
            "processor_version": (
                self.processor_version
            ),
        }


@dataclass
class ProcessingResult:
    """
    Result of processing one or more documents.
    """

    documents: List[
        ProcessedDocument
    ] = field(
        default_factory=list
    )

    rejected: int = 0

    errors: int = 0

    warnings: int = 0

    duplicates: int = 0

    chunks_created: int = 0

    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:

        return {
            "documents": [
                document.to_dict()
                for document in self.documents
            ],
            "accepted": len(
                self.documents
            ),
            "rejected": self.rejected,
            "errors": self.errors,
            "warnings": self.warnings,
            "duplicates": self.duplicates,
            "chunks_created": (
                self.chunks_created
            ),
            "processing_time_ms": (
                self.processing_time_ms
            ),
        }


# ============================================================
# DUPLICATE REGISTRY
# ============================================================


class FingerprintRegistry:
    """
    Tracks fingerprints for exact and approximate duplicate
    detection.

    This is deliberately independent of the index so duplicate
    handling can evolve without changing indexing code.
    """

    def __init__(self):

        self._fingerprints = {}

        self._normalized = {}

        self._lock = threading.RLock()

    def add(
        self,
        document_id: str,
        fingerprint: str,
        normalized_fingerprint: str,
    ):

        with self._lock:

            self._fingerprints[
                fingerprint
            ] = document_id

            self._normalized[
                normalized_fingerprint
            ] = document_id

    def exact_duplicate(
        self,
        fingerprint: str,
    ) -> Optional[str]:

        with self._lock:

            return self._fingerprints.get(
                fingerprint
            )

    def normalized_duplicate(
        self,
        fingerprint: str,
    ) -> Optional[str]:

        with self._lock:

            return self._normalized.get(
                fingerprint
            )

    def remove(
        self,
        document_id: str,
    ):

        with self._lock:

            for mapping in (
                self._fingerprints,
                self._normalized,
            ):

                keys = [
                    key
                    for key, value
                    in mapping.items()
                    if value == document_id
                ]

                for key in keys:

                    del mapping[key]

    def clear(self):

        with self._lock:

            self._fingerprints.clear()

            self._normalized.clear()

    def size(self) -> int:

        with self._lock:

            return len(
                self._fingerprints
            )


# ============================================================
# DOCUMENT PROCESSOR
# ============================================================


class DocumentProcessor:
    """
    Main document processing engine.

    It accepts flexible input and produces deterministic,
    search-ready documents.
    """

    def __init__(
        self,
        config: Optional[
            ProcessorConfig
        ] = None,
    ):

        self.config = (
            config
            or ProcessorConfig()
        )

        self.registry = (
            FingerprintRegistry()
        )

        self.statistics = Counter()

        self._lock = threading.RLock()

    # ========================================================
    # CLOCK
    # ========================================================

    @staticmethod
    def _now() -> str:

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def validate_input(
        self,
        document: Any,
    ) -> Tuple[
        bool,
        List[str],
    ]:

        errors = []

        if document is None:

            errors.append(
                "Document cannot be None."
            )

            return False, errors

        if isinstance(
            document,
            str,
        ):

            if not document.strip():

                errors.append(
                    "Document text is empty."
                )

            return (
                len(errors) == 0,
                errors,
            )

        if isinstance(
            document,
            Mapping,
        ):

            if not document:

                errors.append(
                    "Document mapping is empty."
                )

                return False, errors

            has_content = any(
                key in document
                for key in (
                    "text",
                    "content",
                    "body",
                    "title",
                    "data",
                )
            )

            if not has_content:

                errors.append(
                    "Document has no recognized content field."
                )

        else:

            errors.append(
                "Document must be text or a mapping."
            )

        return (
            len(errors) == 0,
            errors,
        )

    # ========================================================
    # DOCUMENT ID
    # ========================================================

    def generate_document_id(
        self,
        document: Any,
    ) -> str:

        if isinstance(
            document,
            Mapping,
        ):

            for key in (
                "id",
                "document_id",
                "uuid",
                "key",
            ):

                value = document.get(
                    key
                )

                if value is not None:

                    return str(
                        value
                    )

        raw = repr(
            document
        )

        return (
            "doc_"
            + hashlib.sha256(
                raw.encode(
                    "utf-8",
                    errors="replace",
                )
            ).hexdigest()[:24]
        )

    # ========================================================
    # FIELD EXTRACTION
    # ========================================================

    def extract_fields(
        self,
        document: Any,
    ) -> Dict[str, Any]:

        if isinstance(
            document,
            str,
        ):

            return {
                "text": document
            }

        if not isinstance(
            document,
            Mapping,
        ):

            return {
                "text": str(
                    document
                )
            }

        fields = dict(
            document
        )

        text = None

        for key in (
            "text",
            "content",
            "body",
            "description",
            "value",
        ):

            if fields.get(
                key
            ) is not None:

                text = fields.get(
                    key
                )

                break

        if text is None:

            data = fields.get(
                "data"
            )

            if isinstance(
                data,
                str,
            ):

                text = data

            elif data is not None:

                text = self._flatten_value(
                    data
                )

        fields["text"] = (
            ""
            if text is None
            else str(
                text
            )
        )

        return fields

    # ========================================================
    # VALUE FLATTENING
    # ========================================================

    def _flatten_value(
        self,
        value: Any,
    ) -> str:

        if value is None:

            return ""

        if isinstance(
            value,
            str,
        ):

            return value

        if isinstance(
            value,
            Mapping,
        ):

            parts = []

            for key, item in (
                value.items()
            ):

                parts.append(
                    str(key)
                )

                parts.append(
                    self._flatten_value(
                        item
                    )
                )

            return " ".join(
                parts
            )

        if isinstance(
            value,
            (list, tuple, set),
        ):

            return " ".join(
                self._flatten_value(
                    item
                )
                for item in value
            )

        return str(
            value
        )

    # ========================================================
    # TITLE EXTRACTION
    # ========================================================

    def extract_title(
        self,
        fields: Mapping[str, Any],
        text: str,
    ) -> str:

        for key in (
            "title",
            "name",
            "subject",
            "heading",
        ):

            value = fields.get(
                key
            )

            if value:

                title = str(
                    value
                ).strip()

                return title[
                    :self.config.max_title_length
                ]

        headings = self.extract_headings(
            text
        )

        if headings:

            return headings[0][
                :self.config.max_title_length
            ]

        first_line = (
            text.strip()
            .splitlines()
        )

        if first_line:

            return first_line[0][
                :self.config.max_title_length
            ]

        return ""

    # ========================================================
    # TAG EXTRACTION
    # ========================================================

    def extract_tags(
        self,
        fields: Mapping[str, Any],
    ) -> List[str]:

        raw = (
            fields.get(
                "tags"
            )
            or fields.get(
                "keywords"
            )
            or []
        )

        if isinstance(
            raw,
            str,
        ):

            values = re.split(
                r"[,;]",
                raw,
            )

        elif isinstance(
            raw,
            Iterable,
        ):

            values = list(
                raw
            )

        else:

            values = [
                raw
            ]

        result = []

        seen = set()

        for value in values:

            tag = str(
                value
            ).strip()

            if not tag:

                continue

            tag = tag[
                :self.config.max_tag_length
            ]

            normalized = (
                tag.casefold()
            )

            if normalized in seen:

                continue

            seen.add(
                normalized
            )

            result.append(
                tag
            )

            if len(
                result
            ) >= self.config.max_tags:

                break

        return result

    # ========================================================
    # HTML CLEANING
    # ========================================================

    def clean_html(
        self,
        text: str,
    ) -> str:

        text = re.sub(
            r"(?is)<script.*?</script>",
            " ",
            text,
        )

        text = re.sub(
            r"(?is)<style.*?</style>",
            " ",
            text,
        )

        text = HTML_TAG_RE.sub(
            " ",
            text,
        )

        return text

    # ========================================================
    # UNICODE NORMALIZATION
    # ========================================================

    def normalize_unicode(
        self,
        text: str,
    ) -> str:

        return unicodedata.normalize(
            "NFKC",
            text,
        )

    # ========================================================
    # CONTROL CHARACTERS
    # ========================================================

    def remove_control_characters(
        self,
        text: str,
    ) -> str:

        return CONTROL_CHAR_RE.sub(
            " ",
            text,
        )

    # ========================================================
    # WHITESPACE
    # ========================================================

    def normalize_whitespace(
        self,
        text: str,
    ) -> str:

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        lines = [
            DUPLICATE_SPACE_RE.sub(
                " ",
                line,
            ).strip()
            for line
            in text.split("\n")
        ]

        text = "\n".join(
            lines
        )

        text = MULTI_NEWLINE_RE.sub(
            "\n\n",
            text,
        )

        if not self.config.preserve_line_breaks:

            text = WHITESPACE_RE.sub(
                " ",
                text,
            )

        return text.strip()

    # ========================================================
    # FULL TEXT NORMALIZATION
    # ========================================================

    def normalize_text(
        self,
        text: str,
    ) -> Tuple[
        str,
        List[str],
    ]:

        warnings = []

        if len(
            text
        ) > self.config.max_document_length:

            text = text[
                :self.config.max_document_length
            ]

            warnings.append(
                "Document exceeded maximum length and was truncated."
            )

        if self.config.remove_html:

            text = self.clean_html(
                text
            )

        if self.config.normalize_unicode:

            text = self.normalize_unicode(
                text
            )

        text = (
            self.remove_control_characters(
                text
            )
        )

        if self.config.normalize_whitespace:

            text = self.normalize_whitespace(
                text
            )

        return text, warnings

    # ========================================================
    # HEADINGS
    # ========================================================

    def extract_headings(
        self,
        text: str,
    ) -> List[str]:

        headings = []

        for match in HEADING_RE.finditer(
            text
        ):

            heading = match.group(
                2
            ).strip()

            if not heading:

                continue

            headings.append(
                heading
            )

            if len(
                headings
            ) >= self.config.max_headings:

                break

        return headings

    # ========================================================
    # LINKS
    # ========================================================

    def extract_links(
        self,
        text: str,
    ) -> List[str]:

        if not self.config.detect_links:

            return []

        links = []

        seen = set()

        for match in URL_RE.finditer(
            text
        ):

            value = match.group(
                0
            ).rstrip(
                ".,;:!?)]}"
            )

            if value in seen:

                continue

            seen.add(
                value
            )

            links.append(
                value
            )

            if len(
                links
            ) >= self.config.max_links:

                break

        return links

    # ========================================================
    # EMAILS
    # ========================================================

    def extract_emails(
        self,
        text: str,
    ) -> List[str]:

        return list(
            dict.fromkeys(
                match.group(
                    0
                )
                for match
                in EMAIL_RE.finditer(
                    text
                )
            )
        )

    # ========================================================
    # SENTENCES
    # ========================================================

    def split_sentences(
        self,
        text: str,
    ) -> List[str]:

        if not text.strip():

            return []

        sentences = []

        for paragraph in text.split(
            "\n"
        ):

            paragraph = paragraph.strip()

            if not paragraph:

                continue

            parts = SENTENCE_RE.split(
                paragraph
            )

            for part in parts:

                part = part.strip()

                if part:

                    sentences.append(
                        part
                    )

                if len(
                    sentences
                ) >= self.config.max_sentences:

                    return sentences

        return sentences

    # ========================================================
    # WORD EXTRACTION
    # ========================================================

    def extract_words(
        self,
        text: str,
    ) -> List[str]:

        return [
            match.group(
                0
            )
            for match
            in WORD_RE.finditer(
                text
            )
        ]

    # ========================================================
    # TERM STATISTICS
    # ========================================================

    def term_statistics(
        self,
        text: str,
    ) -> Dict[str, Any]:

        words = self.extract_words(
            text
        )

        normalized = [
            word.casefold()
            for word in words
        ]

        frequencies = Counter(
            normalized
        )

        if len(
            frequencies
        ) > self.config.max_terms:

            frequencies = Counter(
                dict(
                    frequencies.most_common(
                        self.config.max_terms
                    )
                )
            )

        return {
            "term_frequencies": dict(
                frequencies
            ),
            "top_terms": [
                {
                    "term": term,
                    "frequency": count,
                }
                for term, count
                in frequencies.most_common(
                    25
                )
            ],
        }

    # ========================================================
    # DOCUMENT TYPE
    # ========================================================

    def detect_document_type(
        self,
        fields: Mapping[str, Any],
        text: str,
    ) -> DocumentType:

        explicit = fields.get(
            "document_type"
        ) or fields.get(
            "type"
        )

        if explicit:

            value = str(
                explicit
            ).lower()

            for item in DocumentType:

                if item.value == value:

                    return item

        if (
            "```"
            in text
        ):

            return DocumentType.CODE

        if (
            "<html"
            in text.lower()
        ):

            return DocumentType.HTML

        if (
            text.lstrip().startswith(
                "# "
            )
            or HEADING_RE.search(
                text
            )
        ):

            return DocumentType.MARKDOWN

        if (
            text.count(",") >= 3
            and "\n" in text
        ):

            return DocumentType.CSV

        return DocumentType.TEXT

    # ========================================================
    # CODE DETECTION
    # ========================================================

    def detect_code(
        self,
        text: str,
    ) -> Dict[str, Any]:

        fence_count = text.count(
            "```"
        )

        code_blocks = list(
            CODE_FENCE_RE.finditer(
                text
            )
        )

        indicators = 0

        patterns = (
            r"\bdef\s+\w+\s*\(",
            r"\bclass\s+\w+",
            r"\bfunction\s+\w+",
            r"\bimport\s+\w+",
            r"\bfrom\s+\w+\s+import",
            r"\bconst\s+\w+\s*=",
            r"\bSELECT\s+.+\s+FROM\b",
            r"\bpublic\s+class\b",
            r"#include\s+<",
        )

        for pattern in patterns:

            if re.search(
                pattern,
                text,
                re.IGNORECASE,
            ):

                indicators += 1

        is_code = (
            fence_count >= 2
            or indicators >= 2
        )

        return {
            "is_code": is_code,
            "code_blocks": len(
                code_blocks
            ),
            "code_indicators": indicators,
        }

    # ========================================================
    # LANGUAGE HEURISTIC
    # ========================================================

    def detect_language(
        self,
        text: str,
    ) -> str:

        if not text.strip():

            return "unknown"

        sample = text[
            :5000
        ].casefold()

        language_markers = {
            "en": (
                " the ",
                " and ",
                " of ",
                " to ",
                " is ",
                " in ",
            ),
            "es": (
                " el ",
                " la ",
                " de ",
                " que ",
                " los ",
                " las ",
            ),
            "fr": (
                " le ",
                " la ",
                " de ",
                " les ",
                " des ",
                " une ",
            ),
            "de": (
                " der ",
                " die ",
                " das ",
                " und ",
                " ist ",
                " den ",
            ),
            "it": (
                " il ",
                " la ",
                " di ",
                " che ",
                " gli ",
                " una ",
            ),
        }

        scores = {}

        padded = (
            " "
            + sample
            + " "
        )

        for language, markers in (
            language_markers.items()
        ):

            scores[language] = sum(
                padded.count(
                    marker
                )
                for marker in markers
            )

        if not scores:

            return "unknown"

        language, score = max(
            scores.items(),
            key=lambda item: item[1],
        )

        if score <= 0:

            return "unknown"

        return language

    # ========================================================
    # QUALITY SCORE
    # ========================================================

    def quality_score(
        self,
        text: str,
        statistics: DocumentStatistics,
    ) -> float:

        if not text.strip():

            return 0.0

        score = 0.0

        # Length contribution.

        if statistics.word_count >= 20:

            score += 0.20

        elif statistics.word_count >= 5:

            score += 0.10

        # Lexical diversity.

        score += min(
            0.25,
            statistics.lexical_diversity
            * 0.25,
        )

        # Sentence structure.

        if statistics.sentence_count >= 2:

            score += 0.15

        # Reasonable average word length.

        if (
            2.5
            <= statistics.average_word_length
            <= 12
        ):

            score += 0.10

        # Very short or extremely repetitive
        # documents receive less confidence.

        if (
            statistics.word_count > 0
            and statistics.unique_word_count
            / statistics.word_count
            > 0.15
        ):

            score += 0.15

        # Headings provide useful structure.

        if statistics.heading_count > 0:

            score += 0.05

        # Penalize extremely short documents.

        if statistics.character_count < 40:

            score *= 0.5

        return round(
            min(
                1.0,
                max(
                    0.0,
                    score,
                ),
            ),
            4,
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    def calculate_statistics(
        self,
        text: str,
        headings: Sequence[str],
        links: Sequence[str],
        document_type: DocumentType,
    ) -> DocumentStatistics:

        words = self.extract_words(
            text
        )

        normalized_words = [
            word.casefold()
            for word in words
        ]

        unique_words = set(
            normalized_words
        )

        sentences = (
            self.split_sentences(
                text
            )
        )

        paragraphs = [
            paragraph
            for paragraph
            in re.split(
                r"\n\s*\n",
                text,
            )
            if paragraph.strip()
        ]

        character_count = len(
            text
        )

        word_count = len(
            words
        )

        unique_count = len(
            unique_words
        )

        average_word_length = (
            sum(
                len(word)
                for word in words
            )
            / word_count
            if word_count
            else 0.0
        )

        lexical_diversity = (
            unique_count
            / word_count
            if word_count
            else 0.0
        )

        reading_time = (
            word_count
            / 200.0
        )

        stats = DocumentStatistics(
            character_count=character_count,
            word_count=word_count,
            unique_word_count=unique_count,
            sentence_count=len(
                sentences
            ),
            paragraph_count=len(
                paragraphs
            ),
            line_count=len(
                text.splitlines()
            ),
            heading_count=len(
                headings
            ),
            link_count=len(
                links
            ),
            email_count=len(
                self.extract_emails(
                    text
                )
            ),
            number_count=len(
                NUMBER_RE.findall(
                    text
                )
            ),
            code_block_count=len(
                CODE_FENCE_RE.findall(
                    text
                )
            ),
            average_word_length=round(
                average_word_length,
                3,
            ),
            lexical_diversity=round(
                lexical_diversity,
                4,
            ),
            estimated_reading_time_minutes=round(
                reading_time,
                2,
            ),
            language_hint=(
                self.detect_language(
                    text
                )
                if self.config.detect_language
                else "unknown"
            ),
            document_type=(
                document_type.value
            ),
        )

        stats.quality_score = (
            self.quality_score(
                text,
                stats,
            )
            if self.config.generate_quality_score
            else 0.0
        )

        return stats

    # ========================================================
    # FINGERPRINTING
    # ========================================================

    def fingerprint(
        self,
        text: str,
    ) -> str:

        value = text

        if self.config.lowercase_fingerprint:

            value = value.casefold()

        if self.config.strip_urls_from_fingerprint:

            value = URL_RE.sub(
                "",
                value,
            )

        value = WHITESPACE_RE.sub(
            " ",
            value,
        ).strip()

        return hashlib.sha256(
            value.encode(
                "utf-8",
                errors="replace",
            )
        ).hexdigest()

    # ========================================================
    # NORMALIZED FINGERPRINT
    # ========================================================

    def normalized_fingerprint(
        self,
        text: str,
    ) -> str:

        normalized, _ = (
            self.normalize_text(
                text
            )
        )

        words = [
            word.casefold()
            for word
            in self.extract_words(
                normalized
            )
        ]

        # Sorting destroys word order intentionally.
        # This fingerprint is for detecting highly similar
        # content whose formatting/order differs slightly.

        normalized_words = " ".join(
            sorted(
                words
            )
        )

        return hashlib.sha256(
            normalized_words.encode(
                "utf-8",
                errors="replace",
            )
        ).hexdigest()

    # ========================================================
    # SIMILARITY FINGERPRINT
    # ========================================================

    def shingle_fingerprint(
        self,
        text: str,
        size: int = 5,
    ) -> Set[int]:

        words = [
            word.casefold()
            for word
            in self.extract_words(
                text
            )
        ]

        if len(
            words
        ) < size:

            return {
                hash(
                    " ".join(
                        words
                    )
                )
            }

        shingles = set()

        for index in range(
            len(words)
            - size
            + 1
        ):

            shingle = " ".join(
                words[
                    index:
                    index + size
                ]
            )

            shingles.add(
                hash(
                    shingle
                )
            )

        return shingles

    def jaccard_similarity(
        self,
        first: Set[int],
        second: Set[int],
    ) -> float:

        if not first and not second:

            return 1.0

        if not first or not second:

            return 0.0

        intersection = len(
            first
            & second
        )

        union = len(
            first
            | second
        )

        return (
            intersection
            / union
            if union
            else 0.0
        )

    # ========================================================
    # CHUNKING
    # ========================================================

    def chunk_text(
        self,
        text: str,
        document_id: str,
        headings: Optional[
            Sequence[str]
        ] = None,
    ) -> List[DocumentChunk]:

        if not self.config.generate_chunks:

            return []

        if not text.strip():

            return []

        chunk_size = max(
            100,
            self.config.chunk_size,
        )

        overlap = min(
            max(
                0,
                self.config.chunk_overlap,
            ),
            chunk_size // 2,
        )

        chunks = []

        start = 0

        chunk_index = 0

        text_length = len(
            text
        )

        heading_list = list(
            headings or []
        )

        while (
            start < text_length
            and chunk_index
            < self.config.max_chunks
        ):

            proposed_end = min(
                text_length,
                start + chunk_size,
            )

            end = proposed_end

            # ------------------------------------------------
            # Prefer sentence boundaries.
            # ------------------------------------------------

            if end < text_length:

                boundary_region = text[
                    start:end
                ]

                sentence_breaks = [
                    match.end()
                    for match
                    in re.finditer(
                        r"[.!?]\s+",
                        boundary_region,
                    )
                ]

                if sentence_breaks:

                    end = (
                        start
                        + sentence_breaks[-1]
                    )

                else:

                    whitespace_breaks = [
                        match.start()
                        for match
                        in re.finditer(
                            r"\s+",
                            boundary_region,
                        )
                    ]

                    if whitespace_breaks:

                        end = (
                            start
                            + whitespace_breaks[-1]
                        )

            if end <= start:

                end = min(
                    text_length,
                    start + chunk_size,
                )

            chunk_text = text[
                start:end
            ].strip()

            if (
                len(chunk_text)
                >= self.config.min_chunk_size
                or not chunks
            ):

                chunk_id = (
                    f"{document_id}"
                    f":chunk:{chunk_index}"
                )

                chunk_heading = (
                    self._heading_for_offset(
                        heading_list,
                        chunk_text,
                    )
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        parent_id=document_id,
                        index=chunk_index,
                        text=chunk_text,
                        start_offset=start,
                        end_offset=end,
                        token_estimate=(
                            self.estimate_tokens(
                                chunk_text
                            )
                        ),
                        heading=chunk_heading,
                        fingerprint=(
                            self.fingerprint(
                                chunk_text
                            )
                        ),
                    )
                )

                chunk_index += 1

            if end >= text_length:

                break

            next_start = (
                end - overlap
            )

            if next_start <= start:

                next_start = end

            start = next_start

        return chunks

    # ========================================================
    # HEADING ASSOCIATION
    # ========================================================

    def _heading_for_offset(
        self,
        headings: Sequence[str],
        chunk_text: str,
    ) -> Optional[str]:

        if not headings:

            return None

        lowered = chunk_text.casefold()

        for heading in reversed(
            headings
        ):

            if heading.casefold() in lowered:

                return heading

        return headings[0]

    # ========================================================
    # TOKEN ESTIMATION
    # ========================================================

    def estimate_tokens(
        self,
        text: str,
    ) -> int:

        if not text:

            return 0

        # Approximation only.
        #
        # A real tokenizer can replace this later.

        return max(
            1,
            int(
                len(text)
                / 4
            ),
        )

    # ========================================================
    # METADATA
    # ========================================================

    def extract_metadata(
        self,
        fields: Mapping[str, Any],
        statistics: DocumentStatistics,
        links: Sequence[str],
        tags: Sequence[str],
    ) -> Dict[str, Any]:

        metadata = {}

        reserved = {
            "text",
            "content",
            "body",
            "data",
            "title",
        }

        for key, value in fields.items():

            if key in reserved:

                continue

            if len(
                metadata
            ) >= self.config.max_metadata_fields:

                break

            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                    type(None),
                ),
            ):

                metadata[
                    str(key)
                ] = value

        metadata.update(
            {
                "language": (
                    statistics.language_hint
                ),
                "document_type": (
                    statistics.document_type
                ),
                "quality_score": (
                    statistics.quality_score
                ),
                "word_count": (
                    statistics.word_count
                ),
                "character_count": (
                    statistics.character_count
                ),
                "reading_time_minutes": (
                    statistics.estimated_reading_time_minutes
                ),
                "link_count": len(
                    links
                ),
                "tag_count": len(
                    tags
                ),
            }
        )

        return metadata

    # ========================================================
    # PROCESS ONE DOCUMENT
    # ========================================================

    def process(
        self,
        document: Any,
        allow_duplicates: bool = False,
    ) -> Optional[
        ProcessedDocument
    ]:

        started = (
            datetime.now(
                timezone.utc
            )
        )

        self.statistics[
            "documents_received"
        ] += 1

        valid, errors = (
            self.validate_input(
                document
            )
        )

        if not valid:

            self.statistics[
                "documents_rejected"
            ] += 1

            return ProcessedDocument(
                document_id=(
                    self.generate_document_id(
                        document
                    )
                ),
                title="",
                text="",
                original_text=None,
                document_type=(
                    DocumentType.UNKNOWN.value
                ),
                metadata={},
                tags=[],
                links=[],
                headings=[],
                statistics=(
                    DocumentStatistics()
                ),
                fingerprint="",
                normalized_fingerprint="",
                chunks=[],
                status=(
                    ProcessingStatus.REJECTED.value
                ),
                errors=errors,
                processed_at=(
                    self._now()
                ),
            )

        fields = self.extract_fields(
            document
        )

        document_id = (
            self.generate_document_id(
                document
            )
        )

        original_text = str(
            fields.get(
                "text",
                "",
            )
        )

        text, warnings = (
            self.normalize_text(
                original_text
            )
        )

        if not text:

            self.statistics[
                "documents_empty"
            ] += 1

            return ProcessedDocument(
                document_id=document_id,
                title="",
                text="",
                original_text=(
                    original_text
                    if self.config.keep_original_text
                    else None
                ),
                document_type=(
                    DocumentType.UNKNOWN.value
                ),
                metadata={},
                tags=[],
                links=[],
                headings=[],
                statistics=(
                    DocumentStatistics()
                ),
                fingerprint="",
                normalized_fingerprint="",
                chunks=[],
                status=(
                    ProcessingStatus.EMPTY.value
                ),
                warnings=warnings,
                processed_at=(
                    self._now()
                ),
            )

        title = self.extract_title(
            fields,
            text,
        )

        tags = self.extract_tags(
            fields
        )

        headings = (
            self.extract_headings(
                text
            )
            if self.config.detect_headings
            else []
        )

        links = (
            self.extract_links(
                text
            )
            if self.config.detect_links
            else []
        )

        document_type = (
            self.detect_document_type(
                fields,
                text,
            )
        )

        statistics = (
            self.calculate_statistics(
                text,
                headings,
                links,
                document_type,
            )
        )

        if self.config.detect_code:

            code_info = (
                self.detect_code(
                    text
                )
            )

            if code_info[
                "is_code"
            ]:

                document_type = (
                    DocumentType.CODE
                )

                statistics.document_type = (
                    document_type.value
                )

                statistics.code_block_count = (
                    code_info[
                        "code_blocks"
                    ]
                )

        metadata = (
            self.extract_metadata(
                fields,
                statistics,
                links,
                tags,
            )
        )

        fingerprint = (
            self.fingerprint(
                text
            )
        )

        normalized_fingerprint = (
            self.normalized_fingerprint(
                text
            )
        )

        # ----------------------------------------------------
        # Duplicate detection
        # ----------------------------------------------------

        duplicate_id = (
            self.registry.exact_duplicate(
                fingerprint
            )
        )

        if (
            duplicate_id
            and duplicate_id != document_id
            and not allow_duplicates
        ):

            self.statistics[
                "duplicates_detected"
            ] += 1

            warnings.append(
                f"Exact duplicate of document {duplicate_id}."
            )

        normalized_duplicate = (
            self.registry.normalized_duplicate(
                normalized_fingerprint
            )
        )

        if (
            normalized_duplicate
            and normalized_duplicate
            != document_id
            and normalized_duplicate
            != duplicate_id
        ):

            warnings.append(
                "Content is normalized-equivalent to another document."
            )

        # ----------------------------------------------------
        # Chunk generation
        # ----------------------------------------------------

        chunks = self.chunk_text(
            text=text,
            document_id=document_id,
            headings=headings,
        )

        metadata[
            "chunk_count"
        ] = len(
            chunks
        )

        metadata[
            "fingerprint"
        ] = fingerprint

        metadata[
            "normalized_fingerprint"
        ] = normalized_fingerprint

        processed = ProcessedDocument(
            document_id=document_id,
            title=title,
            text=text,
            original_text=(
                original_text
                if self.config.keep_original_text
                else None
            ),
            document_type=(
                document_type.value
            ),
            metadata=metadata,
            tags=tags,
            links=links,
            headings=headings,
            statistics=statistics,
            fingerprint=fingerprint,
            normalized_fingerprint=(
                normalized_fingerprint
            ),
            chunks=chunks,
            status=(
                ProcessingStatus.SUCCESS.value
            ),
            warnings=warnings,
            processed_at=self._now(),
        )

        # ----------------------------------------------------
        # Register fingerprint.
        # ----------------------------------------------------

        if (
            not duplicate_id
            or allow_duplicates
        ):

            self.registry.add(
                document_id,
                fingerprint,
                normalized_fingerprint,
            )

        self.statistics[
            "documents_processed"
        ] += 1

        self.statistics[
            "chunks_created"
        ] += len(
            chunks
        )

        if warnings:

            self.statistics[
                "warnings"
            ] += len(
                warnings
            )

        elapsed = (
            datetime.now(
                timezone.utc
            )
            - started
        ).total_seconds()

        self.statistics[
            "processing_time_seconds"
        ] += elapsed

        return processed

    # ========================================================
    # BATCH PROCESSING
    # ========================================================

    def process_many(
        self,
        documents: Iterable[Any],
        allow_duplicates: bool = False,
    ) -> ProcessingResult:

        started = (
            datetime.now(
                timezone.utc
            )
        )

        result = ProcessingResult()

        for document in documents:

            try:

                processed = self.process(
                    document,
                    allow_duplicates=(
                        allow_duplicates
                    ),
                )

                if processed is None:

                    result.errors += 1

                    continue

                if (
                    processed.status
                    == ProcessingStatus.REJECTED.value
                ):

                    result.rejected += 1

                elif (
                    processed.status
                    == ProcessingStatus.ERROR.value
                ):

                    result.errors += 1

                else:

                    result.documents.append(
                        processed
                    )

                result.warnings += len(
                    processed.warnings
                )

                result.chunks_created += len(
                    processed.chunks
                )

            except Exception:

                result.errors += 1

                self.statistics[
                    "processing_errors"
                ] += 1

        elapsed = (
            datetime.now(
                timezone.utc
            )
            - started
        ).total_seconds()

        result.processing_time_ms = (
            elapsed * 1000.0
        )

        return result

    # ========================================================
    # STREAMING PROCESSING
    # ========================================================

    def process_stream(
        self,
        documents: Iterable[Any],
        allow_duplicates: bool = False,
    ) -> Iterator[
        ProcessedDocument
    ]:

        for document in documents:

            processed = self.process(
                document,
                allow_duplicates=(
                    allow_duplicates
                ),
            )

            if processed is None:

                continue

            if (
                processed.status
                == ProcessingStatus.SUCCESS.value
            ):

                yield processed

    # ========================================================
    # SEARCH INDEX FORMAT
    # ========================================================

    def to_index_record(
        self,
        document: ProcessedDocument,
    ) -> Dict[str, Any]:

        """
        Convert a processed document into a generic record
        suitable for index.py.

        This intentionally uses ordinary dictionaries so the
        processor does not become tightly coupled to one
        IndexRecord implementation.
        """

        record = {
            "id": document.document_id,
            "document_id": (
                document.document_id
            ),
            "title": document.title,
            "text": document.text,
            "content": document.text,
            "type": document.document_type,
            "tags": list(
                document.tags
            ),
            "links": list(
                document.links
            ),
            "headings": list(
                document.headings
            ),
            "metadata": dict(
                document.metadata
            ),
            "fingerprint": (
                document.fingerprint
            ),
            "normalized_fingerprint": (
                document.normalized_fingerprint
            ),
            "parent_id": (
                document.parent_id
            ),
            "statistics": (
                document.statistics.to_dict()
            ),
            "chunks": [
                chunk.to_dict()
                for chunk
                in document.chunks
            ],
        }

        return record

    # ========================================================
    # CHUNK INDEX RECORDS
    # ========================================================

    def chunk_records(
        self,
        document: ProcessedDocument,
    ) -> List[
        Dict[str, Any]
    ]:

        records = []

        for chunk in document.chunks:

            records.append(
                {
                    "id": chunk.chunk_id,
                    "document_id": (
                        chunk.chunk_id
                    ),
                    "parent_id": (
                        chunk.parent_id
                    ),
                    "text": chunk.text,
                    "content": chunk.text,
                    "title": (
                        document.title
                    ),
                    "heading": (
                        chunk.heading
                    ),
                    "chunk_index": (
                        chunk.index
                    ),
                    "start_offset": (
                        chunk.start_offset
                    ),
                    "end_offset": (
                        chunk.end_offset
                    ),
                    "token_estimate": (
                        chunk.token_estimate
                    ),
                    "fingerprint": (
                        chunk.fingerprint
                    ),
                    "metadata": dict(
                        chunk.metadata
                    ),
                }
            )

        return records

    # ========================================================
    # DUPLICATE CHECKING
    # ========================================================

    def is_duplicate(
        self,
        text: str,
    ) -> bool:

        fingerprint = (
            self.fingerprint(
                text
            )
        )

        return (
            self.registry.exact_duplicate(
                fingerprint
            )
            is not None
        )

    # ========================================================
    # SIMILARITY CHECK
    # ========================================================

    def similarity_to(
        self,
        first_text: str,
        second_text: str,
        shingle_size: int = 5,
    ) -> float:

        first = (
            self.shingle_fingerprint(
                first_text,
                shingle_size,
            )
        )

        second = (
            self.shingle_fingerprint(
                second_text,
                shingle_size,
            )
        )

        return self.jaccard_similarity(
            first,
            second,
        )

    # ========================================================
    # DOCUMENT SUMMARY
    # ========================================================

    def summarize(
        self,
        document: ProcessedDocument,
    ) -> Dict[str, Any]:

        return {
            "id": document.document_id,
            "title": document.title,
            "type": document.document_type,
            "status": document.status,
            "words": (
                document.statistics.word_count
            ),
            "characters": (
                document.statistics.character_count
            ),
            "chunks": len(
                document.chunks
            ),
            "links": len(
                document.links
            ),
            "headings": len(
                document.headings
            ),
            "quality": (
                document.statistics.quality_score
            ),
            "language": (
                document.statistics.language_hint
            ),
            "fingerprint": (
                document.fingerprint
            ),
        }

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def update_config(
        self,
        **changes: Any,
    ):

        for key, value in changes.items():

            if not hasattr(
                self.config,
                key,
            ):

                raise AttributeError(
                    f"Unknown processor setting: {key}"
                )

            setattr(
                self.config,
                key,
                value,
            )

    def get_config(
        self,
    ) -> Dict[str, Any]:

        return asdict(
            self.config
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    def get_statistics(
        self,
    ) -> Dict[str, Any]:

        result = dict(
            self.statistics
        )

        result[
            "registry_size"
        ] = self.registry.size()

        return result

    # ========================================================
    # RESET
    # ========================================================

    def clear_registry(
        self,
    ):

        self.registry.clear()

    def reset_statistics(
        self,
    ):

        self.statistics.clear()

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:

        problems = []

        if (
            self.config.chunk_overlap
            >= self.config.chunk_size
        ):

            problems.append(
                "Chunk overlap must be smaller than chunk size."
            )

        if (
            self.config.min_chunk_size
            > self.config.chunk_size
        ):

            problems.append(
                "Minimum chunk size exceeds chunk size."
            )

        if (
            self.config.max_document_length
            <= 0
        ):

            problems.append(
                "Maximum document length must be positive."
            )

        return {
            "healthy": not problems,
            "processor_version": (
                PROCESSOR_VERSION
            ),
            "problems": problems,
            "configuration": (
                self.get_config()
            ),
            "registry_size": (
                self.registry.size()
            ),
        }

    # ========================================================
    # DEBUG
    # ========================================================

    def debug_info(
        self,
    ) -> Dict[str, Any]:

        return {
            "processor_version": (
                PROCESSOR_VERSION
            ),
            "configuration": (
                self.get_config()
            ),
            "statistics": (
                self.get_statistics()
            ),
            "health": (
                self.health_check()
            ),
        }


# ============================================================
# DEFAULT INSTANCE
# ============================================================

processor = DocumentProcessor()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def process(
    document: Any,
    **kwargs,
) -> Optional[
    ProcessedDocument
]:

    return processor.process(
        document,
        **kwargs,
    )


def process_many(
    documents: Iterable[Any],
    **kwargs,
) -> ProcessingResult:

    return processor.process_many(
        documents,
        **kwargs,
    )


def process_stream(
    documents: Iterable[Any],
    **kwargs,
) -> Iterator[
    ProcessedDocument
]:

    return processor.process_stream(
        documents,
        **kwargs,
    )


def fingerprint(
    text: str,
) -> str:

    return processor.fingerprint(
        text
    )


def similarity_to(
    first_text: str,
    second_text: str,
) -> float:

    return processor.similarity_to(
        first_text,
        second_text,
    )


def health_check() -> Dict[str, Any]:

    return processor.health_check()


def debug_info() -> Dict[str, Any]:

    return processor.debug_info()