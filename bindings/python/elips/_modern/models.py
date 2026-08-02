r"""Dataclasses exchanged across the modern ELIPS API.

:class:`RecordInput` is the write-side model; :class:`Row`, :class:`Hit`,
:class:`WalRecord`, and :class:`ArenaHealth` are read-side results.

Examples::

    >>> from elips import RecordInput
    >>> record = RecordInput(text="alpha note", meta={"kind": "design"})
    >>> record.text
    'alpha note'
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from elips._native import ChunkInfo, DocumentAttachment, EmbeddingLineage
from elips.types import CodecName, MetaValue, PayloadLike, Vector


def _clone_document_attachment(document: DocumentAttachment) -> DocumentAttachment:
    clone = DocumentAttachment(
        text=document.text,
        uri=document.uri,
        mime_type=document.mime_type,
    )
    return clone


@dataclass(frozen=True, slots=True)
class RecordInput:
    r"""RecordInput(*, vector=None, text=None, meta=None, key=None, document=None, chunk=None, lineage=None) -> RecordInput

    Structured input record for :meth:`elips.Arena.write` and
    :meth:`elips.Arena.write_many`.

    Args:
        vector (Sequence[float], optional): Explicit embedding vector. Default:
            ``None``.
        text (str, optional): Source text used for native text ingestion or
            Python-side embedding. Default: ``None``.
        meta (Mapping[str, MetaValue], optional): Metadata payload. Default:
            ``None``.
        key (str, optional): Explicit record identifier. Default: ``None``.
        document (DocumentAttachment, optional): Full document attachment. When
            both ``text`` and ``document`` are provided, their text values must
            agree. Default: ``None``.
        chunk (ChunkInfo, optional): Chunk lineage information. Default:
            ``None``.
        lineage (EmbeddingLineage, optional): Embedding provenance. Default:
            ``None``.

    .. note::
        Text-only records with custom document metadata such as ``uri`` or a
        non-default ``mime_type`` still require an explicit vector or the
        low-level :meth:`elips.Vault.place` API, because
        :meth:`elips.Vault.place_document` only accepts raw text.

    Examples::

        >>> import elips
        >>> record = elips.RecordInput(
        ...     text="alpha design note",
        ...     meta={"kind": "design"},
        ... )
        >>> record.document_text
        'alpha design note'
        >>> attached = elips.DocumentAttachment(text="beta note", mime_type="text/markdown")
        >>> explicit = elips.RecordInput(
        ...     vector=[0.0, 1.0],
        ...     document=attached,
        ...     meta={"kind": "ops"},
        ... )
        >>> explicit.document_text
        'beta note'
    """

    vector: Vector | None = None
    text: str | None = None
    meta: PayloadLike | None = None
    key: str | None = None
    document: DocumentAttachment | None = None
    chunk: ChunkInfo | None = None
    lineage: EmbeddingLineage | None = None

    def __post_init__(self) -> None:
        if self.vector is None and self.document_text is None:
            raise ValueError(
                "record input requires a vector, text, or document with text"
            )

        if self.text is not None and self.document is not None:
            document_text = self.document.text
            if document_text not in ("", self.text):
                raise ValueError(
                    "text and document.text must match when both are provided"
                )

    @property
    def document_text(self) -> str | None:
        r"""document_text -> Optional[str]

        Return the record text, whether it was provided directly or through a
        :class:`~elips.DocumentAttachment`.

        Examples::

            >>> import elips
            >>> elips.RecordInput(text="alpha note").document_text
            'alpha note'
        """

        if self.text is not None:
            return self.text
        if self.document is None:
            return None
        return self.document.text

    def materialize_meta(self) -> dict[str, MetaValue]:
        r"""materialize_meta() -> dict[str, MetaValue]

        Return the record metadata as a mutable dictionary.

        Examples::

            >>> import elips
            >>> elips.RecordInput(text="alpha", meta={"kind": "design"}).materialize_meta()
            {'kind': 'design'}
        """

        return dict(self.meta or {})

    def materialize_document(self) -> DocumentAttachment | None:
        r"""materialize_document() -> Optional[DocumentAttachment]

        Build a concrete :class:`~elips.DocumentAttachment` for storage.

        Returns:
            DocumentAttachment or None: The materialized document attachment.

        Examples::

            >>> import elips
            >>> record = elips.RecordInput(text="alpha note")
            >>> record.materialize_document().text
            'alpha note'
        """

        if self.document is None:
            if self.text is None:
                return None
            return DocumentAttachment(text=self.text)

        clone = _clone_document_attachment(self.document)
        if self.text is not None:
            clone.text = self.text
        return clone

    def has_custom_document_metadata(self) -> bool:
        r"""has_custom_document_metadata() -> bool

        Return whether the input carries document fields beyond plain text.

        Examples::

            >>> import elips
            >>> doc = elips.DocumentAttachment(text="alpha", uri="notes.md")
            >>> elips.RecordInput(vector=[1.0, 0.0], document=doc).has_custom_document_metadata()
            True
        """

        if self.document is None:
            return False

        return bool(
            self.document.uri
            or self.document.mime_type not in ("", "text/plain")
        )

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> RecordInput:
        r"""from_mapping(record) -> RecordInput

        Convert a mapping into a :class:`RecordInput`.

        The method accepts both the modern field names (``meta`` / ``key``)
        and the low-level batch field names (``data`` / ``id``).

        Args:
            record (Mapping[str, Any]): Mapping describing a single record.

        Returns:
            RecordInput: The normalized structured record.

        Examples::

            >>> import elips
            >>> record = elips.RecordInput.from_mapping(
            ...     {"text": "alpha note", "meta": {"kind": "design"}}
            ... )
            >>> record.key is None
            True
            >>> legacy = elips.RecordInput.from_mapping(
            ...     {"vector": [1.0, 0.0], "data": {"kind": "vector"}}
            ... )
            >>> legacy.materialize_meta()
            {'kind': 'vector'}
        """

        modern_meta = record.get("meta")
        legacy_meta = record.get("data")
        if (
            modern_meta is not None
            and legacy_meta is not None
            and dict(modern_meta) != dict(legacy_meta)
        ):
            raise ValueError(
                "record mapping cannot define both meta and data with different values"
            )

        modern_key = record.get("key")
        legacy_key = record.get("id")
        if modern_key is not None and legacy_key is not None and modern_key != legacy_key:
            raise ValueError(
                "record mapping cannot define both key and id with different values"
            )

        document = record.get("document")
        if document is not None and not isinstance(document, DocumentAttachment):
            raise TypeError("record document must be a DocumentAttachment instance")

        chunk = record.get("chunk")
        if chunk is not None and not isinstance(chunk, ChunkInfo):
            raise TypeError("record chunk must be a ChunkInfo instance")

        lineage = record.get("lineage")
        if lineage is not None and not isinstance(lineage, EmbeddingLineage):
            raise TypeError("record lineage must be an EmbeddingLineage instance")

        return cls(
            vector=record.get("vector"),
            text=record.get("text"),
            meta=modern_meta if modern_meta is not None else legacy_meta,
            key=modern_key if modern_key is not None else legacy_key,
            document=document,
            chunk=chunk,
            lineage=lineage,
        )


@dataclass(frozen=True, slots=True)
class Row:
    r"""Row(key, meta, document=None, vector=None, chunk=None, lineage=None) -> Row

    Materialized record returned by :meth:`elips.Arena.pull` and
    :meth:`elips.Arena.sweep`.

    Args:
        key (str): Record identifier.
        meta (dict[str, MetaValue]): Metadata payload.
        document (DocumentAttachment, optional): Stored document attachment.
            Default: ``None``.
        vector (tuple[float, ...], optional): Stored vector when requested.
            Default: ``None``.
        chunk (ChunkInfo, optional): Chunk lineage. Default: ``None``.
        lineage (EmbeddingLineage, optional): Embedding provenance. Default:
            ``None``.

    Examples::

        >>> import elips
        >>> row = elips.Row(key="demo", meta={"kind": "design"})
        >>> row.text is None
        True
    """

    key: str
    meta: dict[str, MetaValue]
    document: DocumentAttachment | None = None
    vector: tuple[float, ...] | None = None
    chunk: ChunkInfo | None = None
    lineage: EmbeddingLineage | None = None
    approximate: bool = False
    codec: CodecName = "none"

    @property
    def text(self) -> str | None:
        r"""text -> Optional[str]

        Convenience alias for ``document.text``.

        Examples::

            >>> import elips
            >>> document = elips.DocumentAttachment(text="alpha note")
            >>> row = elips.Row(key="demo", meta={}, document=document)
            >>> row.text
            'alpha note'
        """

        if self.document is None:
            return None
        return self.document.text


@dataclass(frozen=True, slots=True)
class Hit:
    r"""Hit(key, distance, meta, document=None, vector=None, chunk=None, lineage=None) -> Hit

    Search hit returned by :meth:`elips.Arena.probe`,
    :meth:`elips.Arena.probe_text`, and :meth:`elips.Arena.probe_hybrid`.

    Args:
        key (str): Record identifier.
        distance (float): Distance from the query.
        meta (dict[str, MetaValue]): Metadata payload.
        document (DocumentAttachment, optional): Stored document attachment.
            Default: ``None``.
        vector (tuple[float, ...], optional): Stored vector when requested.
            Default: ``None``.
        chunk (ChunkInfo, optional): Chunk lineage. Default: ``None``.
        lineage (EmbeddingLineage, optional): Embedding provenance. Default:
            ``None``.

    Examples::

        >>> import elips
        >>> hit = elips.Hit(key="demo", distance=0.0, meta={"kind": "design"})
        >>> hit.distance
        0.0
    """

    key: str
    distance: float
    meta: dict[str, MetaValue]
    document: DocumentAttachment | None = None
    vector: tuple[float, ...] | None = None
    chunk: ChunkInfo | None = None
    lineage: EmbeddingLineage | None = None
    approximate: bool = False
    codec: CodecName = "none"

    @property
    def text(self) -> str | None:
        r"""text -> Optional[str]

        Convenience alias for ``document.text``.

        Examples::

            >>> import elips
            >>> document = elips.DocumentAttachment(text="alpha note")
            >>> hit = elips.Hit(key="demo", distance=0.1, meta={}, document=document)
            >>> hit.text
            'alpha note'
        """

        if self.document is None:
            return None
        return self.document.text


@dataclass(frozen=True, slots=True)
class WalRecord:
    r"""WalRecord(*, op, arena, key, vector=None, meta=None, document=None, chunk=None, lineage=None) -> WalRecord

    One acknowledged write-ahead log record, as returned by
    :meth:`elips.Engine.pending_writes`.

    Args:
        op ("insert" | "erase" | "insert_ex"): Kind of mutation. Transaction
            markers are resolved during replay and never surface here.
        arena (str): Arena the mutation targeted.
        key (str): Record identifier.
        vector (tuple[float, ...], optional): Stored vector; empty for erases.
            Default: ``None``.
        meta (dict[str, MetaValue], optional): Metadata payload. Default:
            ``None``.
        document (DocumentAttachment, optional): Attached document. Default:
            ``None``.
        chunk (ChunkInfo, optional): Chunk lineage. Default: ``None``.
        lineage (EmbeddingLineage, optional): Embedding provenance. Default:
            ``None``.

    Examples::

        >>> import elips
        >>> record = elips.WalRecord(op="insert", arena="documents", key="abc")
        >>> record.is_delete
        False
    """

    op: str
    arena: str
    key: str
    vector: tuple[float, ...] | None = None
    meta: dict[str, MetaValue] | None = None
    document: DocumentAttachment | None = None
    chunk: ChunkInfo | None = None
    lineage: EmbeddingLineage | None = None

    @property
    def is_delete(self) -> bool:
        r"""is_delete -> bool

        True when this record removed a key rather than writing one.

        Examples::

            >>> import elips
            >>> elips.WalRecord(op="erase", arena="a", key="k").is_delete
            True
        """

        return self.op == "erase"

    @classmethod
    def from_entry(cls, entry: Any) -> WalRecord:
        r"""from_entry(entry) -> WalRecord

        Build a typed record from a low-level :class:`elips.WalEntry`.

        Args:
            entry (WalEntry): Entry produced by :func:`elips.replay_wal`.

        Returns:
            WalRecord: Typed, immutable view of the same record.
        """

        return cls(
            op=entry.op.name,
            arena=entry.vault,
            key=entry.id,
            vector=tuple(entry.vector) or None,
            meta=dict(entry.data) or None,
            document=entry.document,
            chunk=entry.chunk,
            lineage=entry.lineage,
        )


@dataclass(frozen=True, slots=True)
class ArenaHealth:
    r"""ArenaHealth(*, name, live, pending_removals, dimension, metric, read_only, sealed) -> ArenaHealth

    A point-in-time health snapshot of one arena, from
    :meth:`elips.Arena.health`.

    Args:
        name (str): Arena name.
        live (int): Records currently searchable.
        pending_removals (int): Deleted records not yet reclaimed. Tombstones
            occupy the search beam and hold memory until compaction, so a large
            value relative to ``live`` is a signal to call
            :meth:`elips.Arena.vacuum`.
        dimension (int): Vector dimension.
        metric ("cosine" | "euclidean" | "dot_product"): Similarity metric.
        read_only (bool): Whether the arena currently refuses writes.
        sealed (bool): Whether the owning engine has been closed.

    Examples::

        >>> import elips
        >>> engine = elips.connect(":memory:", dimension=2)
        >>> arena = engine.arena("documents")
        >>> _ = arena.write(vector=[1.0, 0.0])
        >>> health = arena.health()
        >>> (health.live, health.pending_removals)
        (1, 0)
        >>> engine.close()
    """

    name: str
    live: int
    pending_removals: int
    dimension: int
    metric: str
    read_only: bool
    sealed: bool
    codec: CodecName = "none"
    code_bytes: int = 0

    @property
    def tombstone_ratio(self) -> float:
        r"""tombstone_ratio -> float

        Fraction of graph nodes that are tombstones, in ``[0.0, 1.0]``.

        Compare against the arena's configured ``compaction_ratio`` to predict
        whether the next delete will trigger an automatic rebuild.

        Examples::

            >>> import elips
            >>> health = elips.ArenaHealth(
            ...     name="a", live=8, pending_removals=2,
            ...     dimension=2, metric="cosine",
            ...     read_only=False, sealed=False,
            ... )
            >>> health.tombstone_ratio
            0.2
        """

        total = self.live + self.pending_removals
        if total == 0:
            return 0.0
        return self.pending_removals / total


__all__ = ["ArenaHealth", "Hit", "RecordInput", "Row", "WalRecord"]
