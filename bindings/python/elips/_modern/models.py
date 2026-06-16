from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..core import ChunkInfo, DocumentAttachment, EmbeddingLineage
from ..types import MetaValue, PayloadLike, Vector


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

    vector: Optional[Vector] = None
    text: Optional[str] = None
    meta: Optional[PayloadLike] = None
    key: Optional[str] = None
    document: Optional[DocumentAttachment] = None
    chunk: Optional[ChunkInfo] = None
    lineage: Optional[EmbeddingLineage] = None

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
    def document_text(self) -> Optional[str]:
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

    def materialize_document(self) -> Optional[DocumentAttachment]:
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
    def from_mapping(cls, record: Mapping[str, Any]) -> "RecordInput":
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
    document: Optional[DocumentAttachment] = None
    vector: Optional[tuple[float, ...]] = None
    chunk: Optional[ChunkInfo] = None
    lineage: Optional[EmbeddingLineage] = None

    @property
    def text(self) -> Optional[str]:
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
    document: Optional[DocumentAttachment] = None
    vector: Optional[tuple[float, ...]] = None
    chunk: Optional[ChunkInfo] = None
    lineage: Optional[EmbeddingLineage] = None

    @property
    def text(self) -> Optional[str]:
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


__all__ = ["Hit", "RecordInput", "Row"]
