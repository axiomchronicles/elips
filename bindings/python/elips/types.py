from __future__ import annotations

r"""Typed aliases and ``TypedDict`` models for the ELIPS Python SDK.

Examples::

    >>> from elips.types import MetricName, PayloadLike, RecordInputDict
    >>> metric: MetricName = "cosine"
    >>> payload: PayloadLike = {"kind": "design", "published": True}
    >>> record: RecordInputDict = {"text": "alpha note", "meta": payload}
"""

from typing import TYPE_CHECKING, Literal, Mapping, Optional, Sequence, TypedDict, Union

if TYPE_CHECKING:
    from .core import ChunkInfo, DocumentAttachment, EmbeddingLineage

# Primitive aliases
MetaValue = Union[bool, int, float, str]
Vector = Sequence[float]
PayloadLike = Mapping[str, MetaValue]
Metadata = dict[str, MetaValue]

# Literal-friendly names used by the pure-Python facade
MetricName = Literal["cosine", "euclidean", "dot_product"]
IndexName = Literal["graph", "exact"]
AccessModeName = Literal["read_write", "read_only"]


class RecordInputDict(TypedDict, total=False):
    r"""Mapping input accepted by :meth:`elips.Arena.write_many` and :meth:`elips.Arena.ingest`."""

    vector: Vector
    text: str
    meta: PayloadLike
    key: str
    document: "DocumentAttachment"
    chunk: "ChunkInfo"
    lineage: "EmbeddingLineage"


class BatchRecord(TypedDict, total=False):
    r"""Legacy batch mapping accepted by :meth:`elips.Vault.place_many` and modern compatibility helpers."""

    id: str
    vector: Vector
    text: str
    data: PayloadLike
    document: "DocumentAttachment"
    chunk: "ChunkInfo"
    lineage: "EmbeddingLineage"


class StoredRecord(TypedDict):
    r"""Record dictionary returned by :meth:`elips.Vault.fetch` and :meth:`elips.Vault.scan`."""

    id: str
    vector: tuple[float, ...]
    data: dict[str, MetaValue]
    document: Optional["DocumentAttachment"]
    chunk: Optional["ChunkInfo"]
    lineage: Optional["EmbeddingLineage"]


FetchResult = StoredRecord
ScanResult = StoredRecord
QueryBindings = Mapping[str, Vector]
RecordDict = BatchRecord

__all__ = [
    "AccessModeName",
    "BatchRecord",
    "FetchResult",
    "IndexName",
    "MetaValue",
    "Metadata",
    "MetricName",
    "PayloadLike",
    "QueryBindings",
    "RecordDict",
    "RecordInputDict",
    "ScanResult",
    "StoredRecord",
    "Vector",
]
