r"""Typed aliases and ``TypedDict`` models for the ELIPS Python SDK.

Examples::

    >>> from elips.types import MetricName, PayloadLike, RecordInputDict
    >>> metric: MetricName = "cosine"
    >>> payload: PayloadLike = {"kind": "design", "published": True}
    >>> record: RecordInputDict = {"text": "alpha note", "meta": payload}
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, Union

if TYPE_CHECKING:
    from .core import ChunkInfo, DocumentAttachment, EmbeddingLineage

# Primitive aliases
MetaValue = Union[bool, int, float, str]
Vector = Sequence[float]
PayloadLike = Mapping[str, MetaValue]
Metadata = dict[str, MetaValue]

# A batch of vectors: rows of equal length, as accepted by the GPU kernels.
VectorBatch = Sequence[Vector]
# Rows of distances, as returned by GpuDevice.compute_distances.
DistanceMatrix = Sequence[Sequence[float]]

# Literal-friendly names used by the pure-Python facade
MetricName = Literal["cosine", "euclidean", "dot_product"]
IndexName = Literal["graph", "exact"]
AccessModeName = Literal["read_write", "read_only"]
DurabilityName = Literal["paranoid", "standard", "relaxed", "ephemeral"]
ComparatorName = Literal["eq", "ne", "lt", "le", "gt", "ge", "gte"]
GpuPolicyName = Literal["auto", "prefer_gpu", "require_gpu", "cpu_only", "specific"]
GpuAlgorithmName = Literal["auto", "cagra", "ivf_flat", "ivf_pq", "brute_force"]
GpuPrecisionName = Literal["fp32", "fp16", "int8", "auto"]
WalOpName = Literal["insert", "erase", "insert_ex", "txn_begin", "txn_commit", "insert_q"]
CodecName = Literal["none", "pq", "opq", "sq8"]


class TextEmbedderFn(Protocol):
    r"""Callable accepted by ``Config.text_embedder`` and ``open(embedder=...)``.

    Receives a batch of strings and must return one vector per input, in order,
    all of the database's dimension.
    """

    def __call__(self, texts: Sequence[str], /) -> Sequence[Vector]: ...


class RecordInputDict(TypedDict, total=False):
    r"""Mapping input accepted by :meth:`elips.Arena.write_many` and :meth:`elips.Arena.ingest`."""

    vector: Vector
    text: str
    meta: PayloadLike
    key: str
    document: DocumentAttachment
    chunk: ChunkInfo
    lineage: EmbeddingLineage


class BatchRecord(TypedDict, total=False):
    r"""Legacy batch mapping accepted by :meth:`elips.Vault.place_many` and modern compatibility helpers."""

    id: str
    vector: Vector
    text: str
    data: PayloadLike
    document: DocumentAttachment
    chunk: ChunkInfo
    lineage: EmbeddingLineage


class StoredRecord(TypedDict):
    r"""Record dictionary returned by :meth:`elips.Vault.fetch` and :meth:`elips.Vault.scan`."""

    id: str
    vector: tuple[float, ...]
    data: dict[str, MetaValue]
    document: DocumentAttachment | None
    chunk: ChunkInfo | None
    lineage: EmbeddingLineage | None
    approximate: bool
    codec: CodecName


FetchResult = StoredRecord
ScanResult = StoredRecord
QueryBindings = Mapping[str, Vector]
RecordDict = BatchRecord
# (indices, values) as returned by GpuDevice.top_k.
TopKResult = tuple[Sequence[Sequence[int]], Sequence[Sequence[float]]]

__all__ = [
    "AccessModeName",
    "BatchRecord",
    "CodecName",
    "ComparatorName",
    "DistanceMatrix",
    "DurabilityName",
    "FetchResult",
    "GpuAlgorithmName",
    "GpuPolicyName",
    "GpuPrecisionName",
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
    "TextEmbedderFn",
    "TopKResult",
    "Vector",
    "VectorBatch",
    "WalOpName",
]
