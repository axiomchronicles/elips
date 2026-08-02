r"""Typed aliases and ``TypedDict`` models for the ELIPS Python SDK.

Examples::

    >>> from elips.typing import MetricName, PayloadLike, RecordInputDict
    >>> metric: MetricName = "cosine"
    >>> payload: PayloadLike = {"kind": "design", "published": True}
    >>> record: RecordInputDict = {"text": "alpha note", "meta": payload}
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, TypedDict

if TYPE_CHECKING:
    # Guarded deliberately, and load-bearing. These names are only needed to
    # annotate the TypedDicts below, and the native modules import *this*
    # module for their own aliases. Importing them unguarded would close that
    # loop into a real circular import at module-execution time.
    from elips._native import ChunkInfo, DocumentAttachment, EmbeddingLineage

# Primitive aliases
MetaValue: TypeAlias = bool | int | float | str
Vector: TypeAlias = Sequence[float]
PayloadLike: TypeAlias = Mapping[str, MetaValue]
Metadata: TypeAlias = dict[str, MetaValue]

# A batch of vectors: rows of equal length, as accepted by the GPU kernels.
VectorBatch: TypeAlias = Sequence[Vector]
# Rows of distances, as returned by GpuDevice.compute_distances.
DistanceMatrix: TypeAlias = Sequence[Sequence[float]]

# Literal-friendly names used by the pure-Python facade
MetricName: TypeAlias = Literal["cosine", "euclidean", "dot_product"]
IndexName: TypeAlias = Literal["graph", "exact"]
AccessModeName: TypeAlias = Literal["read_write", "read_only"]
DurabilityName: TypeAlias = Literal["paranoid", "standard", "relaxed", "ephemeral"]
ComparatorName: TypeAlias = Literal["eq", "ne", "lt", "le", "gt", "ge", "gte"]
GpuPolicyName: TypeAlias = Literal[
    "auto", "prefer_gpu", "require_gpu", "cpu_only", "specific"
]
GpuAlgorithmName: TypeAlias = Literal[
    "auto", "cagra", "ivf_flat", "ivf_pq", "brute_force"
]
GpuPrecisionName: TypeAlias = Literal["fp32", "fp16", "int8", "auto"]
WalOpName: TypeAlias = Literal[
    "insert", "erase", "insert_ex", "txn_begin", "txn_commit", "insert_q"
]
CodecName: TypeAlias = Literal["none", "pq", "opq", "sq8"]


class TextEmbedderFn(Protocol):
    r"""Callable accepted by ``Config.text_embedder`` and ``open(embedder=...)``.

    Receives a batch of strings and must return one vector per input, in order,
    all of the database's dimension.
    """

    def __call__(self, texts: Sequence[str], /) -> Sequence[Vector]: ...


class RecordInputDict(TypedDict, total=False):
    r"""Mapping input accepted by :meth:`elips.Collection.write_many` and :meth:`elips.Collection.ingest`."""

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


FetchResult: TypeAlias = StoredRecord
ScanResult: TypeAlias = StoredRecord
QueryBindings: TypeAlias = Mapping[str, Vector]
RecordDict: TypeAlias = BatchRecord
# (indices, values) as returned by GpuDevice.top_k.
TopKResult: TypeAlias = tuple[Sequence[Sequence[int]], Sequence[Sequence[float]]]

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
