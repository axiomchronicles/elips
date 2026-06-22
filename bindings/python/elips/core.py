from __future__ import annotations

r"""ELIPS low-level binding facade.

This module re-exports the compiled :mod:`elips._core` extension through a
regular Python module so the package root can stay small and the low-level
runtime API remains easy to import directly.

Examples::

    >>> import elips.core as core
    >>> db = core.open(":memory:", dimension=2, metric="cosine")
    >>> docs = db.vault("documents")
    >>> docs.count()
    0
"""

from . import _core as _native_core
from ._core import (
    AccessMode,
    ChunkInfo,
    Comparator,
    Config,
    ConfigError,
    Database,
    DimensionMismatch,
    DocumentAttachment,
    Durability,
    ElipsError,
    EmbeddingLineage,
    Filter,
    GraphParams,
    IndexType,
    InvalidVector,
    LocalEmbedderConfig,
    LockConflict,
    Metric,
    NotFound,
    ParseError,
    QueryPlan,
    QueryStrategy,
    Result,
    StorageError,
    TextEmbedderInfo,
    TextEmbedderKind,
    Token,
    TokenKind,
    Transaction,
    TransactionVault,
    Vault,
    VaultInfo,
    distance,
    metric_from_string,
    metric_to_string,
    open,
    open_with_config,
    requires_normalization,
    tokenize_eql,
    validate_eql,
)

__version__ = _native_core.__version__

try:
    from ._core import (
        GpuConfig,
        GpuDeviceInfo,
        GpuError,
        GpuIndexAlgorithm,
        GpuIndexBuildParams,
        GpuMetricsSnapshot,
        GpuPolicy,
        GpuPrecision,
        GraphBuildAlgo,
        GraphIndexBuildParams,
        IndexBuildMode,
        IvfPqBuildParams,
        KernelTiming,
        gpu_devices,
    )
except ImportError:
    _has_gpu = False
    _gpu_exports: list[str] = []
else:
    _has_gpu = True
    _gpu_exports = [
        "GpuConfig",
        "GpuDeviceInfo",
        "GpuError",
        "GpuIndexAlgorithm",
        "GpuIndexBuildParams",
        "GpuMetricsSnapshot",
        "GpuPolicy",
        "GpuPrecision",
        "GraphBuildAlgo",
        "GraphIndexBuildParams",
        "IndexBuildMode",
        "IvfPqBuildParams",
        "KernelTiming",
        "gpu_devices",
    ]

__all__ = [
    "open",
    "open_with_config",
    "Database",
    "Vault",
    "VaultInfo",
    "Filter",
    "Result",
    "Config",
    "GraphParams",
    "LocalEmbedderConfig",
    "Transaction",
    "TransactionVault",
    "TextEmbedderInfo",
    "Metric",
    "IndexType",
    "Durability",
    "AccessMode",
    "Comparator",
    "QueryStrategy",
    "TextEmbedderKind",
    "Token",
    "TokenKind",
    "validate_eql",
    "tokenize_eql",
    "distance",
    "requires_normalization",
    "metric_from_string",
    "metric_to_string",
    "ElipsError",
    "DimensionMismatch",
    "InvalidVector",
    "ConfigError",
    "NotFound",
    "StorageError",
    "LockConflict",
    "ParseError",
    "DocumentAttachment",
    "ChunkInfo",
    "EmbeddingLineage",
    "QueryPlan",
]
__all__.extend(_gpu_exports)
