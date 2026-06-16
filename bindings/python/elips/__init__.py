"""ELIPS — an embedded, local-storage-first vector database.

Zero infrastructure: no server, no port, no daemon. Import it and go.

Quick start::

    import elips

    db = elips.open(":memory:", dimension=128, metric="cosine")
    docs = db.vault("documents")
    docs.place_document("alpha design note", {"kind": "design"})

    print(docs.seek_text("alpha", top=1)[0].document.text)

Advanced configuration::

    from elips import Config, GraphParams, LocalEmbedderConfig

    config = (Config()
        .dimension(768)
        .metric("cosine")
        .index("graph")
        .graph_params(GraphParams(max_connections=32, ef_construction=400))
        .local_text_embedder(LocalEmbedderConfig(model="compact", revision="v1"))
        .durability("standard"))

    db = elips.open_with_config("/data/vectors", config)

Transactions::

    with db.begin_transaction() as txn:
        v = txn.vault("docs")
        v.place([1.0, 2.0], {"title": "A"})
        v.place([3.0, 4.0], {"title": "B"})
        # Auto-committed on exit
"""

# -- version ------------------------------------------------------------------

__version__ = "1.0.0"

# -- module imports -----------------------------------------------------------

from ._core import (
    AccessMode,
    ChunkInfo,
    Comparator,
    Config,
    ConfigError,
    Database,
    DocumentAttachment,
    DimensionMismatch,
    Durability,
    EmbeddingLineage,
    ElipsError,
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
    validate_eql,
    requires_normalization,
    tokenize_eql,
)
from .modern import Arena, Embedder, Engine, Hit, Row, connect, connect_with_config

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
except ImportError:
    _has_gpu = False
    _gpu_exports = []

# -- public API ---------------------------------------------------------------

__all__ = [
    # factory
    "open",
    "open_with_config",
    "connect",
    "connect_with_config",
    # core classes
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
    "Engine",
    "Arena",
    "Row",
    "Hit",
    "Embedder",
    "TextEmbedderInfo",
    # enums
    "Metric",
    "IndexType",
    "Durability",
    "AccessMode",
    "Comparator",
    "QueryStrategy",
    "TextEmbedderKind",
    # EQL
    "Token",
    "TokenKind",
    "validate_eql",
    "tokenize_eql",
    # utilities
    "distance",
    "requires_normalization",
    "metric_from_string",
    "metric_to_string",
    # errors
    "ElipsError",
    "DimensionMismatch",
    "InvalidVector",
    "ConfigError",
    "NotFound",
    "StorageError",
    "LockConflict",
    "ParseError",
    # lineage / docs
    "DocumentAttachment",
    "ChunkInfo",
    "EmbeddingLineage",
    "QueryPlan",
]

__all__.extend(_gpu_exports)
