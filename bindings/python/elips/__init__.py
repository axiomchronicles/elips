"""ELIPS — an embedded, local-storage-first vector database.

Zero infrastructure: no server, no port, no daemon. Import it and go.

Quick start::

    import elips

    db = elips.open(":memory:", dimension=1536, metric="cosine")
    docs = db.vault("documents")
    docs.place(vector=embedding, data={"title": "Example", "year": 2024})

    for hit in docs.seek(vector=query, top=10):
        print(hit.id, hit.distance, hit.data)

Advanced configuration::

    from elips import Config, GraphParams

    config = (Config()
        .dimension(1536)
        .metric("cosine")
        .index("graph")
        .graph_params(GraphParams(max_connections=32, ef_construction=400))
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
    Comparator,
    Config,
    ConfigError,
    Database,
    DimensionMismatch,
    Durability,
    ElipsError,
    Filter,
    GraphParams,
    IndexType,
    InvalidVector,
    LockConflict,
    Metric,
    NotFound,
    ParseError,
    Result,
    StorageError,
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
    )
    _has_gpu = True
except ImportError:
    _has_gpu = False

# -- public API ---------------------------------------------------------------

__all__ = [
    # factory
    "open",
    "open_with_config",
    # core classes
    "Database",
    "Vault",
    "VaultInfo",
    "Filter",
    "Result",
    "Config",
    "GraphParams",
    "Transaction",
    "TransactionVault",
    # enums
    "Metric",
    "IndexType",
    "Durability",
    "Comparator",
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
    # GPU types
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
    # errors
    "ElipsError",
    "DimensionMismatch",
    "InvalidVector",
    "ConfigError",
    "NotFound",
    "StorageError",
    "LockConflict",
    "ParseError",
]