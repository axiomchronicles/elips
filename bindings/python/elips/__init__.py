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
    Config,
    ConfigError,
    Database,
    DimensionMismatch,
    ElipsError,
    Filter,
    GraphParams,
    InvalidVector,
    LockConflict,
    NotFound,
    Result,
    StorageError,
    Transaction,
    TransactionVault,
    Vault,
    VaultInfo,
    open,
    open_with_config,
)

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
    # errors
    "ElipsError",
    "DimensionMismatch",
    "InvalidVector",
    "ConfigError",
    "NotFound",
    "StorageError",
    "LockConflict",
]