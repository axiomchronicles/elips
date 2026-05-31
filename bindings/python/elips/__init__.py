"""ELIPS — an embedded, local-storage-first vector database.

Zero infrastructure: no server, no port, no daemon. Import it and go.

    import elips
    db = elips.open("/data/vectors", dimension=1536, metric="cosine")
    docs = db.vault("documents")
    docs.place(vector=embedding, data={"title": "Example"})
    for hit in docs.seek(vector=query, top=10):
        print(hit.id, hit.distance, hit.data)
"""

from ._core import (
    Database,
    ElipsError,
    Filter,
    LockConflict,
    Result,
    Vault,
    open,
)

__all__ = [
    "open",
    "Database",
    "Vault",
    "Filter",
    "Result",
    "ElipsError",
    "LockConflict",
]

__version__ = "1.0.0"
