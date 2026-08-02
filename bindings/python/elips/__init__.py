r"""ELIPS -- embedded, local-storage-first vector retrieval.

The names exported here are the API: :func:`connect` opens a database,
:class:`Engine` hands out :class:`Collection` objects, and a collection reads
and writes :class:`Row` / :class:`Hit` records.

Native handles (:class:`Database`, :class:`Vault`, :class:`Filter`) sit one
level down, under :mod:`elips.native`, and are reachable from any high-level
object through its ``native`` attribute. They used to be exported here as
co-equal names, which left no way to tell whether :class:`Vault` or
:class:`Collection` was the one to reach for.

Examples::

    >>> import elips
    >>> with elips.connect(":memory:", dimension=2) as db:
    ...     docs = db.collection("documents")
    ...     _ = docs.add(text="alpha design note", meta={"kind": "design"})
    ...     docs.search("alpha", top=1)[0].text
    'alpha design note'
"""

from __future__ import annotations

from typing import Any

from elips import _native
from elips._internal.protocols import Embedder
from elips._native import (
    ChunkInfo,
    Codec,
    Config,
    ConfigError,
    DimensionMismatch,
    DocumentAttachment,
    ElipsError,
    EmbeddingLineage,
    Filter,
    GraphParams,
    InvalidVector,
    LocalEmbedderConfig,
    LockConflict,
    NotFound,
    ParseError,
    QuantParams,
    QueryPlan,
    StorageError,
    distance,
    generate_id,
    is_valid_id,
    magnitude,
    normalize,
)
from elips.collection import Arena, Collection
from elips.config import connect, connect_with_config
from elips.database import Engine
from elips.gpu import Accelerator, AcceleratorSpec, accelerator, accelerators
from elips.records import CollectionHealth, Hit, RecordInput, Row, WalRecord

__version__: str = _native.__version__

#: True when the extension was built with GPU bindings.
has_gpu: bool = _native.has_gpu
_has_gpu = has_gpu

# Submodules resolved on first use rather than at import: `elips.native` and
# the legacy `elips.core` / `elips.modern` facades cost nothing until touched.
_LAZY = frozenset({"core", "modern", "native"})


def __getattr__(name: str) -> Any:
    """Resolve lazy submodules and the remaining native symbols.

    Native names stay reachable as ``elips.Vault`` so existing code keeps
    working, but they are deliberately absent from ``__all__``: the supported
    spelling is ``elips.native.Vault``.

    Engine internals warn on top of that, naming the module that owns them.
    """

    if name in _LAZY:
        import importlib

        module = importlib.import_module(f"elips.{name}")
        globals()[name] = module
        return module

    if not name.startswith("_") and hasattr(_native, name):
        if name in _native.DEMOTED:
            import warnings

            warnings.warn(
                _native.demotion_message(name, via=__name__),
                DeprecationWarning,
                stacklevel=2,
            )
            # Not cached, so every call site warns rather than only the first;
            # see the same note in `elips/native.py`.
            return getattr(_native, name)
        value = getattr(_native, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Offer the supported surface; demoted internals stay out of completion."""

    return sorted({*__all__, *_LAZY, *_native.supported_names()})


__all__ = [
    # opening a database
    "connect",
    "connect_with_config",
    "Config",
    "Engine",
    # collections and records
    "Collection",
    "CollectionHealth",
    "RecordInput",
    "Row",
    "Hit",
    # search
    "Filter",
    "QueryPlan",
    # documents and provenance
    "DocumentAttachment",
    "ChunkInfo",
    "EmbeddingLineage",
    # durability
    "WalRecord",
    # compression
    "Codec",
    "QuantParams",
    # tuning
    "GraphParams",
    "LocalEmbedderConfig",
    "Embedder",
    # GPU
    "Accelerator",
    "AcceleratorSpec",
    "accelerator",
    "accelerators",
    "has_gpu",
    # vector helpers
    "distance",
    "generate_id",
    "is_valid_id",
    "magnitude",
    "normalize",
    # exceptions
    "ElipsError",
    "ConfigError",
    "DimensionMismatch",
    "InvalidVector",
    "LockConflict",
    "NotFound",
    "ParseError",
    "StorageError",
    # prior name for Collection
    "Arena",
    "__version__",
]
