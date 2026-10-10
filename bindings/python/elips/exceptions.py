r"""ELIPS exception hierarchy -- the single source for the public error types.

All ELIPS errors derive from :class:`ElipsError`, which in turn derives from
:class:`RuntimeError`. The native runtime raises the base set
(:class:`DimensionMismatch`, :class:`NotFound`, ...); this module adds the
``*Error``-suffixed spellings (Phase 5 naming alignment) as aliases of the same
classes, so ``except DimensionMismatchError`` catches what the engine raises,
and the typed additions ELIPS surfaces from the Python layer
(:class:`CollectionNotFoundError`, :class:`NotTrainedError`,
:class:`BackendUnavailableError`).

Examples::

    >>> import elips
    >>> try:
    ...     client = elips.Client(dimension=2)
    ...     client.get_or_create_collection("docs").add(embeddings=[[1.0]])
    ... except elips.DimensionMismatchError:
    ...     print("dimension error")
    dimension error
"""

from __future__ import annotations

from elips._native import (
    ConfigError,
    DimensionMismatch,
    ElipsError,
    InvalidVector,
    LockConflict,
    NotFound,
    ParseError,
    StorageError,
)

# ``*Error`` suffix alignment: the engine raises the un-suffixed native class,
# so these are aliases (not subclasses) and still catch native raises.
DimensionMismatchError = DimensionMismatch
InvalidVectorError = InvalidVector
NotFoundError = NotFound
LockConflictError = LockConflict


class NotTrainedError(ElipsError):
    """A compressed/IVF index was searched before its codebook was trained.

    FAISS parity for ``is_trained``: training (``Collection.train`` /
    ``compress``) must run on representative data before encoded search.
    """


class CollectionNotFoundError(NotFoundError):
    """A collection was requested by name but does not exist.

    Raised by :meth:`elips.Client.get_collection`. Subclasses
    :class:`NotFoundError` so existing ``except NotFound`` handlers still catch
    it.
    """


class BackendUnavailableError(ElipsError):
    """A requested compute backend (e.g. a GPU device) is not available.

    Maps the native ``GpuError.backend_unavailable`` condition to a typed
    Python error.
    """


__all__ = [
    "BackendUnavailableError",
    "CollectionNotFoundError",
    "ConfigError",
    "DimensionMismatch",
    "DimensionMismatchError",
    "ElipsError",
    "InvalidVector",
    "InvalidVectorError",
    "LockConflict",
    "LockConflictError",
    "NotFound",
    "NotFoundError",
    "NotTrainedError",
    "ParseError",
    "StorageError",
]
