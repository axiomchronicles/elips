from __future__ import annotations

r"""ELIPS exception hierarchy.

All ELIPS errors derive from :class:`ElipsError`, which in turn derives from
:class:`RuntimeError`.

Examples::

    >>> import elips
    >>> try:
    ...     db = elips.open(":memory:", dimension=2)
    ...     db.vault("docs").place([1.0])
    ... except elips.DimensionMismatch:
    ...     print("dimension error")
    dimension error
"""

from .core import (
    ConfigError,
    DimensionMismatch,
    ElipsError,
    InvalidVector,
    LockConflict,
    NotFound,
    ParseError,
    StorageError,
)

__all__ = [
    "ElipsError",
    "ConfigError",
    "DimensionMismatch",
    "InvalidVector",
    "LockConflict",
    "NotFound",
    "ParseError",
    "StorageError",
]
