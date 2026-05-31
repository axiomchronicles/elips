"""ELIPS exception hierarchy.

All ELIPS errors derive from :class:`ElipsError`, which itself derives from
:exc:`RuntimeError`. Catching ``ElipsError`` catches all ELIPS-specific
failures.

.. code-block:: python

    try:
        docs.place([1.0, 2.0], {"key": "value"})
    except elips.DimensionMismatch as e:
        print("Vector dimension is wrong:", e)
    except elips.ElipsError as e:
        print("Some other ELIPS error:", e)
"""

from ._core import (
    ConfigError,
    DimensionMismatch,
    ElipsError,
    InvalidVector,
    LockConflict,
    NotFound,
    StorageError,
)

__all__ = [
    "ElipsError",
    "ConfigError",
    "DimensionMismatch",
    "InvalidVector",
    "LockConflict",
    "NotFound",
    "StorageError",
]