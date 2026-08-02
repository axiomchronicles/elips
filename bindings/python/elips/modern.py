r"""Compatibility facade for the modern ELIPS API.

.. deprecated:: 1.1
    These names are exported from the package root now. Import them from
    :mod:`elips` directly.
"""

from __future__ import annotations

import warnings

from elips._internal.protocols import Embedder
from elips.collection import Arena, Collection
from elips.config import connect, connect_with_config
from elips.database import Engine
from elips.gpu import Accelerator, AcceleratorSpec, accelerator, accelerators
from elips.records import CollectionHealth, Hit, RecordInput, Row, WalRecord

warnings.warn(
    "elips.modern is deprecated; these names are exported from elips directly",
    DeprecationWarning,
    stacklevel=2,
)

#: Prior name for :class:`~elips.CollectionHealth`.
ArenaHealth = CollectionHealth

__all__ = [
    "Accelerator",
    "AcceleratorSpec",
    "Arena",
    "ArenaHealth",
    "Collection",
    "CollectionHealth",
    "Embedder",
    "Engine",
    "Hit",
    "RecordInput",
    "Row",
    "WalRecord",
    "accelerator",
    "accelerators",
    "connect",
    "connect_with_config",
]
