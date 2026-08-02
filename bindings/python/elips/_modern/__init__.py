r"""Modern ELIPS API package.

Examples::

    >>> from elips._modern import RecordInput, connect
    >>> engine = connect(":memory:", dimension=2)
    >>> arena = engine.arena("documents")
    >>> _ = arena.write(RecordInput(text="alpha note", meta={"kind": "design"}))
"""

from __future__ import annotations

from elips._modern.arena import Arena
from elips._modern.connect import connect, connect_with_config
from elips._modern.engine import Engine
from elips._modern.gpu import Accelerator, AcceleratorSpec, accelerator, accelerators
from elips._modern.models import ArenaHealth, Hit, RecordInput, Row, WalRecord
from elips._modern.typing import Embedder

__all__ = [
    "Accelerator",
    "AcceleratorSpec",
    "Arena",
    "ArenaHealth",
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
