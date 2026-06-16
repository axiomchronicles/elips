from __future__ import annotations

r"""Modern ELIPS API package.

Examples::

    >>> from elips._modern import RecordInput, connect
    >>> engine = connect(":memory:", dimension=2)
    >>> arena = engine.arena("documents")
    >>> _ = arena.write(RecordInput(text="alpha note", meta={"kind": "design"}))
"""

from .arena import Arena
from .connect import connect, connect_with_config
from .engine import Engine
from .models import Hit, RecordInput, Row
from .typing import Embedder

__all__ = [
    "Arena",
    "Embedder",
    "Engine",
    "Hit",
    "RecordInput",
    "Row",
    "connect",
    "connect_with_config",
]
