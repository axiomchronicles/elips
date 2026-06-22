from __future__ import annotations

r"""Compatibility facade for the modern ELIPS API.

Examples::

    >>> from elips.modern import RecordInput, connect
    >>> engine = connect(":memory:", dimension=2)
    >>> arena = engine.arena("documents")
    >>> _ = arena.write(RecordInput(text="alpha note", meta={"kind": "design"}))
    >>> engine.close()
"""

from ._modern import (
    Arena,
    Embedder,
    Engine,
    Hit,
    RecordInput,
    Row,
    connect,
    connect_with_config,
)

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
