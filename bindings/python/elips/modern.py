r"""Compatibility facade for the modern ELIPS API.

Examples::

    >>> from elips.modern import RecordInput, connect
    >>> engine = connect(":memory:", dimension=2)
    >>> arena = engine.arena("documents")
    >>> _ = arena.write(RecordInput(text="alpha note", meta={"kind": "design"}))
    >>> engine.close()
"""

from __future__ import annotations

from ._modern import (
    Accelerator,
    AcceleratorSpec,
    Arena,
    ArenaHealth,
    Embedder,
    Engine,
    Hit,
    RecordInput,
    Row,
    WalRecord,
    accelerator,
    accelerators,
    connect,
    connect_with_config,
)

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
