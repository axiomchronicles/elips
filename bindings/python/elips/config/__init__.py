r"""Opening a database: connection helpers, the config builder, and its enums.

:func:`connect` takes keyword arguments and builds the configuration for you.
:func:`connect_with_config` accepts a prepared :class:`~elips.Config` for the
options the keyword form does not cover.

Examples::

    >>> from elips.config import connect
    >>> db = connect(":memory:", dimension=2, metric="cosine")
    >>> db.collection_names()
    []
    >>> db.close()
"""

from __future__ import annotations

from elips._native import (
    AccessMode,
    Config,
    Durability,
    GraphParams,
    IndexType,
    LocalEmbedderConfig,
    Metric,
    TextEmbedderInfo,
    TextEmbedderKind,
    describe_local_embedder,
)
from elips.config.connect import connect, connect_with_config

__all__ = [
    "AccessMode",
    "Config",
    "Durability",
    "GraphParams",
    "IndexType",
    "LocalEmbedderConfig",
    "Metric",
    "TextEmbedderInfo",
    "TextEmbedderKind",
    "connect",
    "connect_with_config",
    "describe_local_embedder",
]
