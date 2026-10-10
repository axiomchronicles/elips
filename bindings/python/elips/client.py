r"""Chroma-style client entry points.

:class:`Client` opens an ephemeral in-memory database; :class:`PersistentClient`
opens (or creates) one on disk. Both are :class:`~elips.database.engine.Engine`
subclasses, so they carry the full Chroma client surface
(``create_collection`` / ``get_collection`` / ``get_or_create_collection`` /
``list_collections`` / ``delete_collection``) plus ELIPS extensions.

The pre-Phase-5 openers :func:`elips.connect` / :func:`elips.open` /
:class:`elips.Engine` remain as deprecated-but-working shims.

Examples::

    >>> import elips
    >>> client = elips.Client(dimension=2, metric="cosine")
    >>> docs = client.get_or_create_collection("documents")
    >>> _ = docs.add(embeddings=[[1.0, 0.0]], ids=["a"])
    >>> docs.count()
    1
    >>> client.close()
"""

from __future__ import annotations

from typing import Any

from elips.config.connect import connect
from elips.database.engine import Engine

# ponytail: ELIPS is dimension-first (its default text embedder is a hashing
# embedder parameterized by the DB dimension — there is no intrinsic default
# dim), so unlike Chroma's zero-config Client() a dimension must be chosen.
# Full Chroma-style dimension inference from an embedding function is a Phase 6
# item; until then we fail with an actionable message instead of a cryptic
# ConfigError from deep in the engine.
_DIM_HINT = (
    "ELIPS is dimension-first: pass dimension=<int> (e.g. Client(dimension=384)) "
    "or an embedder= that fixes the dimension. Zero-config dimension inference is "
    "not yet supported."
)


def _require_dimension(settings: dict[str, Any]) -> None:
    if not settings.get("dimension") and settings.get("embedder") is None:
        raise ValueError(_DIM_HINT)


class Client(Engine):
    r"""Client(**settings) -> Client

    An ephemeral, in-memory ELIPS client (wraps ``connect(":memory:")``).

    Accepts the same keyword settings as :func:`elips.connect` (``dimension``,
    ``metric``, ``index``, ``embedder``, ``quantization``, ``gpu``, ...).
    A ``dimension`` (or dimension-fixing ``embedder``) is required.
    """

    def __init__(self, **settings: Any) -> None:
        _require_dimension(settings)
        engine = connect(":memory:", **settings)
        super().__init__(engine.native, default_embedder=engine._default_embedder)


class PersistentClient(Engine):
    r"""PersistentClient(path, **settings) -> PersistentClient

    An on-disk ELIPS client (wraps ``connect(path)``).

    Args:
        path (str): Directory to open or create.

    Accepts the same keyword settings as :func:`elips.connect`. A ``dimension``
    (or dimension-fixing ``embedder``) is required when creating a new database.
    """

    def __init__(self, path: str, **settings: Any) -> None:
        engine = connect(path, **settings)
        super().__init__(engine.native, default_embedder=engine._default_embedder)


__all__ = ["Client", "PersistentClient"]
