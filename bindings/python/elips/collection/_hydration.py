r"""Turn native record dicts and search results into typed models.

Split out of ``collection/arena.py``: the native layer returns dicts and
:class:`~elips.Result` handles, and every read method needs the same conversion
into :class:`~elips.Row` / :class:`~elips.Hit`.

A search result already carries metadata, document, chunk, and lineage, so a
hit only needs a fetch when the caller asked for the stored vector. That is why
``include_vectors`` costs an extra read per hit and defaults to off.
"""

from __future__ import annotations

from typing import cast

from elips._native import Result, Vault
from elips.records.models import Hit, Row
from elips.typing import StoredRecord


def row_from_record(record: StoredRecord, *, include_vectors: bool) -> Row:
    """Build a :class:`~elips.Row` from a native record dict."""

    return Row(
        key=record["id"],
        meta=dict(record["data"]),
        document=record["document"],
        vector=tuple(record["vector"]) if include_vectors else None,
        chunk=record["chunk"],
        lineage=record["lineage"],
        approximate=record["approximate"],
        codec=record["codec"],
    )


def hit_from_result(
    result: Result,
    *,
    vault: Vault,
    include_vectors: bool,
) -> Hit:
    """Build a :class:`~elips.Hit`, fetching the stored record only if needed."""

    fetched = (
        cast(StoredRecord | None, vault.fetch(result.id))
        if include_vectors
        else None
    )
    return Hit(
        key=result.id,
        distance=result.distance,
        meta=dict(result.data),
        document=result.document if result.document is not None else (
            fetched["document"] if fetched is not None else None
        ),
        vector=tuple(fetched["vector"]) if fetched is not None else None,
        chunk=result.chunk if result.chunk is not None else (
            fetched["chunk"] if fetched is not None else None
        ),
        lineage=result.lineage if result.lineage is not None else (
            fetched["lineage"] if fetched is not None else None
        ),
        approximate=result.approximate,
        codec=result.codec,
    )


__all__ = ["hit_from_result", "row_from_record"]
