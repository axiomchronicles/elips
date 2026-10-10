r"""Turn native record dicts and search results into typed models.

Split out of ``collection/arena.py``: the native layer returns dicts and
:class:`~elips.Result` handles, and every read method needs the same conversion
into :class:`~elips.Row` / :class:`~elips.Hit`.

A search result already carries metadata, document, chunk, and lineage, so a
hit only needs a fetch when the caller asked for the stored vector. That is why
``include_vectors`` costs an extra read per hit and defaults to off.
"""

from __future__ import annotations

import hashlib
from typing import cast

from elips._native import Result, Vault
from elips.records.models import Hit, Row
from elips.typing import StoredRecord

#: Reserved metadata key holding a record's original Chroma-style string id.
#: Hidden from user-visible metadata on read; see :func:`_externalize_id`.
RESERVED_ID_KEY = "__elips_id__"

_HEX = frozenset("0123456789abcdefABCDEF")


def is_native_id(value: str) -> bool:
    """True when ``value`` is already a native RecordID (32 hex digits).

    Mirrors ``RecordID::from_string`` (hyphens stripped, 32 hex digits). Any
    other string is a Chroma-style arbitrary id that needs derivation.
    """
    digits = value.replace("-", "")
    return len(digits) == 32 and all(c in _HEX for c in digits)


def derive_record_id(value: str) -> str:
    """Map an arbitrary string id to a stable 32-hex native RecordID.

    Deterministic (BLAKE2b-128), so the same string always resolves to the
    same record across calls and reopens. ``RecordID::from_string`` enforces
    only hex length -- not UUIDv7 version/variant bits -- so no bit-fixing is
    needed for the hash to parse.
    """
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).hexdigest()


def resolve_native_id(value: str) -> str:
    """Return the native RecordID for a user-supplied id (native passthrough)."""
    return value if is_native_id(value) else derive_record_id(value)


def _externalize_id(key: str, meta: dict) -> str:
    """Pop the reserved key from ``meta`` and return the user-visible id.

    Falls back to the native ``key`` (e.g. auto-generated records) when the
    record carries no stored string id. ``meta`` is mutated in place; callers
    pass a fresh ``dict`` copy.
    """
    original = meta.pop(RESERVED_ID_KEY, None)
    return original if original is not None else key


def row_from_record(record: StoredRecord, *, include_vectors: bool) -> Row:
    """Build a :class:`~elips.Row` from a native record dict."""

    meta = dict(record["data"])
    key = _externalize_id(record["id"], meta)
    return Row(
        key=key,
        meta=meta,
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
        cast(StoredRecord | None, vault.fetch(result.id)) if include_vectors else None
    )
    meta = dict(result.data)
    key = _externalize_id(result.id, meta)
    return Hit(
        key=key,
        distance=result.distance,
        meta=meta,
        document=result.document
        if result.document is not None
        else (fetched["document"] if fetched is not None else None),
        vector=tuple(fetched["vector"]) if fetched is not None else None,
        chunk=result.chunk
        if result.chunk is not None
        else (fetched["chunk"] if fetched is not None else None),
        lineage=result.lineage
        if result.lineage is not None
        else (fetched["lineage"] if fetched is not None else None),
        approximate=result.approximate,
        codec=result.codec,
    )


__all__ = [
    "RESERVED_ID_KEY",
    "derive_record_id",
    "hit_from_result",
    "is_native_id",
    "resolve_native_id",
    "row_from_record",
]
