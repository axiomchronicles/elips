r"""Embedder resolution for collection writes.

Split out of ``collection/arena.py``: deciding whether a record needs embedding,
and fanning a batch out to the configured embedder, is a separate concern from
the write API that calls it.

A database may carry a native text embedder, in which case the engine embeds on
its own and Python passes text straight through. Otherwise a Python embedder
has to produce the vectors first, in one batched call rather than one call per
record.
"""

from __future__ import annotations

from collections.abc import Sequence

from elips._internal.protocols import Embedder
from elips._native import Database, EmbeddingLineage
from elips.records.models import RecordInput
from elips.typing import Vector


def has_native_text_embedder(db: Database) -> bool:
    """True when the engine can embed text itself, without help from Python."""

    return bool(db.config.has_text_embedder)


def resolve_vectors(
    records: Sequence[RecordInput],
    *,
    db: Database,
    embedder: Embedder | None,
) -> tuple[list[Vector | None], set[int]]:
    """Fill in vectors for records that arrived as text.

    Returns the per-record vectors alongside the indices that were embedded
    here, which the caller needs in order to stamp provenance on exactly those
    records.

    Records are left as ``None`` when the engine has a native text embedder:
    the write path hands the text to the engine and lets it embed, which keeps
    the text on one side of the boundary instead of round-tripping a vector.
    """

    resolved: list[Vector | None] = [record.vector for record in records]
    missing = [index for index, vector in enumerate(resolved) if vector is None]
    if not missing or has_native_text_embedder(db):
        return resolved, set()

    if embedder is None:
        raise ValueError("this collection needs an embedder for text-first operations")

    texts: list[str] = []
    for index in missing:
        text = records[index].document_text
        if text is None:
            raise ValueError("records without vectors require text or document text")
        texts.append(text)

    # One call for the whole batch. The embedder port supports batching and the
    # per-record path did not use it, so ingesting N texts made N crossings.
    embedded = embedder(texts)
    if len(embedded) != len(missing):
        raise ValueError("embedder returned a batch with the wrong length")

    for index, vector in zip(missing, embedded, strict=True):
        resolved[index] = vector

    return resolved, set(missing)


def resolve_lineage(
    record: RecordInput,
    *,
    vector_generated: bool,
) -> EmbeddingLineage | None:
    """Attach provenance to vectors this layer produced.

    An explicit lineage always wins. Otherwise only a vector generated *here*
    gets stamped -- a caller-supplied vector has provenance we cannot know.
    """

    if record.lineage is not None:
        return record.lineage
    if not vector_generated:
        return None

    generated = EmbeddingLineage()
    generated.provider = "python"
    generated.model = "callable"
    generated.revision = ""
    generated.attributes = {}
    return generated


__all__ = ["has_native_text_embedder", "resolve_lineage", "resolve_vectors"]
