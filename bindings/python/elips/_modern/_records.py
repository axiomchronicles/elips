from __future__ import annotations

r"""Internal record normalization helpers for the modern ELIPS API."""

from collections.abc import Mapping, Sequence

from ..core import ChunkInfo, DocumentAttachment, EmbeddingLineage
from ..types import PayloadLike, Vector
from .models import RecordInput
from .typing import RecordInputLike


def coerce_record_input(record: RecordInputLike) -> RecordInput:
    if isinstance(record, RecordInput):
        return record

    if isinstance(record, Mapping):
        return RecordInput.from_mapping(record)

    raise TypeError(
        "record inputs must be RecordInput instances or mapping-style records"
    )


def build_record_inputs_from_columns(
    *,
    vectors: Sequence[Vector | None] | None,
    texts: Sequence[str | None] | None,
    meta: Sequence[PayloadLike | None] | None,
    keys: Sequence[str | None] | None,
    documents: Sequence[DocumentAttachment | None] | None,
    chunks: Sequence[ChunkInfo | None] | None,
    lineages: Sequence[EmbeddingLineage | None] | None,
) -> list[RecordInput]:
    candidates = [
        len(vectors) if vectors is not None else None,
        len(texts) if texts is not None else None,
        len(meta) if meta is not None else None,
        len(keys) if keys is not None else None,
        len(documents) if documents is not None else None,
        len(chunks) if chunks is not None else None,
        len(lineages) if lineages is not None else None,
    ]

    count = next((value for value in candidates if value is not None), 0)
    if count == 0:
        raise ValueError(
            "ingest requires records or at least one vector or text batch"
        )

    for value in candidates:
        if value is not None and value != count:
            raise ValueError("all ingest batch fields must have the same length")

    prepared_vectors = list(vectors) if vectors is not None else [None] * count
    prepared_texts = list(texts) if texts is not None else [None] * count
    prepared_meta = list(meta) if meta is not None else [None] * count
    prepared_keys = list(keys) if keys is not None else [None] * count
    prepared_documents = (
        list(documents) if documents is not None else [None] * count
    )
    prepared_chunks = list(chunks) if chunks is not None else [None] * count
    prepared_lineages = (
        list(lineages) if lineages is not None else [None] * count
    )

    records: list[RecordInput] = []
    for index in range(count):
        try:
            record = RecordInput(
                vector=prepared_vectors[index],
                text=prepared_texts[index],
                meta=prepared_meta[index],
                key=prepared_keys[index],
                document=prepared_documents[index],
                chunk=prepared_chunks[index],
                lineage=prepared_lineages[index],
            )
        except ValueError as error:
            raise ValueError(f"ingest row {index}: {error}") from error

        records.append(record)

    return records
