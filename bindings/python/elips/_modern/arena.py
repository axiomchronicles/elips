from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, cast, overload

from ..core import (
    ChunkInfo,
    Database,
    DocumentAttachment,
    EmbeddingLineage,
    Filter,
    QueryPlan,
    Result,
    Vault,
)
from ..types import PayloadLike, StoredRecord, Vector
from ._records import build_record_inputs_from_columns, coerce_record_input
from .models import Hit, RecordInput, Row
from .typing import Embedder, RecordInputLike


class Arena:
    r"""Arena(db, name, *, embedder=None, text_slot="__elips_text__") -> Arena

    Typed convenience wrapper over a single :class:`elips.Vault`.

    Args:
        db (Database): Open ELIPS database handle.
        name (str): Vault name.
        embedder (Embedder, optional): Batch embedder used when the database
            does not expose a native text embedder. Default: ``None``.
        text_slot (str, optional): Reserved compatibility argument preserved for
            older wrappers. Default: ``"__elips_text__"``.

    Examples::

        >>> import elips
        >>> engine = elips.connect(":memory:", dimension=2)
        >>> arena = engine.arena("documents")
        >>> key = arena.write(text="alpha note", meta={"kind": "design"})
        >>> arena.pull([key])[0].text
        'alpha note'
        >>> engine.close()
    """

    def __init__(
        self,
        db: Database,
        name: str,
        *,
        embedder: Embedder | None = None,
        text_slot: str = "__elips_text__",
    ) -> None:
        self._db = db
        self._vault = db.vault(name)
        self._embedder = embedder
        self._text_slot = text_slot

    @property
    def name(self) -> str:
        r"""name -> str

        Return the vault name wrapped by this arena.
        """

        return self._vault.name

    @property
    def raw(self) -> Vault:
        r"""raw -> Vault

        Return the underlying low-level vault handle.
        """

        return self._vault

    def count(self) -> int:
        r"""count() -> int

        Return the number of records in the arena.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> arena = engine.arena("documents")
            >>> arena.count()
            0
            >>> engine.close()
        """

        return self._vault.count()

    @overload
    def write(self, record: RecordInputLike, /) -> str:
        ...

    @overload
    def write(
        self,
        *,
        vector: Vector | None = ...,
        text: str | None = ...,
        meta: PayloadLike | None = ...,
        key: str | None = ...,
        document: DocumentAttachment | None = ...,
        chunk: ChunkInfo | None = ...,
        lineage: EmbeddingLineage | None = ...,
    ) -> str:
        ...

    def write(
        self,
        record: RecordInputLike | None = None,
        /,
        *,
        vector: Vector | None = None,
        text: str | None = None,
        meta: PayloadLike | None = None,
        key: str | None = None,
        document: DocumentAttachment | None = None,
        chunk: ChunkInfo | None = None,
        lineage: EmbeddingLineage | None = None,
    ) -> str:
        r"""write(record=None, /, *, vector=None, text=None, meta=None, key=None, document=None, chunk=None, lineage=None) -> str

        Write a single record into the arena.

        Args:
            record (RecordInput or mapping, optional): Structured record input.
                Default: ``None``.
            vector (Sequence[float], optional): Explicit embedding vector.
                Default: ``None``.
            text (str, optional): Source text for native text ingestion or
                Python-side embedding. Default: ``None``.
            meta (Mapping[str, MetaValue], optional): Metadata payload.
                Default: ``None``.
            key (str, optional): Explicit record identifier. Default: ``None``.
            document (DocumentAttachment, optional): Full document attachment.
                Default: ``None``.
            chunk (ChunkInfo, optional): Chunk lineage. Default: ``None``.
            lineage (EmbeddingLineage, optional): Embedding provenance.
                Default: ``None``.

        Returns:
            str: Assigned record identifier.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> arena = engine.arena("documents")
            >>> key = arena.write(text="alpha note", meta={"kind": "design"})
            >>> isinstance(key, str)
            True
            >>> record = elips.RecordInput(text="beta note", meta={"kind": "ops"})
            >>> isinstance(arena.write(record), str)
            True
            >>> engine.close()
        """

        if record is not None and any(
            value is not None
            for value in (vector, text, meta, key, document, chunk, lineage)
        ):
            raise ValueError(
                "write accepts either a structured record or keyword fields, not both"
            )

        normalized = (
            coerce_record_input(record)
            if record is not None
            else RecordInput(
                vector=vector,
                text=text,
                meta=meta,
                key=key,
                document=document,
                chunk=chunk,
                lineage=lineage,
            )
        )
        return self.write_many([normalized])[0]

    def write_many(self, records: Sequence[RecordInputLike]) -> list[str]:
        r"""write_many(records) -> list[str]

        Write a batch of structured records into the arena.

        Args:
            records (Sequence[RecordInput or mapping]): Batch of records. Each
                record may provide a vector, text, or both.

        Returns:
            list[str]: Assigned record identifiers in input order.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> arena = engine.arena("documents")
            >>> keys = arena.write_many([
            ...     elips.RecordInput(text="alpha note", meta={"kind": "alpha"}),
            ...     {"text": "beta note", "meta": {"kind": "beta"}},
            ... ])
            >>> len(keys)
            2
            >>> engine.close()
        """

        normalized_records = [coerce_record_input(record) for record in records]
        if not normalized_records:
            raise ValueError("write_many requires at least one record")

        resolved_vectors, embedded_indices = self._resolve_vectors(normalized_records)

        assigned: list[str] = []
        for index, record in enumerate(normalized_records):
            vector = resolved_vectors[index]
            payload = record.materialize_meta()
            lineage = self._resolve_lineage(
                record,
                vector_generated=index in embedded_indices,
            )

            if vector is None:
                text = record.document_text
                if text is None:
                    raise ValueError("records without vectors require text input")
                if record.has_custom_document_metadata():
                    raise ValueError(
                        "text-only writes with custom document metadata require "
                        "an explicit vector or the low-level Vault.place() API"
                    )

                assigned.append(
                    self._vault.place_document(
                        text,
                        payload,
                        id=record.key,
                        chunk=record.chunk,
                        lineage=lineage,
                    )
                )
                continue

            assigned.append(
                self._vault.place(
                    vector,
                    payload,
                    id=record.key,
                    document=record.materialize_document(),
                    chunk=record.chunk,
                    lineage=lineage,
                )
            )

        return assigned

    @overload
    def ingest(self, records: Sequence[RecordInputLike], /) -> list[str]:
        ...

    @overload
    def ingest(
        self,
        *,
        vectors: Sequence[Vector | None] | None = ...,
        texts: Sequence[str | None] | None = ...,
        meta: Sequence[PayloadLike | None] | None = ...,
        keys: Sequence[str | None] | None = ...,
        documents: Sequence[DocumentAttachment | None] | None = ...,
        chunks: Sequence[ChunkInfo | None] | None = ...,
        lineages: Sequence[EmbeddingLineage | None] | None = ...,
    ) -> list[str]:
        ...

    def ingest(
        self,
        records: Sequence[RecordInputLike] | None = None,
        /,
        *,
        vectors: Sequence[Vector | None] | None = None,
        texts: Sequence[str | None] | None = None,
        meta: Sequence[PayloadLike | None] | None = None,
        keys: Sequence[str | None] | None = None,
        documents: Sequence[DocumentAttachment | None] | None = None,
        chunks: Sequence[ChunkInfo | None] | None = None,
        lineages: Sequence[EmbeddingLineage | None] | None = None,
    ) -> list[str]:
        r"""ingest(records=None, /, *, vectors=None, texts=None, meta=None, keys=None, documents=None, chunks=None, lineages=None) -> list[str]

        Write a batch of records using either structured records or the legacy
        column-oriented wrapper syntax.

        Args:
            records (Sequence[RecordInput or mapping], optional): Structured
                batch input. Default: ``None``.
            vectors (Sequence[Sequence[float] or None], optional): Explicit
                vectors for the legacy column-oriented API. Default: ``None``.
            texts (Sequence[str or None], optional): Text inputs for the legacy
                column-oriented API. Default: ``None``.
            meta (Sequence[Mapping[str, MetaValue] or None], optional): Legacy
                metadata batch. Default: ``None``.
            keys (Sequence[str or None], optional): Legacy key batch. Default:
                ``None``.
            documents (Sequence[DocumentAttachment or None], optional): Legacy
                document batch. Default: ``None``.
            chunks (Sequence[ChunkInfo or None], optional): Legacy chunk batch.
                Default: ``None``.
            lineages (Sequence[EmbeddingLineage or None], optional): Legacy
                lineage batch. Default: ``None``.

        Returns:
            list[str]: Assigned record identifiers in input order.

        .. note::
            The structured :meth:`write_many` API is the preferred form for new
            code because it keeps related fields together and is easier to type
            check.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> arena = engine.arena("documents")
            >>> keys = arena.ingest([
            ...     elips.RecordInput(text="alpha note", meta={"kind": "alpha"}),
            ...     {"text": "beta note", "meta": {"kind": "beta"}},
            ... ])
            >>> len(keys)
            2
            >>> legacy = arena.ingest(
            ...     texts=["gamma note"],
            ...     meta=[{"kind": "gamma"}],
            ... )
            >>> len(legacy)
            1
            >>> engine.close()
        """

        if records is not None:
            if any(
                value is not None
                for value in (vectors, texts, meta, keys, documents, chunks, lineages)
            ):
                raise ValueError(
                    "ingest accepts either structured records or column batches, not both"
                )

            return self.write_many(records)

        normalized_records = build_record_inputs_from_columns(
            vectors=vectors,
            texts=texts,
            meta=meta,
            keys=keys,
            documents=documents,
            chunks=chunks,
            lineages=lineages,
        )
        return self.write_many(normalized_records)

    def merge(
        self,
        records: Sequence[RecordInputLike] | None = None,
        /,
        *,
        vectors: Sequence[Vector | None] | None = None,
        texts: Sequence[str | None] | None = None,
        meta: Sequence[PayloadLike | None] | None = None,
        keys: Sequence[str | None] | None = None,
        documents: Sequence[DocumentAttachment | None] | None = None,
        chunks: Sequence[ChunkInfo | None] | None = None,
        lineages: Sequence[EmbeddingLineage | None] | None = None,
    ) -> list[str]:
        r"""merge(records=None, /, *, vectors=None, texts=None, meta=None, keys=None, documents=None, chunks=None, lineages=None) -> list[str]

        Compatibility alias for :meth:`ingest`.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> arena = engine.arena("documents")
            >>> key = arena.write(text="alpha old", meta={"rev": 1})
            >>> rows = arena.merge(
            ...     texts=["alpha new"],
            ...     meta=[{"rev": 2}],
            ...     keys=[key],
            ... )
            >>> len(rows)
            1
            >>> engine.close()
        """

        return self.ingest(
            records,
            vectors=vectors,
            texts=texts,
            meta=meta,
            keys=keys,
            documents=documents,
            chunks=chunks,
            lineages=lineages,
        )

    def probe(
        self,
        vector: Vector,
        *,
        top: int = 10,
        where: Filter | None = None,
        max_distance: float | None = None,
        include_vectors: bool = False,
    ) -> list[Hit]:
        r"""probe(vector, *, top=10, where=None, max_distance=None, include_vectors=False) -> list[Hit]

        Run vector similarity search.

        Args:
            vector (Sequence[float]): Query vector.
            top (int, optional): Maximum number of hits. Default: ``10``.
            where (Filter, optional): Metadata filter. Default: ``None``.
            max_distance (float, optional): Distance threshold. Default:
                ``None``.
            include_vectors (bool, optional): Whether to hydrate stored vectors
                on each hit. Default: ``False``.

        Returns:
            list[Hit]: Search hits sorted by distance.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> arena = engine.arena("documents")
            >>> _ = arena.write(vector=[1.0, 0.0], text="alpha")
            >>> arena.probe([1.0, 0.0], top=1)[0].text
            'alpha'
            >>> engine.close()
        """

        results = self._vault.seek(
            vector,
            top=top,
            where=where if where is not None else Filter(),
            threshold=max_distance,
        )
        return [
            self._hit_from_result(result, include_vectors=include_vectors)
            for result in results
        ]

    def probe_text(
        self,
        text: str,
        *,
        top: int = 10,
        where: Filter | None = None,
        max_distance: float | None = None,
        include_vectors: bool = False,
        lexical_weight: float = 0.25,
    ) -> list[Hit]:
        r"""probe_text(text, *, top=10, where=None, max_distance=None, include_vectors=False, lexical_weight=0.25) -> list[Hit]

        Run text-first retrieval.

        When the database has a native text embedder, the call uses
        :meth:`elips.Vault.seek_text`. Otherwise the arena falls back to the
        configured Python embedder and issues a hybrid search.

        Args:
            text (str): Query text.
            top (int, optional): Maximum number of hits. Default: ``10``.
            where (Filter, optional): Metadata filter. Default: ``None``.
            max_distance (float, optional): Distance threshold. Default:
                ``None``.
            include_vectors (bool, optional): Whether to hydrate stored vectors
                on each hit. Default: ``False``.
            lexical_weight (float, optional): Hybrid lexical weight used when
                the arena falls back to Python-side embedding. Default:
                ``0.25``.

        Returns:
            list[Hit]: Search hits sorted by distance.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> arena = engine.arena("documents")
            >>> _ = arena.write(text="alpha design note", meta={"kind": "design"})
            >>> arena.probe_text("alpha", top=1)[0].text
            'alpha design note'
            >>> engine.close()
        """

        if self._has_native_text():
            results = self._vault.seek_text(
                text,
                top=top,
                where=where if where is not None else Filter(),
                threshold=max_distance,
            )
        else:
            embedder = self._require_embedder()
            query = embedder([text])
            if len(query) != 1:
                raise ValueError(
                    "embedder must return exactly one vector for a single text probe"
                )
            results = self._vault.seek_hybrid(
                query[0],
                text,
                top=top,
                where=where if where is not None else Filter(),
                threshold=max_distance,
                lexical_weight=lexical_weight,
            )

        return [
            self._hit_from_result(result, include_vectors=include_vectors)
            for result in results
        ]

    def probe_hybrid(
        self,
        vector: Vector,
        text: str,
        *,
        top: int = 10,
        where: Filter | None = None,
        max_distance: float | None = None,
        lexical_weight: float = 0.25,
        include_vectors: bool = False,
    ) -> list[Hit]:
        r"""probe_hybrid(vector, text, *, top=10, where=None, max_distance=None, lexical_weight=0.25, include_vectors=False) -> list[Hit]

        Run hybrid retrieval that blends vector distance with lexical overlap.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> arena = engine.arena("documents")
            >>> _ = arena.write(text="alpha design note")
            >>> len(arena.probe_hybrid([1.0, 0.0], "alpha", top=1))
            1
            >>> engine.close()
        """

        results = self._vault.seek_hybrid(
            vector,
            text,
            top=top,
            where=where if where is not None else Filter(),
            threshold=max_distance,
            lexical_weight=lexical_weight,
        )
        return [
            self._hit_from_result(result, include_vectors=include_vectors)
            for result in results
        ]

    def explain(
        self,
        vector: Vector,
        *,
        top: int = 10,
        where: Filter | None = None,
        max_distance: float | None = None,
        has_text_component: bool = False,
    ) -> QueryPlan:
        r"""explain(vector, *, top=10, where=None, max_distance=None, has_text_component=False) -> QueryPlan

        Return the planner decision for a query shape.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> arena = engine.arena("documents")
            >>> _ = arena.write(text="alpha design note", meta={"kind": "design"})
            >>> plan = arena.explain([1.0, 0.0], top=1, has_text_component=True)
            >>> hasattr(plan, "strategy")
            True
            >>> engine.close()
        """

        return self._vault.explain_seek(
            vector,
            top=top,
            where=where if where is not None else Filter(),
            threshold=max_distance,
            has_text_component=has_text_component,
        )

    def pull(
        self,
        keys: Sequence[str],
        *,
        include_vectors: bool = True,
    ) -> list[Row]:
        r"""pull(keys, *, include_vectors=True) -> list[Row]

        Fetch records by key and return typed rows.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> arena = engine.arena("documents")
            >>> key = arena.write(text="alpha note")
            >>> arena.pull([key])[0].text
            'alpha note'
            >>> engine.close()
        """

        rows: list[Row] = []
        for key in keys:
            fetched = cast(Optional[StoredRecord], self._vault.fetch(key))
            if fetched is None:
                continue
            rows.append(self._row_from_record(fetched, include_vectors=include_vectors))
        return rows

    def sweep(
        self,
        *,
        where: Filter | None = None,
        offset: int = 0,
        limit: int | None = None,
        include_vectors: bool = False,
    ) -> list[Row]:
        r"""sweep(*, where=None, offset=0, limit=None, include_vectors=False) -> list[Row]

        Scan arena records and return typed rows.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> arena = engine.arena("documents")
            >>> _ = arena.write(text="alpha note")
            >>> len(arena.sweep())
            1
            >>> engine.close()
        """

        rows = self._vault.scan(
            where=where if where is not None else Filter(),
            offset=offset,
            limit=-1 if limit is None else limit,
        )
        return [
            self._row_from_record(cast(StoredRecord, row), include_vectors=include_vectors)
            for row in rows
        ]

    def discard(
        self,
        keys: Sequence[str] | None = None,
        *,
        where: Filter | None = None,
    ) -> int:
        r"""discard(keys=None, *, where=None) -> int

        Delete records by key, filter, or both.

        Examples::

            >>> import elips
            >>> engine = elips.connect(":memory:", dimension=2)
            >>> arena = engine.arena("documents")
            >>> key = arena.write(text="alpha note", meta={"kind": "design"})
            >>> arena.discard([key])
            1
            >>> engine.close()
        """

        victims: list[str] = []
        seen: set[str] = set()

        if keys is not None:
            for key in keys:
                if key not in seen:
                    seen.add(key)
                    victims.append(key)

        if where is not None:
            for row in self._vault.scan(where=where, offset=0, limit=-1):
                record = cast(StoredRecord, row)
                key = record["id"]
                if key not in seen:
                    seen.add(key)
                    victims.append(key)

        if not victims:
            raise ValueError("discard requires at least one explicit key or a filter")

        removed = 0
        for key in victims:
            removed += int(self._vault.erase(key))
        return removed

    def _resolve_vectors(
        self,
        records: Sequence[RecordInput],
    ) -> tuple[list[Vector | None], set[int]]:
        resolved_vectors = [record.vector for record in records]
        missing_indices = [
            index for index, vector in enumerate(resolved_vectors) if vector is None
        ]
        if not missing_indices:
            return resolved_vectors, set()

        if self._has_native_text():
            return resolved_vectors, set()

        embedder = self._require_embedder()
        texts: list[str] = []
        for index in missing_indices:
            text = records[index].document_text
            if text is None:
                raise ValueError(
                    "records without vectors require text or document text"
                )
            texts.append(text)

        embedded = embedder(texts)
        if len(embedded) != len(missing_indices):
            raise ValueError("embedder returned a batch with the wrong length")

        for index, vector in zip(missing_indices, embedded):
            resolved_vectors[index] = vector

        return resolved_vectors, set(missing_indices)

    def _row_from_record(
        self,
        record: StoredRecord,
        *,
        include_vectors: bool,
    ) -> Row:
        return Row(
            key=record["id"],
            meta=dict(record["data"]),
            document=record["document"],
            vector=tuple(record["vector"]) if include_vectors else None,
            chunk=record["chunk"],
            lineage=record["lineage"],
        )

    def _hit_from_result(self, result: Result, *, include_vectors: bool) -> Hit:
        fetched = (
            cast(Optional[StoredRecord], self._vault.fetch(result.id))
            if include_vectors
            else None
        )
        document = result.document if result.document is not None else (
            fetched["document"] if fetched is not None else None
        )
        return Hit(
            key=result.id,
            distance=result.distance,
            meta=dict(result.data),
            document=document,
            vector=tuple(fetched["vector"]) if fetched is not None else None,
            chunk=result.chunk if result.chunk is not None else (
                fetched["chunk"] if fetched is not None else None
            ),
            lineage=result.lineage if result.lineage is not None else (
                fetched["lineage"] if fetched is not None else None
            ),
        )

    def _resolve_lineage(
        self,
        record: RecordInput,
        *,
        vector_generated: bool,
    ) -> EmbeddingLineage | None:
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

    def _has_native_text(self) -> bool:
        return bool(self._db.config.has_text_embedder)

    def _require_embedder(self) -> Embedder:
        if self._embedder is None:
            raise ValueError("this arena needs an embedder for text-first operations")
        return self._embedder


__all__ = ["Arena"]
