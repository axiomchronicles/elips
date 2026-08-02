r"""Typed collection facade over the native :class:`elips.Vault` handle.

:class:`Collection` owns the overload dispatch that lets one ``write`` accept a
vector, a text, or a :class:`~elips.RecordInput`. Two concerns it used to carry
inline now live beside it: embedder resolution in ``_embedding``, and
native-dict-to-model conversion in ``_hydration``.

Examples::

    >>> import elips
    >>> db = elips.connect(":memory:", dimension=2)
    >>> docs = db.collection("documents")
    >>> _ = docs.add(vector=[1.0, 0.0], meta={"kind": "design"})
    >>> docs.count()
    1
    >>> db.close()
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast, overload

from elips._internal.coercion import (
    build_record_inputs_from_columns,
    coerce_record_input,
)
from elips._internal.protocols import Embedder, RecordInputLike
from elips._native import (
    ChunkInfo,
    Database,
    DocumentAttachment,
    EmbeddingLineage,
    Filter,
    QueryPlan,
    Vault,
)
from elips.collection._embedding import (
    has_native_text_embedder,
    resolve_lineage,
    resolve_vectors,
)
from elips.collection._hydration import hit_from_result, row_from_record
from elips.records.models import CollectionHealth, Hit, RecordInput, Row
from elips.typing import PayloadLike, StoredRecord, Vector


class Collection:
    r"""Collection(db, name, *, embedder=None, text_slot="__elips_text__") -> Collection

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
        >>> db = elips.connect(":memory:", dimension=2)
        >>> docs = db.collection("documents")
        >>> key = docs.write(text="alpha note", meta={"kind": "design"})
        >>> docs.pull([key])[0].text
        'alpha note'
        >>> db.close()
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

        Return the vault name wrapped by this collection.
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

        Return the number of records in the collection.

        Examples::

            >>> import elips
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> docs.count()
            0
            >>> db.close()
        """

        return self._vault.count()

    @overload
    def write(self, record: RecordInputLike, /) -> str:
        ...

    @overload
    def write(
        self,
        record: None = ...,
        /,
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

        Write a single record into the collection.

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
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> key = docs.write(text="alpha note", meta={"kind": "design"})
            >>> isinstance(key, str)
            True
            >>> record = elips.RecordInput(text="beta note", meta={"kind": "ops"})
            >>> isinstance(docs.write(record), str)
            True
            >>> db.close()
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

        Write a batch of structured records into the collection.

        Args:
            records (Sequence[RecordInput or mapping]): Batch of records. Each
                record may provide a vector, text, or both.

        Returns:
            list[str]: Assigned record identifiers in input order.

        Examples::

            >>> import elips
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> keys = docs.write_many([
            ...     elips.RecordInput(text="alpha note", meta={"kind": "alpha"}),
            ...     {"text": "beta note", "meta": {"kind": "beta"}},
            ... ])
            >>> len(keys)
            2
            >>> db.close()
        """

        normalized_records = [coerce_record_input(record) for record in records]
        if not normalized_records:
            raise ValueError("write_many requires at least one record")

        resolved_vectors, embedded_indices = resolve_vectors(
            normalized_records, db=self._db, embedder=self._embedder
        )

        assigned: list[str] = []
        for index, record in enumerate(normalized_records):
            vector = resolved_vectors[index]
            payload = record.materialize_meta()
            lineage = resolve_lineage(
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
        records: None = ...,
        /,
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
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> keys = docs.ingest([
            ...     elips.RecordInput(text="alpha note", meta={"kind": "alpha"}),
            ...     {"text": "beta note", "meta": {"kind": "beta"}},
            ... ])
            >>> len(keys)
            2
            >>> legacy = docs.ingest(
            ...     texts=["gamma note"],
            ...     meta=[{"kind": "gamma"}],
            ... )
            >>> len(legacy)
            1
            >>> db.close()
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
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> key = docs.write(text="alpha old", meta={"rev": 1})
            >>> rows = docs.merge(
            ...     texts=["alpha new"],
            ...     meta=[{"rev": 2}],
            ...     keys=[key],
            ... )
            >>> len(rows)
            1
            >>> db.close()
        """

        # Route explicitly: `ingest` overloads the structured and
        # column-oriented forms as mutually exclusive, so forwarding both
        # argument groups in one call matches neither signature.
        if records is not None:
            return self.ingest(records)
        return self.ingest(
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
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> _ = docs.write(vector=[1.0, 0.0], text="alpha")
            >>> docs.probe([1.0, 0.0], top=1)[0].text
            'alpha'
            >>> db.close()
        """

        results = self._vault.seek(
            vector,
            top=top,
            where=where if where is not None else Filter(),
            threshold=max_distance,
        )
        return [
            hit_from_result(
                result, vault=self._vault, include_vectors=include_vectors
            )
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
        :meth:`elips.Vault.seek_text`. Otherwise the collection falls back to the
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
                the collection falls back to Python-side embedding. Default:
                ``0.25``.

        Returns:
            list[Hit]: Search hits sorted by distance.

        Examples::

            >>> import elips
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> _ = docs.write(text="alpha design note", meta={"kind": "design"})
            >>> docs.probe_text("alpha", top=1)[0].text
            'alpha design note'
            >>> db.close()
        """

        if has_native_text_embedder(self._db):
            results = self._vault.seek_text(
                text,
                top=top,
                where=where if where is not None else Filter(),
                threshold=max_distance,
            )
        else:
            if self._embedder is None:
                raise ValueError(
                    "this collection needs an embedder for text-first operations"
                )
            query = self._embedder([text])
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
            hit_from_result(
                result, vault=self._vault, include_vectors=include_vectors
            )
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
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> _ = docs.write(text="alpha design note")
            >>> len(docs.probe_hybrid([1.0, 0.0], "alpha", top=1))
            1
            >>> db.close()
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
            hit_from_result(
                result, vault=self._vault, include_vectors=include_vectors
            )
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
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> _ = docs.write(text="alpha design note", meta={"kind": "design"})
            >>> plan = docs.explain([1.0, 0.0], top=1, has_text_component=True)
            >>> hasattr(plan, "strategy")
            True
            >>> db.close()
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
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> key = docs.write(text="alpha note")
            >>> docs.pull([key])[0].text
            'alpha note'
            >>> db.close()
        """

        rows: list[Row] = []
        for key in keys:
            fetched = cast(StoredRecord | None, self._vault.fetch(key))
            if fetched is None:
                continue
            rows.append(row_from_record(fetched, include_vectors=include_vectors))
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

        Scan collection records and return typed rows.

        Examples::

            >>> import elips
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> _ = docs.write(text="alpha note")
            >>> len(docs.sweep())
            1
            >>> db.close()
        """

        rows = self._vault.scan(
            where=where if where is not None else Filter(),
            offset=offset,
            limit=-1 if limit is None else limit,
        )
        return [
            row_from_record(cast(StoredRecord, row), include_vectors=include_vectors)
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
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> key = docs.write(text="alpha note", meta={"kind": "design"})
            >>> docs.discard([key])
            1
            >>> db.close()
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

    # ---- maintenance and introspection ----

    @property
    def pending_removals(self) -> int:
        r"""pending_removals -> int

        Records deleted from the index but not yet reclaimed.

        Deletes are tombstones: the node stays in the graph as a routing
        waypoint so live neighbours do not become unreachable. Search widens its
        beam to compensate, so recall holds, but the vectors and edges keep
        consuming memory until compaction.

        The index compacts itself as soon as tombstones reach its configured
        ``compaction_ratio``, so on a small collection this often reads ``0``
        immediately after a delete.

        Examples::

            >>> import elips
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> keys = [docs.write(vector=[float(i), 1.0]) for i in range(20)]
            >>> _ = docs.discard(keys[:2])        # 2/20 = 0.1, under the 0.2 default
            >>> docs.pending_removals
            2
            >>> db.close()
        """

        return self._vault.pending_removals

    @property
    def read_only(self) -> bool:
        r"""read_only -> bool

        Whether this collection currently refuses writes.
        """

        return self._vault.read_only

    @property
    def sealed(self) -> bool:
        r"""sealed -> bool

        Whether the owning engine has been closed.

        Writes to a sealed collection raise :class:`elips.StorageError` rather than
        succeeding and then failing to persist.
        """

        return self._vault.sealed

    def freeze(self, frozen: bool = True) -> None:
        r"""freeze(frozen=True) -> None

        Make this collection reject writes, or allow them again.

        Useful while serving a snapshot you do not want mutated, without
        reopening the whole database read-only.

        Args:
            frozen (bool, optional): ``True`` to refuse writes. Default:
                ``True``.

        Examples::

            >>> import elips
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> docs.freeze()
            >>> try:
            ...     docs.write(vector=[1.0, 0.0])
            ... except elips.StorageError:
            ...     print("refused")
            refused
            >>> docs.freeze(False)
            >>> isinstance(docs.write(vector=[1.0, 0.0]), str)
            True
            >>> db.close()
        """

        self._vault.set_read_only(frozen)

    def vacuum(self) -> None:
        r"""vacuum() -> None

        Reclaim index space held by deleted records in this collection.

        The index compacts itself once tombstones pass its configured
        ``compaction_ratio``, so routine churn needs no intervention. Call this
        after a bulk delete, when waiting for the threshold would hold a large
        amount of dead memory.

        Examples::

            >>> import elips
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> keys = [docs.write(vector=[float(i), 1.0]) for i in range(20)]
            >>> _ = docs.discard(keys[:2])
            >>> docs.vacuum()
            >>> docs.pending_removals
            0
            >>> db.close()
        """

        self._vault.vacuum()

    def rebuild(self) -> None:
        r"""rebuild() -> None

        Rebuild the index from the authoritative record store.

        Worth doing after a bulk load, since inserting in arrival order yields a
        lower-quality graph than rebuilding once the full set is known.
        """

        self._vault.rebuild_index()

    def compress(self) -> None:
        r"""compress() -> None

        Train a codebook over this collection and compress it in place, freeing the
        full-precision vectors.

        Requires a codec on the engine's configuration (pass ``quantization=``
        to :func:`elips.connect`). Raises :class:`ConfigError` if none is set,
        if the collection is empty, or if it has already been compressed.

        Compression is a separate step from configuring it because product
        quantization cannot encode the first record: a codebook has to be
        learned from real data. Before this call the collection stores full fp32;
        after it, writes are encoded on arrival, and :meth:`pull` and
        :meth:`sweep` return reconstructions with ``approximate`` set.

        Holds the collection's writer lock throughout, so treat it like
        :meth:`vacuum` -- a maintenance operation, not part of the write path.

        Examples::

            >>> import elips
            >>> db = elips.connect(":memory:", dimension=8, quantization="sq8")
            >>> docs = db.collection("documents")
            >>> for i in range(32):
            ...     _ = docs.write(vector=[float(i % 4)] * 8)
            >>> docs.health().codec
            'none'
            >>> docs.compress()
            >>> docs.health().codec
            'sq8'
            >>> docs.probe([1.0] * 8, top=1)[0].approximate
            True
            >>> db.close()
        """

        self._vault.quantize()

    def health(self) -> CollectionHealth:
        r"""health() -> CollectionHealth

        Return a point-in-time health snapshot of this collection.

        Returns:
            CollectionHealth: Live count, tombstone count, dimension, metric, and
            lifecycle flags.

        Examples::

            >>> import elips
            >>> db = elips.connect(":memory:", dimension=2)
            >>> docs = db.collection("documents")
            >>> _ = docs.write(vector=[1.0, 0.0])
            >>> health = docs.health()
            >>> (health.live, health.tombstone_ratio)
            (1, 0.0)
            >>> db.close()
        """

        info = self._vault.info()
        return CollectionHealth(
            name=self._vault.name,
            live=info.count,
            pending_removals=self._vault.pending_removals,
            dimension=info.dimension,
            metric=info.metric,
            read_only=self._vault.read_only,
            sealed=self._vault.sealed,
            codec=info.codec,
            code_bytes=info.code_bytes,
        )

    # ---- conventional names ----
    #
    # `add` and `search` are what every comparable library calls these, and are
    # the names the public API leads with. The originals stay: they are not
    # deprecated, just less discoverable.

    add = write
    """add(record=None, /, **fields) -> str: alias for :meth:`write`."""

    add_many = write_many
    """add_many(records) -> list[str]: alias for :meth:`write_many`."""

    search = probe_text
    """search(text, **kwargs) -> list[Hit]: alias for :meth:`probe_text`."""

    search_vector = probe
    """search_vector(vector, **kwargs) -> list[Hit]: alias for :meth:`probe`."""

    search_hybrid = probe_hybrid
    """search_hybrid(vector, text, **kwargs) -> list[Hit]: alias for :meth:`probe_hybrid`."""

    delete = discard
    """delete(keys=None, *, where=None) -> int: alias for :meth:`discard`."""

    get = pull
    """get(keys, *, include_vectors=True) -> list[Row]: alias for :meth:`pull`."""


#: The collection type was called ``Arena`` before 1.1. ``arena`` also collides
#: with the allocator sense of the word in a project that does its own memory
#: management, which is the other reason for the rename.
Arena = Collection

__all__ = ["Arena", "Collection"]
