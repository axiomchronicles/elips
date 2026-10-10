r"""Typed collection facade over the native :class:`elips.native.Vault` handle.

:class:`Collection` is the Chroma-style surface: ``add/upsert/get/update/delete/
query/count/peek/modify`` with ``ids/embeddings/documents/metadatas/n_results/
where/include`` kwargs. The pre-Phase-5 verbs (``write/ingest/merge/probe*/pull/
sweep/discard``) remain as deprecated shims that emit :class:`DeprecationWarning`
and delegate to the private implementations the canonical methods also use.

Examples::

    >>> import elips
    >>> client = elips.Client(dimension=2)
    >>> docs = client.get_or_create_collection("documents")
    >>> _ = docs.add(embeddings=[[1.0, 0.0]], metadatas=[{"kind": "design"}])
    >>> docs.count()
    1
    >>> client.close()
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

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
from elips.collection._hydration import (
    RESERVED_ID_KEY,
    derive_record_id,
    hit_from_result,
    is_native_id,
    resolve_native_id,
    row_from_record,
)
from elips.exceptions import NotFoundError
from elips.records.models import CollectionHealth, Hit, RecordInput, Row
from elips.typing import PayloadLike, Vector


def _deprecated(old: str, new: str) -> None:
    warnings.warn(
        f"Collection.{old} is deprecated; use Collection.{new} instead",
        DeprecationWarning,
        stacklevel=3,
    )


class Collection:
    r"""Chroma-style typed wrapper over a single native :class:`elips.native.Vault`."""

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
        """Return the collection name."""
        return self._vault.name

    @property
    def native(self) -> Vault:
        """The underlying low-level native handle (explicit escape hatch)."""
        return self._vault

    @property
    def raw(self) -> Vault:
        """.. deprecated:: use :attr:`native`."""
        _deprecated("raw", "native")
        return self._vault

    def count(self) -> int:
        """Return the number of records in the collection."""
        return self._vault.count()

    # ---- private implementations (shared by canonical + deprecated names) ----

    def _write(
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
        if record is not None and any(
            value is not None
            for value in (vector, text, meta, key, document, chunk, lineage)
        ):
            raise ValueError(
                "add accepts either a structured record or keyword fields, not both"
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
        return self._write_many([normalized])[0]

    def _write_many(self, records: Sequence[RecordInputLike]) -> list[str]:
        normalized_records = [coerce_record_input(record) for record in records]
        if not normalized_records:
            raise ValueError("add requires at least one record")

        resolved_vectors, embedded_indices = resolve_vectors(
            normalized_records, db=self._db, embedder=self._embedder
        )

        assigned: list[str] = []
        for index, record in enumerate(normalized_records):
            vector = resolved_vectors[index]
            payload = record.materialize_meta()
            key = record.key
            # Chroma allows arbitrary string ids; a native RecordID is 32 hex
            # digits. Non-native ids are hashed to a stable RecordID and the
            # original is stashed under RESERVED_ID_KEY so it survives reopen.
            if RESERVED_ID_KEY in payload:
                raise ValueError(
                    f"{RESERVED_ID_KEY!r} is a reserved metadata key and cannot "
                    "be set by callers"
                )
            if key is not None and not is_native_id(key):
                payload[RESERVED_ID_KEY] = key
                key = derive_record_id(key)
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
                        "an explicit vector or the low-level Vault.add() API"
                    )
                assigned.append(
                    self._external_id(
                        self._vault.place_document(
                            text,
                            payload,
                            id=key,
                            chunk=record.chunk,
                            lineage=lineage,
                        ),
                        payload,
                    )
                )
                continue

            assigned.append(
                self._external_id(
                    self._vault.place(
                        vector,
                        payload,
                        id=key,
                        document=record.materialize_document(),
                        chunk=record.chunk,
                        lineage=lineage,
                    ),
                    payload,
                )
            )
        return assigned

    @staticmethod
    def _external_id(native_id: str, payload: Mapping[str, Any]) -> str:
        """Report the caller's original string id when one was stashed."""
        original = payload.get(RESERVED_ID_KEY)
        return original if original is not None else native_id

    def _ingest(
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
        if records is not None:
            if any(
                value is not None
                for value in (vectors, texts, meta, keys, documents, chunks, lineages)
            ):
                raise ValueError(
                    "add accepts either structured records or column batches, not both"
                )
            return self._write_many(records)

        normalized_records = build_record_inputs_from_columns(
            vectors=vectors,
            texts=texts,
            meta=meta,
            keys=keys,
            documents=documents,
            chunks=chunks,
            lineages=lineages,
        )
        return self._write_many(normalized_records)

    def _probe(
        self,
        vector: Vector,
        *,
        top: int = 10,
        where: Filter | None = None,
        max_distance: float | None = None,
        include_vectors: bool = False,
    ) -> list[Hit]:
        results = self._vault.seek(
            vector,
            top=top,
            where=where if where is not None else Filter(),
            threshold=max_distance,
        )
        return [
            hit_from_result(result, vault=self._vault, include_vectors=include_vectors)
            for result in results
        ]

    def _probe_text(
        self,
        text: str,
        *,
        top: int = 10,
        where: Filter | None = None,
        max_distance: float | None = None,
        include_vectors: bool = False,
        lexical_weight: float = 0.25,
    ) -> list[Hit]:
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
                    "embedder must return exactly one vector for a single text query"
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
            hit_from_result(result, vault=self._vault, include_vectors=include_vectors)
            for result in results
        ]

    def _probe_hybrid(
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
        results = self._vault.seek_hybrid(
            vector,
            text,
            top=top,
            where=where if where is not None else Filter(),
            threshold=max_distance,
            lexical_weight=lexical_weight,
        )
        return [
            hit_from_result(result, vault=self._vault, include_vectors=include_vectors)
            for result in results
        ]

    def _pull(self, keys: Sequence[str], *, include_vectors: bool = True) -> list[Row]:
        rows: list[Row] = []
        for key in keys:
            fetched = self._vault.fetch(resolve_native_id(key))
            if fetched is None:
                continue
            rows.append(row_from_record(fetched, include_vectors=include_vectors))
        return rows

    def _sweep(
        self,
        *,
        where: Filter | None = None,
        offset: int = 0,
        limit: int | None = None,
        include_vectors: bool = False,
    ) -> list[Row]:
        rows = self._vault.scan(
            where=where if where is not None else Filter(),
            offset=offset,
            limit=-1 if limit is None else limit,
        )
        return [row_from_record(row, include_vectors=include_vectors) for row in rows]

    def _discard(
        self,
        keys: Sequence[str] | None = None,
        *,
        where: Filter | None = None,
    ) -> int:
        victims: list[str] = []
        seen: set[str] = set()

        if keys is not None:
            for key in keys:
                native = resolve_native_id(key)
                if native not in seen:
                    seen.add(native)
                    victims.append(native)

        if where is not None:
            for row in self._vault.scan(where=where, offset=0, limit=-1):
                key = row["id"]
                if key not in seen:
                    seen.add(key)
                    victims.append(key)

        if not victims:
            raise ValueError("delete requires at least one explicit id or a filter")

        removed = 0
        for key in victims:
            removed += int(self._vault.erase(key))
        return removed

    @staticmethod
    def _wants(include: Sequence[str] | None, legacy: bool | None, default: bool) -> bool:
        if include is not None:
            return "embeddings" in include
        if legacy is not None:
            return legacy
        return default

    @staticmethod
    def _split_documents(
        documents: Sequence[Any] | None,
    ) -> tuple[Sequence[str] | None, Sequence[DocumentAttachment] | None]:
        """Chroma ``documents`` are raw text; disambiguate attachments by type."""
        if documents is None:
            return None, None
        items = list(documents)
        if any(isinstance(d, DocumentAttachment) for d in items if d is not None):
            return None, items
        return items, None

    # ---- canonical Chroma surface ----

    def add(
        self,
        records: RecordInputLike | Sequence[RecordInputLike] | None = None,
        /,
        *,
        ids: Sequence[str] | None = None,
        embeddings: Sequence[Vector] | None = None,
        documents: Sequence[Any] | None = None,
        metadatas: Sequence[PayloadLike | None] | None = None,
        vectors: Sequence[Vector | None] | None = None,
        texts: Sequence[str | None] | None = None,
        meta: Any = None,
        keys: Sequence[str] | None = None,
        chunks: Sequence[ChunkInfo | None] | None = None,
        lineages: Sequence[EmbeddingLineage | None] | None = None,
        vector: Vector | None = None,
        embedding: Vector | None = None,
        text: str | None = None,
        document: DocumentAttachment | None = None,
        metadata: PayloadLike | None = None,
        key: str | None = None,
        id: str | None = None,
        chunk: ChunkInfo | None = None,
        lineage: EmbeddingLineage | None = None,
    ) -> str | list[str]:
        r"""Add records (Chroma column form, structured records, or a single record).

        Column form uses ``ids/embeddings/documents/metadatas``; a single record
        uses ``vector/text/metadata/id``. Returns the assigned id(s). ``place``
        has upsert semantics, so re-adding an existing id overwrites it.
        """
        if records is not None:
            if isinstance(records, (RecordInput, Mapping)):
                return self._write_many([coerce_record_input(records)])[0]
            return self._write_many([coerce_record_input(r) for r in records])

        text_docs, attachments = self._split_documents(documents)
        columns = (
            ids,
            embeddings,
            documents,
            metadatas,
            vectors,
            texts,
            keys,
            chunks,
            lineages,
        )
        if any(value is not None for value in columns):
            return self._ingest(
                vectors=embeddings if embeddings is not None else vectors,
                texts=texts if texts is not None else text_docs,
                meta=metadatas if metadatas is not None else meta,
                keys=ids if ids is not None else keys,
                documents=attachments,
                chunks=chunks,
                lineages=lineages,
            )

        return self._write(
            vector=vector if vector is not None else embedding,
            text=text,
            meta=metadata if metadata is not None else meta,
            key=key if key is not None else id,
            document=document,
            chunk=chunk,
            lineage=lineage,
        )

    def upsert(
        self,
        records: RecordInputLike | Sequence[RecordInputLike] | None = None,
        /,
        **kwargs: Any,
    ) -> str | list[str]:
        """Insert or overwrite records. Same semantics as :meth:`add`."""
        return self.add(records, **kwargs)

    def update(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Vector] | None = None,
        documents: Sequence[Any] | None = None,
        metadatas: Sequence[PayloadLike | None] | None = None,
    ) -> list[str]:
        """Overwrite existing records; raise :class:`NotFoundError` if an id is absent."""
        id_list = list(ids)
        for rid in id_list:
            if self._vault.fetch(rid) is None:
                raise NotFoundError(f"update: no record with id {rid!r}")
        result = self.add(
            ids=id_list, embeddings=embeddings, documents=documents, metadatas=metadatas
        )
        return result if isinstance(result, list) else [result]

    def get(
        self,
        ids: Sequence[str] | None = None,
        *,
        where: Filter | None = None,
        where_document: Any = None,
        limit: int | None = None,
        offset: int = 0,
        include: Sequence[str] | None = None,
        include_vectors: bool | None = None,
    ) -> list[Row]:
        """Fetch records by id, or scan by ``where``/``limit``/``offset``."""
        if where_document is not None:
            raise NotImplementedError(
                "where_document (full-text filtering) is not implemented yet"
            )
        if ids is not None:
            want = self._wants(include, include_vectors, default=True)
            return self._pull(ids, include_vectors=want)
        want = self._wants(include, include_vectors, default=False)
        return self._sweep(
            where=where, offset=offset, limit=limit, include_vectors=want
        )

    def delete(
        self,
        ids: Sequence[str] | None = None,
        *,
        where: Filter | None = None,
        where_document: Any = None,
    ) -> int:
        """Delete records by id, filter, or both."""
        if where_document is not None:
            raise NotImplementedError(
                "where_document (full-text filtering) is not implemented yet"
            )
        return self._discard(ids, where=where)

    def peek(self, n: int = 10, *, include_vectors: bool = False) -> list[Row]:
        """Return up to ``n`` records (Chroma preview)."""
        return self._sweep(limit=n, include_vectors=include_vectors)

    def modify(self, *, name: str | None = None, metadata: Any = None) -> None:
        """Chroma collection rename/retag -- not supported by ELIPS."""
        raise NotImplementedError(
            "modify (renaming/retagging a collection) is not supported"
        )

    @staticmethod
    def _vec_batch(q: Any) -> tuple[list[list[float]] | None, bool]:
        if q is None:
            return None, True
        seq = list(q)
        if seq and isinstance(seq[0], (int, float)):
            return [[float(x) for x in seq]], True
        return [[float(x) for x in v] for v in seq], False

    @staticmethod
    def _txt_batch(q: Any) -> tuple[list[str] | None, bool]:
        if q is None:
            return None, True
        if isinstance(q, str):
            return [q], True
        return list(q), False

    def query(
        self,
        query_embeddings: Any = None,
        query_texts: Any = None,
        *,
        n_results: int = 10,
        k: int | None = None,
        limit: int | None = None,
        where: Filter | None = None,
        where_document: Any = None,
        include: Sequence[str] | None = None,
        include_vectors: bool | None = None,
        max_distance: float | None = None,
        lexical_weight: float = 0.25,
    ) -> list[Hit] | list[list[Hit]]:
        r"""Vector / text / hybrid search (Chroma ``query``).

        ``query_embeddings`` and/or ``query_texts`` select the mode; pass both
        for hybrid. A single query returns ``list[Hit]``; a batch of queries
        returns ``list[list[Hit]]``. ``n_results`` (aliases ``k``/``limit``)
        caps hits per query.
        """
        if where_document is not None:
            raise NotImplementedError(
                "where_document (full-text filtering) is not implemented yet"
            )
        top = k if k is not None else limit if limit is not None else n_results
        want = self._wants(include, include_vectors, default=False)

        vecs, v_single = self._vec_batch(query_embeddings)
        txts, t_single = self._txt_batch(query_texts)
        count = max(len(vecs) if vecs else 0, len(txts) if txts else 0)
        if count == 0:
            raise ValueError("query requires query_embeddings or query_texts")
        single = (query_embeddings is None or v_single) and (
            query_texts is None or t_single
        )

        results: list[list[Hit]] = []
        for i in range(count):
            v = vecs[i] if vecs else None
            t = txts[i] if txts else None
            if v is not None and t is not None:
                hits = self._probe_hybrid(
                    v, t, top=top, where=where, max_distance=max_distance,
                    lexical_weight=lexical_weight, include_vectors=want,
                )
            elif v is not None:
                hits = self._probe(
                    v, top=top, where=where, max_distance=max_distance,
                    include_vectors=want,
                )
            else:
                hits = self._probe_text(
                    t, top=top, where=where, max_distance=max_distance,
                    include_vectors=want, lexical_weight=lexical_weight,
                )
            results.append(hits)
        return results[0] if single else results

    # ---- ELIPS extensions (kept; no Chroma equivalent) ----

    def explain(
        self,
        vector: Vector,
        *,
        top: int = 10,
        where: Filter | None = None,
        max_distance: float | None = None,
        has_text_component: bool = False,
    ) -> QueryPlan:
        """Return the planner decision for a query shape."""
        return self._vault.explain_seek(
            vector,
            top=top,
            where=where if where is not None else Filter(),
            threshold=max_distance,
            has_text_component=has_text_component,
        )

    @property
    def pending_removals(self) -> int:
        """Records deleted from the index but not yet reclaimed."""
        return self._vault.pending_removals

    @property
    def read_only(self) -> bool:
        """Whether this collection currently refuses writes."""
        return self._vault.read_only

    @property
    def sealed(self) -> bool:
        """Whether the owning engine has been closed."""
        return self._vault.sealed

    def freeze(self, frozen: bool = True) -> None:
        """Make this collection reject writes, or allow them again."""
        self._vault.set_read_only(frozen)

    def vacuum(self) -> None:
        """Reclaim index space held by deleted records in this collection."""
        self._vault.vacuum()

    def rebuild(self) -> None:
        """Rebuild the index from the authoritative record store."""
        self._vault.rebuild_index()

    def compress(self) -> None:
        """Train a codebook over this collection and compress it in place."""
        self._vault.quantize()

    #: FAISS spells codebook training ``train``.
    train = compress

    def health(self) -> CollectionHealth:
        """Return a point-in-time health snapshot of this collection."""
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

    # ---- deprecated pre-Phase-5 verbs (shims; emit DeprecationWarning) ----

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
        """.. deprecated:: use :meth:`add`."""
        _deprecated("write", "add")
        return self._write(
            record,
            vector=vector,
            text=text,
            meta=meta,
            key=key,
            document=document,
            chunk=chunk,
            lineage=lineage,
        )

    def write_many(self, records: Sequence[RecordInputLike]) -> list[str]:
        """.. deprecated:: use :meth:`add`."""
        _deprecated("write_many", "add")
        return self._write_many(records)

    def add_many(self, records: Sequence[RecordInputLike]) -> list[str]:
        """.. deprecated:: use :meth:`add`."""
        _deprecated("add_many", "add")
        return self._write_many(records)

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
        """.. deprecated:: use :meth:`add`."""
        _deprecated("ingest", "add")
        return self._ingest(
            records,
            vectors=vectors,
            texts=texts,
            meta=meta,
            keys=keys,
            documents=documents,
            chunks=chunks,
            lineages=lineages,
        )

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
        """.. deprecated:: use :meth:`upsert`."""
        _deprecated("merge", "upsert")
        if records is not None:
            return self._ingest(records)
        return self._ingest(
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
        """.. deprecated:: use :meth:`query` (``query_embeddings=``)."""
        _deprecated("probe", "query")
        return self._probe(
            vector, top=top, where=where, max_distance=max_distance,
            include_vectors=include_vectors,
        )

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
        """.. deprecated:: use :meth:`query` (``query_texts=``)."""
        _deprecated("probe_text", "query")
        return self._probe_text(
            text, top=top, where=where, max_distance=max_distance,
            include_vectors=include_vectors, lexical_weight=lexical_weight,
        )

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
        """.. deprecated:: use :meth:`query` (hybrid: embeddings + texts)."""
        _deprecated("probe_hybrid", "query")
        return self._probe_hybrid(
            vector, text, top=top, where=where, max_distance=max_distance,
            lexical_weight=lexical_weight, include_vectors=include_vectors,
        )

    def search(self, text: str, **kwargs: Any) -> list[Hit]:
        """.. deprecated:: use :meth:`query`."""
        _deprecated("search", "query")
        return self._probe_text(text, **kwargs)

    def search_vector(self, vector: Vector, **kwargs: Any) -> list[Hit]:
        """.. deprecated:: use :meth:`query`."""
        _deprecated("search_vector", "query")
        return self._probe(vector, **kwargs)

    def search_hybrid(self, vector: Vector, text: str, **kwargs: Any) -> list[Hit]:
        """.. deprecated:: use :meth:`query`."""
        _deprecated("search_hybrid", "query")
        return self._probe_hybrid(vector, text, **kwargs)

    def pull(self, keys: Sequence[str], *, include_vectors: bool = True) -> list[Row]:
        """.. deprecated:: use :meth:`get`."""
        _deprecated("pull", "get")
        return self._pull(keys, include_vectors=include_vectors)

    def sweep(
        self,
        *,
        where: Filter | None = None,
        offset: int = 0,
        limit: int | None = None,
        include_vectors: bool = False,
    ) -> list[Row]:
        """.. deprecated:: use :meth:`get`."""
        _deprecated("sweep", "get")
        return self._sweep(
            where=where, offset=offset, limit=limit, include_vectors=include_vectors
        )

    def discard(
        self,
        keys: Sequence[str] | None = None,
        *,
        where: Filter | None = None,
    ) -> int:
        """.. deprecated:: use :meth:`delete`."""
        _deprecated("discard", "delete")
        return self._discard(keys, where=where)


#: The collection type was called ``Arena`` before 1.1.
Arena = Collection

__all__ = ["Arena", "Collection"]
