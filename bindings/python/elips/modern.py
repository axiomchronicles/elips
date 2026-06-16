from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, Sequence

from ._core import (
    ChunkInfo,
    Config,
    Database,
    DocumentAttachment,
    EmbeddingLineage,
    Filter,
    LocalEmbedderConfig,
    QueryPlan,
    Vault,
    open_with_config as _open_with_config,
)
from .types import MetaValue, PayloadLike, Vector as VectorLike

if TYPE_CHECKING:
    from ._core import GpuConfig

TEXT_SLOT = "__elips_text__"


class Embedder(Protocol):
    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


@dataclass(frozen=True, slots=True)
class Row:
    key: str
    meta: dict[str, MetaValue]
    text: Optional[str] = None
    vector: Optional[tuple[float, ...]] = None
    chunk: Optional[ChunkInfo] = None
    lineage: Optional[EmbeddingLineage] = None


@dataclass(frozen=True, slots=True)
class Hit:
    key: str
    distance: float
    meta: dict[str, MetaValue]
    text: Optional[str] = None
    vector: Optional[tuple[float, ...]] = None
    chunk: Optional[ChunkInfo] = None
    lineage: Optional[EmbeddingLineage] = None


class Engine:
    def __init__(
        self,
        db: Database,
        *,
        default_embedder: Optional[Embedder] = None,
    ) -> None:
        self._db = db
        self._default_embedder = default_embedder

    @property
    def raw(self) -> Database:
        return self._db

    def arena(
        self,
        name: str,
        *,
        embedder: Optional[Embedder] = None,
        text_slot: str = TEXT_SLOT,
    ) -> "Arena":
        return Arena(
            self._db,
            name,
            embedder=embedder if embedder is not None else self._default_embedder,
            text_slot=text_slot,
        )

    def checkpoint(self) -> None:
        self._db.checkpoint()

    def compact(self) -> None:
        self._db.compact()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class Arena:
    def __init__(
        self,
        db: Database,
        name: str,
        *,
        embedder: Optional[Embedder] = None,
        text_slot: str = TEXT_SLOT,
    ) -> None:
        self._db = db
        self._vault = db.vault(name)
        self._embedder = embedder
        self._text_slot = text_slot

    @property
    def name(self) -> str:
        return self._vault.name

    @property
    def raw(self) -> Vault:
        return self._vault

    def count(self) -> int:
        return self._vault.count()

    def write(
        self,
        *,
        vector: Optional[VectorLike] = None,
        text: Optional[str] = None,
        meta: Optional[PayloadLike] = None,
        key: Optional[str] = None,
        chunk: Optional[ChunkInfo] = None,
        lineage: Optional[EmbeddingLineage] = None,
    ) -> str:
        keys = self.ingest(
            vectors=[vector] if vector is not None else None,
            texts=[text] if text is not None else None,
            meta=[meta] if meta is not None else None,
            keys=[key] if key is not None else None,
            chunks=[chunk] if chunk is not None else None,
            lineages=[lineage] if lineage is not None else None,
        )
        return keys[0]

    def merge(
        self,
        *,
        vectors: Optional[Sequence[VectorLike]] = None,
        texts: Optional[Sequence[str]] = None,
        meta: Optional[Sequence[Optional[PayloadLike]]] = None,
        keys: Optional[Sequence[Optional[str]]] = None,
        chunks: Optional[Sequence[Optional[ChunkInfo]]] = None,
        lineages: Optional[Sequence[Optional[EmbeddingLineage]]] = None,
    ) -> list[str]:
        return self.ingest(
            vectors=vectors,
            texts=texts,
            meta=meta,
            keys=keys,
            chunks=chunks,
            lineages=lineages,
        )

    def ingest(
        self,
        *,
        vectors: Optional[Sequence[VectorLike]] = None,
        texts: Optional[Sequence[str]] = None,
        meta: Optional[Sequence[Optional[PayloadLike]]] = None,
        keys: Optional[Sequence[Optional[str]]] = None,
        chunks: Optional[Sequence[Optional[ChunkInfo]]] = None,
        lineages: Optional[Sequence[Optional[EmbeddingLineage]]] = None,
    ) -> list[str]:
        count = self._batch_size(
            vectors=vectors,
            texts=texts,
            meta=meta,
            keys=keys,
            chunks=chunks,
            lineages=lineages,
        )
        prepared_texts = self._normalize_optional_strings(texts, count)
        prepared_vectors = self._resolve_vectors(
            vectors=vectors,
            texts=prepared_texts,
            count=count,
        )
        prepared_meta = self._normalize_payloads(meta, count)
        prepared_keys = self._normalize_optional_strings(keys, count)
        prepared_chunks = self._normalize_sequence(chunks, count)
        prepared_lineages = self._normalize_sequence(lineages, count)

        assigned: list[str] = []
        for index in range(count):
            payload = dict(prepared_meta[index] or {})
            text = prepared_texts[index]
            key = prepared_keys[index]
            chunk = prepared_chunks[index]
            lineage = prepared_lineages[index]
            vector = prepared_vectors[index]

            if text is not None and vector is None:
                assigned.append(
                    self._vault.place_document(
                        text,
                        payload,
                        id=key,
                        chunk=chunk,
                        lineage=lineage,
                    )
                )
                continue

            document = (
                DocumentAttachment(text=text)
                if text is not None
                else None
            )
            assigned.append(
                self._vault.place(
                    vector,
                    payload,
                    id=key,
                    document=document,
                    chunk=chunk,
                    lineage=self._resolve_lineage(text, lineage),
                )
            )
        return assigned

    def probe(
        self,
        vector: VectorLike,
        *,
        top: int = 10,
        where: Optional[Filter] = None,
        max_distance: Optional[float] = None,
        include_vectors: bool = False,
    ) -> list[Hit]:
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
        where: Optional[Filter] = None,
        max_distance: Optional[float] = None,
        include_vectors: bool = False,
        lexical_weight: float = 0.25,
    ) -> list[Hit]:
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
        vector: VectorLike,
        text: str,
        *,
        top: int = 10,
        where: Optional[Filter] = None,
        max_distance: Optional[float] = None,
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
            self._hit_from_result(result, include_vectors=include_vectors)
            for result in results
        ]

    def explain(
        self,
        vector: VectorLike,
        *,
        top: int = 10,
        where: Optional[Filter] = None,
        max_distance: Optional[float] = None,
        has_text_component: bool = False,
    ) -> QueryPlan:
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
        rows: list[Row] = []
        for key in keys:
            fetched = self._vault.fetch(key)
            if fetched is None:
                continue
            rows.append(self._row_from_record(fetched, include_vectors=include_vectors))
        return rows

    def sweep(
        self,
        *,
        where: Optional[Filter] = None,
        offset: int = 0,
        limit: Optional[int] = None,
        include_vectors: bool = False,
    ) -> list[Row]:
        rows = self._vault.scan(
            where=where if where is not None else Filter(),
            offset=offset,
            limit=-1 if limit is None else limit,
        )
        return [
            self._row_from_record(row, include_vectors=include_vectors)
            for row in rows
        ]

    def discard(
        self,
        keys: Optional[Sequence[str]] = None,
        *,
        where: Optional[Filter] = None,
    ) -> int:
        victims: list[str] = []
        seen: set[str] = set()

        if keys is not None:
            for key in keys:
                if key not in seen:
                    seen.add(key)
                    victims.append(key)

        if where is not None:
            for row in self._vault.scan(where=where, offset=0, limit=-1):
                key = row["id"]
                if key not in seen:
                    seen.add(key)
                    victims.append(key)

        if not victims:
            raise ValueError("discard requires at least one explicit key or a filter")

        removed = 0
        for key in victims:
            removed += int(self._vault.erase(key))
        return removed

    def _batch_size(
        self,
        *,
        vectors: Optional[Sequence[VectorLike]],
        texts: Optional[Sequence[str]],
        meta: Optional[Sequence[Optional[PayloadLike]]],
        keys: Optional[Sequence[Optional[str]]],
        chunks: Optional[Sequence[Optional[ChunkInfo]]],
        lineages: Optional[Sequence[Optional[EmbeddingLineage]]],
    ) -> int:
        candidates = [
            len(vectors) if vectors is not None else None,
            len(texts) if texts is not None else None,
            len(meta) if meta is not None else None,
            len(keys) if keys is not None else None,
            len(chunks) if chunks is not None else None,
            len(lineages) if lineages is not None else None,
        ]
        count = next((value for value in candidates if value is not None), 0)
        if count == 0:
            raise ValueError("ingest requires at least one vector batch or text batch")
        for value in candidates:
            if value is not None and value != count:
                raise ValueError("all ingest batch fields must have the same length")
        return count

    def _resolve_vectors(
        self,
        *,
        vectors: Optional[Sequence[VectorLike]],
        texts: list[Optional[str]],
        count: int,
    ) -> list[Optional[VectorLike]]:
        if vectors is not None:
            return list(vectors)

        available_texts = [text for text in texts if text is not None]
        if len(available_texts) != count:
            raise ValueError("text batch contains null values")

        if self._has_native_text():
            return [None] * count

        embedder = self._require_embedder()
        embedded = embedder(available_texts)
        if len(embedded) != count:
            raise ValueError("embedder returned a batch with the wrong length")
        return list(embedded)

    def _normalize_payloads(
        self,
        meta: Optional[Sequence[Optional[PayloadLike]]],
        count: int,
    ) -> list[Optional[PayloadLike]]:
        if meta is None:
            return [None] * count
        return list(meta)

    def _normalize_optional_strings(
        self,
        values: Optional[Sequence[Optional[str]]],
        count: int,
    ) -> list[Optional[str]]:
        if values is None:
            return [None] * count
        return list(values)

    def _normalize_sequence(self, values, count: int):
        if values is None:
            return [None] * count
        return list(values)

    def _row_from_record(
        self,
        record: dict,
        *,
        include_vectors: bool,
    ) -> Row:
        document = record.get("document")
        return Row(
            key=record["id"],
            meta=dict(record["data"]),
            text=document.text if document is not None else None,
            vector=tuple(record["vector"]) if include_vectors else None,
            chunk=record.get("chunk"),
            lineage=record.get("lineage"),
        )

    def _hit_from_result(self, result, *, include_vectors: bool) -> Hit:
        fetched = self._vault.fetch(result.id) if include_vectors else None
        document = result.document if result.document is not None else (
            fetched.get("document") if fetched is not None else None
        )
        return Hit(
            key=result.id,
            distance=result.distance,
            meta=dict(result.data),
            text=document.text if document is not None else None,
            vector=tuple(fetched["vector"]) if fetched is not None else None,
            chunk=result.chunk if result.chunk is not None else (
                fetched.get("chunk") if fetched is not None else None
            ),
            lineage=result.lineage if result.lineage is not None else (
                fetched.get("lineage") if fetched is not None else None
            ),
        )

    def _resolve_lineage(
        self,
        text: Optional[str],
        lineage: Optional[EmbeddingLineage],
    ) -> Optional[EmbeddingLineage]:
        if lineage is not None or text is None or self._embedder is None:
            return lineage
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


def connect(
    path: str,
    *,
    dimension: int = 0,
    metric: str = "cosine",
    index: str = "graph",
    access_mode: str = "read_write",
    segmented_storage: bool = True,
    metadata_acceleration: bool = True,
    embedder: Optional[Embedder | LocalEmbedderConfig] = None,
    embedder_provider: str = "python",
    embedder_model: str = "callable",
    embedder_revision: str = "",
    use_default_text_embedder: bool = True,
    gpu: Optional["GpuConfig"] = None,
) -> Engine:
    config = (
        Config()
        .dimension(dimension)
        .metric(metric)
        .index(index)
        .access_mode(access_mode)
        .segmented_storage(segmented_storage)
        .metadata_acceleration(metadata_acceleration)
        .auto_text_embedder(use_default_text_embedder)
    )
    if gpu is not None:
        if not hasattr(config, "gpu"):
            raise ValueError("gpu config requires ELIPS to be built with GPU bindings")
        config.gpu(gpu)
    runtime_embedder: Optional[Embedder] = None
    if isinstance(embedder, LocalEmbedderConfig):
        config.local_text_embedder(embedder)
    elif embedder is not None:
        config.text_embedder(
            embedder,
            provider=embedder_provider,
            model=embedder_model,
            revision=embedder_revision,
            dimension=dimension,
        )
        runtime_embedder = embedder
    return Engine(_open_with_config(path, config), default_embedder=runtime_embedder)


def connect_with_config(
    path: str,
    config: Config,
    *,
    embedder: Optional[Embedder | LocalEmbedderConfig] = None,
    embedder_provider: str = "python",
    embedder_model: str = "callable",
    embedder_revision: str = "",
) -> Engine:
    if embedder is not None and not config.has_text_embedder:
        if isinstance(embedder, LocalEmbedderConfig):
            config.local_text_embedder(embedder)
        else:
            config.text_embedder(
                embedder,
                provider=embedder_provider,
                model=embedder_model,
                revision=embedder_revision,
            )
    return Engine(
        _open_with_config(path, config),
        default_embedder=embedder if callable(embedder) else None,
    )
