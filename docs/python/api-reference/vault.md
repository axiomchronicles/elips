# Vault API Reference

A `Vault` is a named partition inside a database. It owns the record store,
search planner, metadata accelerator, and backing index.

## Core Methods

### `place()`

```python
rid = vault.place(
    [1.0, 0.0],
    {"kind": "design"},
    document=elips.DocumentAttachment(text="alpha design note"),
    chunk=chunk,
    lineage=lineage,
)
```

Parameters:

- `vector`: embedding vector
- `data`: metadata payload
- `id`: optional explicit UUIDv7 string
- `document`: optional `DocumentAttachment`
- `chunk`: optional `ChunkInfo`
- `lineage`: optional `EmbeddingLineage`

Returns the assigned record id.

### `place_document()`

```python
rid = vault.place_document("alpha design note", {"kind": "design"})
```

Requires a configured text embedder in `Config`. ELIPS embeds the text, stores
the source document, fills chunk defaults when needed, and auto-generates
lineage from the configured embedder when not provided.

### `place_many()`

```python
vault.place_many([
    {"vector": [1.0, 0.0], "data": {"kind": "vector-only"}},
    {"text": "alpha design note", "data": {"kind": "text-first"}},
    {
        "vector": [0.0, 1.0],
        "document": elips.DocumentAttachment(text="beta note"),
        "chunk": chunk,
        "lineage": lineage,
    },
])
```

Each batch record may include:

- `vector`
- `text`
- `data`
- `id`
- `document`
- `chunk`
- `lineage`

Text-only rows require a native text embedder.

## Query Methods

### `seek(vector, top=10, where=Filter(), threshold=None)`

Vector similarity search.

### `seek_text(text, top=10, where=Filter(), threshold=None)`

Text-first query surface. ELIPS embeds the text through the configured text
embedder and routes the resulting vector through the planner. If no text
embedder is configured, `seek_text()` raises `ConfigError` with guidance to
configure a local embedder or use vector-first APIs.

### `seek_hybrid(vector, text, top=10, where=Filter(), threshold=None, lexical_weight=0.25)`

Blends vector distance with lexical overlap from attached documents.

### `explain_seek(vector, top=10, where=Filter(), threshold=None, has_text_component=False)`

Returns a `QueryPlan` with:

- `strategy`
- `candidate_count`
- `metadata_accelerated`
- `gpu_index`
- `index_type`

## Retrieval And Maintenance

### `fetch(id)`

Returns `None` or a dict containing:

- `id`
- `vector`
- `data`
- `document`
- `chunk`
- `lineage`

### `scan(where=Filter(), offset=0, limit=-1)`

Returns a list of the same record dict shape as `fetch()`.

### `erase(id)`

Deletes a record by id. Returns `True` if a record existed.

### `info()`

Returns `VaultInfo(count, dimension, metric)`.

### `count()`

Convenience wrapper for the number of records in the vault.

### `rebuild_index()`

Drops and rebuilds the backing index from the authoritative record store. This
is the vault-scoped maintenance primitive used by `Database.compact()`.

## Result Shape

`seek*()` methods return `Result` objects with:

- `id`
- `distance`
- `data`
- `document`
- `chunk`
- `lineage`

This means document context and embedding provenance are available directly on
query hits, not only through `fetch()`.

## Common Errors

- `DimensionMismatch`
- `InvalidVector`
- `ConfigError` when `place_document()` or `seek_text()` is used without a text embedder
- `StorageError` when mutating a read-only vault
