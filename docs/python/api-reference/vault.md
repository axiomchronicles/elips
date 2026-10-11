# Collection API Reference

A `Collection` is the Chroma-style data plane handed out by a client. All
methods take keyword arguments. A collection wraps one native `Vault`
(reachable as `collection.native`).

Get one from a client:

```python
import elips

client = elips.Client(dimension=4)
docs = client.get_or_create_collection("documents")
```

## Writing

### `add(...)` / `upsert(...)`

Column form (`ids` / `embeddings` / `documents` / `metadatas`, aligned by
position), returning the assigned id(s). `add` has upsert semantics, so
re-adding an existing id overwrites it; `upsert` is an alias.

```python
docs.add(
    ids=["a", "b"],
    embeddings=[[1.0, 0, 0, 0], [0, 1.0, 0, 0]],
    metadatas=[{"kind": "design"}, {"kind": "ops"}],
)
# With the default text embedder you can add raw documents instead:
docs.add(ids=["c"], documents=["gamma appendix note"], metadatas=[{"kind": "appendix"}])
print(docs.count())
```

- `ids` are arbitrary strings; a non-native id is hashed to a stable internal
  id and the original is preserved across reopen.
- supply `embeddings` to skip embedding; supply `documents` to embed via the
  database's text embedder.

### `update(...)`

Chroma `update` overwrites existing records and raises `NotFoundError` for a
missing id. In this build it is limited: it re-adds the record (so an embedding
or document must be supplied) and validates ids through the native store, so it
is unreliable for metadata-only edits and arbitrary string ids. Prefer
`upsert` to change a record by id.

## Querying

### `query(query_embeddings=None, query_texts=None, *, n_results=10, ...)`

Vector / text / hybrid search. Pass `query_embeddings`, `query_texts`, or both
(hybrid). A single query returns `list[Hit]`; a batch returns `list[list[Hit]]`.

```python
hits = docs.query(query_embeddings=[1.0, 0, 0, 0], n_results=2)
print(hits[0].key, round(hits[0].distance, 3))

# batch of two query vectors -> list[list[Hit]]
batch = docs.query(query_embeddings=[[1.0, 0, 0, 0], [0, 1.0, 0, 0]], n_results=1)
print(len(batch), len(batch[0]))
```

| Argument | Meaning |
|---|---|
| `n_results` | Hits per query (aliases: `k`, `limit`). Default `10`. |
| `where` | Metadata `Filter` (see [Filter](filter.md)). |
| `include` | Subset of `"embeddings"` / `"metadatas"` / `"documents"`. |
| `max_distance` | Drop hits beyond this distance (range search). |
| `lexical_weight` | Lexical blend for hybrid queries. Default `0.25`. |
| `where_document` | Full-text filter — **not implemented** (`NotImplementedError`). |

Each hit is a `Hit` with `.key`, `.distance`, `.meta`, `.text`, and (when
requested) `.vector`.

## Reading and deleting

### `get(ids=None, *, where=None, limit=None, offset=0, include=None)`

Fetch by id, or scan by `where` / `limit` / `offset`. Ranking-free. Returns
`list[Row]`.

```python
rows = docs.get(["a"])
page = docs.get(where=elips.Filter().field("kind").equals("ops"), limit=10)
print(rows[0].key, len(page))
```

### `delete(ids=None, *, where=None)`

Delete by id, filter, or both; returns the number removed.

```python
removed = docs.delete(["b"])
print(removed)
```

### `peek(n=10)`

Return up to `n` records (Chroma preview).

```python
print(len(docs.peek(2)))
```

### `count()`

```python
print(docs.count())
```

### `modify(...)`

Collection rename / retag — **not supported** (`NotImplementedError`).

`where_document=` on `get` / `delete` also raises `NotImplementedError`.

## ELIPS extensions

Beyond the Chroma surface, a collection exposes:

| Member | Purpose |
|---|---|
| `explain(vector, *, top=10, where=None, ...)` | Return the planner `QueryPlan` for a query shape. |
| `health()` | `CollectionHealth` snapshot (live count, pending removals, dimension, metric, codec). |
| `vacuum()` | Reclaim index space held by deleted records in this collection. |
| `rebuild()` | Rebuild the index from the record store. |
| `compress()` / `train()` | Train a codebook and compress the collection in place. |
| `freeze(frozen=True)` | Reject (or re-allow) writes. |
| `pending_removals` / `read_only` / `sealed` | State properties. |

```python
plan = docs.explain([1.0, 0, 0, 0], top=1)
print(plan.strategy.name, plan.metadata_accelerated)
client.close()
```

## Deprecated verbs

The pre-Phase-5 methods remain as shims that emit `DeprecationWarning`:
`write`/`write_many`/`add_many` → `add`; `ingest`/`merge` → `add`/`upsert`;
`probe`/`probe_text`/`probe_hybrid`/`search*` → `query`; `pull`/`sweep` →
`get`; `discard` → `delete`. The `Arena` name is a deprecated alias for
`Collection`. See `MIGRATION.md`.

## Native Vault (low-level)

`collection.native` is the native `Vault` — the explicit escape hatch, also
reachable as `elips.native.Vault`. It speaks the low-level vocabulary and
returns native dict rows and `Result` objects, not `Row` / `Hit`.

| Native method | Note |
|---|---|
| `place(vector, data, *, id=None, document=None, chunk=None, lineage=None)` | Insert one record; returns the id. |
| `place_document(text, data, ...)` | Embed text via the configured embedder and store it. |
| `place_many(records)` | Column/record batch insert. |
| `seek(vector, *, top=10, where=Filter(), threshold=None)` | Vector search → `list[Result]`. |
| `seek_text(text, ...)` / `seek_hybrid(vector, text, ..., lexical_weight=0.25)` | Text / hybrid search. |
| `explain_seek(vector, ..., has_text_component=False)` | `QueryPlan`. |
| `fetch(id)` | Record dict (`id`, `vector`, `data`, `document`, `chunk`, `lineage`) or `None`. |
| `scan(where=Filter(), offset=0, limit=-1)` | List of record dicts. |
| `erase(id)` | Delete by id; returns whether a record existed. |
| `info()` | `VaultInfo(count, dimension, metric)`. |
| `count()` | Record count. |

Use the native layer only when the high-level `Collection` does not cover what
you need; the Chroma surface above is the supported API.
