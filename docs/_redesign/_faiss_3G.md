# Phase 3G — Vector-DB Collection API Conventions (DOC/WEB research)

Goal: make ELIPS's Python Collection API instantly recognizable to users of
modern vector DBs. Chroma is the primary model to mirror; Qdrant, Milvus, and
LanceDB are surveyed for contrast to see where Chroma's names are (or aren't)
the de-facto standard.

Every signature below was verified against live 2025/2026 docs (fetched
2026-10-10). Recent-change flags are called out inline. All source URLs are
listed at the end.

> Note on method shapes: Chroma's Python client uses keyword arguments
> everywhere (`name=`, `ids=`, `where=`, ...). The Rust client exposes
> positional signatures — quoted below only where they pin down exact arg
> order/types. ELIPS mirrors the **Python keyword** surface.

---

## 1. CHROMA (primary model)

### 1.1 Clients

| Constructor | Behavior | Key args |
|---|---|---|
| `chromadb.Client()` | In-memory, ephemeral. Docs: "The `Client()` method starts a Chroma server in-memory and also returns a client with which you can connect to it." | `settings`, `tenant="default"`, `database="default"` |
| `chromadb.EphemeralClient(...)` | Explicit in-memory variant; data not persisted. | `settings=None, tenant="default", database="default"` |
| `chromadb.PersistentClient(path=...)` | Saves/loads from local disk; auto-persists, reloads on start. Default path `.chroma`. | `path`, `settings=None, tenant="default", database="default"` |
| `chromadb.HttpClient(host, port)` | Connects to a running Chroma server. | `host="localhost", port=8000, ssl=False, headers=None, settings=None, tenant, database` |
| `chromadb.AsyncHttpClient()` | Awaited async variant of HttpClient. | same as HttpClient (awaited) |
| `chromadb.CloudClient(...)` | Chroma Cloud. Reads `CHROMA_TENANT/DATABASE/API_KEY` from env if omitted. Defaults to `api.trychroma.com` (us-east-1). | `tenant=None, database=None, api_key=None, settings=None` |

Convenience methods on clients: `heartbeat()`, `reset()` (destructive/irreversible).

### 1.2 Collection management (on the client)

```python
client.create_collection(name=..., embedding_function=..., metadata=..., configuration=...)
client.get_collection(name=..., embedding_function=...)
client.get_or_create_collection(name=..., metadata=..., configuration=...)
client.delete_collection(name=...)
client.list_collections(limit=..., offset=...)   # default limit 100
```

- Collection `name` must be 3–512 characters.
- `get_or_create_collection` is the idempotent create.

### 1.3 Collection methods (data plane)

```python
# Write
collection.add(ids=, embeddings=, documents=, metadatas=, uris=)
collection.upsert(ids=, embeddings=, documents=, metadatas=, uris=)
collection.update(ids=, embeddings=, documents=, metadatas=)
collection.delete(ids=, where=, where_document=)

# Read / search
collection.query(query_embeddings=, query_texts=, n_results=10,
                 where=, where_document=, ids=, include=)
collection.get(ids=, where=, where_document=, limit=, offset=, include=)

# Introspection / admin
collection.count()
collection.peek()                    # returns the first 10 records
collection.modify(name=, metadata=)  # rename / change metadata in place
```

Rust `add` signature (pins arg order/types):

```rust
pub async fn add(&self, ids: Vec<String>, embeddings: Vec<Vec<f32>>,
    documents: Option<Vec<Option<String>>>, uris: Option<Vec<Option<String>>>,
    metadatas: Option<Vec<Option<Metadata>>>) -> Result<...>
```

Rust `query` signature:

```rust
pub async fn query(&self, query_embeddings: Vec<Vec<f32>>, n_results: Option<u32>,
    where: Option<Where>, ids: Option<Vec<String>>,
    include: Option<IncludeList>) -> Result<QueryResponse, ...>
```

Semantics worth mirroring:
- `add`: must provide `documents`, `embeddings`, or both; `metadatas` optional.
  If only `documents` are given, the collection's embedding function generates
  embeddings. Re-adding an existing `id` is **ignored** (no error) — use
  `update`/`upsert` to overwrite.
- `query` inputs: Python also supports `query_images` and `query_uris`.
- `get` has no similarity ranking (filter-only retrieval).

### 1.4 Argument shapes

- `ids`: `List[str]` — always required for `add/upsert/update`, always returned.
- `embeddings`: `List[List[float]]`.
- `documents`: `List[str]` (raw text; embedded via the collection's EF).
- `metadatas`: `List[Dict[str, str|int|float|bool]]` — one dict per record.
- `n_results`: `int` (default 10) — **query only** (the "top-k" name).
- `where`: metadata filter dict (see 1.5).
- `where_document`: document-content filter dict (see 1.6).
- `include`: `List[str]` selecting what comes back. Values seen in docs:
  `"documents"`, `"metadatas"`, `"embeddings"`, `"distances"`, `"uris"`.
  Defaults — query returns documents+metadatas+distances; get returns
  documents+metadatas. `ids` are **always** returned regardless of `include`.

### 1.5 `where` operators (metadata filtering)

| Operator | Meaning |
|---|---|
| `$eq` | equals (equivalent to using the field directly: `{"page": 10}`) |
| `$ne` | not equals |
| `$gt` / `$gte` | greater than / greater-or-equal |
| `$lt` / `$lte` | less than / less-or-equal |
| `$in` | value in a list (string, int, float, bool) |
| `$nin` | value not in a list |
| `$and` | match **all** filters in the list |
| `$or` | match **any** filter in the list |

```python
# bare equality shorthand
where={"page": 10}
# explicit operator
where={"page": {"$gt": 10}}
# logical combination (range query)
where={"$and": [{"page": {"$gte": 5}}, {"page": {"$lte": 10}}]}
where={"$or": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]}
```

(The metadata-filtering doc explicitly documents `$eq $gt $gte $lte $in $nin
$and $or`; `$ne`/`$lt` are the standard counterparts shown in the API
reference.)

### 1.6 `where_document` operators (full-text / content filtering)

`$contains`, `$not_contains`, `$regex`, `$not_regex` — all four current.
Full-text search is **case-sensitive**.

```python
where_document={"$contains": "search string"}
where_document={"$or": [{"$contains": "s1"}, {"$not_contains": "s2"}]}
where_document={"$regex": "pattern"}
```

(Rust expresses these as enum variants `Contains / NotContains / Regex /
NotRegex` rather than `$`-prefixed keys.)

### 1.7 Distance / space configuration

Set via the collection `configuration` object (current API), **not** the
legacy `metadata={"hnsw:space": ...}` key:

```python
client.create_collection(
    name="c",
    configuration={"hnsw": {"space": "cosine"}},  # or {"spann": {...}}
)
```

| `space` | Metric |
|---|---|
| `l2` | squared L2 norm — **default**; absolute geometric distance |
| `ip` | inner product: `d = 1.0 - Σ(Aᵢ·Bᵢ)` |
| `cosine` | angle only (magnitude ignored) — best for text embeddings |

> RECENT CHANGE: older Chroma used `metadata={"hnsw:space": "cosine"}`. Current
> docs put HNSW/SPANN settings (and the embedding function) under the
> structured `configuration=` object. The space must be one supported by the
> collection's embedding function.

### 1.8 Embedding function interface

Abstract base class `EmbeddingFunction`; the core method:

```python
def __call__(self, input: Documents) -> Embeddings:
```

A custom EF subclasses `EmbeddingFunction` and implements:
- `__call__(self, input: Documents) -> Embeddings` — the embedding logic.
- `name()` — static, returns a string id (e.g. `"my-ef"`).
- `get_config()` -> `Dict[str, Any]` — serialize config (model name, etc.).
- `build_from_config(config)` — static, reconstructs an instance.
- optional `@register_embedding_function` decorator for recognition.

The EF (and its init params) is stored in the collection configuration so
Chroma can reconstruct it across clients. It is invoked automatically during
`add`, `update`, `upsert`, and `query`. Default EF is a sentence-transformer;
pass `embedding_function=None` to supply raw embeddings yourself.

---

## 2. QDRANT (contrast)

Vocabulary: **collection** of **points** (not "records"), `payload` (not
"metadata"), `upsert` (shared with Chroma).

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (VectorParams, Distance, PointStruct,
                                  Filter, FieldCondition, MatchValue, Range)

client = QdrantClient(url="http://localhost:6333")
client.create_collection(collection_name="c",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE))

client.upsert(collection_name="c", points=[
    PointStruct(id=1, vector=[...], payload={"city": "Berlin", "price": 100})])

client.query_points(collection_name="c", query=[...], limit=10,
    query_filter=Filter(
        must=[FieldCondition(key="city", match=MatchValue(value="Berlin")),
              FieldCondition(key="price", range=Range(gte=50, lte=200))],
        should=[...], must_not=[...]))
```

- `Distance` enum: `COSINE`, `DOT`, `EUCLID`, `MANHATTAN` (vs Chroma's
  `l2/ip/cosine` strings).
- `search()` is the older method (takes `query_vector`); `query_points()` is
  current (takes `query`).
- Filter model: structured objects, **not** Mongo-style `$`-operators. `Filter`
  has `must` (AND) / `should` (OR) / `must_not` (NOT), each a list of
  `FieldCondition(key=, match=MatchValue(...) | range=Range(...))`.

Takeaway: Qdrant shares `upsert` but diverges on everything else — points vs
records, payload vs metadata, enum distances, object-based filters, `limit` not
`n_results`.

## 3. MILVUS (contrast)

Vocabulary: **collection** of **entities**; fields; primary key. `MilvusClient`
is the simplified facade.

```python
from pymilvus import MilvusClient
client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

client.create_collection(collection_name="c", dimension=768)
client.insert(collection_name="c", data=[{...}, ...])   # list of dicts
client.upsert(collection_name="c", data=[...])
client.search(collection_name="c", data=[[...]], limit=10,
              filter="price > 50", output_fields=["city"])
client.query(collection_name="c", filter='color like "red%"',
             output_fields=["color"])
client.delete(collection_name="c", ids=[...] )           # or filter=
client.get(collection_name="c", ids=[...])
```

- `insert` does **not** dedup primary keys; use `upsert` to update.
- Filter is a **string boolean-expression DSL**, not a dict:
  `==, !=, >, >=, <, <=`; `in [..]`; `like "red%"`; `and / or / not`;
  chained ranges `5 < id <= 15`.
- `limit` not `n_results`; `output_fields` is Milvus's `include` analogue (but
  selects schema fields, not record components).

Takeaway: Milvus shares `upsert`, `query`, `delete`, `get`, and `limit`-style
top-k, but uses `insert` (not `add`), `data` (not `ids/embeddings/documents`
split), and a string filter DSL.

## 4. LanceDB (contrast)

Vocabulary: **table** (not collection) of rows; builder-style search chain;
SQL-string filters.

```python
import lancedb
db = lancedb.connect(uri)                       # local path, s3://, db://, ...
tbl = db.create_table("t", data=[{...}], mode="overwrite")
tbl = db.open_table("t")
db.table_names()
tbl.add(new_rows)                               # append list of dicts
tbl.delete("id = '3'")                          # SQL predicate

result = (tbl.search(query_vector)
             .where("stats.magic >= 4")         # SQL boolean expr
             .select(["name", "role"])
             .limit(2)
             .to_pandas())                       # or .to_list() / .to_polars()
```

- Filter is a **SQL boolean expression** (per-row): `= != < <= > >=`,
  `AND/OR/NOT`, `IN (...)`, `BETWEEN`, `IS NULL`; dot-notation for nested
  struct fields; single-quoted string literals.
- No `metadata`/`documents`/`ids` split — rows are arbitrary dicts; a
  `_distance` column is returned from search.
- `limit(n)` on the builder, not `n_results`; `select([...])` is the `include`
  analogue (column projection).

Takeaway: LanceDB is the furthest from Chroma — tables not collections, builder
chain not single `query()` call, SQL strings not dict filters.

---

## 5. Where Chroma is / isn't the de-facto standard

| Concept | Chroma | Qdrant | Milvus | LanceDB | Chroma = standard? |
|---|---|---|---|---|---|
| container | `collection` | collection | collection | **table** | mostly yes (3/4) |
| write-new | `add` | `upsert` | `insert` | `add` | split 2/4 |
| idempotent write | `upsert` | `upsert` | `upsert` | (merge/add) | **yes** — universal |
| top-k search | `query` | `query_points` | `search` | `search` | no — all differ |
| top-k arg | `n_results` | `limit` | `limit` | `.limit()` | **no** — `limit` wins 3/4 |
| metadata | `metadatas` | `payload` | fields | columns | Chroma-specific |
| filter form | dict `$ops` | objects | string DSL | SQL string | **no** — no consensus |
| projection | `include=[...]` | with_payload | `output_fields` | `select([...])` | Chroma-specific |
| distance | `l2/ip/cosine` str | enum | metric str | metric str | partial |
| get-by-id | `get` | retrieve | `get`/`query` | search/scan | partial |

Key judgments:
- **`upsert`** is genuinely universal — safe to adopt verbatim.
- **`collection`, `add`, `query`, `metadatas`, `where`, `include`,
  `n_results`** are Chroma-house conventions, not cross-DB standards — but
  Chroma is the most-copied Python-first vector DB, so mirroring it is the
  right "recognizable" target for ELIPS's RAG/LLM-adjacent audience.
- The only place to consider accommodating the broader crowd is the top-k arg:
  `n_results` is Chroma-only; `limit` is the 3/4 majority.

---

## 6. ADOPT / ADAPT / SKIP — ELIPS public Python symbols

Mapping each ELIPS public symbol to the Chroma name it should use.

### ADOPT verbatim (match Chroma exactly)

| ELIPS symbol | Chroma name | Signature to mirror |
|---|---|---|
| ephemeral client | `Client()` | in-memory, zero-arg |
| persistent client | `PersistentClient(path=)` | `path`, default `.chroma` |
| server client | `HttpClient(host=, port=)` | `host="localhost", port=8000` |
| create | `create_collection(name=, embedding_function=, metadata=, configuration=)` | kw-only |
| get | `get_collection(name=, embedding_function=)` | |
| idempotent create | `get_or_create_collection(name=, ...)` | |
| drop | `delete_collection(name=)` | |
| list | `list_collections(limit=, offset=)` | default limit 100 |
| `Collection` class | `Collection` | the object returned above |
| `add` | `add(ids=, embeddings=, documents=, metadatas=)` | re-add existing id = no-op |
| `upsert` | `upsert(ids=, embeddings=, documents=, metadatas=)` | |
| `get` | `get(ids=, where=, where_document=, limit=, offset=, include=)` | filter-only, no ranking |
| `update` | `update(ids=, embeddings=, documents=, metadatas=)` | |
| `delete` | `delete(ids=, where=, where_document=)` | |
| `query` | `query(query_embeddings=, query_texts=, n_results=, where=, where_document=, include=)` | |
| `count` | `count()` | |
| `peek` | `peek()` | first 10 records |
| `where` | metadata dict + `$eq $ne $gt $gte $lt $lte $in $nin $and $or` | |
| `where_document` | `$contains $not_contains $regex $not_regex` | |
| `include` | `include=[...]` | `"documents","metadatas","embeddings","distances","uris"` |
| `metadatas` | `metadatas` | list of dicts, one per record |
| `ids` | `ids` | list[str], always returned |

### ADAPT (Chroma name, ELIPS tweak)

- **`n_results`** — ADOPT the name for Chroma familiarity, but **also accept
  `limit` as an alias** (3/4 of the field uses `limit`; cheap goodwill for
  Qdrant/Milvus/LanceDB migrants). Document `n_results` as canonical.
- **`peek()`** — adopt the name; make the count a parameter
  (`peek(n=10)`) since a hard-coded 10 is the one rough edge in Chroma's
  surface. Default stays 10.
- **distance/space `configuration`** — adopt Chroma's `configuration={"hnsw":
  {"space": ...}}` shape with values `l2/ip/cosine`, but map internally to
  whatever ELIPS's index actually supports; keep `l2` the default to match.
- **embedding function** — adopt the `EmbeddingFunction.__call__(self, input)
  -> Embeddings` protocol and `embedding_function=` kwarg, but ELIPS can skip
  the `register/build_from_config` persistence plumbing unless cross-client EF
  reconstruction is a real requirement (YAGNI until then).

### SKIP (don't copy)

- `CloudClient` / tenant+database multi-tenancy — Chroma-Cloud-specific; no
  ELIPS analogue. Skip until there's a hosted ELIPS.
- `AsyncHttpClient` — only if/when ELIPS grows an async surface; skip for now.
- `uris` / `query_images` / `query_uris` / data-URI multimodal args — skip
  unless ELIPS targets multimodal; they bloat every signature otherwise.
- Qdrant/Milvus/LanceDB vocab (`points`, `payload`, `entities`, `tables`,
  `insert`, string/SQL filter DSLs, `output_fields`, builder-chain search) —
  skip; adopting Chroma means NOT offering these as the primary surface.

---

## Sources

- https://docs.trychroma.com/docs/collections/manage-collections
- https://docs.trychroma.com/docs/collections/add-data
- https://docs.trychroma.com/docs/collections/update-data
- https://docs.trychroma.com/docs/collections/configure
- https://docs.trychroma.com/docs/querying-collections/query-and-get
- https://docs.trychroma.com/docs/querying-collections/metadata-filtering
- https://docs.trychroma.com/docs/querying-collections/full-text-search
- https://docs.trychroma.com/docs/run-chroma/clients
- https://docs.trychroma.com/docs/run-chroma/client-server
- https://docs.trychroma.com/docs/run-chroma/persistent-client
- https://docs.trychroma.com/reference/python/client
- https://docs.trychroma.com/reference/python/collection
- https://docs.trychroma.com/docs/embeddings/embedding-functions
- https://docs.trychroma.com/docs/overview/getting-started
- https://cookbook.chromadb.dev/core/clients/
- https://qdrant.tech/documentation/concepts/points/
- https://milvus.io/docs/insert-update-delete.md
- https://milvus.io/docs/manage-collections.md
- https://docs.lancedb.com/quickstart
- https://lancedb.github.io/lancedb/
