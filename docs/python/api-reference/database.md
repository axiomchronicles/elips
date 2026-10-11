# Database / Client API Reference

The high-level entry points are `elips.Client` (in-memory) and
`elips.PersistentClient` (on disk). Both subclass `Engine`, so they carry the
full client surface — collection management, EQL, persistence, and the
effective configuration. The native `Database` handle sits one level down,
reachable as `client.native`.

> **Maturity:** CPU engine and Collection API are **Stable**. GPU via Metal is
> **Beta** (Apple M4); CUDA/HIP are **Experimental**, not hardware-validated.

## Clients

### `elips.Client(**settings)`

An ephemeral, in-memory database. ELIPS is dimension-first, so a `dimension`
(or dimension-fixing `embedder`) is required.

```python
import elips

client = elips.Client(dimension=4, metric="cosine")
print(client.config.dimension_val)
client.close()
```

### `elips.PersistentClient(path, **settings)`

An on-disk database; creates `path` if needed.

```python
import elips, tempfile

path = tempfile.mkdtemp()
db = elips.PersistentClient(path, dimension=4)
db.get_or_create_collection("documents").add(ids=["a"], embeddings=[[1.0, 0, 0, 0]])
db.close()
```

Shared settings (both clients): `dimension`, `metric` (`"cosine"` |
`"euclidean"` | `"dot_product"`), `index` (`"graph"` | `"exact"`),
`access_mode` (`"read_write"` | `"read_only"`), `segmented_storage`,
`metadata_acceleration`, `embedder`, `use_default_text_embedder`,
`quantization`, and GPU (`gpu=` / `device=` / `fallback=`).

> **Deprecated openers.** `elips.connect(path, **settings)`,
> `elips.connect_with_config(path, config)`, `elips.open(...)`,
> `elips.open_with_config(...)`, and the `elips.Engine` class remain as working
> shims (`connect`/`connect_with_config` emit `DeprecationWarning`). New code
> uses `Client` / `PersistentClient`.

## Collection management

```python
client = elips.Client(dimension=4)

docs = client.get_or_create_collection("documents")   # idempotent create
# client.create_collection("documents")               # raises: already exists
# client.get_collection("missing")                    # CollectionNotFoundError
print(client.list_collections())                       # -> ['documents']
client.close()
```

| Method | Returns / behavior |
|---|---|
| `get_or_create_collection(name, *, embedder=None)` | The collection, created on first use. |
| `create_collection(name, ...)` | New collection; `ValueError` if it exists. |
| `get_collection(name, *, embedder=None)` | Existing collection; `CollectionNotFoundError` otherwise. |
| `list_collections()` / `collection_names()` | `list[str]` of existing names. |
| `delete_collection(name)` | `NotImplementedError` (no vault drop yet). |
| `reset()` | `NotImplementedError`. |

`metadata=` / `configuration=` are accepted on the create methods for Chroma
compatibility but ignored.

## Persistence and lifecycle

| Method | Effect |
|---|---|
| `checkpoint()` | Flush state to durable storage and truncate the WAL. |
| `compact()` | Rebuild each collection's index and checkpoint. |
| `vacuum()` | Reclaim index space held by deleted records across all collections (works in-memory; no snapshot rewrite). |
| `pending_writes()` | Read the WAL without mutating (`list[WalRecord]`; empty for in-memory). |
| `close()` | Checkpoint, release the lock, seal collections. Idempotent. |

`Client` / `PersistentClient` are context managers:

```python
with elips.Client(dimension=2) as client:
    client.get_or_create_collection("documents").add(ids=["a"], embeddings=[[1.0, 0.0]])
    print(client.collection_names())
```

## EQL

EQL runs through `query_eql` on the client (which delegates to the native
`Database.query`). It returns native `Result` rows (`.id`, `.distance`,
`.data`).

```python
client = elips.Client(dimension=2)
client.get_or_create_collection("documents").add(
    ids=["d1"], embeddings=[[1.0, 0.0]], metadatas=[{"kind": "design"}]
)
rows = client.query_eql(
    'search in documents nearest $q k 5 where kind = "design" yield',
    {"q": [1.0, 0.0]},
)
print(len(list(rows)))
client.close()
```

Note: EQL is not on the high-level handle as `.query(...)` (that name is the
Chroma vector search on `Collection`). Use `client.query_eql(...)` or
`client.native.query(...)`.

## Configuration

```python
client = elips.Client(dimension=4, metric="cosine")
cfg = client.config
print(cfg.dimension_val, cfg.metric_val, cfg.index_val)
client.close()
```

`config` returns the effective `Config` (see
[Config & Enums](config-and-enums.md)).

## Native handle

Everything the client does goes through `client.native`, a native `Database`.
Reach for it when you need something the facade does not surface:

```python
client = elips.Client(dimension=2)
with client.native.begin_transaction() as txn:
    txn.vault("documents").place([1.0, 0.0], {"kind": "design"})
print(client.native.list_vaults())
client.close()
```

The native `Database` also exposes `gpu_info()` / `gpu_stats()` (GPU builds
only); these are **not** on the high-level client.

## Common errors

- `ValueError`: `Client()` opened without a dimension or dimension-fixing embedder.
- `ConfigError`: invalid configuration, or read-only open of a missing database.
- `LockConflict(Error)`: another writer holds the database.
- `StorageError`: IO failure, or a write to a read-only database.
- `BackendUnavailableError`: a named GPU backend is unavailable and `fallback=False`.
- `ParseError`: malformed EQL.
