# Database API Reference

`Database` is the top-level Python handle for an ELIPS database. It owns vaults,
coordinates persistence, exposes EQL, and surfaces the effective configuration.

## Factory Functions

### `elips.open()`

```python
def open(
    path: str,
    dimension: int = 0,
    metric: str = "cosine",
    index: str = "graph",
    access_mode: str = "read_write",
    embedder = None,
    use_default_text_embedder: bool = True,
) -> Database
```

Use `open()` for simple cases.

- `path`: filesystem path or `":memory:"`
- `dimension`: required for new databases and all in-memory opens
- `metric`: `"cosine"`, `"euclidean"`, `"dot_product"`
- `index`: `"graph"` or `"exact"`
- `access_mode`: `"read_write"` or `"read_only"`
- `embedder`: optional Python callable embedder or `LocalEmbedderConfig`
- `use_default_text_embedder`: attach the built-in local embedder automatically
  for new databases when no explicit embedder is supplied

Read-only opens require an existing database.

### `elips.open_with_config()`

```python
def open_with_config(path: str, config: Config) -> Database
```

Use this when you need:

- `segmented_storage()`
- `metadata_acceleration()`
- `auto_text_embedder()`
- `local_text_embedder()`
- `text_embedder()`
- GPU configuration

## `Database`

### `vault(name)`

```python
docs = db.vault("documents")
```

Returns a `Vault`, creating it lazily if needed.

### `list_vaults()`

```python
names = db.list_vaults()
```

### `begin_transaction()`

```python
with db.begin_transaction() as txn:
    txn.vault("documents").place([1.0, 0.0], {"kind": "design"})
```

Transactions buffer vector `place()` and `erase()` calls and commit atomically.

### `query(eql, bindings={})`

```python
rows = db.query(
    "seek in documents nearest $q top 5 where kind = \"design\" yield",
    bindings={"q": [1.0, 0.0]},
)
```

EQL supports declarative vector operations. Text-first retrieval is exposed on
`Vault` rather than through EQL.

### `checkpoint()`

Writes the current state to manifest/segments or snapshot and truncates the WAL.

### `compact()`

Rebuilds each vault index from stored records and checkpoints the compacted
state.

### `close()`

Gracefully checkpoints and releases locks. Safe to call more than once.

### `abandon()`

Testing hook that skips checkpointing so the next open must recover from the
WAL.

### `config`

```python
effective = db.config
```

Returns the effective `Config`, including persisted dimension/metric/index and
runtime options such as access mode, metadata acceleration, and resolved text
embedder metadata.

### `gpu_info()` and `gpu_stats()`

Available only when the Python extension is built with GPU support.

`gpu_info()` reports the active accelerator device when a GPU backend is
selected, and otherwise returns non-empty CPU fallback metadata.

## Context Manager

```python
with elips.open("/tmp/elips-db", dimension=2) as db:
    db.vault("documents").place([1.0, 0.0])
```

This automatically calls `close()` on exit.

## Common Errors

- `ConfigError`: invalid configuration or read-only open on a missing database
- `LockConflict`: another writer already holds the database
- `StorageError`: IO failure or attempted mutation in read-only mode
- `ParseError`: malformed EQL
