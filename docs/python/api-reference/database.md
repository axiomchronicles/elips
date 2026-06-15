# Database API Reference

The `Database` class is the top-level handle for an ELIPS database. One handle per directory. It owns all vaults and coordinates persistence, locking, and query execution.

## Factory Functions

### `elips.open()`

```python
def open(
    path: str,
    dimension: int = 0,
    metric: str = "cosine",
    index: str = "graph",
) -> Database
```

Open (or create) a database with simple parameters.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | required | Filesystem path or `":memory:"` for ephemeral |
| `dimension` | `int` | `0` | Vector dimension. Required for new databases; ignored for existing ones |
| `metric` | `str` | `"cosine"` | Similarity metric: `"cosine"`, `"euclidean"`, or `"dot_product"` |
| `index` | `str` | `"graph"` | Index backend: `"graph"` (HNSW) or `"exact"` (brute-force) |

**Returns:** A `Database` handle.

**Raises:**
- `ConfigError` — dimension/configuration mismatch with an existing database
- `LockConflict` — another process holds a write lock on the directory

**Examples:**

```python
db = elips.open(":memory:", dimension=1536)
db = elips.open("/data/vectors", dimension=768, metric="euclidean")
db = elips.open("/data/exact_db", dimension=384, index="exact")
```

### `elips.open_with_config()`

```python
def open_with_config(path: str, config: Config) -> Database
```

Open (or create) a database with a fully configured `Config` builder.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | required | Filesystem path or `":memory:"` for ephemeral |
| `config` | `Config` | required | A configured `Config` instance |

**Returns:** A `Database` handle.

**Raises:**
- `ConfigError` — configuration conflict with persisted identity
- `LockConflict` — write lock is already held by another process

**Example:**

```python
from elips import Config, GraphParams

config = (Config()
    .dimension(1536)
    .metric("cosine")
    .index("graph")
    .graph_params(GraphParams(max_connections=32, ef_construction=400))
    .durability("standard"))

db = elips.open_with_config("/data/vectors", config)
```

## Database

```python
class Database
```

Top-level database handle. One per directory. Owns all vaults. Use as a context manager for automatic checkpoint and lock release.

### `Database.vault()`

```python
def vault(self, name: str) -> Vault
```

Access or lazily create a vault by name. If the vault does not exist, it is created with the database's configured dimension and metric.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `name` | `str` | The vault name |

**Returns:** A `Vault` handle.

**Raises:**
- `ConfigError` — vault name contains invalid characters

**Example:**

```python
docs = db.vault("documents")
images = db.vault("images")
```

### `Database.list_vaults()`

```python
def list_vaults(self) -> list[str]
```

List all vault names in the database.

**Returns:** List of vault name strings.

**Example:**

```python
for name in db.list_vaults():
    print(name)
```

### `Database.begin_transaction()`

```python
def begin_transaction(self) -> Transaction
```

Begin an atomic write transaction. Operations are buffered and applied only on `commit()`. An un-committed transaction rolls back on destruction or on context-manager exit with an exception.

**Returns:** A `Transaction` handle.

**Example:**

```python
with db.begin_transaction() as txn:
    v = txn.vault("docs")
    v.place([1.0, 2.0], {"title": "A"})
    v.place([3.0, 4.0], {"title": "B"})
    # Auto-committed on clean exit
```

### `Database.checkpoint()`

```python
def checkpoint(self) -> None
```

Flush all state to disk: write a full snapshot, truncate the write-ahead log (WAL). No-op for `":memory:"` databases.

**Example:**

```python
db.checkpoint()
```

### `Database.close()`

```python
def close(self) -> None
```

Checkpoint and release the database lock. Idempotent — safe to call multiple times. After `close()`, the handle should not be used.

**Example:**

```python
db.close()
```

### `Database.abandon()`

```python
def abandon(self) -> None
```

Drop the handle without checkpointing. This simulates a crash exit: only the WAL remains on disk. The next `open()` recovers via WAL replay.

Useful for testing recovery behavior.

**Example:**

```python
db.abandon()
# On next open(), database replays WAL to restore to last checkpointed state
```

### `Database.query()`

```python
def query(
    self,
    eql: str,
    bindings: Mapping[str, Vector] = {},
) -> list[Result]
```

Execute a single EQL statement with optional parameterized vectors.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `eql` | `str` | required | The EQL statement string |
| `bindings` | `Mapping[str, Vector]` | `{}` | Map of variable names to vector values, referenced as `$name` in EQL |

**Returns:** List of `Result` objects.

**Raises:**
- `ParseError` — malformed EQL syntax

**Examples:**

```python
results = db.query("seek in documents nearest $q top 10 yield",
                   bindings={"q": [1.0, 0.0, 0.0]})

db.query('place in items vector [1.0, 0.0] data {"name": "a"}')

db.query('fetch from items id "018f..." yield')

db.query('erase from items id "018f..."')

db.query('scan in items where val >= 10 limit 5 yield')
```

### `Database.gpu_info()`

```python
def gpu_info(self) -> GpuDeviceInfo
```

Return information about the detected GPU device.

**Returns:** A `GpuDeviceInfo` with fields such as `name`, `vendor`, `backend`, `total_memory_bytes`, `memory_gb`, `supports_cagra`, `peak_tflops_fp16`, etc.

**Availability:** Only when the extension is built with GPU support. Call `elips._has_gpu` to check first.

**Example:**

```python
if elips._has_gpu:
    info = db.gpu_info()
    print(f"{info.name} ({info.vendor}) — {info.memory_gb:.1f} GB")
```

### `Database.gpu_stats()`

```python
def gpu_stats(self) -> GpuMetricsSnapshot
```

Return a snapshot of GPU runtime metrics.

**Returns:** A `GpuMetricsSnapshot` with fields such as `search_p50_latency_us`, `search_p99_latency_us`, `batch_avg_size`, `batch_coalescing_ratio`, `index_build_speedup_vs_cpu_avg`, `fallback_events_total`, etc.

**Availability:** Only when the extension is built with GPU support.

**Example:**

```python
if elips._has_gpu:
    stats = db.gpu_stats()
    print(f"P50 latency: {stats.search_p50_latency_us} us")
    print(f"Fallback events: {stats.fallback_events_total}")
```

### `Database.config`

```python
@property
def config(self) -> Config
```

The effective configuration of this database. Read-only.

**Returns:** A `Config` object with populated dimension, metric, index type, durability, and (if built with GPU) GPU configuration.

**Example:**

```python
print(db.config.dimension_val)    # 1536
print(db.config.metric_val)       # "cosine"
print(db.config.metric_enum)      # Metric.cosine
print(db.config.index_enum)       # IndexType.graph
print(db.config.durability_enum)  # Durability.standard
```

### Context Manager

```python
def __enter__(self) -> Database
def __exit__(self, *args: Any) -> None
```

Using `Database` as a context manager automatically calls `close()` on exit, performing a checkpoint and releasing the file lock.

**Example:**

```python
with elips.open("/data/vectors", dimension=384) as db:
    docs = db.vault("documents")
    docs.place([1.0, 2.0, 3.0], {"title": "Hello"})
# close() called here automatically
```

For `":memory:"` databases, `close()` is a no-op but the pattern is safe to use.

### `Database.__repr__()`

```python
def __repr__(self) -> str
```

Returns a string like `<Database vaults=3>` showing the number of vaults.

## Complete Lifecycle Example

```python
import elips

db = elips.open("/data/vectors", dimension=3, metric="cosine")

docs = db.vault("documents")
docs.place([1.0, 0.0, 0.0], {"title": "alpha"})

print(db.list_vaults())           # ["documents"]
print(docs.count())               # 1

print(db.config.dimension_val)    # 3
print(db.config.metric_val)       # "cosine"

db.checkpoint()

results = db.query(
    "seek in documents nearest $q top 5 yield",
    bindings={"q": [1.0, 0.0, 0.0]},
)
print(results[0].id, results[0].distance)

db.close()
```

## Thread Safety

- **Single writer**: Only one write-capable connection per database directory (enforced by file locking)
- **Multiple readers**: Multiple read-only connections can coexist with each other and with a writer
- **Python GIL**: Access to `Database` from multiple threads is safe because pybind11 holds the GIL during C++ calls. Concurrent reads from the same `Database` handle are thread-safe. Concurrent writes (via `begin_transaction()`) are serialized at the C++ level