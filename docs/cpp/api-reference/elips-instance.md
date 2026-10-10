# API Reference: Database

**Maturity:** stable (v1.0+).

`elips::Database` is the top-level database handle returned by `elips::open()`.

> Renamed in v1.1: the class was `ElipsInstance` (now a `[[deprecated]]` alias
> `using ElipsInstance = Database`). The accessor `vault(name)` is now
> `collection(name)`, and `list_vaults()` is `list_collections()` (old names
> kept as deprecated shims). See `MIGRATION.md`.

## Factory

```cpp
std::unique_ptr<Database> open(const std::string& path,
                               const Config& config = {});
```

Behavior:

- `":memory:"` opens are in-memory only and require `config.dimension() > 0`
- new persistent databases require a non-zero dimension
- existing persistent databases reuse the persisted identity
- `AccessMode::read_only` requires an existing database and acquires a shared
  advisory lock
- read-write opens acquire an exclusive advisory lock

On open, ELIPS loads segmented state if `elips.manifest` is present, otherwise
falls back to `elips.snapshot`, then replays the WAL.

## Core Methods

### `collection(name)`

Returns a reference to a named collection, creating it lazily.

### `list_collections()`

Returns all current collection names (`std::vector<std::string>`).

### `begin_transaction()`

Starts an atomic write transaction (`Transaction`).

### `query(eql, bindings={})`

Runs a single EQL statement and returns `std::vector<SearchResult>`.

### `train(collection_name)` / `quantize_all()`

Trains a quantization codebook and compresses a single collection (or every
collection) in place. For a persistent database this checkpoints afterwards,
publishing the codebook and encoded records durably before the WAL is
truncated. `train()` replaces the old `quantize()` (kept as a deprecated shim).

### `checkpoint()`

Writes current state to disk and truncates the WAL.

- segmented mode: rewrites manifest + per-collection segment files
- snapshot mode: rewrites `elips.snapshot`

### `compact()`

Rebuilds every collection index from stored records and then checkpoints.

### `vacuum()`

```cpp
void vacuum();
```

Reclaims tombstoned index space across every collection. Unlike `compact()`,
`vacuum()` does not rewrite the on-disk snapshot and works on in-memory
databases. Calls each collection's `vacuum()` under the instance mutex.

### `close()`

Graceful shutdown: checkpoint, detach WAL, release the advisory lock.

> After `close()`, every collection becomes **sealed**. Any write operation on a
> sealed collection throws `StorageError` immediately rather than silently
> mutating memory that cannot be persisted.

### `abandon()`

Testing hook that suppresses destructor checkpointing so recovery must come from
the WAL.

### `config()` / `path()` / `persistent()` / `closed()`

Accessors for the effective `Config`, the directory the database was opened from
(or `":memory:"`), and lifecycle flags.

### `gpu_info()` / `gpu_stats()`

Available when GPU support is compiled in (`ELIPS_GPU_ENABLED`) and a GPU backend
is active. Return `gpu::GpuDeviceInfo` and `gpu::GpuMetricsSnapshot`.

## Lifecycle Notes

- Persistent databases checkpoint on destruction unless already closed or opened
  read-only.
- Read-only databases never attach a WAL writer.
- Collections created under a read-only database are immediately marked
  read-only.

## Common Failure Modes

- `ConfigError`: missing dimension on new open, dimension mismatch, read-only
  open against a missing database
- `LockConflict`: another process already owns the write lock
- `StorageError`: on-disk IO failure

## Concurrency

Two layers of locking protect the database:

**Cross-process** (via `LockManager`): An exclusive `flock(LOCK_EX)` on
`<dir>/LOCK` prevents a second writer process from opening the same directory.
Read-only databases acquire `LOCK_SH` instead, allowing multiple concurrent
readers.

**In-process** (via mutexes): `Database` owns a `std::recursive_mutex` over the
collection registry, WAL handle, and lifecycle flags. Each `Collection` owns a
`std::shared_mutex`; readers share it, mutators take it exclusively.
`Transaction::commit()` holds the instance mutex for the whole batch, so
concurrent commits serialize.

See `docs/internals/transaction-engine.md` and ADR-0008.
