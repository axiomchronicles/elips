# API Reference: ElipsInstance

`elips::ElipsInstance` is the top-level database handle returned by
`elips::open()`.

## Factory

```cpp
std::unique_ptr<ElipsInstance> open(const std::string& path,
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

### `vault(name)`

Returns a reference to a named vault, creating it lazily.

### `list_vaults()`

Returns all current vault names.

### `begin_transaction()`

Starts an atomic write transaction.

### `query(eql, bindings={})`

Runs a single EQL statement and returns `std::vector<SearchResult>`.

### `checkpoint()`

Writes current state to disk and truncates the WAL.

- segmented mode: rewrites manifest + per-vault segment files
- snapshot mode: rewrites `elips.snapshot`

### `compact()`

Rebuilds every vault index from stored records and then checkpoints.

### `vacuum()`

```cpp
void vacuum();
```
Reclaims tombstoned index space across every vault. Unlike `compact()`, `vacuum()` does not rewrite the on-disk snapshot and works on in-memory databases. Calls each vault's `vacuum()` under the instance mutex.

### `close()`

Graceful shutdown: checkpoint, detach WAL, release the advisory lock.

> After `close()`, every vault becomes **sealed**. Any write operation on a sealed vault throws `StorageError` immediately rather than silently mutating memory that cannot be persisted.

### `abandon()`

Testing hook that suppresses destructor checkpointing so recovery must come from
the WAL.

### `config()`

Returns the effective `Config`.

### `gpu_info()` / `gpu_stats()`

Available when GPU support is compiled in and a GPU backend is active.

## Lifecycle Notes

- Persistent instances checkpoint on destruction unless already closed or opened
  read-only.
- Read-only instances never attach a WAL writer.
- Vaults created under a read-only instance are immediately marked read-only.

## Common Failure Modes

- `ConfigError`: missing dimension on new open, dimension mismatch, read-only
  open against a missing database
- `LockConflict`: another process already owns the write lock
- `StorageError`: on-disk IO failure

## Concurrency

Two layers of locking protect the instance:

**Cross-process** (via `LockManager`): An exclusive `flock(LOCK_EX)` on `<dir>/LOCK` prevents a second writer process from opening the same directory. Read-only instances acquire `LOCK_SH` instead, allowing multiple concurrent readers.

**In-process** (via mutexes): `ElipsInstance` owns a `std::recursive_mutex` over the vault registry, WAL handle, and lifecycle flags. Each `Vault` owns a `std::shared_mutex`; readers share it, mutators take it exclusively. `Transaction::commit()` holds the instance mutex for the whole batch, so concurrent commits serialize.

See `docs/internals/transaction-engine.md` and ADR-0008.
