# Storage & Recovery

ELIPS persists a database directory and supports both manifest-based segmented
storage and the original snapshot format.

## Files On Disk

```text
/my_db/
├── LOCK
├── IDENTITY
├── TEXT_EMBEDDER.manifest
├── wal.log
├── elips.manifest        # segmented mode
├── text_embedder/
│   └── default_v1_<dim>.localembed
├── segments/
│   └── vault_<n>_<epoch>.segment
└── elips.snapshot        # snapshot mode or older databases
```

## Identity

`IDENTITY` is the durable source of truth for:

- vector dimension
- metric
- index type

Existing databases always reopen with the persisted identity. Passing a
conflicting dimension on reopen raises `ConfigError`.

## Text Embedder Manifest

`TEXT_EMBEDDER.manifest` records the database-level text embedding identity:

- provider / model / revision
- output dimension
- fingerprint
- whether the embedder is rehydratable
- whether it was auto-attached as the default local embedder
- the local artifact path (relative to the database when possible)

For the built-in local embedder, ELIPS also stores a deterministic artifact
under `text_embedder/`. On reopen, the manifest is used to restore the same
embedder automatically. For non-rehydratable external embedders, the manifest
stores metadata only; reopening without the same embedder causes text-first
APIs to fail with an actionable `ConfigError`.

## WAL

Every write is appended to `wal.log` before the in-memory vault is mutated.

Supported WAL operations:

- `insert`
- `erase`
- `insert_ex` for records carrying `DocumentAttachment`, `ChunkInfo`, or
  `EmbeddingLineage`

This means crash recovery restores full record state, not only vectors and
payloads.

`Durability` controls flush behavior:

- `paranoid` / `standard`: flush each write
- `relaxed`: buffer until checkpoint or close
- `ephemeral`: no WAL attachment

## Checkpoint

`checkpoint()` writes the current logical state to disk and truncates the WAL.

- Segmented mode writes one fresh segment per vault, rewrites
  `elips.manifest`, and removes obsolete segment files.
- Snapshot mode writes `elips.snapshot.tmp`, atomically renames it to
  `elips.snapshot`, and removes segmented artifacts.

`compact()` rebuilds each vault index from the authoritative record store and
then checkpoints.

## Recovery

`open()` performs:

1. Acquire advisory lock.
2. Read `IDENTITY`.
3. Resolve `TEXT_EMBEDDER.manifest` and attach the matching text embedder when
   possible.
4. Load `elips.manifest` + segments if present, otherwise load
   `elips.snapshot` if present.
5. Replay the valid WAL prefix.
6. Attach a live WAL unless the open is read-only or `ephemeral`.

Corrupt or truncated WAL tails are tolerated: replay stops at the first invalid
record and preserves the valid prefix.

## Read-Only Mode

Read-only opens require an existing database and take a shared lock.

- Multiple readers may coexist.
- No WAL writer is attached.
- Writes and maintenance operations raise `StorageError`.

This is the supported mode for shared-reader analytics or serving flows.
