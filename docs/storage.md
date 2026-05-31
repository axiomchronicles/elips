# Storage & Recovery

A database is a directory on the local filesystem:

```
/my_db/
├── LOCK             # advisory writer lock (flock)
├── IDENTITY         # dimension, metric, index type (authoritative identity)
├── elips.snapshot   # full consistent state (atomic temp-write + rename)
└── wal.log          # write-ahead log since the last checkpoint
```

## IDENTITY

Written when the database is created. Format: `magic(0xE11D0001) | version |
dimension(u16) | metric(u8) | index(u8)`. It is the source of truth for the
database's shape and survives even before the first checkpoint, so a crash
immediately after creation still reopens with the right configuration. A
conflicting `dimension` passed to `open()` raises `ConfigError`.

## Write path (durability)

Every `place`/`erase` is appended to the WAL **before** memory is mutated. A
record is framed:

```
magic(0xE1105E01) | op(u8) | vault(len+bytes) | id(16B)
                  | [ dim(u16) | float32[dim] | payload ]      # insert only
                  | crc32c(u32)
```

`Durability` controls flushing: `paranoid`/`standard` flush every record;
`relaxed` buffers and flushes on checkpoint; `ephemeral` disables the WAL.

## Checkpoint

`checkpoint()` serializes all vaults to `elips.snapshot.tmp`, atomically renames
it over `elips.snapshot`, then truncates the WAL. `close()` checkpoints and
releases the WAL + lock. The destructor checkpoints on graceful teardown.

## Recovery

`open()`:
1. Acquire the writer lock (`LockConflict` if held).
2. Read `IDENTITY` (or create it for a new database).
3. Load `elips.snapshot` if present (rebuilds vaults + indexes).
4. Replay `wal.log` on top, applying inserts/erases.
5. Validate each WAL record's CRC32C; stop at the first corrupt/truncated record
   (the valid prefix is recovered — torn final writes are non-fatal).
6. Attach a live WAL for subsequent appends.

Recovery is deterministic: the same on-disk state always yields the same
in-memory state. `crash → reopen` replays exactly the writes that were durably
logged.

## Notes & limitations (v1.0)

- Serialization uses native byte order (single-machine embedded use). Cross-
  platform artifacts require little-endian normalization (future).
- Snapshots are whole-database rewrites; per-segment incremental persistence and
  compaction are on the roadmap.
