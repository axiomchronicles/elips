# Changelog

All notable changes to ELIPS are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ELIPS_SANITIZE=thread|address` CMake option, building the test suite under
  ThreadSanitizer or ASan+UBSan. The full suite (197 tests) is clean under TSan.

### Fixed

- **In-process reader/writer locking [F6].** Nothing outside `LocalTextEmbedder`
  held a lock. `LockManager` wraps a cross-process `flock` acquired once at open,
  which cannot serialize threads inside one process — so two threads sharing an
  instance raced on `HierarchicalGraphIndex`'s `data_`/`links_`/`id_to_node_` and
  on `Vault::records_`. Each `Vault` now owns a `std::shared_mutex` (readers
  share, mutators exclude) and `ElipsInstance` owns a mutex over the vault
  registry, WAL handle, and lifecycle flags. `Transaction::commit()` holds the
  instance lock for the whole batch so a concurrent writer cannot interleave
  mutations and invalidate the undo log. ADR-0008 and the transaction-engine
  internals doc previously claimed `LockManager` provided this; both corrected.
  *Behavioral change:* `Vault::records()` now returns a copy rather than a
  reference to the live map, since a reference could be mutated under a
  concurrent reader.

- **Writes after `close()` are refused instead of silently discarded [F7].**
  `close()` detached the WAL but left `Vault` objects live and writable.
  `Vault::place()` treated a null WAL as "skip logging" rather than "closed", so
  a post-close write returned a valid `RecordID`, mutated the in-memory index,
  and then vanished at process exit — silent data loss with a success return.
  `close()` now seals every vault (and any vault created afterwards); mutations
  throw `StorageError`. Reads still work.

- **WAL and checkpoint writes now reach stable storage before acknowledgement
  [F1].** `WAL::append()` called `std::ofstream::flush()`, which only pushes
  bytes into the OS page cache. Acknowledged writes were lost on OS crash or
  power loss — the exact failure mode a WAL exists to cover. The WAL now writes
  through a file descriptor and calls `fsync` (`F_FULLFSYNC` on macOS,
  `fdatasync` on Linux) before returning, and every snapshot, segment, manifest,
  and IDENTITY write is synced and published via a durable rename that also syncs
  the containing directory. *Behavioral change:* per-record write latency now
  includes a real device sync under `paranoid`/`standard` durability; `relaxed`
  is unaffected.

- **Length-prefixed reads are bounded before allocating [F2].** `get_string()`
  and `get_payload()` read a 32-bit count straight off disk and allocated for it,
  reachable from `WAL::replay()` *before* the CRC32C check that was supposed to
  reject corrupt records. A truncated or hostile log could trigger a ~4 GiB
  allocation or a 4-billion-iteration loop. Length prefixes are now validated
  against the bytes remaining in the stream (with a 64 MiB cap for non-seekable
  streams), and truncated fields raise `StorageError` instead of silently
  yielding default-constructed values.

- **`WAL::replay()` is linear in record count [F3].** Each iteration copied the
  entire remaining log tail into a fresh `std::string`, making recovery O(n·k).
  Records are now parsed in place over a non-owning stream buffer. *Measured:*
  replaying 8,000 records versus 2,000 (4× the data) now scales linearly rather
  than quadratically.

- **Transactions are atomic under I/O and read-only failure [F4].** `commit()`
  applied operations one at a time with no undo, and `rollback()` merely cleared
  an in-memory vector — so a read-only vault or a failed WAL write left the
  database holding *part* of a batch, silently. `commit()` now pre-checks that
  every target vault is writable, records each touched record's prior state, and
  restores that state in reverse order if any operation throws. The WAL brackets
  each batch with `txn_begin`/`txn_commit` markers; replay discards records
  inside an unterminated window, so a crash mid-commit also recovers to the
  pre-batch state. *Behavioral change:* `commit()` on a read-only vault now
  throws before applying anything (previously it threw partway through).
  ADR-0004 and ADR-0005 were corrected — both previously described guarantees the
  code did not deliver.
