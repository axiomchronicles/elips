# Changelog

All notable changes to ELIPS are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

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
