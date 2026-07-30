# ADR-0004: Record-based WAL with CRC32C

**Status:** Accepted
**Date:** 2024-08-01
**Revised:** 2026-07-30 (durability mechanism corrected; transaction framing added)

## Context
Writes must survive a crash that happens before the next checkpoint. The log
format must allow safe recovery from a torn/partial final write.

## Decision
Use a record-based (logical) WAL. Each record frames `magic | op | vault | id |
[dim | floats | payload] | crc32c`. Records are appended with a single `write()`
and, under `paranoid`/`standard` durability, `fsync`'d (`F_FULLFSYNC` on macOS,
`fdatasync` on Linux) before the write is acknowledged — a stream flush alone
only reaches the OS page cache and would lose acknowledged writes on power loss.
Replay validates each record's CRC32C and stops cleanly at the first corrupt or
truncated record, returning the valid prefix. Length-prefixed fields are checked
against the remaining input before allocation, so a corrupt prefix cannot trigger
a multi-gigabyte allocation ahead of the CRC check.

Batches are bracketed by `txn_begin` / `txn_commit` marker records. Replay
buffers records inside an open window and discards them if no commit marker
follows, so a crash mid-batch cannot resurrect a partial transaction.

## Consequences
- Recovery is deterministic and tolerant of partial tail writes.
- Acknowledged writes survive OS crash and power loss, not just process crash.
- Replay is linear in log size (records are parsed in place rather than by
  copying the remaining tail per record).
- The logical format is replication-friendly (future).
- Per-record framing has modest space overhead vs page-based logs; per-record
  `fsync` bounds write throughput (group commit is a roadmap item).

## Alternatives Considered
- **Page-based WAL:** efficient for B-trees but mismatched to our snapshot model.
- **No WAL (snapshot only):** loses all writes since the last checkpoint on crash.
- **Stream flush without fsync (the original implementation):** rejected. It
  satisfies process-crash durability only; the page cache is lost on OS crash or
  power loss, which is the failure mode a WAL exists to cover.
