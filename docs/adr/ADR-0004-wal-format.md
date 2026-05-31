# ADR-0004: Record-based WAL with CRC32C

**Status:** Accepted
**Date:** 2024-08-01

## Context
Writes must survive a crash that happens before the next checkpoint. The log
format must allow safe recovery from a torn/partial final write.

## Decision
Use a record-based (logical) WAL. Each record frames `magic | op | vault | id |
[dim | floats | payload] | crc32c`. Records are appended and flushed before the
write is acknowledged (`paranoid`/`standard` durability). Replay validates each
record's CRC32C and stops cleanly at the first corrupt or truncated record,
returning the valid prefix.

## Consequences
- Recovery is deterministic and tolerant of partial tail writes.
- The logical format is replication-friendly (future).
- Per-record framing has modest space overhead vs page-based logs.

## Alternatives Considered
- **Page-based WAL:** efficient for B-trees but mismatched to our snapshot model.
- **No WAL (snapshot only):** loses all writes since the last checkpoint on crash.
