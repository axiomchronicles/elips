# ADR-0007: Snapshot + WAL persistence

**Status:** Accepted
**Date:** 2024-08-01

## Context
The spec describes per-segment indexes with compaction. That machinery is large;
v1.0 needs durable, crash-safe persistence with a simple, correct model.

## Decision
Persist each database as a directory holding `IDENTITY` (dimension/metric/index,
authoritative even before the first checkpoint), an atomic `elips.snapshot`
(full state, written to a temp file then renamed), and `wal.log`. `open()` loads
the snapshot then replays the WAL on top; `checkpoint()` rewrites the snapshot
and truncates the WAL. Indexes are rebuilt on load from stored vectors.

## Consequences
- Crash-safe and simple to reason about; recovery = snapshot + WAL replay.
- Whole-snapshot rewrite is O(N) per checkpoint — acceptable at v1.0 scale.
- Per-segment incremental persistence and compaction are deferred (roadmap).

## Alternatives Considered
- **Segments + compaction now:** matches the long-term design, large up-front cost.
- **mmap'd append-only store:** higher complexity for crash semantics.
