# ADR-0005: Atomic batched transactions for v1.0

**Status:** Accepted
**Date:** 2024-08-01

## Context
The spec targets Snapshot Isolation via MVCC version chains. Full MVCC is a
large subsystem; v1.0 needs correct, all-or-nothing writes without it.

## Decision
v1.0 ships a transaction that buffers operations, validates them eagerly at
enqueue time (so `commit()` cannot fail mid-batch), and applies them atomically
on `commit()`. An un-committed transaction rolls back on destruction. Within the
single-writer model this gives in-process atomicity.

## Consequences
- Simple, correct all-or-nothing semantics with rollback.
- No concurrent-writer conflict resolution yet; the single-writer lock makes that
  unnecessary for v1.0.
- Full MVCC version chains and reader snapshots are deferred (roadmap).

## Alternatives Considered
- **Full MVCC now:** matches the long-term goal but is disproportionate for v1.0.
- **Autocommit only:** no multi-write atomicity.
