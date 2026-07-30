# ADR-0005: Atomic batched transactions for v1.0

**Status:** Accepted
**Date:** 2024-08-01
**Revised:** 2026-07-30 (atomicity mechanism corrected — see Revision History)

## Context
The spec targets Snapshot Isolation via MVCC version chains. Full MVCC is a
large subsystem; v1.0 needs correct, all-or-nothing writes without it.

## Decision
v1.0 ships a transaction that buffers operations and applies them atomically on
`commit()`. Atomicity rests on three mechanisms, not on eager validation alone:

1. **Eager validation at enqueue time.** Dimension and finiteness are checked
   when an operation is enqueued, so malformed vectors are rejected before any
   state is touched.
2. **Pre-flight writability check at commit time.** `commit()` verifies every
   target vault is writable before applying the first operation, so the
   read-only case fails without partial application.
3. **In-memory undo log.** Runtime I/O failure (disk full, revoked permission,
   failed `fsync`) cannot be pre-validated away. `commit()` therefore records
   each record's prior state before mutating it, and on any exception restores
   those states in reverse order before rethrowing.

Durability is framed to match: the WAL brackets a batch with `txn_begin` /
`txn_commit` markers, and replay discards any records inside an unterminated
window. A crash mid-commit therefore recovers to the pre-batch state, matching
what the in-memory undo achieves for a caught exception.

`rollback()` discards an unapplied batch. After a failed `commit()` the state has
already been restored, so the destructor's implicit rollback is a no-op by design
rather than by omission.

## Consequences
- All-or-nothing semantics hold for the reachable failure modes: read-only
  vault, WAL/fsync I/O failure, and process crash mid-batch.
- Undo is bounded by batch size: the prior state of every touched record is held
  in memory for the duration of `commit()`. Very large batches cost memory
  proportional to the records they replace.
- Undo restores in-memory structures only. It deliberately bypasses the WAL and
  the read-only guard, because undo must work when the WAL write is the thing
  that failed; the WAL's transaction markers cover the durable side.
- No concurrent-writer conflict resolution yet; the single-writer lock plus
  in-process reader/writer locking makes that unnecessary for v1.0.
- Full MVCC version chains and reader snapshots are deferred (roadmap).

## Alternatives Considered
- **Full MVCC now:** matches the long-term goal but is disproportionate for v1.0.
- **Autocommit only:** no multi-write atomicity.
- **Eager validation alone (the original v1.0 design):** rejected. The claim that
  eager validation makes `commit()` unable to fail mid-batch was false:
  read-only vaults and WAL write failures both threw partway through the apply
  loop, leaving a partially-applied batch that the no-op `rollback()` never
  undid.

## Revision History
- **2026-07-30:** The original decision claimed eager enqueue-time validation
  made `commit()` incapable of failing mid-batch, and `rollback()` cleared only
  an in-memory op vector. An audit found two reachable counter-examples
  (read-only vault; WAL write/fsync failure), each leaving the database with a
  silent partial batch. The pre-flight writability check, undo log, and WAL
  transaction markers were added, and this ADR was corrected to describe the
  mechanism that actually delivers the guarantee.
