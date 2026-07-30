# ADR-0008: File advisory locking for coordination

**Status:** Accepted
**Date:** 2024-08-01

## Context
Multiple processes may point at the same database directory. The single-writer /
multi-reader contract must be enforced without a coordinating daemon.

## Decision
Acquire a non-blocking exclusive advisory lock (`flock(LOCK_EX | LOCK_NB)`) on
`<dir>/LOCK` at `open()`. If another writer holds it, throw `LockConflict`. The
lock is RAII-bound to the instance and released on `close()`/destruction.
(Windows uses `LockFileEx` at the same seam — POSIX ships in v1.0.)

## Consequences
- Cross-process single-writer enforcement with no daemon.
- `close()` must release the lock so the same process can reopen.
- Shared (reader) locks for true multi-reader concurrency are a later refinement.
- **This lock does not synchronize threads.** It is advisory, cross-process, and
  acquired once at open, so it cannot serialize two threads in one process that
  share an instance. In-process safety is a separate mechanism: a
  `std::shared_mutex` per `Vault` plus an instance-level mutex over the vault
  registry and WAL handle (see ADR-0005 and
  `docs/internals/transaction-engine.md`).

## Alternatives Considered
- **Lock files with PID checks:** racy and stale-prone.
- **OS named mutex:** platform-specific and heavier.
- **Relying on this lock for thread safety (the original assumption):** rejected.
  The internals docs previously stated in-process synchronization came "via the
  `LockManager`"; it does not, and mutable index/record state was left
  unprotected against concurrent threads.
