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

## Alternatives Considered
- **Lock files with PID checks:** racy and stale-prone.
- **OS named mutex:** platform-specific and heavier.
