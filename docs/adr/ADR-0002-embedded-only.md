# ADR-0002: Embedded-only deployment (no server)

**Status:** Accepted
**Date:** 2024-08-01

## Context
Most vector databases ship as networked servers, adding operational burden
(ports, daemons, orchestration) that small and embedded applications do not want.

## Decision
ELIPS runs in-process only. A database is a directory on the local filesystem.
There is no listener, no daemon, and no network code. Multiple processes
coordinate through file-level advisory locks (see ADR-0008).

## Consequences
- Zero infrastructure; the database lives and dies with the host process.
- Distribution is a single library/wheel.
- Multi-node features (replication, sharding) are out of scope for v1.0 but the
  WAL is a logical log that a future layer can stream (see roadmap).

## Alternatives Considered
- **Client/server:** scales horizontally but contradicts the SQLite-like goal.
- **Sidecar process:** still requires lifecycle management and IPC.
