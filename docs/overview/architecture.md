# Architecture Overview

This page is the short version of the current ELIPS runtime design. For the
full version, see [../architecture.md](../architecture.md).

## Runtime Shape

```text
SDKs / CLI
   |
ElipsInstance
   |
 Vault
   |-- record store
   |-- metadata index
   |-- planner
   |-- ANN / exact / GPU index
   |-- WAL attachment
```

## Query Flow

1. Validate and normalize the vector when needed.
2. Ask the planner for a `QueryPlan`.
3. Narrow candidates with `MetadataIndex` when an exact metadata filter can be
   accelerated.
4. Use ANN, exact candidates, or full scan depending on query shape.
5. Hydrate results from the authoritative record store, including document,
   chunk, and lineage state.

Text-first queries use the same vault-level planner path through
`seek_text()` and `seek_hybrid()`.

## Persistence Flow

1. Append to the WAL before mutating in-memory state.
2. Resolve `TEXT_EMBEDDER.manifest` and the built-in local artifact when the
   database uses native text embedding.
3. Keep either segmented state (`elips.manifest` + `segments/`) or a snapshot
   file depending on configuration.
4. Replay the valid WAL prefix on open.
5. `compact()` rebuilds indexes and rewrites the persistent layout.

## Concurrency

- One read-write opener via exclusive advisory lock
- Many read-only openers via shared advisory lock
- Read-only vaults reject mutation paths

## Public Surfaces

- C++: `open()`, `Config`, `ElipsInstance`, `Vault`
- Python low-level: `open()`, `open_with_config()`, `Database`, `Vault`
- Python modern: `connect()`, `Engine`, `Arena`

New databases default to a built-in local text embedder, so Python and C++
text-first APIs can be used without explicitly supplying an embedder object.
