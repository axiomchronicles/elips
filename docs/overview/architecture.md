# Architecture Overview

This page is the short version of the current ELIPS runtime design. For the full
version, see [../architecture.md](../architecture.md).

## Runtime Shape

```text
Client / PersistentClient · low-level open() · C++ SDK · CLI
   |
Database                       (core engine handle, formerly ElipsInstance)
   |
Collection                     (built on Index, formerly Vault)
   |-- record store
   |-- MetadataFilterIndex
   |-- query planner
   |-- Index  (IndexFlat / IndexHNSW / IndexIVFFlat / IndexIVFPQ)
   |-- WAL attachment
   |
GPU layer (optional)
   |-- GpuBackend  (Metal / CUDA / HIP)
   |-- GpuDeviceManager + BackendSelector
   |-- QueryBatcher · bf_knn
```

## Index API

`Collection` is built on a FAISS-recognizable `Index` (formerly `IndexPort`):

- `IndexFlat` — exact brute force (the recall oracle; formerly `ExactIndex`)
- `IndexHNSW` — primary ANN graph index (formerly `HierarchicalGraphIndex`)
- `IndexIVFFlat` / `IndexIVFPQ` / `IndexIVFScalarQuantizer` — CPU IVF family
- `index_factory("IVF4096,PQ16")` — FAISS-style string constructor

## Query Flow

1. Validate and normalize the vector when needed.
2. Ask the planner for a `QueryPlan`.
3. Narrow candidates with `MetadataFilterIndex` when an exact metadata filter
   can be accelerated.
4. Use the ANN index, exact candidates, or full scan depending on query shape.
5. Hydrate results from the authoritative record store, including document,
   chunk, and lineage state.

Text-first queries use the same collection-level planner through `search_text()`
and `hybrid_search()`.

## GPU Flow

1. `device=` selects `cpu` / `metal` / `cuda:N` / `hip:N` / `auto`.
2. `GpuDeviceManager` probes compile-time-enabled backends; `BackendSelector`
   picks one per policy.
3. A named-but-absent backend raises `BackendUnavailableError` unless
   `fallback=True`, which demotes to CPU with a **logged** warning (never
   silent).
4. On Metal, distance, top-k, and IVF scan run as real GPU kernels loaded from
   `elips_kernels.metallib`; unified memory makes transfers a `memcpy`.
5. `bf_knn(xq, xb, k, device="metal")` runs exact GPU k-NN with no index build.

## Persistence Flow

1. Append to the WAL (CRC32C) before mutating in-memory state.
2. Resolve `TEXT_EMBEDDER.manifest` and the built-in local artifact when the
   database uses native text embedding.
3. Keep either segmented state (`elips.manifest` + `segments/`) or a snapshot
   file depending on configuration.
4. Replay the valid WAL prefix on open.
5. `compact()` rebuilds indexes and rewrites the persistent layout.

## Concurrency

- One read-write opener via exclusive advisory lock
- Many read-only openers via shared advisory lock
- Read-only collections reject mutation paths
- Search is single-threaded today; parallelize across independent read handles

## Public Surfaces

- C++: `open()`, `Config`, `Database`, `Collection`, and the `Index` layer
- Python low-level: `open()`, `open_with_config()`, `Collection`, `Config`,
  `IndexFlat`/`IndexIVFFlat`/`index_factory`/`bf_knn`
- Python Chroma-style: `Client()`, `PersistentClient(path=)`,
  `create_collection` / `get_or_create_collection` / `list_collections`

Old spellings (`connect`/`Engine`/`Arena`, `ElipsInstance`, `Vault`, `place`,
`seek`) remain as deprecation shims for one minor version.

New databases default to a built-in local text embedder, so Python and C++
text-first APIs can be used without explicitly supplying an embedder object.
