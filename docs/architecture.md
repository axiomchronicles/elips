# Architecture

ELIPS is an embedded retrieval engine organized around a FAISS-recognizable
**Index API**, a Chroma-recognizable **Collection** built on top of that Index,
and an optional **GPU layer**. Every record may carry metadata, a source
document attachment, chunk coordinates, and embedding lineage, so the engine is
document-aware rather than vector-only.

```text
Python Client / PersistentClient  ·  Python low-level  ·  C++ SDK  ·  CLI
                              |
                           Database
        lifecycle · config · collection registry · WAL · GPU backend state
                              |
                          Collection                (built on Index)
   record store · query planner · MetadataFilterIndex · Index · text/hybrid
                              |
      ------------------------------------------------------------------
      |                  |                   |                 |
   Index API         Query path          Persistence        Locking
 IndexFlat/HNSW    search / scan / get   manifest / WAL    shared RO /
 IVFFlat/IVFPQ     search_text /         segments / snap   exclusive RW
 index_factory     hybrid_search
                              |
                     GPU layer (optional)
   GpuBackend interface · Metal / CUDA / HIP · GpuDeviceManager
   BackendSelector · QueryBatcher · bf_knn · logged device selection
```

## Core Objects

- `Database`: top-level engine handle (the C++ core type, formerly
  `ElipsInstance`). Owns collections, persistence, locks, and optional GPU
  backend state. In Python it is reached through `Client()` (ephemeral) and
  `PersistentClient(path=)`, which wrap the same core.
- `Collection`: authoritative record store plus query planner and `Index`
  (formerly `Vault`). The Chroma-style verbs (`add`, `get`, `query`/`search`,
  `update`, `upsert`, `delete`, `count`, `peek`) all route through one
  collection onto its `Index`.
- `Record`: `id`, `vector`, `payload`, and optional `document`, `chunk`,
  `lineage`.
- `MetadataFilterIndex`: exact-match accelerator used by the planner for
  equality and set-membership filters (formerly `MetadataIndex`).
- `TextEmbedderPort`: optional core interface for native text ingestion and
  text-first querying.

## Index API

The Index layer is deliberately FAISS-shaped. `Collection` is built on an
`Index`, and the same index types are exposed directly in both C++ and Python.

- `Index`: abstract base (formerly `IndexPort`) defining the FAISS contract —
  `train`, `add`, `add_with_ids`, `search`, `range_search`, `reconstruct`,
  `remove`, `reset`, `is_trained`, `ntotal`, `d`, `metric_type`.
- `IndexFlat`: exact brute-force scan (formerly `ExactIndex`); the ground-truth
  oracle for recall benchmarks.
- `IndexHNSW`: graph ANN index (formerly `HierarchicalGraphIndex`), the primary
  approximate index. Tuned via `HnswParams` (`M`, `ef_construction`,
  `ef_search`; formerly `GraphParams`).
- `IndexIVFFlat`, `IndexIVFPQ`, `IndexIVFScalarQuantizer`: the CPU IVF family,
  composed from the existing `KMeans` / `ProductQuantizer` / `ScalarQuantizer`.
- `index_factory`: parses a FAISS-style string (e.g. `"IVF4096,PQ16"`) into a
  concrete index (replaces the old fluent `IndexFactory` / `make_index`).

`IndexType::hnsw` / `IndexType::flat` are the canonical enumerators (the old
`graph` / `exact` names remain as deprecated aliases).

## Query Path

Every vector or hybrid query goes through the collection's query planner first.

- `ann_index`: use the configured ANN index (`IndexHNSW`, IVF, …) directly.
- `exact_candidates`: restrict the search domain using `MetadataFilterIndex`.
- `full_scan`: fall back when the candidate set is small or the query shape
  makes a scan cheaper.
- `text_probe`: text-only search when no embedding stage is needed.
- `hybrid_fusion`: combine vector distance with lexical overlap from attached
  documents.

The planner emits `QueryPlan`, exposed in both C++ and Python for inspection and
testing.

## GPU Layer

The GPU layer is optional and compiled in behind backend flags. Domain code
depends only on the abstract `GpuBackend` interface (formerly `GpuPort`), never
on concrete backend types.

- `GpuBackend`: abstract interface — `initialize`/`shutdown`, device info,
  device allocation + `upload`/`download`, `compute_distances_batch`, `top_k`,
  `ivf_scan`, `synchronize`.
- Backends: **Metal** (Apple, validated on Apple M4), **CUDA** (NVIDIA,
  experimental), **HIP** (AMD, experimental). See [the GPU guide](gpu/index.md)
  for the maturity table.
- `GpuDeviceManager`: probes compile-time-enabled backends and reports device
  info; `BackendSelector` (formerly `GpuSelector`) orders candidates and selects
  one per policy.
- `QueryBatcher` (formerly `DynamicBatcher`): coalesces concurrent single
  queries into one batched kernel launch (time window + max batch size).
- `bf_knn`: standalone exact GPU brute-force k-NN over raw arrays, no index
  build (`elips.bf_knn(xq, xb, k, device="metal")`).
- **Device selection is explicit and fallback is never silent.** `device=` picks
  `cpu` / `metal` / `cuda:N` / `hip:N` / `auto`. A named backend that is absent
  raises `BackendUnavailableError` at open time unless `fallback=True`, which
  demotes to CPU with a logged warning (`[elips] warning: GPU backend ...`).
  `Auto` always permits a logged CPU fallback.

The Metal path runs real GPU kernels end to end: tiled-GEMM distance
(`elips_l2_matrix` / `elips_ip_matrix`), a GPU top-k (`topk_select`), and a
two-pass GPU IVF scan (`elips_ivf_scan_list` + `elips_ivf_merge_lists`), all
loaded from a runtime `elips_kernels.metallib`. On Apple Silicon every buffer is
`MTLResourceStorageModeShared`, so CPU↔GPU "transfer" is a `memcpy` over
`[buffer contents]` with no blit.

## Persistence Path

Persistent databases can run in two layouts:

- Segmented mode, enabled by default: root `elips.manifest` plus one segment
  file per collection under `segments/`.
- Snapshot mode: a single `elips.snapshot` file for compatibility or when
  segmented storage is disabled.
- Text embedder identity: `TEXT_EMBEDDER.manifest` plus optional
  `text_embedder/*.localembed` artifacts for the built-in local embedder.

All mutations are WAL-appended (CRC32C-checked) before in-memory mutation. WAL
replay rebuilds the record store on open, including document attachments, chunk
info, and embedding lineage.

## Concurrency Model

- One read-write opener at a time via an exclusive advisory lock.
- Multiple read-only openers via shared advisory locks.
- Read-only collections reject `add`, `add_document`, `remove`,
  `rebuild_index`, and compaction-driven mutation paths with `StorageError`.
- In-process: concurrent readers are safe; writes are serialized under a single
  writer. Search is currently single-threaded (a parallel-search pool is a
  deferred performance item); callers may parallelize across independent
  read-only handles.

ELIPS is an embedded engine, not a server. Process-level coordination is done
through file locks rather than background daemons.

## Python Surface

Two layers over the same core:

- Low-level bindings: `open()`, `open_with_config()`, `Collection`, `Config`,
  plus the FAISS Index layer (`IndexFlat`/`IndexIVFFlat`/`IndexIVFPQ`/
  `IndexHNSW`, `index_factory`, `read_index`/`write_index`, `bf_knn`).
- Chroma-style entry: `Client()`, `PersistentClient(path=)`, and
  `create_collection` / `get_collection` / `get_or_create_collection` /
  `delete_collection` / `list_collections`. Old entry points
  (`connect`/`open`/`Engine`/`Arena`) remain as deprecation shims.

Text-first APIs prefer the native core embedder when the config resolves one
(including the default local embedder attached to new databases); otherwise a
Python callable embedder drives Python-side embedding plus hybrid querying. If
neither exists, text-first APIs raise `ConfigError` / `ValueError` rather than
silently degrading to lexical-only.

## Read More

- [GPU guide](gpu/index.md) — backend maturity, device selection, troubleshooting
- [Which index should I use?](overview/choosing-an-index.md) — with measured benchmarks
- [Architecture Decision Records](adr/README.md)
