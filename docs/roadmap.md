# Roadmap

What ELIPS ships today and what remains, kept honest against the current code.

## Shipped

- C++23 core with hexagonal layering and the C++ Core Guidelines.
- Metrics: cosine, euclidean (true L2), dot product (NEON SIMD + scalar dispatch;
  `ELIPS_OPT_LEVEL` selects generic/neon/avx2/avx512).
- Indexes behind `Index`: `IndexFlat`, `IndexHNSW`, and the **CPU IVF family**
  (`IndexIVFFlat`, `IndexIVFPQ`, `IndexIVFScalarQuantizer`) — quantized indexes
  (PQ / SQ) are **done**, no longer deferred.
- FAISS-style `index_factory` string grammar and standalone `bf_knn`.
- Chroma-style `Client` / `PersistentClient` + Collection API, with arbitrary
  string ids.
- Durable storage: `IDENTITY`, segmented + snapshot formats, record-based WAL with
  CRC32C and `fsync`, deterministic crash recovery.
- Concurrency: per-collection `std::shared_mutex` — **multi-reader shared locks**
  are done. Adaptive-`ef` filtered HNSW search (widens the beam under selective
  filters and tombstone churn) is **done**.
- Metadata `Filter` engine; atomic batched transactions with rollback.
- EQL: lexer, parser, AST, executor (new `add/search/get/delete` verbs).
- `elips` CLI; PyBind11 Python bindings (`py.typed`); benchmark suite.
- Metal GPU backend validated on Apple Silicon, with real brute-force and IVF
  kernels.

## Remaining work

| Capability | Status / hook |
|---|---|
| GPU IVFPQ scan **on-device** (PQ list scan kernels) | IVF scan kernels partially land; PQ-on-device remains |
| CUDA / HIP **hardware validation** | interfaces compile today; no real-kernel run on NVIDIA/AMD hardware (EXPERIMENTAL) |
| Parallel / multi-threaded search (OpenMP or a pool) | search is single-threaded today; threading-policy seam exists |
| `range_search` / `reconstruct` on the **Python** Index | present in C++ (`IndexFlat`, `IndexIVFFlat`); not yet bound to Python |
| Index serialization in Python (`read_index` / `write_index`) | ELIPS snapshots are versioned; Python index I/O not yet exposed |
| GPU resources API (`StandardGpuResources`, `index_cpu_to_gpu`) | GPU indexes exist; the explicit FAISS-style transfer API is not exposed |
| `abi3` / stable-ABI wheels + cpu/gpu packaging split | scikit-build-core build wired; abi3 wheel not yet produced |
| `where_document`, `delete_collection` / `reset` | accepted in the signature; raise `NotImplementedError` today |
| MVCC / snapshot-isolation version chains | single-writer model + txn buffer |
| Cross-platform snapshot encoding (little-endian) | centralized `Serialization` |
| Multi-node replication / sharding; object-storage adapters | WAL is a logical streamable log; `StoragePort` seam |

## Known limitations

- Snapshot serialization uses native byte order (single-machine use).
- Checkpoints rewrite the whole snapshot (O(N)); compaction rebuilds a graph
  rather than merging segments incrementally.
- Per-record `fsync` bounds write throughput under `paranoid`/`standard`
  durability; group commit is not implemented.
- `euclidean` returns true L2, not FAISS's squared L2 (see the [FAQ](faq.md)).
