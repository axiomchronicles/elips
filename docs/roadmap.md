# Roadmap

What ELIPS v1.0 ships and what is intentionally deferred, with the v1.0 hook that
keeps each future capability additive.

## Shipped in v1.0

- C++23 core with hexagonal layering and the C++ Core Guidelines.
- Metrics: cosine, euclidean, dot product (NEON SIMD + scalar dispatch).
- Indexes: `HierarchicalGraphIndex` (HNSW) and `ExactIndex`, behind `IndexPort`.
- Durable storage: `IDENTITY`, atomic snapshot, record-based WAL with CRC32C,
  deterministic crash recovery.
- Single-writer advisory file locking.
- Dynamic metadata + `Filter` predicate engine.
- Atomic batched transactions with rollback.
- EQL: lexer, parser, AST, executor.
- `elips` CLI; PyBind11 Python bindings (`py.typed`); benchmark suite.

## Deferred (with v1.0 hooks)

| Future capability | v1.0 hook |
|---|---|
| Per-segment indexes + compaction | `IndexPort`; snapshot/WAL split per vault |
| Full MVCC version chains / Snapshot Isolation | single-writer model + txn buffer |
| Quantized indexes (PQ/OPQ/SQ), DiskANN | `IndexPort` + `make_index` factory |
| AVX2 / AVX-512 kernels | function-pointer dispatch seam in `Metrics` |
| Columnar metadata, attribute B-trees, inverted/bloom | `Filter` over `Payload` today |
| Multi-reader shared locks | `LockManager` seam |
| Multi-node replication / sharding | WAL is a logical, streamable log |
| Cloud object-storage adapters (S3/GCS/Azure) | `StoragePort` adapter pattern |
| Numpy zero-copy ingestion, async/streaming C++ APIs | binding + SDK extension points |
| Cross-platform snapshot encoding (little-endian) | centralized `Serialization` |

## Known v1.0 limitations

- Snapshot serialization uses native byte order (single-machine use).
- Checkpoints rewrite the whole snapshot (O(N)).
- Filtered ANN search post-filters an over-fetched candidate set rather than
  expanding `ef` adaptively.
