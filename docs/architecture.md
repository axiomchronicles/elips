# Architecture

ELIPS is layered with dependencies pointing inward toward the domain. Each layer
talks to the next through a narrow port (interface), following hexagonal /
clean-architecture principles.

```
        Python SDK (PyBind11)        C++ SDK (elips.hpp)        elips CLI
                     \                    |                    /
                      \                   |                   /
                                 ElipsInstance
                  (lifecycle · config · vault registry · txn)
                                       |
        ┌──────────────┬──────────────┼──────────────┬───────────────┐
   Query Engine    Vault /       Index Engine     Metadata        Storage
     (EQL)       Vector Engine    (IndexPort)      (Filter)     (WAL+snapshot)
                                       |                              |
                              ExactIndex / HNSW              LockManager · disk
```

## Layers

- **Domain** (`include/elips/domain`): pure value types — `Vector`, `RecordID`
  (UUIDv7), `Record`, `Payload`, `SearchResult`, error hierarchy. No
  infrastructure dependencies.
- **Vector engine** (`vector_engine/Metrics`): ordering-normalized `distance()`
  (smaller = more similar for every metric) with NEON SIMD kernels and a scalar
  fallback behind a runtime-dispatch seam.
- **Index engine** (`index_engine`): `IndexPort` abstraction; `ExactIndex`
  (brute force, ground truth) and `HierarchicalGraphIndex` (HNSW). `make_index`
  builds the configured backend.
- **Metadata** (`metadata/Filter`): a predicate tree + fluent builder evaluated
  against a record's `Payload`. Shared by the SDK and EQL.
- **Query engine** (`query_engine`): EQL lexer → parser → AST → executor.
- **Storage** (`storage`): `WAL`, `Serialization` (binary + CRC32C), atomic
  snapshot. `kernel/LockManager` enforces single-writer locking.
- **SDK** (`elips.hpp`): `open()`, `ElipsInstance`, `Vault`, `Transaction`.

## Key invariants

- `distance()` is ordering-normalized; all result lists sort ascending.
- Cosine vectors are L2-normalized on ingest **and** query.
- Writes are WAL-appended (and flushed) before in-memory state changes.
- `RecordID` byte order equals UUIDv7 time order, so the record map iterates in
  insertion-time order (deterministic snapshots).

## Design principles applied

SOLID, dependency inversion (consumers depend on `IndexPort`/`Filter`, never
concretes), RAII everywhere (`LockManager`, WAL, smart pointers), immutability by
default, and value semantics. See [ADRs](adr/) for the rationale behind each
major choice.
