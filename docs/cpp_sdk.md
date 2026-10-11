# C++ SDK

The C++ surface is the runtime's source of truth. It exposes typed
configuration, document-aware records, planner introspection, persistence
control, and optional GPU-backed indexes.

> **Renamed in 1.1.** The core types and verbs were renamed (`Vault`→`Collection`,
> `ElipsInstance`→`Database`, `place`→`add`, `seek`→`search`, `fetch`→`get`,
> `erase`→`remove`, …). Old names remain as `[[deprecated]]` shims for one minor
> version — see [`MIGRATION.md`](../MIGRATION.md) for the full table.

New databases automatically attach the built-in local text embedder unless you
disable it with `Config::auto_text_embedder(false)`.

## Minimal Example

```cpp
#include "elips/elips.hpp"

auto db = elips::open(
    ":memory:",
    elips::Config{}
        .dimension(128)
        .metric(elips::Metric::cosine));

auto& docs = db->collection("documents");
docs.add_document("alpha design note", {{"kind", std::string{"design"}}});
auto hits = docs.search_text("alpha", 1);
```

## Primary Types

- `elips::Config`: fluent configuration builder
- `elips::Database`: top-level database handle (open with `elips::open`)
- `elips::Collection`: per-collection record store and query surface
- `elips::DocumentAttachment`, `elips::ChunkInfo`, `elips::EmbeddingLineage`
- `elips::QueryPlan`: planner output for vector and hybrid queries
- `elips::CollectionInfo`: summary statistics (`count`, `dimension`, `metric`, `codec`, `code_bytes`)

## Collection Operations

```cpp
auto& docs = db->collection("documents");

docs.add(elips::Vector{{1.0F, 0.0F}}, {{"kind", std::string{"vector"}}});
docs.add_document("alpha note", {{"kind", std::string{"text"}}});

auto vector_hits = docs.search(elips::Vector{{1.0F, 0.0F}}, 5);
auto text_hits   = docs.search_text("alpha", 5);
auto hybrid_hits = docs.hybrid_search(elips::Vector{{0.0F, 1.0F}}, "alpha", 5);
auto plan        = docs.explain_query(elips::Vector{{1.0F, 0.0F}}, 5,
                                      elips::Filter{}, std::nullopt, true);

auto record = docs.get(id);        // std::optional<Record>
bool erased = docs.remove(id);
auto rows   = docs.scan(elips::Filter{}, /*offset=*/0, /*limit=*/50);
```

## Persistence

Persistent opens support:

- segmented storage via `Config::segmented_storage(true)`
- metadata acceleration via `Config::metadata_acceleration(true)`
- shared readers via `Config::access_mode(elips::AccessMode::read_only)`
- `checkpoint()` and `compact()`

```cpp
auto db = elips::open(
    "/tmp/elips-cpp",
    elips::Config{}.dimension(2).segmented_storage(true));
db->compact();

auto reader = elips::open(
    "/tmp/elips-cpp",
    elips::Config{}.access_mode(elips::AccessMode::read_only));
```

## Local vs External Embedders

- The built-in local embedder is rehydratable and persisted via
  `TEXT_EMBEDDER.manifest` plus a local artifact under `text_embedder/`.
- Custom `TextEmbedderPort` implementations still work through
  `Config::text_embedder(...)`, but ELIPS can only persist their metadata. A
  later reopen must provide the same embedder explicitly before text-first APIs
  can be used again.

## Transactions And EQL

Transactions buffer `add()` / `remove()` and commit atomically:

```cpp
auto txn = db->begin_transaction();
txn.collection("documents").add(elips::Vector{{1.0F, 0.0F}});
txn.commit();
```

EQL remains available for declarative vector operations:

```cpp
auto rows = db->query(
    "search in documents nearest $q k 10 where kind = \"design\" yield",
    {{"q", elips::Vector{{1.0F, 0.0F}}}});
```

EQL verbs are `search` / `get` / `scan` / `add` / `delete` (the old
`seek`/`fetch`/`place`/`erase` and `top`/`data` keywords are accepted as
deprecated aliases).

## Errors

All runtime errors derive from `elips::ElipsError` (which derives
`std::runtime_error`). Typed subclasses (`include/elips/domain/Errors.hpp`):

| Exception | Thrown when |
|-----------|-------------|
| `DimensionMismatch` | Vector dimension does not match the collection config. Alias: `DimensionMismatchError` (same type). |
| `InvalidVector` | Vector contains NaN/Inf or is otherwise unusable. |
| `ConfigError` | Invalid config, or a config that conflicts with persisted identity. |
| `NotTrainedError` | An IVF-family index was used (`add`/`search`) before `train()` learned its coarse quantizer. Derives `ConfigError`. |
| `NotFound` | Requested record/collection does not exist. |
| `StorageError` | Persistence/IO failure (also raised on writes to a read-only open). |
| `BackendUnavailableError` | A required compute backend (e.g. a GPU device) was unavailable and no CPU fallback was permitted. |
| `LockConflict` | Database directory already locked by another writer/reader (`include/elips/kernel/LockManager.hpp`; derives `ElipsError`). |

## Indexes and `index_factory`

Collections build their index from `Config` (`Config::index(IndexType::graph |
IndexType::exact)` plus `Config::graph_params(HnswParams{...})`). The concrete
index types all implement the `elips::Index` contract
(`train`/`add_with_ids`/`search`/`range_search`/`remove`/`reset`/`ntotal`/
`is_trained`) — see [internals/index-engine.md](cpp/internals/index-engine.md).

To build an index directly from a FAISS-style spec string:

```cpp
#include "elips/index_engine/IndexFactory.hpp"

auto index = elips::index_factory(/*dimension=*/128, elips::Metric::cosine,
                                  "IVF4096,PQ8");
```

| Spec | Builds | Maturity |
|------|--------|----------|
| `"Flat"` | `IndexFlat` (exact brute force / recall oracle) | Stable |
| `"IVF<nlist>,Flat"` | `IndexIVFFlat(nlist)` | Stable |
| `"IVF<nlist>,PQ<M>"` | `IndexIVFPQ(nlist, M)` | Stable |
| `"IVF<nlist>,SQ8"` | `IndexIVFScalarQuantizer(nlist)` | Stable |
| `"HNSW<M>"` | `IndexHNSW(M)` | Stable |

`index_factory` throws `ConfigError` on an unrecognized or malformed spec. The
IVF family must be `train()`-ed before `add_with_ids`/`search` or it raises
`NotTrainedError`.

## C API

A thin `extern "C"` shim (`include/elips/c_api/elips.h`) over the index
lifecycle, for FFI / other-language callers that cannot consume the C++ headers.
Built only with `-DELIPS_ENABLE_C_API=ON` (produces `elips_c`).

**Maturity: Experimental / intentionally minimal.** Covers single-vector add and
k-NN search over factory-built indexes only. NOT covered: batched add/search,
range search, `train()` for IVF before add, persistence, metadata, and the
`Collection`/`Database` surface. IVF specs build but assert untrained until
`train()` is exposed here.

```c
#include "elips/c_api/elips.h"

elips_index* ix = elips_index_create(128, ELIPS_METRIC_COSINE, "Flat");
elips_index_add(ix, /*id=*/42, vec, 128);

int64_t ids[10]; float dist[10]; size_t n = 0;
elips_index_search(ix, query, 128, 10, ids, dist, &n);
size_t total = elips_index_ntotal(ix);

elips_index_free(ix);  // safe on NULL
```

- `elips_metric`: `ELIPS_METRIC_COSINE`, `ELIPS_METRIC_EUCLIDEAN`, `ELIPS_METRIC_DOT_PRODUCT`.
- `elips_status`: `ELIPS_OK`, `ELIPS_ERR_INVALID_ARG`, `ELIPS_ERR_RUNTIME`.
- Identity: the caller's `int64` id is encoded big-endian into the first 8 bytes
  of the 16-byte `RecordID` and decoded back on search; ids must be unique per
  index.
- `elips_index_create` returns `NULL` on a bad spec/metric/dimension.
