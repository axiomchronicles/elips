# C++ SDK

The C++ surface is the runtime's source of truth. It exposes typed
configuration, document-aware records, planner introspection, persistence
control, and optional GPU-backed indexes.

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

auto& docs = db->vault("documents");
docs.place_document("alpha design note", {{"kind", std::string{"design"}}});
auto hits = docs.seek_text("alpha", 1);
```

## Primary Types

- `elips::Config`: fluent configuration builder
- `elips::ElipsInstance`: top-level database handle
- `elips::Vault`: per-collection record store and query surface
- `elips::DocumentAttachment`, `elips::ChunkInfo`, `elips::EmbeddingLineage`
- `elips::QueryPlan`: planner output for vector and hybrid queries

## Vault Operations

```cpp
auto& docs = db->vault("documents");

docs.place(elips::Vector{{1.0F, 0.0F}}, {{"kind", std::string{"vector"}}});
docs.place_document("alpha note", {{"kind", std::string{"text"}}});

auto vector_hits = docs.seek(elips::Vector{{1.0F, 0.0F}}, 5);
auto text_hits = docs.seek_text("alpha", 5);
auto hybrid_hits = docs.seek_hybrid(elips::Vector{{0.0F, 1.0F}}, "alpha", 5);
auto plan = docs.explain_seek(elips::Vector{{1.0F, 0.0F}}, 5,
                              elips::Filter{}, std::nullopt, true);
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

Transactions buffer `place()` / `erase()` and commit atomically:

```cpp
auto txn = db->begin_transaction();
txn.vault("documents").place(elips::Vector{{1.0F, 0.0F}});
txn.commit();
```

EQL remains available for declarative vector operations:

```cpp
auto rows = db->query(
    "seek in documents nearest $q top 10 where kind = \"design\" yield",
    {{"q", elips::Vector{{1.0F, 0.0F}}}});
```

## Errors

All runtime errors derive from `elips::ElipsError`. Common subclasses:

- `DimensionMismatch`
- `InvalidVector`
- `ConfigError`
- `StorageError`
- `LockConflict`
- `NotFound`
