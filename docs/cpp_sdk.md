# C++ SDK

Link against `elips_core` and include `<elips/elips.hpp>`.

```cpp
#include <elips/elips.hpp>

auto db = elips::open("/data/vectors",
    elips::Config{}.dimension(1536).metric(elips::Metric::cosine));

auto& docs = db->vault("documents");

const elips::RecordID id = docs.place(
    elips::Vector{{0.1F, 0.2F, /* ... */}},
    elips::Payload{{"title", std::string{"Doc"}}, {"year", std::int64_t{2024}}});

for (const auto& hit : docs.seek(elips::Vector{query}, 10)) {
    std::cout << hit.id.to_string() << ' ' << hit.distance << '\n';
}
```

## Configuration

```cpp
elips::Config{}
    .dimension(768)
    .metric(elips::Metric::euclidean)
    .index(elips::IndexType::graph)                 // or ::exact
    .graph_params({.max_connections = 16,
                   .ef_construction = 200,
                   .ef_search = 50})
    .durability(elips::Durability::standard);       // paranoid|standard|relaxed|ephemeral
```

## Filters

```cpp
auto f = elips::Filter()
    .field("category").equals(std::string{"tech"})
    .field("year").ge(std::int64_t{2023});

auto results = docs.seek(query, 10, f);             // filtered
auto rows    = docs.scan(f);                        // metadata-only scan
```

## Transactions

```cpp
{
    auto txn = db->begin_transaction();
    auto v = txn.vault("documents");
    v.place(vec1, payload1);
    v.place(vec2, payload2);
    txn.commit();                  // atomic; rolls back if the scope exits early
}
```

## EQL

```cpp
auto hits = db->query(
    "seek in documents nearest $q top 10 where year >= 2022 yield",
    {{"q", elips::Vector{query}}});
```

## Lifecycle

```cpp
db->checkpoint();   // flush snapshot, truncate WAL
db->close();        // checkpoint + release lock (idempotent)
```

Errors derive from `elips::ElipsError` (`DimensionMismatch`, `InvalidVector`,
`ConfigError`, `StorageError`, `NotFound`, `LockConflict`). Throw by value, catch
by reference.
