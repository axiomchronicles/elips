# Quickstart

This quickstart uses the current C++ API: typed config, native text embedding,
document-aware records, hybrid retrieval, and read-only reopen.

## Minimal Example

```cpp
#include "elips/elips.hpp"

auto db = elips::open(
    ":memory:",
    elips::Config{}
        .dimension(2)
        .metric(elips::Metric::cosine));

auto& docs = db->vault("documents");
docs.place_document("alpha design note", {{"kind", std::string{"design"}}});
docs.place_document("beta incident runbook", {{"kind", std::string{"ops"}}});

const auto hits = docs.seek_text("alpha", 2);
```

New databases attach ELIPS' built-in local embedder automatically. Use
`Config::local_text_embedder(...)` when you want to pin model/revision or
artifact location explicitly, and `Config::text_embedder(...)` when you want to
attach a custom runtime `TextEmbedderPort`.

## Open Persistent Storage

```cpp
auto db = elips::open(
    "/tmp/elips-cpp",
    elips::Config{}
        .dimension(2)
        .segmented_storage(true)
        .metadata_acceleration(true));
```

For a new database you must provide a non-zero dimension. Existing databases
reuse their persisted identity.

## Ingest With Lineage

```cpp
elips::ChunkInfo chunk;
chunk.document_key = "doc-alpha";
chunk.ordinal = 0;
chunk.char_start = 0;
chunk.char_end = 17;

elips::EmbeddingLineage lineage;
lineage.provider = "demo";
lineage.model = "toy";
lineage.revision = "v1";
lineage.attributes = {{"stage", std::string{"quickstart"}}};

docs.place_document("alpha design note",
                    {{"kind", std::string{"design"}}},
                    std::nullopt,
                    chunk,
                    lineage);
```

You can also store explicit vectors with an attached source document:

```cpp
docs.place(
    elips::Vector{{1.0F, 0.0F}},
    {{"kind", std::string{"appendix"}}},
    std::nullopt,
    elips::DocumentAttachment{.text = "gamma appendix"}
);
```

## Query

```cpp
const auto vector_hits = docs.seek(elips::Vector{{1.0F, 0.0F}}, 5);
const auto text_hits = docs.seek_text("alpha", 5);
const auto hybrid_hits =
    docs.seek_hybrid(elips::Vector{{0.0F, 1.0F}}, "alpha", 5);
```

Planner inspection:

```cpp
const auto filter =
    elips::Filter().field("kind").equals(std::string{"design"});
const auto plan = docs.explain_seek(elips::Vector{{1.0F, 0.0F}},
                                    1,
                                    filter,
                                    std::nullopt,
                                    true);
```

## Persistence

```cpp
db->checkpoint();
db->compact();
db->close();
```

`compact()` rebuilds every vault index from the authoritative record store and
then checkpoints the new segment set.

## Read-Only Open

```cpp
auto reader = elips::open(
    "/tmp/elips-cpp",
    elips::Config{}.access_mode(elips::AccessMode::read_only));
const auto rows = reader->vault("documents").seek_text("beta", 1);
```

Read-only open requires an existing database and rejects writes with
`StorageError`.

## Next

- [C++ SDK](../../cpp_sdk.md)
- [Config API Reference](../api-reference/config.md)
- [Vault API Reference](../api-reference/vault.md)
- [ElipsInstance API Reference](../api-reference/elips-instance.md)
