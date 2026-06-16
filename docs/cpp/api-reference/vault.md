# API Reference: Vault

`elips::Vault` is the document-aware record store and query surface for a single
named partition.

## Construction

Users do not construct `Vault` directly. Obtain it from:

```cpp
auto& vault = db->vault("documents");
```

## Write Methods

### `place()`

```cpp
RecordID place(const Vector& vector,
               Payload payload = {},
               std::optional<RecordID> id = std::nullopt,
               std::optional<DocumentAttachment> document = std::nullopt,
               std::optional<ChunkInfo> chunk = std::nullopt,
               std::optional<EmbeddingLineage> lineage = std::nullopt);
```

Stores an explicit vector and optional document/lineage metadata.

### `place_document()`

```cpp
RecordID place_document(std::string text,
                        Payload payload = {},
                        std::optional<RecordID> id = std::nullopt,
                        std::optional<ChunkInfo> chunk = std::nullopt,
                        std::optional<EmbeddingLineage> lineage = std::nullopt);
```

Embeds the text through the configured `TextEmbedderPort`, stores the source
document, fills chunk defaults, and auto-generates lineage when omitted.

### `place_many()`

```cpp
void place_many(const std::vector<Record>& records);
```

Each `Record` may include `document`, `chunk`, and `lineage`.

## Query Methods

### `seek()`

```cpp
std::vector<SearchResult> seek(const Vector& query,
                               std::size_t top,
                               const Filter& filter = Filter{},
                               std::optional<float> threshold = std::nullopt) const;
```

Vector similarity search.

### `seek_text()`

```cpp
std::vector<SearchResult> seek_text(std::string_view text,
                                    std::size_t top,
                                    const Filter& filter = Filter{},
                                    std::optional<float> threshold = std::nullopt) const;
```

Text-first query surface. Uses the native text embedder when configured,
otherwise raises `ConfigError` with guidance to configure or reattach a text
embedder. Use `seek_hybrid()` when you already have the query vector and want
to blend it with lexical overlap from attached documents.

### `seek_hybrid()`

```cpp
std::vector<SearchResult> seek_hybrid(const Vector& query,
                                      std::string_view text,
                                      std::size_t top,
                                      const Filter& filter = Filter{},
                                      std::optional<float> threshold = std::nullopt,
                                      float lexical_weight = 0.25F) const;
```

Blends vector distance with lexical overlap from attached documents.

### `explain_seek()`

```cpp
QueryPlan explain_seek(const Vector& query,
                       std::size_t top,
                       const Filter& filter = Filter{},
                       std::optional<float> threshold = std::nullopt,
                       bool has_text_component = false) const;
```

Returns the planner decision:

- `QueryStrategy`
- candidate count
- metadata acceleration flag
- GPU index flag
- index type name

## Retrieval

### `fetch()`

```cpp
std::optional<Record> fetch(const RecordID& id) const;
```

Returns the full stored `Record`, including `document`, `chunk`, and `lineage`.

### `scan()`

```cpp
std::vector<Record> scan(const Filter& filter = Filter{},
                         std::size_t offset = 0,
                         std::size_t limit = std::numeric_limits<std::size_t>::max()) const;
```

Metadata scan in insertion order.

## Maintenance

### `erase()`

Deletes a record by id and returns whether it existed.

### `info()`

Returns `VaultInfo{count, dimension, metric}`.

### `rebuild_index()`

Drops the current index instance, rebuilds it from stored records, and
rebuilds the metadata accelerator if enabled.

## Search Results

`SearchResult` now contains:

- `id`
- `distance`
- `data`
- `document`
- `chunk`
- `lineage`

That data is hydrated directly from the authoritative record store, so fetch,
scan, vector seek, text seek, hybrid seek, and EQL-backed reads can expose the
same record context.
