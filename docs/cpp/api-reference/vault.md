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

### `records()`

```cpp
[[nodiscard]] std::vector<StoredRecord> records() const;
```

Returns a snapshot copy of all stored records. The copy is taken under the vault's shared mutex, so it is safe to read even while another thread is inserting. **Prefer `scan()` with a filter and limit for large vaults** — `records()` copies the entire record set.

## Maintenance

### `erase()`

Deletes a record by id and returns whether it existed. Increments `pending_removals`. The tombstone is not reclaimed until `vacuum()` is called.

### `vacuum()`

```cpp
// Vault maintenance — v1.1.0

void vacuum();
```
Reclaims index space held by deleted records. For `HierarchicalGraphIndex`, rebuilds the graph from live records only, releasing all tombstoned nodes and edges. For `ExactIndex`, compacts the vector array. A no-op when `pending_removals() == 0`. Safe to call at any time; automatically triggered when tombstones exceed `GraphParams::compaction_ratio`.

### `pending_removals()`

```cpp
[[nodiscard]] std::size_t pending_removals() const noexcept;
```
Number of records deleted from the index but not yet reclaimed by `vacuum()`. Use this to monitor tombstone pressure.

### `read_only()` / `set_read_only()`

```cpp
[[nodiscard]] bool read_only() const noexcept;
void set_read_only(bool read_only);
```
`read_only()`: Returns `true` when the vault refuses mutations. Set by `ElipsInstance` when opened with `AccessMode::read_only`, or via `set_read_only()`. Write operations on a read-only vault throw `StorageError`.

### `sealed()`

```cpp
[[nodiscard]] bool sealed() const noexcept;
```
Returns `true` once the owning `ElipsInstance` has been closed. Mutations on a sealed vault throw `StorageError` immediately — they would mutate in-memory state that can never be persisted. Reads still succeed.

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

## Thread Safety

Each `Vault` owns a `std::shared_mutex`. Multiple threads may call read methods (`seek`, `seek_text`, `seek_hybrid`, `fetch`, `scan`, `records`, `explain_seek`, `info`) concurrently. Write methods (`place`, `place_document`, `place_many`, `erase`, `rebuild_index`, `vacuum`, `set_read_only`) take an exclusive lock. A committed transaction holds the instance-level lock for the whole batch, so concurrent commits serialize.

