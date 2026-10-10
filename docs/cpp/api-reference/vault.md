# API Reference: Collection

**Maturity:** stable (v1.0+).

`elips::Collection` is the document-aware record store and query surface for a
single named partition.

> Renamed in v1.1: the class was `Vault` (now a `[[deprecated]]` alias
> `using Vault = Collection`). Verbs were renamed — `place`→`add`,
> `place_document`→`add_document`, `place_many`→`add_many`, `seek`→`search`,
> `seek_text`→`search_text`, `seek_hybrid`→`hybrid_search`,
> `explain_seek`→`explain_query`, `fetch`→`get`, `erase`→`remove`,
> `quantize`→`train` — with the old names kept as deprecated shims. See
> `MIGRATION.md`.

## Construction

Users do not construct `Collection` directly. Obtain it from:

```cpp
auto& docs = db->collection("documents");
```

## Write Methods

### `add()`

```cpp
RecordID add(const Vector& vector,
             Payload payload = {},
             std::optional<RecordID> id = std::nullopt,
             std::optional<DocumentAttachment> document = std::nullopt,
             std::optional<ChunkInfo> chunk = std::nullopt,
             std::optional<EmbeddingLineage> lineage = std::nullopt);
```

Stores an explicit vector and optional document/lineage metadata.

### `add_document()`

```cpp
RecordID add_document(std::string text,
                      Payload payload = {},
                      std::optional<RecordID> id = std::nullopt,
                      std::optional<ChunkInfo> chunk = std::nullopt,
                      std::optional<EmbeddingLineage> lineage = std::nullopt);
```

Embeds the text through the configured `TextEmbedderPort`, stores the source
document, fills chunk defaults, and auto-generates lineage when omitted.

### `add_many()`

```cpp
void add_many(const std::vector<Record>& records);
```

Each `Record` may include `document`, `chunk`, and `lineage`.

### `place_record()`

```cpp
void place_record(const Record& record);
```

Load path — stores a record as-is, bypassing normalization and encoding when it
already carries a code. Used by snapshot, segment, and WAL replay so a persisted
code is restored verbatim rather than decoded and re-encoded. Not a general-use
write method.

## Query Methods

### `search()`

```cpp
std::vector<SearchResult> search(const Vector& query,
                                 std::size_t top,
                                 const Filter& filter = Filter{},
                                 std::optional<float> threshold = std::nullopt) const;
```

Vector similarity search.

### `search_text()`

```cpp
std::vector<SearchResult> search_text(std::string_view text,
                                      std::size_t top,
                                      const Filter& filter = Filter{},
                                      std::optional<float> threshold = std::nullopt) const;
```

Text-first query surface. Uses the native text embedder when configured,
otherwise raises `ConfigError` with guidance to configure or reattach a text
embedder. Use `hybrid_search()` when you already have the query vector and want
to blend it with lexical overlap from attached documents.

### `hybrid_search()`

```cpp
std::vector<SearchResult> hybrid_search(const Vector& query,
                                        std::string_view text,
                                        std::size_t top,
                                        const Filter& filter = Filter{},
                                        std::optional<float> threshold = std::nullopt,
                                        float lexical_weight = 0.25F) const;
```

Blends vector distance with lexical overlap from attached documents.

### `explain_query()`

```cpp
QueryPlan explain_query(const Vector& query,
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

### `get()`

```cpp
std::optional<Record> get(const RecordID& id) const;
```

Returns the full stored `Record`, including `document`, `chunk`, and `lineage`.
Returns `std::nullopt` if the record does not exist.

### `scan()`

```cpp
std::vector<Record> scan(const Filter& filter = Filter{},
                         std::size_t offset = 0,
                         std::size_t limit = std::numeric_limits<std::size_t>::max()) const;
```

Metadata scan in insertion order.

### `records()`

```cpp
[[nodiscard]] std::map<RecordID, Record> records() const;
```

Returns a snapshot copy of all stored records. The copy is taken under the
collection's shared mutex, so it is safe to read even while another thread is
inserting. **Prefer `scan()` with a filter and limit for large collections** —
`records()` copies the entire record set.

## Maintenance

### `remove()`

Deletes a record by id and returns whether it existed (`bool`). Increments
`pending_removals`. The tombstone is not reclaimed until `vacuum()` is called.

### `train()` / `is_trained()` / `codec()`

```cpp
void train();
[[nodiscard]] bool is_trained() const noexcept;
[[nodiscard]] quant::CodecId codec() const noexcept;
```

`train()` learns a quantization codebook over the resident vectors and
compresses the collection in place, freeing the fp32 copies. Requires a codec to
have been selected in the config; throws `ConfigError` otherwise, or if the
collection is empty or already quantized. `is_trained()` reports whether a
codebook has been attached; `codec()` names it. (Replaces the deprecated
`quantize()`/`quantized()`.)

### `vacuum()`

```cpp
void vacuum();
```

Reclaims index space held by deleted records. For `IndexHNSW`, rebuilds the
graph from live records only, releasing all tombstoned nodes and edges. For
`IndexFlat`, compacts the vector array. A no-op when `pending_removals() == 0`.
Safe to call at any time; automatically triggered when tombstones exceed
`HnswParams::compaction_ratio`.

### `pending_removals()`

```cpp
[[nodiscard]] std::size_t pending_removals() const noexcept;
```

Number of records deleted from the index but not yet reclaimed by `vacuum()`.
Use this to monitor tombstone pressure.

### `read_only()` / `set_read_only()`

```cpp
[[nodiscard]] bool read_only() const noexcept;
void set_read_only(bool read_only) noexcept;
```

`read_only()`: Returns `true` when the collection refuses mutations. Set by
`Database` when opened with `AccessMode::read_only`, or via `set_read_only()`.
Write operations on a read-only collection throw `StorageError`.

### `sealed()`

```cpp
[[nodiscard]] bool sealed() const noexcept;
```

Returns `true` once the owning `Database` has been closed. Mutations on a sealed
collection throw `StorageError` immediately — they would mutate in-memory state
that can never be persisted. Reads still succeed.

### `info()`

Returns `CollectionInfo{count, dimension, metric, codec, code_bytes}`.

### `rebuild_index()`

Drops the current index instance, rebuilds it from stored records, and
rebuilds the metadata accelerator if enabled.

## Search Results

`SearchResult` contains:

- `id`
- `distance`
- `data`
- `document`
- `chunk`
- `lineage`
- `codec` (and `approximate()` — true when the result came from a quantized
  collection)

That data is hydrated directly from the authoritative record store, so get,
scan, vector search, text search, hybrid search, and EQL-backed reads can expose
the same record context.

## Thread Safety

Each `Collection` owns a `std::shared_mutex`. Multiple threads may call read
methods (`search`, `search_text`, `hybrid_search`, `get`, `scan`, `records`,
`explain_query`, `info`) concurrently. Write methods (`add`, `add_document`,
`add_many`, `remove`, `rebuild_index`, `vacuum`, `train`, `set_read_only`) take
an exclusive lock. A committed transaction holds the instance-level lock for the
whole batch, so concurrent commits serialize.

