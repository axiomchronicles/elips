# Architecture: Layers

ELIPS is structured into seven layers, each with well-defined responsibilities. Layers communicate through narrow, typed interfaces.

```
┌─────────────────────────────────────────┐
│ SDK (Database, Collection, Transaction)  │  user-facing API
├─────────────────────────────────────────┤
│ Query Engine (EQL parser, executor)      │  text query → operations
├─────────────────────────────────────────┤
│ Metadata (Filter predicate tree)         │  payload matching
├──────────────┬──────────────────────────┤
│ Index Engine │ Vector Engine             │  search + similarity
│              │ (distance, SIMD)          │
├──────────────┴──────────────────────────┤
│ Domain (Vector, RecordID, Record, etc.)  │  value types, errors
├─────────────────────────────────────────┤
│ Storage (WAL, snapshot, serialization)   │  durability
├─────────────────────────────────────────┤
│ Kernel (LockManager)                     │  cross-process concurrency
└─────────────────────────────────────────┘
```

---

## Domain Layer

The domain layer defines the value types and error hierarchy used everywhere else. All types are in namespace `elips`.

### Vector

`elips/domain/Vector.hpp:14`

An owned `float32` vector with value semantics. Not a raw float array — the class enforces a precise interface.

```cpp
class Vector {
public:
    Vector();
    explicit Vector(std::vector<float> values);
    std::span<const float> values() const noexcept;
    std::uint16_t dimension() const noexcept;
    bool empty() const noexcept;
    float magnitude() const noexcept;
    Vector normalized() const;
};
```

- **`Vector(std::vector<float>)`**: Takes ownership of the float data.
- **`values()`**: Returns a read-only span over the raw floats.
- **`dimension()`**: Returns the count of elements as `uint16_t`.
- **`empty()`**: True if `values_` is empty.
- **`magnitude()`**: Computes the Euclidean (L2) norm: `sqrt(sum(v_i^2))`.
- **`normalized()`**: Returns an L2-normalized copy (`v / ||v||`). Zero vectors are returned unchanged (no-op).

### RecordID (UUIDv7)

`elips/domain/RecordID.hpp:13`

A 128-bit value type for stable, time-ordered record identity.

```cpp
class RecordID {
public:
    using Bytes = std::array<std::uint8_t, 16>;

    RecordID();
    explicit RecordID(Bytes bytes);
    const Bytes& bytes() const noexcept;
    std::string to_string() const;

    static RecordID generate();
    static RecordID from_string(const std::string& text);

    bool operator==(const RecordID&) const = default;
    auto operator<=>(const RecordID&) const = default;
};
```

**Byte-level format (UUIDv7):**

| Bytes | Content |
|-------|---------|
| 0–5   | 48-bit big-endian millisecond Unix timestamp (time-ordered prefix) |
| 6     | Version nibble: `0x7` OR'd with lower 4 random bits |
| 7     | Random byte |
| 8     | Variant nibble: `0x8` OR'd with lower 6 random bits |
| 9–15  | Random bytes |

The time prefix is big-endian, meaning **lexicographic byte order equals temporal order** — `operator<=>` is `=default` and correctly sorts by time.

**`generate()`**: Creates a new UUIDv7 with a monotonic-by-time prefix. Uses `std::chrono::system_clock` for the millisecond timestamp and `std::mt19937_64` seeded from `std::random_device` for the random portion.

**`to_string()`**: Returns the canonical 8-4-4-4-12 hyphenated form (e.g., `"01931d0a-1234-7abc-8000-abcdef123456"`).

**`from_string(text)`**: Parses the hyphenated form (hyphens stripped, 32 hex digits required). Throws `ConfigError` on invalid input.

**Hashing**: Specialization of `std::hash<RecordID>` uses FNV-1a hashing over the 16 bytes.

### Record

`elips/domain/Record.hpp:21`

```cpp
struct Record {
    RecordID id;
    Vector vector;
    Payload payload;
};
```

A record is a vector with identity and metadata. Simple aggregate — no invariants beyond those of its fields.

### Payload and MetaValue

`elips/domain/Record.hpp:15-18`

```cpp
using MetaValue = std::variant<std::int64_t, double, bool, std::string>;
using Payload = std::map<std::string, MetaValue>;
```

**MetaValue** is a tagged union supporting four types:
- `std::int64_t` (variant index 0)
- `double` (variant index 1)
- `bool` (variant index 2)
- `std::string` (variant index 3)

Dynamic schema — no upfront field declaration. Keys are arbitrary strings. The variant index is stable and used in serialization.

### SearchResult

`elips/domain/SearchResult.hpp:11`

```cpp
struct SearchResult {
    RecordID id;
    float distance{0.0F};
    Payload data;
    std::optional<DocumentAttachment> document;
    std::optional<ChunkInfo> chunk;
    std::optional<EmbeddingLineage> lineage;
    quant::CodecId codec{quant::CodecId::none};  // approximate() when != none
};
```

One result from `search()`. Distance is ordering-normalized: **smaller means more similar** for all metrics (cosine = 1 - dot, euclidean = sqrt(L2), dot_product = -dot). Results are sorted ascending by `distance`. In a quantized collection `distance` is approximate; `codec` names the responsible quantizer.

### CollectionInfo

`elips/elips.hpp`

```cpp
struct CollectionInfo {
    std::size_t count{0};
    std::uint16_t dimension{0};
    Metric metric{Metric::cosine};
    quant::CodecId codec{quant::CodecId::none};  // none until trained
    std::size_t code_bytes{0};                   // stored code width
};
```

Summary statistics returned by `Collection::info()`: record count, configured
dimension, similarity metric, and compression state.

### Error Hierarchy

`elips/domain/Errors.hpp`

```
std::runtime_error
 └── ElipsError
      ├── DimensionMismatch        (alias DimensionMismatchError)
      ├── InvalidVector
      ├── ConfigError
      │    └── NotTrainedError      (IVF used before train())
      ├── NotFound
      ├── StorageError
      ├── BackendUnavailableError
      ├── LockConflict             (in elips/kernel/LockManager.hpp)
      └── ParseError               (in elips/query_engine/EQLParser.hpp)
```

All exceptions carry a human-readable `what()` message via the `std::runtime_error` base.

| Exception | Thrown When |
|-----------|-------------|
| `DimensionMismatch` | Vector dimension doesn't match the database/collection config |
| `InvalidVector` | Vector contains NaN or infinity |
| `ConfigError` | Configuration conflicts with persisted identity, invalid RecordID hex, unknown metric name |
| `NotTrainedError` | An IVF-family index was used (`add`/`search`) before `train()` (derives `ConfigError`) |
| `NotFound` | Record/collection does not exist; EQL references an unbound variable (`$name`) |
| `StorageError` | IO failure: corrupt snapshot, can't open/truncate WAL, write to a read-only open |
| `BackendUnavailableError` | A required GPU backend was unavailable and no CPU fallback was permitted |
| `LockConflict` | Another writer holds `flock(LOCK_EX)` on the database directory |
| `ParseError` | Malformed EQL query |

---

## Vector Engine

`elips/vector_engine/Metrics.hpp`

The vector engine computes similarity distances and dispatches to SIMD kernels. All functions in namespace `elips`.

### distance()

```cpp
float distance(Metric metric, std::span<const float> a,
               std::span<const float> b) noexcept;
```

Computes ordering-normalized distance between two vectors:

| Metric | Formula | Notes |
|--------|---------|-------|
| `cosine` | `1.0 - dot(a, b)` | Assumes inputs are L2-normalized at ingest |
| `euclidean` | `sqrt(sum((a_i - b_i)^2))` | Standard L2 distance |
| `dot_product` | `-dot(a, b)` | Negated so larger dot sorts first |

All return values follow the convention: **smaller = more similar**. Callers always sort ascending.

### requires_normalization()

```cpp
bool requires_normalization(Metric metric) noexcept;
```

Returns `true` only for `Metric::cosine`. When true, ingested and query vectors are L2-normalized before insertion and search.

### to_string() / metric_from_string()

```cpp
std::string_view to_string(Metric metric) noexcept;
Metric metric_from_string(std::string_view name);
```

String conversions: `"cosine"`, `"euclidean"`, `"dot_product"`. `metric_from_string` throws `ConfigError` for unknown names.

### NEON SIMD Dispatch

The kernel dispatch uses a `Dispatch` struct constructed once (static local). On ARM platforms (`__ARM_NEON` defined), NEON intrinsics are selected; otherwise scalar fallback:

#### NEON dot product (`dot_neon`)
- 4-wide `float32x4_t` accumulator with FMA (`vmlaq_f32`)
- Horizontal reduction via `vaddvq_f32`
- Scalar tail for remainder elements

#### NEON squared L2 (`sql2_neon`)
- 4-wide difference (`vsubq_f32`) followed by FMA (`vmlaq_f32`) of difference squared
- Same horizontal reduction and tail pattern

#### Scalar fallback
- Simple loop: `sum += a[i] * b[i]` for dot, `sum += (a[i] - b[i])^2` for squared L2

The dispatch is decided at compile time by `#if defined(__ARM_NEON)`. A future runtime CPUID dispatch on x86-64 will select AVX2/AVX-512 kernels.

---

## Index Engine

The index engine provides pluggable nearest-neighbor search behind a common interface.

### Index (Abstract Interface)

`elips/index_engine/Index.hpp`

```cpp
class Index {
public:
    using Hit = std::pair<RecordID, float>;  // (id, ordering-normalized distance)

    virtual void train(std::span<const float> training_vectors, std::size_t n);  // no-op by default
    [[nodiscard]] virtual bool is_trained() const noexcept;                      // true by default
    virtual void reset();

    virtual void add_with_ids(const RecordID& id, std::span<const float> vector) = 0;
    virtual void remove(const RecordID& id) = 0;
    [[nodiscard]] virtual std::vector<Hit> search(std::span<const float> query,
                                                  std::size_t k) const = 0;
    [[nodiscard]] virtual std::vector<Hit> range_search(std::span<const float> query,
                                                        float radius) const;  // throws by default
    [[nodiscard]] virtual std::size_t ntotal() const noexcept = 0;
    [[nodiscard]] virtual std::string_view type_name() const noexcept = 0;

    virtual void vacuum();
    [[nodiscard]] virtual std::size_t pending_removals() const noexcept;
};
```

Non-copyable, non-movable. Vectors passed to `add_with_ids()` are pre-validated
and pre-normalized by the SDK layer. `search()` results are sorted ascending by
distance; `k` is an upper bound. Flat/HNSW indexes keep the no-op `train()`;
IVF-family indexes override it to learn coarse centroids first. `range_search()`
is supported by the flat/IVF-flat indexes and throws `ElipsError` on index types
that cannot answer it (ADC-only IVFPQ/SQ, graph HNSW). `IndexPort` remains as a
deprecated alias of `Index`.

### IndexFlat

`elips/index_engine/IndexFlat.hpp`

Brute-force linear scan. Row-major storage: `data_[i * dimension_ + j]` for vector `i`, dimension `j`. Used for:

- Small collections (guaranteed exact results)
- Ground-truth oracle for ANN benchmark recall measurements
- The index built for `IndexType::exact`

Constructor: `IndexFlat(Metric metric, std::uint16_t dimension, quant::QuantizerPtr quantizer = {})`

- `add_with_ids()`: Appends ID to `ids_` and the vector (or its code, when a quantizer is attached) to storage.
- `remove()`: Linear scan to find and erase the ID, with the corresponding row removed.
- `search()` / `range_search()`: Iterate all rows, compute `distance(metric_, query, row)`, collect top-k (or every row within `radius`).
- `type_name()`: `"exact"`, or `"exact_quantized"` when a quantizer is attached.

### IndexHNSW (HNSW)

`elips/index_engine/IndexHNSW.hpp`

The primary ANN index. Implements the classic HNSW algorithm from scratch.

**Constructor**: `IndexHNSW(Metric metric, std::uint16_t dimension, HnswParams params, quant::QuantizerPtr quantizer = {})`

- `level_mult_` = `1.0 / ln(max_connections)`, the normalization factor for level assignment.
- RNG seeded from `std::random_device`.
- With a trained quantizer attached, node vectors are stored as codes and every distance (traversal and query) is computed by asymmetric lookup.

#### Data Structures

- `data_`: Row-major float storage (empty when compressed), `dimension_` floats per node.
- `codes_`: Row-major code storage (used when compressed).
- `ids_`: `RecordID` per node (index = NodeId).
- `deleted_`: Boolean tombstone per node (soft delete).
- `node_levels_`: Integer level per node.
- `links_[node][level]`: Neighbor NodeId list per node per level.
- `id_to_node_`: `unordered_map` from `RecordID` to `NodeId`.
- `entry_point_`: The top-level entry node for search.
- `max_level_`: The highest level currently in the graph.

#### Level Assignment (`random_level()`)

Standard HNSW level assignment: `floor(-ln(uniform(0,1)) * mL)`. Uses `level_mult_ = 1/ln(M)`.

#### Beam Search (`search_layer()`)

Beam search within one graph layer. Maintains two heaps:

- **Min-heap** (candidates): Frontier sorted by distance (closest first to expand).
- **Max-heap** (result): `ef` closest nodes found so far (farthest on top for easy eviction).

Algorithm: Extract closest candidate, visit its neighbors at the given level, add unseen neighbors if they could improve the result set. Terminates when the closest candidate is farther than the farthest result and the result set is full.

Returns results sorted ascending by distance.

#### Add (`add_with_ids()`)

1. Assign a random level.
2. Append to `ids_`, `deleted_`, `node_levels_`, `data_`/`codes_`, initialize `links_`.
3. If the graph is empty, set as entry point.
4. Greedy descent from entry through levels above the new node's level (1-nearest search per layer).
5. For each level from `min(level, max_level_)` down to 0:
   - Beam search with `ef_construction` neighbors.
   - Call `connect()` with diversity heuristic.
6. If the new level exceeds `max_level_`, update the entry point.

#### Connect (`connect()`)

HNSW Algorithm 4 (diversity heuristic / pruning):

For each candidate (sorted by distance from the new node): keep it only if it is **closer to the new node than to any already-selected neighbor**. This produces better-connected graphs than simply taking the top M by distance.

Reciprocal edges are also added, pruning each neighbor's list to `max_connections` by keeping the closest.

The base layer (level 0) uses `max_connections * 2` as the link budget.

#### Remove (`remove()`) and Vacuum (`vacuum()`)

Soft tombstone: sets `deleted_[node] = true`, increments `deleted_count_`. Graph structure is preserved; deleted nodes act as navigation bridges but are excluded from result sets. `pending_removals()` reports the tombstone count; once it passes `HnswParams::compaction_ratio` the index rebuilds itself, and `vacuum()` forces a rebuild on demand.

#### Search (`search()`)

1. Greedy descent from entry point through levels above 0 (1-nearest per layer).
2. Beam search on level 0 with `ef = max(ef_search, k)`, widened in proportion to the live/deleted ratio.
3. Filter out deleted nodes, return up to `k` hits.

#### type_name()

`"graph"`, or `"graph_quantized"` when a quantizer is attached.

### CPU IVF Family

`elips/index_engine/IndexIVFFlat.hpp`, `IndexIVFPQ.hpp`, `IndexIVFScalarQuantizer.hpp`

FAISS-aligned inverted-file indexes. A coarse quantizer of `nlist` k-means
centroids (learned in `train()`) partitions the space; `search()` probes only
the `nprobe` lists nearest the query (public, mutable, FAISS default 1, clamped
to `[1, nlist]`). All require `train()` before `add_with_ids`/`search` and raise
`NotTrainedError` otherwise.

| Type | Posting storage | `type_name()` | Notes |
|------|-----------------|---------------|-------|
| `IndexIVFFlat` | full fp32 | `"ivfflat"` | exact per-list scan; matches `IndexFlat` as `nprobe → nlist`. Supports `range_search`. |
| `IndexIVFPQ` | PQ code of the residual | `"ivfpq"` | `pq_dim` subspaces, `pq_bits` (4–8, default 8); ADC/LUT scoring, lossy. |
| `IndexIVFScalarQuantizer` | SQ8 code of the raw vector | `"ivfsq"` | flat 4× compression, high recall. |

### IndexFactory

`elips/index_engine/IndexFactory.hpp`

```cpp
std::unique_ptr<Index> make_index(const Config& config, std::uint16_t dimension);
std::unique_ptr<Index> index_factory(std::uint16_t dimension, Metric metric,
                                     std::string_view spec);
```

`make_index` switches on `config.index()`:

- `IndexType::exact` → `std::make_unique<IndexFlat>(config.metric(), dimension)`
- `IndexType::graph` → `std::make_unique<IndexHNSW>(config.metric(), dimension, config.graph_params())`

`index_factory` builds from a FAISS-style spec string — `"Flat"`,
`"IVF<nlist>,Flat"`, `"IVF<nlist>,PQ<M>"`, `"IVF<nlist>,SQ8"`, `"HNSW<M>"` —
throwing `ConfigError` on a malformed spec.

DIP (Dependency Inversion Principle): callers depend only on `Index`, never on concrete index types.

---

## Metadata Layer

`elips/metadata/Filter.hpp`

### Filter Predicate Tree

A `Filter` is a boolean predicate over a `Payload` (record metadata). A default-constructed `Filter` matches everything.

#### Node Structure

Internal tree node:

```
Kind: cmp | in | contains | conj | disj | neg | none
```

| Kind | Fields Used | Meaning |
|------|------------|---------|
| `cmp` | `field`, `cmp`, `value` | Compare `payload[field]` against `value` using `cmp` (eq, ne, lt, le, gt, ge) |
| `in` | `field`, `set` | Check if `payload[field]` is in `set` |
| `contains` | `field`, `value` (string) | Check if `payload[field]` (must be string) contains substring `value` |
| `conj` | `a`, `b` (children) | AND: `eval(a) && eval(b)` |
| `disj` | `a`, `b` (children) | OR: `eval(a) || eval(b)` |
| `neg` | `a` (child) | NOT: `!eval(a)` |
| `none` | none | Always false (for `NOT(match-all)`) |

#### compare_values()

Cross-type numeric comparison: `int64_t` and `double` are compared numerically (both promoted to `double`). `bool` compares within type. `string` compares within type. Incompatible type pairs return `std::nullopt` (evaluated as "not a match").

#### Fluent Builder API

Chained predicates are AND-ed together. Each `field()` sets the current target field; subsequent comparator/set/contains calls create a leaf node AND-ed onto the root:

```cpp
Filter f;
f.field("year").ge(2023).field("tag").one_of({"a", "b"});
// => (year >= 2023) AND (tag IN ["a", "b"])
```

#### Leaf Factories (used by EQL executor)

```cpp
static Filter compare(std::string field, Comparator op, MetaValue value);
static Filter in_set(std::string field, std::vector<MetaValue> values);
static Filter has_substring(std::string field, std::string substring);
```

#### Combinators

```cpp
Filter and_(const Filter& other) const;  // conj
Filter or_(const Filter& other) const;   // disj
static Filter not_(const Filter& inner); // neg
```

`or_()` with a match-all filter returns match-all. `not_(match-all)` returns a `none` node (never matches).

#### Evaluation

`matches(payload)` → walks the tree recursively, returning bool. `matches_all()` → `root_ == nullptr` (no filter tree = pass everything).

---

## Query Engine

`elips/query_engine/`

The EQL (ELIPS Query Language) engine provides a text-based query interface. EQL is tokenized, parsed into an AST, and executed against a `Database`.

### EQL Lexer (`EQLLexer.hpp`)

`elips::eql::tokenize(std::string_view source)` → `std::vector<Token>`

Token kinds:

| Kind | Description | Fields |
|------|-------------|--------|
| `word` | Identifier or keyword (alphanumeric + underscore) | `text` |
| `number` | Integer or floating-point literal | `text`, `number`, `is_integer` |
| `string` | Double-quoted string literal | `text` (body without quotes) |
| `punct` | Single or double-character operator/bracket: `=`, `!=`, `<`, `<=`, `>`, `>=`, `[`, `]`, `{`, `}`, `(`, `)`, `:`, `,`, `$` | `text` |
| `end` | End-of-stream sentinel | none |

Tokenization skips whitespace and `#`-prefixed line comments. Two-character operators (`<=`, `>=`, `!=`, `==`) are merged. Unterminated string literals throw `ElipsError`.

### EQL Parser (`EQLParser.hpp`)

`elips::eql::parse(std::string_view source)` → `Statement`

A recursive-descent parser (`Parser` class) over the flat token stream. Grammar:

```
statement   = search_stmt | get_stmt | scan_stmt | add_stmt | delete_stmt

search_stmt = "search" "in" identifier "nearest" vector_ref
              ["k" number] ["threshold" number] ["where" filter]
              ["rank_by" identifier] ["project" projection] "yield"

get_stmt    = "get" "from" identifier "id" string "yield"

scan_stmt   = "scan" "in" identifier
              ["where" filter] ["offset" number] ["limit" number] "yield"

add_stmt    = "add" "in" identifier "vector" vector_literal
              ["metadata" json_object]

delete_stmt = "delete" "from" identifier "id" string

vector_ref  = "$" identifier | vector_literal
vector_literal = "[" [number ("," number)*] "]"

filter     = or_expr
or_expr    = and_expr ("or" and_expr)*
and_expr   = not_expr ("and" not_expr)*
not_expr   = "not" not_expr | "(" or_expr ")" | comparison
comparison = identifier ("in" "[" [value ("," value)*] "]"
                        | "contains" string
                        | comparator value)

comparator = "=" | "!=" | "<" | "<=" | ">" | ">="

projection = "*" | identifier ("," identifier)*
json_object = "{" [string ":" value ("," string ":" value)*] "}"
```

The pre-1.1 verbs/keywords `seek`/`fetch`/`place`/`erase`, `top`, and `data` are
accepted as deprecated aliases of `search`/`get`/`add`/`delete`, `k`, and
`metadata`. Throws `ParseError` (inherits `ElipsError`) on syntax errors.

### AST Node Types (`AST.hpp`)

`elips::eql::Statement` = `std::variant<SearchStatement, GetStatement, ScanStatement, AddStatement, DeleteStatement>`

| Statement Type | Fields |
|---------------|--------|
| `SearchStatement` | `collection`, `query` (VectorRef), `k` (optional int), `threshold` (optional double), `where` (Filter), `rank_by` (optional string), `projection` (vector of string) |
| `GetStatement` | `collection`, `id` (string) |
| `ScanStatement` | `collection`, `where` (Filter), `offset` (optional int), `limit` (optional int) |
| `AddStatement` | `collection`, `vector` (float vector), `metadata` (Payload) |
| `DeleteStatement` | `collection`, `id` (string) |

`InsertStatement` remains as a deprecated alias of `AddStatement`. `VectorRef` has either a `literal` (inline float vector) or a `binding` (variable name, referencing the bindings map).

### Query Executor (`QueryExecutor.hpp`)

`elips::eql::execute(statement, db, bindings)` → `std::vector<SearchResult>`

Uses `std::visit` with an `Executor` struct to dispatch each statement type to the appropriate SDK method:

| Statement | Routing |
|-----------|---------|
| `SearchStatement` | `db.collection(s.collection).search(query, k, s.where, threshold)` → optionally re-ranked by `rank_by` field, projection applied |
| `GetStatement` | `db.collection(s.collection).get(RecordID::from_string(s.id))` → wrap result |
| `ScanStatement` | `db.collection(s.collection).scan(s.where, offset, limit)` → wrap records |
| `AddStatement` | `db.collection(s.collection).add(Vector{s.vector}, s.metadata)` → return new ID |
| `DeleteStatement` | `db.collection(s.collection).remove(RecordID::from_string(s.id))` → empty result |

`Database::query(eql_string, bindings)` wraps `parse()` + `execute()`:

```cpp
std::vector<SearchResult> Database::query(
    const std::string& eql, const std::map<std::string, Vector>& bindings) {
    return eql::execute(eql::parse(eql), *this, bindings);
}
```

Default `k` for search is 10. When `threshold` is set without explicit `k`, the fetch size is unbounded (100000).

**Rank-by**: When `rank_by` specifies a non-distance field, results are re-sorted via `std::stable_sort` using a metadata-aware comparator (`meta_less`). Numeric cross-type comparison is supported (`int64_t` and `double` compared numerically); strings compared lexicographically; mixed types compared by variant index.

**Projection**: When projection fields are specified, only those fields are kept in the result payloads (others removed).

### EQL Examples

```
search in documents nearest [0.9, 0.1, 0.0] k 5 where year >= 2023 yield

search in images nearest $query_vec k 20 threshold 0.5 yield

get from documents id "01931d0a-1234-7abc-8000-abcdef123456" yield

scan in documents where status = "active" offset 0 limit 50 yield

add in documents vector [0.1, 0.2, 0.3] metadata {"title": "new doc", "year": 2025}
```

---

## Storage Layer

### WAL (Write-Ahead Log)

`elips/storage/WAL.hpp`

Every mutation is appended and (optionally) flushed before acknowledgment, ensuring writes survive a crash before the next checkpoint.

**Op types**: `insert = 1`, `erase = 3`.

**On-disk format** (per entry):

| Field | Size | Description |
|-------|------|-------------|
| magic | 4 bytes | `0xE1105E01U` (same as snapshot magic) |
| op | 1 byte | `1` (insert) or `3` (erase) |
| collection | 4 + N bytes | Length-prefixed UTF-8 string (collection name) |
| id | 16 bytes | RecordID raw bytes (UUIDv7) |
| vector (insert only) | 2 + 4*N bytes | Length prefix (`uint16_t`) + float array |
| payload (insert only) | variable | Count (`uint32_t`) + `N` × (key length + string key + type tag byte + value) |
| CRC32C | 4 bytes | CRC32C (Castagnoli) of all preceding bytes in the record |

**Durability modes:**

| Mode | Behavior |
|------|----------|
| `paranoid` / `standard` | `sync_each_write = true` → `out_.flush()` after each append |
| `relaxed` | `sync_each_write = false` → buffered; flushed on close/checkpoint only |
| `ephemeral` | No WAL created (in-memory databases) |

**Replay** (`WAL::replay(path)`):

1. Read the entire log into memory.
2. Iterate record-by-record: read magic, parse header + body through a `std::istringstream` view, verify trailing CRC32C.
3. Stop at the first invalid record (truncated, corrupt magic, or CRC mismatch) — clean truncation, no partial apply.
4. Return valid entries for the recovery loop in `open()`.

**Reset** (`WAL::reset()`): Closes and reopens the log file with `trunc`, dropping all entries. Called after a successful checkpoint.

### Snapshot Format

`elips.snapshot` (binary):

| Field | Size | Description |
|-------|------|-------------|
| magic | 4 bytes | `0xE1105E01U` |
| version | 4 bytes | `1` |
| dimension | 2 bytes | `uint16_t` |
| metric | 1 byte | `uint8_t` enum |
| collection_count | 4 bytes | `uint32_t` |
| [per collection] | | |
|   name | 4 + N bytes | Length-prefixed string |
|   record_count | 4 bytes | `uint32_t` |
|   [per record] | | |
|     id | 16 bytes | RecordID raw bytes |
|     dim | 2 bytes | Vector dimension (`uint16_t`) |
|     vector | 4*dim bytes | Float array |
|     payload | variable | Same encoding as WAL payload |

**Atomic publish**: The snapshot is written to `elips.snapshot.tmp`, then `fs::rename()` atomically replaces the live snapshot. The WAL is then truncated.

### IDENTITY File

`IDENTITY` (binary):

| Field | Size | Description |
|-------|------|-------------|
| magic | 4 bytes | `0xE11D0001U` |
| version | 4 bytes | `1` |
| dimension | 2 bytes | `uint16_t` |
| metric | 1 byte | `uint8_t` enum |
| index_type | 1 byte | `uint8_t` enum |

Written on first `open()` and validated on subsequent opens.

### Serialization Primitives

`elips/storage/Serialization.hpp` (namespace `elips::detail` — internal, not public API)

- `put<T>(ostream, value)`: Binary write of trivially-copyable types.
- `get<T>(istream)` → `T`: Binary read.
- `put_string(ostream, s)`: `uint32_t` length prefix + char data.
- `get_string(istream)` → `string`: Read length then data.
- `put_payload(ostream, payload)`: Count (`uint32_t`) + per-entry (key string, type tag `uint8_t`, typed value).
- `get_payload(istream)` → `Payload`: Reverse.
- `crc32c(data, len)` → `uint32_t`: CRC32C (Castagnoli polynomial `0x82F63B78`) with static software table, used for WAL record integrity verification.

Byte order is **native** (single-machine embedded use; cross-platform normalization planned).

### Recovery Flow (`open()`)

1. If `:memory:`: create non-persistent instance (no WAL, no lock).
2. Create directory, acquire `flock(LOCK_EX | LOCK_NB)` via `LockManager`.
3. Read `IDENTITY` if exists, else write it.
4. Load snapshot (`elips.snapshot`) into collections if present.
5. Replay WAL (`wal.log`) entries on top of snapshot state.
6. Create WAL instance (unless durability is `ephemeral`).
7. Return ready `Database`.

---

## Kernel Layer

### LockManager

`elips/kernel/LockManager.hpp:19`

RAII advisory file lock enforcing single-writer / multi-reader contract across processes.

```cpp
class LockManager {
public:
    explicit LockManager(const std::string& lock_path);  // acquires exclusive
    ~LockManager();
    LockManager(LockManager&&) noexcept;     // move-only
};
```

**Constructor**: Opens `LOCK` file with `O_RDWR | O_CREAT`, then calls `flock(fd, LOCK_EX | LOCK_NB)`. If the lock is held by another process, `flock` returns immediately with `EWOULDBLOCK`, and a `LockConflict` exception is thrown (the fd is closed first).

**Destructor**: Calls `flock(fd, LOCK_UN)` then `close(fd)`.

**Move constructor**: Transfers fd ownership via `std::exchange`.

The lock is held for the lifetime of the `Database` and released on `close()` or destruction.

---

## SDK Layer

The SDK is the user-facing API. Main types:

- **`elips::open(path, config)`** → `unique_ptr<Database>`: Factory function. Handles the full recovery flow.
- **`Database`**: Database handle. Owns collections, WAL, lock. Checkpoints on destruction.
- **`Collection`**: Named partition of records. `add` / `add_document` / `search` / `search_text` / `hybrid_search` / `scan` / `get` / `remove` / `train` / `vacuum`.
- **`Transaction` + `TransactionCollection`**: Buffered atomic batch mutations (`add` / `remove`).

Pre-1.1 type names (`ElipsInstance`, `Vault`, `TransactionVault`) and verb names
(`place`/`seek`/`fetch`/`erase`) remain as deprecated aliases/shims for one minor
version. Detailed API references for each type are in the API Reference section.