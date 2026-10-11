# API Reference: Domain Types

**Maturity:** stable (v1.0+).

Namespace `elips`. The value types and error hierarchy used throughout ELIPS. All domain types are in `include/elips/domain/`.

> Renamed in v1.1: `VaultInfo`→`CollectionInfo`; `DimensionMismatch` gained the
> alias `DimensionMismatchError`; typed errors `NotTrainedError` and
> `BackendUnavailableError` were added. Old names remain as aliases. See
> `MIGRATION.md`.

---

## Vector

`include/elips/domain/Vector.hpp:14`

An owned `float32` vector with value semantics.

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

### Construction

| Constructor | Description |
|-------------|-------------|
| `Vector()` | Default: empty vector (dimension 0). |
| `Vector(std::vector<float> values)` | Takes ownership of float data. Empty vectors allowed. |

### Methods

#### values()

```cpp
std::span<const float> values() const noexcept;
```

Returns a read-only span over the underlying float array. Safe for concurrent reads.

#### dimension()

```cpp
std::uint16_t dimension() const noexcept;
```

Returns `static_cast<uint16_t>(values_.size())`. For an empty vector, returns 0.

#### empty()

```cpp
bool empty() const noexcept;
```

Returns `values_.empty()`. True for a default-constructed Vector.

#### magnitude()

```cpp
float magnitude() const noexcept;
```

Computes the Euclidean (L2) norm: `sqrt(sum(v_i * v_i))`. For an empty vector, returns `0.0f`.

#### normalized()

```cpp
Vector normalized() const;
```

Returns an L2-normalized copy of the vector (each element divided by the magnitude). **Zero vectors are returned unchanged** (magnitude check: if `mag == 0.0f`, returns `*this` by copy). This prevents division-by-zero silently.

Creates a new `std::vector<float>` internally, computes the normalized values, and returns a new `Vector`.

**Example**:

```cpp
elips::Vector v{{3.0F, 4.0F}};
float m = v.magnitude();     // 5.0
auto n = v.normalized();     // {0.6, 0.8}
float nm = n.magnitude();    // ~1.0
auto z = elips::Vector{{0.0F, 0.0F}};
auto zn = z.normalized();    // {0.0, 0.0} (unchanged)
```

---

## RecordID

`include/elips/domain/RecordID.hpp:13`

A 128-bit UUIDv7 value type. Stable, time-ordered record identity.

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

### Construction

| Constructor | Description |
|-------------|-------------|
| `RecordID()` | Default: all zero bytes. Acts as a sentinel for "no ID" (used in `add_many()` to detect records needing auto-generated IDs). |
| `RecordID(Bytes bytes)` | Constructs from a 16-byte array. |

### UUIDv7 Byte Format

| Bytes | Bits | Content |
|-------|------|---------|
| 0–5 | 48 | Big-endian millisecond Unix timestamp (time-ordered prefix) |
| 6 | 4+4 | Version nibble `0x7` (bits 48-51 of timestamp upper) OR'd with lower 4 random bits |
| 7 | 8 | Random byte |
| 8 | 2+6 | Variant nibble `0x8` OR'd with lower 6 random bits |
| 9 | 8 | Random byte |
| 10–15 | 48 | Additional random bytes |

The timestamp occupies the most significant bytes in big-endian order, so **lexicographic byte comparison equals temporal ordering**. `operator<=>` is `=default` and leverages this property.

### bytes()

```cpp
const Bytes& bytes() const noexcept;
```

Returns the raw 16-byte array. Used by serialization and hashing.

### to_string()

```cpp
std::string to_string() const;
```

Returns the canonical 8-4-4-4-12 hyphenated hex representation.

Format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` (36 characters including hyphens).

```
Bytes:  01 93 1d 0a  12 34  7a bc  80 00  ab cd ef 12 34 56
String: "01931d0a-1234-7abc-8000-abcdef123456"
```

The hyphens are placed after bytes 4, 6, 8, and 10, following RFC 9562 UUID string format.

### generate() (Static)

```cpp
static RecordID generate();
```

Generates a new UUIDv7 with the following properties:

- **Timestamp**: `std::chrono::system_clock::now()` converted to milliseconds since epoch. Written as a 48-bit big-endian value into bytes 0–5.
- **Random**: Two 64-bit values from `std::mt19937_64` seeded by `std::random_device`. 40 bits go to bytes 6–10, 40 bits to bytes 11–15.
- **Version 7**: Byte 6 has its upper nibble set to `0x7` (preserving lower 4 random bits).
- **Variant 10**: Byte 8 has its upper 2 bits set to `0b10` (preserving lower 6 random bits).

The RNG is `thread_local`, so `generate()` is safe to call from multiple threads.

**Collision probability**: With 80 random bits per ID and a well-seeded Mersenne Twister, collisions are astronomically unlikely in practice.

### from_string() (Static)

```cpp
static RecordID from_string(const std::string& text);
```

Parses a UUID string (with or without hyphens) into a `RecordID`.

**Input format**: Accepts the hyphenated form (`"01931d0a-1234-7abc-8000-abcdef123456"`) or bare hex (`"01931d0a12347abc8000abcdef123456"`). Hyphens are stripped before parsing.

**Validation**: Requires exactly 32 hex digits (after stripping hyphens). Each hex digit must be valid (`0-9`, `a-f`, `A-F`). Throws `ConfigError` on:
- Wrong number of hex digits
- Invalid hex characters

**Example**:

```cpp
auto id1 = RecordID::from_string("01931d0a-1234-7abc-8000-abcdef123456");
auto id2 = RecordID::from_string("01931d0a12347abc8000abcdef123456");  // same
assert(id1 == id2);

// throws ConfigError:
auto bad = RecordID::from_string("not-a-uuid");
auto bad2 = RecordID::from_string("01931d0a-1234-7abc-8000-abcdef12345");  // 31 digits
```

### Hashing

```cpp
template <>
struct std::hash<elips::RecordID> {
    std::size_t operator()(const elips::RecordID& id) const noexcept;
};
```

FNV-1a hash over the 16 bytes. Offset basis: `1469598103934665603`, prime: `1099511628211`. Used by `std::unordered_map` and internally by HNSW's `id_to_node_` map.

### Operators

- `operator==` / `operator!=`: Defaulted member-wise comparison.
- `operator<=>`: Defaulted three-way comparison. **Lexicographic byte order = UUIDv7 time order** due to big-endian timestamp prefix.

### Usage as Map Key

`RecordID` is used as the key in `std::map<RecordID, Record>` (the collection's record store). This gives O(log N) lookup ordered by insertion time, which also means `scan()` iterates in temporal order.

---

## Record

`include/elips/domain/Record.hpp:21`

```cpp
struct Record {
    RecordID id;
    Vector vector;
    Payload payload;
    std::optional<DocumentAttachment> document;
    std::optional<ChunkInfo> chunk;
    std::optional<EmbeddingLineage> lineage;
    std::vector<std::uint8_t> codes;        // compressed code (quantized collections)
    quant::CodecId codec{quant::CodecId::none};

    [[nodiscard]] bool approximate() const noexcept;  // true when codec != none
};
```

An identity (`RecordID`), an embedding (`Vector`), metadata (`Payload`), and
optional document/chunk/lineage attachments. In a quantized collection `codes`
holds the stored code and `codec` names the quantizer; `vector` then carries a
reconstruction and `approximate()` returns `true`. All members are publicly
accessible.

Used as:
- The value type in the collection's record store (`std::map<RecordID, Record>`)
- The return type of `Collection::scan()` and `Collection::get()`
- Input to `Collection::add_many()`

### DocumentAttachment, ChunkInfo, EmbeddingLineage

```cpp
struct DocumentAttachment { std::string text; std::string uri;
                            std::string mime_type{"text/plain"}; };
struct ChunkInfo { std::string document_key; std::uint32_t ordinal{0};
                   std::uint32_t char_start{0}; std::uint32_t char_end{0}; };
struct EmbeddingLineage { std::string provider; std::string model;
                          std::string revision; Payload attributes; };
```

Optional attachments carried by `Record` and `SearchResult` for document-aware
collections.

---

## Payload and MetaValue

`include/elips/domain/Record.hpp:15-18`

```cpp
using MetaValue = std::variant<std::int64_t, double, bool, std::string>;
using Payload = std::map<std::string, MetaValue>;
```

### MetaValue

A tagged union of four supported metadata types:

| Variant Index | C++ Type | Serialization Tag | Description |
|---------------|----------|-------------------|-------------|
| 0 | `std::int64_t` | `0` | Signed 64-bit integer |
| 1 | `double` | `1` | IEEE 754 double-precision float |
| 2 | `bool` | `2` | Boolean (true/false) |
| 3 | `std::string` | `3` | UTF-8 string |

The variant index is stable and used as the type tag in serialization (both snapshot and WAL).

Type conversions in filtering:
- **Numeric cross-type**: `int64_t` and `double` are promoted to `double` for comparison. `int64_t{5} == double{5.0}` evaluates to true in filter predicates.
- **No implicit conversions**: `bool` is not comparable to numeric types. `string` is not comparable to anything but `string`. Incompatible type pairs in filter comparisons return "not a match".

### Payload

A `std::map<std::string, MetaValue>` — ordered dictionary with string keys and `MetaValue` values. Dynamic schema: keys are arbitrary strings, values are any of the four `MetaValue` alternatives.

**Example**:

```cpp
Payload p = {
    {"title", std::string{"Hello World"}},
    {"year", std::int64_t{2024}},
    {"score", 0.95},
    {"active", true}
};

// accessing values via std::get
std::string title = std::get<std::string>(p.at("title"));
std::int64_t year = std::get<std::int64_t>(p.at("year"));
double score = std::get<double>(p.at("score"));
bool active = std::get<bool>(p.at("active"));

// accessing via std::visit
for (const auto& [key, value] : p) {
    std::visit([](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            std::cout << "string: " << v << '\n';
        } else {
            std::cout << "other: " << v << '\n';
        }
    }, value);
}
```

**Note**: `Payload` uses `std::map` so iteration order is lexicographic by key. Record store iteration is ordered by `RecordID` (time order).

---

## SearchResult

`include/elips/domain/SearchResult.hpp:11`

```cpp
struct SearchResult {
    RecordID id;
    float distance{0.0F};
    Payload data;
    std::optional<DocumentAttachment> document;
    std::optional<ChunkInfo> chunk;
    std::optional<EmbeddingLineage> lineage;
    quant::CodecId codec{quant::CodecId::none};

    [[nodiscard]] bool approximate() const noexcept;  // true when codec != none
};
```

Returned by `Collection::search()` and `Database::query()`. Contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `RecordID` | The matched record's identity |
| `distance` | `float` | Ordering-normalized distance from the query. **Smaller = more similar.** Cosine: `1 - dot`, Euclidean: `sqrt(L2)`, Dot product: `-dot`. |
| `data` | `Payload` | The record's full metadata payload. For EQL queries with projection, only projected fields are present. |
| `document` / `chunk` / `lineage` | optional | Attachments hydrated from the record store. |
| `codec` | `quant::CodecId` | Quantizer behind an approximate score; `approximate()` is `true` when a quantized collection produced the hit. |

Results from `search()` are sorted ascending by `distance`. When `rank_by` is
used in EQL, results are re-sorted by the named payload field.

---

## CollectionInfo

`include/elips/elips.hpp`

```cpp
struct CollectionInfo {
    std::size_t count{0};
    std::uint16_t dimension{0};
    Metric metric{Metric::cosine};
    quant::CodecId codec{quant::CodecId::none};
    std::size_t code_bytes{0};
};

using VaultInfo [[deprecated("renamed to CollectionInfo")]] = CollectionInfo;
```

Returned by `Collection::info()`. Summary statistics:

| Field | Description |
|-------|-------------|
| `count` | Number of records in the collection |
| `dimension` | Configured vector dimension |
| `metric` | Configured similarity metric |
| `codec` | Active compression codec (`none` until `train()` has run) |
| `code_bytes` | Width of one stored code, against `dimension * 4` uncompressed |

---

## ElipsError Hierarchy

All errors in `include/elips/domain/Errors.hpp` unless otherwise noted.

```
std::runtime_error
 └── elips::ElipsError           (domain/Errors.hpp)
      ├── DimensionMismatch      (domain/Errors.hpp)   [alias DimensionMismatchError]
      ├── InvalidVector          (domain/Errors.hpp)
      ├── ConfigError            (domain/Errors.hpp)
      │    └── NotTrainedError   (domain/Errors.hpp)
      ├── NotFound               (domain/Errors.hpp)
      ├── StorageError           (domain/Errors.hpp)
      ├── BackendUnavailableError(domain/Errors.hpp)
      ├── LockConflict           (kernel/LockManager.hpp)
      └── eql::ParseError        (query_engine/EQLParser.hpp)
```

### ElipsError (Base)

```cpp
class ElipsError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
```

Base for all ELIPS exceptions. Carries a human-readable `what()` message. All derived types simply inherit the `std::runtime_error` constructor via `using` declarations.

### DimensionMismatch

```cpp
class DimensionMismatch : public ElipsError {
public:
    using ElipsError::ElipsError;
};
using DimensionMismatchError = DimensionMismatch;  // *Error-suffixed alias
```

Thrown when:
- `Collection::prepare()` detects `vector.dimension() != config_.dimension()`
- `Transaction` enqueue detects a dimension mismatch during eager validation

Message example: `"vector dimension does not match collection"` or `"vector
dimension does not match database"`. `DimensionMismatchError` is an exact alias
(same type), mirroring the Python name — a throw of either is caught by either.

### InvalidVector

```cpp
class InvalidVector : public ElipsError {
public:
    using ElipsError::ElipsError;
};
```

Thrown when:
- `Collection::prepare()` detects NaN or Inf via `!std::isfinite(v)`
- `Transaction` enqueue detects NaN/Inf during eager validation

Message example: `"vector contains NaN or Inf"`.

### ConfigError

```cpp
class ConfigError : public ElipsError {
public:
    using ElipsError::ElipsError;
};
```

Thrown when:
- In-memory database opened without dimension
- New file database opened without dimension
- Configured dimension conflicts with persisted identity
- `RecordID::from_string()` receives malformed hex input
- `metric_from_string()` receives an unknown metric name
- an unrecognized/malformed `index_factory` spec is given

### NotTrainedError

```cpp
class NotTrainedError : public ConfigError {
public:
    using ConfigError::ConfigError;
};
```

Thrown when an IVF-family index (`IndexIVFFlat` / `IndexIVFPQ` /
`IndexIVFScalarQuantizer`) is used (`add_with_ids`/`search`) before `train()`
has learned its coarse quantizer. **Derives from `ConfigError`**, so existing
`catch (const ConfigError&)` handlers keep catching it; catch `NotTrainedError`
directly for the specific condition.

### NotFound

```cpp
class NotFound : public ElipsError {
public:
    using ElipsError::ElipsError;
};
```

Thrown when:
- EQL query references an unbound variable (`$name` not in bindings map)

Message example: `"EQL: unbound query vector $my_vec"`.

**Note**: `Collection::get()` returns `std::nullopt` rather than throwing on
missing records. `Collection::remove()` returns `false`.

### StorageError

```cpp
class StorageError : public ElipsError {
public:
    using ElipsError::ElipsError;
};
```

Thrown on persistence/IO failures:
- Can't open snapshot for reading or writing
- Corrupt `IDENTITY` file (wrong magic)
- Snapshot magic or version mismatch
- Truncated or corrupt snapshot during load
- Can't open/truncate WAL file
- WAL append fails (disk full or stream error)
- Unknown payload value type tag in deserialization (tag byte not 0-3)
- Can't open lock file

### BackendUnavailableError

```cpp
class BackendUnavailableError : public ElipsError {
public:
    using ElipsError::ElipsError;
};
```

Thrown when a required compute backend (e.g. a GPU device) was requested but is
unavailable and no CPU fallback was permitted — the GPU policy failure surfaces
as a typed error instead of a silent CPU demotion. Mirrors the Python
`BackendUnavailableError`.

### LockConflict

```cpp
class LockConflict : public ElipsError {
public:
    using ElipsError::ElipsError;
};
```

`include/elips/kernel/LockManager.hpp:11`

Thrown when `flock(fd, LOCK_EX | LOCK_NB)` returns non-zero (another writer holds the database directory lock). The lock file fd is closed before the exception is thrown.

### ParseError

```cpp
class ParseError : public ElipsError {
public:
    using ElipsError::ElipsError;
};
```

`include/elips/query_engine/EQLParser.hpp:12`

Thrown by `eql::parse()` on malformed EQL syntax. Message describes the expected token and what was found. Examples:

- `"EQL: expected a statement keyword, got 'foo'"`
- `"EQL: expected 'yield'"`
- `"EQL: unexpected token ',' in search"`
- `"EQL: unknown comparator '@'"`

### Catching Errors

```cpp
try {
    auto db = elips::open("./data", Config{}.dimension(768));
    auto& docs = db->collection("docs");
    docs.add(embedding, metadata);
} catch (const elips::DimensionMismatch& e) {
    // wrong vector dimension
} catch (const elips::InvalidVector& e) {
    // NaN/Inf in vector
} catch (const elips::NotTrainedError& e) {
    // IVF index used before train()
} catch (const elips::StorageError& e) {
    // disk / IO issue
} catch (const elips::ConfigError& e) {
    // misconfiguration (also catches NotTrainedError)
} catch (const elips::LockConflict& e) {
    // another process holds the lock
} catch (const elips::ElipsError& e) {
    // catch-all for any ELIPS error
}
```

All exceptions carry messages suitable for logging or user display through `e.what()`.