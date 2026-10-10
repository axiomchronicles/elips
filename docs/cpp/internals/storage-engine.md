# Storage Engine

The storage engine provides durability, crash recovery, and serialisation for
on-disk ELIPS databases. It combines a write-ahead log (WAL), periodic atomic
snapshots, advisory file locking, and binary serialisation primitives.

---

## Database Directory Layout

When a persistent database is opened at a directory path, four files may exist:

```
/path/to/db/
├── LOCK           advisory flock file (created on open, released on close)
├── IDENTITY       immutable configuration record (magic + dimension + metric + index type)
├── elips.snapshot  full database snapshot (created by checkpoint)
└── wal.log         write-ahead log (appended on every mutation)
```

In-memory databases (`path == ":memory:"`) create no files and skip all
persistence machinery.

---

## IDENTITY

**Purpose:** Records the database's dimension, metric, and index type so that
reopening does not require reconfiguration. It is the authoritative schema.
The snapshot file may also carry dimension/metric for historical reasons, but
the IDENTITY file takes precedence.

**Format (binary, native byte order):**

| Offset | Size | Field | Value |
|--------|------|-------|-------|
| 0 | 4 bytes | Magic | `0xE11D0001` (little-endian on-disk) |
| 4 | 4 bytes | Version | `1` (currently always 1) |
| 8 | 2 bytes | Dimension | `dimension` as uint16 |
| 10 | 1 byte | Metric | `0`=cosine, `1`=euclidean, `2`=dot_product |
| 11 | 1 byte | Index type | `0`=graph, `1`=exact |

**Operations:**

- **Write** (`write_identity()` in `Database.cpp`): Creates the file fresh
  with `std::ios::trunc`. Called only when a new (empty) database directory
  is opened.
- **Read** (`read_identity()`): Validates magic, reads version, dimension,
  metric, index. Returns an `Identity` struct. Called on every `open()` of
  an existing database.
- **Conflict resolution:** If the caller's `Config` specifies a non-zero
  dimension that differs from the persisted identity, `open()` throws
  `ConfigError`.

---

## Write-Ahead Log (WAL)

**Header:** `include/elips/storage/WAL.hpp`
**Source:** `src/WAL.cpp`

The WAL provides crash safety: every mutation (insert or erase) is
append-logged **before** in-memory state is mutated, ensuring that writes
survive a crash preceding the next checkpoint.

### Record Framing

Each WAL record has the form:

```
[body: variable length] [CRC32C: 4 bytes]
```

**Body encoding** (`encode_body()` in `WAL.cpp`):

| Field | Size | Encoding |
|-------|------|----------|
| Magic | 4 bytes | `0xE1105E01` (uint32) |
| Op code | 1 byte | `1` = insert, `3` = erase (plus `4` = insert_ex, `7` = quantized insert) |
| Collection name | var | uint32 length + UTF-8 bytes |
| Record ID | 16 bytes | Raw UUIDv7 bytes |
| **If insert:** | | |
| Dimension | 2 bytes | uint16 |
| Vector data | `dim × 4` bytes | Raw float32 array |
| Payload | var | uint32 count + key-value pairs |

**Erase records** omit the vector and payload fields entirely.

### Op Enum

```cpp
enum class Op : std::uint8_t {
    insert = 1, erase = 3, insert_ex = 4, insert_q = 7
};
```

The op codes are a **persisted on-disk format** and are deliberately not renamed
by the Phase-5 vocabulary rename (`insert`/`erase` here are the storage op
semantics, distinct from the public `add`/`remove` verbs). The values skip
2/5/6 intentionally — this is a common defensive pattern so that
accidental zero bytes (from uninitialised memory or disk corruption) are
unlikely to match a valid op code.

### CRC32C Integrity Check

Each record body is checksummed with **CRC32C (Castagnoli)** polynomial
`0x82F63B78`. The static 256-entry lookup table is constructed once at
compile time:

```cpp
inline std::uint32_t crc32c(const void* data, std::size_t len) {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t crc = i;
            for (int j = 0; j < 8; ++j)
                crc = (crc & 1U) ? (crc >> 1) ^ 0x82F63B78U : (crc >> 1);
            t[i] = crc;
        }
        return t;
    }();
    // ... standard CRC32C bytewise loop ...
}
```

CRC32C is hardware-accelerated on modern x86 (`crc32` instruction via SSE4.2)
and ARM (`crc32cb`/`crc32ch` instructions), but the software table fallback is
used here for portability. Future optimisation may add CPUID dispatch to the
hardware-accelerated path.

The CRC covers the entire body (magic through payload), not just the
variable-length portion. On append, the body is serialised, the CRC computed,
and both written to the file stream.

### Append Flow

```cpp
void WAL::append(const Entry& entry) {
    const std::string body = encode_body(entry);
    const std::uint32_t crc = detail::crc32c(body.data(), body.size());
    out_.write(body.data(), body.size());
    detail::put<std::uint32_t>(out_, crc);
    if (sync_each_write_) out_.flush();
    if (!out_) throw StorageError{"WAL append failed"};
}
```

The write-before-memory-mutation invariant is enforced at the collection level:

```cpp
RecordID Collection::add(const Vector& vector, Payload payload,
                         std::optional<RecordID> id) {
    Vector prepared = prepare(vector);
    const RecordID record_id = id.value_or(RecordID::generate());
    if (wal_ != nullptr)  // WAL FIRST
        wal_->append_insert(name_, record_id, prepared.values(), payload);
    index_->add_with_ids(record_id, prepared.values());   // then memory
    records_[record_id] = Record{record_id, std::move(prepared), std::move(payload)};
    return record_id;
}
```

### `sync_each_write` Flag

The `sync_each_write_` boolean controls calling `std::ofstream::flush()` after
each append:

- **`true`:** calls `flush()`, which hands the data to the OS kernel buffer.
  The kernel will write it to disk asynchronously. This is the default for
  `standard` and `paranoid` durability levels.
- **`false`:** no explicit flush. Data remains in the C++ iostream buffer and
  then the OS buffer. Used for `relaxed` durability (see below).

Note: `flush()` is a userspace-to-kernel handoff, not an fsync/fdatasync.
Full `paranoid` durability would additionally call `fsync()` — this is noted
as a future hardening item.

### WAL Replay

```cpp
static std::vector<Entry> WAL::replay(const std::filesystem::path& path);
```

On database open, after loading the snapshot, the WAL is replayed to recover
any writes that occurred after the last checkpoint. The replay process:

1. **Reads the entire WAL file into memory** as a byte blob.
2. **Frame-by-frame validation:**
   - Read 4-byte magic. If not `0xE1105E01`, stop immediately (corrupt or
     truncated tail).
   - Parse op, collection name, record ID, and optionally vector + payload via
     a temporary `std::istringstream`.
   - If the stream fails mid-parse, stop (truncated record).
   - Verify CRC32C. If mismatch, stop (corruption detected).
3. **Truncated tail is non-fatal:** Replay stops cleanly at the first invalid
   record. All records up to that point are successfully replayed. This
   handles the common crash-during-append scenario where the last record may
   be partially written.
4. **Returns** a vector of `Entry` structs, which the caller applies to
   collections in order.

The replay loop:

```cpp
while (pos < n) {
    const std::size_t record_start = pos;
    std::uint32_t magic = 0;
    if (!read_u32(magic) || magic != wal_magic) break;  // stop
    // ... parse ...
    if (!body) break;                                     // truncated
    if (record_start + body_len + 4 > n) break;           // CRC missing
    if (stored_crc != actual) break;                      // corrupt
    entries.push_back(std::move(entry));
    pos = record_start + body_len + 4;
}
```

### WAL Reset

After a checkpoint has durably captured the entire database state, the WAL
is truncated to zero length:

```cpp
void WAL::reset() {
    out_.close();
    out_.open(path_, std::ios::binary | std::ios::trunc);
}
```

---

## Serialization Primitives

**Header:** `include/elips/storage/Serialization.hpp`

All binary serialisation lives in `namespace elips::detail` and is shared by
the WAL and snapshot subsystems. **Native byte order** is used (no network
byte order conversion) since ELIPS targets single-machine embedded use.

### `put()` / `get()` — Fixed-width Types

```cpp
template <typename T>
void put(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T get(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}
```

Trivially copyable types (integers, floats) are written/read as raw bytes.
Used for: `uint16_t`, `uint32_t`, `uint8_t`, `int64_t`, `double`, `bool`.

### `put_string()` / `get_string()` — Variable-length Strings

```cpp
inline void put_string(std::ostream& out, const std::string& s) {
    put<std::uint32_t>(out, static_cast<std::uint32_t>(s.size()));
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}
```

Length-prefixed (uint32) followed by raw bytes.

### `put_payload()` / `get_payload()` — Metadata Payloads

A `Payload` is a `std::map<std::string, MetaValue>` where `MetaValue` is a
`std::variant<int64_t, double, bool, std::string>`.

**Encoding:**
1. `uint32_t` count of key-value pairs.
2. For each pair:
   - `put_string(key)`
   - `uint8_t` type tag (the `variant::index()`):
     - `0` → `int64_t` (raw 8 bytes)
     - `1` → `double` (raw 8 bytes)
     - `2` → `bool` (raw 1 byte)
     - `3` → `std::string` (length-prefixed)
3. Unknown tags throw `StorageError`.

This tag-based dispatch is implemented with `std::visit`:

```cpp
put<std::uint8_t>(out, static_cast<std::uint8_t>(value.index()));
std::visit(
    [&out](const auto& v) {
        using V = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<V, std::string>) put_string(out, v);
        else put<V>(out, v);
    },
    value);
```

---

## Snapshot

### Format

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 B | Magic `0xE1105E01` |
| 4 | 4 B | Version `1` |
| 8 | 2 B | Dimension |
| 10 | 1 B | Metric |
| 11 | 4 B | Collection count (N) |
| 15 | var | Collection 0…N-1 |

Each collection:
```
[uint32 name_len][name bytes][uint32 record_count][record × N]
```

Each record:
```
[16 B RecordID][uint16 dim][dim × 4 B float vector][payload]
```

### Atomic Publish

Checkpoints use a **write-to-temp, rename-over-live** pattern:

```cpp
void Database::checkpoint() {
    const fs::path tmp = path_ / (std::string(snapshot_file) + ".tmp");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        // ... write snapshot ...
    }
    fs::rename(tmp, path_ / snapshot_file);  // atomic on same filesystem
    if (wal_) wal_->reset();                 // truncate WAL
}
```

The `fs::rename` is atomic on POSIX (same filesystem) — readers will see
either the old complete snapshot or the new complete snapshot, never a
partially-written file. After rename, the WAL is truncated since its
contents are now captured in the snapshot.

### Snapshot Full-Rewrite

Every checkpoint rewrites the **entire** database state — all collections, all
records. This is O(total records × dimension) and produces a self-contained
snapshot. The WAL is then reset to zero. This full-rewrite model is chosen
for its simplicity and self-contained snapshot format; incremental snapshots
or copy-on-write are future optimisation items.

---

## Durability Levels

Defined in `Config.hpp`:

```cpp
enum class Durability { paranoid, standard, relaxed, ephemeral };
```

| Level | WAL flush | WAL fsync | Behavior |
|-------|-----------|-----------|----------|
| `paranoid` | ✅ | (future) | WAL flushed on every write; eventual fsync |
| `standard` | ✅ | ❌ | WAL flushed (= kernel handoff) on every write |
| `relaxed` | ❌ | ❌ | WAL buffered, flushed on checkpoint/close only |
| `ephemeral` | N/A | N/A | No WAL created; database is purely in-memory |

The effective WAL `sync_each_write` flag is set at open time:

```cpp
if (effective.durability() != Durability::ephemeral) {
    const bool sync = effective.durability() != Durability::relaxed;
    instance->attach_wal(std::make_unique<WAL>(walpath, sync));
}
```

**Standard** is the default. It balances crash safety (writes are in the OS
buffer before acknowledgement) with throughput (avoids per-write fsync cost).

**Relaxed** is suitable for bulk loading or non-critical data where some loss
on crash is acceptable.

**Ephemeral** is used for `:memory:` databases.

---

## LockManager

**Header:** `include/elips/kernel/LockManager.hpp`
**Source:** `src/LockManager.cpp`

The LockManager enforces a **single-writer contract** across processes sharing
a database directory.

### Implementation

```cpp
LockManager::LockManager(const std::string& lock_path) {
    fd_ = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) throw StorageError{"cannot open lock file"};
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
        ::close(fd_); fd_ = -1;
        throw LockConflict{"database is already open by another writer"};
    }
}
```

- **`LOCK_EX`:** Exclusive (write) lock — only one process may hold it.
- **`LOCK_NB`:** Non-blocking — if the lock is held by another process, `flock`
  returns immediately with an error rather than blocking. ELIPS throws
  `LockConflict` rather than waiting.
- **RAII:** The destructor calls `flock(LOCK_UN)` and `close(fd_)`. The lock
  is released when the `Database` is destroyed or `close()` is called.
- **Move semantics:** The fd is transferred via `std::exchange` on move
  construction. Copy is deleted (unique ownership).

The lock file is a zero-length marker file at `{path}/LOCK`. It exists solely
for the advisory lock; it carries no data. Multiple readers on the same host
would not conflict (readers don't acquire the exclusive lock — though v1.0
exclusively uses a single-writer model).

---

## Recovery Flow — `open()`

The complete database open sequence (`src/Database.cpp:354–418`):

1. **`LockManager` acquisition:** Opens or creates the LOCK file and acquires
   an exclusive `flock(LOCK_EX|LOCK_NB)`. Fails with `LockConflict` if another
   writer holds the directory.

2. **IDENTITY resolution:** If IDENTITY exists, read dimension/metric/index
   from it, validate against caller's config. If not, create IDENTITY from
   the caller's config (new database).

3. **`Database` construction:** Creates the instance handle with the
   resolved config, persistent flag, and lock.

4. **Snapshot load:** If `elips.snapshot` exists, parse it and populate all
   collections. Vectors are added through the normal `Collection::add()` path
   (which writes through the index), but the WAL is not attached yet
   (WAL writes during snapshot load are skipped).

5. **WAL replay:** Attach a temporary WAL reader (not the live appender),
   replay all valid records on top of the snapshot state. Each replayed
   entry is applied through `Collection::add()` (for inserts) or
   `Collection::remove()` (for erases), again without live WAL appends.

6. **Live WAL attach:** Create a `WAL` in append mode at `wal.log`. The
   `sync_each_write` flag is `false` for `relaxed` durability, `true`
   otherwise. `ephemeral` databases skip this step entirely.

After step 6, the database is fully operational: reads hit the in-memory
state (index + record map), writes are WAL-appended before in-memory
mutation, and the next checkpoint will publish a fresh snapshot and truncate
the WAL.