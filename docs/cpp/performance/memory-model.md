# Memory Model

This document describes the memory layout, allocation patterns, and ownership
model of ELIPS data structures. Understanding the memory model is essential for
capacity planning, performance tuning, and diagnosing memory-related issues.

---

## Row-Major Vector Storage

Both index implementations (`IndexFlat` and `IndexHNSW`) store
vectors as a single contiguous `std::vector<float>` in **row-major** order:

```
data_ = [ row_0[0], row_0[1], ..., row_0[dim-1],
          row_1[0], row_1[1], ..., row_1[dim-1],
          ...
          row_{N-1}[0], row_{N-1}[1], ..., row_{N-1}[dim-1] ]
```

Access is by pointer arithmetic: `data_.data() + i × dimension_` gives the start
of row `i`. A `std::span<const float>` is constructed with size `dimension_`.

**Cache benefits:** Row-major layout means sequential access (linear scans in
`IndexFlat`) exhibits excellent spatial locality. The NEON kernel loads 4 floats
(16 bytes) per SIMD instruction, fitting exactly into a cache line on most
architectures.

**Memory for N vectors of dimension D:**
- Vector data: `N × D × 4` bytes (float32).
- ID storage: `N × 16` bytes (UUIDv7 raw bytes in `std::vector<RecordID>`).

---

## Vector Copies at Insert (No Zero-Copy)

When a vector is inserted into an index, **a copy is made** in the flat `data_`
array:

```cpp
// IndexFlat::add_with_ids
data_.insert(data_.end(), vector.begin(), vector.end());

// IndexHNSW::add_with_ids (uncompressed path)
data_.insert(data_.end(), vector.begin(), vector.end());
```

Similarly, the `Collection` stores the full `Vector` object (which owns a
`std::vector<float>`) in its `records_` map. This means:

- Each added vector exists in **two locations**: the index's flat `data_`
  array AND the collection's `records_` map (as a `Vector`). This doubles the
  vector memory footprint.
- The index copy is tightly packed (row-major flat array, no per-entry
  overhead beyond the floats themselves).
- The collection copy is inside a `Record` struct with a `Vector` member — the
  `Vector` itself is a `std::vector<float>` with its own dynamic allocation.

**Rationale:** Zero-copy from the collection into the index would require either
shared ownership (complicating lifecycle) or move semantics (leaving the
collection record empty). The current design prioritises correctness and
simplicity — the collection copy is the single source of truth for `get()` and
`scan()` operations, and the index copy is optimised for search.

**Capacity estimate:** Total vector memory ≈ `2 × N × D × 4` bytes (index +
collection copies). For 1M vectors at dimension 768, this is ~6.1 GB for vectors
alone.

---

## Index Memory Overhead

### IndexFlat

Overhead beyond vectors and IDs is minimal:
- `ids_` vector overhead: `N × 16` bytes.
- `data_` vector overhead: `N × D × 4` bytes.
- No additional index structures.

**Total:** `N × (16 + D × 4)` bytes. No multiplicative overhead factor.

### IndexHNSW (HNSW)

The HNSW index adds substantial graph structure overhead:

| Component | Storage | Bytes per element |
|-----------|---------|-------------------|
| `data_` | `vector<float>` | `N × D × 4` |
| `ids_` | `vector<RecordID>` | `N × 16` |
| `deleted_` | `vector<bool>` | ~`N / 8` (bit-packed) |
| `node_levels_` | `vector<int>` | `N × 4` |
| `links_` | `vector<vector<vector<uint32_t>>>` | Variable (see below) |
| `id_to_node_` | `unordered_map` | ~`N × (16 + 4 + hash overhead)` |

**Links memory** is the dominant overhead. For each node at each of its levels
(max level `l`), there is a neighbour list of up to `M` NodeIds (4 bytes each).
At layer 0, the limit is `2M`. With default `M = 16` and geometric level
distribution (approximately `N/M` nodes per level):

- Average links per node: roughly `M × avg_levels × 4` bytes.
- With `M = 16`, `mL = 1/ln(16) ≈ 0.36`, average level ≈ 0.36.
- Average links per node ≈ `16 × 1.36 × 4 ≈ 87` bytes.
- Plus list overhead (`std::vector` overhead ≈ 24 bytes per list + amortised
  capacity).

**Approximate HNSW total memory:** `N × (D × 4 + 16 + 4 + ~100-200)` bytes
for links and metadata. For D=768, the graph overhead (~200 bytes) is
only ~6% of the vector data (~3072 bytes).

**`id_to_node_` map:** Uses `std::unordered_map` with hash table overhead. Each
entry consumes approximately 40–64 bytes (RecordID key + uint32_t value +
bucket pointer overhead). This grows linearly with N and is never pruned
(soft deletes leave entries in place).

---

## WAL Per-Record Framing Overhead

Each WAL insert record carries:

| Field | Size |
|-------|------|
| Magic (uint32) | 4 B |
| Op code (uint8) | 1 B |
| Collection name (len prefix + bytes) | 4 B + len(name) |
| Record ID (UUIDv7) | 16 B |
| Dimension (uint16) | 2 B |
| Vector data | `D × 4` B |
| Payload (len prefix + key-value pairs) | variable |
| CRC32C | 4 B |

**Overhead per insert (excl. vector and payload):** 4 + 1 + 4 + len(collection_name)
+ 16 + 2 + 4 = 31 + len(collection_name) bytes.

For a typical collection name like `"default"` (7 bytes), framing overhead is ~38
bytes per insert. For D=768, the vector is 3072 bytes, making framing overhead
~1.2%.

**WAL growth:** Each insert adds `38 + D×4 + payload_bytes` to the log. The
log grows unbounded between checkpoints. A checkpoint truncates the WAL to
zero.

---

## Snapshot Full-Rewrite — O(N)

Every checkpoint rewrites the entire database state:

1. **All collections** are iterated in insertion order.
2. **All records** in each collection are serialised with their vectors, IDs, and
   payloads.
3. **The WAL is truncated** to zero after successful atomic rename.

This means:
- Checkpoint time is linear in the total number of records (`O(N × D)` for
  vector serialisation).
- Checkpoint disk I/O equals the full database size.
- Between checkpoints, only the WAL grows (incremental).
- No incremental snapshot or copy-on-write mechanism exists in v1.0.

**Implications for large databases:** For 10M vectors at D=768, a checkpoint
writes ~30 GB to disk. Checkpoints should be scheduled during low-traffic
periods or triggered explicitly via `Database::checkpoint()`.

---

## GPU Memory Pool Strategy

When compiled with `ELIPS_GPU_ENABLED`, GPU-accelerated indices manage device
memory through a pooling subsystem designed for throughput and minimal
fragmentation.

### `GpuMemoryPool`

**Header:** `include/elips/gpu_engine/GpuMemoryPool.hpp`

The GPU memory pool pre-allocates a single large device buffer at
initialisation time (size determined by `GpuConfig::device_memory_pool_bytes`).
Subsequent allocations carve sub-regions from this pool:

```
pool_buffer_ (single pre-allocated GpuBuffer)
├── Block 0: offset=0,     bytes=X, free=false
├── Block 1: offset=X,     bytes=Y, free=true
├── Block 2: offset=X+Y,   bytes=Z, free=false
└── Block N: ...
```

- `acquire(bytes, alignment)` scans for a free block ≥ the requested size
  (first-fit strategy). Returns the block marked as used.
- `release(buf)` marks the corresponding block as free. Adjacent free blocks
  are coalesced to reduce fragmentation.
- Thread safety is ensured by `std::mutex` on all allocation operations.

**Benefits:** Eliminates per-operation CUDA/Metal allocation overhead
(typically 10–100 μs per `cudaMalloc`/`MTLBuffer::newBuffer`). Pool
allocations are O(N) in the number of blocks (first-fit linear scan), but
the number of blocks is kept small by coalescing.

### `GpuConfig` memory parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device_memory_pool_bytes` | 0 (auto) | Size of pre-allocated device pool; 0 means auto-detect from device |
| `pinned_host_pool_bytes` | 256 MB | Size of host pinned memory pool for DMA transfers |
| `use_unified_memory` | false | Use CUDA unified memory or Metal shared memory |

---

## Pinned (Page-Locked) Buffers

**Header:** `include/elips/gpu_engine/PinnedBuffer.hpp`

`PinnedBuffer` represents a host-side memory allocation that is page-locked
(pinned) for efficient DMA transfers to/from the GPU:

```cpp
class PinnedBuffer {
    void* data_;
    size_t bytes_;
    // move-only, non-copyable
};
```

**Usage:** Pinned buffers are used as staging areas for:
- Uploading vector batches to the GPU for index building.
- Downloading search results from the GPU.
- Holding query vectors for repeated GPU searches.

Pinned memory avoids the driver-internal copy that would otherwise occur with
pageable host memory. On Apple Silicon (Metal), since CPU and GPU share unified
memory, pinned buffers are equivalent to regular allocations and the abstraction
collapses to a no-op. On discrete GPUs (CUDA), `cudaMallocHost` (or equivalent)
is used.

**Capacity:** The default pinned pool is 256 MB (`GpuConfig::pinned_host_pool_bytes`).
For typical workloads, 256 MB accommodates ~340K float32 vectors at dimension 192,
or ~85K at dimension 768.

---

## Memory Ownership Summary

| Component | Owns | Shared? |
|-----------|------|---------|
| `Collection::records_` | `std::map<RecordID, Record>` → each `Record` owns its `Vector` and `Payload` | No |
| `Index::data_` | `std::vector<float>` — flat copy of all vectors | No, independent copy |
| `WAL::Entry` | Temporary vector copies during append | Transient (on stack) |
| `GpuMemoryPool` | Single pre-allocated device buffer, subdivided | Yes (mutex-guarded) |
| `PinnedBuffer` | Page-locked host buffer | Move-only ownership |

There is no reference counting, no copy-on-write sharing, and no
`std::shared_ptr` for large data structures. The data duplication between
collection and index is the primary memory multiplier (2× for vectors).