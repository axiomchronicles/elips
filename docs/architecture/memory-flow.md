# ELIPS Memory Flow

This document describes how data flows through ELIPS memory structures: vector storage, index data, WAL, snapshots, record storage, and GPU memory management.

## Memory Architecture Overview

```
+-------------------------------------------------------------+
|  Vault                                                      |
|  +-------------------------------------------------------+ |
|  | records_: std::map<RecordID, Record>                  | |
|  |   Record: { RecordID id, Vector vector, Payload }     | |
|  |   Vector: { std::vector<float> values_ }              | |
|  |   Payload: std::map<std::string, MetaValue>           | |
|  +-------------------------------------------------------+ |
|  | index_: std::unique_ptr<IndexPort>                    | |
|  |   ExactIndex or HierarchicalGraphIndex                | |
|  +-------------------------------------------------------+ |
|  | wal_*: WAL* (non-owning)                              | |
|  +-------------------------------------------------------+ |
+-------------------------------------------------------------+
                          │
                          │ WAL pointer shared with ElipsInstance
                          ▼
+-------------------------------------------------------------+
|  ElipsInstance                                              |
|  +-------------------------------------------------------+ |
|  | wal_: std::unique_ptr<WAL>                             | |
|  |   path_: wal.log                                       | |
|  |   out_: std::ofstream (append mode, binary)            | |
|  |   sync_each_write_: bool                               | |
|  +-------------------------------------------------------+ |
|  | vaults_: std::map<std::string, unique_ptr<Vault>>     | |
|  +-------------------------------------------------------+ |
+-------------------------------------------------------------+
                          │
                          │ optional (ELIPS_GPU_ENABLED)
                          ▼
+-------------------------------------------------------------+
|  GPU Memory (Device)                                        |
|  +-------------------------------------------------------+ |
|  | GpuMemoryManager                                       | |
|  |   pool_bytes_: size_t (80% of device memory)           | |
|  |   free_blocks_: vector<FreeBlock> (best-fit pool)      | |
|  |   pinned_blocks_: vector<void*> (aligned_alloc)        | |
|  +-------------------------------------------------------+ |
|  | GpuMemoryPool (fixed-size)                             | |
|  |   pool_buffer_: GpuBuffer                              | |
|  |   blocks_: vector<Block> (free/used tracking)          | |
|  +-------------------------------------------------------+ |
|  | GpuIndexPort buffers                                   | |
|  |   GpuBruteForceIndex: GpuBuffer database_buffer_       | |
|  |   GpuGraphIndex: GpuBuffer graph_data_                 | |
|  +-------------------------------------------------------+ |
+-------------------------------------------------------------+
```

## 1. Vector Memory

The `Vector` class is the fundamental data unit:

```cpp
class Vector {
    std::vector<float> values_;  // owned float array
};
```

- **Ownership**: `Vector` owns its `std::vector<float>` via value semantics. Copying a `Vector` copies all float data.
- **Access**: `values()` returns a `std::span<const float>` (non-owning view) for zero-copy access to the underlying data.
- **Dimension**: `dimension()` returns `values_.size()` as `uint16_t`.
- **Normalization**: `normalized()` returns a NEW `Vector` with L2-normalized values. The original is unchanged. Zero vectors are returned unchanged (avoid division by zero).
- **Move semantics**: `Vector(std::vector<float>&&)` efficiently transfers ownership without copying.

### Memory Footprint per Record

| Component | Size |
|-----------|------|
| RecordID | 16 bytes (128-bit UUIDv7) |
| Vector values (dimension N) | N × 4 bytes (float32) |
| Payload | Variable (map of string → variant) |

For a 768-dimensional embedding with sparse metadata: ~3,088+ bytes per record.

## 2. Index Memory

### ExactIndex (Brute-Force)

```cpp
class ExactIndex {
    std::vector<RecordID> ids_;         // N × 16 bytes
    std::vector<float> data_;           // N × dim × 4 bytes, row-major
};
```

- **Row-major layout**: Vector `i` starts at `data_[i * dimension_]`. Cache-friendly for linear scanning.
- **Removal**: `std::find` over `ids_` then `erase` from both vectors (O(N) shift). Suitable only for small collections.

### HierarchicalGraphIndex (HNSW)

```cpp
class HierarchicalGraphIndex {
    std::vector<float> data_;                                      // N × dim × 4 bytes
    std::vector<RecordID> ids_;                                    // N × 16 bytes
    std::vector<bool> deleted_;                                    // N bits
    std::vector<int> node_levels_;                                 // N × 4 bytes
    std::vector<std::vector<std::vector<NodeId>>> links_;          // adjacency lists
    std::unordered_map<RecordID, NodeId> id_to_node_;              // N × (16+4+overhead)
};
```

- **Row-major vectors**: Same layout as ExactIndex.
- **Multi-layer adjacency**: `links_[node][level]` is a `std::vector<NodeId>` of neighbor node indices for that level. Size depends on `max_connections` (M) and the node's level.
- **Soft tombstones**: `deleted_` flags nodes as removed without modifying graph structure. `deleted_count_` tracks active deletions. `size()` returns `ids_.size() - deleted_count_`.
- **Memory estimate**: For 1M vectors of 768 dimensions with M=16, average level ~5:
  - Vectors: 1M × 768 × 4 = ~3.0 GB
  - IDs: 1M × 16 = ~16 MB
  - Adjacency lists (approximate): M × 2 × avg_level × 4 bytes per node = ~640 MB
  - Total: ~3.7 GB

## 3. WAL Memory (Append-Only)

```cpp
class WAL {
    std::filesystem::path path_;        // path to wal.log
    std::ofstream out_;                 // append-mode binary stream
    bool sync_each_write_;              // flush behavior
};
```

- **Append-only**: `out_` is opened in `std::ios::app` mode. All writes are sequential to the end of the file.
- **No in-memory buffer in WAL**: Records are serialized directly to the `std::ofstream`. The OS page cache buffers writes; `flush()` forces dirty pages to disk.
- **CRC32C**: Each WAL entry has a 4-byte CRC32C checksum appended after the record body. The CRC table (256 × 4 bytes) is computed once at static init.
- **Sync behavior by durability**:
  - `paranoid` / `standard`: `sync_each_write_=true`, `out_.flush()` after every append.
  - `relaxed`: `sync_each_write_=false`, no explicit flush (OS flushes lazily).
  - `ephemeral`: No WAL at all (`wal_` is nullptr).
- **Replay memory**: `WAL::replay()` reads the entire log into a `std::string` blob, then parses sequentially. The blob is deallocated when replay completes.

## 4. Snapshot Memory (Full Serialization)

Snapshots are full database serializations written to a temp file then atomically renamed:

- **No incremental snapshots**: Every checkpoint writes the complete state of all vaults.
- **Binary format**: Uses `elips::detail::put<T>()` and `elips::detail::get<T>()` for native-byte-order serialization.
- **Payload serialization**: `put_payload()` writes each key-value pair as: key length (4 bytes) + key string + type tag (1 byte) + value (variable size per type).
- **Memory during checkpoint**: The entire database state is iterated in-memory. No additional large allocations (streamed to file incrementally).

## 5. Record Store

```cpp
class Vault {
    std::map<RecordID, Record> records_;  // ordered by UUIDv7 (time order)
};
```

- **Ordered by RecordID**: `std::map` with `operator<=>` on `RecordID` (lexicographic byte order = UUIDv7 time order).
- **Lookup**: `fetch(id)` uses `std::map::find()` — O(log N).
- **Scan**: Sequential iteration with offset/limit and filter predicate.
- **No secondary indices**: Metadata filtering is a linear scan during `seek()` post-filtering and `scan()`.

## 6. GPU Memory Management

### 6.1 Device Memory Pool (GpuMemoryManager)

```
GpuMemoryManager
  │
  ├─ pool_bytes_: 80% of total device memory by default
  ├─ allocated_: currently used device bytes
  ├─ peak_allocated_: peak bytes tracked
  │
  ├─ Best-fit allocator:
  │   └─ free_blocks_: vector<FreeBlock{{ptr, bytes}>
  │       ├─ On allocate(bytes): scan free_blocks_ for best fit
  │       ├─ Best fit found → reclaim block, track allocated_
  │       ├─ No fit → allocate_device(alloc_size), add remainder to free_blocks_
  │       └─ On deallocate: push to free_blocks_
  │
  └─ Pinned memory:
      └─ pinned_blocks_: vector<void*>
          └─ allocate_pinned(bytes): std::aligned_alloc(4096, bytes)
          └─ deallocate_pinned(ptr): std::free(ptr)
```

- **Thread safety**: All public methods acquire `std::lock_guard<std::mutex>`.
- **Best-fit strategy**: Iterates over free blocks, selects the one with smallest waste (`bytes - requested`). Minimizes fragmentation.
- **Peak tracking**: `peak_allocated_` is updated on every allocation.
- **Shutdown**: Frees all device buffers via `backend_.free_device()` and all pinned blocks via `std::free()`.

### 6.2 Fixed-Size Pool (GpuMemoryPool)

```
GpuMemoryPool
  │
  ├─ pool_buffer_: single GpuBuffer (allocated once at construction)
  ├─ blocks_: vector<Block{{ptr, bytes, free}>
  │
  ├─ acquire(bytes):
  │   ├─ Scan blocks_ for first free block >= bytes
  │   ├─ Split if remaining > bytes + alignment
  │   └─ Return GpuBuffer{ptr, bytes}
  │
  ├─ release(buf):
  │   ├─ Mark block as free
  │   └─ Coalesce with adjacent free block (next sequential ptr)
  │
  └─ Tracking: available() sums free bytes, used() = total - available
```

- **First-fit with splitting**: Acquires the first free block that fits, splits remainder into a new free block.
- **Coalescing on release**: After marking a block free, checks if the next sequential block is also free and merges them.
- **Fallback**: If the pool is exhausted, falls back to direct `backend_.allocate_device()`.
- **Thread safety**: All public methods acquire `std::lock_guard<std::mutex>`.

### 6.3 Unified Memory (Apple Silicon)

On Apple Silicon (M1/M2/M3), Metal's unified memory architecture means the CPU and GPU share the same physical memory pool:

- **UnifiedBuffer**: Wraps a host pointer (`void* host_ptr()`) and a GPU buffer view (`GpuBuffer gpu_view_`). Both point to the same physical memory — no `upload()`/`download()` transfers needed.
- **Auto-detection**: `GpuDeviceInfo::has_unified_memory` is set by the Metal backend when probing. Apple Silicon always reports `true`.
- **Config control**: `GpuConfig::use_unified_memory` can be set to explicitly enable/disable this optimization.

### 6.4 Pinned (Page-Locked) Memory

- **Allocation**: `std::aligned_alloc(4096, bytes)` — page-aligned for efficient DMA.
- **Usage**: Host-side staging buffers for non-unified-memory GPUs. Pinned memory allows the GPU DMA engine to access host memory directly without page faults.
- **Pool**: `pinned_host_pool_bytes` (default 256 MB) in `GpuConfig`.
- **Lifecycle**: Tracked in `GpuMemoryManager::pinned_blocks_`, freed on `shutdown()`.

### 6.5 GPU Buffer Types

| Type | Purpose | Backend Handle |
|------|---------|----------------|
| `GpuBuffer` | General device buffer | `void* device_ptr_` + `void* backend_handle_` (e.g., `MTLBuffer*`, `CUdeviceptr`) |
| `UnifiedBuffer` | Apple Silicon shared memory | `void* host_ptr_` + `GpuBuffer gpu_view_` |
| `PinnedBuffer` | Host-side page-locked memory | `void* data_` (from `aligned_alloc`) |

## 7. RecordID Memory

```cpp
class RecordID {
    std::array<std::uint8_t, 16> bytes_;  // 128-bit UUIDv7
};
```

- **16 bytes**: Fixed size, value type. No heap allocation.
- **UUIDv7 format**: First 48 bits are a Unix timestamp in milliseconds (big-endian), followed by version/variant bits and random data.
- **Time ordering**: Lexicographic byte order equals UUIDv7 time order, making `std::map<RecordID, ...>` a time-ordered container.
- **FNV-1a hash**: `std::hash<RecordID>` uses FNV-1a with 64-bit mixing for hash table use (e.g., `id_to_node_` in HNSW).

## 8. Filter Memory

```cpp
class Filter {
    std::shared_ptr<const Node> root_;  // nullptr = match-all
    std::string current_field_;          // builder state
};
```

- **Immutable tree**: `Node` objects are `shared_ptr<const Node>`, enabling safe copy and sharing of filter trees.
- **Node types**: `cmp` (comparison), `in` (set membership), `contains` (substring), `conj` (AND), `disj` (OR), `neg` (NOT), `none` (never-match).
- **Tree sharing**: `and_()` / `or_()` create new nodes that share subtrees via `shared_ptr`.
- **No allocation during matching**: `matches()` is a pure recursive evaluation over `const Node*` pointers.