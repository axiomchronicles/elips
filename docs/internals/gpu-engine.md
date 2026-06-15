# ELIPS GPU Engine

Exhaustive documentation of the GPU acceleration subsystem, covering the abstract interface, device management, backend implementations, memory allocation, index types, batching, pipelines, profiling, and the fallback chain.

## Architecture Overview

The GPU engine is an optional, compile-time-selectable layer that accelerates vector distance computation and approximate nearest neighbor search. It follows a **Ports & Adapters** pattern: the `GpuPort` abstract interface defines the contract, and backend-specific implementations (Metal, CUDA, HIP, SYCL, Vulkan) fulfill it.

```
+--------------------------------------------------------------+
|  elips_core (Database.cpp)                                   |
|    GpuDeviceManager::probe_all_devices()                     |
|    GpuSelector::select(config, devices)                      |
+--------------------------------------------------------------+
        │
        ▼
+--------------------------------------------------------------+
|  elips_gpu (static library)                                  |
|  +--------------------------------------------------------+  |
|  | GpuPort (abstract interface)                           |  |
|  +--------------------------------------------------------+  |
|  | GpuMemoryPort, GpuKernelPort, GpuStreamPort            |  |
|  +--------------------------------------------------------+  |
|  | GpuDeviceManager, GpuSelector                          |  |
|  +--------------------------------------------------------+  |
|  | GpuMemoryManager, GpuMemoryPool                        |  |
|  +--------------------------------------------------------+  |
|  | GpuIndexPort + implementations                         |  |
|  |   GpuBruteForceIndex, GpuGraphIndex                    |  |
|  |   GpuIVFFlatIndex, GpuIVFPQIndex, GpuHybridIndex       |  |
|  +--------------------------------------------------------+  |
|  | DynamicBatcher, GpuSearchPipeline                      |  |
|  | GpuQuantizationPipeline, GpuIngestionPipeline          |  |
|  | GpuProfiler                                            |  |
|  +--------------------------------------------------------+  |
+--------------------------------------------------------------+
        │ (compile-time backend inclusion)
        ▼
+--------------------------------------------------------------+
|  Backend Libraries                                           |
|  elips_gpu_metal  → MetalBackend.mm  + Metal/MPS/Foundation |
|  elips_gpu_cuda   → CudaBackend.cpp  + CUDA toolkit         |
|  elips_gpu_hip    → HipBackend.cpp   + ROCm/HIP             |
|  elips_gpu_sycl   → SyclBackend.cpp  + oneAPI/SYCL          |
|  elips_gpu_vulkan → VulkanBackend.cpp + Vulkan SDK          |
+--------------------------------------------------------------+
```

## 1. GpuPort — Abstract Backend Interface

`GpuPort` is the base class for all GPU backend adapters, defined in `include/elips/gpu_engine/GpuPort.hpp`. It declares 13 pure virtual methods:

### Initialization and Lifecycle

```cpp
[[nodiscard]] virtual std::expected<void, GpuError>
initialize(const GpuConfig& config) = 0;

virtual void shutdown() noexcept = 0;
```

- `initialize()`: Configure and initialize the backend. Returns `std::expected<void, GpuError>` for error handling. The `GpuConfig` provides policy, memory pool sizes, precision preferences, and profiling flags.
- `shutdown()`: Release all backend resources. Called during `GpuDeviceManager::select()` probe cleanup and on `GpuMemoryManager` shutdown.

### Device Information

```cpp
[[nodiscard]] virtual GpuDeviceInfo device_info() const noexcept = 0;
[[nodiscard]] virtual bool is_available() const noexcept = 0;
```

- `device_info()`: Returns a populated `GpuDeviceInfo` struct with device name, vendor, backend name, memory capacity, compute capabilities, precision support, bandwidth, and feature flags.
- `is_available()`: Whether the backend successfully initialized and the device is ready for operations.

### Memory Allocation

```cpp
[[nodiscard]] virtual std::expected<GpuBuffer, GpuError>
allocate_device(size_t bytes) = 0;

virtual void free_device(GpuBuffer&& buf) noexcept = 0;
```

- `allocate_device(bytes)`: Allocates `bytes` on the GPU device. Returns a `GpuBuffer` wrapping a `void* device_ptr` and an opaque `void* backend_handle` (e.g., `MTLBuffer*` for Metal, `CUdeviceptr` for CUDA).
- `free_device(buf)`: Releases a device buffer. Takes by rvalue-reference to transfer ownership.

### Data Transfer

```cpp
[[nodiscard]] virtual std::expected<void, GpuError>
upload(const void* host_src, GpuBuffer& dst, size_t bytes) = 0;

[[nodiscard]] virtual std::expected<void, GpuError>
download(const GpuBuffer& src, void* host_dst, size_t bytes) = 0;
```

- `upload()`: Copy data from host memory to a device buffer. On Apple Silicon with unified memory, this may be a no-op or minimal copy.
- `download()`: Copy data from a device buffer to host memory.

### Distance Computation

```cpp
[[nodiscard]] virtual std::expected<void, GpuError>
compute_distances_batch(
    std::span<const float> queries,
    std::span<const float> database,
    std::span<float> distances_out,
    size_t nq, size_t nb, size_t dim,
    elips::Metric metric) = 0;
```

Computes all-pairs distances between `nq` query vectors and `nb` database vectors of dimension `dim`. The result matrix is stored in `distances_out` (size `nq * nb`, row-major). Supports `cosine`, `euclidean`, and `dot_product` metrics via the `elips::Metric` parameter.

### Top-K Selection

```cpp
[[nodiscard]] virtual std::expected<void, GpuError>
top_k(
    std::span<const float> distances,
    std::span<uint32_t> indices_out,
    std::span<float> values_out,
    size_t nq, size_t nb, size_t k) = 0;
```

Selects the top-k smallest distances for each query row from a pre-computed distance matrix. Results are written to `indices_out` and `values_out` (row-major, `nq * k` each).

### Synchronization

```cpp
virtual void synchronize() = 0;
[[nodiscard]] virtual bool is_idle() const noexcept = 0;
```

- `synchronize()`: Block until all queued GPU operations complete.
- `is_idle()`: Check if the GPU command queue is empty.

### GpuError Enum

```cpp
enum class GpuError {
    DeviceNotFound,
    InsufficientMemory,
    KernelLaunchFailed,
    TransferFailed,
    IndexBuildFailed,
    UnsupportedMetric,
    InitializationFailed,
    BackendUnavailable,
};
```

All GPU operations use `std::expected<T, GpuError>` for error handling.

## 2. Port Interfaces

### GpuMemoryPort

```cpp
class GpuMemoryPort {
    virtual std::expected<GpuBuffer, GpuError> allocate(size_t bytes, size_t alignment = 256) = 0;
    virtual void deallocate(GpuBuffer&& buf) noexcept = 0;
    virtual std::expected<void*, GpuError> allocate_pinned(size_t bytes) = 0;
    virtual void deallocate_pinned(void* ptr) noexcept = 0;
    virtual size_t bytes_used() const noexcept = 0;
    virtual size_t bytes_available() const noexcept = 0;
    virtual size_t peak_bytes_used() const noexcept = 0;
};
```

Higher-level memory management interface. Adds alignment support, pinned (page-locked) host memory, and usage tracking compared to bare `GpuPort` allocation.

### GpuKernelPort

```cpp
class GpuKernelPort {
    virtual std::expected<void, GpuError>
    cosine_fp32(span<const float> queries, span<const float> database,
                span<float> distances, size_t nq, size_t nb, size_t dim) = 0;
    virtual std::expected<void, GpuError>
    euclidean_fp32(span<const float> queries, span<const float> database,
                   span<float> distances, size_t nq, size_t nb, size_t dim) = 0;
    virtual std::expected<void, GpuError>
    dot_product_fp32(span<const float> queries, span<const float> database,
                     span<float> distances, size_t nq, size_t nb, size_t dim) = 0;
};
```

Specialized interface for individual kernel launches. Each metric has its own function, enabling backend-specific shader/kernel dispatch.

### GpuStreamPort

```cpp
class GpuStreamPort {
    virtual std::expected<void, GpuError> synchronize() = 0;
    virtual bool is_complete() const noexcept = 0;
    virtual void wait_for_completion() = 0;
};
```

Stream-level synchronization interface for async execution patterns.

## 3. GpuDeviceManager

```cpp
class GpuDeviceManager {
    std::vector<GpuDeviceInfo> probe_all_devices() const;
    std::optional<std::unique_ptr<GpuPort>> select(const GpuConfig&, const vector<GpuDeviceInfo>&) const;
    bool can_fit_index(const GpuDeviceInfo&, size_t n_vectors, size_t dim, const GpuConfig&) const;
};
```

### probe_all_devices()

The probing order is determined at compile time by `#if defined()` guards:

1. **Metal** (`__APPLE__ && ELIPS_METAL_ENABLED`): Creates a `metal::MetalBackend`, calls `initialize()`, extracts `device_info()`, calls `shutdown()`.
2. **CUDA** (`ELIPS_CUDA_ENABLED`): Same probe pattern with `cuda::CudaBackend`.
3. **HIP** (`ELIPS_HIP_ENABLED`): Same with `hip::HipBackend`.
4. **SYCL** (`ELIPS_SYCL_ENABLED`): Same with `sycl::SyclBackend`.
5. **Vulkan** (`ELIPS_VULKAN_ENABLED`): Same with `vulkan::VulkanBackend`.

Each probe is wrapped in a try-catch to handle initialization failures gracefully.

After probing, devices are sorted:
1. **CAGRA support first** (`a.supports_cagra != b.supports_cagra`)
2. **Then by memory capacity** (`total_device_memory_bytes > ...`)

This ensures the most capable GPU (CAGRA-capable, most memory) is selected by default.

### select()

Selects the best available backend matching the `GpuConfig` policy:
- `GpuPolicy::CpuOnly` → returns `std::nullopt`
- `GpuPolicy::RequireGpu` with no devices → returns `std::nullopt`
- Matches the chosen device's `.backend` string against compile-time-enabled backends (same order as probing)
- If no backend matches and policy is `RequireGpu`, returns `std::nullopt`

### can_fit_index()

Estimates whether an index can fit in device memory:
```cpp
size_t required = n_vectors * dim * sizeof(float);  // vector data
if (algorithm == CagraGraph) {
    required += n_vectors * graph_params.graph_degree * sizeof(uint32_t);  // adjacency
    required += n_vectors * sizeof(int);                                    // levels
}
return dev.free_device_memory_bytes > required;
```

## 4. GpuSelector

```cpp
class GpuSelector {
    std::optional<std::unique_ptr<GpuPort>>
    select(const GpuConfig& config, const std::vector<GpuDeviceInfo>& devices) const;
private:
    int rank_backend(std::string_view backend) const noexcept;
};
```

### Backend Ranking

| Backend | Score | Rationale |
|---------|-------|-----------|
| `"cuda"` | **100** | Highest performance, most mature ecosystem, CAGRA support |
| `"hip"` | **90** | AMD ROCm, comparable to CUDA for supported GPUs |
| `"metal"` | **80** | Native Apple GPU, excellent for unified memory architectures |
| `"sycl"` | **50** | Intel oneAPI, cross-vendor but less mature for ANN |
| `"vulkan"` | **30** | Cross-platform compute, highest abstraction overhead |
| `"cpu"` | **0** | CPU SIMD fallback (no GPU) |

### select() Policy Logic

```cpp
if (config.policy == GpuPolicy::CpuOnly) return nullopt;     // user explicitly wants CPU
if (devices.empty()) {
    if (config.policy == RequireGpu) return nullopt;         // no GPU available, user requires one
    return nullopt;                                           // no GPU, fall back to CPU
}

// Use best device (front after sorting by CAGRA + memory)
const GpuDeviceInfo& best = devices.front();

// If Specific policy and preferred_backend is set, find matching device
if (config.policy == Specific && !config.preferred_backend.empty()) {
    for each device: if dev.backend == preferred_backend, use it
}

// Instantiate the backend via GpuDeviceManager::select()
```

## 5. MetalBackend Implementation

The Metal backend (`backends/metal/MetalBackend.mm`) is the primary GPU backend on Apple platforms.

### Initialization

1. **MTLDevice Detection**: `MTLCreateSystemDefaultDevice()` — obtains the default GPU.
2. **Command Queue**: Creates an `MTLCommandQueue` for serializing GPU work.
3. **Shader Compilation**: Compiles MSL (Metal Shading Language) kernels from embedded source strings:
   - `cosine_fp32`: Computes `(a · b) / (||a|| * ||b||)` for normalized input vectors
   - `euclidean_fp32`: Computes `||a - b||^2` squared Euclidean distance
   - `dot_product_fp32`: Computes `a · b` dot product
4. **Pipeline State**: Creates `MTLComputePipelineState` objects for each kernel.
5. **Device Info**: Populates `GpuDeviceInfo` with Metal-specific capabilities:
   - `name`: GPU name (e.g., "Apple M3 Pro")
   - `vendor`: "Apple"
   - `backend`: "metal"
   - `has_unified_memory`: `true` on Apple Silicon
   - `supports_fp16`: true on most Apple GPUs
   - `supports_cagra`: true (CAGRA graph index supported on Metal)
   - `supports_ivf_pq`: true
   - `host_to_device_bandwidth_gb_s`: effectively infinite (unified memory)

### Memory Operations

- **Unified Memory**: On Apple Silicon, `allocate_device()` uses `[MTLDevice newBufferWithBytesNoCopy:length:options:deallocator:]` to create a Metal buffer backed by existing host memory — no copy needed.
- **upload() / download()**: When unified memory is active, these are no-ops (both CPU and GPU access the same physical memory). The `MTLBlitCommandEncoder` is used for explicit copies when needed.
- **Auto-detection**: `GpuDeviceInfo::has_unified_memory` is set based on `[MTLDevice hasUnifiedMemory]`.

### Kernel Execution

Metal kernels are dispatched via `MTLComputeCommandEncoder`:
1. Set compute pipeline state for the appropriate kernel
2. Set input buffers at binding indices 0 (query), 1 (database), 2 (distance output)
3. Set kernel parameters via `setBytes:` for nq, nb, dim
4. Dispatch threadgroups with appropriate grid sizing
5. End encoding and commit the command buffer

### Precision Strategy

- **FP32 storage**: All vectors are stored as `float` (32-bit).
- **FP16 compute**: When `enable_fp16_search` is true and the device supports FP16, intermediate computations use `half` precision for 2x throughput.
- **Auto-selection**: `GpuPrecision::Auto` selects FP16 when available and the dimension is large enough to benefit.

## 6. GpuMemoryManager

Implements `GpuMemoryPort` with a best-fit pool allocator.

```cpp
class GpuMemoryManager : public GpuMemoryPort {
    GpuPort& backend_;                   // backend for raw allocations
    size_t pool_bytes_{0};               // total pool capacity (80% of device memory)
    size_t allocated_{0};                // currently allocated
    size_t peak_allocated_{0};           // peak usage
    std::vector<FreeBlock> free_blocks_; // best-fit free list
    std::vector<void*> pinned_blocks_;   // host-side pinned allocations
    mutable std::mutex mutex_;           // thread safety
};
```

### Best-Fit Allocation Algorithm

```
allocate(bytes):
  lock mutex
  
  // Search free blocks for best fit (smallest waste)
  best_idx = SIZE_MAX, best_waste = SIZE_MAX
  for each free block:
    if block.bytes >= bytes:
      waste = block.bytes - bytes
      if waste < best_waste: best_idx = i, best_waste = waste
  
  if best_idx found:
    remove block from free_blocks_
    allocated_ += bytes
    return GpuBuffer{block.ptr, bytes}
  
  // No suitable free block → allocate from backend
  alloc_size = max(bytes, pool_bytes_ / 16)   // Minimum chunk = 6.25% of pool
  if allocated_ + alloc_size > pool_bytes_:
    return InsufficientMemory
  
  result = backend_.allocate_device(alloc_size)
  
  // If oversized, add remainder to free list
  if alloc_size > bytes:
    free_blocks_.push_back({ptr + bytes, alloc_size - bytes})
  
  allocated_ += bytes
  return GpuBuffer{ptr, bytes}
```

- **Minimum chunk size**: `pool_bytes_ / 16` to prevent excessive fragmentation.
- **Peak tracking**: `peak_allocated_` is updated on every allocation.
- **No defragmentation**: Blocks are not compacted; free blocks are coalesced when deallocated adjacent blocks are detected (done in `GpuMemoryPool`).

### Deallocation

```
deallocate(buf):
  lock mutex
  allocated_ -= buf.bytes
  free_blocks_.push_back({buf.ptr, buf.bytes})
```

### Pinned Memory

```cpp
allocate_pinned(bytes):
  ptr = std::aligned_alloc(4096, bytes)   // page-aligned for DMA
  pinned_blocks_.push_back(ptr)
  return ptr

deallocate_pinned(ptr):
  remove from pinned_blocks_
  std::free(ptr)
```

- **Page alignment**: 4096-byte alignment ensures DMA compatibility.
- **Thread safety**: Protected by `mutex_`.

### Shutdown

```cpp
shutdown():
  lock mutex
  for each free block: backend_.free_device(block)
  free_blocks_.clear()
  for each pinned: std::free(ptr)
  pinned_blocks_.clear()
  pool_bytes_ = 0
```

Called by `~GpuMemoryManager()`.

## 7. GpuMemoryPool

A fixed-size pool that acquires a single large device buffer at construction and carves it into sub-allocations.

```cpp
class GpuMemoryPool {
    GpuPort& backend_;
    GpuBuffer pool_buffer_;         // single large allocation
    size_t total_bytes_;
    std::vector<Block> blocks_;     // tracking (free/used) sub-blocks
    mutable std::mutex mutex_;
};
```

### Block Tracking

```cpp
struct Block {
    void* ptr;
    size_t bytes;
    bool free;
};
```

### Acquire (First-Fit with Splitting)

```
acquire(bytes, alignment):
  lock mutex
  if no pool_buffer_: fall back to backend_.allocate_device(bytes)
  
  for each block:
    if block.free && block.bytes >= bytes:
      block.free = false
      if block.bytes > bytes + alignment:
        // Split: create new free block from remainder
        remaining.ptr = block.ptr + bytes
        remaining.bytes = block.bytes - bytes
        remaining.free = true
        blocks_.push_back(remaining)
        block.bytes = bytes
      return GpuBuffer{block.ptr, bytes}
  
  return InsufficientMemory
```

### Release with Coalescing

```
release(buf):
  lock mutex
  for each block:
    if block.ptr == buf.ptr:
      block.free = true
      // Coalesce: if next block is free and adjacent, merge
      next = find block where ptr == block.ptr + block.bytes
      if next.free:
        block.bytes += next.bytes
        erase next
      return
```

### Usage Tracking

```cpp
available() → sum of all free block bytes
used() → total_bytes_ - available()
```

## 8. GpuIndexPort and Implementations

`GpuIndexPort` extends `elips::IndexPort` with GPU-specific batch operations:

```cpp
class GpuIndexPort : public elips::IndexPort {
    virtual std::expected<void, GpuError>
    build_from_batch(span<const float> vectors, span<const RecordID> ids,
                     const GpuIndexBuildParams& params) = 0;

    virtual std::expected<vector<vector<SearchResult>>, GpuError>
    search_batch(span<const float> queries, size_t k, size_t ef_search) const = 0;

    virtual std::expected<void, GpuError>
    export_to_cpu_index(IndexPort& cpu_index_out) const = 0;

    virtual std::expected<void, GpuError>
    import_from_cpu_index(const IndexPort& cpu_index) = 0;

    virtual size_t device_bytes_used() const noexcept = 0;
    virtual std::string_view backend_name() const noexcept = 0;
};
```

### GpuBruteForceIndex

```cpp
class GpuBruteForceIndex final : public GpuIndexPort {
    GpuPort& backend_;
    Metric metric_;
    uint16_t dimension_;
    GpuBuffer database_buffer_;          // device-side vector data
    std::vector<RecordID> ids_;          // host-side ID mapping
    size_t count_{0};
};
```

- **Exact search on GPU**: Stores vectors in a single device buffer. `search()` calls `compute_distances_batch()` then `top_k()`.
- **type_name()**: Returns `"gpu_brute_force"`.
- **Build**: `build_from_batch()` uploads all vectors to the device buffer in one transfer.
- **Export/Import**: Can sync with a CPU `ExactIndex`.

### GpuGraphIndex (CAGRA)

```cpp
class GpuGraphIndex final : public GpuIndexPort {
    GpuPort& backend_;
    Metric metric_;
    uint16_t dimension_;
    size_t count_{0};
    GpuBuffer graph_data_;               // device-side graph + vectors
    std::vector<RecordID> ids_;
};
```

- **CAGRA (CUDA Accelerated Graph)**: GPU-native hierarchical graph index. Based on NVIDIA's CAGRA algorithm ported to Metal/CUDA/HIP.
- **type_name()**: Returns `"gpu_graph"`.
- **Build parameters**: Uses `GraphIndexBuildParams` with `intermediate_graph_degree` (default 128), `graph_degree` (default 64), build algorithms: `IvfPq`, `NnDescent`, `IterativeSearch`.

### GpuIVFFlatIndex

```cpp
class GpuIVFFlatIndex final : public GpuIndexPort {
    GpuPort& backend_;
    Metric metric_;
    uint16_t dimension_;
    size_t n_lists_{1024};               // number of IVF clusters
    size_t count_{0};
};
```

- **IVF-Flat**: Inverted File with Flat quantization. Partition vectors into `n_lists` clusters via k-means; search probes the nearest clusters.
- **type_name()**: Returns `"gpu_ivf_flat"`.

### GpuIVFPQIndex

```cpp
class GpuIVFPQIndex final : public GpuIndexPort {
    GpuPort& backend_;
    Metric metric_;
    uint16_t dimension_;
    size_t n_lists_{1024};
    uint32_t pq_dim_{0};                 // dimension after PQ reduction
    uint32_t pq_bits_{8};                // bits per PQ sub-vector
    size_t count_{0};
};
```

- **IVF-PQ**: Inverted File with Product Quantization. Reduces memory footprint by encoding vectors as short PQ codes.
- **type_name()**: Returns `"gpu_ivf_pq"`.
- **Parameters**: `IvfPqBuildParams` with `n_lists`, `pq_dim` (auto if 0), `pq_bits` (8), `add_data_on_build`, `kmeans_n_iters` (20), `kmeans_trainset_fraction` (0.5).

### GpuHybridIndex

```cpp
class GpuHybridIndex final : public GpuIndexPort {
    GpuPort& backend_;
    std::unique_ptr<IndexPort> cpu_index_;    // CPU fallback (HNSW or ExactIndex)
    std::unique_ptr<GpuIndexPort> gpu_index_; // GPU primary
    Metric metric_;
    uint16_t dimension_;
};
```

- **Dual-index architecture**: Maintains BOTH a GPU index AND a CPU index. Queries are routed to the GPU index first; if GPU is unavailable or the index is too small, falls back to CPU.
- **type_name()**: Returns `"gpu_hybrid"`.
- **Synchronization**: `export_to_cpu_index()` and `import_from_cpu_index()` keep both indices in sync.
- **Use case**: `IndexBuildMode::Hybrid` — GPU accelerates queries for large indices while CPU provides guaranteed availability.

## 9. DynamicBatcher

Coalesces individual search queries into batches for higher GPU throughput.

```cpp
class DynamicBatcher {
    size_t window_us_;                // max wait time before processing (default 500μs)
    size_t max_batch_;                // max queries per batch (default 256)
    std::vector<PendingQuery> pending_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    // Search function: (queries, k) → vector<vector<SearchResult>>
    std::function<...> search_fn_;
};
```

### Push Model with Worker Thread

```
enqueue(query_vector, k):
  pq = PendingQuery{copy(query_vector), k, promise}
  future = pq.promise.get_future()
  lock: pending_.push_back(pq)
  notify_one()
  return future
```

The worker thread loop:

```
worker_loop():
  while running_:
    wait_for(window_us_) or until pending_.size() >= max_batch_ or !running_
    
    // Take up to max_batch_ queries
    batch = move up to max_batch_ from pending_
    unlock
    
    // Execute batch search
    results = search_fn_(queries, k)
    
    // Resolve promises
    for each query: promise.set_value(results[i])
```

### Batching Behavior

- **Time window**: Waits up to `window_us_` microseconds for additional queries to arrive before processing. Default 500μs.
- **Max batch**: Processes at most `max_batch_` queries at once. If the queue fills faster than the window, processes immediately.
- **Stats**: `BatchStats` tracks `queries_coalesced`, `kernel_launches`, `avg_batch_size`, `p99_latency_us`.

### Integration

```cpp
batcher.set_search_fn([&](const auto& queries, size_t k) {
    return gpu_index_.search_batch(...);
});
batcher.start();
auto future = batcher.enqueue(query, 10);
auto results = future.get();  // blocks until batch completes
batcher.stop();
```

## 10. GpuSearchPipeline

Orchestrates a complete batch search pipeline:

```cpp
class GpuSearchPipeline {
    GpuPort& backend_;
    
    batch_search(queries, nq, database, db_ids, nb, dim, k, metric)
      → vector<vector<SearchResult>>
};
```

The pipeline:
1. Allocates temporary device buffers for distance matrix and top-k indices/values
2. Uploads queries if not already on device
3. Calls `compute_distances_batch()` to compute the distance matrix
4. Calls `top_k()` to extract the k-nearest neighbors
5. Downloads results back to host
6. Pairs results with record IDs and constructs `SearchResult` objects
7. Cleans up temporary buffers

## 11. GpuQuantizationPipeline

Handles Product Quantization codebook training and encoding:

```cpp
class GpuQuantizationPipeline {
    GpuPort& backend_;
    
    train_pq_codebook(training_vectors, n, dim, pq_dim, n_lists, codebook_out)
      → std::expected<void, GpuError>
    
    encode_pq(vectors, codebook, n, dim, pq_dim, codes_out)
      → std::expected<void, GpuError>
};
```

- **`train_pq_codebook()`**: Runs k-means clustering on a training subset to learn PQ codebooks. Accelerated on GPU.
- **`encode_pq()`**: Encodes vectors as `uint8_t` PQ codes using the trained codebook. The output size is `n * pq_dim * (pq_bits/8)` bytes.

## 12. GpuIngestionPipeline

Pre-processing pipeline for data ingestion:

```cpp
class GpuIngestionPipeline {
    GpuPort& backend_;
    
    normalize_batch(vectors, n, dim)
      → std::expected<void, GpuError>
    
    quantize_batch(vectors, codes_out, n, dim, pq_dim, pq_bits)
      → std::expected<void, GpuError>
};
```

- **`normalize_batch()`**: L2-normalizes a batch of vectors on GPU. Used for cosine metric pre-processing.
- **`quantize_batch()`**: Quantizes vectors to `uint8_t` codes for IVF-PQ index building.

## 13. GpuProfiler

Records kernel execution timing:

```cpp
struct KernelTiming {
    std::string kernel_name;
    std::chrono::microseconds duration;
    size_t work_items;
};

class GpuProfiler {
    void record(kernel, duration, items);
    vector<KernelTiming> recent_timings(max_count = 100);
    size_t total_launches() const noexcept;
    void clear();
};
```

- **Recording**: Called after each kernel launch to store timing data.
- **Retention**: Keeps the last 100 timing records (configurable via `max_count`).
- **Atomic counter**: `total_launches_` tracks lifetime kernel launch count.
- **Enabled via config**: `GpuConfig::enable_profiling` and `emit_kernel_timings`.

## 14. CUDA/HIP/SYCL/Vulkan Stubs

Each non-Metal backend has a stub implementation in its respective directory:

```
backends/
  cuda/CudaBackend.cpp      → implements GpuPort via CUDA Runtime API
  hip/HipBackend.cpp        → implements GpuPort via HIP API (ROCm)
  sycl/SyclBackend.cpp      → implements GpuPort via SYCL 2020
                            + kernels/cosine_fp32.cpp
  vulkan/VulkanBackend.cpp  → implements GpuPort via Vulkan Compute
```

All stubs follow the same pattern: they implement the 13 pure virtual methods of `GpuPort` using their respective APIs. The SYCL backend includes separate kernel source files (`kernels/cosine_fp32.cpp`) for device kernels. Each backend is compiled as a separate static library.

## 15. Fallback Chain

When a GPU search cannot be satisfied, ELIPS falls back through a prioritized chain:

```
GPU CAGRA (GpuGraphIndex)
    │ failure / not available
    ▼
GPU BruteForce (GpuBruteForceIndex)
    │ failure / too small for GPU
    ▼
CPU HNSW (HierarchicalGraphIndex)
    │ fallback
    ▼
CPU ExactIndex (brute-force ground truth)
```

### Fallback Conditions

| Condition | Action |
|-----------|--------|
| No GPU device detected | Skip GPU layer, start at CPU HNSW |
| `GpuPolicy::CpuOnly` | Skip GPU entirely |
| Index too small for GPU benefit | Fall back to CPU (GPU overhead exceeds benefit) |
| `GpuError::InsufficientMemory` | Try next GPU index type, then CPU |
| `GpuError::BackendUnavailable` | Skip to next backend or CPU |
| `GpuError::UnsupportedMetric` | CPU handles all metrics |

The `GpuMetricsSnapshot::fallback_events_total` counter tracks how many times fallback occurred.

## 16. Precision Strategy

| Level | Type | When Used |
|-------|------|-----------|
| **Storage** | FP32 | Always. All vectors stored as `float`. Indices store FP32. |
| **Compute** | FP32 | Default. Full-precision distance computation. |
| **Compute** | FP16 | When `enable_fp16_search` and device `supports_fp16`. Half-precision kernels for 2x throughput. |
| **Quantized** | Int8 | IVF-PQ indices. PQ-encoded vectors use 8-bit codes. |
| **Auto** | Dynamic | `GpuPrecision::Auto` selects FP16 when dimension ≥ threshold and FP16 is supported. |

### GPU Device Info Precision Flags

```
GpuDeviceInfo:
  supports_fp16: bool           → FP16 compute support
  supports_bf16: bool           → bfloat16 support
  supports_int8: bool           → INT8 compute support
  supports_half_precision_search: bool → FP16 search optimization
  peak_tflops_fp32: float       → theoretical FP32 throughput
  peak_tflops_fp16: float       → theoretical FP16 throughput
```

## 17. GpuConfig

Complete GPU configuration structure:

```cpp
struct GpuConfig {
    GpuPolicy policy{Auto};                    // Auto, PreferGpu, RequireGpu, CpuOnly, Specific
    std::string preferred_backend;             // "metal", "cuda", "hip", "sycl", "vulkan"
    int32_t device_index{-1};                  // -1 = auto-select
    IndexBuildMode index_build_mode{GpuBuild_GpuServe};
    GpuIndexAlgorithm algorithm{Auto};          // Auto, CagraGraph, IvfFlat, IvfPq, BruteForce
    size_t device_memory_pool_bytes{0};        // 0 = auto (80% of device memory)
    size_t pinned_host_pool_bytes{256 MB};
    bool use_unified_memory{false};            // false = auto-detect
    size_t default_ef_search_gpu{64};          // beam width for CAGRA search
    size_t dynamic_batch_window_us{500};       // batcher coalescing window
    size_t dynamic_batch_max_size{256};        // max queries per batch
    bool enable_fp16_search{false};            // half-precision compute
    GraphIndexBuildParams graph_params;        // CAGRA build parameters
    IvfPqBuildParams ivf_pq_params;           // IVF-PQ build parameters
    GpuPrecision search_precision{Auto};       // FP32, FP16, Int8, Auto
    bool auto_rebuild_on_startup{false};       // auto-rebuild index on open
    float rebuild_threshold_ratio{0.1f};       // ratio threshold for auto-rebuild
    bool enable_profiling{false};              // kernel timing recording
    bool emit_kernel_timings{false};           // emit timing data
};
```