# Configuration & Enums Reference

Complete reference for the `Config` builder, every enum, and every configuration-related class in the ELIPS Python API.

## Configuration

### `Config`

```python
class Config
```

Fluent builder for database configuration. Every setter method returns `self` for chaining.

**Constructor:**

```python
def __init__(self) -> None
```

Creates an empty configuration.

#### Builder Methods

##### `Config.dimension()`

```python
def dimension(self, dim: int) -> Config
```

Set the vector dimension. Required for new databases.

##### `Config.metric()`

```python
def metric(self, metric: str) -> Config
```

Set the similarity metric by string name.

| Argument | Meaning |
|---|---|
| `"cosine"` | Cosine similarity (requires L2-normalized vectors) |
| `"euclidean"` | Euclidean (L2) distance |
| `"dot_product"` | Inner product (negative for ranking) |

##### `Config.index()`

```python
def index(self, type: str) -> Config
```

Set the index backend.

| Argument | Meaning |
|---|---|
| `"graph"` | Hierarchical Navigable Small World (HNSW) — good for high-dimensional search |
| `"exact"` | Brute-force linear scan — exact results but slower for large datasets |

##### `Config.graph_params()`

```python
def graph_params(self, params: GraphParams) -> Config
```

Set tunable parameters for the graph index.

##### `Config.durability()`

```python
def durability(self, level: str) -> Config
```

Set the durability level.

| Argument | Meaning |
|---|---|
| `"paranoid"` | fsync after every write |
| `"standard"` | Periodic fsync (default) |
| `"relaxed"` | OS handles flushing |
| `"ephemeral"` | No persistence (":memory:"-like semantics) |

##### `Config.gpu()`

```python
def gpu(self, config: GpuConfig) -> Config
```

Attach GPU configuration. Only available when the extension is built with GPU support.

#### Read-Only Properties

| Property | Type | Description |
|---|---|---|
| `dimension_val` | `int` | The configured dimension |
| `metric_val` | `str` | Metric as string (legacy alias for `metric_enum`) |
| `metric_enum` | `Metric` | The `Metric` enum value |
| `index_val` | `str` | Index type as string: `"graph"` or `"exact"` |
| `index_enum` | `IndexType` | The `IndexType` enum value |
| `graph_params_val` | `GraphParams` | The configured graph parameters |
| `durability_enum` | `Durability` | The `Durability` enum value |
| `gpu_val` | `GpuConfig` or `None` | GPU config if set, else `None` |

**Example:**

```python
from elips import Config, GraphParams

config = (Config()
    .dimension(1536)
    .metric("cosine")
    .index("graph")
    .graph_params(GraphParams(max_connections=32, ef_construction=400))
    .durability("standard"))

print(config.dimension_val)     # 1536
print(config.metric_val)        # "cosine"
print(config.metric_enum)       # Metric.cosine
print(config.index_val)         # "graph"
print(config.index_enum)        # IndexType.graph
print(config.graph_params_val)  # <GraphParams M=32 ef_c=400 ef_s=50>
print(config.durability_enum)   # Durability.standard
print(config.gpu_val)           # None (no GPU configured)
```

### `GraphParams`

```python
class GraphParams
```

Tunable parameters for the Hierarchical Graph Index (HNSW).

**Constructor:**

```python
def __init__(
    self,
    max_connections: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50,
) -> None
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_connections` | `int` | `16` | Maximum number of connections per node (M). Higher = more accurate, uses more memory |
| `ef_construction` | `int` | `200` | Beam width during index construction. Higher = better recall, slower build |
| `ef_search` | `int` | `50` | Beam width during search. Higher = better recall, slower search |

| Property | Type | Read/Write |
|---|---|---|
| `max_connections` | `int` | read/write |
| `ef_construction` | `int` | read/write |
| `ef_search` | `int` | read/write |

**Example:**

```python
params = elips.GraphParams(max_connections=32, ef_construction=400, ef_search=100)
print(params.max_connections)   # 32
print(params.ef_construction)   # 400
params.ef_search = 80           # Modify after construction
```

## Core Enums

### `Metric`

```python
class Metric(IntEnum)
```

Similarity metrics supported by ELIPS.

| Member | Value | Description |
|---|---|---|
| `cosine` | — | Cosine similarity. Vectors should be L2-normalized |
| `euclidean` | — | Euclidean (L2) distance |
| `dot_product` | — | Inner product (negated for ranking: lower = more similar) |

### `IndexType`

```python
class IndexType(IntEnum)
```

Index backends.

| Member | Description |
|---|---|
| `graph` | Hierarchical Navigable Small World (HNSW) |
| `exact` | Brute-force linear scan |

### `Durability`

```python
class Durability(IntEnum)
```

Write durability levels, trading throughput against crash safety.

| Member | Description |
|---|---|
| `paranoid` | `fsync` after every write. Safest, slowest |
| `standard` | Periodic `fsync`. Good balance (default) |
| `relaxed` | Delegate to OS page cache flushing |
| `ephemeral` | No persistence. Data lost on process exit |

### `Comparator`

```python
class Comparator(IntEnum)
```

Metadata comparison operators. Used internally; not typically needed by users.

| Member | Operation |
|---|---|
| `eq` | `==` |
| `ne` | `!=` |
| `lt` | `<` |
| `le` | `<=` |
| `gt` | `>` |
| `ge` | `>=` |

## GPU Enums

All GPU types are only available when the extension is compiled with GPU support. Check `elips._has_gpu` at runtime.

### `GpuPolicy`

```python
class GpuPolicy(IntEnum)
```

GPU usage policy.

| Member | Description |
|---|---|
| `auto` | Let ELIPS decide based on dataset size and hardware |
| `prefer_gpu` | Use GPU if available, fall back to CPU |
| `require_gpu` | Fail if no GPU is available |
| `cpu_only` | Never use GPU |
| `specific` | Use only the device specified by `device_index` in `GpuConfig` |

### `GpuIndexAlgorithm`

```python
class GpuIndexAlgorithm(IntEnum)
```

GPU index algorithm selection.

| Member | Description |
|---|---|
| `auto` | Let ELIPS choose the best algorithm |
| `cagra` | CUDA-Accelerated Graph (NVIDIA) |
| `ivf_flat` | Inverted File with flat quantization |
| `ivf_pq` | Inverted File with Product Quantization |
| `brute_force` | Exhaustive GPU-accelerated search |

### `GpuPrecision`

```python
class GpuPrecision(IntEnum)
```

GPU computation precision.

| Member | Description |
|---|---|
| `fp32` | 32-bit floating point (highest accuracy) |
| `fp16` | 16-bit floating point (2x throughput, slight accuracy loss) |
| `int8` | 8-bit integer quantization (fastest, lower accuracy) |
| `auto` | Let ELIPS choose based on hardware capabilities |

### `GpuError`

```python
class GpuError(IntEnum)
```

GPU error codes returned by the runtime.

| Member | Description |
|---|---|
| `device_not_found` | No compatible GPU device detected |
| `insufficient_memory` | Not enough GPU memory for the requested operation |
| `kernel_launch_failed` | GPU kernel failed to launch |
| `transfer_failed` | Host-to-device or device-to-host data transfer failed |
| `index_build_failed` | GPU index construction failed |
| `unsupported_metric` | The selected metric is not supported on GPU |
| `initialization_failed` | GPU backend initialization failed |
| `backend_unavailable` | Requested GPU backend (CUDA/Metal/etc.) is not available |

### `IndexBuildMode`

```python
class IndexBuildMode(IntEnum)
```

Controls where index construction and serving happen.

| Member | Description |
|---|---|
| `gpu_build_cpu_serve` | Build on GPU, serve queries from CPU |
| `gpu_build_gpu_serve` | Build and serve both on GPU |
| `hybrid` | Hybrid CPU/GPU strategy |

### `GraphBuildAlgo`

```python
class GraphBuildAlgo(IntEnum)
```

Algorithm for GPU-accelerated graph index construction.

| Member | Description |
|---|---|
| `ivf_pq` | IVF-PQ based construction |
| `nn_descent` | NN-Descent (nearest neighbor descent) |
| `iterative_search` | Iterative refinement search |

## GPU Configuration Classes

### `GpuConfig`

```python
class GpuConfig
```

GPU acceleration configuration. All fields are read/write.

| Field | Type | Default | Description |
|---|---|---|---|
| `policy` | `GpuPolicy` | `GpuPolicy.auto` | GPU usage policy |
| `preferred_backend` | `str` | `""` | Preferred backend name (`"cuda"`, `"metal"`, etc.) |
| `device_index` | `int` | `0` | Which GPU device to use (when `policy == specific`) |
| `build_mode` | `IndexBuildMode` | `IndexBuildMode.gpu_build_cpu_serve` | Where index is built vs. served |
| `algorithm` | `GpuIndexAlgorithm` | `GpuIndexAlgorithm.auto` | Index algorithm |
| `device_memory_pool_mb` | `int` | — | GPU memory pool size in megabytes (property: converts to/from bytes) |
| `fp16_search` | `bool` | `False` | Enable FP16 for search queries |
| `unified_memory` | `bool` | `False` | Use unified memory (when supported) |
| `batch_window_us` | `int` | — | Dynamic batching time window in microseconds |
| `max_batch_size` | `int` | — | Maximum queries to coalesce into one GPU call |
| `ef_search` | `int` | — | Default ef_search for GPU HNSW queries |
| `precision` | `GpuPrecision` | `GpuPrecision.auto` | Computation precision |
| `profiling` | `bool` | `False` | Enable GPU kernel profiling |
| `graph_params` | `GraphIndexBuildParams` | — | Graph index build parameters |
| `ivf_pq_params` | `IvfPqBuildParams` | — | IVF-PQ build parameters |

**Example:**

```python
gpu = elips.GpuConfig()
gpu.policy = elips.GpuPolicy.prefer_gpu
gpu.algorithm = elips.GpuIndexAlgorithm.cagra
gpu.device_memory_pool_mb = 512
gpu.fp16_search = True
gpu.max_batch_size = 128
gpu.precision = elips.GpuPrecision.fp16

config = Config().dimension(1536).gpu(gpu)
```

### `GpuDeviceInfo`

```python
class GpuDeviceInfo
```

Information about a detected GPU device. All fields are read-only.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Device name (e.g., `"Apple M2 Pro"`, `"NVIDIA RTX 4090"`) |
| `vendor` | `str` | Vendor name |
| `backend` | `str` | Backend name: `"metal"`, `"cuda"`, `"hip"`, `"sycl"`, `"vulkan"` |
| `device_index` | `int` | Device index within the backend |
| `total_memory_bytes` | `int` | Total device memory in bytes |
| `free_memory_bytes` | `int` | Free device memory in bytes |
| `memory_gb` | `float` | Total device memory in gigabytes (property) |
| `has_unified_memory` | `bool` | Whether unified (host+device) memory is available |
| `supports_fp16` | `bool` | FP16 computation support |
| `supports_bf16` | `bool` | BF16 computation support |
| `supports_int8` | `bool` | INT8 computation support |
| `supports_cagra` | `bool` | CAGRA graph index support |
| `supports_ivf_pq` | `bool` | IVF-PQ index support |
| `supports_dynamic_batching` | `bool` | Dynamic query batching support |
| `supports_half_precision_search` | `bool` | FP16 search support |
| `compute_capability_major` | `int` | Major compute capability (CUDA CC) |
| `compute_capability_minor` | `int` | Minor compute capability |
| `max_threads_per_block` | `int` | Max threads per compute block |
| `multiprocessor_count` | `int` | Number of multiprocessors |
| `shared_memory_per_block_bytes` | `int` | Shared memory per block in bytes |
| `l2_cache_bytes` | `int` | L2 cache size in bytes |
| `peak_tflops_fp32` | `float` | Peak FP32 throughput (TFLOPS) |
| `peak_tflops_fp16` | `float` | Peak FP16 throughput (TFLOPS) |
| `host_to_device_bandwidth_gb_s` | `float` | Host → Device bandwidth (GB/s) |
| `device_to_host_bandwidth_gb_s` | `float` | Device → Host bandwidth (GB/s) |

### `GpuMetricsSnapshot`

```python
class GpuMetricsSnapshot
```

Snapshot of GPU runtime metrics. All fields are read-only.

| Field | Type | Description |
|---|---|---|
| `backend` | `str` | Backend name |
| `device_name` | `str` | Device name |
| `device_memory_used_bytes` | `int` | Currently used device memory |
| `device_memory_total_bytes` | `int` | Total device memory |
| `index_build_count` | `int` | Number of GPU index builds |
| `index_build_time_total_ms` | `int` | Total time spent building indexes |
| `index_build_speedup_vs_cpu_avg` | `float` | Average speedup vs CPU index build |
| `search_kernel_launches_total` | `int` | Total search kernel launches |
| `search_p50_latency_us` | `int` | P50 search latency in microseconds |
| `search_p99_latency_us` | `int` | P99 search latency in microseconds |
| `batch_avg_size` | `float` | Average dynamic batch size |
| `batch_coalescing_ratio` | `float` | Ratio of coalesced queries to total |
| `fp16_search_enabled` | `bool` | Whether FP16 search is active |
| `fallback_events_total` | `int` | Number of GPU→CPU fallback events |
| `kernel_errors_total` | `int` | Total kernel error count |
| `pinned_memory_pool_used_bytes` | `int` | Pinned (page-locked) memory in use |

### `GraphIndexBuildParams`

```python
class GraphIndexBuildParams
```

Parameters for GPU graph index construction. All fields are read/write.

| Field | Type | Description |
|---|---|---|
| `intermediate_graph_degree` | `int` | Degree of intermediate graph during construction |
| `graph_degree` | `int` | Final graph degree |
| `build_algo` | `GraphBuildAlgo` | Construction algorithm |
| `nn_descent_iterations` | `int` | Number of NN-Descent iterations |
| `compression_ratio` | `float` | Compression ratio for quantized representation |

### `IvfPqBuildParams`

```python
class IvfPqBuildParams
```

Parameters for IVF-PQ index construction. All fields are read/write.

| Field | Type | Description |
|---|---|---|
| `n_lists` | `int` | Number of inverted lists (clusters) |
| `pq_dim` | `int` | Product Quantization sub-vector dimension |
| `pq_bits` | `int` | Bits per PQ code |
| `add_data_on_build` | `bool` | Whether to add data vectors during build |
| `kmeans_n_iters` | `int` | Number of k-means iterations for clustering |
| `kmeans_trainset_fraction` | `float` | Fraction of dataset used for k-means training |

### `GpuIndexBuildParams`

```python
class GpuIndexBuildParams
```

Variant wrapper for GPU index build parameters.

| Field | Type | Description |
|---|---|---|
| `params` | `Union[GraphIndexBuildParams, IvfPqBuildParams]` | The specific build parameters |

### `KernelTiming`

```python
class KernelTiming
```

Recorded GPU kernel timing.

| Field | Type | Description |
|---|---|---|
| `kernel_name` | `str` | Name of the GPU kernel |
| `work_items` | `int` | Number of work items processed |
| `duration_us` | `int` | Duration in microseconds (property) |

## Configuration Examples

### Minimal Config

```python
from elips import Config

config = Config().dimension(384)
db = elips.open_with_config(":memory:", config)
```

### Production Config

```python
from elips import Config, GraphParams

config = (Config()
    .dimension(1536)
    .metric("cosine")
    .index("graph")
    .graph_params(GraphParams(
        max_connections=32,
        ef_construction=400,
        ef_search=80,
    ))
    .durability("standard"))
```

### GPU-Accelerated Config

```python
from elips import Config, GpuConfig, GpuPolicy, GpuIndexAlgorithm, GpuPrecision

gpu = GpuConfig()
gpu.policy = GpuPolicy.prefer_gpu
gpu.algorithm = GpuIndexAlgorithm.cagra
gpu.precision = GpuPrecision.fp16
gpu.device_memory_pool_mb = 2048
gpu.fp16_search = True
gpu.max_batch_size = 512

config = (Config()
    .dimension(1536)
    .metric("cosine")
    .index("graph")
    .gpu(gpu)
    .durability("standard"))

db = elips.open_with_config("/data/vectors", config)
```