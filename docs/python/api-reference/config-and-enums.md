# Config & Enums

`Config` is the authoritative Python configuration builder for advanced ELIPS
features.

## `Config`

```python
config = (
    elips.Config()
    .dimension(128)
    .metric("cosine")
    .index("graph")
    .durability("standard")
    .access_mode("read_write")
    .segmented_storage(True)
    .metadata_acceleration(True)
    .auto_text_embedder(True)
)
```

### Builder Methods

- `dimension(dim: int)`
- `metric(metric: str)`
- `index(type: str)`
- `graph_params(params: GraphParams)`
- `durability(level: str)`
- `access_mode(mode: str)`
- `segmented_storage(enabled: bool)`
- `metadata_acceleration(enabled: bool)`
- `auto_text_embedder(enabled: bool)`
- `local_text_embedder(config=LocalEmbedderConfig(...))`
- `text_embedder(embedder, provider="...", model="...", revision="", dimension=0)`
- `gpu(config)` when GPU support is built

### Read-Only Properties

- `dimension_val`
- `metric_val` / `metric_enum`
- `index_val` / `index_enum`
- `graph_params_val`
- `durability_enum`
- `access_mode_val` / `access_mode_enum`
- `segmented_storage_enabled`
- `metadata_acceleration_enabled`
- `auto_text_embedder_enabled`
- `has_text_embedder`
- `text_embedder_info`
- `gpu_val`

## Local Embedding Types

### `LocalEmbedderConfig`

```python
local = elips.LocalEmbedderConfig(
    model="default",
    revision="v1",
    storage_path="/tmp/elips/default.localembed",
    dimension=128,
)
```

Use this with `Config.local_text_embedder(local)` or `elips.open(..., embedder=local)`.

### `TextEmbedderInfo`

Resolved metadata for the current or expected text embedder:

- `kind`
- `provider`
- `model`
- `revision`
- `backend`
- `dimension`
- `fingerprint`
- `storage_path`
- `rehydratable`
- `loaded`
- `auto_attached`

## `GraphParams`

```python
params = elips.GraphParams(
    max_connections=32,
    ef_construction=400,
    ef_search=100,
)
```

These tune the graph index when `index("graph")` is used.

## Core Enums

### `Metric`

- `cosine`
- `euclidean`
- `dot_product`

### `IndexType`

- `graph`
- `exact`

### `Durability`

- `paranoid`
- `standard`
- `relaxed`
- `ephemeral`

### `AccessMode`

- `read_write`
- `read_only`

### `Comparator`

- `eq`
- `ne`
- `lt`
- `le`
- `gt`
- `ge`

### `QueryStrategy`

Planner output enum used by `QueryPlan`:

- `ann_index`
- `exact_candidates`
- `full_scan`
- `text_probe`
- `hybrid_fusion`

## Query Planning Types

### `QueryPlan`

Returned by `Vault.explain_seek(...)`.

Fields:

- `strategy`
- `candidate_count`
- `metadata_accelerated`
- `gpu_index`
- `index_type`

## Lineage Types

### `DocumentAttachment`

- `text`
- `uri`
- `mime_type`

### `ChunkInfo`

- `document_key`
- `ordinal`
- `char_start`
- `char_end`

### `EmbeddingLineage`

- `provider`
- `model`
- `revision`
- `attributes`

## GPU Types

When the extension is built with GPU support, ELIPS also exposes:

- `GpuConfig`
- `GpuPolicy`
- `IndexBuildMode`
- `GpuIndexAlgorithm`
- `GpuPrecision`
- `GpuError`
- `GraphBuildAlgo`
- `GraphIndexBuildParams`
- `IvfPqBuildParams`
- `GpuDeviceInfo`
- `GpuMetricsSnapshot`

Use `elips._has_gpu` to detect whether GPU bindings were compiled into the
extension module. Use `elips.GpuDeviceInfo()` or `db.gpu_info()` to inspect the
active runtime device, which may still be the CPU fallback when no GPU backend
is selected.

### `GpuConfig` Example

```python
gpu = elips.GpuConfig()
gpu.policy = elips.GpuPolicy.prefer_gpu
gpu.build_mode = elips.IndexBuildMode.gpu_build_gpu_serve
gpu.algorithm = elips.GpuIndexAlgorithm.ivf_pq
gpu.precision = elips.GpuPrecision.auto
gpu.ef_search = 48

gpu.ivf_pq_params.n_lists = 2048
gpu.ivf_pq_params.pq_dim = 48
gpu.ivf_pq_params.pq_bits = 8
gpu.ivf_pq_params.kmeans_n_iters = 20

config = (
    elips.Config()
    .dimension(384)
    .metric("cosine")
    .index("exact")
    .gpu(gpu)
)
```

### `GpuIndexAlgorithm`

- `auto`
  - Lets Elips choose a GPU path from backend capabilities. Today it prefers
    the IVF path when supported and otherwise falls back toward brute force.
- `brute_force`
  - Exact GPU scan.
- `ivf_flat`
  - Coarse-list routing on GPU followed by exact reranking of gathered
    candidates.
- `ivf_pq`
  - Coarse-list routing plus residual PQ approximation on GPU, followed by
    exact reranking of a shortlist.
- `cagra`
  - Elips' graph-oriented GPU path. In the current implementation this keeps a
    CPU graph mirror as the authoritative topology and mirrors vectors on
    device; it is not vendor-specific CAGRA kernels.

Recommended pairings:

- `index("exact")` with `brute_force`, `ivf_flat`, or `ivf_pq`
- `index("graph")` with `cagra`

### `IndexBuildMode`

- `gpu_build_cpu_serve`
- `gpu_build_gpu_serve`
- `hybrid`

The full enum is available in Python, but today Elips already keeps CPU and GPU
state synchronized for the non-brute-force GPU algorithms through a hybrid
runtime path.

### `IvfPqBuildParams`

```python
ivf = elips.IvfPqBuildParams()
ivf.n_lists = 1024
ivf.pq_dim = 48
ivf.pq_bits = 8
ivf.kmeans_n_iters = 20

gpu = elips.GpuConfig()
gpu.algorithm = elips.GpuIndexAlgorithm.ivf_pq
gpu.ivf_pq_params = ivf
```
