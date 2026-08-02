# Python API Reference

Comprehensive index of the ELIPS Python API, organized by subsystem.

## Core API

- [**elips module**](core/module.md) — Top-level module
  - `elips.__version__` — Library version string
  - `elips._has_gpu` — Whether GPU support is compiled in

### Database Management

- [**open()**](core/open.md) — Open or create a database
  - `elips.open(path, *, dimension, metric, index, durability, graph_params, gpu_config)` — Convenience constructor
  - `elips.open_with_config(path, config)` — Open with a pre-built Config object
  - Returns `ElipsInstance` (supports context manager `with` protocol)
  - In-memory databases: `elips.open(":memory:", dimension=768)`

- [**ElipsInstance**](core/instance.md) — Database handle
  - `db.vault(name)` — Get or create a named vault
  - `db.list_vaults()` — List all vault names
  - `db.query(eql, bindings={})` — Execute EQL query
  - `db.begin_transaction()` — Start an atomic transaction
  - `db.checkpoint()` — Persist state, truncate WAL
  - `db.close()` — Checkpoint and release lock
  - `db.config` — The active configuration object
  - `db.gpu_info()` — Active accelerator device information or CPU fallback (if GPU enabled)
  - `db.gpu_stats()` — GPU performance metrics (if GPU enabled)

### Configuration

- [**Config**](core/config.md) — Configuration builder
  - `elips.Config()` — Fluent builder
  - `.dimension(n)` — Set vector dimension
  - `.metric("cosine"|"euclidean"|"dot_product")` — Set similarity metric
  - `.index("graph"|"exact")` — Set index backend
  - `.durability("paranoid"|"standard"|"relaxed"|"ephemeral")` — Set durability level
  - `.graph_params(GraphParams(...))` — Set HNSW parameters
  - `.gpu(GpuConfig())` — Set GPU configuration
  - `.dimension_val`, `.metric_val`, `.index_val` — String getters
  - `.metric_enum`, `.index_enum`, `.durability_enum` — Enum getters
  - `.graph_params_val` — GraphParams getter

### Enums

All enums are available as `elips.<EnumName>.<value>`:

| Enum | Module Path | Values |
|---|---|---|
| `Metric` | `elips.Metric` | `cosine`, `euclidean`, `dot_product` |
| `IndexType` | `elips.IndexType` | `graph`, `exact` |
| `Durability` | `elips.Durability` | `paranoid`, `standard`, `relaxed`, `ephemeral` |
| `Comparator` | `elips.Comparator` | `eq`, `ne`, `lt`, `le`, `gt`, `ge` |
| `TokenKind` | `elips.query.TokenKind` | `word`, `number`, `string`, `punct`, `end` |
| `GpuPolicy` | `elips.GpuPolicy` | `auto`, `prefer_gpu`, `require_gpu`, `cpu_only`, `specific` |
| `GpuIndexAlgorithm` | `elips.GpuIndexAlgorithm` | `auto`, `cagra`, `ivf_flat`, `ivf_pq`, `brute_force` |
| `GpuPrecision` | `elips.GpuPrecision` | `fp32`, `fp16`, `int8`, `auto` |
| `GpuError` | `elips.GpuError` | `device_not_found`, `insufficient_memory`, `kernel_launch_failed`, etc. |
| `GraphBuildAlgo` | `elips.GraphBuildAlgo` | `ivf_pq`, `nn_descent`, `iterative_search` |

### GraphParams

- [**GraphParams**](core/graph_params.md) — HNSW tuning
  - `elips.GraphParams(max_connections=16, ef_construction=200, ef_search=50)`
  - `.max_connections` — M parameter
  - `.ef_construction` — Beam width during index construction
  - `.ef_search` — Beam width during search

## Vault API

- [**Vault**](vault/vault.md) — Named vector collection
  - `vault.name` — Vault name (string)
  - `vault.place(vector, data={}, id=None)` — Insert a vector with metadata
  - `vault.place_many(records)` — Batch insert vectors
  - `vault.seek(query, top=10, where=None, threshold=None)` — Nearest-neighbor search
  - `vault.scan(where=None, offset=0, limit=sys.maxsize)` — Metadata-only scan
  - `vault.fetch(id)` — Retrieve a record by ID (returns dict or None)
  - `vault.erase(id)` — Delete a record (returns bool)
  - `vault.count()` — Number of records
  - `vault.info()` — VaultInfo with count, dimension, metric

### Search Result

- [**SearchResult**](vault/search_result.md) — One result hit
  - `hit.id` — RecordID string (UUIDv7)
  - `hit.distance` — Distance to query (float, smaller = more similar)
  - `hit.data` — Metadata payload (dict)

### VaultInfo

- [**VaultInfo**](vault/vault_info.md) — Vault statistics
  - `info.count` — Number of records
  - `info.dimension` — Vector dimensionality
  - `info.metric` — Metric string

## Filter API

- [**Filter**](filter/filter.md) — Metadata predicate engine
  - `elips.Filter()` — Create a filter (default matches everything)
  - `.field(name)` — Select metadata field
  - `.equals(value)` — Equality comparison
  - `.not_equals(value)` — Inequality comparison
  - `.lt(value)`, `.le(value)`, `.gt(value)`, `.ge(value)` — Numeric comparisons
  - `.one_of([values])` — Set membership
  - `.contains(substring)` — String substring match
  - `.and_(other_filter)` — Logical AND with another filter
  - `.or_(other_filter)` — Logical OR with another filter
  - `Filter.not_(filter)` — Logical NOT (static factory)
  - Chained predicates on the same Filter object are AND-ed

## Transaction API

- [**Transaction**](transaction/transaction.md) — Atomic writes
  - `db.begin_transaction()` — Start a transaction (use with `with` statement)
  - `txn.vault(name)` — Get a transaction-scoped vault proxy
  - `txn.commit()` — Apply all buffered writes atomically
  - `txn.rollback()` — Discard all buffered writes
  - Auto-rollback on exception when used as context manager
  - Auto-commit on clean context manager exit

- [**TransactionVault**](transaction/transaction_vault.md) — Transaction-scoped vault
  - `tv.place(vector, data={}, id=None)` — Buffer a write
  - `tv.erase(id)` — Buffer a delete

## EQL API

- [**EQL Query**](eql/query.md) — Query language interface
  - `db.query(eql_string, bindings={})` — Execute EQL, returns list of SearchResult
  - `elips.validate_eql(eql_string)` — Validate EQL syntax (None if valid, raises ParseError)
  - `elips.tokenize_eql(eql_string)` — Tokenize EQL for tooling
  - `token.kind` — TokenKind enum
  - `token.text` — Token text

### EQL Statements

| Statement | Syntax | Description |
|---|---|---|
| `seek` | `seek in <vault> nearest <vector> [top n] [threshold x] [where <filter>] [project fields] yield` | Nearest-neighbor vector search |
| `fetch` | `fetch from <vault> id "<uuid>" yield` | Retrieve record by ID |
| `scan` | `scan in <vault> [where <filter>] [offset n] [limit n] yield` | Metadata-only record scan |
| `place` | `place in <vault> vector [x, y, ...] [data {"k": v}]` | Insert a record |
| `erase` | `erase from <vault> id "<uuid>"` | Delete a record |

## Vector Utilities

- [**Distance Functions**](utilities/distance.md) — Metric computation
  - `elips.distance(metric, a, b)` — Compute distance between two lists
  - `elips.distance("cosine"|"euclidean"|"dot_product", a, b)` — String metric overload
  - `elips.distance(elips.Metric.cosine, a, b)` — Enum metric overload
  - `elips.requires_normalization(metric)` — Check if metric needs L2-normalization
  - `elips.metric_to_string(metric_enum)` — Enum → string
  - `elips.metric_from_string("cosine")` — String → enum

## Error Handling

- [**Exception Hierarchy**](errors/exceptions.md) — Exception types
  - `elips.ElipsError` — Base of all ELIPS errors (inherits `Exception`)
  - `elips.DimensionMismatch` — Vector dimension mismatch
  - `elips.InvalidVector` — NaN, Inf in vector
  - `elips.ConfigError` — Configuration conflict
  - `elips.NotFound` — Missing record or vault
  - `elips.StorageError` — IO or persistence failure
  - `elips.LockConflict` — Exclusive writer lock held
  - `elips.ParseError` — Malformed EQL syntax

## GPU Configuration (conditional on `_has_gpu`)

- [**GpuConfig**](gpu/config.md) — GPU settings
  - `elips.GpuConfig()` — GPU configuration builder
  - `.policy` — `GpuPolicy` enum
  - `.build_mode` — `IndexBuildMode` enum
  - `.algorithm` — `GpuIndexAlgorithm` enum
  - `.precision` — `GpuPrecision` enum
  - `.graph_params` — Graph-oriented GPU build parameters
  - `.ivf_pq_params` — IVF and IVF-PQ tuning parameters
  - `.device_memory_pool_mb` — Pool size in MB
  - `.fp16_search` — Half-precision flag (bool)
  - `.max_batch_size` — Maximum queries per batch
  - `.ef_search` — GPU beam width

  Common selections:
  - `brute_force` — exact GPU scan
  - `ivf_flat` — coarse-quantized ANN on GPU
  - `ivf_pq` — compressed ANN on GPU
  - `cagra` — Elips' graph-oriented GPU path

- [**GpuDeviceInfo**](gpu/device_info.md) — Active accelerator device snapshot
  - `info.name` — Device name
  - `info.vendor` — Vendor name
  - `info.memory_gb` — Total device memory
  - `info.peak_tflops_fp32` / `info.peak_tflops_fp16` — Compute throughput
  - `info.supports_cagra`, `info.supports_dynamic_batching`, `info.supports_half_precision_search`
  - `info.supports_bf16`, `info.compute_capability_major`
  - `info.host_to_device_bandwidth_gb_s` — Transfer bandwidth

- [**GpuIndexBuildParams**](gpu/build_params.md) — GPU index build parameters
  - `elips.GpuIndexBuildParams()` — Variant wrapper for build parameters
  - `.params` — Underlying GraphIndexBuildParams, IvfPqBuildParams, etc.

- [**GraphIndexBuildParams**](gpu/graph_build_params.md) — GPU graph build config
  - `elips.GraphIndexBuildParams()`
  - `.graph_degree`, `.build_algo`

- [**IvfPqBuildParams**](gpu/ivf_pq_build_params.md) — IVF-PQ build config
  - `elips.IvfPqBuildParams()`
  - `.n_lists`, `.pq_dim`

## Getting Started

- [Installation & build](../python/getting-started/installation.md)
- [Quickstart](../python/getting-started/quickstart.md)
- [Python SDK overview](../../python_sdk.md)
- [Cookbook](../../cookbook.md) — Worked usage patterns
