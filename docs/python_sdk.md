# Python SDK

Build and import:

```bash
cmake -S . -B build -G Ninja -DELIPS_BUILD_PYTHON=ON
cmake --build build --target elips_pymodule
export PYTHONPATH=$PWD/bindings/python
```

```python
import elips
```

## Open

```python
db = elips.open("/data/vectors", dimension=1536, metric="cosine")  # or "euclidean", "dot_product"
db = elips.open(":memory:", dimension=768)        # ephemeral
db = elips.open("/data/vectors", index="exact")   # brute-force backend
```

Use it as a context manager to checkpoint + release the lock on exit:

```python
with elips.open("/data/vectors", dimension=384) as db:
    ...
```

## Configuration

Build configuration fluently with `Config`:

```python
from elips import Config, GraphParams, Metric, IndexType, Durability

config = (Config()
    .dimension(1536)
    .metric("cosine")
    .index("graph")
    .graph_params(GraphParams(max_connections=32, ef_construction=400, ef_search=80))
    .durability("standard"))

db = elips.open_with_config("/data/vectors", config)

# Inspect configuration with enum getters
print(config.metric_enum)   # Metric.cosine
print(config.index_enum)    # IndexType.graph
print(config.durability_enum)  # Durability.standard
```

## Vaults & writes

```python
docs = db.vault("documents")
rid = docs.place([0.1, 0.2, ...], {"title": "Doc", "year": 2024})
docs.place([0.3, 0.4, ...], {"title": "Other"}, id="custom-uuid")
```

## Search

```python
for hit in docs.seek([0.1, 0.2, ...], top=10):
    print(hit.id, hit.distance, hit.data)

# filtered
flt = elips.Filter().field("year").gte(2023).field("category").equals("tech")
docs.seek(query, top=10, where=flt)

# range search
docs.seek(query, top=1000, where=flt, threshold=0.25)
```

## Filters

```python
f = (elips.Filter()
     .field("category").equals("tech")
     .field("score").gte(0.8)
     .field("country").one_of(["US", "GB"])
     .field("title").contains("intro"))

either = elips.Filter().field("tier").equals("pro").or_(
         elips.Filter().field("year").gte(2023))
negated = elips.Filter.not_(elips.Filter().field("draft").equals(True))
```

## Transactions

```python
with db.begin_transaction() as txn:
    v = txn.vault("docs")
    v.place([1.0, 2.0], {"title": "A"})
    v.place([3.0, 4.0], {"title": "B"})
    # Auto-committed on clean exit; rolled back on exception
```

## EQL

```python
results = db.query("""
    seek in documents nearest $q top 20
    where category = "finance" and year >= 2022
    project title, author yield
""", bindings={"q": query_vector})

# Validate EQL without executing
elips.validate_eql("seek in docs nearest [1.0] top 5 yield")

# Tokenize EQL for tooling
tokens = elips.tokenize_eql("seek in docs nearest $q top 5 yield")
```

## EQL Tooling (Validate & Tokenize)

```python
# Validate EQL syntax without executing
elips.validate_eql("seek in docs nearest [1.0] top 5 yield")  # None if valid

# Tokenize EQL source for tooling / syntax highlighting
tokens = elips.tokenize_eql("seek in docs nearest $q top 5")
for t in tokens:
    print(t.kind, t.text)  # TokenKind.word, "seek" etc.
```

## Vector Utilities

```python
# Compute distance directly
dist = elips.distance("cosine", [1.0, 0.0], [0.0, 1.0])  # -> 1.0

# Also accepts enum values
dist = elips.distance(elips.Metric.cosine, [1.0, 0.0], [0.0, 1.0])

# Check if normalization is needed
elips.requires_normalization("cosine")  # -> True
elips.requires_normalization(elips.Metric.euclidean)  # -> False

# Enum <-> string conversion
elips.metric_to_string(elips.Metric.dot_product)  # -> "dot_product"
elips.metric_from_string("euclidean")  # -> Metric.euclidean
```

## Fetch, scan, erase, maintenance

```python
record = docs.fetch(rid)             # dict | None
rows = docs.scan(where=flt)          # list of {"id", "data"}
docs.erase(rid)
db.checkpoint()
db.list_vaults()
```

## GPU Configuration

```python
from elips import GpuConfig, GpuPolicy, GpuIndexAlgorithm, GpuPrecision

gpu = GpuConfig()
gpu.policy = GpuPolicy.prefer_gpu
gpu.algorithm = GpuIndexAlgorithm.brute_force
gpu.precision = GpuPrecision.fp16
gpu.device_memory_pool_mb = 512

config = Config().dimension(1536).gpu(gpu)
db = elips.open_with_config("/data/vectors", config)

# Inspect GPU info
info = db.gpu_info()
print(info.name, info.vendor, info.memory_gb)
print(info.peak_tflops_fp16, info.supports_cagra)

# GPU metrics
stats = db.gpu_stats()
print(stats.search_p50_latency_us, stats.batch_avg_size)
```

## Error Handling

All ELIPS errors derive from `elips.ElipsError`:

```python
try:
    docs.place([1.0, 2.0], {"key": "value"})
except elips.DimensionMismatch:
    print("Vector dimension doesn't match vault")
except elips.InvalidVector:
    print("Vector contains NaN or Inf")
except elips.ConfigError:
    print("Configuration conflict")
except elips.NotFound:
    print("Record/vault not found")
except elips.StorageError:
    print("IO/persistence failure")
except elips.LockConflict:
    print("Another writer holds the lock")
except elips.ParseError:
    print("Malformed EQL query")
except elips.ElipsError:
    print("Some other ELIPS error")
```

All public APIs are fully typed (`py.typed` + `_core.pyi` with complete stubs). IDE autocompletion (PyCharm, VSCode, Pylance, Pyright, MyPy) is supported.

## Enums Reference

| Enum | Values |
|---|---|
| `Metric` | `cosine`, `euclidean`, `dot_product` |
| `IndexType` | `graph`, `exact` |
| `Durability` | `paranoid`, `standard`, `relaxed`, `ephemeral` |
| `Comparator` | `eq`, `ne`, `lt`, `le`, `gt`, `ge` |
| `GpuPolicy` | `auto`, `prefer_gpu`, `require_gpu`, `cpu_only`, `specific` |
| `IndexBuildMode` | `gpu_build_cpu_serve`, `gpu_build_gpu_serve`, `hybrid` |
| `GpuIndexAlgorithm` | `auto`, `cagra`, `ivf_flat`, `ivf_pq`, `brute_force` |
| `GpuPrecision` | `fp32`, `fp16`, `int8`, `auto` |
| `GpuError` | `device_not_found`, `insufficient_memory`, `kernel_launch_failed`, `transfer_failed`, `index_build_failed`, `unsupported_metric`, `initialization_failed`, `backend_unavailable` |
| `GraphBuildAlgo` | `ivf_pq`, `nn_descent`, `iterative_search` |
| `TokenKind` | `word`, `number`, `string`, `punct`, `end` |