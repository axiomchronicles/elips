# Quick Start

This walkthrough uses the current ELIPS v2 Python surface: native document
ingestion, text and hybrid retrieval, segmented persistence, and the modern
`connect()` wrapper.

## 1. Build The Python Module

```bash
cmake -S . -B build -G Ninja -DELIPS_BUILD_PYTHON=ON
cmake --build build --target elips_pymodule
export PYTHONPATH=$PWD/bindings/python
```

You can smoke-test the Python bindings with:

```bash
PYTHONPATH=bindings/python python3 bindings/python/test.py
PYTHONPATH=bindings/python python3 bindings/python/test.py --gpu-algorithm ivf_pq
```

## 2. Open A Database

Low-level open:

```python
import elips

db = elips.open(":memory:", dimension=2, metric="cosine")
```

Full configuration:

```python
config = (
    elips.Config()
    .dimension(2)
    .metric("cosine")
    .segmented_storage(True)
    .metadata_acceleration(True)
    .auto_text_embedder(True)
)

db = elips.open_with_config("/tmp/elips-quickstart", config)
```

New databases attach the built-in local embedder automatically when
`auto_text_embedder(True)` is left enabled. Use `LocalEmbedderConfig` when you
want to pin the model name, revision, or storage path explicitly, or
`Config.text_embedder(...)` when you want to attach a live Python callable.

## 3. Optional GPU Configuration

If the extension was built with GPU support, the Python bindings expose
`GpuConfig`, `GpuIndexAlgorithm`, `GraphIndexBuildParams`, and
`IvfPqBuildParams`.

Exact GPU scan:

```python
gpu = elips.GpuConfig()
gpu.policy = elips.GpuPolicy.require_gpu
gpu.build_mode = elips.IndexBuildMode.gpu_build_gpu_serve
gpu.algorithm = elips.GpuIndexAlgorithm.brute_force

config = (
    elips.Config()
    .dimension(384)
    .metric("cosine")
    .index("exact")
    .gpu(gpu)
)
```

IVF-Flat on GPU:

```python
gpu = elips.GpuConfig()
gpu.policy = elips.GpuPolicy.prefer_gpu
gpu.algorithm = elips.GpuIndexAlgorithm.ivf_flat
gpu.ivf_pq_params.n_lists = 1024
gpu.ef_search = 32

config = (
    elips.Config()
    .dimension(384)
    .metric("cosine")
    .index("exact")
    .gpu(gpu)
)
```

IVF-PQ on GPU:

```python
gpu = elips.GpuConfig()
gpu.policy = elips.GpuPolicy.prefer_gpu
gpu.algorithm = elips.GpuIndexAlgorithm.ivf_pq
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

Graph-oriented GPU path:

```python
gpu = elips.GpuConfig()
gpu.policy = elips.GpuPolicy.prefer_gpu
gpu.algorithm = elips.GpuIndexAlgorithm.cagra
gpu.graph_params.graph_degree = 48
gpu.graph_params.intermediate_graph_degree = 96

config = (
    elips.Config()
    .dimension(384)
    .metric("cosine")
    .index("graph")
    .gpu(gpu)
)
```

Notes:

- Use `index("exact")` with `brute_force`, `ivf_flat`, and `ivf_pq`.
- Use `index("graph")` with `cagra`.
- In the current Elips runtime, the non-brute-force GPU algorithms serve through
  a hybrid wrapper that keeps a CPU mirror synchronized with the GPU leaf.
- `GpuIndexAlgorithm.cagra` currently selects Elips' graph-oriented GPU path,
  not the FAISS/NVIDIA-specific CAGRA kernel stack.

## 4. Ingest Documents

```python
docs = db.vault("documents")

chunk = elips.ChunkInfo()
chunk.document_key = "doc-alpha"
chunk.ordinal = 0
chunk.char_start = 0
chunk.char_end = 17

rid = docs.place_document(
    "alpha design note",
    {"kind": "design"},
    chunk=chunk,
)
docs.place_document("beta incident runbook", {"kind": "ops"})
```

You can also attach document metadata to explicit vectors:

```python
attachment = elips.DocumentAttachment(text="gamma appendix", mime_type="text/plain")
docs.place(
    [1.0, 0.0],
    {"kind": "appendix"},
    document=attachment,
)
```

## 5. Query

Vector search:

```python
hits = docs.seek([1.0, 0.0], top=2)
```

Text-first search:

```python
hits = docs.seek_text("alpha", top=2)
print(hits[0].document.text)
```

Hybrid search:

```python
hits = docs.seek_hybrid([0.0, 1.0], "alpha", top=2, lexical_weight=0.35)
```

Planner inspection:

```python
where = elips.Filter().field("kind").equals("design")
plan = docs.explain_seek([1.0, 0.0], top=1, where=where, has_text_component=True)
print(plan.strategy.name, plan.metadata_accelerated, plan.candidate_count)
```

## 6. Scan, Fetch, And Maintain

```python
record = docs.fetch(rid)
print(record["document"].text, record["chunk"].document_key)

rows = docs.scan(where=elips.Filter().field("kind").equals("design"))
print(rows[0]["id"], rows[0]["data"])

db.checkpoint()
db.compact()
db.close()
```

## 7. Shared Read-Only Reopen

```python
reader = elips.open("/tmp/elips-quickstart", access_mode="read_only")
print(reader.vault("documents").seek_text("beta", top=1)[0].data)
reader.close()
```

Read-only mode requires an existing database and rejects writes.

## 8. Modern Wrapper

If you prefer a more Pythonic, typed surface:

```python
engine = elips.connect(
    "/tmp/elips-modern",
    dimension=2,
    metric="cosine",
)
arena = engine.arena("documents")

keys = arena.ingest(
    texts=["alpha design note", "beta incident runbook"],
    meta=[{"kind": "design"}, {"kind": "ops"}],
)
hits = arena.probe_text("alpha", top=2)
hybrid = arena.probe_hybrid([0.0, 1.0], "alpha", top=2)
```

`Arena` maps directly onto the same core APIs and supports typed `Row` and
`Hit` objects. By default it reuses the database's native local embedder, so
you only need to pass `embedder=...` when you want a custom Python callable or
an explicit `LocalEmbedderConfig`.

## Next

- [Python SDK](../../python_sdk.md)
- [Database API Reference](../api-reference/database.md)
- [Vault API Reference](../api-reference/vault.md)
- [Config & Enums](../api-reference/config-and-enums.md)
