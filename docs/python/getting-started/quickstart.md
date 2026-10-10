# Quick Start

This walkthrough uses the current ELIPS Python surface: a Chroma-style
client and collection, text / vector / hybrid retrieval, the FAISS-style index
API, and persistence.

> **Maturity:** the CPU engine, Collection API, and Index API are **Stable**.
> GPU via Apple **Metal** is **Beta** (validated on Apple M4). GPU via
> **CUDA / HIP (ROCm)** is **Experimental** — compiled but not
> hardware-validated.

## 1. Build the Python module

```bash
cmake -S . -B build -G Ninja -DELIPS_ENABLE_PYTHON=ON
cmake --build build --target elips_pymodule
export PYTHONPATH=$PWD/bindings/python
```

Verify the import:

```bash
PYTHONPATH=bindings/python python3 -c "import elips; print(elips.__version__)"
```

## 2. Open a client

`Client()` is in-memory; `PersistentClient(path)` is on disk. ELIPS is
dimension-first, so pass `dimension=` (or an `embedder=` that fixes it).

```python
import elips

client = elips.Client(dimension=2, metric="cosine")
```

Settings mirror the database config: `metric` (`"cosine"`, `"euclidean"`,
`"dot_product"`), `index` (`"graph"` or `"exact"`), `access_mode`,
`quantization`, and GPU (`gpu=` / `device=`). New databases attach the built-in
local text embedder automatically unless you pass
`use_default_text_embedder=False`.

## 3. Add records

A collection takes Chroma column form: `ids`, `embeddings`, `documents`,
`metadatas`. With the default embedder you can add raw documents and skip
embeddings.

```python
docs = client.get_or_create_collection("documents")

docs.add(
    ids=["alpha", "beta"],
    documents=["alpha design note", "beta incident runbook"],
    metadatas=[{"kind": "design"}, {"kind": "ops"}],
)

# Or supply explicit vectors.
docs.add(ids=["gamma"], embeddings=[[1.0, 0.0]], metadatas=[{"kind": "appendix"}])
print(docs.count())
```

`ids` may be arbitrary strings; ELIPS preserves them across reopen. `add` has
upsert semantics (re-adding an id overwrites it).

## 4. Query

```python
# Text search (default embedder).
hits = docs.query(query_texts="alpha", n_results=2)
print(hits[0].key, hits[0].text)

# Vector search with a metadata filter.
hits = docs.query(
    query_embeddings=[1.0, 0.0],
    n_results=2,
    where=elips.Filter().field("kind").equals("appendix"),
)

# Hybrid: pass both; lexical_weight tunes the lexical blend.
hits = docs.query(query_embeddings=[0.0, 1.0], query_texts="alpha", n_results=2)
```

`n_results` is canonical (aliases `k` / `limit`). A single query returns
`list[Hit]`; a batch of queries returns `list[list[Hit]]`.

## 5. Read, scan, delete

```python
rows = docs.get(["alpha"])                      # by id
page = docs.get(where=elips.Filter().field("kind").equals("ops"), limit=10)
preview = docs.peek(5)                           # first N
docs.delete(["gamma"])                          # by id, or where=, or both
print(docs.count())
client.close()
```

## 6. Persist and reopen

```python
import tempfile

path = tempfile.mkdtemp()
db = elips.PersistentClient(path, dimension=2)
db.get_or_create_collection("documents").add(ids=["x"], embeddings=[[1.0, 0.0]])
db.checkpoint()        # flush to durable storage
db.close()

reader = elips.PersistentClient(path, access_mode="read_only")
print(reader.get_or_create_collection("documents").count())
reader.close()
```

Read-only mode requires an existing database and rejects writes.

## 7. FAISS-style index

For a bare vector index with no record store, use the `Index` surface
(`float32` NumPy arrays).

```python
import numpy as np

d = 16
xb = np.random.rand(200, d).astype("float32")
index = elips.IndexFlat(d, metric="l2")
index.add(xb)
distances, labels = index.search(xb[:3], 5)
print(index.ntotal, labels.shape)
```

IVF indexes train first (`index_factory`, `IndexIVFFlat`, `IndexIVFPQ`,
`IndexIVFScalarQuantizer`) and expose `nprobe`; `IndexHNSW` is the graph index.
See the [Python SDK](../../python_sdk.md) for the full index catalog.

## 8. Optional GPU

GPU is opt-in through a device string. The example below is shape-only; it
needs a GPU backend to run.

```python
# no-exec — requires a GPU backend
gpu = elips.Client(dimension=384, device="metal")   # "metal" | "cuda:0" | "auto"
corpus = gpu.get_or_create_collection("corpus")
hits = corpus.query(query_embeddings=[0.0] * 384, n_results=10)
gpu.close()
```

`device="metal"` is Beta on Apple; `"cuda:N"` / `"hip:N"` are experimental. A
named backend that is unavailable raises `BackendUnavailableError` unless you
pass `fallback=True`.

## Next

- [Python SDK](../../python_sdk.md)
- [Collection API Reference](../api-reference/vault.md)
- [Database API Reference](../api-reference/database.md)
- [Config & Enums](../api-reference/config-and-enums.md)
- [Filter API Reference](../api-reference/filter.md)
