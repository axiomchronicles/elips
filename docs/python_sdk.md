# Python SDK

ELIPS ships a Chroma-style Python surface on top of its C++ core. You open a
client, hand out collections, and read and write records with the familiar
`add` / `upsert` / `get` / `update` / `delete` / `query` / `count` / `peek`
verbs. A separate FAISS-style `Index` surface is available when you want raw
vector indexes with no record store or metadata.

> **Maturity**
>
> | Surface | Status |
> |---|---|
> | CPU engine, Collection API, Index API | **Stable** |
> | GPU via Apple **Metal** | **Beta** — validated on Apple M4 |
> | GPU via **CUDA** / **HIP (ROCm)** | **Experimental** — compiled but not hardware-validated |

> **ELIPS is dimension-first.** Unlike Chroma's zero-config `Client()`, ELIPS
> has no intrinsic default embedding dimension, so you must pass
> `dimension=<int>` (or an `embedder=` that fixes the dimension) when creating a
> database.

## Build

```bash
cmake -S . -B build -G Ninja -DELIPS_ENABLE_PYTHON=ON
cmake --build build --target elips_pymodule
export PYTHONPATH=$PWD/bindings/python
```

The compiled module already lives in `bindings/python/elips/`. Any example
below runs with `PYTHONPATH=bindings/python python3`.

## Clients

`Client()` opens an ephemeral in-memory database; `PersistentClient(path)`
opens (or creates) one on disk. Both accept the same keyword settings
(`dimension`, `metric`, `index`, `embedder`, `quantization`, `gpu`, `device`).

```python
import elips

client = elips.Client(dimension=4, metric="cosine")
docs = client.get_or_create_collection("documents")
docs.add(ids=["a", "b"], embeddings=[[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
print(docs.count())
client.close()
```

```python
import tempfile

path = tempfile.mkdtemp()
pdb = elips.PersistentClient(path, dimension=4)
pdb.get_or_create_collection("documents").add(ids=["a"], embeddings=[[1.0, 0, 0, 0]])
pdb.close()

# Reopen; a persistent database remembers its dimension, so none is needed.
reader = elips.PersistentClient(path, access_mode="read_only")
print(reader.get_or_create_collection("documents").count())
reader.close()
```

### Collection management (on the client)

| Method | Behavior |
|---|---|
| `get_or_create_collection(name)` | Idempotent create; returns the collection. |
| `create_collection(name)` | Create; raises if `name` already exists. |
| `get_collection(name)` | Return existing; raises `CollectionNotFoundError`. |
| `list_collections()` | Return the names of existing collections (`list[str]`). |
| `delete_collection(name)` | Not supported — raises `NotImplementedError`. |

`metadata=` / `configuration=` are accepted on the create methods for Chroma
compatibility but ignored: ELIPS index configuration is set at the database
level (see `Config`).

## Collections

A `Collection` is the Chroma-style data plane. All methods take keyword
arguments: `ids`, `embeddings`, `documents`, `metadatas`, plus `where`,
`include`, and `n_results` on reads.

```python
client = elips.Client(dimension=8)
docs = client.get_or_create_collection("corpus")

# Column form: one list per field, aligned by position.
docs.add(
    ids=["d1", "d2"],
    documents=["alpha design note", "beta ops runbook"],
    metadatas=[{"kind": "design"}, {"kind": "ops"}],
)
```

- `ids` are **arbitrary strings**. A non-native id is hashed to a stable
  internal id and the original is preserved across reopen, so `get(["d1"])`
  returns a `Row` whose `.key` is `"d1"`.
- `documents` are raw text; the collection embeds them through the database's
  text embedder (the built-in local embedder is attached by default).
- `embeddings` are `list[list[float]]`; supply them to skip embedding.
- `add` has upsert semantics: re-adding an existing id overwrites it.

### Querying

`query` does vector, text, or hybrid search. A single query returns
`list[Hit]`; a batch returns `list[list[Hit]]`.

```python
# Text query (uses the default embedder).
hits = docs.query(query_texts="alpha", n_results=2)
print(hits[0].key, hits[0].distance, hits[0].text)

# Vector query with a metadata filter.
docs.add(ids=["y24", "y20"], embeddings=[[1.0] + [0] * 7, [0, 1.0] + [0] * 6],
         metadatas=[{"year": 2024}, {"year": 2020}])
recent = docs.query(
    query_embeddings=[1.0] + [0] * 7,
    n_results=5,
    where=elips.Filter().field("year").gte(2023),
)
print(len(recent))
```

`n_results` is canonical; `k` and `limit` are accepted aliases. Pass both
`query_embeddings` and `query_texts` for a hybrid query (`lexical_weight`
tunes the lexical blend, default `0.25`).

### Reading and deleting

```python
# get: by id (ranking-free), or scan by where/limit/offset.
rows = docs.get(["d1"])
page = docs.get(where=elips.Filter().field("kind").equals("ops"), limit=10)
preview = docs.peek(3)              # first N records
docs.delete(["y20"])               # by id, or where=, or both
print(docs.count())
client.close()
```

`include=[...]` selects what comes back (`"embeddings"`, `"metadatas"`,
`"documents"`). `ids` are always present on the returned `Row`/`Hit`.

### Methods not yet implemented

These exist for Chroma parity but raise `NotImplementedError` in this build:

- `where_document=` (full-text content filtering) on `get` / `query` / `delete`.
- `Collection.modify(...)` (collection rename / retag).
- `Client.delete_collection(...)` and `Client.reset()`.

`update(ids=..., ...)` exists and overwrites existing records, but in this
build is limited (it validates ids through the native store and re-adds the
record, so it needs the embedding re-supplied and works reliably only for
records ELIPS can resolve). Prefer `upsert` to change a record by id:

```python
# no-exec — illustrative; upsert is the reliable overwrite path
docs.upsert(ids=["d1"], documents=["alpha design note v2"])
```

## FAISS-style Index API

For raw vector indexing without a record store, ELIPS exposes a FAISS-shaped
`Index` surface. Vectors are `float32` NumPy arrays; `search` returns
`(distances, labels)`.

```python
import numpy as np
import elips

d = 16
xb = np.random.rand(500, d).astype("float32")
xq = np.random.rand(5, d).astype("float32")

index = elips.IndexFlat(d, metric="l2")   # brute-force exact
index.add(xb)
distances, labels = index.search(xq, 4)
print(index.ntotal, distances.shape, labels.shape)
```

IVF indexes must be trained before `add`; tune recall with `nprobe`:

```python
ivf = elips.IndexIVFFlat(d, nlist=32, metric="l2")
ivf.train(xb)
ivf.add(xb)
ivf.nprobe = 8
_ = ivf.search(xq, 4)
print(ivf.is_trained, ivf.ntotal)
```

| Constructor | Index |
|---|---|
| `IndexFlat(d, metric="l2")` | Brute-force exact. |
| `IndexIVFFlat(d, nlist, metric="l2")` | Inverted file, full-precision postings. |
| `IndexIVFPQ(d, nlist, M, metric="l2")` | Inverted file, product-quantized postings (`M` subspaces). |
| `IndexIVFScalarQuantizer(d, nlist, metric="l2")` | Inverted file, SQ8 postings. |
| `IndexHNSW(d, M=16, metric="l2")` | HNSW graph. |

`metric` is `"l2"`/`"euclidean"`, `"ip"`/`"dot_product"`, or `"cosine"`
(cosine assumes L2-normalized inputs). Index methods: `train(x)`, `add(x)`,
`add_with_ids(x, ids)`, `search(x, k) -> (D, I)`, `reset()`; attributes `d`,
`ntotal`, `is_trained`, `nprobe`.

Build an index from a FAISS factory string with `index_factory`:

```python
fx = elips.index_factory(d, "IVF32,PQ2", metric="l2")
fx.train(xb)
fx.add(xb)
print(fx.ntotal)
```

Supported factory specs: `"Flat"`, `"IVF<nlist>,Flat"`, `"IVF<nlist>,PQ<M>"`,
`"IVF<nlist>,SQ8"`, `"HNSW<M>"`.

For a one-shot exact k-NN over raw arrays with no index build, use `bf_knn`
(runs on Metal when available, else CPU):

```python
D, I = elips.bf_knn(xq, xb, k=4, metric="l2", device="cpu")
print(D.shape, I.shape)
```

## Exceptions

All ELIPS errors derive from `ElipsError` (itself a `RuntimeError`).

| Exception | Raised when |
|---|---|
| `DimensionMismatchError` | A vector length differs from the database dimension. |
| `InvalidVectorError` | A vector contains `NaN`/`Inf`. |
| `NotFoundError` | A requested record is absent. |
| `CollectionNotFoundError` | `get_collection` names a collection that does not exist. |
| `NotTrainedError` | A compressed/IVF index is searched before training. |
| `BackendUnavailableError` | A named GPU backend is unavailable and no CPU fallback was permitted. |
| `ConfigError`, `LockConflict(Error)`, `StorageError`, `ParseError` | Config/lock/IO/EQL failures. |

The `*Error`-suffixed names are the Phase-5 spellings; the native runtime
raises the un-suffixed classes (`DimensionMismatch`, `NotFound`, ...) and the
suffixed names are aliases, so either spelling catches the same error.

```python
client = elips.Client(dimension=2)
try:
    client.get_collection("missing")
except elips.CollectionNotFoundError:
    pass
client.close()
```

## Low-level native handles

The Chroma surface wraps native handles (`Database`, `Vault`, `Filter`),
reachable as `elips.native.*` or through the `.native` attribute of any
high-level object. Drop down only when you need something the facade does not
cover (e.g. transactions, `engine.native.begin_transaction()`). See the
[Collection](python/api-reference/vault.md) and
[Database](python/api-reference/database.md) references.

## Deprecated vocabulary

The pre-Phase-5 API (`connect`/`open`/`Engine`; collection verbs
`write`/`ingest`/`merge`/`probe*`/`pull`/`sweep`/`discard`; `Arena`; kwargs
`top=`/`meta=`/`keys=`/`vector(s)=`/`text=`) still works as deprecated shims
that emit `DeprecationWarning` and delegate to the canonical methods. They are
removed in the next minor version. See `MIGRATION.md` for the full mapping.

## Typing

The package ships `py.typed` and a `_core.pyi` stub so IDEs and type checkers
understand the public API. See [Type Stubs & IDE Support](python/typing/type-stubs.md).
