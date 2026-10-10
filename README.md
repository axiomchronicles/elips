# ELIPS

Embedded Local Index & Persistence System.

ELIPS is an in-process vector and document retrieval engine built in C++23 with
native Python bindings. It keeps the embedded deployment model of SQLite, but
adds ANN indexes, typed metadata filters, first-class document lineage, hybrid
retrieval, segmented persistence, and optional GPU-backed indexes. The public
API is organized into two recognizable layers: a **Chroma-style Collection API**
and a **FAISS-style Index API**.

## Quick Start (Python)

The Chroma-style client is the high-level entry point. `Client` is in-memory;
`PersistentClient(path)` opens a database on disk. ELIPS is **dimension-first** —
pass `dimension=` (or a dimension-fixing `embedder=`).

```python
import elips

client = elips.Client(dimension=4)                 # in-memory
docs = client.get_or_create_collection("documents")
docs.add(
    ids=["a", "b"],
    embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]],
    metadatas=[{"kind": "design"}, {"kind": "ops"}],
)

# query_embeddings is a batch; results are one hit-list per query.
for hit in docs.query(query_embeddings=[[1, 0, 0, 0]], n_results=2)[0]:
    print(hit.key, hit.distance, hit.meta)

print(docs.count())
client.close()
```

Text in, with the built-in local embedder (auto-provisioned per database):

```python
import elips

client = elips.Client(dimension=16)
docs = client.get_or_create_collection("documents")
docs.add(
    ids=["d1", "d2"],
    documents=["alpha design note", "beta incident runbook"],
    metadatas=[{"kind": "design"}, {"kind": "ops"}],
)
hit = docs.query(query_texts=["alpha"], n_results=1)[0][0]
print(hit.key, hit.document.text)
client.close()
```

Persist to disk with `PersistentClient`:

```python
import elips

client = elips.PersistentClient("/tmp/elips-demo", dimension=4)
docs = client.get_or_create_collection("documents")
docs.add(ids=["x"], embeddings=[[1, 0, 0, 0]])
client.close()
```

## FAISS-style Index API

For low-level, numpy-first indexing, use the Index API directly. Inputs are
float32 / C-contiguous; `search` returns `(D, I)` — float32 distances and int64
labels, each `(n_queries, k)`, with `-1` labels / `+inf` distances padding slots
past `ntotal`.

```python
import numpy as np
import elips

index = elips.index_factory(8, "IVF16,Flat")       # or elips.IndexFlat(8)
xb = np.random.rand(1000, 8).astype(np.float32)
index.train(xb)                                     # IVF requires train() first
index.add(xb)
D, I = index.search(xb[:5], k=4)
print(D.shape, I.shape)                             # (5, 4) (5, 4)
```

`index_factory` specs mirror FAISS token spellings: `"Flat"`, `"IVF<nlist>,Flat"`,
`"IVF<nlist>,PQ<M>"`, `"IVF<nlist>,SQ8"`, `"HNSW<M>"`. For a one-shot exact search
with no index build, use `elips.bf_knn(xq, xb, k, metric="l2", device="auto")`.

## Backend Maturity

| Backend | Status |
|---|---|
| CPU | **Stable** — the reference path; all indexes and metrics. |
| Metal (Apple Silicon) | **Beta** — validated on Apple M4; real GPU kernels (brute-force + IVF). |
| CUDA (NVIDIA) | **Experimental** — compiles against the interfaces; not hardware-validated. |
| HIP / ROCm (AMD) | **Experimental** — compiles against the interfaces; not hardware-validated. |

`elips.has_gpu` reports whether the build has a GPU backend;
`elips.accelerators()` lists the devices it can drive.

## What Is Implemented

- Exact (`IndexFlat`), HNSW (`IndexHNSW`), and the CPU IVF family
  (`IndexIVFFlat`, `IndexIVFPQ`, `IndexIVFScalarQuantizer`), behind `Index`.
- FAISS-style `index_factory` grammar and standalone `bf_knn`.
- Chroma-style `Client` / `PersistentClient` and the Collection API
  (`add/upsert/get/update/delete/query/count/peek/modify`) with arbitrary string ids.
- First-class `DocumentAttachment`, `ChunkInfo`, and `EmbeddingLineage`;
  built-in local text embedding with auto-provisioning.
- Metadata-accelerated equality filters, hybrid lexical+vector search, and EQL.
- Segmented persistence (`elips.manifest` + per-collection segment files),
  `compact()`, WAL crash recovery, snapshot compatibility, and shared read-only mode.
- Metal GPU indexes with real kernels; CUDA/HIP interface-complete (experimental).

> Migrating from the pre-rename API (`connect`/`open`, `Vault`, `place`/`seek`,
> `arena`/`ingest`/`probe`)? Those names still work this release via deprecation
> shims and are documented in [MIGRATION.md](MIGRATION.md).

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DELIPS_ENABLE_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
PYTHONPATH=bindings/python python3 tests/python/test_bindings.py
```

GPU backends: `-DELIPS_ENABLE_METAL=ON` (default on Apple), `-DELIPS_ENABLE_CUDA=ON`,
`-DELIPS_ENABLE_ROCM=ON` (both experimental).

## Storage Model

Persistent databases use:

```text
/my_db/
├── LOCK
├── IDENTITY
├── TEXT_EMBEDDER.manifest
├── wal.log
├── elips.manifest          # when segmented storage is enabled
├── text_embedder/
│   └── default_v1_<dim>.localembed
├── segments/
│   ├── vault_0_<epoch>.segment
│   └── vault_1_<epoch>.segment
└── elips.snapshot          # compatibility / non-segmented mode
```

The `vault_` segment-file prefix is a legacy on-disk name kept for backward
compatibility (renaming it would break existing databases). Single-writer opens
take an exclusive lock; read-only opens take a shared lock and reject writes.

## Documentation

- [Architecture](docs/architecture.md)
- [Storage](docs/storage.md)
- [EQL](docs/eql.md)
- [CLI](docs/cli.md)
- [FAQ](docs/faq.md)
- [Python SDK](docs/python_sdk.md) / [C++ SDK](docs/cpp_sdk.md)
- [Developer guide](docs/developer-guide/)
- [Roadmap](docs/roadmap.md) · [Migration](MIGRATION.md) · [Changelog](CHANGELOG.md)
