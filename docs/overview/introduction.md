# ELIPS — Introduction

ELIPS is an embedded retrieval engine for vectors and documents.

It is designed for applications that want local, in-process retrieval without
operating a separate service: CLIs, desktop apps, notebooks, edge workloads,
single-process APIs, and test environments.

## What You Get

- ANN and exact vector search
- First-class documents, chunk coordinates, and embedding lineage
- Native text-first and hybrid query APIs
- Metadata filters with planner-side candidate narrowing
- WAL recovery plus segmented persistence
- Shared read-only mode for multi-reader serving
- Python and C++ SDKs over the same core

## Mental Model

An ELIPS database is a directory. A collection is a named partition inside that
database, built on a FAISS-recognizable `Index`. Each record can carry:

- a vector
- metadata
- an optional source document
- optional chunk coordinates
- optional embedding provenance

The core APIs are:

- `add()` for explicit vectors (upsert semantics)
- `add_document()` for text ingestion through a configured or auto-attached
  text embedder
- `search()`, `search_text()`, and `hybrid_search()`
- `get()` and `scan()`
- `checkpoint()` and `compact()`

The index types themselves (`IndexFlat`, `IndexHNSW`, `IndexIVFFlat`,
`IndexIVFPQ`) and `index_factory` are exposed directly for FAISS-style use. See
[Which index should I use?](choosing-an-index.md).

## Example

```python
import elips

db = elips.open(":memory:", dimension=128, metric="cosine")
docs = db.collection("documents")
docs.add_document("alpha design note", {"kind": "design"})
print(docs.search_text("alpha", top=1)[0].document.text)
```

The older `place`/`seek`/`vault`/`top=` spellings still work as deprecation
shims for one minor version.

## Read More

- [Architecture](../architecture.md)
- [Storage](../storage.md)
- [Python SDK](../python_sdk.md)
- [C++ SDK](../cpp_sdk.md)
