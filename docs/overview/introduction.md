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

An ELIPS database is a directory. A vault is a named partition inside that
database. Each record can carry:

- a vector
- metadata
- an optional source document
- optional chunk coordinates
- optional embedding provenance

The core APIs are:

- `place()` for explicit vectors
- `place_document()` for text ingestion through a configured or auto-attached
  text embedder
- `seek()`, `seek_text()`, and `seek_hybrid()`
- `fetch()` and `scan()`
- `checkpoint()` and `compact()`

## Example

```python
import elips

db = elips.open(":memory:", dimension=128, metric="cosine")
docs = db.vault("documents")
docs.place_document("alpha design note", {"kind": "design"})
print(docs.seek_text("alpha", top=1)[0].document.text)
```

## Read More

- [Architecture](../architecture.md)
- [Storage](../storage.md)
- [Python SDK](../python_sdk.md)
- [C++ SDK](../cpp_sdk.md)
