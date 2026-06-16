# ELIPS

Embedded Local Index & Persistence System.

ELIPS is an in-process vector and document retrieval engine built in C++23 with
native Python bindings. It keeps the embedded deployment model of SQLite, but
adds ANN indexes, typed metadata filters, first-class document lineage, hybrid
retrieval, segmented persistence, and optional GPU-backed indexes.

```python
from __future__ import annotations

import elips

engine = elips.connect(":memory:", dimension=128)
arena = engine.arena("documents")
arena.ingest(
    texts=["alpha design note", "beta incident runbook"],
    meta=[{"kind": "design"}, {"kind": "ops"}],
)

for hit in arena.probe_text("alpha", top=2):
    print(hit.key, hit.distance, hit.text, hit.meta)
```

## What Is Implemented

- Vector search with `graph` (HNSW) and `exact` indexes
- GPU index selection through the main `open()` / index factory path
- First-class `DocumentAttachment`, `ChunkInfo`, and `EmbeddingLineage`
- Native `place_document()`, `seek_text()`, `seek_hybrid()`, and `explain_seek()`
- Built-in local text embedding with automatic default provisioning for new databases
- Metadata acceleration for equality filters via `MetadataIndex`
- Segmented persistence with `elips.manifest` plus per-vault segment files
- `compact()` to rebuild indexes and rewrite the segment set
- Shared read-only mode with advisory locks
- WAL crash recovery, snapshot compatibility, typed filters, EQL, Python bindings

## Quick Start

Build the core and run the C++ and Python tests:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DELIPS_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
PYTHONPATH=bindings/python python3 tests/python/test_bindings.py
```

Minimal low-level Python usage:

```python
import elips

db = elips.open("/tmp/elips-demo", dimension=128, metric="cosine")
docs = db.vault("documents")
docs.place_document("alpha design note", {"kind": "design"})
docs.place_document("beta runbook", {"kind": "ops"})
print(db.config.text_embedder_info.provider, db.config.text_embedder_info.model)
print(docs.seek_text("alpha", top=1)[0].document.text)
db.compact()
db.close()
```

Vector-first ingestion still works unchanged:

```python
docs.place(
    embedding_vector,
    document=elips.DocumentAttachment(text="source document"),
)
```

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

Single-writer opens take an exclusive lock. Read-only opens take a shared lock
and reject writes.

## Documentation

- [Architecture](docs/architecture.md)
- [Local Embedding Architecture](docs/architecture/local-embedding-system.md)
- [Storage](docs/storage.md)
- [Python SDK](docs/python_sdk.md)
- [C++ SDK](docs/cpp_sdk.md)
- [Python Quickstart](docs/python/getting-started/quickstart.md)
- [C++ Quickstart](docs/cpp/getting-started/quickstart.md)
- [API references](docs/python/api-reference/) / [C++ API references](docs/cpp/api-reference/)

## Status

The remaining roadmap is now focused on future work such as MVCC/versioned
reads, deeper text planning in EQL, quantized indexes, and broader distributed
or cloud-oriented integrations. See [docs/roadmap.md](docs/roadmap.md).
