# ELIPS (Python Bindings)

Embedded Local Index & Persistence System (ELIPS) Python package.

ELIPS is an in-process vector and document retrieval engine built in C++23 with native Python bindings. It keeps the embedded deployment model of SQLite, but adds ANN indexes, typed metadata filters, first-class document lineage, hybrid retrieval, segmented persistence, and optional GPU-backed indexes.

## Installation

### From PyPI
```bash
pip install elips
```

### From Source
```bash
pip install .
```
*Note: Building from source requires a C++23 compatible compiler (GCC 13+ or Clang 17+), CMake >= 3.24, and Ninja.*

## Quick Start

```python
import elips

# Connect to database (either path or ":memory:")
engine = elips.connect(":memory:", dimension=128)

# Access or create a documents arena
arena = engine.arena("documents")

# Ingest documents with text and optional metadata
arena.ingest(
    texts=["alpha design note", "beta incident runbook"],
    meta=[{"kind": "design"}, {"kind": "ops"}],
)

# Run semantic search
for hit in arena.probe_text("alpha", top=2):
    print(hit.key, hit.distance, hit.text, hit.meta)
```

## Features

- **Vector Search**: Optimized HNSW (`graph`) and `exact` index backends.
- **Embedded Deployment**: Simple SQLite-like single-writer/multi-reader lock model.
- **Local Embedding**: Built-in local text embedding using local models.
- **Metadata Filters**: Rich metadata equality filters via `MetadataIndex` and `Filter`.
- **Hybrid Retrieval**: Combine dense semantic search and document retrieval out of the box.
- **Segmented Persistence**: Durable writes with WAL crash recovery and snapshot checkpoints.
- **GPU Acceleration**: Optional Metal backend on Apple Silicon and CUDA/Vulkan backends on Linux.

## License

ELIPS Python bindings are released under the [MIT License](LICENSE).
