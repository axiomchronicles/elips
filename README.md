# ELIPS

**E**mbedded **L**ocal **I**ndex & **P**ersistence **S**ystem — an embedded,
in-process vector database. SQLite for vectors: no server, no port, no daemon.
`#include` it or `import` it and go.

```python
import elips

db = elips.open("/data/vectors", dimension=1536, metric="cosine")
docs = db.vault("documents")
docs.place(vector=embedding, data={"title": "Example", "year": 2024})

for hit in docs.seek(vector=query, top=10):
    print(hit.id, hit.distance, hit.data)
```

## Status

This repository implements the ELIPS core and its surfaces from first
principles in C++23:

| Subsystem | Status |
|---|---|
| Domain model (Vector, Record, RecordID/UUIDv7, Payload) | ✅ |
| Metrics: cosine, euclidean, dot product (NEON SIMD + scalar) | ✅ |
| Indexes: `HierarchicalGraphIndex` (HNSW) + `ExactIndex` | ✅ |
| Storage: snapshots, write-ahead log, crash recovery | ✅ |
| Single-writer / multi-reader file locking | ✅ |
| Metadata filtering + atomic transactions | ✅ |
| EQL (lexer, parser, AST, executor) | ✅ |
| `elips` CLI (info/vaults/query/export/import/verify/bench/…) | ✅ |
| Python bindings (PyBind11) + `py.typed` | ✅ |
| Benchmark suite | ✅ |

See [docs/roadmap.md](docs/roadmap.md) for what is intentionally deferred
(segments/compaction, MVCC version chains, quantized indexes, cloud adapters).

## Build

Requires CMake ≥ 3.24, Ninja, and a C++23 compiler (Clang 17+/GCC 13+).

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Targets: `elips_core` (library), `elips` (CLI), `elips_bench`, `elips_tests`.

### Python bindings

```bash
cmake -S . -B build -G Ninja -DELIPS_BUILD_PYTHON=ON
cmake --build build --target elips_pymodule
PYTHONPATH=bindings/python python3 -c "import elips; print(elips.__version__)"
```

## Documentation

- [Architecture](docs/architecture.md)
- [Storage & recovery](docs/storage.md)
- [EQL language guide](docs/eql.md)
- [Python SDK](docs/python_sdk.md) · [C++ SDK](docs/cpp_sdk.md) · [CLI](docs/cli.md)
- [Cookbook](docs/cookbook.md)
- [Architecture Decision Records](docs/adr/)

## License

See repository for license terms.
