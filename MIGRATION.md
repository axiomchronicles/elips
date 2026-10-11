# Migration Guide

This release re-architects ELIPS's public API around two recognizable layers — a
**FAISS-style Index API** and a **Chroma-style Collection API**. All old names below are
**deprecated but still work this minor version** (they emit `DeprecationWarning` in Python,
`[[deprecated]]` in C++, a parse-time synonym in EQL, and `message(DEPRECATION)` in CMake).
**They will be removed in the next minor release** — migrate at your convenience.

## Python

### Entry points
| Old | New |
|---|---|
| `elips.connect(":memory:", dimension=d)` | `elips.Client(dimension=d)` |
| `elips.connect(path, …)` / `elips.open(path, …)` | `elips.PersistentClient(path, …)` |
| `elips.Engine` | `elips.Client` / `elips.PersistentClient` |

> Note: ELIPS is **dimension-first** — `Client(...)` requires `dimension=` (or a dimension-fixing
> `embedder=`). Unlike Chroma it does not yet infer the dimension from a default embedding function.

### Collection methods
| Old | New |
|---|---|
| `write` / `write_many` / `ingest` | `add` |
| `merge` | `upsert` |
| `probe` / `probe_text` / `probe_hybrid` / `search_vector` | `query` |
| `pull` / `sweep` | `get` |
| `discard` | `delete` |
| *(new)* | `update`, `peek`, `modify`, `count` |

Keyword arguments: `top=` → `n_results=` (also accepts `k=` / `limit=`), `meta=` → `metadatas=`,
`keys=` → `ids=`, `vector(s)=` → `embeddings=`, `text=` → `documents=`, `include_vectors=` → `include=[...]`.
New: `where=`, `where_document=` (raises `NotImplementedError` until implemented), `offset=`.

### Collections management
`create_collection` / `get_collection` / `get_or_create_collection` / `list_collections` on the client.
(`delete_collection` / `reset` are not yet implemented and raise `NotImplementedError`.)

### FAISS-style Index API (new)
A low-level numpy surface sits alongside the Collection API: `elips.IndexFlat`,
`IndexIVFFlat`, `IndexIVFPQ`, `IndexIVFScalarQuantizer`, `IndexHNSW`, the
`index_factory(d, spec)` builder, and standalone `bf_knn`. There is no old name to
migrate from — this layer is new. (`read_index`/`write_index`, `get_num_gpus`,
`index_cpu_to_gpu`, `StandardGpuResources`, and Python `range_search`/`reconstruct`
are not yet exposed; see `docs/roadmap.md`.)

### Exceptions
Suffix-aligned to `*Error`; new `NotTrainedError`, `CollectionNotFoundError`, `BackendUnavailableError`.
Removed from the public surface (never produced by any API): `IndexSnapshot`, `IvfSnapshot`,
`PqSnapshot`, `IndexSnapshotKind`.

## C++

### Classes / types
| Old | New |
|---|---|
| `Vault` / `VaultInfo` / `TransactionVault` | `Collection` / `CollectionInfo` / `TransactionCollection` |
| `ElipsInstance` | `Database` |
| `IndexPort` | `Index` |
| `ExactIndex` | `IndexFlat` |
| `HierarchicalGraphIndex` | `IndexHNSW` |
| `GpuPort` | `GpuBackend` |
| `GpuIndexPort` | `GpuIndex` |
| `GpuSelector` | `BackendSelector` |
| `DynamicBatcher` | `QueryBatcher` |
| `MetadataIndex` | `MetadataFilterIndex` |
| `GraphParams` | `HnswParams` |
| `IndexFactory` / `make_index` | `index_factory` |

> Note: `IndexType::graph` / `IndexType::exact` are **unchanged** (the enumerators were not renamed to `hnsw`/`flat`). The class `IndexFlat`/`IndexHNSW` names apply to the Index API types, not the `Config::IndexType` enum.

### Methods
`place`→`add` (+`_many`/`_document`), `seek`→`search` (+`_text`), `seek_hybrid`→`hybrid_search`,
`explain_seek`→`explain_query`, `fetch`→`get`, `erase`→`remove`, `quantize`→`train`,
`quantized`→`is_trained`, `Database::vault`→`collection`, `list_vaults`→`list_collections`;
Index contract `insert`→`add_with_ids`, `size`→`ntotal`.

## EQL
New primary verbs/clauses with old accepted as synonyms: `place`→`add`, `seek`→`search`,
`fetch`→`get`, `erase`→`delete`, `data`→`metadata`, `top`→`k`.
(The uppercase `SEARCH FROM … SEEK :q TOP` dialect some old docs showed was never implemented and is gone.)

## CMake options
| Old | New |
|---|---|
| `ELIPS_GPU_METAL` | `ELIPS_ENABLE_METAL` |
| `ELIPS_GPU_CUDA` | `ELIPS_ENABLE_CUDA` |
| `ELIPS_GPU_HIP` | `ELIPS_ENABLE_ROCM` |
| `ELIPS_BUILD_PYTHON` | `ELIPS_ENABLE_PYTHON` |
| `ELIPS_BUILD_TESTS` | `BUILD_TESTING` |

## Not changed (intentionally)
- On-disk format is unchanged: segment filenames (`vault_<n>_<epoch>.segment`), `elips.manifest`,
  snapshot magic, `IDENTITY`/`LOCK` — existing databases open unmodified.
- `nprobe` / `probe_lists` / `probe_all_devices` are IVF/device-probe parameters (not the old search-`probe`).
