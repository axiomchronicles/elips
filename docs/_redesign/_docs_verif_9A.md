# Phase 9A — Python Docs Verification (`_docs_verif_9A.md`)

Scope: `docs/python/**`, `docs/python_sdk.md`, `docs/cookbook.md`, rewritten to
the post-rename (Phase 5) Python API. Every documented symbol/signature/default
checked against `bindings/python/elips/**.py` and the built module
(`PYTHONPATH=bindings/python python3`, elips **1.1.0**). Every runnable python
block executed via `scripts/run_doc_snippets.py` — final run: **69 executed,
19 skipped (no-exec), 0 failed across 13 files**.

Legend: "ran?" = appears in an executed (non-`no-exec`) snippet that passed.

## Core high-level surface

| Doc page | Symbol / signature | Code (file:line) | Ran? |
|---|---|---|---|
| python_sdk, quickstart, database | `Client(**settings)` (dimension required) | client.py:48,58 | yes |
| python_sdk, quickstart, database | `PersistentClient(path, **settings)` | client.py:64,76 | yes |
| database | `get_or_create_collection` / `create_collection` / `get_collection` | engine.py:148,163,176 | yes |
| database | `list_collections()` / `collection_names()` → `list[str]` | engine.py:184,244 | yes |
| database | `delete_collection` / `reset` → `NotImplementedError` | engine.py:188,194 | verified (not run) |
| database | `query_eql(eql, bindings)` (EQL entry on client) | engine.py:198 | yes |
| database | `checkpoint`/`compact`/`vacuum`/`pending_writes`/`close` | engine.py:202,210,218,264,297 | yes (checkpoint/close) |
| vault (Collection) | `add` / `upsert` (ids/embeddings/documents/metadatas) | collection.py:408,477 | yes |
| vault | `get(ids=, where=, limit=, offset=, include=)` | collection.py:504 | yes |
| vault | `query(query_embeddings=, query_texts=, n_results=(+k/limit), where=, max_distance=, lexical_weight=)` | collection.py:569 | yes |
| vault | `delete` / `count` / `peek(n=10)` | collection.py:528,98,542 | yes |
| vault | `update` — present but limited (re-adds via native fetch) | collection.py:486 | documented limitation (not run) |
| vault | `modify` / `where_document` → `NotImplementedError` | collection.py:546,516,536 | verified (not run) |
| vault | native `Vault` escape hatch (`place/seek/fetch/scan/erase`) | collection.py:88; collection.py:194,256,334,348 | yes (`.native`) |

## Records, filter, config, utilities

| Doc page | Symbol | Code (file:line) | Ran? |
|---|---|---|---|
| vault, python_sdk | `Hit` (.key/.distance/.meta/.text/.vector) | records/models.py:313 | yes |
| vault | `Row` (.key/.meta/.vector/.text) | records/models.py:258 | yes |
| filter | `Filter.field/equals/not_equals/lt/le/gt/gte/one_of/contains/and_/or_/not_` | _core (verified at runtime) | yes |
| filter | **No `.ge()` / `.lte()`** (enum-only); use `.gte()` / `.le()` | _core (runtime: `ge`/`lte` absent) | yes |
| config-and-enums | `Config` builder + `dimension_val/metric_val/index_val/durability_val` | _core (runtime) | yes |
| config-and-enums | `GraphParams`, `LocalEmbedderConfig`, `QuantParams`, `Codec` | __init__.py:30-46 | yes (GraphParams/LocalEmbedderConfig/QuantParams) |
| config-and-enums | **No `HnswParams`; `IndexType` = {graph, exact} only** | runtime check | corrected |
| utilities | `distance/requires_normalization/metric_to_string/metric_from_string` | __init__.py:40-46 | yes |
| utilities | `generate_id/is_valid_id/magnitude/normalize` | __init__.py:40-46 | yes |
| utilities | `metric_from_string("bad")` raises **ConfigError** (not ValueError) | runtime check | yes |

## FAISS Index API

| Doc page | Symbol | Code | Ran? |
|---|---|---|---|
| python_sdk, cookbook, quickstart | `IndexFlat/IndexIVFFlat/IndexIVFPQ/IndexIVFScalarQuantizer/IndexHNSW` | _core (runtime) | yes |
| python_sdk, cookbook | `index_factory(d, spec, metric)`, `bf_knn(xq, xb, k, metric, device)` | _core (runtime) | yes |
| python_sdk | `Index.train/add/add_with_ids/search/reset`, props `d/ntotal/is_trained/nprobe` | _core (runtime) | yes |
| typing | **Absent (planned, not built): `read_index`/`write_index`/`get_num_gpus`/`StandardGpuResources`/`index_cpu_to_gpu`** | runtime check | not documented |

## Exceptions (exceptions.py)

| Symbol | Code | Ran? |
|---|---|---|
| `ElipsError` base; `DimensionMismatch(Error)`, `InvalidVector(Error)`, `NotFound(Error)`, `LockConflict(Error)` aliases | exceptions.py:26-44 | yes (CollectionNotFoundError) |
| `CollectionNotFoundError`, `NotTrainedError`, `BackendUnavailableError`, `ConfigError`, `StorageError`, `ParseError` | exceptions.py:46-60 | yes (CollectionNotFoundError, ConfigError) |

## EQL (verified against the built parser)

| Form | Result | Ran? |
|---|---|---|
| `add in <coll> vector [..] data {..}` (old `place` ok) | works | yes |
| `search in <coll> nearest <vec> k <n> [where..][project..] yield` (old `seek`/`top` ok) | works | yes |
| `get from <coll> id "<id>" yield` (old `fetch` ok) | works | yes |
| `scan in <coll> [where..][limit n] yield` | works | yes |
| `delete from <coll> id "<id>"` (old `erase` ok) | works | yes |
| `validate_eql`, `tokenize_eql`, `TokenKind` (import from `elips.query`) | works | yes |

## Doc lies corrected (from `_phase1_docs.md`)

- "There is no add/ingest/search/get/delete" — **false**; all exist (collection.py). Removed; replaced with the real Chroma nine.
- `Engine.query(...)` / `db.query` on the high-level handle — **nonexistent**; EQL is `client.query_eql(...)` / `client.native.query(...)`. Fixed in database.md, cookbook, eql.
- `Filter.ge(...)` — method is `.gte(...)`; `ge` is enum-only. Fixed in filter.
- `connect(gpu=True)` / `engine.gpu_info()` — wrong; GPU is `gpu=GpuConfig`/`device=` string, `gpu_info` is on native `Database`. Fixed.
- Build flags `ELIPS_BUILD_PYTHON`/`ELIPS_GPU_*`/`ELIPS_BUILD_TESTS` → `ELIPS_ENABLE_PYTHON`/`ELIPS_ENABLE_METAL|CUDA|ROCM`/`BUILD_TESTING`. Package files `errors.py`/`types.py` → `exceptions.py`/`typing.py`. Version `1.0.0` → `1.1.0`. `_core.pyi` 924 → 1803 lines.

## Maturity labels applied

CPU engine / Collection API / Index API = **Stable**; GPU via Metal =
**Beta** (Apple M4); GPU via CUDA/HIP = **Experimental** (not hardware-validated).
Applied in python_sdk, quickstart, installation, database, config-and-enums,
cookbook, pybind11-architecture.

## Dimension-first + caveats noted

`Client(dimension=...)` required (client.py:36-45 `_DIM_HINT`); arbitrary
string ids hashed + preserved (collection.py:155-163); `where_document`
NotImplemented (collection.py:516,536).
