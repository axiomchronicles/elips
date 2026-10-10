# ELIPS Phase 1B — Python Wrapper Inventory

Scope: `bindings/python/elips/**/*.py`, `.pyi` stubs, and the pybind layer
`bindings/python/elips_python.cpp`. Evidence gathered by reading every package
module and by introspecting the prebuilt extension
`bindings/python/elips/_core.cpython-313-darwin.so` at runtime (Python 3.13.14,
Apple Metal GPU present → `has_gpu == True`, `__version__ == "1.0.0"`).

Verification run facts:
- `import elips` works only with cwd/package dir on `sys.path`; `python3 -I` fails
  (`ModuleNotFoundError`) because `-I` drops cwd. Not a package bug.
- `bindings/python/test.py` **runs green** (exit 0) against the native vocabulary
  (`open` / `vault` / `place_document` / `seek_text`).
- `tests/python` suite: **239 passed, 5 skipped** (skips are GPU-device-absent and
  single-core-GIL guards) via `.venv/bin/python -m pytest`.
- No doctest runner is configured (`pyproject.toml` has no `--doctest-*`), so the
  modern-vocabulary docstring examples are **not executed as tests**.

---

## Architecture (layering)

The package is a set of **pure-Python typed facades** over a single compiled
extension. Nothing in Python re-implements engine logic; every facade delegates
to a native handle. The seam is deliberate and documented.

```
_core.*.so            compiled pybind11 extension (the only native code)
  └─ elips._core      raw extension module
       └─ elips._native   THE SEAM: mirrors _core programmatically + has_gpu,
                          DEMOTED map, supported_names()/demotion_message()
            ├─ elips.native    curated low-level facade (warns on demoted names)
            ├─ elips.core      DEPRECATED shim → _native  (warns on import)
            ├─ elips.modern    DEPRECATED shim → high-level names (warns on import)
            └─ elips           high-level API (connect/Engine/Collection/records)
                 ├─ elips.config        connect(), connect_with_config(), Config enums
                 ├─ elips.database       Engine
                 ├─ elips.collection     Collection (+ Arena alias), _embedding, _hydration
                 ├─ elips.records        RecordInput/Row/Hit/WalRecord/CollectionHealth
                 ├─ elips.gpu            Accelerator/AcceleratorSpec facade
                 ├─ elips.query          EQL AST + parse/tokenize/validate (re-export)
                 ├─ elips.search         Filter/Comparator/QueryPlan/Result (re-export)
                 ├─ elips.storage        WalEntry/WalOp/replay_wal + WalRecord
                 ├─ elips.quantization   Codec/QuantParams (re-export)
                 ├─ elips.typing         type aliases + TypedDicts (no runtime objects)
                 └─ elips._internal      coercion + protocols (private)
```

Key consequence: the high-level verbs (`connect`, `collection`, `add`/`write`,
`search`/`probe_text`, `ingest`, `discard`) are **Python renames** of the native
verbs (`open`, `vault`, `place`/`place_document`, `seek_text`, `place_many`,
`erase`). Both vocabularies are live today; see the vocabulary table.

---

## Public API surface (top-level `elips`)

High-level objects are Python; native handles are C++. "Tested" cites a
`tests/python` path exercising it; the native CRUD underneath is additionally
covered by `test_bindings.py` / `test_exposed_surface.py`.

| Name | Kind | Backed by | Status | Defined at |
|------|------|-----------|--------|-----------|
| `connect(path, *, dimension=0, metric="cosine", index="graph", access_mode="read_write", segmented_storage=True, metadata_acceleration=True, embedder=None, embedder_provider="python", embedder_model="callable", embedder_revision="", use_default_text_embedder=True, quantization=None, gpu=None) -> Engine` | func | native `open_with_config` | complete, tested (`test_bindings.py` via `connect`) | config/connect.py:58 |
| `connect_with_config(path, config, *, embedder=None, ...) -> Engine` | func | native `open_with_config` | complete, tested | config/connect.py:159 |
| `open(...) -> Database` | native func | C++ | complete, tested (test.py; test_bindings) | _core (elips_python.cpp) |
| `open_with_config(path, config) -> Database` | native func | C++ | complete, tested | _core |
| `Engine` | class (Py) | wraps native `Database` | complete, tested | database/engine.py:29 |
| `Collection` / `Arena` | class (Py) | wraps native `Vault` | complete, tested (`arena(...)`) | collection/collection.py:48 |
| `Config` | native class | C++ builder | complete, tested (`test_config`) | _core |
| `RecordInput` | dataclass | pure Py | complete, tested | records/models.py:34 |
| `Row` | dataclass | pure Py | complete, tested | records/models.py:258 |
| `Hit` | dataclass | pure Py | complete, tested | records/models.py:313 |
| `WalRecord` | dataclass | pure Py (+native `WalEntry`) | complete, tested (`pending_writes`) | records/models.py:370 |
| `CollectionHealth` | dataclass | pure Py | complete, tested (`health`) | records/models.py:449 |
| `Filter` | native class | C++ predicate builder | complete, tested (`test_filter_*`) | _core |
| `QueryPlan` | native class | C++ | complete, tested (`explain`) | _core |
| `DocumentAttachment`, `ChunkInfo`, `EmbeddingLineage` | native classes | C++ | complete, tested | _core |
| `Codec`, `QuantParams`, `GraphParams`, `LocalEmbedderConfig` | native | C++ | complete, tested | _core |
| `Embedder` | Protocol | pure Py | complete, tested (local embedder tests) | _internal/protocols.py:23 |
| `Accelerator`, `AcceleratorSpec`, `accelerator()`, `accelerators()` | Py facade | wraps native `GpuDevice` + free fns | complete; **GPU paths untested in `tests/python`** (device-gated skips) | gpu/accelerator.py |
| `has_gpu` | bool | `hasattr(_core, "GpuDevice")` | complete | __init__.py:63 |
| `distance`, `magnitude`, `normalize`, `generate_id`, `is_valid_id` | native funcs | C++ | complete, tested (`test_utilities`, `test_exposed_surface`) | _core |
| `ElipsError` + `ConfigError`/`DimensionMismatch`/`InvalidVector`/`LockConflict`/`NotFound`/`ParseError`/`StorageError` | native exc | C++ | complete, tested (`test_exceptions`) | _core |
| `__version__` | str | `_native.__version__` | complete | __init__.py:60 |

Native names reachable through `elips.*` (via `__getattr__`) but intentionally
absent from `__all__` (supported spelling is `elips.native.*`): `Database`,
`Vault`, `Transaction`, `TransactionVault`, `VaultInfo`, `BatchStats`, `Result`,
`Comparator`, `WalEntry`, `WalOp`, `AccessMode`, `Durability`, `Metric`,
`IndexType`, `QueryStrategy`, `TextEmbedderInfo`, `TextEmbedderKind`, all `Gpu*`,
`Graph*`, `Ivf*`, metric/eql/gpu free functions, `replay_wal`,
`describe_local_embedder`.

---

## Module: `elips/__init__.py`

The package root. Curates `__all__` (the supported high-level surface), re-exports
exceptions and vector helpers straight from `_native`, and defines a module
`__getattr__`/`__dir__` that (a) lazily imports the submodules `core`, `modern`,
`native` and (b) resolves bare native names like `elips.Vault`, warning for
`DEMOTED` ones. `__version__ = _native.__version__`; `has_gpu = _native.has_gpu`.

- Module docstring (quoted): *"ELIPS -- embedded, local-storage-first vector
  retrieval. The names exported here are the API: `connect` opens a database,
  `Engine` hands out `Collection` objects, and a collection reads and writes
  `Row` / `Hit` records. Native handles (`Database`, `Vault`, `Filter`) sit one
  level down, under `elips.native` ..."*
- Status: complete, tested (`test_native_seam.py`, `test_exposed_surface.py`).
- GOTCHA (not a bug, worth noting): only `{core, modern, native}` are in `_LAZY`.
  `elips.query` / `elips.search` / `elips.storage` / `elips.quantization` are
  **not** attribute-resolvable from a bare `import elips`; `elips.query` raises
  `AttributeError` until `import elips.query` runs. The subpackage docstrings use
  `from elips.query import ...`, which works. file:line __init__.py:68, 71.

## Module: `elips/_native.py` — THE SEAM

Single seam between package and extension. `globals().update(...)` binds every
public `_core` symbol programmatically (so new `py::class_` appears with zero
Python edits). Defines `has_gpu`, `supported_names()`, `demotion_message()`, and
`DEMOTED` (name → new home, or `None` for "no public home").

- Docstring (quoted): *"The single seam between the Python package and the
  compiled extension. ... Anything imported here is re-exported automatically, so
  that tax is gone. This module is private."*
- Status: complete, tested (`test_native_seam.py::test_the_seam_exposes_every_public_extension_symbol`,
  `::test_the_seam_itself_stays_complete`). file:line _native.py:56, 72, 82, 97.

## Modules: `elips/native.py`, `elips/core.py`, `elips/modern.py`

- `native.py`: supported low-level facade. `__all__ = supported_names() + has_gpu`.
  `__getattr__` resolves demoted names with a `DeprecationWarning` naming the new
  home; deliberately **not cached** so every call site warns. Complete, tested
  (`test_native_seam::test_demoted_names_warn_and_still_resolve`). native.py:38.
- `core.py`: **deprecated** compatibility shim. Emits `DeprecationWarning` at
  import, re-exports all of `_native`. Complete, tested
  (`test_native_seam::test_core_facade_matches_the_seam`). core.py:19.
- `modern.py`: **deprecated** shim re-exporting the high-level names (plus
  `ArenaHealth = CollectionHealth`). Warns at import. Complete. modern.py:19.

## Module: `elips/config/connect.py` + `config/__init__.py`

`connect(...)` builds a `Config` fluently and calls native `open_with_config`;
wraps result in `Engine`. `_attach_embedder` routes a `LocalEmbedderConfig` to
`config.local_text_embedder(...)` or a Python callable to `config.text_embedder(...)`.
`connect_with_config(...)` takes a prepared `Config`, attaching an embedder only
if the config has none. `config/__init__` re-exports the native config enums
(`AccessMode`, `Durability`, `GraphParams`, `IndexType`, `Metric`,
`TextEmbedderInfo`, `TextEmbedderKind`, `describe_local_embedder`).

- `connect` docstring quoted in table above. Status: complete, tested.
  file:line config/connect.py:58, 159.
- GPU guard: passing `gpu=` on a non-GPU build raises `ValueError`
  ("gpu config requires ELIPS to be built with GPU bindings"). connect.py:144.

## Module: `elips/database/engine.py` — `Engine`

Typed context-manager over native `Database`. All methods delegate to the native
handle.

| Member | Signature | Delegates to | Status |
|--------|-----------|--------------|--------|
| `native` / `raw` (prop) | `-> Database` | `self._db` | complete, tested |
| `config` (prop) | `-> Config` | `db.config` | complete, tested |
| `collection` / `arena` | `(name, *, embedder=None, text_slot="__elips_text__") -> Collection` | constructs `Collection` (which calls `db.vault(name)`) | complete, tested (`arena`) |
| `checkpoint` | `() -> None` | `db.checkpoint()` | complete, tested |
| `compact` | `() -> None` | `db.compact()` | complete, tested |
| `vacuum` | `() -> None` | `db.vacuum()` | complete, tested |
| `collection_names` / `vault_names` | `() -> list[str]` | `db.list_vaults()` | complete; `vault_names` tested, `collection_names` by-name untested |
| `pending_writes` | `() -> list[WalRecord]` | reads `<path>/wal.log` via `replay_wal` | complete, tested |
| `close` | `() -> None` | `db.close()` | complete, tested |
| `__enter__/__exit__` | ctx mgr | `close()` | complete, tested |

- Docstring (quoted): *"`Engine` is a context manager that hands out `Collection`
  handles and exposes WAL replay, checkpointing, **and transactions**."*
- **MISMATCH:** the docstring claims `Engine` exposes transactions, but `Engine`
  has **no** transaction method. Transactions live only on native `Database`
  (`begin_transaction()` → `Transaction`), reachable as `engine.native.begin_transaction()`.
  file:line database/engine.py:1 (docstring), class body engine.py:29–254.
- **MISSING high-level EQL:** native `Database.query(eql, bindings={})` exists and
  is tested (`test_bindings.py:492`), but `Engine` exposes **no** `query`. See the
  `elips.query` mismatch below. file:line engine.py.

## Module: `elips/collection/collection.py` — `Collection` (alias `Arena`)

Typed facade over one native `Vault`. Constructor: `Collection(db, name, *,
embedder=None, text_slot="__elips_text__")` → stores `db.vault(name)`.
`text_slot` is explicitly a **reserved no-op** compatibility arg (docstring:
*"Reserved compatibility argument preserved for older wrappers"*).

| Method | Signature | Native call | Status |
|--------|-----------|-------------|--------|
| `name` (prop) | `-> str` | `vault.name` | complete, tested |
| `raw` (prop) | `-> Vault` | `vault` | complete, tested |
| `count` | `() -> int` | `vault.count()` | complete, tested |
| `write` / **`add`** | `(record=None, /, *, vector=None, text=None, meta=None, key=None, document=None, chunk=None, lineage=None) -> str` | → `write_many([one])` | complete, tested (`write`; `add` by-name untested) |
| `write_many` / **`add_many`** | `(records) -> list[str]` | `vault.place` / `vault.place_document` per row | complete, tested (`write_many`; `add_many` untested) |
| `ingest` | `(records=None, /, *, vectors=None, texts=None, meta=None, keys=None, documents=None, chunks=None, lineages=None) -> list[str]` | → `write_many` | complete, tested |
| `merge` | same as `ingest` | → `ingest` | complete (compat alias), tested |
| `probe` / **`search_vector`** | `(vector, *, top=10, where=None, max_distance=None, include_vectors=False) -> list[Hit]` | `vault.seek(...)` | complete, tested (`probe`; `search_vector` untested) |
| `probe_text` / **`search`** | `(text, *, top=10, where=None, max_distance=None, include_vectors=False, lexical_weight=0.25) -> list[Hit]` | `vault.seek_text` (native embedder) else Python-embed + `vault.seek_hybrid` | complete, tested (`probe_text`; `search` by-name tested x2) |
| `probe_hybrid` / **`search_hybrid`** | `(vector, text, *, top=10, where=None, max_distance=None, lexical_weight=0.25, include_vectors=False) -> list[Hit]` | `vault.seek_hybrid` | complete, tested |
| `explain` | `(vector, *, top=10, where=None, max_distance=None, has_text_component=False) -> QueryPlan` | `vault.explain_seek` | complete, tested |
| `pull` / **`get`** | `(keys, *, include_vectors=True) -> list[Row]` | `vault.fetch` per key | complete, tested (`pull`, `get`) |
| `sweep` | `(*, where=None, offset=0, limit=None, include_vectors=False) -> list[Row]` | `vault.scan` | complete, tested |
| `discard` / **`delete`** | `(keys=None, *, where=None) -> int` | `vault.scan` + `vault.erase` per victim | complete, tested (`discard`; `delete` by-name untested) |
| `pending_removals` (prop) | `-> int` | `vault.pending_removals` | complete, tested |
| `read_only` (prop) | `-> bool` | `vault.read_only` | complete, tested |
| `sealed` (prop) | `-> bool` | `vault.sealed` | complete, tested |
| `freeze` | `(frozen=True) -> None` | `vault.set_read_only` | complete, tested |
| `vacuum` | `() -> None` | `vault.vacuum()` | complete, tested |
| `rebuild` | `() -> None` | `vault.rebuild_index()` | complete, tested |
| `compress` | `() -> None` | `vault.quantize()` | complete, tested (`compress` x2) |
| `health` | `() -> CollectionHealth` | `vault.info()` + flags | complete, tested |

- `discard` with neither `keys` nor `where` raises `ValueError`. collection.py:734.
- `write` with both `record` and keyword fields raises `ValueError`. collection.py:188.
- text-only write with custom document metadata raises `ValueError` (routes to
  `place_document` which takes raw text only). collection.py:259.
- Helpers `collection/_embedding.py` (`has_native_text_embedder`,
  `resolve_vectors`, `resolve_lineage`) and `collection/_hydration.py`
  (`row_from_record`, `hit_from_result`): pure Py, complete, tested transitively.

## Module: `elips/records/models.py` — exchange dataclasses

All `@dataclass(frozen=True, slots=True)`, pure Python. Complete, tested.

- `RecordInput(*, vector=None, text=None, meta=None, key=None, document=None,
  chunk=None, lineage=None)` — write-side input. `__post_init__` requires a vector
  or some text; validates text/document.text agreement. Props/methods:
  `document_text`, `materialize_meta()`, `materialize_document()`,
  `has_custom_document_metadata()`, classmethod `from_mapping()` (accepts both
  `meta`/`key` and legacy `data`/`id`). models.py:34.
- `Row(key, meta, document=None, vector=None, chunk=None, lineage=None,
  approximate=False, codec="none")` — read result; `.text` prop aliases
  `document.text`. models.py:258.
- `Hit(key, distance, meta, document=None, vector=None, chunk=None, lineage=None,
  approximate=False, codec="none")` — search hit; `.text` prop. models.py:313.
- `WalRecord(op, arena, key, vector=None, meta=None, document=None, chunk=None,
  lineage=None)` — `.is_delete` prop, classmethod `from_entry(WalEntry)`. Note
  field is named `arena` (built from `entry.vault`). models.py:370.
- `CollectionHealth(name, live, pending_removals, dimension, metric, read_only,
  sealed, codec="none", code_bytes=0)` — `.tombstone_ratio` prop. Docstring
  signature line says `-> ArenaHealth` (legacy return-name wording). models.py:449.

## Module: `elips/gpu/accelerator.py` — GPU facade

Imports native GPU symbols inside a `try/except ImportError` → `_GPU_AVAILABLE`.
Degrades gracefully: `accelerators()` returns `[]` and `accelerator()` returns
`None` on CPU-only builds.

- `AcceleratorSpec` (frozen dataclass): `name, backend, index, memory_bytes,
  free_memory_bytes, unified_memory, supports_fp16, raw(GpuDeviceInfo)`; `.memory_gb`
  prop; `.can_fit(n_vectors, dimension, *, config=None) -> bool`; classmethod
  `from_info()`. gpu/accelerator.py:42.
- `Accelerator(device)` ctx-mgr wrapping `GpuDevice`: props `raw, spec, backend,
  idle, closed`; methods `reserve(pool_bytes=0)`, `distances(queries, corpus, *,
  metric="cosine")`, `nearest(distances, *, top)`, `search(queries, corpus, *, top,
  metric="cosine")`, `synchronize()`, `memory_usage() -> (used, available, peak)`,
  `kernel_timings(limit=100)`, `close()`. gpu/accelerator.py:147.
- `accelerators() -> list[AcceleratorSpec]`, `accelerator(config=None) ->
  Accelerator | None`. gpu/accelerator.py:399, 422.
- Status: complete; **the GPU facade is not exercised by `tests/python`** — GPU
  coverage lives in C++ (`tests/unit/gpu`, `tests/integration/gpu`) and the Python
  GPU surface tests in `test_exposed_surface.py` are `skipif(no device)`. On a
  Metal host `has_gpu` is True but these facade methods remain **untested in Python**.

## Re-export subpackages (thin, no new runtime objects)

- `elips/query/__init__.py` — EQL: `parse`(=`parse_eql`), `tokenize`(=`tokenize_eql`),
  `validate`(=`validate_eql`), plus AST types `SearchStatement`, `FetchStatement`,
  `ScanStatement`, `InsertStatement`, `DeleteStatement`, `Token`, `TokenKind`,
  `VectorRef`, `ParseError`. These AST types are the **sanctioned home** for the
  `DEMOTED` names (importing them here does **not** warn). Verified at runtime:
  `parse('seek in docs nearest $q top 5 yield')` → `SearchStatement(vault='docs',
  top=5)`. Tested: `test_bindings.py::test_eql_tooling` (imports `TokenKind`).
- `elips/search/__init__.py` — re-exports native `Comparator`, `Filter`,
  `QueryPlan`, `QueryStrategy`, `Result`. Tested (`test_filter_*`).
- `elips/storage/__init__.py` — native `WalEntry`, `WalOp`, `replay_wal` + Py
  `WalRecord`. Tested via `pending_writes`.
- `elips/quantization/__init__.py` — native `Codec`, `QuantParams`. Tested.

## Supporting modules

- `elips/typing.py` — type aliases + `TypedDict`s (`RecordInputDict`, `BatchRecord`,
  `StoredRecord`) + `Protocol`s (`TextEmbedderFn`). No runtime objects. typing.py.
- `elips/_internal/protocols.py` — `Embedder` (runtime-checkable Protocol),
  `RecordInputLike`. protocols.py:23.
- `elips/_internal/coercion.py` — `coerce_record_input`,
  `build_record_inputs_from_columns`. Pure Py, tested transitively.
- `elips/exceptions.py` — re-exports the native exception hierarchy. **ORPHAN:**
  nothing in the package imports it (`__init__.py` pulls exceptions straight from
  `_native`, not from this module); only reachable as `elips.exceptions`. Its
  docstring example uses legacy `elips.open(...).vault("docs").place([1.0])`.
  file:line exceptions.py:1.
- `elips/core.pyi`, `elips/_core.pyi` — type stubs; `_core.pyi` declares the full
  (incl. GPU) native surface unconditionally. `py.typed` marker present.

---

## Native handles (compiled `_core`, reached via `elips.native.*`)

All C++-backed; signatures from the compiled docstrings. The high-level facades
are thin wrappers over these.

- **`Database`**: `open`/`open_with_config` return it. Methods: `vault(name)`,
  `list_vaults()`, `begin_transaction() -> Transaction`, `query(eql, bindings={})
  -> list[Result]`, `checkpoint()`, `compact()`, `vacuum()`, `quantize()`,
  `quantize_all()`, `abandon()`, `close()`; props `closed`, `config`, `path`,
  `persistent`, `gpu_info()`, `gpu_stats()`. Tested (`test_database_crud`,
  `test_bindings` EQL + transaction).
- **`Vault`**: `place(vector, data={}, id=None, document=None, chunk=None,
  lineage=None) -> str`, `place_document(text, data={}, id=None, chunk=None,
  lineage=None) -> str`, `place_many(records) -> None`, `seek(vector, top=10,
  where=<all>, threshold=None)`, `seek_text(text, top=10, where, threshold)`,
  `seek_hybrid(vector, text, top=10, where, threshold, lexical_weight=0.25)`,
  `explain_seek(...)`, `fetch(id) -> object|None`, `scan(where, offset=0, limit=-1)`,
  `records()`, `erase(id) -> bool`, `count()`, `info() -> VaultInfo`,
  `rebuild_index()`, `vacuum()`, `quantize()`, `set_read_only(bool)`; props
  `name, read_only, sealed, pending_removals, codec, quantized`. Tested heavily.
- **`Config`**: fluent builder — `dimension/metric/index/access_mode/durability/
  segmented_storage/metadata_acceleration/auto_text_embedder/graph_params/
  quantization/gpu/text_embedder/local_text_embedder` setters, each returning
  `Config`; `*_val`/`*_enum` getters; `has_text_embedder`, `has_quantization`,
  `has_gpu`, `has_pending_local_text_embedder`. Tested (`test_config`,
  `test_config_v2_surface`).
- **`Filter`**: `compare/equals/not_equals/gt/gte/lt/le/in_set/one_of/contains/
  has_substring/field` builders, `and_/or_/not_` combinators, `matches/matches_all`,
  `exact_constraints`. Tested extensively (`test_exposed_surface`).
- **`Result`**: props `id, distance, data, document, chunk, lineage, approximate,
  codec`. **`VaultInfo`**: `count, dimension, metric, codec, code_bytes,
  compression_ratio`. **`Transaction`**: `vault(name) -> TransactionVault`,
  `commit()`, `rollback()`. **`TransactionVault`**: `place()`, `erase()`. Tested
  (`test_transaction`).
- **GPU natives** (`GpuDevice`, `GpuConfig`, `GpuDeviceInfo`, `GpuMemory`,
  `GpuProfiler`, `GpuPolicy`, `GpuPrecision`, `GpuIndexAlgorithm`, …) plus free
  functions `gpu_select`, `gpu_devices`, `gpu_can_fit_index`,
  `gpu_runtime_device_info`, `gpu_cpu_fallback_info`, `gpu_error_message`. Present
  (Metal). Python-side **untested** (C++ tests cover them).
- Vector/EQL/metric free functions: `distance`, `magnitude`, `normalize`,
  `generate_id`, `is_valid_id`, `replay_wal`, `parse_eql`, `tokenize_eql`,
  `validate_eql`, `metric_from_string`, `metric_to_string`,
  `requires_normalization`, `describe_local_embedder`. Tested.

---

## Public vocabulary — what exists today

| Term | Exists? | Where / note |
|------|---------|--------------|
| `connect` | ✅ | high-level `elips.connect` (config/connect.py:58) |
| `open` | ✅ | native free fn `elips.open` → `Database` (used by `test.py`) |
| `Engine` | ✅ | high-level DB wrapper (database/engine.py) |
| `Collection` | ✅ | high-level vault wrapper (collection/collection.py:48) |
| `arena` / `Arena` | ✅ | legacy aliases: `Arena = Collection`, `Engine.arena = collection` |
| `vault` | ✅ | native `Database.vault(name) -> Vault` |
| `add` / `write` | ✅ | `Collection.add` is an alias of `.write` |
| `ingest` | ✅ | `Collection.ingest` (structured or column batches) |
| `merge` | ✅ | `Collection.merge` → alias of `ingest` |
| `place` | ✅ | native `Vault.place` |
| `place_document` | ✅ | native `Vault.place_document` (used by `test.py`) |
| `place_many` | ✅ | native `Vault.place_many` |
| `probe` | ✅ | `Collection.probe` → `Vault.seek` |
| `search` | ✅ | `Collection.search` alias of `probe_text` |
| `seek` | ✅ | native `Vault.seek` |
| `seek_text` | ✅ | native `Vault.seek_text` (used by `test.py`) |
| `seek_hybrid` | ✅ | native `Vault.seek_hybrid` |
| `explain_seek` | ✅ | native `Vault.explain_seek` (`Collection.explain`) |
| `explain_seek` as `explain` | ✅ | `Collection.explain` |
| `erase` | ✅ | native `Vault.erase`; high-level `discard`/`delete` |
| `top=` | ✅ | kwarg on all seek/probe/search |
| `meta` | ✅ | high-level kwarg; native equivalent is `data` |
| `probe_text` / `probe_hybrid` | ✅ | `Collection` methods |
| EQL `.query(...)` | ⚠️ native only | `Database.query`; **no** `Engine.query` (see mismatch) |
| `Client` / `PersistentClient` | ❌ | **do not exist** (Chroma vocabulary; absent) |
| `collection.get` / `delete` / `search_vector` | ✅ | `Collection` aliases |
| `Database.persistent` | ✅ | native prop (`False` for `:memory:`) |

---

## Dead / mismatched bindings

**No Python-only shim pretends to be native.** Every facade (`Engine`,
`Collection`, `Accelerator`, record dataclasses) is openly Python and delegates
to a native handle. Conversely, every name bound in Python resolves to a real
`_core` object — there are no names exported that lack a C++ definition. The
findings below are *reachable-but-unproductive* bindings and *doc/behavior*
disagreements, not fabricated natives.

### Dead bindings (reachable in C++, nothing in the API produces them)
1. **`IndexSnapshot`, `IvfSnapshot`, `PqSnapshot`, `IndexSnapshotKind`** — fully
   built pybind classes (real accessors: `IndexSnapshot.{dimension,ids,ivf,kind,
   metric,pq,vectors}`, etc.) but **no binding returns one**. `_native.DEMOTED`
   maps them to `None` ("engine internal with no public replacement, and nothing
   in the API returns one"). Constructible placeholders only. Accessing via
   `elips.*` or `elips.native.*` warns `DeprecationWarning`. Evidence:
   `_native.py:56–69`; `elips_python.cpp:834+` binds the enum; runtime confirms
   the classes exist with accessors but no producer. Status: **dead**, slated for
   removal (deprecated this minor version).

### Mismatched bindings (docstring/signature vs behavior)
2. **`Engine` docstring claims transactions, `Engine` has none.** `database/engine.py:1`
   docstring: *"...exposes WAL replay, checkpointing, and transactions."* `Engine`
   exposes no transaction entry point; transactions exist only on native
   `Database.begin_transaction()`. Fix: drop the claim or add
   `Engine.transaction()`.
3. **`elips.query` docstring claims `Engine.query`, which does not exist.**
   `query/__init__.py` docstring: *"Executing a query is `elips.Engine.query`."*
   There is **no** `Engine.query`. EQL execution exists only as native
   `Database.query(eql, bindings={})` (tested at `test_bindings.py:492`). Reachable
   today only via `engine.native.query(...)`. Fix: add `Engine.query` or correct
   the docstring to point at `Database.query`.
4. **`elips.query` / `.search` / `.storage` / `.quantization` are not attributes
   of a bare `elips`.** Only `core`/`modern`/`native` are in `_LAZY`
   (`__init__.py:68`). `import elips; elips.query` raises `AttributeError` until an
   explicit `import elips.query`. The docstrings' `from elips.query import ...`
   form works, so impact is low, but `elips.query.*` dotted access after a plain
   `import elips` silently fails. Candidate: add the four subpackages to `_LAZY`.
5. **`elips/exceptions.py` is an orphan module.** Not imported anywhere in the
   package (root pulls exceptions from `_native`). Harmless duplicate; its
   docstring example uses legacy `open/vault/place`. Candidate for deletion or
   wiring-in.

### Vocabulary drift (not a bug, but a re-architecture signal)
6. **Two parallel, both-live vocabularies.** High-level (`connect/collection/add/
   search/ingest/discard/probe`) and native (`open/vault/place/seek_text/erase`)
   both work. `test.py` (the shipped example) uses the **native** vocabulary;
   the package docstrings lead with the **high-level** one. `meta` (high-level)
   vs `data` (native) name the same payload. `Arena`/`arena` are retained legacy
   aliases of `Collection`/`collection`. No `Client`/`PersistentClient` exists.

---

## Test coverage summary

- `tests/python/`: 239 passed / 5 skipped. Native CRUD, filters, transactions,
  EQL tooling, config, local embedder, GIL release, the seam, and packaging are
  well covered. High-level API is tested through the **legacy `arena`/`write`/
  `probe`** spelling.
- **Untested by name** (back onto tested code, but the alias/method itself is never
  invoked in `tests/python`): `Collection.add`, `add_many`, `search_vector`,
  `delete`; `Engine.collection_names` (only `vault_names` tested); the GPU
  `Accelerator`/`AcceleratorSpec` facade methods (device-gated skips).
- **Not run as tests:** the modern-vocabulary docstring doctests (no doctest
  runner configured).
- `test.py` (package root): runs green against the native API.






