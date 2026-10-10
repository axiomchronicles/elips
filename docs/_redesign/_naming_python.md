# Phase 5B — Python Naming Map (old → new), DEEP rename (ADR-R2)

Planning only. No code changed. Enumerates every public Python symbol in
`bindings/python/elips/**.py` + the pybind names in `bindings/python/elips_python.cpp`
and proposes a new name per ADR-R2 (`04_architecture.md:22`). Target vocabulary =
Chroma Collection API (`get_or_create_collection`; `add/upsert/get/update/delete/query/count/peek/modify`;
`ids/embeddings/documents/metadatas/n_results/where/where_document/include/limit/offset`,
`01_faiss_study.md:79-85`, `_faiss_3G.md:39-99`) over a FAISS-recognizable `Index` layer
(`index_factory`, `read_index`/`write_index`, `IndexFlat`/`IndexIVFFlat`/`IndexIVFPQ`/`IndexHNSW`).

Path shorthand: `cpp` = `bindings/python/elips_python.cpp`; all other paths are under
`bindings/python/elips/`. **Shim?** = whether a `DeprecationWarning` alias keeps the old
name for one minor (removed next minor, per ADR-R2).

**Collision rule (ADR-R2):** where a Chroma name is a C++ keyword (`delete`), the C++
method is `remove`; Python/pybind still exposes `delete` (a method-name string, legal in
pybind `.def("delete", &Collection::remove)`).

---

## 1. Entry points (Client tier)

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| _(none)_ | `Client()` | class (NEW) | — | Chroma ephemeral in-memory client; wraps engine opened at `":memory:"`. B3. |
| _(none)_ | `PersistentClient(path, *, ...)` | class (NEW) | — | Chroma on-disk client; `path=` first positional. B3. |
| `Engine` (database/engine.py:29) | `Client` | class | yes (`Engine = Client`) | ADR-R2 `Database`→`Client`-tier. The typed facade becomes `Client`; `Client()`/`PersistentClient()` are its constructors. |
| `connect` (config/connect.py:58) | `PersistentClient` / `Client` | func | yes | Deprecated shim → `PersistentClient(path, **kw)` (or `Client()` when `path==":memory:"`). |
| `connect_with_config` (config/connect.py:159) | `Client.from_settings` | func | yes | Deprecated shim; `Config`→`Settings` (see §9). |
| `open` (cpp:2801) | `_open` (private) + `PersistentClient` | func | yes | Low-level native open; stops being public. `open(index=…, access_mode=…)` kwargs move to Settings. |
| `open_with_config` (cpp:2862) | `_open_with_settings` (private) | func | yes | Native, demoted to private seam. |

**New `Client`/`PersistentClient` methods to ADD (Chroma, `_faiss_3G.md:39-45`):**
`create_collection(name, *, metadata=, configuration=)`, `get_collection(name)`,
`get_or_create_collection(name, ...)`, `delete_collection(name)`,
`list_collections(*, limit=100, offset=0)`, `reset()`.

---

## 2. `Engine`→`Client` members (database/engine.py)

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `native` (engine.py:58) | `native` | property | — | Keep; ELIPS escape hatch, no Chroma equiv. Returns the native `Collection`-tier handle (§6). |
| `raw` (engine.py:82) | `native` | property alias | yes | Already a legacy alias; stays deprecated. |
| `config` (engine.py:84) | `settings` | property | yes (`config`) | Chroma spells it `Settings`; keep `config` as shim. |
| `collection` (engine.py:93) | `get_or_create_collection` | method | yes (`collection`) | Lazy create matches Chroma idempotent create. Add `create_collection`/`get_collection` (§1). |
| `arena` (engine.py:137) | `get_or_create_collection` | method alias | yes | Pre-1.1 name; stays deprecated. |
| `checkpoint` (engine.py:139) | `checkpoint` | method | — | ELIPS durability extension; keep. |
| `compact` (engine.py:147) | `compact` | method | — | Keep (ELIPS; FAISS/Chroma have none). |
| `vacuum` (engine.py:155) | `vacuum` | method | — | Keep. |
| `collection_names` (engine.py:181) | `list_collections` | method | yes (`collection_names`) | Chroma name. NOTE semantic drift: Chroma returns Collection objects; ELIPS returns `list[str]`. Open question §11. |
| `vault_names` (engine.py:199) | `list_collections` | method alias | yes | Deprecated. |
| `pending_writes` (engine.py:201) | `pending_writes` | method | — | ELIPS WAL extension; keep. |
| `close` (engine.py:234) | `close` | method | — | Keep. |
| `DEFAULT_TEXT_SLOT` (engine.py:26) | `_DEFAULT_TEXT_SLOT` | module const | yes | Vestigial (text_slot is a reserved no-op); make private. |
| `collection(..., text_slot=)` kwarg (engine.py:98) | drop (deprecated) | kwarg | yes | Reserved compat arg, no runtime effect; keep accepting + ignore for one minor. |

---

## 3. `Collection` facade (collection/collection.py)

Container class name `Collection` already matches Chroma — **no rename**
(collection/collection.py:48). Methods below collapse ELIPS's verb sprawl onto the Chroma
nine. Shims keep every current spelling for one minor.

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `name` (collection.py:85) | `name` | property | — | Chroma Collection has `.name`. Keep. |
| `raw` (collection.py:94) | `native` | property | yes (`raw`) | Align with `Client.native` naming. |
| `count` (collection.py:103) | `count` | method | — | Chroma `count()`. Keep. |
| `write` (collection.py:138) | `add` | method | yes (`write`) | Chroma `add`. `add` already aliases `write` (collection.py:939) — promote `add` to canonical. |
| `write_many` (collection.py:211) | `add` (batch) | method | yes (`write_many`,`add_many`) | Fold into `add` column form (`ids=/embeddings=/documents=/metadatas=`). `add_many` (collection.py:942) stays deprecated. |
| `ingest` (collection.py:306) | `add` / `upsert` | method | yes (`ingest`) | Column-oriented bulk write → `add`. ADR verb map: `ingest`→`add`. |
| `merge` (collection.py:392) | `upsert` | method | yes (`merge`) | `merge` was the upsert-ish alias; promote to Chroma `upsert`. See §3a (native `place` is already upsert, `02_gap_analysis.md`). |
| _(none)_ | `upsert` | method (NEW) | — | Chroma; explicit id-overwrite write. Backed by native `add` (upsert semantics). |
| _(none)_ | `update` | method (NEW) | — | Chroma; update existing records only (error if id absent → `NotFound`). |
| `probe` (collection.py:440) | `query` | method | yes (`probe`,`search_vector`) | Chroma `query` (vector form via `query_embeddings=`). `search_vector` alias (collection.py:948) deprecated. |
| `probe_text` (collection.py:487) | `query` | method | yes (`probe_text`,`search`) | Chroma `query` (text form via `query_texts=`). `search` alias (collection.py:945) deprecated. Unify both into one `query`. |
| `probe_hybrid` (collection.py:562) | `query(..., lexical_weight=)` | method | yes (`probe_hybrid`,`search_hybrid`) | ELIPS hybrid has no Chroma equiv; keep as `query` with `lexical_weight=`. `search_hybrid` (collection.py:951) deprecated. |
| `explain` (collection.py:601) | `explain` | method | — | ELIPS planner extension; keep (no Chroma name). |
| `pull` (collection.py:634) | `get` | method | yes (`pull`,`get`) | Chroma `get(ids=, ...)`. `get` already aliases `pull` (collection.py:957); promote. |
| `sweep` (collection.py:663) | `get` (filter form) | method | yes (`sweep`) | Chroma `get(where=, limit=, offset=)` subsumes scan. Merge into `get`. |
| `discard` (collection.py:696) | `delete` | method | yes (`discard`) | Chroma `delete(ids=, where=, where_document=)`. `delete` already aliases (collection.py:954); promote. |
| `pending_removals` (collection.py:744) | `pending_removals` | property | — | ELIPS extension; keep. |
| `read_only` (collection.py:773) | `read_only` | property | — | Keep. |
| `sealed` (collection.py:782) | `sealed` | property | — | Keep. |
| `freeze` (collection.py:794) | `freeze` | method | — | ELIPS extension; keep. |
| `vacuum` (collection.py:825) | `vacuum` | method | — | Keep. |
| `rebuild` (collection.py:850) | `rebuild` | method | — | Keep (FAISS has no public rebuild). |
| `compress` (collection.py:861) | `compress` | method | yes (adds `train` alias) | FAISS calls codebook training `train`; expose `train` alias. |
| `health` (collection.py:899) | `health` | method | — | ELIPS extension; keep. |
| `add`/`add_many`/`search`/`search_vector`/`search_hybrid`/`delete`/`get` aliases (collection.py:939-958) | see rows above | aliases | — | Several become the canonical name; the rest demote to deprecated shims. |
| `Arena = Collection` (collection.py:964; collection/__init__.py:5) | `Collection` | class alias | yes | Deprecated. |

### 3a. Collection kwargs (collection.py)

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `top=` (probe collection.py:443; probe_text:491; probe_hybrid:567; explain:605) | `n_results` (alias `k`) | kwarg | yes (`top`) | Chroma `n_results`, default 10. `k` alias per task; `limit` reserved for `get`. |
| `meta=` (write collection.py:145) | `metadatas` (batch) / `metadata` (single) | kwarg | yes (`meta`) | Chroma: `metadatas` on `add`/batch, `metadata` on single. |
| `keys=` / `key=` (write:146; pull:636; discard:698) | `ids` / `id` | kwarg | yes (`keys`,`key`) | Chroma `ids`. |
| `vector=` / `vectors=` (write:143; ingest:311) | `embeddings` / `embedding` | kwarg | yes | Chroma `embeddings`. |
| `text=` / `texts=` (write:144; ingest:312) | `documents` / `document_text` | kwarg | yes | Chroma `documents`. NOTE collides with existing `document=` (DocumentAttachment); keep `document=` as the attachment, `documents=` as text list. §11. |
| `where=` (probe:445 etc.) | `where` | kwarg | — | Already Chroma-named; adopt `$`-operator dict grammar (`$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$and/$or`). |
| `max_distance=` (probe:446) | `max_distance` | kwarg | — | ELIPS extension; keep (no Chroma equiv). |
| `include_vectors=` (probe:447; pull:638; sweep:669) | `include=[...]` | kwarg | yes (`include_vectors`) | Chroma `include=["embeddings","documents","metadatas","distances"]`. Replace the bool with the list form. |
| `lexical_weight=` (probe_text:495; probe_hybrid:570) | `lexical_weight` | kwarg | — | ELIPS extension; keep. |
| `offset=` / `limit=` (sweep:667-668) | `offset` / `limit` | kwarg | — | Already correct; surface on `get`. Chroma `list_collections` also uses them. |
| _(none)_ | `where_document=` | kwarg (NEW) | — | Chroma content filter (`$contains`/`$not_contains`/`$regex`), on `query`/`get`/`delete`. |
| _(none)_ | `query_embeddings=` / `query_texts=` | kwarg (NEW) | — | Chroma `query` input names (plural, batched). |
| _(none)_ | `uris=` | kwarg (NEW, optional) | — | Chroma accepts; maps to `DocumentAttachment.uri`. Low priority. |

---

## 4. Record models (records/models.py)

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `RecordInput` (models.py:34) | `RecordInput` | dataclass | — | Internal write model; keep name. Rename its fields to match §3a. |
| `RecordInput.vector` (models.py:81) | `embedding` | field | yes | — |
| `RecordInput.text` (models.py:82) | `document` (text) | field | yes | Clashes with `.document` attachment; see §11. |
| `RecordInput.meta` (models.py:83) | `metadata` | field | yes | — |
| `RecordInput.key` (models.py:84) | `id` | field | yes | — |
| `RecordInput.document/chunk/lineage` (models.py:85-87) | same | field | — | ELIPS extensions; keep. |
| `RecordInput.from_mapping` (models.py:183) | `from_mapping` | classmethod | — | Accept both old (`meta`/`key`/`data`/`id`) and new (`metadata`/`id`) keys. |
| `Row` (models.py:258) | `Row` | dataclass | — | Read model; keep. |
| `Row.key/meta` (models.py:283-284) | `id`/`metadata` | field | yes | Align. |
| `Row.vector` (models.py:286) | `embedding` | field | yes | — |
| `Hit` (models.py:313) | `Hit` | dataclass | — | Keep. |
| `Hit.key/meta/vector` (models.py:339-343) | `id`/`metadata`/`embedding` | field | yes | — |
| `Hit.distance` (models.py:340) | `distance` | field | — | Chroma returns `distances`; field name stays. |
| `WalRecord` (models.py:370) | `WalRecord` | dataclass | — | ELIPS extension; keep. |
| `WalRecord.arena` (models.py:400) | `collection` | field | yes | ADR `Vault`/arena→collection. |
| `WalRecord.key/meta` (models.py:401-403) | `id`/`metadata` | field | yes | — |
| `WalRecord.op` values `erase` (models.py:399) | `delete` | enum string | yes | Align WAL op names (see §7 `WalOp`). |
| `CollectionHealth` (models.py:449) | `CollectionHealth` | dataclass | — | Keep (docstring return-type says `ArenaHealth` — fix, models.py:450). |
| `CollectionHealth.name` (models.py:479) | `name` | field | — | Keep. |
| `ArenaHealth = CollectionHealth` (modern.py:26) | `CollectionHealth` | alias | yes | Deprecated. |

---

## 5. GPU facade (gpu/accelerator.py)

ELIPS keeps its `Accelerator`/`AcceleratorSpec` ergonomic layer; FAISS GPU names are
added as thin aliases for recognizability (`index_cpu_to_gpu` etc. land in §6).

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `Accelerator` (gpu/accelerator.py:147) | `Accelerator` | class | — | Keep; ELIPS extension over FAISS `GpuResources`. |
| `AcceleratorSpec` (gpu/accelerator.py:43) | `AcceleratorSpec` | dataclass | — | Keep. |
| `accelerator(config=)` (gpu/accelerator.py:422) | `accelerator(config=)` | func | — | Keep. |
| `accelerators()` (gpu/accelerator.py:399) | `accelerators()` | func | — | Keep. Back `get_num_gpus()` with `len(accelerators())` (§6). |
| `Accelerator.distances/nearest/search` (gpu/accelerator.py:232/272/293) | same | method | — | Keep; `top=` on `nearest`/`search` (gpu/accelerator.py:272,298) → `k=` (FAISS top-k name). Shim `top`. |
| `Accelerator.reserve/synchronize/memory_usage/kernel_timings/close` | same | method | — | Keep. |
| `AcceleratorSpec.can_fit(..., config=)` (gpu/accelerator.py:88) | same | method | — | Keep. |
| props `backend/idle/closed/spec/raw/memory_gb` | same | property | — | `raw`→`native` for consistency (shim `raw`). |

---

## 6. FAISS Index API in Python (all NEW — Phase 6 exposure, B2)

ELIPS's C++ `ExactIndex`/`HierarchicalGraphIndex`(HNSW)/`IndexFactory` exist but are **not
exposed in Python** (`00_inventory.md:32`). These are additions, not renames.

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| _(none)_ | `IndexFlat` | class (NEW) | — | Wrap `ExactIndex`. FAISS parity. |
| _(none)_ | `IndexHNSW` | class (NEW) | — | Wrap `HierarchicalGraphIndex`. |
| _(none)_ | `IndexIVFFlat` | class (NEW) | — | Needs CPU IVF (Phase 6, currently GPU-only). |
| _(none)_ | `IndexIVFPQ` | class (NEW) | — | Needs CPU IVF/PQ wiring. |
| _(none)_ | `index_factory(d, "IVF4096,PQ16", metric)` | func (NEW) | — | B2 grammar parser. |
| _(none)_ | `read_index(path)` / `write_index(index, path)` | func (NEW) | — | Index-level serialization. |
| _(none)_ | `index_cpu_to_gpu(resources, device, index)` | func (NEW) | — | Chroma/FAISS GPU transfer; backed by former `IndexSnapshot` transfer path (now dead, §8). |
| _(none)_ | `index_gpu_to_cpu(index)` | func (NEW) | — | — |
| _(none)_ | `get_num_gpus()` | func (NEW) | — | FAISS Python-level helper (`01_faiss_study.md:23`); `== len(accelerators())`. |
| _(none)_ | `StandardGpuResources` | class (NEW) | alias→`Accelerator` | FAISS-recognizable name for the resources handle; alias ELIPS `Accelerator`/`GpuDevice`. |

---

## 7. Native pybind symbols (cpp) — deep rename per ADR-R2

Native handles live under `elips.native.*` (and legacy `elips.core`, deprecated). Module
scoping means native `Collection` ≠ high-level `elips.Collection` (different modules).

### 7a. Core handles

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `Vault` (cpp:2297) | `Collection` | pybind class | yes (`Vault`) | ADR `Vault`→`Collection`. Native-module scoped. |
| `Database` (cpp:2682) | `Client` | pybind class | yes (`Database`) | ADR `Database`→`Client`-tier. |
| `TransactionVault` (cpp:2243) | `TransactionCollection` | pybind class | yes | — |
| `Transaction` (cpp:2266) | `Transaction` | pybind class | — | Keep. |
| `VaultInfo` (cpp:2008) | `CollectionInfo` | pybind class | yes | — |
| `Result` (SearchResult, cpp:2044) | `Result` | pybind class | — | Keep (website drift called it `SearchResult`; `00_inventory.md:75`). |
| `Filter` (cpp:2072) | `Filter` | pybind class | — | Keep; adopt `$`-op dict parsing alongside fluent builder. |
| `QueryPlan` (cpp:1992) | `QueryPlan` | pybind class | — | Keep. |

### 7b. `Vault`→`Collection` methods (cpp)

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `place` (cpp:2351) | `add` | method | yes (`place`) | ADR `place`→`add`. Upsert semantics stay. |
| `place_document` (cpp:2388) | `add_document` | method | yes | — |
| `place_many` (cpp:2417) | `add_many` | method | yes | — |
| `seek` (cpp:2480) | `search` | method | yes (`seek`) | ADR `seek`→`search`. |
| `seek_text` (cpp:2494) | `search_text` | method | yes | — |
| `seek_hybrid` (cpp:2510) | `search_hybrid` | method | yes | — |
| `explain_seek` (cpp:2528) | `explain_search` | method | yes | — |
| `fetch` (cpp:2545) | `get` | method | yes | Chroma `get` (by id). |
| `erase` (cpp:2559) | `delete` (C++ method `remove`) | method | yes (`erase`) | Collision rule: pybind `.def("delete", &remove)`. |
| `scan` (cpp:2565) | `scan` | method | — | Keep native `scan`; high-level `get(where=)` routes here (avoids `get` name clash with `fetch`). §11. |
| `count` (cpp:2592) | `count` | method | — | Keep. |
| `info` (cpp:2587) | `info` | method | — | Returns `CollectionInfo`. |
| `rebuild_index` (cpp:2597) | `rebuild` | method | yes | — |
| `vacuum` (cpp:2601) | `vacuum` | method | — | Keep. |
| `quantize` (cpp:2604) | `train` | method | yes (`quantize`) | FAISS codebook-training term; high-level stays `compress`. |
| `quantized` (cpp:2626) | `is_trained` | property | yes | FAISS `is_trained`. |
| `codec` (cpp:2630) | `codec` | property | — | Keep. |
| `records` (cpp:2636) | `records` | method | — | Keep (ELIPS). |
| `set_read_only` (cpp:2657) | `set_read_only` | method | — | Keep. |
| `read_only` (cpp:2661) | `read_only` | property | — | Keep. |
| `sealed` (cpp:2663) | `sealed` | property | — | Keep. |
| `pending_removals` (cpp:2667) | `pending_removals` | property | — | Keep. |
| `name` (cpp:2671) | `name` | property | — | Keep. |
| kwargs `top=` (seek cpp:2491), `data=` (place cpp:2255), `id=` (cpp:2256) | `n_results`/`k`, `metadatas`, `ids` | kwarg | yes | Mirror §3a at native layer. |

### 7c. `Database`→`Client` methods (cpp)

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `vault` (cpp:2732) | `collection` | method | yes (`vault`) | — |
| `list_vaults` (cpp:2734) | `list_collections` | method | yes | — |
| `begin_transaction` (cpp:2735) | `begin_transaction` | method | — | Keep (ELIPS). |
| `checkpoint`/`compact`/`vacuum` (cpp:2741/2743/2745) | same | method | — | Keep. |
| `quantize` (cpp:2748) | `train_collection` | method | yes (`quantize`) | — |
| `quantize_all` (cpp:2753) | `train_all` | method | yes | — |
| `close`/`abandon` (cpp:2756/2757) | same | method | — | Keep. |
| `path`/`persistent`/`closed` (cpp:2758/2761/2764) | same | property | — | Keep. |
| `query` (cpp:2767) | `query_eql` | method | yes (`query`) | Disambiguate from Chroma collection `query`; EQL executor. Fixes the `elips.query` docstring lie (§10). |
| `gpu_info`/`gpu_stats` (cpp:2780/2781) | same | method | — | Keep (GPU build only). |
| `config` (cpp:2783) | `settings` | property | yes (`config`) | — |

### 7d. EQL AST + lexer (query/__init__.py; cpp)

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `VectorRef` (cpp:528) | `VectorRef` | class | — | Keep in `elips.query`. |
| `SearchStatement` (cpp:544) | `SearchStatement` | class | — | Keep. Field `.vault` (cpp:547) → `.collection` (shim). `.top` (cpp:549) → `.n_results` (shim). |
| `FetchStatement` (cpp:563) | `FetchStatement` | class | — | `.vault`→`.collection`. |
| `ScanStatement` (cpp:572) | `ScanStatement` | class | — | `.vault`→`.collection`. |
| `InsertStatement` (cpp:583) | `InsertStatement` | class | — | `.vault`→`.collection`. |
| `DeleteStatement` (cpp:601) | `DeleteStatement` | class | — | `.vault`→`.collection`. |
| `Token`/`TokenKind` (cpp:639/632) | same | class/enum | — | Keep. |
| `parse_eql`/`tokenize_eql`/`validate_eql` (cpp:610/523/515) | re-exported `parse`/`tokenize`/`validate` (query/__init__.py:33-35) | func | — | Keep public spelling; underlying `*_eql` names unchanged. |

### 7e. Config / params / embedder (cpp)

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `Config` (cpp:998) | `Settings` | pybind class | yes (`Config`) | Chroma `Settings`; fluent builder stays. Docstring refs `open`/`open_with_config` (cpp:1000-1002) → `PersistentClient`. |
| `Config.index(type)` (cpp:1058) | `.index_type(type)` + accept factory string | method | yes | Bridge to `index_factory` grammar (§6). |
| `Config` setters `dimension/metric/graph_params/quantization/durability/access_mode/segmented_storage/metadata_acceleration/auto_text_embedder/local_text_embedder/text_embedder/gpu` (cpp:1046-1157) | same names | method | — | Keep; `quantization`→ also accept `train`-ready params. |
| `Config` readonly getters `*_val`/`*_enum`/`has_*` (cpp:1158-1269) | same (group) | property | — | Introspection surface; keep as-is (not individually renamed). |
| `GraphParams` (cpp:652) | `GraphParams` | class | — | Keep ELIPS descriptive names; **record FAISS aliases** `max_connections=M`, `ef_construction=efConstruction`, `ef_search=efSearch` (`02_gap_analysis.md` Phase 5). |
| `QuantParams` (cpp:702) | `QuantParams` | class | — | Keep; props `pq_dim/pq_bits` note FAISS `m`/`nbits` aliases. `validate` (cpp:783)/`code_bytes` (cpp:776) keep. |
| `Codec` enum (cpp:696) | `Codec` | enum | — | Keep. |
| `LocalEmbedderConfig` (cpp:797) | `LocalEmbedderConfig` | class | — | Keep (ELIPS extension). |
| `TextEmbedderInfo` (cpp:813) | `TextEmbedderInfo` | class | — | Keep. |
| `describe_local_embedder` (cpp:974) | `describe_local_embedder` | func | — | Keep. |

### 7f. Enums (cpp)

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `Metric` (cpp:390) | `Metric` | enum | — | Keep. `euclidean` returns true √L2 — document vs FAISS squared L2 (`02_gap_analysis.md` Phase 6). |
| `IndexType` (cpp:395) | `IndexType` | enum | — | Keep; extend when CPU IVF lands. |
| `Durability` (cpp:399) | `Durability` | enum | — | Keep (ELIPS extension). |
| `Comparator` (cpp:405) | `Comparator` | enum | — | Keep; `$`-op dict is the public form, enum stays for native `Filter`. |
| `AccessMode` (cpp:413) | `AccessMode` | enum | — | Keep. |
| `QueryStrategy` (cpp:417) | `QueryStrategy` | enum | — | Keep (note `text_probe`/`hybrid_fusion` are labels only, `00_inventory.md:42`). |
| `TextEmbedderKind` (cpp:424) | `TextEmbedderKind` | enum | — | Keep. |

### 7g. WAL (storage/__init__.py; cpp)

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `WalOp` (cpp:910) | `WalOp` | enum | — | Keep; value `erase` (cpp:912) → `delete` (shim), `insert_ex`→`insert_extended` (shim). |
| `WalEntry` (cpp:921) | `WalEntry` | class | — | `.vault` (cpp:924) → `.collection` (shim). |
| `replay_wal` (cpp:951) | `replay_wal` | func | — | Keep (ELIPS). |

### 7h. Vector helpers + attachments (cpp)

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `generate_id` (cpp:430) | `generate_id` | func | — | Keep. Docstring refs `Vault.place` (cpp:437) → `Collection.add`. |
| `is_valid_id` (cpp:440) | `is_valid_id` | func | — | Keep. |
| `normalize` (cpp:452) | `normalize` | func | — | Keep. Docstring `Vault.place/seek` (cpp:460-461) → `Collection.add/search`. |
| `magnitude` (cpp:465) | `magnitude` | func | — | Keep. |
| `distance` (cpp:472) | `distance` | func | — | Keep. |
| `requires_normalization` (cpp:488) | `requires_normalization` | func | — | Keep. |
| `metric_to_string`/`metric_from_string` (cpp:501/507) | same | func | — | Keep. |
| `DocumentAttachment` (cpp:1951) | `DocumentAttachment` | class | — | Keep (ELIPS extension). |
| `ChunkInfo` (cpp:1963) | `ChunkInfo` | class | — | Keep. |
| `EmbeddingLineage` (cpp:1974) | `EmbeddingLineage` | class | — | Keep. |

### 7i. GPU classes/funcs (cpp, GPU build)

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `GpuDevice` (cpp:1592) | `GpuDevice` | class | — | Keep; back `StandardGpuResources` alias (§6). `top_k(..., k=)` already FAISS-shaped. |
| `gpu_select` (cpp:1778) | `gpu_select` | func | — | Keep. |
| `gpu_devices` (cpp:1449) | `gpu_devices` | func | — | Keep; `get_num_gpus()`=len. |
| `gpu_runtime_device_info`/`gpu_cpu_fallback_info` (cpp:1469/1462) | same | func | — | Keep. |
| `gpu_can_fit_index` (cpp:1476) | `gpu_can_fit_index` | func | — | Keep. |
| `gpu_error_message` (cpp:1498) | `gpu_error_message` | func | — | Keep. |
| `GpuConfig` (cpp:1372) | `GpuConfig` | class | — | Keep. |
| `GpuDeviceInfo` (cpp:1412) | `GpuDeviceInfo` | class | — | Keep. |
| `GpuError` (cpp:1283) | `GpuError` | enum | — | Keep; `backend_unavailable` pairs with new `BackendUnavailableError` (§9). |
| `GpuPolicy`/`IndexBuildMode`/`GpuIndexAlgorithm`/`GpuPrecision`/`GraphBuildAlgo` (cpp:1295/1302/1307/1314/1322) | same | enum | — | Keep. |
| `GraphIndexBuildParams`/`IvfPqBuildParams`/`GpuIndexBuildParams` (cpp:1327/1338/1351) | same | class | — | Keep. |
| `KernelTiming`/`GpuMemory`/`GpuProfiler`/`BatchStats`/`GpuMetricsSnapshot` (cpp:1360/1507/1537/1566/1807) | same | class | — | Keep (telemetry). |

---

## 8. Dead bindings to RETIRE (not renamed — removed)

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `IndexSnapshot` (cpp:877; `_native.py:62` DEMOTED) | removed | class | no | Never returned by any API (`00_inventory.md:53`). Transfer role absorbed by new `index_cpu_to_gpu` (§6). |
| `IndexSnapshotKind` (cpp:834) | removed | enum | no | Dead. |
| `IvfSnapshot` (cpp:846) | removed | class | no | Dead. |
| `PqSnapshot` (cpp:861) | removed | class | no | Dead. |

Remove their `#include "elips/index_engine/IndexSnapshot.hpp"` (cpp:16) and the `DEMOTED`
entries (`_native.py:62-64`) once gone.

---

## 9. Exceptions (exceptions.py; cpp:372-386)

| Current (file:line) | New | Kind | Shim? | Notes |
|---|---|---|---|---|
| `ElipsError` (cpp:372) | `ElipsError` | exception | — | Base; keep. |
| `DimensionMismatch` (cpp:375) | `DimensionMismatchError` | exception | yes (`DimensionMismatch`) | `*Error` suffix for consistency with new typed errors (task). |
| `InvalidVector` (cpp:377) | `InvalidVectorError` | exception | yes | Suffix align. |
| `ConfigError` (cpp:379) | `ConfigError` | exception | — | Already suffixed; keep. |
| `NotFound` (cpp:380) | `NotFoundError` | exception | yes | Suffix align. |
| `StorageError` (cpp:381) | `StorageError` | exception | — | Keep. |
| `LockConflict` (cpp:383) | `LockConflictError` | exception | yes | Suffix align. |
| `ParseError` (cpp:385) | `ParseError` | exception | — | Keep. |
| _(none)_ | `DimensionMismatchError` | exception (NEW) | — | = renamed `DimensionMismatch` (above); task lists it as a target. |
| _(none)_ | `NotTrainedError` | exception (NEW) | — | Raised by `search` on an untrained/uncompressed-required index (FAISS parity). |
| _(none)_ | `CollectionNotFoundError` | exception (NEW) | — | Subclass of `NotFoundError`; raised by `get_collection`. |
| _(none)_ | `BackendUnavailableError` | exception (NEW) | — | Maps `GpuError::backend_unavailable` (cpp:1291) to a typed Python error. |

`exceptions.py` is currently orphaned (not imported by `__init__.py`, `00_inventory.md:55`) —
wire it into the public surface as the single source for the hierarchy.

---

## 10. Docstring lies to FIX (no symbol rename; correctness)

| Location (file:line) | Problem | Fix |
|---|---|---|
| `database/engine.py:1-13` + class doc (engine.py:30) | `Engine` doc implies transactions it does not implement (`00_inventory.md:54`). | State transactions live on the native `Client`/`Database.begin_transaction`, not the facade; or add `Client.transaction()`. |
| `query/__init__.py:8-9` | "Executing a query is `elips.Engine.query`" — `Engine` has **no** `query` method (engine.py has none). | Point to native `Client.query_eql` (§7c), or add `Client.query_eql` passthrough. |
| `__init__.py:6-11` + `exceptions.py:9-14` | Package/exception docstrings teach old `open()`/`Vault.place()` surface. | Rewrite to `PersistentClient`/`Collection.add`. |
| `collection.py:106` ("vaults are created lazily") & many method docstrings | Say "vault" throughout. | Replace with "collection". |
| `records/models.py:450` | `CollectionHealth` docstring return type reads `ArenaHealth`. | Fix to `CollectionHealth`. |
| website `/docs/python-sdk` (`00_inventory.md:73`) | Denies shipped `add/search/get/delete`. | Phase 9, gated on this map. |

---

## 11. Open questions / collisions (recommended defaults, then proceed)

1. **Native `Collection` vs high-level `Collection`.** ADR-R2 renames C++ `Vault`→`Collection`,
   but the facade is already `elips.Collection`. **Default:** keep both, module-scoped —
   high-level `elips.Collection` (collection/collection.py) and native `elips.native.Collection`
   (the pybind handle). They never share a namespace. Proceed.
2. **`get` name clash at native layer.** Chroma `get` covers both by-id (`fetch`) and
   by-filter (`scan`). **Default:** native keeps distinct `get`(by id)+`scan`(by filter);
   only the facade unifies them into one `get(ids=, where=, limit=, offset=)`. Proceed.
3. **`document` (attachment) vs `documents` (text list).** ELIPS `document=`/`.document` is a
   `DocumentAttachment`; Chroma `documents=` is raw text. **Default:** keep `document=` for the
   attachment, add `documents=`/`document=` text kwargs that build attachments; disambiguate by
   type. Flag for review — mild foot-gun.
4. **`list_collections` return type.** Chroma returns Collection objects; ELIPS returns names.
   **Default:** `list_collections()`→`list[str]` for one minor (shim of `collection_names`),
   add `list_collection_objects()` if full Chroma parity is wanted later. Proceed.
5. **`n_results` vs `limit` alias.** Task says alias `k`; `01_faiss_study.md:83-85` found `limit`
   beats `n_results` 3-of-4 vendors. **Default:** `query` uses `n_results` (alias `k` per task);
   `get`/`list_collections` use `limit`/`offset`. Proceed.
