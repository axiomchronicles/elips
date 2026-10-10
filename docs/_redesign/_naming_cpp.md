# Phase 5A — C++ Naming Map (old → new, DEEP rename per ADR-R2)

Planning artifact. **No code is changed here.** This enumerates every public C++ symbol in
`include/elips/**.hpp` + its `src/` definition and proposes a new name, per ADR-R2
(`04_architecture.md:22`) — the deep rename of the mature core into FAISS-recognizable
`Index*` names + Chroma-recognizable `Collection`/`Client` names.

Sources: `00_inventory.md`, `_phase1_cpp.md`, `_phase1_gpu.md`, `_faiss_3A.md`, `01_faiss_study.md`,
`_gap_4A.md`. Line numbers are the declaration line (cat -n).

Legend for **Notes/collision**: `keep` = stay unchanged; `proposed` = my proposal where the mission
table was silent; `COLLISION` = C++ keyword/reserved clash with a safe identifier; `ADD` = new
symbol the FAISS/Chroma contract wants that ELIPS lacks today (listed for completeness, not a rename).

**Two reality corrections to the mission brief, confirmed in code:**
1. There is **no C++ `class Database`** (grep: none). The engine type is **`ElipsInstance`**
   (`elips.hpp:290`); `src/Database.cpp` is only the impl filename. So "`Database`→Client-tier"
   maps in C++ to: rename `ElipsInstance`→`Database` (engine type, now matches the filename and the
   Python-native `Database`) and add thin `Client`/`PersistentClient` wrappers over `open()`.
2. The container is **`Vault`**; insert verbs are **`place*`**, query verbs **`seek*`**, delete
   **`erase`**, point-get **`fetch`**, iterate **`scan`** (no `ingest`/`probe`/`meta`/`top=` verbs).

---

## facade — `include/elips/elips.hpp` (+ `src/Database.cpp`), namespace `elips`

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| `VaultInfo` (elips.hpp:87) | `CollectionInfo` | struct | proposed (follows Vault→Collection) |
| `VaultInfo::count` (:88) | `count` | field | keep; FAISS `ntotal` alias ADD |
| `VaultInfo::dimension` (:89) | `dimension` | field | keep; FAISS `d` alias ADD |
| `VaultInfo::metric` (:90) | `metric` | field | keep |
| `VaultInfo::codec` (:94) | `codec` | field | keep |
| `VaultInfo::code_bytes` (:95) | `code_bytes` | field | keep |
| `QueryStrategy` (:98) | `QueryStrategy` | enum class | keep (internal plan labels) |
| `QueryStrategy::ann_index` (:99) | `ann_index` | enumerator | keep |
| `QueryStrategy::exact_candidates` (:100) | `exact_candidates` | enumerator | keep |
| `QueryStrategy::full_scan` (:101) | `full_scan` | enumerator | keep |
| `QueryStrategy::text_probe` (:102) | `text_search` | enumerator | proposed (align w/ seek_text→search_text) |
| `QueryStrategy::hybrid_fusion` (:103) | `hybrid_fusion` | enumerator | keep |
| `QueryPlan` (:106) | `QueryPlan` | struct | keep |
| `QueryPlan::strategy/candidate_count/metadata_accelerated/gpu_index/index_type/codec` (:107-112) | (same) | fields | keep |
| `Vault` (:122) | `Collection` | class | mission core rename |
| `Vault::Vault(...)` (:124) | `Collection(...)` | ctor | rename w/ class |
| `Vault::place` (:130) | `add` | method | mission: place→add |
| `Vault::place_document` (:135) | `add_document` | method | mission |
| `Vault::place_many` (:140) | `add_many` | method | mission |
| `Vault::place_record` (:147) | `add_record` | method | proposed (load-path; follows place→add) |
| `Vault::seek` (:149) | `search` | method | mission: seek→search; map `top`→`n_results` (alias `k`) |
| `Vault::seek_text` (:152) | `search_text` | method | mission |
| `Vault::seek_hybrid` (:155) | `hybrid_search` | method | mission; param `lexical_weight` keep |
| `Vault::explain_seek` (:160) | `explain_query` | method | mission: explain_seek→explain_query |
| `Vault::scan` (:165) | `scan` | method | keep (filtered range iter); Python `get(where=)` maps here |
| `Vault::fetch` (:169) | `get` | method | proposed (Chroma point-get; FAISS `reconstruct(id)` equiv) |
| `Vault::erase` (:170) | `remove` | method | **COLLISION**: Chroma/Py name `delete` is a C++ keyword → C++ `remove` (FAISS `remove_ids`); Python exposes `delete` |
| `Vault::info` (:172) | `info` | method | keep (returns CollectionInfo) |
| `Vault::name` (:173) | `name` | method | keep |
| `Vault::records` (:178) | `records` | method | keep |
| `Vault::set_wal` (:180) | `set_wal` | method | keep |
| `Vault::set_read_only` (:181) | `set_read_only` | method | keep |
| `Vault::read_only` (:182) | `read_only` | method | keep |
| `Vault::rebuild_index` (:183) | `rebuild_index` | method | keep |
| `Vault::vacuum` (:188) | `vacuum` | method | keep |
| `Vault::pending_removals` (:190) | `pending_removals` | method | keep |
| `Vault::quantize` (:208) | `quantize` | method | keep (Python exposes `compress`) |
| `Vault::quantized` (:209) | `quantized` | method | keep |
| `Vault::codec` (:210) | `codec` | method | keep |
| `Vault::config` (:214) | `config` | method | keep |
| `Vault::install_quantizer` (:217) | `install_quantizer` | method | keep |
| `Vault::seal` (:222) | `seal` | method | keep |
| `Vault::sealed` (:223) | `sealed` | method | keep |
| `Vault::restore_for_undo` (:230) | `restore_for_undo` | method | keep |
| _(none)_ | `ntotal` / `count` | method | ADD (FAISS `ntotal` over `info().count`) |

### facade — engine, instance opener, transactions

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| `ElipsInstance` (elips.hpp:290) | `Database` | class | proposed; engine type (matches `src/Database.cpp` + Python-native `Database`). `Client`/`PersistentClient` wrap it (see ADD rows) |
| `ElipsInstance::ElipsInstance(...)` (:292) | `Database(...)` | ctor | rename w/ class |
| `ElipsInstance::~ElipsInstance` (:294) | `~Database` | dtor | rename w/ class |
| `ElipsInstance::vault` (:301) | `collection` | method | Chroma: `get_collection`/`get_or_create_collection` wrap this |
| `ElipsInstance::list_vaults` (:302) | `list_collections` | method | Chroma `list_collections` |
| `ElipsInstance::begin_transaction` (:304) | `begin_transaction` | method | keep |
| `ElipsInstance::query` (:306) | `query` | method | keep (EQL entrypoint; impl `QueryExecutor.cpp:151`) |
| `ElipsInstance::checkpoint` (:310) | `checkpoint` | method | keep (FAISS `write_index` equiv) |
| `ElipsInstance::compact` (:311) | `compact` | method | keep |
| `ElipsInstance::quantize` (:316) | `quantize` | method | keep |
| `ElipsInstance::quantize_all` (:317) | `quantize_all` | method | keep |
| `ElipsInstance::vacuum` (:320) | `vacuum` | method | keep |
| `ElipsInstance::close` (:321) | `close` | method | keep |
| `ElipsInstance::abandon` (:322) | `abandon` | method | keep |
| `ElipsInstance::wal` (:323) | `wal` | method | keep |
| `ElipsInstance::config` (:325) | `config` | method | keep |
| `ElipsInstance::path` (:327) | `path` | method | keep |
| `ElipsInstance::persistent` (:328) | `persistent` | method | keep |
| `ElipsInstance::closed` (:329) | `closed` | method | keep |
| `ElipsInstance::gpu_info` (:332) | `gpu_info` | method | keep (GPU-gated) |
| `ElipsInstance::gpu_stats` (:333) | `gpu_stats` | method | keep (GPU-gated) |
| `ElipsInstance::set_gpu_available/set_gpu_info/set_gpu_backend` (:334-336) | (same) | methods | keep (GPU-gated) |
| `ElipsInstance::adopt_vault` (:341) | `adopt_collection` | method | proposed (follows Vault→Collection) |
| `ElipsInstance::attach_wal` (:342) | `attach_wal` | method | keep |
| `elips::open` (elips.hpp:370 / Config.hpp:23 / Database.cpp:1775) | `open` | free func | keep (engine opener); `connect` alias + Client-tier wrap it |
| `TransactionVault` (:375) | `TransactionCollection` | class | proposed (follows Vault→Collection) |
| `TransactionVault::place` (:377) | `add` | method | mission: place→add |
| `TransactionVault::erase` (:379) | `remove` | method | COLLISION (erase→remove; Py `delete`) |
| `Transaction` (:389) | `Transaction` | class | keep |
| `Transaction::Transaction` (:391) | (same) | ctor | keep |
| `Transaction::~Transaction` (:392) | (same) | dtor | keep |
| `Transaction::vault` (:399) | `collection` | method | proposed (returns TransactionCollection) |
| `Transaction::commit` (:405) | `commit` | method | keep |
| `Transaction::rollback` (:408) | `rollback` | method | keep |
| `SharedMutex` (:40, MinGW only) | `SharedMutex` | class | keep (platform shim) |
| `WAL` fwd-decl (:83), `Transaction` fwd-decl (:84) | — | fwd-decl | track w/ their defs |
| _(none)_ | `Client` | class | ADD (Chroma ephemeral client; wraps `open(":memory:")`) |
| _(none)_ | `PersistentClient` | class | ADD (Chroma `PersistentClient(path=)`; wraps `open(path)`) |
| _(none)_ | `create_collection` / `get_collection` / `get_or_create_collection` / `delete_collection` | methods | ADD (Chroma Client surface over `collection()`/`list_collections`) |

---

## config — `include/elips/Config.hpp`, namespace `elips`

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| `Metric` (:27) | `Metric` | enum class | keep |
| `Metric::cosine` (:27) | `cosine` | enumerator | keep (ELIPS: 1−dot; FAISS = IP-on-normalized) |
| `Metric::euclidean` (:27) | `euclidean` | enumerator | keep; FAISS `METRIC_L2` alias (NB ELIPS returns true sqrt-L2, not squared) |
| `Metric::dot_product` (:27) | `dot_product` | enumerator | keep; FAISS `METRIC_INNER_PRODUCT`/`inner_product` alias |
| `IndexType` (:31) | `IndexType` | enum class | keep |
| `IndexType::graph` (:31) | `hnsw` | enumerator | proposed (align w/ HierarchicalGraphIndex→IndexHNSW) |
| `IndexType::exact` (:31) | `flat` | enumerator | proposed (align w/ ExactIndex→IndexFlat) |
| _(none)_ | `ivf_flat` / `ivf_pq` | enumerator | ADD (CPU IVF tier, Phase 6) |
| `Durability` (:38) | `Durability` | enum class | keep |
| `Durability::paranoid/standard/relaxed/ephemeral` (:38) | (same) | enumerators | keep |
| `AccessMode` (:40) | `AccessMode` | enum class | keep |
| `AccessMode::read_write/read_only` (:40) | (same) | enumerators | keep |
| `to_string(Metric)` (:42 / Metrics.cpp:121) | `to_string` | free func | keep |
| `metric_from_string` (:43 / Metrics.cpp:133) | `metric_from_string` | free func | keep |
| `GraphParams` (:46) | `HnswParams` | struct | proposed (align w/ IndexHNSW; FAISS `HNSW`) |
| `GraphParams::max_connections` (:47) | `max_connections` | field | keep; FAISS `M` alias |
| `GraphParams::ef_construction` (:48) | `ef_construction` | field | keep; FAISS `efConstruction` alias |
| `GraphParams::ef_search` (:49) | `ef_search` | field | keep; FAISS `efSearch` alias |
| `GraphParams::compaction_ratio` (:52) | `compaction_ratio` | field | keep (ELIPS-specific) |
| `Config` (:56) | `Config` | class | keep (FAISS/Chroma "configuration") |
| `Config::dimension` (set :58 / get :121) | `dimension` | method | keep; FAISS `d` alias |
| `Config::metric` (:62 / :122) | `metric` | method | keep |
| `Config::index` (:66 / :123) | `index` | method | keep |
| `Config::graph_params` (:70 / :124) | `hnsw_params` | method | proposed (follows GraphParams→HnswParams) |
| `Config::quantization` (:78 / :125) | `quantization` | method | keep |
| `Config::has_quantization` (:128) | `has_quantization` | method | keep |
| `Config::durability` (:82 / :131) | `durability` | method | keep |
| `Config::access_mode` (:86 / :132) | `access_mode` | method | keep |
| `Config::segmented_storage` (:90 / :133) | `segmented_storage` | method | keep |
| `Config::metadata_acceleration` (:94 / :134) | `metadata_acceleration` | method | keep |
| `Config::text_embedder` (:98 / :135) | `text_embedder` | method | keep |
| `Config::has_text_embedder` (:138) | `has_text_embedder` | method | keep |
| `Config::local_text_embedder` (:104) | `local_text_embedder` | method | keep |
| `Config::auto_text_embedder` (:110 / :141) | `auto_text_embedder` | method | keep |
| `Config::has_pending_local_text_embedder` (:144) | (same) | method | keep |
| `Config::local_text_embedder_options` (:147) | (same) | method | keep |
| `Config::text_embedder_info` (:151) | `text_embedder_info` | method | keep |
| `Config::gpu` (:115 / :165) | `gpu` | method | keep (GPU-gated) |
| `Config::has_gpu` (:166) | `has_gpu` | method | keep (GPU-gated) |
| `Config::attach_runtime_text_embedder` (:171) | (same) | method | keep |
| `Config::remember_expected_text_embedder` (:178) | (same) | method | keep |
| `Config::attach_quantizer` (:187) | `attach_quantizer` | method | keep |
| `Config::quantizer` (:190) | `quantizer` | method | keep |
| `Config::has_trained_quantizer` (:193) | `has_trained_quantizer` | method | keep |

---

## index_engine — namespace `elips`

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| **FILE** `index_engine/IndexPort.hpp` | `index_engine/Index.hpp` | header file | rename |
| `IndexPort` (IndexPort.hpp:16) | `Index` | class | mission: IndexPort→Index (FAISS base) |
| `IndexPort::Hit` (:18) | `Hit` | alias | keep (ELIPS `pair<RecordID,float>`; FAISS uses raw arrays) |
| `IndexPort::IndexPort` (:20) | `Index` | ctor | rename w/ class |
| `IndexPort::~IndexPort` (:21) | `~Index` | dtor | rename w/ class |
| `IndexPort::insert` (:28) | `add_with_ids` | method | proposed (ELIPS rows always carry an id; FAISS single-id add) |
| `IndexPort::remove` (:29) | `remove` | method | keep (FAISS `remove_ids`); NB Py `delete` collision handled at Collection tier |
| `IndexPort::search` (:32) | `search` | method | keep (already FAISS-named) |
| `IndexPort::size` (:35) | `ntotal` | method | proposed (FAISS `ntotal`) |
| `IndexPort::type_name` (:36) | `type_name` | method | keep |
| `IndexPort::vacuum` (:40) | `vacuum` | method | keep |
| `IndexPort::pending_removals` (:43) | `pending_removals` | method | keep |
| _(none)_ | `add` / `train` / `reset` / `reconstruct` / `range_search` / `is_trained` / `d` / `metric_type` | methods | ADD (FAISS `Index` contract; default no-op/throw per `_faiss_3A.md §1.2`) |
| **FILE** `index_engine/ExactIndex.hpp` + `src/ExactIndex.cpp` | `index_engine/IndexFlat.hpp` + `src/IndexFlat.cpp` | files | rename |
| `ExactIndex` (ExactIndex.hpp:26) | `IndexFlat` | class | mission: ExactIndex→IndexFlat |
| `ExactIndex::ExactIndex(...)` (:28) | `IndexFlat(...)` | ctor | rename w/ class |
| `ExactIndex::insert/remove/search/size/type_name` (:36-42) | (inherited new names) | methods | override renames follow `Index` |
| `ExactIndex::export_snapshot` (:46) | `export_snapshot` | method | keep |
| `ExactIndex::import_snapshot` (:48) | `import_snapshot` | method | keep |
| `"exact"` / `"exact_quantized"` (ExactIndex.hpp:43) | `"flat"` / `"flat_quantized"` | string literal | type_name values — rename w/ class |
| _(none)_ | `IndexFlatL2` / `IndexFlatIP` | class | ADD (FAISS metric-fixing subclasses, `_faiss_3A §2.1`) |
| **FILE** `index_engine/HierarchicalGraphIndex.hpp` + `src/HierarchicalGraphIndex.cpp` | `index_engine/IndexHNSW.hpp` + `src/IndexHNSW.cpp` | files | rename |
| `HierarchicalGraphIndex` (HGI.hpp:39) | `IndexHNSW` | class | mission: HierarchicalGraphIndex→IndexHNSW |
| `HierarchicalGraphIndex::HierarchicalGraphIndex(...)` (:41) | `IndexHNSW(...)` | ctor | rename w/ class |
| `HierarchicalGraphIndex::insert/remove/search/size/type_name/vacuum/pending_removals` (:45-58) | (inherited new names) | methods | override renames follow `Index` |
| `HierarchicalGraphIndex::export_snapshot/import_snapshot` (:62-64) | (same) | methods | keep |
| `"graph"` / `"graph_quantized"` (HGI.hpp:54) | `"hnsw"` / `"hnsw_quantized"` | string literal | type_name values — rename w/ class |
| `HierarchicalGraphIndex::Probe` (:79), `NodeId` (:68), `Scored` (:69) | — | private nested | not public — out of scope (rename internally w/ class) |
| **FILE** `index_engine/IndexFactory.hpp` + `src/IndexFactory.cpp` | `index_engine/index_factory.hpp` + `src/index_factory.cpp` | files | rename (FAISS `index_factory.h`) |
| `make_index` (IndexFactory.hpp:20) | `index_factory` | free func | mission: IndexFactory→index_factory. Keep `Config` overload; ADD string-desc overload (Phase 6) |
| `IndexSnapshotKind` (IndexSnapshot.hpp:14) | `IndexSnapshotKind` | enum class | keep |
| `IndexSnapshotKind::unknown` (:15) | `unknown` | enumerator | keep |
| `IndexSnapshotKind::exact` (:16) | `flat` | enumerator | proposed (align ExactIndex→IndexFlat) |
| `IndexSnapshotKind::graph` (:17) | `hnsw` | enumerator | proposed (align HGI→IndexHNSW) |
| `IndexSnapshotKind::gpu_brute_force` (:18) | `gpu_flat` | enumerator | proposed (align GpuBruteForceIndex→GpuIndexFlat) |
| `IndexSnapshotKind::gpu_ivf_flat` (:19) | `gpu_ivf_flat` | enumerator | keep |
| `IndexSnapshotKind::gpu_ivf_pq` (:20) | `gpu_ivf_pq` | enumerator | keep |
| `IndexSnapshotKind::gpu_graph` (:21) | `gpu_hnsw` | enumerator | proposed (align GpuGraphIndex→GpuIndexHNSW) |
| `IndexSnapshotKind::gpu_hybrid` (:22) | `gpu_hybrid` | enumerator | keep |
| `IndexSnapshotKind::gpu_distributed` (:23) | `gpu_distributed` | enumerator | keep |
| `IvfSnapshot` (:26) | `IvfSnapshot` | struct | keep |
| `IvfSnapshot::n_lists/n_probe/centroids/assignments` (:27-30) | (same) | fields | keep |
| `PqSnapshot` (:36) | `PqSnapshot` | struct | keep |
| `PqSnapshot::codec/pq_dim/pq_bits/codebook/rotation/codes` (:37-42) | (same) | fields | keep |
| `IndexSnapshot` (:48) | `IndexSnapshot` | struct | keep |
| `IndexSnapshot::kind/metric/dimension/ids/vectors/ivf/pq` (:49-55) | (same) | fields | keep |
| `IndexTransferPort` (IndexTransferPort.hpp:11) | `IndexTransferPort` | class | keep (ELIPS CPU↔GPU seam; no FAISS equiv — FAISS uses clone) |
| `IndexTransferPort::~IndexTransferPort` (:13) | (same) | dtor | keep |
| `IndexTransferPort::export_snapshot` (:15) | `export_snapshot` | method | keep |
| `IndexTransferPort::import_snapshot` (:18) | `import_snapshot` | method | keep |

---

## metadata — namespace `elips`

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| `Comparator` (Filter.hpp:13) | `Comparator` | enum class | keep |
| `Comparator::eq/ne/lt/le/gt/ge` (:13) | (same) | enumerators | keep (Chroma `$eq/$ne/$lt/$lte/$gt/$gte` map at Python) |
| `Filter` (Filter.hpp:23) | `Filter` | class | keep (ELIPS predicate; Chroma `where`) |
| `Filter::ExactConstraint` (:25) | `ExactConstraint` | alias | keep |
| `Filter::Filter` (:27) | (same) | ctor | keep |
| `Filter::field/equals/not_equals/lt/le/gt/ge/one_of/contains` (:30-38) | (same) | methods | keep (fluent builder) |
| `Filter::compare/in_set/has_substring` (:41-45) | (same) | static methods | keep |
| `Filter::and_/or_/not_` (:49-51) | (same) | methods | keep |
| `Filter::matches` (:53) | `matches` | method | keep |
| `Filter::exact_constraints` (:54) | `exact_constraints` | method | keep |
| `Filter::matches_all` (:56) | `matches_all` | method | keep |
| `Filter::Kind` (:59), `Node` (:60) | — | private nested | not public — out of scope |
| **FILE** `metadata/MetadataIndex.hpp` + `src/MetadataIndex.cpp` | `metadata/MetadataFilterIndex.hpp` + `src/MetadataFilterIndex.cpp` | files | rename |
| `MetadataIndex` (MetadataIndex.hpp:16) | `MetadataFilterIndex` | class | mission: MetadataIndex→MetadataFilterIndex |
| `MetadataIndex::insert` (:18) | `insert` | method | keep |
| `MetadataIndex::remove` (:19) | `remove` | method | keep |
| `MetadataIndex::exact_candidates` (:20) | `exact_candidates` | method | keep |

---

## domain — namespace `elips` (header-only value types)

Entire layer is **keep** — ELIPS domain primitives have no FAISS/Chroma rename (FAISS uses `idx_t`
int64 ids + raw `float*`; ELIPS keeps `RecordID`/`Vector`/`Record`). Noted for completeness.

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| `ElipsError` (Errors.hpp:9) | `ElipsError` | class | keep (base; inherits `std::runtime_error`) |
| `DimensionMismatch` (:15), `InvalidVector` (:21), `ConfigError` (:27), `NotFound` (:33), `StorageError` (:39) | (same) | classes | keep |
| `MetaValue` (Record.hpp:17) | `MetaValue` | alias | keep (`variant<int64,double,bool,string>`) |
| `Payload` (:20) | `Payload` | alias | keep (`map<string,MetaValue>`); Python arg `meta`→`metadatas` maps here |
| `DocumentAttachment` (:22) + fields `text/uri/mime_type` (:23-25) | (same) | struct | keep |
| `ChunkInfo` (:28) + fields `document_key/ordinal/char_start/char_end` (:29-32) | (same) | struct | keep |
| `EmbeddingLineage` (:35) + fields `provider/model/revision/attributes` (:36-39) | (same) | struct | keep |
| `Record` (:50) + fields `id/vector/payload/document/chunk/lineage/codes/codec` (:51-58) | (same) | struct | keep |
| `Record::approximate` (:63) | `approximate` | method | keep |
| `RecordID` (RecordID.hpp:13) | `RecordID` | class | keep (UUIDv7; FAISS `idx_t` int64 equivalent) |
| `RecordID::Bytes` (:15) | `Bytes` | alias | keep |
| `RecordID` ctors (:17-18), `bytes` (:20), `to_string` (:21), `generate` (:23), `from_string` (:24), `operator==` (:26), `operator<=>` (:28) | (same) | methods | keep |
| `std::hash<elips::RecordID>` (:37) + `operator()` (:38) | (same) | struct spec | keep |
| `SearchResult` (SearchResult.hpp:15) | `SearchResult` | struct | keep (also the Chroma result row) |
| `SearchResult` fields `id/distance/data/document/chunk/lineage/codec` (:16-22) | (same) | fields | keep; `data`→Python `metadatas` |
| `SearchResult::approximate` (:24) | `approximate` | method | keep |
| `Vector` (Vector.hpp:14) | `Vector` | class | keep |
| `Vector` ctors (:16-17), `values` (:19), `dimension` (:20), `empty` (:23), `magnitude` (:25), `normalized` (:34) | (same) | methods | keep |
| `std::expected`/`std::unexpected` polyfill (Expected.hpp:17-88) | (same) | class templates | keep (dead on C++23; `_phase1_cpp` flags as removable — a deletion, not a rename) |

---

## vector_engine — `include/elips/vector_engine/Metrics.hpp`, namespace `elips`

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| `distance(Metric, span, span)` (Metrics.hpp:16 / Metrics.cpp:104) | `distance` | free func | keep |
| `requires_normalization(Metric)` (:20 / Metrics.cpp:117) | `requires_normalization` | free func | keep |

---

## kernel — `include/elips/kernel/LockManager.hpp`, namespace `elips`

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| `LockConflict` (LockManager.hpp:11) | `LockConflict` | class | keep |
| `LockMode` (:16) + `exclusive`/`shared` | `LockMode` | enum class | keep |
| `LockManager` (:21) + ctor (:23), dtor (:25), move-ctor (:29) | `LockManager` | class | keep |

---

## quant_engine — namespace `elips::quant` (`detail` → `elips::quant::detail`)

Entire layer is **keep** — ELIPS already uses FAISS-standard names (`ProductQuantizer`,
`ScalarQuantizer`) and clearer param names; `_gap_4A.md` / `3B` say keep and only record FAISS
aliases (`M=pq_dim`, `nbits=pq_bits`, `dsub=sub_dim`). `refine_fraction` stays (the one
to-implement knob).

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| `Quantizer` fwd (QuantParams.hpp:12) | `Quantizer` | class fwd | keep |
| `QuantizerPtr` (QuantParams.hpp:17 / Quantizer.hpp:79) | `QuantizerPtr` | alias | keep |
| `CodecId` (QuantParams.hpp:21) + `none=0/pq=1/opq=2/sq8=3` (:22-25) | `CodecId` | enum class | keep (persisted on disk — never renumber) |
| `to_string(CodecId)` (:28) / `codec_from_string` (:29) | (same) | free funcs | keep |
| `QuantParams` (:33) + fields `codec/pq_dim/pq_bits/train_iters/opq_iters/refine_fraction` (:34-47) | (same) | struct | keep; FAISS aliases `M=pq_dim`, `nbits=pq_bits` |
| `Quantizer` (Quantizer.hpp:30) + virtuals `codec/code_bytes/dimension/metric/encode/decode/make_lut/lut_distance/serialize` + convenience `encode/decode` | (same) | class | keep |
| `train` (:87), `load` (:93), `train_sample_cap` (:98), `choose_pq_dim` (:102), `validate` (:108), `code_bytes_for` (:112) | (same) | free funcs/const | keep |
| `ProductQuantizer` (ProductQuantizer.hpp:28) + `train`/`deserialize`/overrides/accessors `codebook/rotation/pq_dim/pq_bits` | `ProductQuantizer` | class | keep (same name as FAISS) |
| `ScalarQuantizer` (ScalarQuantizer.hpp:26) + `train`/`deserialize`/overrides/`mins`/`scales` | `ScalarQuantizer` | class | keep (same name as FAISS) |
| `kmeans` (KMeans.hpp:22), `train_pq_codebook` (:30), `encode_pq` (:36), `nearest_centroid` (:41), `solve_procrustes` (:53), `identity_matrix` (:59), `apply_matrix` (:62), `apply_matrix_transposed` (:66) | (same) | free funcs | keep (`elips::quant::detail`) |

---

## query_engine — namespace `elips::eql`

C++ AST **type** names mostly keep; EQL **keyword** renames (`seek`→`search`, `place`→`add`,
`erase`→`delete`/`remove`, etc.) are a separate surface tracked in `03_naming_map.md`'s EQL section.

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| `VectorRef` (AST.hpp:15) + fields `literal/binding` | `VectorRef` | struct | keep |
| `SearchStatement` (:20) + fields `vault/query/top/threshold/where/rank_by/projection` | `SearchStatement` | struct | keep; field `vault`→`collection` (proposed), `top`→`n_results` (proposed) |
| `FetchStatement` (:30) + `vault/id` | `FetchStatement` | struct | keep (EQL `fetch`); `vault`→`collection` field proposed |
| `ScanStatement` (:35) + `vault/where/offset/limit` | `ScanStatement` | struct | keep; `vault`→`collection` field proposed |
| `InsertStatement` (:42) + `vault/vector/data` | `AddStatement` | struct | proposed (EQL place→add); `vault`→`collection` field proposed |
| `DeleteStatement` (:48) + `vault/id` | `DeleteStatement` | struct | keep (parse node for erase/delete); `vault`→`collection` field proposed |
| `Statement` (:53) | `Statement` | alias | keep (variant; update member list w/ AddStatement) |
| `TokenKind` (EQLLexer.hpp:10) + `word/number/string/punct/end` | `TokenKind` | enum class | keep |
| `Token` (:12) + fields `kind/text/number/is_integer` | `Token` | struct | keep |
| `tokenize` (:20) | `tokenize` | free func | keep |
| `ParseError` (EQLParser.hpp:12) | `ParseError` | class | keep |
| `parse(string_view)` (:18) | `parse` | free func | keep |
| `execute(Statement, ElipsInstance&, bindings)` (QueryExecutor.hpp:20) | `execute` | free func | keep; param type `ElipsInstance&`→`Database&` (follows facade rename) |

---

## storage — namespaces `elips::detail` (FileSync, Serialization) and `elips` (WAL)

Entire layer is **keep** — WAL/serialization/fsync are ELIPS internals with no FAISS/Chroma
counterpart (ELIPS is ahead here; `_gap_4A.md`). Listed for completeness.

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| `SyncProbe` (FileSync.hpp:28) + fields, `sync_probe` (:35), `sync_file_data` (:42), `sync_file_path` (:79), `sync_directory` (:110), `durable_rename` (:125) | (same) | struct/free funcs | keep (`elips::detail`) |
| `put`/`get` templates (Serialization.hpp:22/27), `stream_remaining` (:35), `max_field_bytes` (:54), `check_length` (:60), `put_string`/`get_string` (:73/78), `put_payload`/`get_payload` (:89/107), `put_document_attachment`/`get_document_attachment` (:141/152), `put_chunk_info`/`get_chunk_info` (:163/174), `put_embedding_lineage`/`get_embedding_lineage` (:186/198), `crc32c` (:211) | (same) | free funcs/const | keep (`elips::detail`) |
| `WAL` (WAL.hpp:21) | `WAL` | class | keep |
| `WAL::Op` (:23) + `insert=1/erase=3/insert_ex=4/txn_begin=5/txn_commit=6/insert_q=7` (:24-33) | `Op` | enum class | keep (on-disk op tags — never renumber; `Op::erase` stays despite erase→remove, it's a wire tag) |
| `WAL::Entry` (:36) + fields `op/vault/id/vector/payload/document/chunk/lineage/codes/codec` (:37-46) | `Entry` | struct | keep; `vault` field→`collection` (proposed) |
| `WAL` ctor (:49), dtor (:50), `append_insert` (:57), `append_insert_quantized` (:62), `append_erase` (:69), `append_txn_begin` (:74), `append_txn_commit` (:75), `sync` (:79), `reset` (:82), `replay` (:85) | (same) | methods | keep (internal verbs; not the public CRUD surface) |

---

## text_engine — namespace `elips`

Entire layer is **keep** — ELIPS-specific embedder abstraction, no FAISS/Chroma rename.

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| `TextEmbedderKind` (TextEmbedderPort.hpp:14) + `external`/`local_builtin` | (same) | enum class | keep |
| `TextEmbedderInfo` (:16) + 11 fields (`kind/provider/model/revision/backend/dimension/fingerprint/storage_path/rehydratable/loaded/auto_attached`, :17-27) | (same) | struct | keep |
| `TextEmbedderPort` (:30) + `embed` (:34), `embed_batch` (:36), `provider_name` (:46), `model_name` (:47), `revision_name` (:48), `backend_name` (:51), `kind` (:54), `output_dimension` (:57), `rehydratable` (:60), `loaded` (:61), `unload` (:62), `set_output_dimension` (:63), `fingerprint` (:64), `storage_path` (:65), `info` (:67) | (same) | class | keep |
| `TextEmbedderPtr` (:84) | (same) | alias | keep |
| `LocalTextEmbedderOptions` (LocalTextEmbedder.hpp:12) + fields `model/revision/storage_path/dimension` (:13-16) | (same) | struct | keep |
| `LocalTextEmbedder` (:19) + overrides + `options` (:37) | (same) | class | keep |
| `describe_local_text_embedder` (:54), `make_local_text_embedder` (:59), `make_default_local_text_embedder` (:62) | (same) | free funcs | keep |

---

## gpu_engine — namespace `elips::gpu`

Mission renames: `GpuPort`→`GpuBackend`, `GpuSelector`→`BackendSelector`,
`DynamicBatcher`→`QueryBatcher`, `GpuIndexPort`→`GpuIndex`. Concrete index classes are proposed to
mirror FAISS `GpuIndex*` names. Index method renames (`insert`→`add_with_ids`, `size`→`ntotal`, …)
follow the `Index` base (above) and are not repeated per class. The sibling port interfaces
`GpuKernelPort`/`GpuStreamPort` have **zero implementers** (`_phase1_gpu.md:12`) — kept here, but
flagged as deletion candidates for Phase 2/6, not renames.

### config, device info, buffers

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| `GpuPolicy` (GpuConfig.hpp:11) + `Auto/PreferGpu/RequireGpu/CpuOnly/Specific` (:12-16) | (same) | enum class | keep |
| `IndexBuildMode` (:19) + `GpuBuild_CpuServe/GpuBuild_GpuServe/Hybrid` (:20-22) | (same) | enum class | keep |
| `GpuIndexAlgorithm` (:25) + `Auto/CagraGraph/IvfFlat/IvfPq/BruteForce` (:26-30) | (same) | enum class | keep; `BruteForce`→`Flat` proposed (align IndexFlat) |
| `GpuPrecision` (:33) + `FP32/FP16/Int8/Auto` | (same) | enum class | keep |
| `GraphIndexBuildParams` (:35) + nested `BuildAlgo` (:38, `IvfPq/NnDescent/IterativeSearch`) + fields (:36-40) | (same) | struct/enum | keep |
| `IvfPqBuildParams` (:43) + fields `n_lists/pq_dim/pq_bits/add_data_on_build/kmeans_n_iters/kmeans_trainset_fraction` (:44-49) | (same) | struct | keep |
| `GpuIndexBuildParams` (:52) + field `params` (:53) | (same) | struct | keep |
| `GpuConfig` (:56) + all fields (:57-84) | `GpuConfig` | struct | keep (FAISS `GpuIndexConfig` is per-index; ELIPS keeps one engine config) |
| `GpuDeviceInfo` (GpuDeviceInfo.hpp:9) + all fields (:10-37) | (same) | struct | keep |
| `GpuMetricsSnapshot` (GpuMetricsSnapshot.hpp:9) + all fields (:10-25) | (same) | struct | keep |
| `GpuBuffer` (GpuBuffer.hpp:9) + ctors/dtor/move + `device_ptr/bytes/backend_handle/operator bool` | (same) | class | keep |
| `PinnedBuffer` (PinnedBuffer.hpp:9) + ctors/dtor/move + `data/bytes/operator bool` | (same) | class | keep |
| `UnifiedBuffer` (UnifiedBuffer.hpp:10) + ctors + `host_ptr/bytes/gpu_view` | (same) | class | keep |

### ports

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| **FILE** `gpu_engine/GpuPort.hpp` | `gpu_engine/GpuBackend.hpp` | header file | rename |
| `GpuError` (GpuPort.hpp:15) + `DeviceNotFound/InsufficientMemory/KernelLaunchFailed/TransferFailed/IndexBuildFailed/UnsupportedMetric/InitializationFailed/BackendUnavailable` (:16-23) | (same) | enum class | keep |
| `to_string(GpuError)` (:26) | `to_string` | free func | keep |
| `GpuPort` (:40) | `GpuBackend` | class | mission: GpuPort→GpuBackend (abstract backend iface) |
| `GpuPort::~GpuPort` (:42) + `initialize` (:44), `shutdown` (:47), `device_info` (:49), `is_available` (:50), `allocate_device` (:52), `free_device` (:57), `upload` (:59), `download` (:62), `compute_distances_batch` (:65), `top_k` (:75), `synchronize` (:84), `is_idle` (:85) | (same) | methods | keep (rename dtor w/ class) |
| `GpuKernelPort` (GpuKernelPort.hpp:12) + `cosine_fp32` (:16), `euclidean_fp32` (:23), `dot_product_fp32` (:30) | `GpuKernelPort` | class | keep — but **zero implementers**, deletion candidate (Phase 2/6) |
| `GpuMemoryPort` (GpuMemoryPort.hpp:12) + `allocate` (:16), `deallocate` (:19), `allocate_pinned` (:21), `deallocate_pinned` (:24), `bytes_used` (:26), `bytes_available` (:27), `peak_bytes_used` (:28) | `GpuMemoryPort` | class | keep |
| `GpuStreamPort` (GpuStreamPort.hpp:9) + `synchronize` (:13), `is_complete` (:14), `wait_for_completion` (:15) | `GpuStreamPort` | class | keep — **zero implementers**, deletion candidate |
| **FILE** `gpu_engine/GpuIndexPort.hpp` | `gpu_engine/GpuIndex.hpp` | header file | rename |
| `GpuIndexPort` (GpuIndexPort.hpp:20, `: public IndexPort, public IndexTransferPort`) | `GpuIndex` | class | mission: GpuIndexPort→GpuIndex (base → `Index`,`IndexTransferPort`) |
| `GpuIndexPort::~GpuIndexPort` (:22) + `build_from_batch` (:24), `search_batch` (:30), `export_to_cpu_index` (:36), `import_from_cpu_index` (:39), `device_bytes_used` (:42), `backend_name` (:44) | (same) | methods | keep (dtor renames w/ class) |

### concrete GPU index implementations (all `final : public GpuIndexPort`→`GpuIndex`)

Each exposes the same override set — `insert`→`add_with_ids`, `remove`, `search`, `size`→`ntotal`,
`type_name`, (some) `vacuum`/`pending_removals`, plus `GpuIndex` verbs and
`export_snapshot`/`import_snapshot` — so only the class + file rename is listed per row.

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| **FILE** `GpuBruteForceIndex.hpp`/`.cpp` | `GpuIndexFlat.hpp`/`.cpp` | files | rename |
| `GpuBruteForceIndex` (GpuBruteForceIndex.hpp:16) | `GpuIndexFlat` | class | proposed (FAISS `GpuIndexFlat`) |
| **FILE** `GpuIVFFlatIndex.hpp`/`.cpp` | `GpuIndexIVFFlat.hpp`/`.cpp` | files | rename |
| `GpuIVFFlatIndex` (GpuIVFFlatIndex.hpp:20) | `GpuIndexIVFFlat` | class | proposed (FAISS `GpuIndexIVFFlat`) |
| **FILE** `GpuIVFPQIndex.hpp`/`.cpp` | `GpuIndexIVFPQ.hpp`/`.cpp` | files | rename |
| `GpuIVFPQIndex` (GpuIVFPQIndex.hpp:20) | `GpuIndexIVFPQ` | class | proposed (FAISS `GpuIndexIVFPQ`) |
| **FILE** `GpuGraphIndex.hpp`/`.cpp` | `GpuIndexHNSW.hpp`/`.cpp` | files | rename |
| `GpuGraphIndex` (GpuGraphIndex.hpp:22) | `GpuIndexHNSW` | class | proposed (align IndexHNSW; FAISS has no GpuIndexHNSW — ELIPS-specific) |
| `GpuHybridIndex` (GpuHybridIndex.hpp:17) | `GpuHybridIndex` | class | keep (ELIPS-specific; no FAISS equiv) |
| `GpuDistributedIndex` (GpuDistributedIndex.hpp:20) | `GpuDistributedIndex` | class | keep (ELIPS; FAISS `IndexShards`/replicas are separate) |
| `DistributedMode` (GpuDistributedIndex.hpp:15) + `replicate`/`shard` (:16-17) | (same) | enum class | keep |
| `detail::IvfIndexState` (GpuIVFFlatIndex.hpp:16 / GpuIVFPQIndex.hpp:16, fwd) | (same) | class fwd | keep (`elips::gpu::detail`) |
| `elips::HierarchicalGraphIndex` fwd (GpuGraphIndex.hpp:16) | `elips::IndexHNSW` fwd | class fwd | rename w/ the CPU class |

### GPU managers, pipelines, utilities

| Current (file:line) | New | Kind | Notes/collision |
|---|---|---|---|
| **FILE** `gpu_engine/DynamicBatcher.hpp`/`.cpp` | `gpu_engine/QueryBatcher.hpp`/`.cpp` | files | rename |
| `DynamicBatcher` (DynamicBatcher.hpp:17) | `QueryBatcher` | class | mission: DynamicBatcher→QueryBatcher |
| `DynamicBatcher::BatchStats` (:32) + fields `queries_coalesced/kernel_launches/avg_batch_size/p99_latency_us` (:33-36) | `BatchStats` | nested struct | keep |
| `DynamicBatcher` ctor (:19), dtor (:20), `enqueue` (:27), `flush` (:30), `stats` (:39), `start` (:41), `stop` (:42), `set_search_fn` (:44) | (same) | methods | keep |
| **FILE** `gpu_engine/GpuSelector.hpp`/`.cpp` | `gpu_engine/BackendSelector.hpp`/`.cpp` | files | rename |
| `GpuSelector` (GpuSelector.hpp:14) + `select` (:16) | `BackendSelector` | class | mission: GpuSelector→BackendSelector |
| `GpuDeviceManager` (GpuDeviceManager.hpp:14) + `cpu_fallback_info` (:18), `runtime_device_info` (:19), `probe_all_devices` (:21), `select` (:23), `can_fit_index` (:26) | `GpuDeviceManager` | class | keep |
| `FreeBlock` (GpuMemoryManager.hpp:21) + fields `ptr/bytes/root` | (same) | struct | keep |
| `GpuMemoryManager` (GpuMemoryManager.hpp:27, `: public GpuMemoryPort`) + `initialize` (:37), `allocate`/`deallocate`/`allocate_pinned`/`deallocate_pinned`/`bytes_used`/`bytes_available`/`peak_bytes_used`/`shutdown` | (same) | class | keep |
| `GpuMemoryPool` (GpuMemoryPool.hpp:14) + `acquire` (:24), `release` (:25), `available` (:27), `used` (:28) | (same) | class | keep |
| `GpuIndexTransferManager` (GpuIndexTransferManager.hpp:12) + static `clone` (:14), `clone_cpu_to_gpu` (:18), `clone_gpu_to_cpu` (:22) | (same) | class | keep |
| `GpuIngestionPipeline` (GpuIngestionPipeline.hpp:14) + `normalize_batch` (:18), `quantize_batch` (:21) | (same) | class | keep |
| `GpuQuantizationPipeline` (GpuQuantizationPipeline.hpp:12) + `train_pq_codebook` (:16), `encode_pq` (:23) | (same) | class | keep |
| `GpuSearchPipeline` (GpuSearchPipeline.hpp:15) + `batch_search` (:19) | (same) | class | keep |
| `KernelTiming` (GpuProfiler.hpp:11) + fields `kernel_name/duration/work_items` | (same) | struct | keep |
| `GpuProfiler` (GpuProfiler.hpp:17) + `record` (:19), `recent_timings` (:20), `total_launches` (:21), `clear` (:22) | (same) | class | keep |
| `CudaBackend` / `HipBackend` / `MetalBackend` (src/gpu_engine/backends/*) | (same) | classes | keep (concrete `GpuBackend` impls; CUDA/HIP are `#error` stubs per `_phase1_gpu.md`) |

---

## Collisions (C++ keyword / reserved) and their safe identifiers

| Chroma/intended name | C++ problem | Safe C++ identifier | Python exposes |
|---|---|---|---|
| `delete` (Collection delete; EQL erase) | `delete` is a C++ keyword | **`remove`** (FAISS `remove_ids`) — maps `Vault::erase`, `TransactionVault::erase`, `IndexPort::remove` | `delete` (+ `discard`) |
| `d` (FAISS dimension field) | one-letter, collides with locals; style | `dimension()` accessor; `d()` optional thin alias | n/a |
| `and`/`or`/`not` (filter ops) | reserved operator-keywords | already handled: `Filter::and_`/`or_`/`not_` (trailing `_`) | `$and`/`$or`/`$not` |
| `is_trained`/`reset`/`train` (FAISS) | none (safe) — listed to confirm no clash | use as-is on `Index` | n/a |

No other mission-target name collides with a C++ keyword. `add`, `search`, `get`, `query`,
`index`, `reconstruct`, `count`, `ntotal`, `metric_type`, `range_search` are all valid identifiers.

## Open proposals (mission table was silent — my proposals, all marked `proposed` above)

1. `ElipsInstance`→`Database` (engine type) + new `Client`/`PersistentClient` wrappers over `open()`.
   The brief said "keep `Database` as the engine type" but there is **no** `class Database` in C++;
   `ElipsInstance` is it. Renaming unifies C++ with `src/Database.cpp` and the Python-native name.
2. `Vault::fetch`→`get` (Chroma point-get; FAISS `reconstruct(id)` equivalent). `scan` kept distinct
   (Python `get(where=)` folds both).
3. `Vault::place_record`→`add_record`; `ElipsInstance::adopt_vault`→`adopt_collection`;
   `TransactionVault`→`TransactionCollection`; `Transaction::vault`→`collection`.
4. `GraphParams`→`HnswParams` + `Config::graph_params`→`hnsw_params`; `IndexType::graph/exact`→
   `hnsw`/`flat`; `IndexSnapshotKind::exact/graph/gpu_brute_force/gpu_graph`→
   `flat/hnsw/gpu_flat/gpu_hnsw`; `GpuIndexAlgorithm::BruteForce`→`Flat`.
5. `IndexPort::insert`→`add_with_ids` and `size`→`ntotal` (FAISS contract); add missing FAISS
   virtuals (`add`,`train`,`reset`,`reconstruct`,`range_search`,`is_trained`,`d`,`metric_type`).
6. GPU concrete indexes → FAISS `GpuIndex*`: `GpuBruteForceIndex`→`GpuIndexFlat`,
   `GpuIVFFlatIndex`→`GpuIndexIVFFlat`, `GpuIVFPQIndex`→`GpuIndexIVFPQ`,
   `GpuGraphIndex`→`GpuIndexHNSW`.
7. EQL AST: `InsertStatement`→`AddStatement`; all `vault` fields across AST/`WAL::Entry`→`collection`;
   `SearchStatement::top`→`n_results`.
8. `QueryStrategy::text_probe`→`text_search`.

## Symbols that stay UNCHANGED (whole layers)

- **domain** (Errors, Record, RecordID, SearchResult, Vector, Expected): all keep.
- **quant_engine** (QuantParams, Quantizer, ProductQuantizer, ScalarQuantizer, KMeans, CodecId): all
  keep — already FAISS-standard / clearer; record FAISS param aliases only.
- **storage** (WAL, Serialization, FileSync): all keep (ELIPS internals, no counterpart).
- **text_engine** (TextEmbedderPort, LocalTextEmbedder): all keep.
- **vector_engine** (`distance`, `requires_normalization`), **kernel** (LockManager, LockMode,
  LockConflict): keep.
- **metadata/Filter** + `Comparator`: keep (only `MetadataIndex`→`MetadataFilterIndex` renames).
- Most of **Config** and all GPU support types (buffers, device info, metrics, managers, pipelines,
  profiler, memory ports): keep.

## File renames (headers + src)

| Current | New |
|---|---|
| `include/elips/index_engine/IndexPort.hpp` | `index_engine/Index.hpp` |
| `include/elips/index_engine/ExactIndex.hpp` + `src/ExactIndex.cpp` | `index_engine/IndexFlat.hpp` + `src/IndexFlat.cpp` |
| `include/elips/index_engine/HierarchicalGraphIndex.hpp` + `src/HierarchicalGraphIndex.cpp` | `index_engine/IndexHNSW.hpp` + `src/IndexHNSW.cpp` |
| `include/elips/index_engine/IndexFactory.hpp` + `src/IndexFactory.cpp` | `index_engine/index_factory.hpp` + `src/index_factory.cpp` |
| `include/elips/metadata/MetadataIndex.hpp` + `src/MetadataIndex.cpp` | `metadata/MetadataFilterIndex.hpp` + `src/MetadataFilterIndex.cpp` |
| `include/elips/gpu_engine/GpuPort.hpp` | `gpu_engine/GpuBackend.hpp` |
| `include/elips/gpu_engine/GpuIndexPort.hpp` | `gpu_engine/GpuIndex.hpp` |
| `include/elips/gpu_engine/GpuSelector.hpp` + `src/gpu_engine/GpuSelector.cpp` | `gpu_engine/BackendSelector.hpp` + `.cpp` |
| `include/elips/gpu_engine/DynamicBatcher.hpp` + `src/gpu_engine/DynamicBatcher.cpp` | `gpu_engine/QueryBatcher.hpp` + `.cpp` |
| `include/elips/gpu_engine/GpuBruteForceIndex.hpp` + `.cpp` | `gpu_engine/GpuIndexFlat.hpp` + `.cpp` |
| `include/elips/gpu_engine/GpuIVFFlatIndex.hpp` + `.cpp` | `gpu_engine/GpuIndexIVFFlat.hpp` + `.cpp` |
| `include/elips/gpu_engine/GpuIVFPQIndex.hpp` + `.cpp` | `gpu_engine/GpuIndexIVFPQ.hpp` + `.cpp` |
| `include/elips/gpu_engine/GpuGraphIndex.hpp` + `.cpp` | `gpu_engine/GpuIndexHNSW.hpp` + `.cpp` |

`src/Database.cpp` keeps its name (now matches the renamed `Database` engine class). `src/EQLLexer`,
`EQLParser`, `QueryExecutor`, `Metrics`, `Filter`, `RecordID`, `WAL`, `LocalTextEmbedder`,
`LockManager`, quant `.cpp`s keep their names.





