# Phase 5C — Naming Map: EQL + CMake + tests + errors + file names

Planning only (no code touched). Decision basis: **ADR-R2 DEEP rename** (`04_architecture.md:22`).
Canonical verb mapping from the mission + ADR-R2: `place`→`add`, `seek`→`search`, `erase`→`delete`,
`fetch`→`get`, `Vault`→`Collection`, `Database`→`Client`-tier. EQL is a DSL, so `delete` as a
keyword is fine (no C++-keyword collision inside the grammar). CMake targets stay recognizable;
options re-aligned to FAISS (`_faiss_3F.md`). "(mission silent)" marks a proposal I chose where the
brief gives no direction. Columns are exactly `Current (file:line) | New | Kind | Notes`.

---

## 1. EQL keywords / grammar

### 1a. Statement verbs (dispatch at `src/EQLParser.cpp:13-21`)

| Current (file:line) | New | Kind | Notes |
|---|---|---|---|
| `place` (EQLParser.cpp:18, :249; EQLLexer n/a) | `add` | EQL keyword | Canonical (ADR-R2). `parse_insert` dispatch + `expect_word("place")`. |
| `seek` (EQLParser.cpp:15, :175) | `search` | EQL keyword | Canonical. `parse_search` already named for the target verb. |
| `fetch` (EQLParser.cpp:17, :217) | `get` | EQL keyword | Chroma by-id read is `get`. (mission silent on EQL spelling; recommended). |
| `scan` (EQLParser.cpp:16, :228) | `scan` (keep) | EQL keyword | No clean Chroma equivalent (filter+offset+limit pagination). Keep. (mission silent). |
| `erase` (EQLParser.cpp:19, :262) | `delete` | EQL keyword | Canonical; DSL keyword so `delete` is safe (unlike the C++ method, which takes `remove`). |

### 1b. Clause / preposition keywords (`src/EQLParser.cpp`)

| Current (file:line) | New | Kind | Notes |
|---|---|---|---|
| `in` — seek/scan/place (EQLParser.cpp:176, :229, :250) | `in` (keep) | EQL keyword | Reads correctly with new verbs (`add in c`, `search in c`). Keep. |
| `from` — fetch/erase (EQLParser.cpp:218, :263) | `from` (keep) | EQL keyword | Could unify to `in` for consistency; keep `from` (`get from c`, `delete from c` read fine). (mission silent). |
| `nearest` (EQLParser.cpp:179) | `nearest` (keep) | EQL keyword | Query-vector marker. Keep. (mission silent). |
| `top` (EQLParser.cpp:182) | `k` | EQL keyword | FAISS-aligned, terse for a DSL. Python kwarg unifies to `n_results` (alias `k`) per `02_gap_analysis.md:21`; EQL uses `k`. (mission silent on EQL spelling; recommended). |
| `threshold` (EQLParser.cpp:184) | `threshold` (keep) | EQL keyword | Radius/range search. Keep. |
| `where` (EQLParser.cpp:186, :233) | `where` (keep) | EQL keyword | Matches Chroma `where`. Keep. |
| `rank_by` (EQLParser.cpp:188) | `rank_by` (keep) | EQL keyword | No Chroma/FAISS analog. Keep. (mission silent). |
| `project` (EQLParser.cpp:193) | `project` (keep) | EQL keyword | Chroma uses `include=`; EQL field-projection stays `project`. Keep. (mission silent). |
| `yield` (EQLParser.cpp:200) | `yield` (keep) | EQL keyword | Statement terminator. Keep. |
| `vector` (EQLParser.cpp:253) | `vector` (keep) | EQL keyword | ELIPS keeps the `Vector` domain type (not renamed), so keep `vector`; Chroma's `embeddings` is Python-surface only. (mission silent; recommended keep). |
| `data` (EQLParser.cpp:255) | `metadata` | EQL keyword | Aligns with `meta`→`metadatas` unification (`02_gap_analysis.md:21`). Payload metadata object. |
| `id` (EQLParser.cpp:221, :266) | `id` (keep) | EQL keyword | Keep. |
| `offset` / `limit` (EQLParser.cpp:235, :237) | keep | EQL keyword | scan pagination. Keep. |
| `in` set-membership (EQLParser.cpp:142) | `in` (keep) | EQL filter op | Keep. |
| `contains` (EQLParser.cpp:154) | `contains` (keep) | EQL filter op | Substring match. Keep. |
| `and` / `or` / `not` (EQLParser.cpp:117, :124, :130) | keep | EQL filter op | Keep. |
| `true` / `false` (EQLParser.cpp:107) | keep | EQL literal | Keep. |
| `distance` — rank_by sentinel (EQLParser.cpp:190) | `distance` (keep) | EQL keyword | Keep. |

Lexer (`src/EQLLexer.cpp`, `EQLLexer.hpp`) is keyword-agnostic (generic word/number/string/punct);
**no lexer change** — all verb/keyword knowledge lives in `EQLParser.cpp`. `TokenKind` enum stays.

### 1c. AST artifacts (`include/elips/query_engine/AST.hpp`) — overlaps the C++ core sub-doc; listed because scope names the query_engine files

| Current (file:line) | New | Kind | Notes |
|---|---|---|---|
| `InsertStatement` (AST.hpp:42) | `AddStatement` | C++ struct | Matches `place`→`add`. Also `parse_insert`→`parse_add` (EQLParser.cpp:248). |
| `FetchStatement` (AST.hpp:30) | `GetStatement` | C++ struct | Matches `fetch`→`get`. Also `parse_fetch`→`parse_get` (EQLParser.cpp:216). |
| `SearchStatement` (AST.hpp:20) | `SearchStatement` (keep) | C++ struct | Already target-named. |
| `ScanStatement` (AST.hpp:35) | `ScanStatement` (keep) | C++ struct | `scan` kept. |
| `DeleteStatement` (AST.hpp:48) | `DeleteStatement` (keep) | C++ struct | Already target-named. |
| `.vault` field — all 5 structs (AST.hpp:21, :31, :36, :43, :49) | `.collection` | C++ member | `Vault`→`Collection`. Executor reads these (QueryExecutor.cpp:87, :105, :118, :129, :135). |
| `Statement` variant (AST.hpp:53) | update member list | C++ alias | Reflect `AddStatement`/`GetStatement` renames. |

### 1d. FABRICATED EQL dialect in the website docs — grammar does NOT implement it (flag for Phase 9)

The real grammar is **case-sensitive lowercase**, uses `$name` bindings, verb-first `seek`/`place`
(there is **no** `SEARCH` or `INSERT` keyword), and `fetch`/`erase` take `id "<uuid>"` (not `WHERE id = N`).

| Current (file:line) | New | Kind | Notes |
|---|---|---|---|
| `SEARCH FROM documents SEEK :q TOP 10 WHERE ...` (docs.python.eql.tsx:45, :51, :76, :112) | lowercase real grammar `search in docs nearest $q k 10 where ... yield` | Doc fix | **Fabricated.** No `SEARCH` verb, no `:` bindings, no `TOP`/`WHERE id` uppercase forms in `EQLParser.cpp`. |
| `SCAN FROM vault_name ...` (docs.python.eql.tsx:79) | `scan in <collection> ...` | Doc fix | Fabricated uppercase + `FROM` (grammar uses `in`). |
| `FETCH FROM vault_name WHERE id = 1042` (docs.python.eql.tsx:82) | `get from <collection> id "<uuid>" yield` | Doc fix | Fabricated: grammar has no `WHERE id =` fetch form; id is a string literal. |
| `:name` colon bindings (docs.python.eql.tsx:57, :66, :93) | `$name` | Doc fix | Grammar binds with `$` (EQLParser.cpp:91). |
| `INSERT`/`DELETE` listed as EQL verbs (docs.python.eql.tsx:12) | `add`/`delete` | Doc fix | `INSERT` never existed; verb is `place`→`add`. |
| "declarative, SQL-like query interface" (docs.python.eql.tsx:34; api-reference/eql.md:3) | "small expression-oriented DSL" | Doc fix | Overstated; it is not SQL. |
| Uppercase section headers `PLACE/SEEK/FETCH/SCAN/ERASE/PROJECT/YIELD` (api-reference/eql.md:36, :48, :77, :87, :97, :109, :138) | lowercase `add/search/get/scan/delete/project/yield` | Doc fix | Keywords are lowercase; uppercase misleads. |
| `ParseError: Expected number after TOP at position 30` (api-reference/eql.md:168) | real msg `EQL: expected a number` | Doc fix | **Fabricated error.** No positional tracking in parser; no `TOP`. |
| `ParseError: Unexpected token at position 0` (api-reference/eql.md:171) | real msg `EQL: expected a statement keyword, got '...'` (EQLParser.cpp:20) | Doc fix | Fabricated positional error. |

`docs/eql.md` and `elips_docs/src/routes/docs.eql.tsx` ARE accurate (lowercase) — only re-verb them
(`seek`→`search`, etc.); they do not carry the fabricated dialect.

---

## 2. CMake targets & options

### 2a. Options (`CMakeLists.txt`) — re-align to FAISS (`_faiss_3F.md:197-214`)

| Current (file:line) | New | Kind | Notes |
|---|---|---|---|
| `ELIPS_GPU_METAL` (CMakeLists.txt:46) | `ELIPS_ENABLE_METAL` | CMake option | FAISS `FAISS_ENABLE_METAL` (`_faiss_3F.md:41`). Default `${APPLE}` kept. |
| `ELIPS_GPU_CUDA` (CMakeLists.txt:47) | `ELIPS_ENABLE_CUDA` | CMake option | FAISS `FAISS_ENABLE_GPU`→CUDA. Default OFF; EXPERIMENTAL. |
| `ELIPS_GPU_HIP` (CMakeLists.txt:48) | `ELIPS_ENABLE_ROCM` | CMake option | FAISS `FAISS_ENABLE_ROCM` (`_faiss_3F.md:35`). Default OFF; EXPERIMENTAL. |
| _(none — computed var)_ `ELIPS_GPU_ENABLED` (CMakeLists.txt:56) | `ELIPS_ENABLE_GPU` (user umbrella) + keep internal computed flag | CMake option/var | NEW umbrella (`_faiss_3F.md:199`): default ON only when a backend is viable. Keep the internal "a-backend-is-on" boolean (rename to `ELIPS_HAS_GPU_BACKEND` to avoid clashing with the user option). **Also a C++ macro** `ELIPS_GPU_ENABLED` (CMakeLists.txt:101, :131, :196, :264) — if renamed, update `#ifdef ELIPS_GPU_ENABLED` sites (code change, cross-ref C++ sub-doc). |
| `ELIPS_BUILD_PYTHON` (CMakeLists.txt:238) | `ELIPS_ENABLE_PYTHON` | CMake option | FAISS `FAISS_ENABLE_PYTHON` (`_faiss_3F.md:37`). pybind11, default OFF. |
| `ELIPS_BUILD_TESTS` (CMakeLists.txt:141) | `BUILD_TESTING` | CMake option | Standard CTest var (`_faiss_3F.md:44`); adopt verbatim via `include(CTest)`. |
| `ELIPS_BUILD_CLI` (CMakeLists.txt:105) | `ELIPS_ENABLE_CLI` | CMake option | FAISS has no CLI; `ENABLE_` prefix for consistency. (mission silent). |
| `ELIPS_BUILD_BENCH` (CMakeLists.txt:118) | `ELIPS_ENABLE_BENCH` | CMake option | FAISS folds benches into `FAISS_ENABLE_EXTRAS`; keep a dedicated bench flag. (mission silent). |
| `ELIPS_BUILD_FUZZERS` (CMakeLists.txt:224) | `ELIPS_ENABLE_FUZZERS` | CMake option | Consistency. (mission silent). |
| `ELIPS_SANITIZE` (CMakeLists.txt:22) | `ELIPS_SANITIZE` (keep) | CMake option | No FAISS equivalent; keep. (mission silent). |
| `ELIPS_PYBIND11_SOURCE_DIR` (CMakeLists.txt:242) | keep | CMake cache | Impl detail. Keep. |
| `ELIPS_PYTHON_OUTPUT_DIR` (CMakeLists.txt:272) | keep | CMake cache | Impl detail. Keep. |
| _(none)_ | `ELIPS_ENABLE_C_API` | NEW CMake option | FAISS `FAISS_ENABLE_C_API` (`_faiss_3F.md:38`). No current ELIPS C API; option is forward-looking, default OFF. (mission requests the name; feature is Phase 8+). |
| _(none)_ | `BUILD_SHARED_LIBS` | NEW CMake option | Standard CMake var (`_faiss_3F.md:45`). Today ELIPS forces static+PIC (CMakeLists.txt:13); honor the standard var. |
| _(none)_ | `ELIPS_OPT_LEVEL` | NEW CMake option | FAISS `FAISS_OPT_LEVEL`; ELIPS enum `generic/neon/avx2/avx512` (`_faiss_3F.md:204`). NEON mandatory baseline on arm64. |

### 2b. Targets — keep recognizable (`add_library`/`add_executable`)

| Current (file:line) | New | Kind | Notes |
|---|---|---|---|
| `elips_core` (CMakeLists.txt:66) | `elips_core` (keep) | CMake target | FAISS core is `faiss`; keep `elips_core` recognizable. |
| `elips_gpu` (src/gpu_engine/CMakeLists.txt) | `elips_gpu` (keep) | CMake target | Keep. |
| `elips_gpu_metal` (CMakeLists.txt:198) | `elips_gpu_metal` (keep) | CMake target | Matches FAISS `faiss_metal`/`libfaiss_metal.a`. Keep. |
| `elips_cli` (CMakeLists.txt:107) | `elips_cli` (keep) | CMake target | `OUTPUT_NAME elips` kept. |
| `elips_bench` (CMakeLists.txt:120) | `elips_bench` (keep) | CMake target | Keep. |
| `elips_gpu_bench` (CMakeLists.txt:129) | `elips_gpu_bench` (keep) | CMake target | Keep. |
| `elips_tests` (CMakeLists.txt:154) | `elips_tests` (keep) | CMake target | FAISS `faiss_test`. Keep. |
| `elips_gpu_lifecycle_probe` (CMakeLists.txt:205) | `elips_gpu_lifecycle_test` | CMake target | `Database`→`Client` + "probe"→"test" to avoid vocab clash; also renames the source file (see §5). (mission silent on the "probe" word; recommended). |
| `elips_fuzz_wal` (CMakeLists.txt:229) | `elips_fuzz_wal` (keep) | CMake target | Keep. |
| `elips_pymodule` (CMakeLists.txt:261) | `elips_pymodule` (keep) | CMake target | `OUTPUT_NAME _core` kept. |

CUDA/HIP backend sub-CMake (`src/gpu_engine/backends/{cuda,hip}/CMakeLists.txt`) also gate on the old
option names and reference nonexistent `.cu`/`.hip` sources (`00_inventory.md:81`) — update the option
references here when renaming; the source-name fix is a Phase 8 build repair, not a naming change.

---

## 3. Error messages / types that embed old vocabulary

### 3a. Error strings (`src/`) — user-facing

| Current (file:line) | New | Kind | Notes |
|---|---|---|---|
| `"...vector-first APIs such as place() and seek()."` (Database.cpp:275, :279, :282) | `"...such as add() and search()."` | error string | Three copies of the embedder-missing message. |
| `"cannot write vault segment"` (Database.cpp:615) | `"cannot write collection segment"` | error string | StorageError. |
| `"error while writing vault segment"` (Database.cpp:628) | `"error while writing collection segment"` | error string | StorageError. |
| `"missing vault segment: " + ...` (Database.cpp:725) | `"missing collection segment: "` | error string | StorageError. |
| `"segment vault mismatch for " + ...` (Database.cpp:736) | `"segment collection mismatch for "` | error string | StorageError. |
| `"vector dimension does not match vault"` (Database.cpp:825) | `"vector dimension does not match collection"` | error string | DimensionMismatch. |
| `"vault is opened in read-only mode"` (Database.cpp:840, :1928) | `"collection is opened in read-only mode"` | error string | StorageError. |
| `"record is quantized but the vault has no matching codebook"` (Database.cpp:939) | `...the collection has no...` | error string | StorageError. |
| `"quantized record does not match the vault codebook"` (Database.cpp:973) | `...the collection codebook` | error string | StorageError. |
| `"persisted codebook dimension does not match the vault"` (Database.cpp:1451) | `...the collection` | error string | StorageError. |
| `"vault has no quantization codec configured; ..."` (Database.cpp:1476) | `"collection has no quantization codec..."` | error string | ConfigError. |
| `"vault is already quantized"` (Database.cpp:1480) | `"collection is already quantized"` | error string | ConfigError. |
| `"cannot train a codebook on an empty vault: ..."` (Database.cpp:1484) | `"...on an empty collection: ..."` | error string | ConfigError. |
| `"' in seek"` parse-error suffix (EQLParser.cpp:197) | `"' in search"` | error string | ParseError in `parse_search`. |
| `"' in scan"` parse-error suffix (EQLParser.cpp:241) | keep (`scan` retained) | error string | No change. |
| `"vault_" + ordinal + "_" + epoch + ".segment"` (Database.cpp:1687) | **keep `vault_` OR migrate** | on-disk filename | **Not a UI string — on-disk segment-file prefix**, read back at Database.cpp:725/736. Renaming to `collection_` changes the persisted layout and breaks existing DBs without a migration/compat reader. Recommend KEEP the on-disk prefix (or add a dual-read + migration). Flag for Phase 6/8. |

### 3b. Error TYPES (`include/elips/domain/Errors.hpp`) — class names carry no old vocab; only doc-comments do

| Current (file:line) | New | Kind | Notes |
|---|---|---|---|
| `// ...match the database/vault configuration.` (Errors.hpp:14) | `.../collection configuration.` | comment | `DimensionMismatch` class name unchanged. |
| `// Requested record/vault does not exist.` (Errors.hpp:32) | `// Requested record/collection does not exist.` | comment | `NotFound` class name unchanged (generic). `02_gap_analysis.md:37` separately proposes NEW typed errors `CollectionNotFoundError`/`NotTrainedError`/`BackendUnavailableError` (Phase 6 additions, not renames). |
| `DimensionMismatch`/`InvalidVector`/`ConfigError`/`NotFound`/`StorageError`/`ElipsError` (Errors.hpp:9-42) | keep | C++ type | No old vocabulary in the names; keep. |

**Not old vocabulary — do NOT rename:** `nprobe` / `probe_lists` / `probe_lists_batch`
(IvfIndexState.cpp:395, :426; GpuIVFFlatIndex.cpp:95) are **FAISS-standard IVF terms** (keep), and
`probe_all_devices` (GpuDeviceManager.cpp:60) is hardware enumeration. The deprecated "probe" is only
the Python `Collection.probe`/`probe_text` **search alias** (`00_inventory.md:25`) — that lives in the
Python sub-doc, not here.

### 3c. CLI strings (`cli/src/main.cpp`) — user-facing, outside the literal `src/` grep scope; listed for completeness

| Current (file:line) | New | Kind | Notes |
|---|---|---|---|
| `--vault NAME` option + `option(a, "vault")` (main.cpp:167, :191, :220) | `--collection NAME` | CLI flag | Accept `--collection`; keep `--vault` as a deprecated alias. |
| `commands: info vaults query ...` + `vaults` subcommand (main.cpp:298, :319; dispatch) | `collections` | CLI command | Rename subcommand `vaults`→`collections`, keep alias. |
| `"vaults: " << vaults.size()` / `vault.<name>.records` / `imported ... into <vault>` (main.cpp:116, :258, :234, :245, :175, :302) | `collection(s)` wording | CLI output | Cosmetic output strings. |

---

## 4. Test names / files that embed old vocabulary

### 4a. Test FILE names

| Current (file:line) | New | Kind | Notes |
|---|---|---|---|
| `tests/unit/metadata_index_test.cpp` | `tests/unit/metadata_filter_index_test.cpp` | test file | `MetadataIndex`→`MetadataFilterIndex` (ADR-R2). Update CMakeLists.txt:162 + fixture `MetadataIndexTest` (metadata_index_test.cpp:13 etc.)→`MetadataFilterIndexTest`. |
| `tests/integration/gpu/DatabaseGpuLifecycleProbe.cpp` | `tests/integration/gpu/ClientGpuLifecycleTest.cpp` | test file | `Database`→`Client`; "Probe"→"Test" (vocab-clash avoidance). Update CMakeLists.txt:206 target + add_test name `GpuDatabaseLifecycleProbe` (CMakeLists.txt:213)→`GpuClientLifecycleTest`. (mission silent on "probe"→"test"). |
| `tests/unit/scan_test.cpp` | keep | test file | `scan` retained. No rename. |
| other test files (eql_test.cpp, roundtrip_test.cpp, etc.) | keep file names | test file | File names carry no old vocab; only internal TEST() names change (§4b). |

### 4b. GTest `TEST()`/`TEST_F()` names (representative; apply the verb map mechanically across the suite)

| Current (file:line) | New | Kind | Notes |
|---|---|---|---|
| `EqlTest.PlaceThenSeekViaEql` (tests/unit/eql_test.cpp:23) | `EqlTest.AddThenSearchViaEql` | test name | place→add, seek→search. |
| `EqlTest.FetchAndEraseViaEql` (eql_test.cpp:47) | `EqlTest.GetAndDeleteViaEql` | test name | fetch→get, erase→delete. |
| `DatabaseTest.InMemoryPlaceAndSeek` (tests/integration/roundtrip_test.cpp:55) | `ClientTest.InMemoryAddAndSearch` | test name + fixture | `DatabaseTest` fixture→`ClientTest`; place/seek→add/search. |
| `DatabaseTest.EraseRemovesRecord` (roundtrip_test.cpp:125) | `ClientTest.DeleteRemovesRecord` | test name | erase→delete. |
| `DatabaseTest.ExplainSeekUsesMetadataAccelerationAndHybridPlanner` (roundtrip_test.cpp:203) | `ClientTest.ExplainSearchUses...` | test name | seek→search. |
| `FilterTest.FilteredSeekRestrictsByMetadata` (tests/integration/transaction_filter_test.cpp:15) | `FilterTest.FilteredSearchRestrictsByMetadata` | test name | seek→search. |
| `M2Test.ConcurrentWriterAndReadersOnOneVault` (tests/concurrency/m2_thread_safety_test.cpp:35) | `...OnOneCollection` | test name | vault→collection (M2Test fixture kept — milestone name). |
| `M2Test.ConcurrentEraseAndSearchStayConsistent` (m2_...:84) | `ConcurrentDeleteAndSearch...` | test name | erase→delete. |
| `M2Test.ConcurrentVaultCreationIsSafe` (m2_...:118) | `ConcurrentCollectionCreationIsSafe` | test name | vault→collection. |
| `M2Test.PlaceAfterCloseThrowsInsteadOfVanishing` (m2_...:192) | `AddAfterCloseThrows...` | test name | place→add. |
| `M2Test.EraseAfterCloseThrows` (m2_...:201) | `DeleteAfterCloseThrows` | test name | erase→delete. |
| `M2Test.VaultCreatedAfterCloseRefusesWrites` (m2_...:210) | `CollectionCreatedAfterCloseRefusesWrites` | test name | vault→collection. |
| `DurabilityTest.MultipleVaultsSurviveCheckpoint` (tests/unit/durability_test.cpp:83) | `MultipleCollectionsSurviveCheckpoint` | test name | vault→collection. |
| `M1Test.CommitOnReadOnlyVaultAppliesNothing` (tests/unit/m1_durability_atomicity_test.cpp:165) | `CommitOnReadOnlyCollection...` | test name | vault→collection. |
| `M1Test.TransactionEraseRollsBackOnFailure` (m1_...:237) | `TransactionDeleteRollsBackOnFailure` | test name | erase→delete. |
| `M3IndexHealth.DatabaseVacuumReclaimsAcrossVaults` (tests/unit/m3_index_health_test.cpp:235) | `ClientVacuumReclaimsAcrossCollections` | test name | Database→Client, vaults→collections. |
| `M5QuantizationPersistence.VaultInfoReportsTheCompressionRatio` (tests/unit/m5_quantization_persistence_test.cpp:501) | `CollectionInfoReportsTheCompressionRatio` | test name | vault→collection. |
| `MetadataIndexTest.*` (tests/unit/metadata_index_test.cpp:13, :40, :69) | `MetadataFilterIndexTest.*` | test name/fixture | MetadataIndex→MetadataFilterIndex. |
| `GpuDeviceManagerTest.probe_returns_results_on_supported_system` (tests/unit/gpu/GpuDeviceManagerTest.cpp:50) | keep (`probe`=device probe) | test name | NOT search-vocab; hardware enumeration. Keep. (mission silent). |

### 4c. Python test names (`tests/python/*.py`)

| Current (file:line) | New | Kind | Notes |
|---|---|---|---|
| `test_place_many` (tests/python/test_bindings.py:362) | `test_add_many` | py test | place→add. |
| `test_place_many_text_batch_with_lineage` (test_bindings.py:316) | `test_add_many_text_batch_with_lineage` | py test | place→add. |
| `test_generated_id_round_trips_through_place` (tests/python/test_exposed_surface.py:51) | `..._through_add` | py test | place→add. |
| `test_place_with_malformed_id_raises` (test_exposed_surface.py:66) | `test_add_with_malformed_id_raises` | py test | place→add. |
| `populated_vault` fixture (tests/python/test_gil_release.py:68, :91) | `populated_collection` | py fixture | vault→collection. |
| `test_gpu_devices_probe` (test_bindings.py:607) | keep (device probe) | py test | NOT search-vocab. Keep. (mission silent). |

---

## 5. Source FILE names that embed old vocabulary

ADR-R2 renames the classes; the files follow. (The class/method renames themselves are enumerated in
the C++/Python naming sub-doc — this table is file paths only.)

| Current (file:line) | New | Kind | Notes |
|---|---|---|---|
| `include/elips/metadata/MetadataIndex.hpp` + `src/MetadataIndex.cpp` (CMakeLists.txt:80) | `MetadataFilterIndex.hpp` + `src/MetadataFilterIndex.cpp` | source file | ADR-R2 explicit example. Update `#include` sites + CMakeLists.txt source list + include guard `ELIPS_METADATA_METADATA_INDEX_HPP`. |
| `include/elips/gpu_engine/GpuSelector.hpp` + `src/gpu_engine/GpuSelector.cpp` | `BackendSelector.hpp` + `BackendSelector.cpp` | source file | ADR-R2 explicit example (`GpuSelector`→`BackendSelector`). Update gpu_engine/CMakeLists.txt + includes. |
| `include/elips/gpu_engine/DynamicBatcher.hpp` + `src/gpu_engine/DynamicBatcher.cpp` | `QueryBatcher.hpp` + `QueryBatcher.cpp` | source file | ADR-R2 (`DynamicBatcher`→`QueryBatcher`). |
| `include/elips/gpu_engine/GpuPort.hpp` | `GpuBackend.hpp` | source file | ADR-R2 (`GpuPort`→`GpuBackend`). Widely included across gpu_engine (`GpuPort&` params in GpuIVFFlatIndex.cpp etc.). |
| `include/elips/index_engine/IndexPort.hpp` | `Index.hpp` | source file | ADR-R2 (`IndexPort`→`Index`). |
| `src/Database.cpp` (CMakeLists.txt:84) | `src/Client.cpp` | source file | ADR-R2 `Database`→`Client`-tier. **Caveat:** facade header is `include/elips/elips.hpp` (no `Database.hpp`); confirm whether the `Database` class splits into `Client`/`PersistentClient` (`02_gap_analysis.md:14`) before fixing the filename. (recommended; verify in C++ sub-doc.) |
| `tests/integration/gpu/DatabaseGpuLifecycleProbe.cpp` | `ClientGpuLifecycleTest.cpp` | source file | See §4a. |
| `tests/unit/metadata_index_test.cpp` | `metadata_filter_index_test.cpp` | source file | See §4a. |

**Mission-silent / do NOT rename in this phase:**
- `*Port.hpp` siblings — `IndexTransferPort`, `TextEmbedderPort`, `GpuKernelPort`, `GpuStreamPort`,
  `GpuMemoryPort`, `GpuIndexPort` — ADR-R2 only names `IndexPort`→`Index` and `GpuPort`→`GpuBackend`.
  The remaining `*Port` (hexagonal-architecture suffix) are mission-silent; recommend deciding the
  suffix policy in the C++ sub-doc, not here.
- `src/gpu_engine/GpuIngestionPipeline.*` — "ingest" is a deprecated **Python** alias of `add`, but
  this is an internal GPU pipeline class; keep (mission silent).
- `src/gpu_engine/backends/metal/shaders/elips_distance_kernels.metal` — brand name only, no vocab.
- EQL files (`EQLLexer.*`, `EQLParser.*`, `QueryExecutor.*`, `AST.hpp`) — "EQL" = ELIPS Query
  Language, retained. File names keep; only internal symbols change (§1c).
