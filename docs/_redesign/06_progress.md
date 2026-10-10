# 06 — Progress Log (single source of truth for orchestration)

Running checklist. Update at every gate. Dates are absolute (today: 2026-10-10).

## Phase status

| Phase | Title | Status | Gate met? |
|---|---|---|---|
| 1 | Inventory & baseline | DONE (2026-10-10) | ☑ |
| 2 | Prune GPU backends (SYCL, Vulkan) | DONE (2026-10-10) | ☑ |
| 3 | Study FAISS | DONE (2026-10-10) | ☑ |
| 4 | Gap analysis | DONE (2026-10-10) | ☑ |
| 5 | Naming & public API redesign | DONE (2026-10-10) | ☑ |
| 6 | Core engine maturity | IN PROGRESS (2026-10-10) | ☐ |
| 7 | GPU acceleration redesign | not started | ☐ |
| 8 | Build / packaging / test infra | not started | ☐ |
| 9 | Documentation | not started | ☐ |
| 10 | Release readiness | not started | ☐ |

## Phase 1 checklist
- [x] 1A C++ inventory (`_phase1_cpp.md`) — core essentially COMPLETE, not stubby
- [x] 1B Python inventory (`_phase1_python.md`) — modern Chroma-ish surface already partly exists
- [x] 1C GPU real-status inventory (`_phase1_gpu.md`) — only Metal partial-real; CUDA/HIP/SYCL/Vulkan = `#error` stubs
- [x] 1D docs-vs-code audit (`_phase1_docs.md`) — severe vocabulary drift; GPU ADRs overclaim
- [x] Build baseline recorded (`05_test_matrix.md`) — green: build 0, ctest 255/256 (1 skip), pytest 239/244
- [x] Test baseline recorded (`05_test_matrix.md`)
- [x] Synthesized into `00_inventory.md`
- [x] One descriptive commit for Phase 1 (f496d41)

## Phase 1 verdict (gate)
Baseline is GREEN and the true state is captured. Mission premise corrected: **the CPU engine is
mature**; immaturity = GPU backends + naming/vocabulary + docs truth + a few test gaps. Phase 1 gate MET
pending the commit. Recommended re-scoping to feed later phases (does not change phase order):
- Phase 2 (prune SYCL/Vulkan) is now also the natural time to fix the latent GPU-CMake build breaks (1C).
- Phase 5 (naming) is **reconcile 3 layers + add Client/PersistentClient + expose FAISS Index API**, not greenfield.
- Phase 7 (GPU) real work on Metal: move top-k onto the GPU, wire the dead `.metal` file, make real-kernel tests not skip.

## Phase 5 checklist (deep rename — ADR-R2)
- [x] Naming map drafted (3 agents → `_naming_cpp.md` 344 rows, `_naming_python.md`, `_naming_other.md`)
- [x] `03_naming_map.md` ADR recorded (collisions resolved, DO-NOT-RENAME safety list, shim policy, batch plan)
- [x] Batch 1: C++ core types/methods → build+ctest green — `DynamicBatcher`→`QueryBatcher`, `GpuSelector`→`BackendSelector`, `MetadataIndex`→`MetadataFilterIndex` (3 classes + files + deprecated `using` aliases). Independently verified: build 0, ctest 256/256 (255+1skip), pytest 34/34.
- [x] Batch 2: core index/port classes → build+ctest+pytest green — `IndexPort`→`Index`, `ExactIndex`→`IndexFlat`, `HierarchicalGraphIndex`→`IndexHNSW`, `GpuPort`→`GpuBackend`, `GpuIndexPort`→`GpuIndex`, `GraphParams`→`HnswParams`, `ElipsInstance`→`Database` (7 classes + files + deprecated aliases; no collisions). Verified: ctest 256/256, pytest 34/34.
- [x] Batch 3: `Vault`→`Collection` (type + `VaultInfo`/`TransactionVault` compounds, lowercase identifiers, 13 user-facing strings; on-disk `vault_` prefix + manifest keys PRESERVED; pybind Python names + methods deferred). Verified: ctest 256/256 (persistence tests pass = format intact), pytest 34/34.
- [x] Batch 4: API verbs/methods → green — place→add, seek→search, erase→remove, fetch→get, explain_seek→explain_query, insert→add_with_ids, size→ntotal, Database::vault→collection, quantize→train (+ `[[deprecated]]` forwarder shims). std `.erase/.insert/.size` + EQL AST + pybind name strings left intact. Verified: ctest 256/256, pytest 34/34.
- [x] Batch 5: EQL verbs + AST → green — new primary keywords add/search/get/delete/metadata/k with old (place/seek/fetch/erase/data/top) accepted as deprecated synonyms; AST `InsertStatement`→`AddStatement`, `FetchStatement`→`GetStatement` (+aliases), `.vault`→`.collection`; new test `EqlTest.NewVocabularyAddSearchGetDelete`. Verified: ctest 257 (256+1skip), pytest 34/34.
- [x] Batch 6: Python Chroma surface → green — renamed Python-exposed names (DeprecationWarning shims), added `Client()`/`PersistentClient(path=)`, Chroma Collection methods (`add/upsert/get/update/delete/query/count/peek/modify`) + kwargs (`ids/embeddings/documents/metadatas/n_results(+k,+limit)/where/include/offset`), wired typed exceptions (`NotTrainedError`/`CollectionNotFoundError`/`BackendUnavailableError`), retired dead `IndexSnapshot*` (Python), fixed docstring lies. Verified: build 0, test_bindings 34/34; Client(dimension=2)→get_or_create_collection→add→query→list_collections works end-to-end. Deferred to Phase 6: exposing FAISS Index API + index_factory, `where_document`, `delete_collection`/`modify`/`reset`.
  - **Two Chroma-parity gaps found during verification (→ Phase 6, recorded in 02):** (a) `Client()` is dimension-first (no zero-config dim inference — ELIPS default embedder is dimension-parameterized); now raises a clear actionable error instead of a cryptic ConfigError. (b) Chroma arbitrary string `ids` need an id-mapping layer (ELIPS `RecordID` is a 32-hex UUID); `add(embeddings=…)` with auto-generated ids works today.
- [x] Batch 7: CMake options → both paths green — `ELIPS_GPU_METAL/CUDA/HIP`→`ELIPS_ENABLE_METAL/CUDA/ROCM`, `ELIPS_BUILD_PYTHON`→`ELIPS_ENABLE_PYTHON`, `ELIPS_BUILD_TESTS`→`BUILD_TESTING`, each with a `message(DEPRECATION)` back-compat alias; first-party callers (setup.py, example, CI workflows) moved to new names. Verified: new-flag fresh configure clean + ctest 257; old-flag configure emits deprecation + ctest 257. (C++ typed-exception raising + test/file-name cosmetics moved to Phase 6/8.)
- [x] Batch 8: `MIGRATION.md` (full old→new tables + deprecation timeline) + `CHANGELOG.md` Unreleased (BREAKING/Added/Removed/Deprecated). Final code-level grep: all renamed class/option tokens appear ONLY in shim/alias lines (verified). Docs-prose old-vocab cleanup = Phase 9.
- [x] Full suite green; committed per batch

## Phase 5 gate: MET
Deep rename applied across C++ / Python / EQL / CMake in 8 verified batches; build + ctest 257 (256+1skip) + pytest 34/34 green at every commit; deprecation shims for all old public names; on-disk format untouched (persistence tests green). Deferred to Phase 6 (recorded in 02): expose FAISS Index API + `index_factory` in Python, IDMap for arbitrary string ids, `delete_collection`/`modify`/`reset`, `where_document`, C++ typed-exception raising.

## Phase 6 checklist (core maturity — fill gaps, don't rebuild)
- [x] 6.1 CPU IVFFlat + `index_factory` + Index-contract extension (train/is_trained/reset defaults). ctest 268 (267+1skip); recall@10 vs Flat oracle: euclidean .999/cosine .985/dot .995 @nprobe8, →1.0 @nprobe=nlist.
- [x] 6.2 IVFPQ + IVF-SQ (CPU) over the same factory. ctest 287 (286+1skip). IVFPQ 16× compression, recall@10 ~0.72 plateau (by_residual PQ); IVF-SQ 4× compression, recall@10 0.96. factory `IVF<n>,PQ<M>` / `IVF<n>,SQ8`.
- [x] 6.6 expose FAISS Index API in Python — `elips.IndexFlat/IndexIVFFlat/IndexIVFPQ/IndexIVFScalarQuantizer/IndexHNSW` + `index_factory`, numpy float32/C-contig in, `(D float32, I int64)` out with -1 pad, settable `nprobe`. PyIndex adapter holds int64↔RecordID map. test_index_api.py 8/8; test_bindings 34/34; ctest 287. **Phase 8 note:** numpy is now a real runtime dep of the Index API — add to pyproject.
- [ ] 6.3 IDMap (arbitrary string ids ⇄ RecordID) — unblocks Chroma `add(ids=…)`
- [ ] 6.4 IndexRefine (`refine_fraction`), batched search, true `range_search`
- [ ] 6.5 metric squared/true option; k>ntotal pad policy; numpy float32/C-contig boundary; C++ typed-exception raising
- [ ] 6.6 expose FAISS Index API in Python (IndexFlat/IVFFlat/HNSW, index_factory, read/write_index) — deferred from Phase 5
- [ ] 6.7 documented concurrency/threading policy

## Open questions
(see `04_architecture.md`)

## Log
- 2026-10-10: Phase 1 kicked off. Repo oriented: ~32k LOC source (82 cpp / 67 hpp / 36 py). 5 GPU backends present as directories; SYCL+Vulkan marked for Phase 2 removal (24 / 28 ref'd files resp.). Prebuilt `_core.cpython-313/314-darwin.so` present in bindings.
- 2026-10-10: Phase 1 DONE. 4 inventory agents + baseline landed; synthesized `00_inventory.md`; committed (`redesign` branch, f496d41). Premise corrected: CPU engine mature.
- 2026-10-10: Phase 2 DONE. Deleted SYCL + Vulkan entirely: source dirs (`src/gpu_engine/backends/{sycl,vulkan}`), CMake options/wiring (root + gpu-engine), device probe/construct/rank in `GpuDeviceManager.cpp`/`GpuSelector.cpp`, 2 ADRs (GPU-009/010), 16 doc files + 3 website files scrubbed, 1 test renamed (`…cuda_over_hip`), accelerator.py + README backend lists. Backend set now Metal/CUDA/HIP/CPU. Repo-wide grep for `sycl|vulkan|oneapi|spirv|glsl|vk` CLEAN outside `docs/_redesign`. Rebuild green: ctest 256 (255 pass/1 skip), pytest 34/34. Gate MET.
- NOTE for Phase 8: latent build issues left untouched (out of Phase 2 scope): `src/gpu_engine/CMakeLists.txt` `ELIPS_GPU_TESTS` adds a missing `tests/` subdir (gated OFF); CUDA/HIP `CMakeLists` name nonexistent `.cu`/`.hip` sources (actual files `.cpp`); linker "ignoring duplicate libraries" (elips_core/elips_gpu linked twice).
- 2026-10-10: Phase 3 started. Cloned facebookresearch/faiss @ v1.15.1 (`75c755c5`) to /tmp/faiss. Fanned out 3A–3H study agents. **MAJOR CORRECTION (overrides mission ground rule #8):** upstream FAISS v1.15.1 now has a REAL Metal backend (`faiss/gpu_metal/`, ~7700 LOC, incl. GPU top-k `topk_threadgroup_2048`). Added agent 3I to study it; recorded as ADR-R1 in `04_architecture.md`. Phase 7 Metal work is now adapt-a-precedent, not greenfield.
- 2026-10-10: Phase 3 DONE. All 9 section reports written (`_faiss_3A..3H.md` + `_faiss_3I_metal.md`); synthesized into `01_faiss_study.md` with 8 reality-corrections + adopt/adapt/skip per section + consolidated decisions feeding Phases 5–8. Gate MET (study non-destructive, nothing built). Next: Phase 4 gap analysis.
- 2026-10-10: Phase 4 DONE. 3 gap agents (4A index/search/CRUD/persistence, 4B GPU/Metal, 4C API/build/concurrency/docs) → merged into `02_gap_analysis.md`. 7 blockers identified; every gap carries an owner phase (gate MET). Headlines: 3 live vocabularies (B1), no index_factory (B2), no Client/PersistentClient (B3), GPU top-k on CPU + naive kernel + no real-kernel test (B4-6), docs actively wrong (B7). ELIPS has NO CPU IVF (GPU-only) — biggest core gap. ELIPS ahead on durability/hybrid/batcher/unified-memory. Committed `02_gap_analysis.md`.
