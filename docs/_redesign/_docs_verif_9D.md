# Phase 9D — Docs verification (EQL / storage / dev-guide / README / CHANGELOG / MIGRATION / roadmap / FAQ)

Date: 2026-10-11. Verified against FINAL code; runnable Python checked with the
built extension (`PYTHONPATH=bindings/python`, Python 3.14, elips 1.1.0, Apple M4,
`has_gpu=True`). No code, website, or out-of-scope docs touched.

## Files updated / created

| File | Change |
|---|---|
| `docs/eql.md` | Rewritten to the REAL grammar: primary verbs `add/search/get/scan/delete`, deprecated synonyms `place/seek/fetch/erase`, `k`/`metadata` clauses, `$`-bindings, real error strings. Explicit note that the uppercase `SEARCH FROM … SEEK :q TOP` dialect never existed. |
| `README.md` | Rewritten quickstart to the current API (Chroma `Client`/`PersistentClient`/Collection + FAISS Index API + `bf_knn`). Removed arena/vault/ingest/probe/place_document/seek_text/top=. Added backend-maturity table. |
| `CHANGELOG.md` | Added to Unreleased: `bf_knn`, Metal GPU (real kernels, M4-validated), device selection (`has_gpu`/`accelerators`/`bf_knn device=`), build & packaging (`install(EXPORT)`/`find_package`, `ELIPS_OPT_LEVEL`, C API, shared libs, CI). |
| `MIGRATION.md` | Added FAISS Index-API section (new layer, no old name), `IndexType::graph/exact`→`hnsw/flat` and `make_index`→`index_factory` rows. |
| `docs/roadmap.md` | Made honest: moved shared locks, adaptive-`ef`, quantized indexes to Shipped. Remaining = GPU IVFPQ-on-device, CUDA/HIP hardware validation, parallel search, Python `range_search`/`reconstruct`, `read_index`/`write_index`, GPU resources API, abi3 wheels. |
| `docs/cli.md` | `vaults`→`collections`, `--vault`→`--collection`, EQL examples re-verbed, added `stats`/`quantize`. |
| `docs/storage.md` | WAL ops add `insert_q` + `txn_begin`/`txn_commit`; noted `vault_<n>_<epoch>.segment` is a legacy on-disk prefix. |
| `docs/faq.md` | NEW: dimension-first, string ids, backend selection, which backends work, true-L2 vs FAISS squared, Collection vs Index API, deprecation. |
| `docs/developer-guide/contributing.md` | CMake options → `ELIPS_ENABLE_PYTHON`/`ELIPS_ENABLE_METAL/CUDA/ROCM`/`BUILD_TESTING`; gtest filters `ExactIndex*`→`IndexFlat*`; `IndexPort`→`Index` in tree. |
| `docs/developer-guide/debugging.md` | API names `place/fetch/erase`→`add/get/remove`, `vault`→`collection`, `ElipsInstance`→`Database`, `ExactIndexEdgeTest`→`IndexFlatEdgeTest`, `ELIPS_GPU_METAL`→`ELIPS_ENABLE_METAL`, `ELIPS_BUILD_PYTHON`→`ELIPS_ENABLE_PYTHON`, EQL `seek…top`→`search…k`. |
| `docs/developer-guide/release-process.md` | Version `1.0.0`→`1.1.0`, CLI `vaults`→`collections` (+`quantize`), wheel/pre-release CMake flags. |
| `docs/developer-guide/testing.md` | Build/python-test CMake flags; `ExactIndexTest`→`IndexFlatTest`; renamed test-file paths (`exact_index_test.cpp`→`index_flat_test.cpp`, `exact_edge_cases_test.cpp`→`index_flat_edge_cases_test.cpp`, `DynamicBatcherTest.cpp`→`QueryBatcherTest.cpp`); `ParsesSearchStatement` desc. |

## Fabricated EQL removed / corrected

`docs/eql.md` did not carry the uppercase SQL-like dialect (that lived in the
website, out of scope) but used old verbs/clauses. It now documents only what
`src/EQLParser.cpp` implements and explicitly states the following were never
implemented and are gone:
- uppercase `SEARCH FROM … SEEK :q TOP …` / `INSERT` / `SCAN FROM` / `FETCH FROM … WHERE id = N`
- `:colon` bindings (real grammar uses `$name`, `EQLParser.cpp:100`)
- positional errors like `Expected number after TOP at position 30` (parser has no position tracking; real messages are `EQL: expected a number`, `EQL: expected a statement keyword, got '…'`, etc.)

## Ground-truth checks (code)

- EQL verbs & aliases: `EQLParser.cpp:17-21` — `search|seek`, `get|fetch`, `scan`, `add|place`, `delete|erase`; `k|top` (`:191`), `metadata|data` (`:264`); `nearest`/`in`/`from`/`where`/`threshold`/`rank_by`/`project`/`yield`.
- Python Index API present via `_native`: `IndexFlat/IndexIVFFlat/IndexIVFPQ/IndexIVFScalarQuantizer/IndexHNSW`, `index_factory`, `bf_knn`. ABSENT: `read_index`/`write_index`, `get_num_gpus`, `index_cpu_to_gpu`/`gpu_to_cpu`, `StandardGpuResources`, and Index `range_search`/`reconstruct` (C++-only). Captured in roadmap + MIGRATION.
- CMake FINAL: `ELIPS_ENABLE_METAL/CUDA/ROCM/PYTHON/C_API/BLAS`, `ELIPS_OPT_LEVEL`, `BUILD_TESTING`, `BUILD_SHARED_LIBS` — but `ELIPS_BUILD_CLI/BENCH/FUZZERS` kept the `BUILD_` prefix (NOT renamed). Deprecated aliases honored (`CMakeLists.txt:111-121`). `find_package(elips)` + `elips::elips_core`, NAMESPACE `elips::`.
- CLI FINAL (`cli/src/main.cpp`): subcommand `collections` (not `vaults`), flag `--collection`, commands `info/collections/query/checkpoint/export/import/verify/stats/bench/quantize`.
- Metric: `IndexFlat(2,"l2")` of `[3,4]` from origin → `5.0` (true L2, not 25).
- Device: `accelerators()` → `[Apple M4 / metal]`; `bf_knn(..., device=...)`.

## Runnable snippets

`scripts/run_doc_snippets.py` (honors `no-exec`) over the edited docs:
- `docs/eql.md` (1), `docs/faq.md` (2) → 3 executed, 0 failed.
- `README.md` (4 Python blocks) → 4 executed, 0 failed.
- `docs/cli.md`, `docs/storage.md`, `docs/roadmap.md`, `docs/developer-guide/**` → bash/cpp/text only, no Python to run.

## Remaining gaps (out of 9D scope)

- `docs/developer-guide/coding-standards.md` still shows pre-rename C++ symbols
  (`ElipsInstance`, `IndexPort`, `ExactIndex`, `make_index`, `place`/`seek`) in
  convention examples. It documents C++ style, not build/test/CLI, so left as-is;
  belongs with the C++ SDK/reference pass.
- `docs/developer-guide/testing.md` per-test catalog tables (lines ~210-400) still
  carry some old vocab in test descriptions. The renamed FILE paths and the broken
  CMake/filter commands were fixed; individual TEST() rows left alone because
  several old-vocab names (e.g. `PlaceThenSeekViaEql`, `FetchAndEraseViaEql`) are
  still the REAL test names (they exercise the deprecated-synonym path), so rewriting
  them wholesale would introduce wrong names.
- Website (`elips_docs/`) and `docs/python`, `docs/cpp`, `docs/architecture`,
  `docs/internals`, `docs/adr`, `docs/overview` untouched per scope.
- CHANGELOG C++ `range_search` entry is accurate for C++ but not yet Python-exposed;
  roadmap tracks the Python binding as remaining.
