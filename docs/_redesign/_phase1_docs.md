# Phase 1D — Docs vs Code Audit (`_phase1_docs.md`)

Scope: `docs/` tree (markdown + ADRs), the website source under `elips_docs/src/`, and
the live site https://elips.tubox.cloud. Checked against `include/`, `src/`,
`bindings/python/elips/`, `cli/`, `CMakeLists.txt`. No examples executed (Phase 9).
Evidence: every row carries a docs location and a code `file:line` (or "no code found").

---

## Ground truth (what the code actually ships)

- **Python high-level public API** (`bindings/python/elips/__init__.py:113` `__all__`):
  `connect()` → **`Engine`**; `Engine.collection()` → **`Collection`** (alias `Arena`).
  Collection leads with conventional names **`add`/`search`/`get`/`delete`** (aliases of
  `write`/`probe_text`/`pull`/`discard`) — `collection.py:939-958`. Records are
  `RecordInput`/`Row`/`Hit`. There is **no `query` and no bare `seek`/`place`** on the
  high-level surface.
- **Native layer** (`_core.pyi`, class names as bound): **`Database`** (not `ElipsInstance`),
  `Vault`, `Filter`, result type **`Result`**. Module funcs `open`/`open_with_config`.
  Reachable as `elips.native.*` or `Engine.native` / `Collection.raw`.
- **C++** (`include/elips/elips.hpp:290`): top class is **`elips::ElipsInstance`**;
  `elips::open(path, config={})` → `unique_ptr<ElipsInstance>`. Vault/Config/EQL use the
  low-level vocabulary (vault/seek/place/erase).
- **EQL** (`src/EQLParser.cpp`): verbs **seek, fetch, scan, place, erase** (lowercase),
  `in/from/nearest/top/threshold/where/rank_by/project/yield/offset/limit`. Not SQL.
- **GPU** (`src/gpu_engine/backends/`): **only Metal is real** (`metal/MetalBackend.mm`, 272
  lines, default ON on Apple). `cuda/hip/sycl/vulkan` host `.cpp` are **3-line stubs that
  `#error` if enabled** (`CudaBackend.cpp` etc.), default **OFF**, and are **never
  constructed** (only `MetalBackend` is). `ELIPS_GPU_ENABLED` can be ON "with no backend
  behind" (`CMakeLists.txt:43-44`) → CPU fallback reporting only. GPU indexes ARE wired
  into `IndexFactory.cpp:40-49` and `Database.cpp:772`, so they function **only when a real
  (Metal) backend is present**.
- **CLI** (`cli/src/main.cpp:318-327`): `elips` binary, commands info, vaults, query,
  checkpoint, export, import, verify, stats, bench, **quantize**.

---

## Headline findings

1. **Vocabulary split is the dominant defect.** The shipped high-level API leads with
   `connect → Engine`, `Collection`, `add`/`search`/`get`/`delete`. Nearly all prose docs
   and the **entire website** instead teach the old/low-level surface
   (`open → Database`, `Vault`/`Arena`, `place`/`place_document`/`seek`/`probe`/`ingest`/
   `erase`, `top=`, `meta=`). Website vocabulary counts: `vault` 666, `arena` 396, `seek`
   212, `erase` 91, `probe` 78, `ingest` 65, `place_document` 41 vs `Collection` 12,
   `.add(` 1, `.search(` 1. Only `docs/python/architecture/package-structure.md` uses the
   new names. The live site https://elips.tubox.cloud/docs/python-sdk explicitly states
   **"There is no add, ingest, search, get, or delete"** — the exact opposite of code
   (`collection.py:939-958`).
2. **GPU docs claim a five-backend engine; only Metal exists.** 10 GPU ADRs (all status
   *Accepted*) and `docs/internals/gpu-engine.md` present CUDA/HIP/SYCL/Vulkan as
   implemented (cuVS/CAGRA, rocBLAS, SPIR-V, speedup numbers). In code those four are
   `#error` stubs. Only two audit docs tell the truth.
3. **Three different `open()` signatures across docs**, none exactly matching
   `_core.pyi:1681`; `index.md` invents `durability=`/`graph_params=`/`gpu_config=` kwargs
   that do not exist.
4. **`connect(gpu=True)` (website) is wrong** — code takes `gpu: GpuConfig | None`
   (`connect.py:73`). `engine.gpu_info()` (website) does not exist on `Engine`.
5. **Name/field conflicts within the docs themselves**: Python `ElipsInstance` vs code
   `Database`; `SearchResult` vs code `Result`; Filter `.ge()` vs code `.gte()`; C++
   `SearchResult` documented as both 3-field and 6-field (code: 6 + codec).
6. **Undocumented-but-shipped surface**: `Collection`/`Engine`/`add`/`search`/`compress`/
   `health`/`CollectionHealth`/`freeze`/`pending_writes`/`merge`, CLI `quantize`,
   `QuantParams`/`Codec`.

---

## GPU maturity claims vs reality

Code reality: `MetalBackend.mm` (272 lines) is the only implemented backend;
`cuda/hip/sycl/vulkan/*Backend.cpp` are 3-line stubs that `#error` when their
`ELIPS_*_ENABLED` flag is set; defaults OFF (`CMakeLists.txt:46-50`); only `MetalBackend`
is ever constructed; `GpuSelector.cpp:8-14` still ranks all five.

| Backend | Docs claim (location) | Reality | Verdict |
|---|---|---|---|
| **Metal** | "primary GPU backend on Apple" `internals/gpu-engine.md:294`; "Working, verified on M4" `architecture/gpu-device-reporting-audit.md:132`; website "Hardware-accelerated matmul on M1–M4" | Real (`MetalBackend.mm`), default ON Apple | **accurate** (but `supports_cagra/ivf_pq=true` advertised at `gpu-engine.md:311-313` overstate — Metal has distance/top-k kernels only) |
| **CUDA** | ADR-GPU-002 *Accepted* "cuVS CAGRA/IVF-PQ primary + cuBLAS" `ADR-GPU-002:13-18`; "implements GpuPort via CUDA Runtime API" `gpu-engine.md:799`; selector rank 100 `gpu-engine.md:265`; website "Native CUDA C++ kernels … Tensor Cores" | 3-line `#error` stub `cuda/CudaBackend.cpp`; OFF; never constructed | **documented-but-nonexistent** (over-claim) |
| **HIP** | ADR-GPU-001:19 / ADR-GPU-009:17 "hipVS/CAGRA, rocBLAS"; "implements GpuPort via HIP/ROCm" `gpu-engine.md:800`; website "ROCm acceleration for Instinct MI200/MI300" | 3-line `#error` stub `hip/HipBackend.cpp`; OFF | **documented-but-nonexistent** |
| **SYCL** | ADR-GPU-009 *Accepted* "Intel Arc/Flex/Max, oneMKL … 2× speedup vs CPU" `ADR-GPU-009:18,33`; "implements GpuPort via SYCL 2020" `gpu-engine.md:801` | 3-line `#error` stub `sycl/SyclBackend.cpp`; OFF; absent from website gpu/overview | **documented-but-nonexistent** |
| **Vulkan** | ADR-GPU-010 *Accepted* "SPIR-V universal fallback … ~1.5× vs CPU, binaries checked in" `ADR-GPU-010:13-34`; `gpu-engine.md:803` | 3-line `#error` stub `vulkan/VulkanBackend.cpp`; OFF; absent from website | **documented-but-nonexistent** |

GPU feature-level over-claims (all `documented-but-nonexistent` on non-Metal, or placeholder):
- **CAGRA / cuVS / GPU IVF-Flat / IVF-PQ**: `ADR-GPU-002:16`, `gpu-engine.md:578-625`,
  `faiss-gpu-acceleration-design.md:574-584`. Design doc itself admits `GpuGraphIndex` "is
  not a true GPU graph traversal" and `GpuIVF*` "use generic backend distance/top-k, not
  FAISS-grade kernels" (`faiss-gpu-acceleration-design.md:914-916`). No cuVS in tree.
- **GpuDistributedIndex (replicate/shard)**: `faiss-gpu-acceleration-design.md:593-812`;
  admitted "not wired into IndexFactory/Config/Database" (`:917-918`).
- **`gpu_stats` / `GpuMetricsSnapshot`**: "largely unimplemented / placeholder"
  (`gpu-device-reporting-audit.md:188`, `gpu-runtime-stability-audit.md:187`) — accurate admissions.
- **ADR index** `docs/adr/README.md:5-16` lists only ADR-0001..0010; the 10 GPU ADRs are
  omitted from the index.
- Accurate counter-notes exist: `gpu-device-reporting-audit.md:21-22,133-136` and
  `gpu-runtime-stability-audit.md:132-135` correctly label CUDA/HIP/SYCL/Vulkan
  "Stubbed/incomplete" and explain `_has_gpu` is a compile flag, not device detection.

---

## Mismatch table

Types: `nonexistent` = documented-but-nonexistent · `undocumented` = exists-but-undocumented ·
`sig` = wrong-signature · `default` = wrong-default · `ret/err` = wrong-return/error ·
`vocab` = vocabulary-drift.

### Python — high-level surface

| Documented symbol | Docs location | Code status | Type |
|---|---|---|---|
| Whole prose/site teaches `open→Database`, `Vault`/`Arena`, `place`/`seek`/`probe`/`ingest` | `python_sdk.md`, `cookbook.md`, `api/python/index.md`, website (vault 666/arena 396 hits) | Shipped lead API is `connect→Engine`, `Collection`, `add`/`search`/`get`/`delete` (`collection.py:48,939-958`; `__init__.py:113`) | vocab |
| "There is no add, ingest, search, get, or delete" | live `/docs/python-sdk`; `elips_docs/src/routes/docs.python-sdk.tsx` | All exist: `add`/`add_many`/`search`/`search_vector`/`search_hybrid`/`get`/`delete` (`collection.py:939-958`); `ingest`/`merge` (`collection.py:306,392`) | nonexistent (false denial) |
| `Collection`, `add`, `search`, `get`, `delete`, `compress`, `health`, `CollectionHealth`, `freeze`, `merge` | only `python/architecture/package-structure.md:18` names Collection; absent elsewhere | Exist (`collection.py`; `models.py:449`) | undocumented |
| `Engine`, `Engine.collection_names`/`pending_writes`/`vacuum` | absent from main Python ref | Exist (`engine.py:93,181,201,155`) | undocumented |
| `connect(..., gpu=True)` | website gpu/overview; `docs.gpu.overview.tsx:120`; `/docs/gpu/overview` | `connect(gpu: GpuConfig \| None=None)` (`connect.py:73`) | sig |
| `engine.gpu_info()` | website gpu/overview | `Engine` has no `gpu_info`; only native `Database.gpu_info` (`_core.pyi:1018`; `engine.py` lacks it) | nonexistent |
| `db.query(...)` where `db` = `connect()` result | implied by connect-returns-Engine framing + EQL examples | `query` is on native `Database` only (`_core.pyi:1007`), **not** on `Engine`; needs `db.native.query` | nonexistent (on Engine) |
| `elips.ElipsInstance` (Python handle) | `api/python/index.md:16` | Python binds native handle as `Database` (`_core.pyi:948`); `ElipsInstance` is the C++ name | vocab/sig |
| `SearchResult` (result class) | `api/python/index.md:85` | Native class is `Result` (`_core.pyi:621`); high-level returns `Hit`/`Row` | ret/err |
| `Filter.ge(value)` | `api/python/index.md:103` | Method is `gte` (`_core.pyi:675`); `ge` is only the `Comparator` enum member | sig |
| `open(..., durability=, graph_params=, gpu_config=)` | `api/python/index.md:14` | `open(... gpu=, embedder=, use_default_text_embedder=, quantization=)` (`_core.pyi:1681`); no such kwargs | sig/nonexistent |
| `open(...)` signatures disagree across files | `database.md:11` vs `api/python/index.md:14` vs `bindings/pybind11-architecture.md` | Single real signature `_core.pyi:1681` | sig |
| `scan(limit=sys.maxsize)` vs `scan(limit=-1)` | `api/python/index.md:76` vs `python/api-reference/vault.md:110` | Native default all; wrapper uses `-1` sentinel (`collection.py:689`); C++ `max()` (`elips.hpp:167`) | default |
| package files `modern.py`,`errors.py`,`types.py` | `python/getting-started/installation.md:112-120` | Real: `exceptions.py`, `typing.py`, split pkgs; `modern.py` is a deprecated shim; no `errors.py`/`types.py` | nonexistent |
| `metric="cosine"`, `dimension=0`, `index="graph"`, `lexical_weight=0.25`, `top=10` | python docs | Match code (`connect.py:62-71`; `collection.py:495`; `_core.pyi:149`) | OK (no mismatch) |

### Website source (`elips_docs/src/`)

| Documented symbol | Docs location | Code status | Type |
|---|---|---|---|
| Collection route titled "Arena"; collections called `Arena`/`arena` throughout | `routes/docs.python.arena.tsx:8,23,48`; `routes/index.tsx:84-92` | Primary name is `Collection`; `Arena` is a back-compat alias (`collection.py:964`) | vocab |
| Methods `write/write_many/ingest/merge/probe/probe_text/probe_hybrid/pull/sweep/discard` as the API | `routes/docs.python.arena.tsx:177-732` | Exist but are the non-conventional names; site omits `add/search/get/delete` | vocab/undocumented |
| **SQL-like EQL dialect**: `SEARCH FROM documents SEEK :q TOP 10 WHERE …`, `DELETE FROM`, `HYBRID_WEIGHT 0.3`, `:name` bindings | `routes/docs.python.eql.tsx:45,76,79,82,85,97` | Grammar has none of `SEARCH/DELETE FROM/HYBRID_WEIGHT`; verbs are `seek/erase`, bindings are `$name` (`EQLParser.cpp:174-269`, `EQLLexer.cpp`) | nonexistent |
| `connect(index_type="hnsw" \| "ivf_flat" \| "ivf_pq", durability="sync_on_commit" \| "wal_only" \| "in_memory", dimension=128)` | `routes/docs.python.connect.tsx:43-53` | Code: `index` ∈ {`graph`,`exact`}; durability ∈ {`paranoid`,`standard`,`relaxed`,`ephemeral`}; `dimension` default `0`; kwarg is `index` not `index_type` (`connect.py:61-66`, `_core.pyi:22-23`) | default/sig |
| GPU `connect(..., gpu=True)`; `gpu_info()` → `{'backend':'cuda'}` | `routes/docs.gpu.overview.tsx:120-125` | `gpu: GpuConfig\|None` (`connect.py:73`); cuda backend is a stub | sig/nonexistent |
| CUDA/HIP/Metal "backend support" with no maturity labels; SYCL/Vulkan only in `lib/tutorial.ts:122,1101` | `routes/docs.gpu.overview.tsx:43-52`; `lib/tutorial.ts` | Only Metal real | nonexistent (CUDA/HIP) |
| Lowercase EQL dialect (correct) | `routes/docs.eql.tsx:47,89` | Matches grammar | OK |

### Python / native — EQL

| Documented symbol | Docs location | Code status | Type |
|---|---|---|---|
| EQL verbs `seek/fetch/scan/place/erase`, clauses `in/nearest/top/threshold/where/rank_by/project/yield/offset/limit` | `docs/eql.md:20-61`; `python/api-reference/eql.md:34` | Match `EQLParser.cpp:174-269` exactly | OK |
| "SQL-like declarative interface" | `python/api-reference/eql.md:3,293` | EQL is not SQL (no SELECT/INSERT/ORDER BY) — loose prose | vocab (minor) |
| `python/api-reference/eql.md` omits `threshold`/`rank_by`/`offset` | `python/api-reference/eql.md:53,90` | All exist in grammar (`EQLParser.cpp:184,188,235`) | undocumented |
| `Database.query`, `validate_eql`, `tokenize_eql`, `TokenKind`, `Token` | `python/api-reference/eql.md:8,144,176,216` | Exist (`_core.pyi:1007,1079,1092,123,132`) | OK |

### C++ surface

| Documented symbol | Docs location | Code status | Type |
|---|---|---|---|
| `records()` returns `std::vector<StoredRecord>` | `cpp/api-reference/vault.md:131` | Returns `std::map<RecordID, Record>` (`elips.hpp:178`); `StoredRecord` is a Python TypedDict, not a C++ type | ret/err + nonexistent |
| C++ `SearchResult` = 3 fields `{id,distance,data}` | `cpp/api-reference/domain-types.md`, `cpp/architecture/layers.md:136` | 6 fields + codec + `approximate()` (`SearchResult.hpp:15-27`) | ret/err |
| C++ `SearchResult` = 6 fields | `cpp/api-reference/vault.md:184` | Matches code (`SearchResult.hpp:15-21`) | OK |
| `Vault::place` 3-param impl | `cpp/internals/storage-engine.md:146` | 6-param (`elips.hpp:130`); 6-param form in `vault.md` is correct | sig (internal conflict) |
| build flag `ELIPS_BUILD_BENCHMARKS` | `cpp/performance/benchmarking.md:236` | Option is `ELIPS_BUILD_BENCH` (`CMakeLists.txt:120`) | nonexistent |
| `elips::ElipsInstance`, `elips::open`, Vault signatures, Config builder, GraphParams 16/200/50/0.2, `seek_hybrid lexical_weight=0.25` | `cpp_sdk.md`, `cpp/api-reference/*` | Match code (`elips.hpp:290,370,130-167`; `Config.hpp:46-52,159`) | OK |
| `QueryStrategy` named, values undocumented | `cpp/api-reference/vault.md:102` | Values exist `{ann_index,exact_candidates,full_scan,text_probe,hybrid_fusion}` (`elips.hpp:98`) | undocumented |

### CLI

| Documented symbol | Docs location | Code status | Type |
|---|---|---|---|
| commands info/vaults/stats/verify/query/checkpoint/export/import/bench | `docs/cli.md:17-35` | All exist (`cli/src/main.cpp:318-326`) | OK |
| `quantize` command | not in `docs/cli.md` | Exists (`cli/src/main.cpp:327`) | undocumented |

### Cross-doc staleness (roadmap vs changelog)

| Claim | Docs location | Reality | Type |
|---|---|---|---|
| Multi-reader shared locks "deferred" | `roadmap.md:28` | Shipped v1.1.0 per `CHANGELOG.md`, `storage.md:98`, `lock-manager.md` | stale |
| ef-adaptive filtered search "planned v1.1" | `overview/project-history.md:207` | Shipped per `CHANGELOG.md:76-80` | stale |
| GPU absent from roadmap entirely | `roadmap.md` | GPU (Metal) wired + shipped (`IndexFactory.cpp:40`) | undocumented |

---

## Doc examples that call nonexistent / wrong symbols (flag for Phase 9)

- `connect(..., gpu=True)` and `engine.gpu_info()` — website gpu/overview examples
  (wrong gpu type; no Engine.gpu_info).
- Any example doing `db.query(...)` / `db.vacuum()` on a `connect()` handle treated as the
  native Database — `query` is not on `Engine` (use `db.native.query`); `vacuum` IS on
  Engine (`engine.py:155`), so only `query` fails.
- `api/python/index.md` snippets using `elips.ElipsInstance`, `SearchResult`, `Filter().ge(...)`,
  and `open(..., durability=/graph_params=/gpu_config=)`.
- C++ `records()` examples typed as `std::vector<StoredRecord>` (`cpp/api-reference/vault.md`).
- GPU examples assuming CUDA/HIP/SYCL/Vulkan selection (ADRs, `gpu-engine.md`) — only Metal
  resolves at runtime.

## Unverified / notes

- Website rows are backed by the `elips_docs/src` extraction, two direct live-site
  WebFetches (`/docs/python-sdk`, `/docs/gpu/overview`), and a direct `grep` of
  `elips_docs/src` (vocabulary + GPU-backend hit counts in Headline findings). The live
  site matches the source (both old-vocabulary). Website line numbers in the table are
  from source files; a few per-line citations beyond those shown are approximate.
- The website carries **two incompatible EQL dialects** — `routes/docs.eql.tsx` (lowercase,
  matches grammar) and `routes/docs.python.eql.tsx` (uppercase SQL-like, does not exist).
- `docs/internals/transaction-engine.md` referenced by `cpp/api-reference/elips-instance.md:102`
  **does exist** (not a broken link).
- Exact C++ `SearchResult` field list verified at `include/elips/domain/SearchResult.hpp:15-27`.

