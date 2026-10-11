# Phase 3I — FAISS v1.15.1 upstream Metal backend study (`/tmp/faiss/faiss/gpu_metal/`)

Read-only study of the ~7700 LOC Metal backend FAISS now ships. Every claim cites
`file:line`. Goal: a precedent map for ELIPS's own Metal-first backend (Phase 7),
with an explicit **adopt / adapt / skip** verdict per area.

ELIPS baseline this maps onto (for grounding):
- `src/gpu_engine/backends/metal/MetalBackend.mm:249` — `top_k()` runs on CPU via
  `std::partial_sort` (`:257`). **This is the gap FAISS closes with a GPU kernel.**
- `src/gpu_engine/backends/metal/MetalBackend.mm:267` — `synchronize()` is a no-op.
- `src/gpu_engine/backends/metal/shaders/elips_distance_kernels.metal` (111 lines) —
  real shader, currently dead code (distance compute runs elsewhere; top-k is CPU).

---

## 0. State map of the FAISS backend (what is really GPU vs CPU)

| Index | Distance | Top-k | Verdict |
|-------|----------|-------|---------|
| `MetalIndexFlat` | GPU (`l2_squared_matrix`/`ip_matrix`) | **GPU** (`topk_threadgroup_*`) | fully GPU |
| `MetalIndexIVFFlat` | GPU (scan kernels) | **GPU** (bitonic per-list + merge) | fully GPU, CPU coarse-assign fallback |
| `MetalIndexIVFPQ` | **CPU** | **CPU** | GPU buffers built but search delegates to CPU |

The single most important takeaway: **Flat and IVFFlat do top-k on the GPU**; the
`.metal` top-k kernel is the artifact ELIPS is missing. IVFPQ is a stub —
`MetalIndexIVFPQ::search` just calls `cpuIndex_->search` (`MetalIndexIVFPQ.mm:225`);
the `.metal` file even has `sq4/sq6_decode_component` helpers (`MetalDistance.metal:1054,1060`)
but **no PQ/SQ scan kernel uses them** (dead scaffolding — note the empty
`BITONIC_MERGE_TWO_SORTED_VARIANT` macro at `:353` too).

## 1. RESOURCE layer

### `MetalResources.h/.mm`
- Owns exactly two things: `id<MTLDevice> device_` and `id<MTLCommandQueue> commandQueue_`
  (`MetalResources.h:74-75`). Mirrors `GpuResources`' role but far thinner.
- Device acquisition: `MTLCreateSystemDefaultDevice()` in the ctor, then one
  `[device newCommandQueue]` (`MetalResources.mm:15-18`). **Single command queue, created once.**
- `MetalAllocType` enum (`MetalResources.h:25-33`) mirrors `faiss::gpu::AllocType` roles
  (FlatData, IVFLists, Quantizer, TemporaryMemoryBuffer, …) but the type is **ignored** by
  the allocator (`allocBuffer(size_t, MetalAllocType /*type*/)`, `.mm:26-28`) — it's a
  future hook, not wired to any pool.
- **Temp-memory / heap management: there is none.** `allocBuffer` is a raw
  `[device newBufferWithLength:size options:MTLResourceStorageModeShared]` (`.mm:36-37`);
  `deallocBuffer` is a no-op relying on ARC (`.mm:40-46`). No pre-reserved temp arena, no
  `MTLHeap`, no sizing default. This is a deliberate simplification vs CUDA's
  `StandardGpuResources` 1.5 GB temp pool. **FAISS leaves a comment** that large GPU-only
  data should switch to `MTLResourceStorageModePrivate` (`.mm:33-35`).
- **Unified / shared memory:** everything is `MTLResourceStorageModeShared` (`.mm:37`) — CPU
  and GPU see the same bytes on Apple Silicon, so "upload" is a `memcpy` into
  `[buf contents]`, never a blit. This is the backbone trick used everywhere (see §5c).
- **Stream/command-buffer model:** no persistent encoder. Each dispatch batch creates a
  fresh `commandBuffer` → `computeCommandEncoder` → `commit` → `waitUntilCompleted`
  (e.g. `MetalFlatKernels.mm:52-64`). `synchronize()` commits an empty command buffer and
  waits (`MetalResources.mm:48-55`) — a full-queue drain.
- **Thread-safety:** `MetalResources` itself is not locked, but the kernel/pipeline cache
  is: `getMetalKernels` guards a static per-device map with a `std::mutex`
  (`MetalKernels.mm:277-286`). Copy/assign deleted (`MetalResources.h:42-43`).

### `StandardMetalResources.h/.mm`
- Trivial: wraps a `std::shared_ptr<MetalResources>` (`.h:31`), constructed eagerly
  (`.mm:14-15`). This is the object Python/cloner hand around. `isAvailable()` forwards.

**Verdict — Resource layer: ADAPT.** Adopt the shared-ptr `MetalResources` shape and the
single-queue model. ELIPS should add the one thing FAISS skipped: a small reusable temp
buffer (see §9b) — but keep FAISS's shared-storage default and no-op dealloc. Skip the
unused `MetalAllocType` plumbing until a pool actually exists (YAGNI).

## 2. INDEX layer

### `MetalIndex` (base) — `MetalIndex.h/.mm`
- `class MetalIndex : public faiss::Index` (`MetalIndex.h:33`). Holds the resources ptr and
  a `MetalIndexConfig` (`.h:53-54`). Ctor forwards `(dims, metric)` to `faiss::Index`, sets
  `metric_arg`, and asserts device==0 and resources available (`MetalIndex.mm:21-29`).
- `MetalIndexConfig` (`MetalIndex.h:22-30`): `device=0`, `useFloat16CoarseQuantizer=false`,
  `indicesOptions=INDICES_64_BIT`, `interleavedLayout=true`. This is the FAISS
  `GpuIndexConfig` analog, trimmed. **Single-GPU only** — `device` must be 0
  (`MetalIndex.mm:25-27`).

### `MetalIndexFlat` — `.h/.mm`
- Stores vectors row-major in one growable `MTLBuffer` (`.h:59`, `capacityVecs_` at `:61`).
- `add` = `ensureCapacity` (2× growth, `.mm:43-45`) then `memcpy` into `[buf contents]`
  (`.mm:67-72`). `add_with_ids` throws — unsupported (`.mm:75-80`).
- **`search` (`.mm:91-167`) is the clean reference path:** assert `k <= getMetalFlatSearchMaxK()`
  (=2048, `.mm:106`); alloc temp query/outDist/outIdx buffers; `memcpy` query in; call
  `runFlatSearchGPU` (distance matrix + GPU top-k, §5); `memcpy` results out; remap int32
  GPU indices → `idx_t`, out-of-range → -1 (`.mm:159-162`). Distances/labels copied straight
  from shared buffers — no blit.
- `copyTo` reads `[buffer contents]` into a host vector and re-adds to a CPU `IndexFlat`
  (`.mm:169-182`).

### `MetalIndexIVFFlat` — `.h` + `.mm` (2308 lines) + `impl/MetalIVFFlat`
- Hybrid: owns a CPU `IndexIVFFlat` for the coarse quantizer + a GPU-resident
  `MetalIVFFlatImpl` for list storage (`.h:139-140`). Coarse quantizer stays CPU; **list
  scan + top-k are GPU.**
- Config knobs are rich: `interleavedLayout_`, `indicesOptions_` (`.h:141-142`), plus many
  persistent, lazily-grown, `mutable` search buffers so `const search()` can resize them
  (`.h:144-167`) — queries, coarse out, per-list dist/idx, centroid buffer, centroid-norms
  buffer, dist-matrix scratch.
- `train` trains CPU index then `uploadCentroids_()` (`.mm:590-597`); `add` assigns on CPU
  (`quantizer->assign`), adds to CPU core, and appends raw vectors to GPU impl
  (`.mm:599-622`).
- **`uploadCentroids_` (`.mm:513-573`)**: reconstructs centroids to host, uploads as float32
  or packed fp16 if `useFloat16CoarseQuantizer` (`:527-545`), and **pre-computes centroid L2
  norms on GPU** via `runMetalComputeNorms` (float32 path only, `:549-569`) — enabling the
  fused `l2_with_norms` coarse step.
- **`search` (`.mm:661`+)**: `k <= 1024` (`kIVFFlatSupportedMaxK`, `.mm:30,671`). Query-tiled
  (`chooseIvfSearchTileRows`, 256 MB default budget `:26`). Two GPU strategies:
  1. **Full GPU** `runMetalIVFFlatFullSearch` (coarse dist-matrix → coarse top-nprobe →
     scan → merge) when `useFullCoarseGpuForIvf()` and `nprobe <= maxK` and matrix fits
     (`.mm:813-886`).
  2. **CPU-coarse + GPU-scan**: CPU `quantizer->search` for assignments, upload as int32,
     then GPU scan+merge (`.mm:888-928`).
  CPU fallback is gated by env `FAISS_METAL_IVF_ALLOW_CPU_FALLBACK` and otherwise **throws**
  (`.mm:747-761`) — i.e. production can force GPU-only.
- Env-tunable everything (`.mm:26-33` constants; `getenv` at `:36,51,109,121,133,145,158,170,182`):
  tile bytes, auto-reserve batch, full-coarse max bytes, exact-candidate budget, force-chunked
  selection. Heavy instrumentation (fallback counts, sync profile).

### `MetalIndexIVFPQ` — `.h/.mm` + `impl/MetalIVFPQ`
- Constrained: **8-bit codes only** (`.mm:60,88`, `verifyPQSettings_:34-37`), `M ∈ [1,64]`
  (`:44-47`), `d % M == 0`, `by_residual` required, no polysemous (`copyFrom:239-243`).
- `train`/`add` run on the CPU `IndexIVFPQ`; codes are PQ-encoded on CPU and appended to a
  GPU buffer (`encodeResidualAndAppend_:130-163`, `impl/MetalIVFPQ.mm:90-170`).
- **`search` is CPU** (`.mm:210-226`): sets nprobe, calls `cpuIndex_->search`. The GPU PQ
  LUT/scan kernel was never written. GPU buffers exist only to support `copyTo`/roundtrip.

**Verdict — Index layer:**
- `MetalIndex` base + `MetalIndexConfig`: **ADOPT** (clean GpuIndex analog, single-GPU).
- `MetalIndexFlat`: **ADOPT** its search skeleton verbatim — it is exactly ELIPS's missing
  "distance-on-GPU + top-k-on-GPU" wiring.
- `MetalIndexIVFFlat` full/scan machinery: **ADAPT**, but skip the env-var zoo and chunked
  experimental path initially (keep full-GPU + CPU-coarse-GPU-scan + one fallback flag).
- `MetalIndexIVFPQ`: **SKIP for now** — it's a CPU stub; no GPU precedent to copy. Revisit
  only when ELIPS needs PQ, and expect to write the scan kernel yourself.

## 3. CLONER — `MetalCloner.h/.mm`

- `get_num_gpus()` returns 1 if a Metal device exists else 0 (`.mm:23-26`).
- `index_cpu_to_metal_gpu` (`.mm:28-77`): `dynamic_cast` ladder IVFPQ → IVFFlat → Flat
  (most-derived first), each checked for L2/IP only, constructs the matching Metal index.
  For Flat it `add`s the CPU vectors (`.mm:67-70`). `device` must be 0 (`.mm:35`).
- `index_metal_gpu_to_cpu` (`.mm:79-121`): reverse ladder; builds a fresh CPU quantizer
  (`IndexFlatL2`/`IndexFlatIP`) and `copyTo`s.
- **Unified-memory exploitation:** the cloner itself does no special trick — the win is that
  because storage is `MTLResourceStorageModeShared`, `copyTo`/`add` are plain host
  `memcpy`s off `[buffer contents]` (`MetalIndexFlat.mm:176-181`), with **zero device↔host
  blit and no staging buffer**. On Apple Silicon the "transfer" is a pointer read.

**Verdict — Cloner: ADAPT.** Adopt the `dynamic_cast` ladder + factory pattern and the
shared-memory zero-copy transfer. ELIPS only needs the Flat (and later IVFFlat) arms; skip
the IVFPQ arm until §2's IVFPQ is real.

---

## 4. KERNELS — `MetalDistance.metal` (1079 lines) + dispatch (`.mm` files)

Dispatch wrappers live in `MetalKernels.mm` (encode + pipeline cache) and
`MetalDistance.mm` (IVF orchestration). The shader is pre-compiled to a `.metallib`
loaded at runtime (§6). `MetalKernels` caches `MTLComputePipelineState` per function name
in an `unordered_map` (`MetalKernels.mm:101-116`), one `MetalKernels` per device behind a
mutex (`:277-286`).

### 4.1 Distance kernels
- **`l2_squared_matrix` (`MetalDistance.metal:18-76`)** and **`ip_matrix` (`:78-134`)**:
  hand-written **tiled GEMM**, *not* MPS. 32×32 output tile, TILE_K=16, 256 threads, each
  thread computes a 2×2 micro-tile (`acc00..acc11`) with threadgroup-staged Q/V tiles and
  two barriers per k-step (`:55-67`). Dispatched 16×16 threads over `ceil(nb/32)×ceil(nq/32)`
  groups (`MetalKernels.mm:138-145`). L2 accumulates `(q-v)²`; IP accumulates `q*v`.
- **`compute_norms` (`:355-374`)**: one thread per base vector, `float4`-vectorised
  `dot(v4,v4)` (`:366-369`) + scalar tail. Used to precompute centroid ‖c‖².
- **`l2_with_norms` (`:379-415`)**: fused L2 = ‖q‖² + ‖v‖² − 2·⟨q,v⟩ using **precomputed**
  vecNorms and an inline query-norm, `float4` loads (`:395-412`). This is the fast coarse
  path when centroid norms are cached. **No fp16 compute path exists** — fp16 is storage-only
  for centroids (`centroidsAreFP16` is accepted then ignored, `MetalDistance.mm:237`).

### 4.2 GPU top-k — `topk_threadgroup_<K>` (THE kernel ELIPS lacks)
Defined by macro `TOPK_THREADGROUP_VARIANT(K)` (`MetalDistance.metal:141-235`) instantiated
for K=32,64,128,256,512,1024 (`:237-242`), plus a hand-written `topk_threadgroup_2048`
(`:244-347`). `MetalKernels::kMaxK = 2048` (`MetalKernels.h:30`).

How it works (per query = one threadgroup of **256 threads**):
1. **Register pre-reduction.** Each thread strides over the `nb` distances
   (`for j=tid; j<nb; j+=256`, `:165`) keeping its best `R` locally via insertion sort
   (`R=4` for K≤1024 → 1024 candidates; `R=16` for K=2048 → 4096 candidates, `:150-152,253-254`).
2. **Spill to threadgroup memory** `tgDist[CANDIDATES]/tgIdx[]` (`:192-201`), padding unused
   slots with `±1e38` sentinels.
3. **Full bitonic sort** over all CANDIDATES in threadgroup memory
   (`for k2=2..CAND; for j=k2>>1..1; idx^j partner compare-exchange`, `:204-225`), one
   barrier per inner stage. `want_min` picks ascending (L2) vs descending (IP).
4. **Write** the first `K_out = min(K, min(k,nb))` sorted entries, pad the rest with
   sentinel/-1 (`:227-234`).

**Tie handling:** deterministic — on equal distance the **smaller index wins**
(`tgIdx[partner] < tgIdx[idx]` tiebreak baked into every compare, `:211-215`). This is what
makes GPU results bit-identical to CPU `std::partial_sort` in tests.

**Max K / variant selection:** `selectTopKVariantIndex(k)` picks the smallest variant ≥ k
(`MetalKernels.mm:40-46`); `encodeTopKThreadgroup` always dispatches `nq` groups × 256
threads (`:206-207`). So K is a **compile-time threadgroup-memory size**, chosen at dispatch.
No simdgroup-reduction top-k — it's threadgroup-memory bitonic.

### 4.3 IVF scan kernels (two-pass, mirrors CUDA IVF)
Design comment at `MetalDistance.metal:417-435`. Shared params `[nq,d,k,nprobe,want_min]`.

- **`ivf_scan_list` (`:455-618`)** — standard variant, 256 threads, grid = `nq*nprobe`
  threadgroups (one list per query-probe, `MetalKernels.mm:254-256`). Caches the query vector
  in threadgroup memory `tgQuery[2048]` (`:505-510`), reused across all list vectors.
  `float4` L2/IP distance (`:530-549`). Each thread keeps `LOCAL_K=4` best, then the
  threadgroup bitonic-sorts `CAND=1024` candidates (`:573-605`), writing per-list top-k and
  **resolving 32-bit vecIdx → 64-bit id only at write** (`:607-613`) — a nice memory trick.
- **`ivf_scan_list_small` (`:623-758`)** — 32-thread / `LOCAL_K=1` / 32-candidate variant for
  lists ≤32 (`MetalDistance.metal:620-622`); saves ~90% of barriers & threadgroup memory.
  Chosen when `k<=32` and all lists fit (`MetalDistance.mm:263-268`).
- **`ivf_scan_list_interleaved` (`:764-935`)** — operates on **32-vector blocks with
  dimensions interleaved** (`[v0d0 v1d0 … v31d0][v0d1 …]`). Each **simdgroup (32 lanes)**
  processes one block; all lanes read the same dimension at once → fully **coalesced**
  reads (`:835-866`). Requires `d%4==0` (`MetalDistance.mm:144,248`). This is the fast layout
  (`interleavedLayout=true` by default) and mirrors CUDA's interleaved code store.
- **`ivf_merge_lists` (`:941-1040`)** — pass 2, grid = `nq` groups. Strided scan over all
  `nprobe*k` per-list candidates keeping `LOCAL_K=4` in registers, then one bitonic sort over
  1024 to produce the final per-query top-k (`:973-1039`). Handles arbitrary `nprobe*k`
  without overflow by the register-prefilter.
- **Exactness guards** (`MetalDistance.mm:20-83,146-150,251-255`): the GPU path only runs when
  `nprobe*k <= 1024` (`kIvfReduceExactCandidates`) and every scanned list length fits the
  candidate budget — otherwise it returns false and the index falls back to CPU-coarse or
  CPU search. This is how FAISS keeps the fixed-size bitonic networks *exact*.
- `ivf_better_int/long` (`:437-453`) centralise the compare with a `tie_eps=1e-6` and
  index-ascending tiebreak.

### 4.4 Observations for ELIPS
- GEMM is **hand-written tiled**, not MPS/`MPSMatrixMultiplication`. Simpler to ship, no MPS
  dependency, and keeps distances in one kernel. fp16 is storage-only; all math is fp32.
- `float4` vectorised loads everywhere (`compute_norms`, `l2_with_norms`, every scan).
- The **only** simdgroup use is the interleaved scan's 32-lane block coordination; top-k and
  standard scan are plain threadgroup-memory bitonic. No `simd_shuffle`/`simd_min` reductions.

**Verdict — Kernels:**
- `topk_threadgroup_*`: **ADOPT wholesale** — this is the headline deliverable (see §9a).
- `l2_squared_matrix`/`ip_matrix` tiled GEMM: **ADOPT** (replaces whatever ELIPS does for the
  distance matrix; no MPS dep).
- `compute_norms` + `l2_with_norms` fused coarse path: **ADAPT** (only needed once ELIPS has
  IVF; adopt the precompute-norms idea).
- IVF scan trio + merge + exactness guards: **ADAPT** for ELIPS Phase 7 IVF; start with
  `ivf_scan_list` (standard) + `ivf_merge_lists`, add interleaved later for throughput.
- SQ4/SQ6 decode helpers + empty merge macro: **SKIP** (dead code even in FAISS).

## 5. BUILD chain

### Top-level language seeding — `/tmp/faiss/CMakeLists.txt:36-61`
- `if(FAISS_ENABLE_METAL) list(APPEND FAISS_LANGUAGES OBJCXX)` (`:36-38`).
- Because `option()` runs after `project()`, it **pre-seeds the cache before `project()`**
  (`:48-61`) and auto-enables Metal **only on `Darwin` + `arm64`** (`:56`), preferring
  `CMAKE_SYSTEM_*` (toolchain/cross) over `CMAKE_HOST_*` (`:49-55`). Intel Macs excluded.

### `gpu_metal/CMakeLists.txt`
- Static lib `faiss_metal` from the `.mm` sources (`:11-27`); links `-framework Metal MetalKit
  MetalPerformanceShaders Foundation` (`:29-37`). `OBJCXX_STANDARD 17` (`:51-54`).
- **metallib build (`:56-81`)** — the part to copy:
  1. `find_program(XCRUN xcrun REQUIRED)`.
  2. `xcrun -sdk macosx metal -c MetalDistance.metal -o MetalDistance.air` (`:64`).
  3. `xcrun -sdk macosx metallib MetalDistance.air -o MetalDistance.metallib` (`:71`).
  4. `add_custom_target(metal_shaders)` + `add_dependencies(faiss_metal metal_shaders)` (`:76-77`).
  5. Compiled-in path `-DFAISS_METALLIB_BUILD_PATH="<buildpath>"` (`:79-81`).
- Install ships the `.metallib` next to the lib (`:113-115`); headers installed under
  `faiss/gpu_metal` (`:90-111`).
- **Runtime discovery** — `MetalKernels.mm:54-81` `findMetalLibPath()`: (1) `FAISS_METALLIB_PATH`
  env override, (2) compiled-in build path, (3) `dladdr` next to the loaded dylib (for
  pip/conda wheels). `newLibraryWithURL` loads it (`:90-93`). Missing → throws (`:86-89`).

**Verdict — Build: ADOPT the metallib chain exactly** (see §9d). ELIPS already has a `.metal`
file; wiring the `xcrun metal → .air → metallib` custom commands + the 3-tier runtime
discovery is precisely what turns ELIPS's dead shader into a loadable library. Drop the
`MetalPerformanceShaders` link (FAISS links it but the code never calls MPS).

---

## 6. PYTHON — `MetalPythonBridge.h/.mm`
- **SWIG-safe C++-only header** (no ObjC types) so SWIG can parse it (`.h:8-9`). Exposes the
  *same symbol names* as the CUDA GPU API — `get_num_gpus`, `index_cpu_to_gpu`,
  `index_gpu_to_cpu`, `gpu_profiler_start/stop`, `gpu_sync_all_devices`
  (`.h:30-42`) — so `faiss.index_cpu_to_gpu(res, 0, idx)` works unchanged in Python.
- `StandardMetalResourcesHolder` is an **opaque `void* impl`** struct (`.h:20-27`) hiding the
  ObjC++ `StandardMetalResources` from SWIG; ctor/dtor new/delete it (`.mm:26-33`).
- `index_cpu_to_gpu` just unwraps and forwards to `index_cpu_to_metal_gpu` (`.mm:63-72`).
  Profiler calls are depth counters gated on `FAISS_METAL_PROFILE_LOG` (`.mm:35-54`);
  `gpu_sync_all_devices` drains the queue (`.mm:56-61`).

**Verdict — Python: ADAPT.** The opaque-holder + name-parity trick is the clean way to expose
an ObjC++ backend through a C++/SWIG (or pybind) layer. ELIPS should mirror the opaque-void*
holder so its Python bindings never see Metal types.

## 7. TESTS — parity assertions & tolerances

### C++ gtest (`test/*.mm`)
- `TestMetalIndexFlat.mm`: `kTolerance = 1e-5f` (`:25`); asserts GPU vs CPU **distances
  `EXPECT_NEAR` and labels `EXPECT_EQ`** (`:35-40`) — i.e. **exact index parity**. Covers
  L2, IP, add_with_ids-throws, reset, empty search, `get_num_gpus ∈ [0,1]`, cloner roundtrip.
  All skip gracefully if no Metal device (`:49-51`).
- `TestMetalIndexIVFFlat.mm`: `kTolerance = 1e-4f` (`:25`) — looser than Flat; same
  distance-near + label-equal parity vs CPU IVFFlat (nlist=8, nprobe=4, k=5), reset, roundtrip.
- `TestMetalIndexIVFPQ.mm`: **recall-based, not exact** — `EXPECT_GT(matches, nq*k/2)` (`:39`):
  only requires >50% of GPU neighbours to appear in the CPU top-k set. Reflects that IVFPQ
  runs on CPU and/or is lossy; no distance tolerance asserted.

### Python (`test/test_metal_python.py`)
- Flat L2/IP: `assert_allclose(rtol=1e-5, atol=1e-5)` on D + `assert_array_equal` on I
  (`:46-47,65-66`). IVFFlat: `rtol=atol=1e-4` + exact I (`:112-113`). IVFPQ: same >nq*k//2
  recall check (`:208-212`). Roundtrip tests confirm `ntotal` and CPU-back search parity.

**Takeaway:** FAISS holds **Flat/IVFFlat to exact label parity + tight distance tolerance**
against CPU (achievable precisely because the GPU top-k tiebreak = smaller-index, §4.2), and
drops IVFPQ to a recall floor. **Verdict — Tests: ADOPT this exact-parity bar for ELIPS Flat
top-k** — it's the regression gate proving the GPU top-k matches the retired `std::partial_sort`.

---

## 8. Consolidated adopt / adapt / skip

| Area | Verdict | One-line rationale |
|------|---------|--------------------|
| `MetalResources` single-queue + shared ptr | ADOPT | Minimal, correct GpuResources analog |
| Shared-storage buffers + no-op dealloc | ADOPT | Unified memory = zero-copy; ARC frees |
| Reusable temp-memory arena | ADAPT (add) | FAISS skipped it; ELIPS should add a tiny one (§9b) |
| `MetalAllocType` pool plumbing | SKIP | Unused in FAISS; YAGNI |
| `MetalIndex` base + `MetalIndexConfig` | ADOPT | Clean single-GPU GpuIndex analog |
| `MetalIndexFlat::search` skeleton | ADOPT | Exactly ELIPS's missing GPU dist+topk wiring |
| `MetalIndexIVFFlat` full/scan machinery | ADAPT | Keep 2 GPU paths + 1 fallback; drop env zoo |
| `MetalIndexIVFPQ` | SKIP | CPU stub, no GPU precedent |
| `MetalCloner` dynamic_cast ladder | ADAPT | Flat/IVFFlat arms only |
| `topk_threadgroup_*` | **ADOPT** | The GPU top-k ELIPS needs (§9a) |
| `l2_squared_matrix`/`ip_matrix` tiled GEMM | ADOPT | No MPS dependency, one kernel |
| `compute_norms` + `l2_with_norms` | ADAPT | For IVF coarse step later |
| IVF scan trio + merge + exactness guards | ADAPT | Phase 7 IVF; start standard+merge |
| SQ4/SQ6 helpers, empty merge macro | SKIP | Dead code even in FAISS |
| metallib `xcrun` build + 3-tier discovery | **ADOPT** | Turns ELIPS's dead shader loadable (§9d) |
| `MetalPerformanceShaders` link | SKIP | Linked but never called |
| Python opaque-holder + name parity | ADAPT | Hide ObjC types from SWIG/pybind |
| Exact-parity test bar (Flat/IVFFlat) | ADOPT | Regression gate vs retired CPU top-k |

## 9. The four specific call-outs

### (a) Moving ELIPS top-k onto the GPU via `topk_threadgroup`
Today ELIPS computes distances on GPU then copies the full `nq×nb` distance matrix back and
runs `std::partial_sort` per query on CPU (`MetalBackend.mm:249-257`). Replace with:
1. Keep the distances in a shared `MTLBuffer` (no copy-back).
2. Port `topk_threadgroup_<K>` (`MetalDistance.metal:141-235`) into ELIPS's `.metal`. One
   threadgroup (256 threads) per query; each thread keeps `R` best in registers over a strided
   pass, spill to threadgroup memory, bitonic-sort, write top-k.
3. Dispatch `nq` groups × 256 threads (`MetalKernels.mm:206-207`); pick the variant with
   `selectTopKVariantIndex(k)` (`:40-46`). Pre-compile all variants into the metallib.
4. **Reproduce the smaller-index tiebreak** (`:211-215`) so results match the old
   `partial_sort` exactly — gate it with FAISS's exact-parity test (§7).
This single kernel + the `MetalIndexFlat::search` wiring (`MetalIndexFlat.mm:91-167`) deletes
the CPU top-k and the matrix copy-back in one move. ELIPS's current `.metal` stops being dead.

### (b) Temp-memory default to copy
FAISS's own default is **"no arena at all"** — `allocBuffer` is a bare
`newBufferWithLength:options:MTLResourceStorageModeShared` per call (`MetalResources.mm:36-37`).
The *reusable* default worth copying is in the **index**, not the resources: IVFFlat keeps
persistent search scratch buffers grown **2× on demand** and never freed between searches
(`ensureSearchBuf_`, `MetalIndexIVFFlat.mm:575-588`; `MetalIndexFlat` grows its vector buffer
the same way, `MetalIndexFlat.mm:43-45`). For a tiling budget default, FAISS uses
`kDefaultIvfQueryTileBudgetBytes = 256 MiB` (`MetalIndexIVFFlat.mm:26`). **Copy these two
numbers: 2× growth, 256 MiB query-tile budget.** Skip CUDA's 1.5 GB temp pool — unified memory
makes it pointless.

### (c) The unified-memory cloner trick
There is no DMA. Because every buffer is `MTLResourceStorageModeShared`, CPU↔GPU "transfer" is
reading `[buffer contents]` as a host pointer: `MetalIndexFlat::copyTo` memcpys straight from
`[vectorsBuffer_ contents]` (`MetalIndexFlat.mm:176-181`), and `add` memcpys straight in
(`:67-72`). The cloner (`MetalCloner.mm:28-121`) just constructs the Metal index and calls
`add`/`copyTo`. ELIPS on M4 should do the same: **allocate shared, treat upload/download as
`memcpy` on `[buf contents]`, never a blit encoder.** This is the biggest simplification Metal
buys over CUDA.

### (d) The metallib build chain to copy
From `gpu_metal/CMakeLists.txt:56-81` + runtime loader `MetalKernels.mm:54-93`:
```cmake
find_program(XCRUN xcrun REQUIRED)
add_custom_command(OUTPUT x.air  COMMAND ${XCRUN} -sdk macosx metal    -c x.metal -o x.air)
add_custom_command(OUTPUT x.metallib COMMAND ${XCRUN} -sdk macosx metallib x.air -o x.metallib)
add_custom_target(shaders DEPENDS x.metallib)
add_dependencies(<lib> shaders)
target_compile_definitions(<lib> PRIVATE ELIPS_METALLIB_BUILD_PATH="${x.metallib}")
install(FILES x.metallib DESTINATION ${CMAKE_INSTALL_LIBDIR})
```
Plus the 3-tier runtime discovery (`env override → compiled-in build path → dladdr next to the
dylib`) so dev builds and installed wheels both find the `.metallib`. Install it beside the
library. This is the exact recipe that makes ELIPS's `elips_distance_kernels.metal` compile to
a loadable `.metallib` instead of being ignored.

---
*End Phase 3I. Source: `/tmp/faiss/faiss/gpu_metal/` (FAISS v1.15.1, `/tmp/faiss/CMakeLists.txt:77`).*







