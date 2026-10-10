# 3D — FAISS GPU Architecture (the layering ELIPS mirrors)

Studied: **FAISS v1.15.1** @ `/tmp/faiss` (read-only, 2026-10-10). Every class/field verified against source; `file:line` throughout. Paths are relative to `/tmp/faiss` unless absolute.

**Why this study matters.** ELIPS copies FAISS's GPU *layering* but implements **Metal first** — FAISS has no Metal backend (CUDA / ROCm / cuVS only). The value here is the architecture: the clean split of RESOURCE / CONFIG / INDEX / CLONER / STANDALONE / KERNEL, the paging strategy, and the CPU-interchangeability contract. Each layer below ends with an **adopt / adapt / skip** verdict for ELIPS-on-Metal.

ELIPS context (from `_phase1_gpu.md`): Metal is the only real backend; ports are `GpuPort` (backend), `GpuMemoryPort` (`GpuMemoryManager`), `GpuIndexPort` (host-side index orchestrators), plus orphan `GpuKernelPort`/`GpuStreamPort`. The FAISS mapping below is written against that structure.

## Corrections to prompt assumptions (reality wins)

- **Temp memory is NOT a fraction of device memory.** It is a flat tiered cap: ≤4 GiB GPU → 512 MiB, ≤8 GiB → 1 GiB, else **1.5 GiB** hard cap (`StandardGpuResources.cpp:180-203`). The prompt's "fraction/formula" guess is wrong — it only ever lowers a request to the tier cap.
- **Max-k is 2048, not a universal constant.** `GPU_MAX_SELECTION_K = 2048` on CUDA > 9.0, **1024** on CUDA ≤ 9.0, 2048 on ROCm (`DeviceDefs.cuh:39,65,67`). cuVS raises it to **16384** (`IndexUtils.cu:21-25`) — but the native GEMM+select path stays bound to 2048 (`Distance.cu:205` hardcodes `getMaxKSelection(false)`).
- **No mutexes anywhere in the resource layer** — concurrency is handled by CUDA stream ordering, not CPU locks.
- **`bfKnnBatch` does not exist** — the batching API is `bfKnn_tiling` (`GpuDistance.cu:516`).
- **`index_cpu_to_all_gpus` / `index_cpu_to_gpus_list` are Python-only** wrappers (`faiss/python/gpu_wrappers.py:38,43`), not C++ symbols. Likewise `get_num_gpus` is a SWIG wrapper over C++ `getNumDevices()`.
- A third vector dtype exists: **BF16** (`DistanceDataType::BF16`, `GpuDistance.h:19-23`), beyond F32/F16.
- `GpuIndexConfig` has only **3** fields (`device`, `memorySpace`, `use_cuvs`) — `GpuIndex.h:32-46`.
- `storeTransposed` is **deprecated / dead** in both `GpuIndexFlatConfig` (`GpuIndexFlat.h:30-33`) and `GpuClonerOptions` (declared, never read).

---

# Layer 1 — RESOURCE (`GpuResources` / `StandardGpuResources`)

The resource layer owns everything device-global: streams, the cuBLAS handle, pinned host memory, the temp-memory stack, and all allocations. Indexes borrow it; they never allocate raw device memory themselves.

## Abstract interface — `GpuResources` (`GpuResources.h:200-253`)

Pure virtuals every backend must implement: `initializeForDevice(int)` (:207), `supportsBFloat16(int)` (:210), `getBlasHandle(int)` (:213), `getDefaultStream(int)` (:217), `setDefaultStream(int, cudaStream_t)` (:229), `getAlternateStreams(int)` (:232), `allocMemory(const AllocRequest&)` (:240), `deallocMemory(int, void*)` (:243), `getTempMemoryAvailable(int) const` (:247), `getPinnedMemory()` → `pair<void*,size_t>` (:250), `getAsyncCopyStream(int)` (:253). (`getRaftHandle` :222 only under `USE_NVIDIA_CUVS`.) Base provides `*CurrentDevice()` conveniences + `syncDefaultStream` + `allocMemoryHandle` (RAII) at :259-286.

**Alloc taxonomy** (`GpuResources.h:45-169`):
- `AllocType`: `Other`, `FlatData`, `IVFLists`, `Quantizer`, `QuantizerPrecomputedCodes`, `TemporaryMemoryBuffer` (=10), `TemporaryMemoryOverflow` (=11) (:45-83). The split lets the allocator tell "the up-front temp stack" from "temp that spilled to cudaMalloc".
- `MemorySpace`: `Temporary=0`, `Device=1`, `Unified=2` (:89-101).
- `AllocInfo{ type, device, space, stream }` (:107-134); `AllocRequest : AllocInfo { size_t size }` (:146-169); factory helpers `makeDevAlloc/makeTempAlloc/makeSpaceAlloc` (:137-143).
- `GpuMemoryReservation` — move-only RAII handle, dtor calls `res->deallocMemory` (:172-196).

## Pimpl + shared ownership (`StandardGpuResources.h`)

`StandardGpuResourcesImpl : public GpuResources` (:46) is the real impl; `StandardGpuResources : public GpuResourcesProvider` (:199) is the user-facing wrapper holding `std::shared_ptr<StandardGpuResourcesImpl> res_` (:265) and forwarding every call (`.cpp:706-767`). `getResources()` hands out the *same* shared_ptr, so **all indexes built from one `StandardGpuResources` share one impl**: one temp stack, one cuBLAS handle, one stream set per device.

**Per-device state** — all `std::unordered_map<int, …>` keyed by device (`StandardGpuResources.h:135-157`): `allocs_` (outstanding allocations: `unordered_map<void*, AllocRequest>`), `tempMemory_` (`unique_ptr<StackDeviceMemory>`), `defaultStreams_`, `userDefaultStreams_`, `alternateStreams_` (`vector<cudaStream_t>`), `asyncCopyStreams_`, `blasHandles_`. Scalars: `tempMemSize_`, `pinnedMemSize_`, `allocLogging_`.

**Thread-safety: NONE at this layer.** No mutexes; maps are mutated unguarded in `initializeForDevice`/`allocMemory`/`deallocMemory`. `isInitialized` uses presence in `defaultStreams_` as the init flag (`.cpp:331-335`). Concurrency correctness comes from **CUDA stream ordering** (cross-stream `streamWait`), not locks.

## Temp-memory manager — the real default (`StandardGpuResources.cpp`)

Constants (`:45-58`): `kDefaultPinnedMemoryAllocation = 256 MiB`, `k4GiBTempMem = 512 MiB`, `k8GiBTempMem = 1 GiB`, `kMaxTempMem = 1.5 GiB`, `kNumStreams = 2`.

`getDefaultTempMemForGPU(device, requested)` (`:180-207`), verified by direct read:
```
totalMem = device==-1 ? SIZE_MAX : props.totalGlobalMem
if   totalMem <= 4 GiB:  cap = 512 MiB
elif totalMem <= 8 GiB:  cap = 1 GiB
else:                    cap = 1.5 GiB
return min(requested, cap)
```
Ctor seeds `tempMemSize_ = getDefaultTempMemForGPU(-1, SIZE_MAX)` = **1.5 GiB** provisional (`.cpp:102`); the real per-device clamp happens in `initializeForDevice` (`.cpp:449-453`). `setTempMemory`/`noTempMemory` rebuild the stack (`.cpp:216-242`).

**`StackDeviceMemory`** (`utils/StackDeviceMemory.cpp`) — a bump allocator over one up-front buffer: `getAlloc` carves `[head_, head_+size)`, issues cross-stream `streamWait` when a prior user of that range ran on a different stream (`:99-102`), tracks `highWaterMemoryUsed_`; `returnAlloc` is strict **LIFO** (`FAISS_ASSERT(p+size==head_)`, `:143-145`). Sizes rounded to 16 bytes.

**Overflow:** in `allocMemory`, a `Temporary` request bigger than current stack availability is re-tagged `MemorySpace::Device` + `TemporaryMemoryOverflow` and recurses → a plain `cudaMalloc` (`.cpp:525-545`). Every request is `roundUp(size, 256)` to match cudaMalloc alignment (`:518-521`).

## Streams, pinned, cuBLAS

- Default stream per device created `cudaStreamNonBlocking` (`.cpp:404-408`); 2 alternate streams (`kNumStreams`) also non-blocking (`:420-428`); separate async-copy stream (`:414-418`). User override via `setDefaultStream` issues a `streamWait` so the new stream waits on prior work (`:269-271`); object does not own user streams.
- Pinned host memory: default 256 MiB, lazily `cudaHostAlloc` on first device init (`.cpp:361-362`); `getPinnedMemory()` returns `{ptr,size}` (`:501-503`); must be sized *before* any device init.
- cuBLAS handle per device (`.cpp:430-434`); on CUDA ≥ 11 sets `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` (`:436-443`) — TF32/reduced-precision accumulation **disallowed** to protect numerics.

## Verdict — RESOURCE: **ADAPT**

Adopt the *shape* wholesale: one resources object shared by all indexes, per-device state maps, a temp-memory stack with high-water tracking, LIFO bump allocation, and stream-ordered (lock-free) concurrency. This is ELIPS's `GpuMemoryManager` + `GpuPort` territory.

Metal-specific redesign:
- **MTLHeap replaces the StackDeviceMemory buffer.** A `MTLHeap` with `MTLHeapTypePlacement` gives you the same up-front-reservation + sub-allocation model; `makeAliasable` + a bump pointer reproduces the LIFO stack cheaply.
- **Unified memory is the default on Apple Silicon, not an opt-in oversubscription mode.** FAISS's `MemorySpace::Unified` (CC 6+ oversubscribe) becomes the *normal* case — host and device share the buffer, so FAISS's upload/paging machinery (Layer 7) is largely unnecessary for resident data. Keep `MemorySpace` as an enum but `Shared` is the Metal default (`MTLResourceStorageModeShared`); `Private` for GPU-only.
- **Streams → `MTLCommandQueue` + `MTLCommandBuffer`.** `kNumStreams=2` alternate streams map to multiple command buffers for overlap. Stream-ordered `streamWait` → `MTLSharedEvent` / `encodeWait`.
- **cuBLAS handle → nothing to hold** (MPS has no persistent handle); the "per-device BLAS handle" slot becomes a per-device `MPSMatrixMultiplication` cache or just the `MTLDevice`.
- **Temp-memory default: re-tune the tiers.** The 512 MiB/1 GiB/1.5 GiB caps are CUDA-era guesses; on a 16-64 GiB unified-memory Mac the cap should scale differently (leave this a **calibration knob**, not a hardcoded constant). Keep the "cap, never a fraction" principle.
- **Pinned memory is a no-op on unified memory** — `getPinnedMemory()` can return the shared buffer directly; the triple-buffered pinned pipeline (Layer 7) collapses.

---

# Layer 2 — CONFIG (`GpuIndexConfig` + per-index configs)

Configs are plain structs passed by value into index constructors. Inheritance mirrors the index hierarchy.

**`GpuIndexConfig`** (`GpuIndex.h:32-46`) — base, only 3 fields:
| Field | Type | Default |
|---|---|---|
| `device` | `int` | `0` |
| `memorySpace` | `MemorySpace` | `MemorySpace::Device` |
| `use_cuvs` | `bool` | `false` |

`should_use_cuvs(config)` (`GpuIndex.cu:46-53`) returns `use_cuvs` only if device CC ≥ 7.0, else false.

**`IndicesOptions`** (`GpuIndicesOptions.h:14-28`) — note the type is `IndicesOptions`, not `GpuIndicesOptions`:
| Enumerator | Val | Meaning |
|---|---|---|
| `INDICES_CPU` | 0 | indices only on CPU; GPU returns (list,offset), CPU translates |
| `INDICES_IVF` | 1 | not stored at all; user gets (list,offset) as the id |
| `INDICES_32_BIT` | 2 | stored 32-bit on GPU, returned as 64-bit |
| `INDICES_64_BIT` | 3 | stored 64-bit on GPU (default) |

**`GpuIndexFlatConfig`** (`GpuIndexFlat.h:26-34`): `useFloat16=false` (`ALIGNED(8)`); `storeTransposed=false` **deprecated/unused**.

**`GpuIndexIVFConfig`** (`GpuIndexIVF.h:23-35`): `indicesOptions=INDICES_64_BIT`; `flatConfig` (`GpuIndexFlatConfig`, for the coarse quantizer); `allowCpuCoarseQuantizer=false`.

**`GpuIndexIVFFlatConfig`** (`GpuIndexIVFFlat.h:25-29`): `interleavedLayout=true` (default on).
**`GpuIndexIVFScalarQuantizerConfig`** (`GpuIndexIVFScalarQuantizer.h:20-24`): `interleavedLayout=true`.
**`GpuIndexIVFPQConfig`** (`GpuIndexIVFPQ.h:25-49`):
| Field | Default | Notes |
|---|---|---|
| `useFloat16LookupTables` | `false` | **required** when `subQuantizers · 2^bitsPerCode > 16384` |
| `usePrecomputedTables` | `false` | big memory cost |
| `interleavedLayout` | `false` | under development, **cuVS-only** (note: default differs from IVFFlat/SQ) |
| `useMMCodeDistance` | `false` | GEMM PQ code distance; auto-on for odd dims-per-subquantizer |

**CAGRA/cuVS configs** (`GpuIndexCagra.h`) — graph ANN, cuVS-only: `GpuIndexCagraConfig` (`:206-244`, `graph_degree=64`, `intermediate_graph_degree=128`, `build_algo=IVF_PQ`, nested `IVFPQBuildCagraConfig`/`IVFPQSearchCagraConfig` shared_ptrs, multi-GPU `devices` + `AllNeighborsCagraConfig`); `SearchParametersCagra` (`:257-302`, `itopk_size=64` is the main accuracy knob, `search_algo=AUTO`). Enums `graph_build_algo`/`codebook_gen`/`search_algo`/`hash_mode` at `:44-59,246-255`.

## Verdict — CONFIG: **ADOPT** (structurally)

The by-value POD-struct-per-index-type pattern with inheritance is clean and maps directly to ELIPS's `GpuConfig`. Adopt `IndicesOptions` almost verbatim — the CPU/IVF/32/64-bit storage choice is backend-agnostic and directly relevant (32-bit index storage halves index-id memory on-device). Adopt `useFloat16`, `usePrecomputedTables`, `interleavedLayout` as concepts.

Adapt: drop `use_cuvs` (replace with an ELIPS `backend` selector if ever needed); `memorySpace` default becomes `Shared` on Metal; drop the dead `storeTransposed`. CAGRA config → **skip for now** (graph ANN is a later phase; keep the struct shape as a reference when ELIPS adds a graph index).

---

# Layer 3 — INDEX (`GpuIndex` and subclasses)

## The interchangeability contract

`class GpuIndex : public faiss::Index` (`GpuIndex.h:52`) — **every GPU index IS-A `faiss::Index`**, so it is drop-in for any CPU `Index` consumer. This is the single most important architectural property to copy.

Hierarchy: `GpuIndexFlat : GpuIndex` (`GpuIndexFlat.h:39`), then `GpuIndexFlatL2`/`GpuIndexFlatIP`. `GpuIndexIVF : public GpuIndex, public IndexIVFInterface` (`GpuIndexIVF.h:40`) — deliberately **does NOT inherit `IndexIVF`** because "many of the public data members and functionality in IndexIVF [are] not supported in the same manner on the GPU" (`:37-39`). Then `GpuIndexIVFFlat`/`GpuIndexIVFPQ`/`GpuIndexIVFScalarQuantizer : GpuIndexIVF`; `GpuIndexCagra : GpuIndex`.

Overridden `Index` virtuals on the base (`GpuIndex.h`): `add`/`add_ex` (:78-79), `add_with_ids`/`_ex` (:84-85), `assign` (:93), `search`/`search_ex` (:98-105), `search_and_reconstruct` (:116), `compute_residual`/`_n` (:127-132). Subclass contract = 3 pure virtuals: `addImplRequiresIDs_()` (:161), `addImpl_()` (:165), `searchImpl_()` (:208). Protected state: `shared_ptr<GpuResources> resources_` (:289), `const GpuIndexConfig config_` (:292), `size_t minPagedSize_` (:295).

`isGpuIndexImplemented` (`GpuIndex.cu:819-834`): only `IndexFlat`, `IndexIVFFlat`, `IndexIVFPQ`, `IndexIVFScalarQuantizer` have GPU equivalents.

## How "unsupported on GPU" is reported — exceptions, not silent fallback

All via `FAISS_THROW_*` macros (→ `FaissException`), checked eagerly:
- **k / nprobe limits** (`impl/IndexUtils.cu`): `validateKSelect` throws `"GPU index only supports min/max-K selection up to %d (requested %d)"` for `k ∉ (0, maxK]` (`:28-34`); `validateNProbe` reuses the same cap (`:36-42`). Called from `search_ex` (`GpuIndex.cu:384`).
- **Metric**: GPU IVF supports only `METRIC_L2` / `METRIC_INNER_PRODUCT`, else `FAISS_THROW_FMT("unsupported metric type %d")` (`GpuIndexIVF.cu:34-38`).
- **Numeric type**: add/search accept Float32 / Float16 / Int8 only; anything else → `"... unsupported numeric type"` (`GpuIndex.cu:228-236,332-340`).
- `GpuIndexFlat`: `add_with_ids` → `"add_with_ids not supported"` (no id storage) (`GpuIndexFlat.cu:210`); `reconstruct`/`reconstruct_n` bounds-checked (`:247-282`); `train` is a no-op.
- `GpuIndexIVF`: `range_search` → `"range search not implemented"` (`:490-501`); `search_preassigned` rejects `stats`, `store_pairs`, `max_codes` (`:419-445`); only `SearchParametersIVF` accepted (`:365-375`).
- `GpuIndexIVFPQ` `verifyPQSettings_` (`:544-619`): non-cuVS non-interleaved requires `bitsPerCode==8`; interleaved 4/5/6/8; `subQuantizers` must divide `d`; shared-memory bound check suggests `useFloat16LookupTables`; `copyFrom` requires `nbits==8`, `by_residual`, `polysemous_ht==0`.
- Base `GpuIndex`: `compute_residual`/`_n` → `"not implemented for this type of index"` (`:793-805`); ctor validates `device < getNumDevices()`, `dims > 0`, unified-memory CC.

## Verdict — INDEX: **ADOPT**

This layer is the heart of what ELIPS mirrors and it is backend-agnostic:
- **Adopt the `GpuIndex : Index` interchangeability contract directly.** ELIPS already has `GpuIndexPort` extending `IndexPort`/`IndexTransferPort` — align it so a GPU index satisfies the same CPU index interface (`export_to_cpu_index`/`import_from_cpu_index` ≈ FAISS `copyTo`/`copyFrom`).
- **Adopt the "deliberately don't inherit the CPU IVF impl" split** — compose an interface, don't inherit storage you can't honor.
- **Adopt the 3-method subclass contract** (`addImplRequiresIDs_`/`addImpl_`/`searchImpl_`) — it is exactly what ELIPS's per-index orchestrators need.
- **Adopt eager `FAISS_THROW`-style validation with explicit messages.** ELIPS should throw (not silently fall back) for: k over the Metal select cap, unsupported metric, unsupported dtype, range_search, reconstruct bounds. Mirror the exact error-message style for parity debugging.

Metal notes: the numeric-type set (F32/F16/Int8) maps to Metal natively (`half`, `float`, `char`). The k-cap value is a Metal kernel property (Layer 6), not 2048. Everything else here is pure host logic — no CUDA in it.

---

# Layer 4 — CLONER (CPU↔GPU conversion + multi-GPU)

The cloner walks a CPU index tree and dispatches by `dynamic_cast` to build the GPU equivalent (and back). It is how `IndexFlat`/`IndexIVF*` become GPU indexes and vice versa.

**`GpuClonerOptions`** (`GpuClonerOptions.h:16`): `indicesOptions=INDICES_64_BIT` (:19), `useFloat16CoarseQuantizer=false` (:22), `useFloat16=false` (:28, IVFPQ intermediate calc), `usePrecomputed=false` (:31), `reserveVecs=0` (:34), `storeTransposed=false` (:37, **dead — never read**), `verbose=false` (:40), `use_cuvs=false` (:43), `allowCpuCoarseQuantizer=false` (:49, when false the cloner throws for coarse quantizers not on GPU).

**`GpuMultipleClonerOptions : GpuClonerOptions`** (`:52`): `shard=false` (:55), `shard_type=1` (:58), `common_ivf_quantizer=false` (:63). **shard=false → replicate** (each GPU gets a full copy → `IndexReplicas`); **shard=true → split** data across GPUs (`IndexShards`/`IndexShardsIVF`).

**Cloner classes** (`GpuCloner.{h,cpp}`):
- `ToCPUCloner : faiss::Cloner` — GPU→CPU `clone_Index` dispatch (`.cpp:71-122`): each `GpuIndex*` → its CPU type via `copyTo`; recurses `IndexShards`/`IndexReplicas` and `merge_index`es shards.
- `ToGpuCloner : faiss::Cloner, GpuClonerOptions` holds `GpuResourcesProvider* provider`, `int device` — CPU→1-GPU dispatch (`.cpp:139-255`): `IndexFlat`→`GpuIndexFlat`; `IndexScalarQuantizer(QT_fp16)`→`GpuIndexFlat(useFloat16)` transferred in 1 M-vector blocks; `IndexIVFFlat`/`IVFSQ`/`IVFPQ`→GPU equivalents via `copyFrom`; else if `IndexIDMap`/`IndexPreTransform` recurse through base `Cloner`, otherwise `FAISS_THROW_MSG("This index type is not implemented on GPU.")` (`:253`).
- `ToGpuClonerMultiple : faiss::Cloner, GpuMultipleClonerOptions` holds `vector<ToGpuCloner> sub_cloners` — replicate vs `clone_Index_to_shards` (`.cpp:325-438`). `copy_ivf_shard` interprets `shard_type` (`:287-323`): **1**=id modulo (`id%n==i`, default), **2**=id range, **4**=by inverted list; else `FAISS_THROW_FMT("shard_type %d not implemented")`.

**Free functions** (C++, `GpuCloner.cpp`): `index_gpu_to_cpu(const Index*)` (:124), `index_cpu_to_gpu(provider, device, index, opts=nullptr)` (:257), `index_cpu_to_gpu_multiple(vector<provider>&, vector<int>& devices, index, opts)` (:491) — providers/devices zipped index-for-index (sizes must match). Binary variants also exist (:529-611).

**Python-only** (`faiss/python/gpu_wrappers.py`): `index_cpu_to_all_gpus(index, co, ngpu)` (:38), `index_cpu_to_gpus_list(index, co, gpus, ngpu)` (:43) — allocate one `StandardGpuResources` per GPU and call `index_cpu_to_gpu_multiple_py` which marshals `GpuResourcesVector`+`Int32Vector` (:21-35). `get_num_gpus` is a SWIG wrapper (`swigfaiss.swig:386`) over C++ `getNumDevices()` (`utils/DeviceUtils.cu:30-41`, returns 0 on no-device/insufficient-driver).

**`GpuParameterSpace : faiss::ParameterSpace`** (`GpuAutoTune.{h,cpp}`): `initialize` exposes `nprobe` as powers of two up to `getNumLists()`/maxK (`:55-64`) plus `quantizer_*` sub-params; `set_index_parameter` handles `nprobe`, `use_precomputed_table` (→`setPrecomputedCodes`), `quantizer_*` (`:81-114`).

## Verdict — CLONER: **ADAPT**

The CPU↔GPU cloner with `copyFrom`/`copyTo` round-tripping is essential for interchangeability — ELIPS's `GpuIndexPort::import_from_cpu_index`/`export_to_cpu_index` is the same idea and should adopt the dispatch-by-type + explicit-throw-on-unsupported pattern. Adopt the single-GPU `index_cpu_to_gpu(provider, device, index, opts)` entry point and the `GpuClonerOptions` field set (minus dead `storeTransposed`, minus `use_cuvs`).

Multi-GPU (shard/replicate, `shard_type`, `IndexShards`/`IndexReplicas`) → **skip for the first Metal cut**: a single Apple GPU is the target; Mac Pro multi-GPU is rare and unified memory changes the sharding calculus. Keep `GpuMultipleClonerOptions`' shard-vs-replicate vocabulary documented for a future `GpuDistributedIndex` (ELIPS already stubs one). `GpuParameterSpace` autotune → **adapt later**: the `nprobe`/`use_precomputed_table` tunables are relevant but autotune is a Phase-N concern.

---

# Layer 5 — STANDALONE distance (`bfKnn` / `GpuDistanceParams`)

Brute-force kNN with no index object — the raw "queries × database → top-k" primitive. Inputs may live on host **or** device.

**`GpuDistanceParams`** (`GpuDistance.h:32-113`): `metric=METRIC_L2` (:38), `metricArg=0` (:42, p for Lp), `k=0` (:47; **`k=-1` ⇒ all pairwise distances, no top-k**), `dims=0` (:50); database side `vectors`/`vectorType=F32`/`vectorsRowMajor=true`/`numVectors`/`vectorNorms` (:59-66); query side `queries`/`queryType`/`queriesRowMajor`/`numQueries` (:76-79); output `outDistances`/`ignoreOutDistances`/`outIndicesType=I64`/`outIndices` (:88-97); `device=-1` (:108, current device); `use_cuvs=false` (:112). Layout: `*RowMajor=true` ⇒ `num×dims`.

**Enums** (`GpuDistance.h:19-29`): `DistanceDataType { F32=1, F16, BF16 }`; `IndicesDataType { I64=1, I32 }`.

**Functions**: `bfKnn(GpuResourcesProvider*, const GpuDistanceParams&)` (`.h:131`, `.cu:229`, requires `vectorType==queryType`); `bfKnn_tiling(res, args, vectorsMemoryLimit, queriesMemoryLimit)` (`.h:145`, `.cu:516` — the memory-bounded batching API; row-major + `k>0` + CPU-resident input only); `bruteForceKnn(...)` deprecated flat-`float*` alias that warns and forwards (`.cu:573-610`). **No `bfKnnBatch`.**

**Host-or-device detection**: `getDeviceForAddress(ptr)` (`utils/DeviceUtils.cu:107-141`) calls `cudaPointerGetAttributes` → returns `-1` for host/unregistered memory, else the owning GPU id. `toDeviceTemporary` (`CopyUtils.cuh:18-41`): if the pointer already lives on the target device, use it in place (no copy); else temp-alloc + upload. `fromDevice` copies results back to host when the output buffer is host-resident. So the same call works whether you pass numpy arrays or already-resident GPU pointers.

**Tiling** (`bfKnn_tiling`, `.cu:457-569`): shards queries by `queriesMemoryLimit`, shards the database by `vectorsMemoryLimit` with `shard_size = vectorsMemoryLimit / (dims·distance_size)`; merges DB shards through a `HeapArray<C>` (CMin/CMax by metric) with label offsetting.

**Python** (`faiss/python/gpu_wrappers.py`): `knn_gpu(res, xq, xb, k, D, I, metric, device, use_cuvs, vectorsMemoryLimit, queriesMemoryLimit)` (:60) infers row/col-major per-array from numpy `c_contiguous`/`f_contiguous` (:118-146), dtype→`DistanceDataType` (f32/f16 only; :130-155), allocates D (f32) / I (i64 or i32), maps **`xb`→`vectors*`, `xq`→`queries*`**, dispatches to `bfKnn_tiling` if a memory limit is set else `bfKnn`. `pairwise_distance_gpu` (:211) sets `k=-1`. (Two pre-existing bugs noted in `pairwise_distance_gpu`: xq-dtype fallback reads `xb` at :259; non-contiguous xq raises instead of coercing.)

## Verdict — STANDALONE: **ADOPT**

`bfKnn` is the cleanest, most reusable primitive in FAISS GPU and maps near-perfectly to ELIPS's `GpuPort::compute_distances_batch` + `top_k`. Adopt:
- The `GpuDistanceParams` struct shape (metric, k, dims, row/col-major flags, dtype enums, host-or-device pointers, optional precomputed norms).
- **The host-or-device transparency** via a `getDeviceForAddress`-equivalent — on Metal this is even simpler: `MTLBuffer` vs a raw host pointer, and with unified memory a host pointer is often already GPU-visible (wrap via `newBufferWithBytesNoCopy`). This is a major Metal simplification.
- `k=-1` all-pairwise mode and the `bfKnn_tiling` DB/query sharding contract (needed when the dataset exceeds working memory — still relevant on Metal for very large DBs, see Layer 7).

Adapt: dtype enum keeps F32/F16, consider dropping BF16 initially (Metal bf16 support is newer/limited); the Python `knn_gpu` wrapper translates directly to ELIPS's pybind layer. Note the `xb→vectors / xq→queries` mapping convention to avoid a parity bug.

---

# Layer 6 — KERNEL (k-selection, GEMM distance, fp16)

This is the layer that must be **rewritten from first principles for Metal** — but the *algorithms* carry over, and the warp/simdgroup width matches.

## The max-k limit and why

`GPU_MAX_SELECTION_K` = **2048** (CUDA > 9.0 / ROCm), **1024** (CUDA ≤ 9.0) (`DeviceDefs.cuh:39,65,67`). Verified comment (`:62-64`): the limit is a **register-allocation / shared-memory budget** on the selection heap — larger k needs more registers+shared memory per thread, so the 2048 kernels drop to 64 threads/block to fit. cuVS raises the ceiling to **16384** (`IndexUtils.cu:21-25`) but the native GEMM+select path stays at 2048 (`Distance.cu:205`). Enforced in `validateKSelect` and re-asserted in every select kernel (`BlockSelectFloat.cu:53`, `WarpSelectFloat.cu:53`, `L2Select.cu:203`).

## WarpSelect vs BlockSelect (`utils/Select.cuh`)

Two-level selection heap, same concept in both: a small per-thread **register** queue (`threadK[NumThreadQ]`, insert only if the candidate beats the current k-th boundary `warpKTop`) feeding a larger sorted **warp queue** (`NumWarpQ` entries). When a thread queue fills (detected warp-wide via `__any_sync`), a bitonic `mergeWarpQ` merges it into the warp queue.
- **WarpSelect**: warp queue lives in **registers** (`warpK[NumWarpQ/kWarpSize]`); one warp handles one input row. Used for **many short rows** (e.g. IVF list scan). Grid = `divUp(numRows, warpsPerBlock)`, 128 threads/block.
- **BlockSelect**: warp queue in **shared memory** (one per warp, `smemK[kNumWarps*NumWarpQ]`); a whole block cooperatively reduces one row. Used for **long rows / few rows** (the GEMM distance output). 128 threads (64 at k=2048).
- The k value only selects the `NumWarpQ` template bucket (next pow2 ≥ k). Instantiation table (identical in `BlockSelectFloat.cu:14-22` / `WarpSelectFloat.cu`): `(WarpQ,ThreadQ)` = (1,1)(32,2)(64,3)(128,3)(256,4)(512,8)(1024,8)(2048,8). 2048 kernels gated `#if GPU_MAX_SELECTION_K >= 2048`.

## GEMM-based distance (`impl/Distance.cu`, `MatrixMult-inl.cuh`)

Identity: **L2 = `||c||² − 2·q·c + ||q||²`**, **IP = `q·c`**. The `−2qc` / `qc` term is a cuBLAS GEMM (alpha `−2.0f` / `1.0f`), fp32 accumulate always (`:90-91,286-287`). Pipeline (`runDistance`, `:120-406`): compute norms → `chooseTileSize` → per-tile GEMM → **fused `runL2SelectMin`** that adds `||c||²` *and* does k-selection in one kernel (comment `:302-305`: only two passes over the big output region), then adds `||q||²` to the k selected values. IP path just `runBlockSelect`s the GEMM output.

Tiling: outer loop over queries (`tileRows`), inner over database (`tileCols`), double-buffered across 2 alternate streams. `chooseTileSize` (`DistanceUtils.cuh:389-431`): `targetUsage` = 512 MiB (≤4 GiB GPU) / 768 MiB (≤8 GiB) / 1 GiB, `/= 2·elementSize` (reserve for double-streaming), `preferredTileRows=512` (1024 if `dim≤32`), `tileCols = targetUsage/preferredTileRows`. GEMM backend `rawGemm` (`MatrixMult-inl.cuh:66-167`): fp16/bf16 inputs, **fp32 output always**; CUDA ≥ 11 uses `cublasGemmEx` with `CUBLAS_COMPUTE_32F_PEDANTIC`.

## Float16 / BF16 (`utils/Float16.cuh`)

fp16 full math gated `__CUDA_ARCH__ ≥ 530` (:14-17); bf16 `≥ 800` (:19-22). Packed `Half4`/`Half8` for vectorized loads. **Pattern: fp16/bf16 is a storage/bandwidth optimization only — all reductions and GEMM accumulate in fp32** (`L2Norm.cu:48`, select output buffers always `float`).

## Metal portability map

| CUDA construct | Metal equivalent |
|---|---|
| `kWarpSize = 32` (`DeviceDefs.cuh:49`) | **simdgroup width = 32 on Apple** — maps directly (biggest win) |
| `__syncwarp()` / `warpFence` | `simdgroup_barrier(mem_threadgroup)` |
| `__any_sync(mask, pred)` | `simd_any()` |
| `shfl` / `warpReduceAll[Sum]` | `simd_shuffle` / `simd_min`/`simd_max`/`simd_sum` |
| `__syncthreads()` | `threadgroup_barrier` |
| `__shared__` / `extern __shared__` | `threadgroup` memory |
| `cublasGemmEx`/`cublasSgemmEx` | **MPSMatrixMultiplication** (or custom simdgroup-matrix GEMM) |
| `half2`/`__nv_bfloat16`/`float4` | native Metal `half`/`half4`/`float4`; bf16 newer/limited |
| `maxThreadsPerBlock` | `maxTotalThreadsPerThreadgroup` (~1024 Apple) |
| `#pragma unroll` | supported |

## Verdict — KERNEL: **REDESIGN from first principles (algorithm ADOPT, code SKIP)**

Rewrite every kernel for Metal, but **keep the algorithm designs**:
- **The two-level register/threadgroup selection heap is the right design for Metal too.** The 32-wide warp queue maps to Apple's 32-wide simdgroup; `kNumWarpQRegisters = NumWarpQ/32` register sizing carries over. `simd_any`/`simd_shuffle`/`simdgroup_barrier` replace the CUDA warp intrinsics 1:1. This is the single highest-value kernel to port (it is the `top_k` that ELIPS's `MetalBackend` currently does on CPU via `std::partial_sort` — see `_phase1_gpu.md`).
- **GEMM-then-fused-select is the right distance strategy.** Use `MPSMatrixMultiplication` for the `−2qc`/`qc` term (fp32 accumulate), then a fused `L2SelectMin`-equivalent Metal kernel that adds `||c||²` and selects in one pass. Query×DB tiling with double command-buffering reproduces the double-streaming.
- **fp16-storage / fp32-accumulate is correct and native on Metal** — adopt directly.
- **The max-k value must be re-derived empirically for Apple GPUs** — do NOT copy 2048. It is a function of register file + threadgroup memory per the comment; measure it per Apple GPU family and expose it as the Metal analog of `GPU_MAX_SELECTION_K` (a **calibration knob**). Likewise the tile-memory targets (512/768 MiB/1 GiB), block sizes (128/64), `preferredTileRows` (512/1024), `kRowsPerBlock=8`, `rowTileSize=8` are CUDA-occupancy tunings to re-tune for Apple threadgroup limits, not copy verbatim.

---

# Layer 7 — MEMORY / PAGING (dataset > working memory)

Two independent paging systems: (a) **host→device query/add paging** in `GpuIndex.cu` (CPU data too big to upload at once), and (b) **GEMM tiling** in `Distance.cu` (output distance matrix too big for temp memory, Layer 6).

**Page constants** (`GpuIndex.cu:28-44`): `kMinPageSize=256 MiB` (default `minPagedSize_`, the threshold at/above which CPU search data is paged), `kNonPinnedPageSize=256 MiB`, `kAddPageSize=256 MiB`, `kAddVecSize=512K vectors`, `kSearchVecSize=32K vectors` (smaller because IVFPQ precomputed codes are memory-heavy; marked FIXME). Settable via `setMinPagingSize`.

**add() paging** (`addPaged_ex_`, `:181-237`): pages only if `!cuVS && (totalSize > kAddPageSize || n > kAddVecSize)`; per-page vectors = `kAddPageSize/(d·sizeof)`, clamped to `kSearchVecSize` (≤32,768); loops `addPage_ex_` over slices.

**search() paging** (`search_ex` → `searchFromCpuPaged_ex_`, `:554-773`): triggers only when the **query input is host-resident** (`getDeviceForAddress(x)==-1`) AND `n·d·sizeof ≥ minPagedSize_`. When pinned memory is available, a **triple-buffered pipeline** overlaps CPU→pinned memcpy, pinned→GPU async copy (on the async copy stream), and GPU compute, using two pinned buffers + two GPU temp buffers synced via CUDA events (ASCII diagram at `:618-631`). No-pinned fallback pages without overlap at `kNonPinnedPageSize`. **Output `n·k` is assumed to fit on GPU** — explicitly not handled (`:415-416`).

**OOM behavior**: device alloc failure → `FaissException` `"StandardGpuResources: alloc fail <n> (cudaMalloc error ...)"` (`StandardGpuResources.cpp:557-577`); temp-stack exhaustion first logs `"...(no temp space); retrying as MemorySpace::Device"` and retries as a real `cudaMalloc` (`:535-541`). The temp stack itself asserts (does not throw) on over-budget (`StackDeviceMemory.cpp:84`). `TestGpuMemoryException.cpp` verifies an oversized `add` throws `FaissException` and leaves the index usable afterward.

## Verdict — MEMORY/PAGING: **ADAPT (mostly SIMPLER on Metal)**

The GEMM output tiling (Layer 6) must stay — the distance matrix can exceed working memory regardless of unified memory, so keep `chooseTileSize`-equivalent query×DB tiling. But the **host→device query/add paging largely disappears on Apple Silicon unified memory**: there is no separate device address space to copy *into*, so the triple-buffered pinned pipeline (the most complex code here) collapses to "wrap the host pointer in a shared `MTLBuffer`." Keep:
- A `minPagedSize_`-style threshold and `setMinPagingSize` knob (still useful for discrete-GPU Macs and for bounding temp-matrix size).
- The **GEMM/distance-matrix tiling** when `numQueries × numDB × k` output exceeds the temp budget.
- The DB-sharding path (`bfKnn_tiling`, Layer 5) for datasets larger than total memory.
- **OOM → throw a typed exception and keep the index usable** (adopt the `TestGpuMemoryException` contract directly). On Metal, allocation failure surfaces as a nil `MTLBuffer`/`newBuffer...` returning nil — convert to an ELIPS exception.

Mark as a **calibration knob**: the 256 MiB page sizes and 32K/512K vector caps are CUDA-era; on unified memory the add/search "page" is really just a tile-size for kernel dispatch, re-tuned per device.

---

# Layer 8 — TESTING (CPU-vs-GPU consistency)

The test philosophy: GPU is **not** required to bit-match CPU — it must agree on *ranks* within a tolerance and on *distances* within a relative error. This is the parity contract ELIPS must copy.

**Harness** (`test/TestUtils.{h,cpp}`): `compareIndices` searches CPU reference and GPU on the same queries at the same k, defaults `maxRelativeError=6e-5, pctMaxDiff1=0.1, pctMaxDiffN=0.005` (`.h:77-92`). `relativeError(a,b)=|a−b|/(0.5(|a|+|b|))` (`.cpp:92-94`). `compareLists` (`.cpp:234-356`) is the assertion engine: for each returned index it finds its position in the reference and buckets the **position difference** — `diff1` (off by exactly 1), `diffN` (off by >1), `diffInf` (not in reference). Assertions: per-element `relErr ≤ maxRelativeError`; `(diff1+diffN+diffInf) ≤ numResults·pctMaxDiff1` (total mis-rank fraction); `diffN ≤ numResults·pctMaxDiffN`. `-1` (IVF misses) must match slot-for-slot; duplicate GPU indices flagged. **This is a recall-like metric, not exact match.**

**Per-test tolerances:**
- Flat (`TestGpuIndexFlat.cpp:21-22`): `F32 relErr 6e-3`, `F16 relErr 0.07`; fp16 rank bounds deliberately loose (0.99/0.65, "useless when accumulator is fp16").
- IVFFlat (`:39-40`): `F32 0.03`, `F16 0.3`.
- IVFPQ (`:106-120`): epsilon `0.15` if `dim≤8` else `0.035`; `pctMaxDiff1` 0.10 (f32) / 0.30 (f16); `pctMaxDiffN` 0.02 / 0.05.
- `TestGpuDistance.cu:64-99`: Normal `6e-3/0.1/0.015`, BF16 `1.5e-2/0.3/0.1`.
- Python (`test_gpu_index.py`): deterministic preassigned paths use exact `assert_array_equal` + `assert_allclose(rtol=atol=1e-5)`; approximate paths require **≥0.9 recall / kNN-intersection** (`:271,313`) with `rtol=atol=5e-4/5e-5` on distances.

## Verdict — TESTING: **ADOPT**

Adopt the whole methodology — it is backend-agnostic and exactly what ELIPS's `CpuGpuRecallParityTest` should enforce:
- **Compare GPU against a CPU reference index** on identical queries; assert (a) distance relative error ≤ ε and (b) rank-agreement fraction, not bit-exactness.
- **Tier tolerances by precision**: tight for f32 exact (Flat), looser for fp16/PQ/approximate. Reuse FAISS's actual numbers as starting points (f32 Flat `6e-3`, f16 `0.07`, IVFPQ `0.035`, recall ≥ 0.9) — then calibrate for Metal fp32/fp16 accumulation behavior.
- **Adopt the OOM-recoverability test** (oversized op throws, index still usable).
- Metal note: Apple fp32 GEMM via MPS may differ slightly from cuBLAS; expect to re-tune ε but keep the *structure* (relErr + mis-rank fraction + ≥0.9 recall).

---

# Cross-layer adopt / adapt / skip summary

| Layer | Verdict | One-line rationale |
|---|---|---|
| RESOURCE | **ADAPT** | Shared resources + temp stack + lock-free stream ordering: keep the shape; `MTLHeap`/unified-memory/`MTLCommandQueue` replace the mechanics; re-tune temp caps. |
| CONFIG | **ADOPT** | POD-struct-per-index with inheritance + `IndicesOptions` (32/64-bit storage) are backend-agnostic. Drop `use_cuvs`/dead `storeTransposed`; default `memorySpace=Shared`. |
| INDEX | **ADOPT** | `GpuIndex : Index` interchangeability, the 3-method subclass contract, and eager `FAISS_THROW` validation are the core of the design. Pure host logic. |
| CLONER | **ADAPT** | CPU↔GPU `copyFrom`/`copyTo` round-trip + dispatch-by-type: adopt single-GPU; skip multi-GPU shard/replicate for the first Metal cut. |
| STANDALONE | **ADOPT** | `bfKnn`/`GpuDistanceParams` host-or-device primitive maps straight to `GpuPort::compute_distances_batch`+`top_k`; even simpler with unified memory. |
| KERNEL | **REDESIGN** | Rewrite kernels for Metal but keep the algorithms: 2-level simdgroup selection heap, MPS GEMM + fused select, fp16-store/fp32-accumulate. Re-derive max-k empirically (not 2048). |
| MEMORY/PAGING | **ADAPT** | GEMM/distance-matrix tiling stays; host→device query/add paging + pinned pipeline mostly dissolves under unified memory. OOM→typed exception, index stays usable. |
| TESTING | **ADOPT** | CPU-reference parity via distance-relErr + rank-agreement fraction + ≥0.9 recall; tier tolerances by precision. Re-tune ε for Metal/MPS. |

## What maps cleanly to Metal vs needs first-principles work

**Clean maps:** simdgroup width 32 = CUDA warp 32 (selection heap register sizing carries over); unified memory removes most upload/paging code and makes host-or-device pointers trivial; `MTLHeap` placement = StackDeviceMemory bump stack; MPSMatrixMultiplication = cuBLAS GEMM (fp32 accumulate); native `half`/`half4` = fp16 paths; all host-layer logic (CONFIG/INDEX/CLONER/STANDALONE dispatch, validation, parity tests) is CUDA-free and ports as-is.

**First-principles redesign:** every compute kernel (selection, L2Norm, fused L2-select, PQ scan) must be rewritten in MSL using `simd_*` intrinsics + `threadgroup` memory + `threadgroup_barrier`; the max-k ceiling, tile-memory targets, block/threadgroup sizes, and temp-memory caps are all CUDA-occupancy tunings that must be re-derived for each Apple GPU family (treat as calibration knobs, never copy the CUDA numbers verbatim); bf16 is newer/limited on Metal (defer); the triple-buffered pinned-copy pipeline is replaced by shared-memory buffers, not ported.







