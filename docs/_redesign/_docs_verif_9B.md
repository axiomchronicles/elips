# Phase 9B — C++ SDK + internals docs verification

Scope: fix `docs/cpp/**`, `docs/cpp_sdk.md`, `docs/internals/**` to the FINAL
post-rename code. Ground truth = `03_naming_map.md` + the real headers in
`include/elips/**` and `src/`. Rule #7 reality-wins applied where the naming map
and the shipped code diverged (see below). No code / other-docs / website / git
touched.

## Files updated (19)

- `docs/cpp_sdk.md`
- `docs/cpp/getting-started/{installation,quickstart}.md`
- `docs/cpp/architecture/layers.md`
- `docs/cpp/api-reference/{config,domain-types,elips-instance,filter,transaction,vault}.md`
- `docs/cpp/internals/{index-engine,metrics-engine,query-engine,storage-engine}.md`
- `docs/cpp/performance/{benchmarking,memory-model}.md`
- `docs/internals/{gpu-engine,lock-manager,transaction-engine}.md`

## Symbols corrected (verified against headers)

| Old | New | Verified in |
|---|---|---|
| `IndexPort` | `Index` | `index_engine/Index.hpp` |
| `ExactIndex` | `IndexFlat` | `index_engine/IndexFlat.hpp` |
| `HierarchicalGraphIndex` | `IndexHNSW` | `index_engine/IndexHNSW.hpp` |
| *(new)* | `IndexIVFFlat`, `IndexIVFPQ`, `IndexIVFScalarQuantizer` | `index_engine/IndexIVF*.hpp` |
| `GraphParams` | `HnswParams` | `Config.hpp` |
| `MetadataIndex` | `MetadataFilterIndex` | `metadata/MetadataFilterIndex.hpp` |
| `GpuPort` | `GpuBackend` | `gpu_engine/GpuBackend.hpp` |
| `GpuIndexPort` | `GpuIndex` | `gpu_engine/GpuIndex.hpp` |
| `GpuSelector` | `BackendSelector` | `gpu_engine/BackendSelector.hpp` |
| `DynamicBatcher` | `QueryBatcher` | `gpu_engine/QueryBatcher.hpp` |
| `Vault` / `VaultInfo` | `Collection` / `CollectionInfo` | `elips.hpp` |
| `ElipsInstance` | `Database` | `elips.hpp` |
| `TransactionVault` | `TransactionCollection` | `elips.hpp` |
| `place*/seek*/fetch/erase/quantize` | `add*/search*/get/remove/train` | `elips.hpp` |
| `insert` / `size` | `add_with_ids` / `ntotal` | `index_engine/Index.hpp` |

## Topics documented

- **Index contract** (`Index.hpp`): train / is_trained / reset / add_with_ids /
  remove / search / range_search (default throws `ElipsError`; only IndexFlat &
  IndexIVFFlat override) / ntotal / type_name / vacuum / pending_removals.
- **CPU IVF family**: IndexIVFFlat (fp32 postings, type_name `ivfflat`),
  IndexIVFPQ (PQ-coded residuals, `ivfpq`), IndexIVFScalarQuantizer (SQ8 raw,
  `ivfsq`); nprobe knob (default 1, clamped `[1,nlist]`); train() required →
  `NotTrainedError`.
- **index_factory grammar** (`IndexFactory.hpp`): `Flat`, `IVF<nlist>,Flat`,
  `IVF<nlist>,PQ<M>`, `IVF<nlist>,SQ8`, `HNSW<M>`; `ConfigError` on bad spec.
  `make_index(config, dim[, gpu_backend])` table: `IndexType::exact`→IndexFlat,
  `IndexType::graph`→IndexHNSW.
- **C API** (`c_api/elips.h`): elips_index_create / add / search / ntotal /
  free; `elips_metric`, `elips_status`; `-DELIPS_ENABLE_C_API=ON`; documented as
  intentionally minimal (single-vector add + k-NN; IVF specs assert untrained).
- **Typed exceptions** (`domain/Errors.hpp` + `kernel/LockManager.hpp`):
  ElipsError base; DimensionMismatch (+`DimensionMismatchError` alias),
  InvalidVector, ConfigError, NotTrainedError (⊂ ConfigError), NotFound,
  StorageError, BackendUnavailableError, LockConflict.
- **Build/install** (`08_build_packaging.md`): `ELIPS_OPT_LEVEL`,
  `ELIPS_ENABLE_{C_API,BLAS,PYTHON,GPU,METAL,CUDA,ROCM}`, `BUILD_TESTING`,
  `BUILD_SHARED_LIBS`; `find_package(elips REQUIRED)` +
  `elips::elips_core`; targets elips_core / elips_gpu / elips_gpu_metal;
  deprecated option aliases; version 1.1.0.
- **GPU internals** (`gpu-engine.md`): real GpuBackend interface (12 pure + 1
  defaulted `ivf_scan`; `free_device(GpuBuffer)` by value); Metal backend layering
  (validated); CUDA/HIP experimental/container-only.
- Maturity labels and a one-line MIGRATION pointer added across the set.

## Reality-wins corrections (code over naming map)

- `enum class IndexType { graph, exact }` — NOT renamed to `hnsw/flat`; no such
  enumerators exist. `Config::index(IndexType::graph)` and `Config::graph_params()`
  keep their names. Docs follow the code.
- No `class Client`/`PersistentClient` in C++ — entry point is `elips::open()`.
- `GpuMemoryPort` is real (implemented by GpuMemoryManager) and kept; only the
  zero-implementer `GpuKernelPort`/`GpuStreamPort` were dropped from the docs
  (their headers still exist, noted as vestigial). No SYCL/Vulkan backends exist.

## Remaining gaps

- `vault.md` / `elips-instance.md` filenames kept (now document Collection /
  Database) to avoid breaking site nav — content is correct; rename deferred.
- `MIGRATION.md` is referenced by the new pointers but was not created/verified
  (outside the 9B file set).
- IVF family documented at header-contract level; `src/` algorithmic internals
  (k-means iteration counts, residual LUT details) not line-by-line like HNSW.
- Not re-audited (not rename targets): BackendSelector ranking table, GpuConfig
  defaults, GpuIndex method signatures, WAL on-disk `insert_ex/insert_q` enum
  variants (documented as kept-format), `elips.snapshot` vs `elips.manifest`
  artifact naming in lock-manager.md.
