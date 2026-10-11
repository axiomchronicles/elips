# Phase 9E — Website + Tutorial verification (`_docs_verif_9E.md`)

Scope: `elips_docs/**` only (the TanStack site + tutorial). No `docs/` markdown, no code, no git.
Ground truth: `03_naming_map.md`, `_phase1_docs.md`, `_naming_other.md`, `_benchmarks.md`, and the
real Python API in `bindings/python/elips/**.py` + `_core.pyi` + `src/EQLParser.cpp` (verified).
Shared API cheat-sheet written to `tmp/_api_spec_9E.md` and used by every pass.

## Method
Core/fabricated pages fixed directly; the bulk Python-reference, C++-reference, tutorial, and
concept/practice pages fixed by five parallel sub-agents against the same spec. No build run
(no `node_modules`); validated by balanced `<CodeBlock>`/`<DocsShell>` tags site-wide, a stale-term
grep sweep, and sub-agent typechecks (B/C/E reported `tsc --noEmit` clean; D reported bun bundle clean).

## Files changed (≈57)
- Directly (17): `routes/docs.python.eql.tsx`, `routes/docs.eql.tsx`, `routes/docs.python-sdk.tsx`,
  `routes/docs.python.connect.tsx`, `routes/docs.gpu.overview.tsx`, `routes/docs.design-decisions.tsx`,
  `routes/docs.getting-started.tsx`, `routes/docs.index.tsx`, `routes/docs.api.tsx`, `routes/index.tsx`,
  `routes/docs.benchmarks.tsx`, `routes/docs.project-history.tsx`, `routes/contributing.tsx`,
  `routes/chat.tsx`, `routes/api/chat.ts`, `lib/content.ts`, `components/Code.tsx`.
- Sub-agent A (4): `routes/docs.python.{arena,vault,database,engine}.tsx`.
- Sub-agent B (10): `routes/docs.python.{models,config,filtering,maintenance,transaction,durability,wal,errors,gpu,utilities}.tsx`.
- Sub-agent C (7): `routes/docs.cpp-sdk.tsx`, `routes/docs.cpp.{overview,elips-instance,vault,domain,config,transaction}.tsx`.
- Sub-agent D (1): `lib/tutorial.ts` (all lessons).
- Sub-agent E (18): `routes/docs.{core-concepts,architecture,algorithms,advanced,guides,examples,cli,storage,configuration,installation,performance,roadmap,gpu-engine}.tsx`,
  `routes/docs.internals.{transaction-engine,lock-manager}.tsx`, `routes/changelog.tsx`, `routes/faq.tsx`,
  `components/SketchDiagram.tsx`.

## Old surface → current surface (applied everywhere user-facing)
- Opening: `elips.open`/`connect`/`Engine`/`arena` → `elips.Client(dimension=…)` / `PersistentClient(path, …)`
  + `get_or_create_collection`. `connect`/`open` kept only as explicitly-labelled deprecated shims.
- Collection verbs: `write/ingest/probe*/seek*/pull/sweep/discard/place*` → Chroma
  `add/upsert/get/update/delete/query/count/peek`; kwargs `top=→n_results`, `meta=→metadatas`,
  `keys=→ids`, `vector(s)=→embeddings`, `text=→documents`, `include_vectors→include`, `where=`.
- Results: `.id/.payload` → `Row`/`Hit` `.key/.meta/.text/.distance`.
- FAISS layer surfaced: `IndexFlat/IndexIVFFlat/IndexIVFPQ/IndexIVFScalarQuantizer/IndexHNSW`,
  `index_factory`, `bf_knn(…, device="metal")`, `train/add/search/nprobe/ntotal`.
- C++: `ElipsInstance→Database`(+`Client`/`PersistentClient`), `Vault→Collection`, `place→add`,
  `seek→search`, `erase→remove`, `fetch→get`, `GraphParams→HnswParams`, `IndexType::graph/exact→hnsw/flat`,
  `IndexPort→Index`, `GpuPort→GpuBackend`, `GpuSelector→BackendSelector`, `DynamicBatcher→QueryBatcher`.
- CLI: `vaults→collections`, added `quantize`.

## Fabricated content removed
- **False denial** ("There is no add/ingest/search/get/delete"): not present in the current source
  (removed in an earlier pass); confirmed no equivalent denial remains. `docs.python-sdk.tsx` now
  teaches the real `add/query/get/delete` + FAISS Index surface.
- **Uppercase EQL dialect** (`SEARCH FROM … SEEK :q TOP 10 WHERE id = N`, `:name` bindings,
  `HYBRID_WEIGHT`, `INSERT`, positional ParseErrors) deleted from `docs.python.eql.tsx` and
  `docs.python.database.tsx`; replaced with the real lowercase grammar
  (`search in <c> nearest $q k 10 where … yield`, `$name` bindings, `get/scan/add/delete`).
  `docs.eql.tsx` re-verbed (seek→search, fetch→get, place/erase→add/delete, top→k, data→metadata).
- **GPU over-claims**: `gpu=True`/`engine.gpu_info()`/`NVIDIA RTX 4090 185k QPS`/CAGRA/cuVS/Tensor-Core
  removed. GPU now presented as Metal (production) + CUDA/HIP (experimental) + CPU with a maturity table
  (`docs.gpu.overview.tsx`, `docs.design-decisions.tsx` GPU-ADR cards, `api/chat.ts`). Selection via
  `device="metal"|"cuda:N"|…"` and `client.native.gpu_info()`; enumerate via `elips.accelerators()`.
- **Benchmarks**: `docs.benchmarks.tsx` fabricated numbers replaced with the real M4 suite from
  `_benchmarks.md` (IndexFlat cpu 1.000@489, bf_knn metal batched 1.000@914, IVFFlat nprobe sweep
  0.030/0.152/0.381/0.801, faiss references); removed the invented durability-throughput and
  RTX/DynamicBatcher rows.
- **SYCL / Vulkan**: already absent from source (removed in Phase 2); verified zero live refs remain
  (only "…were removed" negative statements).
- `connect` fabricated kwargs `index_type`/`durability=`/`gpu=True`/`dimension=128` default corrected to
  the real `index`/`gpu`/`device`/`dimension=0` surface.

## Dead refs / deliberate retentions
- `routeTree.gen.ts` untouched; every `createFileRoute("/…")` path and `<Link>`/`href` target preserved
  (e.g. `/docs/python/{arena,vault,database}`, `/docs/cpp/elips-instance`) — content retitled, URLs kept.
- Native low-level pages (`python/vault`, `python/database`) intentionally keep `place/seek/fetch/erase/scan`
  as the correct native escape-hatch API reached via `.native`.
- `components/SketchDiagram.tsx` keeps the `DynamicBatcherDiagram` **export name** and the
  `GpuIndexPort`/`GpuMemoryPort`/`GpuKernelPort`/`GpuStreamPort` ISP sub-port labels (import stability +
  naming-map leaves these sub-ports mission-silent); rendered batcher label reads QueryBatcher.
- `lib/tutorial.ts` keeps the `DiagramName` union value `"DynamicBatcher"` (maps to the component via
  `LessonRenderer`); renaming it would break the lesson→diagram mapping.

## Remaining gaps
- No Vite/tsc build was run (no `node_modules`); correctness rests on grep sweeps, tag-balance checks,
  and the sub-agents' own typechecks. A real `bun install && tsc --noEmit && vite build` is still advisable.
- `elips` CLI transaction/`begin_transaction` surface is documented from the Python shim + naming map;
  if a dedicated CLI/transaction spec exists it is worth a cross-check.
