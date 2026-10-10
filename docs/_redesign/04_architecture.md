# 04 — Target Architecture (ADR-style decisions + open questions)

Target: FAISS-recognizable Index/GPU architecture + Chroma-recognizable Collection API,
two layers (Collection built on Index). Metal validated first; CUDA/HIP experimental.

## Decisions
- **ADR-R1 (Metal precedent exists — overrides mission ground rule #8).** The mission states
  "Metal has no FAISS precedent … design kernels from first principles." **This is now false.**
  Verified against the real clone (origin facebookresearch/faiss, tag v1.15.1,
  `75c755c5`): **upstream FAISS v1.15.1 ships a full Metal backend** at `faiss/gpu_metal/`
  (~7700 LOC): `MetalResources`/`StandardMetalResources`, `MetalIndex`/`MetalIndexFlat`/
  `MetalIndexIVFFlat`/`MetalIndexIVFPQ`, `MetalCloner`, `MetalDistance.metal` (40 KB of MSL
  incl. `l2_squared_matrix`, `ip_matrix`, `compute_norms`, `l2_with_norms`,
  **`topk_threadgroup_2048` — a real on-GPU top-k**, `ivf_scan_list*`, `ivf_merge_lists`),
  `MetalFlatKernels`, `MetalPythonBridge`, and parity tests; `FAISS_ENABLE_METAL` auto-ON on
  Darwin arm64. **Reality wins (ground rule #7).**
  **Consequence:** Phase 7's Metal work is now *adapt a real FAISS precedent*, not greenfield.
  In particular ELIPS's current CPU-side top-k should be replaced using FAISS's
  `topk_threadgroup` approach, and the metallib build chain + resource/cloner design are
  directly referenceable. A dedicated study (`_faiss_3I_metal.md`) is in progress.

- **ADR-R2 (DEEP rename through the core — user decision 2026-10-10).** The user chose the
  deep rename over the public-boundary-only approach I recommended. The mission's naming table is
  applied literally into the mature C++ core: core classes/methods/enums/files are renamed
  (`Vault`→`Collection`, `place`→`add`, `seek`→`search`, `Database`→`Client`-tier, `IndexPort`→
  `Index`, `GpuPort`→`GpuBackend`, `GpuSelector`→`BackendSelector`, `DynamicBatcher`→`QueryBatcher`,
  `MetadataIndex`→`MetadataFilterIndex`, …), plus Python + EQL + CMake targets + test names + docs,
  with a deprecation shim mapping old public names → new (DeprecationWarning, removed next minor).
  **Collision rule:** where a Chroma name is a C++ keyword/reserved (`delete`), the C++ method takes
  a safe synonym (`remove`) while Python exposes the Chroma name (`delete`). Accepted risk: large
  invasive diff across the tested engine; mitigated by applying renames mechanically in batches,
  keeping ctest+pytest green after each batch, and the naming map (`03_naming_map.md`) being written
  and reviewed BEFORE any code change (Phase 5 gate).
  **Superseded recommendation:** public-boundary-only (lower risk) — not chosen.

## Open questions
_(record here with a recommended default, then proceed)_
