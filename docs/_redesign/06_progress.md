# 06 — Progress Log (single source of truth for orchestration)

Running checklist. Update at every gate. Dates are absolute (today: 2026-10-10).

## Phase status

| Phase | Title | Status | Gate met? |
|---|---|---|---|
| 1 | Inventory & baseline | DONE (2026-10-10) | ☑ |
| 2 | Prune GPU backends (SYCL, Vulkan) | DONE (2026-10-10) | ☑ |
| 3 | Study FAISS | DONE (2026-10-10) | ☑ |
| 4 | Gap analysis | IN PROGRESS (2026-10-10) | ☐ |
| 5 | Naming & public API redesign | not started | ☐ |
| 6 | Core engine maturity | not started | ☐ |
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

## Open questions
(see `04_architecture.md`)

## Log
- 2026-10-10: Phase 1 kicked off. Repo oriented: ~32k LOC source (82 cpp / 67 hpp / 36 py). 5 GPU backends present as directories; SYCL+Vulkan marked for Phase 2 removal (24 / 28 ref'd files resp.). Prebuilt `_core.cpython-313/314-darwin.so` present in bindings.
- 2026-10-10: Phase 1 DONE. 4 inventory agents + baseline landed; synthesized `00_inventory.md`; committed (`redesign` branch, f496d41). Premise corrected: CPU engine mature.
- 2026-10-10: Phase 2 DONE. Deleted SYCL + Vulkan entirely: source dirs (`src/gpu_engine/backends/{sycl,vulkan}`), CMake options/wiring (root + gpu-engine), device probe/construct/rank in `GpuDeviceManager.cpp`/`GpuSelector.cpp`, 2 ADRs (GPU-009/010), 16 doc files + 3 website files scrubbed, 1 test renamed (`…cuda_over_hip`), accelerator.py + README backend lists. Backend set now Metal/CUDA/HIP/CPU. Repo-wide grep for `sycl|vulkan|oneapi|spirv|glsl|vk` CLEAN outside `docs/_redesign`. Rebuild green: ctest 256 (255 pass/1 skip), pytest 34/34. Gate MET.
- NOTE for Phase 8: latent build issues left untouched (out of Phase 2 scope): `src/gpu_engine/CMakeLists.txt` `ELIPS_GPU_TESTS` adds a missing `tests/` subdir (gated OFF); CUDA/HIP `CMakeLists` name nonexistent `.cu`/`.hip` sources (actual files `.cpp`); linker "ignoring duplicate libraries" (elips_core/elips_gpu linked twice).
- 2026-10-10: Phase 3 started. Cloned facebookresearch/faiss @ v1.15.1 (`75c755c5`) to /tmp/faiss. Fanned out 3A–3H study agents. **MAJOR CORRECTION (overrides mission ground rule #8):** upstream FAISS v1.15.1 now has a REAL Metal backend (`faiss/gpu_metal/`, ~7700 LOC, incl. GPU top-k `topk_threadgroup_2048`). Added agent 3I to study it; recorded as ADR-R1 in `04_architecture.md`. Phase 7 Metal work is now adapt-a-precedent, not greenfield.
- 2026-10-10: Phase 3 DONE. All 9 section reports written (`_faiss_3A..3H.md` + `_faiss_3I_metal.md`); synthesized into `01_faiss_study.md` with 8 reality-corrections + adopt/adapt/skip per section + consolidated decisions feeding Phases 5–8. Gate MET (study non-destructive, nothing built). Next: Phase 4 gap analysis.
