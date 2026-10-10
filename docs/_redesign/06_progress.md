# 06 — Progress Log (single source of truth for orchestration)

Running checklist. Update at every gate. Dates are absolute (today: 2026-10-10).

## Phase status

| Phase | Title | Status | Gate met? |
|---|---|---|---|
| 1 | Inventory & baseline | IN PROGRESS (started 2026-10-10) | ☐ |
| 2 | Prune GPU backends (SYCL, Vulkan) | not started | ☐ |
| 3 | Study FAISS | not started | ☐ |
| 4 | Gap analysis | not started | ☐ |
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
- [ ] One descriptive commit for Phase 1

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
