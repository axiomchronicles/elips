# 05 — Test Matrix (what's tested, on what hardware, status, recall/latency)

Hardware available for validation: **Apple M4 (Metal + CPU)**. CUDA/HIP: compile-check only, no hardware.

## Baseline (Phase 1) — captured 2026-10-10

**Toolchain:** cmake 4.4.2, ninja 1.13.2, AppleClang 21.0.0, Python 3.14.7. Host: Apple M4 (darwin 25.5.0).

**Configure/build:** `cmake -S . -B build-baseline -G Ninja -DCMAKE_BUILD_TYPE=Release -DELIPS_BUILD_PYTHON=ON` → CONFIGURE_EXIT=0, BUILD_EXIT=0. Metal backend ON (Apple default); `elips_gpu` + `elips_gpu_metal` static libs built; `_core.cpython-314-darwin.so` produced. Build warnings only (unused param in `MetalBackend.mm:32`; several `[[nodiscard]]`-ignored in benchmarks/tests; one gtest char8_t warning). Linker warns "ignoring duplicate libraries" (elips_core/elips_gpu linked twice — cosmetic, Phase 8 cleanup).

**ctest:** `ctest --test-dir build-baseline` → **255 passed / 256, 1 skipped, 0 failed**, 43.65s. Skipped: `GpuMemoryPoolTest.acquire_and_release`. Slowest: `QuantizedIndexTest.QuantizedGraphIndexKeepsUsefulRecall` 13.6s.

**CAUTION (not yet verified):** The 41 GPU-labelled tests (#216–256) PASS, but this does NOT prove real Metal compute. `MetalBackend.mm:32 initialize()` ignores its config arg, and GPU unit tests appear to use a recording/mock backend (`tests/unit/gpu/TestRecordingBackend.hpp`) and CPU-mirror paths. Whether `GpuBackendIntegrationTest.cosine_distance_matches_cpu` (#251) and `CpuGpuRecallParityTest` (#255) exercise GPU kernels vs CPU fallback is pending 1C's verdict. Do NOT record Metal as "validated" until 1C + a real-kernel test confirm it.

**pytest:** `PYTHONPATH=bindings/python python3 tests/python/test_bindings.py` → **34/34 passed, 0 failed**. (8 python test files exist under tests/python/; only test_bindings.py run for baseline — full pytest sweep pending Phase 8.) GPU-labelled python tests (gpu_config/device_info/probe/fallback) pass but, as above, do not prove real Metal compute. Note: `python3 -I` ignores PYTHONPATH (implies -E), so plain `import elips` under -I fails — harness artifact, not a code defect; the bindings import fine via the documented command.

## Per-index parity / recall (filled Phase 6–7)
| Index | Backend | Hardware | recall@10 | p50 lat | status |
|---|---|---|---|---|---|
| IndexIVFFlat (euclidean) | CPU | Apple M4 | 0.999 @nprobe8, 1.000 @nprobe16, 1.000 @nprobe64 | — | PASS (nlist=64, d=32, n=5000 clustered; oracle=IndexFlat) |
| IndexIVFFlat (cosine) | CPU | Apple M4 | 0.985 @nprobe8, 0.992 @nprobe16, 1.000 @nprobe64 | — | PASS (same config) |
| IndexIVFFlat (dot_product) | CPU | Apple M4 | 0.995 @nprobe8, 0.996 @nprobe16, 1.000 @nprobe64 | — | PASS (same config) |

Phase 6.1 (2026-10-10): added CPU `IndexIVFFlat` + FAISS-style `index_factory("Flat"/"IVF<nlist>,Flat"/"HNSW<M>")`. nprobe→nlist is exhaustive, so recall→1.0 matches the IndexFlat oracle exactly. Suite now 268 tests (257 prior + 11: 7 ivfflat, 4 factory), all green / 1 skipped (GpuMemoryPoolTest). Data is mildly clustered (50 centers, σ=0.15 jitter) — the regime IVF targets; pure i.i.d. noise is adversarial and needs nprobe≈nlist for the same recall.
