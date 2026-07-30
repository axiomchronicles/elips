# Testing, Sanitizers & Quality Assurance — v1.1.0 Updates

This document details the test infrastructure, sanitizer suites, fuzz testing harnesses, and CI automation introduced in ELIPS v1.1.0.

---

## 1. Sanitizer Suite Integration (`ELIPS_SANITIZE`)

ELIPS v1.1.0 adds native CMake support for LLVM/GCC sanitizers via `-DELIPS_SANITIZE=address|thread|undefined`.

### Supported Sanitizer Flags
- `-DELIPS_SANITIZE=address` — Enables **AddressSanitizer (ASan)** and **UndefinedBehaviorSanitizer (UBSan)**. Detects out-of-bounds access, use-after-free, double-free, memory leaks, and undefined scalar math.
- `-DELIPS_SANITIZE=thread` — Enables **ThreadSanitizer (TSan)**. Detects data races between concurrent threads accessing shared memory locations.

### Executing Local Sanitizer Runs

```bash
# 1. ASan + UBSan Clean Verification Build
cmake -S . -B build_asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DELIPS_SANITIZE=address

cmake --build build_asan
ctest --test-dir build_asan --output-on-failure

# 2. ThreadSanitizer Concurrency Audit
cmake -S . -B build_tsan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DELIPS_SANITIZE=thread

cmake --build build_tsan
ctest --test-dir build_tsan --output-on-failure
```

---

## 2. LLVM libFuzzer Harness (`elips_fuzz_wal`)

v1.1.0 introduces a dedicated `libFuzzer` target to continuously test parser robustness against corrupted, mutated, or malicious storage payload streams.

### Fuzz Target Details
- Source: `tests/fuzz/wal_replay_fuzz.cpp`
- CMake Flag: `-DELIPS_BUILD_FUZZERS=ON`
- Tested Component: `WAL::replay()` byte-stream parsing.

### Verification Milestone
The `elips_fuzz_wal` target completed **895,000 continuous fuzz executions** under ASan + UBSan without registering a single crash, assertion failure, memory leak, or unhandled exception.

```bash
# Build and Run libFuzzer Target
cmake -S . -B build_fuzz -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DELIPS_BUILD_FUZZERS=ON \
  -DELIPS_SANITIZE=address

cmake --build build_fuzz
./build_fuzz/elips_fuzz_wal -runs=500000 -max_len=65536
```

---

## 3. Parser Robustness & Mutation Test Suite

In addition to continuous fuzzing, v1.1.0 adds deterministic parser mutation tests (`tests/fuzz/parser_robustness_test.cpp`):

1. **Single-Byte Bit Flipping:** Systematically flips every bit position across valid WAL and snapshot binary files to verify clean recovery termination.
2. **Truncation Sweeps:** Truncates valid log files at every single byte boundary offset ($0 \dots L$) to verify bounded parsing without infinite loops or invalid memory access.
3. **Random Byte Stream Replay:** Feeds pure pseudo-random garbage bytes to `WAL::replay()`, asserting that `elips::StorageError` is raised cleanly.

---

## 4. Continuous Integration (CI) Workflow Matrix

GitHub Actions workflows execute the following mandatory verification jobs on every commit:

| Job Name | Operating System | Build Flags | Verification Target |
| :--- | :--- | :--- | :--- |
| `asan_ubsan_check` | Ubuntu 22.04 | `-DELIPS_SANITIZE=address` | C++ test suite + Python bindings under ASan/UBSan |
| `thread_sanitizer` | Ubuntu 22.04 | `-DELIPS_SANITIZE=thread` | Multi-threaded concurrency test suite (`m2_thread_safety_test`) |
| `macos_metal_test` | macOS 14 (Apple Silicon) | `-DELIPS_GPU_METAL=ON` | Metal GPU backend verification & device memory allocation |
| `linux_no_gpu_check`| Ubuntu 22.04 | Default (No GPU flags) | Asserts GPU engine is excluded (`_has_gpu = False`) |
| `python_ruff_lint` | Ubuntu 22.04 | `ruff check` | Python style, type annotations, and linting |
