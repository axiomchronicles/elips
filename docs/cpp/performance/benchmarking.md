# Benchmarking

ELIPS ships with a self-contained benchmark suite that measures insert
throughput, search latency percentiles, recall vs exact ground truth, and
crash-recovery time. A separate GPU benchmark compares CPU vs GPU-accelerated
search performance across embedding dimensions.

---

## Benchmark Suite — `BenchMain.cpp`

**Location:** `benchmarks/BenchMain.cpp`

This is a standalone executable (`elips_bench`) with no dependencies beyond
the ELIPS library itself. It does not use any external benchmarking framework
(no Google Benchmark dependency).

### Command-Line Interface

```
elips_bench [--count N] [--dim D] [--queries Q]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--count` | 20000 | Number of vectors to insert |
| `--dim` | 128 | Vector dimension |
| `--queries` | 1000 | Number of query vectors to test |

### Benchmark Components

#### 1. Insert + Search Benchmark (`bench_insert_and_search`)

**Setup:**
- Generates `count` random vectors with `std::normal_distribution<float>(0.0, 1.0)`
  using a fixed seed (`rng(7)`) for reproducibility.
- Generates `queries` query vectors from the same distribution.
- Creates both an `IndexHNSW` (HNSW, default params) and an
  `IndexFlat` with `Metric::euclidean`.

**Insert measurement:**
```cpp
const auto t_insert = clock_type::now();
for (std::size_t i = 0; i < count; ++i) {
    ids[i] = RecordID::generate();
    graph.add_with_ids(ids[i], data[i]);
}
const double insert_s = seconds_since(t_insert);
```

Measures wall-clock time for all HNSW adds. `IndexFlat` is also populated
(after the timer stops) to serve as the ground-truth oracle. Results are
reported as records/second.

**Search + recall measurement:**
```cpp
for (const auto& probe : probes) {
    const auto t_q = clock_type::now();
    const auto approx = graph.search(probe, k);   // k = 10
    latencies_us.push_back(seconds_since(t_q) * 1e6);

    const auto truth = exact.search(probe, k);
    // compute recall@10
}
```

For each query, both indices return their top-10. The HNSW results are compared
against the `IndexFlat` results. Recall is computed as:

```
recall@10 = |HNSW_top10 ∩ Exact_top10| / |Exact_top10|
```

Latency percentiles (p50, p95, p99) are computed from the sorted vector of
per-query latencies.

**Output format:**
```
[insert]  20000 x 128  12345 rec/s (1.62s)
[search]  p50=12.3us  p95=45.6us  p99=89.2us
[recall]  recall@10 = 0.965
```

#### 2. Crash Recovery Benchmark (`bench_recovery`)

**Setup:**
- Creates a temporary database at `$TMPDIR/elips_bench_recovery`.
- Uses `Durability::standard` (WAL with flush on every write).
- Inserts `count/4` vectors.
- Calls `db->abandon()` — simulates a crash by skipping the checkpoint so
  that only the WAL contains the data.

**Recovery measurement:**
```cpp
const auto t_recover = clock_type::now();
auto db = elips::open(dir.string());
const double recover_s = seconds_since(t_recover);
```

Measures the time to open the database (load identity, replay WAL). Reports
the number of replayed records and recovery time.

**Cleanup:** Removes the temporary directory after measurement.

### Fixed Random Seeds

Both the insert/search benchmark (`rng(7)`) and recovery benchmark (`rng(11)`)
use fixed seeds. This ensures reproducibility across runs and machines — the
exact same vectors are generated each time.

---

## GPU Benchmark — `BenchGpuSearch.cpp`

**Location:** `benchmarks/gpu/BenchGpuSearch.cpp`

A separate standalone executable (`elips_gpu_bench`) that compares CPU
brute-force search against GPU-accelerated search across dimensions.

### Benchmark Structure

```cpp
BenchResult bench_search(size_t n_vectors, size_t dim, Metric metric,
                         size_t n_queries) {
    // 1. Generate random dataset (uniform [-1, 1])
    // 2. Build CPU IndexFlat
    // 3. Time CPU search over n_queries queries
    // 4. Probe GPU devices via GpuDeviceManager + BackendSelector
    // 5. Build GpuBruteForceIndex on GPU
    // 6. Time GPU search over n_queries queries
    // 7. Report speedup = cpu_ms / gpu_ms
}
```

**Dataset:** `n_vectors = 10000`, `n_queries = 100`, uniform distribution
`[-1.0, 1.0]`, metric = cosine. Each query seeks `k = 10` results.

**Dimensions tested:** 128, 384, 768, 1536.

**Output format:**
```
=== ELIPS GPU Benchmark ===

Config      CPU (ms)    GPU (ms)    Speedup
------------------------------------------------
dim=128     0.123       0.018       6.833x
dim=384     0.456       0.052       8.769x
dim=768     1.234       0.098       12.592x
dim=1536    3.456       0.234       14.769x
```

### GPU Build

The `GpuBruteForceIndex` is used for the benchmark:
```cpp
GpuBruteForceIndex gpu_idx(**backend, metric, dim, config);
gpu_idx.build_from_batch(db_vecs, ids, GpuIndexBuildParams{});
```

`BackendSelector::select()` chooses the best available backend (Metal,
CUDA, HIP) based on the device probes and `GpuConfig::policy`.

### Interpreting GPU Results

- **Speedup** is computed as `cpu_ms / max(gpu_ms, 0.001)`. The `max` avoids
  division by zero if no GPU is available.
- GPU advantage increases with dimension — higher dimensions have more data
  parallelism to exploit.
- GPU build time (upload + index construction) is NOT included in the per-query
  latency measurement — only the search time is compared.
- On Apple Silicon (unified memory), the GPU advantage is smaller because CPU
  NEON SIMD is already well-optimised, and the GPU shares memory bandwidth
  with the CPU. On discrete GPUs (CUDA), speedups of 10–50× are typical.

---

## Recall Benchmarks (HNSW vs IndexFlat)

### Unit Test Recall Target

**Location:** `tests/unit/hnsw_recall_test.cpp`

The dedicated recall test asserts that HNSW achieves `recall@10 ≥ 0.90` on
2000 random vectors in 64 dimensions with Euclidean metric and default HNSW
parameters (M=16, ef_construction=200, ef_search=50):

```cpp
TEST(HnswRecallTest, RecallAtTenIsHighVsExactGroundTruth) {
    constexpr std::size_t count = 2000;
    constexpr std::uint16_t dim = 64;
    // ... add into both IndexHNSW and IndexFlat ...
    EXPECT_GE(recall, 0.90) << "recall@" << k << " = " << recall;
}
```

This test is run as part of the standard CTest suite. A recall below 0.90
indicates a regression in the insertion algorithm, the beam search, or the
diversity heuristic.

### Benchmark Suite Recall Measurement

The `BenchMain.cpp` benchmark also measures recall@10 but at a larger scale
(20K vectors, 128 dimensions by default). The benchmark reports the actual
recall value without a threshold assertion — it is informational. Typical
observed values:

| Dimensions | Vectors | Default params | Typical recall@10 |
|-----------|---------|---------------|-------------------|
| 64 | 2000 | M=16, ef=200, ef=50 | 0.95–0.99 |
| 128 | 20000 | M=16, ef=200, ef=50 | 0.94–0.98 |
| 384 | 100000 | M=16, ef=200, ef=50 | 0.90–0.95 |

### Tuning for Higher Recall

If recall falls below the 0.90 target, adjust `HnswParams`:

| Parameter | Increase effect | Trade-off |
|-----------|----------------|-----------|
| `ef_construction` | Larger candidate set during insert yields better-connected graph | Slower inserts, more memory (larger neighbour lists) |
| `ef_search` | Larger beam width during query | Slower queries, linear in ef_search |
| `max_connections` (M) | More edges per node, denser graph | More memory, slower inserts (more edge pruning) |

For production deployments targeting recall@10 ≥ 0.95, recommended settings:
- `M = 32` (doubled from default 16)
- `ef_construction = 400` (doubled from default 200)
- `ef_search = 100` (doubled from default 50)

---

## How to Run Benchmarks

### Building

```bash
cd /path/to/elips
cmake -B build -DCMAKE_BUILD_TYPE=Release -DELIPS_BUILD_BENCH=ON
cmake --build build --target elips_bench
```

For GPU benchmarks (enable a backend — Metal shown; use `-DELIPS_ENABLE_CUDA=ON`
or `-DELIPS_ENABLE_ROCM=ON` for discrete GPUs):
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DELIPS_BUILD_BENCH=ON -DELIPS_ENABLE_METAL=ON
cmake --build build --target elips_gpu_bench
```

### Running

```bash
# Default parameters (20K vectors, dim=128, 1K queries)
./build/benchmarks/elips_bench

# Custom scale
./build/benchmarks/elips_bench --count 100000 --dim 384 --queries 5000

# GPU benchmark
./build/benchmarks/gpu/elips_gpu_bench
```

### CTest Integration

The recall unit test runs with the standard CTest suite:
```bash
cmake --build build && cd build && ctest -R HnswRecallTest
```

No separate benchmark CI job exists currently — benchmarks are intended for
local development and profiling.

---

## Interpreting Results

### Insert Throughput

Insertion throughput (`rec/s`) is driven primarily by:
- **Vector dimension:** O(D) per distance computation in beam search.
- **ef_construction:** Higher values visit more neighbours per insert.
- **Graph complexity:** O(M × log N) beam searches per insert.

### Search Latency

Per-query latency is dominated by:
- **Beam search:** O(ef_search × M) distance computations at layer 0.
- **Greedy descent:** O(log N × M) distance computations through upper layers.
- **Distance computation:** O(D) per distance call (amortised by NEON SIMD).

The p50/p95/p99 distribution reveals tail latency, which is important for
interactive applications. HNSW's greedy search typically has low variance.

### Recovery Time

Recovery time scales with WAL size (number of records to replay) and is
dominated by:
- **WAL parsing:** O(records) byte-level parsing and CRC validation.
- **Index insertion during replay:** Each replayed record is inserted into
  the index, which has the same cost as a live insert.
- **Recovery is sequential** (single-threaded replay), so throughput is
  the same as single-threaded insert throughput.

For large WALs, periodic checkpointing reduces recovery time by truncating
the WAL. The benchmark simulates a worst-case scenario (checkpoint never
happened).

### GPU vs CPU

GPU benchmarks measure brute-force (`IndexFlat`) search only — HNSW on GPU
(CAGRA) is a separate feature. The GPU advantage comes from massive
parallelism: thousands of distance computations in parallel vs CPU SIMD
lanes (4 for NEON, 8 for AVX). For smaller datasets (< 10K vectors), CPU
search may be faster due to lower overhead (GPU kernel launch latency).

---

## Python ANN Benchmarks — `bench_ann.py` (Phase 7.5 / 8)

**Location:** `benchmarks/bench_ann.py`

A fixed-seed Python harness that compares ELIPS against **faiss** on identical
data through the FAISS-style Index API (`elips.IndexFlat`, `elips.IndexIVFFlat`,
`elips.bf_knn`), following the recall@k + QPS methodology in
`docs/_redesign/_faiss_3H.md` §4 (exact-Flat `R@1==1.0` sanity gate, warmup
before timing, seeded data). It is the reference harness Phase 8/9 should cite.

Configurations, all on the same vectors (metric L2):

- **ELIPS CPU** `IndexFlat` (exact) and `IndexIVFFlat` (nprobe sweep)
- **ELIPS Metal GPU** `bf_knn(device="metal")` — exact brute force, single-query
  and batched
- **faiss-cpu** `IndexFlatL2` + `IndexIVFFlat` (nprobe sweep), when importable

```bash
# smoke (CI-sized): dim=64 nb=10k nq=200
PYTHONPATH=bindings/python python benchmarks/bench_ann.py --smoke
# full: dim=128 nb=100k nq=1k
PYTHONPATH=bindings/python python benchmarks/bench_ann.py --full
```

Metrics per config: recall@10 vs the exact numpy oracle, QPS, p50/p99 latency,
and build/train time. The run prints a table and writes it to
**`docs/_redesign/_benchmarks.md`** with exact hardware/OS/versions.

### Reference hardware and results

The committed reference run in `docs/_redesign/_benchmarks.md` was measured on
**Apple M4 (arm64)**, macOS 26.5.1, Python 3.13, ELIPS 1.1.0 (GPU bindings on),
faiss-cpu 1.15.1, numpy 2.5.3 — dim=128, nb=100k, nq=1k, k=10, metric L2. QPS and
p50/p99 are single-query unless a row is marked "batched". Because the Metal path
is validated on Apple M4, these are the numbers cited across the docs. For the
honest interpretation (faiss wins exact-CPU and IVF build; ELIPS's batched Metal
`bf_knn` is the bright spot at 914 QPS / recall 1.000), see
[Which index should I use?](../../overview/choosing-an-index.md).

**faiss availability:** faiss-cpu ships **no wheel for Python 3.14**; run the
harness under Python 3.13 to include faiss. With faiss absent the harness
compares ELIPS CPU vs ELIPS Metal only and labels faiss `UNAVAILABLE` (it never
fabricates faiss numbers — ground rule #1).
