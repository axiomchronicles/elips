# Which index should I use?

A short, honest decision guide, backed by measured numbers. ELIPS exposes the
FAISS-recognizable index types directly (`IndexFlat`, `IndexHNSW`,
`IndexIVFFlat`, `IndexIVFPQ`) plus `index_factory`, and a standalone GPU
`bf_knn`. This page helps you pick one and sets honest expectations against
`faiss-cpu`.

Ground rule: numbers below are reported as measured — wins and losses. The full
run (environment, seeds, methodology) lives in
[`_benchmarks.md`](../_redesign/_benchmarks.md) and is produced by
`benchmarks/bench_ann.py`.

## Quick decision

| Situation | Use |
|---|---|
| Small collection, need exact results, simplest thing | `IndexFlat` (CPU) |
| General-purpose ANN, good recall/latency, no training | `IndexHNSW` (CPU) |
| Large collection, want to trade recall for speed/memory, OK to train | `IndexIVFFlat` / `IndexIVFPQ` (CPU) |
| Apple Silicon, many queries at once, want exact answers fast | `bf_knn(device="metal")`, **batched** |
| One query at a time, latency-sensitive | CPU (`IndexFlat` / `IndexHNSW`) — GPU launch overhead dominates single queries |

If you are unsure: start with `IndexFlat` (exact, zero tuning) and move to
`IndexHNSW` or IVF only when Flat is too slow for your dataset size.

## Measured results (Apple M4)

Hardware: **Apple M4 (arm64)**, macOS 26.5.1, Python 3.13, ELIPS 1.1.0 (GPU
bindings on), faiss-cpu 1.15.1, numpy 2.5.3. Dataset: dim=128, nb=100k, nq=1k,
k=10, L2. Exact-Flat sanity gate: PASS. QPS/p50/p99 are **single-query** unless
the row says "batched".

| Config | recall@10 | QPS | p50 ms | build s |
|---|---|---|---|---|
| ELIPS `IndexFlat` (cpu) | 1.000 | 489 | 1.98 | 0.02 |
| ELIPS `bf_knn` metal, 1-query | 1.000 | 24 | 41.3 | 0.00 |
| **ELIPS `bf_knn` metal, batched** | **1.000** | **914** | **1.09** | 0.00 |
| ELIPS `IndexIVFFlat` nprobe=1 (cpu) | 0.030 | 46,614 | 0.020 | 71.4 |
| ELIPS `IndexIVFFlat` nprobe=8 (cpu) | 0.152 | 14,123 | 0.070 | 40.3 |
| ELIPS `IndexIVFFlat` nprobe=32 (cpu) | 0.381 | 6,273 | 0.159 | 28.9 |
| ELIPS `IndexIVFFlat` nprobe=128 (cpu) | 0.801 | 1,616 | 0.617 | 25.1 |
| faiss `IndexFlatL2` (cpu) | 1.000 | 989 | 0.926 | 0.01 |
| faiss `IndexIVFFlat` nprobe=1 (cpu) | 0.030 | 107,010 | 0.009 | 0.26 |
| faiss `IndexIVFFlat` nprobe=8 (cpu) | 0.152 | 31,079 | 0.032 | 0.26 |
| faiss `IndexIVFFlat` nprobe=32 (cpu) | 0.379 | 9,390 | 0.106 | 0.25 |
| faiss `IndexIVFFlat` nprobe=128 (cpu) | 0.793 | 2,587 | 0.385 | 0.25 |

(nlist=316 for all IVF rows. Recall tracks nprobe identically for ELIPS and
faiss, confirming the IVF implementations agree.)

## Honest reading of the numbers

**faiss-cpu is faster on exact CPU search.** faiss `IndexFlatL2` does ~989 QPS vs
ELIPS `IndexFlat` ~489 QPS — about 2x. faiss's BLAS-backed exact search is more
optimized than ELIPS's current CPU scan.

**faiss-cpu is much faster to build IVF, and faster per-query at a given recall.**
faiss builds IVF in ~0.25s; ELIPS currently takes **25–70s** to build the same
`IndexIVFFlat` — a large gap, and a known optimization target. At matched nprobe,
faiss also delivers higher QPS (e.g. nprobe=128: faiss 2,587 vs ELIPS 1,616).
Recall is identical, so the difference is pure engine speed, not quality.

**ELIPS's bright spot is batched exact search on Apple Metal.** Batched
`bf_knn(device="metal")` hits **914 QPS at recall 1.000** — roughly on par with
faiss's exact CPU Flat and ~1.9x ELIPS's own CPU Flat — while staying exact. This
is the validated GPU win: on-GPU distance + top-k over a batched call, amortizing
launch and sync.

**GPU does not help one query at a time.** Single-query `bf_knn` metal is only 24
QPS — launch + sync overhead dwarfs the work. Do not reach for the GPU for
latency-sensitive single lookups; batch, or stay on CPU.

**IVF is a recall/speed knob, not free accuracy.** At nlist=316, nprobe=1 gives
3% recall at huge QPS; you need nprobe≈128 for ~80% recall. Tune nprobe to your
recall target; for exact answers use Flat (CPU) or batched `bf_knn` (Metal).

## Takeaways

- Need exact + simple → **`IndexFlat`** (CPU).
- Need exact + fast + many queries on Apple Silicon → **batched `bf_knn(device="metal")`**.
- Need scalable ANN with a recall/speed tradeoff → **IVF**, tuning `nprobe`
  (accept the current slow build; it is being optimized).
- Comparing to faiss on raw CPU throughput → faiss currently wins exact-CPU and
  IVF-build; ELIPS matches recall and wins batched Metal exact search.
- Reproduce: `PYTHONPATH=bindings/python python benchmarks/bench_ann.py --full`
  (see [Benchmarking](../cpp/performance/benchmarking.md)).
