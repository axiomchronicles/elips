# ADR-0003: HNSW as the primary ANN index

**Status:** Accepted  **Revised:** 2026-07-30
**Date:** 2024-08-01

## Context
The index must deliver a strong recall/latency trade-off for in-memory datasets
without GPUs, and support incremental inserts and deletes.

## Decision
Implement `HierarchicalGraphIndex`, a from-scratch HNSW: a layered navigable
small-world graph with probabilistic level assignment (`mL = 1/ln(M)`), beam
search (`ef`), and the diversity neighbor-selection heuristic. `ExactIndex`
(brute force) ships alongside as the recall ground-truth oracle and for small
collections. Both sit behind the `IndexPort` interface.

## Consequences
- High recall (≈0.97 recall@10 on structured data) with sub-millisecond search.
- Deletes are soft tombstones; the node stays for navigation, excluded from results.
- Quantized/disk-resident indexes (PQ, DiskANN) are deferred but plug into the same port.
- **Tombstone cost is now bounded.** Before v1.1.0, `remove()` set a flag and nothing reclaimed it: dead vectors and edges accumulated indefinitely under insert/delete churn. `search()` would silently return fewer than k live results because the fixed ef beam filled with tombstones. v1.1.0 fixes this via:
  - `search()` scales ef by `total/live` ratio so the beam still yields k live hits despite tombstones.
  - `vacuum()` rebuilds the index from live records, reclaiming dead vectors and edges. Triggered automatically when `tombstones/total >= compaction_ratio` (default 0.2), or explicitly by the caller.
  - `pending_removals()` exposes the outstanding tombstone count.
  - `GraphParams::compaction_ratio` controls the automatic trigger (0 disables auto-compaction).
  - *Measured:* at 50% delete ratio, search returns full k results with recall within 0.15 of baseline; 4,000 records churned through a 200-record live set keep the graph under 400 nodes instead of growing to 4,200.
- **Filtered ANN search re-probes adaptively.** The old fixed `top * 20` over-fetch returned short sets for selective filters. `seek()` now widens the beam (4× per round, bounded by vault size) until k matches are found or the vault is exhausted.


## Alternatives Considered
- **IVF / IVF-PQ:** great memory profile, but lower recall without large nprobe.
- **LSH:** simple but weaker recall/latency at these scales.
- **Flat only:** exact but linear; impractical beyond ~10⁴ vectors.

## Revision History
- **2026-07-30 (v1.1.0):** Added tombstone bounds analysis, adaptive ef scaling, vacuum()/pending_removals()/compaction_ratio, and adaptive filtered ANN re-probe. The original ADR noted "deletes are soft tombstones" but did not describe the growth bound or recall degradation path they caused.
