# ADR-0003: HNSW as the primary ANN index

**Status:** Accepted
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

## Alternatives Considered
- **IVF / IVF-PQ:** great memory profile, but lower recall without large nprobe.
- **LSH:** simple but weaker recall/latency at these scales.
- **Flat only:** exact but linear; impractical beyond ~10⁴ vectors.
