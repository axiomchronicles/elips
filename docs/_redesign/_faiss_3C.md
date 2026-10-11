# 3C — FAISS HNSW / Graph Indexes

Source: **FAISS v1.15.1** at `/tmp/faiss` (read-only, not built). All citations are `file:line`
from that tree. Reality wins over assumption; corrections to the brief are flagged inline.

ELIPS already ships a complete HNSW (`HierarchicalGraphIndex`). This section documents what
FAISS exposes so we can align param *names* and confirm we are not missing capabilities.

---

## 1. `IndexHNSW` and the `HNSW` struct

`IndexHNSW : Index` is a thin index that holds two things (`faiss/IndexHNSW.h:30-58`):

- `HNSW hnsw;` — the link structure only (no vectors).
- `Index* storage;` — a *separate* random-access index that owns the actual vectors.
  `own_fields` controls whether the HNSW deletes it (`IndexHNSW.h:34-38`).

So FAISS factors HNSW into "graph" + "pluggable vector store". The graph never stores vectors;
it calls `storage->get_distance_computer()` to score candidates (`IndexHNSW.h:128`).

### Core parameters (the `HNSW` struct, `faiss/impl/HNSW.h:63-173`)

| FAISS field | default | meaning | citation |
|---|---|---|---|
| `M` (ctor arg only) | 32 | neighbors/node; **not stored as a field** — baked into `cum_nneighbor_per_level` | `HNSW.h:197`, `HNSW.cpp:62-65` |
| `efConstruction` | 40 | beam width while building | `HNSW.h:141` |
| `efSearch` | 16 | beam width at query time | `HNSW.h:144` |
| `assign_probas` | — | per-level assignment probability (sums to 1) | `HNSW.h:114` |
| `cum_nneighbor_per_level` | — | cumulative max-degree per level | `HNSW.h:118` |
| `levels` | — | level of each vector (base level stored as 1) | `HNSW.h:121` |
| `offsets` / `neighbors` | — | CSR-style flat neighbor table | `HNSW.h:123-129` |
| `entry_point` | -1 | one of the max-level nodes | `HNSW.h:133` |
| `max_level` | -1 | current top level | `HNSW.h:138` |
| `prune_headroom` | 0.2 | leave slack when pruning to cut O(n²) + lock contention | `HNSW.h:146-148` |
| `check_relative_distance` | true | early-stop test during search | `HNSW.h:152` |
| `search_bounded_queue` | true | bounded vs unbounded beam | `HNSW.h:155` |
| `is_similarity` | false | flips comparator (CMax→CMin) for IP/cosine | `HNSW.h:170` |

> **Correction to brief:** `M` is *not* a live field on `HNSW`. The ctor calls
> `set_default_probas(M, 1/log(M))` which bakes `2*M` links on level 0 and `M` on levels >0
> into `cum_nneighbor_per_level` (`HNSW.cpp:80-92`). After the first `add`, `set_nb_neighbors`
> asserts `levels.size()==0` — so **M is immutable post-build** (`HNSW.cpp:39-45`). efConstruction
> and efSearch stay mutable.

### Level generation (`HNSW.cpp:67-92`)

`random_level()` draws `f = rand_float()` and walks `assign_probas` (`HNSW.cpp:67-78`).
`set_default_probas` builds a geometric decay `proba = exp(-level/levelMult)*(1-exp(-1/levelMult))`
with `levelMult = 1/log(M)`, stopping when proba < 1e-9 (`HNSW.cpp:80-92`). Standard Malkov HNSW.

### Entry point & greedy descent

Entry point is a max-level node (`HNSW.h:131-133`). Build and search both start there and do a
greedy single-neighbor descent on upper levels via `greedy_update_nearest_impl` until no neighbor
improves (`HNSW.cpp:702-775`), then switch to beam search at the point's own levels
(build: `HNSW.cpp:813-820`; search: `HNSW.cpp:1745-1751`).

### Neighbor selection / pruning (`shrink_neighbor_list`, `HNSW.cpp:244-290`)

The classic HNSW heuristic (Algorithm 4): enumerate candidates nearest→farthest, keep `v1` only
if **no already-kept neighbor `v2` is closer to `v1` than the query is** (`HNSW.cpp:263-273`).
This produces a diverse, navigable graph rather than just the k closest. `keep_max_size_level0`
back-fills pruned "outsiders" to hit `2*M` on level 0 (needed for CAGRA copy) (`HNSW.cpp:280-289`).
`prune_headroom` lets degree run slightly over before a prune is forced (`HNSW.cpp:310-313`).

### Build is deterministic & lock-free (notable divergence from stock HNSW)

v1.15.1 uses a two-phase deterministic build (`hnsw_add_vertices_deterministic`,
`HNSW.h:286-297`): phase A writes each point's forward links against an immutable snapshot
(`compute_forward_links_deterministic`, `HNSW.h:219-225` / `HNSW.cpp:800-856`); phase B merges
reciprocal edges in a total order `(distance, id)` so results are order-independent
(`merge_reverse_links_deterministic`, `HNSW.h:231-236`). The older locked incremental path
(`add_links_starting_from`, `HNSW.cpp:635-667`) still exists but add() uses the deterministic one.

### `search_bounded_queue`

Two beam-search backends, selected by `search_bounded_queue` / `SearchParametersHNSW::bounded_queue`
(`HNSW.cpp:1731-1739`):

- **bounded** → `MinimaxHeapT` of fixed capacity `ef = max(efSearch, k)`, via
  `search_from_candidates` (`HNSW.cpp:1752-1786`, impl `1117-1239`). Stop condition: once
  `ef` distances already seen beat the next candidate `d0`, break (`HNSW.cpp:1163-1166`); or when
  `check_relative_distance` is off, after `efSearch` hops (`HNSW.cpp:1224-1226`).
- **unbounded** → grows two heaps until exhausted, then trims to k
  (`search_from_candidate_unbounded`, `HNSW.cpp:1787-1822`). Higher recall, more work.

The 4-at-a-time `distances_batch_4` batching in both build and search is a FAISS perf trick
(`HNSW.cpp:527-544`, `1202-1221`), not an algorithmic change.

**Decision — HNSW struct/algorithm: ADOPT the model, SKIP re-architecting ELIPS.**
ELIPS already has this. Keep ELIPS's own graph; the FAISS reading only confirms the design is
sound. One idea worth noting (not required this phase): FAISS's deterministic build gives
reproducible indexes — flag as a possible future ELIPS enhancement, do not implement now.

---

## 2. Storage over a flat vs quantized index + the "2-level" idea

HNSW is agnostic to how vectors are stored; the variant classes just pick a different `storage`:

| Variant | storage index | citation |
|---|---|---|
| `IndexHNSWFlat` | `IndexFlatL2` / `IndexFlat` (raw fp32) | `IndexHNSW.cpp:889-892` |
| `IndexHNSWPQ` | `IndexPQ(d, pq_m, pq_nbits)` | `IndexHNSW.cpp:960-966` |
| `IndexHNSWSQ` | `IndexScalarQuantizer(d, qtype)` | `IndexHNSW.cpp:980-985` |
| `IndexHNSWRaBitQ` | `IndexRaBitQ(d, nb_bits)` | `IndexHNSW.cpp:1011-1022` |
| `IndexHNSW2Level` | `Index2Layer(quantizer, nlist, m_pq)` | `IndexHNSW.cpp:1028-1036` |

Quantized storage shrinks memory and speeds distance calc at a recall cost; the graph itself is
identical. PQ needs `pq.compute_sdc_table()` after train (`IndexHNSW.cpp:971-974`). RaBitQ with
`nb_bits>=2` sets `search_method = SM_RABITQ` for a staged 1-bit-estimate-then-refine search
(`IndexHNSW.cpp:1021`, evaluator at `HNSW.cpp:1081+`).

**"2-level" (`IndexHNSW2Level`)** = HNSW over an `Index2Layer`, which codes each vector as
*(coarse centroid id, PQ residual)* — a flat IVF-like 2-level code with fast random access
(`IndexHNSW.h:243-247`). It has its own `search` that, on the L2 Index2Layer path, runs a
specialized candidate loop and can `flip_to_ivf()` to convert into a real IVF index afterwards
(`IndexHNSW.cpp:1111-1121`, `1237-1238`). It is a niche memory-optimization combo.

**Decision — storage variants:**
- Flat storage: **ADOPT** (ELIPS graph already stores fp32; it is the baseline). 
- PQ/SQ-backed HNSW: **ADAPT later** — ELIPS has its own `quant::QuantParams` + GPU IVFPQ; a
  quantized-storage CPU HNSW can reuse that codec behind the same graph when memory pressure demands.
- `IndexHNSW2Level` / `flip_to_ivf`: **SKIP** — niche, and ELIPS already separates IVF and graph
  index types (`Config::IndexType {graph, exact}`).

---

## 3. `IndexNSG` (contrast only, `faiss/IndexNSG.h`)

NSG is a *single-layer* monotonic graph (no hierarchy, no efConstruction/level machinery). Its
knobs are different (`IndexNSG.h:24-54`):

- `R = 32` — max out-degree (its "M") (`IndexNSG.h:49`).
- `GK = 64` — size of the k-NN graph it is built from (`IndexNSG.h:36`).
- `build_type`, plus nndescent params `nndescent_S/R/L/iter` for building the initial kNN graph
  (`IndexNSG.h:41-47`).
- search beam is `search_L` (set on the `NSG` sub-struct), not `efSearch`.

NSG trades HNSW's layered entry-point descent for one navigating-spread-out graph with a fixed
medoid entry. **Decision: SKIP.** ELIPS standardizes on HNSW; NSG is only useful as prior art.

---

## 4. Runtime param setting (how `efSearch` changes at query time)

Two mechanisms, both real:

**(a) Per-query via `SearchParametersHNSW`** (`HNSW.h:55-61`):
```cpp
struct SearchParametersHNSW : SearchParameters {
    int  efSearch = 16;
    bool check_relative_distance = true;
    bool bounded_queue = true;
};
```
Passed into `search(...)`; `extract_search_params` does a `dynamic_cast` and, if present,
*overrides* the index-level `hnsw.efSearch` for that call only (`HNSW.cpp:1024-1042`,
also `IndexHNSW.cpp:420-426`). This is thread-safe and does not mutate the index.

**(b) Persistent via `ParameterSpace`** (AutoTune, `faiss/AutoTune.cpp`):
`set_index_parameter(index, "efSearch"|"efConstruction", val)` mutates `index->hnsw.efSearch`
etc. directly, including when the HNSW is an IVF quantizer (`AutoTune.cpp:590-614`). `initialize()`
registers an `efSearch` sweep over `{4,8,…,512}` (`1<<2 .. 1<<9`) for any `IndexHNSW`
(`AutoTune.cpp:408-413`). This is the auto-tuning/grid-search entry point.

**Decision — runtime params:** **ADOPT the `SearchParametersHNSW` pattern** (per-query ef
override) — ELIPS already threads `ef_search` as a `search_batch(..., size_t ef_search)` argument
(`include/elips/gpu_engine/GpuIndexPort.hpp:34`, `GpuGraphIndex.hpp:40`, etc.), which is the same
idea with a cleaner signature. **SKIP `ParameterSpace`/AutoTune** — grid auto-tuning is out of
scope.

---

## 5. Param-naming map: ELIPS ↔ FAISS

ELIPS's current graph params (`include/elips/Config.hpp:45-53`) already map cleanly:

| ELIPS (`GraphParams`) | FAISS | note |
|---|---|---|
| `max_connections{16}` *// M* | `M` (ctor arg, baked into degree table) | **already annotated `// M`** in Config.hpp:47 |
| `ef_construction{200}` | `efConstruction` | same meaning, FAISS default is only 40 |
| `ef_search{50}` | `efSearch` | FAISS default 16; per-query override exists both sides |
| `compaction_ratio{0.2}` | *(no FAISS equivalent)* | ELIPS-only tombstone/rebuild knob — keep |
| GPU `graph_degree{64}` / `intermediate_graph_degree{128}` (`GpuConfig.hpp:36-37`) | CAGRA concept, not CPU `M` | leave as GPU-specific |

**Naming decision:**
- **ADAPT (cosmetic, Phase 5 naming map only):** ELIPS already uses `max_connections`,
  `ef_construction`, `ef_search`. These are correct and self-documenting. FAISS uses bare
  `M`/`efConstruction`/`efSearch`. Recommend keeping ELIPS's descriptive snake_case names and
  recording the FAISS aliases in `03_naming_map.md` rather than renaming to `M`. The one-word
  `M` is cryptic; `max_connections` is strictly better for a public API.
- **SKIP:** no behavioral change. ELIPS semantics already match FAISS.

---

## Summary of decisions

| Item | Decision | Rationale |
|---|---|---|
| HNSW struct & algorithm | ADOPT model / SKIP rewrite | ELIPS already has an equivalent HNSW |
| Deterministic lock-free build | NOTE as future option | reproducible indexes; not needed now |
| Flat storage | ADOPT | matches ELIPS baseline fp32 graph |
| PQ/SQ-backed HNSW | ADAPT later | reuse ELIPS quant codec when memory-bound |
| 2-level / `flip_to_ivf` | SKIP | niche; ELIPS keeps IVF and graph separate |
| `IndexNSG` | SKIP | single-layer prior art only |
| `SearchParametersHNSW` per-query ef | ADOPT | ELIPS already passes `ef_search` per call |
| `ParameterSpace`/AutoTune | SKIP | grid auto-tune out of scope |
| Param names M/efC/efS | ADAPT (keep ELIPS names, record aliases) | descriptive > cryptic `M` |
