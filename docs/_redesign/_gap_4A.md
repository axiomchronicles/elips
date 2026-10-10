# Phase 4A — Gap Analysis: index / search / data axes

ELIPS (`redesign`) vs **FAISS v1.15.1** and **Chroma** (current 2025/26 docs). Analysis only.
Inputs: `00_inventory.md`, `_phase1_cpp.md`, `_phase1_python.md`, `_faiss_3A/3B/3C/3E/3G.md`.
Claims verified against real code in `include/`, `src/` (cites below). Owner phase ∈
{5 naming, 6 core, 7 gpu, 8 build/test, 9 docs}.

Baseline facts confirmed in code:
- CPU index types are **only** `graph` (HNSW) + `exact` (brute force) — `Config.hpp:31`
  (`enum class IndexType { graph, exact }`). No CPU IVF/InvertedLists anywhere outside
  `gpu_engine` + the `IndexSnapshot.hpp` transfer struct.
- `euclidean` returns **true L2** `sqrt(squared_l2(a,b))` — `src/Metrics.cpp:107` — not FAISS's
  squared L2. `cosine = 1-dot`, `dot_product = -dot` (ascending-normalized), `Metrics.cpp:104-113`.
- `Vault::place(..., id, ...)` is **upsert**: on an existing id it removes the old record from
  index+metadata then re-inserts — `src/Database.cpp:898-905`. Auto-id = UUIDv7 if none given
  (`Database.cpp:889`). IDs are 128-bit `RecordID`, not int64 positions.
- Index construction is a **fluent `Config`** → `IndexFactory::make_index(Config, dim[, GpuPort*])`
  (`IndexFactory.cpp:15`); there is **no `index_factory` string grammar**.
- `seek(query, top, Filter, optional<float> threshold)` — `elips.hpp:149`. `threshold` is a
  max-distance cutoff applied inside a top-k search (widening beam re-probes), not an unbounded
  radius search. No batch (multi-query) overload on the CPU facade; batching exists only in GPU
  `search_batch`.
- `reconstruct` exists as `Vault::reconstruct(Record)` / `hydrate(Record)` (`elips.hpp:261-262`)
  and HNSW private `reconstruct(NodeId)` (`HierarchicalGraphIndex.hpp:101`).

---

## Gap table

### Index types & params

| Area | FAISS/Chroma has | ELIPS has | Severity | Effort | Decision | Owner phase |
|---|---|---|---|---|---|---|
| Flat / brute force | `IndexFlat`/`IndexFlatL2`/`IndexFlatIP` (3A §2.1) | `ExactIndex` (brute force, dual fp32/codes) `ExactIndex.cpp:55` | minor | S | Keep; record Flat/L2/IP as naming aliases | 5 naming / 9 |
| HNSW | `IndexHNSW` + `HNSW` struct (3C §1) | `HierarchicalGraphIndex` complete HNSW, recall@10≥0.90 (`_phase1_cpp.md`) | minor | S | Keep; map `M/efC/efS`↔`max_connections/ef_construction/ef_search` (3C §5) | 5 naming |
| IVFFlat | `IndexIVFFlat` (3B §2) | **CPU: none**; GPU `GpuIVFFlatIndex` only (`Config.hpp:31`) | major | L | Adapt: add CPU IVFFlat tier, or document IVF as GPU-only | 6 core / 7 gpu |
| IVFPQ | `IndexIVFPQ` + residual codes + precomputed table (3B §3) | CPU: none; GPU IVFPQ path only | major | L | Adapt: CPU IVFPQ reuses existing PQ codec behind an IVF list layer | 6 core / 7 gpu |
| IVF-SQ | IVF + `ScalarQuantizer` storage | None (SQ8 codec exists, no IVF-SQ) | major | L | Adapt: falls out of IVF tier once lists exist | 6 core |
| IDMap / IDMap2 | int64 id wrapper + `reconstruct(id)` (3A §2.2) | Folded in: every `Record` has a `RecordID`; `fetch(id)`/`reconstruct` built in (`Database.cpp`) | minor | S | Keep ELIPS shape; document as the IDMap equivalent | 5 / 9 |
| Refine / RefineFlat | over-fetch `k*k_factor` then exact rerank (3A §2.3) | `QuantParams::refine_fraction` **declared, unimplemented** (`QuantParams.hpp:45-47`, cites nonexistent ADR-0011); `seek_hybrid` over-fetches ×5 (hybrid only), HNSW widening beam | major | M | Implement: exact-rerank of quantized candidates with a `k_factor`/`refine_fraction` knob | 6 core |
| PreTransform (OPQ/PCA) | `IndexPreTransform` chain (3A §2.4) | OPQ folded into `ProductQuantizer` rotation (`ProductQuantizer.hpp:20-27`); no standalone PCA/Pad/L2norm transform | minor | M | Skip standalone class (3A verdict); add only if factory prefixes need it | 6 (if needed) |
| **`index_factory` grammar** | regex factory `"IVF…,PQ…"` etc. (3A §3) — "most FAISS-recognizable feature" | **None** — fluent `Config` + `make_index(Config)` (`IndexFactory.cpp:15`) | blocker | M | Adapt: a curated string parser (`Flat`, `IVF{n},Flat/PQ/SQ8`, `HNSW{M}[,…]`, `PQ{M}x{b}`, `OPQ…`, `…,RFlat`, `IDMap[2]`) over the engines ELIPS has | 6 core (+5 tokens) |

### Metrics & ordering

| Area | FAISS/Chroma has | ELIPS has | Severity | Effort | Decision | Owner phase |
|---|---|---|---|---|---|---|
| L2: squared vs true | FAISS `METRIC_L2` returns **squared** L2 (3E §1); Chroma `l2` squared, default | `euclidean` returns **true** `sqrt` L2 (`Metrics.cpp:107`) — ranking identical (monotonic), but distance *values* and `threshold` cutoffs differ | major | S | Document the divergence; optionally expose a squared-L2 option for FAISS/Chroma value-compat | 5 naming / 9 |
| Inner product | `METRIC_INNER_PRODUCT` (max) (3E §1) | `dot_product` = `-dot` (ascending-normalized) `Metrics.cpp:111` | minor | S | Keep; map name + note heap-direction normalization | 5 / 9 |
| Cosine | FAISS: IP on L2-normalized (no cosine metric) (3E §2); Chroma `cosine` | Explicit `cosine = 1-dot` + `requires_normalization` only for cosine (`Metrics.cpp:104,117`) | minor | S | Keep (cleaner than FAISS); document cosine=normalized-IP equivalence | 9 |
| Ordering / result padding | min/max heap by `is_similarity`; `k>ntotal` legal; empty slots label **`-1`** + sentinel; caller-alloc `n*k` (3A §1.3) | Ordering-normalized smaller=closer; returns `vector<Hit>`/`SearchResult` of ≤k, **no `-1` padding**, no fixed `n*k` array (`IndexPort.hpp:18`) | minor | S | Keep ELIPS contract; document the difference; add `-1`-padded path only if a FAISS-shaped C API is wanted | 9 |
| scipy family (L1/Linf/Lp/Canberra/…) | flat-only via `knn_extra_metrics` (3E §1) | None | minor | — | Skip (YAGNI, 3E verdict) | — |

### Training / quantization

| Area | FAISS/Chroma has | ELIPS has | Severity | Effort | Decision | Owner phase |
|---|---|---|---|---|---|---|
| PQ | `ProductQuantizer` M/nbits (3B §4) | `ProductQuantizer` complete, 8-bit only (`code_bytes==pq_dim`) | minor | S | Keep; alias `M=pq_dim`, `nbits=pq_bits`, `dsub=sub_dim` (3B alias table); mark "8-bit only" | 5 / 9 |
| OPQ | `OPQMatrix` niter=50/4/40 (3B §6) | OPQ via from-scratch Newton-Schulz Procrustes, single `opq_iters` (`_phase1_cpp.md` quant) | minor | S | Keep; document FAISS iter defaults as reference | 9 |
| SQ types | `QT_8bit/4bit/6bit/fp16/bf16/…` (3B §5) | `sq8` per-dim int8 only (= FAISS `QT_8bit` default) | minor | M | Keep sq8 as default; `fp16`/4-bit are future tiers (3B verdict) | 6 (future) |
| k-means defaults | `niter=25`, `nredo=1`, `min/max_points_per_centroid=39/256`, random/kmeans++ init (3B §7) | Deterministic strided init (byte-reproducible, no RNG), explicit `train_iters`, **no `nredo`, no points-per-centroid caps** (`KMeans.hpp:16-23`) | minor | S/M | Adopt `nredo`, `max_points_per_centroid=256` training cap, `<39`/centroid warning; **keep** deterministic init | 6 core |
| Residual coding (`by_residual`) | PQ/SQ trained on `x−centroid`, biggest IVFPQ recall lever (3B §1,§4) | N/A on CPU (no IVF); verify GPU IVFPQ trains on residuals | major | M | Adopt with the IVF tier; part of the IVFPQ build | 6 / 7 |
| Precomputed per-list table | IVFPQ L2+residual centroid-term cache (3B §3) | None (no CPU IVF) | major | M | Adopt with IVF tier (`use_precomputed_table` −1/0/1, drop MultiIndex) | 6 / 7 |
| PQ bit-packing (4/6-bit) | `code_size=(nbits·M+7)/8` (3B §4) | 8-bit/subspace only | minor | M | Skip until a sub-8-bit tier is requested | 6 (future) |

### Search features

| Area | FAISS/Chroma has | ELIPS has | Severity | Effort | Decision | Owner phase |
|---|---|---|---|---|---|---|
| k-NN search | `search(n,x,k,…)` (3A §1.2) | `seek(query, top, …)` single-query (`elips.hpp:149`) | minor | S | Keep; map `k`↔`top` | 5 |
| range_search (radius) | `range_search(radius, RangeSearchResult)` (3A §1.2) | **No unbounded radius search**; `threshold` is a max-distance cutoff *inside* top-k (`elips.hpp:150`) | major | M | Adapt: add a true radius/range variant (threshold is close but top-capped) | 6 core |
| Batched search | `search` takes `n` queries, query-parallel (3E §5) | CPU facade is single-query; batch only in GPU `search_batch` | major | M | Adapt: add a batch (multi-query) overload on the CPU seek path | 6 core |
| Filtered search | `IDSelector` in `SearchParameters` (3A §5) | `Filter` metadata-predicate pushdown via `exact_candidates`+widening beam (`Filter.cpp`, `Database.cpp:1228-1254`) — richer (predicate, not just id) | minor | S | Keep; it's the EQL/Filter path. Map concept to SearchParameters.sel | 5 / 9 |
| ID selectors (range/batch/bitmap) | `IDSelectorRange/Batch/Bitmap/Not/And/Or` over int64 (3A §5.2) | None — IDs are UUIDv7; filtering is metadata-based, not id-set-based | minor | M | Covered differently; a by-id selector is low value with UUID ids — document, add only on demand | 9 / 6 |
| SearchParameters struct | per-call `{sel, nprobe, efSearch}` thread-safe (3A §5.1, 3C §4) | Per-call args (`top, Filter, threshold`; GPU `ef_search` arg) — no struct | minor | S | Keep per-call args; record as the SearchParameters equivalent | 5 |
| Hybrid (lexical+vector) fusion | FAISS: **none** | `seek_hybrid` (vector KNN + Jaccard-ish lexical rerank, over-fetch ×5) + `seek_text` (`elips.hpp:155`, `Database.cpp:1272`) — **ELIPS ahead** | minor | — | Keep; feature ELIPS has over FAISS; document | 9 |
| Document/content filter | Chroma `where_document` `$contains/$not_contains/$regex/$not_regex` (3G §1.6) | EQL `contains`/`Filter::has_substring` = substring only, **no regex** (`EQLParser`, `Filter.cpp`) | minor | M | Adapt: map `$contains`/`$not_contains`; add `$regex` later | 6 / 9 |
| Metadata filter operators | Chroma `$eq $ne $gt $gte $lt $lte $in $nin $and $or` (3G §1.5) | `Comparator{eq,ne,lt,le,gt,ge}` + `in_set` + `and_/or_/not_` (`Filter.hpp:13`) — covers all but `$nin` directly | minor | S | Adopt `$`-operator mapping; `$nin` = `not_(in_set)` | 5 / 6 |

### reconstruct / reconstruct_n

| Area | FAISS/Chroma has | ELIPS has | Severity | Effort | Decision | Owner phase |
|---|---|---|---|---|---|---|
| reconstruct(id) | `reconstruct(key, recons)`, may be lossy (3A §1.2) | `fetch(id)->Record` + `Vault::reconstruct(Record)`/`hydrate` (fp32 or codec decode) `elips.hpp:261` | minor | S | Keep; map name `reconstruct` | 5 / 9 |
| reconstruct_n / batch | `reconstruct_n(i0, ni, …)` positional range (3A §1.2) | `scan(Filter, offset, limit)->vector<Record>` (decoded) `elips.hpp:165` — no positional-range form (ids are UUID, not positional) | minor | S | Covered by `scan`; document the mapping | 9 |
| reconstruct lossy flag | approximate when codec lossy | `Record::approximate()`/`codec` stamped on hydrate (`Record.hpp:60-63`) | minor | — | Keep | 9 |

### CRUD

| Area | FAISS/Chroma has | ELIPS has | Severity | Effort | Decision | Owner phase |
|---|---|---|---|---|---|---|
| add | FAISS `add` (auto ids); Chroma `add(ids=…)` | `place` (auto UUIDv7) / Py `Collection.add`/`write` (`_phase1_python.md`) | minor | S | Keep; `add` alias already in Py | 5 |
| add_with_ids | `add_with_ids(…, xids)` (3A §1.2) | `place(…, id=…)` (`Database.cpp:889`) | minor | S | Keep; map name | 5 |
| upsert | Chroma `upsert` (universal across DBs, 3G §5) | `place` **is** upsert (`Database.cpp:898-905`) but **no `upsert` name** in Py `Collection` | major | S | Adopt `upsert` name alias on `Collection` (semantics already there) | 5 naming |
| get | Chroma `get(ids,where,…)` (3G §1.3) | `fetch(id)` + `scan(where)`; Py `Collection.get`/`pull` | minor | S | Keep; `get` alias exists in Py | 5 |
| update | Chroma `update` (partial embeddings/meta) | **No partial update** — `place` replaces whole record; no `update` name in Py Collection | major | S/M | Adapt: `update` = fetch+merge+place, or add `Collection.update` | 6 core |
| delete / remove_ids | FAISS `remove_ids(IDSelector)` (3A §1.2); Chroma `delete(ids,where)` | `erase(id)` single (`elips.hpp:170`); Py `Collection.delete`/`discard(keys|where)` loops erase | minor | S | Keep; add bulk `remove_ids`-style only if measured | 5 / 6 |
| reset / clear | FAISS `reset()` clears all (3A §1.2); Chroma `reset()` | **No clear-all verb** (`rebuild_index` drops the index but keeps records; no truncate) | minor | S | Adapt: add a `reset`/`clear` that empties a vault | 6 core |
| peek / count | Chroma `count()`, `peek(n=10)` (3G §1.3) | `count()` present; **no `peek`** | minor | S | Adopt `peek(n=10)` (3G verdict) | 6 / 5 |

### Persistence / durability

| Area | FAISS/Chroma has | ELIPS has | Severity | Effort | Decision | Owner phase |
|---|---|---|---|---|---|---|
| Format magic + version | FAISS: **no magic, no version int** — fourcc per class (3A §4) | Snapshot format v3 (`snapshot_version` `Database.cpp:57`), WAL magic `0xE1105E01`, CRC32C, length-bounded — **ELIPS ahead / safer** | minor | — | Keep ELIPS header; do NOT clone fourcc scheme (3A verdict) | 9 |
| WAL | **None** | WAL complete: append/replay/txn-framing, recovery-tested + fuzz target (`WAL.cpp`, `_phase1_cpp.md` storage) — **ELIPS ahead** | minor | — | Keep; test `append_insert_quantized`/`reset` (currently untested) | 8 build/test |
| Snapshots / checkpoints | `write_index(idx, path)` (3A §4) | `checkpoint()` (snapshot or segmented manifest+segments, durable-rename) `Database.cpp:1662` — richer, DB-level not per-index | minor | S | Keep; document as the write_index equivalent | 9 |
| read_index / write_index API shape | free fns returning `Index*` + `_up` unique_ptr + `io_flags` (3A §4) | `open(path)->Database` + `checkpoint()`; whole-DB granularity, not per-index | minor | M | Document the mapping; optionally expose per-index save/load for FAISS familiarity | 9 / 6 |
| Deserialization size caps | `set_deserialization_*_limit` anti-DoS (3A §4) | `max_field_bytes=64MiB` + `check_length` before alloc (`Serialization.hpp:54,60`) — already adopted | minor | — | Keep; expose the cap as a tunable if needed | 8 |
| Chroma `PersistentClient`/`Client` | `Client()` / `PersistentClient(path)` / `HttpClient` (3G §1.1) | **None** — `connect(path)`/`open(path)` only; no Client layer (`_phase1_python.md`) | blocker | M | Adopt `Client`/`PersistentClient(path=)` over `connect` for Chroma recognizability (3G ADOPT) | 5 naming / 6 |

---

## Biggest gaps (ranked)

1. **No `index_factory` string grammar** (blocker, M) — the single most FAISS-recognizable
   feature is absent; ELIPS only has a fluent `Config` + `make_index(Config)`
   (`IndexFactory.cpp:15`). A curated string parser over the engines ELIPS already has is the
   highest-leverage recognizability win. Phase 6.
2. **No CPU IVF family** (major, L) — IVFFlat/IVFPQ/IVF-SQ exist only on GPU; CPU is HNSW+exact
   (`Config.hpp:31`). This also blocks `by_residual` training and the IVFPQ precomputed table —
   FAISS's signature mid-tier. Phase 6 (+7 for GPU parity).
3. **No Chroma `Client`/`PersistentClient`** (blocker, M) and **missing `upsert`/`update`/`peek`
   names** on the Python `Collection` (major, S) — the data-plane upsert semantics already exist
   (`place`, `Database.cpp:898`); the surface names do not. Mostly Phase 5 naming.
4. **Refine / exact-rerank unimplemented** (major, M) — `QuantParams::refine_fraction` is a
   declared-but-dead knob citing a nonexistent ADR-0011 (`QuantParams.hpp:45-47`); the one real
   "to-build" from Phase 1. Phase 6.
5. **L2 is true (`sqrt`) not squared** (major, S) — `Metrics.cpp:107` diverges from FAISS/Chroma
   `l2`. Ranking is identical (monotonic) but distance values and `threshold` cutoffs differ;
   needs documenting and an optional squared-L2 mode. Phase 5/9.
6. **No batched (multi-query) k-NN and no true range/radius search on the CPU facade** (major, M
   each) — `seek` is single-query; `threshold` is top-capped, not an unbounded radius
   (`elips.hpp:149`). Phase 6.
7. **ELIPS is ahead on durability and hybrid search** — WAL + CRC + magic/versioned snapshots and
   `seek_hybrid` have no FAISS equivalent; keep them, don't regress. Only test debt remains
   (`WAL::append_insert_quantized`/`reset` untested — Phase 8).
