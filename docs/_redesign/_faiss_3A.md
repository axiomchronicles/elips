# Phase 3A — FAISS Core CPU Index API Study

Source: FAISS **v1.15.1** at `/tmp/faiss` (read-only; not built). All citations are `file:line`
against that tree. Goal: make ELIPS's public C++ API FAISS-recognizable. Each item ends with an
**adopt / adapt / skip** call for ELIPS.

Version is baked into the header, not the file format: `faiss/Index.h:18-20`
(`FAISS_VERSION_MAJOR/MINOR/PATCH = 1/15/1`).

---

## 1. The `Index` base contract (`faiss/Index.h`)

### 1.1 Public data members (`Index.h:101-123`)
```cpp
int        d;           // vector dimension (NB: int, not idx_t)   Index.h:105
idx_t      ntotal;      // total nb of indexed vectors              Index.h:106
bool       verbose;     // verbosity flag                           Index.h:107
bool       is_trained;  // true if no training needed / done        Index.h:111
MetricType metric_type; // metric for search                        Index.h:114
float      metric_arg;  // metric parameter (e.g. p for Lp)         Index.h:115
```
- `idx_t = int64_t` (`MetricType.h:52`). Vectors are **always** `float*`, row-major, compact:
  component j of vector i is `x[i*d + j]` (`Index.h:34-47`). Internal storage may differ.
- Ctor defaults: `d=0, ntotal=0, verbose=false, is_trained=true, metric=METRIC_L2, metric_arg=0`
  (`Index.h:117-123`). **Reality check: `is_trained` defaults to `true`** — an index is "trained"
  until a subclass that needs training sets it false. `d` is a plain `int`, so max dim ~2.1B but
  deserialization caps it at `1<<20` (`index_read.cpp:286`).

### 1.2 The virtual contract (signatures exact)
| Method | Signature / semantics | Cite |
|---|---|---|
| `train` | `virtual void train(idx_t n, const float* x)` — **default is a no-op** (does nothing, leaves `is_trained=true`). | `Index.h:132`, `Index.cpp` train body |
| `add` | `virtual void add(idx_t n, const float* x) = 0` — **pure virtual**. Implicitly assigns ids `ntotal .. ntotal+n-1`. | `Index.h:164` |
| `add_with_ids` | `virtual void add_with_ids(idx_t n, const float* x, const idx_t* xids)` — **default THROWS** "not implemented" (only IDMap/IVF support it). | `Index.h:183`, `Index.cpp` |
| `search` | `virtual void search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels, const SearchParameters* params=nullptr) const = 0` — **pure virtual**, `const`. Fills `distances` and `labels` (both size `n*k`, row-major per query). | `Index.h:207-213` |
| `range_search` | `virtual void range_search(idx_t n, const float* x, float radius, RangeSearchResult* result, const SearchParameters* params=nullptr) const` — **default THROWS** "range search not implemented". | `Index.h:252`, `Index.cpp` |
| `assign` | `virtual void assign(idx_t n, const float* x, idx_t* labels, idx_t k=1) const` — like search but labels only. | `Index.h:268` |
| `reset` | `virtual void reset() = 0` — **pure virtual**. Removes all vectors. | `Index.h:272` |
| `remove_ids` | `virtual size_t remove_ids(const IDSelector& sel)` — **default THROWS**. Returns #removed. | `Index.h:277` |
| `reconstruct` | `virtual void reconstruct(idx_t key, float* recons) const` — **default THROWS**. May be approximate (lossy codecs). | `Index.h:285` |
| `reconstruct_batch` | `virtual void reconstruct_batch(idx_t n, const idx_t* keys, float* recons) const` | `Index.h:295` |
| `reconstruct_n` | `virtual void reconstruct_n(idx_t i0, idx_t ni, float* recons) const` — **default loops `reconstruct(i0+i)`** (OpenMP if ni>1000). | `Index.h:305`, `Index.cpp` |
| `search_and_reconstruct` | search + reconstruct results, `recons` size `(n,k,d)`. | `Index.h:320` |
| `compute_residual` / `_n` | residual = x − reconstruction(key). | `Index.h:363,379` |
| `get_distance_computer` | returns a `DistanceComputer` for random-access indexes. | `Index.h:391` |
| `sa_code_size` / `sa_encode` / `sa_decode` | standalone codec interface. | `Index.h:396-412` |
| `merge_from` / `check_compatible_for_merge` | move entries from another index. | `Index.h:418,423` |
| `add_sa_codes` | add pre-encoded codes. | `Index.h:430` |

Only three methods are pure virtual: `add`, `search`, `reset` (`Index.h:164,213,272`). Everything
else has a default (no-op, throw, or loop). **This is the minimal surface a concrete index must
implement.**

### 1.3 Search output semantics (the important fine print)
Verified against `IndexFlat::search` (`faiss/IndexFlat.cpp`) and `faiss/utils/Heap.h`:
- `k > 0` is **asserted** (`FAISS_THROW_IF_NOT(k > 0)` in `IndexFlat.cpp`).
- Results are maintained as a per-query heap: `maxheap` for L2, `minheap` for inner product
  (`IndexFlat.cpp`, `knn_L2sqr`/`knn_inner_product`).
- **`k > ntotal` is NOT an error.** The heap is pre-initialized with `bh_ids[i] = -1` and a sentinel
  distance (`Heap.h:341`, `heap_heapify`). Any slot not filled by a real neighbor stays **label
  `-1`** with sentinel distance (`+HUGE_VALF` for L2 maxheap). So callers must treat `-1` labels as
  "no result".
- L2 returns **squared** L2 (`METRIC_L2` doc, `MetricType.h:32`). Inner product returns raw dot
  product (larger = better); `is_similarity_metric()` distinguishes min/max indexes
  (`MetricType.h:56-59`).
- `distances`/`labels` are caller-allocated, size `n*k`, row-major (query-major).

### 1.4 `NumericType` + `*_ex` variants — reality divergence
v1.15.1 adds an `enum NumericType {Float32, Float16, UInt8, Int8}` (`Index.h:61-66`) and `_ex`
twins (`train_ex`, `add_ex`, `add_with_ids_ex`, `search_ex`) that take `const void* x` +
`NumericType` (`Index.h:148-233`). Default impls only accept `Float32` and throw otherwise. This is
newer than most FAISS docs show — the canonical API is still the `float*` set; the `_ex` layer is an
optional typed-input shim.

**Decision — ADOPT (the float32 contract) / SKIP (the `_ex` layer).** The `d/ntotal/is_trained/
metric_type` fields and the `train/add/add_with_ids/search/range_search/reconstruct/reconstruct_n/
remove_ids/reset` virtuals are the recognizable core; ELIPS should mirror names, signatures, and the
`-1`-padding + squared-L2 semantics exactly. Skip `NumericType`/`_ex` until a typed-input need is
real (YAGNI) — ELIPS can take `float*` only and add typed overloads later.

---

## 2. Concrete leaf / wrapper indexes

### 2.1 `IndexFlat` family (`faiss/IndexFlat.h`, `faiss/IndexFlatCodes.h`)
- `IndexFlat : IndexFlatCodes : Index`. Stores full vectors as fixed-size codes in
  `MaybeOwnedVector<uint8_t> codes` (`IndexFlatCodes.h:22-26`); for plain Flat the "code" is just the
  raw float vector (`code_size = d*sizeof(float)`), add/encode are memcpy (`IndexFlat.h:70-73`).
- `IndexFlatL2(d)` and `IndexFlatIP(d)` are thin ctor-only subclasses fixing the metric
  (`IndexFlat.h:76-101`). `IndexFlatL2` additionally caches L2 norms (`cached_l2norms`,
  `sync_l2norms()`) to accelerate its DistanceComputer (`IndexFlat.h:81-100`).
- `reconstruct`/`reconstruct_n` and exact `remove_ids` live in `IndexFlatCodes`
  (`IndexFlatCodes.h:37-46`). **Reality check: Flat's `remove_ids` SHIFTS remaining ids down** —
  "the new ids are shifted" (`IndexFlatCodes.h:43-45`). So positional ids are not stable across
  removals; that is exactly why IDMap exists.
- Extras present in 1.15.1 but niche: `IndexFlatPanorama`/`IndexFlat1D` (`IndexFlat.h:103-224`).

**Decision — ADOPT `IndexFlat`/`IndexFlatL2`/`IndexFlatIP` as the brute-force baseline and the
reconstruct reference.** ELIPS already has storage; expose these three names over it. Note the
id-shift-on-remove behavior in docs. **SKIP** Panorama/Flat1D (speculative).

### 2.2 `IndexIDMap` / `IndexIDMap2` (`faiss/IndexIDMap.h`)
- Templated `IndexIDMapTemplate<IndexT>` wrapping a sub-`index` with a `std::vector<idx_t> id_map`
  (`IndexIDMap.h:20-27`). `IndexIDMap = IndexIDMapTemplate<Index>` (`IndexIDMap.h:88`).
- Translates: `add_with_ids` stores user ids in `id_map`; `search` runs the sub-index then maps
  internal positions → user ids via `id_map` (`IndexIDMap.h:32-50`). **`add` without ids FAILS** on
  IDMap — "this will fail. Use add_with_ids" (`IndexIDMap.h:40-41`).
- `IndexIDMap2` adds a reverse map `std::unordered_map<idx_t,idx_t> rev_map` for efficient
  `reconstruct(user_id)` (`IndexIDMap.h:93-120`).
- When searching with an `IDSelector`, IDMap wraps it in `IDSelectorTranslated` so the user's
  selector sees user ids while the sub-index scans internal positions (`IndexIDMap.h:132-152`).
- `own_fields` controls whether the wrapped index is deleted in the dtor (`IndexIDMap.h:26`).

**Decision — ADOPT the concept, ADAPT the shape.** ELIPS needs arbitrary 64-bit external ids, and
IDMap is FAISS's canonical answer. But a template-wrapper-over-pointer with manual `own_fields` is
clunky; ELIPS should fold id↔position mapping into its engine (or a single `IdMappedIndex`) rather
than a separately-owned wrapper. Keep the `add_with_ids` / translated-selector / `reconstruct(id)`
semantics so it stays recognizable.

### 2.3 `IndexRefine` / `IndexRefineFlat` (`faiss/IndexRefine.h`)
- Two-stage search: a fast `base_index` over-fetches `k * k_factor` candidates, then a `refine_index`
  (exact) re-ranks to the final `k` (`IndexRefine.h:21-55`). `k_factor` default 1 (`IndexRefine.h:36`).
- `IndexRefineFlat` fixes the refine index to an `IndexFlat` and can be built from a raw table
  (`IndexRefine.h:83-96`).
- Per-call tuning via `IndexRefineSearchParameters { float k_factor; SearchParameters*
  base_index_params; }` (`IndexRefine.h:14-19`) — overrides the member `k_factor` at search time.
- `reconstruct` routed to `refine_index`; codec interleaves base+refine codes (`IndexRefine.h:64-75`).

**Decision — ADAPT.** The over-fetch-then-exact-rerank pattern maps directly onto ELIPS's existing
PQ/SQ8 + exact-reorder path; expose it as a refine wrapper (or a search option) with the `k_factor`
knob. Mirror `IndexRefineSearchParameters` as a per-call override.

### 2.4 `IndexPreTransform` (`faiss/IndexPreTransform.h`)
- Holds a `std::vector<VectorTransform*> chain` applied to vectors before the sub-`index`
  (`IndexPreTransform.h:25-38`). Transforms are prepended (`prepend_transform`), so the last added is
  applied first; `apply_chain`/`reverse_chain` run forward/inverse (`IndexPreTransform.h:38,90-96`).
- `train` trains each transform then the sub-index; `search` transforms the query then delegates
  (`IndexPreTransform.h:40-58`). Used for PCA / OPQ / L2norm / Pad front-ends.

**Decision — ADAPT (thin) / mostly SKIP for now.** ELIPS already owns OPQ/PCA inside its engine, so a
general transform-chain wrapper is largely redundant. Keep the *idea* (a pre-transform hook) if the
factory grammar needs `OPQ...,` / `PCA...,` prefixes to parse; otherwise skip the standalone class.

---

## 3. `index_factory` string grammar (`faiss/index_factory.cpp`)

Entry: `Index* index_factory(int d, const char* description, MetricType metric=METRIC_L2, bool
own_invlists=true)` (`index_factory.cpp:1234-1240`). It's a **regex-based recursive descent**, not a
formal grammar — `index_factory_sub` (`:965`) peels tokens in a fixed priority order. Tokens are
comma-separated; the recognizer order is what defines the grammar.

**Parse order (reality — order matters, first match wins):**
1. **IDMap / IDMap2** wrapper, accepted as **prefix OR suffix**: `"...,IDMap2"` or `"IDMap2,..."`,
   same for `IDMap` (`index_factory.cpp:983-997`).
2. **Refine**: `",RefinePanorama(...)"`, `",RFlat"`, `",Refine(...)"` (`:1001-1030`).
3. **PreTransform** prefixes — any leading token parseable as a `VectorTransform` is stripped and the
   rest recursed, wrapped in `IndexPreTransform` (`:1032-1061`). Transforms: `PCA[W][R]k`, `L2norm`,
   `RR[k]`, `HR[k]`, `ITQ[k]`, `OPQM[_d]`, `Pad k` (`parse_VectorTransform :226-258`).
4. **Parentheses** — `(...)` sub-descriptions are factored out as `Index0`, `Index1`, …
   (`:1067-1078`), used for custom IVF coarse quantizers.
5. **Basic leaf indexes** (`parse_other_indexes :781-960`): `Flat`, `SQ{4,6,8,fp16,bf16,...}`
   (`sq_pattern :178-179`), `PQ{M}[x{nbits}][np]`, `PQ{M}x4fs[_bbs]`, `LSH[bits][r][t]`, `RQ.../RCQ`,
   `LSQ.../LSCQ`, `PRQ`, `PLSQ`, `RaBitQ[1-9]`, `EDEN...`, `ZnLattice...`.
6. **HNSW**: `HNSW{M}[,|_ code]` where code ∈ `Flat|PQ..|SQ..|RaBitQ..|{n}+PQ..` (`:1096-1115`,
   `parse_IndexHNSW :548-606`). Separator can be `,` or `_`.
7. **NSG**: `NSG{R}[,|_ code]` (`:1138-1157`).
8. **MinMax / MinMaxFP16** prefix wrappers (`:1160-1181`).
9. **IVF** (fallback): `coarse,code`. Coarse (`parse_coarse_quantizer :278-363`):
   `IVF{nlist}` (nlist accepts `k`/`M` suffix → ×1024 / ×1024², `parse_nlist :264-275`),
   `IMI2x{nbit}`, `IVF{nlist}_HNSW{M}`, `IVF{nlist}_NSG{R}`, `IVF{nlist}(Index0)`, `Residual...`.
   Code (`parse_IndexIVF :367-542`): `Flat`, `FlatDedup`, `SQ*`, `PQ{M}[x{nbits}][np]`,
   `PQ{M1}+{M2}` (→IVFPQR), `PQ{M}x4fs[r][_bbs]`, `RQ/LSQ...`, `PRQ/PLSQ...`, `RaBitQ`, `EDEN`.

**Reality notes:**
- IDMap works as **both prefix and suffix** — a historical quirk explicitly commented at
  `:981-982`. Most docs show only the suffix form (`"...,IDMap"`).
- `np` suffix on PQ **disables** polysemous training (`:406,844`).
- Unparseable input → `FAISS_THROW_FMT("could not parse index string ...")` (`:1228`).
- A separate `index_binary_factory` uses `sscanf`, not regex (`:1242-1304`) — `BFlat`, `BIVF%d`, etc.

**Decision — ADAPT (curated subset).** The factory string is the single most "FAISS-recognizable"
feature — ELIPS should accept the common tokens that map to engines it actually has: `Flat`,
`IVF{nlist},Flat`, `IVF{nlist},PQ{M}x{nbits}`, `IVF{nlist},SQ8`, `HNSW{M}[,Flat|,PQ..|,SQ8]`,
`PQ{M}x{nbits}`, `OPQ{M},...`, `...,RFlat`/`,Refine(...)`, and `IDMap`/`IDMap2` (support both
affix positions). **SKIP** the long tail (EDEN, RaBitQ, PRQ/PLSQ, Lattice, SVS, Panorama, binary) —
re-regex only what ELIPS can build. Reuse FAISS's token spellings verbatim so strings copy-paste.

---

## 4. `read_index` / `write_index` — format & versioning (`index_io.h`, `impl/index_write.cpp`, `impl/index_read.cpp`)

**Format philosophy (`index_write.cpp:76-93`):** the serialized form **is the class's fields**, in
declaration order. Each polymorphic object is prefixed by a **4-char-code (fourcc)** `uint32_t`
identifying the concrete subclass; parent fields are written before child fields.
`fourcc("IxF2")` = little-endian pack of the 4 bytes (`io.cpp:241-245`).

**Key reality: there is NO global file magic and NO version integer.** A FAISS index file begins
*directly* with the top-level index's fourcc. "Versioning" is done by **swapping the fourcc** when a
class's layout changes; new readers keep branches for old fourccs (`index_write.cpp:87-89`). E.g.
Flat is written as `IxFI`/`IxF2`/`IxFl` by metric (`index_write.cpp:492-497`) and read back by
matching those codes (`index_read.cpp:1744-1762`); LSH reads both `IxHE` (old) and `IxHe` (new)
(`index_read.cpp:1763`). This is the opposite of a versioned-header format.

**Header payload (`write_index_header :100-111` / `read_index_header :283-302`):**
`d, ntotal, dummy(1<<20), dummy, is_trained, metric_type` and, **only if `metric_type > 1`**,
`metric_arg`. Two `dummy` i64 slots (legacy `max_codes`/range fields) are written as `1<<20` and
ignored on read. `metric_type` round-trips through `metric_type_from_int()` with range validation
(`index_read.cpp:297`).

**Dispatch:** `write_index` is a long `dynamic_cast` chain (`index_write.cpp:475+`); `read_index` /
`read_index_up` reads the fourcc then an `if (h == fourcc(...))` chain (`index_read.cpp:3260-3262`,
leaf branches throughout). Public API: `write_index(idx, fname|FILE*|IOWriter*, io_flags=0)` and
`read_index(...)` returning raw `Index*`, plus `read_index_up(...)` returning `unique_ptr`
(`index_io.h:42-81`). Composite indexes recurse (`write_index(idxrf->base_index, f)` etc.,
`index_write.cpp:841,856-857,866`).

**`io_flags`** (`index_io.h:40-71`): `IO_FLAG_SKIP_STORAGE=1`, `READ_ONLY=2`, `ONDISK_SAME_DIR=4`,
`SKIP_IVF_DATA=8`, `SKIP_PRECOMPUTE_TABLE=16`, `PQ_SKIP_SDC_TABLE=32`, `MMAP`, `MMAP_IFC=1<<9`.
Hardening knobs exist: `set_deserialization_loop_limit` / `_vector_byte_limit` (default 1<<40) /
`_lattice_r2_limit` (`index_io.h:120-154`) — anti-DoS caps on attacker-controlled sizes.

**Decision — ADAPT, do NOT clone the format.** ELIPS should keep FAISS's *API shape*
(`write_index(idx, path)` / `read_index(path) -> Index*` + `_up` unique_ptr variant + an `io_flags`
int). But the on-disk fourcc-per-class, no-magic, no-version scheme is a liability for an embedded DB:
**ELIPS should use a real file with a magic + explicit version integer + length-prefixed sections**,
and keep per-engine type tags. Also adopt the deserialization size caps (`index_io.h:120-154`) — they
are cheap and prevent malicious-file OOM. Binary cross-compat with FAISS files is not worth it (skip).

---

## 5. `IDSelector` family (`faiss/impl/IDSelector.h`) & `SearchParameters` (`faiss/Index.h`)

### 5.1 `SearchParameters` (`Index.h:83-93`)
```cpp
struct SearchParameters {
    IDSelector* sel = nullptr;   // if non-null, only these ids considered   Index.h:90
    virtual ~SearchParameters() {}
};
```
Base per-call params; **the only base field is `sel`**. Subclasses add index-specific knobs
(`IndexRefineSearchParameters` :3.3, `SearchParametersPreTransform`, IVF's `nprobe`/`max_codes` in
`IndexIVF.h`). Ownership of fields is the caller's (`Index.h:86-87`). Passed as the trailing
`const SearchParameters* params = nullptr` on every `search`/`range_search`. **Reality: this replaced
the old global-mutable-nprobe style — it is the thread-safe, per-query filtering mechanism.**

### 5.2 `IDSelector` hierarchy (`IDSelector.h`)
Abstract: `struct IDSelector { virtual bool is_member(idx_t id) const = 0; }` (`IDSelector.h:21-24`).
Concrete selectors:
| Selector | Semantics | Cite |
|---|---|---|
| `IDSelectorRange(imin, imax[, assume_sorted])` | `[imin, imax)`; sorted fast-path | `:71-91` |
| `IDSelectorArray(n, ids)` | array; `is_member` is slow (linear), ids reusable directly | `:98-111` |
| `IDSelectorBatch(n, ids)` | hash-set + bloom filter; for large id sets | `:123-141` |
| `IDSelectorBitmap(n, bitmap)` | one bit per id, size `ceil(n/8)` | `:145-158` |
| `IDSelectorNot(sel)` | negation | `:161-168` |
| `IDSelectorAll` | selects everything (benchmarking) | `:171-176` |
| `IDSelectorAnd/Or/XOr(lhs, rhs)` | boolean combinators | `:180-215` |
| `IDSelectorTranslated` | maps through an IDMap's `id_map` (in `IndexIDMap.h:132-152`) | — |

**Reality divergence (newer than docs):** `IDSelectorWithContext` + `IDScanContext` +
`IDSelectorContextDispatch` (`:26-68`) let a selector see the surrounding scan block (for prefetch/
locality). The dispatcher does one `dynamic_cast` per inverted-list scan and otherwise falls back to
plain `is_member`. This is an optimization hook, not core.

**Decision — ADOPT `SearchParameters{sel}` + the common selectors.** This is the recognizable
filtering API and ELIPS (which has EQL) needs predicate pushdown anyway. Implement `IDSelector` with
`is_member`, plus `Range`, `Batch` (hash+bloom), `Bitmap`, `Not`, `And`/`Or`. **ADAPT:** back
ELIPS's EQL predicate behind an `IDSelector` so filtered search reuses one path. **SKIP**
`IDSelectorContextDispatch`/`WithContext` (micro-opt), `IDSelectorArray` (linear, strictly worse than
Batch), and `IDSelectorXOr`/`All` (rarely used) until needed.

---

## Summary (10 lines)
1. `Index` has exactly 3 pure virtuals — `add`, `search`, `reset`; everything else defaults to no-op/throw/loop (`Index.h:164,213,272`).
2. Vectors are always row-major `float*`, `d` is an `int`, `idx_t=int64_t`, L2 returns **squared** distance (`Index.h:34-47`, `MetricType.h`).
3. `search` fills caller-allocated `n*k` arrays; **`k>ntotal` is legal** and unused slots get label **`-1`** + sentinel distance (`Heap.h:341`).
4. `is_trained` defaults **true**; `train` default is a no-op; `add_with_ids`/`remove_ids`/`reconstruct`/`range_search` default to **throw** (`Index.cpp`).
5. Flat's `remove_ids` **shifts ids** — IDMap/IDMap2 exist to give stable 64-bit external ids and `reconstruct(id)` (`IndexFlatCodes.h:43`, `IndexIDMap.h`).
6. `IndexRefine` (over-fetch `k*k_factor` then exact rerank) and `IndexPreTransform` (OPQ/PCA front-end) are the composable wrappers (`IndexRefine.h`, `IndexPreTransform.h`).
7. `index_factory` is regex recursive-descent; IDMap accepts prefix **or** suffix; curated token set is the main portability win (`index_factory.cpp:965-1230`).
8. Serialization has **no file magic and no version int** — a per-class fourcc identifies subclasses and layout changes swap the fourcc (`index_write.cpp:76-93`); ELIPS should instead use a magic+version header but keep the API shape and the deserialization size caps.
9. `SearchParameters{IDSelector* sel}` is the thread-safe per-call filter; selectors are `Range/Array/Batch/Bitmap/Not/And/Or/XOr/All` plus a newer context-dispatch opt (`IDSelector.h`).
10. Net plan: **adopt** the Index contract + Flat/IDMap names + SearchParameters/IDSelector; **adapt** factory strings, refine, serialization API, EQL-as-IDSelector; **skip** `NumericType/_ex`, Panorama, binary, SVS, and FAISS binary file-format compat.
