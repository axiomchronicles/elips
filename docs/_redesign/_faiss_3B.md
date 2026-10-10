# Phase 3B — FAISS IVF family & quantization study

Source: FAISS **v1.15.1** (`/tmp/faiss/CMakeLists.txt:77`), read-only. All citations are
`file:line` into that tree. Goal: align ELIPS C++23 IVF/PQ/OPQ/SQ naming, params, defaults,
and training flow with FAISS. Verdicts: **adopt** (mirror name+default), **adapt** (take the
idea, keep ELIPS naming/shape), **skip** (not worth it for ELIPS).

ELIPS today (for reference): `n_lists`/`n_probe` (IVF), `pq_dim`/`pq_bits`/`sub_dim`/`ksub`
(`include/elips/quant_engine/ProductQuantizer.hpp`), `train_iters`/`opq_iters`,
per-dim int8 `sq8` only (`include/elips/quant_engine/ScalarQuantizer.hpp`), deterministic
strided-init k-means (`include/elips/quant_engine/detail/KMeans.hpp`).

---

## 1. IndexIVF base (`faiss/IndexIVF.h`, `faiss/IndexIVF.cpp`)

### Coarse quantizer / Level1Quantizer (`IndexIVF.h:30-66`)
- `Index* quantizer` + `size_t nlist` live in `Level1Quantizer`, deliberately split out so
  training is independent of list storage (`IndexIVF.h:25-29`). The coarse quantizer is itself
  an `Index` (usually `IndexFlatL2`), i.e. centroids are just a flat index you `assign()` against.
- `char quantizer_trains_alone = 0` (`IndexIVF.h:42`): `0` = run k-means using the quantizer as
  the assignment index; `1` = pass training set straight to `quantizer->train()`; `2` = k-means on
  a flat index then `add` centroids to the quantizer. Dispatched in `train_q1`
  (`IndexIVF.cpp:68,77,105`).
- `bool own_fields = false` (`IndexIVF.h:43`) — ownership of the quantizer pointer.
- `ClusteringParameters cp` embedded (`IndexIVF.h:45`) so IVF training overrides k-means defaults.
- `Index* clustering_index = nullptr` (`IndexIVF.h:47`) — optional separate index used only during
  clustering (e.g. a GPU index to cluster faster, CPU flat for storage).

**Surprise:** spherical clustering is auto-enabled for inner-product metric in the ctor —
`if (metric_type == METRIC_INNER_PRODUCT) cp.spherical = true;` (`IndexIVF.cpp:194-195`). Not a
user decision; the metric forces it.

Verdict:
- `nlist` → **adapt**: keep ELIPS `n_lists`, but document it as the FAISS `nlist` synonym.
- `quantizer_trains_alone` → **skip**: ELIPS has one trainer; the 3-way enum is for pluggable
  quantizer indices ELIPS doesn't expose. Revisit only if a GPU coarse quantizer lands.
- `clustering_index` separation → **adapt**: ELIPS GPU pipeline already delegates clustering to
  `detail::KMeans`; keep that split, no need for the pointer indirection.
- auto-spherical on IP → **adopt**: cheap correctness win, mirror the "IP ⇒ normalize centroids"
  rule rather than making callers set it.

### Search params (`IndexIVF.h:68-95, 104-111`)
- `nprobe = 1` default (`IndexIVF.h:69,105`) — number of lists probed at query time.
- `max_codes = 0` (`IndexIVF.h:70,106`) — 0 = unlimited; else cap on codes visited per query.
- `SearchParametersIVF` carries per-query overrides (`nprobe`, `max_codes`, plus newer knobs
  `max_lists_num`, `ensure_topk_full`, `max_empty_result_buckets` — `IndexIVF.h:72-88`).

Verdict:
- `nprobe` default `1` → **adopt** name as synonym, but **adapt** the default: ELIPS maps
  `n_probe` from `default_ef_search_gpu` clamped to ≥1 (`src/gpu_engine/detail/IvfIndexState.cpp:45`).
  `1` is a poor recall default; keep ELIPS's larger default and note the FAISS name.
- `max_codes` → **adopt**: name + `0 = unlimited` sentinel. Cheap early-stop budget worth having.
- `ensure_topk_full` / `max_empty_result_buckets` → **skip**: FastScan/range-specific tuning,
  premature for ELIPS.

### Storage & flags (`IndexIVF.h:194-220`)
- `InvertedLists* invlists` + `bool own_invlists`; `size_t code_size` bytes/vector (`IndexIVF.h:196-199`).
- `int parallel_mode = 0` (`IndexIVF.h:201-212`): 0 split over queries, 1 over lists, 2 both,
  3 finer-grained over queries; `PARALLEL_MODE_NO_HEAP_INIT = 1024` as an OR flag.
- `bool by_residual = true` (`IndexIVF.h:220`) — do codes encode the vector **minus its centroid**?
- `DirectMap direct_map` for `reconstruct()` (`IndexIVF.h:216`).

Verdict:
- `by_residual = true` → **adopt**: this is the single most important IVF+PQ default. Residual
  coding is why IVFPQ recall beats flat PQ; ELIPS should encode residuals by default and name the
  flag `by_residual`.
- `code_size` (bytes/vector) → **adopt** concept; ELIPS already exposes `code_bytes()` per codec —
  align the IVF-side field name to `code_size` or document the mapping.
- `parallel_mode` enum → **skip**: ELIPS parallelism is backend-driven; a 4-way OpenMP mode enum
  doesn't map onto the GPU/port design.

### Training flow (`IndexIVF.cpp:1311-1344`)
`IndexIVF::train` is the canonical two-stage flow worth mirroring exactly:
1. `train_q1(n, x, ...)` trains the coarse quantizer (`IndexIVF.cpp:1316`).
2. optional subsample to `train_encoder_num_vectors()` via `fvecs_maybe_subsample`
   (`IndexIVF.cpp:1322-1329`).
3. **if `by_residual`**: `quantizer->assign` → `compute_residual_n` → `train_encoder(residuals,
   assign)`; else `train_encoder(x, nullptr)` (`IndexIVF.cpp:1331-1341`).

**Surprise:** residual computation is unconditional whenever `by_residual` — the sub-quantizer
(PQ/SQ) is *always* trained on residuals, never raw vectors, in the default path. ELIPS must train
PQ codebooks on `x - centroid[assign]`, not on `x`, to match recall.

Verdict: **adopt** this exact ordering (coarse → subsample → residual → encoder-train) as the ELIPS
IVF training contract.

---

## 2. IndexIVFFlat (`faiss/IndexIVFFlat.h`)
- Stores raw float vectors in lists, no encoding; `code_size = d * sizeof(float)`
  (`IndexIVFFlat.h:76`, in the scanner). Pre-selection only, exact distances within probed lists.
- `IndexIVFFlatDedup` (`IndexIVFFlat.h:97-146`) dedups identical vectors via an
  `unordered_multimap<idx_t,idx_t>`.

Verdict:
- IVFFlat → **adopt** as a named index tier (ELIPS already has `GpuIVFFlatIndex`). It's the
  "IVF with no quantizer" baseline; keep it.
- Dedup variant → **skip**: niche; add only if a dataset with heavy exact-duplicate load appears.

---

## 3. IndexIVFPQ (`faiss/IndexIVFPQ.h`, `faiss/IndexIVFPQ.cpp`)
- `ProductQuantizer pq` member produces residual codes (`IndexIVFPQ.h:40`).
- Ctor `(quantizer, d, nlist, M, nbits_per_idx, metric=METRIC_L2)` (`IndexIVFPQ.h:58-65`) — note
  **no default for M/nbits**; caller must choose.
- `int use_precomputed_table` (`IndexIVFPQ.h:52`) + `AlignedTable<float> precomputed_table` of
  size `nlist * pq.M * pq.ksub` (`IndexIVFPQ.h:55-56`). Heuristic selector
  `initialize_IVFPQ_precomputed_table` (`IndexIVFPQ.h:178-184`): `-1` force off, `0` decide by
  `precomputed_table_max_bytes`, `1` generic table, `2` MultiIndex-specific.
- Precomputed tables are **only for `by_residual` + `METRIC_L2`** (`IndexIVFPQ.h:49-51`): they
  cache the per-list centroid term so the ADC LUT reduces to a lookup. `precompute_table()`
  builds it (`IndexIVFPQ.h:159`).
- Polysemous: `do_polysemous_training`, `polysemous_ht` (`IndexIVFPQ.h:42-47`) — reorders PQ
  centroids so Hamming distance on codes pre-filters before ADC.
- `train_encoder` trains the PQ on residuals (`IndexIVFPQ.h:101`), `train_encoder_num_vectors()`
  caps training set size (`IndexIVFPQ.h:103`).

Verdict:
- `M` / `nbits` → **adapt**: ELIPS already names these `pq_dim` / `pq_bits`
  (`ProductQuantizer.hpp:31-33`). Keep ELIPS names; add a doc-level alias table (`M=pq_dim`,
  `nbits=pq_bits`, `ksub=ksub`, `dsub=sub_dim`). Do **not** rename — ELIPS names are clearer.
- precomputed table (`by_residual`+L2 only, `nlist*M*ksub` floats) → **adopt** the design and the
  `use_precomputed_table` heuristic (-1/0/1 is enough; drop the `2` MultiIndex case). This is the
  main IVFPQ query-time speedup and ELIPS lacks the per-list centroid cache.
- polysemous training → **skip**: significant complexity for a Hamming pre-filter; ELIPS has no
  polysemous path and ADC-only is fine for now.

---

## 4. Product Quantizer (`faiss/impl/ProductQuantizer.h/.cpp`)
- Core params: `size_t M` (subquantizers), `size_t nbits` (bits/index); derived `dsub = d/M`,
  `ksub = 2^nbits` (`ProductQuantizer.h:30-36`; set in `ProductQuantizer.cpp:67-70`).
- `code_size = (nbits * M + 7) / 8` (`ProductQuantizer.cpp:69`) — bit-packed, not byte-per-sub
  unless `nbits==8`. **Surprise vs ELIPS**: ELIPS assumes 1 byte/subspace (`code_bytes()==pq_dim`,
  `ProductQuantizer.hpp:44-46`), i.e. `nbits==8` hard-coded. FAISS packs 4/6/non-8-bit codes.
- `nbits > 24` throws (`ProductQuantizer.cpp:68`).
- Centroid layout `(M, ksub, dsub)` (`ProductQuantizer.h:54-56`); also keeps
  `transposed_centroids (dsub, M, ksub)` and `centroids_sq_lengths (M, ksub)` for fast ADC
  (`ProductQuantizer.h:58-64`).
- Train type enum `Train_default / hot_start / shared / hypercube / hypercube_pca`
  (`ProductQuantizer.h:39-46`), default `Train_default` (`ProductQuantizer.cpp:73`). Training runs
  one `Clustering(dsub, ksub, cp)` per subspace (`ProductQuantizer.cpp:151`,`188`), or all-at-once
  for `Train_shared` (`ProductQuantizer.cpp:193`).
- LUTs: `compute_distance_table` → `dis_table(m,j) = ||x_m - c_(m,j)||²`, size `M*ksub`
  (`ProductQuantizer.h:113-126`) = **ADC**. `sdc_table` + `compute_sdc_table` for
  **SDC** (symmetric, code-vs-code) (`ProductQuantizer.h:166-178`).

Verdict:
- `M`/`nbits`/`dsub`/`ksub` names → **adapt** (alias to ELIPS `pq_dim`/`pq_bits`/`sub_dim`/`ksub`).
- `code_size = (nbits*M+7)/8` bit-packing → **adapt**: ELIPS's byte-per-subspace is fine as the
  default (`nbits=8`), but if ELIPS ever wants `pq_bits ∈ {4,6}` the packed formula is the one to
  copy. Mark current ELIPS as "8-bit-only" explicitly.
- ADC vs SDC → **adopt** the distinction and FAISS's names. ELIPS `make_lut`/`lut_distance`
  (`ProductQuantizer.hpp:60-66`) is the ADC path; name it ADC. SDC → **skip** unless code-vs-code
  (dedup/clustering-on-codes) is needed.
- `transposed_centroids` / `centroids_sq_lengths` precompute → **adapt**: worth it for ADC speed;
  ELIPS can keep its own LUT layout but should cache squared norms.
- `Train_hypercube*` inits → **skip**: ELIPS deterministic strided init
  (`KMeans.hpp:16-23`) is intentional for byte-reproducibility; don't trade that away.

---

## 5. ScalarQuantizer (`faiss/impl/ScalarQuantizer.h`, `.cpp`)
- `QuantizerType qtype = QT_8bit` default (`ScalarQuantizer.h:59`). Classic range:
  `QT_8bit`, `QT_4bit`, `QT_8bit_uniform`, `QT_4bit_uniform`, `QT_fp16`, `QT_8bit_direct`,
  `QT_6bit`, `QT_bf16`, `QT_8bit_direct_signed`, `QT_0bit` (`ScalarQuantizer.h:27-38`). (v1.15.1
  also adds TurboQuant `QT_*_tqmse`/`_tq` and EDEN `QT_*_eden` families, `.h:39-55` — new, niche.)
- `bits` per code set from qtype in `set_derived_sizes` (`ScalarQuantizer.h:77,86`): 8bit→8,
  4bit→4, 6bit→6, fp16→16, bf16→16.
- Range estimation `RangeStat rangestat = RS_minmax` + `float rangestat_arg = 0`
  (`ScalarQuantizer.h:66-74`): `RS_minmax [min-rs·(max-min), max+rs·(max-min)]`, `RS_meanstd`,
  `RS_quantiles`, `RS_optim`.
- "uniform" = one shared range for all dims; default (non-uniform) = per-dimension range
  (`ScalarQuantizer.h:20-24`). `trained` holds the range (`ScalarQuantizer.h:80`).

**Surprise:** `QT_8bit` is **per-dimension** min/max by default — exactly what ELIPS `sq8` does
(`ScalarQuantizer.hpp:28-33`, per-dim `mins_`/`scales_`). ELIPS and FAISS already agree here.

Verdict:
- `QT_8bit` per-dim min/max → **adopt**: ELIPS `sq8` is the FAISS default SQ; name the ELIPS codec
  as the `QT_8bit` equivalent and keep per-dim ranges.
- `QT_fp16` → **adapt**: cheap, high-recall tier (half-precision passthrough, no training). Good
  candidate for a future ELIPS codec; name it `fp16`.
- `QT_4bit` / `QT_6bit` → **adapt** (future): adds a smaller tier; needs bit-packing like PQ.
- `RangeStat` variants → **adapt**: ELIPS hard-codes min/max (`RS_minmax, rs=0`). Keep that as the
  default but note `rangestat_arg` as the clip knob to add if outliers hurt recall. `RS_optim`/
  TurboQuant/EDEN → **skip** (research-grade, large surface).
- `QT_8bit_direct` / `_direct_signed` (no training, raw byte reinterpret) → **skip**: niche.

---

## 6. OPQ / OPQMatrix (`faiss/VectorTransform.h:258-276`)
- `OPQMatrix : LinearTransform`, params: `int M` (subquantizers), `int niter = 50` (outer iters),
  `int niter_pq = 4` (PQ iters per outer step), `int niter_pq_0 = 40` (PQ iters on first outer
  step), `size_t max_train_points = 256*256` resample cap (`VectorTransform.h:259-265`).
- Learns a `d×d` (or `d×d2`) orthonormal rotation applied **before** PQ to spread variance across
  subspaces; optional external `ProductQuantizer* pq` for training (`VectorTransform.h:267-273`).
- Precomputed tables: OPQ composes with IVFPQ's `precomputed_table` — the rotation is folded into
  the stored vectors, so query-side the precompute machinery from §3 applies unchanged.

Verdict:
- OPQ as a pre-rotation on PQ → **adopt** (ELIPS already does this: `rotation_` empty ⇒ PQ,
  populated ⇒ OPQ, one class — `ProductQuantizer.hpp:20-27`). Design matches FAISS intent.
- `niter=50` / `niter_pq=4` / `niter_pq_0=40` → **adapt**: ELIPS exposes a single `opq_iters`
  (`ProductQuantizer.hpp:33`). Mirror FAISS's split only if OPQ quality is insufficient; the
  outer=50 default is the number to document as the reference.
- `max_train_points = 65536` → **adopt**: sane resample cap for OPQ training; ELIPS should bound
  its OPQ training set similarly.
- ELIPS Procrustes-via-Newton-Schulz rotation solve (`KMeans.hpp:46-57`) is a legitimate
  **adapt** that avoids FAISS's LAPACK/eigen path — keep it, note the divergence.

---

## 7. Clustering / k-means (`faiss/Clustering.h`, `.cpp`)
`ClusteringParameters` defaults (`Clustering.h:22-86`) — the numbers ELIPS should mirror:
- `int niter = 25` (`Clustering.h:24`).
- `int nredo = 1` (`Clustering.h:27`) — redo N times, keep best objective.
- `bool spherical = false` (`Clustering.h:32`) — L2-normalize centroids each iter (IP clustering).
- `int min_points_per_centroid = 39` (`Clustering.h:44`) — below this → warning; <1/centroid throws.
- `int max_points_per_centroid = 256` (`Clustering.h:46`) — above this, training set is subsampled.
- `int seed = 1234` (`Clustering.h:50`); negative ⇒ clock-seeded.
- `int_centroids`, `update_index`, `frozen_centroids`, `check_input_data_for_NaNs=true`
  (`Clustering.h:34-56`).
- `ClusteringInitMethod init_method = RANDOM` (`Clustering.h:62-66`); also `KMEANS_PLUS_PLUS`,
  `AFK_MC2`. `use_super_kmeans = false` (`Clustering.h:79-85`, newer).

**Surprise:** default init is `RANDOM` uniform sampling, **not** k-means++ (`Clustering.h:63-66`).
k-means++ exists but is opt-in. ELIPS's deterministic strided init is a *third* choice and
intentionally reproducible (`KMeans.hpp:16-23`).

**Surprise:** `max_points_per_centroid=256` means FAISS silently subsamples the PQ/IVF training set
— training on fewer vectors than you passed. The `256 * ksub` rule-of-thumb for PQ training size
comes directly from this.

Verdict:
- `niter=25` → **adapt**: ELIPS passes `train_iters`/`iters` explicitly (`KMeans.hpp:22`); adopt
  **25** as the documented default value, keep the ELIPS param name.
- `nredo=1` → **adopt** (both the knob and default); cheap robustness lever ELIPS lacks.
- `max_points_per_centroid=256` / `min_points_per_centroid=39` → **adopt**: these drive the
  training-set-size heuristic (and the subsample in `IndexIVF::train`, §1). ELIPS should adopt the
  `256·ksub` training cap and the <39/centroid warning.
- `seed=1234` → **adapt**: ELIPS init is deterministic without RNG, so seed is moot — but if a
  randomized init is ever added, mirror `1234` and the "negative ⇒ clock" convention.
- `spherical` → **adopt** (ties to the auto-spherical-on-IP rule in §1).
- `init_method` enum / k-means++ / AFK-MC² / SuperKMeans → **skip**: ELIPS's deterministic strided
  init is a deliberate reproducibility choice (`KMeans.hpp:16-23`); do not trade byte-reproducibility
  for marginally better init. Keep ELIPS init, document it as a 4th `init_method`.

---

## Alias table (ELIPS ↔ FAISS) to adopt in docs
| ELIPS | FAISS | note |
|---|---|---|
| `n_lists` | `nlist` | IVF coarse list count |
| `n_probe` | `nprobe` | FAISS default 1; ELIPS keeps a larger default |
| `pq_dim` | `M` | PQ subquantizers |
| `pq_bits` | `nbits` | ELIPS currently 8-only; FAISS packs 4/6/… |
| `sub_dim` | `dsub` | `d / pq_dim` |
| `ksub` | `ksub` | `2^pq_bits` |
| `opq_iters` | `OPQMatrix.niter` | FAISS default 50 |
| `train_iters` | `ClusteringParameters.niter` | FAISS default 25 |
| codebook residual training | `by_residual=true` | train PQ/SQ on `x − centroid` |
| `make_lut`/`lut_distance` | ADC (`compute_distance_table`) | asymmetric |

## Top correctness items for ELIPS (ranked)
1. **`by_residual=true` + residual-trained sub-quantizers** (§1, §4) — biggest recall lever; verify
   ELIPS IVFPQ trains codebooks on residuals, not raw vectors.
2. **Precomputed per-list centroid table** for IVFPQ+L2 (§3) — main query-time speedup ELIPS lacks.
3. **`max_points_per_centroid=256` training cap + `min_points_per_centroid=39` warning** (§7).
4. **Adopt defaults**: `niter=25`, `nredo=1`, `max_codes=0`, auto-spherical on IP.
5. **Document the alias table** so naming disputes stop (ELIPS names win; FAISS names are synonyms).
