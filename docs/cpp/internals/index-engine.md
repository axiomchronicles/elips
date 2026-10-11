# Index Engine

The index engine is the similarity-search backbone of ELIPS. It defines an
abstract base class (`Index`) that decouples query logic from concrete index
implementations, then provides a family of backends: an exact brute-force
scanner used as a ground-truth oracle, a from-scratch HNSW graph, and a CPU
inverted-file (IVF) family (flat / PQ / scalar-quantized) built on the same
FAISS-aligned contract.

> **Naming:** the base class was `IndexPort`, the brute-force index was
> `ExactIndex`, and the graph was `HierarchicalGraphIndex`. Those names remain
> as `[[deprecated]]` aliases for one minor version — see `MIGRATION.md`. This
> page documents the final names.

**Maturity:** `IndexFlat` and `IndexHNSW` are stable. The IVF family
(`IndexIVFFlat`, `IndexIVFPQ`, `IndexIVFScalarQuantizer`) is newer CPU surface,
reachable through `index_factory`.

---

## Index — Abstract Interface

**Header:** `include/elips/index_engine/Index.hpp`

`Index` is a pure-virtual base class following the Dependency Inversion
Principle (DIP). Callers (`Collection`, `QueryExecutor`, benchmarks) depend
solely on `Index`, never on a concrete index type. The `make_index()` and
`index_factory()` factories select the concrete implementation at construction.

```cpp
class Index {
public:
    using Hit = std::pair<RecordID, float>;  // (id, ordering-normalized distance)

    // Train path (FAISS-aligned). Flat/graph indexes keep the no-op defaults;
    // the IVF family learns coarse centroids here before any add_with_ids.
    virtual void train(std::span<const float> training_vectors, std::size_t n);
    [[nodiscard]] virtual bool is_trained() const noexcept;   // default true

    virtual void reset();                                     // default no-op

    virtual void add_with_ids(const RecordID& id, std::span<const float> vector) = 0;
    virtual void remove(const RecordID& id) = 0;

    [[nodiscard]] virtual std::vector<Hit> search(
        std::span<const float> query, std::size_t k) const = 0;
    [[nodiscard]] virtual std::vector<Hit> range_search(
        std::span<const float> query, float radius) const;   // default throws

    [[nodiscard]] virtual std::size_t ntotal() const noexcept = 0;
    [[nodiscard]] virtual std::string_view type_name() const noexcept = 0;

    virtual void vacuum();                                    // default no-op
    [[nodiscard]] virtual std::size_t pending_removals() const noexcept;  // default 0
};
```

### Contract summary

| Method | Default | Meaning |
|--------|---------|---------|
| `train(vectors, n)` | no-op | Learn coarse structure. IVF family overrides; flat/graph need none. `training_vectors` is `n × dimension` row-major floats. |
| `is_trained()` | `true` | IVF family returns `false` until `train()` runs. |
| `reset()` | no-op | Drop all stored records (training/centroids may be retained). |
| `add_with_ids(id, vector)` | pure | Write path (was `insert`). Vectors are pre-validated/normalized by the vector engine. |
| `remove(id)` | pure | Removing an id that was never added is a no-op. |
| `search(query, k)` | pure | Up to `k` nearest, sorted ascending by distance. May return fewer. |
| `range_search(query, radius)` | **throws** `ElipsError` | Every record within `radius`, ascending. Overridden by `IndexFlat` and `IndexIVFFlat`; the default throws for index types that cannot answer it (ADC-only IVFPQ/SQ, graph HNSW). |
| `ntotal()` | pure | Live record count (was `size()`). |
| `type_name()` | pure | Short identifier, e.g. `"exact"`, `"graph"`, `"ivfflat"`. |
| `vacuum()` | no-op | Reclaim space held by removed records; the HNSW graph rebuilds here. |
| `pending_removals()` | `0` | Records tombstoned but not yet reclaimed. |

### The `Hit` Type

`Hit` is a `std::pair<RecordID, float>`. The `float` is an **ordering-normalized
distance**: smaller always means more similar, regardless of metric. This
contract is enforced by `distance()` in the vector engine (see
[metrics-engine.md](metrics-engine.md)). Callers sort ascending by `second` and
keep the smallest `k`.

### Non-copyable, Non-movable

Copy and move constructors/operators are explicitly deleted:

```cpp
Index(const Index&) = delete;
Index& operator=(const Index&) = delete;
Index(Index&&) = delete;
Index& operator=(Index&&) = delete;
```

Index instances are exclusively owned by `std::unique_ptr<Index>` and are never
sliced, shared, or moved. Indexes contain large internal data (vectors, graphs)
whose shallow moves would leave dangling references in adjacent state (e.g.
`id_to_node_` map entries pointing into `data_`). Deleting move forces all
transfers to go through the factory and remain behind a pointer.

---

## IndexFlat — Brute-force Ground Truth

**Header:** `include/elips/index_engine/IndexFlat.hpp`
**Source:** `src/IndexFlat.cpp`
**Maturity:** stable. `type_name()` is `"exact"`, or `"exact_quantized"` when a
quantizer is attached.

```cpp
IndexFlat(Metric metric, std::uint16_t dimension,
          quant::QuantizerPtr quantizer = {}) noexcept;
```

`IndexFlat` performs a brute-force linear scan over every stored vector and
serves two roles:

1. The default index for small collections where O(N × dim) is acceptable.
2. The **recall ground-truth oracle**: benchmarks and recall tests compare ANN
   results against `IndexFlat` results to measure recall@K.

It also implements `IndexTransferPort` (`export_snapshot()` / `import_snapshot()`)
and overrides `range_search()`.

### Quantized storage

With a trained `quant::Quantizer` supplied, rows are stored as codes rather than
fp32 and `search()` computes distances by asymmetric lookup (ADC), which makes
the scan both smaller and faster. Note "exact" then refers to the scan being
exhaustive, not to the distances being exact — a quantized `IndexFlat` is no
longer a ground-truth oracle. Only one storage buffer is ever populated:
`data_` (uncompressed fp32) **or** `codes_` (one `code_bytes()` row each).

### Data Structures

```
| id_0 | id_1 | ... | id_{N-1} |
| v_0[0..dim-1] | v_1[0..dim-1] | ... | v_{N-1}[0..dim-1] |
```

- `ids_` — `std::vector<RecordID>`, one entry per record, in insertion order.
- `data_` — `std::vector<float>`, **row-major** flat storage. Row `i` occupies
  indices `[i × dim, (i+1) × dim)`.

### add_with_ids — O(1) amortised

```cpp
void IndexFlat::add_with_ids(const RecordID& id, std::span<const float> vector) {
    ids_.push_back(id);
    data_.insert(data_.end(), vector.begin(), vector.end());  // fp32 path
}
```

Appends the ID and copies the vector data (or its code). No index structure is
rebuilt; adds are O(1) amortised.

### search — O(N × dim) + O(K log K)

Iterates over every row, computes `distance()` (NEON SIMD on ARM, see
[metrics-engine.md](metrics-engine.md)) or the ADC lookup for the compressed
path, then uses `std::partial_sort` to extract the smallest K — O(N × log K),
more efficient than a full sort when K ≪ N.

### remove — O(N)

Erases the ID and its row from both parallel arrays, shifting subsequent
elements. Acceptable because `IndexFlat` targets small collections / offline
benchmarking. `ntotal()` returns `ids_.size()`.

### Use as Ground-truth Oracle

In recall benchmarks, `IndexFlat` (unquantized) is populated with the same data
as the ANN index. For each query, both return `k` results and recall is:

```
recall@K = |ANN_topK ∩ Flat_topK| / |Flat_topK|
```

---

## IndexHNSW — Hierarchical Navigable Small World

**Header:** `include/elips/index_engine/IndexHNSW.hpp`
**Source:** `src/IndexHNSW.cpp`
**Maturity:** stable (primary ANN index). `type_name()` is `"graph"`, or
`"graph_quantized"` with a quantizer attached.

```cpp
IndexHNSW(Metric metric, std::uint16_t dimension, HnswParams params,
          quant::QuantizerPtr quantizer = {});
```

`IndexHNSW` is a from-scratch HNSW implementation providing approximate
nearest-neighbour search with configurable recall/latency/memory trade-offs. It
implements `IndexTransferPort` snapshot export/import.

### Configuration (`HnswParams`)

Defined in `include/elips/Config.hpp` (the old name `GraphParams` is a
`[[deprecated]]` alias).

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `max_connections` (M) | 16 | Max edges per node per layer above layer 0 |
| `ef_construction` | 200 | Beam width during add (greedy neighbour search) |
| `ef_search` | 50 | Beam width during query search (layer 0) |
| `compaction_ratio` | 0.2 | Tombstone fraction that triggers an automatic `vacuum()`; 0 disables auto-compaction |

At layer 0, each node may have up to `2 × M` (= 32 by default) links. The
internal `level_mult_` (mL) is `1.0 / ln(M)`, driving geometric level assignment.

### Data Structures

- **`NodeId`** (`std::uint32_t`) — each added vector gets a sequentially
  increasing node ID (the index into the flat arrays).
- **`data_`** — row-major fp32 (empty when compressed); **`codes_`** — row-major
  codes (empty when not). Same layout as `IndexFlat`.
- **`ids_`**, **`deleted_`** (tombstones), **`node_levels_`**.
- **`links_`** — a 3D ragged vector: `links_[node][level]` is a vector of
  neighbour `NodeId`s. Edges are maintained bidirectionally during construction.
- **`id_to_node_`** — `unordered_map<RecordID, NodeId>`.
- **`entry_point_`** / **`max_level_`** — global entry and top level; `-1` when
  empty. **`deleted_count_`** — running tombstone count.

### Quantized distance — the `Probe`

With a trained quantizer, node vectors are stored as codes and every distance
(during traversal and at query time) is computed by asymmetric lookup. The
`Probe` binds the per-query setup (O(dimension) table build) to the lifetime of
one traversal so ADC stays cheaper than the fp32 scan and the compressed/
uncompressed branch stays out of `search_layer()`'s inner loop. The rvalue
`Probe(const IndexHNSW&, std::vector<float>&&)` overload is deleted so a
temporary query cannot dangle.

### Level Assignment — `random_level()`

Standard HNSW formula: `u` is drawn uniformly from (0, 1] and the level is
`floor(-ln(u) × mL)` where `mL = 1/ln(M)`. This yields an exponentially-decaying
distribution: roughly 1/M of nodes appear at each successive level. A `u ≤ 0`
guard avoids `-inf` from `log(0)`.

### Beam Search — `search_layer()`

```cpp
std::vector<Scored> search_layer(const Probe& probe, NodeId entry,
                                 std::size_t ef, int level) const;
```

Beam search within a single layer using a min-heap of candidates and a max-heap
of results (both seeded with `entry`): pop the closest candidate, stop if it is
worse than the current worst result once `ef` results are held, otherwise visit
unvisited neighbours at `level`, scoring each via `probe.to(node)`. Used in three
contexts: greedy descent (`ef = 1` on upper layers), add-time neighbour search
(`ef = ef_construction`), and query search (`ef = max(ef_search, k)` on layer 0).

### Connect with Diversity Heuristic — `connect()`

Implements HNSW Algorithm 4: keep a candidate only if it is closer to the new
node than to any already-selected neighbour (prunes redundant edges), add forward
edges, then add reverse edges, pruning each neighbour's list back to `max_links`
(`2M` at layer 0, `M` above) by distance. Yields higher-recall graphs than
simply taking the M closest.

### Add Flow — `add_with_ids()`

1. **Allocate:** assign the next `NodeId`, generate a random level, push the
   id/vector(or code)/deleted-flag/level, size `links_` to `level + 1` empty
   lists, record in `id_to_node_`.
2. **Bootstrap:** if first node, set it as the entry point and return.
3. **Greedy descent:** from the global entry point, descend `max_level_` → `level + 1`
   with `search_layer(ef = 1)`.
4. **Connect intersecting layers:** for each layer `min(level, max_level_)` → 0,
   `search_layer(ef = ef_construction)` then `connect()`.
5. **Update entry point** if `level > max_level_`.

### Soft Delete — `remove()`

Nodes are **never physically removed** on `remove()`; the node is tombstoned
(`deleted_[node] = true`, `++deleted_count_`). This preserves graph connectivity
(tombstones still route traversal) and mirrors the collection's logical delete.
Consequences: `ntotal()` returns `ids_.size() - deleted_count_`, and
`id_to_node_` accumulates entries until a rebuild.

Tombstones are not free: they occupy the search beam and keep consuming memory.
Two mechanisms bound that cost — `search()` widens `ef` in proportion to the
live/deleted ratio so a tombstoned graph still returns `k` live results, and once
tombstones exceed `HnswParams::compaction_ratio` the index rebuilds itself in
`vacuum()`. Callers can force a rebuild anytime via `Collection::vacuum()`.

### Query Search — `search()`

Greedy descent `max_level_` → layer 1 (`ef = 1`), then layer-0 beam search
(`ef = max(ef_search, k)`), then post-filter: skip tombstones, collect up to `k`
hits. Construction complexity is **O(N × log N × M)**.

---

## CPU IVF Family

**Maturity:** newer CPU surface, reached through `index_factory`. All three
partition the space with a coarse quantizer of `nlist` k-means centroids and
**require `train()` before any `add_with_ids` / `search`** — using an untrained
IVF index raises `NotTrainedError` (a `ConfigError` subclass). `nprobe` is the
public, mutable recall/latency knob (default 1, FAISS-style, clamped to
`[1, nlist]` at search time): as `nprobe → nlist` the scan becomes exhaustive.

### IndexIVFFlat

**Header:** `include/elips/index_engine/IndexIVFFlat.hpp` · `type_name()` `"ivfflat"`

```cpp
IndexIVFFlat(Metric metric, std::uint16_t dimension, std::size_t nlist);
```

Full-precision (flat) postings: each vector is stored uncompressed in the
inverted list of its nearest centroid. `search()` probes the `nprobe` nearest
lists and scans them with the exact Metric kernel. As `nprobe → nlist` results
match `IndexFlat` exactly. Overrides `train()`, `reset()`, and `range_search()`.

### IndexIVFPQ

**Header:** `include/elips/index_engine/IndexIVFPQ.hpp` · `type_name()` `"ivfpq"`

```cpp
IndexIVFPQ(Metric metric, std::uint16_t dimension, std::size_t nlist,
           std::uint32_t pq_dim, std::uint32_t pq_bits = 8);
```

Product-quantized postings. Within each list a vector is stored as a PQ code of
its **residual** (vector minus its assigned centroid, FAISS `by_residual=true`) —
residual coding is the single biggest recall lever for IVFPQ. `search()` scores
codes with the `ProductQuantizer` ADC/LUT (no per-candidate decode) and merges
top-k. Constraints: `dimension % pq_dim == 0`, `pq_bits ∈ [4, 8]`. **LOSSY**:
recall rises monotonically with `nprobe` but never reaches the exact oracle,
trading that for a `code_bytes()`-byte footprint per vector vs `dimension × 4`.
No `range_search` (ADC-only).

### IndexIVFScalarQuantizer

**Header:** `include/elips/index_engine/IndexIVFScalarQuantizer.hpp` · `type_name()` `"ivfsq"`

```cpp
IndexIVFScalarQuantizer(Metric metric, std::uint16_t dimension, std::size_t nlist);
```

SQ8 postings: each component is linearly mapped onto a byte against a
per-dimension `[min, max]` range learned once over the whole training set.
Unlike IVFPQ, postings encode the **raw** vector, not the residual — SQ8's
per-dimension range is already near-lossless, and a single global codec keeps
the LUT shareable across lists. `search()` uses the `ScalarQuantizer` ADC/LUT.
LOSSY but high-recall at a flat ~4× compression (`dimension` bytes vs
`dimension × 4`). No `range_search` (ADC-only).

---

## Factories — Dependency Injection

**Header:** `include/elips/index_engine/IndexFactory.hpp`
**Source:** `src/IndexFactory.cpp`

Two free functions build an `Index` so callers depend only on the abstract base.

### `make_index()` — config-driven

```cpp
std::unique_ptr<Index> make_index(const Config& config, std::uint16_t dimension
#ifdef ELIPS_GPU_ENABLED
                                  , gpu::GpuBackend* gpu_backend = nullptr
#endif
);
```

Selects the concrete index from `Config::index()`. The enum is still
`IndexType { graph, exact }`:

| `config.index()` | Constructed type |
|------------------|------------------|
| `IndexType::exact` | `IndexFlat(metric, dimension)` |
| `IndexType::graph` | `IndexHNSW(metric, dimension, config.graph_params())` |

When GPU is compiled in and a backend is passed, the eligible index is placed on
the GPU behind the same `Index` interface.

### `index_factory()` — FAISS-style spec string

```cpp
std::unique_ptr<Index> index_factory(std::uint16_t dimension, Metric metric,
                                     std::string_view spec);
```

| Spec | Builds | Example |
|------|--------|---------|
| `"Flat"` | `IndexFlat` | `"Flat"` |
| `"IVF<nlist>,Flat"` | `IndexIVFFlat(nlist)` | `"IVF4096,Flat"` |
| `"IVF<nlist>,PQ<M>"` | `IndexIVFPQ(nlist, M)` | `"IVF4096,PQ8"` |
| `"IVF<nlist>,SQ8"` | `IndexIVFScalarQuantizer(nlist)` | `"IVF4096,SQ8"` |
| `"HNSW<M>"` | `IndexHNSW(M)` | `"HNSW32"` |

Throws `ConfigError` on an unrecognized or malformed spec. The same grammar
backs the C API (`elips_index_create`) and the Python `elips.index_factory`.

The returned `std::unique_ptr<Index>` is passed to the `Collection` constructor,
which stores it as `std::unique_ptr<Index> index_`. All callers interact through
the abstract base — no code outside the factories needs to know the concrete
type, which enables A/B testing, ground-truth verification, and new index types
with no API breakage.

