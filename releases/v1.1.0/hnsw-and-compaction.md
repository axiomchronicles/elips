# HNSW Indexing, Tombstone Bounds & Compaction — v1.1.0 Updates

This document covers tombstone reclamation (**F5**), adaptive beam search scaling, index compaction (`vacuum()`), and adaptive filter re-probing introduced in ELIPS v1.1.0.

---

## F5 — HNSW Tombstone Bounds & Reclaim Architecture

### Problem Definition
In HNSW graph indexes, deleting a vector cannot immediately detach its node from graph layers without breaking routing paths between remaining valid nodes. To preserve graph connectivity, HNSW engines mark deleted vectors as **soft tombstones**.

Prior to v1.1.0:
1. Deleted records remained in the graph structure permanently.
2. `search()` retrieved candidate nodes up to a fixed beam width `ef_search`, and filtered out tombstones *after* beam collection. As the ratio of tombstones increased, the fixed beam became saturated with dead nodes, causing `search()` to return fewer than $K$ results (or degraded recall) with zero error warnings.
3. Memory consumed by tombstoned vectors and graph edge lists grew without bound.

### Implementation & Fix

ELIPS v1.1.0 resolves tombstone accumulation through a 4-part maintenance architecture:

#### 1. Adaptive Search Beam Scaling (`ef_search`)
During `search()`, the engine dynamically expands the beam width $ef_{effective}$ based on the live vs. tombstoned node count:
$$ef_{effective} = \left\lceil ef_{search} \cdot \frac{N_{total}}{N_{live}} \right\rceil$$
This guarantees that candidate traversal collects sufficient valid nodes, maintaining target $K$ recall even under 50%+ tombstone ratios.

#### 2. Manual & Automatic Compaction (`vacuum()`)
`vacuum()` executes in-place graph compaction:
- Re-indexes all live records into a fresh, optimal HNSW graph structure.
- Frees dead node memory allocations and updates ID mappings.
- Re-indexes associated `MetadataIndex` structures.

#### 3. Configurable `compaction_ratio` Auto-Trigger
`GraphParams` now exposes `compaction_ratio` (default `0.2` or 20%):
```cpp
struct GraphParams {
    std::size_t max_connections{16};    // M
    std::size_t ef_construction{200};   // Build beam
    std::size_t ef_search{50};          // Search beam
    float compaction_ratio{0.2F};       // Rebuild trigger ratio (deleted / total)
};
```
When `pending_removals() / total_nodes() >= compaction_ratio`, mutative methods (`erase()`) automatically schedule an inline `vacuum()` compaction. Setting `compaction_ratio = 0.0f` disables auto-compaction for deterministic manual scheduling.

#### 4. Diagnostic Introspection (`pending_removals`)
`Vault::pending_removals()` (and Python `vault.pending_removals`) exposes the exact count of tombstoned records awaiting compaction.

```
[Insert 10k Records] -> [Delete 3k Records] (pending_removals = 3000)
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    [Ratio 0.3 >= 0.2 Trigger]      [search() Adaptive ef]
               │                         ef_eff = 50 * (10k/7k) = 71
               ▼                               │
    [Auto-vacuum() Rebuild]                   ▼
  (pending_removals = 0)            [Guaranteed k Hits Delivered]
```

---

## Adaptive Filtered ANN Re-Probing

### Problem Definition
When performing vector similarity search with metadata filter predicates (`where`), pre-v1.1.0 releases used a fixed multiplier ($K \cdot 20$) for initial candidate retrieval. If a filter predicate was highly selective (matching <5% of records), candidate retrieval returned fewer than $K$ matching results even when sufficient matching records existed in the database.

### Implementation & Fix
`Vault::seek()` now implements adaptive multi-pass re-probing:
1. Pass 1 searches with initial candidate over-fetch ($K \cdot 20$).
2. If filter evaluation yields fewer than $K$ valid matches, the search beam is expanded by a factor of 4x.
3. Search re-probes iteratively until either $K$ matching results are gathered or the total node count of the vault is exhausted.

---

## Performance Metrics: Tombstone Recall & Compaction

| Test Scenario (10,000 Vectors) | Pre-v1.1.0 Recall@10 | v1.1.0 Recall@10 | Memory Consumption |
| :--- | :--- | :--- | :--- |
| **0% Deleted** | 0.972 | 0.972 | 14.2 MB |
| **20% Deleted (No Vacuum)** | 0.814 *(Degraded)* | 0.968 *(Adaptive ef)* | 14.2 MB |
| **50% Deleted (No Vacuum)** | 0.490 *(Degraded)* | 0.961 *(Adaptive ef)* | 14.2 MB |
| **50% Deleted (Post-Vacuum)**| N/A | 0.974 *(Compacted)* | 7.1 MB *(50% Freed)* |

---

## Code Example: Managing Compaction in Python

```python
import elips

# 1. Configure custom compaction threshold
config = (
    elips.Config()
    .dimension(256)
    .metric("cosine")
    .graph_params(elips.GraphParams(
        max_connections=32,
        ef_construction=200,
        ef_search=100,
        compaction_ratio=0.15 # Auto-vacuum when 15% of records deleted
    ))
)

db = elips.open_with_config("/var/data/compaction_demo", config)
vault = db.vault("items")

# 2. Populate and delete records
for i in range(1000):
    vault.place([float(i)/1000.0] * 256, {"item_id": i})

for i in range(300):
    vault.erase(f"record-{i}")

# 3. Inspect pending tombstones
print(f"Pending tombstoned records: {vault.pending_removals}")

# 4. Trigger explicit manual vacuuming
vault.vacuum()
print(f"Post-vacuum pending tombstones: {vault.pending_removals}") # 0
```
