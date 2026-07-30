# ELIPS v1.1.0 Release Notes

**Release Date:** July 30, 2026  
**Version:** v1.1.0  
**Type:** Production Correctness & Hardening Release  

---

## Executive Summary

ELIPS v1.1.0 is a major infrastructure and reliability release that resolves 9 critical architectural issues (F1–F9) identified during rigorous auditing, fuzzing, and sanitizer runs. This release guarantees disk durability across OS power losses, all-or-nothing transaction atomicity, thread-safe in-process memory sharing, bounded HNSW tombstone memory overhead, and leak-free GPU suballocation.

---

## Release Architecture & Navigation

The documentation for v1.1.0 is organized into specialized sections:

- [Storage & Write-Ahead Log (WAL)](storage-and-wal.md) — F1 (Disk Syncing), F2 (Bounded Length Prefixes), F3 (Linear O(n) WAL Replay).
- [Transactions & Concurrency Control](transactions-and-locking.md) — F4 (Undo Log Atomicity), F6 (In-Process `std::shared_mutex`), F7 (Sealed Vault Lifecycles).
- [HNSW Indexing & Compaction](hnsw-and-compaction.md) — F5 (Adaptive `ef` Beam, Tombstone Compaction, `vacuum()`, Adaptive Filter Re-probing).
- [GPU Acceleration & Suballocator](gpu-engine.md) — F8 (Build Gating), F9 (Suballocator Coalescing & Remainder Splitting).
- [Python SDK Reference](python-sdk.md) — Full coverage of new Python APIs (`Filter`, `GpuDevice`, `Accelerator`, `replay_wal`, `vacuum`, `durability`).
- [Testing & Quality Assurance](testing-and-tooling.md) — Sanitizers (ASan, UBSan, TSan), libFuzzer (`elips_fuzz_wal`), and Parser Hardening.

---

## Highlights at a Glance

### 1. Hardened Persistence (F1, F2, F3)
- **Power-Loss Durability (F1):** Replaced stream flushing with kernel-level sync (`fdatasync` on Linux, `F_FULLFSYNC` on macOS) and directory fsync on atomic manifest/segment renames.
- **Safety Against Corrupt Logs (F2):** Length-prefixed string and payload reads are validated against remaining stream size before allocating memory.
- **O(n) WAL Replay (F3):** Eliminated $O(n^2)$ tail-copying during WAL recovery.

### 2. Transaction Atomicity & Isolation (F4, F6, F7)
- **In-Memory Undo Log (F4):** Batched mutations in `commit()` log previous record states before mutating. If an I/O error occurs mid-batch, prior states are restored in reverse order.
- **In-Process Concurrency (F6):** Reader/writer locking using `std::shared_mutex` per vault and an instance-level `std::recursive_mutex`. `Vault::records()` now safely returns a snapshot copy.
- **Sealed Vault Protection (F7):** Closed instances seal all child vaults, rejecting subsequent writes with `StorageError` to prevent silent memory-only mutations.

### 3. Graph Maintenance & Search Precision (F5)
- **Tombstone Compaction:** Soft-deleted HNSW nodes no longer degrade search recall or cause unbounded memory growth. `search()` dynamically widens `ef` beam based on tombstone ratios, while `vacuum()` reclaims dead nodes when tombstones cross `GraphParams::compaction_ratio` (default 0.2).
- **Adaptive Filter Search:** ANN search under metadata filters dynamically re-probes with an expanding beam (4x expansion) until requested `top` matches are satisfied.

### 4. GPU Suballocator & Build Gating (F8, F9)
- **Leak-Free Suballocator (F9):** Splitting remainders on allocation, coalescing adjacent spans on deallocation, tracking live block sizes, and providing accurate `bytes_available()`.
- **Platform-Gated GPU Build (F8):** GPU engine builds only on supported platforms (Metal defaults to Apple host).

---

## Summary Table of Behavioral Changes

| Feature / Area | Pre-v1.1.0 Behavior | v1.1.0 Behavior |
| :--- | :--- | :--- |
| **WAL Sync** | `std::ofstream::flush()` (Page cache only) | `fdatasync` / `F_FULLFSYNC` + directory sync |
| **WAL Replay** | $O(n^2)$ due to string tail-copies | $O(n)$ in-place stream buffer parsing |
| **Failed Commit** | Partial batch left in memory | Full rollback via in-memory Undo Log |
| **Post-Close Writes** | Silently mutated un-persisted memory | Throws `StorageError` ("vault is sealed") |
| **Thread Safety** | Inter-process `flock` only (thread races) | `std::shared_mutex` per Vault + instance lock |
| **`Vault::records()`**| Reference to live map (race hazard) | Returns thread-safe `std::vector<StoredRecord>` copy |
| **Deleted Vectors** | Soft-deleted forever (recall degrades) | Adaptive `ef` + automatic `vacuum()` compaction |
| **GPU Suballocator** | Dropped leftovers & unmerged frees | Immediate remainder split & free span coalescing |

---

## Quick Start / Code Snippets

### Python Batch Atomicity & Vacuuming
```python
import elips

# Open with standard durability (full fsync guarantees)
db = elips.open("/var/data/elips_db")
vault = db.vault("vectors")

# Atomic Transaction
with db.begin_transaction() as txn:
    v = txn.vault("vectors")
    v.place([0.1, 0.2, 0.3], {"tag": "doc1"})
    v.place([0.4, 0.5, 0.6], {"tag": "doc2"})

# Maintenance & Tombstone Reclaim
print("Pending tombstones:", vault.pending_removals)
vault.vacuum()  # Reclaims deleted HNSW nodes
```

### C++ Modern In-Process Locking & Config
```cpp
#include "elips/Config.hpp"
#include "elips/elips.hpp"
#include <iostream>

int main() {
    elips::Config config;
    config.dimension(128)
          .metric(elips::Metric::cosine)
          .durability(elips::Durability::standard)
          .graph_params({.max_connections = 16, .ef_construction = 200, .compaction_ratio = 0.15f});

    auto db = elips::open("./db_data", config);
    auto& vault = db->vault("docs");

    // Thread-safe records snapshot
    std::vector<elips::StoredRecord> snapshot = vault.records();
    std::cout << "Records count: " << snapshot.size() << std::endl;

    return 0;
}
```
