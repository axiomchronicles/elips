# Transactions, Concurrency & Locking — v1.1.0 Updates

This document covers transaction atomicity improvements (**F4**), multi-threaded reader/writer locking (**F6**), and sealed vault lifecycle safety (**F7**) introduced in ELIPS v1.1.0.

---

## F4 — Full Transaction Atomicity with Undo Logs

### Problem Definition
In pre-v1.1.0 releases, `Transaction::commit()` applied enqueued operations sequentially by calling `Vault::place()` or `Vault::erase()` in a simple `for` loop. If an operation failed mid-batch (e.g., target vault set to read-only, disk full during WAL write, or file descriptor IO failure):
- Operations prior to the failing point remained applied in memory and persisted to the WAL.
- Operations after the failing point were skipped.
- `Transaction::rollback()` only cleared the uncommitted pending operation queue, leaving the database in a partially-applied state (violating transaction atomicity).

### Implementation & Fix
v1.1.0 introduces three coupled mechanisms to deliver complete all-or-nothing transaction atomicity:

1. **Pre-Flight Writability Verification:**
   Before executing any mutations, `commit()` iterates through all queued operations and verifies that every target `Vault` is constructible and writable. If any target vault is read-only or sealed, `commit()` aborts immediately before modifying any state.

2. **In-Memory Undo Log:**
   During `commit()`, each record modification creates an `UndoEntry` capturing its prior state:
   ```cpp
   struct UndoEntry {
       std::string vault;
       RecordID id;
       std::optional<Record> previous; // std::nullopt if record did not exist prior to batch
   };
   ```
   If any exception occurs during vector preparation, index insertion, or WAL write, `commit()` catches the exception, iterates through the `UndoEntry` stack in reverse order, restores all modified records to their exact pre-batch state, and re-throws the exception.

3. **WAL Transaction Framing (`txn_begin` / `txn_commit`):**
   Batched transaction records are written to the WAL surrounded by `Op::txn_begin` and `Op::txn_commit` markers. During crash recovery replay, if a `txn_begin` is encountered without a matching `txn_commit` before EOF, all operations within that unclosed transaction frame are discarded.

```
[Begin Transaction] -> Pre-flight check -> Record Undo Entry -> Apply Op & WAL
                                                                     │
  ┌───────────────── Error Encountered (IO/Disk) ────────────────────┘
  ▼
[Rollback In-Memory via Undo Log (Reverse Order)] -> Re-throw Exception
```

---

## F6 — In-Process Thread-Safe Reader/Writer Locking

### Problem Definition
`LockManager` utilizes POSIX file advisory locks (`flock`). Operating system file locks coordinate processes across the OS, but offer zero protection between multiple C++ threads sharing a single `ElipsInstance` pointer in the same process address space.

In pre-v1.1.0 releases, concurrent calls to `Vault::place()` or `Vault::seek()` from different threads resulted in data races on `HierarchicalGraphIndex` adjacency lists (`links_`), vector buffers (`data_`), and internal maps (`id_to_node_`). Additionally, `Vault::records()` returned a direct reference to internal `std::map<RecordID, Record>`, allowing concurrent modifications to invalidate iterators during iteration.

### Implementation & Fix
v1.1.0 implements a two-tier in-process synchronization hierarchy:

1. **Vault-Level Shared Mutex (`std::shared_mutex`):**
   - Read operations (`seek()`, `fetch()`, `scan()`, `explain_seek()`) acquire a **shared read lock** (`std::shared_lock`). Unlimited concurrent readers are supported.
   - Write operations (`place()`, `erase()`, `rebuild_index()`, `vacuum()`) acquire an **exclusive write lock** (`std::unique_lock`).

2. **Instance-Level Recursive Mutex (`std::recursive_mutex`):**
   - Guards vault creation, vault lookup maps, WAL handle operations, and database lifecycle flags.
   - `Transaction::commit()` acquires the instance lock for the duration of the entire batch commit, preventing concurrent threads from interleaving operations into a committing transaction.

3. **Thread-Safe Snapshot Returns:**
   - `Vault::records()` signature updated: now returns `std::vector<StoredRecord>` (a deep snapshot copy taken under shared read lock) instead of a live map reference.

```cpp
// C++ Thread Safety Example
auto db = elips::open("./db_data");

// Thread 1: Concurrent Writer
std::thread t1([&]() {
    db->vault("vectors").place(vector1, {{"tag", "writer1"}});
});

// Thread 2: Concurrent Reader (Safe under shared_mutex)
std::thread t2([&]() {
    auto results = db->vault("vectors").seek(query_vec, 10);
});

t1.join(); t2.join();
```

---

## F7 — Sealed Vault State Lifecycle Safety

### Problem Definition
Calling `Database.close()` or `ElipsInstance::close()` checkpoints persistent state, flushes the WAL, and releases file locks. 

In pre-v1.1.0 releases, calling `close()` left child `Vault` handles alive and holding valid memory pointers. If user code subsequently invoked `vault.place()`, the vault detected a closed WAL handle and bypassed logging—updating in-memory indexes while returning a successful `RecordID`. When the process exited, those post-close writes disappeared completely, resulting in silent data loss.

### Implementation & Fix
1. `ElipsInstance::close()` sets a `sealed_ = true` atomic flag on the instance and seals all created child vaults.
2. Any subsequent write operation (`place()`, `erase()`, `vacuum()`, `rebuild_index()`) on a sealed vault immediately throws `elips::StorageError("Operation failed: vault is sealed because database instance has been closed")`.
3. Read operations (`fetch()`, `seek()`, `scan()`) on sealed vaults continue to function normally over the in-memory state.

```python
import elips

db = elips.open("/var/data/vec_db")
vault = db.vault("documents")

# Close database
db.close()

# Reads remain functional
doc = vault.fetch("018f3c7a-0000-7000-8000-000000000001")

# Writes fail explicitly (F7 fix)
try:
    vault.place([0.1, 0.2, 0.3], {"title": "post-close write"})
except elips.StorageError as err:
    print("Caught expected error:", err)
```
