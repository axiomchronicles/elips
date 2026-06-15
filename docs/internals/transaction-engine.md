# ELIPS Transaction Engine

This document describes the transaction system: the `Transaction` and `TransactionVault` classes, eager validation, the pending operation queue, commit/rollback semantics, and the RAII auto-rollback guarantee.

## Class Definitions

```cpp
// include/elips/elips.hpp
namespace elips {

class TransactionVault {
public:
    RecordID place(const Vector& vector, Payload payload = {},
                   std::optional<RecordID> id = std::nullopt);
    void erase(const RecordID& id);

private:
    friend class Transaction;
    TransactionVault(Transaction& txn, std::string vault)
        : txn_(&txn), vault_(std::move(vault)) {}
    Transaction* txn_;
    std::string vault_;
};

class Transaction {
public:
    explicit Transaction(ElipsInstance& db) : db_(&db) {}
    ~Transaction();

    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    Transaction(Transaction&&) = delete;
    Transaction& operator=(Transaction&&) = delete;

    [[nodiscard]] TransactionVault vault(const std::string& name) {
        return TransactionVault{*this, name};
    }

    void commit();
    void rollback() noexcept { ops_.clear(); done_ = true; }

private:
    friend class TransactionVault;
    struct PendingOp {
        bool is_erase{false};
        std::string vault;
        Vector vector;
        Payload payload;
        std::optional<RecordID> id;
    };
    void enqueue_place(std::string vault, const Vector& vector, Payload payload,
                       std::optional<RecordID> id);
    void enqueue_erase(std::string vault, const RecordID& id);

    ElipsInstance* db_;
    std::vector<PendingOp> ops_;
    bool done_{false};
};

} // namespace elips
```

## Architecture

```
+------------------+         +-----------------------+         +------------------+
|  User Code       |         |  Transaction          |         |  ElipsInstance   |
|                  |         |                       |         |                  |
| begin_transaction| ------> | db_: ElipsInstance*   | -owns-> | vaults_: map<>   |
|                  |         | ops_: vector<PendingOp>|        | wal_: WAL*       |
| txn.vault("v")   | ------> | done_: bool           |         | ...              |
|                  |         |                       |         |                  |
| tv.place(v, p)   | ------> | enqueue_place()       |         |                  |
|                  |         | enqueue_erase()       |         |                  |
| txn.commit()     | ------> | commit()              | ------> | vault.place()    |
|                  |         |                       |         | vault.erase()    |
| txn.rollback()   | ------> | rollback()            |         |                  |
|                  |         |                       |         |                  |
| ~Transaction()   | ------> | auto-rollback         |         |                  |
+------------------+         +-----------------------+         +------------------+
        │                                 │
        │                                 │
        ▼                                 ▼
+------------------+         +-----------------------+
| TransactionVault |         |  PendingOp            |
|                  |         |                       |
| txn_: Transaction*|        | is_erase: bool        |
| vault_: string   |         | vault: string         |
+------------------+         | vector: Vector        |
                             | payload: Payload      |
                             | id: optional<RecordID>|
                             +-----------------------+
```

## PendingOp Queue

Operations within a transaction are buffered in a `std::vector<PendingOp>`. Each `PendingOp` captures the full state needed to apply the operation:

```cpp
struct PendingOp {
    bool is_erase{false};              // true = erase, false = place
    std::string vault;                 // target vault name
    Vector vector;                     // vector data (empty for erase)
    Payload payload;                   // metadata (empty for erase)
    std::optional<RecordID> id;        // explicit ID or nullopt for auto-generate
};
```

- **Insert operations**: `is_erase=false`, `vector` contains the user-provided vector, `payload` contains metadata, `id` is the pre-generated RecordID.
- **Erase operations**: `is_erase=true`, `vector` and `payload` are empty, `id` contains the target RecordID.

## Eager Validation

`enqueue_place()` validates inputs immediately, before the operation is buffered:

```cpp
void Transaction::enqueue_place(std::string vault, const Vector& vector,
                                Payload payload, std::optional<RecordID> id) {
    // Validate eagerly so commit() can never fail mid-batch (atomicity).
    if (vector.dimension() != db_->config().dimension()) {
        throw DimensionMismatch{"vector dimension does not match database"};
    }
    if (!all_finite(vector.values())) {
        throw InvalidVector{"vector contains NaN or Inf"};
    }
    ops_.push_back(PendingOp{false, std::move(vault), vector, std::move(payload),
                             std::move(id)});
}
```

### Why Eager Validation?

Without eager validation, `commit()` would need to handle validation failures mid-batch, breaking the atomicity promise:

```
# WITHOUT eager validation (BAD):
txn.vault("a").place(vec1, {});
txn.vault("b").place(vec2, {});    # this could fail during commit()
txn.commit();                       # what happens to vec1? Already applied? Not applied?
```

With eager validation:
```
# WITH eager validation (GOOD):
txn.vault("a").place(vec1, {});    # validated immediately → OK
txn.vault("b").place(vec2, {});    # DimensionMismatch thrown immediately
# Transaction is still clean. User can fix and retry, or rollback.
```

**Validation performed eagerly:**
1. **Dimension check**: `vector.dimension() != db_->config().dimension()` → `DimensionMismatch`
2. **Finiteness check**: `!std::isfinite(v)` for any value → `InvalidVector`

**Validation deferred to commit (performed by Vault):**
- Vector normalization (`Vault::prepare()` normalizes cosine vectors)
- RecordID uniqueness (inserts with duplicate IDs will overwrite — the record store is a map)

## RecordID Pre-Generation

```cpp
RecordID TransactionVault::place(const Vector& vector, Payload payload,
                                 std::optional<RecordID> id) {
    const RecordID assigned = id.value_or(RecordID::generate());
    txn_->enqueue_place(vault_, vector, std::move(payload), assigned);
    return assigned;  // ID is returned immediately, before commit
}
```

- If the caller provides an `id`, it is used as-is.
- If no `id` is provided, `RecordID::generate()` creates a UUIDv7 **immediately**.
- The generated ID is returned to the caller right away, even though the operation hasn't been committed yet.
- This allows the caller to log the ID or use it in follow-up operations within the same transaction.

## Commit

```cpp
void Transaction::commit() {
    for (auto& op : ops_) {
        Vault& vault = db_->vault(op.vault);
        if (op.is_erase) {
            vault.erase(*op.id);                     // WAL append + index remove + store erase
        } else {
            vault.place(op.vector, op.payload, op.id); // prepare + WAL + index + store
        }
    }
    ops_.clear();
    done_ = true;
}
```

### Commit Semantics

1. **Sequential application**: Operations are applied one by one in the order they were enqueued.
2. **Each operation triggers WAL**: `vault.place()` and `vault.erase()` both write to the WAL before mutating in-memory state, ensuring each operation is durably recorded.
3. **Not atomic across operations**: If the process crashes mid-commit, WAL replay on recovery will re-apply the already-WAL'd operations. Unapplied operations in the batch (after the crash point) are lost, but the pre-crash state is consistent.
4. **Single-writer safety**: Under the single-writer model, no concurrent mutations can interleave with the commit batch.
5. **Idempotent-ish**: Calling `commit()` after `done_=true` would be a no-op (though the current implementation doesn't guard against double-commit — `ops_` is empty, so the loop does nothing).

### Commit Failure Handling

Since eager validation ensures all vectors are dimensionally valid and finite, the only failure points during `commit()` are:
- **Storage errors**: WAL write failures, filesystem full → `StorageError` exception
- **Vault not found**: `db_->vault(name)` creates the vault lazily, so this always succeeds

If a `StorageError` occurs mid-commit, the exception propagates to the caller. The transaction is left in a partially-applied state (the WAL already contains committed operations). If the process continues, the caller would typically handle the error by checkpointing. If the process crashes, WAL replay recovers the committed portion.

## Rollback

```cpp
void rollback() noexcept { ops_.clear(); done_ = true; }
```

- **Trivially safe**: Simply clears the pending operations vector. No in-memory state was modified, so no undo is needed.
- **No exceptions**: Marked `noexcept` — guaranteed to succeed.
- **Marks transaction as done**: `done_ = true` prevents the destructor from calling `rollback()` again.

## RAII Auto-Rollback (Destructor)

```cpp
Transaction::~Transaction() {
    if (!done_) {
        rollback();  // discard buffered, never-applied ops
    }
}
```

- If the `Transaction` object is destroyed without `commit()` or `rollback()` being called explicitly, the destructor automatically calls `rollback()`.
- This ensures that uncommitted operations are never silently applied.
- Example: exception thrown while building a transaction → stack unwinding destroys `Transaction` → auto-rollback.

### Scenarios

| Scenario | `done_` | Destructor Action |
|----------|---------|-------------------|
| `commit()` called successfully | `true` | No-op |
| `rollback()` called explicitly | `true` | No-op |
| Exception after some enqueued ops | `false` | Auto-rollback |
| Transaction object goes out of scope | `false` | Auto-rollback |
| Python context manager exit with exception | `false` | Auto-rollback |

## Python Integration

The `TransactionHolder` pattern integrates the C++ transaction with Python's lifetime management:

```cpp
struct TransactionHolder {
    py::object db_ref;              // Keeps Database alive
    elips::Transaction txn;         // C++ transaction

    TransactionHolder(py::object db, elips::ElipsInstance& instance)
        : db_ref(std::move(db)), txn(instance) {}
};
```

### Python Context Manager

```python
# Python binding:
.def("__enter__", [](TransactionHolder& h) -> TransactionHolder& { return h; })
.def("__exit__", [](TransactionHolder& h, const py::object& exc_type, ...) -> bool {
         if (exc_type.is_none()) {
             h.txn.commit();       # clean exit → commit
         }
         return false;            # don't suppress exception
     })
```

Python usage:
```python
with db.begin_transaction() as txn:
    v = txn.vault("data")
    v.place(vector1, {"tag": "a"})
    v.place(vector2, {"tag": "b"})
    # no explicit commit needed — __exit__ calls commit() on clean exit
    # if an exception is raised, __exit__ does NOT commit → destructor rollbacks

# Equivalent to:
txn = db.begin_transaction()
try:
    v = txn.vault("data")
    v.place(vector1, {"tag": "a"})
    v.place(vector2, {"tag": "b"})
    txn.commit()
except:
    # ~Transaction() auto-rollback
    raise
```

## TransactionVault

`TransactionVault` is a lightweight handle that bridges transaction operations to specific vaults:

```cpp
class TransactionVault {
public:
    RecordID place(const Vector& vector, Payload payload = {},
                   std::optional<RecordID> id = std::nullopt);
    void erase(const RecordID& id);

private:
    // Only constructible by Transaction::vault()
    TransactionVault(Transaction& txn, std::string vault)
        : txn_(&txn), vault_(std::move(vault)) {}
    Transaction* txn_;
    std::string vault_;
};
```

- **Private constructor**: Only `Transaction::vault()` can create a `TransactionVault`. This ensures the vault name is always bound to a transaction.
- **Delegate to Transaction**: `place()` calls `txn_->enqueue_place()`; `erase()` calls `txn_->enqueue_erase()`.
- **No own state**: `TransactionVault` is essentially a (transaction-pointer, vault-name) pair with convenience methods.
- **Copyable**: `TransactionVault` has default copy semantics (both members are trivially copyable). This is safe because the underlying `Transaction` outlives any copies.

## Lifetime Dependencies

```
ElipsInstance (owner)
  │
  ├── owns Vault objects (via vaults_ map)
  │
  └── referenced by Transaction (raw pointer db_)
        │
        └── referenced by TransactionVault (raw pointer txn_)
```

- `Transaction` holds a raw `ElipsInstance*`. The `ElipsInstance` MUST outlive the `Transaction`.
- `TransactionVault` holds a raw `Transaction*`. The `Transaction` MUST outlive the `TransactionVault`.
- In Python, `TransactionHolder` ensures the Python `Database` object stays alive via `db_ref`.

## Concurrency Model

The transaction engine operates under the **single-writer model**:

- **No internal locking**: The transaction itself is not thread-safe. It assumes external synchronization via the `LockManager`.
- **No concurrent transactions**: Only one process can hold the write lock. Within that process, only one `Transaction` should be active.
- **Nested transactions**: Not supported. Creating a second `Transaction` while one is active will reference the same `ElipsInstance` but will not coordinate with the first.
- **Read-your-writes**: Not available within a transaction. Writes are deferred to `commit()` and are not visible to reads within the same transaction. This is by design — the Vault's `seek()` always reads from the committed state.

## Isolation Level

**Atomic Batched Writes** under the single-writer model:

- **Atomicity**: All operations in a transaction are applied together (or none, via rollback). WAL ensures durability.
- **Consistency**: Eager validation ensures dimension/finiteness constraints. The WAL records every mutation atomically.
- **Isolation**: Single-writer ensures no dirty reads or lost updates. Other processes see a consistent snapshot via checkpoint + WAL replay.
- **Durability**: Each operation writes to the WAL during `commit()`. The WAL is flushed per the `Durability` setting.

This is effectively **serializable isolation** under the single-writer constraint — all transactions are serialized by the lock.

## Transaction Lifecycle State Machine

```
                    begin_transaction()
                           │
                           ▼
                    ┌──────────────┐
                    │   ACTIVE     │
                    │ done_=false   │
                    │ ops_=[]       │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         enqueue ops    commit()    ~Transaction()
              │            │         (exception/unwind)
              ▼            ▼            │
      ┌──────────────┐ ┌──────────┐    │
      │   ACTIVE     │ │ COMMITTED│    │
      │ ops_=[...]   │ │ done_=T  │    │
      └──────┬───────┘ │ ops_=[]  │    │
             │          └──────────┘    │
        commit()                        │
             │                          ▼
             ▼                   ┌──────────────┐
      ┌──────────────┐          │  ROLLED BACK  │
      │  COMMITTED   │          │ done_=true    │
      │ done_=true   │          │ ops_=[]       │
      │ ops_=[]      │          └──────────────┘
      └──────────────┘
```