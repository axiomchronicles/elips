# API Reference: Transaction & TransactionCollection

**Maturity:** stable (v1.0+).

Namespace `elips`. Buffered atomic batch mutations, declared in
`include/elips/elips.hpp`.

> Renamed in v1.1: `TransactionVault`→`TransactionCollection`,
> `Transaction::vault(name)`→`Transaction::collection(name)`, verbs
> `place`→`add` and `erase`→`remove` (old names kept as deprecated shims). See
> `MIGRATION.md`.

Transactions buffer multiple `add()` and `remove()` operations, then apply them
atomically via `commit()`. Operations are validated eagerly during enqueue so
`commit()` cannot fail mid-batch on validation; if a late failure still occurs,
the already-applied ops are undone and the error is rethrown.

---

## TransactionCollection

A collection-scoped handle within a transaction. Obtained from
`Transaction::collection()`. Tied to its parent transaction's lifetime.

```cpp
class TransactionCollection {
public:
    RecordID add(const Vector& vector, Payload payload = {},
                 std::optional<RecordID> id = std::nullopt);
    void remove(const RecordID& id);
    // deprecated shims: place(...) -> add(...), erase(id) -> remove(id)
private:
    friend class Transaction;
    TransactionCollection(Transaction& txn, std::string collection);
    Transaction* txn_;
    std::string collection_;
};
```

### add()

```cpp
RecordID add(const Vector& vector, Payload payload = {},
             std::optional<RecordID> id = std::nullopt);
```

Enqueues a vector insertion for batch commit.

| Parameter | Type | Description |
|-----------|------|-------------|
| `vector` | `const Vector&` | The embedding vector to insert |
| `payload` | `Payload` | Metadata key-value pairs |
| `id` | `std::optional<RecordID>` | Explicit RecordID; auto-generated if omitted |

**Returns**: `RecordID` — the assigned record identity. Generated eagerly (if
not provided) so the caller has the ID immediately, even before `commit()`.

**Validation** (eager, at enqueue time):
1. Checks `vector.dimension() != db_->config().dimension()` — throws `DimensionMismatch`.
2. Checks vector values are finite — throws `InvalidVector` if any NaN/Inf.

This ensures that if enqueue succeeds for all operations in the transaction,
`commit()` cannot fail due to validation errors (maintaining atomicity).

**Note**: Unlike `Collection::add()`, the vector is **not normalized** at enqueue
time. Normalization happens inside `Collection::add()` when `commit()` is called.

### remove()

```cpp
void remove(const RecordID& id);
```

Enqueues a record deletion for batch commit.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `const RecordID&` | The record identity to remove |

No immediate validation — the remove is applied during `commit()` via
`Collection::remove()`, which silently returns `false` if the record does not
exist.

### Constructor (Private)

```cpp
TransactionCollection(Transaction& txn, std::string collection);
```

Constructed by `Transaction::collection()`. Associates the collection name with
the parent transaction. Not directly constructible by users.

---

## Transaction

```cpp
class Transaction {
public:
    explicit Transaction(Database& db);
    ~Transaction();

    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    Transaction(Transaction&&) = delete;
    Transaction& operator=(Transaction&&) = delete;

    TransactionCollection collection(const std::string& name);
    void commit();
    void rollback() noexcept;
    // deprecated shim: vault(name) -> collection(name)
};
```

### Construction

```cpp
explicit Transaction(Database& db);
```

Created via `Database::begin_transaction()`:

```cpp
auto txn = db->begin_transaction();
```

The transaction holds a raw pointer to the database. The database must outlive
the transaction. **Not copyable, not movable** — a `Transaction` is a
stack-local value.

### destructor (Auto-Rollback)

```cpp
~Transaction();
```

If `commit()` was not called (`done_ == false`), the destructor calls
`rollback()` to clear buffered operations; nothing is applied. This ensures
abandoned transactions never leave partial state. Always explicitly call
`commit()` or `rollback()` for clarity.

### collection()

```cpp
TransactionCollection collection(const std::string& name);
```

Obtains a `TransactionCollection` handle to enqueue operations on the named
collection. Multiple collections can be used within the same transaction:

```cpp
auto txn = db->begin_transaction();
txn.collection("docs").add(vec1, payload1);
txn.collection("images").add(vec2, payload2);
txn.collection("docs").remove(some_id);
txn.commit();  // atomic across both collections
```

### commit()

```cpp
void commit();
```

Applies all buffered operations to the database atomically.

**Behavior**:
1. Iterates `ops_` in order, recording an undo entry for each.
2. For each op: resolves the collection via `db_->collection(op.collection)`,
   then calls `Collection::add()` or `Collection::remove()`.
3. Each `add()` call triggers validation, normalization, WAL logging, index
   insertion, and record-store update. Each `remove()` triggers WAL logging,
   index removal, and record-store removal.
4. On success, clears `ops_` and sets `done_ = true`.

**Atomicity**: a failure partway through (read-only collection, disk-full WAL
write) **undoes the ops already applied** and rethrows, so the batch applies
fully or not at all. Because validation is eager at enqueue time, such a late
failure is rare in normal use.

**After commit**: the transaction is finalized; calling `commit()` again is a
no-op.

### rollback()

```cpp
void rollback() noexcept;
```

Discards all buffered operations without applying them. Safe to call after a
failed `commit()` — the failed commit has already restored the pre-batch state.
Guaranteed not to throw.

---

## Usage Patterns

### Basic Atomic Batch

```cpp
auto txn = db->begin_transaction();

txn.collection("documents").add(elips::Vector{{1.0F, 0.0F, 0.0F}},
                                {{"title", std::string{"gamma"}}});
txn.collection("documents").add(elips::Vector{{0.0F, 1.0F, 0.0F}},
                                {{"title", std::string{"delta"}}});

txn.commit();  // both inserted atomically
```

### Cross-Collection Atomicity

```cpp
auto txn = db->begin_transaction();

txn.collection("source").remove(source_id);
txn.collection("target").add(source_vector, source_payload);

txn.commit();  // remove + insert as one atomic unit
```

### RAII Guard (Destructor Auto-Rollback)

```cpp
{
    auto txn = db->begin_transaction();
    txn.collection("docs").add(vec1, payload1);
    // If we exit scope without commit(), the destructor auto-rolls back
}
// nothing was applied
```
