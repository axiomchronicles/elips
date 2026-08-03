# Python SDK Reference Updates — v1.1.0

This document summarizes all Python SDK additions, type stub expansions (`_core.pyi`), and module surface additions introduced in ELIPS v1.1.0.

---

## Complete Python API Reference Overview

The v1.1.0 Python bindings export both low-level native extension bindings (`elips`) and high-level Pythonic abstractions (`elips._modern`).

---

## 1. Filter & Metadata Acceleration (`elips.Filter`)

A fluent, immutable predicate builder for metadata filtering.

### Methods
- `Filter()` — Empty filter (matches all records).
- `.field(name)` — Target a payload key.
- `.equals(val)` / `.not_equals(val)` — Equality / inequality comparison.
- `.lt(val)` / `.le(val)` / `.gt(val)` / `.ge(val)` — Range comparison.
- `.one_of(list)` — Set membership check.
- `.contains(substr)` — Metadata string substring match.
- `.and_(other)` / `.or_(other)` — Boolean combinators.
- `Filter.not_(inner)` — Static boolean negation.
- `Filter.compare(field, op, val)` — Static factory using `Comparator` enum (`eq`, `ne`, `lt`, `le`, `gt`, `ge`).
- `Filter.in_set(field, values)` / `Filter.has_substring(field, substr)` — Static leaf factories.
- `.matches(dict)` — Test filter against a Python dictionary without database context.
- `.matches_all()` — Returns `True` if filter is a no-op.
- `.exact_constraints()` — Returns equality tuples `[("field", [values])]` if the filter is accelerated by `MetadataIndex`.

```python
import elips

# Fluent Filter Builder
f = (
    elips.Filter()
    .field("tenant")
    .equals("acme")
    .field("score")
    .gte(0.8)
    .field("region")
    .one_of(["us-east", "eu-west"])
)

# Test filter standalone
print(
    "Matches sample:", f.matches({"tenant": "acme", "score": 0.9, "region": "us-east"})
)  # True

# Query Vault with Filter
db = elips.open("/var/data/vecs")
results = db.vault("docs").seek([0.1, 0.2, 0.3], top=10, where=f)
```

---

## 2. Maintenance & Lifecycle API (`Vault` & `Database`)

Exposes tombstone vacuuming, read-only modes, and instance sealing.

### Vault Methods & Properties
- `vault.vacuum()` — Reclaims index space held by deleted records in-place.
- `vault.pending_removals` — Integer count of tombstoned records awaiting compaction.
- `vault.rebuild_index()` — Force full rebuild of backing index.
- `vault.records()` — Snapshot copy of all stored records (`list[StoredRecord]`).
- `vault.read_only` / `vault.set_read_only(bool)` — Inspect or toggle read-only mode.
- `vault.sealed` — `True` if parent database instance has been closed.

### Database Methods & Properties
- `db.vacuum()` — Vacuums all vaults across the database.
- `db.compact()` — Rebuilds indexes and flushes snapshot checkpoint.
- `db.checkpoint()` — Flushes WAL mutations to durable disk storage.
- `db.close()` — Checkpoints, releases advisory file locks, and seals child vaults.
- `db.closed` — `True` after `close()` or `abandon()` has run.
- `db.path` / `db.persistent` — Inspect database path or in-memory state.

```python
import elips

db = elips.open("/var/data/vecs")
vault = db.vault("docs")

# Delete records
vault.erase("doc-123")

# Inspect & Reclaim
print("Outstanding tombstones:", vault.pending_removals)
vault.vacuum()

# Toggle Read-Only
vault.set_read_only(True)
assert vault.read_only is True
```

---

## 3. Durability & Transaction Management (`Config.durability`)

### Durability Levels
- `"paranoid"` — Full device sync (`fdatasync` / `F_FULLFSYNC`) on every record append.
- `"standard"` — Default level; full device sync on every record append.
- `"relaxed"` — Buffered in OS page cache; synced on `checkpoint()` or `close()`.
- `"ephemeral"` — In-memory only; no WAL written.

### Context Manager Transaction Atomicity
```python
import elips

config = elips.Config().dimension(128).durability("standard")
db = elips.open_with_config("/var/data/durable_db", config)

# Context Manager automatically commits on clean exit, rolls back on exception
with db.begin_transaction() as txn:
    v = txn.vault("items")
    v.place([0.1] * 128, {"sku": "A100"})
    v.place([0.2] * 128, {"sku": "B200"})
```

---

## 4. Crash Recovery Forensics & WAL Replay (`elips.replay_wal`)

Exposes offline WAL file parsing and EQL AST tokenization for crash investigation.

### Functions
- `elips.replay_wal(path)` — Reads WAL file records without opening database. Returns `list[WalEntry]`.
- `elips.parse_eql(query)` — Parses EQL string into AST `Statement` object.
- `elips.validate_eql(query)` — Validates EQL syntax (raises `ParseError` if invalid).
- `elips.tokenize_eql(query)` — Returns `list[Token]` for syntax highlighting/analysis.

```python
import elips

# Inspect WAL offline after crash
entries = elips.replay_wal("/var/data/durable_db/wal.log")
print(f"Replayed {len(entries)} entries.")

# EQL AST Parsing
stmt = elips.parse_eql("seek in docs nearest $q top 10 where status = 'active' yield")
print("Target vault:", stmt.vault)
print("Top K:", stmt.top)
```
