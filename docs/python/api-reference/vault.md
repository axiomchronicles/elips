# Vault API Reference

A `Vault` is a named partition of records within a database. Each vault owns its own index and is obtained via `db.vault("name")`.

## Vault

```python
class Vault
```

A named partition of records within a database. Owns its index and the authoritative record store.

### `Vault.name`

```python
@property
def name(self) -> str
```

The vault's name.

**Example:**

```python
docs = db.vault("documents")
print(docs.name)  # "documents"
```

### `Vault.place()`

```python
def place(
    self,
    vector: Vector,
    data: PayloadLike = {},
    id: Optional[str] = None,
) -> str
```

Ingest a single record. The vector is indexed immediately for future search.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `vector` | `Vector` | required | The embedding vector (list or tuple of floats) |
| `data` | `PayloadLike` | `{}` | Optional metadata payload — dict of `str` → `int` / `float` / `bool` / `str` |
| `id` | `str` or `None` | `None` | Optional custom UUIDv7 record ID as hex string (36 chars). If `None`, a random UUIDv7 is generated |

**Returns:** The record's UUIDv7 ID as a 36-character hex string.

**Raises:**
- `DimensionMismatch` — vector length does not match the vault's configured dimension
- `InvalidVector` — vector contains `NaN`, `Inf`, or is otherwise unusable

**Examples:**

```python
rid = docs.place([0.1, 0.2, 0.3])
# → "018f3c7a-0000-7000-8000-000000000001"

rid = docs.place(
    [0.1, 0.2, 0.3],
    data={"title": "Getting Started", "year": 2024, "pinned": True},
)

rid = docs.place(
    [0.4, 0.5, 0.6],
    id="00000000-0000-7000-8000-000000000001",
)
```

Metadata values must be one of: `bool`, `int`, `float`, or `str`. Other types raise `TypeError`:

```python
# OK
docs.place([1.0], {"count": 42})         # int
docs.place([1.0], {"ratio": 0.95})       # float
docs.place([1.0], {"active": True})      # bool
docs.place([1.0], {"name": "hello"})     # str

# TypeError
docs.place([1.0], {"tags": ["a", "b"]})  # list — not supported
docs.place([1.0], {"obj": {"key": 1}})   # dict — not supported
```

### `Vault.place_many()`

```python
def place_many(self, records: Iterable[Mapping[str, Any]]) -> None
```

Batch-ingest multiple records. Each record is a dict with the following keys:

| Key | Type | Required | Description |
|---|---|---|---|
| `vector` | `list[float]` | Yes | The embedding vector |
| `data` | `dict` | No | Metadata payload |
| `id` | `str` | No | Custom UUIDv7 record ID |

**Raises:**
- `DimensionMismatch` — any vector has wrong dimension
- `InvalidVector` — any vector is invalid
- `KeyError` — any record is missing the required `"vector"` key

**Example:**

```python
docs.place_many([
    {"vector": [1.0, 0.0, 0.0], "data": {"n": 1}},
    {"vector": [0.0, 1.0, 0.0], "data": {"n": 2}},
    {"vector": [0.0, 0.0, 1.0], "id": "00000000-0000-7000-8000-000000000002", "data": {"n": 3}},
])
```

### `Vault.seek()`

```python
def seek(
    self,
    vector: Vector,
    top: int = 10,
    where: Filter = Filter(),
    threshold: Optional[float] = None,
) -> list[Result]
```

Top-k nearest neighbors sorted ascending by distance. Smaller distance means higher similarity.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `vector` | `Vector` | required | The query embedding |
| `top` | `int` | `10` | Maximum number of results to return |
| `where` | `Filter` | `Filter()` | Optional metadata filter applied before distance ranking |
| `threshold` | `float` or `None` | `None` | Optional max distance for range search. Records with distance > threshold are excluded |

**Returns:** List of `Result` objects sorted by distance (closest first).

**Raises:**
- `DimensionMismatch` — query vector dimension does not match the vault
- `InvalidVector` — query vector contains `NaN` or `Inf`

**Examples:**

```python
# Basic search
hits = docs.seek([1.0, 0.0, 0.0], top=5)
for r in hits:
    print(r.id, r.distance, r.data)

# Filtered search
f = elips.Filter().field("category").equals("tech")
hits = docs.seek([1.0, 0.0, 0.0], top=10, where=f)

# Range search (threshold)
hits = docs.seek([1.0, 0.0, 0.0], top=100, threshold=0.25)
```

**Result object:**

Each `Result` has three accessors:

| Attribute | Type | Description |
|---|---|---|
| `id` | `str` | Record UUIDv7 hex string (36 chars) |
| `distance` | `float` | Distance from query vector. Smaller = more similar. For `cosine`: 0.0 = identical, 1.0 = orthogonal, 2.0 = opposite |
| `data` | `dict[str, MetaValue]` | Metadata payload attached to the record |

### `Vault.fetch()`

```python
def fetch(self, id: str) -> Optional[dict[str, Any]]
```

Fetch a record's full data by ID, including its vector.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `id` | `str` | The record's UUIDv7 hex ID (36 chars) |

**Returns:** A dict with keys `"id"`, `"vector"`, and `"data"`, or `None` if the record does not exist.

**Example:**

```python
record = docs.fetch("018f3c7a-0000-7000-8000-000000000001")
if record:
    print(record["id"])       # "018f3c7a-0000-7000-8000-000000000001"
    print(record["vector"])   # (0.1, 0.2, 0.3)
    print(record["data"])     # {"title": "Getting Started", "year": 2024}

missing = docs.fetch("00000000-0000-7000-8000-999999999999")
print(missing)  # None
```

### `Vault.erase()`

```python
def erase(self, id: str) -> bool
```

Remove a record by ID.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `id` | `str` | The record's UUIDv7 hex ID |

**Returns:** `True` if the record was found and removed, `False` if it did not exist.

**Example:**

```python
removed = docs.erase("018f3c7a-0000-7000-8000-000000000001")
print(removed)  # True

removed = docs.erase("nonexistent-id-0000000000000001")
print(removed)  # False
```

### `Vault.scan()`

```python
def scan(
    self,
    where: Filter = Filter(),
    offset: int = 0,
    limit: int = -1,
) -> list[dict[str, Any]]
```

Iterate records matching a filter in insertion order. No vector search — purely metadata iteration.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `where` | `Filter` | `Filter()` | Optional metadata filter |
| `offset` | `int` | `0` | Number of matching records to skip |
| `limit` | `int` | `-1` | Maximum records to return. `-1` means all matching records |

**Returns:** List of dicts, each with `"id"` and `"data"` keys.

**Examples:**

```python
# All records
rows = docs.scan()
for row in rows:
    print(row["id"], row["data"])

# Filtered
rows = docs.scan(where=elips.Filter().field("year").gte(2023))

# Paginated
page_1 = docs.scan(offset=0, limit=50)
page_2 = docs.scan(offset=50, limit=50)
```

### `Vault.info()`

```python
def info(self) -> VaultInfo
```

Return summary statistics for the vault.

**Returns:** A `VaultInfo` with properties:

| Property | Type | Description |
|---|---|---|
| `count` | `int` | Number of records in the vault |
| `dimension` | `int` | Vector dimension of the vault |
| `metric` | `str` | Similarity metric: `"cosine"`, `"euclidean"`, or `"dot_product"` |

**Example:**

```python
info = docs.info()
print(info.count)      # 42
print(info.dimension)  # 1536
print(info.metric)     # "cosine"
```

### `Vault.count()`

```python
def count(self) -> int
```

Return the number of records in this vault. Equivalent to `vault.info().count`.

**Example:**

```python
n = docs.count()
print(n)  # 42
```

### `Vault.__repr__()`

```python
def __repr__(self) -> str
```

Returns a string like `<Vault name='documents' count=42 dimension=1536>`.

## VaultInfo

```python
class VaultInfo
```

Summary statistics for a vault. Obtained via `vault.info()`.

| Property | Type | Description |
|---|---|---|
| `count` | `int` | Number of records in the vault |
| `dimension` | `int` | Vector dimension |
| `metric` | `str` | Similarity metric string |

## Result

```python
class Result
```

A single result from a `seek()` or `query()` call.

| Property | Type | Description |
|---|---|---|
| `id` | `str` | Record UUIDv7 hex string |
| `distance` | `float` | Distance from query vector. Smaller = more similar |
| `data` | `dict[str, MetaValue]` | Metadata payload |

## TransactionVault

```python
class TransactionVault
```

Vault-scoped handle for operations within a transaction. Obtained via `txn.vault(name)`. Only supports `place()` and `erase()` — search operations must go through the regular `Vault`.

### `TransactionVault.place()`

```python
def place(
    self,
    vector: Vector,
    data: PayloadLike = {},
    id: Optional[str] = None,
) -> str
```

Buffered insert within a transaction. Not visible to reads until the transaction commits.

### `TransactionVault.erase()`

```python
def erase(self, id: str) -> None
```

Buffered delete within a transaction. Not applied until the transaction commits.

## Complete Example

```python
import elips

db = elips.open(":memory:", dimension=3, metric="cosine")
docs = db.vault("documents")

rid_a = docs.place([1.0, 0.0, 0.0], {"title": "alpha", "year": 2024})
rid_b = docs.place([0.0, 1.0, 0.0], {"title": "beta", "year": 2019})
rid_c = docs.place([0.9, 0.1, 0.0], {"title": "gamma", "year": 2023})

print(docs.count())     # 3
print(docs.name)        # "documents"

info = docs.info()
print(info.dimension)   # 3

hits = docs.seek([1.0, 0.0, 0.0], top=2)
print(hits[0].data["title"])  # "alpha"
print(hits[0].distance)       # 0.0 (identical)

record = docs.fetch(rid_a)
print(record["vector"])       # (1.0, 0.0, 0.0)
print(record["data"])         # {"title": "alpha", "year": 2024}

rows = docs.scan(where=elips.Filter().field("year").gte(2023))
print(len(rows))  # 2

docs.erase(rid_b)
print(docs.count())  # 2
```