# Filter API Reference

The `Filter` class provides a fluent builder API for constructing metadata predicates used in `seek()` and `scan()`.

## Filter

```python
class Filter
```

Metadata filter for search and scan operations. Uses a fluent builder pattern for chaining predicates.

**Constructor:**

```python
def __init__(self) -> None
```

Creates an empty filter that matches all records.

### Filter Building Model

Filters combine **field-level predicates** with **boolean combinators**:

1. **Field predicates**: call `.field(name)` to select a field, then chain a comparison method. Multiple field-level predicates are AND-ed together.
2. **Boolean combinators**: `.and_(other)`, `.or_(other)`, and `Filter.not_(inner)` build compound expressions.

### Field-Level Predicates

#### `Filter.field()`

```python
def field(self, name: str) -> Filter
```

Select a metadata field for comparison. Returns `self` for chaining.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `name` | `str` | The metadata field name |

**Returns:** `self` (the same Filter object) for chaining.

#### Comparison Methods

Each of the following is called on the result of `.field(name)`:

##### `Filter.equals()`

```python
def equals(self, value: MetaValue) -> Filter
```

Field equals the given value.

```python
elips.Filter().field("category").equals("tech")
elips.Filter().field("year").equals(2024)
elips.Filter().field("active").equals(True)
```

##### `Filter.not_equals()`

```python
def not_equals(self, value: MetaValue) -> Filter
```

Field does **not** equal the given value.

```python
elips.Filter().field("status").not_equals("deleted")
```

##### `Filter.lt()`

```python
def lt(self, value: MetaValue) -> Filter
```

Field is less than the given value.

```python
elips.Filter().field("score").lt(0.5)
elips.Filter().field("year").lt(2020)
```

##### `Filter.le()`

```python
def le(self, value: MetaValue) -> Filter
```

Field is less than or equal to the given value.

```python
elips.Filter().field("score").le(0.5)
```

##### `Filter.gt()`

```python
def gt(self, value: MetaValue) -> Filter
```

Field is greater than the given value.

```python
elips.Filter().field("score").gt(0.8)
```

##### `Filter.gte()`

```python
def gte(self, value: MetaValue) -> Filter
```

Field is greater than or equal to the given value.

```python
elips.Filter().field("year").gte(2023)
```

##### `Filter.one_of()`

```python
def one_of(self, values: Iterable[MetaValue]) -> Filter
```

Field value is contained in the given set of values (IN clause).

```python
elips.Filter().field("country").one_of(["US", "GB", "DE"])
elips.Filter().field("priority").one_of([1, 2, 3])
```

##### `Filter.contains()`

```python
def contains(self, substring: str) -> Filter
```

String field contains the given substring (case-sensitive).

```python
elips.Filter().field("title").contains("intro")
elips.Filter().field("description").contains("machine learning")
```

### Boolean Combinators

#### `Filter.and_()`

```python
def and_(self, other: Filter) -> Filter
```

Logical AND. Both this filter and `other` must match.

```python
f = elips.Filter().field("cat").equals("tech").and_(
    elips.Filter().field("score").gte(90)
)
# Both conditions must be true
```

Note: Chaining field predicates on the same filter is already an AND. Use `.and_()` only when combining separate `Filter` instances:

```python
# These are equivalent:
f1 = elips.Filter().field("cat").equals("tech").field("score").gte(90)
f2 = (elips.Filter().field("cat").equals("tech")
      .and_(elips.Filter().field("score").gte(90)))
```

#### `Filter.or_()`

```python
def or_(self, other: Filter) -> Filter
```

Logical OR. Either this filter or `other` (or both) must match.

```python
f = elips.Filter().field("tier").equals("pro").or_(
    elips.Filter().field("year").gte(2023)
)
# Matches if tier == "pro" OR year >= 2023
```

#### `Filter.not_()` (static)

```python
@staticmethod
def not_(inner: Filter) -> Filter
```

Logical NOT. Negates the inner filter.

```python
f = elips.Filter.not_(
    elips.Filter().field("draft").equals(True)
)
# Matches records where draft is NOT True
```

## Composite Examples

### Multi-field AND

```python
f = (elips.Filter()
     .field("category").equals("tech")
     .field("score").gte(80)
     .field("country").one_of(["US", "GB"])
     .field("title").contains("intro"))
```

### Complex Boolean Expression

```python
# (category == "tech" AND score >= 80) OR (category == "finance" AND year >= 2022)
tech_high = elips.Filter().field("category").equals("tech").field("score").gte(80)
finance_recent = elips.Filter().field("category").equals("finance").field("year").gte(2022)

combined = tech_high.or_(finance_recent)
```

### Negation

```python
# NOT (status == "deleted" OR status == "archived")
active = elips.Filter.not_(
    elips.Filter().field("status").one_of(["deleted", "archived"])
)
```

### With seek()

```python
f = (elips.Filter()
     .field("category").equals("tech")
     .field("score").gte(0.8))

results = docs.seek([1.0, 0.0, 0.0], top=10, where=f)
```

### With scan()

```python
f = elips.Filter().field("year").gte(2023).field("year").lt(2025)
rows = docs.scan(where=f, limit=100)
```

### With threshold (range search)

```python
f = elips.Filter().field("category").equals("tech")
results = docs.seek([1.0, 0.0, 0.0], top=1000, where=f, threshold=0.25)
```

## `Filter.__repr__()`

```python
def __repr__(self) -> str
```

Returns `"<Filter match-all>"` for an empty filter, or `"<Filter>"` for a populated one.

## `MetaValue` Type

```python
MetaValue = Union[bool, int, float, str]
```

The type accepted by comparison methods. Only these four types are valid:

| Python Type | C++ Type | Example |
|---|---|---|
| `bool` | `bool` | `True`, `False` |
| `int` | `std::int64_t` | `42`, `-1`, `0` |
| `float` | `double` | `0.95`, `3.14` |
| `str` | `std::string` | `"tech"`, `"hello world"` |

Passing a `list`, `dict`, `None`, or any other type raises `TypeError: metadata values must be int, float, bool, or str`.

**Important:** `bool` is checked before `int` in the C++ conversion (`to_meta()`) because `bool` is a subclass of `int` in Python.

## Filter and Index Interaction

- Filter predicates are evaluated **after** vector distance computation, not before. The search first retrieves candidate vectors by distance, then applies the filter.
- For large datasets with selective filters, a filtered search may return fewer than `top` results if not enough filtered candidates exist.
- For range searches with `threshold`, the filter is applied in addition to the distance constraint.

## Filter and Transactions

Within a transaction, filters only see **committed** data, not in-flight writes from the same transaction. This is consistent with the transaction's isolation model.