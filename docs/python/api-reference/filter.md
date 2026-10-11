# Filter API Reference

The `Filter` class is a fluent builder for metadata predicates. Pass one as the
`where=` argument to `Collection.query()`, `Collection.get()`, and
`Collection.delete()` (and to the native `Vault.seek()` / `Vault.scan()`).

An empty `Filter()` matches all records.

## Filter building model

Filters combine **field-level predicates** with **boolean combinators**:

1. **Field predicates:** call `.field(name)` to select a field, then chain a
   comparison method. Multiple field predicates on the same filter are AND-ed.
2. **Boolean combinators:** `.and_(other)`, `.or_(other)`, and the static
   `Filter.not_(inner)` build compound expressions.

```python
import elips

# field + comparison
elips.Filter().field("category").equals("tech")
# chained predicates are AND-ed
elips.Filter().field("category").equals("tech").field("score").gte(0.8)
```

## Field-level predicates

Each is called on the result of `.field(name)`:

| Method | Meaning |
|---|---|
| `equals(value)` | Field equals value. |
| `not_equals(value)` | Field does not equal value. |
| `lt(value)` | Less than. |
| `le(value)` | Less than or equal. |
| `gt(value)` | Greater than. |
| `gte(value)` | Greater than or equal. |
| `one_of(values)` | Field value is in the given set (IN). |
| `contains(substring)` | String field contains the substring (case-sensitive). |

```python
import elips

elips.Filter().field("year").equals(2024)
elips.Filter().field("status").not_equals("deleted")
elips.Filter().field("score").gte(0.8).field("score").lt(1.0)
elips.Filter().field("country").one_of(["US", "GB", "DE"])
elips.Filter().field("title").contains("intro")
```

> **Spelling:** the comparison methods are `lt`, `le`, `gt`, and `gte`. There is
> **no** `.ge()` or `.lte()` method (`ge` / `le` exist only as `Comparator`
> enum members). Use `.gte(...)` for greater-or-equal and `.le(...)` for
> less-or-equal.

## Boolean combinators

```python
import elips

# AND of two separate filters (chaining fields already ANDs)
tech_high = elips.Filter().field("cat").equals("tech").and_(
    elips.Filter().field("score").gte(90)
)

# OR
pro_or_recent = elips.Filter().field("tier").equals("pro").or_(
    elips.Filter().field("year").gte(2023)
)

# NOT (static)
active = elips.Filter.not_(elips.Filter().field("draft").equals(True))
```

`.and_()` / `.or_()` combine separate `Filter` instances; chaining field
predicates on one filter is already an AND, so these two are equivalent:

```python
import elips

f1 = elips.Filter().field("cat").equals("tech").field("score").gte(90)
f2 = elips.Filter().field("cat").equals("tech").and_(
    elips.Filter().field("score").gte(90)
)
```

## Composite examples

```python
import elips

# (cat == "tech" AND score >= 80) OR (cat == "finance" AND year >= 2022)
tech = elips.Filter().field("cat").equals("tech").field("score").gte(80)
finance = elips.Filter().field("cat").equals("finance").field("year").gte(2022)
combined = tech.or_(finance)

# NOT (status in {deleted, archived})
live = elips.Filter.not_(elips.Filter().field("status").one_of(["deleted", "archived"]))
```

## Using a filter

```python
import elips

client = elips.Client(dimension=3)
docs = client.get_or_create_collection("docs")
docs.add(
    ids=["a", "b"],
    embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    metadatas=[{"category": "tech", "score": 0.9}, {"category": "bio", "score": 0.3}],
)

f = elips.Filter().field("category").equals("tech").field("score").gte(0.8)

# vector search with a filter
hits = docs.query(query_embeddings=[1.0, 0.0, 0.0], n_results=10, where=f)
# filter-only scan
rows = docs.get(where=f, limit=100)
# range search: drop hits beyond max_distance
near = docs.query(query_embeddings=[1.0, 0.0, 0.0], n_results=1000, where=f, max_distance=0.25)
print(len(hits), len(rows), len(near))
client.close()
```

## `MetaValue` type

Comparison methods accept only four metadata types:

| Python type | C++ type | Example |
|---|---|---|
| `bool` | `bool` | `True`, `False` |
| `int` | `std::int64_t` | `42`, `-1` |
| `float` | `double` | `0.95` |
| `str` | `std::string` | `"tech"` |

Passing a `list`, `dict`, `None`, or any other type raises `TypeError:
metadata values must be int, float, bool, or str`. `bool` is checked before
`int` (because `bool` is a subclass of `int` in Python), so `True` stays a bool
rather than collapsing to `1`.

## Filter and index interaction

- Filter predicates are evaluated **after** vector distance computation: the
  search retrieves candidates by distance, then applies the filter. A selective
  filter may return fewer than `n_results` hits if not enough candidates match.
- With `max_distance`, the filter applies in addition to the distance bound.
- Within a native transaction, filters see only **committed** data, not
  in-flight writes from the same transaction.
