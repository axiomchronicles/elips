# API Reference: Filter

**Maturity:** stable (v1.0+).

Namespace `elips`. A metadata predicate over a record's `Payload`. Declaration at `include/elips/metadata/Filter.hpp:23`.

A default-constructed `Filter` matches everything (`matches_all()` returns `true`). Filters are immutable value types - composable via combinators that return new `Filter` objects.

## Internal Structure

A `Filter` holds a tree of `Node` objects (shared via `shared_ptr<const Node>`). The root being `nullptr` means "match all".

### Node Kind Enum

```cpp
enum class Kind { cmp, in, contains, conj, disj, neg, none };
```

| Kind | Purpose | Node Fields Used |
|------|---------|-----------------|
| `cmp` | Field comparison (eq, ne, lt, le, gt, ge) | `field`, `cmp`, `value` |
| `in` | Field value in set | `field`, `set` |
| `contains` | String field contains substring | `field`, `value` (string) |
| `conj` | Boolean AND of two sub-trees | `a`, `b` |
| `disj` | Boolean OR of two sub-trees | `a`, `b` |
| `neg` | Boolean NOT of one sub-tree | `a` |
| `none` | Always false (for NOT(match-all)) | none |

### Comparator Enum

```cpp
enum class Comparator { eq, ne, lt, le, gt, ge };
```

Used by `cmp` nodes. Maps to standard comparison operators on `MetaValue`.

### Cross-Type Numeric Comparison

`compare_values(a, b)` is an internal function that compares two `MetaValue` instances:

- **Numeric vs numeric**: `int64_t` and `double` are promoted to `double` and compared numerically. Example: `compare_values(int64_t{5}, double{5.0})` returns `0` (equal).
- **bool vs bool**: Compared as integers (false=0, true=1).
- **string vs string**: Lexicographic comparison via `std::string::compare`.
- **Mixed non-numeric types**: Returns `std::nullopt` (not comparable). Evaluates to "not a match" for the predicate.

---

## Fluent Builder API

The fluent builder chains predicates by AND-ing them together. Set the target field with `field()`, then apply a comparator or set operation. Multiple chains are AND-ed.

```cpp
Filter f;
f.field("year").ge(std::int64_t{2023}).field("tag").one_of({std::string{"a"}, std::string{"b"}});
// => (year >= 2023) AND (tag IN ["a", "b"])
```

### field()

```cpp
Filter& field(std::string name);
```

Sets the current target field for subsequent comparator / set / contains calls. Returns `*this` for chaining.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `std::string` | The payload key to filter on |

### equals()

```cpp
Filter& equals(MetaValue value);
```

Creates a `cmp` node with `Comparator::eq`. AND-ed onto the current tree.

```cpp
Filter().field("status").equals(std::string{"active"});
// => status = "active"
```

### not_equals()

```cpp
Filter& not_equals(MetaValue value);
```

Creates a `cmp` node with `Comparator::ne`.

```cpp
Filter().field("status").not_equals(std::string{"deleted"});
// => status != "deleted"
```

### lt()

```cpp
Filter& lt(MetaValue value);
```

Creates a `cmp` node with `Comparator::lt`.

```cpp
Filter().field("score").lt(std::int64_t{50});
// => score < 50
```

### le()

```cpp
Filter& le(MetaValue value);
```

Creates a `cmp` node with `Comparator::le`.

```cpp
Filter().field("score").le(std::int64_t{100});
// => score <= 100
```

### gt()

```cpp
Filter& gt(MetaValue value);
```

Creates a `cmp` node with `Comparator::gt`.

```cpp
Filter().field("score").gt(std::int64_t{0});
// => score > 0
```

### ge()

```cpp
Filter& ge(MetaValue value);
```

Creates a `cmp` node with `Comparator::ge`.

```cpp
Filter().field("year").ge(std::int64_t{2023});
// => year >= 2023
```

### one_of()

```cpp
Filter& one_of(std::vector<MetaValue> values);
```

Creates an `in` node - matches if the field value equals any value in the set.

```cpp
Filter().field("category").one_of({
    std::string{"news"}, std::string{"sports"}, std::string{"tech"}
});
// => category IN ["news", "sports", "tech"]
```

Cross-type numeric comparison is supported: `one_of({int64_t{5}, double{5.1}})` will match an `int64_t{5}` record field.

### contains()

```cpp
Filter& contains(std::string substring);
```

Creates a `contains` node - matches if the field value is a string that contains the given substring.

```cpp
Filter().field("title").contains(std::string{"hello"});
// => title contains "hello"
```

The field must hold a `std::string` MetaValue. If the field is missing or non-string, `contains` evaluates to false.

**Evaluation**: Uses `std::get<std::string>(field_value).find(substring) != std::string::npos`.

---

## Leaf Factory Methods (Static)

These create standalone `Filter` objects with a single leaf node. Used by the EQL parser/executor rather than the fluent builder.

### compare()

```cpp
static Filter compare(std::string field, Comparator op, MetaValue value);
```

Creates a filter with a single `cmp` node.

```cpp
auto f = Filter::compare("year", Comparator::ge, std::int64_t{2023});
// => year >= 2023
```

### in_set()

```cpp
static Filter in_set(std::string field, std::vector<MetaValue> values);
```

Creates a filter with a single `in` node.

```cpp
auto f = Filter::in_set("tag", {std::string{"a"}, std::string{"b"}});
// => tag IN ["a", "b"]
```

### has_substring()

```cpp
static Filter has_substring(std::string field, std::string substring);
```

Creates a filter with a single `contains` node.

```cpp
auto f = Filter::has_substring("title", std::string{"search term"});
// => title contains "search term"
```

---

## Combinator Methods

### and_()

```cpp
Filter and_(const Filter& other) const;
```

Returns a new Filter representing `this AND other`.

```cpp
auto f1 = Filter::compare("a", Comparator::eq, std::int64_t{1});
auto f2 = Filter::compare("b", Comparator::gt, std::int64_t{0});
auto combined = f1.and_(f2);
// => (a = 1) AND (b > 0)
```

**Edge cases**:
- `match_all.and_(x)` returns `x`.
- `x.and_(match_all)` returns `x`.

### or_()

```cpp
Filter or_(const Filter& other) const;
```

Returns a new Filter representing `this OR other`.

```cpp
auto f1 = Filter::compare("cat", Comparator::eq, std::string{"news"});
auto f2 = Filter::compare("cat", Comparator::eq, std::string{"sports"});
auto combined = f1.or_(f2);
// => (cat = "news") OR (cat = "sports")
```

**Edge cases**:
- If either operand is match-all, `or_()` returns match-all (disjunction with universal truth is universal truth).

### not_()

```cpp
static Filter not_(const Filter& inner);
```

Returns a new Filter representing `NOT inner`.

```cpp
auto active = Filter::compare("status", Comparator::eq, std::string{"active"});
auto not_active = Filter::not_(active);
// => NOT (status = "active")
```

**Edge case**: `Filter::not_(Filter{})` returns a `none`-kind node (never matches), since NOT(match-all) matches nothing.

---

## Evaluation Methods

### matches()

```cpp
bool matches(const Payload& payload) const;
```

Evaluates the filter tree against a record's payload.

**Returns**: `true` if the record satisfies the predicate, `false` otherwise.

**Behavior**: Recursively evaluates the node tree via `eval_node()`:

| Kind | Logic |
|------|-------|
| `cmp` | Look up `node.field` in payload. If missing, return false. Compute `compare_values(payload_value, node.value)`. If comparable and satisfies `node.cmp`, return true. |
| `in` | Look up `node.field` in payload. If missing, return false. For each value in `node.set`, check if `compare_values(payload_value, candidate) == 0`. Return true if any match. |
| `contains` | Look up `node.field` in payload. If missing or not a string, return false. Return `payload_string.find(node_value_string) != npos`. |
| `conj` | `matches(node.a) && matches(node.b)` |
| `disj` | `matches(node.a) \|\| matches(node.b)` |
| `neg` | `!matches(node.a)` |
| `none` | `false` |

### matches_all()

```cpp
bool matches_all() const noexcept;
```

Returns `true` if the filter has no predicate tree (`root_ == nullptr`), meaning it matches every record. Used as an optimization in `Collection::search()` to skip post-filtering when no filter is applied.

---

## Full Examples

**Complex filter with AND**:

```cpp
auto f = Filter()
    .field("year").ge(std::int64_t{2020})
    .field("category").one_of({std::string{"news"}, std::string{"tech"}});
// => (year >= 2020) AND (category IN ["news", "tech"])
```

**OR filter using combinators**:

```cpp
auto cat_news = Filter().field("category").equals(std::string{"news"});
auto cat_sports = Filter().field("category").equals(std::string{"sports"});
auto f = cat_news.or_(cat_sports);
// => (category = "news") OR (category = "sports")
```

**NOT filter**:

```cpp
auto deleted = Filter().field("status").equals(std::string{"deleted"});
auto not_deleted = Filter::not_(deleted);
// => NOT (status = "deleted")
```

**Nested boolean logic**:

```cpp
auto is_car = Filter().field("type").equals(std::string{"car"});
auto is_red = Filter().field("color").equals(std::string{"red"});
auto is_blue = Filter().field("color").equals(std::string{"blue"});

auto f = is_car.and_(is_red.or_(is_blue));
// => (type = "car") AND ((color = "red") OR (color = "blue"))
```

**Substring match**:

```cpp
auto f = Filter().field("description").contains(std::string{"important"});
// => description contains "important"
```

**Using with Collection methods**:

```cpp
// Filtered vector search
auto results = docs.search(query, 10, filter);

// Filtered metadata scan
auto records = docs.scan(filter, 0, 50);

// Combined: filter + threshold search
auto results = docs.search(query, 100, filter, 0.5f);
```

## Thread Safety

`Filter` objects are immutable value types. All methods return new objects or const references. It is safe to share `Filter` instances across threads for read-only access.