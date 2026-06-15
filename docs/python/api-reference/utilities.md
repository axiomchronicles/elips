# Utilities API Reference

Module-level utility functions for distance computation, metric introspection, and metric string/enum conversion.

## `elips.distance()`

```python
def distance(
    metric: Union[str, Metric],
    a: Vector,
    b: Vector,
) -> float
```

Compute the ordering-normalized distance between two vectors. The result is the **ELIPS distance** — a scalar where smaller values indicate higher similarity across all metrics.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `metric` | `str` or `Metric` | The similarity metric — either a string (`"cosine"`, `"euclidean"`, `"dot_product"`) or a `Metric` enum value (`Metric.cosine`, etc.) |
| `a` | `Vector` | First vector (any iterable of floats) |
| `b` | `Vector` | Second vector (any iterable of floats) |

**Returns:** A float distance value. Smaller = more similar for all metrics.

**Raises:**
- `ValueError` — vectors have different lengths
- `ValueError` — unrecognized metric string

**Metric-specific formulas:**

| Metric | Distance Formula | Range |
|---|---|---|
| `cosine` | `1.0 - cos_sim(a, b)` = `1.0 - (a·b) / (|a|·|b|)` | `[0.0, 2.0]` |
| `euclidean` | `sqrt(Σ(aᵢ - bᵢ)²)` | `[0.0, ∞)` |
| `dot_product` | `-(a·b)` | `(-∞, ∞)` (negative for ranking) |

For cosine: `0.0` = identical, `1.0` = orthogonal, `2.0` = opposite direction.

**Examples:**

```python
# String argument
d = elips.distance("cosine", [1.0, 0.0], [0.0, 1.0])
print(d)  # 1.0 (orthogonal vectors)

# Enum argument
d = elips.distance(elips.Metric.cosine, [1.0, 0.0], [1.0, 0.0])
print(d)  # 0.0 (identical vectors)

# Euclidean
d = elips.distance("euclidean", [0.0, 0.0], [3.0, 4.0])
print(d)  # 5.0 (3-4-5 triangle)

# Dot product
d = elips.distance("dot_product", [1.0, 2.0], [1.0, 1.0])
print(d)  # -3.0 (negative of inner product)
```

The `distance()` function computes the same value as `Vault.seek()` would return for a query vector. You can use it to pre-compute distances, verify results, or implement custom ranking logic.

## `elips.requires_normalization()`

```python
def requires_normalization(
    metric: Union[str, Metric],
) -> bool
```

Return `True` if vectors should be L2-normalized before use with this metric.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `metric` | `str` or `Metric` | The similarity metric |

**Returns:** `True` for `cosine`, `False` for `euclidean` and `dot_product`.

**Raises:**
- `ValueError` — unrecognized metric string

**Why this matters:**

- **Cosine**: Requires L2-normalized vectors for correct results. If you compute your own embeddings, normalize them before inserting.
- **Euclidean**: No normalization needed. Raw vectors work as-is.
- **Dot product**: No normalization needed. The raw inner product is used.

**Examples:**

```python
elips.requires_normalization("cosine")              # True
elips.requires_normalization("euclidean")           # False
elips.requires_normalization("dot_product")         # False
elips.requires_normalization(elips.Metric.cosine)   # True
elips.requires_normalization(elips.Metric.euclidean)  # False
```

## `elips.metric_to_string()`

```python
def metric_to_string(metric: Metric) -> str
```

Convert a `Metric` enum value to its canonical string name.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `metric` | `Metric` | The enum value |

**Returns:** One of `"cosine"`, `"euclidean"`, `"dot_product"`.

**Examples:**

```python
elips.metric_to_string(elips.Metric.cosine)      # "cosine"
elips.metric_to_string(elips.Metric.euclidean)   # "euclidean"
elips.metric_to_string(elips.Metric.dot_product) # "dot_product"
```

## `elips.metric_from_string()`

```python
def metric_from_string(name: str) -> Metric
```

Parse a string into a `Metric` enum value.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `name` | `str` | One of `"cosine"`, `"euclidean"`, `"dot_product"` |

**Returns:** The corresponding `Metric` enum value.

**Raises:**
- `ValueError` — the string is not a recognized metric name

**Examples:**

```python
m = elips.metric_from_string("cosine")
print(m)  # Metric.cosine
print(int(m) == int(elips.Metric.cosine))  # True

elips.metric_from_string("euclidean")    # Metric.euclidean
elips.metric_from_string("dot_product")  # Metric.dot_product

elips.metric_from_string("unknown")      # ValueError
```

## String/Enum Dual Overloads

The `distance()` and `requires_normalization()` functions accept both string and enum arguments. The implementation dispatches based on argument type:

```python
# Equivalent calls
elips.distance("cosine", a, b)
elips.distance(elips.Metric.cosine, a, b)

elips.requires_normalization("cosine")
elips.requires_normalization(elips.Metric.cosine)
```

The string overload allows simple usage without importing enums. The enum overload is preferred when you already have a `Metric` value (e.g., from `db.config.metric_enum`).

## Complete Example

```python
import elips

# Compute distances
v1 = [1.0, 0.0, 0.0]
v2 = [0.0, 1.0, 0.0]
v3 = [0.0, 0.0, 1.0]

print(elips.distance("cosine", v1, v2))       # 1.0
print(elips.distance("cosine", v1, v1))       # 0.0
print(elips.distance("euclidean", v1, v2))    # 1.414...

# Check normalization requirements
for metric in [elips.Metric.cosine, elips.Metric.euclidean, elips.Metric.dot_product]:
    name = elips.metric_to_string(metric)
    needs_norm = elips.requires_normalization(metric)
    print(f"{name:12} requires_normalization = {needs_norm}")

# Roundtrip conversion
for name in ["cosine", "euclidean", "dot_product"]:
    m = elips.metric_from_string(name)
    back = elips.metric_to_string(m)
    assert name == back
    print(f"'{name}' → {m} → '{back}'")
```

Output:

```
cosine       requires_normalization = True
euclidean    requires_normalization = False
dot_product  requires_normalization = False
'cosine' → Metric.cosine → 'cosine'
'euclidean' → Metric.euclidean → 'euclidean'
'dot_product' → Metric.dot_product → 'dot_product'
```

## Type Signatures

The `Vector` and `Metric` types are available for annotation:

```python
from typing import Union
from elips import Metric
from elips.types import Vector

def my_distance(metric: Union[str, Metric], a: Vector, b: Vector) -> float:
    return elips.distance(metric, a, b)
```

Type checkers (MyPy, Pyright) can validate these annotations using the `_core.pyi` stub.