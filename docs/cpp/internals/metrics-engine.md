# Metrics Engine

The metrics engine provides ordering-normalised distance computation over
float32 vectors with automatic SIMD dispatch on ARM (NEON) and a portable scalar
fallback.

---

## The `distance()` Function and Ordering-Normalisation Contract

**Header:** `include/elips/vector_engine/Metrics.hpp`
**Source:** `src/Metrics.cpp`

```cpp
[[nodiscard]] float distance(Metric metric, std::span<const float> a,
                             std::span<const float> b) noexcept;
```

The fundamental contract is that `distance()` returns a value where **smaller
means more similar** for every metric. This allows all index implementations
(`IndexFlat`, `IndexHNSW`, and the IVF family) and the query engine to sort results ascending by distance
and keep the smallest K, without needing to know which metric is in use.

This is achieved by transforming each underlying metric:

| Metric | Raw computation | Returned value |
|--------|----------------|----------------|
| `cosine` | `dot(a, b)` on L2-normalised inputs | `1.0 - dot(a, b)` |
| `euclidean` | `sqrt(Σ(aᵢ - bᵢ)²)` | `sqrt(Σ(aᵢ - bᵢ)²)` |
| `dot_product` | `dot(a, b)` | `-dot(a, b)` |

For cosine: since `dot(a,b)` ∈ [−1, 1] for unit vectors, `1 − dot(a,b)` ∈ [0, 2],
with 0 meaning identical direction.

For euclidean: the raw squared L2 is already monotonically increasing with
dissimilarity, so the raw value is ordering-correct.

For dot_product: larger values mean more similar, so negation converts it to
the ascending-dissimilarity ordering. The caller sorts ascending, so the
largest-dot vectors appear first.

```cpp
float distance(Metric metric, std::span<const float> a,
               std::span<const float> b) noexcept {
    switch (metric) {
        case Metric::cosine:      return 1.0F - dot(a, b);
        case Metric::euclidean:   return std::sqrt(squared_l2(a, b));
        case Metric::dot_product: return -dot(a, b);
    }
    return 0.0F;
}
```

---

## Three Metrics

### Cosine — `1 − dot(a, b)`

Assumes both vectors are L2-normalised (unit length). ELIPS enforces this
through `requires_normalization()`, which returns `true` for cosine only.
`Collection::prepare()` calls `vector.normalized()` for cosine metrics before
adding or querying, so the index engine always receives unit vectors.

The returned value 1 − dot measures the cosine distance, with a range of
[0, 2] for normalised inputs.

### Euclidean — `sqrt(Σ(aᵢ - bᵢ)²)`

Standard Euclidean (L2) distance. Computed as the square root of the
sum of squared differences. Because `√` is monotonic, sorting by `squared_l2`
would also produce the correct ordering, but `distance()` returns the true
Euclidean value.

### Dot Product — `−dot(a, b)`

Used for maximum-inner-product search (MIPS). Negated so that larger dot
products sort earlier in ascending order. Vectors are used as-is; no
normalisation is performed.

---

## NEON SIMD Kernels (ARM / Apple Silicon)

Enabled when `__ARM_NEON` is defined at compile time. Two kernels are
provided:

### `dot_neon` — NEON Vector Dot Product

```cpp
float dot_neon(const float* a, const float* b, std::size_t n) noexcept {
    float32x4_t acc = vdupq_n_f32(0.0F);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));  // FMA
    }
    float sum = vaddvq_f32(acc);  // horizontal reduction
    for (; i < n; ++i) sum += a[i] * b[i];  // scalar tail
    return sum;
}
```

- **`vld1q_f32`** loads 4 packed `float32_t` values from each input pointer.
- **`vmlaq_f32`** performs a fused multiply-add: `acc = acc + a × b` on all
  four lanes simultaneously. This is a single NEON instruction (`FMLA`).
- **`vaddvq_f32`** performs a horizontal reduction: adds the four lanes of
  the accumulator into a single scalar float.
- **Tail loop:** handles any remaining elements (n mod 4) with scalar operations.

Throughput: 4 multiply-adds per NEON instruction, plus one reduction per
chunk of 4. For typical embedding dimensions (128–1536), the speedup vs scalar
is roughly 2.5–3.5× (limited by memory bandwidth for larger dimensions).

### `sql2_neon` — NEON Squared L2

```cpp
float sql2_neon(const float* a, const float* b, std::size_t n) noexcept {
    float32x4_t acc = vdupq_n_f32(0.0F);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        acc = vmlaq_f32(acc, d, d);  // acc += d*d
    }
    float sum = vaddvq_f32(acc);
    for (; i < n; ++i) { const float d = a[i] - b[i]; sum += d * d; }
    return sum;
}
```

- **`vsubq_f32`** computes element-wise difference `a − b` in a single NEON
  instruction.
- **`vmlaq_f32`** with `d` as both multiplicands computes `d²` for all four
  lanes in a single FMA instruction.
- Same reduction and tail strategy as `dot_neon`.

---

## Scalar Fallback Kernels

When `__ARM_NEON` is not defined (x86 without AVX, or any other architecture),
portable scalar kernels are used:

```cpp
float dot_scalar(const float* a, const float* b, std::size_t n) noexcept {
    float sum = 0.0F;
    for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

float sql2_scalar(const float* a, const float* b, std::size_t n) noexcept {
    float sum = 0.0F;
    for (std::size_t i = 0; i < n; ++i) { const float d = a[i] - b[i]; sum += d * d; }
    return sum;
}
```

These are straightforward loops. The compiler typically auto-vectorises them on
x86 with SSE/AVX when optimisations are enabled (`-O2`/`-O3`/`-march=native`).

---

## Runtime Dispatch

```cpp
struct Dispatch {
    KernelFn dot;
    KernelFn sql2;
    Dispatch() {
#if defined(__ARM_NEON)
        dot  = &dot_neon;
        sql2 = &sql2_neon;
#else
        dot  = &dot_scalar;
        sql2 = &sql2_scalar;
#endif
    }
};

const Dispatch& kernels() {
    static const Dispatch instance;
    return instance;
}
```

The `Dispatch` struct is a compile-time + run-once dispatch table. On ARM,
NEON kernels are unconditionally selected (NEON is mandatory on ARMv8, so
runtime CPUID probing is unnecessary). On x86, the scalar fallback is used
and the compiler's auto-vectorisation takes over.

The `kernels()` function returns a reference to a **static local** `Dispatch`
instance, constructed exactly once (C++11 thread-safe static initialisation).
All subsequent calls return the same instance without any branch or lookup
overhead — the function pointers are dereferenced directly.

The internal helper functions `dot()` and `squared_l2()` call through the
dispatch table:

```cpp
float dot(std::span<const float> a, std::span<const float> b) noexcept {
    return kernels().dot(a.data(), b.data(), a.size());
}
float squared_l2(std::span<const float> a, std::span<const float> b) noexcept {
    return kernels().sql2(a.data(), b.data(), a.size());
}
```

This is a classic **strategy pattern with compile-time seam**: new kernels
(AVX2, AVX-512, SVE) can be added by extending the `Dispatch` constructor
with runtime CPUID checks without changing any callers.

---

## `requires_normalization()`

```cpp
bool requires_normalization(Metric metric) noexcept {
    return metric == Metric::cosine;
}
```

Only the cosine metric requires L2-normalisation. The caller (`Collection::prepare()`)
calls `vector.normalized()` on both ingested and query vectors when this returns
`true`.

`Vector::normalized()` computes the L2 norm and divides each component. Zero
vectors are returned unchanged to avoid division by zero:

```cpp
Vector Vector::normalized() const {
    const float mag = magnitude();
    if (mag == 0.0F) return *this;
    std::vector<float> out(values_.size());
    for (std::size_t i = 0; i < values_.size(); ++i)
        out[i] = values_[i] / mag;
    return Vector{std::move(out)};
}
```

---

## String Conversion

```cpp
std::string_view to_string(Metric metric) noexcept {
    switch (metric) {
        case Metric::cosine:      return "cosine";
        case Metric::euclidean:   return "euclidean";
        case Metric::dot_product: return "dot_product";
    }
    return "cosine";
}

Metric metric_from_string(std::string_view name) {
    if (name == "cosine")      return Metric::cosine;
    if (name == "euclidean")   return Metric::euclidean;
    if (name == "dot_product") return Metric::dot_product;
    throw ConfigError{"unknown metric name"};
}
```

`to_string` is used for display. `metric_from_string` is used when parsing
configuration (IDENTITY file, Python SDK parameters). It throws `ConfigError`
on unrecognised input rather than silently defaulting.

---

## Performance Considerations

- On Apple Silicon (M1/M2/M3), NEON kernels provide approximately 2.5–3.5×
  throughput over scalar code for typical 128–768 dimensional vectors.
- The squared L2 distance involves one extra subtraction per element vs dot
  product but is otherwise similar in cost.
- Cosine distance adds one extra subtraction (1 − dot) after the dot product
  — negligible overhead.
- The `distance()` function is the hottest code path in the system. Every
  candidate evaluation in `IndexFlat` linear scan and every neighbour probe
  in `IndexHNSW` beam search calls `distance()`. The NEON dispatch ensures this
  path is as close to hardware as possible with zero abstraction overhead.