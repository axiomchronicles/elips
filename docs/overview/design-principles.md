# Design Principles

> **Name update.** The code examples below use pre-rename class names to
> illustrate C++ Core Guidelines principles; the principles are unchanged, only
> the names drifted. Current names: `Database` (was `ElipsInstance`),
> `Collection` (was `Vault`), `Index`/`IndexFlat`/`IndexHNSW` (was
> `IndexPort`/`ExactIndex`/`HierarchicalGraphIndex`), `index_factory` (was
> `IndexFactory`/`make_index`), `GpuBackend` (was `GpuPort`), and verbs
> `add`/`search`/`get` (was `place`/`seek`/`fetch`). The DI-seam example
> (`Collection` depends on the abstract `Index`, domain code depends on
> `GpuBackend`) still holds exactly as drawn.

ELIPS is built from first principles in C++23 following the
[C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
and modern software engineering practices. Every design choice is intentional,
and the codebase deliberately avoids many C++ anti-patterns.

---

## C++ Core Guidelines Compliance

The project follows the C++ Core Guidelines per [ADR-0001](../adr/ADR-0001-cpp23-core.md).
Key guideline references:

| Guideline | Category | Application in ELIPS |
|-----------|----------|---------------------|
| **P.1** | Philosophy | Express intent directly in code — `Vector` instead of `float*`, `RecordID` instead of `std::array<uint8_t, 16>`, `IndexPort` instead of `void*` |
| **P.3** | Philosophy | No surprises — `distance()` always returns smaller = more similar, `Filter` default-constructed matches everything |
| **P.10** | Philosophy | Prefer immutable data — `Config` immutable after `open()`, Filter tree nodes are `shared_ptr<const Node>` |
| **I.4** | Interfaces | Precisely typed interfaces — `Vector` is a class, not a raw float array; `MetaValue` is `std::variant<>`, not `void*` |
| **I.13** | Interfaces | Never pass raw arrays — `std::span<const float>` everywhere for vector data |
| **I.23** | Interfaces | Keep function argument counts low — distance kernels take precisely 2 vectors + metric |
| **F.15** | Functions | Prefer simple, conventional info passing — value semantics on `Vector`, `RecordID`, `Record`, `SearchResult` |
| **F.16** | Functions | Pass cheaply-copied types by value — `RecordID` (128-bit, register-passable) is passed by value |
| **F.21** | Functions | Return multiple values via struct — `SearchResult`, `VaultInfo`, `GpuDeviceInfo` |
| **C.2** | Classes | Use `class` if any member is non-public — all classes use `class` keyword |
| **C.4** | Classes | Make a function a member only if it needs direct access — constructors, accessors, mutators are members; distance computation is a free function |
| **C.9** | Classes | Minimize exposure of members — all data members are `private` |
| **C.21** | Classes | Default operations or delete them — `ElipsInstance` is non-copyable, non-movable; `Vector` has value semantics; `LockManager` has move semantics |
| **C.35** | Classes | Base class destructor should be virtual — `IndexPort::~IndexPort()`, `GpuPort::~GpuPort()` are `virtual` |
| **C.128** | Classes | Virtual functions should specify `virtual`, `override`, or `final` — `HierarchicalGraphIndex` uses `final`, all port implementations use `override` |
| **C.133** | Classes | Avoid `protected` data — no `protected` data members exist in ELIPS |
| **Enum.3** | Enums | Prefer scoped enums — `enum class` for `Metric`, `IndexType`, `Durability`, `Comparator`, `GpuPolicy`, etc. |
| **Enum.7** | Enums | Specify the underlying type when necessary — `WAL::Op : std::uint8_t` |
| **R.1** | Resource mgmt | Manage resources via RAII — `LockManager`, `WAL`, smart pointers |
| **R.3** | Resource mgmt | Raw pointer is non-owning — `WAL*` in `Vault` is a non-owning reference |
| **R.11** | Resource mgmt | Avoid `new`/`delete` — `std::make_unique`, `std::make_shared` |
| **R.20** | Resource mgmt | Use `unique_ptr` or `shared_ptr` — `std::unique_ptr<IndexPort>`, `std::shared_ptr<const Node>` |
| **ES.23** | Expressions | Prefer `{}` initializer — `GraphParams{}`, `Config{}`, `Payload{}` |
| **E.14** | Error handling | Use purpose-designed exception types — `ElipsError` hierarchy with 6 concrete types |
| **E.16** | Error handling | Destructors must not throw — `ElipsInstance::~ElipsInstance()` catches all exceptions |

---

## Dependency Inversion Principle (DIP)

The most important architectural principle in ELIPS: **consumers depend on
abstractions, never on concrete implementations.**

### Index Engine

```cpp
// Vault depends on IndexPort — never HierarchicalGraphIndex, never ExactIndex.
class Vault {
    std::unique_ptr<IndexPort> index_;  // abstract
};

// make_index() is the composition root.
auto index = make_index(config, dimension);  // returns unique_ptr<IndexPort>
```

Every piece of code that performs vector search calls `index_->search()` on the
abstract `IndexPort`. The concrete index type (`ExactIndex` or
`HierarchicalGraphIndex`) is decided at construction time by `IndexFactory` and
the `Config`. This means:

- New indexes (PQ, DiskANN, segment-based) plug in behind the same interface.
- Tests can inject a mock `IndexPort`.
- `Vault` is entirely decoupled from HNSW internals.

### Metadata Engine

```cpp
// Vault::seek() and Vault::scan() depend on Filter — never on concrete predicates.
std::vector<SearchResult> Vault::seek(const Vector& query, std::size_t top,
                                       const Filter& filter, ...) const;
```

The `Filter` object is a value type representing a predicate tree. Both the SDK
(fluent builder) and the EQL parser construct `Filter` objects that are evaluated
identically. No component knows or cares which path created the filter.

### GPU Engine

```cpp
// Domain code depends on GpuPort — never CudaBackend, never MetalBackend.
// Backend classes are never included outside src/gpu_engine/.
#ifdef ELIPS_GPU_ENABLED
#include "elips/gpu_engine/GpuDeviceInfo.hpp"
#include "elips/gpu_engine/GpuMetricsSnapshot.hpp"
#endif
```

The `elips.hpp` public header includes only `GpuDeviceInfo` and
`GpuMetricsSnapshot` (pure data structs). No backend header (`CudaBackend.hpp`,
`MetalBackend.hpp`) is ever included in domain or SDK code. Backend selection
happens at startup via `GpuDeviceManager::select()`.

### Storage

```cpp
// Vault depends on WAL* (non-owning raw pointer) — never on WAL internals.
class Vault {
    WAL* wal_{nullptr};  // set by ElipsInstance::attach_wal()
};
```

`Vault::place()` calls `wal_->append_insert(...)` without knowing whether the WAL
syncs on every write or buffers — that behavior is configured at construction.

---

## RAII (Resource Acquisition Is Initialization)

Every resource is acquired in a constructor and released in a destructor. No
manual `close()`, `release()`, or `free()` calls needed in normal code paths.

### LockManager

```cpp
// src/LockManager.cpp
LockManager::LockManager(const std::string& lock_path) {
    fd_ = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0644);
    ::flock(fd_, LOCK_EX | LOCK_NB);  // acquire exclusive lock
}

LockManager::~LockManager() {
    if (fd_ >= 0) {
        ::flock(fd_, LOCK_UN);  // release on destruction
        ::close(fd_);
    }
}
```

`LockManager` is movable (`fd_` transferred via `std::exchange`), non-copyable.
The lock is held for the lifetime of the object and released on destruction or
`close()`. No dangling locks, no `atexit()` handlers.

### WAL

```cpp
WAL::WAL(std::filesystem::path path, bool sync_each_write)
    : path_(std::move(path)), sync_each_write_(sync_each_write) {
    out_.open(path_, std::ios::binary | std::ios::app);
}
```

`std::ofstream out_` is an RAII member. When the `WAL` object is destroyed, the
file stream's destructor closes the file handle.

### ElipsInstance

```cpp
ElipsInstance::~ElipsInstance() {
    if (persistent_ && !closed_) {
        try { checkpoint(); } catch (...) { /* E.16: destructors must not throw */ }
    }
}
```

On graceful teardown, the database checkpoints automatically. The catch block
prevents exception propagation through the destructor (E.16).

### Transaction

```cpp
Transaction::~Transaction() {
    if (!done_) rollback();
}
```

An un-committed transaction rolls back automatically when it goes out of scope.
This makes the "commit-or-rollback" pattern ergonomic:

```cpp
{
    auto txn = db->begin_transaction();
    auto v = txn.vault("docs");
    v.place(vec1, payload1);
    v.place(vec2, payload2);
    txn.commit();  // atomic; if this line is skipped, auto-rollback on scope exit
}
```

### Smart Pointers

- `std::unique_ptr<IndexPort> index_` — exclusive ownership of the index.
- `std::unique_ptr<WAL> wal_` — exclusive ownership of the write-ahead log.
- `std::map<std::string, std::unique_ptr<Vault>> vaults_` — exclusive vault ownership.
- `std::shared_ptr<const Node>` — shared, immutable filter tree nodes (enables structural sharing).
- `WAL* wal_` — non-owning raw pointer (C++ Core Guidelines R.3).

No `new` or `delete` appears in the codebase. All allocations use
`std::make_unique`, `std::make_shared`, or value semantics.

---

## Immutability by Default

### Config

`Config` uses a fluent builder pattern: chain `.dimension()`, `.metric()`,
`.index()` etc. to construct a configuration, then pass it to `open()`. After
construction, the config's intent is immutable — no setters exist to change
values mid-lifecycle. This prevents an entire class of bugs where a running
database's settings are accidentally modified.

### Filter Tree

Filter tree nodes are stored as `std::shared_ptr<const Node>`. Once constructed,
a filter predicate is immutable. The fluent builder accumulates a chain of
predicates, and when the chain terminates (the filter is passed to `seek()` or
`scan()`), the tree is structurally frozen. This enables:

- Safe sharing of filter trees across threads without synchronization.
- Composition without mutation (`and_()`, `or_()`, `not_()` return new `Filter` objects).
- Predictable evaluation — no stateful side effects during `matches()`.

### Vector

`Vector` is a value type. `normalized()` returns a new `Vector` — it never
mutates the original. `values()` returns `std::span<const float>`, providing
read-only access to the underlying data.

### RecordID

`RecordID` is a 128-bit value type with defaulted `==` and `<=>`. It behaves
like an integer — assign it, compare it, hash it — with no mutable state.

---

## Value Semantics

Domain types are designed as proper value types with copy, move, comparison, and
hash support:

| Type | Copy | Move | Comparison | Hash |
|------|------|------|------------|------|
| `Vector` | Yes | Yes | — | — |
| `RecordID` | Default | Default | Default `<=>` | FNV-1a via `std::hash` |
| `Record` | Default | Default | — | — |
| `SearchResult` | Default | Default | — | — |
| `Payload` | Default (map) | Default | — | — |
| `Filter` | Default (shared_ptr) | Default | — | — |
| `Config` | Default | Default | — | — |
| `GraphParams` | Default | Default | — | — |

This contrasts with pointer semantics seen in many C++ libraries. You rarely see
`const Vector&` in function signatures — vectors are passed by value or by
`std::span<const float>` when only the raw data is needed.

---

## Scoped Enums

All enumerations use `enum class` (C++ Core Guidelines Enum.3):

```cpp
enum class Metric { cosine, euclidean, dot_product };
enum class IndexType { graph, exact };
enum class Durability { paranoid, standard, relaxed, ephemeral };
enum class Comparator { eq, ne, lt, le, gt, ge };
enum class GpuPolicy { Auto, PreferGpu, RequireGpu, CpuOnly, Specific };
enum class GpuIndexAlgorithm { Auto, CagraGraph, IvfFlat, IvfPq, BruteForce };
enum class GpuPrecision { FP32, FP16, Int8, Auto };
enum class IndexBuildMode { GpuBuild_CpuServe, GpuBuild_GpuServe, Hybrid };
enum class TokenKind { word, number, string, punct, end };
```

Benefits:
- No implicit conversion to `int` — prevents accidental comparison with integers.
- Namespacing — `Metric::cosine` never conflicts with `Comparator::eq`.
- Type safety — a function taking `Metric` cannot receive `Durability` without a compile error.

Where an underlying type is needed for serialization or ABI stability, it is
explicitly specified:

```cpp
enum class Op : std::uint8_t { insert = 1, erase = 3 };  // WAL operation encoding
```

String conversion is provided via free functions:
```cpp
std::string_view to_string(Metric metric) noexcept;
Metric metric_from_string(std::string_view name);  // throws ConfigError
```

---

## Runtime Dispatch (SIMD Kernels)

The vector engine uses a runtime dispatch pattern to select the fastest SIMD
kernel for the current CPU:

```cpp
// src/Metrics.cpp:75-92
using KernelFn = float (*)(const float*, const float*, std::size_t) noexcept;

struct Dispatch {
    KernelFn dot;
    KernelFn sql2;
    Dispatch() {
#if defined(__ARM_NEON)
        dot = &dot_neon;
        sql2 = &sql2_neon;
#else
        dot = &dot_scalar;
        sql2 = &sql2_scalar;
#endif
    }
};

const Dispatch& kernels() {
    static const Dispatch instance;  // Meyer's singleton — initialized once
    return instance;
}
```

Key characteristics:
- **Zero-overhead.** The dispatch is resolved once at static initialization.
  Every call to `distance()` indirects through a function pointer (`kernels().dot(...)`),
  but the target is a leaf function that the compiler can devirtualize via LTO.
- **Extensible.** Future kernels (AVX2, AVX-512) plug into the same `Dispatch`
  struct at compile time. A runtime CPUID check can select the best kernel.
- **Scalar fallback always present.** The scalar loops in `dot_scalar` and
  `sql2_scalar` serve as the correctness baseline. SIMD kernels can be validated
  against them.
- **No dynamic library loading.** All kernels are compiled into `elips_core`.
  The dispatch is a compile-time `#ifdef` plus (future) runtime CPU feature check.

---

## Interface Segregation in the GPU Engine

The GPU engine applies the Interface Segregation Principle (ISP) aggressively.
Rather than one monolithic `GpuBackend` interface, five narrow ports each serve
one client:

### GpuPort (`GpuPort.hpp`)

The primary device port. Clients that need general GPU interaction depend only
on this:

```cpp
class GpuPort {
    virtual std::expected<void, GpuError> initialize(const GpuConfig& config) = 0;
    virtual void shutdown() noexcept = 0;
    virtual GpuDeviceInfo device_info() const noexcept = 0;
    virtual std::expected<GpuBuffer, GpuError> allocate_device(size_t bytes) = 0;
    virtual std::expected<void, GpuError> upload(const void*, GpuBuffer&, size_t) = 0;
    virtual std::expected<void, GpuError> download(const GpuBuffer&, void*, size_t) = 0;
    virtual std::expected<void, GpuError> compute_distances_batch(...) = 0;
    virtual std::expected<void, GpuError> top_k(...) = 0;
    virtual void synchronize() = 0;
};
```

### GpuMemoryPort (`GpuMemoryPort.hpp`)

For components that only need memory allocation (`GpuMemoryManager`,
`GpuSearchPipeline`):

```cpp
class GpuMemoryPort {
    virtual std::expected<GpuBuffer, GpuError> allocate(size_t, size_t alignment) = 0;
    virtual void deallocate(GpuBuffer&&) noexcept = 0;
    virtual std::expected<void*, GpuError> allocate_pinned(size_t) = 0;
    virtual void deallocate_pinned(void*) noexcept = 0;
    virtual size_t bytes_used() const noexcept = 0;
    virtual size_t bytes_available() const noexcept = 0;
};
```

### GpuKernelPort (`GpuKernelPort.hpp`)

For components that need raw distance computation (the quantization pipeline,
brute-force index fallback):

```cpp
class GpuKernelPort {
    virtual std::expected<void, GpuError> cosine_fp32(...) = 0;
    virtual std::expected<void, GpuError> euclidean_fp32(...) = 0;
    virtual std::expected<void, GpuError> dot_product_fp32(...) = 0;
};
```

### GpuStreamPort (`GpuStreamPort.hpp`)

For components that need async operation tracking (`DynamicBatcher`,
`GpuIngestionPipeline`):

```cpp
class GpuStreamPort {
    virtual std::expected<void, GpuError> synchronize() = 0;
    virtual bool is_complete() const noexcept = 0;
    virtual void wait_for_completion() = 0;
};
```

### GpuIndexPort (`GpuIndexPort.hpp`)

For components that manage GPU-side indexes. Extends `IndexPort`, bridging GPU
and CPU index abstraction:

```cpp
class GpuIndexPort : public elips::IndexPort {
    virtual std::expected<void, GpuError> build_from_batch(...) = 0;
    virtual std::expected<std::vector<std::vector<SearchResult>>, GpuError>
        search_batch(...) const = 0;
    virtual std::expected<void, GpuError> export_to_cpu_index(IndexPort&) const = 0;
    virtual std::expected<void, GpuError> import_from_cpu_index(const IndexPort&) = 0;
    virtual size_t device_bytes_used() const noexcept = 0;
};
```

This segregation means:
- A component that only needs memory management can inject `GpuMemoryPort` and
  be tested with a minimal mock — no need to mock 15 unrelated methods.
- Backend implementations can be built incrementally (e.g., a backend can
  implement only `GpuKernelPort` for brute-force, skipping `GpuIndexPort` for
  advanced indexes).
- The `GpuIndexPort` extends `IndexPort`, so GPU indexes plug into the existing
  DI seam without modifying any existing code.

---

## Error Handling Strategy

### The ElipsError Hierarchy

All errors derive from `ElipsError : public std::runtime_error`:

```
std::runtime_error
└── elips::ElipsError
    ├── elips::DimensionMismatch
    ├── elips::InvalidVector
    ├── elips::ConfigError
    ├── elips::NotFound
    ├── elips::StorageError
    ├── elips::LockConflict          (in kernel/)
    └── elips::eql::ParseError       (in query_engine/)
```

### Principles

1. **Purpose-designed exception types (E.14).** Each error category has its own
   type, making `catch` blocks precise:
   ```python no-exec
   try:
       docs.place(invalid_vector)
   except elips.DimensionMismatch:
       ...  # specific handling
   except elips.ElipsError:
       ...  # catch-all
   ```

2. **Throw by value, catch by reference.** All `throw` statements construct an
   exception on the stack using inheriting constructors:
   ```cpp
   throw DimensionMismatch{"vector dimension does not match vault"};
   throw InvalidVector{"vector contains NaN or Inf"};
   ```

3. **Destructors must not throw (E.16).** `ElipsInstance::~ElipsInstance()` wraps
   its checkpoint call in `try/catch(...)`. If the checkpoint fails during
   destruction, the exception is silently swallowed — the alternative
   (`std::terminate`) is worse.

4. **GPU errors use `std::expected`.** GPU operations return
   `std::expected<T, GpuError>` rather than throwing, because GPU errors are
   expected in normal operation (device lost, memory exhaustion) and should be
   handled as part of the fallback chain:
   ```cpp
   enum class GpuError {
       DeviceNotFound, InsufficientMemory, KernelLaunchFailed,
       TransferFailed, IndexBuildFailed, UnsupportedMetric,
       InitializationFailed, BackendUnavailable,
   };
   ```

5. **Python translation.** PyBind11 maps each C++ exception type to a
   corresponding Python exception (`elips.DimensionMismatch`,
   `elips.InvalidVector`, etc.) so Python users get idiomatic `try/except`
   semantics.

---

## Summary of Prohibitions

The codebase deliberately avoids:

| Anti-pattern | Why | What ELIPS does instead |
|-------------|-----|------------------------|
| Raw `new`/`delete` | C.149, R.11 | `std::make_unique`, `std::make_shared` |
| Raw owning pointers | R.3 | `std::unique_ptr`, `std::shared_ptr` |
| C-style arrays as parameters | I.13 | `std::span<const float>` |
| `void*` parameters | I.4 | Typed abstractions (`IndexPort`, `Vector`) |
| Mutable global state | CP.3 | No globals beyond the Meyer's singleton `kernels()` dispatch |
| `protected` data | C.133 | All data members are `private` |
| Unscoped enums | Enum.3 | `enum class` everywhere |
| `catch(...)` without rethrow | E.15 | Only in `ElipsInstance::~ElipsInstance()` per E.16 |
| Virtual calls in constructors/destructors | C.48 | Not done |
| Slicing (polymorphic copy) | C.67 | `IndexPort` and `GpuPort` delete copy/move |
| Unchecked optional access | ES.49 | `std::optional::value_or()`, explicit has_value checks |
| Macros for constants/config | ES.31 | `constexpr` variables, scoped enums |
| Type punning via unions | F.55 | `std::variant<MetaValue>` |
| `reinterpret_cast` outside serialization | ES.48 | Only in `put<>`/`get<>` template helpers |