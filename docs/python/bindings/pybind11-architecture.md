# PyBind11 Binding Architecture

This document describes how the C++ types in `elips_python.cpp` map to Python via pybind11, including type conversion, exception mapping, RAII patterns, enum registration, and text embedder bridging.

## Overview

The binding file (`bindings/python/elips_python.cpp`, 1326 lines) is a single pybind11 module named `_core`. It wraps the `elips_core` library (and optionally `elips_gpu`) and is the sole compiled source of all Python-facing classes.

```
┌──────────────────────────────────────────────────┐
│  Python                                         │
│  elips/__init__.py  ←  imports from  ────────┐  │
│  elips/_core.pyi    ←  annotates             │  │
│                                               │  │
│  ┌────────────────────────────────────────┐   │  │
│  │  _core.cpython-314-darwin.so           │   │  │
│  │  (pybind11 module)                     │   │  │
│  │                                        │   │  │
│  │  to_meta()  to_vector()  to_payload()  │   │  │
│  │  from_meta()        from_payload()     │   │  │
│  │                                        │   │  │
│  │  TransactionHolder (RAII keep_alive)   │   │  │
│  │                                        │   │  │
│  │  PYBIND11_MODULE(_core, m) { ... }     │◄──┘  │
│  └────────────────────────────────────────┘      │
│                                                   │
│  C++                                              │
│  elips_core (elips::ElipsInstance, etc.)          │
│  elips_gpu  (elips::gpu::GpuConfig, etc.)         │
└──────────────────────────────────────────────────┘
```

## Type Conversion Layer

## Text Embedder Bridge

Python text embedding reaches the C++ core through three paths:

- `Config.text_embedder(...)` wraps a Python callable in `PythonTextEmbedder`,
  which implements `elips::TextEmbedderPort`.
- `LocalEmbedderConfig` maps directly to
  `elips::LocalTextEmbedderOptions` for the built-in rehydratable local
  embedder.
- `open(..., use_default_text_embedder=True)` forwards the default-local
  preference into the core resolver, which auto-attaches the built-in local
  embedder for new databases.

The binding also exposes `TextEmbedderInfo` so Python can inspect provider,
model, revision, backend, fingerprint, storage path, rehydratable state, and
whether the embedder was auto-attached.

### Vector Conversion: `to_vector`

```cpp
elips::Vector to_vector(const py::iterable& values) {
    std::vector<float> out;
    for (const auto& v : values) {
        out.push_back(v.cast<float>());
    }
    return elips::Vector{std::move(out)};
}
```

Accepts any Python iterable of floats (`list[float]`, `tuple[float, ...]`, `numpy.ndarray` of float32, etc.). Each element is cast to `float` via pybind11's `cast<float>()`. The result is an `elips::Vector` backed by `std::vector<float>`.

Reverse direction (Vector → Python tuple):

```cpp
py::tuple tuple_from_vector(const elips::Vector& vector) {
    const auto vals = vector.values();
    py::tuple t(vals.size());
    for (std::size_t i = 0; i < vals.size(); ++i) {
        t[i] = py::float_(vals[i]);
    }
    return t;
}
```

This is used by `Vault.fetch()` to return vectors as Python tuples.

### Metadata Conversion: `to_meta` / `from_meta`

Python metadata values map to C++ `elips::MetaValue` (`std::variant<bool, int64_t, double, std::string>`):

```cpp
elips::MetaValue to_meta(const py::handle& value) {
    if (py::isinstance<py::bool_>(value)) {       // bool BEFORE int (bool is subclass of int in Python)
        return value.cast<bool>();
    }
    if (py::isinstance<py::int_>(value)) {
        return value.cast<std::int64_t>();
    }
    if (py::isinstance<py::float_>(value)) {
        return value.cast<double>();
    }
    if (py::isinstance<py::str>(value)) {
        return value.cast<std::string>();
    }
    throw py::type_error("metadata values must be int, float, bool, or str");
}
```

Critical ordering: `bool` is checked before `int` because in Python, `bool` is a subclass of `int`. Checking `int` first would incorrectly cast `True` → `1` (int64) instead of `True` (bool).

The reverse:

```cpp
py::object from_meta(const elips::MetaValue& value) {
    return std::visit([](const auto& v) -> py::object { return py::cast(v); },
                      value);
}
```

Uses `std::visit` on the variant to dispatch the correct pybind11 cast for each type.

### Payload Conversion: `to_payload` / `from_payload`

A `Payload` is `std::unordered_map<std::string, MetaValue>`:

```cpp
elips::Payload to_payload(const py::dict& data) {
    elips::Payload payload;
    for (const auto& [key, value] : data) {
        payload.emplace(key.cast<std::string>(), to_meta(value));
    }
    return payload;
}

py::dict from_payload(const elips::Payload& payload) {
    py::dict out;
    for (const auto& [key, value] : payload) {
        out[py::str(key)] = from_meta(value);
    }
    return out;
}
```

### Optional ID Conversion

```cpp
std::optional<elips::RecordID> to_optional_id(const py::object& id) {
    if (id.is_none()) {
        return std::nullopt;
    }
    return elips::RecordID::from_string(id.cast<std::string>());
}
```

## Exception Mapping

C++ exceptions are registered with pybind11 to appear as Python exception classes:

```cpp
auto elips_error =
    py::register_exception<elips::ElipsError>(m, "ElipsError", PyExc_RuntimeError);
py::register_exception<elips::DimensionMismatch>(m, "DimensionMismatch", elips_error);
py::register_exception<elips::InvalidVector>(m, "InvalidVector", elips_error);
py::register_exception<elips::ConfigError>(m, "ConfigError", elips_error);
py::register_exception<elips::NotFound>(m, "NotFound", elips_error);
py::register_exception<elips::StorageError>(m, "StorageError", elips_error);
py::register_exception<elips::LockConflict>(m, "LockConflict", elips_error);
py::register_exception<elips::eql::ParseError>(m, "ParseError", elips_error);
```

Inheritance hierarchy:

```
RuntimeError
  └── ElipsError
        ├── DimensionMismatch
        ├── InvalidVector
        ├── ConfigError
        ├── NotFound
        ├── StorageError
        ├── LockConflict
        └── ParseError
```

When any of these C++ exception types is thrown through a pybind11 boundary, pybind11 automatically translates it to the corresponding Python exception. Python `except elips.DimensionMismatch:` catches `elips::DimensionMismatch` thrown in C++.

## The TransactionHolder RAII Pattern

The C++ `elips::Transaction` holds a raw pointer to `elips::ElipsInstance`. Python must ensure the owning `Database` object outlives any `Transaction` obtained from it. The `TransactionHolder` struct solves this:

```cpp
struct TransactionHolder {
    py::object db_ref;          // Keeps the Python Database object alive
    elips::Transaction txn;     // Holds a raw pointer to ElipsInstance

    TransactionHolder(py::object db, elips::ElipsInstance& instance)
        : db_ref(std::move(db)), txn(instance) {}
};
```

When `db.begin_transaction()` is called, a `TransactionHolder` is created that stores both a reference to the Python `Database` object (incrementing its refcount) and the C++ `Transaction`:

```cpp
.def("begin_transaction",
     [](py::object db_ref) {
         auto& db = db_ref.cast<elips::ElipsInstance&>();
         return std::make_unique<TransactionHolder>(std::move(db_ref), db);
     })
```

The `TransactionVault` returned by `Transaction.vault()` uses `py::keep_alive<0, 1>()` to ensure the `Transaction` outlives the `TransactionVault`:

```cpp
.def("vault",
     [](TransactionHolder& h, const std::string& name) {
         return h.txn.vault(name);
     },
     py::keep_alive<0, 1>())   // TransactionVault keeps Transaction alive
```

### Context Manager Integration

```cpp
.def("__enter__",
     [](TransactionHolder& h) -> TransactionHolder& { return h; })
.def("__exit__",
     [](TransactionHolder& h, const py::object& exc_type,
        const py::object&, const py::object&) -> bool {
         if (exc_type.is_none()) {
             h.txn.commit();          // Clean exit → commit
         }
         return false;                // Don't suppress exceptions
     });
```

On `__exit__`, if no exception occurred (`exc_type.is_none()`), the transaction is automatically committed. If an exception was raised, nothing happens — the `Transaction` destructor rolls back.

## Config Builder Pattern

The `Config` class uses `py::return_value_policy::reference_internal` for its fluent builder methods. This ensures each chained call returns a reference to the same `Config` object rather than a copy:

```cpp
.def("dimension",
     [](elips::Config& c, std::uint16_t dim) -> elips::Config& {
         return c.dimension(dim);
     },
     py::return_value_policy::reference_internal)
.def("metric",
     [](elips::Config& c, const std::string& metric) -> elips::Config& {
         return c.metric(elips::metric_from_string(metric));
     },
     py::return_value_policy::reference_internal)
```

Python string arguments (`"cosine"`, `"graph"`, `"standard"`) are converted to their C++ enum equivalents inside the lambda before calling the C++ method.

The `gpu()` method is conditionally compiled:

```cpp
#ifdef ELIPS_GPU_ENABLED
.def("gpu",
     [](elips::Config& c, const elips::gpu::GpuConfig& gc) -> elips::Config& {
         return c.gpu(gc);
     },
     py::arg("config"),
     py::return_value_policy::reference_internal)
#endif
```

Property getters expose configuration values:

```cpp
.def_property_readonly("dimension_val",   [](const elips::Config& c) { return c.dimension(); })
.def_property_readonly("metric_val",      ...)   // returns string
.def_property_readonly("metric_enum",     ...)   // returns Metric enum
.def_property_readonly("index_val",       ...)   // returns string ("graph" | "exact")
.def_property_readonly("index_enum",      ...)   // returns IndexType enum
.def_property_readonly("graph_params_val",...)   // returns GraphParams
.def_property_readonly("durability_enum", ...)   // returns Durability enum
.def_property_readonly("gpu_val",         ...)   // returns GpuConfig or None
```

## Filter Chaining

The `Filter` class returns `reference_internal` from all builder methods, enabling fluent chaining. Each comparison method converts the Python value to `MetaValue` via `to_meta()`:

```cpp
.def("field", &elips::Filter::field, py::return_value_policy::reference_internal)
.def("equals",
     [](elips::Filter& f, const py::handle& v) -> elips::Filter& {
         return f.equals(to_meta(v));
     },
     py::return_value_policy::reference_internal)
// ... same pattern for not_equals, lt, le, gt, gte ...
```

`one_of` accepts a Python iterable and converts each element:

```cpp
.def("one_of",
     [](elips::Filter& f, const py::iterable& vs) -> elips::Filter& {
         std::vector<elips::MetaValue> set;
         for (const auto& v : vs) set.push_back(to_meta(v));
         return f.one_of(std::move(set));
     },
     py::return_value_policy::reference_internal)
```

Boolean combinators (`and_`, `or_`, `not_`) are bound directly:

```cpp
.def("and_", &elips::Filter::and_)
.def("or_", &elips::Filter::or_)
.def_static("not_", &elips::Filter::not_)
```

## Enum Registration Pattern

All enums are registered with `py::enum_` using lowercase Python-style names for values:

```cpp
py::enum_<elips::Metric>(m, "Metric")
    .value("cosine", elips::Metric::cosine)
    .value("euclidean", elips::Metric::euclidean)
    .value("dot_product", elips::Metric::dot_product)
    .export_values();
```

`export_values()` makes enum members available at module scope. The C++ enum names use CamelCase or snake_case internally, but the Python-facing names are always snake_case.

GPU enums are wrapped in `#ifdef ELIPS_GPU_ENABLED` guards and are not registered in CPU-only builds.

## Vault Lifetime

Vaults returned by `Database.vault()` use `py::return_value_policy::reference_internal`, meaning the Vault keeps the Database alive:

```cpp
.def("vault", &elips::ElipsInstance::vault,
     py::return_value_policy::reference_internal)
```

The Vault class itself is registered with `std::unique_ptr` and `py::nodelete` to indicate that the C++ `ElipsInstance` owns vault objects:

```cpp
py::class_<elips::Vault, std::unique_ptr<elips::Vault, py::nodelete>>(m, "Vault")
```

Python never deletes a Vault; the owning `ElipsInstance` manages its lifetime.

## GPU Conditional Compilation

The entire GPU section is wrapped in `#ifdef ELIPS_GPU_ENABLED` / `#endif`:

```cpp
#ifdef ELIPS_GPU_ENABLED
#include "elips/gpu_engine/GpuConfig.hpp"
// ... more includes ...

// Inside PYBIND11_MODULE:
py::enum_<elips::gpu::GpuError>(m, "GpuError")  // ... etc.
py::class_<elips::gpu::GpuConfig>(m, "GpuConfig") // ... etc.

// On Database:
.def("gpu_info", ...)
.def("gpu_stats", ...)
#endif
```

In CPU-only builds, the GPU classes are simply absent from the `_core` module. Python's `__init__.py` catches the `ImportError` and sets `_has_gpu = False`.

## `open()` / `open_with_config()` Factories

The module-level factory functions construct a `Config` internally (for the simple API) or use a provided one (for the advanced API), then call `elips::open()`:

```cpp
m.def("open",
      [](const std::string& path, std::uint16_t dimension,
         const std::string& metric, const std::string& index) {
          elips::Config config;
          config.dimension(dimension)
              .metric(elips::metric_from_string(metric));
          if (index == "exact") config.index(elips::IndexType::exact);
          return elips::open(path, config);
      },
      py::arg("path"), py::arg("dimension") = 0,
      py::arg("metric") = "cosine", py::arg("index") = "graph");

m.def("open_with_config",
      [](const std::string& path, const elips::Config& config) {
          return elips::open(path, config);
      },
      py::arg("path"), py::arg("config"));
```

Both return `std::unique_ptr<elips::ElipsInstance>`, which pybind11 auto-converts to a Python-owning `Database` object.
