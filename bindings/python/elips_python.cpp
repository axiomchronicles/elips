// PyBind11 bindings for ELIPS. Exposes the embedded database, vaults, filters,
// transactions, configuration, GPU config, and the EQL query interface to Python.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <filesystem>
#include <map>
#include <memory>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "elips/elips.hpp"
#include "elips/index_engine/IndexSnapshot.hpp"
#include "elips/query_engine/AST.hpp"
#include "elips/query_engine/EQLLexer.hpp"
#include "elips/quant_engine/Quantizer.hpp"
#include "elips/query_engine/EQLParser.hpp"
#include "elips/storage/WAL.hpp"
#include "elips/vector_engine/Metrics.hpp"

#include "elips/gpu_engine/GpuDeviceInfo.hpp"

#ifdef ELIPS_GPU_ENABLED
#include "elips/gpu_engine/DynamicBatcher.hpp"
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuDeviceManager.hpp"
#include "elips/gpu_engine/GpuMemoryManager.hpp"
#include "elips/gpu_engine/GpuMetricsSnapshot.hpp"
#include "elips/gpu_engine/GpuPort.hpp"
#include "elips/gpu_engine/GpuProfiler.hpp"
#include "elips/gpu_engine/GpuSearchPipeline.hpp"
#endif

namespace py = pybind11;

namespace {

elips::MetaValue to_meta(const py::handle& value) {
    if (py::isinstance<py::bool_>(value)) {
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

py::object from_meta(const elips::MetaValue& value) {
    return std::visit([](const auto& v) -> py::object { return py::cast(v); },
                      value);
}

elips::Payload to_payload(const py::dict& data) {
    elips::Payload payload;
    for (const auto& [key, value] : data) {
        payload.emplace(key.cast<std::string>(), to_meta(value));
    }
    return payload;
}

elips::Comparator comparator_from_string(std::string_view name) {
    if (name == "eq") return elips::Comparator::eq;
    if (name == "ne") return elips::Comparator::ne;
    if (name == "lt") return elips::Comparator::lt;
    if (name == "le") return elips::Comparator::le;
    if (name == "gt") return elips::Comparator::gt;
    if (name == "ge" || name == "gte") return elips::Comparator::ge;
    throw py::value_error(
        "unknown comparator: expected one of eq, ne, lt, le, gt, ge");
}

py::dict from_payload(const elips::Payload& payload) {
    py::dict out;
    for (const auto& [key, value] : payload) {
        out[py::str(key)] = from_meta(value);
    }
    return out;
}

elips::Vector to_vector(const py::iterable& values) {
    std::vector<float> out;
    for (const auto& v : values) {
        out.push_back(v.cast<float>());
    }
    return elips::Vector{std::move(out)};
}

py::tuple tuple_from_vector(const elips::Vector& vector) {
    const auto vals = vector.values();
    py::tuple t(vals.size());
    for (std::size_t i = 0; i < vals.size(); ++i) {
        t[i] = py::float_(vals[i]);
    }
    return t;
}

std::optional<elips::RecordID> to_optional_id(const py::object& id) {
    if (id.is_none()) {
        return std::nullopt;
    }
    return elips::RecordID::from_string(id.cast<std::string>());
}

// RAII holder for Transaction that keeps the Database alive.
// The C++ Transaction holds a raw pointer to ElipsInstance; we must ensure
// the owning Python Database object outlives it.
struct TransactionHolder {
    py::object db_ref;
    elips::Transaction txn;

    TransactionHolder(py::object db, elips::ElipsInstance& instance)
        : db_ref(std::move(db)), txn(instance) {}
};

#ifdef ELIPS_GPU_ENABLED
// Owning Python-facing handle for a selected GPU backend.
//
// Deliberately narrower than the C++ GpuPort: the raw allocate/upload/download
// methods traffic in void* plus a caller-supplied byte count, where a wrong
// size is an out-of-bounds device write rather than an exception. Those stay
// C++-only. What is exposed here -- device metadata, stream synchronization,
// the distance and top-k kernels, and memory/profiler telemetry -- is
// bounds-checked against the array shapes the caller passes in.
class GpuDeviceHandle {
public:
    explicit GpuDeviceHandle(std::unique_ptr<elips::gpu::GpuPort> port)
        : port_(std::move(port)),
          memory_(std::make_unique<elips::gpu::GpuMemoryManager>(*port_)) {}

    [[nodiscard]] elips::gpu::GpuPort& port() {
        if (closed_) {
            throw elips::StorageError{"GPU device handle is closed"};
        }
        return *port_;
    }

    [[nodiscard]] elips::gpu::GpuMemoryManager& memory() {
        if (closed_) {
            throw elips::StorageError{"GPU device handle is closed"};
        }
        return *memory_;
    }

    [[nodiscard]] elips::gpu::GpuProfiler& profiler() noexcept {
        return profiler_;
    }

    [[nodiscard]] bool closed() const noexcept { return closed_; }

    void close() noexcept {
        if (closed_) {
            return;
        }
        closed_ = true;
        memory_->shutdown();
        port_->shutdown();
    }

    ~GpuDeviceHandle() { close(); }

    GpuDeviceHandle(const GpuDeviceHandle&) = delete;
    GpuDeviceHandle& operator=(const GpuDeviceHandle&) = delete;
    GpuDeviceHandle(GpuDeviceHandle&&) = delete;
    GpuDeviceHandle& operator=(GpuDeviceHandle&&) = delete;

private:
    std::unique_ptr<elips::gpu::GpuPort> port_;
    std::unique_ptr<elips::gpu::GpuMemoryManager> memory_;
    elips::gpu::GpuProfiler profiler_;
    bool closed_{false};
};

// Turn a GpuError arm of std::expected into a Python exception. Returning a
// union type would push error handling onto every call site; raising matches
// how the rest of the binding reports failure.
void raise_gpu_error(elips::gpu::GpuError error, std::string_view context) {
    throw elips::StorageError{std::string(context) + " failed: " +
                              std::string(elips::gpu::to_string(error))};
}

std::vector<float> flatten_matrix(const py::iterable& rows, std::size_t expected_cols,
                                  std::size_t& row_count,
                                  std::string_view label) {
    std::vector<float> flat;
    row_count = 0;
    for (const auto& row : rows) {
        std::size_t cols = 0;
        for (const auto& value : py::cast<py::iterable>(row)) {
            flat.push_back(value.cast<float>());
            ++cols;
        }
        if (expected_cols != 0 && cols != expected_cols) {
            throw py::value_error(std::string(label) +
                                  " rows must all have the same length");
        }
        expected_cols = cols;
        ++row_count;
    }
    return flat;
}
#endif  // ELIPS_GPU_ENABLED

class PythonTextEmbedder final : public elips::TextEmbedderPort {
public:
    PythonTextEmbedder(py::object callable, std::string provider,
                       std::string model, std::string revision,
                       std::uint16_t expected_dimension)
        : callable_(std::move(callable)),
          provider_(std::move(provider)),
          model_(std::move(model)),
          revision_(std::move(revision)),
          expected_dimension_(expected_dimension) {}

    [[nodiscard]] elips::Vector embed(std::string_view text) const override {
        const auto batch = embed_batch({std::string(text)});
        if (batch.size() != 1) {
            throw py::value_error(
                "text embedder must return exactly one vector");
        }
        return batch.front();
    }

    [[nodiscard]] std::vector<elips::Vector> embed_batch(
        const std::vector<std::string>& texts) const override {
        py::gil_scoped_acquire gil;
        py::list batch;
        for (const auto& text : texts) {
            batch.append(py::str(text));
        }
        const py::sequence embedded =
            py::cast<py::sequence>(callable_(batch));
        if (py::len(embedded) != static_cast<py::ssize_t>(texts.size())) {
            throw py::value_error(
                "text embedder returned a batch with the wrong length");
        }

        std::vector<elips::Vector> vectors;
        vectors.reserve(texts.size());
        for (const auto& row : embedded) {
            auto vector = to_vector(py::reinterpret_borrow<py::iterable>(row));
            if (expected_dimension_ != 0 &&
                vector.dimension() != expected_dimension_) {
                throw py::value_error(
                    "text embedder returned a vector with the wrong dimension");
            }
            vectors.push_back(std::move(vector));
        }
        return vectors;
    }

    [[nodiscard]] std::string_view provider_name() const noexcept override {
        return provider_;
    }

    [[nodiscard]] std::string_view model_name() const noexcept override {
        return model_;
    }

    [[nodiscard]] std::string_view revision_name() const noexcept override {
        return revision_;
    }

    [[nodiscard]] std::string_view backend_name() const noexcept override {
        return "python-callable";
    }

    [[nodiscard]] std::uint16_t output_dimension() const noexcept override {
        return expected_dimension_;
    }

    void set_output_dimension(const std::uint16_t dimension) noexcept override {
        if (expected_dimension_ == 0) {
            expected_dimension_ = dimension;
        }
    }

private:
    py::object callable_;
    std::string provider_;
    std::string model_;
    std::string revision_;
    std::uint16_t expected_dimension_{0};
};

py::dict record_to_dict(const elips::Record& record) {
    py::dict out;
    out["id"] = record.id.to_string();
    out["vector"] = tuple_from_vector(record.vector);
    out["data"] = from_payload(record.payload);
    out["document"] = py::cast(record.document);
    out["chunk"] = py::cast(record.chunk);
    out["lineage"] = py::cast(record.lineage);
    // In a quantized vault "vector" is a reconstruction, not the bytes that
    // were written. Say so rather than let a caller assume exactness.
    out["approximate"] = record.approximate();
    out["codec"] = elips::quant::to_string(record.codec);
    return out;
}

}  // namespace

#ifdef ELIPS_GPU_ENABLED
namespace {

[[nodiscard]] std::string_view gpu_policy_name(
    const elips::gpu::GpuPolicy policy) noexcept {
    switch (policy) {
        case elips::gpu::GpuPolicy::Auto:
            return "auto";
        case elips::gpu::GpuPolicy::PreferGpu:
            return "prefer_gpu";
        case elips::gpu::GpuPolicy::RequireGpu:
            return "require_gpu";
        case elips::gpu::GpuPolicy::CpuOnly:
            return "cpu_only";
        case elips::gpu::GpuPolicy::Specific:
            return "specific";
    }
    return "unknown";
}

[[nodiscard]] std::string_view gpu_build_mode_name(
    const elips::gpu::IndexBuildMode mode) noexcept {
    switch (mode) {
        case elips::gpu::IndexBuildMode::GpuBuild_CpuServe:
            return "gpu_build_cpu_serve";
        case elips::gpu::IndexBuildMode::GpuBuild_GpuServe:
            return "gpu_build_gpu_serve";
        case elips::gpu::IndexBuildMode::Hybrid:
            return "hybrid";
    }
    return "unknown";
}

[[nodiscard]] std::string_view gpu_algorithm_name(
    const elips::gpu::GpuIndexAlgorithm algorithm) noexcept {
    switch (algorithm) {
        case elips::gpu::GpuIndexAlgorithm::Auto:
            return "auto";
        case elips::gpu::GpuIndexAlgorithm::CagraGraph:
            return "cagra";
        case elips::gpu::GpuIndexAlgorithm::IvfFlat:
            return "ivf_flat";
        case elips::gpu::GpuIndexAlgorithm::IvfPq:
            return "ivf_pq";
        case elips::gpu::GpuIndexAlgorithm::BruteForce:
            return "brute_force";
    }
    return "unknown";
}

}  // namespace
#endif

PYBIND11_MODULE(_core, m) {
    m.doc() = "ELIPS — embedded local vector database (C extension)";

    // ---- version ----
    m.attr("__version__") = "1.0.0";

    // =====================  Error hierarchy  =====================

    auto elips_error =
        py::register_exception<elips::ElipsError>(m, "ElipsError",
                                                   PyExc_RuntimeError);
    py::register_exception<elips::DimensionMismatch>(m, "DimensionMismatch",
                                                      elips_error);
    py::register_exception<elips::InvalidVector>(m, "InvalidVector",
                                                  elips_error);
    py::register_exception<elips::ConfigError>(m, "ConfigError", elips_error);
    py::register_exception<elips::NotFound>(m, "NotFound", elips_error);
    py::register_exception<elips::StorageError>(m, "StorageError",
                                                 elips_error);
    py::register_exception<elips::LockConflict>(m, "LockConflict",
                                                 elips_error);
    py::register_exception<elips::eql::ParseError>(m, "ParseError",
                                                    elips_error);

    // =====================  Core enums  =====================

    py::enum_<elips::Metric>(m, "Metric")
        .value("cosine", elips::Metric::cosine)
        .value("euclidean", elips::Metric::euclidean)
        .value("dot_product", elips::Metric::dot_product);

    py::enum_<elips::IndexType>(m, "IndexType")
        .value("graph", elips::IndexType::graph)
        .value("exact", elips::IndexType::exact);

    py::enum_<elips::Durability>(m, "Durability")
        .value("paranoid", elips::Durability::paranoid)
        .value("standard", elips::Durability::standard)
        .value("relaxed", elips::Durability::relaxed)
        .value("ephemeral", elips::Durability::ephemeral);

    py::enum_<elips::Comparator>(m, "Comparator")
        .value("eq", elips::Comparator::eq)
        .value("ne", elips::Comparator::ne)
        .value("lt", elips::Comparator::lt)
        .value("le", elips::Comparator::le)
        .value("gt", elips::Comparator::gt)
        .value("ge", elips::Comparator::ge);

    py::enum_<elips::AccessMode>(m, "AccessMode")
        .value("read_write", elips::AccessMode::read_write)
        .value("read_only", elips::AccessMode::read_only);

    py::enum_<elips::QueryStrategy>(m, "QueryStrategy")
        .value("ann_index", elips::QueryStrategy::ann_index)
        .value("exact_candidates", elips::QueryStrategy::exact_candidates)
        .value("full_scan", elips::QueryStrategy::full_scan)
        .value("text_probe", elips::QueryStrategy::text_probe)
        .value("hybrid_fusion", elips::QueryStrategy::hybrid_fusion);

    py::enum_<elips::TextEmbedderKind>(m, "TextEmbedderKind")
        .value("external", elips::TextEmbedderKind::external)
        .value("local_builtin", elips::TextEmbedderKind::local_builtin);

    // =====================  Utility functions  =====================

    m.def("generate_id",
          [] { return elips::RecordID::generate().to_string(); },
          R"doc(Generate a fresh record identifier.

Returns:
    A new unique record ID as a string, suitable for passing as the ``id``
    argument to :meth:`Vault.place`. Useful when the caller needs to know
    the identifier before the write happens.
)doc");

    m.def("is_valid_id",
          [](const std::string& id) {
              try {
                  (void)elips::RecordID::from_string(id);
                  return true;
              } catch (const elips::ElipsError&) {
                  return false;
              }
          },
          py::arg("id"),
          "Return True when ``id`` is a well-formed record identifier.");

    m.def("normalize",
          [](const py::iterable& values) {
              return tuple_from_vector(to_vector(values).normalized());
          },
          py::arg("vector"),
          R"doc(Return ``vector`` scaled to unit length.

A zero vector is returned unchanged. Cosine similarity requires normalized
inputs; :meth:`Vault.place` and :meth:`Vault.seek` already normalize
internally when the vault metric needs it, so this is exposed for callers
doing their own pre-processing.
)doc");

    m.def("magnitude",
          [](const py::iterable& values) {
              return to_vector(values).magnitude();
          },
          py::arg("vector"),
          "Return the Euclidean length (L2 norm) of ``vector``.");

    m.def("distance",
          [](const py::object& metric_arg, const py::iterable& a,
             const py::iterable& b) {
              elips::Metric metric;
              if (py::isinstance<py::str>(metric_arg)) {
                  metric = elips::metric_from_string(metric_arg.cast<std::string>());
              } else {
                  metric = metric_arg.cast<elips::Metric>();
              }
              const auto va = to_vector(a);
              const auto vb = to_vector(b);
              return elips::distance(metric, va.values(), vb.values());
          },
          py::arg("metric"), py::arg("a"), py::arg("b"),
          "Compute the ordering-normalized distance between two vectors.");

    m.def("requires_normalization",
          [](const py::object& metric_arg) {
              elips::Metric metric;
              if (py::isinstance<py::str>(metric_arg)) {
                  metric = elips::metric_from_string(metric_arg.cast<std::string>());
              } else {
                  metric = metric_arg.cast<elips::Metric>();
              }
              return elips::requires_normalization(metric);
          },
          py::arg("metric"),
          "Return True if vectors should be L2-normalized for this metric.");

    m.def("metric_to_string",
          [](elips::Metric metric) -> std::string {
              return std::string(elips::to_string(metric));
          },
          py::arg("metric"), "Convert a Metric enum value to its string name.");

    m.def("metric_from_string",
          [](const std::string& name) -> elips::Metric {
              return elips::metric_from_string(name);
          },
          py::arg("name"), "Parse a string into a Metric enum value.");

    // =====================  EQL parsing functions  =====================

    m.def("validate_eql",
          [](const std::string& source) -> py::object {
              (void)elips::eql::parse(source);
              return py::none();
          },
          py::arg("source"),
          "Validate an EQL statement. Returns None on success, raises ParseError on invalid syntax.");

    m.def("tokenize_eql", &elips::eql::tokenize, py::arg("source"),
          "Tokenize an EQL source string. Returns a list of Token objects.");

    // =====================  EQL abstract syntax tree  =====================

    py::class_<elips::eql::VectorRef>(
        m, "VectorRef",
        "A query vector in a parsed EQL statement: either an inline literal or "
        "a named binding supplied at execution time.")
        .def(py::init<>())
        .def_readwrite("literal", &elips::eql::VectorRef::literal,
                       "Inline vector values; empty when a binding is used.")
        .def_readwrite("binding", &elips::eql::VectorRef::binding,
                       "Binding name (``$name`` in EQL); empty for literals.")
        .def("__repr__", [](const elips::eql::VectorRef& ref) {
            return ref.binding.empty()
                       ? "<VectorRef literal dim=" +
                             std::to_string(ref.literal.size()) + ">"
                       : "<VectorRef binding='" + ref.binding + "'>";
        });

    py::class_<elips::eql::SearchStatement>(
        m, "SearchStatement", "A parsed EQL ``SEARCH`` statement.")
        .def(py::init<>())
        .def_readwrite("vault", &elips::eql::SearchStatement::vault)
        .def_readwrite("query", &elips::eql::SearchStatement::query)
        .def_readwrite("top", &elips::eql::SearchStatement::top,
                       "Result limit, or ``None`` when unspecified.")
        .def_readwrite("threshold", &elips::eql::SearchStatement::threshold,
                       "Maximum distance, or ``None``.")
        .def_readwrite("where", &elips::eql::SearchStatement::where,
                       "Metadata filter; matches everything when absent.")
        .def_readwrite("rank_by", &elips::eql::SearchStatement::rank_by,
                       "Ranking field, or ``None`` to rank by distance.")
        .def_readwrite("projection", &elips::eql::SearchStatement::projection,
                       "Requested fields; empty means all fields.")
        .def("__repr__", [](const elips::eql::SearchStatement& s) {
            return "<SearchStatement vault='" + s.vault + "'>";
        });

    py::class_<elips::eql::FetchStatement>(
        m, "FetchStatement", "A parsed EQL ``FETCH`` statement.")
        .def(py::init<>())
        .def_readwrite("vault", &elips::eql::FetchStatement::vault)
        .def_readwrite("id", &elips::eql::FetchStatement::id)
        .def("__repr__", [](const elips::eql::FetchStatement& s) {
            return "<FetchStatement vault='" + s.vault + "' id='" + s.id + "'>";
        });

    py::class_<elips::eql::ScanStatement>(
        m, "ScanStatement", "A parsed EQL ``SCAN`` statement.")
        .def(py::init<>())
        .def_readwrite("vault", &elips::eql::ScanStatement::vault)
        .def_readwrite("where", &elips::eql::ScanStatement::where)
        .def_readwrite("offset", &elips::eql::ScanStatement::offset)
        .def_readwrite("limit", &elips::eql::ScanStatement::limit)
        .def("__repr__", [](const elips::eql::ScanStatement& s) {
            return "<ScanStatement vault='" + s.vault + "'>";
        });

    py::class_<elips::eql::InsertStatement>(
        m, "InsertStatement", "A parsed EQL ``INSERT`` statement.")
        .def(py::init<>())
        .def_readwrite("vault", &elips::eql::InsertStatement::vault)
        .def_readwrite("vector", &elips::eql::InsertStatement::vector)
        .def_property(
            "data",
            [](const elips::eql::InsertStatement& s) {
                return from_payload(s.data);
            },
            [](elips::eql::InsertStatement& s, const py::dict& data) {
                s.data = to_payload(data);
            },
            "Payload to store with the record.")
        .def("__repr__", [](const elips::eql::InsertStatement& s) {
            return "<InsertStatement vault='" + s.vault + "'>";
        });

    py::class_<elips::eql::DeleteStatement>(
        m, "DeleteStatement", "A parsed EQL ``DELETE`` statement.")
        .def(py::init<>())
        .def_readwrite("vault", &elips::eql::DeleteStatement::vault)
        .def_readwrite("id", &elips::eql::DeleteStatement::id)
        .def("__repr__", [](const elips::eql::DeleteStatement& s) {
            return "<DeleteStatement vault='" + s.vault + "' id='" + s.id + "'>";
        });

    m.def("parse_eql",
          [](const std::string& source) {
              return elips::eql::parse(source);
          },
          py::arg("source"),
          R"doc(Parse an EQL statement into its abstract syntax tree.

Args:
    source: EQL source text.

Returns:
    One of :class:`SearchStatement`, :class:`FetchStatement`,
    :class:`ScanStatement`, :class:`InsertStatement`, or
    :class:`DeleteStatement`. Use ``isinstance`` to discriminate.

Raises:
    ParseError: If the statement is not valid EQL.

Unlike :func:`validate_eql`, which discards the result, this returns the
parsed tree so callers can build query linters, rewriters, or builders.
)doc");

    py::enum_<elips::eql::TokenKind>(m, "TokenKind")
        .value("word", elips::eql::TokenKind::word)
        .value("number", elips::eql::TokenKind::number)
        .value("string", elips::eql::TokenKind::string)
        .value("punct", elips::eql::TokenKind::punct)
        .value("end", elips::eql::TokenKind::end);

    py::class_<elips::eql::Token>(m, "Token")
        .def(py::init<>())
        .def_readonly("kind", &elips::eql::Token::kind)
        .def_readonly("text", &elips::eql::Token::text)
        .def_readonly("number", &elips::eql::Token::number)
        .def_readonly("is_integer", &elips::eql::Token::is_integer)
        .def("__repr__", [](const elips::eql::Token& t) {
            return "<Token kind=" + std::to_string(static_cast<int>(t.kind)) +
                   " text='" + t.text + "'>";
        });

    // =====================  GraphParams  =====================

    py::class_<elips::GraphParams>(m, "GraphParams")
        .def(py::init<>())
        .def(py::init([](std::size_t max_connections, std::size_t ef_construction,
                         std::size_t ef_search, float compaction_ratio) {
                 return elips::GraphParams{max_connections, ef_construction,
                                           ef_search, compaction_ratio};
             }),
             py::arg("max_connections") = 16,
             py::arg("ef_construction") = 200, py::arg("ef_search") = 50,
             py::arg("compaction_ratio") = 0.2F,
             R"doc(Tuning parameters for the HNSW graph index.

Args:
    max_connections: HNSW ``M``. Neighbours kept per node per layer. Higher
        values improve recall and increase memory and build time.
    ef_construction: Beam width during index construction. Higher values
        build a better-connected graph at the cost of insert throughput.
    ef_search: Beam width during search. Higher values improve recall at
        the cost of query latency.
    compaction_ratio: Fraction of tombstoned nodes that triggers an
        automatic index rebuild. ``0.0`` disables automatic compaction, in
        which case call :meth:`Vault.vacuum` explicitly.
)doc")
        .def_readwrite("max_connections",
                        &elips::GraphParams::max_connections,
                        "HNSW ``M``: neighbours kept per node per layer.")
        .def_readwrite("ef_construction",
                        &elips::GraphParams::ef_construction,
                        "Beam width used while building the graph.")
        .def_readwrite("ef_search", &elips::GraphParams::ef_search,
                        "Beam width used while searching the graph.")
        .def_readwrite("compaction_ratio",
                        &elips::GraphParams::compaction_ratio,
                        "Tombstone fraction that triggers automatic compaction; "
                        "0.0 disables it.")
        .def("__repr__", [](const elips::GraphParams& p) {
            return "<GraphParams M=" + std::to_string(p.max_connections) +
                   " ef_c=" + std::to_string(p.ef_construction) +
                   " ef_s=" + std::to_string(p.ef_search) +
                   " compaction=" + std::to_string(p.compaction_ratio) + ">";
        });

    // =====================  QuantParams / Codec  =====================

    py::enum_<elips::quant::CodecId>(m, "Codec")
        .value("none", elips::quant::CodecId::none)
        .value("pq", elips::quant::CodecId::pq)
        .value("opq", elips::quant::CodecId::opq)
        .value("sq8", elips::quant::CodecId::sq8);

    py::class_<elips::quant::QuantParams>(m, "QuantParams")
        .def(py::init<>())
        .def(py::init([](const std::string& codec, std::uint32_t pq_dim,
                         std::uint32_t pq_bits, std::uint32_t train_iters,
                         std::uint32_t opq_iters) {
                 elips::quant::QuantParams params;
                 params.codec = elips::quant::codec_from_string(codec.c_str());
                 params.pq_dim = pq_dim;
                 params.pq_bits = pq_bits;
                 params.train_iters = train_iters;
                 params.opq_iters = opq_iters;
                 return params;
             }),
             py::arg("codec") = "none", py::arg("pq_dim") = 0,
             py::arg("pq_bits") = 8, py::arg("train_iters") = 10,
             py::arg("opq_iters") = 4,
             R"doc(Vector compression parameters.

Selecting a codec does not compress anything on its own: a codebook has to be
trained on real data first. Call :meth:`Vault.quantize` (or
:meth:`Database.quantize`) once the vault holds a representative sample.

Args:
    codec: One of ``"none"``, ``"pq"``, ``"opq"``, or ``"sq8"``.

        * ``pq`` -- product quantization. Highest compression (one byte per
          subspace regardless of dimension), largest recall cost.
        * ``opq`` -- product quantization with a learned rotation. Same code
          width as ``pq``, better recall when dimensions are correlated, which
          is the usual case for learned embeddings.
        * ``sq8`` -- per-dimension int8. A flat 4x with near-exact recall and
          no codebook. The right default when memory is a concern rather than
          the binding constraint.
    pq_dim: Number of subspaces for ``pq``/``opq``, which is also the code
        width in bytes. Must divide the vector dimension; ``0`` picks the
        largest divisor at or below ``dimension / 8``. Ignored by ``sq8``.
    pq_bits: Bits per subquantizer, 4 to 8. The codebook holds ``2**pq_bits``
        centroids per subspace. Ignored by ``sq8``.
    train_iters: Lloyd iterations per subspace codebook.
    opq_iters: Rotation/codebook alternations. Ignored unless the codec is
        ``opq``.

Examples:
    32x compression on 768-dimensional embeddings::

        >>> import elips
        >>> params = elips.QuantParams(codec="pq", pq_dim=24)
        >>> config = elips.Config().dimension(768).quantization(params)

    Near-lossless 4x instead::

        >>> params = elips.QuantParams(codec="sq8")
)doc")
        .def_property(
            "codec",
            [](const elips::quant::QuantParams& p) {
                return std::string(elips::quant::to_string(p.codec));
            },
            [](elips::quant::QuantParams& p, const std::string& name) {
                p.codec = elips::quant::codec_from_string(name.c_str());
            },
            "Codec name: ``none``, ``pq``, ``opq``, or ``sq8``.")
        .def_readwrite("pq_dim", &elips::quant::QuantParams::pq_dim,
                       "Subspace count, which is also the code width in bytes. "
                       "0 selects automatically.")
        .def_readwrite("pq_bits", &elips::quant::QuantParams::pq_bits,
                       "Bits per subquantizer code, 4 to 8.")
        .def_readwrite("train_iters", &elips::quant::QuantParams::train_iters,
                       "Lloyd iterations per subspace codebook.")
        .def_readwrite("opq_iters", &elips::quant::QuantParams::opq_iters,
                       "Rotation/codebook alternations; ``opq`` only.")
        .def_property_readonly(
            "codec_enum", [](const elips::quant::QuantParams& p) { return p.codec; },
            "The codec as a :class:`Codec` enum member.")
        .def("code_bytes",
             [](const elips::quant::QuantParams& p, std::uint16_t dimension) {
                 return elips::quant::code_bytes_for(p, dimension);
             },
             py::arg("dimension"),
             "Bytes one vector will occupy at ``dimension``, for computing a "
             "compression ratio before training.")
        .def("validate",
             [](const elips::quant::QuantParams& p, std::uint16_t dimension) {
                 elips::quant::validate(p, dimension);
             },
             py::arg("dimension"),
             "Raises :class:`ConfigError` if these parameters cannot describe a "
             "valid codec at ``dimension``.")
        .def("__repr__", [](const elips::quant::QuantParams& p) {
            return "<QuantParams codec=" +
                   std::string(elips::quant::to_string(p.codec)) +
                   " pq_dim=" + std::to_string(p.pq_dim) +
                   " pq_bits=" + std::to_string(p.pq_bits) + ">";
        });

    py::class_<elips::LocalTextEmbedderOptions>(m, "LocalEmbedderConfig")
        .def(py::init<>())
        .def(py::init<std::string, std::string, std::string, std::uint16_t>(),
             py::arg("model") = "default", py::arg("revision") = "v1",
             py::arg("storage_path") = "", py::arg("dimension") = 0)
        .def_readwrite("model", &elips::LocalTextEmbedderOptions::model)
        .def_readwrite("revision", &elips::LocalTextEmbedderOptions::revision)
        .def_readwrite("storage_path",
                       &elips::LocalTextEmbedderOptions::storage_path)
        .def_readwrite("dimension", &elips::LocalTextEmbedderOptions::dimension)
        .def("__repr__", [](const elips::LocalTextEmbedderOptions& options) {
            return "<LocalEmbedderConfig model='" + options.model +
                   "' revision='" + options.revision +
                   "' dimension=" + std::to_string(options.dimension) + ">";
        });

    py::class_<elips::TextEmbedderInfo>(m, "TextEmbedderInfo")
        .def(py::init<>())
        .def_readonly("kind", &elips::TextEmbedderInfo::kind)
        .def_readonly("provider", &elips::TextEmbedderInfo::provider)
        .def_readonly("model", &elips::TextEmbedderInfo::model)
        .def_readonly("revision", &elips::TextEmbedderInfo::revision)
        .def_readonly("backend", &elips::TextEmbedderInfo::backend)
        .def_readonly("dimension", &elips::TextEmbedderInfo::dimension)
        .def_readonly("fingerprint", &elips::TextEmbedderInfo::fingerprint)
        .def_readonly("storage_path", &elips::TextEmbedderInfo::storage_path)
        .def_readonly("rehydratable", &elips::TextEmbedderInfo::rehydratable)
        .def_readonly("loaded", &elips::TextEmbedderInfo::loaded)
        .def_readonly("auto_attached", &elips::TextEmbedderInfo::auto_attached)
        .def("__repr__", [](const elips::TextEmbedderInfo& info) {
            return "<TextEmbedderInfo provider='" + info.provider +
                   "' model='" + info.model +
                   "' dimension=" + std::to_string(info.dimension) + ">";
        });

    // =====================  Index snapshots  =====================

    py::enum_<elips::IndexSnapshotKind>(m, "IndexSnapshotKind",
                                        "Which index produced a snapshot.")
        .value("unknown", elips::IndexSnapshotKind::unknown)
        .value("exact", elips::IndexSnapshotKind::exact)
        .value("graph", elips::IndexSnapshotKind::graph)
        .value("gpu_brute_force", elips::IndexSnapshotKind::gpu_brute_force)
        .value("gpu_ivf_flat", elips::IndexSnapshotKind::gpu_ivf_flat)
        .value("gpu_ivf_pq", elips::IndexSnapshotKind::gpu_ivf_pq)
        .value("gpu_graph", elips::IndexSnapshotKind::gpu_graph)
        .value("gpu_hybrid", elips::IndexSnapshotKind::gpu_hybrid)
        .value("gpu_distributed", elips::IndexSnapshotKind::gpu_distributed);

    py::class_<elips::IvfSnapshot>(
        m, "IvfSnapshot", "Inverted-file clustering state from an IVF index.")
        .def(py::init<>())
        .def_readwrite("n_lists", &elips::IvfSnapshot::n_lists,
                       "Number of coarse clusters.")
        .def_readwrite("n_probe", &elips::IvfSnapshot::n_probe,
                       "Clusters visited per query.")
        .def_readwrite("centroids", &elips::IvfSnapshot::centroids,
                       "Row-major cluster centroids.")
        .def_readwrite("assignments", &elips::IvfSnapshot::assignments,
                       "Per-vector cluster assignment.")
        .def("__repr__", [](const elips::IvfSnapshot& s) {
            return "<IvfSnapshot n_lists=" + std::to_string(s.n_lists) + ">";
        });

    py::class_<elips::PqSnapshot>(
        m, "PqSnapshot", "Product-quantization codebook and codes.")
        .def(py::init<>())
        .def_readwrite("pq_dim", &elips::PqSnapshot::pq_dim,
                       "Number of subquantizers.")
        .def_readwrite("pq_bits", &elips::PqSnapshot::pq_bits,
                       "Bits per subquantizer code.")
        .def_readwrite("codebook", &elips::PqSnapshot::codebook,
                       "Trained centroids for every subquantizer.")
        .def_readwrite("codes", &elips::PqSnapshot::codes,
                       "Encoded vectors.")
        .def("__repr__", [](const elips::PqSnapshot& s) {
            return "<PqSnapshot pq_dim=" + std::to_string(s.pq_dim) +
                   " pq_bits=" + std::to_string(s.pq_bits) + ">";
        });

    py::class_<elips::IndexSnapshot>(
        m, "IndexSnapshot",
        "A portable dump of index contents, used to move an index between "
        "backends (CPU to GPU and back) or to inspect it offline.")
        .def(py::init<>())
        .def_readwrite("kind", &elips::IndexSnapshot::kind)
        .def_readwrite("metric", &elips::IndexSnapshot::metric)
        .def_readwrite("dimension", &elips::IndexSnapshot::dimension)
        .def_property_readonly(
            "ids",
            [](const elips::IndexSnapshot& s) {
                py::list out;
                for (const auto& id : s.ids) {
                    out.append(py::str(id.to_string()));
                }
                return out;
            },
            "Record identifiers, aligned with :attr:`vectors`.")
        .def_readwrite("vectors", &elips::IndexSnapshot::vectors,
                       "Row-major vector data.")
        .def_readwrite("ivf", &elips::IndexSnapshot::ivf,
                       "IVF clustering state, when present.")
        .def_readwrite("pq", &elips::IndexSnapshot::pq,
                       "Product-quantization state, when present.")
        .def("__len__",
             [](const elips::IndexSnapshot& s) { return s.ids.size(); })
        .def("__repr__", [](const elips::IndexSnapshot& s) {
            return "<IndexSnapshot vectors=" + std::to_string(s.ids.size()) +
                   " dimension=" + std::to_string(s.dimension) + ">";
        });

    // =====================  Write-ahead log  =====================

    py::enum_<elips::WAL::Op>(m, "WalOp", "Kind of a write-ahead log record.")
        .value("insert", elips::WAL::Op::insert)
        .value("erase", elips::WAL::Op::erase)
        .value("insert_ex", elips::WAL::Op::insert_ex,
               "Insert carrying document, chunk, or lineage attachments.")
        .value("txn_begin", elips::WAL::Op::txn_begin,
               "Start of a transaction batch.")
        .value("txn_commit", elips::WAL::Op::txn_commit,
               "End of a transaction batch. Records inside an unterminated "
               "begin..commit window are discarded on replay.");

    py::class_<elips::WAL::Entry>(m, "WalEntry",
                                  "One replayed write-ahead log record.")
        .def_readonly("op", &elips::WAL::Entry::op)
        .def_readonly("vault", &elips::WAL::Entry::vault)
        .def_property_readonly(
            "id",
            [](const elips::WAL::Entry& e) { return e.id.to_string(); },
            "Record identifier this entry applies to.")
        .def_property_readonly(
            "vector",
            [](const elips::WAL::Entry& e) {
                py::tuple out(e.vector.size());
                for (std::size_t i = 0; i < e.vector.size(); ++i) {
                    out[i] = py::float_(e.vector[i]);
                }
                return out;
            },
            "Vector payload; empty for erases and transaction markers.")
        .def_property_readonly(
            "data",
            [](const elips::WAL::Entry& e) { return from_payload(e.payload); },
            "Metadata payload.")
        .def_readonly("document", &elips::WAL::Entry::document)
        .def_readonly("chunk", &elips::WAL::Entry::chunk)
        .def_readonly("lineage", &elips::WAL::Entry::lineage)
        .def("__repr__", [](const elips::WAL::Entry& e) {
            return "<WalEntry vault='" + e.vault + "' id='" + e.id.to_string() +
                   "'>";
        });

    m.def("replay_wal",
          [](const std::string& path) {
              // Reads and parses a whole log file: pure I/O plus native
              // decoding, with nothing Python-facing until the return.
              py::gil_scoped_release unlocked;
              return elips::WAL::replay(std::filesystem::path{path});
          },
          py::arg("path"),
          R"doc(Replay a write-ahead log file without opening the database.

Args:
    path: Path to a ``wal.log`` file.

Returns:
    A list of :class:`WalEntry` in log order. Records inside an
    unterminated transaction window are omitted, matching what recovery
    would apply. A corrupt or truncated tail is dropped rather than raised,
    so a partial log still yields its valid prefix.

Intended for crash forensics and recovery tooling: it answers "what did the
database actually acknowledge before it died?" without mutating anything.
)doc");

    m.def("describe_local_embedder",
          [](const elips::LocalTextEmbedderOptions& options,
             std::uint16_t fallback_dimension, bool auto_attached) {
              return elips::describe_local_text_embedder(
                  options, fallback_dimension, auto_attached);
          },
          py::arg("config") = elips::LocalTextEmbedderOptions{},
          py::arg("fallback_dimension") = 0,
          py::arg("auto_attached") = false,
          R"doc(Describe a local text embedder without instantiating it.

Args:
    config: The :class:`LocalEmbedderConfig` to describe.
    fallback_dimension: Dimension to assume when ``config.dimension`` is 0.
    auto_attached: Whether the embedder would be attached automatically.

Returns:
    A :class:`TextEmbedderInfo` with the resolved dimension, fingerprint,
    and storage path. Use this to check compatibility with an existing
    database before opening it.
)doc");

    // =====================  Config  =====================

    py::class_<elips::Config>(m, "Config", R"doc(A fluent builder for everything fixed at database-open time.

Use this instead of :func:`open`'s keyword arguments when you need durability,
graph tuning, segmented storage, or GPU settings. Pass the result to
:func:`open_with_config`.

Dimension, metric, and index type become part of the database's persisted
identity: reopening with conflicting values raises ``ConfigError`` rather than
silently reinterpreting stored vectors.

Example:
    Tune for a write-heavy ingest, then reopen the same database read-only for
    serving -- the two-phase pattern behind most production deployments::

        import elips

        # 1. Ingest phase: relaxed durability trades crash safety for
        #    throughput, since the source data can be replayed. A wide
        #    ef_construction builds a higher-recall graph once, up front.
        ingest_cfg = (elips.Config()
                      .dimension(768)
                      .metric("cosine")
                      .index("graph")
                      .durability("relaxed")
                      .graph_params(elips.GraphParams(
                          max_connections=32,      # denser graph, better recall
                          ef_construction=400,
                          ef_search=100,
                          compaction_ratio=0.0,    # no compaction mid-ingest
                      ))
                      .segmented_storage(True)     # per-vault segment files
                      .metadata_acceleration(True))

        db = elips.open_with_config("/data/index", ingest_cfg)
        ...                                        # bulk load
        db.compact()                               # rebuild + checkpoint
        db.close()

        # 2. Serving phase: read-only takes a shared lock, so many worker
        #    processes can serve the same directory concurrently.
        serve_cfg = elips.Config().access_mode("read_only")
        db = elips.open_with_config("/data/index", serve_cfg)

        # 3. Dimension and metric come from the persisted identity; they do not
        #    need repeating, and conflicting values would raise ConfigError.
        print(db.config().dimension_val, db.config().metric_val)
)doc")
        .def(py::init<>())
        .def("dimension",
             [](elips::Config& c, std::uint16_t dim) -> elips::Config& {
                 return c.dimension(dim);
             },
             py::arg("dim"),
             py::return_value_policy::reference_internal)
        .def("metric",
             [](elips::Config& c, const std::string& metric) -> elips::Config& {
                 return c.metric(elips::metric_from_string(metric));
             },
             py::arg("metric"),
             py::return_value_policy::reference_internal)
        .def("index",
             [](elips::Config& c, const std::string& type) -> elips::Config& {
                 if (type == "exact") {
                     c.index(elips::IndexType::exact);
                 } else {
                     c.index(elips::IndexType::graph);
                 }
                 return c;
             },
             py::arg("type"),
             py::return_value_policy::reference_internal)
        .def("graph_params",
             [](elips::Config& c,
                const elips::GraphParams& params) -> elips::Config& {
                 return c.graph_params(params);
             },
             py::arg("params"),
             py::return_value_policy::reference_internal)
        .def("quantization",
             [](elips::Config& c,
                const elips::quant::QuantParams& params) -> elips::Config& {
                 return c.quantization(params);
             },
             py::arg("params"), py::return_value_policy::reference_internal,
             "Selects a compression codec. Call :meth:`Vault.quantize` to train "
             "it and compress the vault.")
        .def("durability",
             [](elips::Config& c, const std::string& level) -> elips::Config& {
                 if (level == "paranoid") {
                     c.durability(elips::Durability::paranoid);
                 } else if (level == "relaxed") {
                     c.durability(elips::Durability::relaxed);
                 } else if (level == "ephemeral") {
                     c.durability(elips::Durability::ephemeral);
                 } else {
                     c.durability(elips::Durability::standard);
                 }
                 return c;
             },
             py::arg("level"),
             py::return_value_policy::reference_internal)
        .def("access_mode",
             [](elips::Config& c, const std::string& mode) -> elips::Config& {
                 if (mode == "read_only") {
                     c.access_mode(elips::AccessMode::read_only);
                 } else {
                     c.access_mode(elips::AccessMode::read_write);
                 }
                 return c;
             },
             py::arg("mode"),
             py::return_value_policy::reference_internal)
        .def("segmented_storage",
             [](elips::Config& c, bool enabled) -> elips::Config& {
                 return c.segmented_storage(enabled);
             },
             py::arg("enabled"),
             py::return_value_policy::reference_internal)
        .def("metadata_acceleration",
             [](elips::Config& c, bool enabled) -> elips::Config& {
                 return c.metadata_acceleration(enabled);
             },
             py::arg("enabled"),
             py::return_value_policy::reference_internal)
        .def("auto_text_embedder",
             [](elips::Config& c, bool enabled) -> elips::Config& {
                 return c.auto_text_embedder(enabled);
             },
             py::arg("enabled"),
             py::return_value_policy::reference_internal)
        .def("local_text_embedder",
             [](elips::Config& c,
                const elips::LocalTextEmbedderOptions& options)
                 -> elips::Config& { return c.local_text_embedder(options); },
             py::arg("config") = elips::LocalTextEmbedderOptions{},
             py::return_value_policy::reference_internal)
        .def("text_embedder",
             [](elips::Config& c, const py::object& embedder,
                const std::string& provider,
                const std::string& model, const std::string& revision,
                std::uint16_t dimension) -> elips::Config& {
                 if (embedder.is_none()) {
                     return c.text_embedder(elips::TextEmbedderPtr{});
                 }
                 return c.text_embedder(std::make_shared<PythonTextEmbedder>(
                     embedder, provider, model, revision,
                     dimension == 0 ? c.dimension() : dimension));
             },
             py::arg("embedder"), py::arg("provider") = "python",
             py::arg("model") = "callable",
             py::arg("revision") = "",
             py::arg("dimension") = 0,
             py::return_value_policy::reference_internal)
#ifdef ELIPS_GPU_ENABLED
        .def("gpu",
             [](elips::Config& c, const elips::gpu::GpuConfig& gc) -> elips::Config& {
                 return c.gpu(gc);
             },
             py::arg("config"),
             py::return_value_policy::reference_internal)
        .def_property_readonly("gpu_val", [](const elips::Config& c) -> py::object {
            if (!c.has_gpu()) return py::none();
            return py::cast(c.gpu());
        })
#endif
        .def_property_readonly(
            "dimension_val",
            [](const elips::Config& c) { return c.dimension(); })
        .def_property_readonly("metric_val", [](const elips::Config& c) {
            return std::string(elips::to_string(c.metric()));
        })
        .def_property_readonly(
            "index_val",
            [](const elips::Config& c) -> std::string {
                return c.index() == elips::IndexType::graph ? "graph"
                                                             : "exact";
            })
        .def_property_readonly("graph_params_val",
                               [](const elips::Config& c) {
                                   return c.graph_params();
                               })
        .def_property_readonly("quantization_val",
                               [](const elips::Config& c) {
                                   return c.quantization();
                               })
        .def_property_readonly("has_quantization",
                               &elips::Config::has_quantization,
                               "True when a compression codec is configured.")
        .def_property_readonly("metric_enum", [](const elips::Config& c) {
            return c.metric();
        })
        .def_property_readonly("index_enum", [](const elips::Config& c) {
            return c.index();
        })
        .def_property_readonly("durability_enum", [](const elips::Config& c) {
            return c.durability();
        })
        .def_property_readonly(
            "durability_val",
            [](const elips::Config& c) -> std::string {
                switch (c.durability()) {
                    case elips::Durability::paranoid: return "paranoid";
                    case elips::Durability::standard: return "standard";
                    case elips::Durability::relaxed: return "relaxed";
                    case elips::Durability::ephemeral: return "ephemeral";
                }
                return "standard";
            },
            "Durability level as a string.")
        .def_property_readonly(
            "has_gpu",
            [](const elips::Config& c) {
#ifdef ELIPS_GPU_ENABLED
                return c.has_gpu();
#else
                (void)c;
                return false;
#endif
            },
            "True when a GPU policy other than CPU-only is configured.")
        .def_property_readonly(
            "has_pending_local_text_embedder",
            [](const elips::Config& c) {
                return c.has_pending_local_text_embedder();
            },
            "True when a local text embedder is configured but not yet "
            "instantiated (it is created when the database is opened).")
        .def_property_readonly(
            "local_text_embedder_config",
            [](const elips::Config& c) -> py::object {
                const auto& options = c.local_text_embedder_options();
                if (!options.has_value()) {
                    return py::none();
                }
                return py::cast(*options);
            },
            "The pending :class:`LocalEmbedderConfig`, or ``None``.")
        .def_property_readonly("access_mode_val",
                               [](const elips::Config& c) -> std::string {
                                   return c.access_mode() ==
                                                  elips::AccessMode::read_only
                                              ? "read_only"
                                              : "read_write";
                               })
        .def_property_readonly("access_mode_enum",
                               [](const elips::Config& c) {
                                   return c.access_mode();
                               })
        .def_property_readonly("segmented_storage_enabled",
                               [](const elips::Config& c) {
                                   return c.segmented_storage();
                               })
        .def_property_readonly("metadata_acceleration_enabled",
                               [](const elips::Config& c) {
                                   return c.metadata_acceleration();
                               })
        .def_property_readonly("auto_text_embedder_enabled",
                               [](const elips::Config& c) {
                                   return c.auto_text_embedder();
                               })
        .def_property_readonly("has_text_embedder",
                               [](const elips::Config& c) {
                                   return c.has_text_embedder();
                               })
        .def_property_readonly("text_embedder_info",
                               [](const elips::Config& c) -> py::object {
                                   const auto info = c.text_embedder_info();
                                   if (!info.has_value()) {
                                       return py::none();
                                   }
                                   return py::cast(*info);
                               })
        .def("__repr__", [](const elips::Config& c) {
            return "<Config dimension=" + std::to_string(c.dimension()) +
                   " metric=" +
                   std::string(elips::to_string(c.metric())) +
                   " index=" +
                   (c.index() == elips::IndexType::graph ? "graph"
                                                           : "exact") +
                   ">";
        });

#ifdef ELIPS_GPU_ENABLED
    // =====================  GpuError  =====================

    py::enum_<elips::gpu::GpuError>(m, "GpuError")
        .value("device_not_found", elips::gpu::GpuError::DeviceNotFound)
        .value("insufficient_memory", elips::gpu::GpuError::InsufficientMemory)
        .value("kernel_launch_failed", elips::gpu::GpuError::KernelLaunchFailed)
        .value("transfer_failed", elips::gpu::GpuError::TransferFailed)
        .value("index_build_failed", elips::gpu::GpuError::IndexBuildFailed)
        .value("unsupported_metric", elips::gpu::GpuError::UnsupportedMetric)
        .value("initialization_failed", elips::gpu::GpuError::InitializationFailed)
        .value("backend_unavailable", elips::gpu::GpuError::BackendUnavailable);

    // =====================  GpuConfig  =====================

    py::enum_<elips::gpu::GpuPolicy>(m, "GpuPolicy")
        .value("auto", elips::gpu::GpuPolicy::Auto)
        .value("prefer_gpu", elips::gpu::GpuPolicy::PreferGpu)
        .value("require_gpu", elips::gpu::GpuPolicy::RequireGpu)
        .value("cpu_only", elips::gpu::GpuPolicy::CpuOnly)
        .value("specific", elips::gpu::GpuPolicy::Specific);

    py::enum_<elips::gpu::IndexBuildMode>(m, "IndexBuildMode")
        .value("gpu_build_cpu_serve", elips::gpu::IndexBuildMode::GpuBuild_CpuServe)
        .value("gpu_build_gpu_serve", elips::gpu::IndexBuildMode::GpuBuild_GpuServe)
        .value("hybrid", elips::gpu::IndexBuildMode::Hybrid);

    py::enum_<elips::gpu::GpuIndexAlgorithm>(m, "GpuIndexAlgorithm")
        .value("auto", elips::gpu::GpuIndexAlgorithm::Auto)
        .value("cagra", elips::gpu::GpuIndexAlgorithm::CagraGraph)
        .value("ivf_flat", elips::gpu::GpuIndexAlgorithm::IvfFlat)
        .value("ivf_pq", elips::gpu::GpuIndexAlgorithm::IvfPq)
        .value("brute_force", elips::gpu::GpuIndexAlgorithm::BruteForce);

    py::enum_<elips::gpu::GpuPrecision>(m, "GpuPrecision")
        .value("fp32", elips::gpu::GpuPrecision::FP32)
        .value("fp16", elips::gpu::GpuPrecision::FP16)
        .value("int8", elips::gpu::GpuPrecision::Int8)
        .value("auto", elips::gpu::GpuPrecision::Auto);

    // =====================  GPU build params structures  =====================

    py::enum_<elips::gpu::GraphIndexBuildParams::BuildAlgo>(m, "GraphBuildAlgo")
        .value("ivf_pq", elips::gpu::GraphIndexBuildParams::BuildAlgo::IvfPq)
        .value("nn_descent", elips::gpu::GraphIndexBuildParams::BuildAlgo::NnDescent)
        .value("iterative_search", elips::gpu::GraphIndexBuildParams::BuildAlgo::IterativeSearch);

    py::class_<elips::gpu::GraphIndexBuildParams>(m, "GraphIndexBuildParams")
        .def(py::init<>())
        .def_readwrite("intermediate_graph_degree", &elips::gpu::GraphIndexBuildParams::intermediate_graph_degree)
        .def_readwrite("graph_degree", &elips::gpu::GraphIndexBuildParams::graph_degree)
        .def_readwrite("build_algo", &elips::gpu::GraphIndexBuildParams::build_algo)
        .def_readwrite("nn_descent_iterations", &elips::gpu::GraphIndexBuildParams::nn_descent_iterations)
        .def_readwrite("compression_ratio", &elips::gpu::GraphIndexBuildParams::compression_ratio)
        .def("__repr__", [](const elips::gpu::GraphIndexBuildParams& p) {
            return "<GraphIndexBuildParams degree=" + std::to_string(p.graph_degree) + ">";
        });

    py::class_<elips::gpu::IvfPqBuildParams>(m, "IvfPqBuildParams")
        .def(py::init<>())
        .def_readwrite("n_lists", &elips::gpu::IvfPqBuildParams::n_lists)
        .def_readwrite("pq_dim", &elips::gpu::IvfPqBuildParams::pq_dim)
        .def_readwrite("pq_bits", &elips::gpu::IvfPqBuildParams::pq_bits)
        .def_readwrite("add_data_on_build", &elips::gpu::IvfPqBuildParams::add_data_on_build)
        .def_readwrite("kmeans_n_iters", &elips::gpu::IvfPqBuildParams::kmeans_n_iters)
        .def_readwrite("kmeans_trainset_fraction", &elips::gpu::IvfPqBuildParams::kmeans_trainset_fraction)
        .def("__repr__", [](const elips::gpu::IvfPqBuildParams& p) {
            return "<IvfPqBuildParams n_lists=" + std::to_string(p.n_lists) +
                   " pq_dim=" + std::to_string(p.pq_dim) + ">";
        });

    py::class_<elips::gpu::GpuIndexBuildParams>(m, "GpuIndexBuildParams")
        .def(py::init<>())
        .def_readwrite("params", &elips::gpu::GpuIndexBuildParams::params)
        .def("__repr__", [](const elips::gpu::GpuIndexBuildParams&) {
            return "<GpuIndexBuildParams>";
        });

    // =====================  KernelTiming  =====================

    py::class_<elips::gpu::KernelTiming>(m, "KernelTiming")
        .def(py::init<>())
        .def_readonly("kernel_name", &elips::gpu::KernelTiming::kernel_name)
        .def_readonly("work_items", &elips::gpu::KernelTiming::work_items)
        .def_property_readonly("duration_us", [](const elips::gpu::KernelTiming& t) {
            return std::chrono::duration_cast<std::chrono::microseconds>(t.duration).count();
        })
        .def("__repr__", [](const elips::gpu::KernelTiming& t) {
            return "<KernelTiming name='" + t.kernel_name +
                   "' items=" + std::to_string(t.work_items) + ">";
        });

    py::class_<elips::gpu::GpuConfig>(m, "GpuConfig")
        .def(py::init<>())
        .def_readwrite("policy", &elips::gpu::GpuConfig::policy)
        .def_readwrite("preferred_backend", &elips::gpu::GpuConfig::preferred_backend)
        .def_readwrite("device_index", &elips::gpu::GpuConfig::device_index)
        .def_readwrite("build_mode", &elips::gpu::GpuConfig::index_build_mode)
        .def_readwrite("algorithm", &elips::gpu::GpuConfig::algorithm)
        .def_property("device_memory_pool_mb",
             [](const elips::gpu::GpuConfig& c) -> size_t {
                 return c.device_memory_pool_bytes / (1024 * 1024);
             },
             [](elips::gpu::GpuConfig& c, size_t mb) {
                 c.device_memory_pool_bytes = mb * 1024 * 1024;
             })
        .def_property("pinned_host_pool_mb",
             [](const elips::gpu::GpuConfig& c) -> size_t {
                 return c.pinned_host_pool_bytes / (1024 * 1024);
             },
             [](elips::gpu::GpuConfig& c, size_t mb) {
                 c.pinned_host_pool_bytes = mb * 1024 * 1024;
             })
        .def_readwrite("fp16_search", &elips::gpu::GpuConfig::enable_fp16_search)
        .def_readwrite("unified_memory", &elips::gpu::GpuConfig::use_unified_memory)
        .def_readwrite("batch_window_us", &elips::gpu::GpuConfig::dynamic_batch_window_us)
        .def_readwrite("max_batch_size", &elips::gpu::GpuConfig::dynamic_batch_max_size)
        .def_readwrite("ef_search", &elips::gpu::GpuConfig::default_ef_search_gpu)
        .def_readwrite("precision", &elips::gpu::GpuConfig::search_precision)
        .def_readwrite("profiling", &elips::gpu::GpuConfig::enable_profiling)
        .def_readwrite("auto_rebuild_on_startup", &elips::gpu::GpuConfig::auto_rebuild_on_startup)
        .def_readwrite("rebuild_threshold_ratio", &elips::gpu::GpuConfig::rebuild_threshold_ratio)
        .def_readwrite("emit_kernel_timings", &elips::gpu::GpuConfig::emit_kernel_timings)
        .def_readwrite("graph_params", &elips::gpu::GpuConfig::graph_params)
        .def_readwrite("ivf_pq_params", &elips::gpu::GpuConfig::ivf_pq_params)
        .def("__repr__", [](const elips::gpu::GpuConfig& c) {
            return "<GpuConfig policy=" + std::string(gpu_policy_name(c.policy)) +
                   " algorithm=" + std::string(gpu_algorithm_name(c.algorithm)) +
                   " build_mode=" + std::string(gpu_build_mode_name(c.index_build_mode)) +
                   " device_index=" + std::to_string(c.device_index) + ">";
        });

    py::class_<elips::gpu::GpuDeviceInfo>(m, "GpuDeviceInfo")
        .def(py::init([]() {
            elips::gpu::GpuDeviceManager manager;
            return manager.runtime_device_info();
        }))
        .def_readonly("name", &elips::gpu::GpuDeviceInfo::name)
        .def_readonly("vendor", &elips::gpu::GpuDeviceInfo::vendor)
        .def_readonly("backend", &elips::gpu::GpuDeviceInfo::backend)
        .def_readonly("total_memory_bytes", &elips::gpu::GpuDeviceInfo::total_device_memory_bytes)
        .def_readonly("free_memory_bytes", &elips::gpu::GpuDeviceInfo::free_device_memory_bytes)
        .def_readonly("has_unified_memory", &elips::gpu::GpuDeviceInfo::has_unified_memory)
        .def_readonly("supports_fp16", &elips::gpu::GpuDeviceInfo::supports_fp16)
        .def_readonly("supports_cagra", &elips::gpu::GpuDeviceInfo::supports_cagra)
        .def_readonly("supports_ivf_pq", &elips::gpu::GpuDeviceInfo::supports_ivf_pq)
        .def_readonly("device_index", &elips::gpu::GpuDeviceInfo::device_index)
        .def_readonly("supports_bf16", &elips::gpu::GpuDeviceInfo::supports_bf16)
        .def_readonly("supports_int8", &elips::gpu::GpuDeviceInfo::supports_int8)
        .def_readonly("compute_capability_major", &elips::gpu::GpuDeviceInfo::compute_capability_major)
        .def_readonly("compute_capability_minor", &elips::gpu::GpuDeviceInfo::compute_capability_minor)
        .def_readonly("max_threads_per_block", &elips::gpu::GpuDeviceInfo::max_threads_per_block)
        .def_readonly("multiprocessor_count", &elips::gpu::GpuDeviceInfo::multiprocessor_count)
        .def_readonly("shared_memory_per_block_bytes", &elips::gpu::GpuDeviceInfo::shared_memory_per_block_bytes)
        .def_readonly("l2_cache_bytes", &elips::gpu::GpuDeviceInfo::l2_cache_bytes)
        .def_readonly("peak_tflops_fp32", &elips::gpu::GpuDeviceInfo::peak_tflops_fp32)
        .def_readonly("peak_tflops_fp16", &elips::gpu::GpuDeviceInfo::peak_tflops_fp16)
        .def_readonly("host_to_device_bandwidth_gb_s", &elips::gpu::GpuDeviceInfo::host_to_device_bandwidth_gb_s)
        .def_readonly("device_to_host_bandwidth_gb_s", &elips::gpu::GpuDeviceInfo::device_to_host_bandwidth_gb_s)
        .def_readonly("supports_dynamic_batching", &elips::gpu::GpuDeviceInfo::supports_dynamic_batching)
        .def_readonly("supports_half_precision_search", &elips::gpu::GpuDeviceInfo::supports_half_precision_search)
        .def_property_readonly("memory_gb", [](const elips::gpu::GpuDeviceInfo& i) {
            return static_cast<double>(i.total_device_memory_bytes) / (1024.0 * 1024.0 * 1024.0);
        })
        .def("__repr__", [](const elips::gpu::GpuDeviceInfo& i) {
            return "<GpuDeviceInfo name='" + i.name + "' backend=" + i.backend +
                   " device_index=" + std::to_string(i.device_index) + ">";
        });

    m.def("gpu_devices", []() {
        elips::gpu::GpuDeviceManager manager;
        return manager.probe_all_devices();
    },
    R"doc(Probe every GPU backend compiled into this build.

Returns:
    A list of :class:`GpuDeviceInfo`, one per usable device. Empty when no
    compatible device is present, which is the normal case on CPU-only
    machines -- callers should treat an empty list as "run on the CPU"
    rather than an error.
)doc");

    m.def("gpu_cpu_fallback_info", []() {
        elips::gpu::GpuDeviceManager manager;
        return manager.cpu_fallback_info();
    },
    "Return the synthetic :class:`GpuDeviceInfo` describing the CPU fallback "
    "path, used when no GPU is selected.");

    m.def("gpu_runtime_device_info", []() {
        elips::gpu::GpuDeviceManager manager;
        return manager.runtime_device_info();
    },
    "Return :class:`GpuDeviceInfo` for the device this process would select "
    "right now, or the CPU fallback when there is none.");

    m.def("gpu_can_fit_index",
          [](const elips::gpu::GpuDeviceInfo& device, std::size_t n_vectors,
             std::size_t dimension, const elips::gpu::GpuConfig& config) {
              elips::gpu::GpuDeviceManager manager;
              return manager.can_fit_index(device, n_vectors, dimension, config);
          },
          py::arg("device"), py::arg("n_vectors"), py::arg("dimension"),
          py::arg("config") = elips::gpu::GpuConfig{},
          R"doc(Check whether an index of the given shape fits in device memory.

Args:
    device: Target device, from :func:`gpu_devices`.
    n_vectors: Number of vectors the index will hold.
    dimension: Vector dimensionality.
    config: GPU configuration; its precision and pool settings affect the
        estimate.

Returns:
    True when the index is expected to fit. Use this for capacity planning
    before a large ingest, rather than discovering the limit mid-build.
)doc");

    m.def("gpu_error_message",
          [](elips::gpu::GpuError error) {
              return std::string(elips::gpu::to_string(error));
          },
          py::arg("error"),
          "Return the human-readable name of a :class:`GpuError` value.");

    // =====================  GPU memory / profiling  =====================

    py::class_<elips::gpu::GpuMemoryManager,
               std::unique_ptr<elips::gpu::GpuMemoryManager, py::nodelete>>(
        m, "GpuMemory",
        "Read-only view of a device memory pool. Obtained from "
        ":attr:`GpuDevice.memory`; allocation itself stays in C++, since a "
        "Python-held device pointer outliving its pool is unrecoverable.")
        .def("initialize",
             [](elips::gpu::GpuMemoryManager& mem, std::size_t pool_bytes) {
                 const auto result = mem.initialize(pool_bytes);
                 if (!result.has_value()) {
                     raise_gpu_error(result.error(), "pool initialization");
                 }
             },
             py::arg("pool_bytes") = 0,
             "Size the suballocator's pool. ``0`` uses 80% of device memory.")
        .def_property_readonly("bytes_used",
                               &elips::gpu::GpuMemoryManager::bytes_used,
                               "Bytes currently handed out to callers.")
        .def_property_readonly("bytes_available",
                               &elips::gpu::GpuMemoryManager::bytes_available,
                               "Bytes a caller could still obtain: the free "
                               "list plus uncommitted pool headroom.")
        .def_property_readonly("peak_bytes_used",
                               &elips::gpu::GpuMemoryManager::peak_bytes_used,
                               "High-water mark of :attr:`bytes_used`.")
        .def("__repr__", [](const elips::gpu::GpuMemoryManager& mem) {
            return "<GpuMemory used=" + std::to_string(mem.bytes_used()) +
                   " available=" + std::to_string(mem.bytes_available()) + ">";
        });

    py::class_<elips::gpu::GpuProfiler,
               std::unique_ptr<elips::gpu::GpuProfiler, py::nodelete>>(
        m, "GpuProfiler",
        "Per-kernel timing log. Obtained from :attr:`GpuDevice.profiler`.")
        .def("record",
             [](elips::gpu::GpuProfiler& profiler, const std::string& kernel,
                std::int64_t duration_us, std::size_t work_items) {
                 profiler.record(kernel,
                                 std::chrono::microseconds(duration_us),
                                 work_items);
             },
             py::arg("kernel"), py::arg("duration_us"),
             py::arg("work_items") = 0,
             "Record a kernel execution. Exposed so callers can log their own "
             "GPU work alongside the engine's.")
        .def("recent_timings", &elips::gpu::GpuProfiler::recent_timings,
             py::arg("max_count") = 100,
             "Return up to ``max_count`` recent :class:`KernelTiming` entries, "
             "newest last.")
        .def_property_readonly("total_launches",
                               &elips::gpu::GpuProfiler::total_launches,
                               "Total kernel launches recorded.")
        .def("clear", &elips::gpu::GpuProfiler::clear,
             "Discard all recorded timings.")
        .def("__repr__", [](const elips::gpu::GpuProfiler& profiler) {
            return "<GpuProfiler launches=" +
                   std::to_string(profiler.total_launches()) + ">";
        });

    py::class_<elips::gpu::DynamicBatcher::BatchStats>(
        m, "BatchStats",
        "Coalescing statistics from the dynamic query batcher.")
        .def(py::init<>())
        .def_readonly("queries_coalesced",
                      &elips::gpu::DynamicBatcher::BatchStats::queries_coalesced,
                      "Queries merged into shared kernel launches.")
        .def_readonly("kernel_launches",
                      &elips::gpu::DynamicBatcher::BatchStats::kernel_launches,
                      "Kernel launches issued.")
        .def_readonly("avg_batch_size",
                      &elips::gpu::DynamicBatcher::BatchStats::avg_batch_size,
                      "Mean queries per launch.")
        .def_readonly("p99_latency_us",
                      &elips::gpu::DynamicBatcher::BatchStats::p99_latency_us,
                      "99th-percentile end-to-end batch latency.")
        .def("__repr__",
             [](const elips::gpu::DynamicBatcher::BatchStats& stats) {
                 return "<BatchStats coalesced=" +
                        std::to_string(stats.queries_coalesced) +
                        " launches=" + std::to_string(stats.kernel_launches) +
                        ">";
             });

    // =====================  GpuDevice  =====================

    py::class_<GpuDeviceHandle>(m, "GpuDevice", R"doc(A live handle to a selected GPU backend.

Obtain one with :func:`gpu_select`. The handle owns the backend, so keep it
alive for as long as you use it, and call :meth:`close` (or use it as a
context manager) when done::

    with elips.gpu_select(cfg) as device:
        dists = device.compute_distances(queries, database, metric="cosine")
        idx, vals = device.top_k(dists, k=10)

Raw device allocation, upload, and download are intentionally not exposed:
they take a caller-supplied byte count against a raw pointer, where a wrong
value corrupts device memory instead of raising. The kernels below derive
their sizes from the arrays you pass in.
)doc")
        .def_property_readonly(
            "device_info",
            [](GpuDeviceHandle& handle) { return handle.port().device_info(); },
            "The :class:`GpuDeviceInfo` for this backend.")
        .def_property_readonly(
            "available",
            [](GpuDeviceHandle& handle) { return handle.port().is_available(); },
            "True while the backend is usable.")
        .def_property_readonly(
            "idle",
            [](GpuDeviceHandle& handle) { return handle.port().is_idle(); },
            "True when no work is outstanding on the device.")
        .def_property_readonly(
            "backend",
            [](GpuDeviceHandle& handle) {
                return handle.port().device_info().backend;
            },
            "Backend name, e.g. ``\"metal\"`` or ``\"cuda\"``.")
        .def_property_readonly(
            "memory",
            [](GpuDeviceHandle& handle) { return &handle.memory(); },
            py::return_value_policy::reference_internal,
            "The :class:`GpuMemory` telemetry view for this device.")
        .def_property_readonly(
            "profiler",
            [](GpuDeviceHandle& handle) { return &handle.profiler(); },
            py::return_value_policy::reference_internal,
            "The :class:`GpuProfiler` for this device.")
        .def("synchronize",
             [](GpuDeviceHandle& handle) {
                 py::gil_scoped_release release;
                 handle.port().synchronize();
             },
             "Block until all outstanding device work completes.")
        .def("compute_distances",
             [](GpuDeviceHandle& handle, const py::iterable& queries,
                const py::iterable& database, const py::object& metric_arg) {
                 elips::Metric metric = elips::Metric::cosine;
                 if (py::isinstance<py::str>(metric_arg)) {
                     metric = elips::metric_from_string(
                         metric_arg.cast<std::string>());
                 } else {
                     metric = metric_arg.cast<elips::Metric>();
                 }

                 std::size_t nq = 0;
                 std::size_t nb = 0;
                 auto query_flat = flatten_matrix(queries, 0, nq, "queries");
                 auto db_flat = flatten_matrix(database, 0, nb, "database");
                 if (nq == 0 || nb == 0) {
                     throw py::value_error(
                         "queries and database must both be non-empty");
                 }
                 const std::size_t dim = query_flat.size() / nq;
                 if (db_flat.size() / nb != dim) {
                     throw elips::DimensionMismatch{
                         "queries and database have different dimensions"};
                 }

                 std::vector<float> out(nq * nb);
                 {
                     py::gil_scoped_release release;
                     const auto result = handle.port().compute_distances_batch(
                         query_flat, db_flat, out, nq, nb, dim, metric);
                     if (!result.has_value()) {
                         py::gil_scoped_acquire acquire;
                         raise_gpu_error(result.error(), "distance computation");
                     }
                 }

                 py::list rows;
                 for (std::size_t q = 0; q < nq; ++q) {
                     py::list row;
                     for (std::size_t b = 0; b < nb; ++b) {
                         row.append(py::float_(out[(q * nb) + b]));
                     }
                     rows.append(row);
                 }
                 return rows;
             },
             py::arg("queries"), py::arg("database"),
             py::arg("metric") = "cosine",
             R"doc(Compute all pairwise distances between two batches on the GPU.

Args:
    queries: Sequence of query vectors, all the same length.
    database: Sequence of database vectors, same dimension as ``queries``.
    metric: A :class:`Metric` or one of ``"cosine"``, ``"euclidean"``,
        ``"dot_product"``.

Returns:
    A list of ``len(queries)`` rows, each holding ``len(database)``
    distances, ordering-normalized so smaller always means closer.

Raises:
    DimensionMismatch: If query and database dimensions differ.
    StorageError: If the GPU kernel fails.
)doc")
        .def("top_k",
             [](GpuDeviceHandle& handle, const py::iterable& distances,
                std::size_t k) {
                 std::size_t nq = 0;
                 auto flat = flatten_matrix(distances, 0, nq, "distances");
                 if (nq == 0 || k == 0) {
                     return py::make_tuple(py::list{}, py::list{});
                 }
                 const std::size_t nb = flat.size() / nq;
                 if (k > nb) {
                     throw py::value_error(
                         "k exceeds the number of database entries per row");
                 }

                 std::vector<std::uint32_t> indices(nq * k);
                 std::vector<float> values(nq * k);
                 {
                     py::gil_scoped_release release;
                     const auto result = handle.port().top_k(flat, indices,
                                                             values, nq, nb, k);
                     if (!result.has_value()) {
                         py::gil_scoped_acquire acquire;
                         raise_gpu_error(result.error(), "top-k selection");
                     }
                 }

                 py::list index_rows;
                 py::list value_rows;
                 for (std::size_t q = 0; q < nq; ++q) {
                     py::list index_row;
                     py::list value_row;
                     for (std::size_t i = 0; i < k; ++i) {
                         index_row.append(py::int_(indices[(q * k) + i]));
                         value_row.append(py::float_(values[(q * k) + i]));
                     }
                     index_rows.append(index_row);
                     value_rows.append(value_row);
                 }
                 return py::make_tuple(index_rows, value_rows);
             },
             py::arg("distances"), py::arg("k"),
             R"doc(Select the k smallest entries per row on the GPU.

Args:
    distances: Rows of distances, e.g. the output of
        :meth:`compute_distances`.
    k: Number of results per row. Must not exceed the row width.

Returns:
    A ``(indices, values)`` tuple of parallel row lists, ascending by
    distance.

Raises:
    ValueError: If ``k`` exceeds the row width.
    StorageError: If the GPU kernel fails.
)doc")
        .def("close", &GpuDeviceHandle::close,
             "Release the backend and its memory pool. Idempotent.")
        .def_property_readonly("closed", &GpuDeviceHandle::closed,
                               "True once :meth:`close` has run.")
        .def("__enter__", [](py::object self) { return self; })
        .def("__exit__",
             [](GpuDeviceHandle& handle, const py::object&, const py::object&,
                const py::object&) { handle.close(); })
        .def("__repr__", [](GpuDeviceHandle& handle) {
            if (handle.closed()) {
                return std::string("<GpuDevice closed>");
            }
            const auto info = handle.port().device_info();
            return "<GpuDevice name='" + info.name + "' backend=" +
                   info.backend + ">";
        });

    m.def("gpu_select",
          [](const elips::gpu::GpuConfig& config) -> std::unique_ptr<GpuDeviceHandle> {
              elips::gpu::GpuDeviceManager manager;
              const auto devices = manager.probe_all_devices();
              if (devices.empty()) {
                  return nullptr;
              }
              auto selected = manager.select(config, devices);
              if (!selected.has_value() || *selected == nullptr) {
                  return nullptr;
              }
              return std::make_unique<GpuDeviceHandle>(std::move(*selected));
          },
          py::arg("config") = elips::gpu::GpuConfig{},
          R"doc(Select and initialize a GPU backend.

Args:
    config: Selection policy and tuning. The default picks the best
        available device.

Returns:
    A :class:`GpuDevice` handle, or ``None`` when no compatible device is
    present. ``None`` is the expected result on CPU-only machines; check
    for it rather than assuming a device exists.

The returned handle owns the backend independently of any
:class:`Database`. Close it when done, or use it as a context manager.
)doc");

    py::class_<elips::gpu::GpuMetricsSnapshot>(m, "GpuMetricsSnapshot")
        .def(py::init<>())
        .def_readonly("backend", &elips::gpu::GpuMetricsSnapshot::backend)
        .def_readonly("device_name", &elips::gpu::GpuMetricsSnapshot::device_name)
        .def_readonly("device_memory_used_bytes", &elips::gpu::GpuMetricsSnapshot::device_memory_used_bytes)
        .def_readonly("device_memory_total_bytes", &elips::gpu::GpuMetricsSnapshot::device_memory_total_bytes)
        .def_readonly("index_build_count", &elips::gpu::GpuMetricsSnapshot::index_build_count)
        .def_readonly("index_build_time_total_ms", &elips::gpu::GpuMetricsSnapshot::index_build_time_total_ms)
        .def_readonly("index_build_speedup_vs_cpu_avg", &elips::gpu::GpuMetricsSnapshot::index_build_speedup_vs_cpu_avg)
        .def_readonly("search_kernel_launches_total", &elips::gpu::GpuMetricsSnapshot::search_kernel_launches_total)
        .def_readonly("search_p50_latency_us", &elips::gpu::GpuMetricsSnapshot::search_p50_latency_us)
        .def_readonly("search_p99_latency_us", &elips::gpu::GpuMetricsSnapshot::search_p99_latency_us)
        .def_readonly("batch_avg_size", &elips::gpu::GpuMetricsSnapshot::batch_avg_size)
        .def_readonly("batch_coalescing_ratio", &elips::gpu::GpuMetricsSnapshot::batch_coalescing_ratio)
        .def_readonly("fp16_search_enabled", &elips::gpu::GpuMetricsSnapshot::fp16_search_enabled)
        .def_readonly("fallback_events_total", &elips::gpu::GpuMetricsSnapshot::fallback_events_total)
        .def_readonly("kernel_errors_total", &elips::gpu::GpuMetricsSnapshot::kernel_errors_total)
        .def_readonly("pinned_memory_pool_used_bytes", &elips::gpu::GpuMetricsSnapshot::pinned_memory_pool_used_bytes)
        .def("__repr__", [](const elips::gpu::GpuMetricsSnapshot& s) {
            return "<GpuMetricsSnapshot backend=" + s.backend + ">";
        });
#else
    // ===================== CPU Fallback Surface for GPU Control =====================
    enum class CpuGpuError {
        device_not_found,
        insufficient_memory,
        kernel_launch_failed,
        transfer_failed,
        index_build_failed,
        unsupported_metric,
        initialization_failed,
        backend_unavailable
    };

    py::enum_<CpuGpuError>(m, "GpuError")
        .value("device_not_found", CpuGpuError::device_not_found)
        .value("insufficient_memory", CpuGpuError::insufficient_memory)
        .value("kernel_launch_failed", CpuGpuError::kernel_launch_failed)
        .value("transfer_failed", CpuGpuError::transfer_failed)
        .value("index_build_failed", CpuGpuError::index_build_failed)
        .value("unsupported_metric", CpuGpuError::unsupported_metric)
        .value("initialization_failed", CpuGpuError::initialization_failed)
        .value("backend_unavailable", CpuGpuError::backend_unavailable);

    py::class_<elips::gpu::GpuDeviceInfo>(m, "GpuDeviceInfo")
        .def(py::init([]() {
            elips::gpu::GpuDeviceInfo info;
            info.name = "CPU Fallback Engine";
            info.vendor = "Software";
            info.backend = "cpu";
            return info;
        }))
        .def_readonly("name", &elips::gpu::GpuDeviceInfo::name)
        .def_readonly("vendor", &elips::gpu::GpuDeviceInfo::vendor)
        .def_readonly("backend", &elips::gpu::GpuDeviceInfo::backend)
        .def_readonly("total_memory_bytes", &elips::gpu::GpuDeviceInfo::total_device_memory_bytes)
        .def_readonly("free_memory_bytes", &elips::gpu::GpuDeviceInfo::free_device_memory_bytes)
        .def_readonly("has_unified_memory", &elips::gpu::GpuDeviceInfo::has_unified_memory)
        .def_readonly("supports_fp16", &elips::gpu::GpuDeviceInfo::supports_fp16)
        .def_readonly("supports_cagra", &elips::gpu::GpuDeviceInfo::supports_cagra)
        .def_readonly("supports_ivf_pq", &elips::gpu::GpuDeviceInfo::supports_ivf_pq)
        .def_readonly("device_index", &elips::gpu::GpuDeviceInfo::device_index)
        .def_readonly("supports_bf16", &elips::gpu::GpuDeviceInfo::supports_bf16)
        .def_readonly("supports_int8", &elips::gpu::GpuDeviceInfo::supports_int8)
        .def_readonly("compute_capability_major", &elips::gpu::GpuDeviceInfo::compute_capability_major)
        .def_readonly("compute_capability_minor", &elips::gpu::GpuDeviceInfo::compute_capability_minor)
        .def_readonly("max_threads_per_block", &elips::gpu::GpuDeviceInfo::max_threads_per_block)
        .def_readonly("multiprocessor_count", &elips::gpu::GpuDeviceInfo::multiprocessor_count)
        .def_readonly("shared_memory_per_block_bytes", &elips::gpu::GpuDeviceInfo::shared_memory_per_block_bytes)
        .def_readonly("l2_cache_bytes", &elips::gpu::GpuDeviceInfo::l2_cache_bytes)
        .def_readonly("peak_tflops_fp32", &elips::gpu::GpuDeviceInfo::peak_tflops_fp32)
        .def_readonly("peak_tflops_fp16", &elips::gpu::GpuDeviceInfo::peak_tflops_fp16)
        .def_readonly("host_to_device_bandwidth_gb_s", &elips::gpu::GpuDeviceInfo::host_to_device_bandwidth_gb_s)
        .def_readonly("device_to_host_bandwidth_gb_s", &elips::gpu::GpuDeviceInfo::device_to_host_bandwidth_gb_s)
        .def_readonly("supports_dynamic_batching", &elips::gpu::GpuDeviceInfo::supports_dynamic_batching)
        .def_readonly("supports_half_precision_search", &elips::gpu::GpuDeviceInfo::supports_half_precision_search)
        .def_property_readonly("memory_gb", [](const elips::gpu::GpuDeviceInfo& i) {
            return static_cast<double>(i.total_device_memory_bytes) / (1024.0 * 1024.0 * 1024.0);
        })
        .def("__repr__", [](const elips::gpu::GpuDeviceInfo& i) {
            return "<GpuDeviceInfo name='" + i.name + "' backend=" + i.backend +
                   " device_index=" + std::to_string(i.device_index) + ">";
        });

    struct CpuBatchStats {
        size_t queries_coalesced{0};
        size_t kernel_launches{0};
        double avg_batch_size{0.0};
        double p99_latency_us{0.0};
    };

    py::class_<CpuBatchStats>(m, "BatchStats")
        .def(py::init<>())
        .def_readonly("queries_coalesced", &CpuBatchStats::queries_coalesced)
        .def_readonly("kernel_launches", &CpuBatchStats::kernel_launches)
        .def_readonly("avg_batch_size", &CpuBatchStats::avg_batch_size)
        .def_readonly("p99_latency_us", &CpuBatchStats::p99_latency_us)
        .def("__repr__", [](const CpuBatchStats&) {
            return "<BatchStats coalesced=0 launches=0>";
        });

    m.def("gpu_devices", []() {
        return std::vector<elips::gpu::GpuDeviceInfo>{};
    });

    m.def("gpu_cpu_fallback_info", []() {
        elips::gpu::GpuDeviceInfo info;
        info.name = "CPU Fallback Engine";
        info.vendor = "Software";
        info.backend = "cpu";
        return info;
    });

    m.def("gpu_runtime_device_info", []() {
        elips::gpu::GpuDeviceInfo info;
        info.name = "CPU Engine";
        info.vendor = "Software";
        info.backend = "cpu";
        return info;
    });

    m.def("gpu_error_message", [](CpuGpuError err) {
        switch (err) {
            case CpuGpuError::device_not_found: return "device not found";
            case CpuGpuError::insufficient_memory: return "insufficient memory";
            case CpuGpuError::kernel_launch_failed: return "kernel launch failed";
            case CpuGpuError::transfer_failed: return "transfer failed";
            case CpuGpuError::index_build_failed: return "index build failed";
            case CpuGpuError::unsupported_metric: return "unsupported metric";
            case CpuGpuError::initialization_failed: return "initialization failed";
            case CpuGpuError::backend_unavailable: return "backend unavailable";
        }
        return "unknown gpu error";
    });

    m.def("gpu_can_fit_index", [](const elips::gpu::GpuDeviceInfo&, size_t, size_t) {
        return false;
    });

    m.def("gpu_select", [](py::object) {
        return py::none();
    }, py::arg("config") = py::none());
#endif

    py::class_<elips::DocumentAttachment>(m, "DocumentAttachment")
        .def(py::init<>())
        .def(py::init<std::string, std::string, std::string>(),
             py::arg("text"), py::arg("uri") = "",
             py::arg("mime_type") = "text/plain")
        .def_readwrite("text", &elips::DocumentAttachment::text)
        .def_readwrite("uri", &elips::DocumentAttachment::uri)
        .def_readwrite("mime_type", &elips::DocumentAttachment::mime_type)
        .def("__repr__", [](const elips::DocumentAttachment& d) {
            return "<DocumentAttachment mime_type='" + d.mime_type + "'>";
        });

    py::class_<elips::ChunkInfo>(m, "ChunkInfo")
        .def(py::init<>())
        .def_readwrite("document_key", &elips::ChunkInfo::document_key)
        .def_readwrite("ordinal", &elips::ChunkInfo::ordinal)
        .def_readwrite("char_start", &elips::ChunkInfo::char_start)
        .def_readwrite("char_end", &elips::ChunkInfo::char_end)
        .def("__repr__", [](const elips::ChunkInfo& chunk) {
            return "<ChunkInfo key='" + chunk.document_key +
                   "' ordinal=" + std::to_string(chunk.ordinal) + ">";
        });

    py::class_<elips::EmbeddingLineage>(m, "EmbeddingLineage")
        .def(py::init<>())
        .def_readwrite("provider", &elips::EmbeddingLineage::provider)
        .def_readwrite("model", &elips::EmbeddingLineage::model)
        .def_readwrite("revision", &elips::EmbeddingLineage::revision)
        .def_property(
            "attributes",
            [](const elips::EmbeddingLineage& lineage) {
                return from_payload(lineage.attributes);
            },
            [](elips::EmbeddingLineage& lineage, const py::dict& attrs) {
                lineage.attributes = to_payload(attrs);
            })
        .def("__repr__", [](const elips::EmbeddingLineage& lineage) {
            return "<EmbeddingLineage provider='" + lineage.provider +
                   "' model='" + lineage.model + "'>";
        });

    py::class_<elips::QueryPlan>(m, "QueryPlan")
        .def(py::init<>())
        .def_readonly("strategy", &elips::QueryPlan::strategy)
        .def_readonly("candidate_count", &elips::QueryPlan::candidate_count)
        .def_readonly("metadata_accelerated",
                      &elips::QueryPlan::metadata_accelerated)
        .def_readonly("gpu_index", &elips::QueryPlan::gpu_index)
        .def_readonly("index_type", &elips::QueryPlan::index_type)
        .def("__repr__", [](const elips::QueryPlan& plan) {
            return "<QueryPlan candidates=" +
                   std::to_string(plan.candidate_count) + " index='" +
                   plan.index_type + "'>";
        });

    // =====================  VaultInfo  =====================

    py::class_<elips::VaultInfo>(m, "VaultInfo")
        .def_property_readonly(
            "count", [](const elips::VaultInfo& vi) { return vi.count; })
        .def_property_readonly("dimension", [](const elips::VaultInfo& vi) {
            return vi.dimension;
        })
        .def_property_readonly("metric", [](const elips::VaultInfo& vi) {
            return std::string(elips::to_string(vi.metric));
        })
        .def_property_readonly("codec", [](const elips::VaultInfo& vi) {
            return std::string(elips::quant::to_string(vi.codec));
        })
        .def_property_readonly(
            "code_bytes", [](const elips::VaultInfo& vi) { return vi.code_bytes; },
            "Bytes per stored vector, or 0 when uncompressed.")
        .def_property_readonly(
            "compression_ratio",
            [](const elips::VaultInfo& vi) {
                if (vi.code_bytes == 0 || vi.dimension == 0) {
                    return 1.0;
                }
                return static_cast<double>(vi.dimension * sizeof(float)) /
                       static_cast<double>(vi.code_bytes);
            },
            "Stored bytes saved per vector against fp32; 1.0 when uncompressed.")
        .def("__repr__", [](const elips::VaultInfo& vi) {
            return "<VaultInfo count=" + std::to_string(vi.count) +
                   " dimension=" + std::to_string(vi.dimension) +
                   " metric=" +
                   std::string(elips::to_string(vi.metric)) +
                   " codec=" +
                   std::string(elips::quant::to_string(vi.codec)) + ">";
        });

    // =====================  SearchResult  =====================

    py::class_<elips::SearchResult>(m, "Result")
        .def_property_readonly(
            "id", [](const elips::SearchResult& r) { return r.id.to_string(); })
        .def_readonly("distance", &elips::SearchResult::distance)
        .def_property_readonly(
            "data", [](const elips::SearchResult& r) { return from_payload(r.data); })
        .def_readonly("document", &elips::SearchResult::document)
        .def_readonly("chunk", &elips::SearchResult::chunk)
        .def_readonly("lineage", &elips::SearchResult::lineage)
        .def_property_readonly(
            "approximate",
            [](const elips::SearchResult& r) { return r.approximate(); },
            "True when ``distance`` is estimated from a compressed vector "
            "rather than computed exactly.")
        .def_property_readonly(
            "codec",
            [](const elips::SearchResult& r) {
                return std::string(elips::quant::to_string(r.codec));
            },
            "Codec that produced this hit's stored vector, or ``none``.")
        .def("__repr__", [](const elips::SearchResult& r) {
            return "<Result id=" + r.id.to_string() +
                   " distance=" + std::to_string(r.distance) +
                   (r.approximate() ? " approximate" : "") + ">";
        });

    // =====================  Filter  =====================

    py::class_<elips::Filter>(m, "Filter", R"doc(A metadata predicate applied during search and scan.

Two styles, which compose:

* fluent builder -- ``Filter().field("year").ge(2023)``; chained predicates are
  AND-ed together
* static leaf factories -- :meth:`compare`, :meth:`in_set`,
  :meth:`has_substring` -- combined with :meth:`and_`, :meth:`or_`, :meth:`not_`

Equality and set predicates can be served straight from the metadata index
(see :meth:`exact_constraints`), which lets the planner skip the vector scan
entirely. Range and substring predicates are evaluated per candidate.

Example:
    Build a tenant-scoped filter, check it before issuing the query, and reuse
    it for client-side validation::

        import elips

        # 1. Hard tenant scope, AND-ed with a freshness window.
        scope = (elips.Filter()
                 .field("tenant").equals("acme")
                 .field("year").ge(2023))

        # 2. OR in a second branch: either tier is acceptable.
        tier = elips.Filter.in_set("tier", ["pro", "enterprise"])
        where = scope.and_(tier)

        # 3. Before running an expensive query, confirm the planner can use
        #    the metadata index rather than falling back to a full scan.
        if where.exact_constraints() is None:
            print("warning: this filter needs a scan")

        hits = db.vault("docs").seek(query_vector, top=10, where=where)

        # 4. The same filter object evaluates payloads locally -- handy for
        #    testing, or for filtering results that arrived from elsewhere.
        assert where.matches({"tenant": "acme", "year": 2024, "tier": "pro"})
        assert not where.matches({"tenant": "other", "year": 2024, "tier": "pro"})

        # 5. Negation wraps a whole subtree.
        exclude_drafts = elips.Filter.not_(
            elips.Filter().field("status").equals("draft"))
)doc")
        .def(py::init<>())
        .def("field", &elips::Filter::field,
             py::return_value_policy::reference_internal)
        .def("equals",
             [](elips::Filter& f, const py::handle& v) -> elips::Filter& {
                 return f.equals(to_meta(v));
             },
             py::return_value_policy::reference_internal)
        .def("not_equals",
             [](elips::Filter& f, const py::handle& v) -> elips::Filter& {
                 return f.not_equals(to_meta(v));
             },
             py::return_value_policy::reference_internal)
        .def("lt",
             [](elips::Filter& f, const py::handle& v) -> elips::Filter& {
                 return f.lt(to_meta(v));
             },
             py::return_value_policy::reference_internal)
        .def("le",
             [](elips::Filter& f, const py::handle& v) -> elips::Filter& {
                 return f.le(to_meta(v));
             },
             py::return_value_policy::reference_internal)
        .def("gt",
             [](elips::Filter& f, const py::handle& v) -> elips::Filter& {
                 return f.gt(to_meta(v));
             },
             py::return_value_policy::reference_internal)
        .def("gte",
             [](elips::Filter& f, const py::handle& v) -> elips::Filter& {
                 return f.ge(to_meta(v));
             },
             py::return_value_policy::reference_internal)
        .def("one_of",
             [](elips::Filter& f,
                const py::iterable& vs) -> elips::Filter& {
                 std::vector<elips::MetaValue> set;
                 for (const auto& v : vs) set.push_back(to_meta(v));
                 return f.one_of(std::move(set));
             },
             py::return_value_policy::reference_internal)
        .def("contains", &elips::Filter::contains,
             py::return_value_policy::reference_internal)
        .def("and_", &elips::Filter::and_)
        .def("or_", &elips::Filter::or_)
        .def_static("not_", &elips::Filter::not_)
        .def_static("compare",
                    [](std::string field, const py::object& op,
                       const py::handle& value) {
                        elips::Comparator comparator{};
                        if (py::isinstance<py::str>(op)) {
                            comparator = comparator_from_string(
                                op.cast<std::string>());
                        } else {
                            comparator = op.cast<elips::Comparator>();
                        }
                        return elips::Filter::compare(std::move(field),
                                                      comparator,
                                                      to_meta(value));
                    },
                    py::arg("field"), py::arg("op"), py::arg("value"),
                    R"doc(Build a single comparison predicate.

Args:
    field: Payload key to test.
    op: A :class:`Comparator` or one of ``"eq"``, ``"ne"``, ``"lt"``,
        ``"le"``, ``"gt"``, ``"ge"``.
    value: Value to compare against (``int``, ``float``, ``bool``, ``str``).

Returns:
    A :class:`Filter` matching records whose ``field`` satisfies ``op``.
)doc")
        .def_static("in_set",
                    [](std::string field, const py::iterable& values) {
                        std::vector<elips::MetaValue> set;
                        for (const auto& value : values) {
                            set.push_back(to_meta(value));
                        }
                        return elips::Filter::in_set(std::move(field),
                                                     std::move(set));
                    },
                    py::arg("field"), py::arg("values"),
                    "Build a predicate matching records whose ``field`` equals "
                    "any of ``values``.")
        .def_static("has_substring", &elips::Filter::has_substring,
                    py::arg("field"), py::arg("substring"),
                    "Build a predicate matching records whose string ``field`` "
                    "contains ``substring``.")
        .def("matches",
             [](const elips::Filter& f, const py::dict& payload) {
                 return f.matches(to_payload(payload));
             },
             py::arg("payload"),
             "Evaluate this filter against a payload dict, without touching "
             "the database. Useful for client-side filtering and tests.")
        .def("matches_all", &elips::Filter::matches_all,
             "True when this filter is empty and therefore matches every "
             "record.")
        .def("exact_constraints",
             [](const elips::Filter& f) -> py::object {
                 const auto constraints = f.exact_constraints();
                 if (!constraints.has_value()) {
                     return py::none();
                 }
                 py::list out;
                 for (const auto& [field, values] : *constraints) {
                     py::list value_list;
                     for (const auto& value : values) {
                         value_list.append(from_meta(value));
                     }
                     out.append(py::make_tuple(py::str(field), value_list));
                 }
                 return out;
             },
             R"doc(Equality constraints this filter can satisfy via the metadata index.

Returns:
    A list of ``(field, [values])`` tuples the query planner can resolve
    without a scan, or ``None`` when the filter is not index-accelerable.
)doc")
        .def("__repr__", [](const elips::Filter& f) {
            return f.matches_all() ? "<Filter match-all>"
                                    : "<Filter>";
        });

    // =====================  TransactionVault  =====================

    py::class_<elips::TransactionVault>(m, "TransactionVault")
        .def("place",
             [](elips::TransactionVault& tv, const py::iterable& vector,
                const py::dict& data, const py::object& id) {
                 auto native_vector = to_vector(vector);
                 auto payload = to_payload(data);
                 auto record_id = to_optional_id(id);
                 py::gil_scoped_release unlocked;
                 return tv.place(native_vector, std::move(payload),
                                 std::move(record_id))
                     .to_string();
             },
             py::arg("vector"), py::arg("data") = py::dict(),
             py::arg("id") = py::none())
        .def("erase",
             [](elips::TransactionVault& tv, const std::string& id) {
                 const auto record_id = elips::RecordID::from_string(id);
                 py::gil_scoped_release unlocked;
                 tv.erase(record_id);
             });

    // =====================  Transaction  =====================

    py::class_<TransactionHolder>(m, "Transaction")
        .def("vault",
             [](TransactionHolder& h, const std::string& name) {
                 return h.txn.vault(name);
             },
             py::keep_alive<0, 1>())
        // commit applies the whole batch: index inserts plus one WAL fsync.
        .def("commit",
             [](TransactionHolder& h) {
                 py::gil_scoped_release unlocked;
                 h.txn.commit();
             })
        .def("rollback",
             [](TransactionHolder& h) {
                 py::gil_scoped_release unlocked;
                 h.txn.rollback();
             })
        .def("__enter__",
             [](TransactionHolder& h) -> TransactionHolder& { return h; })
        .def("__exit__",
             [](TransactionHolder& h, const py::object& exc_type,
                const py::object&, const py::object&) -> bool {
                 if (exc_type.is_none()) {
                     py::gil_scoped_release unlocked;
                     h.txn.commit();
                 }
                 return false;
             });

    // =====================  Vault  =====================

    py::class_<elips::Vault, std::unique_ptr<elips::Vault, py::nodelete>>(
        m, "Vault", R"doc(A named partition of records: the unit of search.

A vault owns one index and one record store. Records in a vault share a
dimension and metric, inherited from the database. Vaults are created lazily by
:meth:`Database.vault`, so there is no separate "create collection" step.

Thread safety: every method is safe to call concurrently. Readers share access;
writers exclude. A background ingest thread and a request-handling thread pool
can share one vault without external locking.

Example:
    A full ingest and retrieval cycle, including the parts most tutorials skip
    -- metadata filtering, deletion, and index maintenance::

        import elips

        db = elips.open("/data/kb", dimension=384, metric="cosine")
        chunks = db.vault("chunks")

        # 1. Ingest. Payload keys are indexed for filtering, so put anything
        #    you will filter on here rather than only in the document text.
        ids = []
        for page, text in enumerate(pages):
            ids.append(chunks.place(
                embed(text),                       # your embedding model
                {"doc": "handbook", "page": page, "public": True},
            ))

        # 2. Retrieve, restricted to public handbook pages. The filter is
        #    applied inside the search and the beam widens automatically when
        #    the filter is selective, so you still get `top` results back.
        hits = chunks.seek(
            embed("how do I request leave?"),
            top=5,
            where=elips.Filter().field("doc").equals("handbook")
                                .field("public").equals(True),
        )
        for hit in hits:
            print(f"{hit.distance:.4f}  page {hit.data['page']}")

        # 3. The document was updated: drop its old chunks.
        for rid in ids:
            chunks.erase(rid)

        # 4. Deletes are tombstones. The index self-compacts once they pass
        #    GraphParams.compaction_ratio, but after a bulk delete it is worth
        #    reclaiming immediately.
        print("tombstones:", chunks.pending_removals)
        chunks.vacuum()
        print("after vacuum:", chunks.pending_removals)

        db.close()
)doc")
        .def("place",
             [](elips::Vault& v, const py::iterable& vector,
                const py::dict& data, const py::object& id,
                const py::object& document, const py::object& chunk,
                const py::object& lineage) {
                 std::optional<elips::DocumentAttachment> doc;
                 std::optional<elips::ChunkInfo> chunk_info;
                 std::optional<elips::EmbeddingLineage> embedding_lineage;
                 if (!document.is_none()) {
                     doc = document.cast<elips::DocumentAttachment>();
                 }
                 if (!chunk.is_none()) {
                     chunk_info = chunk.cast<elips::ChunkInfo>();
                 }
                 if (!lineage.is_none()) {
                     embedding_lineage =
                         lineage.cast<elips::EmbeddingLineage>();
                 }
                 // Everything Python-facing is converted above, while the GIL
                 // is held. What remains -- index insert, WAL append, fsync --
                 // is native and holds the vault's writer lock, so running it
                 // with the GIL held would serialize every other thread behind
                 // this vault's lock for no reason.
                 auto native_vector = to_vector(vector);
                 auto payload = to_payload(data);
                 auto record_id = to_optional_id(id);
                 py::gil_scoped_release unlocked;
                 return v.place(native_vector, std::move(payload),
                                std::move(record_id), std::move(doc),
                                std::move(chunk_info),
                                std::move(embedding_lineage))
                     .to_string();
             },
             py::arg("vector"), py::arg("data") = py::dict(),
             py::arg("id") = py::none(), py::arg("document") = py::none(),
             py::arg("chunk") = py::none(),
             py::arg("lineage") = py::none())
        .def("place_document",
             [](elips::Vault& v, const std::string& text, const py::dict& data,
                const py::object& id, const py::object& chunk,
                const py::object& lineage) {
                 std::optional<elips::ChunkInfo> chunk_info;
                 std::optional<elips::EmbeddingLineage> embedding_lineage;
                 if (!chunk.is_none()) {
                     chunk_info = chunk.cast<elips::ChunkInfo>();
                 }
                 if (!lineage.is_none()) {
                     embedding_lineage =
                         lineage.cast<elips::EmbeddingLineage>();
                 }
                 auto payload = to_payload(data);
                 auto record_id = to_optional_id(id);
                 // Safe to release even though this path embeds: place_document
                 // embeds *before* taking the vault lock, and PythonTextEmbedder
                 // re-acquires the GIL itself. Releasing here is what lets a
                 // native embedder run off the GIL at all.
                 py::gil_scoped_release unlocked;
                 return v.place_document(text, std::move(payload),
                                         std::move(record_id),
                                         std::move(chunk_info),
                                         std::move(embedding_lineage))
                     .to_string();
             },
             py::arg("text"), py::arg("data") = py::dict(),
             py::arg("id") = py::none(), py::arg("chunk") = py::none(),
             py::arg("lineage") = py::none())
        .def("place_many",
             [](elips::Vault& v, const py::iterable& records) {
                 std::vector<elips::Record> recs;
                 for (const auto& item : records) {
                     py::dict d = py::reinterpret_borrow<py::dict>(item);
                     py::object id = py::none();
                     py::object chunk = py::none();
                     py::object lineage = py::none();
                     const py::dict payload =
                         d.contains("data")
                             ? py::reinterpret_borrow<py::dict>(d["data"])
                             : py::dict();
                     if (d.contains("id")) {
                         id = py::reinterpret_borrow<py::object>(d["id"]);
                     }
                     if (d.contains("chunk")) {
                         chunk = py::reinterpret_borrow<py::object>(d["chunk"]);
                     }
                     if (d.contains("lineage")) {
                         lineage =
                             py::reinterpret_borrow<py::object>(d["lineage"]);
                     }
                     if (d.contains("text") && !d.contains("vector")) {
                         std::optional<elips::ChunkInfo> chunk_info;
                         std::optional<elips::EmbeddingLineage> embedding_lineage;
                         if (!chunk.is_none()) {
                             chunk_info = chunk.cast<elips::ChunkInfo>();
                         }
                         if (!lineage.is_none()) {
                             embedding_lineage =
                                 lineage.cast<elips::EmbeddingLineage>();
                         }
                         v.place_document(d["text"].cast<std::string>(),
                                          to_payload(payload),
                                          to_optional_id(id), chunk_info,
                                          embedding_lineage);
                         continue;
                     }

                     elips::Record rec;
                     rec.vector = to_vector(d["vector"]);
                     rec.payload = to_payload(payload);
                     if (d.contains("id")) {
                         rec.id = elips::RecordID::from_string(
                             d["id"].cast<std::string>());
                     }
                     if (d.contains("document") && !d["document"].is_none()) {
                         rec.document =
                             d["document"].cast<elips::DocumentAttachment>();
                     }
                     if (!chunk.is_none()) {
                         rec.chunk = chunk.cast<elips::ChunkInfo>();
                     }
                     if (!lineage.is_none()) {
                         rec.lineage =
                             lineage.cast<elips::EmbeddingLineage>();
                     }
                     recs.push_back(std::move(rec));
                 }
                 py::gil_scoped_release unlocked;
                 v.place_many(recs);
             },
             py::arg("records"))
        .def("seek",
             [](const elips::Vault& v, const py::iterable& vector,
                std::size_t top, const elips::Filter& where,
                const py::object& threshold) {
                 std::optional<float> th;
                 if (!threshold.is_none())
                     th = threshold.cast<float>();
                 auto query = to_vector(vector);
                 py::gil_scoped_release unlocked;
                 return v.seek(query, top, where, th);
             },
             py::arg("vector"), py::arg("top") = 10,
             py::arg("where") = elips::Filter{},
             py::arg("threshold") = py::none())
        .def("seek_text",
             [](const elips::Vault& v, const std::string& text,
                std::size_t top, const elips::Filter& where,
                const py::object& threshold) {
                 std::optional<float> th;
                 if (!threshold.is_none()) {
                     th = threshold.cast<float>();
                 }
                 // seek_text embeds the query before taking the reader lock,
                 // and PythonTextEmbedder re-acquires the GIL for itself.
                 py::gil_scoped_release unlocked;
                 return v.seek_text(text, top, where, th);
             },
             py::arg("text"), py::arg("top") = 10,
             py::arg("where") = elips::Filter{},
             py::arg("threshold") = py::none())
        .def("seek_hybrid",
             [](const elips::Vault& v, const py::iterable& vector,
                const std::string& text, std::size_t top,
                const elips::Filter& where, const py::object& threshold,
                float lexical_weight) {
                 std::optional<float> th;
                 if (!threshold.is_none()) {
                     th = threshold.cast<float>();
                 }
                 auto query = to_vector(vector);
                 py::gil_scoped_release unlocked;
                 return v.seek_hybrid(query, text, top, where, th,
                                      lexical_weight);
             },
             py::arg("vector"), py::arg("text"), py::arg("top") = 10,
             py::arg("where") = elips::Filter{},
             py::arg("threshold") = py::none(),
             py::arg("lexical_weight") = 0.25F)
        .def("explain_seek",
             [](const elips::Vault& v, const py::iterable& vector,
                std::size_t top, const elips::Filter& where,
                const py::object& threshold, bool has_text_component) {
                 std::optional<float> th;
                 if (!threshold.is_none()) {
                     th = threshold.cast<float>();
                 }
                 auto query = to_vector(vector);
                 py::gil_scoped_release unlocked;
                 return v.explain_seek(query, top, where, th,
                                       has_text_component);
             },
             py::arg("vector"), py::arg("top") = 10,
             py::arg("where") = elips::Filter{},
             py::arg("threshold") = py::none(),
             py::arg("has_text_component") = false)
        .def("fetch",
             [](const elips::Vault& v,
                const std::string& id) -> py::object {
                 const auto record_id = elips::RecordID::from_string(id);
                 std::optional<elips::Record> rec;
                 {
                     py::gil_scoped_release unlocked;
                     rec = v.fetch(record_id);
                 }
                 if (!rec) {
                     return py::none();
                 }
                 return py::object(record_to_dict(*rec));
             })
        .def("erase",
             [](elips::Vault& v, const std::string& id) {
                 const auto record_id = elips::RecordID::from_string(id);
                 py::gil_scoped_release unlocked;
                 return v.erase(record_id);
             })
        .def("scan",
             [](const elips::Vault& v, const elips::Filter& where,
                std::size_t offset, int limit) {
                 std::size_t lim =
                     limit < 0
                         ? std::numeric_limits<std::size_t>::max()
                         : static_cast<std::size_t>(limit);
                 // The scan itself runs unlocked; building the Python list has
                 // to hold the GIL, so it happens after the release ends.
                 std::vector<elips::Record> records;
                 {
                     py::gil_scoped_release unlocked;
                     records = v.scan(where, offset, lim);
                 }
                 py::list out;
                 for (const auto& rec : records) {
                     out.append(record_to_dict(rec));
                 }
                 return out;
             },
             py::arg("where") = elips::Filter{},
             py::arg("offset") = 0, py::arg("limit") = -1)
        .def("info",
             [](const elips::Vault& v) {
                 py::gil_scoped_release unlocked;
                 return v.info();
             })
        .def("count",
             [](const elips::Vault& v) {
                 py::gil_scoped_release unlocked;
                 return v.info().count;
             })
        .def("rebuild_index", &elips::Vault::rebuild_index,
             py::call_guard<py::gil_scoped_release>(),
             "Rebuild the index from the authoritative record store. Useful "
             "after bulk loads, or to recover index quality after heavy churn.")
        .def("vacuum", &elips::Vault::vacuum,
             py::call_guard<py::gil_scoped_release>(),
             "Reclaim index space held by deleted records.")
        .def("quantize", &elips::Vault::quantize,
             py::call_guard<py::gil_scoped_release>(),
             R"doc(Train a codebook over this vault and compress it in place.

Requires a codec on the config (see :class:`QuantParams`). Raises
:class:`ConfigError` if none is set, if the vault is empty, or if it is already
quantized.

Compression is a separate step from configuring it because product quantization
cannot encode the first record: a codebook has to be learned from real data.
Before this call the vault stores full fp32; after it, inserts are encoded on
arrival. Searches work correctly either way.

This holds the vault's writer lock for its whole duration, so treat it as a
maintenance operation alongside :meth:`Database.compact`. Training samples at
most 100,000 vectors, so its cost tracks the dimension rather than the vault
size, but encoding is linear in the record count.

Once compressed, :meth:`fetch` and :meth:`scan` return reconstructions rather
than the exact vectors written; each record dict carries ``approximate`` and
``codec`` so this is visible.
)doc")
        .def_property_readonly(
            "quantized", &elips::Vault::quantized,
            "True once :meth:`quantize` has trained a codebook and compressed "
            "this vault.")
        .def_property_readonly(
            "codec",
            [](const elips::Vault& v) {
                return std::string(elips::quant::to_string(v.codec()));
            },
            "Active codec name, or ``none`` while uncompressed.")
        .def("records",
             [](const elips::Vault& v) {
                 // records() copies the whole store under the vault lock; that
                 // copy is exactly the part worth doing off the GIL.
                 std::map<elips::RecordID, elips::Record> snapshot;
                 {
                     py::gil_scoped_release unlocked;
                     snapshot = v.records();
                 }
                 py::list out;
                 for (const auto& [id, record] : snapshot) {
                     (void)id;
                     out.append(record_to_dict(record));
                 }
                 return out;
             },
             R"doc(Snapshot every record in this vault.

Returns a list of record dicts. This copies the whole vault, so prefer
:meth:`scan` with a filter or ``limit`` for large vaults.
)doc")
        .def("set_read_only", &elips::Vault::set_read_only,
             py::arg("read_only"),
             "Toggle runtime read-only mode. While read-only, any mutation "
             "raises :class:`StorageError`.")
        .def_property_readonly("read_only", &elips::Vault::read_only,
                               "True when this vault refuses mutations.")
        .def_property_readonly(
            "sealed", &elips::Vault::sealed,
            "True once the owning database has been closed. Writes to a "
            "sealed vault raise instead of silently failing to persist.")
        .def_property_readonly("pending_removals",
                               &elips::Vault::pending_removals,
                               "Records deleted from the index but not yet "
                               "reclaimed. See :meth:`vacuum`.")
        .def_property_readonly("name", &elips::Vault::name,
                               "This vault's name.")
        .def("__repr__", [](const elips::Vault& v) {
            const auto vi = v.info();
            return "<Vault name='" + v.name() +
                   "' count=" + std::to_string(vi.count) +
                   " dimension=" + std::to_string(vi.dimension) + ">";
        });

    // =====================  Database  =====================

    py::class_<elips::ElipsInstance>(m, "Database", R"doc(A database handle: one per directory, owning every vault and all persistence.

Durability model: writes are appended to a write-ahead log and ``fsync``'d
before :meth:`Vault.place` returns, so an acknowledged write survives an OS
crash or power loss. :meth:`checkpoint` folds the log into a snapshot.
:meth:`close` checkpoints, releases the cross-process lock, and seals every
vault so later writes raise instead of silently vanishing.

Concurrency: a ``flock`` keeps other processes out; in-process reader/writer
locks make the handle safe to share across threads.

Example:
    Batch ingest with atomic transactions, then verify durability across a
    reopen -- the shape of a real ingest job::

        import elips

        db = elips.open("/data/catalog", dimension=768, metric="cosine")

        # 1. Batch writes in a transaction. Either every record in the batch
        #    lands or none does: a disk-full error mid-batch rolls back the
        #    records already applied, and a crash mid-commit recovers to the
        #    pre-batch state.
        for batch in chunked(products, 1000):
            with db.begin_transaction() as txn:
                shelf = txn.vault("products")
                for product in batch:
                    shelf.place(embed(product.text), {"sku": product.sku})
            # commit() runs on a clean __exit__; an exception rolls back.

        # 2. Fold the WAL into a snapshot so recovery has less to replay.
        db.checkpoint()

        # 3. Query with EQL when the query text comes from elsewhere -- a
        #    saved report, a config file, an API request.
        rows = db.query(
            'seek in products nearest $q top 10 where sku = "A-1" yield',
            {"q": embed("wireless keyboard")},
        )

        # 4. Rebuild indexes and reclaim tombstones, then close cleanly.
        db.compact()
        db.close()

        # 5. Reopening reads the snapshot plus any WAL tail. The dimension and
        #    metric come from the persisted identity, so they need not be
        #    repeated -- and conflicting values raise ConfigError.
        db = elips.open("/data/catalog")
        print(db.vault("products").count())
)doc")
        .def("vault", &elips::ElipsInstance::vault,
             py::return_value_policy::reference_internal)
        .def("list_vaults", &elips::ElipsInstance::list_vaults)
        .def("begin_transaction",
             [](py::object db_ref) {
                 auto& db = db_ref.cast<elips::ElipsInstance&>();
                 return std::make_unique<TransactionHolder>(
                     std::move(db_ref), db);
             })
        .def("checkpoint", &elips::ElipsInstance::checkpoint,
             py::call_guard<py::gil_scoped_release>())
        .def("compact", &elips::ElipsInstance::compact,
             py::call_guard<py::gil_scoped_release>())
        .def("vacuum", &elips::ElipsInstance::vacuum,
             py::call_guard<py::gil_scoped_release>(),
             "Reclaim tombstoned index space across every vault.")
        .def("quantize", &elips::ElipsInstance::quantize, py::arg("vault"),
             py::call_guard<py::gil_scoped_release>(),
             "Compress one vault (see :meth:`Vault.quantize`), then checkpoint "
             "so the codebook and encoded records are durable before the WAL is "
             "truncated.")
        .def("quantize_all", &elips::ElipsInstance::quantize_all,
             py::call_guard<py::gil_scoped_release>(),
             "Compress every vault in this database.")
        .def("close", &elips::ElipsInstance::close)
        .def("abandon", &elips::ElipsInstance::abandon)
        .def_property_readonly(
            "path", &elips::ElipsInstance::path,
            "The directory this database was opened from, or ``\":memory:\"``.")
        .def_property_readonly(
            "persistent", &elips::ElipsInstance::persistent,
            "False for in-memory databases, which have no WAL or snapshot.")
        .def_property_readonly(
            "closed", &elips::ElipsInstance::closed,
            "True once :meth:`close` or :meth:`abandon` has run.")
        .def("query",
             [](elips::ElipsInstance& db, const std::string& eql,
                const py::dict& bindings) {
                 std::map<std::string, elips::Vector> binds;
                 for (const auto& [k, v] : bindings) {
                     binds.emplace(k.cast<std::string>(),
                                   to_vector(py::cast<py::iterable>(v)));
                 }
                 py::gil_scoped_release unlocked;
                 return db.query(eql, binds);
             },
             py::arg("eql"), py::arg("bindings") = py::dict())
#ifdef ELIPS_GPU_ENABLED
        .def("gpu_info", [](const elips::ElipsInstance& db) { return db.gpu_info(); })
        .def("gpu_stats", [](const elips::ElipsInstance& db) { return db.gpu_stats(); })
#endif
        .def_property_readonly(
            "config",
            [](const elips::ElipsInstance& db) -> elips::Config {
                return db.config();
            })
        .def("__enter__",
             [](elips::ElipsInstance& db) { return &db; })
        .def("__exit__",
             [](elips::ElipsInstance& db, const py::object&,
                const py::object&, const py::object&) { db.close(); })
        .def("__repr__", [](const elips::ElipsInstance& db) {
            auto vaults = db.list_vaults();
            return "<Database vaults=" +
                   std::to_string(vaults.size()) + ">";
        });

    // =====================  open()  =====================

    m.def("open",
          [](const std::string& path, std::uint16_t dimension,
             const std::string& metric, const std::string& index,
             const std::string& access_mode, const py::object& gpu_config,
             const py::object& embedder, const std::string& embedder_provider,
             const std::string& embedder_model,
             const std::string& embedder_revision,
             bool use_default_text_embedder, const py::object& quantization) {
              elips::Config config;
              config.dimension(dimension)
                  .metric(elips::metric_from_string(metric))
                  .auto_text_embedder(use_default_text_embedder);
              if (index == "exact") config.index(elips::IndexType::exact);
              if (!quantization.is_none()) {
                  // Accept either a QuantParams or a bare codec name, so
                  // open(path, quantization="sq8") works for the common case.
                  if (py::isinstance<elips::quant::QuantParams>(quantization)) {
                      config.quantization(
                          quantization.cast<elips::quant::QuantParams>());
                  } else {
                      elips::quant::QuantParams params;
                      params.codec = elips::quant::codec_from_string(
                          quantization.cast<std::string>().c_str());
                      config.quantization(params);
                  }
              }
              if (access_mode == "read_only") {
                  config.access_mode(elips::AccessMode::read_only);
              }
#ifdef ELIPS_GPU_ENABLED
              if (!gpu_config.is_none()) {
                  config.gpu(gpu_config.cast<elips::gpu::GpuConfig>());
              }
#else
              if (!gpu_config.is_none()) {
                  throw py::value_error(
                      "gpu config requires ELIPS to be built with GPU bindings");
              }
#endif
              if (!embedder.is_none()) {
                  if (py::isinstance<elips::LocalTextEmbedderOptions>(embedder)) {
                      config.local_text_embedder(
                          embedder.cast<elips::LocalTextEmbedderOptions>());
                  } else {
                      config.text_embedder(std::make_shared<PythonTextEmbedder>(
                          embedder, embedder_provider, embedder_model,
                          embedder_revision, dimension));
                  }
              }
              return elips::open(path, config);
          },
          py::arg("path"), py::arg("dimension") = 0,
          py::arg("metric") = "cosine", py::arg("index") = "graph",
          py::arg("access_mode") = "read_write", py::arg("gpu") = py::none(),
          py::arg("embedder") = py::none(),
          py::arg("embedder_provider") = "python",
          py::arg("embedder_model") = "callable",
          py::arg("embedder_revision") = "",
          py::arg("use_default_text_embedder") = true,
          py::arg("quantization") = py::none());

    m.def("open_with_config",
          [](const std::string& path, const elips::Config& config) {
              return elips::open(path, config);
          },
          py::arg("path"), py::arg("config"));
}
