// PyBind11 bindings for ELIPS. Exposes the embedded database, vaults, filters,
// transactions, configuration, GPU config, and the EQL query interface to Python.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "elips/elips.hpp"
#include "elips/query_engine/EQLLexer.hpp"
#include "elips/query_engine/EQLParser.hpp"
#include "elips/vector_engine/Metrics.hpp"

#ifdef ELIPS_GPU_ENABLED
#include "elips/gpu_engine/GpuConfig.hpp"
#include "elips/gpu_engine/GpuDeviceInfo.hpp"
#include "elips/gpu_engine/GpuMetricsSnapshot.hpp"
#include "elips/gpu_engine/GpuPort.hpp"
#include "elips/gpu_engine/GpuProfiler.hpp"
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

}  // namespace

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
        .value("dot_product", elips::Metric::dot_product)
        .export_values();

    py::enum_<elips::IndexType>(m, "IndexType")
        .value("graph", elips::IndexType::graph)
        .value("exact", elips::IndexType::exact)
        .export_values();

    py::enum_<elips::Durability>(m, "Durability")
        .value("paranoid", elips::Durability::paranoid)
        .value("standard", elips::Durability::standard)
        .value("relaxed", elips::Durability::relaxed)
        .value("ephemeral", elips::Durability::ephemeral)
        .export_values();

    py::enum_<elips::Comparator>(m, "Comparator")
        .value("eq", elips::Comparator::eq)
        .value("ne", elips::Comparator::ne)
        .value("lt", elips::Comparator::lt)
        .value("le", elips::Comparator::le)
        .value("gt", elips::Comparator::gt)
        .value("ge", elips::Comparator::ge)
        .export_values();

    // =====================  Utility functions  =====================

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

    py::enum_<elips::eql::TokenKind>(m, "TokenKind")
        .value("word", elips::eql::TokenKind::word)
        .value("number", elips::eql::TokenKind::number)
        .value("string", elips::eql::TokenKind::string)
        .value("punct", elips::eql::TokenKind::punct)
        .value("end", elips::eql::TokenKind::end)
        .export_values();

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
        .def(py::init<std::size_t, std::size_t, std::size_t>(),
             py::arg("max_connections") = 16,
             py::arg("ef_construction") = 200, py::arg("ef_search") = 50)
        .def_readwrite("max_connections",
                        &elips::GraphParams::max_connections)
        .def_readwrite("ef_construction",
                        &elips::GraphParams::ef_construction)
        .def_readwrite("ef_search", &elips::GraphParams::ef_search)
        .def("__repr__", [](const elips::GraphParams& p) {
            return "<GraphParams M=" + std::to_string(p.max_connections) +
                   " ef_c=" + std::to_string(p.ef_construction) +
                   " ef_s=" + std::to_string(p.ef_search) + ">";
        });

    // =====================  Config  =====================

    py::class_<elips::Config>(m, "Config")
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
                 if (type == "exact") c.index(elips::IndexType::exact);
                 else c.index(elips::IndexType::graph);
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
        .def("durability",
             [](elips::Config& c, const std::string& level) -> elips::Config& {
                 if (level == "paranoid")
                     c.durability(elips::Durability::paranoid);
                 else if (level == "relaxed")
                     c.durability(elips::Durability::relaxed);
                 else if (level == "ephemeral")
                     c.durability(elips::Durability::ephemeral);
                 else
                     c.durability(elips::Durability::standard);
                 return c;
             },
             py::arg("level"),
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
        .def_property_readonly("metric_enum", [](const elips::Config& c) {
            return c.metric();
        })
        .def_property_readonly("index_enum", [](const elips::Config& c) {
            return c.index();
        })
        .def_property_readonly("durability_enum", [](const elips::Config& c) {
            return c.durability();
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
        .value("backend_unavailable", elips::gpu::GpuError::BackendUnavailable)
        .export_values();

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
        .value("iterative_search", elips::gpu::GraphIndexBuildParams::BuildAlgo::IterativeSearch)
        .export_values();

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
        .def_readwrite("fp16_search", &elips::gpu::GpuConfig::enable_fp16_search)
        .def_readwrite("unified_memory", &elips::gpu::GpuConfig::use_unified_memory)
        .def_readwrite("batch_window_us", &elips::gpu::GpuConfig::dynamic_batch_window_us)
        .def_readwrite("max_batch_size", &elips::gpu::GpuConfig::dynamic_batch_max_size)
        .def_readwrite("ef_search", &elips::gpu::GpuConfig::default_ef_search_gpu)
        .def_readwrite("precision", &elips::gpu::GpuConfig::search_precision)
        .def_readwrite("profiling", &elips::gpu::GpuConfig::enable_profiling)
        .def_readwrite("graph_params", &elips::gpu::GpuConfig::graph_params)
        .def_readwrite("ivf_pq_params", &elips::gpu::GpuConfig::ivf_pq_params)
        .def("__repr__", [](const elips::gpu::GpuConfig& c) {
            return "<GpuConfig policy=" + std::to_string(static_cast<int>(c.policy)) + ">";
        });

    py::class_<elips::gpu::GpuDeviceInfo>(m, "GpuDeviceInfo")
        .def(py::init<>())
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
            return "<GpuDeviceInfo name='" + i.name + "' backend=" + i.backend + ">";
        });

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
#endif

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
        .def("__repr__", [](const elips::VaultInfo& vi) {
            return "<VaultInfo count=" + std::to_string(vi.count) +
                   " dimension=" + std::to_string(vi.dimension) +
                   " metric=" +
                   std::string(elips::to_string(vi.metric)) + ">";
        });

    // =====================  SearchResult  =====================

    py::class_<elips::SearchResult>(m, "Result")
        .def_property_readonly(
            "id", [](const elips::SearchResult& r) { return r.id.to_string(); })
        .def_readonly("distance", &elips::SearchResult::distance)
        .def_property_readonly(
            "data", [](const elips::SearchResult& r) { return from_payload(r.data); })
        .def("__repr__", [](const elips::SearchResult& r) {
            return "<Result id=" + r.id.to_string() +
                   " distance=" + std::to_string(r.distance) + ">";
        });

    // =====================  Filter  =====================

    py::class_<elips::Filter>(m, "Filter")
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
        .def("__repr__", [](const elips::Filter& f) {
            return f.matches_all() ? "<Filter match-all>"
                                    : "<Filter>";
        });

    // =====================  TransactionVault  =====================

    py::class_<elips::TransactionVault>(m, "TransactionVault")
        .def("place",
             [](elips::TransactionVault& tv, const py::iterable& vector,
                const py::dict& data, const py::object& id) {
                 return tv.place(to_vector(vector), to_payload(data),
                                 to_optional_id(id))
                     .to_string();
             },
             py::arg("vector"), py::arg("data") = py::dict(),
             py::arg("id") = py::none())
        .def("erase",
             [](elips::TransactionVault& tv, const std::string& id) {
                 tv.erase(elips::RecordID::from_string(id));
             });

    // =====================  Transaction  =====================

    py::class_<TransactionHolder>(m, "Transaction")
        .def("vault",
             [](TransactionHolder& h, const std::string& name) {
                 return h.txn.vault(name);
             },
             py::keep_alive<0, 1>())
        .def("commit", [](TransactionHolder& h) { h.txn.commit(); })
        .def("rollback", [](TransactionHolder& h) { h.txn.rollback(); })
        .def("__enter__",
             [](TransactionHolder& h) -> TransactionHolder& { return h; })
        .def("__exit__",
             [](TransactionHolder& h, const py::object& exc_type,
                const py::object&, const py::object&) -> bool {
                 if (exc_type.is_none()) {
                     h.txn.commit();
                 }
                 return false;
             });

    // =====================  Vault  =====================

    py::class_<elips::Vault, std::unique_ptr<elips::Vault, py::nodelete>>(
        m, "Vault")
        .def("place",
             [](elips::Vault& v, const py::iterable& vector,
                const py::dict& data, const py::object& id) {
                 return v.place(to_vector(vector), to_payload(data),
                                to_optional_id(id))
                     .to_string();
             },
             py::arg("vector"), py::arg("data") = py::dict(),
             py::arg("id") = py::none())
        .def("place_many",
             [](elips::Vault& v, const py::iterable& records) {
                 std::vector<elips::Record> recs;
                 for (const auto& item : records) {
                     py::dict d = py::reinterpret_borrow<py::dict>(item);
                     elips::Record rec;
                     rec.vector = to_vector(d["vector"]);
                     if (d.contains("data")) {
                         rec.payload = to_payload(
                             py::reinterpret_borrow<py::dict>(d["data"]));
                     }
                     if (d.contains("id")) {
                         rec.id = elips::RecordID::from_string(
                             d["id"].cast<std::string>());
                     }
                     recs.push_back(std::move(rec));
                 }
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
                 return v.seek(to_vector(vector), top, where, th);
             },
             py::arg("vector"), py::arg("top") = 10,
             py::arg("where") = elips::Filter{},
             py::arg("threshold") = py::none())
        .def("fetch",
             [](const elips::Vault& v,
                const std::string& id) -> py::object {
                 const auto rec =
                     v.fetch(elips::RecordID::from_string(id));
                 if (!rec) return py::none();
                 py::dict out;
                 out["id"] = rec->id.to_string();
                 out["vector"] = tuple_from_vector(rec->vector);
                 out["data"] = from_payload(rec->payload);
                 return out;
             })
        .def("erase",
             [](elips::Vault& v, const std::string& id) {
                 return v.erase(elips::RecordID::from_string(id));
             })
        .def("scan",
             [](const elips::Vault& v, const elips::Filter& where,
                std::size_t offset, int limit) {
                 std::size_t lim =
                     limit < 0
                         ? std::numeric_limits<std::size_t>::max()
                         : static_cast<std::size_t>(limit);
                 py::list out;
                 for (const auto& rec : v.scan(where, offset, lim)) {
                     py::dict row;
                     row["id"] = rec.id.to_string();
                     row["data"] = from_payload(rec.payload);
                     out.append(row);
                 }
                 return out;
             },
             py::arg("where") = elips::Filter{},
             py::arg("offset") = 0, py::arg("limit") = -1)
        .def("info",
             [](const elips::Vault& v) { return v.info(); })
        .def("count",
             [](const elips::Vault& v) { return v.info().count; })
        .def_property_readonly("name", &elips::Vault::name)
        .def("__repr__", [](const elips::Vault& v) {
            const auto vi = v.info();
            return "<Vault name='" + v.name() +
                   "' count=" + std::to_string(vi.count) +
                   " dimension=" + std::to_string(vi.dimension) + ">";
        });

    // =====================  Database  =====================

    py::class_<elips::ElipsInstance>(m, "Database")
        .def("vault", &elips::ElipsInstance::vault,
             py::return_value_policy::reference_internal)
        .def("list_vaults", &elips::ElipsInstance::list_vaults)
        .def("begin_transaction",
             [](py::object db_ref) {
                 auto& db = db_ref.cast<elips::ElipsInstance&>();
                 return std::make_unique<TransactionHolder>(
                     std::move(db_ref), db);
             })
        .def("checkpoint", &elips::ElipsInstance::checkpoint)
        .def("close", &elips::ElipsInstance::close)
        .def("abandon", &elips::ElipsInstance::abandon)
        .def("query",
             [](elips::ElipsInstance& db, const std::string& eql,
                const py::dict& bindings) {
                 std::map<std::string, elips::Vector> binds;
                 for (const auto& [k, v] : bindings) {
                     binds.emplace(k.cast<std::string>(),
                                   to_vector(py::cast<py::iterable>(v)));
                 }
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
}