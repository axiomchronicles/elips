// PyBind11 bindings for ELIPS. Exposes the embedded database, vaults, filters,
// transactions, configuration, and the EQL query interface to Python. Built as
// the extension module `elips._core`; the Python package re-exports it.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "elips/elips.hpp"

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
        .def("__repr__", [](const elips::Config& c) {
            return "<Config dimension=" + std::to_string(c.dimension()) +
                   " metric=" +
                   std::string(elips::to_string(c.metric())) +
                   " index=" +
                   (c.index() == elips::IndexType::graph ? "graph"
                                                          : "exact") +
                   ">";
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