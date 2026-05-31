// PyBind11 bindings for ELIPS. Exposes the embedded database, vaults, filters,
// transactions, and the EQL query interface to Python. Built as the extension
// module `elips._core`; the Python package re-exports it.
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

std::optional<elips::RecordID> to_optional_id(const py::object& id) {
    if (id.is_none()) {
        return std::nullopt;
    }
    return elips::RecordID::from_string(id.cast<std::string>());
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "ELIPS — embedded local vector database";

    py::register_exception<elips::ElipsError>(m, "ElipsError");
    py::register_exception<elips::LockConflict>(m, "LockConflict");

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

    py::class_<elips::Filter>(m, "Filter")
        .def(py::init<>())
        .def("field", &elips::Filter::field, py::return_value_policy::reference_internal)
        .def("equals", [](elips::Filter& f, const py::handle& v) -> elips::Filter& {
            return f.equals(to_meta(v));
        }, py::return_value_policy::reference_internal)
        .def("not_equals", [](elips::Filter& f, const py::handle& v) -> elips::Filter& {
            return f.not_equals(to_meta(v));
        }, py::return_value_policy::reference_internal)
        .def("lt", [](elips::Filter& f, const py::handle& v) -> elips::Filter& {
            return f.lt(to_meta(v));
        }, py::return_value_policy::reference_internal)
        .def("le", [](elips::Filter& f, const py::handle& v) -> elips::Filter& {
            return f.le(to_meta(v));
        }, py::return_value_policy::reference_internal)
        .def("gt", [](elips::Filter& f, const py::handle& v) -> elips::Filter& {
            return f.gt(to_meta(v));
        }, py::return_value_policy::reference_internal)
        .def("gte", [](elips::Filter& f, const py::handle& v) -> elips::Filter& {
            return f.ge(to_meta(v));
        }, py::return_value_policy::reference_internal)
        .def("one_of", [](elips::Filter& f, const py::iterable& vs) -> elips::Filter& {
            std::vector<elips::MetaValue> set;
            for (const auto& v : vs) set.push_back(to_meta(v));
            return f.one_of(std::move(set));
        }, py::return_value_policy::reference_internal)
        .def("contains", &elips::Filter::contains,
             py::return_value_policy::reference_internal)
        .def("and_", &elips::Filter::and_)
        .def("or_", &elips::Filter::or_)
        .def_static("not_", &elips::Filter::not_);

    py::class_<elips::Vault, std::unique_ptr<elips::Vault, py::nodelete>>(m, "Vault")
        .def("place",
             [](elips::Vault& v, const py::iterable& vector, const py::dict& data,
                const py::object& id) {
                 return v.place(to_vector(vector), to_payload(data),
                                to_optional_id(id))
                     .to_string();
             },
             py::arg("vector"), py::arg("data") = py::dict(),
             py::arg("id") = py::none())
        .def("seek",
             [](const elips::Vault& v, const py::iterable& vector, std::size_t top,
                const elips::Filter& where, const py::object& threshold) {
                 std::optional<float> th;
                 if (!threshold.is_none()) th = threshold.cast<float>();
                 return v.seek(to_vector(vector), top, where, th);
             },
             py::arg("vector"), py::arg("top") = 10,
             py::arg("where") = elips::Filter{}, py::arg("threshold") = py::none())
        .def("fetch",
             [](const elips::Vault& v, const std::string& id) -> py::object {
                 const auto rec = v.fetch(elips::RecordID::from_string(id));
                 if (!rec) return py::none();
                 return from_payload(rec->payload);
             })
        .def("erase",
             [](elips::Vault& v, const std::string& id) {
                 return v.erase(elips::RecordID::from_string(id));
             })
        .def("scan",
             [](const elips::Vault& v, const elips::Filter& where) {
                 py::list out;
                 for (const auto& rec : v.scan(where)) {
                     py::dict row;
                     row["id"] = rec.id.to_string();
                     row["data"] = from_payload(rec.payload);
                     out.append(row);
                 }
                 return out;
             },
             py::arg("where") = elips::Filter{})
        .def("count", [](const elips::Vault& v) { return v.info().count; });

    py::class_<elips::ElipsInstance>(m, "Database")
        .def("vault", &elips::ElipsInstance::vault,
             py::return_value_policy::reference_internal)
        .def("list_vaults", &elips::ElipsInstance::list_vaults)
        .def("checkpoint", &elips::ElipsInstance::checkpoint)
        .def("close", &elips::ElipsInstance::close)
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
        .def("__enter__", [](elips::ElipsInstance& db) { return &db; })
        .def("__exit__",
             [](elips::ElipsInstance& db, const py::object&, const py::object&,
                const py::object&) { db.close(); });

    m.def("open",
          [](const std::string& path, std::uint16_t dimension,
             const std::string& metric, const std::string& index) {
              elips::Config config;
              config.dimension(dimension).metric(elips::metric_from_string(metric));
              if (index == "exact") config.index(elips::IndexType::exact);
              return elips::open(path, config);
          },
          py::arg("path"), py::arg("dimension") = 0,
          py::arg("metric") = "cosine", py::arg("index") = "graph");
}
