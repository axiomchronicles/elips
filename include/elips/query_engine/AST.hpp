#ifndef ELIPS_QUERY_ENGINE_AST_HPP
#define ELIPS_QUERY_ENGINE_AST_HPP

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "elips/domain/Record.hpp"
#include "elips/metadata/Filter.hpp"

namespace elips::eql {

// A query vector: either an inline literal or a bound variable ($name).
struct VectorRef {
    std::vector<float> literal;
    std::string binding;  // non-empty => use bindings[binding]
};

struct SearchStatement {
    std::string vault;
    VectorRef query;
    std::optional<int> top;
    std::optional<double> threshold;
    Filter where;                       // matches all if no where-clause
    std::optional<std::string> rank_by;  // nullopt => rank by distance
    std::vector<std::string> projection;  // empty => all fields
};

struct FetchStatement {
    std::string vault;
    std::string id;
};

struct ScanStatement {
    std::string vault;
    Filter where;
    std::optional<int> offset;
    std::optional<int> limit;
};

struct InsertStatement {
    std::string vault;
    std::vector<float> vector;
    Payload data;
};

struct DeleteStatement {
    std::string vault;
    std::string id;
};

using Statement = std::variant<SearchStatement, FetchStatement, ScanStatement,
                               InsertStatement, DeleteStatement>;

}  // namespace elips::eql

#endif  // ELIPS_QUERY_ENGINE_AST_HPP
