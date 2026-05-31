#ifndef ELIPS_QUERY_ENGINE_QUERY_EXECUTOR_HPP
#define ELIPS_QUERY_ENGINE_QUERY_EXECUTOR_HPP

#include <map>
#include <string>
#include <vector>

#include "elips/domain/SearchResult.hpp"
#include "elips/domain/Vector.hpp"
#include "elips/query_engine/AST.hpp"

namespace elips {
class ElipsInstance;
}

namespace elips::eql {

// Execute a parsed statement against a database. Read statements return their
// rows; insert returns a single row carrying the new id; delete returns empty.
[[nodiscard]] std::vector<SearchResult> execute(
    const Statement& statement, ElipsInstance& db,
    const std::map<std::string, Vector>& bindings);

}  // namespace elips::eql

#endif  // ELIPS_QUERY_ENGINE_QUERY_EXECUTOR_HPP
