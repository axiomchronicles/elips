#ifndef ELIPS_QUERY_ENGINE_EQL_PARSER_HPP
#define ELIPS_QUERY_ENGINE_EQL_PARSER_HPP

#include <string_view>

#include "elips/domain/Errors.hpp"
#include "elips/query_engine/AST.hpp"

namespace elips::eql {

// Raised on malformed EQL input. Carries a human-readable message.
class ParseError : public ElipsError {
public:
    using ElipsError::ElipsError;
};

// Parse a single EQL statement. Throws ParseError on invalid syntax.
[[nodiscard]] Statement parse(std::string_view source);

}  // namespace elips::eql

#endif  // ELIPS_QUERY_ENGINE_EQL_PARSER_HPP
