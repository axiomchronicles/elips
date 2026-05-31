#ifndef ELIPS_QUERY_ENGINE_EQL_LEXER_HPP
#define ELIPS_QUERY_ENGINE_EQL_LEXER_HPP

#include <string>
#include <string_view>
#include <vector>

namespace elips::eql {

enum class TokenKind { word, number, string, punct, end };

struct Token {
    TokenKind kind{TokenKind::end};
    std::string text;     // identifier/keyword, operator/bracket, or string body
    double number{0.0};   // for TokenKind::number
    bool is_integer{false};
};

// Lexes EQL source into tokens. Skips whitespace and `#` line comments.
[[nodiscard]] std::vector<Token> tokenize(std::string_view source);

}  // namespace elips::eql

#endif  // ELIPS_QUERY_ENGINE_EQL_LEXER_HPP
