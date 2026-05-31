#include "elips/query_engine/EQLLexer.hpp"

#include <cctype>

#include "elips/domain/Errors.hpp"

namespace elips::eql {
namespace {

bool is_word_start(char c) {
    return (std::isalpha(static_cast<unsigned char>(c)) != 0) || c == '_';
}
bool is_word_char(char c) {
    return (std::isalnum(static_cast<unsigned char>(c)) != 0) || c == '_';
}

}  // namespace

std::vector<Token> tokenize(std::string_view src) {
    std::vector<Token> tokens;
    std::size_t i = 0;
    const std::size_t n = src.size();

    while (i < n) {
        const char c = src[i];
        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            ++i;
            continue;
        }
        if (c == '#') {  // line comment
            while (i < n && src[i] != '\n') {
                ++i;
            }
            continue;
        }
        if (c == '"') {
            std::string body;
            ++i;
            while (i < n && src[i] != '"') {
                body.push_back(src[i]);
                ++i;
            }
            if (i >= n) {
                throw ElipsError{"EQL: unterminated string literal"};
            }
            ++i;  // closing quote
            tokens.push_back(Token{TokenKind::string, std::move(body), 0.0, false});
            continue;
        }
        // number: optional leading '-' followed by a digit
        if ((std::isdigit(static_cast<unsigned char>(c)) != 0) ||
            (c == '-' && i + 1 < n &&
             std::isdigit(static_cast<unsigned char>(src[i + 1])) != 0)) {
            const std::size_t start = i;
            if (src[i] == '-') {
                ++i;
            }
            bool is_int = true;
            while (i < n && (std::isdigit(static_cast<unsigned char>(src[i])) != 0)) {
                ++i;
            }
            if (i < n && src[i] == '.') {
                is_int = false;
                ++i;
                while (i < n &&
                       (std::isdigit(static_cast<unsigned char>(src[i])) != 0)) {
                    ++i;
                }
            }
            const std::string text(src.substr(start, i - start));
            tokens.push_back(Token{TokenKind::number, text, std::stod(text), is_int});
            continue;
        }
        if (is_word_start(c)) {
            const std::size_t start = i;
            while (i < n && is_word_char(src[i])) {
                ++i;
            }
            tokens.push_back(
                Token{TokenKind::word, std::string(src.substr(start, i - start)),
                      0.0, false});
            continue;
        }
        // punctuation, including two-char comparators
        std::string op(1, c);
        if ((c == '<' || c == '>' || c == '!' || c == '=') && i + 1 < n &&
            src[i + 1] == '=') {
            op.push_back('=');
        }
        i += op.size();
        tokens.push_back(Token{TokenKind::punct, std::move(op), 0.0, false});
    }

    tokens.push_back(Token{TokenKind::end, "", 0.0, false});
    return tokens;
}

}  // namespace elips::eql
