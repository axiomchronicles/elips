#include "elips/query_engine/EQLParser.hpp"

#include "elips/query_engine/EQLLexer.hpp"

namespace elips::eql {
namespace {

// Recursive-descent parser over a flat token stream.
class Parser {
public:
    explicit Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

    Statement parse_statement() {
        const std::string& kw = peek().text;
        // New verbs are primary; the old ones remain accepted as deprecated
        // synonyms so existing EQL queries keep working.
        if (is_word("search") || is_word("seek")) return parse_search();
        if (is_word("get") || is_word("fetch")) return parse_get();
        if (is_word("scan")) return parse_scan();
        if (is_word("add") || is_word("place")) return parse_add();
        if (is_word("delete") || is_word("erase")) return parse_delete();
        throw ParseError{"EQL: expected a statement keyword, got '" + kw + "'"};
    }

private:
    const Token& peek() const { return tokens_[pos_]; }
    const Token& advance() { return tokens_[pos_++]; }
    bool at_end() const { return peek().kind == TokenKind::end; }

    bool is_word(std::string_view w) const {
        return peek().kind == TokenKind::word && peek().text == w;
    }
    bool is_punct(std::string_view p) const {
        return peek().kind == TokenKind::punct && peek().text == p;
    }
    bool match_word(std::string_view w) {
        if (is_word(w)) {
            ++pos_;
            return true;
        }
        return false;
    }
    bool match_punct(std::string_view p) {
        if (is_punct(p)) {
            ++pos_;
            return true;
        }
        return false;
    }
    void expect_word(std::string_view w) {
        if (!match_word(w)) {
            throw ParseError{"EQL: expected '" + std::string(w) + "'"};
        }
    }
    // Consume a verb keyword, accepting either the new (primary) spelling or a
    // deprecated alias, so old-vocabulary queries keep parsing.
    void expect_verb(std::string_view primary, std::string_view alias) {
        if (!match_word(primary) && !match_word(alias)) {
            throw ParseError{"EQL: expected '" + std::string(primary) + "'"};
        }
    }
    void expect_punct(std::string_view p) {
        if (!match_punct(p)) {
            throw ParseError{"EQL: expected '" + std::string(p) + "'"};
        }
    }
    std::string expect_identifier() {
        if (peek().kind != TokenKind::word) {
            throw ParseError{"EQL: expected an identifier"};
        }
        return advance().text;
    }
    std::string expect_string() {
        if (peek().kind != TokenKind::string) {
            throw ParseError{"EQL: expected a string literal"};
        }
        return advance().text;
    }
    double expect_number() {
        if (peek().kind != TokenKind::number) {
            throw ParseError{"EQL: expected a number"};
        }
        return advance().number;
    }

    std::vector<float> parse_vector_literal() {
        expect_punct("[");
        std::vector<float> values;
        if (!is_punct("]")) {
            values.push_back(static_cast<float>(expect_number()));
            while (match_punct(",")) {
                values.push_back(static_cast<float>(expect_number()));
            }
        }
        expect_punct("]");
        return values;
    }

    VectorRef parse_vector_ref() {
        if (match_punct("$")) {
            return VectorRef{{}, expect_identifier()};
        }
        return VectorRef{parse_vector_literal(), {}};
    }

    MetaValue parse_value() {
        const Token& t = peek();
        if (t.kind == TokenKind::string) {
            return advance().text;
        }
        if (t.kind == TokenKind::number) {
            const Token n = advance();
            return n.is_integer ? MetaValue{static_cast<std::int64_t>(n.number)}
                                : MetaValue{n.number};
        }
        if (is_word("true") || is_word("false")) {
            return MetaValue{advance().text == "true"};
        }
        throw ParseError{"EQL: expected a value literal"};
    }

    Filter parse_filter() { return parse_or(); }

    Filter parse_or() {
        Filter left = parse_and();
        while (match_word("or")) {
            left = left.or_(parse_and());
        }
        return left;
    }
    Filter parse_and() {
        Filter left = parse_not();
        while (match_word("and")) {
            left = left.and_(parse_not());
        }
        return left;
    }
    Filter parse_not() {
        if (match_word("not")) {
            return Filter::not_(parse_not());
        }
        if (match_punct("(")) {
            Filter inner = parse_or();
            expect_punct(")");
            return inner;
        }
        return parse_comparison();
    }
    Filter parse_comparison() {
        const std::string field = expect_identifier();
        if (match_word("in")) {
            expect_punct("[");
            std::vector<MetaValue> set;
            if (!is_punct("]")) {
                set.push_back(parse_value());
                while (match_punct(",")) {
                    set.push_back(parse_value());
                }
            }
            expect_punct("]");
            return Filter::in_set(field, std::move(set));
        }
        if (match_word("contains")) {
            return Filter::has_substring(field, expect_string());
        }
        const Comparator op = parse_comparator();
        return Filter::compare(field, op, parse_value());
    }
    Comparator parse_comparator() {
        if (peek().kind != TokenKind::punct) {
            throw ParseError{"EQL: expected a comparator"};
        }
        const std::string op = advance().text;
        if (op == "=") return Comparator::eq;
        if (op == "!=") return Comparator::ne;
        if (op == "<") return Comparator::lt;
        if (op == "<=") return Comparator::le;
        if (op == ">") return Comparator::gt;
        if (op == ">=") return Comparator::ge;
        throw ParseError{"EQL: unknown comparator '" + op + "'"};
    }

    SearchStatement parse_search() {
        expect_verb("search", "seek");
        expect_word("in");
        SearchStatement stmt;
        stmt.collection = expect_identifier();
        expect_word("nearest");
        stmt.query = parse_vector_ref();
        while (!is_word("yield") && !at_end()) {
            if (match_word("k") || match_word("top")) {
                stmt.top = static_cast<int>(expect_number());
            } else if (match_word("threshold")) {
                stmt.threshold = expect_number();
            } else if (match_word("where")) {
                stmt.where = parse_filter();
            } else if (match_word("rank_by")) {
                const std::string by = expect_identifier();
                if (by != "distance") {
                    stmt.rank_by = by;
                }
            } else if (match_word("project")) {
                stmt.projection = parse_projection();
            } else {
                throw ParseError{"EQL: unexpected token '" + peek().text +
                                 "' in search"};
            }
        }
        expect_word("yield");
        return stmt;
    }

    std::vector<std::string> parse_projection() {
        std::vector<std::string> fields;
        if (match_punct("*")) {
            return fields;  // empty => all
        }
        fields.push_back(expect_identifier());
        while (match_punct(",")) {
            fields.push_back(expect_identifier());
        }
        return fields;
    }

    GetStatement parse_get() {
        expect_verb("get", "fetch");
        expect_word("from");
        GetStatement stmt;
        stmt.collection = expect_identifier();
        expect_word("id");
        stmt.id = expect_string();
        expect_word("yield");
        return stmt;
    }

    ScanStatement parse_scan() {
        expect_word("scan");
        expect_word("in");
        ScanStatement stmt;
        stmt.collection = expect_identifier();
        while (!is_word("yield") && !at_end()) {
            if (match_word("where")) {
                stmt.where = parse_filter();
            } else if (match_word("offset")) {
                stmt.offset = static_cast<int>(expect_number());
            } else if (match_word("limit")) {
                stmt.limit = static_cast<int>(expect_number());
            } else {
                throw ParseError{"EQL: unexpected token '" + peek().text +
                                 "' in scan"};
            }
        }
        expect_word("yield");
        return stmt;
    }

    AddStatement parse_add() {
        expect_verb("add", "place");
        expect_word("in");
        AddStatement stmt;
        stmt.collection = expect_identifier();
        expect_word("vector");
        stmt.vector = parse_vector_literal();
        if (match_word("metadata") || match_word("data")) {
            stmt.data = parse_json_object();
        }
        return stmt;
    }

    DeleteStatement parse_delete() {
        expect_verb("delete", "erase");
        expect_word("from");
        DeleteStatement stmt;
        stmt.collection = expect_identifier();
        expect_word("id");
        stmt.id = expect_string();
        return stmt;
    }

    Payload parse_json_object() {
        Payload payload;
        expect_punct("{");
        if (!is_punct("}")) {
            do {
                std::string key = expect_string();
                expect_punct(":");
                payload.emplace(std::move(key), parse_value());
            } while (match_punct(","));
        }
        expect_punct("}");
        return payload;
    }

    std::vector<Token> tokens_;
    std::size_t pos_{0};
};

}  // namespace

Statement parse(std::string_view source) {
    Parser parser(tokenize(source));
    return parser.parse_statement();
}

}  // namespace elips::eql
