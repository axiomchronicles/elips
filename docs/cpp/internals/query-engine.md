# Query Engine

The query engine translates EQL (ELIPS Query Language) text into executable
operations against a database. It consists of a lexer, a recursive-descent
parser producing an AST, and a visitor-style executor.

---

## EQL Lexer

**Header:** `include/elips/query_engine/EQLLexer.hpp`
**Source:** `src/EQLLexer.cpp`

### Token Types

```cpp
enum class TokenKind { word, number, string, punct, end };

struct Token {
    TokenKind kind{TokenKind::end};
    std::string text;     // identifier/keyword text, operator string, or string body
    double number{0.0};   // parsed numeric value (for TokenKind::number)
    bool is_integer{false};
};
```

| Kind | Lexed by | Examples |
|------|----------|----------|
| `word` | `[a-zA-Z_][a-zA-Z0-9_]*` | `search`, `in`, `k`, `collection_name` |
| `number` | Optional `-`, digits, optional `.digits` | `10`, `-3`, `0.5`, `-0.75` |
| `string` | `"` ... `"` | `"hello"`, `"abc123"` |
| `punct` | Single or two-char comparator | `[`, `]`, `{`, `}`, `,`, `$`, `=`, `!=`, `<`, `<=`, `>`, `>=` |
| `end` | End of input | N/A |

### `tokenize()` Function

```cpp
[[nodiscard]] std::vector<Token> tokenize(std::string_view source);
```

Single-pass tokeniser over the source text:

1. **Whitespace:** Skipped (`std::isspace`).

2. **Line comments:** `#` through end of line is skipped (no block comments).

3. **String literals:** Quoted with `"`. The body is accumulated character by
   character (no escape sequences in v1.0). Unterminated strings throw
   `ElipsError{"EQL: unterminated string literal"}`.

4. **Numbers:** A leading `-` followed by a digit starts a number (single `-`
   alone is punctuation). After the optional leading `-`, digits are consumed.
   If a `.` follows, more digits are consumed and `is_integer` is set to false.
   The text is converted via `std::stod()` for the `number` field. Both
   `integer` and `float` are stored as `double` in the token; the parser uses
   `is_integer` to decide whether to store as `int64_t` or `double` in a
   `MetaValue`.

5. **Words:** Start with `[a-zA-Z_]`, continue with `[a-zA-Z0-9_]`.

6. **Punctuation / comparators:** Single-character punctuation is emitted as a
   one-char token. Two-character comparators (`<=`, `>=`, `!=`, `==`) are
   merged into a single two-char `punct` token:

   ```cpp
   if ((c == '<' || c == '>' || c == '!' || c == '=') && i + 1 < n &&
       src[i + 1] == '=') op.push_back('=');
   ```

7. **`end` token:** Always appended as the final token so the parser can
   detect end-of-input uniformly.

---

## EQL Parser

**Header:** `include/elips/query_engine/EQLParser.hpp`
**Source:** `src/EQLParser.cpp`

The parser is a hand-written **recursive-descent** parser operating over a flat
token stream. It is implemented as a private `Parser` class inside the
translation unit.

### Entry Point

```cpp
Statement parse(std::string_view source) {
    Parser parser(tokenize(source));
    return parser.parse_statement();
}
```

The public `parse()` function lexes the source, constructs a `Parser`,
and calls `parse_statement()`. On any syntax error, `ParseError` is thrown.

### Statement Dispatch

```cpp
Statement Parser::parse_statement() {
    const std::string& kw = peek().text;
    if (is_word("search") || is_word("seek")) return parse_search();
    if (is_word("get")    || is_word("fetch")) return parse_get();
    if (is_word("scan"))  return parse_scan();
    if (is_word("add")    || is_word("place")) return parse_add();
    if (is_word("delete") || is_word("erase")) return parse_delete();
    throw ParseError{"EQL: expected a statement keyword, got '" + kw + "'"};
}
```

The first word determines the statement type. The current verbs are `search`,
`get`, `scan`, `add`, `delete`; the pre-Phase-5 verbs (`seek`/`fetch`/`place`/
`erase`) are still accepted as aliases (see `MIGRATION.md`). There is no
look-ahead beyond one token.

### Filter Parsing (Recursive-Descent Expressions)

Filters use recursive descent with standard precedence:

```
parse_or()    → parse_and() ( "or" parse_and() )*
parse_and()   → parse_not()  ( "and" parse_not() )*
parse_not()   → "not" parse_not()  |  "(" parse_or() ")"  |  parse_comparison()
parse_comparison() → identifier ( "in" "[" values "]" | "contains" string | comparator value )
```

**Comparators** are parsed from `punct` tokens:

```cpp
Comparator parse_comparator() {
    if (peek().kind != TokenKind::punct)
        throw ParseError{"EQL: expected a comparator"};
    const std::string op = advance().text;
    if (op == "=")  return Comparator::eq;
    if (op == "!=") return Comparator::ne;
    if (op == "<")  return Comparator::lt;
    if (op == "<=") return Comparator::le;
    if (op == ">")  return Comparator::gt;
    if (op == ">=") return Comparator::ge;
    throw ParseError{"EQL: unknown comparator '" + op + "'"};
}
```

**Values** are parsed as `MetaValue`:

```cpp
MetaValue parse_value() {
    const Token& t = peek();
    if (t.kind == TokenKind::string) return advance().text;
    if (t.kind == TokenKind::number) {
        const Token n = advance();
        return n.is_integer ? MetaValue{static_cast<int64_t>(n.number)}
                            : MetaValue{n.number};
    }
    if (is_word("true") || is_word("false"))
        return MetaValue{advance().text == "true"};
    throw ParseError{"EQL: expected a value literal"};
}
```

Booleans are parsed as keywords `true`/`false`, not as special tokens.

### Vector Literal Parsing

```cpp
std::vector<float> parse_vector_literal() {
    expect_punct("[");
    std::vector<float> values;
    if (!is_punct("]")) {
        values.push_back(static_cast<float>(expect_number()));
        while (match_punct(","))
            values.push_back(static_cast<float>(expect_number()));
    }
    expect_punct("]");
    return values;
}
```

Vectors are comma-separated float literals inside square brackets. Empty
vectors `[]` are allowed.

### Vector Reference Parsing

```cpp
VectorRef parse_vector_ref() {
    if (match_punct("$")) return VectorRef{{}, expect_identifier()};
    return VectorRef{parse_vector_literal(), {}};
}
```

A `$` prefix denotes a bound variable (e.g. `$query_vec`), resolved at execute
time against the bindings map. Otherwise, an inline vector literal is expected.

### JSON Object Parsing (Payload)

```cpp
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
```

Braces `{ }` with comma-separated `"key": value` pairs. Keys are always
quoted strings. Values follow the same `parse_value()` rules.

### Projection Parsing

```cpp
std::vector<std::string> parse_projection() {
    std::vector<std::string> fields;
    if (match_punct("*")) return fields;  // empty vector → all fields
    fields.push_back(expect_identifier());
    while (match_punct(",")) fields.push_back(expect_identifier());
    return fields;
}
```

Projections appear after the `project` keyword in SEARCH statements. `*` means
return all payload fields. A comma-separated list selects specific fields.

### Statement-Specific Parsers

**SEARCH:**
```
search in <collection> nearest <vector_ref>
  [k <int>] [threshold <float>] [where <filter>]
  [rank_by <field>] [project <fields>]
  yield
```
`k` (alias `top`) sets the result count; `yield` is the required statement
terminator.

**GET:**
```
get from <collection> id <string> yield
```

**SCAN:**
```
scan in <collection> [where <filter>] [offset <int>] [limit <int>] yield
```

**ADD:**
```
add in <collection> vector <vector_literal> [metadata <json_object>]
```
`metadata` (alias `data`) carries the payload; `yield` is optional for ADD.

**DELETE:**
```
delete from <collection> id <string>
```

Each verb also accepts its pre-Phase-5 alias (`seek`/`fetch`/`place`/`erase`,
and `data` for `metadata`).

---

## AST Types

**Header:** `include/elips/query_engine/AST.hpp`

```cpp
using Statement = std::variant<SearchStatement, GetStatement, ScanStatement,
                               AddStatement, DeleteStatement>;

// Deprecated pre-Phase-5 aliases (one minor version):
using FetchStatement  [[deprecated("renamed to GetStatement")]] = GetStatement;
using InsertStatement [[deprecated("renamed to AddStatement")]] = AddStatement;
```

### `VectorRef` — Query Vector Representation

```cpp
struct VectorRef {
    std::vector<float> literal;   // inline literal values
    std::string binding;          // non-empty → resolve from bindings[this]
};
```

Exactly one of `literal` (non-empty) or `binding` (non-empty) is set.

### `SearchStatement`

```cpp
struct SearchStatement {
    std::string collection;
    VectorRef query;
    std::optional<int> top;                  // k; default 10 (or unbounded for threshold)
    std::optional<double> threshold;         // range query distance cutoff
    Filter where;                            // payload filter; matches_all if absent
    std::optional<std::string> rank_by;      // sort by a payload field; nullopt = by distance
    std::vector<std::string> projection;     // empty = all fields; else field whitelist
};
```

### `GetStatement`

```cpp
struct GetStatement {
    std::string collection;
    std::string id;
};
```

### `ScanStatement`

```cpp
struct ScanStatement {
    std::string collection;
    Filter where;
    std::optional<int> offset;
    std::optional<int> limit;
};
```

### `AddStatement`

```cpp
struct AddStatement {
    std::string collection;
    std::vector<float> vector;
    Payload data;
};
```

### `DeleteStatement`

```cpp
struct DeleteStatement {
    std::string collection;
    std::string id;
};
```

All AST types use `Filter` from the metadata engine rather than a
query-engine-specific representation — Filter is shared between the SDK
builder and EQL parser.

---

## QueryExecutor

**Header:** `include/elips/query_engine/QueryExecutor.hpp`
**Source:** `src/QueryExecutor.cpp`

### Execution Entry

```cpp
std::vector<SearchResult> execute(const Statement& statement, Database& db,
                                  const std::map<std::string, Vector>& bindings) {
    return std::visit(Executor{db, bindings}, statement);
}
```

Uses `std::visit` over the `Statement` variant. The `Executor` is a callable
struct with `operator()` overloads for each statement type.

### Binding Resolution

```cpp
Vector resolve_query(const VectorRef& ref,
                     const std::map<std::string, Vector>& bindings) {
    if (!ref.binding.empty()) {
        const auto it = bindings.find(ref.binding);
        if (it == bindings.end())
            throw NotFound{"EQL: unbound query vector $" + ref.binding};
        return it->second;
    }
    return Vector{ref.literal};
}
```

When a SEARCH statement uses `$variable` syntax, the executor looks up the
binding in the provided map. Missing bindings throw `NotFound`.

### Statement Dispatch to Collection

| Statement | Collection method called | Notes |
|-----------|-------------------|-------|
| `SearchStatement` | `collection.search(query, top, filter, threshold)` | Resolves vector, computes `top` from `k`/`top` or default 10 (or unbounded for threshold queries). Applies optional `rank_by` re-sort and `project` field filtering post-hoc. |
| `GetStatement` | `collection.get(RecordID::from_string(id))` | Looks up by string ID, wraps in SearchResult |
| `ScanStatement` | `collection.scan(filter, offset, limit)` | Iterates ordered record map. Offset/limit control pagination. |
| `AddStatement` | `collection.add(Vector{vector}, data)` | Returns single SearchResult with the new RecordID |
| `DeleteStatement` | `collection.remove(RecordID::from_string(id))` | Returns empty vector |

### SEARCH Statement Execution Details

```cpp
std::vector<SearchResult> operator()(const SearchStatement& s) const {
    const Vector query = resolve_query(s.query, bindings);
    const std::optional<float> threshold =
        s.threshold ? std::optional<float>{static_cast<float>(*s.threshold)} : std::nullopt;
    const std::size_t top =
        s.top ? static_cast<std::size_t>(*s.top)
              : (s.threshold ? unbounded_top : static_cast<std::size_t>(default_top));
    auto results = db.collection(s.collection).search(query, top, s.where, threshold);
    if (s.rank_by) {
        // re-sort by a payload field using meta_less()
    }
    apply_projection(results, s.projection);
    return results;
}
```

**Default top:** 10 results when no `k`/`top` or `threshold` is specified.

**Threshold queries:** When `threshold` is set but `top` is not, `top` is
set to `unbounded_top` (= 100,000) to get all candidates below the threshold.
The collection's `search()` method then filters by the threshold value.

**Ranking by payload field:** When `rank_by` is specified (and not `"distance"`),
results are re-sorted by the named payload field using `meta_less()`, which
compares numeric types across int64/double, and falls back to string
comparison or variant-index ordering for non-comparable types.

**Projection:** `apply_projection()` strips each result's payload down to
only the requested fields. If `project *` was used, the projection list is
empty and all fields are retained.

### Execution as Part of Database

The `Database::query()` method combines parse + execute:

```cpp
std::vector<SearchResult> Database::query(
    const std::string& eql, const std::map<std::string, Vector>& bindings) {
    return eql::execute(eql::parse(eql), *this, bindings);
}
```

This is the primary entry point for programmatic EQL usage from C++.

### Error Handling

- **Parse errors** (invalid syntax, unexpected tokens) throw `ParseError`,
  which inherits from `ElipsError`.
- **Runtime errors** (unbound variables, dimension mismatches) throw
  domain errors (`NotFound`, `DimensionMismatch`).
- **WAL/storage errors** during writes throw `StorageError`.

All errors propagate to the caller — neither the lexer, parser, nor executor
return error codes or null values.