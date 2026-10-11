# EQL — ELIPS Query Language

EQL is a small, expression-oriented DSL — not SQL. It has a keyword-agnostic
lexer, a recursive-descent parser, an AST, and an executor
(`src/EQLLexer.cpp`, `src/EQLParser.cpp`, `include/elips/query_engine/`).
Keywords are **lowercase**, query vectors are either inline literals or
`$name` bindings, and statements read verb-first.

> There is no uppercase `SEARCH FROM … SEEK :q TOP …` SQL-like dialect, no
> `:colon` bindings, and no `WHERE id = N` by-id form. Earlier docs showed those;
> they were never implemented. The grammar below is the whole language.

## Verbs

The primary verbs are `add`, `search`, `get`, `scan`, and `delete`. The
pre-rename spellings `place`, `seek`, `fetch`, and `erase` are still accepted as
**deprecated synonyms** (the parser maps them to the new verbs), so existing
queries keep working.

| Primary | Deprecated synonym |
|---|---|
| `add` | `place` |
| `search` | `seek` |
| `get` | `fetch` |
| `delete` | `erase` |
| `scan` | *(no rename)* |

## Running EQL

EQL executes through the native database handle. From Python, reach it with
`client.native.query(eql, bindings)`:

```python
import elips

client = elips.Client(dimension=3)
docs = client.get_or_create_collection("docs")
docs.add(ids=["a"], embeddings=[[0.1, 0.2, 0.3]], metadatas=[{"year": 2024}])

results = client.native.query(
    "search in docs nearest $q k 5 where year >= 2023 yield",
    {"q": [0.1, 0.2, 0.3]},
)
assert results and results[0].distance == 0.0
client.close()
```

From the CLI:

```bash
elips query /my_db --eql 'search in docs nearest [0.1, 0.2, 0.3] k 5 yield'
```

## Statements

### search — nearest-neighbor search
```
search in <collection>
    nearest <[literal] | $binding>
    [k <int>]
    [threshold <number>]
    [where <filter>]
    [rank_by <distance | field>]
    [project <* | field, field, ...>]
    yield
```
- `k` defaults to 10 (or unbounded when a `threshold` range is given). `top` is a
  deprecated synonym for `k`.
- `nearest` takes an inline `[...]` vector literal or a `$binding` resolved from
  the `bindings` dict.
- `rank_by <field>` re-sorts the result by a metadata field; the default,
  `rank_by distance`, keeps distance order.
- `project` trims the returned payload to the named fields (`project *` keeps all).

### get — by id
```
get from <collection> id "<id-string>" yield
```
Ids are string literals (`fetch` is the deprecated synonym).

### scan — iterate (no vector)
```
scan in <collection> [where <filter>] [offset <int>] [limit <int>] yield
```

### add / delete — mutate
```
add in <collection> vector [0.1, 0.2, ...] [metadata { "key": value, ... }]
delete from <collection> id "<id-string>"
```
`metadata` is the payload object (`data` is the deprecated synonym). `place` and
`erase` are the deprecated synonyms for `add` and `delete`. Note `add` and
`delete` have no trailing `yield`.

## Filters

```
filter   := term (("and" | "or") term)* | "not" filter | "(" filter ")"
term     := field <comparator> value
          | field "in" [ value, value, ... ]
          | field "contains" "substring"
comparator := = | != | < | <= | > | >=
value      := "string" | number | true | false
```

Numbers without a decimal point are integers; `contains` is a substring match on
string fields. Comments start with `#` and run to end of line.

## Examples

```eql
# filtered search with projection
search in articles
    nearest $embedding
    k 10
    where category = "technology" and published_year >= 2023
    project title, author
    yield

# range search
search in faces nearest $probe threshold 0.15 where verified = true yield

# membership scan
scan in users where country in ["US", "GB", "CA"] limit 100 yield

# mutate
add in docs vector [0.1, 0.2, 0.3] metadata { "kind": "design" }
delete from docs id "019e3c7a-..."
```

## Errors

Parse errors are plain messages with no positional (`at position N`) tracking:

- `EQL: expected a statement keyword, got '<word>'` — the statement did not start
  with a verb.
- `EQL: expected '<word>'` — a required keyword (e.g. `in`, `nearest`, `yield`) was
  missing.
- `EQL: unexpected token '<word>' in search` — an unrecognized clause inside a
  `search`.
- `EQL: expected a number` / `expected a string literal` / `expected a value literal`.
