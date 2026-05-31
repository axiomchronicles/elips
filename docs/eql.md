# EQL — ELIPS Query Language

EQL is a small, expression-oriented language with a lexer, recursive-descent
parser, AST, and executor. Run it from any surface:

```python
db.query("seek in docs nearest $q top 10 where year >= 2023 yield",
         bindings={"q": query_vector})
```
```bash
elips query /my_db --eql 'seek in docs nearest [0.1, 0.2, 0.3] top 5 yield'
```

## Statements

### seek — nearest-neighbor search
```
seek in <vault>
    nearest <[literal] | $binding>
    [top <int>]
    [threshold <number>]
    [where <filter>]
    [rank_by <distance | field>]
    [project <* | field, field, ...>]
    yield
```
- `top` defaults to 10 (or unbounded when a `threshold` range is given).
- `rank_by <field>` re-sorts the result by a metadata field; default is distance.
- `project` trims the returned payload to the named fields.

### fetch — by id
```
fetch from <vault> id "<uuid>" yield
```

### scan — iterate (no vector)
```
scan in <vault> [where <filter>] [offset <int>] [limit <int>] yield
```

### place / erase — mutate
```
place in <vault> vector [0.1, 0.2, ...] [data { "key": value, ... }]
erase from <vault> id "<uuid>"
```

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
seek in articles
    nearest $embedding
    top 10
    where category = "technology" and published_year >= 2023
    project title, author
    yield

# range search
seek in faces nearest $probe threshold 0.15 where verified = true yield

# membership scan
scan in users where country in ["US", "GB", "CA"] limit 100 yield
```
