# EQL API Reference

The ELIPS Query Language (EQL) is a small declarative query language for vector
operations. It is **not** SQL — there is no `SELECT`/`INSERT`/`ORDER BY`, and
keywords are lowercase. This page covers running, validating, and tokenizing
EQL from Python.

## Running EQL

EQL executes through the native `Database.query(eql, bindings={})`. From a
high-level client, call `query_eql(...)` (which delegates to the native handle)
or reach `client.native.query(...)`:

```python
# no-exec — signature
def query(self, eql: str, bindings: Mapping[str, Vector] = {}) -> list[Result]
```

`$name` placeholders in the EQL are filled from `bindings`. Results are native
`Result` rows with `.id`, `.distance`, and `.data`.

> EQL is not exposed as `.query(...)` on the high-level handle — that name is
> the Chroma vector search on `Collection`. Use `client.query_eql(...)`.

## Statements

EQL has five statement types. The canonical verbs are **`add`**, **`search`**,
**`get`**, **`scan`**, and **`delete`**; the pre-Phase-5 verbs
(`place` / `seek` / `fetch` / `erase`) and the top-k keyword `top` are still
accepted as deprecated spellings.

| Statement | Form |
|---|---|
| **add** | `add in <coll> vector <literal> data <json>` |
| **search** | `search in <coll> nearest <vec> k <n> [where <cond>] [project <fields>] yield` |
| **get** | `get from <coll> id "<id>" [project <fields>] yield` |
| **scan** | `scan in <coll> [where <cond>] [limit <n>] yield` |
| **delete** | `delete from <coll> id "<id>"` |

- `k` is the top-k count (alias of the deprecated `top`).
- `search`/`scan` take `in <coll>`; `get`/`delete` take `from <coll> id "..."`.
- `yield` terminates statements that return rows (`search`, `get`, `scan`).
- `where` supports infix `=`, `!=`, `>`, `>=`, `<`, `<=`, combined with `and`;
  values are numbers, quoted strings, or `true`/`false`.
- `project <fields>` selects which metadata fields appear (all when omitted).

### End-to-end example

```python
import elips

client = elips.Client(dimension=2)
db = client.native

db.query('add in items vector [1.0, 0.0] data {"name": "alpha", "year": 2024}')
db.query('add in items vector [0.0, 1.0] data {"name": "beta", "year": 2019}')

rows = client.query_eql(
    'search in items nearest $q k 5 where year >= 2020 project name yield',
    {"q": [1.0, 0.0]},
)
for r in rows:
    print(r.id[:8], round(r.distance, 3), r.data.get("name"))

# get by id, then delete it
rid = list(rows)[0].id
got = db.query(f'get from items id "{rid}" yield')
db.query(f'delete from items id "{rid}"')
client.close()
```

## `elips.validate_eql(source)`

Validate EQL syntax without executing it. Returns `None` on success; raises
`ParseError` on invalid syntax.

```python
import elips

elips.validate_eql('search in docs nearest [1.0] k 5 yield')   # ok -> None

try:
    elips.validate_eql("garbage !!!")
except elips.ParseError as e:
    print("invalid:", str(e)[:40])
```

## `elips.tokenize_eql(source)`

Tokenize an EQL string into a list of `Token` objects (useful for highlighting
and tooling).

```python
import elips

for t in elips.tokenize_eql("search in docs nearest $q k 5 yield"):
    print(t.kind.name, repr(t.text))
```

### `TokenKind`

Token categories (`IntEnum`): `word`, `number`, `string`, `punct`, `end`.
Import it from `elips.query` (reaching it as `elips.TokenKind` still works but
emits a `DeprecationWarning`):

```python
from elips.query import TokenKind, Token
```

### `Token`

| Attribute | Type | Description |
|---|---|---|
| `kind` | `TokenKind` | Token category. |
| `text` | `str` | Raw token text. |
| `number` | `float` | Numeric value when `kind == number`, else `0.0`. |
| `is_integer` | `bool` | `True` when the number literal has no fractional part. |

## Error handling

All EQL functions raise `ParseError` (a subclass of `ElipsError`) on malformed
input; execution against a missing collection raises `NotFound`.

```python
import elips

client = elips.Client(dimension=2)
try:
    client.query_eql('search in nope nearest $q k 5 yield', {"q": [1.0, 0.0]})
except (elips.ParseError, elips.NotFound) as e:
    print(type(e).__name__)
client.close()
```

## EQL vs. the Python API

`client.query_eql('search in docs nearest $q k 5 yield', {"q": v})` produces the
same ranking as `collection.query(query_embeddings=v, n_results=5)`. Choose the
Python API for programmatic filter building and IDE autocomplete; choose EQL for
a declarative, statically analyzable string form with `$param` bindings.
