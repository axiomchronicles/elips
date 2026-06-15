# EQL API Reference

The ELIPS Query Language (EQL) provides a SQL-like declarative interface for vector operations. This document covers the Python API for EQL execution, validation, and tokenization.

## `Database.query()`

```python
def query(
    self,
    eql: str,
    bindings: Mapping[str, Vector] = {},
) -> list[Result]
```

Execute a single EQL statement with optional parameterized vectors.

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `eql` | `str` | required | The EQL statement string |
| `bindings` | `Mapping[str, Vector]` | `{}` | Map of variable names to vector values. Referenced as `$name` in the EQL |

**Returns:** List of `Result` objects (same type as `Vault.seek()` returns).

**Raises:**
- `ParseError` — malformed EQL syntax
- `DimensionMismatch` — bound vector dimension does not match the target vault
- `NotFound` — referenced vault does not exist
- `InvalidVector` — vector contains `NaN` or `Inf`

### EQL Statements

EQL supports five statement types: `place`, `seek`, `fetch`, `scan`, and `erase`.

#### PLACE — Insert Records

Insert a record with a literal vector and metadata:

```
place in <vault> vector <literal> data <json>
```

```python
db.query('place in items vector [1.0, 0.0] data {"name": "alpha", "val": 10}')
```

#### SEEK — Vector Search

Search for nearest neighbors with optional filtering:

```
seek in <vault> nearest <vector> top <n> [where <conditions>] [project <fields>] yield
```

```python
# Basic search
results = db.query("seek in documents nearest [1.0, 0.0] top 5 yield")

# Parameterized vector
results = db.query(
    "seek in documents nearest $q top 10 project title, year yield",
    bindings={"q": [1.0, 0.0, 0.0]},
)

# Filtered
results = db.query(
    'seek in documents nearest $q top 20 where category = "finance" and year >= 2022 yield',
    bindings={"q": [1.0, 0.0, 0.0]},
)
```

Each result is a `Result` with `.id`, `.distance`, and `.data`.

#### FETCH — Retrieve by ID

```
fetch from <vault> id "<uuid>" [project <fields>] yield
```

```python
results = db.query('fetch from items id "018f3c7a-0000-7000-8000-000000000001" project name, val yield')
```

#### SCAN — Metadata Iteration

```
scan in <vault> [where <conditions>] [limit <n>] yield
```

```python
results = db.query('scan in items where val >= 10 limit 100 yield')
```

#### ERASE — Remove by ID

```
erase from <vault> id "<uuid>"
```

```python
db.query('erase from items id "018f3c7a-0000-7000-8000-000000000001"')
```

No results are returned.

### PROJECT Clause

The `project` clause selects which metadata fields appear in results:

```
seek in documents nearest [1.0, 0.0] top 5 project title, author yield
```

When `project` is omitted, all metadata fields are returned.

### WHERE Clause Filtering

EQL `where` clauses support comparison operators in infix form:

```
where field = value
where field != value
where field > value
where field >= value
where field < value
where field <= value
where field1 = value1 and field2 > value2
```

Values support:
- Numbers: `42`, `3.14`, `-1`
- Strings: `"hello"`, `'single quoted'`
- Booleans: `true`, `false`

### YIELD Keyword

The `yield` keyword terminates statements that return results (`seek`, `fetch`, `scan`). It is optional for `place` and absent for `erase`.

## `elips.validate_eql()`

```python
def validate_eql(source: str) -> None
```

Validate an EQL statement string without executing it.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `source` | `str` | The EQL source string |

**Returns:** `None` if the statement is syntactically valid.

**Raises:**
- `ParseError` — on invalid EQL syntax (with a descriptive error message)

**Examples:**

```python
elips.validate_eql("seek in docs nearest [1.0] top 5 yield")
# → None (valid)

elips.validate_eql("seek in docs nearest [1.0] top five yield")
# → ParseError: Expected number after TOP at position 30

elips.validate_eql("garbage !!!")
# → ParseError: Unexpected token at position 0
```

## `elips.tokenize_eql()`

```python
def tokenize_eql(source: str) -> list[Token]
```

Tokenize an EQL source string into its constituent tokens. Useful for syntax highlighting, IDE tooling, and debugging.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `source` | `str` | The EQL source string |

**Returns:** List of `Token` objects.

**Example:**

```python
tokens = elips.tokenize_eql("seek in docs nearest $q top 5 yield")
for t in tokens:
    print(f"{t.kind.name:8}  {t.text!r}")
```

Output:

```
word      'seek'
word      'in'
word      'docs'
word      'nearest'
punct     '$'
word      'q'
word      'top'
number    '5'
word      'yield'
end       ''
```

## TokenKind

```python
class TokenKind(IntEnum)
```

EQL token categories.

| Member | Description |
|---|---|
| `word` | Keyword or identifier (`seek`, `in`, `docs`, `$q`) |
| `number` | Numeric literal (`5`, `3.14`, `-1`) |
| `string` | Quoted string literal (`"hello"`, `'world'`) |
| `punct` | Punctuation or operator (`$`, `=`, `>=`, `[`, `]`, `,`, `{`, `}`) |
| `end` | End-of-input sentinel |

## Token

```python
class Token
```

A single EQL token produced by the lexer.

| Attribute | Type | Description |
|---|---|---|
| `kind` | `TokenKind` | Token category |
| `text` | `str` | Raw text of the token |
| `number` | `float` | Numeric value if `kind == TokenKind.number`, else `0.0` |
| `is_integer` | `bool` | `True` if the number literal has no fractional part |

**Example:**

```python
tokens = elips.tokenize_eql("seek in docs nearest [1.0, 0.5] top 5 yield")

# First token: keyword
print(tokens[0].kind)   # TokenKind.word
print(tokens[0].text)   # "seek"
print(tokens[0].number) # 0.0 (irrelevant for non-number tokens)

# Number token
num_token = tokens[6]   # The "5" token
print(num_token.kind)       # TokenKind.number
print(num_token.text)       # "5"
print(num_token.number)     # 5.0
print(num_token.is_integer) # True

# Float number token
float_token = tokens[4]  # The "1.0" token
print(float_token.number)     # 1.0
print(float_token.is_integer) # False
```

## Error Handling

All EQL functions raise `ParseError` (a subclass of `ElipsError`) on malformed input:

```python
try:
    elips.validate_eql("invalid syntax")
except elips.ParseError as e:
    print(f"Parse error: {e}")

try:
    db.query("seek in nonexistent_vault nearest [1.0] top 5 yield")
except elips.NotFound as e:
    print(f"Vault not found: {e}")
except elips.ParseError as e:
    print(f"Syntax error: {e}")
```

## EQL vs. Python API

EQL and the Python API are equivalent — `db.query("seek in docs nearest $q top 5 yield", {"q": v})` produces the same results as `db.vault("docs").seek(v, top=5)`. Choose based on your use case:

| Consideration | Python API | EQL |
|---|---|---|
| Dynamic construction | Programmatic filter building | String interpolation |
| Parameterization | Native Python types | `$param` bindings map |
| Readability | Pythonic chained calls | SQL-like declarative syntax |
| Tooling | IDE autocomplete on methods | `validate_eql()` + `tokenize_eql()` |
| Flexibility | Full Python expressiveness | Fixed grammar, statically analyzable |

You can mix both styles in the same application.