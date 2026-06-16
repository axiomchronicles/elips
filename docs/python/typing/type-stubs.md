# Type Stubs & IDE Support

ELIPS ships a complete type stub file (`_core.pyi`, 924 lines) alongside the
compiled extension, enabling full static analysis and IDE support. The higher
level `modern.py` surface (`connect()`, `Engine`, `Arena`, `Row`, `Hit`) also
ships inline Python annotations.

## How Type Stubs Work

When a Python package contains a compiled extension (`.so`/`.dylib`/`.pyd`), type checkers cannot inspect it at runtime to determine argument types, return types, or class members. A `.pyi` stub file provides this information in a form static analyzers can consume.

The combination required for ELIPS:

| File | Purpose |
|---|---|
| `elips/_core.pyi` | Type stubs for the `_core` extension module |
| `elips/py.typed` | PEP 561 marker — signals that the package is typed |
| `elips/__init__.py` | Runtime imports from `_core`, type checkers follow the imports |

## PEP 561: `py.typed`

The `py.typed` file is an **empty marker file** that tells type checkers (MyPy, Pyright, Pylance) that the `elips` package provides inline type information. Without it, type checkers would fall back to treating the extension module as untyped (`Any`).

`setup.py` declares both files as package data:

```python
package_data={"elips": ["py.typed", "_core.pyi"]},
```

## IDE Support

### PyCharm

PyCharm Professional and Community editions natively support `.pyi` stub files and the `py.typed` marker. Features enabled:

- Autocompletion for all ELIPS classes, methods, and parameters
- Type inference for `elips.open()`, vault operations, and filter chains
- Inline documentation from docstrings in `_core.pyi`
- Go-to-definition on ELIPS symbols navigates to the stub

### VSCode / Pylance

Pylance (the default Python language server in VSCode) fully supports PEP 561. Features:

- IntelliSense with parameter info, return types, and member access
- Type checking in strict mode via `python.analysis.typeCheckingMode`
- Hover tooltips showing signatures from the stub file
- Import completion for `elips.*` symbols

### MyPy

MyPy recognizes the `py.typed` marker and uses `_core.pyi` for type checking:

```bash
mypy my_script.py
```

Example type-checked code:

```python
import elips

db: elips.Database = elips.open(":memory:", dimension=3)
docs: elips.Vault = db.vault("documents")
rid: str = docs.place([1.0, 2.0, 3.0], {"title": "Test"})

result: elips.Result = docs.seek([1.0, 0.0, 0.0], top=5)[0]
reveal_type(result.id)       # Revealed type is "str"
reveal_type(result.distance) # Revealed type is "float"
reveal_type(result.data)     # Revealed type is "dict[str, MetaValue]"
```

### Pyright

Pyright (used by Pylance under the hood) also works with the stub:

```bash
pyright my_script.py
```

Pyright's strict mode is especially useful for catching type errors with ELIPS filter chains and EQL bindings.

## Type Aliases

The stub defines reusable type aliases at module level:

```python
MetaValue = Union[bool, int, float, str]
Vector = Sequence[float]
PayloadLike = Mapping[str, MetaValue]
```

These aliases are duplicated in `elips/types.py` for runtime use. The stub versions serve static analysis.

| Alias | Definition | Usage |
|---|---|---|
| `MetaValue` | `Union[bool, int, float, str]` | A typed metadata value |
| `Vector` | `Sequence[float]` | A vector of floats |
| `PayloadLike` | `Mapping[str, MetaValue]` | A metadata payload dict |

## Class Stub Coverage

The `_core.pyi` stub covers every Python-facing type. Below is the complete inventory.

### Error Hierarchy (8 classes)

```
ElipsError(Exception)
├── DimensionMismatch(ElipsError)
├── InvalidVector(ElipsError)
├── ConfigError(ElipsError)
├── NotFound(ElipsError)
├── StorageError(ElipsError)
├── LockConflict(ElipsError)
└── ParseError(ElipsError)
```

Each is declared as a simple `class` inheriting from its parent exception:

```python
class ElipsError(Exception):
    """Base exception for all ELIPS errors."""

class DimensionMismatch(ElipsError):
    """Vector dimension does not match the database/vault configuration."""
```

### Core Enums (4 enums, all `IntEnum`)

| Enum | Members |
|---|---|
| `Metric` | `cosine`, `euclidean`, `dot_product` |
| `IndexType` | `graph`, `exact` |
| `Durability` | `paranoid`, `standard`, `relaxed`, `ephemeral` |
| `Comparator` | `eq`, `ne`, `lt`, `le`, `gt`, `ge` |

Each member is typed with `: int`:

```python
class Metric(IntEnum):
    """Similarity metrics supported by ELIPS."""
    cosine: int
    euclidean: int
    dot_product: int
```

### EQL Types (1 enum, 1 class)

| Type | Description |
|---|---|
| `TokenKind` | `IntEnum` — `word`, `number`, `string`, `punct`, `end` |
| `Token` | Has `kind: TokenKind`, `text: str`, `number: float`, `is_integer: bool` |

### Configuration And Text Embedding Types

| Class | Key Members |
|---|---|
| `GraphParams` | `max_connections: int`, `ef_construction: int`, `ef_search: int` |
| `LocalEmbedderConfig` | `model`, `revision`, `storage_path`, `dimension` |
| `TextEmbedderInfo` | `kind`, `provider`, `model`, `revision`, `backend`, `dimension`, `fingerprint`, `storage_path`, `rehydratable`, `loaded`, `auto_attached` |
| `Config` | Fluent builder with `.dimension(dim)`, `.metric(str)`, `.index(str)`, `.graph_params(GraphParams)`, `.durability(str)`, `.access_mode(str)`, `.segmented_storage(bool)`, `.metadata_acceleration(bool)`, `.auto_text_embedder(bool)`, `.local_text_embedder(LocalEmbedderConfig)`, `.text_embedder(callable, ...)`, `.gpu(GpuConfig)`. Properties include `has_text_embedder`, `text_embedder_info`, and `auto_text_embedder_enabled`. |

### GPU Types (7 enums, 7 classes) — Conditional

GPU types are defined in the stub unconditionally (type checkers see them regardless of build). At runtime, they only exist if the extension was compiled with GPU support.

| Type | Kind | Members / Fields |
|---|---|---|
| `GpuPolicy` | `IntEnum` | `auto`, `prefer_gpu`, `require_gpu`, `cpu_only`, `specific` |
| `IndexBuildMode` | `IntEnum` | `gpu_build_cpu_serve`, `gpu_build_gpu_serve`, `hybrid` |
| `GpuIndexAlgorithm` | `IntEnum` | `auto`, `cagra`, `ivf_flat`, `ivf_pq`, `brute_force` |
| `GpuPrecision` | `IntEnum` | `fp32`, `fp16`, `int8`, `auto` |
| `GpuError` | `IntEnum` | `device_not_found`, `insufficient_memory`, `kernel_launch_failed`, `transfer_failed`, `index_build_failed`, `unsupported_metric`, `initialization_failed`, `backend_unavailable` |
| `GraphBuildAlgo` | `IntEnum` | `ivf_pq`, `nn_descent`, `iterative_search` |
| `GraphIndexBuildParams` | Class | `intermediate_graph_degree`, `graph_degree`, `build_algo`, `nn_descent_iterations`, `compression_ratio` |
| `IvfPqBuildParams` | Class | `n_lists`, `pq_dim`, `pq_bits`, `add_data_on_build`, `kmeans_n_iters`, `kmeans_trainset_fraction` |
| `GpuIndexBuildParams` | Class | `params: Union[GraphIndexBuildParams, IvfPqBuildParams]` |
| `KernelTiming` | Class | `kernel_name: str`, `work_items: int`, `duration_us: int` (property) |
| `GpuConfig` | Class | `policy`, `preferred_backend`, `device_index`, `build_mode`, `algorithm`, `device_memory_pool_mb`, `fp16_search`, `unified_memory`, `batch_window_us`, `max_batch_size`, `ef_search`, `precision`, `profiling`, `graph_params`, `ivf_pq_params` |
| `GpuDeviceInfo` | Class | 25 fields including `name`, `vendor`, `backend`, `memory_gb`, `peak_tflops_fp32`, `supports_cagra`, etc. |
| `GpuMetricsSnapshot` | Class | 17 fields including `device_memory_used_bytes`, `search_p50_latency_us`, `batch_avg_size`, `fp16_search_enabled`, etc. |

### Document, Planning, And Core Classes

| Class | Key Methods and Properties |
|---|---|
| `DocumentAttachment` | `text`, `uri`, `mime_type` |
| `ChunkInfo` | `document_key`, `ordinal`, `char_start`, `char_end` |
| `EmbeddingLineage` | `provider`, `model`, `revision`, `attributes` |
| `QueryPlan` | `strategy`, `candidate_count`, `metadata_accelerated`, `gpu_index`, `index_type` |
| `VaultInfo` | Properties: `count: int`, `dimension: int`, `metric: str` |
| `Result` | Properties: `id: str`, `distance: float`, `data: dict[str, MetaValue]` |
| `Filter` | Methods: `.field(name)`, `.equals(v)`, `.not_equals(v)`, `.lt(v)`, `.le(v)`, `.gt(v)`, `.gte(v)`, `.one_of(values)`, `.contains(substring)`, `.and_(other)`, `.or_(other)`, `Filter.not_(inner)` (static) |
| `TransactionVault` | Methods: `.place(vector, data, id) -> str`, `.erase(id) -> None` |
| `Transaction` | Methods: `.vault(name) -> TransactionVault`, `.commit()`, `.rollback()`, context manager ( `__enter__`, `__exit__` ) |
| `Vault` | Properties: `name: str`. Methods: `.place(...)`, `.place_document(...)`, `.place_many(...)`, `.seek(...)`, `.seek_text(...)`, `.seek_hybrid(...)`, `.explain_seek(...)`, `.fetch(id)`, `.erase(id)`, `.scan(...)`, `.info()`, `.count()` |
| `Database` | Methods: `.vault(name) -> Vault`, `.list_vaults()`, `.begin_transaction()`, `.checkpoint()`, `.close()`, `.abandon()`, `.query(eql, bindings)`, `.gpu_info()`, `.gpu_stats()`. Properties: `config: Config`. Context manager. |

### Modern Python Types

The `modern.py` layer is not part of `_core.pyi`; it is regular Python code
with inline annotations. IDEs still provide full completion for:

- `Engine`
- `Arena`
- `Row`
- `Hit`
- `Embedder`

### Module-Level Functions

| Function | Signature |
|---|---|
| `open(path, dimension, metric, index) -> Database` | Open a database with simple parameters |
| `open_with_config(path, config) -> Database` | Open with a Config builder |
| `distance(metric, a, b) -> float` | Compute distance between two vectors (string or enum overload) |
| `requires_normalization(metric) -> bool` | Whether vectors should be L2-normalized |
| `metric_to_string(metric) -> str` | Convert Metric enum to string name |
| `metric_from_string(name) -> Metric` | Parse string to Metric enum |
| `validate_eql(source) -> None` | Validate EQL syntax (raises ParseError) |
| `tokenize_eql(source) -> list[Token]` | Tokenize EQL source string |

## Docstrings in Stubs

The stub includes comprehensive docstrings for every public method and property. These appear in IDE hover tooltips and are rendered by documentation generators:

```python
def place(
    self,
    vector: Vector,
    data: PayloadLike = ...,
    id: Optional[str] = ...,
) -> str:
    """Ingest a single record. Returns the assigned UUIDv7 id.

    Args:
        vector: The embedding vector (list or tuple of floats).
        data: Optional metadata payload (dict of str -> int/float/bool/str).
        id: Optional custom UUIDv7 record ID.

    Returns:
        The record's ID as a hex string.
    """
```

Default values use `...` (Ellipsis literal) because the actual defaults are provided by the C++ extension at runtime and cannot be expressed in the stub.

## Type Checking Configuration

### MyPy

In `mypy.ini` or `pyproject.toml`:

```ini
[mypy]
plugins = pydantic.mypy  ; if using pydantic with ELIPS
```

No special configuration is needed for ELIPS itself — the `py.typed` marker is recognized automatically.

### Pyright / Pylance

In `pyrightconfig.json`:

```json
{
    "typeCheckingMode": "basic"
}
```

Or in VSCode settings:

```json
{
    "python.analysis.typeCheckingMode": "basic"
}
```

### Strict Mode Considerations

When using strict type checking, note:

- `elips.open()` returns `Database` but the stub uses `...` for default parameter values. Function calls with omitted keyword arguments pass strict checks.
- `Result.data` returns `dict[str, MetaValue]`. You may need to narrow `MetaValue` with `isinstance()` checks before using values.
- Filter chain methods return `Filter` (self). The fluent API is fully type-safe even in strict mode.
