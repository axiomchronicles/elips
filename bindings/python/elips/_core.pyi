from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union
from enum import IntEnum

# -- Type aliases --------------------------------------------------------------

MetaValue = Union[bool, int, float, str]
Vector = Sequence[float]
PayloadLike = Mapping[str, MetaValue]

# -- Error hierarchy ----------------------------------------------------------

class ElipsError(Exception):
    """Base exception for all ELIPS errors."""

class DimensionMismatch(ElipsError):
    """Vector dimension does not match the database/vault configuration."""

class InvalidVector(ElipsError):
    """Vector contains NaN/Inf or is otherwise unusable."""

class ConfigError(ElipsError):
    """Configuration is invalid or conflicts with persisted identity."""

class NotFound(ElipsError):
    """Requested record/vault does not exist."""

class StorageError(ElipsError):
    """Persistence/IO failure."""

class LockConflict(ElipsError):
    """A second writer tried to open a database directory already held."""

class ParseError(ElipsError):
    """Malformed EQL input."""

# -- Core enums ---------------------------------------------------------------

class Metric(IntEnum):
    """Similarity metrics supported by ELIPS."""
    cosine: int
    euclidean: int
    dot_product: int

class IndexType(IntEnum):
    """Index backends."""
    graph: int
    exact: int

class Durability(IntEnum):
    """Durability levels trading write throughput against crash safety."""
    paranoid: int
    standard: int
    relaxed: int
    ephemeral: int

class Comparator(IntEnum):
    """Metadata comparison operators."""
    eq: int
    ne: int
    lt: int
    le: int
    gt: int
    ge: int

class AccessMode(IntEnum):
    """Database access mode."""
    read_write: int
    read_only: int

class QueryStrategy(IntEnum):
    """Planner strategy chosen for a query."""
    ann_index: int
    exact_candidates: int
    full_scan: int
    text_probe: int
    hybrid_fusion: int

# -- EQL token types ----------------------------------------------------------

class TokenKind(IntEnum):
    """EQL token categories."""
    word: int
    number: int
    string: int
    punct: int
    end: int

class Token:
    """A single EQL token produced by the lexer."""
    kind: TokenKind
    text: str
    number: float
    is_integer: bool

    def __repr__(self) -> str: ...

# -- GraphParams ---------------------------------------------------------------

class GraphParams:
    """Tunable parameters for the HierarchicalGraphIndex (HNSW)."""

    def __init__(
        self,
        max_connections: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> None: ...

    max_connections: int
    """Maximum number of connections per node (M)."""

    ef_construction: int
    """Beam width during index construction."""

    ef_search: int
    """Beam width during search."""

    def __repr__(self) -> str: ...

# -- Config -------------------------------------------------------------------

class Config:
    """Fluent builder for database configuration.

    Example:
        config = Config().dimension(1536).metric("cosine").index("graph")
        db = open_with_config("/data/vectors", config)
    """

    def __init__(self) -> None: ...
    def dimension(self, dim: int) -> "Config": ...
    def metric(self, metric: str) -> "Config": ...
    def index(self, type: str) -> "Config": ...
    def graph_params(self, params: GraphParams) -> "Config": ...
    def durability(self, level: str) -> "Config": ...
    def access_mode(self, mode: str) -> "Config": ...
    def segmented_storage(self, enabled: bool) -> "Config": ...
    def metadata_acceleration(self, enabled: bool) -> "Config": ...
    def text_embedder(
        self,
        embedder: Callable[[Sequence[str]], Sequence[Vector]],
        provider: str = ...,
        model: str = ...,
    ) -> "Config": ...
    def gpu(self, config: "GpuConfig") -> "Config": ...

    @property
    def dimension_val(self) -> int:
        """Get the configured dimension."""
    @property
    def metric_val(self) -> str:
        """Get the metric as a string (legacy alias for metric_enum)."""
    @property
    def metric_enum(self) -> Metric:
        """Get the configured Metric enum value."""
    @property
    def index_val(self) -> str:
        """Get the index type as a string (legacy alias)."""
    @property
    def index_enum(self) -> IndexType:
        """Get the configured IndexType enum value."""
    @property
    def graph_params_val(self) -> GraphParams:
        """Get the configured graph parameters."""
    @property
    def durability_enum(self) -> Durability:
        """Get the configured Durability enum value."""
    @property
    def access_mode_val(self) -> str:
        """Get the access mode as a string."""
    @property
    def access_mode_enum(self) -> AccessMode:
        """Get the configured AccessMode enum value."""
    @property
    def segmented_storage_enabled(self) -> bool:
        """Return whether segmented storage is enabled."""
    @property
    def metadata_acceleration_enabled(self) -> bool:
        """Return whether metadata acceleration is enabled."""
    @property
    def has_text_embedder(self) -> bool:
        """Return whether a text embedder is configured."""
    @property
    def gpu_val(self) -> Optional["GpuConfig"]:
        """Get the GPU configuration if set, else None."""

    def __repr__(self) -> str: ...

# -- GPU enums and types ------------------------------------------------------

class GpuPolicy(IntEnum):
    """GPU usage policy."""
    auto: int
    prefer_gpu: int
    require_gpu: int
    cpu_only: int
    specific: int

class IndexBuildMode(IntEnum):
    """GPU index build vs. serve mode."""
    gpu_build_cpu_serve: int
    gpu_build_gpu_serve: int
    hybrid: int

class GpuIndexAlgorithm(IntEnum):
    """GPU index algorithm selection."""
    auto: int
    cagra: int
    ivf_flat: int
    ivf_pq: int
    brute_force: int

class GpuPrecision(IntEnum):
    """GPU computation precision."""
    fp32: int
    fp16: int
    int8: int
    auto: int

class GpuError(IntEnum):
    """GPU error codes."""
    device_not_found: int
    insufficient_memory: int
    kernel_launch_failed: int
    transfer_failed: int
    index_build_failed: int
    unsupported_metric: int
    initialization_failed: int
    backend_unavailable: int

class GraphBuildAlgo(IntEnum):
    """Graph index build algorithm."""
    ivf_pq: int
    nn_descent: int
    iterative_search: int

class GraphIndexBuildParams:
    """Parameters for GPU graph index construction."""
    intermediate_graph_degree: int
    graph_degree: int
    build_algo: GraphBuildAlgo
    nn_descent_iterations: int
    compression_ratio: float

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class IvfPqBuildParams:
    """Parameters for IVF-PQ index construction."""
    n_lists: int
    pq_dim: int
    pq_bits: int
    add_data_on_build: bool
    kmeans_n_iters: int
    kmeans_trainset_fraction: float

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class GpuIndexBuildParams:
    """GPU index build parameter variant."""
    params: Union[GraphIndexBuildParams, IvfPqBuildParams]

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class KernelTiming:
    """Recorded GPU kernel timing."""
    kernel_name: str
    work_items: int

    @property
    def duration_us(self) -> int:
        """Duration in microseconds."""

    def __repr__(self) -> str: ...

class GpuConfig:
    """GPU acceleration configuration."""

    def __init__(self) -> None: ...

    policy: GpuPolicy
    preferred_backend: str
    device_index: int
    build_mode: IndexBuildMode
    algorithm: GpuIndexAlgorithm
    device_memory_pool_mb: int
    pinned_host_pool_mb: int
    fp16_search: bool
    unified_memory: bool
    batch_window_us: int
    max_batch_size: int
    ef_search: int
    precision: GpuPrecision
    profiling: bool
    auto_rebuild_on_startup: bool
    rebuild_threshold_ratio: float
    emit_kernel_timings: bool
    graph_params: GraphIndexBuildParams
    ivf_pq_params: IvfPqBuildParams

    def __repr__(self) -> str: ...

class GpuDeviceInfo:
    """Information about the active accelerator device or CPU fallback."""

    def __init__(self) -> None: ...

    name: str
    vendor: str
    backend: str
    device_index: int
    total_memory_bytes: int
    free_memory_bytes: int
    has_unified_memory: bool
    supports_fp16: bool
    supports_bf16: bool
    supports_int8: bool
    supports_cagra: bool
    supports_ivf_pq: bool
    supports_dynamic_batching: bool
    supports_half_precision_search: bool
    compute_capability_major: int
    compute_capability_minor: int
    max_threads_per_block: int
    multiprocessor_count: int
    shared_memory_per_block_bytes: int
    l2_cache_bytes: int
    peak_tflops_fp32: float
    peak_tflops_fp16: float
    host_to_device_bandwidth_gb_s: float
    device_to_host_bandwidth_gb_s: float

    @property
    def memory_gb(self) -> float:
        """Total device memory in gigabytes."""

    def __repr__(self) -> str: ...

class GpuMetricsSnapshot:
    """Snapshot of GPU runtime metrics."""

    def __init__(self) -> None: ...

    backend: str
    device_name: str
    device_memory_used_bytes: int
    device_memory_total_bytes: int
    index_build_count: int
    index_build_time_total_ms: int
    index_build_speedup_vs_cpu_avg: float
    search_kernel_launches_total: int
    search_p50_latency_us: int
    search_p99_latency_us: int
    batch_avg_size: float
    batch_coalescing_ratio: float
    fp16_search_enabled: bool
    fallback_events_total: int
    kernel_errors_total: int
    pinned_memory_pool_used_bytes: int

    def __repr__(self) -> str: ...

class DocumentAttachment:
    """Attached source document for a record."""

    def __init__(
        self,
        text: str = ...,
        uri: str = ...,
        mime_type: str = ...,
    ) -> None: ...

    text: str
    uri: str
    mime_type: str

    def __repr__(self) -> str: ...

class ChunkInfo:
    """Chunk lineage information for a record."""

    def __init__(self) -> None: ...

    document_key: str
    ordinal: int
    char_start: int
    char_end: int

    def __repr__(self) -> str: ...

class EmbeddingLineage:
    """Embedding provenance for a record."""

    def __init__(self) -> None: ...

    provider: str
    model: str
    revision: str
    attributes: dict[str, MetaValue]

    def __repr__(self) -> str: ...

class QueryPlan:
    """Plan selected by the vault query planner."""

    def __init__(self) -> None: ...

    strategy: QueryStrategy
    candidate_count: int
    metadata_accelerated: bool
    gpu_index: bool
    index_type: str

    def __repr__(self) -> str: ...

# -- VaultInfo ----------------------------------------------------------------

class VaultInfo:
    """Summary statistics for a vault."""

    @property
    def count(self) -> int:
        """Number of records in the vault."""

    @property
    def dimension(self) -> int:
        """Vector dimension of the vault."""

    @property
    def metric(self) -> str:
        """Similarity metric used by the vault (cosine|euclidean|dot_product)."""

    def __repr__(self) -> str: ...

# -- Result -------------------------------------------------------------------

class Result:
    """A single result from a seek() or query() call."""

    @property
    def id(self) -> str:
        """Record identifier (UUIDv7 hex string)."""

    distance: float
    """Distance from the query vector (smaller = more similar)."""

    @property
    def data(self) -> dict[str, MetaValue]:
        """Metadata payload attached to the record."""

    document: Optional[DocumentAttachment]
    chunk: Optional[ChunkInfo]
    lineage: Optional[EmbeddingLineage]

    def __repr__(self) -> str: ...

# -- Filter -------------------------------------------------------------------

class Filter:
    """Metadata filter for search and scan operations.

    Uses a fluent builder pattern for chaining predicates. Chained predicates
    are AND-ed together. Combinator methods (and_, or_, not_) construct
    boolean expressions.

    Example:
        f = (Filter()
             .field("category").equals("tech")
             .field("score").gte(0.8)
             .field("country").one_of(["US", "GB"]))

        either = Filter().field("tier").equals("pro").or_(
            Filter().field("year").gte(2023))
    """

    def __init__(self) -> None: ...
    def field(self, name: str) -> "Filter": ...
    def equals(self, value: MetaValue) -> "Filter": ...
    def not_equals(self, value: MetaValue) -> "Filter": ...
    def lt(self, value: MetaValue) -> "Filter": ...
    def le(self, value: MetaValue) -> "Filter": ...
    def gt(self, value: MetaValue) -> "Filter": ...
    def gte(self, value: MetaValue) -> "Filter": ...
    def one_of(self, values: Iterable[MetaValue]) -> "Filter": ...
    def contains(self, substring: str) -> "Filter": ...
    def and_(self, other: "Filter") -> "Filter": ...
    def or_(self, other: "Filter") -> "Filter": ...

    @staticmethod
    def not_(inner: "Filter") -> "Filter": ...

    def __repr__(self) -> str: ...

# -- TransactionVault ---------------------------------------------------------

class TransactionVault:
    """Vault-scoped handle for operations within a transaction.

    Buffers writes into the owning Transaction. The transaction must be
    committed for changes to be applied.
    """

    def place(
        self,
        vector: Vector,
        data: PayloadLike = ...,
        id: Optional[str] = ...,
    ) -> str: ...
    def erase(self, id: str) -> None: ...

# -- Transaction --------------------------------------------------------------

class Transaction:
    """Atomic, all-or-nothing batch of writes.

    Operations are buffered and applied only on commit(). An un-committed
    transaction is rolled back on destruction (or on context-manager exit
    with an exception).

    Use as a context manager for automatic commit/rollback::

        with db.begin_transaction() as txn:
            v = txn.vault("docs")
            v.place([1.0, 2.0], {"title": "A"})
            v.place([3.0, 4.0], {"title": "B"})
            # Committed automatically on exit
    """

    def vault(self, name: str) -> TransactionVault: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def __enter__(self) -> "Transaction": ...
    def __exit__(self, *args: Any) -> bool: ...

# -- Vault --------------------------------------------------------------------

class Vault:
    """A named partition of records within a database.

    Owns its index and the authoritative record store. Obtained via
    ``db.vault("name")``.
    """

    @property
    def name(self) -> str:
        """The vault's name."""

    def place(
        self,
        vector: Vector,
        data: PayloadLike = ...,
        id: Optional[str] = ...,
        document: Optional[DocumentAttachment] = ...,
        chunk: Optional[ChunkInfo] = ...,
        lineage: Optional[EmbeddingLineage] = ...,
    ) -> str:
        """Ingest a single record. Returns the assigned UUIDv7 id.

        Args:
            vector: The embedding vector (list or tuple of floats).
            data: Optional metadata payload (dict of str -> int/float/bool/str).
            id: Optional custom UUIDv7 record ID.
            document: Optional source document attachment.
            chunk: Optional chunk lineage.
            lineage: Optional embedding provenance.

        Returns:
            The record's ID as a hex string.
        """

    def place_document(
        self,
        text: str,
        data: PayloadLike = ...,
        id: Optional[str] = ...,
        chunk: Optional[ChunkInfo] = ...,
        lineage: Optional[EmbeddingLineage] = ...,
    ) -> str:
        """Embed and ingest a text document using the configured text embedder."""

    def place_many(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Batch-ingest records.

        Each record is a dict with:
            vector: list[float]    (required)
            text: str              (optional, requires native text embedder or wrapper embedder)
            data: dict             (optional)
            id: str                (optional)
            document: DocumentAttachment (optional)
            chunk: ChunkInfo       (optional)
            lineage: EmbeddingLineage (optional)

        Example:
            vault.place_many([
                {"vector": [1.0, 2.0], "data": {"t": 1}},
                {"vector": [3.0, 4.0], "data": {"t": 2}},
            ])
        """

    def seek(
        self,
        vector: Vector,
        top: int = ...,
        where: Filter = ...,
        threshold: Optional[float] = ...,
    ) -> list[Result]:
        """Top-k nearest neighbors sorted ascending by distance.

        Args:
            vector: The query vector.
            top: Number of results to return.
            where: Optional metadata filter.
            threshold: Optional maximum distance for range search.

        Returns:
            List of Result objects sorted by distance (closest first).
        """

    def seek_text(
        self,
        text: str,
        top: int = ...,
        where: Filter = ...,
        threshold: Optional[float] = ...,
    ) -> list[Result]:
        """Query using text directly."""

    def seek_hybrid(
        self,
        vector: Vector,
        text: str,
        top: int = ...,
        where: Filter = ...,
        threshold: Optional[float] = ...,
        lexical_weight: float = ...,
    ) -> list[Result]:
        """Blend vector similarity with lexical overlap over attached documents."""

    def explain_seek(
        self,
        vector: Vector,
        top: int = ...,
        where: Filter = ...,
        threshold: Optional[float] = ...,
        has_text_component: bool = ...,
    ) -> QueryPlan:
        """Return the planner decision for a query shape."""

    def fetch(self, id: str) -> Optional[dict[str, Any]]:
        """Fetch a record's full data by ID.

        Returns:
            A dict with ``id``, ``vector``, and ``data`` keys, or None if
            the record does not exist.
        """

    def erase(self, id: str) -> bool:
        """Remove a record by ID. Returns False if not found."""

    def scan(
        self,
        where: Filter = ...,
        offset: int = ...,
        limit: int = ...,
    ) -> list[dict[str, Any]]:
        """Iterate records matching a filter in insertion order.

        Args:
            where: Optional metadata filter.
            offset: Number of matching records to skip.
            limit: Maximum records to return (-1 = all).

        Returns:
            List of dicts with ``id`` and ``data`` keys.
        """

    def info(self) -> VaultInfo:
        """Return summary statistics (count, dimension, metric)."""

    def count(self) -> int:
        """Return the number of records in this vault."""

    def rebuild_index(self) -> None:
        """Rebuild the backing index from authoritative stored records."""

    def __repr__(self) -> str: ...

# -- Database -----------------------------------------------------------------

class Database:
    """Top-level database handle. One per directory. Owns all vaults.

    Use as a context manager for automatic checkpoint + lock release::

        with elips.open("/data/vectors", dimension=384) as db:
            docs = db.vault("documents")
            ...
    """

    def vault(self, name: str) -> Vault:
        """Access or lazily create a vault by name."""

    def list_vaults(self) -> list[str]:
        """List all vault names in the database."""

    def begin_transaction(self) -> Transaction:
        """Begin an atomic write transaction."""

    def checkpoint(self) -> None:
        """Flush all state to disk (no-op for in-memory databases)."""

    def compact(self) -> None:
        """Compact persistent state and rebuild indexes."""

    def close(self) -> None:
        """Checkpoint and release the lock."""

    def abandon(self) -> None:
        """Drop handle without checkpointing (simulates crash exit).

        Only the WAL remains on disk. The next open() recovers via WAL replay.
        """

    def query(
        self, eql: str, bindings: Mapping[str, Vector] = ...
    ) -> list[Result]:
        """Execute a single EQL statement.

        Args:
            eql: The EQL statement string.
            bindings: Map of variable names to vector values for $bindings.

        Returns:
            List of Result objects.
        """

    def gpu_info(self) -> "GpuDeviceInfo":
        """Return information about the active accelerator device or CPU fallback."""

    def gpu_stats(self) -> "GpuMetricsSnapshot":
        """Return a snapshot of GPU runtime metrics."""

    @property
    def config(self) -> Config:
        """The effective configuration of this database."""

    def __enter__(self) -> "Database": ...
    def __exit__(self, *args: Any) -> None: ...
    def __repr__(self) -> str: ...

# -- Module-level utility functions -------------------------------------------

def distance(metric: Union[str, Metric], a: Vector, b: Vector) -> float:
    """Compute the ordering-normalized distance between two vectors.

    Args:
        metric: One of ``"cosine"``, ``"euclidean"``, ``"dot_product"``, or a Metric enum value.
        a: First vector.
        b: Second vector.

    Returns:
        The distance: smaller = more similar for all metrics.
    """

def requires_normalization(metric: Union[str, Metric]) -> bool:
    """Return True if vectors should be L2-normalized for this metric.

    Args:
        metric: One of ``"cosine"``, ``"euclidean"``, ``"dot_product"``, or a Metric enum value.

    Returns:
        True only for cosine metric.
    """

def metric_to_string(metric: Metric) -> str:
    """Convert a Metric enum value to its string name.

    Args:
        metric: A Metric enum value.

    Returns:
        One of ``"cosine"``, ``"euclidean"``, ``"dot_product"``.
    """

def metric_from_string(name: str) -> Metric:
    """Parse a string into a Metric enum value.

    Args:
        name: One of ``"cosine"``, ``"euclidean"``, ``"dot_product"``.

    Returns:
        The corresponding Metric enum value.

    Raises:
        ValueError: If the name is not a recognized metric.
    """

def validate_eql(source: str) -> None:
    """Validate an EQL statement string without executing it.

    Args:
        source: EQL source string.

    Returns:
        None if the statement is syntactically valid.

    Raises:
        ParseError: On invalid EQL syntax.
    """

def tokenize_eql(source: str) -> list[Token]:
    """Tokenize an EQL source string.

    Args:
        source: EQL source string.

    Returns:
        A list of Token objects.
    """

def gpu_devices() -> list[GpuDeviceInfo]:
    """Probe all compile-time-enabled GPU backends and return detected devices."""

# -- Module-level factory functions -------------------------------------------

def open(
    path: str,
    dimension: int = ...,
    metric: str = ...,
    index: str = ...,
    access_mode: str = ...,
    gpu: Optional[GpuConfig] = ...,
) -> Database:
    """Open (or create) a database with simple parameters.

    Args:
        path: Filesystem path, or ``\":memory:\"`` for ephemeral.
        dimension: Vector dimension (required for new databases).
        metric: Similarity metric (``\"cosine\"``, ``\"euclidean\"``,
            ``\"dot_product\"``).
        index: Index backend (``\"graph\"`` for HNSW, ``\"exact\"`` for brute-force).
        access_mode: ``\"read_write\"`` or ``\"read_only\"``.
        gpu: Optional GPU runtime configuration applied before open.

    Returns:
        A Database handle.
    """

def open_with_config(path: str, config: Config) -> Database:
    """Open (or create) a database with a full Config builder.

    Args:
        path: Filesystem path, or ``\":memory:\"`` for ephemeral.
        config: A configured Config instance.

    Returns:
        A Database handle.
    """
