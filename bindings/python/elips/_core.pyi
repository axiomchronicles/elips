from enum import IntEnum
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypedDict,
    Union,
)

# -- Type aliases --------------------------------------------------------------

MetaValue = Union[bool, int, float, str]
Vector = Sequence[float]
PayloadLike = Mapping[str, MetaValue]

MetricName = Literal["cosine", "euclidean", "dot_product"]
IndexName = Literal["graph", "exact"]
DurabilityName = Literal["paranoid", "standard", "relaxed", "ephemeral"]
AccessModeName = Literal["read_write", "read_only"]
ComparatorName = Literal["eq", "ne", "lt", "le", "gt", "ge", "gte"]
CodecName = Literal["none", "pq", "opq", "sq8"]

class StoredRecord(TypedDict):
    """Record mapping returned by :meth:`Vault.fetch`, :meth:`Vault.scan`, and
    :meth:`Vault.records`."""

    id: str
    vector: tuple[float, ...]
    data: dict[str, MetaValue]
    document: Optional["DocumentAttachment"]
    chunk: Optional["ChunkInfo"]
    lineage: Optional["EmbeddingLineage"]
    approximate: bool
    codec: CodecName

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
    cosine = ...
    euclidean = ...
    dot_product = ...

class IndexType(IntEnum):
    """Index backends."""
    graph = ...
    exact = ...

class Durability(IntEnum):
    """Durability levels trading write throughput against crash safety."""
    paranoid = ...
    standard = ...
    relaxed = ...
    ephemeral = ...

class Comparator(IntEnum):
    """Metadata comparison operators."""
    eq = ...
    ne = ...
    lt = ...
    le = ...
    gt = ...
    ge = ...

class AccessMode(IntEnum):
    """Database access mode."""
    read_write = ...
    read_only = ...

class QueryStrategy(IntEnum):
    """Planner strategy chosen for a query."""
    ann_index = ...
    exact_candidates = ...
    full_scan = ...
    text_probe = ...
    hybrid_fusion = ...

class TextEmbedderKind(IntEnum):
    """Runtime kind of text embedder attached to the database."""
    external = ...
    local_builtin = ...

# -- EQL token types ----------------------------------------------------------

class TokenKind(IntEnum):
    """EQL token categories."""
    word = ...
    number = ...
    string = ...
    punct = ...
    end = ...

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
        compaction_ratio: float = 0.2,
    ) -> None: ...

    max_connections: int
    """Maximum number of connections per node (M)."""

    ef_construction: int
    """Beam width during index construction."""

    ef_search: int
    """Beam width during search."""

    compaction_ratio: float
    """Tombstone fraction that triggers automatic compaction; 0.0 disables it."""

    def __repr__(self) -> str: ...

class Codec:
    """Vector compression codec. Values are persisted on disk."""

    none: Codec
    pq: Codec
    opq: Codec
    sq8: Codec

class QuantParams:
    """Vector compression parameters.

    Selecting a codec does not compress anything on its own: a codebook has to
    be trained on real data first, which is what ``Vault.quantize()`` does.
    """

    def __init__(
        self,
        codec: CodecName = "none",
        pq_dim: int = 0,
        pq_bits: int = 8,
        train_iters: int = 10,
        opq_iters: int = 4,
    ) -> None: ...

    codec: CodecName
    """Codec name: ``none``, ``pq``, ``opq``, or ``sq8``."""

    pq_dim: int
    """Subspace count, also the code width in bytes. 0 selects automatically."""

    pq_bits: int
    """Bits per subquantizer code, 4 to 8."""

    train_iters: int
    """Lloyd iterations per subspace codebook."""

    opq_iters: int
    """Rotation/codebook alternations; ``opq`` only."""

    @property
    def codec_enum(self) -> Codec: ...
    def code_bytes(self, dimension: int) -> int: ...
    def validate(self, dimension: int) -> None: ...
    def __repr__(self) -> str: ...

class LocalEmbedderConfig:
    """Configuration for the built-in local text embedder."""

    def __init__(
        self,
        model: str = ...,
        revision: str = ...,
        storage_path: str = ...,
        dimension: int = ...,
    ) -> None: ...

    model: str
    revision: str
    storage_path: str
    dimension: int

    def __repr__(self) -> str: ...

class TextEmbedderInfo:
    """Resolved metadata for the configured or expected text embedder."""

    kind: TextEmbedderKind
    provider: str
    model: str
    revision: str
    backend: str
    dimension: int
    fingerprint: str
    storage_path: str
    rehydratable: bool
    loaded: bool
    auto_attached: bool

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
    def quantization(self, params: QuantParams) -> "Config": ...
    def durability(self, level: str) -> "Config": ...
    def access_mode(self, mode: str) -> "Config": ...
    def segmented_storage(self, enabled: bool) -> "Config": ...
    def metadata_acceleration(self, enabled: bool) -> "Config": ...
    def auto_text_embedder(self, enabled: bool) -> "Config": ...
    def local_text_embedder(
        self,
        config: LocalEmbedderConfig = ...,
    ) -> "Config": ...
    def text_embedder(
        self,
        embedder: Callable[[Sequence[str]], Sequence[Vector]],
        provider: str = ...,
        model: str = ...,
        revision: str = ...,
        dimension: int = ...,
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
    def quantization_val(self) -> QuantParams:
        """Get the configured quantization parameters."""
    @property
    def has_quantization(self) -> bool:
        """True when a compression codec is configured."""
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
    def auto_text_embedder_enabled(self) -> bool:
        """Return whether automatic default local embedding is enabled."""
    @property
    def durability_val(self) -> str:
        """Get the durability level as a string."""
    @property
    def has_gpu(self) -> bool:
        """True when a GPU policy other than CPU-only is configured."""
    @property
    def has_pending_local_text_embedder(self) -> bool:
        """True when a local embedder is configured but not yet instantiated."""
    @property
    def local_text_embedder_config(self) -> Optional[LocalEmbedderConfig]:
        """The pending local embedder configuration, or None."""
    @property
    def has_text_embedder(self) -> bool:
        """Return whether a text embedder is configured."""
    @property
    def text_embedder_info(self) -> Optional[TextEmbedderInfo]:
        """Return resolved metadata for the current or expected text embedder."""
    @property
    def gpu_val(self) -> Optional["GpuConfig"]:
        """Get the GPU configuration if set, else None."""

    def __repr__(self) -> str: ...

# -- GPU enums and types ------------------------------------------------------

class GpuPolicy(IntEnum):
    """GPU usage policy."""
    auto = ...
    prefer_gpu = ...
    require_gpu = ...
    cpu_only = ...
    specific = ...

class IndexBuildMode(IntEnum):
    """GPU index build vs. serve mode."""
    gpu_build_cpu_serve = ...
    gpu_build_gpu_serve = ...
    hybrid = ...

class GpuIndexAlgorithm(IntEnum):
    """GPU index algorithm selection."""
    auto = ...
    cagra = ...
    ivf_flat = ...
    ivf_pq = ...
    brute_force = ...

class GpuPrecision(IntEnum):
    """GPU computation precision."""
    fp32 = ...
    fp16 = ...
    int8 = ...
    auto = ...

class GpuError(IntEnum):
    """GPU error codes."""
    device_not_found = ...
    insufficient_memory = ...
    kernel_launch_failed = ...
    transfer_failed = ...
    index_build_failed = ...
    unsupported_metric = ...
    initialization_failed = ...
    backend_unavailable = ...

class GraphBuildAlgo(IntEnum):
    """Graph index build algorithm."""
    ivf_pq = ...
    nn_descent = ...
    iterative_search = ...

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

    @property
    def codec(self) -> CodecName:
        """Active compression codec (none|pq|opq|sq8)."""

    @property
    def code_bytes(self) -> int:
        """Bytes per stored vector, or 0 when uncompressed."""

    @property
    def compression_ratio(self) -> float:
        """Stored bytes saved per vector against fp32; 1.0 when uncompressed."""

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

    @property
    def approximate(self) -> bool:
        """True when distance is estimated from a compressed vector."""

    @property
    def codec(self) -> CodecName:
        """Codec that produced this hit's stored vector, or ``none``."""

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

    @staticmethod
    def compare(
        field: str,
        op: Union[Comparator, ComparatorName],
        value: MetaValue,
    ) -> "Filter":
        """Build a single comparison predicate."""

    @staticmethod
    def in_set(field: str, values: Iterable[MetaValue]) -> "Filter":
        """Build a predicate matching records whose field equals any value."""

    @staticmethod
    def has_substring(field: str, substring: str) -> "Filter":
        """Build a predicate matching records whose string field contains substring."""

    def matches(self, payload: PayloadLike) -> bool:
        """Evaluate this filter against a payload dict, without touching the database."""

    def matches_all(self) -> bool:
        """True when this filter is empty and therefore matches every record."""

    def exact_constraints(self) -> Optional[list[tuple[str, list[MetaValue]]]]:
        """Equality constraints resolvable via the metadata index, or None."""

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
        """Query using text directly. Requires a configured text embedder."""

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

    def vacuum(self) -> None:
        """Reclaim index space held by deleted records."""

    def quantize(self) -> None:
        """Train a codebook over this vault and compress it in place.

        Requires a codec on the config. Raises :class:`ConfigError` if none is
        set, if the vault is empty, or if it is already quantized.
        """

    @property
    def quantized(self) -> bool:
        """True once :meth:`quantize` has compressed this vault."""

    @property
    def codec(self) -> CodecName:
        """Active codec name, or ``none`` while uncompressed."""

    def records(self) -> list[StoredRecord]:
        """Snapshot every record in this vault.

        Copies the whole vault; prefer :meth:`scan` with a filter or limit for
        large vaults.
        """

    def set_read_only(self, read_only: bool) -> None:
        """Toggle runtime read-only mode. Mutations then raise StorageError."""

    @property
    def read_only(self) -> bool:
        """True when this vault refuses mutations."""

    @property
    def sealed(self) -> bool:
        """True once the owning database has been closed.

        Writes to a sealed vault raise rather than silently failing to persist.
        """

    @property
    def pending_removals(self) -> int:
        """Records deleted from the index but not yet reclaimed by vacuum()."""

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

    def vacuum(self) -> None:
        """Reclaim tombstoned index space across every vault.

        Unlike :meth:`compact`, this does not rewrite the on-disk snapshot and
        works on in-memory databases.
        """

    def quantize(self, vault: str) -> None:
        """Compress one vault, then checkpoint so the codebook is durable."""

    def quantize_all(self) -> None:
        """Compress every vault in this database."""

    def close(self) -> None:
        """Checkpoint, release the lock, and seal every vault against writes."""

    def abandon(self) -> None:
        """Drop handle without checkpointing (simulates crash exit).

        Only the WAL remains on disk. The next open() recovers via WAL replay.
        """

    @property
    def path(self) -> str:
        """The directory this database was opened from, or ``":memory:"``."""

    @property
    def persistent(self) -> bool:
        """False for in-memory databases, which have no WAL or snapshot."""

    @property
    def closed(self) -> bool:
        """True once :meth:`close` or :meth:`abandon` has run."""

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
    """Probe every GPU backend compiled into this build.

    Returns:
        One :class:`GpuDeviceInfo` per usable device. An empty list is the
        normal result on CPU-only machines and should be treated as "run on
        the CPU", not as an error.

    Example:
        Pick a device and size an index to it before ingesting::

            import elips

            devices = elips.gpu_devices()
            if not devices:
                print("no GPU; the CPU HNSW index will be used")
            else:
                dev = devices[0]
                print(f"{dev.name} ({dev.backend}) {dev.memory_gb:.1f} GiB")

                cfg = elips.GpuConfig()
                cfg.policy = elips.GpuPolicy.prefer_gpu
                cfg.algorithm = elips.GpuIndexAlgorithm.ivf_pq

                # 2M vectors of 768 floats: will it fit in VRAM?
                if elips.gpu_can_fit_index(dev, 2_000_000, 768, cfg):
                    db = elips.open("/data/vectors", dimension=768, gpu=cfg)
                else:
                    cfg.algorithm = elips.GpuIndexAlgorithm.ivf_pq
                    cfg.ivf_pq_params.pq_bits = 4   # compress harder
                    db = elips.open("/data/vectors", dimension=768, gpu=cfg)
    """

def gpu_cpu_fallback_info() -> GpuDeviceInfo:
    """Return the synthetic device info describing the CPU fallback path."""

def gpu_runtime_device_info() -> GpuDeviceInfo:
    """Return info for the device this process would select right now.

    Falls back to :func:`gpu_cpu_fallback_info` when no GPU is present, so the
    result is always usable for logging.
    """

def gpu_can_fit_index(
    device: GpuDeviceInfo,
    n_vectors: int,
    dimension: int,
    config: GpuConfig = ...,
) -> bool:
    """Check whether an index of the given shape fits in device memory.

    Args:
        device: Target device, from :func:`gpu_devices`.
        n_vectors: Number of vectors the index will hold.
        dimension: Vector dimensionality.
        config: GPU configuration; precision and pool settings affect the estimate.

    Returns:
        True when the index is expected to fit.

    Use this for capacity planning before a large ingest, rather than
    discovering the limit part-way through a multi-hour build.
    """

def gpu_error_message(error: GpuError) -> str:
    """Return the human-readable name of a :class:`GpuError` value."""

def gpu_select(config: GpuConfig = ...) -> Optional[GpuDevice]:
    """Select and initialize a GPU backend.

    Args:
        config: Selection policy and tuning. The default picks the best
            available device.

    Returns:
        A :class:`GpuDevice` handle, or ``None`` when no compatible device is
        present. Check for ``None`` rather than assuming a device exists.

    The handle owns the backend independently of any :class:`Database`, so it
    can be used for standalone GPU work.

    Example:
        Brute-force top-k over an in-memory matrix, entirely on the GPU --
        the FAISS ``IndexFlatIP`` pattern, without an index object::

            import elips

            device = elips.gpu_select()
            if device is None:
                raise SystemExit("this example needs a GPU")

            with device:
                # 1. Size the pool up front so the suballocator does not have
                #    to grow mid-run.
                device.memory.initialize(512 * 1024 * 1024)

                # 2. Your corpus and queries as plain nested sequences.
                corpus = [[0.1, 0.2, 0.9], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]
                queries = [[0.1, 0.2, 0.88], [0.85, 0.15, 0.0]]

                # 3. One kernel launch computes every pairwise distance.
                #    Result is len(queries) rows x len(corpus) columns.
                dists = device.compute_distances(queries, corpus, metric="cosine")

                # 4. A second kernel selects the nearest 2 per row.
                indices, values = device.top_k(dists, k=2)
                for qi, (row_idx, row_val) in enumerate(zip(indices, values)):
                    print(f"query {qi}: {list(zip(row_idx, row_val))}")

                # 5. Telemetry: what did that cost?
                print("peak VRAM bytes:", device.memory.peak_bytes_used)
                print("kernel launches:", device.profiler.total_launches)
    """

# -- GPU device handle --------------------------------------------------------

class GpuMemory:
    """Read-only view of a device memory pool.

    Obtained from :attr:`GpuDevice.memory`. Allocation itself stays in C++: a
    Python-held device pointer that outlives its pool is unrecoverable, so only
    pool sizing and telemetry are exposed.
    """

    def initialize(self, pool_bytes: int = 0) -> None:
        """Size the suballocator's pool. ``0`` uses 80% of device memory."""

    @property
    def bytes_used(self) -> int:
        """Bytes currently handed out to callers."""
    @property
    def bytes_available(self) -> int:
        """Bytes still obtainable: free list plus uncommitted pool headroom."""
    @property
    def peak_bytes_used(self) -> int:
        """High-water mark of :attr:`bytes_used`."""
    def __repr__(self) -> str: ...

class GpuProfiler:
    """Per-kernel timing log. Obtained from :attr:`GpuDevice.profiler`."""

    def record(self, kernel: str, duration_us: int, work_items: int = 0) -> None:
        """Record a kernel execution, so caller GPU work appears alongside the engine's."""

    def recent_timings(self, max_count: int = 100) -> list[KernelTiming]:
        """Return up to ``max_count`` recent timings, newest last."""

    @property
    def total_launches(self) -> int:
        """Total kernel launches recorded."""
    def clear(self) -> None:
        """Discard all recorded timings."""
    def __repr__(self) -> str: ...

class BatchStats:
    """Coalescing statistics from the dynamic query batcher."""

    def __init__(self) -> None: ...

    queries_coalesced: int
    """Queries merged into shared kernel launches."""
    kernel_launches: int
    """Kernel launches issued."""
    avg_batch_size: float
    """Mean queries per launch."""
    p99_latency_us: float
    """99th-percentile end-to-end batch latency."""

    def __repr__(self) -> str: ...

class GpuDevice:
    """A live handle to a selected GPU backend.

    Obtain one with :func:`gpu_select`. The handle owns the backend, so keep it
    alive while in use and close it when done -- ideally via ``with``.

    Raw device allocation, upload, and download are intentionally absent: they
    take a caller-supplied byte count against a raw pointer, where a wrong
    value corrupts device memory instead of raising. The kernels below derive
    every size from the arrays passed in.
    """

    @property
    def device_info(self) -> GpuDeviceInfo:
        """The device metadata for this backend."""
    @property
    def available(self) -> bool:
        """True while the backend is usable."""
    @property
    def idle(self) -> bool:
        """True when no work is outstanding on the device."""
    @property
    def backend(self) -> str:
        """Backend name, e.g. ``"metal"`` or ``"cuda"``."""
    @property
    def memory(self) -> GpuMemory:
        """Memory-pool telemetry for this device."""
    @property
    def profiler(self) -> GpuProfiler:
        """Kernel timing log for this device."""

    def synchronize(self) -> None:
        """Block until all outstanding device work completes."""

    def compute_distances(
        self,
        queries: Iterable[Vector],
        database: Iterable[Vector],
        metric: Union[str, Metric] = "cosine",
    ) -> list[list[float]]:
        """Compute all pairwise distances between two batches on the GPU.

        Args:
            queries: Query vectors, all the same length.
            database: Database vectors, same dimension as ``queries``.
            metric: A :class:`Metric` or ``"cosine"`` / ``"euclidean"`` /
                ``"dot_product"``.

        Returns:
            ``len(queries)`` rows of ``len(database)`` distances,
            ordering-normalized so smaller always means closer.

        Raises:
            DimensionMismatch: If query and database dimensions differ.
            StorageError: If the GPU kernel fails.
        """

    def top_k(
        self,
        distances: Iterable[Sequence[float]],
        k: int,
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Select the k smallest entries per row on the GPU.

        Args:
            distances: Rows of distances, e.g. from :meth:`compute_distances`.
            k: Results per row; must not exceed the row width.

        Returns:
            An ``(indices, values)`` pair of parallel row lists, ascending by
            distance.

        Raises:
            ValueError: If ``k`` exceeds the row width.
            StorageError: If the GPU kernel fails.
        """

    def close(self) -> None:
        """Release the backend and its memory pool. Idempotent."""

    @property
    def closed(self) -> bool:
        """True once :meth:`close` has run."""

    def __enter__(self) -> "GpuDevice": ...
    def __exit__(self, *exc: object) -> None: ...
    def __repr__(self) -> str: ...

# -- EQL abstract syntax tree -------------------------------------------------

class VectorRef:
    """A query vector in a parsed EQL statement.

    Either an inline literal or a named binding supplied at execution time.
    """

    def __init__(self) -> None: ...

    literal: list[float]
    """Inline vector values; empty when a binding is used."""
    binding: str
    """Binding name (``$name`` in EQL); empty for literals."""

    def __repr__(self) -> str: ...

class SearchStatement:
    """A parsed EQL ``seek`` statement."""

    def __init__(self) -> None: ...

    vault: str
    query: VectorRef
    top: Optional[int]
    """Result limit, or None when unspecified."""
    threshold: Optional[float]
    """Maximum distance, or None."""
    where: Filter
    """Metadata filter; matches everything when absent."""
    rank_by: Optional[str]
    """Ranking field, or None to rank by distance."""
    projection: list[str]
    """Requested fields; empty means all fields."""

    def __repr__(self) -> str: ...

class FetchStatement:
    """A parsed EQL ``fetch`` statement."""

    def __init__(self) -> None: ...
    vault: str
    id: str
    def __repr__(self) -> str: ...

class ScanStatement:
    """A parsed EQL ``scan`` statement."""

    def __init__(self) -> None: ...
    vault: str
    where: Filter
    offset: Optional[int]
    limit: Optional[int]
    def __repr__(self) -> str: ...

class InsertStatement:
    """A parsed EQL insert statement."""

    def __init__(self) -> None: ...
    vault: str
    vector: list[float]
    data: dict[str, MetaValue]
    def __repr__(self) -> str: ...

class DeleteStatement:
    """A parsed EQL ``erase`` statement."""

    def __init__(self) -> None: ...
    vault: str
    id: str
    def __repr__(self) -> str: ...

Statement = Union[
    SearchStatement,
    FetchStatement,
    ScanStatement,
    InsertStatement,
    DeleteStatement,
]

def parse_eql(source: str) -> Statement:
    """Parse an EQL statement into its abstract syntax tree.

    Args:
        source: EQL source text.

    Returns:
        One of the statement classes above. Use ``isinstance`` to discriminate.

    Raises:
        ParseError: If the statement is not valid EQL.

    Unlike :func:`validate_eql`, which discards the result, this returns the
    tree, so callers can build linters, rewriters, and query builders.

    Example:
        Reject queries that would scan an entire vault unbounded -- the kind of
        guardrail a multi-tenant service puts in front of user-supplied
        queries::

            import elips

            MAX_TOP = 100

            def check(query: str) -> None:
                stmt = elips.parse_eql(query)          # raises ParseError

                if isinstance(stmt, elips.SearchStatement):
                    # 1. Refuse unbounded result sets.
                    if stmt.top is None or stmt.top > MAX_TOP:
                        raise ValueError(f"seek needs top <= {MAX_TOP}")

                    # 2. Require a filter that the metadata index can serve,
                    #    so the planner does not fall back to a full scan.
                    if stmt.where.matches_all():
                        raise ValueError("seek needs a where clause")
                    if stmt.where.exact_constraints() is None:
                        raise ValueError("where clause is not index-accelerable")

                elif isinstance(stmt, elips.ScanStatement):
                    # 3. Scans must be paginated.
                    if stmt.limit is None:
                        raise ValueError("scan needs a limit")

            check('seek in docs nearest $q top 10 where tenant = "acme" yield')
    """

# -- Index snapshots ----------------------------------------------------------

class IndexSnapshotKind(IntEnum):
    """Which index produced a snapshot."""

    unknown = ...
    exact = ...
    graph = ...
    gpu_brute_force = ...
    gpu_ivf_flat = ...
    gpu_ivf_pq = ...
    gpu_graph = ...
    gpu_hybrid = ...
    gpu_distributed = ...

class IvfSnapshot:
    """Inverted-file clustering state from an IVF index."""

    def __init__(self) -> None: ...
    n_lists: int
    """Number of coarse clusters."""
    n_probe: int
    """Clusters visited per query."""
    centroids: list[float]
    """Row-major cluster centroids."""
    assignments: list[int]
    """Per-vector cluster assignment."""
    def __repr__(self) -> str: ...

class PqSnapshot:
    """Product-quantization codebook and codes."""

    def __init__(self) -> None: ...
    pq_dim: int
    """Number of subquantizers."""
    pq_bits: int
    """Bits per subquantizer code."""
    codebook: list[float]
    """Trained centroids for every subquantizer."""
    codes: list[int]
    """Encoded vectors."""
    def __repr__(self) -> str: ...

class IndexSnapshot:
    """A portable dump of index contents.

    Used to move an index between backends (CPU to GPU and back) or to inspect
    it offline.
    """

    def __init__(self) -> None: ...
    kind: IndexSnapshotKind
    metric: Metric
    dimension: int
    @property
    def ids(self) -> list[str]:
        """Record identifiers, aligned with :attr:`vectors`."""
    vectors: list[float]
    """Row-major vector data."""
    ivf: Optional[IvfSnapshot]
    """IVF clustering state, when present."""
    pq: Optional[PqSnapshot]
    """Product-quantization state, when present."""
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

# -- Write-ahead log ----------------------------------------------------------

class WalOp(IntEnum):
    """Kind of a write-ahead log record."""

    insert = ...
    erase = ...
    insert_ex = ...
    """Insert carrying document, chunk, or lineage attachments."""
    txn_begin = ...
    """Start of a transaction batch."""
    txn_commit = ...
    """End of a transaction batch.

    Records inside an unterminated begin..commit window are discarded on replay.
    """

class WalEntry:
    """One replayed write-ahead log record."""

    @property
    def op(self) -> WalOp: ...
    @property
    def vault(self) -> str: ...
    @property
    def id(self) -> str:
        """Record identifier this entry applies to."""
    @property
    def vector(self) -> tuple[float, ...]:
        """Vector payload; empty for erases and transaction markers."""
    @property
    def data(self) -> dict[str, MetaValue]:
        """Metadata payload."""
    @property
    def document(self) -> Optional[DocumentAttachment]: ...
    @property
    def chunk(self) -> Optional[ChunkInfo]: ...
    @property
    def lineage(self) -> Optional[EmbeddingLineage]: ...
    def __repr__(self) -> str: ...

def replay_wal(path: str) -> list[WalEntry]:
    """Replay a write-ahead log file without opening the database.

    Args:
        path: Path to a ``wal.log`` file.

    Returns:
        :class:`WalEntry` records in log order. Records inside an unterminated
        transaction window are omitted, matching what recovery would apply. A
        corrupt or truncated tail is dropped rather than raised, so a partial
        log still yields its valid prefix.

    Intended for crash forensics and recovery tooling: it answers "what did the
    database actually acknowledge before it died?" while mutating nothing.

    Example:
        After an unclean shutdown, reconcile the log against the recovered
        state before letting an ingest job resume::

            import elips
            from collections import Counter

            entries = elips.replay_wal("/data/vectors/wal.log")

            # 1. What kinds of operations were pending?
            print(Counter(e.op.name for e in entries))

            # 2. Which record IDs did the WAL acknowledge but the last
            #    checkpoint may not have captured?
            acked = {e.id for e in entries if e.op != elips.WalOp.erase}
            erased = {e.id for e in entries if e.op == elips.WalOp.erase}

            # 3. Confirm recovery actually applied them.
            db = elips.open("/data/vectors")
            vault = db.vault(entries[0].vault) if entries else None
            missing = [rid for rid in acked - erased
                       if vault is not None and vault.fetch(rid) is None]
            if missing:
                raise SystemExit(f"{len(missing)} acknowledged writes lost")

            # 4. Re-drive the upstream job from the last acknowledged offset.
            print("recovered cleanly;", len(acked - erased), "live writes")
    """

def describe_local_embedder(
    config: LocalEmbedderConfig = ...,
    fallback_dimension: int = 0,
    auto_attached: bool = False,
) -> TextEmbedderInfo:
    """Describe a local text embedder without instantiating it.

    Args:
        config: The embedder configuration to describe.
        fallback_dimension: Dimension to assume when ``config.dimension`` is 0.
        auto_attached: Whether the embedder would be attached automatically.

    Returns:
        Info with the resolved dimension, fingerprint, and storage path.

    Use this to check compatibility with an existing database before opening
    it, which otherwise raises ``ConfigError`` on a fingerprint mismatch.
    """

def generate_id() -> str:
    """Generate a fresh record identifier.

    Returns:
        A new unique record ID, suitable as the ``id`` argument to
        :meth:`Vault.place`.

    Useful when the caller must know the identifier before the write happens --
    for example to publish it to a queue in the same transaction.
    """

def is_valid_id(id: str) -> bool:
    """Return True when ``id`` is a well-formed record identifier."""

def normalize(vector: Vector) -> tuple[float, ...]:
    """Return ``vector`` scaled to unit length.

    A zero vector is returned unchanged. Cosine similarity requires normalized
    inputs, but :meth:`Vault.place` and :meth:`Vault.seek` already normalize
    internally when the vault metric needs it -- this is for callers doing
    their own pre-processing.
    """

def magnitude(vector: Vector) -> float:
    """Return the Euclidean length (L2 norm) of ``vector``."""

# -- Module-level factory functions -------------------------------------------

def open(
    path: str,
    dimension: int = ...,
    metric: str = ...,
    index: str = ...,
    access_mode: str = ...,
    gpu: Optional[GpuConfig] = ...,
    embedder: Optional[Union[Callable[[Sequence[str]], Sequence[Vector]], LocalEmbedderConfig]] = ...,
    embedder_provider: str = ...,
    embedder_model: str = ...,
    embedder_revision: str = ...,
    use_default_text_embedder: bool = ...,
    quantization: Union[QuantParams, str, None] = ...,
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
        embedder: Optional Python callable embedder or ``LocalEmbedderConfig``.
        embedder_provider: Provider metadata for Python callable embedders.
        embedder_model: Model metadata for Python callable embedders.
        embedder_revision: Revision metadata for Python callable embedders.
        quantization: A :class:`QuantParams`, or a bare codec name such as
            ``"sq8"``. Selects a codec; call ``Vault.quantize()`` to train it.
        use_default_text_embedder: Attach the built-in local embedder automatically
            for new databases when no explicit embedder is supplied.

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
