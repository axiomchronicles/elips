from typing import Any, Iterable, Mapping, Sequence, Union

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

    @property
    def dimension_val(self) -> int: ...
    @property
    def metric_val(self) -> str: ...
    @property
    def index_val(self) -> str: ...
    @property
    def graph_params_val(self) -> GraphParams: ...

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
        id: str | None = ...,
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
        id: str | None = ...,
    ) -> str:
        """Ingest a single record. Returns the assigned UUIDv7 id.

        Args:
            vector: The embedding vector (list or tuple of floats).
            data: Optional metadata payload (dict of str -> int/float/bool/str).
            id: Optional custom UUIDv7 record ID.

        Returns:
            The record's ID as a hex string.
        """

    def place_many(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Batch-ingest records.

        Each record is a dict with:
            vector: list[float]    (required)
            data: dict             (optional)
            id: str                (optional)

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
        threshold: float | None = ...,
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

    def fetch(self, id: str) -> dict[str, Any] | None:
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

    @property
    def config(self) -> Config:
        """The effective configuration of this database."""

    def __enter__(self) -> "Database": ...
    def __exit__(self, *args: object) -> None: ...
    def __repr__(self) -> str: ...

# -- Module-level functions ---------------------------------------------------

def open(
    path: str,
    dimension: int = ...,
    metric: str = ...,
    index: str = ...,
) -> Database:
    """Open (or create) a database with simple parameters.

    Args:
        path: Filesystem path, or ``\":memory:\"`` for ephemeral.
        dimension: Vector dimension (required for new databases).
        metric: Similarity metric (``\"cosine\"``, ``\"euclidean\"``,
            ``\"dot_product\"``).
        index: Index backend (``\"graph\"`` for HNSW, ``\"exact\"`` for brute-force).

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