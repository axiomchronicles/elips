import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/errors")({
  head: () => ({
    meta: [
      { title: "Exceptions & Error Handling — Python API" },
      {
        name: "description",
        content:
          "Complete reference for ELIPS Python exceptions: ElipsError, ConfigError, DimensionMismatch, InvalidVector, LockConflict, NotFound, CollectionNotFoundError, NotTrainedError, BackendUnavailableError, ParseError, and StorageError.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="Exceptions & Error Handling"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "exception-hierarchy", label: "Exception Hierarchy" },
        { id: "config-error", label: "ConfigError" },
        { id: "dimension-mismatch", label: "DimensionMismatch" },
        { id: "invalid-vector", label: "InvalidVector" },
        { id: "lock-conflict", label: "LockConflict" },
        { id: "not-found", label: "NotFound & CollectionNotFoundError" },
        { id: "not-trained", label: "NotTrainedError" },
        { id: "backend-unavailable", label: "BackendUnavailableError" },
        { id: "parse-error", label: "ParseError" },
        { id: "storage-error", label: "StorageError" },
        { id: "best-practices", label: "Retry Strategies & Best Practices" },
      ]}
    >
      <p className="text-[18px] text-ink">
        All Python exceptions raised by ELIPS derive from{" "}
        <code>elips.ElipsError</code>. The C++ engine exceptions are mapped
        cleanly to idiomatic Python exception classes.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        ELIPS uses specific exception types so you can catch precisely and build
        targeted retry logic — for cross-process lock contention, mismatched
        vector dimensions, malformed vectors, missing collections or ids,
        untrained quantizers, absent compute backends, bad EQL, and storage I/O
        failures.
      </p>

      <h2 id="exception-hierarchy">Exception Hierarchy</h2>
      <CodeBlock lang="python">{`ElipsError                      # base class for every ELIPS error
├── ConfigError                 # invalid or conflicting configuration
├── DimensionMismatch           # vector dimension != collection dimension
│                               #   (alias: DimensionMismatchError)
├── InvalidVector               # malformed vector (NaN/inf, wrong element type)
│                               #   (alias: InvalidVectorError)
├── LockConflict                # cross-process write lock could not be acquired
│                               #   (alias: LockConflictError)
├── NotFound                    # id missing on update()/get()
│                               #   (alias: NotFoundError)
├── CollectionNotFoundError     # get_collection() on a collection that does not exist
├── NotTrainedError             # query/compress before the codebook was trained
├── BackendUnavailableError     # a named compute backend (e.g. "cuda") is absent
├── ParseError                  # malformed EQL statement
└── StorageError                # WAL / snapshot / disk I/O failure`}</CodeBlock>

      <h2 id="config-error">ConfigError</h2>
      <p>
        Raised when a configuration value is invalid, or when a persisted
        identity field (dimension, metric, index type) conflicts with the value
        already stored on disk.
      </p>
      <CodeBlock lang="python">{`import elips

# Opening an existing database with a mismatched dimension
try:
    client = elips.PersistentClient("/data/v", dimension=512)  # on-disk value is 768
except elips.ConfigError as e:
    print(f"Config conflict: {e}")`}</CodeBlock>

      <h2 id="dimension-mismatch">DimensionMismatch</h2>
      <p>
        Raised when a vector whose length does not match the collection's
        configured dimension is added or used as a query. Also exported as{" "}
        <code>DimensionMismatchError</code>.
      </p>
      <CodeBlock lang="python">{`with elips.Client(dimension=128) as client:
    docs = client.get_or_create_collection("vectors")
    try:
        # 3-dimensional vector added to a 128-dim collection
        docs.add(vector=[1.0, 2.0, 3.0])
    except elips.DimensionMismatch as e:
        print(f"Dimension mismatch: {e}")`}</CodeBlock>

      <h2 id="invalid-vector">InvalidVector</h2>
      <p>
        Raised when a vector is structurally malformed — for example it contains{" "}
        <code>NaN</code>/<code>inf</code> values or non-numeric elements. Also
        exported as <code>InvalidVectorError</code>.
      </p>
      <CodeBlock lang="python">{`with elips.Client(dimension=2) as client:
    docs = client.get_or_create_collection("vectors")
    try:
        docs.add(vector=[float("nan"), 1.0])
    except elips.InvalidVector as e:
        print(f"Invalid vector: {e}")`}</CodeBlock>

      <h2 id="lock-conflict">LockConflict</h2>
      <p>
        Raised when another OS process holds an exclusive write lock on the
        database directory and a <code>read_write</code> handle cannot be
        acquired. Also exported as <code>LockConflictError</code>.
      </p>
      <CodeBlock lang="python">{`import elips
import time

def open_with_retry(db_path: str, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            return elips.PersistentClient(db_path, dimension=128)
        except elips.LockConflict:
            print(f"Locked by another process; retrying ({attempt + 1}/{max_retries})...")
            time.sleep(1.0)
    raise RuntimeError("Failed to acquire database lock.")`}</CodeBlock>

      <h2 id="not-found">NotFound &amp; CollectionNotFoundError</h2>
      <ul>
        <li>
          <code>NotFound</code> (alias <code>NotFoundError</code>) — raised by{" "}
          <code>collection.update()</code> (and other id-targeted operations)
          when the record id does not exist.
        </li>
        <li>
          <code>CollectionNotFoundError</code> — raised by{" "}
          <code>client.get_collection(name)</code> when the named collection
          does not exist. Use <code>get_or_create_collection()</code> if you
          want lazy creation instead.
        </li>
      </ul>
      <CodeBlock lang="python">{`with elips.Client(dimension=2) as client:
    try:
        client.get_collection("missing")
    except elips.CollectionNotFoundError as e:
        print(f"No such collection: {e}")

    docs = client.get_or_create_collection("docs")
    try:
        docs.update(ids=["does-not-exist"], metadatas=[{"k": 1}])
    except elips.NotFound as e:
        print(f"No such record: {e}")`}</CodeBlock>

      <h2 id="not-trained">NotTrainedError</h2>
      <p>
        Raised when an operation requires a trained quantization codebook that
        has not been built yet — for example querying a quantized collection, or
        calling <code>collection.compress()</code> before{" "}
        <code>train</code> has run.
      </p>
      <CodeBlock lang="python">{`try:
    docs.compress()  # requires a trained codebook
except elips.NotTrainedError as e:
    print(f"Codebook not trained: {e}")`}</CodeBlock>

      <h2 id="backend-unavailable">BackendUnavailableError</h2>
      <p>
        Raised when a named compute backend is requested but is not present in
        the current build or on the current hardware — for example opening with{" "}
        <code>device="cuda:0"</code> on a machine with no CUDA device. Pass{" "}
        <code>fallback=True</code> to fall back to CPU instead of raising.
      </p>
      <CodeBlock lang="python">{`try:
    client = elips.Client(dimension=128, device="cuda:0")
except elips.BackendUnavailableError as e:
    print(f"Backend unavailable: {e}")

# Or request graceful fallback to CPU
client = elips.Client(dimension=128, device="cuda:0", fallback=True)`}</CodeBlock>

      <h2 id="parse-error">ParseError</h2>
      <p>
        Raised when an EQL statement is syntactically invalid. Surfaced by{" "}
        <code>client.query_eql()</code> and the standalone{" "}
        <code>elips.parse_eql()</code> / <code>elips.validate_eql()</code>{" "}
        helpers.
      </p>
      <CodeBlock lang="python">{`try:
    client.query_eql("search in docs nearest")  # incomplete statement
except elips.ParseError as e:
    print(f"Bad EQL: {e}")`}</CodeBlock>

      <h2 id="storage-error">StorageError</h2>
      <p>
        Raised on WAL, snapshot, or disk I/O failures — disk-full conditions,
        permission errors on storage files, or a corrupt WAL sequence. A write
        attempted against a read-only or sealed collection also surfaces as a{" "}
        <code>StorageError</code>.
      </p>
      <CodeBlock lang="python">{`try:
    docs.add(vector=[1.0, 0.0])
except elips.StorageError as e:
    print(f"Storage I/O failure: {e}")`}</CodeBlock>

      <h2 id="best-practices">Retry Strategies &amp; Best Practices</h2>
      <ul>
        <li>
          In multi-process worker environments (Gunicorn, Celery), wrap database
          opening in a <code>LockConflict</code> retry loop with exponential
          back-off, or open secondary processes with{" "}
          <code>access_mode="read_only"</code>.
        </li>
        <li>
          Catch <code>DimensionMismatch</code> and <code>InvalidVector</code> at
          your ingest boundary to reject bad input before it reaches a batch
          write.
        </li>
        <li>
          Prefer <code>get_or_create_collection()</code> over{" "}
          <code>get_collection()</code> when a missing collection should be
          created lazily rather than raise <code>CollectionNotFoundError</code>.
        </li>
        <li>
          Catch the base <code>ElipsError</code> as a last-resort handler so new
          error types are never silently swallowed as generic exceptions.
        </li>
      </ul>
    </DocsShell>
  );
}
