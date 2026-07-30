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
          "Complete reference for ELIPS Python exceptions: ElipsError, LockConflictError, ValidationError, TransactionError, DiskError, and GpuError handling strategies.",
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
        { id: "lock-conflict", label: "LockConflictError" },
        { id: "validation-error", label: "ValidationError" },
        { id: "transaction-error", label: "TransactionError" },
        { id: "disk-gpu-errors", label: "DiskError & GpuError" },
        { id: "best-practices", label: "Retry Strategies & Best Practices" },
      ]}
    >
      <p className="text-[18px] text-ink">
        All Python exceptions raised by ELIPS derive from <code>elips.ElipsError</code>. The C++ engine exceptions are mapped cleanly to idiomatic Python exception classes.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        ELIPS uses specific exception types to allow precise error catching and retry logic for concurrent file locks, invalid vector dimensions, transaction aborted rollbacks, or disk space issues.
      </p>

      <h2 id="exception-hierarchy">Exception Hierarchy</h2>
      <CodeBlock lang="python">{`ElipsError
├── LockConflictError       # Raised when file lock cannot be acquired
├── ValidationError          # Raised on dimension mismatch or malformed filter
├── TransactionError        # Raised when a transaction commit fails
├── VaultSealedError        # Raised on write attempt to closed/sealed vault
├── ReadOnlyError           # Raised on mutation attempt in read-only mode
├── DiskError               # Raised on WAL or storage I/O errors
└── GpuError                # Raised on GPU device memory allocation failure`}</CodeBlock>

      <h2 id="lock-conflict">LockConflictError</h2>
      <p>
        Raised when another OS process holds an exclusive write lock on the database directory.
      </p>
      <CodeBlock lang="python">{`import elips
import time

def acquire_with_retry(db_path: str, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            return elips.connect(db_path, dimension=128)
        except elips.LockConflictError:
            print(f"Database locked by another process. Retrying in 1s ({attempt+1}/{max_retries})...")
            time.sleep(1.0)
    raise RuntimeError("Failed to acquire database lock.")`}</CodeBlock>

      <h2 id="validation-error">ValidationError</h2>
      <p>
        Raised when attempting to insert a vector whose dimension does not match the vault configuration, or when supplying a invalid metadata filter construct.
      </p>
      <CodeBlock lang="python">{`with elips.connect(":memory:", dimension=128) as engine:
    arena = engine.arena("vectors")
    try:
        # Invalid 3-dimensional vector passed to 128-dim vault
        arena.write(vector=[1.0, 2.0, 3.0])
    except elips.ValidationError as e:
        print(f"Validation failed: {e}")`}</CodeBlock>

      <h2 id="transaction-error">TransactionError</h2>
      <p>
        Raised during <code>txn.commit()</code> if a batched write fails or a constraint is violated.
      </p>

      <h2 id="disk-gpu-errors">DiskError & GpuError</h2>
      <ul>
        <li>
          <code>DiskError</code> — Triggered by disk full events, permission denied on storage files, or corrupt WAL sequence numbers.
        </li>
        <li>
          <code>GpuError</code> — Triggered by CUDA out-of-memory (`cudaErrorMemoryAllocation`) or missing GPU hardware drivers.
        </li>
      </ul>

      <h2 id="best-practices">Retry Strategies & Best Practices</h2>
      <p>
        In multi-process worker environments (such as Gunicorn or Celery), wrap database opening in a <code>LockConflictError</code> retry loop. For transaction blocks, always catch <code>TransactionError</code> and invoke explicit rollback.
      </p>
    </DocsShell>
  );
}
