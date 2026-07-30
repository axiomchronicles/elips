import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/transaction")({
  head: () => ({
    meta: [
      { title: "Transaction — ELIPS Python API Docs" },
      {
        name: "description",
        content:
          "Complete reference for the ELIPS Python Transaction API: atomic multi-vault mutations, commit, rollback, crash-safety guarantees, and the WAL framing model.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="PYTHON API · LOW-LEVEL"
      title="Transaction"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "begin", label: "begin_transaction()" },
        { id: "transaction-vault", label: "TransactionVault" },
        { id: "commit", label: "commit()" },
        { id: "rollback", label: "rollback()" },
        { id: "context-manager", label: "Context Manager" },
        { id: "wal-framing", label: "WAL Framing" },
        { id: "crash-safety", label: "Crash Safety" },
        { id: "undo-log", label: "Undo Log" },
        { id: "lock-conflict", label: "LockConflict" },
        { id: "thread-safety", label: "Thread Safety" },
        { id: "examples", label: "Examples" },
        { id: "common-mistakes", label: "Common Mistakes" },
      ]}
    >
      {/* ── Overview ───────────────────────────────────────────── */}
      <h2 id="overview">Overview</h2>
      <p>
        A <code>Transaction</code> lets you group multiple mutations — spanning
        one or more vaults — into a single <strong>atomic</strong> operation.
        Either every pending operation in the transaction is applied, or none
        of them are. This guarantee holds across process crashes: the WAL
        framing ensures that a partially written transaction is never replayed
        on recovery.
      </p>
      <p>
        Transactions are obtained from a{" "}
        <a href="/docs/python/database">
          <code>Database</code>
        </a>{" "}
        object and are most conveniently used as Python context managers.
      </p>

      <CodeBlock lang="python">{`with db.begin_transaction() as txn:
    products = txn.vault("products")
    sessions = txn.vault("sessions")

    products.place([0.1] * 768, data={"sku": "BOOT-42"})
    sessions.erase("session:expired-001")
# Both ops committed atomically
`}</CodeBlock>

      {/* ── begin_transaction() ────────────────────────────────── */}
      <h2 id="begin">
        <code>Database.begin_transaction()</code>
      </h2>
      <CodeBlock lang="python">{`Database.begin_transaction() -> Transaction
`}</CodeBlock>
      <p>
        Allocates a new <code>Transaction</code> context against the database.
        This call writes a <code>txn_begin</code> marker to the WAL immediately.
        No vault-level write lock is acquired until the first mutating operation
        is enqueued or commit is called.
      </p>
      <p>
        You may call <code>begin_transaction()</code> from any thread; however,
        the returned <code>Transaction</code> object must not be shared across
        threads (see <a href="#thread-safety">Thread Safety</a>).
      </p>

      {/* ── TransactionVault ───────────────────────────────────── */}
      <h2 id="transaction-vault">
        <code>Transaction.vault(name) -&gt; TransactionVault</code>
      </h2>
      <CodeBlock lang="python">{`Transaction.vault(name: str) -> TransactionVault
`}</CodeBlock>
      <p>
        Returns a <code>TransactionVault</code> — a thin proxy over the named
        vault that buffers all mutations into the transaction's pending-operation
        log. The underlying vault is created if it does not exist (same lazy
        semantics as <code>Database.vault()</code>).
      </p>
      <p>
        Multiple calls to <code>txn.vault("same-name")</code> within the same
        transaction return the same <code>TransactionVault</code> proxy
        (idempotent).
      </p>

      <h3>
        <code>TransactionVault.place()</code>
      </h3>
      <CodeBlock lang="python">{`TransactionVault.place(
    vector: list[float] | np.ndarray,
    data:   dict        = {},
    id:     str | None  = None,
) -> str
`}</CodeBlock>
      <p>
        Enqueues a place (insert/upsert) operation into the transaction log.
        The record is <strong>not</strong> visible to other readers until the
        transaction is committed. Returns the record ID (auto-generated if{" "}
        <code>id</code> is <code>None</code>).
      </p>

      <h3>
        <code>TransactionVault.erase()</code>
      </h3>
      <CodeBlock lang="python">{`TransactionVault.erase(id: str) -> None
`}</CodeBlock>
      <p>
        Enqueues a delete operation into the transaction log. The record
        remains visible to readers outside this transaction until commit.
      </p>

      {/* ── commit() ───────────────────────────────────────────── */}
      <h2 id="commit">
        <code>Transaction.commit()</code>
      </h2>
      <CodeBlock lang="python">{`Transaction.commit() -> None
`}</CodeBlock>
      <p>
        Applies all pending operations atomically:
      </p>
      <ol>
        <li>
          Acquires exclusive write locks on every vault touched by the
          transaction (in a deterministic order to avoid deadlocks).
        </li>
        <li>
          Replays the pending-operation buffer against each vault's in-memory
          state.
        </li>
        <li>
          Writes a <code>txn_commit</code> marker to the WAL, then flushes
          according to the database's <code>durability</code> setting.
        </li>
        <li>
          Releases all vault write locks.
        </li>
      </ol>
      <p>
        If <em>any</em> step fails (e.g., a WAL write error, a dimension
        mismatch discovered during replay), ELIPS automatically rolls back
        the in-memory state to its pre-transaction snapshot using the undo log
        before propagating the exception. The transaction is then invalid and
        must be discarded.
      </p>
      <p>
        After a successful <code>commit()</code>, calling any method on the
        <code>Transaction</code> raises <code>RuntimeError</code>.
      </p>

      <CodeBlock lang="python">{`txn = db.begin_transaction()
v = txn.vault("items")
v.place([0.5] * 768, data={"label": "a"})
v.place([0.6] * 768, data={"label": "b"})

try:
    txn.commit()
except Exception as e:
    # In-memory state has already been restored.
    # No partial data is visible to readers.
    print(f"Commit failed: {e}")
`}</CodeBlock>

      {/* ── rollback() ─────────────────────────────────────────── */}
      <h2 id="rollback">
        <code>Transaction.rollback()</code>
      </h2>
      <CodeBlock lang="python">{`Transaction.rollback() -> None
`}</CodeBlock>
      <p>
        Discards all pending operations without writing anything to the vaults
        or WAL. A <code>txn_rollback</code> marker is written to the WAL so
        that WAL recovery skips any prior <code>txn_begin</code> for this
        transaction. After rollback, the <code>Transaction</code> is invalid.
      </p>

      <CodeBlock lang="python">{`txn = db.begin_transaction()
v = txn.vault("items")
v.place([0.1] * 768, data={"label": "draft"})

# Decided not to proceed
txn.rollback()
# The record is never visible to anyone
`}</CodeBlock>

      {/* ── Context Manager ────────────────────────────────────── */}
      <h2 id="context-manager">Context Manager</h2>
      <p>
        <code>Transaction</code> implements the Python context manager protocol
        (<code>__enter__</code> / <code>__exit__</code>):
      </p>
      <ul>
        <li>
          <strong>Clean exit (<code>__exit__</code> with no exception)</strong>{" "}
          — automatically calls <code>commit()</code>.
        </li>
        <li>
          <strong>Exception exit</strong> — automatically calls{" "}
          <code>rollback()</code>, then re-raises the original exception.
        </li>
      </ul>
      <p>
        This is the recommended usage pattern for all application code.
      </p>

      <CodeBlock lang="python">{`# Clean exit → auto-commit
with db.begin_transaction() as txn:
    txn.vault("items").place([0.1] * 768, data={"x": 1})
    txn.vault("items").place([0.2] * 768, data={"x": 2})
# commit() called here automatically

# Exception → auto-rollback
try:
    with db.begin_transaction() as txn:
        txn.vault("items").place([0.3] * 768, data={"x": 3})
        raise ValueError("something went wrong")
except ValueError:
    pass  # rollback() was called; the record is not in the vault
`}</CodeBlock>

      {/* ── WAL Framing ────────────────────────────────────────── */}
      <h2 id="wal-framing">WAL Framing</h2>
      <p>
        Every transaction writes a pair of WAL markers that bracket all
        operation records:
      </p>
      <ol>
        <li>
          <code>txn_begin(txn_id, timestamp)</code> — written by{" "}
          <code>begin_transaction()</code>.
        </li>
        <li>
          Individual operation records (<code>op_place</code>,{" "}
          <code>op_erase</code>) — written during commit replay.
        </li>
        <li>
          <code>txn_commit(txn_id)</code> — written as the last record of a
          successful commit.
        </li>
      </ol>
      <p>
        During WAL recovery (after a crash), ELIPS scans the WAL and only
        applies transactions for which it finds a matching{" "}
        <code>txn_commit</code> marker. Any transaction with a{" "}
        <code>txn_begin</code> but no corresponding <code>txn_commit</code>{" "}
        is silently skipped — no partial data is ever applied.
      </p>

      <CodeBlock lang="bash">{`# Conceptual WAL layout:
# [txn_begin:42]
#   [op_place vault=products id=abc ...]
#   [op_place vault=products id=def ...]
#   [op_erase vault=sessions id=xyz]
# [txn_commit:42]       ← only if this is present does recovery apply the above
`}</CodeBlock>

      {/* ── Crash Safety ───────────────────────────────────────── */}
      <h2 id="crash-safety">Crash Safety Guarantee</h2>
      <p>
        ELIPS provides the following crash-safety guarantee for transactions:
      </p>
      <ul>
        <li>
          <strong>Before commit</strong> — A crash at any point before{" "}
          <code>txn_commit</code> is flushed to disk means the transaction
          is completely absent after recovery. Zero data is applied.
        </li>
        <li>
          <strong>During commit flush</strong> — If the process crashes while
          flushing <code>txn_commit</code> to the WAL, the incomplete record is
          detected by a checksum mismatch during recovery and the transaction
          is skipped.
        </li>
        <li>
          <strong>After commit flush</strong> — The transaction is fully
          durable (subject to the database's <code>durability</code> setting).
          With <code>"paranoid"</code> durability, an <code>fsync</code> is
          issued before <code>commit()</code> returns, guaranteeing the data
          survives even a hard power loss.
        </li>
      </ul>

      {/* ── Undo Log ───────────────────────────────────────────── */}
      <h2 id="undo-log">Undo Log</h2>
      <p>
        Internally, a <code>Transaction</code> maintains two structures:
      </p>
      <ul>
        <li>
          <strong>
            <code>PendingOp</code> buffer
          </strong>{" "}
          — An ordered list of operations to apply at commit time. Each entry
          is a discriminated union of <code>Place</code> or <code>Erase</code>{" "}
          variants.
        </li>
        <li>
          <strong>
            <code>UndoEntry</code> list
          </strong>{" "}
          — A snapshot of the pre-commit state for every record that will be
          mutated. Built during commit replay just before each mutation is
          applied. If any mutation fails mid-batch, the undo entries are
          replayed in reverse order via their{" "}
          <code>restore_for_undo()</code> mechanism, returning every affected
          vault to its exact pre-transaction state.
        </li>
      </ul>
      <p>
        This means a <strong>failed commit is always clean</strong>: you never
        observe a vault that has some-but-not-all of the transaction's
        operations applied.
      </p>

      {/* ── LockConflict ───────────────────────────────────────── */}
      <h2 id="lock-conflict">
        <code>LockConflict</code>
      </h2>
      <p>
        <code>elips.LockConflict</code> (subclass of <code>RuntimeError</code>)
        is raised when a transaction's commit cannot acquire a required vault
        write lock within the configured timeout. This can happen if:
      </p>
      <ul>
        <li>
          Another long-running transaction holds the write lock on the same
          vault.
        </li>
        <li>
          A <code>rebuild_index()</code> or <code>vacuum()</code> call is in
          progress on the vault.
        </li>
      </ul>
      <p>
        When <code>LockConflict</code> is raised, the transaction is
        automatically rolled back — no partial mutations have been applied.
        You may retry the transaction.
      </p>

      <CodeBlock lang="python">{`import time, elips

def place_with_retry(db, vault_name, vector, data, max_retries=3):
    for attempt in range(max_retries):
        try:
            with db.begin_transaction() as txn:
                txn.vault(vault_name).place(vector, data=data)
            return  # success
        except elips.LockConflict:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.05 * (2 ** attempt))  # exponential back-off
`}</CodeBlock>

      {/* ── Thread Safety ──────────────────────────────────────── */}
      <h2 id="thread-safety">Thread Safety</h2>
      <p>
        <strong>
          <code>Transaction</code> objects are NOT thread-safe.
        </strong>{" "}
        Do not share a single <code>Transaction</code> across threads. The
        correct pattern is one transaction per thread:
      </p>

      <CodeBlock lang="python">{`import threading, elips

db = elips.open("/data/v.elips", dimension=384, metric="cosine")

def worker(items):
    # Each thread creates its own transaction
    with db.begin_transaction() as txn:
        v = txn.vault("items")
        for vec, meta in items:
            v.place(vec, data=meta)

threads = [
    threading.Thread(target=worker, args=(batch,))
    for batch in batches
]
for t in threads: t.start()
for t in threads: t.join()
`}</CodeBlock>

      <p>
        The <code>Database</code> object itself is thread-safe and can be shared
        freely. Concurrent calls to <code>begin_transaction()</code> from
        different threads each produce independent <code>Transaction</code>{" "}
        objects with separate pending-op buffers.
      </p>

      {/* ── Examples ───────────────────────────────────────────── */}
      <h2 id="examples">Examples</h2>

      <h3>Minimal example</h3>
      <CodeBlock lang="python">{`import elips

db = elips.open("/data/v.elips", dimension=128, metric="cosine")

with db.begin_transaction() as txn:
    v = txn.vault("default")
    v.place([0.1] * 128, data={"label": "hello"}, id="rec:1")
    v.place([0.2] * 128, data={"label": "world"}, id="rec:2")

# Both records are now visible
print(db.vault("default").count)  # 2
`}</CodeBlock>

      <h3>Multi-vault atomic update</h3>
      <CodeBlock lang="python">{`"""
Atomically move a record from 'staging' to 'production' and
record the transition in an 'audit' vault.
"""
import elips, time

db = elips.open("/data/pipeline.elips", dimension=768, metric="cosine")

def promote(record_id: str, embedding: list[float], meta: dict):
    with db.begin_transaction() as txn:
        staging    = txn.vault("staging")
        production = txn.vault("production")
        audit      = txn.vault("audit")

        staging.erase(record_id)
        production.place(embedding, data=meta, id=record_id)
        audit.place(
            embedding,
            data={
                "action":    "promoted",
                "record_id": record_id,
                "at":        time.time(),
            },
        )
    # All three operations committed atomically, or none
`}</CodeBlock>

      <h3>Error handling example</h3>
      <CodeBlock lang="python">{`import elips

db = elips.open("/data/v.elips", dimension=128, metric="cosine")

def safe_batch_insert(records: list[dict]) -> bool:
    """Returns True on success, False if the transaction could not be applied."""
    try:
        with db.begin_transaction() as txn:
            v = txn.vault("items")
            for r in records:
                v.place(r["vector"], data=r.get("data", {}), id=r.get("id"))
        return True
    except elips.LockConflict as e:
        print(f"Lock conflict — will retry: {e}")
        return False
    except elips.DimensionMismatch as e:
        print(f"Bad vector dimension in batch: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error (transaction rolled back): {e}")
        return False
`}</CodeBlock>

      {/* ── Common Mistakes ────────────────────────────────────── */}
      <h2 id="common-mistakes">Common Mistakes</h2>

      <h3>Sharing a Transaction across threads</h3>
      <CodeBlock lang="python">{`# ❌ WRONG — TransactionVault is not thread-safe
txn = db.begin_transaction()
v = txn.vault("items")

def bad_worker(vec, meta):
    v.place(vec, data=meta)  # Race condition!

threads = [threading.Thread(target=bad_worker, args=(vec, {})) for vec in vecs]

# ✅ CORRECT — one transaction per thread
def good_worker(vecs_metas):
    with db.begin_transaction() as txn:
        v = txn.vault("items")
        for vec, meta in vecs_metas:
            v.place(vec, data=meta)
`}</CodeBlock>

      <h3>Using a Transaction after commit or rollback</h3>
      <CodeBlock lang="python">{`# ❌ WRONG
txn = db.begin_transaction()
txn.vault("items").place([0.1] * 128)
txn.commit()
txn.vault("items").place([0.2] * 128)  # RuntimeError: transaction already finalised

# ✅ CORRECT — use the context manager or open a new transaction
with db.begin_transaction() as txn:
    txn.vault("items").place([0.1] * 128)

with db.begin_transaction() as txn:
    txn.vault("items").place([0.2] * 128)
`}</CodeBlock>

      <h3>Forgetting to handle LockConflict</h3>
      <CodeBlock lang="python">{`# ❌ FRAGILE — no retry on lock contention
with db.begin_transaction() as txn:
    txn.vault("items").place(vec, data={})

# ✅ ROBUST — wrap with retry logic for high-concurrency scenarios
import time

def insert_with_retry(db, vault_name, vec, data, retries=5):
    for i in range(retries):
        try:
            with db.begin_transaction() as txn:
                txn.vault(vault_name).place(vec, data=data)
            return
        except elips.LockConflict:
            time.sleep(0.02 * (2 ** i))
    raise RuntimeError(f"Failed to insert after {retries} attempts")
`}</CodeBlock>
    </DocsShell>
  );
}
