import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/durability")({
  head: () => ({
    meta: [
      {
        title: "Durability & Transactions",
      },
    ],
    links: [],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="Durability & Transactions"
      toc={[
        { id: "overview", title: "Overview" },
        { id: "durability-levels", title: "Durability Levels" },
        { id: "what-changed", title: "Storage Improvements (F1-F4)" },
        { id: "transaction-semantics", title: "Transaction Semantics" },
        { id: "transaction-examples", title: "Transaction Examples" },
        { id: "access-modes", title: "Access Modes & Locking" },
        { id: "read-only-serving", title: "Read-Only Serving" },
        { id: "monitoring-durability", title: "Monitoring Durability" },
        { id: "crash-recovery", title: "Crash Recovery" },
        { id: "pitfalls", title: "Common Pitfalls" },
      ]}
    >
      <h2 id="overview">Overview</h2>
      <p>
        ELIPS provides strong guarantees around data persistence and atomicity,
        allowing you to tune performance versus crash safety according to your application's needs.
        With the introduction of the F1-F4 storage engine updates, Python developers now have access to robust Write-Ahead Log (WAL) behaviors, reliable undo logs, and secure cross-process access controls.
      </p>

      <h2 id="durability-levels">Durability Levels</h2>
      <p>
        Durability levels determine how aggressively ELIPS forces the operating system to flush writes to physical media. You can configure this using the <code>durability()</code> builder on the <code>Config</code> object:
      </p>
      <ul>
        <li>
          <strong>paranoid:</strong> <code>F_FULLFSYNC</code> on macOS, <code>fdatasync</code> on Linux before every acknowledgment. Provides maximum crash safety, suitable for financial or medical data where every single write must survive power loss.
        </li>
        <li>
          <strong>standard:</strong> OS-synced writes (fsync at checkpoint). This is the default setting and is suitable for most production uses.
        </li>
        <li>
          <strong>relaxed:</strong> Writes directly to the OS page cache. Data will survive a process crash but not an OS crash or power loss. Ideal for staging or batch ingest pipelines where you can afford to re-ingest data if a system goes down.
        </li>
        <li>
          <strong>ephemeral:</strong> No persistence at all (in-memory mode, no WAL). Suitable for unit tests, CI, and in-process search over preloaded data.
        </li>
      </ul>

      <CodeBlock lang="python">{`import elips

# paranoid = strongest durability (sync every ack)
config = elips.Config().durability("paranoid")

# standard = default
config = elips.Config().durability("standard")

# relaxed = throughput-optimized (page cache)
config = elips.Config().durability("relaxed")

# ephemeral = no persistence
config = elips.Config().durability("ephemeral")`}</CodeBlock>

      <p>
        You can inspect the currently active durability via properties on the config object: <code>config.durability_enum</code> and <code>config.durability_val</code>.
      </p>
      <CodeBlock lang="python">{`# Throughput-optimized (relaxed)
config = elips.Config().dimension(128).durability("relaxed")
db = elips.open_with_config("/data/staging", config)

# In-memory (no WAL at all)
db = elips.open(":memory:", dimension=128)
print(db.persistent)  # False`}</CodeBlock>

      <h2 id="what-changed">Storage Improvements (F1-F4)</h2>
      <p>
        The F1 through F4 patches significantly hardened the ELIPS storage engine, resolving critical issues around atomicity and corruption recovery. It is important for users to understand what these fixes guarantee:
      </p>
      <ul>
        <li>
          <strong>F1: True Sync Guarantees.</strong> The Write-Ahead Log (WAL) now correctly fsyncs (or <code>F_FULLFSYNC</code> on macOS) before returning from an append. Additionally, snapshot, segment, and manifest writes are published via a durable rename, which syncs both the file and directory metadata. Previously, acknowledged writes could be lost on OS crash or power loss even under the "standard" durability setting.
        </li>
        <li>
          <strong>F2: Bounded Allocations.</strong> Length-prefixed payloads (like strings or vectors) are now bounded by the remaining stream length before allocation. This means a corrupt WAL can no longer cause the system to attempt unbounded memory allocations.
        </li>
        <li>
          <strong>F3: Faster Recovery.</strong> WAL replay time was optimized from <code>O(n*k)</code> to <code>O(n)</code> by avoiding unnecessary tail-copying per record.
        </li>
        <li>
          <strong>F4: Robust Atomicity.</strong> The <code>commit()</code> process now pre-checks writability and maintains an undo log. A failed WAL write automatically restores the prior state. Commits on a read-only vault throw an error before applying anything. Furthermore, the WAL brackets batches with <code>txn_begin</code> and <code>txn_commit</code>, ensuring that replay completely drops unterminated transaction windows.
        </li>
      </ul>

      <h2 id="transaction-semantics">Transaction Semantics</h2>
      <p>
        Following the F4 updates, transactions are strictly all-or-nothing:
      </p>
      <ul>
        <li><strong>All-or-nothing:</strong> A failure partway through a transaction uses the undo log to revert any operations that were already applied.</li>
        <li><strong>WAL framing:</strong> Transactions are bracketed with <code>txn_begin</code> and <code>txn_commit</code> markers in the WAL. If a process dies mid-commit, the replay engine will drop the unterminated window.</li>
        <li><strong>Pre-checks:</strong> <code>commit()</code> verifies that a vault is not read-only <em>before</em> attempting to apply operations.</li>
        <li><strong>Safe Rollbacks:</strong> Calling <code>rollback()</code> is always safe, even after a failed <code>commit()</code>, since the state will have already been restored by the undo log.</li>
      </ul>

      <h2 id="transaction-examples">Transaction Examples</h2>
      <CodeBlock lang="python">{`import elips

# Configure maximum durability
config = (
    elips.Config()
    .dimension(384)
    .metric("cosine")
    .durability("paranoid")
)
db = elips.open_with_config("/data/critical", config)

# Transaction - all or nothing using a context manager
with db.begin_transaction() as txn:
    v = txn.vault("docs")
    v.place([1.0, 0.0], {"title": "A"})
    v.place([0.0, 1.0], {"title": "B"})
    # Clean exit: both writes committed atomically
    # Exception: both writes rolled back

# Failed commit restores state automatically
txn = db.begin_transaction()
v = txn.vault("docs")
v.place([1.0, 0.0], {"title": "C"})
try:
    txn.commit()
except elips.StorageError:
    # State restored. Safe to retry or rollback explicitly.
    txn.rollback()  # Also safe; state already restored`}</CodeBlock>

      <h2 id="access-modes">Access Modes & Locking</h2>
      <p>
        ELIPS uses POSIX advisory <code>flock</code> to manage cross-process locking. There are two primary modes for opening a database:
      </p>
      <ul>
        <li>
          <strong>read_write (default):</strong> Acquires an exclusive writer lock. Only one process can open the database in this mode at a time.
        </li>
        <li>
          <strong>read_only:</strong> Acquires a shared advisory lock. Multiple processes can open the database in <code>read_only</code> mode. Write operations (like place, erase, rebuild, or compaction) will raise a <code>StorageError</code>.
        </li>
      </ul>
      <p>
        If a second process attempts to open a database in <code>read_write</code> mode while another process already holds the exclusive lock, ELIPS raises a <code>LockConflict</code> exception. To read from a database while it is being written to by another process, use <code>read_only</code> mode.
      </p>

      <h2 id="read-only-serving">Read-Only Serving</h2>
      <p>
        The shared-reader pattern allows you to scale read traffic without interfering with a single writer process. 
      </p>
      <CodeBlock lang="python">{`# Read-only serving (shared readers)
# A separate process might hold the exclusive writer lock
reader = elips.open("/data/critical", access_mode="read_only")
results = reader.vault("docs").seek([1.0, 0.0], top=5)`}</CodeBlock>

      <h2 id="monitoring-durability">Monitoring Durability</h2>
      <p>
        You can inspect operations that are currently in the WAL but not yet checkpointed using <code>vault.pending_writes()</code>. This returns a typed list of <code>WalRecord</code> objects, which is extremely useful for monitoring durability lag and deciding when to force a checkpoint.
      </p>

      <h2 id="crash-recovery">Crash Recovery</h2>
      <p>
        Recovery after an unclean shutdown happens automatically. When you call <code>elips.open()</code>, ELIPS replays the WAL to recover any committed transactions that were not yet checkpointed. 
      </p>
      <CodeBlock lang="python">{`# Simulate unclean shutdown by abandoning the handle
db.abandon()  # Leaves WAL on disk, bypassing clean shutdown

# Next open() replays the WAL automatically
db2 = elips.open("/data/critical")`}</CodeBlock>

      <h2 id="pitfalls">Common Pitfalls</h2>
      <ul>
        <li>
          <strong>LockConflict in multiprocessing:</strong> Remember that only one process can have a <code>read_write</code> handle. Use <code>read_only</code> for read replicas in secondary processes.
        </li>
        <li>
          <strong>Unnecessary durability:</strong> Using <code>paranoid</code> durability introduces significant latency (often 10-20ms per transaction due to syncing physical media). Only use it if losing a single write during power loss is unacceptable.
        </li>
        <li>
          <strong>Leaving transactions open:</strong> Long-running transactions delay checkpointing. Keep transactions scoped tightly around the required batch of operations.
        </li>
      </ul>

      <p>
        <strong>Learn more:</strong> Check out the <Link to="/docs/internals/transaction-engine">Transaction Engine</Link>, <Link to="/docs/storage">Storage format</Link>, and <Link to="/docs/internals/lock-manager">Lock Manager</Link> documentation.
      </p>
    </DocsShell>
  );
}
