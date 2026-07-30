import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/maintenance")({
  head: () => ({
    meta: [{ title: "Index Maintenance" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="Index Maintenance"
      toc={[
        { id: "overview", title: "Overview" },
        { id: "hnsw-compaction", title: "HNSW Compaction & Tombstones" },
        { id: "vacuum", title: "Vacuum" },
        { id: "rebuild-index", title: "Rebuild Index" },
        { id: "graph-params", title: "Graph Params" },
        { id: "read-only-sealing", title: "Read-Only Mode & Sealing" },
        { id: "records-snapshot", title: "Records Snapshot" },
        { id: "database-lifecycle", title: "Database Lifecycle" },
        { id: "modern-engine-maintenance", title: "Modern Engine Maintenance" },
        { id: "arena-health", title: "Arena Health" },
        { id: "operational-patterns", title: "Operational Patterns" },
        { id: "pitfalls", title: "Common Pitfalls" }
      ]}
    >
      <h2 id="overview">Overview</h2>
      <p>
        Maintaining the performance and health of your ELIPS database is critical for long-running deployments. This page documents the APIs available for managing vault and database maintenance in Python, including managing HNSW tombstones, reclaiming disk space, and monitoring the lifecycle of your database.
      </p>

      <h2 id="hnsw-compaction">HNSW Compaction & Tombstones</h2>
      <p>
        When you delete a record in ELIPS using <code>Vault.erase()</code>, the node is not immediately removed from the HNSW graph to avoid heavy lock contention. Instead, it is marked as a <strong>tombstone</strong>.
      </p>
      <p>
        The search algorithm uses an <strong>adaptive <code>ef</code> beam</strong>: when tombstones build up, <code>search()</code> dynamically scales the search width by the ratio of total nodes to live nodes. This ensures that the search beam still yields enough live hits, allowing recall to stay within 0.15 of baseline even at a 50% delete ratio, while the graph size remains bounded.
      </p>
      <p>
        You can check how many tombstones are pending removal using the <code>Vault.pending_removals</code> property.
      </p>

      <h2 id="vacuum">Vacuum</h2>
      <p>
        To reclaim the space held by tombstones in the index, use <code>vacuum()</code>. It performs a self-compaction of the graph index without rewriting the on-disk snapshot, making it cheap and safe to use even on in-memory databases.
      </p>
      <ul>
        <li><code>Vault.vacuum()</code> — Vacuums a specific vault. It's a cheap no-op if nothing is tombstoned.</li>
        <li><code>Database.vacuum()</code> — Reclaims tombstoned index space across every vault in the database.</li>
      </ul>

      <CodeBlock lang="python">{`import elips

db = elips.open("/data/vec", dimension=128)
vault = db.vault("docs")

# Ingest then delete a batch
ids = [vault.place([0.0]*128, {"i": i}) for i in range(1000)]
for rid in ids[:200]:
    vault.erase(rid)

# Check tombstone pressure
print(vault.pending_removals)  # 200

# Vacuum: cheap, works on in-memory DBs too
vault.vacuum()
print(vault.pending_removals)  # 0

# Database-level vacuum across all vaults
db.vacuum()`}</CodeBlock>

      <h2 id="rebuild-index">Rebuild Index</h2>
      <p>
        Unlike <code>vacuum()</code>, <code>Vault.rebuild_index()</code> completely reconstructs the backing index from stored records. It is more expensive but creates a perfectly fresh HNSW graph without the fragmentation that might accumulate after millions of updates.
      </p>

      <h2 id="graph-params">Graph Params</h2>
      <p>
        The <code>compaction_ratio</code> parameter controls the tombstone fraction that triggers auto-compaction. By default, it is set to 0.2 (20%). Setting it to 0.0 disables auto-compaction.
      </p>

      <CodeBlock lang="python">{`# GraphParams with compaction_ratio
params = elips.GraphParams(
    max_connections=16,
    ef_construction=200,
    ef_search=50,
    compaction_ratio=0.15,  # compact when 15% tombstoned
)
config = elips.Config().dimension(128).graph_params(params)
db = elips.open_with_config("/data/vec", config)`}</CodeBlock>

      <h2 id="read-only-sealing">Read-Only Mode & Sealing</h2>
      <p>
        You can explicitly control the mutability of your vaults:
      </p>
      <ul>
        <li><code>Vault.read_only</code>: Returns <code>True</code> when the vault is in read-only mode and refuses mutations.</li>
        <li><code>Vault.set_read_only(read_only: bool)</code>: Toggles runtime read-only mode. Further mutations will raise a <code>StorageError</code>.</li>
        <li><code>Vault.sealed</code>: Returns <code>True</code> once the owning database is closed. Writes to a sealed vault will immediately raise an error rather than silently failing to persist data to memory that will never be checkpointed.</li>
      </ul>

      <h2 id="records-snapshot">Records Snapshot</h2>
      <p>
        You can snapshot every stored record in a vault using <code>Vault.records()</code>, which returns a copy of the live map (safely guarded by a mutex) as a list of <code>StoredRecord</code> (a TypedDict).
      </p>
      <CodeBlock lang="python">{`records = vault.records()  # list[StoredRecord]
print(len(records))  # 800`}</CodeBlock>
      <p>
        <strong>Note:</strong> For large vaults, prefer using <code>scan()</code> with a filter and limit instead to avoid copying the entire dataset into memory at once.
      </p>

      <h2 id="database-lifecycle">Database Lifecycle</h2>
      <p>
        The <code>Database</code> object exposes several maintenance and lifecycle methods:
      </p>
      <ul>
        <li><code>Database.compact()</code> — Rebuilds every vault index from scratch and checkpoints the state. Ideal for periodic cleanup on on-disk databases (unlike <code>vacuum</code>, it does not work on in-memory DBs).</li>
        <li><code>Database.checkpoint()</code> — Flushes current state and WAL to disk for durability.</li>
        <li><code>Database.close()</code> — Triggers a final checkpoint, releases locks, and seals every vault.</li>
      </ul>
      <p>Lifecycle properties:</p>
      <ul>
        <li><code>Database.path</code> — The filesystem directory or <code>":memory:"</code> for transient databases.</li>
        <li><code>Database.persistent</code> — Returns <code>False</code> for in-memory databases.</li>
        <li><code>Database.closed</code> — Returns <code>True</code> once <code>close()</code> or <code>abandon()</code> has run.</li>
      </ul>

      <h2 id="modern-engine-maintenance">Modern Engine Maintenance</h2>
      <p>
        The modern <code>Engine</code> API exposes analogous maintenance tools:
      </p>
      <ul>
        <li><code>Engine.vacuum()</code> — Vacuums all arenas.</li>
        <li><code>Engine.vault_names() {"->"} list[str]</code> — Lists all vault/arena names.</li>
        <li><code>Engine.pending_writes() {"->"} list[WalRecord]</code> — Returns all pending WAL entries.</li>
      </ul>

      <h2 id="arena-health">Arena Health</h2>
      <p>
        For detailed diagnostics, <code>Arena.health()</code> returns an <code>ArenaHealth</code> TypedDict which includes metrics like the <code>tombstone_ratio: float</code>. This is useful for building operational dashboards and determining when to trigger a manual vacuum.
      </p>

      <h2 id="operational-patterns">Operational Patterns</h2>
      <p>
        For production applications, follow this checklist to ensure consistent performance:
      </p>
      <ul>
        <li><strong>Monitor tombstones:</strong> Periodically check <code>Vault.pending_removals</code> or <code>Arena.health()['tombstone_ratio']</code>.</li>
        <li><strong>Vacuum proactively:</strong> Call <code>vacuum()</code> when the <code>tombstone_ratio</code> exceeds your configured <code>compaction_ratio</code> threshold, or rely on auto-compaction.</li>
        <li><strong>Use checkpoints:</strong> Periodically call <code>checkpoint()</code> in long-running serving loops to ensure regular durability without pausing queries.</li>
        <li><strong>Scheduled compactions:</strong> Call <code>compact()</code> before planned long downtime windows to flush the WAL and completely rebuild indices on disk.</li>
      </ul>

      <h2 id="pitfalls">Common Pitfalls</h2>
      <ul>
        <li>Never call <code>Vault.rebuild_index()</code> on a hot vault under active concurrent search load unless you have a second replica serving the traffic.</li>
        <li>Don't expect <code>compact()</code> to work on an in-memory database; use <code>vacuum()</code> instead.</li>
        <li>Writing to a vault after calling <code>Database.close()</code> will raise an error because the vault is <strong>sealed</strong>. Be sure to stop ingest traffic before closing the database.</li>
      </ul>

      <p>
        See also: <Link to="/docs/python-sdk">Python SDK</Link>, <Link to="/docs/algorithms">Algorithms</Link>, <Link to="/docs/storage">Storage</Link>.
      </p>
    </DocsShell>
  );
}
