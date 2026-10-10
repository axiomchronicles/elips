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
        { id: "rebuild-index", title: "Rebuild" },
        { id: "graph-params", title: "HNSW Params" },
        { id: "read-only-sealing", title: "Read-Only Mode & Sealing" },
        { id: "records-snapshot", title: "Inspecting Records" },
        { id: "database-lifecycle", title: "Client Lifecycle" },
        { id: "modern-engine-maintenance", title: "Client Maintenance" },
        { id: "arena-health", title: "Collection Health" },
        { id: "operational-patterns", title: "Operational Patterns" },
        { id: "pitfalls", title: "Common Pitfalls" }
      ]}
    >
      <h2 id="overview">Overview</h2>
      <p>
        Maintaining the performance and health of your ELIPS database is critical for long-running deployments. This page documents the APIs available for managing collection and client maintenance in Python, including managing HNSW tombstones, reclaiming disk space, and monitoring the lifecycle of your database.
      </p>

      <h2 id="hnsw-compaction">HNSW Compaction & Tombstones</h2>
      <p>
        When you delete a record in ELIPS using <code>collection.delete()</code>, the node is not immediately removed from the HNSW graph to avoid heavy lock contention. Instead, it is marked as a <strong>tombstone</strong>.
      </p>
      <p>
        The search algorithm uses an <strong>adaptive <code>ef</code> beam</strong>: when tombstones build up, <code>query()</code> dynamically scales the search width by the ratio of total nodes to live nodes. This ensures that the search beam still yields enough live hits, allowing recall to stay within 0.15 of baseline even at a 50% delete ratio, while the graph size remains bounded.
      </p>
      <p>
        You can check how many tombstones are pending removal using the <code>collection.pending_removals</code> property.
      </p>

      <h2 id="vacuum">Vacuum</h2>
      <p>
        To reclaim the space held by tombstones in the index, use <code>vacuum()</code>. It performs a self-compaction of the graph index without rewriting the on-disk snapshot, making it cheap and safe to use even on in-memory databases.
      </p>
      <ul>
        <li><code>collection.vacuum()</code> — Vacuums a specific collection. It's a cheap no-op if nothing is tombstoned.</li>
        <li><code>client.vacuum()</code> — Reclaims tombstoned index space across every collection in the database.</li>
      </ul>

      <CodeBlock lang="python">{`import elips

client = elips.PersistentClient("/data/vec", dimension=128)
docs = client.get_or_create_collection("docs")

# Ingest then delete a batch
ids = [docs.add(embedding=[0.0] * 128, metadata={"i": i}) for i in range(1000)]
docs.delete(ids[:200])

# Check tombstone pressure
print(docs.pending_removals)  # 200

# Vacuum: cheap, works on in-memory DBs too
docs.vacuum()
print(docs.pending_removals)  # 0

# Client-level vacuum across all collections
client.vacuum()`}</CodeBlock>

      <h2 id="rebuild-index">Rebuild</h2>
      <p>
        Unlike <code>vacuum()</code>, <code>collection.rebuild()</code> completely reconstructs the backing index from stored records. It is more expensive but creates a perfectly fresh HNSW graph without the fragmentation that might accumulate after millions of updates.
      </p>
      <p>
        For quantized collections, <code>collection.compress()</code> (alias <code>train</code>) trains the codebook from the stored vectors so compressed search can be served.
      </p>

      <h2 id="graph-params">HNSW Params</h2>
      <p>
        The <code>compaction_ratio</code> parameter controls the tombstone fraction that triggers auto-compaction. By default, it is set to 0.2 (20%). Setting it to 0.0 disables auto-compaction.
      </p>

      <CodeBlock lang="python">{`# HnswParams with compaction_ratio
params = elips.HnswParams(
    max_connections=16,
    ef_construction=200,
    ef_search=50,
    compaction_ratio=0.15,  # compact when 15% tombstoned
)
client = elips.PersistentClient("/data/vec", dimension=128, index="graph", hnsw=params)`}</CodeBlock>

      <h2 id="read-only-sealing">Read-Only Mode & Sealing</h2>
      <p>
        You can explicitly control the mutability of your collections:
      </p>
      <ul>
        <li><code>collection.read_only</code>: Returns <code>True</code> when the collection is in read-only mode and refuses mutations.</li>
        <li><code>collection.freeze(frozen=True)</code>: Freezes the collection so further mutations are rejected. Pass <code>frozen=False</code> to unfreeze.</li>
        <li><code>collection.sealed</code>: Returns <code>True</code> once the owning client is closed. Writes to a sealed collection immediately raise an error rather than silently failing to persist data to memory that will never be checkpointed.</li>
      </ul>

      <h2 id="records-snapshot">Inspecting Records</h2>
      <p>
        To look at stored records without running a similarity search, use <code>collection.peek(n)</code> (a quick sample) or <code>collection.get(...)</code> (by ids or a <code>where</code> filter). Both return <code>Row</code> objects.
      </p>
      <CodeBlock lang="python">{`sample = docs.peek(10)        # list[Row]
print(len(sample))            # up to 10
print(docs.count())           # total live records

# Scan a filtered page without loading everything
rows = docs.get(where=elips.Filter().field("kind").equals("design"), limit=100)`}</CodeBlock>
      <p>
        <strong>Note:</strong> For large collections, prefer <code>get()</code> with a filter and limit instead of fetching every record, to avoid copying the entire dataset into memory at once.
      </p>

      <h2 id="database-lifecycle">Client Lifecycle</h2>
      <p>
        The <code>Client</code> / <code>PersistentClient</code> object exposes several maintenance and lifecycle methods:
      </p>
      <ul>
        <li><code>client.compact()</code> — Rebuilds every collection index from scratch and checkpoints the state. Ideal for periodic cleanup on on-disk databases (unlike <code>vacuum</code>, it does not work on in-memory DBs).</li>
        <li><code>client.checkpoint()</code> — Flushes current state and WAL to disk for durability.</li>
        <li><code>client.close()</code> — Triggers a final checkpoint, releases locks, and seals every collection.</li>
      </ul>
      <p>Use the client as a context manager so it is always closed cleanly:</p>
      <CodeBlock lang="python">{`with elips.PersistentClient("/data/vec", dimension=128) as client:
    docs = client.get_or_create_collection("docs")
    # ... work ...
    client.checkpoint()
# close() runs automatically on exit`}</CodeBlock>
      <p>
        The low-level native handle is available via <code>client.native</code>, and the resolved settings via <code>client.config</code>.
      </p>

      <h2 id="modern-engine-maintenance">Client Maintenance</h2>
      <p>
        The client exposes database-wide maintenance helpers:
      </p>
      <ul>
        <li><code>client.vacuum()</code> — Vacuums all collections.</li>
        <li><code>client.collection_names() {"->"} list[str]</code> — Lists all collection names (alias <code>list_collections()</code>).</li>
        <li><code>client.pending_writes() {"->"} list[WalRecord]</code> — Returns all pending WAL entries.</li>
      </ul>

      <h2 id="arena-health">Collection Health</h2>
      <p>
        For detailed diagnostics, <code>collection.health()</code> returns a <code>CollectionHealth</code> dataclass which includes metrics like the <code>tombstone_ratio</code> property. This is useful for building operational dashboards and determining when to trigger a manual vacuum.
      </p>

      <h2 id="operational-patterns">Operational Patterns</h2>
      <p>
        For production applications, follow this checklist to ensure consistent performance:
      </p>
      <ul>
        <li><strong>Monitor tombstones:</strong> Periodically check <code>collection.pending_removals</code> or <code>collection.health().tombstone_ratio</code>.</li>
        <li><strong>Vacuum proactively:</strong> Call <code>vacuum()</code> when the <code>tombstone_ratio</code> exceeds your configured <code>compaction_ratio</code> threshold, or rely on auto-compaction.</li>
        <li><strong>Use checkpoints:</strong> Periodically call <code>checkpoint()</code> in long-running serving loops to ensure regular durability without pausing queries.</li>
        <li><strong>Scheduled compactions:</strong> Call <code>compact()</code> before planned long downtime windows to flush the WAL and completely rebuild indices on disk.</li>
      </ul>

      <h2 id="pitfalls">Common Pitfalls</h2>
      <ul>
        <li>Never call <code>collection.rebuild()</code> on a hot collection under active concurrent search load unless you have a second replica serving the traffic.</li>
        <li>Don't expect <code>compact()</code> to work on an in-memory database; use <code>vacuum()</code> instead.</li>
        <li>Writing to a collection after calling <code>client.close()</code> will raise an error because the collection is <strong>sealed</strong>. Be sure to stop ingest traffic before closing the client.</li>
      </ul>

      <p>
        See also: <Link to="/docs/python-sdk">Python SDK</Link>, <Link to="/docs/algorithms">Algorithms</Link>, <Link to="/docs/storage">Storage</Link>.
      </p>
    </DocsShell>
  );
}
