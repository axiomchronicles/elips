import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/cpp/elips-instance")({
  head: () => ({
    meta: [
      { title: "Database — C++ Core API Reference" },
      {
        name: "description",
        content:
          "Detailed reference for elips::Database: database handle lifecycle, factory functions, collection management, checkpoints, compaction, and WAL recovery.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="C++ API"
      title="Database"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "factory-open", label: "open() & Constructor" },
        { id: "vault-management", label: "Collection Management" },
        { id: "transactions", label: "Transactions" },
        { id: "query-eql", label: "query() (EQL)" },
        { id: "maintenance", label: "Maintenance & Checkpoints" },
        { id: "introspection", label: "Introspection & GPU" },
        { id: "thread-safety", label: "Thread Safety & Concurrency" },
      ]}
    >
      <p className="text-[18px] text-ink">
        <code>elips::Database</code> is the primary database class in the ELIPS engine.
        It manages persistent file descriptors, write-ahead logs (WAL), advisory file locking,
        and collection life-cycles.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        A <code>Database</code> represents one open database instance bound to a directory path
        (or <code>":memory:"</code>). It acts as the container and coordinator for all named{" "}
        <Link to="/docs/cpp/vault"><code>Collection</code></Link> partitions within the database.
      </p>

      <h2 id="factory-open">open() & Constructor</h2>
      <CodeBlock lang="cpp">{`namespace elips {
    std::unique_ptr<Database> open(
        const std::string& path,
        const Config& config = Config{}
    );
}`}</CodeBlock>
      <p>
        The factory function <code>elips::open()</code> is the recommended way to open or create an
        ELIPS database instance.
      </p>
      <ul>
        <li>
          <code>path</code> — File system directory path for disk-backed persistence, or <code>":memory:"</code> for volatile in-memory operation.
        </li>
        <li>
          <code>config</code> — Default <Link to="/docs/cpp/config"><code>Config</code></Link> applied to newly spawned collections.
        </li>
      </ul>

      <h3 id="direct-constructor">Direct Constructor</h3>
      <CodeBlock lang="cpp">{`Database(
    std::string path,
    Config config,
    bool persistent,
    std::optional<LockManager> lock = std::nullopt
);`}</CodeBlock>

      <h2 id="vault-management">Collection Management</h2>
      <CodeBlock lang="cpp">{`Collection& collection(const std::string& name);
std::vector<std::string> list_collections() const;
Collection& adopt_collection(std::unique_ptr<Collection> collection);`}</CodeBlock>
      <ul>
        <li>
          <code>collection(name)</code> — Gets a reference to an existing collection or lazily creates a new collection with the instance's default <code>Config</code>.
        </li>
        <li>
          <code>list_collections()</code> — Returns a vector containing the names of all active collections.
        </li>
        <li>
          <code>adopt_collection(collection)</code> — Attaches a custom initialized <code>Collection</code> to the instance registry.
        </li>
      </ul>

      <h2 id="transactions">Transactions</h2>
      <CodeBlock lang="cpp">{`Transaction begin_transaction();`}</CodeBlock>
      <p>
        Starts an atomic multi-vault write transaction. Returns a <Link to="/docs/cpp/transaction"><code>Transaction</code></Link> object.
      </p>

      <h2 id="query-eql">query() (EQL)</h2>
      <CodeBlock lang="cpp">{`std::vector<SearchResult> query(
    const std::string& eql,
    const std::map<std::string, Vector>& bindings = {}
);`}</CodeBlock>
      <p>
        Executes a declarative ELIPS Query Language (EQL) string against the database instance with vector binding parameters.
      </p>

      <h2 id="maintenance">Maintenance & Checkpoints</h2>
      <CodeBlock lang="cpp">{`void checkpoint();
void compact();
void vacuum();
void close();
void abandon() noexcept;`}</CodeBlock>
      <ul>
        <li>
          <code>checkpoint()</code> — Flushes memory-mapped pages and truncates the WAL log up to the latest sequence number.
        </li>
        <li>
          <code>compact()</code> — Rewrites storage segment files on disk to eliminate unreferenced tombstones and shrink file sizes.
        </li>
        <li>
          <code>vacuum()</code> — Reclaims deleted vector nodes within HNSW graph indices across all open collections.
        </li>
        <li>
          <code>close()</code> — Safely seals all collections, flushes WAL, and releases process lock handlers.
        </li>
      </ul>

      <h2 id="introspection">Introspection & GPU</h2>
      <CodeBlock lang="cpp">{`WAL* wal() const noexcept;
const Config& config() const noexcept;
const std::string& path() const noexcept;
bool persistent() const noexcept;
bool closed() const noexcept;

#ifdef ELIPS_GPU_ENABLED
gpu::GpuDeviceInfo gpu_info() const;
gpu::GpuMetricsSnapshot gpu_stats() const;
#endif`}</CodeBlock>

      <h2 id="thread-safety">Thread Safety & Concurrency</h2>
      <p>
        <code>Database</code> is completely thread-safe. Concurrent calls to <code>collection()</code>, <code>list_collections()</code>, and <code>checkpoint()</code> across threads are serialized via internal recursive locks.
      </p>
    </DocsShell>
  );
}
