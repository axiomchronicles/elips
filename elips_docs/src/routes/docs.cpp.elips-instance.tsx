import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/cpp/elips-instance")({
  head: () => ({
    meta: [
      { title: "ElipsInstance — C++ Core API Reference" },
      {
        name: "description",
        content:
          "Detailed reference for elips::ElipsInstance: database handle lifecycle, factory functions, vault management, checkpoints, compaction, and WAL recovery.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="C++ API"
      title="ElipsInstance"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "factory-open", label: "open() & Constructor" },
        { id: "vault-management", label: "Vault Management" },
        { id: "transactions", label: "Transactions" },
        { id: "query-eql", label: "query() (EQL)" },
        { id: "maintenance", label: "Maintenance & Checkpoints" },
        { id: "introspection", label: "Introspection & GPU" },
        { id: "thread-safety", label: "Thread Safety & Concurrency" },
      ]}
    >
      <p className="text-[18px] text-ink">
        <code>elips::ElipsInstance</code> is the primary database class in the ELIPS engine.
        It manages persistent file descriptors, write-ahead logs (WAL), advisory file locking,
        and vault collection life-cycles.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        An <code>ElipsInstance</code> represents one open database instance bound to a directory path
        (or <code>":memory:"</code>). It acts as the container and coordinator for all named{" "}
        <Link to="/docs/cpp/vault"><code>Vault</code></Link> partitions within the database.
      </p>

      <h2 id="factory-open">open() & Constructor</h2>
      <CodeBlock lang="cpp">{`namespace elips {
    std::unique_ptr<ElipsInstance> open(
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
          <code>config</code> — Default <Link to="/docs/cpp/config"><code>Config</code></Link> applied to newly spawned vaults.
        </li>
      </ul>

      <h3 id="direct-constructor">Direct Constructor</h3>
      <CodeBlock lang="cpp">{`ElipsInstance(
    std::string path,
    Config config,
    bool persistent,
    std::optional<LockManager> lock = std::nullopt
);`}</CodeBlock>

      <h2 id="vault-management">Vault Management</h2>
      <CodeBlock lang="cpp">{`Vault& vault(const std::string& name);
std::vector<std::string> list_vaults() const;
Vault& adopt_vault(std::unique_ptr<Vault> vault);`}</CodeBlock>
      <ul>
        <li>
          <code>vault(name)</code> — Gets a reference to an existing vault or lazily creates a new vault with the instance's default <code>Config</code>.
        </li>
        <li>
          <code>list_vaults()</code> — Returns a vector containing the names of all active vaults.
        </li>
        <li>
          <code>adopt_vault(vault)</code> — Attaches a custom initialized <code>Vault</code> to the instance registry.
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
          <code>vacuum()</code> — Reclaims deleted vector nodes within HNSW graph indices across all open vaults.
        </li>
        <li>
          <code>close()</code> — Safely seals all vaults, flushes WAL, and releases process lock handlers.
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
        <code>ElipsInstance</code> is completely thread-safe. Concurrent calls to <code>vault()</code>, <code>list_vaults()</code>, and <code>checkpoint()</code> across threads are serialized via internal recursive locks.
      </p>
    </DocsShell>
  );
}
