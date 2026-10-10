import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/database")({
  head: () => ({
    meta: [
      { title: "Database — ELIPS Python API Docs" },
      {
        name: "description",
        content:
          "Reference for the low-level native ELIPS Database: vaults, transactions, EQL queries, checkpoint/compact, and lifecycle. The high-level Client wraps it.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="PYTHON API · LOW-LEVEL"
      title="Database"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "reaching", label: "Reaching a Database" },
        { id: "vault", label: "vault()" },
        { id: "list-vaults", label: "list_vaults()" },
        { id: "transactions", label: "begin_transaction()" },
        { id: "query", label: "query() — EQL" },
        { id: "eql", label: "EQL grammar" },
        { id: "maintenance", label: "Maintenance" },
        { id: "properties", label: "Properties" },
        { id: "thread-safety", label: "Thread Safety" },
        { id: "examples", label: "Full Examples" },
      ]}
    >
      <h2 id="overview">Overview</h2>
      <p>
        <code>Database</code> is the low-level native handle to an ELIPS instance
        (the C++ core type, formerly <code>ElipsInstance</code>). It is the
        escape hatch beneath the high-level{" "}
        <a href="/docs/python/engine">Client / PersistentClient</a> — those wrap
        a <code>Database</code> and expose the Chroma-style{" "}
        <a href="/docs/python/arena">Collection</a> surface on top of it.
      </p>
      <p>
        Prefer the client surface for application code. Reach the native{" "}
        <code>Database</code> only when you need a capability the client does not
        wrap: cross-vault transactions, raw EQL, or direct vault handles.
      </p>

      <h2 id="reaching">Reaching a Database</h2>
      <p>
        The supported way to obtain a native <code>Database</code> is{" "}
        <code>client.native</code> on an open client:
      </p>
      <CodeBlock lang="python">{`import elips

client = elips.PersistentClient("/data/products", dimension=1536, metric="cosine")
db = client.native            # low-level native Database
print(db.list_vaults())
`}</CodeBlock>
      <p>
        The module-level <code>elips.open(path, ...)</code> /{" "}
        <code>elips.open_with_config(path, config)</code> entry points still
        return a <code>Database</code> directly, but they are deprecated shims —
        new code should open a{" "}
        <a href="/docs/python/engine">
          <code>Client</code> or <code>PersistentClient</code>
        </a>{" "}
        and use <code>client.native</code>. There is no <code>durability=</code>{" "}
        open keyword — durability is a native <code>Config</code> concern.
      </p>
      <h2 id="vault">
        <code>Database.vault(name)</code>
      </h2>
      <p>
        Return a native{" "}
        <a href="/docs/python/vault">
          <code>Vault</code>
        </a>{" "}
        — a named namespace that stores its own vectors and records. Vaults are
        created lazily: the first call for a name creates and registers it; later
        calls return the same object.
      </p>
      <CodeBlock lang="python">{`Database.vault(name: str) -> elips.native.Vault
`}</CodeBlock>
      <CodeBlock lang="python">{`db = client.native
products = db.vault("products")
reviews  = db.vault("reviews")
assert db.vault("products") is products   # cached
`}</CodeBlock>

      <h2 id="list-vaults">
        <code>Database.list_vaults()</code>
      </h2>
      <p>
        Return the names of all vaults registered in the database, including
        those on disk that have not yet been accessed this session.
      </p>
      <CodeBlock lang="python">{`names = db.list_vaults()
print(names)   # ['products', 'reviews', 'sessions']
`}</CodeBlock>

      <h2 id="transactions">
        <code>Database.begin_transaction()</code>
      </h2>
      <p>
        Return a <code>Transaction</code> that groups mutations across one or
        more vaults into a single atomic operation. Use it as a context manager
        so rollback is automatic on error.
      </p>
      <CodeBlock lang="python">{`with db.begin_transaction() as txn:
    v = txn.vault("products")
    v.place([0.1] * 1536, {"sku": "ABC-001"})
    v.place([0.2] * 1536, {"sku": "ABC-002"})
# Committed atomically — both records appear or neither does
`}</CodeBlock>
      <p>
        See the dedicated{" "}
        <a href="/docs/python/transaction">Transaction reference</a> for commit,
        rollback, and crash-safety details.
      </p>
      <h2 id="query">
        <code>Database.query(eql, bindings=None)</code>
      </h2>
      <p>
        Execute an <strong>ELIPS Query Language (EQL)</strong> statement and
        return the results. EQL is a small, expression-oriented DSL — it is{" "}
        <em>not</em> SQL.
      </p>
      <CodeBlock lang="python">{`Database.query(eql: str, bindings: dict | None = None) -> list
`}</CodeBlock>
      <ul>
        <li>
          <strong>
            <code>eql</code>
          </strong>{" "}
          — an EQL statement (see the grammar below).
        </li>
        <li>
          <strong>
            <code>bindings</code>
          </strong>{" "}
          — a mapping from <code>$name</code> placeholders in the statement to
          Python values (e.g. a query vector).
        </li>
      </ul>
      <CodeBlock lang="python">{`# Nearest-neighbour search via EQL
hits = db.query(
    'search in products nearest $q k 10 where category = "electronics" and price < 999 yield',
    {"q": query_vector},
)
for h in hits:
    print(h.key, h.distance, h.meta)

# Fetch one record by id
rows = db.query('get from products id "product:BOOT-42" yield')

# Scan with a filter and pagination
rows = db.query('scan in products where in_stock = true offset 0 limit 100 yield')

# The high-level client exposes the same thing as query_eql(...)
hits = client.query_eql('search in products nearest $q k 5 yield', {"q": query_vector})
`}</CodeBlock>
      <h2 id="eql">EQL grammar</h2>
      <p>
        EQL verbs are lowercase. Bindings use <code>$name</code> (not{" "}
        <code>:name</code>). Comments start with <code>#</code>.
      </p>
      <ul>
        <li>
          <code>
            search in &lt;collection&gt; nearest &lt;$binding | [1.0,0.0]&gt; [k
            &lt;int&gt;] [threshold &lt;num&gt;] [where &lt;filter&gt;] [rank_by
            &lt;distance|field&gt;] [project &lt;*|f,f&gt;] yield
          </code>
        </li>
        <li>
          <code>get from &lt;collection&gt; id "&lt;string-id&gt;" yield</code>
        </li>
        <li>
          <code>
            scan in &lt;collection&gt; [where &lt;filter&gt;] [offset
            &lt;int&gt;] [limit &lt;int&gt;] yield
          </code>
        </li>
        <li>
          <code>
            add in &lt;collection&gt; vector [..] [metadata {"{"} "k": v {"}"}]
          </code>
        </li>
        <li>
          <code>delete from &lt;collection&gt; id "&lt;string-id&gt;"</code>
        </li>
      </ul>
      <p>
        Filters: <code>field = value</code> (also <code>!=</code>,{" "}
        <code>&lt;</code>, <code>&lt;=</code>, <code>&gt;</code>,{" "}
        <code>&gt;=</code>), <code>field in [a, b]</code>,{" "}
        <code>field contains "sub"</code>, combined with <code>and</code> /{" "}
        <code>or</code> / <code>not</code> and parentheses. Values are strings,
        numbers, <code>true</code>, or <code>false</code>.
      </p>
      <p>
        The old verbs <code>seek</code>, <code>fetch</code>, <code>place</code>,
        and <code>erase</code>, and the <code>top</code> clause (in place of{" "}
        <code>k</code>), are accepted as deprecated synonyms; prefer the forms
        above.
      </p>
      <h2 id="maintenance">Maintenance</h2>
      <h3>
        <code>Database.checkpoint()</code>
      </h3>
      <p>
        Flush all pending WAL entries to the main data files and truncate the
        WAL. Reduces WAL size and shortens recovery on the next open. Does not
        rebuild the graph index.
      </p>
      <h3>
        <code>Database.compact()</code>
      </h3>
      <p>
        Run a full compaction pass: checkpoint the WAL, rewrite the data segment
        to drop dead space from deleted records, and rebuild the HNSW graph for
        vaults whose tombstone ratio exceeds their configured threshold. An
        expensive, blocking operation — schedule it off-peak.
      </p>
      <h3>
        <code>Database.close()</code>
      </h3>
      <p>
        Flush pending writes, checkpoint (for persistent databases), release file
        locks, and invalidate all vault handles from this instance. Further calls
        raise. Using the owning client as a context manager closes it for you.
      </p>
      <CodeBlock lang="python">{`db.checkpoint()
db.compact()
db.close()
`}</CodeBlock>

      <h2 id="properties">Properties</h2>
      <h3>
        <code>Database.path: str</code>
      </h3>
      <p>
        The filesystem path backing the database, or <code>":memory:"</code> for
        an ephemeral in-memory instance.
      </p>
      <h3>
        <code>Database.config: Config</code>
      </h3>
      <p>
        The fully-resolved <code>Config</code>, including values read back from
        persisted metadata.
      </p>
      <CodeBlock lang="python">{`print(db.config.dimension_val)   # e.g. 768
print(db.config.metric_val)      # 'cosine'
print(db.config.index_val)       # 'graph'
`}</CodeBlock>
      <h2 id="thread-safety">Thread Safety</h2>
      <p>ELIPS has a layered concurrency model:</p>
      <ul>
        <li>
          <strong>Instance mutex</strong> — guards the vault registry;{" "}
          <code>vault()</code> and <code>list_vaults()</code> lock it briefly. It
          is not held during search or insert.
        </li>
        <li>
          <strong>Per-vault reader-writer lock</strong> — many threads may read
          the same vault concurrently; writes take an exclusive lock.
        </li>
        <li>
          <strong>
            <code>Database</code> is thread-safe
          </strong>{" "}
          — share one instance across threads freely.
        </li>
        <li>
          <strong>
            <code>Transaction</code> is NOT thread-safe
          </strong>{" "}
          — use one per thread.
        </li>
      </ul>

      <h2 id="examples">Full Examples</h2>
      <h3>Dropping to the native layer from a client</h3>
      <CodeBlock lang="python">{`import elips

client = elips.PersistentClient("/var/lib/myapp/vectors", dimension=1536, metric="cosine")

# High-level work goes through collections
docs = client.get_or_create_collection("products")
docs.add(ids=["p1"], embeddings=[[0.1] * 1536], metadatas=[{"sku": "ABC"}])

# Drop to native for a cross-vault transaction
db = client.native
with db.begin_transaction() as txn:
    txn.vault("audit").place([0.0] * 1536, {"event": "import"})

# Raw EQL
hits = db.query('search in products nearest $q k 5 yield', {"q": [0.1] * 1536})
client.close()
`}</CodeBlock>

      <h3>Ephemeral / test database</h3>
      <CodeBlock lang="python">{`import elips

def test_vector_search():
    client = elips.Client(dimension=4, metric="cosine")
    v = client.native.vault("test")
    v.place([1.0, 0.0, 0.0, 0.0], {"label": "x-axis"})
    v.place([0.0, 1.0, 0.0, 0.0], {"label": "y-axis"})
    hits = v.seek([1.0, 0.1, 0.0, 0.0], top=1)
    assert hits[0].meta["label"] == "x-axis"
    client.close()
`}</CodeBlock>

      <h3>Read-only replica</h3>
      <CodeBlock lang="python">{`# A secondary process can open the same path read-only
reader = elips.PersistentClient("/var/lib/myapp/vectors", access_mode="read_only")
hits = reader.get_collection("products").query(query_embeddings=[0.1] * 1536, n_results=5)
`}</CodeBlock>
    </DocsShell>
  );
}
