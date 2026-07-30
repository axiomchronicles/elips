import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/wal")({
  head: () => ({
    meta: [
      { title: "Recovery & Introspection | ELIPS Python Reference" }
    ],
    links: []
  }),
  component: Page
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="Recovery & Introspection"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "wal-replay", label: "WAL Replay" },
        { id: "crash-forensics-example", label: "Crash Forensics Example" },
        { id: "eql-ast", label: "EQL AST" },
        { id: "eql-guardrails-example", label: "EQL Guardrails Example" },
        { id: "index-snapshots", label: "Index Snapshots" },
        { id: "embedder-introspection", label: "Embedder Introspection" },
        { id: "vector-utilities", label: "Vector Utilities" },
        { id: "metric-utilities", label: "Metric Utilities" }
      ]}
    >
      <h2 id="overview">Overview</h2>
      <p>
        This page documents the advanced diagnostic and utility APIs available in the ELIPS Python client.
        These APIs allow you to replay the write-ahead log (WAL) for forensic analysis, parse EQL queries into abstract syntax trees (AST) for linting and security checks, introspect index snapshots and embedder configurations, and perform common vector and metric mathematical operations.
      </p>

      <h2 id="wal-replay">WAL Replay API</h2>
      <p>
        The WAL replay API enables you to read the contents of a Write-Ahead Log without opening the full database.
        This is primarily used for post-crash reconciliation to answer "what did the DB actually acknowledge before it died?".
        The replay is entirely read-only and mutates no database files.
        Corrupt tails are dropped silently (not raised), and unterminated transactions (begin without a commit) are omitted, matching what the database recovery mechanism would apply.
      </p>
      <ul>
        <li><code>replay_wal(path: str) -&gt; list[WalEntry]</code> — Parses the WAL file at the given path and returns a sequence of entries.</li>
      </ul>
      <p>
        The returned <code>WalEntry</code> objects have the following properties:
      </p>
      <ul>
        <li><code>op</code> (<code>WalOp</code>) — The operation type.</li>
        <li><code>vault</code> (<code>str</code>) — The target vault.</li>
        <li><code>id</code> (<code>str</code>) — The record ID.</li>
        <li><code>vector</code> (<code>tuple[float, ...]</code>) — The vector payload (for inserts).</li>
        <li><code>data</code> (<code>dict</code>) — The metadata payload.</li>
        <li><code>document</code> (<code>Optional[DocumentAttachment]</code>) — Attached document data.</li>
        <li><code>chunk</code> (<code>Optional[ChunkInfo]</code>) — Chunking metadata.</li>
        <li><code>lineage</code> (<code>Optional[EmbeddingLineage]</code>) — Embedding traceability info.</li>
      </ul>
      <p>
        The <code>WalOp</code> enum includes:
      </p>
      <ul>
        <li><code>insert</code></li>
        <li><code>erase</code></li>
        <li><code>insert_ex</code> (insertion with attachments)</li>
        <li><code>txn_begin</code></li>
        <li><code>txn_commit</code></li>
      </ul>
      <p>
        For details on transaction semantics, see <Link to="/docs/internals/transaction-engine">Transaction Engine</Link>. For details on the underlying binary format, see <Link to="/docs/storage">Storage Architecture</Link>.
      </p>

      <h2 id="crash-forensics-example">Crash Forensics Example</h2>
      <p>
        The following example demonstrates how to reconcile acknowledged writes with the live database state after a crash.
      </p>
      <CodeBlock lang="python">
{`import elips
from collections import Counter

entries = elips.replay_wal("/data/vectors/wal.log")
print(Counter(e.op.name for e in entries))

acked = {e.id for e in entries if e.op != elips.WalOp.erase}
erased = {e.id for e in entries if e.op == elips.WalOp.erase}

db = elips.open("/data/vectors")
vault = db.vault(entries[0].vault) if entries else None
missing = [rid for rid in acked - erased
           if vault is not None and vault.fetch(rid) is None]
if missing:
    raise SystemExit(f"{len(missing)} acknowledged writes lost")
print("recovered cleanly;", len(acked - erased), "live writes")`}
      </CodeBlock>

      <h2 id="eql-ast">EQL AST API</h2>
      <p>
        The EQL AST API allows you to parse, validate, and tokenize EQL (ELIPS Query Language) statements without executing them.
      </p>
      <ul>
        <li><code>parse_eql(source: str) -&gt; Statement</code> — Parses the query string and returns an AST root node. Raises <code>ParseError</code> on syntax failure.</li>
        <li><code>validate_eql(source: str) -&gt; None</code> — Validates the syntax without constructing the full AST tree.</li>
        <li><code>tokenize_eql(source: str) -&gt; list[Token]</code> — Lexes the source into tokens.</li>
      </ul>
      
      <p>A <code>Token</code> contains:</p>
      <ul>
        <li><code>kind</code> (<code>TokenKind</code>: <code>word</code>, <code>number</code>, <code>string</code>, <code>punct</code>, or <code>end</code>)</li>
        <li><code>text</code> (<code>str</code>)</li>
        <li><code>number</code> (<code>float</code>)</li>
        <li><code>is_integer</code> (<code>bool</code>)</li>
      </ul>

      <p>The <code>Statement</code> returned by <code>parse_eql</code> is a union of one of the following classes:</p>
      
      <h3>SearchStatement</h3>
      <ul>
        <li><code>vault</code> (<code>str</code>)</li>
        <li><code>query</code> (<code>VectorRef</code>)</li>
        <li><code>top</code> (<code>Optional[int]</code>)</li>
        <li><code>threshold</code> (<code>Optional[float]</code>)</li>
        <li><code>where</code> (<code>Filter</code>)</li>
        <li><code>rank_by</code> (<code>Optional[str]</code>)</li>
        <li><code>projection</code> (<code>list[str]</code>)</li>
      </ul>

      <h3>FetchStatement</h3>
      <ul>
        <li><code>vault</code> (<code>str</code>)</li>
        <li><code>id</code> (<code>str</code>)</li>
      </ul>

      <h3>ScanStatement</h3>
      <ul>
        <li><code>vault</code> (<code>str</code>)</li>
        <li><code>where</code> (<code>Filter</code>)</li>
        <li><code>offset</code> (<code>Optional[int]</code>)</li>
        <li><code>limit</code> (<code>Optional[int]</code>)</li>
      </ul>

      <h3>InsertStatement</h3>
      <ul>
        <li><code>vault</code> (<code>str</code>)</li>
        <li><code>vector</code> (<code>VectorRef</code>)</li>
        <li><code>data</code> (<code>dict</code>)</li>
      </ul>

      <h3>DeleteStatement</h3>
      <ul>
        <li><code>vault</code> (<code>str</code>)</li>
        <li><code>id</code> (<code>str</code>)</li>
      </ul>

      <p>
        A <code>VectorRef</code> represents a vector literal or parameter binding. It has fields <code>literal (list[float])</code> and <code>binding (str)</code>, one of which will be non-empty.
      </p>

      <h2 id="eql-guardrails-example">EQL Guardrails Example</h2>
      <p>
        By leveraging <code>parse_eql</code>, you can implement multi-tenant query linters and security guardrails that inspect queries before dispatching them.
      </p>
      <CodeBlock lang="python">
{`import elips
MAX_TOP = 100

def check(query: str) -> None:
    stmt = elips.parse_eql(query)  # raises ParseError on bad syntax
    if isinstance(stmt, elips.SearchStatement):
        if stmt.top is None or stmt.top > MAX_TOP:
            raise ValueError(f"seek needs top <= {MAX_TOP}")
        if stmt.where.matches_all():
            raise ValueError("seek needs a where clause")
        if stmt.where.exact_constraints() is None:
            raise ValueError("where clause is not index-accelerable")
    elif isinstance(stmt, elips.ScanStatement):
        if stmt.limit is None:
            raise ValueError("scan needs a limit")

check('seek in docs nearest $q top 10 where tenant = "acme" yield')`}
      </CodeBlock>

      <h2 id="index-snapshots">Index Snapshots API</h2>
      <p>
        Index snapshots provide read-only introspective access to internal index layouts.
      </p>
      <p>
        The <code>IndexSnapshotKind</code> enum defines the hardware and layout target: <code>unknown</code>, <code>exact</code>, <code>graph</code>, <code>gpu_brute_force</code>, <code>gpu_ivf_flat</code>, <code>gpu_ivf_pq</code>, <code>gpu_graph</code>, <code>gpu_hybrid</code>, <code>gpu_distributed</code>.
      </p>
      <p>
        The <code>IndexSnapshot</code> object exposes the following properties:
      </p>
      <ul>
        <li><code>kind</code> (<code>IndexSnapshotKind</code>)</li>
        <li><code>metric</code> (<code>Metric</code>)</li>
        <li><code>dimension</code> (<code>int</code>)</li>
        <li><code>ids</code> (<code>list[str]</code>)</li>
        <li><code>vectors</code> (<code>list[float]</code>) (if stored natively)</li>
        <li><code>ivf</code> (<code>Optional[IvfSnapshot]</code>)</li>
        <li><code>pq</code> (<code>Optional[PqSnapshot]</code>)</li>
        <li><code>__len__()</code> returns the number of items.</li>
      </ul>
      <p>
        <code>IvfSnapshot</code> represents Inverted File structures with properties: <code>n_lists</code>, <code>n_probe</code>, <code>centroids (list[float])</code>, and <code>assignments (list[int])</code>.
      </p>
      <p>
        <code>PqSnapshot</code> represents Product Quantization structures with properties: <code>pq_dim</code>, <code>pq_bits</code>, <code>codebook (list[float])</code>, and <code>codes (list[int])</code>.
      </p>

      <h2 id="embedder-introspection">Embedder Introspection API</h2>
      <p>
        You can inspect how the Python client resolves and configures embedding models using <code>describe_local_embedder</code>.
      </p>
      <ul>
        <li><code>describe_local_embedder(config=..., fallback_dimension=0, auto_attached=False) -&gt; TextEmbedderInfo</code></li>
      </ul>
      <p>
        The returned <code>TextEmbedderInfo</code> exposes properties such as: <code>kind</code> (<code>TextEmbedderKind</code>), <code>provider</code>, <code>model</code>, <code>revision</code>, <code>backend</code>, <code>dimension</code>, <code>fingerprint</code>, <code>storage_path</code>, <code>rehydratable</code>, <code>loaded</code>, and <code>auto_attached</code>.
      </p>
      <p>
        <code>TextEmbedderKind</code> specifies whether the model is <code>external</code> (e.g. remote API) or <code>local_builtin</code> (running directly within ELIPS).
      </p>

      <h2 id="vector-utilities">Vector Utilities</h2>
      <p>
        These utilities are commonly used for manipulating vectors and IDs.
      </p>
      <ul>
        <li><code>generate_id() -&gt; str</code> — Returns a fresh UUIDv7 hex string. This is useful when the caller must know the ID prior to the write operation (e.g., to publish to an event queue within the same transaction).</li>
        <li><code>is_valid_id(id: str) -&gt; bool</code> — Validates the format of a record ID.</li>
        <li><code>normalize(vector) -&gt; tuple[float, ...]</code> — L2-normalizes the given vector (zero vectors are returned unchanged).</li>
        <li><code>magnitude(vector) -&gt; float</code> — Computes the Euclidean L2 norm of the vector.</li>
      </ul>

      <h2 id="metric-utilities">Metric Utilities</h2>
      <ul>
        <li><code>distance(metric, a, b) -&gt; float</code> — Computes the ordering-normalized distance between two vectors according to the specified metric.</li>
        <li><code>requires_normalization(metric) -&gt; bool</code> — Returns <code>True</code> only for the cosine metric.</li>
        <li><code>metric_to_string(metric) -&gt; str</code> — Returns the string name for a given <code>Metric</code>.</li>
        <li><code>metric_from_string(name) -&gt; Metric</code> — Returns the corresponding <code>Metric</code> enum from its string name.</li>
      </ul>
    </DocsShell>
  );
}
