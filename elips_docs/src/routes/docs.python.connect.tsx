import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/connect")({
  head: () => ({
    meta: [
      { title: "elips.connect() — Python Connection Reference" },
      {
        name: "description",
        content:
          "Complete reference for elips.connect() and elips.connect_with_config(): database creation, embedder integration, durability levels, and context manager lifecycle.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="elips.connect()"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "connect-signature", label: "elips.connect()" },
        { id: "connect-with-config", label: "elips.connect_with_config()" },
        { id: "embedder-integration", label: "Embedder Integration" },
        { id: "context-manager", label: "Context Manager Usage" },
        { id: "examples", label: "Code Examples" },
      ]}
    >
      <p className="text-[18px] text-ink">
        <code>elips.connect()</code> is the primary, modern entry point for opening or initializing an ELIPS database in Python. It creates the underlying C++ database instance and wraps it in a high-level <Link to="/docs/python/engine"><code>Engine</code></Link> handle.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        The modern Python surface favors <code>elips.connect()</code> over raw database handle instantiation. It automatically configures vector dimensions, distance metrics, HNSW parameters, text embedder models, and durability guarantees.
      </p>

      <h2 id="connect-signature">elips.connect()</h2>
      <CodeBlock lang="python">{`def connect(
    path: str = ":memory:",
    *,
    dimension: int = 128,
    metric: str = "cosine",             # "cosine", "euclidean", "dot_product"
    index_type: str = "hnsw",           # "hnsw", "exact", "ivf_flat", "ivf_pq"
    embedder: Embedder | None = None,   # Python callable or sentence-transformers model
    durability: str = "sync_on_commit", # "sync_on_commit", "wal_only", "in_memory"
    access_mode: str = "read_write",    # "read_write", "read_only"
    gpu: bool = False,
) -> Engine: ...`}</CodeBlock>

      <h2 id="connect-with-config">elips.connect_with_config()</h2>
      <CodeBlock lang="python">{`def connect_with_config(
    path: str,
    config: Config,
    *,
    embedder: Embedder | None = None
) -> Engine: ...`}</CodeBlock>

      <h2 id="embedder-integration">Embedder Integration</h2>
      <p>
        You can pass any text-to-vector embedding function or object matching the <code>Embedder</code> protocol (e.g. SentenceTransformers, OpenAI, Ollama):
      </p>
      <CodeBlock lang="python">{`from sentence_transformers import SentenceTransformer
import elips

model = SentenceTransformer("all-MiniLM-L6-v2")

# Pass model directly to connect
with elips.connect("./my_db", dimension=384, embedder=model.encode) as engine:
    arena = engine.arena("docs")
    arena.write(text="ELIPS makes vector search embedded and fast.")
    
    hits = arena.probe_text("fast vector search", top=3)
    print(hits[0].text)`}</CodeBlock>

      <h2 id="context-manager">Context Manager Usage</h2>
      <p>
        Using <code>connect()</code> inside a Python <code>with</code> block guarantees that the database is flushed, checkpointed, and closed cleanly upon exiting the block:
      </p>
      <CodeBlock lang="python">{`with elips.connect("vectors.elips", dimension=1536) as engine:
    # Perform read/write operations
    pass
# Database handle is safely closed here`}</CodeBlock>

      <h2 id="examples">Code Examples</h2>
      <h3 id="in-memory">In-Memory Volatile DB</h3>
      <CodeBlock lang="python">{`import elips

with elips.connect(":memory:", dimension=64, metric="euclidean") as engine:
    arena = engine.arena("quick_test")
    arena.write(vector=[0.1] * 64, meta={"tag": "test"})`}</CodeBlock>

      <h3 id="persistent-gpu">Persistent Disk DB with GPU Acceleration</h3>
      <CodeBlock lang="python">{`import elips

with elips.connect(
    "./gpu_db",
    dimension=1536,
    metric="cosine",
    durability="wal_only",
    gpu=True
) as engine:
    print(f"Connected to GPU: {engine.gpu_info()}")`}</CodeBlock>
    </DocsShell>
  );
}
