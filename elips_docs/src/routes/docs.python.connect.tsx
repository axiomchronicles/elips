import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/connect")({
  head: () => ({
    meta: [
      { title: "Opening a database — Client & PersistentClient" },
      {
        name: "description",
        content:
          "Reference for elips.Client() and elips.PersistentClient(): database creation, settings, embedder integration, and context-manager lifecycle. connect() is the deprecated shim.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="Opening a database"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "clients", label: "Client & PersistentClient" },
        { id: "settings", label: "Settings" },
        { id: "connect-legacy", label: "connect() (deprecated)" },
        { id: "embedder-integration", label: "Embedder integration" },
        { id: "context-manager", label: "Context manager usage" },
        { id: "examples", label: "Code examples" },
      ]}
    >
      <p className="text-[18px] text-ink">
        <code>elips.Client()</code> and <code>elips.PersistentClient()</code> are the entry points
        for opening an ELIPS database in Python. They create the underlying C++ database and expose
        the Chroma-style client surface (<code>create_collection</code>,{" "}
        <code>get_or_create_collection</code>, <code>list_collections</code>, …) through a high-level{" "}
        <Link to="/docs/python/engine"><code>Engine</code></Link> handle.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        <code>Client</code> opens an ephemeral, in-memory database; <code>PersistentClient</code>{" "}
        opens (or creates) one on disk. ELIPS is dimension-first, so a <code>dimension</code> (or a
        dimension-fixing <code>embedder</code>) is required. The pre-1.1 openers{" "}
        <code>elips.connect()</code> / <code>elips.open()</code> / <code>elips.Engine</code> still
        work as deprecated shims.
      </p>

      <h2 id="clients">Client &amp; PersistentClient</h2>
      <CodeBlock lang="python">{`import elips

# Ephemeral, in-memory
client = elips.Client(dimension=384, metric="cosine")

# On disk (opens or creates)
client = elips.PersistentClient("./my_db", dimension=384)

docs = client.get_or_create_collection("documents")`}</CodeBlock>

      <h2 id="settings">Settings</h2>
      <p>
        Both clients accept the same keyword settings:
      </p>
      <ul>
        <li>
          <code>dimension</code> — vector dimension. Required for in-memory and new databases.
        </li>
        <li>
          <code>metric</code> — <code>"cosine"</code> (default), <code>"euclidean"</code>, or{" "}
          <code>"dot_product"</code>.
        </li>
        <li>
          <code>index</code> — <code>"graph"</code> (default, HNSW) or <code>"exact"</code> (flat).
        </li>
        <li>
          <code>access_mode</code> — <code>"read_write"</code> (default) or <code>"read_only"</code>.
        </li>
        <li>
          <code>embedder</code> — a Python callable or <code>LocalEmbedderConfig</code>.
        </li>
        <li>
          <code>quantization</code> — a <code>QuantParams</code> or a codec name (<code>"sq8"</code>,{" "}
          <code>"pq"</code>, <code>"opq"</code>).
        </li>
        <li>
          <code>device</code> — <code>"cpu"</code>, <code>"auto"</code>, <code>"metal"</code>,{" "}
          <code>"cuda:N"</code>, or <code>"hip:N"</code> (see{" "}
          <Link to="/docs/python/gpu">GPU</Link>); or pass a <code>gpu=GpuConfig(...)</code>.
        </li>
        <li>
          <code>segmented_storage</code>, <code>metadata_acceleration</code>,{" "}
          <code>use_default_text_embedder</code> — persistence and embedder toggles.
        </li>
      </ul>
      <p>
        There is no <code>index_type</code>, no <code>durability=</code>, and no{" "}
        <code>gpu=True</code> keyword — those never existed on the opener. Durability is configured
        on the native <code>Config</code>; see <Link to="/docs/python/durability">Durability</Link>.
      </p>

      <h2 id="connect-legacy">connect() (deprecated)</h2>
      <CodeBlock lang="python">{`def connect(
    path: str,
    *,
    dimension: int = 0,
    metric: str = "cosine",            # "cosine" | "euclidean" | "dot_product"
    index: str = "graph",              # "graph" (HNSW) | "exact" (flat)
    access_mode: str = "read_write",   # "read_write" | "read_only"
    embedder = None,
    quantization = None,               # QuantParams | "sq8" | "pq" | "opq"
    gpu = None,                        # GpuConfig | None
    device: str | None = None,         # "cpu" | "auto" | "metal" | "cuda:N" | "hip:N"
    fallback: bool = False,
) -> Engine: ...  # deprecated shim; prefer Client / PersistentClient`}</CodeBlock>

      <h2 id="embedder-integration">Embedder integration</h2>
      <p>
        Pass any text-to-vector embedding function matching the <code>Embedder</code> protocol
        (e.g. SentenceTransformers, OpenAI, Ollama):
      </p>
      <CodeBlock lang="python">{`from sentence_transformers import SentenceTransformer
import elips

model = SentenceTransformer("all-MiniLM-L6-v2")

with elips.PersistentClient("./my_db", dimension=384, embedder=model.encode) as client:
    docs = client.get_or_create_collection("docs")
    docs.add(documents=["ELIPS makes vector search embedded and fast."])

    hits = docs.query(query_texts="fast vector search", n_results=3)
    print(hits[0].text)`}</CodeBlock>

      <h2 id="context-manager">Context manager usage</h2>
      <p>
        Using a client inside a <code>with</code> block flushes, checkpoints, and closes the
        database cleanly on exit:
      </p>
      <CodeBlock lang="python">{`with elips.PersistentClient("vectors.elips", dimension=1536) as client:
    # read/write operations
    pass
# database handle is safely closed here`}</CodeBlock>

      <h2 id="examples">Code examples</h2>
      <h3 id="in-memory">In-memory volatile DB</h3>
      <CodeBlock lang="python">{`import elips

with elips.Client(dimension=64, metric="euclidean") as client:
    quick = client.get_or_create_collection("quick_test")
    quick.add(embeddings=[[0.1] * 64], metadatas=[{"tag": "test"}])`}</CodeBlock>

      <h3 id="persistent-gpu">Persistent disk DB with GPU acceleration</h3>
      <CodeBlock lang="python">{`import elips

with elips.PersistentClient("./gpu_db", dimension=1536, device="metal") as client:
    print(f"Backend: {client.native.gpu_info().backend}")`}</CodeBlock>
    </DocsShell>
  );
}
