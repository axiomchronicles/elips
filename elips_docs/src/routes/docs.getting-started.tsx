import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/getting-started")({
  head: () => ({
    meta: [
      { title: "Quick Start — ELIPS Docs" },
      {
        name: "description",
        content:
          "In-memory hello world, persistent DB, embedders, filtering, hybrid search — complete runnable ELIPS examples.",
      },
      { property: "og:title", content: "Quick Start — ELIPS" },
    ],
    links: [{ rel: "canonical", href: "/docs/getting-started" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Start"
      title="Quick start"
      toc={[
        { id: "what-is-elips", label: "What is ELIPS?" },
        { id: "install", label: "Install" },
        { id: "hello-world", label: "Hello world (in-memory)" },
        { id: "persistent", label: "Persistent database" },
        { id: "embedder", label: "Custom embedder" },
        { id: "filtering", label: "Metadata filtering" },
        { id: "hybrid", label: "Hybrid search" },
        { id: "transactions", label: "Transactions" },
        { id: "gpu", label: "GPU acceleration" },
        { id: "next", label: "Next steps" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS is an embedded, local-first vector database. No server to run — just open a path and
        start searching. Five minutes to your first semantic search.
      </p>

      <h2 id="what-is-elips">What is ELIPS?</h2>
      <p>
        ELIPS (Embedded Local Index &amp; Persistence System) is a C++23 library that gives you an
        in-process vector database with the same deployment philosophy as SQLite: open a file, use
        it, close it. No daemon, no network, no container.
      </p>
      <ul>
        <li>
          <strong>Embedded:</strong> ships as a shared/static library + Python extension. Zero
          external services.
        </li>
        <li>
          <strong>Persistent:</strong> WAL-backed on-disk storage with crash recovery. Also works
          fully in-memory.
        </li>
        <li>
          <strong>Fast:</strong> HNSW graph index for ANN search; optional GPU acceleration via
          CUDA, HIP, or Metal.
        </li>
        <li>
          <strong>Flexible:</strong> cosine, Euclidean, dot-product metrics; metadata filtering;
          hybrid vector+lexical search; EQL query language.
        </li>
      </ul>

      <h2 id="install">Install</h2>
      <p>Build from the repository — there is no PyPI wheel yet:</p>
      <CodeBlock lang="bash">{`# 1. Clone and configure
git clone https://github.com/axiomchronicles/elips
cd elips
cmake -S . -B build -G Ninja \\
  -DELIPS_BUILD_PYTHON=ON \\
  -DCMAKE_BUILD_TYPE=Release

# 2. Build
cmake --build build --target elips_pymodule -j$(nproc)

# 3. Put the package on PYTHONPATH
export PYTHONPATH=$PWD/bindings/python

# 4. Verify
python -c "import elips; print(elips.__version__)"
# 1.0.0`}</CodeBlock>
      <p>
        For GPU support add <code>-DELIPS_GPU_ENABLED=ON</code>. See{" "}
        <Link to="/docs/installation">Installation</Link> for full toolchain requirements.
      </p>

      <h2 id="hello-world">Hello world (in-memory)</h2>
      <p>
        Open an in-memory database, write three records, and run a text search — no embedder
        configuration required:
      </p>
      <CodeBlock lang="python">{`import elips

# Open an in-memory engine. Dimension=2 for demo.
# The built-in local text embedder auto-attaches.
engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("documents")

# Write records with text + metadata
arena.write(text="alpha design note", meta={"kind": "design", "year": 2024})
arena.write(text="beta deployment runbook", meta={"kind": "ops", "year": 2023})
arena.write(text="gamma security policy", meta={"kind": "security", "year": 2024})

# Text search — returns Hit objects sorted by similarity
hits = arena.probe_text("alpha", top=2)
for hit in hits:
    print(hit.text, hit.distance, hit.meta)

engine.close()

# Output (approx):
# alpha design note 0.0 {'kind': 'design', 'year': 2024}
# gamma security policy 0.42 {'kind': 'security', 'year': 2024}`}</CodeBlock>
      <p>
        <strong>What just happened:</strong> <code>connect()</code> auto-attached the built-in
        local text embedder. <code>arena.write(text=...)</code> embedded the text and stored both
        the vector and the source text. <code>probe_text()</code> embedded the query and ran ANN
        search.
      </p>

      <h2 id="persistent">Persistent database</h2>
      <p>
        Pass a filesystem path instead of <code>":memory:"</code>. The database persists across
        process restarts:
      </p>
      <CodeBlock lang="python">{`import elips, tempfile, os

path = tempfile.mkdtemp()

# Write session
with elips.connect(path, dimension=128) as engine:
    arena = engine.arena("papers")
    key = arena.write(
        vector=[0.1] * 128,
        meta={"title": "Attention is all you need", "year": 2017},
    )
    engine.checkpoint()          # flush WAL to snapshot
    print("wrote key:", key)

# Reopen and search
with elips.connect(path) as engine:
    arena = engine.arena("papers")
    print("count after reopen:", arena.count())
    # 1 — record survived the restart`}</CodeBlock>
      <p>
        ELIPS writes mutations to a WAL (<code>wal.log</code>) before acknowledging them. On reopen
        it replays the WAL on top of the last snapshot. <code>checkpoint()</code> merges the WAL
        into the snapshot and truncates the log.
      </p>

      <h2 id="embedder">Custom embedder</h2>
      <p>
        Bring your own model — anything that turns a list of strings into a list of float vectors:
      </p>
      <CodeBlock lang="python">{`import elips
from sentence_transformers import SentenceTransformer

# Load model once, outside connect()
model = SentenceTransformer("all-MiniLM-L6-v2")  # dim=384

def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()

engine = elips.connect(
    ":memory:",
    dimension=384,
    metric="cosine",
    embedder=embed,
    embedder_provider="sentence-transformers",
    embedder_model="all-MiniLM-L6-v2",
    use_default_text_embedder=False,   # don't auto-attach builtin
)
arena = engine.arena("docs")

arena.write(text="Machine learning fundamentals")
arena.write(text="Neural network architecture")

hits = arena.probe_text("deep learning basics", top=2)
print(hits[0].text)   # Machine learning fundamentals

engine.close()`}</CodeBlock>

      <h2 id="filtering">Metadata filtering</h2>
      <p>
        Combine vector search with metadata predicates using the <code>Filter</code> API:
      </p>
      <CodeBlock lang="python">{`import elips

engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("articles")

arena.write(vector=[1.0, 0.0], meta={"kind": "design", "year": 2024, "author": "alice"})
arena.write(vector=[0.9, 0.1], meta={"kind": "ops",    "year": 2023, "author": "bob"})
arena.write(vector=[0.8, 0.2], meta={"kind": "design", "year": 2023, "author": "carol"})

# Only design articles from 2024+
f = elips.Filter().field("kind").equals("design").field("year").ge(2024)
hits = arena.probe([1.0, 0.0], top=10, where=f)
print([h.meta["author"] for h in hits])  # ['alice']

# OR combinator: design OR ops from 2023+
f2 = (
    elips.Filter().field("kind").equals("design")
    .or_(elips.Filter().field("kind").equals("ops"))
    .and_(elips.Filter().field("year").ge(2023))
)
hits2 = arena.probe([1.0, 0.0], top=10, where=f2)
print(len(hits2))  # 3

engine.close()`}</CodeBlock>

      <h2 id="hybrid">Hybrid search</h2>
      <p>
        Blend vector similarity with lexical (keyword) overlap using{" "}
        <code>probe_hybrid()</code>:
      </p>
      <CodeBlock lang="python">{`import elips

engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("docs")

arena.write(vector=[1.0, 0.0], text="ELIPS vector database overview")
arena.write(vector=[0.5, 0.5], text="PostgreSQL relational database")
arena.write(vector=[0.9, 0.1], text="ELIPS GPU acceleration guide")

# Hybrid: 75% vector score + 25% lexical overlap on "ELIPS"
hits = arena.probe_hybrid(
    vector=[1.0, 0.0],
    text="ELIPS",
    top=3,
    lexical_weight=0.25,
)
print([h.text for h in hits])
# ELIPS records rank higher due to lexical boost

engine.close()`}</CodeBlock>

      <h2 id="transactions">Transactions</h2>
      <p>Group multiple writes into an atomic batch — all succeed or none apply:</p>
      <CodeBlock lang="python">{`import elips

engine = elips.connect(":memory:", dimension=2)

# Use begin_transaction() from the low-level Database handle
with engine.raw.begin_transaction() as txn:
    docs_txn = txn.vault("documents")
    ops_txn  = txn.vault("ops-logs")

    k1 = docs_txn.place([1.0, 0.0], {"title": "Proposal"})
    k2 = ops_txn.place([0.5, 0.5], {"event": "proposal_created", "doc_id": k1})
    # txn.commit() called automatically on clean exit
    # txn.rollback() called automatically on exception

engine.close()`}</CodeBlock>

      <h2 id="gpu">GPU acceleration</h2>
      <p>
        If your build includes GPU support, ELIPS auto-selects the best backend. Opt in via{" "}
        <code>GpuConfig</code>:
      </p>
      <CodeBlock lang="python">{`import elips

if not elips._has_gpu:
    raise RuntimeError("build without -DELIPS_GPU_ENABLED=ON")

# Discover devices
devices = elips.gpu_devices()
print(devices[0].name, devices[0].total_device_memory_bytes)

# Use GPU for index build and search
cfg = elips.Config().dimension(1024).metric("cosine").gpu(
    elips.GpuConfig(
        policy=elips.GpuPolicy.PreferGpu,
        algorithm=elips.GpuIndexAlgorithm.CagraGraph,
        enable_fp16_search=True,
    )
)
engine = elips.connect_with_config(":memory:", cfg)
arena = engine.arena("embeddings")

# ... write vectors, probe as normal — GPU handles the search
engine.close()`}</CodeBlock>

      <h2 id="next">Next steps</h2>
      <ul>
        <li>
          <Link to="/docs/python/arena">Arena reference</Link> — full method documentation with
          all parameters
        </li>
        <li>
          <Link to="/docs/python/connect">connect() reference</Link> — all configuration options
        </li>
        <li>
          <Link to="/docs/python/filtering">Filtering</Link> — Filter API in depth
        </li>
        <li>
          <Link to="/docs/python/config">Config &amp; GraphParams</Link> — HNSW tuning,
          durability, access modes
        </li>
        <li>
          <Link to="/docs/gpu/overview">GPU overview</Link> — GPU backends, memory, algorithms
        </li>
        <li>
          <Link to="/docs/guides">Guides</Link> — task-shaped walkthroughs for common workflows
        </li>
        <li>
          <Link to="/docs/eql">EQL</Link> — query language for complex programmatic queries
        </li>
        <li>
          <Link to="/docs/cpp/overview">C++ quick start</Link> — embed ELIPS directly in C++23
          applications
        </li>
      </ul>
    </DocsShell>
  );
}
