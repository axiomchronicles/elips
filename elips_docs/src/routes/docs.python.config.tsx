import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/config")({
  head: () => ({
    meta: [
      { title: "Config — ELIPS Python API Docs" },
      {
        name: "description",
        content:
          "Complete reference for the ELIPS Python Config fluent builder, HnswParams, LocalEmbedderConfig, and GpuConfig — including the persisted identity concept and ConfigError.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="PYTHON API · LOW-LEVEL"
      title="Config"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "persisted-identity", label: "Persisted Identity" },
        { id: "constructor", label: "Config() Constructor" },
        { id: "dimension", label: ".dimension()" },
        { id: "metric", label: ".metric()" },
        { id: "index", label: ".index()" },
        { id: "graph-params", label: ".graph_params() & HnswParams" },
        { id: "access-mode", label: ".access_mode()" },
        { id: "storage", label: "Storage Options" },
        { id: "quantization", label: "Quantization" },
        { id: "embedder", label: "Text Embedder" },
        { id: "local-embedder", label: "LocalEmbedderConfig" },
        { id: "gpu", label: "GPU Selection" },
        { id: "read-properties", label: "Read Properties" },
        { id: "config-error", label: "ConfigError" },
        { id: "ingest-serve", label: "Two-Phase Ingest/Serve" },
        { id: "examples", label: "Full Examples" },
      ]}
    >
      {/* ── Overview ───────────────────────────────────────────── */}
      <h2 id="overview">Overview</h2>
      <p>
        <code>Config</code> is a fluent (method-chaining) builder that
        collects all settings for a database before it is opened. In most code
        you pass settings directly as keyword arguments to{" "}
        <a href="/docs/python/database">
          <code>elips.Client</code>
        </a>{" "}
        (in-memory) or <code>elips.PersistentClient</code> (on disk). A{" "}
        <code>Config</code> object is useful when you want to assemble and reuse
        a configuration programmatically; the legacy{" "}
        <code>elips.connect_with_config(path, config)</code> entry point accepts
        one directly.
      </p>

      <CodeBlock lang="python">{`import elips
from elips import Config, HnswParams

# Primary: settings as keyword arguments
client = elips.PersistentClient(
    "/data/vectors.elips",
    dimension=768,
    metric="cosine",
    index="graph",
    metadata_acceleration=True,
)

# Or build a Config and open with the legacy config entry point
cfg = (
    Config()
    .dimension(768)
    .metric("cosine")
    .index("graph")
    .graph_params(HnswParams(max_connections=32, ef_construction=300))
    .metadata_acceleration(True)
)
client = elips.connect_with_config("/data/vectors.elips", cfg)  # legacy
`}</CodeBlock>

      {/* ── Persisted Identity ─────────────────────────────────── */}
      <h2 id="persisted-identity">Persisted Identity</h2>
      <p>
        Certain fields are written to the database's on-disk metadata on first
        open and <strong>cannot be changed</strong> for the lifetime of the
        database file. These are called <em>persisted identity</em> fields:
      </p>
      <ul>
        <li>
          <strong>dimension</strong>
        </li>
        <li>
          <strong>metric</strong>
        </li>
        <li>
          <strong>index type</strong> (<code>"graph"</code> vs{" "}
          <code>"exact"</code>)
        </li>
      </ul>
      <p>
        If you open an existing database with a <code>Config</code> that
        conflicts with these stored values, ELIPS raises a{" "}
        <code>ConfigError</code> immediately — before any I/O is done on the
        data files. This prevents silent data corruption.
      </p>
      <p>
        Non-identity fields (<code>access_mode</code>,{" "}
        <code>graph_params</code>, <code>metadata_acceleration</code>,{" "}
        <code>embedder</code>) can be changed between opens.
      </p>

      <CodeBlock lang="python">{`import elips

# First open — establishes identity
client = elips.PersistentClient(
    "/data/v.elips", dimension=768, metric="cosine", index="graph"
)
client.close()

# Reopen read-only — fine (non-identity change)
client = elips.PersistentClient("/data/v.elips", access_mode="read_only")
client.close()

# Reopen with a different dimension — ConfigError
try:
    client = elips.PersistentClient("/data/v.elips", dimension=512)
except elips.ConfigError as e:
    print(e)  # Dimension mismatch: expected 768, got 512
`}</CodeBlock>

      {/* ── Constructor ────────────────────────────────────────── */}
      <h2 id="constructor">
        <code>Config()</code> Constructor
      </h2>
      <CodeBlock lang="python">{`from elips import Config
cfg = Config()
`}</CodeBlock>
      <p>
        Creates a blank <code>Config</code> with all fields at their default
        values. Every subsequent method call returns <code>self</code>, enabling
        chaining. You must call at least <code>.dimension()</code> when creating
        a new database.
      </p>

      {/* ── dimension() ────────────────────────────────────────── */}
      <h2 id="dimension">
        <code>{"Config.dimension(dim) -> Config"}</code>
      </h2>
      <CodeBlock lang="python">{`Config.dimension(dim: int) -> Config
`}</CodeBlock>
      <p>
        Sets the vector dimensionality. Required for new databases; ignored (but
        validated) on reopen. Valid range: 1–65 536.
      </p>
      <ul>
        <li>
          <strong>Persisted identity</strong> — cannot be changed after first
          open.
        </li>
        <li>
          All vaults within a database share this dimension.
        </li>
        <li>
          Common values: <code>384</code> (MiniLM-L6), <code>768</code>{" "}
          (BERT-base), <code>1536</code> (OpenAI text-embedding-3-small),{" "}
          <code>3072</code> (text-embedding-3-large).
        </li>
      </ul>

      {/* ── metric() ───────────────────────────────────────────── */}
      <h2 id="metric">
        <code>{"Config.metric(metric_str) -> Config"}</code>
      </h2>
      <CodeBlock lang="python">{`Config.metric(metric_str: str) -> Config
# metric_str ∈ {"cosine", "euclidean", "dot_product"}
`}</CodeBlock>
      <p>
        Chooses the distance function used for all vector comparisons. Must
        match the metric used to train/produce the embeddings.
      </p>
      <ul>
        <li>
          <strong>
            <code>"cosine"</code>
          </strong>{" "}
          (default) — Angular distance. Normalises vectors before comparison.
          Best for sentence embeddings from models like BERT, OpenAI, and
          Sentence-Transformers.
        </li>
        <li>
          <strong>
            <code>"euclidean"</code>
          </strong>{" "}
          — L2 distance. Does not normalise. Use when magnitude matters (e.g.,
          audio embeddings, pixel features).
        </li>
        <li>
          <strong>
            <code>"dot_product"</code>
          </strong>{" "}
          — Inner product. Highest score = most similar (note: results are
          returned in descending order unlike cosine/euclidean). Appropriate
          for embeddings already normalised to unit length where speed matters.
        </li>
      </ul>
      <p>
        <strong>Persisted identity</strong> — cannot be changed after first
        open.
      </p>

      {/* ── index() ────────────────────────────────────────────── */}
      <h2 id="index">
        <code>{"Config.index(type_str) -> Config"}</code>
      </h2>
      <CodeBlock lang="python">{`Config.index(type_str: str) -> Config
# type_str ∈ {"graph", "exact"}
`}</CodeBlock>
      <ul>
        <li>
          <strong>
            <code>"graph"</code>
          </strong>{" "}
          (default) — HNSW approximate nearest-neighbour index. O(log n)
          search time. Suitable for &gt; 10 k vectors.
        </li>
        <li>
          <strong>
            <code>"exact"</code>
          </strong>{" "}
          — Brute-force linear scan. O(n) search time. Perfect recall
          (100 %). Appropriate only for small datasets (&lt; ~50 k vectors) or
          ground-truth benchmarking.
        </li>
      </ul>
      <p>
        <strong>Persisted identity</strong> — cannot be changed after first
        open.
      </p>

      {/* ── graph_params() ─────────────────────────────────────── */}
      <h2 id="graph-params">
        <code>Config.graph_params()</code> &amp; <code>HnswParams</code>
      </h2>
      <CodeBlock lang="python">{`Config.graph_params(params: HnswParams) -> Config
`}</CodeBlock>

      <h3>
        <code>HnswParams</code>
      </h3>
      <CodeBlock lang="python">{`from elips import HnswParams

params = HnswParams(
    max_connections:  int   = 16,
    ef_construction:  int   = 200,
    ef_search:        int   = 50,
    compaction_ratio: float = 0.2,
)
`}</CodeBlock>

      <h3>Parameters in depth</h3>
      <ul>
        <li>
          <strong>
            <code>max_connections</code> (M)
          </strong>{" "}
          — Maximum number of bi-directional links each node has at each HNSW
          layer. Higher values improve recall and graph connectivity but
          increase memory usage (~8 bytes per connection per record) and slow
          down insertions.
          <br />
          Typical values: <code>8</code>–<code>64</code>. Default:{" "}
          <code>16</code>.
        </li>
        <li>
          <strong>
            <code>ef_construction</code>
          </strong>{" "}
          — Beam width during graph construction. Controls how many candidate
          neighbours are explored when inserting a new node. Higher values
          produce a better-quality graph (higher recall at search time) at the
          cost of slower insertion throughput.
          <br />
          Rule of thumb: <code>ef_construction</code> &ge;{" "}
          <code>2 × max_connections</code>. Default: <code>200</code>.
        </li>
        <li>
          <strong>
            <code>ef_search</code>
          </strong>{" "}
          — Beam width during query. Controls the search-time recall/latency
          trade-off. Can be tuned at runtime by changing <code>HnswParams</code>{" "}
          without rebuilding the graph (unlike <code>max_connections</code> and{" "}
          <code>ef_construction</code>).
          <br />
          Must satisfy <code>ef_search</code> &ge; <code>n_results</code>{" "}
          (number of results requested). Default: <code>50</code>.
        </li>
        <li>
          <strong>
            <code>compaction_ratio</code>
          </strong>{" "}
          — Fraction of live records that may be tombstoned (erased but not
          yet purged) before an automatic graph rebuild is triggered on the
          next write. Set to <code>0.0</code> to disable automatic compaction.
          Default: <code>0.2</code> (20 %).
        </li>
      </ul>

      <CodeBlock lang="python">{`# High-recall configuration (favours quality over speed)
high_recall = HnswParams(
    max_connections=32,
    ef_construction=400,
    ef_search=200,
)

# Low-latency configuration (favours speed, slight recall trade-off)
low_latency = HnswParams(
    max_connections=16,
    ef_construction=100,
    ef_search=25,
)

cfg = Config().dimension(768).metric("cosine").graph_params(high_recall)
`}</CodeBlock>

      <p>
        <strong>Note on durability:</strong> durability levels (
        <code>paranoid</code>, <code>standard</code>, <code>relaxed</code>,{" "}
        <code>ephemeral</code>) are a native <code>Config</code>/
        <code>Durability</code> concept — not a <code>Client</code> keyword
        argument. See the{" "}
        <a href="/docs/python/durability">
          <code>Durability</code>
        </a>{" "}
        reference for details.
      </p>

      {/* ── access_mode() ──────────────────────────────────────── */}
      <h2 id="access-mode">
        <code>{"Config.access_mode(mode_str) -> Config"}</code>
      </h2>
      <CodeBlock lang="python">{`Config.access_mode(mode_str: str) -> Config
# mode_str ∈ {"read_write", "read_only"}
`}</CodeBlock>
      <ul>
        <li>
          <strong>
            <code>"read_write"</code>
          </strong>{" "}
          (default) — Full read/write access. Acquires an exclusive process
          lock on the database directory.
        </li>
        <li>
          <strong>
            <code>"read_only"</code>
          </strong>{" "}
          — Allows concurrent read access without an exclusive lock. Multiple
          processes can open the same database read-only simultaneously.
          Attempting any write operation raises <code>StorageError</code>.
        </li>
      </ul>

      {/* ── Storage Options ────────────────────────────────────── */}
      <h2 id="storage">Storage Options</h2>

      <h2 id="segmented-storage">
        <code>{"Config.segmented_storage(enabled: bool) -> Config"}</code>
      </h2>
      <p>
        When <code>True</code>, ELIPS splits the vector data file across
        multiple fixed-size memory-mapped segments rather than a single large
        mapping. This allows databases larger than the available contiguous
        virtual address space on 32-bit platforms. On 64-bit systems this
        option has no practical benefit and adds a small indirection overhead.
        Defaults to <code>False</code>.
      </p>

      <h2 id="metadata-acceleration">
        <code>{"Config.metadata_acceleration(enabled: bool) -> Config"}</code>
      </h2>
      <p>
        When <code>True</code> (default), ELIPS maintains a secondary
        in-memory hash map of all record metadata fields. This enables O(1)
        predicate evaluation during search filtering rather than decoding
        each record's metadata from the data file on every comparison.
      </p>
      <p>
        Memory cost is approximately <code>64 + (avg_metadata_bytes)</code>{" "}
        per record. Disable only if you are operating under severe memory
        constraints and primarily use full-scan workloads.
      </p>

      <h2 id="quantization">
        <code>{"Config.quantization(spec) -> Config"}</code>
      </h2>
      <p>
        Compresses stored vectors to reduce memory and disk footprint. Accepts
        a <code>QuantParams</code> object or one of the shorthand strings{" "}
        <code>"sq8"</code>, <code>"pq"</code>, <code>"opq"</code>, or{" "}
        <code>"none"</code> (default). The same value may also be passed as the{" "}
        <code>quantization=</code> keyword argument to <code>Client</code> /
        <code>PersistentClient</code>.
      </p>
      <CodeBlock lang="python">{`import elips

# Scalar quantization (8-bit) via the keyword argument
client = elips.PersistentClient(
    "/data/v.elips", dimension=768, quantization="sq8"
)

# Or on a Config builder
cfg = elips.Config().dimension(768).metric("cosine").quantization("sq8")
`}</CodeBlock>

      {/* ── Text Embedder ──────────────────────────────────────── */}
      <h2 id="embedder">Text Embedder</h2>

      <h3>
        <code>
          {"Config.text_embedder(callable, *, provider, model, revision, dimension) -> Config"}
        </code>
      </h3>
      <CodeBlock lang="python">{`Config.text_embedder(
    fn:        callable,
    *,
    provider:  str = "",
    model:     str = "",
    revision:  str = "",
    dimension: int = 0,
) -> Config
`}</CodeBlock>
      <p>
        Attaches a custom Python callable as the text embedder. The callable
        must accept a single <code>str</code> argument and return a{" "}
        <code>list[float]</code> or <code>np.ndarray</code> of the correct
        dimension.
      </p>
      <p>
        The optional keyword arguments populate the{" "}
        <code>EmbeddingLineage</code> stored with each record when using text
        APIs — useful for auditing which model version produced an embedding.
      </p>

      <CodeBlock lang="python">{`import openai, elips

client = openai.OpenAI()

def embed(text: str) -> list[float]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

cfg = (
    Config()
    .dimension(1536)
    .metric("cosine")
    .text_embedder(
        embed,
        provider="openai",
        model="text-embedding-3-small",
        revision="2024-01",
        dimension=1536,
    )
)
`}</CodeBlock>

      <h3>
        <code>{"Config.local_text_embedder(LocalEmbedderConfig) -> Config"}</code>
      </h3>
      <p>
        Attaches a built-in local embedder backed by a bundled ONNX model.
        Requires the optional <code>elips[local]</code> package extra.
      </p>

      <h3>
        <code>{"Config.auto_text_embedder(enabled: bool) -> Config"}</code>
      </h3>
      <p>
        When <code>True</code>, ELIPS automatically selects and attaches a
        bundled lightweight embedder for new databases that do not have one
        configured. On subsequent reopens, the same embedder is reused
        (identified from persisted lineage metadata). Useful for prototyping
        when you don't want to wire up an external embedding service.
      </p>

      {/* ── LocalEmbedderConfig ────────────────────────────────── */}
      <h2 id="local-embedder">
        <code>LocalEmbedderConfig</code>
      </h2>
      <CodeBlock lang="python">{`from elips import LocalEmbedderConfig

lcfg = LocalEmbedderConfig(
    model:        str = "default",
    revision:     str = "v1",
    storage_path: str = "",
    dimension:    int = 0,
)
`}</CodeBlock>
      <p>
        Configuration for the bundled ONNX-backed local embedder:
      </p>
      <ul>
        <li>
          <strong>
            <code>model</code>
          </strong>{" "}
          — Model identifier. <code>"default"</code> selects the recommended
          general-purpose model bundled with the current ELIPS release.
        </li>
        <li>
          <strong>
            <code>revision</code>
          </strong>{" "}
          — Model version tag. Used for lineage tracking.
        </li>
        <li>
          <strong>
            <code>storage_path</code>
          </strong>{" "}
          — Directory where model weights are cached after first download.
          Empty string uses the default platform cache directory (e.g.,{" "}
          <code>~/.cache/elips/models</code> on Linux/macOS).
        </li>
        <li>
          <strong>
            <code>dimension</code>
          </strong>{" "}
          — Override output dimension. <code>0</code> uses the model's native
          dimension.
        </li>
      </ul>

      <CodeBlock lang="python">{`from elips import Config, LocalEmbedderConfig

cfg = (
    Config()
    .dimension(384)
    .metric("cosine")
    .local_text_embedder(LocalEmbedderConfig(model="default", revision="v1"))
)
`}</CodeBlock>

      {/* ── GPU ────────────────────────────────────────────────── */}
      <h2 id="gpu">GPU Selection</h2>
      <p>
        The modern way to run on an accelerator is the <code>device</code>{" "}
        setting on <code>Client</code> / <code>PersistentClient</code>:{" "}
        <code>device="auto"</code>, <code>"cpu"</code>, <code>"metal"</code>,{" "}
        <code>"cuda:0"</code>, or <code>"hip:0"</code>. A named but absent
        backend raises <code>BackendUnavailableError</code> unless you also pass{" "}
        <code>fallback=True</code>.
      </p>
      <CodeBlock lang="python">{`import elips

# Apple GPU (Metal is the default-on backend on Apple hardware)
client = elips.PersistentClient("/data/v.elips", dimension=1536, device="metal")

# Named CUDA device, with CPU fallback if unavailable
client = elips.PersistentClient(
    "/data/v.elips", dimension=1536, device="cuda:0", fallback=True
)
`}</CodeBlock>
      <p>
        For fine-grained control you may instead pass a{" "}
        <code>GpuConfig</code> via the <code>gpu=</code> setting (or{" "}
        <code>Config.gpu(GpuConfig(...))</code>). There is no{" "}
        <code>gpu=True</code> shorthand. See the{" "}
        <a href="/docs/python/gpu">
          <code>GPU API</code>
        </a>{" "}
        reference for backend maturity (Metal production on Apple; CUDA and
        HIP/ROCm experimental; CPU always available) and the accelerator
        discovery helpers.
      </p>

      {/* ── Read Properties ────────────────────────────────────── */}
      <h2 id="read-properties">Read Properties</h2>
      <p>
        After a database is opened, the resolved <code>Config</code> is
        available via <code>client.config</code> and exposes read-only
        properties:
      </p>
      <ul>
        <li>
          <code>dimension_val: int</code> — Resolved dimension.
        </li>
        <li>
          <code>metric_val: str</code> — Resolved metric string.
        </li>
        <li>
          <code>index_val: str</code> — Resolved index type string.
        </li>
        <li>
          <code>graph_params_val: HnswParams</code> — Active HNSW parameters.
        </li>
        <li>
          <code>access_mode_val: str</code> — Active access mode.
        </li>
        <li>
          <code>has_text_embedder: bool</code> — Whether a text embedder is
          configured.
        </li>
        <li>
          <code>text_embedder_info: dict | None</code> — Provider/model/revision
          info, if an embedder is attached.
        </li>
      </ul>

      <CodeBlock lang="python">{`client = elips.PersistentClient(
    "/data/v.elips", dimension=768, metric="cosine"
)
c = client.config
print(c.dimension_val)         # 768
print(c.metric_val)            # 'cosine'
print(c.graph_params_val.ef_search)  # 50
print(c.has_text_embedder)     # True/False
`}</CodeBlock>

      {/* ── ConfigError ────────────────────────────────────────── */}
      <h2 id="config-error">
        <code>ConfigError</code>
      </h2>
      <p>
        <code>elips.ConfigError</code> (subclass of <code>ValueError</code>) is
        raised when:
      </p>
      <ul>
        <li>
          A persisted identity field conflicts with the value stored on disk
          (dimension, metric, index type mismatch).
        </li>
        <li>
          An invalid value is supplied (e.g., <code>dimension=0</code>,
          unknown metric string, <code>ef_search &lt; n_results</code>).
        </li>
        <li>
          <code>text_embedder</code> is configured but the callable returns
          vectors of wrong dimension.
        </li>
      </ul>
      <CodeBlock lang="python">{`try:
    client = elips.PersistentClient("/data/v.elips", dimension=512)
except elips.ConfigError as e:
    # e.field  → 'dimension'
    # e.stored → 768
    # e.given  → 512
    print(f"Config conflict on field '{e.field}': stored={e.stored}, given={e.given}")
`}</CodeBlock>

      {/* ── Two-Phase Ingest/Serve ─────────────────────────────── */}
      <h2 id="ingest-serve">Two-Phase Ingest/Serve Pattern</h2>
      <p>
        A common production pattern is to separate the <em>ingest phase</em>{" "}
        (writing many vectors) from the <em>serve phase</em> (read-only query
        serving) using two different <code>Config</code> objects against the
        same database path:
      </p>

      <CodeBlock lang="python">{`import elips

# --- INGEST PHASE ---
client = elips.PersistentClient(
    "/data/prod.elips",
    dimension=1536,
    metric="cosine",
    index="graph",
    metadata_acceleration=True,
)
embeddings = client.get_or_create_collection("embeddings")

# ... bulk insert via collection.add(...) ...

# Compact and checkpoint after ingest
embeddings.rebuild()
client.compact()
client.checkpoint()
client.close()

# --- SERVE PHASE ---
# Reopen read-only for query serving.
client = elips.PersistentClient("/data/prod.elips", access_mode="read_only")
`}</CodeBlock>

      {/* ── Full Examples ──────────────────────────────────────── */}
      <h2 id="examples">Full Examples</h2>

      <h3>Minimal new database</h3>
      <CodeBlock lang="python">{`import elips

client = elips.PersistentClient(
    "/data/simple.elips", dimension=384, metric="cosine"
)
`}</CodeBlock>

      <h3>Production configuration from environment</h3>
      <CodeBlock lang="python">{`import os, elips
from elips import HnswParams

def open_from_env() -> elips.PersistentClient:
    return elips.PersistentClient(
        os.environ["ELIPS_PATH"],
        dimension=int(os.environ.get("ELIPS_DIM", "768")),
        metric=os.environ.get("ELIPS_METRIC", "cosine"),
        index="graph",
        graph_params=HnswParams(
            max_connections=int(os.environ.get("ELIPS_M", "16")),
            ef_construction=int(os.environ.get("ELIPS_EF_CONSTRUCTION", "200")),
            ef_search=int(os.environ.get("ELIPS_EF_SEARCH", "50")),
        ),
        metadata_acceleration=True,
    )

client = open_from_env()
`}</CodeBlock>
    </DocsShell>
  );
}
