import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/cpp/config")({
  head: () => ({
    meta: [
      { title: "Config & GraphParams — C++ Engine Tuning" },
      {
        name: "description",
        content:
          "Complete reference for C++ elips::Config and elips::GraphParams: metric selection, index algorithms, HNSW graph parameters, durability policies, and segment sizing.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="C++ API"
      title="Config & GraphParams"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "config-struct", label: "Config Struct" },
        { id: "graph-params", label: "GraphParams Struct" },
        { id: "enums", label: "Metrics & Policies Enums" },
        { id: "tuning-recipes", label: "Tuning Recipes" },
      ]}
    >
      <p className="text-[18px] text-ink">
        The <code>elips::Config</code> struct specifies all indexing algorithms, distance metrics, HNSW graph connectivity, durability policies, and segment sizes.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        Configuration objects are passed when opening an <Link to="/docs/cpp/elips-instance"><code>ElipsInstance</code></Link> or initializing individual <Link to="/docs/cpp/vault"><code>Vault</code></Link> instances.
      </p>

      <h2 id="config-struct">Config Struct</h2>
      <CodeBlock lang="cpp">{`struct Config {
    std::uint16_t dimension{128};
    Metric metric{Metric::cosine};
    IndexType index_type{IndexType::hnsw};
    GraphParams graph_params{};
    Durability durability{Durability::sync_on_commit};
    AccessMode access_mode{AccessMode::read_write};
    std::size_t segment_size_bytes{64 * 1024 * 1024}; // 64MB default
};`}</CodeBlock>

      <h2 id="graph-params">GraphParams Struct</h2>
      <CodeBlock lang="cpp">{`struct GraphParams {
    std::size_t m{16};                   // Max bi-directional links per node
    std::size_t ef_construction{200};    // Search queue capacity during construction
    std::size_t ef_search{50};           // Search queue capacity during query time
    float compaction_ratio{0.2f};       // Tombstone threshold to trigger auto-vacuum
};`}</CodeBlock>

      <h2 id="enums">Metrics & Policies Enums</h2>
      <CodeBlock lang="cpp">{`enum class Metric {
    cosine,
    euclidean,
    dot_product
};

enum class IndexType {
    hnsw,
    exact,
    ivf_flat,
    ivf_pq
};

enum class Durability {
    sync_on_commit,  // fsync WAL on every commit
    wal_only,        // write WAL without forcing fsync immediately
    in_memory        // skip disk writing completely
};

enum class AccessMode {
    read_write,
    read_only
};`}</CodeBlock>

      <h2 id="tuning-recipes">Tuning Recipes</h2>
      <h3 id="high-recall">1. High Recall / Accuracy Setup</h3>
      <CodeBlock lang="cpp">{`elips::Config config;
config.dimension = 1536;
config.metric = elips::Metric::cosine;
config.index_type = elips::IndexType::hnsw;
config.graph_params.m = 32;
config.graph_params.ef_construction = 400;
config.graph_params.ef_search = 128;`}</CodeBlock>

      <h3 id="low-latency">2. Ultra Low Latency Setup</h3>
      <CodeBlock lang="cpp">{`elips::Config config;
config.dimension = 384;
config.metric = elips::Metric::dot_product;
config.index_type = elips::IndexType::hnsw;
config.graph_params.m = 12;
config.graph_params.ef_construction = 100;
config.graph_params.ef_search = 24;`}</CodeBlock>
    </DocsShell>
  );
}
