import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";
import { SketchCard, ErrorHierarchyDiagram } from "../components/SketchDiagram";

export const Route = createFileRoute("/docs/cpp-sdk")({
  head: () => ({
    meta: [
      { title: "C++ SDK — ELIPS Docs" },
      {
        name: "description",
        content:
          "The complete ELIPS C++ surface — Config, ElipsInstance, Vault, transactions, embedders, query plans, GPU configuration, and error handling.",
      },
      { property: "og:title", content: "C++ SDK — ELIPS" },
      {
        property: "og:description",
        content:
          "C++23 surface: typed config, document-aware records, planner introspection, transactions, and locking.",
      },
      { property: "og:url", content: "/docs/cpp-sdk" },
    ],
    links: [{ rel: "canonical", href: "/docs/cpp-sdk" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · C++"
      title="C++ SDK"
      toc={[
        { id: "build", label: "Build & install" },
        { id: "hello", label: "Minimal example" },
        { id: "types", label: "Primary types" },
        { id: "config", label: "Config" },
        { id: "instance", label: "ElipsInstance" },
        { id: "vault", label: "Vault" },
        { id: "query", label: "Query & planner" },
        { id: "txn", label: "Transactions" },
        { id: "embedders", label: "Embedders" },
        { id: "persistence", label: "Persistence" },
        { id: "locking", label: "Locking & threading" },
        { id: "errors", label: "Errors" },
        { id: "gpu", label: "GPU configuration" },
        { id: "principles", label: "Design principles" },
        { id: "pitfalls", label: "Pitfalls" },
      ]}
    >
      <p className="text-[18px] text-ink">
        The C++ surface is the runtime's source of truth. It exposes typed configuration,
        document-aware records, planner introspection, persistence control, and optional GPU-backed
        indexes. Everything else — the Python bindings, the CLI, EQL — runs through the same
        headers.
      </p>

      <h2 id="build">Build &amp; install</h2>
      <CodeBlock lang="bash">
        {`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure`}
      </CodeBlock>
      <p>
        Link against the produced static/shared library and include <code>elips/elips.hpp</code>.
        C++23 is required; the project targets the C++ Core Guidelines (see{" "}
        <Link to="/docs/design-decisions">ADR-0001</Link>).
      </p>

      <h2 id="hello">Minimal example</h2>
      <CodeBlock lang="cpp">
        {`#include "elips/elips.hpp"

auto db = elips::open(
    ":memory:",
    elips::Config{}
        .dimension(2)
        .metric(elips::Metric::cosine));

auto& docs = db->vault("documents");
docs.place_document("alpha design note", {{"kind", std::string{"design"}}});
docs.place_document("beta incident runbook", {{"kind", std::string{"ops"}}});

const auto hits = docs.seek_text("alpha", 2);`}
      </CodeBlock>
      <p>
        New databases attach ELIPS' built-in local embedder automatically unless you disable it with{" "}
        <code>Config::auto_text_embedder(false)</code>.
      </p>

      <h2 id="types">Primary types</h2>
      <ul>
        <li>
          <code>elips::Config</code> — fluent configuration builder.
        </li>
        <li>
          <code>elips::ElipsInstance</code> — top-level database handle returned by{" "}
          <code>elips::open()</code>. Move-only, non-copyable.
        </li>
        <li>
          <code>elips::Vault</code> — per-collection record store and query surface.
        </li>
        <li>
          <code>elips::Record</code>, <code>elips::DocumentAttachment</code>,{" "}
          <code>elips::ChunkInfo</code>, <code>elips::EmbeddingLineage</code> — domain types.
        </li>
        <li>
          <code>elips::Filter</code> — predicate tree used by both the fluent builder and EQL.
        </li>
        <li>
          <code>elips::QueryPlan</code> — planner output for vector and hybrid queries.
        </li>
        <li>
          <code>elips::Transaction</code> / <code>elips::TransactionVault</code> — atomic batched
          writes.
        </li>
        <li>
          <code>elips::TextEmbedderPort</code> — pluggable text embedder interface.
        </li>
      </ul>

      <h2 id="config">Config</h2>
      <CodeBlock lang="cpp">
        {`enum class Metric      { cosine, euclidean, dot_product };
enum class IndexType   { graph, exact };
enum class Durability  { paranoid, standard, relaxed, ephemeral };
enum class AccessMode  { read_write, read_only };

struct GraphParams {
    std::size_t max_connections{16};
    std::size_t ef_construction{200};
    std::size_t ef_search{50};
};

elips::Config{}
    .dimension(768)
    .metric(elips::Metric::cosine)
    .index(elips::IndexType::graph)
    .graph_params({.max_connections = 32, .ef_construction = 400, .ef_search = 100})
    .durability(elips::Durability::standard)
    .access_mode(elips::AccessMode::read_write)
    .segmented_storage(true)
    .metadata_acceleration(true)
    .auto_text_embedder(true);`}
      </CodeBlock>
      <p>
        Setters mirror getters one-for-one. Notable behaviour:
        <code>dimension()</code> must be non-zero for new persistent databases and every{" "}
        <code>":memory:"</code> open; existing databases reopen with the persisted identity;{" "}
        <code>access_mode(read_only)</code> requires an existing database;{" "}
        <code>metadata_acceleration(true)</code> enables exact candidate narrowing through{" "}
        <code>MetadataIndex</code>.
      </p>

      <h2 id="instance">ElipsInstance</h2>
      <CodeBlock lang="cpp">
        {`std::unique_ptr<ElipsInstance> open(const std::string& path,
                                    const Config& config = {});`}
      </CodeBlock>
      <ul>
        <li>
          <code>vault(name)</code> — returns a reference, creating lazily.
        </li>
        <li>
          <code>list_vaults()</code> — current vault names.
        </li>
        <li>
          <code>begin_transaction()</code> — atomic write transaction.
        </li>
        <li>
          <code>query(eql, bindings={"{}"})</code> — single EQL statement, returns{" "}
          <code>std::vector&lt;SearchResult&gt;</code>.
        </li>
        <li>
          <code>checkpoint()</code> — flush manifest+segments (or snapshot) and truncate the WAL.
        </li>
        <li>
          <code>compact()</code> — rebuild every vault index from stored records and checkpoint.
        </li>
        <li>
          <code>close()</code> — graceful shutdown: checkpoint, detach WAL, release lock.
        </li>
        <li>
          <code>abandon()</code> — testing hook that suppresses destructor checkpointing.
        </li>
        <li>
          <code>config()</code> — effective <code>Config</code>.
        </li>
        <li>
          <code>gpu_info()</code> / <code>gpu_stats()</code> — only in GPU builds.
        </li>
      </ul>
      <p>
        Persistent instances checkpoint on destruction unless already closed or opened read-only.
        Read-only instances never attach a WAL writer; vaults under a read-only instance are
        immediately marked read-only.
      </p>

      <h2 id="vault">Vault</h2>
      <CodeBlock lang="cpp">
        {`RecordID place(const Vector& vector,
               Payload payload = {},
               std::optional<RecordID> id = std::nullopt,
               std::optional<DocumentAttachment> document = std::nullopt,
               std::optional<ChunkInfo> chunk = std::nullopt,
               std::optional<EmbeddingLineage> lineage = std::nullopt);

RecordID place_document(std::string text,
                        Payload payload = {},
                        std::optional<RecordID> id = std::nullopt,
                        std::optional<ChunkInfo> chunk = std::nullopt,
                        std::optional<EmbeddingLineage> lineage = std::nullopt);

void place_many(const std::vector<Record>& records);

bool erase(const RecordID& id);
std::optional<Record> fetch(const RecordID& id) const;
std::vector<Record> scan(const Filter& filter = {},
                         std::size_t offset = 0,
                         std::size_t limit  = std::numeric_limits<std::size_t>::max()) const;

VaultInfo info() const;
void      rebuild_index();`}
      </CodeBlock>

      <h2 id="query">Query &amp; planner</h2>
      <CodeBlock lang="cpp">
        {`std::vector<SearchResult> seek       (const Vector& q, std::size_t top,
                                      const Filter& = {},
                                      std::optional<float> threshold = std::nullopt) const;

std::vector<SearchResult> seek_text  (std::string_view text, std::size_t top,
                                      const Filter& = {},
                                      std::optional<float> threshold = std::nullopt) const;

std::vector<SearchResult> seek_hybrid(const Vector& q, std::string_view text,
                                      std::size_t top, const Filter& = {},
                                      std::optional<float> threshold = std::nullopt,
                                      float lexical_weight = 0.25F) const;

QueryPlan explain_seek(const Vector& q, std::size_t top,
                       const Filter& = {},
                       std::optional<float> threshold = std::nullopt,
                       bool has_text_component = false) const;`}
      </CodeBlock>
      <p>
        Every vector or hybrid query passes through <code>Vault::plan_seek()</code> first.{" "}
        <code>QueryPlan</code> exposes the chosen strategy (<code>ann_index</code>,{" "}
        <code>exact_candidates</code>, <code>full_scan</code>, <code>text_probe</code>,{" "}
        <code>hybrid_fusion</code>), candidate count, the metadata acceleration flag, the GPU flag,
        and the index type name. <code>SearchResult</code> carries <code>id</code>,{" "}
        <code>distance</code>, <code>data</code>, <code>document</code>, <code>chunk</code>, and{" "}
        <code>lineage</code> hydrated from the authoritative record store.
      </p>

      <h2 id="txn">Transactions</h2>
      <CodeBlock lang="cpp">
        {`auto txn = db->begin_transaction();
txn.vault("documents").place(elips::Vector{{1.0F, 0.0F}});
txn.commit();   // or txn.rollback();`}
      </CodeBlock>
      <p>
        <code>Transaction</code> is RAII: if the destructor runs without an explicit{" "}
        <code>commit()</code> or <code>rollback()</code>, it calls <code>rollback()</code>{" "}
        automatically — buffered operations are discarded. <code>enqueue_place</code> validates
        dimension and finiteness eagerly, so <code>commit()</code> never fails mid-batch on
        validation grounds. See{" "}
        <Link to="/docs/internals/transaction-engine">Transaction engine</Link>.
      </p>

      <h2 id="embedders">Embedders</h2>
      <CodeBlock lang="cpp">
        {`class TextEmbedderPort {
public:
    virtual ~TextEmbedderPort() = default;
    virtual Vector embed(std::string_view text) const = 0;
    virtual std::vector<Vector> embed_batch(
        const std::vector<std::string>& texts) const;
    virtual std::string_view provider_name() const noexcept = 0;
    virtual std::string_view model_name()    const noexcept = 0;
    virtual std::string_view revision_name() const noexcept;
    virtual std::string_view backend_name()  const noexcept;
    virtual std::uint16_t    output_dimension() const noexcept;
};`}
      </CodeBlock>
      <p>
        The built-in local embedder is rehydratable — its identity lives in{" "}
        <code>TEXT_EMBEDDER.manifest</code> plus an artifact under <code>text_embedder/</code>.
        Custom <code>TextEmbedderPort</code> implementations work through{" "}
        <code>Config::text_embedder(...)</code>, but ELIPS can only persist their metadata; a later
        reopen must provide the same embedder before text-first APIs can be used again.
      </p>

      <h2 id="persistence">Persistence</h2>
      <p>
        Two on-disk layouts. Segmented (default) writes a root <code>elips.manifest</code> plus one
        segment file per vault under <code>segments/</code>. Snapshot mode writes a single{" "}
        <code>elips.snapshot</code> for compatibility. Every mutation is WAL-appended before the
        in-memory store changes; WAL replay rebuilds documents, chunks, and lineage on open. See{" "}
        <Link to="/docs/storage">Storage &amp; recovery</Link>.
      </p>

      <h2 id="locking">Locking, threading, ownership</h2>
      <ul>
        <li>
          <strong>Single writer, many readers.</strong> The writer holds an exclusive{" "}
          <code>flock</code> on <code>LOCK</code>; read-only opens take a shared lock. See{" "}
          <Link to="/docs/internals/lock-manager">Lock manager</Link>.
        </li>
        <li>
          <strong>RAII everywhere.</strong> <code>LockManager</code> releases on destruction,{" "}
          <code>Transaction</code> auto-rolls back if not committed, <code>ElipsInstance</code>'s
          destructor checkpoints and swallows exceptions (Core Guideline E.16).
        </li>
        <li>
          <strong>Not thread-safe.</strong> A single <code>ElipsInstance</code> assumes a single
          thread of mutation. The locking model is process-level, not intra-process.
        </li>
        <li>
          <strong>Move-only.</strong> <code>ElipsInstance</code> is non-copyable; pass{" "}
          <code>std::unique_ptr&lt;ElipsInstance&gt;</code> by move.
        </li>
      </ul>

      <h2 id="errors">Errors</h2>
      <SketchCard caption="Six concrete throws, one base. EQL parsing and GPU calls deliberately sit on their own surfaces.">
        <ErrorHierarchyDiagram
          leaves={[
            "DimensionMismatch",
            "InvalidVector",
            "ConfigError",
            "NotFound",
            "StorageError",
            "LockConflict",
          ]}
          asides={[
            "elips::eql::ParseError — EQL surface",
            "std::expected<T, GpuError> — GPU surface",
          ]}
        />
      </SketchCard>

      <h2 id="gpu">GPU configuration</h2>
      <p>
        Available only in GPU builds. Configure through{" "}
        <code>Config::gpu(gpu::GpuConfig{"{}"})</code>. Supported algorithms include{" "}
        <code>brute_force</code>, <code>ivf_flat</code>, <code>ivf_pq</code>, and <code>cagra</code>
        . See <Link to="/docs/algorithms">Algorithms</Link>.
      </p>

      <h2 id="principles">Design principles you can rely on</h2>
      <ul>
        <li>
          <strong>Dependency Inversion.</strong> <code>Vault</code> depends on{" "}
          <code>IndexPort</code>, never on a concrete index; the composition root is{" "}
          <code>make_index()</code>.
        </li>
        <li>
          <strong>RAII.</strong> Locks, transactions, and the instance are bound to scope; nothing
          leaks on exception unwinding.
        </li>
        <li>
          <strong>Interface Segregation.</strong> The GPU engine splits into <code>GpuPort</code>,{" "}
          <code>GpuMemoryPort</code>, <code>GpuKernelPort</code>, <code>GpuStreamPort</code>, and{" "}
          <code>GpuIndexPort</code> so each consumer depends only on the slice it needs.
        </li>
        <li>
          <strong>Purpose-built errors.</strong> One root, narrow subclasses, and{" "}
          <code>std::expected</code> for GPU paths where failure is expected rather than
          exceptional.
        </li>
      </ul>

      <h2 id="pitfalls">Pitfalls</h2>
      <ul>
        <li>
          Forgetting to <code>commit()</code> a <code>Transaction</code> — the destructor will roll
          it back. This is intentional, not a bug.
        </li>
        <li>
          Calling <code>seek_text</code> / <code>place_document</code> without a configured text
          embedder — raises <code>ConfigError</code> with an actionable message; ELIPS never
          silently switches to lexical-only behaviour.
        </li>
        <li>
          Opening the same database twice for writing — the second open raises{" "}
          <code>LockConflict</code>.
        </li>
        <li>
          Reusing a <code>Vault&amp;</code> after the owning <code>ElipsInstance</code> has been
          moved or destroyed — references are non-owning and the instance is the lifetime root.
        </li>
      </ul>
    </DocsShell>
  );
}
