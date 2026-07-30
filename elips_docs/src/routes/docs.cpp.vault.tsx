import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/cpp/vault")({
  head: () => ({
    meta: [
      { title: "Vault — C++ Collection API Reference" },
      {
        name: "description",
        content:
          "Complete C++ reference for elips::Vault: vector and document placement, seek, seek_hybrid, explain_seek, filtering, tombstone vacuum, and index rebuilding.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="C++ API"
      title="Vault"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "placement", label: "Placement & Ingestion" },
        { id: "seeking", label: "Seeking & Querying" },
        { id: "hybrid-seek", label: "Hybrid Search" },
        { id: "query-planner", label: "explain_seek()" },
        { id: "retrieval-scan", label: "Scan & Fetch" },
        { id: "erasure-vacuum", label: "Erasure & Vacuum" },
        { id: "concurrency", label: "Concurrency & Locking" },
      ]}
    >
      <p className="text-[18px] text-ink">
        <code>elips::Vault</code> is the core collection container in ELIPS. Each vault owns its HNSW/exact vector index, metadata inverted index, and payload store.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        Vaults are named collections created via <Link to="/docs/cpp/elips-instance"><code>db-&gt;vault("name")</code></Link>.
        They expose methods for vector ingestion, text embedding placement, multi-modal seeking, boolean metadata filtering, and graph compaction.
      </p>

      <h2 id="placement">Placement & Ingestion</h2>
      <CodeBlock lang="cpp">{`RecordID place(
    const Vector& vector,
    Payload payload = {},
    std::optional<RecordID> id = std::nullopt,
    std::optional<DocumentAttachment> document = std::nullopt,
    std::optional<ChunkInfo> chunk = std::nullopt,
    std::optional<EmbeddingLineage> lineage = std::nullopt
);

RecordID place_document(
    std::string text,
    Payload payload = {},
    std::optional<RecordID> id = std::nullopt,
    std::optional<ChunkInfo> chunk = std::nullopt,
    std::optional<EmbeddingLineage> lineage = std::nullopt
);

void place_many(const std::vector<Record>& records);`}</CodeBlock>
      <ul>
        <li>
          <code>place()</code> — Inserts a raw floating-point vector into the vault index and returns its <code>RecordID</code>.
        </li>
        <li>
          <code>place_document()</code> — Invokes the configured text embedder engine to convert raw text into a vector before indexing.
        </li>
        <li>
          <code>place_many()</code> — Efficient batch placement of pre-constructed records.
        </li>
      </ul>

      <h2 id="seeking">Seeking & Querying</h2>
      <CodeBlock lang="cpp">{`std::vector<SearchResult> seek(
    const Vector& query,
    std::size_t top,
    const Filter& filter = Filter{},
    std::optional<float> threshold = std::nullopt
) const;

std::vector<SearchResult> seek_text(
    std::string_view text,
    std::size_t top,
    const Filter& filter = Filter{},
    std::optional<float> threshold = std::nullopt
) const;`}</CodeBlock>
      <p>
        Executes vector similarity search. Returns an ordered list of <code>SearchResult</code> items sorted by score.
      </p>

      <h2 id="hybrid-seek">Hybrid Search</h2>
      <CodeBlock lang="cpp">{`std::vector<SearchResult> seek_hybrid(
    const Vector& query,
    std::string_view text,
    std::size_t top,
    const Filter& filter = Filter{},
    std::optional<float> threshold = std::nullopt,
    float lexical_weight = 0.25F
) const;`}</CodeBlock>
      <p>
        Fuses dense vector similarity with sparse BM25 text relevance scores using <code>lexical_weight</code>.
      </p>

      <h2 id="query-planner">explain_seek()</h2>
      <CodeBlock lang="cpp">{`QueryPlan explain_seek(
    const Vector& query,
    std::size_t top,
    const Filter& filter = Filter{},
    std::optional<float> threshold = std::nullopt,
    bool has_text_component = false
) const;`}</CodeBlock>
      <p>
        Returns an inspection <code>QueryPlan</code> object detailing whether HNSW graph traversal, exact scan, or metadata-accelerated filtering will be selected by the optimizer.
      </p>

      <h2 id="retrieval-scan">Scan & Fetch</h2>
      <CodeBlock lang="cpp">{`std::optional<Record> fetch(const RecordID& id) const;

std::vector<Record> scan(
    const Filter& filter = Filter{},
    std::size_t offset = 0,
    std::size_t limit = std::numeric_limits<std::size_t>::max()
) const;`}</CodeBlock>

      <h2 id="erasure-vacuum">Erasure & Vacuum</h2>
      <CodeBlock lang="cpp">{`bool erase(const RecordID& id);
void vacuum();
std::size_t pending_removals() const noexcept;
void rebuild_index();`}</CodeBlock>
      <ul>
        <li>
          <code>erase()</code> — Tombstones a record ID.
        </li>
        <li>
          <code>vacuum()</code> — Purges tombstoned node slots from HNSW graph indices and reclaims memory.
        </li>
        <li>
          <code>pending_removals()</code> — Returns the count of un-vacuumed tombstones.
        </li>
      </ul>

      <h2 id="concurrency">Concurrency & Locking</h2>
      <p>
        Every public method on <code>Vault</code> is thread-safe. Read operations take a shared reader lock (<code>std::shared_mutex::lock_shared</code>), enabling parallel seek operations across CPU threads.
      </p>
    </DocsShell>
  );
}
