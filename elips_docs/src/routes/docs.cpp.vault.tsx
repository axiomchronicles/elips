import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/cpp/vault")({
  head: () => ({
    meta: [
      { title: "Collection — C++ API Reference" },
      {
        name: "description",
        content:
          "Complete C++ reference for elips::Collection: vector and document add, search, hybrid_search, explain_query, filtering, tombstone vacuum, and index rebuilding.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="C++ API"
      title="Collection"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "placement", label: "Add & Ingestion" },
        { id: "seeking", label: "Searching & Querying" },
        { id: "hybrid-seek", label: "Hybrid Search" },
        { id: "query-planner", label: "explain_query()" },
        { id: "retrieval-scan", label: "Scan & Get" },
        { id: "erasure-vacuum", label: "Removal & Vacuum" },
        { id: "concurrency", label: "Concurrency & Locking" },
      ]}
    >
      <p className="text-[18px] text-ink">
        <code>elips::Collection</code> is the core collection container in ELIPS. Each collection owns its HNSW/flat vector index, metadata inverted index, and payload store.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        Collections are named partitions created via <Link to="/docs/cpp/elips-instance"><code>db-&gt;collection("name")</code></Link>.
        They expose methods for vector ingestion, text embedding, multi-modal search, boolean metadata filtering, and graph compaction.
      </p>

      <h2 id="placement">Add &amp; Ingestion</h2>
      <CodeBlock lang="cpp">{`RecordID add(
    const Vector& vector,
    Payload payload = {},
    std::optional<RecordID> id = std::nullopt,
    std::optional<DocumentAttachment> document = std::nullopt,
    std::optional<ChunkInfo> chunk = std::nullopt,
    std::optional<EmbeddingLineage> lineage = std::nullopt
);

RecordID add_document(
    std::string text,
    Payload payload = {},
    std::optional<RecordID> id = std::nullopt,
    std::optional<ChunkInfo> chunk = std::nullopt,
    std::optional<EmbeddingLineage> lineage = std::nullopt
);

void add_many(const std::vector<Record>& records);`}</CodeBlock>
      <ul>
        <li>
          <code>add()</code> — Inserts a raw floating-point vector into the collection index and returns its <code>RecordID</code>.
        </li>
        <li>
          <code>add_document()</code> — Invokes the configured text embedder engine to convert raw text into a vector before indexing.
        </li>
        <li>
          <code>add_many()</code> — Efficient batch add of pre-constructed records.
        </li>
      </ul>

      <h2 id="seeking">Searching &amp; Querying</h2>
      <CodeBlock lang="cpp">{`std::vector<SearchResult> search(
    const Vector& query,
    std::size_t top,
    const Filter& filter = Filter{},
    std::optional<float> threshold = std::nullopt
) const;

std::vector<SearchResult> search_text(
    std::string_view text,
    std::size_t top,
    const Filter& filter = Filter{},
    std::optional<float> threshold = std::nullopt
) const;`}</CodeBlock>
      <p>
        Executes vector similarity search. Returns an ordered list of <code>SearchResult</code> items sorted by distance.
      </p>

      <h2 id="hybrid-seek">Hybrid Search</h2>
      <CodeBlock lang="cpp">{`std::vector<SearchResult> hybrid_search(
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

      <h2 id="query-planner">explain_query()</h2>
      <CodeBlock lang="cpp">{`QueryPlan explain_query(
    const Vector& query,
    std::size_t top,
    const Filter& filter = Filter{},
    std::optional<float> threshold = std::nullopt,
    bool has_text_component = false
) const;`}</CodeBlock>
      <p>
        Returns an inspection <code>QueryPlan</code> object detailing whether HNSW graph traversal, flat exact scan, or metadata-accelerated filtering will be selected by the optimizer.
      </p>

      <h2 id="retrieval-scan">Scan &amp; Get</h2>
      <CodeBlock lang="cpp">{`std::optional<Record> get(const RecordID& id) const;

std::vector<Record> scan(
    const Filter& filter = Filter{},
    std::size_t offset = 0,
    std::size_t limit = std::numeric_limits<std::size_t>::max()
) const;`}</CodeBlock>

      <h2 id="erasure-vacuum">Removal &amp; Vacuum</h2>
      <CodeBlock lang="cpp">{`bool remove(const RecordID& id);
void vacuum();
std::size_t pending_removals() const noexcept;
void rebuild_index();`}</CodeBlock>
      <ul>
        <li>
          <code>remove()</code> — Tombstones a record ID. (C++ uses <code>remove</code>; the Python binding exposes <code>delete</code>.)
        </li>
        <li>
          <code>vacuum()</code> — Purges tombstoned node slots from HNSW graph indices and reclaims memory.
        </li>
        <li>
          <code>pending_removals()</code> — Returns the count of un-vacuumed tombstones.
        </li>
      </ul>

      <h2 id="concurrency">Concurrency &amp; Locking</h2>
      <p>
        Every public method on <code>Collection</code> is thread-safe. Read operations take a shared reader lock (<code>std::shared_mutex::lock_shared</code>), enabling parallel search operations across CPU threads.
      </p>
    </DocsShell>
  );
}
