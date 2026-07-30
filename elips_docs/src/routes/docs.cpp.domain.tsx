import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/cpp/domain")({
  head: () => ({
    meta: [
      { title: "C++ Domain Types & Record Model — ELIPS Documentation" },
      {
        name: "description",
        content:
          "Complete C++ reference for ELIPS domain types: RecordID, Vector, Record, SearchResult, Payload, attachments, and C++ error exception hierarchies.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="C++ API"
      title="Domain Types & Record Model"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "record-id", label: "RecordID" },
        { id: "vector-type", label: "Vector" },
        { id: "payload-type", label: "Payload" },
        { id: "record-struct", label: "Record" },
        { id: "search-result", label: "SearchResult" },
        { id: "attachments", label: "Domain Attachments" },
        { id: "exceptions", label: "C++ Exception Hierarchy" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS domain types define the data structures and exception types used throughout the native C++ engine.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        Domain types live in the <code>elips::</code> namespace and are declared under headers in <code>elips/domain/</code>.
      </p>

      <h2 id="record-id">RecordID</h2>
      <CodeBlock lang="cpp">{`using RecordID = std::uint64_t;`}</CodeBlock>
      <p>
        Records in ELIPS are uniquely identified by a 64-bit unsigned integer <code>RecordID</code>. If an explicit ID is omitted during insertion, the engine assigns an auto-incrementing 64-bit key.
      </p>

      <h2 id="vector-type">Vector</h2>
      <CodeBlock lang="cpp">{`using Vector = std::vector<float>;`}</CodeBlock>
      <p>
        Dense vector embeddings are represented as 32-bit floating point vectors (<code>std::vector&lt;float&gt;</code>).
      </p>

      <h2 id="payload-type">Payload</h2>
      <CodeBlock lang="cpp">{`using PayloadValue = std::variant<
    std::int64_t,
    double,
    std::string,
    bool,
    std::vector<std::string>
>;

using Payload = std::map<std::string, PayloadValue>;`}</CodeBlock>
      <p>
        Metadata attributes associated with a vector record are stored in a key-value map supporting primitive scalar types and string arrays.
      </p>

      <h2 id="record-struct">Record</h2>
      <CodeBlock lang="cpp">{`struct Record {
    RecordID id{0};
    Vector vector;
    Payload payload;
    std::optional<DocumentAttachment> document;
    std::optional<ChunkInfo> chunk;
    std::optional<EmbeddingLineage> lineage;
};`}</CodeBlock>

      <h2 id="search-result">SearchResult</h2>
      <CodeBlock lang="cpp">{`struct SearchResult {
    RecordID id{0};
    float distance{0.0f};
    float score{0.0f};
    Record record;
};`}</CodeBlock>
      <p>
        Vector similarity search returns <code>SearchResult</code> instances containing distance metrics, similarity scores, and hydrated <code>Record</code> payloads.
      </p>

      <h2 id="attachments">Domain Attachments</h2>
      <CodeBlock lang="cpp">{`struct DocumentAttachment {
    std::string text;
    std::string mime_type{"text/plain"};
};

struct ChunkInfo {
    std::size_t index{0};
    std::size_t start_char{0};
    std::size_t end_char{0};
};

struct EmbeddingLineage {
    std::string model_name;
    std::uint32_t dimension{0};
};`}</CodeBlock>

      <h2 id="exceptions">C++ Exception Hierarchy</h2>
      <CodeBlock lang="cpp">{`namespace elips {
    class ElipsException : public std::runtime_error;
    class ValidationException : public ElipsException;
    class LockConflictException : public ElipsException;
    class TransactionException : public ElipsException;
    class DiskException : public ElipsException;
    class GpuException : public ElipsException;
}`}</CodeBlock>
      <p>
        All C++ exceptions derive from <code>elips::ElipsException</code>. Catching <code>ElipsException</code> guarantees catching all engine-thrown runtime errors.
      </p>
    </DocsShell>
  );
}
