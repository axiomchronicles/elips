import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/models")({
  head: () => ({
    meta: [
      { title: "Data Models — ELIPS Python Reference" },
      {
        name: "description",
        content:
          "Complete reference for all modern ELIPS Python data models: RecordInput, Row, Hit, WalRecord, ArenaHealth, DocumentAttachment, ChunkInfo, EmbeddingLineage, and type aliases.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="Data Models"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "record-input", label: "RecordInput" },
        { id: "row", label: "Row" },
        { id: "hit", label: "Hit" },
        { id: "wal-record", label: "WalRecord" },
        { id: "arena-health", label: "ArenaHealth" },
        { id: "domain-types", label: "Domain types" },
        { id: "document-attachment", label: "DocumentAttachment" },
        { id: "chunk-info", label: "ChunkInfo" },
        { id: "embedding-lineage", label: "EmbeddingLineage" },
        { id: "type-aliases", label: "Type aliases" },
      ]}
    >
      <p className="text-[18px] text-ink">
        This page documents every data class and type alias in the modern ELIPS
        Python API. All classes live in{" "}
        <code>elips._modern.models</code> and are re-exported from the top-level{" "}
        <code>elips</code> namespace. Domain types (<code>DocumentAttachment</code>
        , <code>ChunkInfo</code>, <code>EmbeddingLineage</code>) are C++ binding
        types re-exported from <code>elips._core</code>.
      </p>

      <h2 id="overview">Overview</h2>
      <p>The modern API uses four frozen dataclasses for its public surface:</p>
      <ul>
        <li>
          <code>RecordInput</code> — structured input to{" "}
          <Link to="/docs/python/arena">
            <code>arena.write()</code> / <code>arena.write_many()</code>
          </Link>
          .
        </li>
        <li>
          <code>Row</code> — a materialized record from{" "}
          <code>arena.pull()</code> or <code>arena.sweep()</code>.
        </li>
        <li>
          <code>Hit</code> — a search result from{" "}
          <code>arena.probe()</code>, <code>arena.probe_text()</code>, or{" "}
          <code>arena.probe_hybrid()</code>.
        </li>
        <li>
          <code>WalRecord</code> — an acknowledged WAL entry from{" "}
          <code>engine.pending_writes()</code>.
        </li>
      </ul>
      <p>
        All four are <code>frozen=True, slots=True</code> dataclasses —
        immutable and memory-efficient. The domain attachment types (
        <code>DocumentAttachment</code>, <code>ChunkInfo</code>,{" "}
        <code>EmbeddingLineage</code>) are C++ extension types and are mutable.
      </p>

      <h2 id="record-input">
        <code>RecordInput</code>
      </h2>
      <CodeBlock lang="python">{`@dataclass(frozen=True, slots=True)
class RecordInput:
    vector:   Sequence[float] | None = None
    text:     str | None = None
    meta:     Mapping[str, MetaValue] | None = None
    key:      str | None = None
    document: DocumentAttachment | None = None
    chunk:    ChunkInfo | None = None
    lineage:  EmbeddingLineage | None = None`}</CodeBlock>
      <p>
        Structured input record for a single arena write. At construction time,
        ELIPS validates that at least one of <code>vector</code>,{" "}
        <code>text</code>, or <code>document.text</code> is present; the record
        must carry enough information to produce an embedding.
      </p>
      <p>
        When both <code>text</code> and <code>document</code> are supplied,
        their text values must match (or <code>document.text</code> must be
        empty). This prevents silent divergence between the embedded text and
        the stored document.
      </p>

      <h3>Validation rules</h3>
      <ul>
        <li>
          <code>vector is None and text is None and document.text is None</code>{" "}
          → <code>ValueError</code>.
        </li>
        <li>
          <code>text != document.text</code> (when both non-empty) →{" "}
          <code>ValueError</code>.
        </li>
        <li>
          Custom document metadata (<code>uri</code> or non-plain MIME type)
          with no explicit vector → <code>ValueError</code> at write time
          (because the text-only ingest path cannot attach custom document
          fields).
        </li>
      </ul>
      <CodeBlock lang="python">{`import elips

# Minimal text record
r = elips.RecordInput(text="Alpha design note", meta={"kind": "design"})

# Explicit vector with attached text document
r = elips.RecordInput(
    vector=[1.0, 0.0],
    text="Alpha design note",
    meta={"kind": "design"},
)

# With a full DocumentAttachment
doc = elips.DocumentAttachment(
    text="Alpha design note",
    uri="notes/alpha.md",
    mime_type="text/markdown",
)
# MUST supply explicit vector because document has a custom URI
r = elips.RecordInput(vector=[1.0, 0.0], document=doc, meta={"kind": "design"})

# Caller-assigned key
r = elips.RecordInput(text="Beta", key=elips.generate_id(), meta={"rev": 1})

# Validation error: no content
try:
    elips.RecordInput(meta={"kind": "empty"})
except ValueError as exc:
    print(exc)   # record input requires a vector, text, or document with text`}</CodeBlock>

      <h3>
        <code>record.document_text</code> → <code>str | None</code>
      </h3>
      <p>
        Returns the text used for embedding resolution, regardless of whether it
        came from <code>text</code> or <code>document.text</code>.{" "}
        <code>text</code> takes precedence over <code>document.text</code>.
      </p>
      <CodeBlock lang="python">{`elips.RecordInput(text="hello").document_text               # "hello"
elips.RecordInput(
    vector=[1.0, 0.0],
    document=elips.DocumentAttachment(text="world"),
).document_text                                             # "world"
elips.RecordInput(vector=[1.0, 0.0]).document_text          # None`}</CodeBlock>

      <h3>
        <code>record.materialize_meta()</code> → <code>dict</code>
      </h3>
      <p>
        Returns a mutable copy of <code>meta</code> as a plain{" "}
        <code>dict</code>. Returns an empty dict when <code>meta</code> is{" "}
        <code>None</code>.
      </p>
      <CodeBlock lang="python">{`r = elips.RecordInput(text="Alpha", meta={"kind": "design"})
payload = r.materialize_meta()
payload["extra"] = "injected"   # does not mutate the frozen record
print(r.meta)                   # {"kind": "design"} — unchanged`}</CodeBlock>

      <h3>
        <code>record.materialize_document()</code> →{" "}
        <code>DocumentAttachment | None</code>
      </h3>
      <p>
        Build a concrete <code>DocumentAttachment</code> ready for storage. When
        only <code>text</code> is supplied (no explicit <code>document</code>),
        constructs <code>DocumentAttachment(text=self.text)</code>. When both
        are supplied, clones the existing attachment and overrides its text.
        Returns <code>None</code> for vector-only records.
      </p>
      <CodeBlock lang="python">{`r = elips.RecordInput(text="hello")
r.materialize_document().text   # "hello"

r = elips.RecordInput(vector=[1.0, 0.0])
r.materialize_document()        # None`}</CodeBlock>

      <h3>
        <code>record.has_custom_document_metadata()</code> → <code>bool</code>
      </h3>
      <p>
        Returns <code>True</code> when the record's document has a non-empty{" "}
        <code>uri</code> or a MIME type other than <code>text/plain</code>.
        Used internally to decide whether the native{" "}
        <code>place_document()</code> path is sufficient or a full{" "}
        <code>place()</code> call (with explicit vector) is required.
      </p>
      <CodeBlock lang="python">{`plain = elips.RecordInput(text="hello")
plain.has_custom_document_metadata()   # False

doc = elips.DocumentAttachment(text="readme", uri="README.md")
rich = elips.RecordInput(vector=[1.0, 0.0], document=doc)
rich.has_custom_document_metadata()    # True`}</CodeBlock>

      <h3>
        <code>RecordInput.from_mapping(record)</code> → <code>RecordInput</code>
      </h3>
      <p>
        Classmethod that converts a plain dict (or any mapping) into a{" "}
        <code>RecordInput</code>. Accepts both modern field names (
        <code>meta</code> / <code>key</code>) and the low-level batch field
        names (<code>data</code> / <code>id</code>) for backward compatibility.
        Both cannot be present with conflicting values.
      </p>
      <CodeBlock lang="python">{`# Modern names
r = elips.RecordInput.from_mapping({"text": "alpha", "meta": {"k": 1}})

# Legacy names (from Vault.place_many dict format)
r = elips.RecordInput.from_mapping({"vector": [1.0, 0.0], "data": {"k": 1}})

# Mixed (key + id with matching value is OK)
r = elips.RecordInput.from_mapping({
    "text": "beta", "key": "abc", "id": "abc"
})

# Conflict → ValueError
try:
    elips.RecordInput.from_mapping({
        "text": "beta", "key": "abc", "id": "xyz"
    })
except ValueError: ...`}</CodeBlock>

      <h2 id="row">
        <code>Row</code>
      </h2>
      <CodeBlock lang="python">{`@dataclass(frozen=True, slots=True)
class Row:
    key:      str
    meta:     dict[str, MetaValue]
    document: DocumentAttachment | None = None
    vector:   tuple[float, ...] | None = None
    chunk:    ChunkInfo | None = None
    lineage:  EmbeddingLineage | None = None`}</CodeBlock>
      <p>
        A materialized record, returned by{" "}
        <Link to="/docs/python/arena">
          <code>arena.pull()</code>
        </Link>{" "}
        and <code>arena.sweep()</code>. Fields map directly to what was stored
        at write time.
      </p>
      <ul>
        <li>
          <code>key</code> — The record identifier (UUIDv7 hex or caller-supplied).
        </li>
        <li>
          <code>meta</code> — Metadata dict (always present, may be empty).
        </li>
        <li>
          <code>document</code> — Present when the record was written with
          attached text (either via <code>text=</code>,{" "}
          <code>document=</code>, or <code>place_document()</code>).
        </li>
        <li>
          <code>vector</code> — Present when <code>include_vectors=True</code>{" "}
          was passed to the fetch call. Always a <code>tuple</code> (not a
          list).
        </li>
        <li>
          <code>chunk</code> / <code>lineage</code> — Optional provenance
          attachments stored at write time.
        </li>
      </ul>

      <h3>
        <code>row.text</code> → <code>str | None</code>
      </h3>
      <p>
        Convenience alias for <code>row.document.text</code>. Returns{" "}
        <code>None</code> when there is no document attachment.
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("docs")
key = arena.write(text="Hello, world!", meta={"source": "test"})

row = arena.pull([key])[0]
print(row.key)        # UUIDv7 hex
print(row.meta)       # {"source": "test"}
print(row.text)       # "Hello, world!"
print(row.vector)     # (x, y)  — included by default
print(row.document)   # DocumentAttachment(text="Hello, world!", ...)
engine.close()`}</CodeBlock>

      <h2 id="hit">
        <code>Hit</code>
      </h2>
      <CodeBlock lang="python">{`@dataclass(frozen=True, slots=True)
class Hit:
    key:      str
    distance: float
    meta:     dict[str, MetaValue]
    document: DocumentAttachment | None = None
    vector:   tuple[float, ...] | None = None
    chunk:    ChunkInfo | None = None
    lineage:  EmbeddingLineage | None = None`}</CodeBlock>
      <p>
        A search result from any of the three probe methods. Fields are
        identical to <code>Row</code> with the addition of{" "}
        <code>distance</code>.
      </p>
      <ul>
        <li>
          <code>distance</code> — Metric-normalized distance from the query.
          For <strong>cosine</strong>: <code>0.0</code> = identical direction,{" "}
          <code>2.0</code> = opposite. For <strong>euclidean</strong>: L2
          distance. For <strong>dot product</strong>: negated dot product (lower
          = better).
        </li>
      </ul>

      <h3>
        <code>hit.text</code> → <code>str | None</code>
      </h3>
      <p>
        Same alias as <code>Row.text</code>.
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("docs")
arena.write(text="Design note", meta={"kind": "design"})
arena.write(text="Ops runbook", meta={"kind": "ops"})

hits = arena.probe_text("design", top=2)
for h in hits:
    print(f"{h.text!r:30s}  d={h.distance:.4f}")
# 'Design note'                   d=0.0000
# 'Ops runbook'                   d=0.1234   (example)

engine.close()`}</CodeBlock>

      <h2 id="wal-record">
        <code>WalRecord</code>
      </h2>
      <CodeBlock lang="python">{`@dataclass(frozen=True, slots=True)
class WalRecord:
    op:       "insert" | "erase" | "insert_ex"
    arena:    str
    key:      str
    vector:   tuple[float, ...] | None = None
    meta:     dict[str, MetaValue] | None = None
    document: DocumentAttachment | None = None
    chunk:    ChunkInfo | None = None
    lineage:  EmbeddingLineage | None = None`}</CodeBlock>
      <p>
        One acknowledged write-ahead log record, as returned by{" "}
        <Link to="/docs/python/engine">
          <code>engine.pending_writes()</code>
        </Link>
        . Transaction markers (<code>txn_begin</code>, <code>txn_commit</code>)
        are resolved during replay and never appear here.
      </p>
      <ul>
        <li>
          <code>op</code> — Operation kind:
          <ul>
            <li>
              <code>"insert"</code> — Vector and metadata only (no document
              attachment).
            </li>
            <li>
              <code>"insert_ex"</code> — Insert with document, chunk, or lineage
              attachments.
            </li>
            <li>
              <code>"erase"</code> — Delete. Vector and meta are empty for erase
              records.
            </li>
          </ul>
        </li>
        <li>
          <code>arena</code> — The vault name the mutation targeted.
        </li>
        <li>
          <code>key</code> — Record identifier.
        </li>
        <li>
          <code>vector</code> — <code>None</code> for erase records.
        </li>
      </ul>

      <h3>
        <code>record.is_delete</code> → <code>bool</code>
      </h3>
      <p>
        <code>True</code> when <code>op == "erase"</code>.
      </p>

      <h3>
        <code>WalRecord.from_entry(entry)</code> → <code>WalRecord</code>
      </h3>
      <p>
        Classmethod that wraps a low-level{" "}
        <code>elips.WalEntry</code> (produced by <code>elips.replay_wal()</code>
        ) into a typed, frozen <code>WalRecord</code>. Used internally by{" "}
        <code>engine.pending_writes()</code>; you rarely need to call it
        directly.
      </p>
      <CodeBlock lang="python">{`import tempfile, elips

path = tempfile.mkdtemp()
engine = elips.connect(path, dimension=2)
arena = engine.arena("docs")

k1 = arena.write(text="Alpha", meta={"v": 1})
k2 = arena.write(vector=[0.0, 1.0])
arena.discard([k1])

records = engine.pending_writes()
for r in records:
    print(r.op, r.arena, r.key, r.is_delete)
# insert  docs  <uuid>   False
# insert  docs  <uuid>   False
# erase   docs  <uuid>   True

engine.close()`}</CodeBlock>

      <h2 id="arena-health">
        <code>ArenaHealth</code>
      </h2>
      <CodeBlock lang="python">{`@dataclass(frozen=True, slots=True)
class ArenaHealth:
    name:             str
    live:             int
    pending_removals: int
    dimension:        int
    metric:           str
    read_only:        bool
    sealed:           bool`}</CodeBlock>
      <p>
        A point-in-time health snapshot, returned by{" "}
        <Link to="/docs/python/arena">
          <code>arena.health()</code>
        </Link>
        .
      </p>
      <ul>
        <li>
          <code>live</code> — Records currently searchable (excludes
          tombstones).
        </li>
        <li>
          <code>pending_removals</code> — Deleted records not yet reclaimed.
          Tombstones widen the search beam and consume memory until{" "}
          <code>arena.vacuum()</code> or automatic compaction.
        </li>
        <li>
          <code>dimension</code> / <code>metric</code> — Inherited from the
          database config.
        </li>
        <li>
          <code>read_only</code> — <code>True</code> if writes are currently
          rejected.
        </li>
        <li>
          <code>sealed</code> — <code>True</code> after <code>engine.close()</code>.
        </li>
      </ul>

      <h3>
        <code>health.tombstone_ratio</code> → <code>float</code>
      </h3>
      <p>
        The fraction of graph nodes that are tombstones:{" "}
        <code>pending_removals / (live + pending_removals)</code>. Returns{" "}
        <code>0.0</code> for empty arenas. Compare against the arena's
        configured <code>compaction_ratio</code> (default 0.2) to predict
        whether the next delete will trigger an automatic rebuild.
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("docs")
keys = [arena.write(vector=[float(i), 1.0]) for i in range(10)]
arena.discard(keys[:2])

health = arena.health()
print(health.live)              # 8
print(health.pending_removals)  # 0–2 (auto-compacts at 0.2)
print(f"{health.tombstone_ratio:.2f}")   # 0.00–0.20
engine.close()`}</CodeBlock>

      <h2 id="domain-types">Domain types</h2>
      <p>
        These types are C++ extension types re-exported from{" "}
        <code>elips._core</code> and available at <code>elips.DocumentAttachment</code>
        , <code>elips.ChunkInfo</code>, <code>elips.EmbeddingLineage</code>.
        They are mutable (not frozen dataclasses).
      </p>

      <h2 id="document-attachment">
        <code>DocumentAttachment</code>
      </h2>
      <CodeBlock lang="python">{`class DocumentAttachment:
    text:      str
    uri:       str = ""
    mime_type: str = "text/plain"

    def __init__(
        self,
        text: str,
        uri: str = "",
        mime_type: str = "text/plain",
    ) -> None: ...`}</CodeBlock>
      <p>
        Represents text content attached to a vector record. Stored in the
        record store alongside the vector and metadata; exposed on search hits
        and fetched rows.
      </p>
      <ul>
        <li>
          <code>text</code> — The source text. Used for lexical overlap scoring
          in hybrid search and for re-embedding.
        </li>
        <li>
          <code>uri</code> — Optional source URI (file path, URL, etc.). Purely
          informational; ELIPS does not fetch from it.
        </li>
        <li>
          <code>mime_type</code> — MIME type of the document.{" "}
          <code>text/plain</code> is the default; you may use{" "}
          <code>text/markdown</code>, <code>text/html</code>, etc.
        </li>
      </ul>
      <CodeBlock lang="python">{`# Plain text (the most common case)
doc = elips.DocumentAttachment(text="Alpha design note")

# With a source URI and MIME type
doc = elips.DocumentAttachment(
    text="# Proposal\n\nSee attached.",
    uri="proposals/q4.md",
    mime_type="text/markdown",
)
print(doc.text)        # # Proposal\n\nSee attached.
print(doc.uri)         # proposals/q4.md
print(doc.mime_type)   # text/markdown`}</CodeBlock>
      <p>
        <strong>When to use <code>DocumentAttachment</code> directly:</strong>{" "}
        when you need to attach a URI or MIME type, you must supply an explicit{" "}
        <code>vector</code> alongside the document (the native{" "}
        <code>place_document()</code> path cannot carry custom attachment
        fields). For plain text with no URI, just pass{" "}
        <code>text="..."</code> to <code>RecordInput</code>.
      </p>

      <h2 id="chunk-info">
        <code>ChunkInfo</code>
      </h2>
      <CodeBlock lang="python">{`class ChunkInfo:
    document_key: str
    ordinal:      int
    char_start:   int
    char_end:     int`}</CodeBlock>
      <p>
        Describes the position of this record within a parent document.
        Typically used in RAG pipelines where a long document is split into
        overlapping chunks, each embedded and stored as a separate record.{" "}
        <code>ChunkInfo</code> lets you reconstruct the full document from hits
        and trace each chunk back to its source.
      </p>
      <ul>
        <li>
          <code>document_key</code> — The key of the parent document record (or
          any stable external identifier).
        </li>
        <li>
          <code>ordinal</code> — Zero-based chunk index within the document.
        </li>
        <li>
          <code>char_start</code> / <code>char_end</code> — Character byte
          offsets within the original document text (half-open interval:{" "}
          <code>[char_start, char_end)</code>).
        </li>
      </ul>
      <CodeBlock lang="python">{`import elips

def ingest_chunked_document(arena, doc_key: str, text: str, chunk_size: int = 512):
    """Split text into chunks and ingest each with ChunkInfo."""
    chunks = []
    for ordinal, start in enumerate(range(0, len(text), chunk_size)):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        ci = elips.ChunkInfo()
        ci.document_key = doc_key
        ci.ordinal = ordinal
        ci.char_start = start
        ci.char_end = end

        chunks.append(elips.RecordInput(text=chunk_text, chunk=ci))

    return arena.write_many(chunks)

engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("chunks")
doc_key = "report-q4-2026"
text = "Annual report content... " * 50

keys = ingest_chunked_document(arena, doc_key, text)

rows = arena.pull(keys[:3])
for row in rows:
    ci = row.chunk
    if ci:
        print(f"chunk {ci.ordinal}: chars [{ci.char_start}, {ci.char_end})")
engine.close()`}</CodeBlock>

      <h2 id="embedding-lineage">
        <code>EmbeddingLineage</code>
      </h2>
      <CodeBlock lang="python">{`class EmbeddingLineage:
    provider:   str
    model:      str
    revision:   str
    attributes: dict[str, str]`}</CodeBlock>
      <p>
        Records the provenance of the embedding stored in a record. ELIPS
        automatically creates an <code>EmbeddingLineage</code> with{" "}
        <code>provider="python"</code> and <code>model="callable"</code> when a
        Python embedder generates a vector. For native text embedders, the
        runtime fills in the correct provider/model/revision from the embedder
        metadata.
      </p>
      <ul>
        <li>
          <code>provider</code> — Source system: <code>"python"</code>,{" "}
          <code>"openai"</code>, <code>"local"</code>, etc.
        </li>
        <li>
          <code>model</code> — Model identifier, e.g.{" "}
          <code>"all-MiniLM-L6-v2"</code> or <code>"text-embedding-3-small"</code>.
        </li>
        <li>
          <code>revision</code> — Model version or commit hash.
        </li>
          <code>attributes</code> — Arbitrary string key/value metadata (e.g.,{" "}
          <code>quantization="int8"</code>).
      </ul>
      <CodeBlock lang="python">{`lineage = elips.EmbeddingLineage()
lineage.provider = "openai"
lineage.model = "text-embedding-3-small"
lineage.revision = "2024-02"
lineage.attributes = {"dimensions": "1536"}

key = arena.write(
    vector=my_openai_vector,
    text="The source text",
    lineage=lineage,
)

row = arena.pull([key])[0]
print(row.lineage.provider)     # openai
print(row.lineage.model)        # text-embedding-3-small
print(row.lineage.attributes)   # {"dimensions": "1536"}`}</CodeBlock>

      <h2 id="type-aliases">Type aliases</h2>

      <h3>
        <code>Embedder</code> — Protocol
      </h3>
      <CodeBlock lang="python">{`from collections.abc import Sequence
from typing import Protocol, runtime_checkable

@runtime_checkable
class Embedder(Protocol):
    def __call__(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]: ...`}</CodeBlock>
      <p>
        A <code>runtime_checkable</code> Protocol for Python batch embedders
        accepted by <code>elips.connect(embedder=...)</code> and{" "}
        <code>engine.arena(embedder=...)</code>. Any callable with the right
        signature satisfies it.
      </p>
      <p>
        The callable receives a batch of strings and must return one embedding
        vector per input string. Returning a different-length list raises{" "}
        <code>ValueError</code> at write or probe time.
      </p>
      <CodeBlock lang="python">{`# numpy / sentence-transformers style
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()

assert isinstance(embed, elips.Embedder)   # runtime_checkable

engine = elips.connect(
    ":memory:",
    dimension=384,
    embedder=embed,
    use_default_text_embedder=False,
)
engine.close()

# Toy embedder for tests
def toy_embed(texts):
    return [[float(len(t)), 0.0] for t in texts]`}</CodeBlock>

      <h3>
        <code>RecordInputLike</code> — Type alias
      </h3>
      <CodeBlock lang="python">{`RecordInputLike = Union[RecordInput, RecordInputDict, BatchRecord]`}</CodeBlock>
      <p>
        The union of types accepted wherever a single record can be passed:{" "}
        <code>RecordInput</code>, a modern-format dict (fields{" "}
        <code>vector</code>, <code>text</code>, <code>meta</code>,{" "}
        <code>key</code>, …), or a legacy low-level batch record dict (fields{" "}
        <code>vector</code>, <code>data</code>, <code>id</code>). In all cases
        the value is internally normalized via{" "}
        <code>RecordInput.from_mapping()</code>.
      </p>
      <CodeBlock lang="python">{`# All three forms are accepted by write() and write_many()
arena.write(elips.RecordInput(text="a", meta={"k": 1}))      # RecordInput
arena.write({"text": "b", "meta": {"k": 2}})                 # RecordInputDict
arena.write({"text": "c", "data": {"k": 3}})                 # BatchRecord (legacy)`}</CodeBlock>
    </DocsShell>
  );
}
