import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/arena")({
  head: () => ({
    meta: [
      { title: "Collection — ELIPS Python Reference" },
      {
        name: "description",
        content:
          "Reference for the ELIPS Collection — the Chroma-style high-level surface: add, upsert, get, update, delete, query, count, peek, plus ELIPS extensions.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="Collection"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "open", label: "Opening a collection" },
        { id: "add", label: "add()" },
        { id: "upsert", label: "upsert()" },
        { id: "get", label: "get()" },
        { id: "update", label: "update()" },
        { id: "delete", label: "delete()" },
        { id: "query", label: "query()" },
        { id: "count", label: "count()" },
        { id: "peek", label: "peek()" },
        { id: "extensions", label: "ELIPS extensions" },
        { id: "results", label: "Result types" },
        { id: "thread-safety", label: "Thread safety" },
      ]}
    >
      <p className="text-[18px] text-ink">
        <code>Collection</code> is the high-level, Chroma-compatible surface in
        ELIPS. Obtain one from a{" "}
        <Link to="/docs/python/engine">
          <code>Client</code>
        </Link>{" "}
        via <code>client.get_or_create_collection("docs")</code>. It exposes the
        canonical verbs — <code>add</code>, <code>upsert</code>,{" "}
        <code>get</code>, <code>update</code>, <code>delete</code>,{" "}
        <code>query</code>, <code>count</code>, and <code>peek</code> — plus
        ELIPS-specific maintenance extensions.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        A <code>Collection</code> is a named set of records that share a
        dimension and distance metric. Each record carries an id (its{" "}
        <code>key</code>), a vector embedding, optional document text, and
        arbitrary metadata. Reads return typed <code>Hit</code> and{" "}
        <code>Row</code> objects whose fields are <code>.key</code>,{" "}
        <code>.distance</code>, <code>.meta</code>, and <code>.text</code>.
      </p>

      <h2 id="open">Opening a collection</h2>
      <p>
        Collections live inside a{" "}
        <Link to="/docs/python/engine">
          <code>Client</code>
        </Link>
        . Open an in-memory <code>Client</code> (which requires a{" "}
        <code>dimension</code>) or an on-disk <code>PersistentClient</code>, then
        get or create a collection by name.
      </p>
      <CodeBlock lang="python">{`import elips

# Ephemeral, in-memory
client = elips.Client(dimension=384, metric="cosine")

# Or persistent on disk (dimension restored from the manifest when reopening)
client = elips.PersistentClient("/tmp/db", dimension=384)

docs = client.get_or_create_collection("docs")`}</CodeBlock>
      <h2 id="add">
        <code>add()</code> → <code>str | list[str]</code>
      </h2>
      <CodeBlock lang="python">{`# Column form (Chroma)
collection.add(
    *,
    ids: str | list[str] | None = None,
    embeddings: Sequence | None = None,
    documents: str | list[str] | None = None,
    metadatas: dict | list[dict] | None = None,
)

# Single-record form (ELIPS convenience)
collection.add(
    *,
    embedding: Sequence[float] | None = None,   # alias: vector=
    text: str | None = None,
    metadata: dict | None = None,
    id: str | None = None,
)`}</CodeBlock>
      <p>
        Insert one or more records. <code>documents=</code> are raw text strings
        (or a <code>DocumentAttachment</code>). Returns the assigned id as a{" "}
        <code>str</code> for a single record, or a <code>list[str]</code> for the
        column form. <code>add</code> has upsert semantics: re-adding the same id
        overwrites the existing record.
      </p>
      <CodeBlock lang="python">{`client = elips.Client(dimension=2)
docs = client.get_or_create_collection("docs")

# Column form — bulk insert
ids = docs.add(
    ids=["doc-1", "doc-2"],
    embeddings=[[1.0, 0.0], [0.0, 1.0]],
    documents=["Alpha design note", "Beta runbook"],
    metadatas=[{"kind": "design"}, {"kind": "ops"}],
)

# Single-record form
rid = docs.add(embedding=[0.5, 0.5], text="Gamma spec", metadata={"kind": "spec"})

client.close()`}</CodeBlock>
      <h2 id="upsert">
        <code>upsert()</code> → <code>str | list[str]</code>
      </h2>
      <p>
        Identical to <code>add()</code> in ELIPS — both have upsert semantics.
        Provided for Chroma compatibility, to make the intent to
        overwrite-or-insert explicit.
      </p>
      <CodeBlock lang="python">{`docs.upsert(
    ids=["doc-1"],
    embeddings=[[0.9, 0.1]],
    metadatas=[{"kind": "design", "rev": 2}],
)`}</CodeBlock>

      <h2 id="get">
        <code>get()</code> → <code>list[Row]</code>
      </h2>
      <CodeBlock lang="python">{`collection.get(
    ids: str | list[str] | None = None,
    *,
    where: dict | None = None,
    limit: int | None = None,
    offset: int = 0,
    include: list[str] | None = None,
) -> list[Row]`}</CodeBlock>
      <p>
        Fetch records by id, or scan with a metadata <code>where</code> filter.
        Returns typed{" "}
        <Link to="/docs/python/models">
          <code>Row</code>
        </Link>{" "}
        objects. Pass <code>include=["embeddings"]</code> to hydrate stored
        vectors.
      </p>
      <CodeBlock lang="python">{`# By id
rows = docs.get(ids=["doc-1", "doc-2"])
print(rows[0].key, rows[0].text, rows[0].meta)

# Scan with a metadata filter + pagination
page = docs.get(where={"kind": "design"}, limit=50, offset=0)

# Include stored vectors
rows = docs.get(ids=["doc-1"], include=["embeddings"])
print(rows[0].vector)`}</CodeBlock>
      <h2 id="update">
        <code>update()</code> → <code>None</code>
      </h2>
      <CodeBlock lang="python">{`collection.update(
    *,
    ids: str | list[str],
    embeddings: Sequence | None = None,
    documents: str | list[str] | None = None,
    metadatas: dict | list[dict] | None = None,
)`}</CodeBlock>
      <p>
        Update existing records in place. Raises <code>NotFoundError</code> if an
        id does not exist — unlike <code>add()</code> / <code>upsert()</code>, it
        never creates new records.
      </p>
      <CodeBlock lang="python">{`docs.update(
    ids=["doc-1"],
    metadatas=[{"kind": "design", "status": "approved"}],
)`}</CodeBlock>

      <h2 id="delete">
        <code>delete()</code> → <code>int</code>
      </h2>
      <CodeBlock lang="python">{`collection.delete(
    ids: str | list[str] | None = None,
    *,
    where: dict | None = None,
) -> int`}</CodeBlock>
      <p>
        Remove records by id, by metadata <code>where</code> filter, or both.
        Returns the count of records removed. At least one of <code>ids</code> or{" "}
        <code>where</code> is required — there is no "delete everything"
        shorthand.
      </p>
      <CodeBlock lang="python">{`# By id
removed = docs.delete(ids=["doc-1", "doc-2"])

# By metadata filter
removed = docs.delete(where={"kind": "ops"})
print(removed)`}</CodeBlock>
      <h2 id="query">
        <code>query()</code> → <code>list[Hit] | list[list[Hit]]</code>
      </h2>
      <CodeBlock lang="python">{`collection.query(
    query_embeddings: Sequence | None = None,
    query_texts: str | list[str] | None = None,
    *,
    n_results: int = 10,          # aliases: k=, limit=
    where: dict | None = None,
    include: list[str] | None = None,
    max_distance: float | None = None,
    lexical_weight: float = 0.25,
) -> list[Hit] | list[list[Hit]]`}</CodeBlock>
      <p>
        Nearest-neighbour search. A single query returns a flat{" "}
        <code>list[Hit]</code>; a batch of queries returns a{" "}
        <code>list[list[Hit]]</code> aligned to the inputs. Hits are sorted by
        ascending <code>distance</code>.
      </p>
      <ul>
        <li>
          <code>query_embeddings</code> — one query vector, or a list of vectors
          for a batch.
        </li>
        <li>
          <code>query_texts</code> — query text, embedded by the collection's
          embedder. Pass <em>both</em> <code>query_embeddings</code> and{" "}
          <code>query_texts</code> for hybrid retrieval, blended by{" "}
          <code>lexical_weight</code>.
        </li>
        <li>
          <code>n_results</code> — maximum hits (aliases <code>k</code> /{" "}
          <code>limit</code>).
        </li>
        <li>
          <code>where</code> — metadata{" "}
          <Link to="/docs/python/filtering">
            <code>filter</code>
          </Link>
          .
        </li>
        <li>
          <code>max_distance</code> — drop hits beyond this distance.
        </li>
        <li>
          <code>include</code> — e.g. <code>["embeddings"]</code> to return
          vectors on each hit.
        </li>
      </ul>
      <CodeBlock lang="python">{`client = elips.Client(dimension=2)
docs = client.get_or_create_collection("docs")
docs.add(
    ids=["a", "b"],
    embeddings=[[1.0, 0.0], [0.0, 1.0]],
    documents=["Alpha", "Beta"],
    metadatas=[{"kind": "a"}, {"kind": "b"}],
)

# Single vector query -> list[Hit]
hits = docs.query(query_embeddings=[1.0, 0.0], n_results=1)
print(hits[0].text)       # Alpha
print(hits[0].distance)   # ~ 0.0

# Filtered query
hits = docs.query(query_embeddings=[0.0, 1.0], n_results=5, where={"kind": "b"})
print(hits[0].text)       # Beta

# Batch query -> list[list[Hit]]
batch = docs.query(query_embeddings=[[1.0, 0.0], [0.0, 1.0]], n_results=1)
print(batch[0][0].text, batch[1][0].text)   # Alpha Beta

# Hybrid (vector + text)
hits = docs.query(query_embeddings=[1.0, 0.0], query_texts="alpha", lexical_weight=0.3)

client.close()`}</CodeBlock>

      <h2 id="count">
        <code>count()</code> → <code>int</code>
      </h2>
      <p>Return the number of live (non-tombstoned) records.</p>
      <CodeBlock lang="python">{`print(docs.count())   # e.g. 2`}</CodeBlock>

      <h2 id="peek">
        <code>peek()</code> → <code>list[Row]</code>
      </h2>
      <p>
        Return the first <code>n</code> records (default 10) as <code>Row</code>{" "}
        objects — a quick way to inspect a collection's contents.
      </p>
      <CodeBlock lang="python">{`for row in docs.peek(n=5):
    print(row.key, row.meta)`}</CodeBlock>
      <h2 id="extensions">ELIPS extensions</h2>
      <p>
        Beyond the Chroma surface, a <code>Collection</code> exposes maintenance
        and introspection methods specific to ELIPS.
      </p>
      <ul>
        <li>
          <code>explain(vector, ...)</code> → <code>QueryPlan</code> — return the
          planner's decision for a hypothetical query without running it.
        </li>
        <li>
          <code>health()</code> →{" "}
          <Link to="/docs/python/models">
            <code>CollectionHealth</code>
          </Link>{" "}
          — a point-in-time snapshot (live count, dimension, metric, tombstone
          ratio, read-only / sealed state).
        </li>
        <li>
          <code>freeze(frozen=True)</code> — refuse writes without reopening the
          database read-only; <code>freeze(False)</code> re-enables them.
        </li>
        <li>
          <code>vacuum()</code> — reclaim graph nodes held by deleted records.
        </li>
        <li>
          <code>rebuild()</code> — rebuild the HNSW graph from the record store
          (higher recall than incremental inserts; run after a bulk load).
        </li>
        <li>
          <code>compress()</code> / <code>train()</code> — train the
          quantization codebook for the collection.
        </li>
      </ul>
      <p>
        Properties: <code>name</code>, <code>native</code> (the low-level{" "}
        <Link to="/docs/python/vault">
          <code>Vault</code>
        </Link>{" "}
        escape hatch), <code>pending_removals</code>, <code>read_only</code>, and{" "}
        <code>sealed</code>.
      </p>
      <CodeBlock lang="python">{`# Health snapshot
h = docs.health()
print(h.live, h.dimension, h.metric, h.tombstone_ratio)

# Maintenance after a bulk load
docs.rebuild()
docs.vacuum()

# Freeze writes during a read-only serving window
docs.freeze(True)
docs.freeze(False)

# Drop to the native handle for a low-level capability
vault = docs.native`}</CodeBlock>
      <h2 id="results">Result types</h2>
      <p>
        Reads return typed records, never raw dicts. Both types live in{" "}
        <Link to="/docs/python/models">the models reference</Link>.
      </p>
      <ul>
        <li>
          <code>Hit(key, distance, meta, document=None, vector=None, ...)</code>{" "}
          — returned by <code>query()</code>. Fields: <code>.key</code>,{" "}
          <code>.distance</code>, <code>.meta</code>, and <code>.text</code>{" "}
          (shortcut for <code>document.text</code>).
        </li>
        <li>
          <code>Row(key, meta, document=None, vector=None, ...)</code> — returned
          by <code>get()</code> and <code>peek()</code>. Fields:{" "}
          <code>.key</code>, <code>.meta</code>, <code>.text</code>, and{" "}
          <code>.vector</code> (when included).
        </li>
      </ul>
      <p>
        Note the field names: records expose <code>.key</code> and{" "}
        <code>.meta</code> — not <code>.id</code> or <code>.metadata</code>.
      </p>
      <CodeBlock lang="python">{`hits = docs.query(query_embeddings=[1.0, 0.0], n_results=3)
for hit in hits:
    print(hit.key, hit.distance, hit.meta, hit.text)

rows = docs.get(ids=["a", "b"], include=["embeddings"])
for row in rows:
    print(row.key, row.meta, row.vector)`}</CodeBlock>

      <h2 id="thread-safety">Thread safety</h2>
      <p>
        A <code>Collection</code> carries no locks of its own. Thread safety is
        provided by the underlying native{" "}
        <Link to="/docs/python/vault">
          <code>Vault</code>
        </Link>{" "}
        and{" "}
        <Link to="/docs/python/database">
          <code>Database</code>
        </Link>
        , which serialize concurrent writes and allow concurrent reads. If you
        pass a Python embedder, make sure the callable is itself thread-safe.
      </p>
    </DocsShell>
  );
}
