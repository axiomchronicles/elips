import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/arena")({
  head: () => ({
    meta: [
      { title: "Arena — ELIPS Python Reference" },
      {
        name: "description",
        content:
          "Complete reference for the Arena class — the modern typed wrapper around Vault that handles ingestion, vector search, text search, hybrid retrieval, and maintenance.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="Arena"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "properties", label: "Properties" },
        { id: "count", label: "count()" },
        { id: "write", label: "write()" },
        { id: "write-many", label: "write_many()" },
        { id: "ingest", label: "ingest() / merge()" },
        { id: "probe", label: "probe()" },
        { id: "probe-text", label: "probe_text()" },
        { id: "probe-hybrid", label: "probe_hybrid()" },
        { id: "explain", label: "explain()" },
        { id: "pull", label: "pull()" },
        { id: "sweep", label: "sweep()" },
        { id: "discard", label: "discard()" },
        { id: "maintenance", label: "Maintenance" },
        { id: "health", label: "health()" },
        { id: "embedding-resolution", label: "Embedding resolution" },
        { id: "thread-safety", label: "Thread safety" },
      ]}
    >
      <p className="text-[18px] text-ink">
        <code>Arena</code> is the typed high-level wrapper around a single ELIPS
        vault. Obtain one via{" "}
        <Link to="/docs/python/engine">
          <code>engine.arena(name)</code>
        </Link>
        . It handles writing, searching, fetching, scanning, and deleting
        records, and automatically routes between native C++ text embedding and
        Python-side embedding depending on how the database was configured.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        An <code>Arena</code> wraps exactly one <code>Vault</code>. All
        operations go through the same WAL-backed record store and the same HNSW
        or exact index. The arena adds:
      </p>
      <ul>
        <li>
          Typed <Link to="/docs/python/models"><code>RecordInput</code></Link>{" "}
          / <code>Row</code> / <code>Hit</code> data classes instead of raw
          dicts.
        </li>
        <li>
          Automatic batch embedding: records without an explicit vector are
          embedded in a single batch call before insertion.
        </li>
        <li>
          Transparent routing between{" "}
          <code>Vault.seek_text</code> (native embedder path) and{" "}
          <code>Vault.seek_hybrid</code> (Python embedder fallback).
        </li>
        <li>
          Column-oriented bulk ingestion via <code>ingest()</code> for
          compatibility with pipeline-style data frames.
        </li>
      </ul>
      <CodeBlock lang="python">{`import elips

with elips.connect(":memory:", dimension=2) as engine:
    arena = engine.arena("articles")

    # Write records
    keys = arena.write_many([
        elips.RecordInput(text="The quick brown fox", meta={"lang": "en"}),
        elips.RecordInput(text="Le renard brun rapide", meta={"lang": "fr"}),
    ])

    # Vector search
    hits = arena.probe([1.0, 0.0], top=5)

    # Text search (uses native embedder)
    hits = arena.probe_text("quick fox", top=3)

    # Fetch by key
    rows = arena.pull(keys, include_vectors=True)
    print(rows[0].text)   # The quick brown fox

    # Delete
    removed = arena.discard(keys[:1])
    print(removed)        # 1`}</CodeBlock>

      <h2 id="properties">Properties</h2>

      <h3>
        <code>arena.name</code> → <code>str</code>
      </h3>
      <p>The vault name this arena was opened against.</p>
      <CodeBlock lang="python">{`arena = engine.arena("documents")
print(arena.name)  # documents`}</CodeBlock>

      <h3>
        <code>arena.raw</code> → <code>Vault</code>
      </h3>
      <p>
        The underlying low-level <code>Vault</code> handle. Drop to this when
        you need a capability the arena does not expose — for example,{" "}
        <code>vault.place_many()</code>, <code>vault.info()</code>, or direct
        EQL targeting.
      </p>
      <CodeBlock lang="python">{`arena = engine.arena("documents")
info = arena.raw.info()
print(info.dimension, info.metric)  # 128  cosine`}</CodeBlock>

      <h3>
        <code>arena.read_only</code> → <code>bool</code>
      </h3>
      <p>
        <code>True</code> if this arena currently refuses writes — either
        because the database was opened with{" "}
        <code>access_mode="read_only"</code> or because{" "}
        <code>arena.freeze(True)</code> was called.
      </p>

      <h3>
        <code>arena.sealed</code> → <code>bool</code>
      </h3>
      <p>
        <code>True</code> once the owning engine has been closed. Writes to a
        sealed arena raise <code>elips.StorageError</code>.
      </p>

      <h3>
        <code>arena.pending_removals</code> → <code>int</code>
      </h3>
      <p>
        Number of deleted records that have been tombstoned in the HNSW graph
        but not yet reclaimed. Tombstones act as routing waypoints so live
        neighbours stay reachable, but they consume memory and widen the search
        beam. Call <code>arena.vacuum()</code> to reclaim them immediately, or
        let the arena auto-compact when they cross its{" "}
        <code>compaction_ratio</code> (default 0.2).
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("documents")
keys = [arena.write(vector=[float(i), 1.0]) for i in range(20)]
arena.discard(keys[:2])           # 2 / 20 = 0.10 — below auto-compact
print(arena.pending_removals)     # 2
engine.close()`}</CodeBlock>

      <h2 id="count">
        <code>count()</code> → <code>int</code>
      </h2>
      <p>
        Return the number of live (non-tombstoned) records. Does not count
        deleted records awaiting compaction.
      </p>
      <CodeBlock lang="python">{`arena = engine.arena("documents")
print(arena.count())   # 0
arena.write(vector=[1.0, 0.0])
print(arena.count())   # 1`}</CodeBlock>

      <h2 id="write">
        <code>write()</code> → <code>str</code>
      </h2>
      <CodeBlock lang="python">{`arena.write(
    record: RecordInput | dict | None = None,
    /,
    *,
    vector: Sequence[float] | None = None,
    text: str | None = None,
    meta: dict | None = None,
    key: str | None = None,
    document: DocumentAttachment | None = None,
    chunk: ChunkInfo | None = None,
    lineage: EmbeddingLineage | None = None,
) -> str`}</CodeBlock>
      <p>
        Write a single record. Returns the assigned record key (UUIDv7 hex
        unless <code>key</code> was supplied).
      </p>
      <p>
        You can pass either a positional structured record (<code>RecordInput</code>{" "}
        or a <code>dict</code> with the right shape) <em>or</em> keyword
        arguments — never both.
      </p>
      <ul>
        <li>
          <strong>Vector path</strong>: supply <code>vector</code>. If{" "}
          <code>text</code> or <code>document</code> is also supplied, the text
          is stored as a <code>DocumentAttachment</code> for hybrid search but
          the embedding is taken from <code>vector</code>.
        </li>
        <li>
          <strong>Text path</strong>: supply <code>text</code> (or{" "}
          <code>document</code> with text). The arena embeds via the native
          embedder if present, otherwise via the configured Python embedder.
        </li>
        <li>
          <code>key</code> — Optional caller-supplied identifier. Must be a
          valid ELIPS ID (use <code>elips.generate_id()</code> if you need a
          pre-allocated key).
        </li>
        <li>
          <code>document</code> — An <code>elips.DocumentAttachment</code> that
          carries text, a URI, and a MIME type. Use this when you want to attach
          a URI or a non-<code>text/plain</code> MIME type; otherwise a plain{" "}
          <code>text="..."</code> is sufficient.
        </li>
        <li>
          <code>chunk</code> — A <code>ChunkInfo</code> describing where in a
          parent document this chunk came from.
        </li>
        <li>
          <code>lineage</code> — An <code>EmbeddingLineage</code> recording the
          embedding provider and model.
        </li>
      </ul>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("docs")

# Keyword form — most common
key = arena.write(text="Alpha design note", meta={"kind": "design"})

# Structured form
record = elips.RecordInput(
    vector=[1.0, 0.0],
    text="Beta runbook",
    meta={"kind": "ops"},
)
key = arena.write(record)

# Dict form (same as RecordInput.from_mapping)
key = arena.write({"text": "Gamma spec", "meta": {"kind": "spec"}})

# With explicit document attachment (URI + MIME type)
doc = elips.DocumentAttachment(
    text="# Proposal\n\nSee attached.",
    uri="proposals/q4.md",
    mime_type="text/markdown",
)
chunk = elips.ChunkInfo()
chunk.document_key = "doc-q4"
chunk.ordinal = 0
chunk.char_start = 0
chunk.char_end = 27
key = arena.write(vector=[0.8, 0.2], document=doc, chunk=chunk)

engine.close()`}</CodeBlock>

      <h2 id="write-many">
        <code>write_many()</code> → <code>list[str]</code>
      </h2>
      <CodeBlock lang="python">{`arena.write_many(
    records: Sequence[RecordInput | dict],
) -> list[str]`}</CodeBlock>
      <p>
        Write a batch of records. Returns assigned keys in input order. At least
        one record is required.
      </p>
      <p>
        Records without a vector are collected, their texts are passed to the
        embedder in a single batch call, and the resulting vectors are
        distributed back. This avoids per-record embedding overhead. Records
        that already have a vector bypass the embedder entirely.
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("docs")

keys = arena.write_many([
    elips.RecordInput(text="Alpha",    meta={"order": 1}),
    elips.RecordInput(text="Beta",     meta={"order": 2}),
    elips.RecordInput(vector=[1.0, 0.0], meta={"order": 3}),
    {"text": "Delta", "meta": {"order": 4}},   # dict form
])

print(len(keys))   # 4
engine.close()`}</CodeBlock>
      <p>
        <strong>Performance note:</strong> for bulk imports, prefer{" "}
        <code>write_many()</code> or <code>ingest()</code> over repeated{" "}
        <code>write()</code> calls. The embedder batch is amortized across the
        whole list, and the WAL absorbs appends efficiently.
      </p>

      <h2 id="ingest">
        <code>ingest()</code> / <code>merge()</code> → <code>list[str]</code>
      </h2>
      <CodeBlock lang="python">{`# Structured form — same as write_many
arena.ingest(records: Sequence[RecordInput | dict]) -> list[str]

# Column-oriented form — pipeline / dataframe style
arena.ingest(
    *,
    vectors:   Sequence[Sequence[float] | None] | None = None,
    texts:     Sequence[str | None] | None = None,
    meta:      Sequence[dict | None] | None = None,
    keys:      Sequence[str | None] | None = None,
    documents: Sequence[DocumentAttachment | None] | None = None,
    chunks:    Sequence[ChunkInfo | None] | None = None,
    lineages:  Sequence[EmbeddingLineage | None] | None = None,
) -> list[str]`}</CodeBlock>
      <p>
        <code>ingest()</code> is the column-oriented bulk ingestion API. It
        accepts either a list of structured records (identical to{" "}
        <code>write_many()</code>) or parallel column sequences. All column
        sequences must have the same length.
      </p>
      <p>
        <code>merge()</code> is a compatibility alias for <code>ingest()</code>{" "}
        with identical semantics.
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("docs")

# Column-oriented — natural for data-pipeline output
keys = arena.ingest(
    texts=["Alpha note", "Beta note", "Gamma note"],
    meta=[{"category": "A"}, {"category": "B"}, {"category": "C"}],
)
print(len(keys))   # 3

# Column-oriented with explicit vectors for some rows
keys = arena.ingest(
    vectors=[[1.0, 0.0], None, [0.0, 1.0]],
    texts=[None, "only text", None],
    meta=[{"v": True}, {"t": True}, {"v": True}],
)
# Row 1: vector provided directly
# Row 2: embedded via configured embedder
# Row 3: vector provided directly

engine.close()`}</CodeBlock>
      <p>
        <strong>When to prefer <code>ingest()</code> over <code>write_many()</code>:</strong>{" "}
        when your upstream pipeline already produces column-oriented outputs
        (e.g., NumPy arrays, Pandas DataFrames sliced into lists). The
        structured <code>write_many()</code> form is recommended for new code
        because it keeps related fields together and is easier to type-check.
      </p>

      <h2 id="probe">
        <code>probe()</code> → <code>list[Hit]</code>
      </h2>
      <CodeBlock lang="python">{`arena.probe(
    vector: Sequence[float],
    *,
    top: int = 10,
    where: Filter | None = None,
    max_distance: float | None = None,
    include_vectors: bool = False,
) -> list[Hit]`}</CodeBlock>
      <p>
        Approximate nearest-neighbour search. Returns hits sorted by distance
        (ascending).
      </p>
      <ul>
        <li>
          <code>vector</code> — Query vector. Must match the database dimension.
        </li>
        <li>
          <code>top</code> — Maximum number of hits returned.
        </li>
        <li>
          <code>where</code> — Optional{" "}
          <Link to="/docs/python/filtering">
            <code>Filter</code>
          </Link>{" "}
          to narrow the candidate set.
        </li>
        <li>
          <code>max_distance</code> — Hard upper bound on distance. Hits beyond
          this threshold are dropped even if fewer than <code>top</code> remain.
        </li>
        <li>
          <code>include_vectors</code> — Whether to hydrate the stored embedding
          on each hit. Incurs one record-store fetch per hit; leave{" "}
          <code>False</code> unless you need the vectors.
        </li>
      </ul>
      <CodeBlock lang="python">{`import elips

engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("docs")
arena.write(vector=[1.0, 0.0], text="Alpha", meta={"kind": "a"})
arena.write(vector=[0.0, 1.0], text="Beta",  meta={"kind": "b"})

# Plain ANN
hits = arena.probe([1.0, 0.0], top=1)
print(hits[0].text)      # Alpha
print(hits[0].distance)  # ≈ 0.0

# Filtered ANN
f = elips.Filter().field("kind").equals("b")
hits = arena.probe([0.0, 1.0], top=5, where=f)
print(len(hits))         # 1
print(hits[0].text)      # Beta

# With distance threshold
hits = arena.probe([1.0, 0.0], top=10, max_distance=0.5)

# With stored vectors hydrated
hits = arena.probe([1.0, 0.0], top=1, include_vectors=True)
print(hits[0].vector)    # (1.0, 0.0)

engine.close()`}</CodeBlock>

      <h2 id="probe-text">
        <code>probe_text()</code> → <code>list[Hit]</code>
      </h2>
      <CodeBlock lang="python">{`arena.probe_text(
    text: str,
    *,
    top: int = 10,
    where: Filter | None = None,
    max_distance: float | None = None,
    include_vectors: bool = False,
    lexical_weight: float = 0.25,
) -> list[Hit]`}</CodeBlock>
      <p>
        Text-first retrieval. The call routes itself based on the database
        configuration:
      </p>
      <ul>
        <li>
          <strong>Native text embedder present</strong> (i.e., the database was
          opened with the built-in local embedder or a{" "}
          <code>LocalEmbedderConfig</code>): uses{" "}
          <code>Vault.seek_text()</code> — the C++ core embeds the query and
          runs ANN in one step.
        </li>
        <li>
          <strong>Python embedder configured</strong>: embeds the query with the
          Python callable and then calls{" "}
          <code>Vault.seek_hybrid()</code> with the <code>lexical_weight</code>{" "}
          blending factor.
        </li>
        <li>
          <strong>Neither</strong>: raises <code>ValueError</code>. ELIPS never
          silently falls back to lexical-only retrieval.
        </li>
      </ul>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("docs")
arena.write(text="The quick brown fox", meta={"lang": "en"})
arena.write(text="Le renard brun rapide", meta={"lang": "fr"})

# Text-first, no filter
hits = arena.probe_text("quick fox", top=5)
print(hits[0].text)          # The quick brown fox

# Filtered by language
f = elips.Filter().field("lang").equals("fr")
hits = arena.probe_text("renard", top=3, where=f)
print(hits[0].meta["lang"])  # fr

engine.close()`}</CodeBlock>
      <p>
        <code>lexical_weight</code> is only used on the Python-embedder path. It
        controls how much the BM25-style lexical overlap score contributes to
        the fused ranking. <code>0.0</code> = pure ANN; <code>1.0</code> = pure
        lexical; the default <code>0.25</code> gives a modest lexical boost.
      </p>

      <h2 id="probe-hybrid">
        <code>probe_hybrid()</code> → <code>list[Hit]</code>
      </h2>
      <CodeBlock lang="python">{`arena.probe_hybrid(
    vector: Sequence[float],
    text: str,
    *,
    top: int = 10,
    where: Filter | None = None,
    max_distance: float | None = None,
    lexical_weight: float = 0.25,
    include_vectors: bool = False,
) -> list[Hit]`}</CodeBlock>
      <p>
        Explicit hybrid retrieval: you supply both the query vector and the
        query text. The core fuses ANN distance with lexical overlap from stored
        documents according to <code>lexical_weight</code>. Use this when you
        have already embedded the query and want control over the blend rather
        than leaving routing to <code>probe_text()</code>.
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("docs")
arena.write(vector=[1.0, 0.0], text="Alpha design note")
arena.write(vector=[0.0, 1.0], text="Beta security note")

hits = arena.probe_hybrid(
    [1.0, 0.0],    # query vector
    "alpha",       # query text for lexical overlap
    top=5,
    lexical_weight=0.3,
)
print(hits[0].text)   # Alpha design note
engine.close()`}</CodeBlock>

      <h2 id="explain">
        <code>explain()</code> → <code>QueryPlan</code>
      </h2>
      <CodeBlock lang="python">{`arena.explain(
    vector: Sequence[float],
    *,
    top: int = 10,
    where: Filter | None = None,
    max_distance: float | None = None,
    has_text_component: bool = False,
) -> QueryPlan`}</CodeBlock>
      <p>
        Return the planner's decision for a hypothetical query without executing
        it. Useful for debugging filter and index interactions.
      </p>
      <p>
        The returned <code>QueryPlan</code> exposes:
      </p>
      <ul>
        <li>
          <code>strategy</code> — One of <code>ann_index</code>,{" "}
          <code>exact_candidates</code>, <code>full_scan</code>,{" "}
          <code>text_probe</code>, <code>hybrid_fusion</code>.
        </li>
        <li>
          <code>metadata_accelerated</code> — Whether the{" "}
          <code>MetadataIndex</code> was used to narrow candidates.
        </li>
        <li>
          <code>candidate_count</code> — Estimated pre-filter candidate set
          size.
        </li>
      </ul>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2, metadata_acceleration=True)
arena = engine.arena("docs")
for i in range(50):
    arena.write(vector=[float(i), 1.0], meta={"group": i % 5})

f = elips.Filter().field("group").equals(0)
plan = arena.explain([1.0, 0.0], top=10, where=f, has_text_component=False)
print(plan.strategy.name)           # ann_index or exact_candidates
print(plan.metadata_accelerated)    # True
print(plan.candidate_count)         # ≈ 10 (5 * 10 / 5 groups)
engine.close()`}</CodeBlock>

      <h2 id="pull">
        <code>pull()</code> → <code>list[Row]</code>
      </h2>
      <CodeBlock lang="python">{`arena.pull(
    keys: Sequence[str],
    *,
    include_vectors: bool = True,
) -> list[Row]`}</CodeBlock>
      <p>
        Fetch records by key and return typed{" "}
        <Link to="/docs/python/models">
          <code>Row</code>
        </Link>{" "}
        objects. Keys that no longer exist (deleted or never written) are
        silently skipped — the returned list may be shorter than{" "}
        <code>keys</code>.
      </p>
      <p>
        By default <code>include_vectors=True</code> — the stored embedding is
        included in each row. Pass <code>False</code> if you only need metadata
        and text.
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("docs")

k1 = arena.write(text="Alpha", meta={"order": 1})
k2 = arena.write(vector=[0.0, 1.0], meta={"order": 2})

rows = arena.pull([k1, k2])
print(rows[0].text)             # Alpha
print(rows[1].vector)           # (0.0, 1.0)

# Missing key is silently skipped
rows = arena.pull([k1, "nonexistent-key"])
print(len(rows))                # 1

engine.close()`}</CodeBlock>

      <h2 id="sweep">
        <code>sweep()</code> → <code>list[Row]</code>
      </h2>
      <CodeBlock lang="python">{`arena.sweep(
    *,
    where: Filter | None = None,
    offset: int = 0,
    limit: int | None = None,
    include_vectors: bool = False,
) -> list[Row]`}</CodeBlock>
      <p>
        Full scan of the arena, optionally filtered. Returns{" "}
        <code>Row</code> objects in storage order (UUIDv7 insert order
        approximately). Use for export, reindexing, or auditing — not as a
        substitute for <code>probe()</code>.
      </p>
      <ul>
        <li>
          <code>where</code> — Optional filter to apply during the scan.
        </li>
        <li>
          <code>offset</code> / <code>limit</code> — Pagination. <code>limit=None</code>{" "}
          returns all matching records.
        </li>
        <li>
          <code>include_vectors</code> — Hydrate stored vectors. Expensive for
          large arenas.
        </li>
      </ul>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("docs")
for i in range(100):
    arena.write(vector=[float(i), 0.0], meta={"page": i // 10})

# All records
rows = arena.sweep()
print(len(rows))    # 100

# Paginated
page1 = arena.sweep(offset=0, limit=10)
page2 = arena.sweep(offset=10, limit=10)

# Filtered
f = elips.Filter().field("page").equals(0)
rows = arena.sweep(where=f)
print(len(rows))    # 10

engine.close()`}</CodeBlock>

      <h2 id="discard">
        <code>discard()</code> → <code>int</code>
      </h2>
      <CodeBlock lang="python">{`arena.discard(
    keys: Sequence[str] | None = None,
    *,
    where: Filter | None = None,
) -> int`}</CodeBlock>
      <p>
        Delete records by key, metadata filter, or both. Returns the count of
        records actually removed. At least one of <code>keys</code> or{" "}
        <code>where</code> must be provided.
      </p>
      <p>
        Deletes are tombstone operations — the node remains in the graph as a
        routing waypoint until the arena compacts. Live neighbours are
        unaffected; recall is maintained by a widened search beam.
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("docs")
keys = [arena.write(vector=[float(i), 0.0], meta={"group": i % 3}) for i in range(9)]

# Delete by key
removed = arena.discard([keys[0], keys[1]])
print(removed)   # 2

# Delete by filter
f = elips.Filter().field("group").equals(2)
removed = arena.discard(where=f)
print(removed)   # 3  (indices 2, 5, 8)

# Delete by both (union, no double-counting)
removed = arena.discard([keys[3]], where=elips.Filter().field("group").equals(1))
print(removed)   # up to 4 (key 3 + group-1 records 4 and 7)

engine.close()`}</CodeBlock>
      <p>
        <strong>Common mistake:</strong> calling <code>discard()</code> with
        neither <code>keys</code> nor <code>where</code> raises{" "}
        <code>ValueError</code> — there is no "delete all" shorthand. To clear
        an arena, sweep for all keys first.
      </p>

      <h2 id="maintenance">Maintenance</h2>

      <h3>
        <code>arena.freeze(frozen=True)</code> → <code>None</code>
      </h3>
      <p>
        Temporarily refuse writes on this arena without reopening the database
        read-only. Pass <code>frozen=False</code> to re-enable writes.
      </p>
      <CodeBlock lang="python">{`arena.freeze()           # reject subsequent writes
arena.freeze(False)      # allow writes again`}</CodeBlock>

      <h3>
        <code>arena.vacuum()</code> → <code>None</code>
      </h3>
      <p>
        Reclaim graph nodes held by deleted records in this arena. The arena
        auto-compacts once tombstones pass the <code>compaction_ratio</code>{" "}
        (default 0.2). Call this after a bulk delete to free memory without
        waiting.
      </p>
      <CodeBlock lang="python">{`keys = [arena.write(vector=[float(i), 1.0]) for i in range(40)]
arena.discard(keys[:20])
print(arena.pending_removals)   # 20 (0.5 — above threshold, may already be 0)
arena.vacuum()
print(arena.pending_removals)   # 0`}</CodeBlock>

      <h3>
        <code>arena.rebuild()</code> → <code>None</code>
      </h3>
      <p>
        Rebuild the HNSW index from the authoritative record store. Incremental
        inserts in arrival order yield a lower-quality graph than a single build
        over the full set. Call this after a large bulk load to maximise recall
        before opening to search traffic.
      </p>
      <CodeBlock lang="python">{`# Bulk load — insert-order graph quality is suboptimal
arena.write_many(large_batch)

# Rebuild — one pass over all records yields a better graph
arena.rebuild()

# Now compact the on-disk layout too
engine.compact()`}</CodeBlock>

      <h2 id="health">
        <code>health()</code> → <code>ArenaHealth</code>
      </h2>
      <p>
        Return a point-in-time health snapshot. Useful for monitoring and
        capacity planning.
      </p>
      <CodeBlock lang="python">{`engine = elips.connect(":memory:", dimension=2)
arena = engine.arena("docs")
for i in range(10):
    arena.write(vector=[float(i), 1.0])
arena.discard([arena.sweep()[0].key])

health = arena.health()
print(health.name)              # docs
print(health.live)              # 9
print(health.pending_removals)  # 0 or 1 (auto-compacts at ratio)
print(health.tombstone_ratio)   # 0.0 – 0.1
print(health.dimension)         # 2
print(health.metric)            # cosine
print(health.read_only)         # False
print(health.sealed)            # False
engine.close()`}</CodeBlock>
      <p>
        See{" "}
        <Link to="/docs/python/models">
          <code>ArenaHealth</code>
        </Link>{" "}
        for the full model reference.
      </p>

      <h2 id="embedding-resolution">Embedding resolution</h2>
      <p>
        When a record lacks an explicit <code>vector</code>, the arena resolves
        it in this priority order:
      </p>
      <ol>
        <li>
          <strong>Native text embedder</strong> — If{" "}
          <code>engine.config.has_text_embedder</code> is truthy, the core
          embeds via the C++ runtime. <code>Vault.place_document()</code> is
          called; no Python-side embedding occurs.
        </li>
        <li>
          <strong>Python embedder</strong> — If the arena or engine has a
          configured Python callable embedder, texts are batched and passed to
          it. The returned vectors are used to call{" "}
          <code>Vault.place()</code>.
        </li>
        <li>
          <strong>Error</strong> — If neither is present,{" "}
          <code>ValueError</code> is raised. ELIPS never silently degrades to
          storing text without a vector.
        </li>
      </ol>
      <p>
        Records with custom document metadata (a non-empty <code>uri</code> or
        non-<code>text/plain</code> MIME type) always require an explicit
        vector. The native <code>place_document()</code> path accepts only raw
        text, not a full <code>DocumentAttachment</code>.
      </p>

      <h2 id="thread-safety">Thread safety</h2>
      <p>
        <code>Arena</code> itself carries no locks. All thread safety is
        provided by the underlying C++ <code>Vault</code> and{" "}
        <code>Database</code>, which serialize concurrent writes. Concurrent
        reads and writes from multiple threads through the same{" "}
        <code>Arena</code> are safe. If you call a Python embedder from multiple
        threads, ensure the callable is itself thread-safe.
      </p>
    </DocsShell>
  );
}
