// ELIPS tutorial — 30 lessons, three levels.
// Every factual claim traces to docs/ in github.com/axiomchronicles/ellips.
// Prose is tight; diagrams + real code carry the weight.

export type LessonBlock =
  | { kind: "lede"; text: string }
  | { kind: "p"; text: string }
  | { kind: "h"; text: string }
  | { kind: "code"; lang: "python" | "cpp" | "bash" | "text"; filename?: string; code: string }
  | { kind: "diagram"; name: DiagramName; caption: string }
  | { kind: "note"; text: string }
  | { kind: "ul"; items: string[] };

export type DiagramName =
  | "SystemShape"
  | "QueryPath"
  | "Persistence"
  | "ObjectModel"
  | "HnswLayers"
  | "RecoveryFlow"
  | "OnDiskLayout"
  | "GpuEngine"
  | "DynamicBatcher"
  | "TransactionLifecycle"
  | "WhyVectorDB"
  | "AgenticFlow"
  | "ElipsAgenticSystem"
  | "GpuAccelGlance"
  | "AlgorithmPorts"
  | "TutorialRoadmap";

export type Level = "Beginner" | "Intermediate" | "Advanced";

export type Lesson = {
  slug: string;
  number: number;
  level: Level;
  title: string;
  eyebrow: string;
  oneLiner: string;
  blocks: LessonBlock[];
};

// ---------------- BEGINNER ----------------

const beginner: Lesson[] = [
  {
    slug: "01-what-is-elips",
    number: 1,
    level: "Beginner",
    title: "What ELIPS is (and isn't)",
    eyebrow: "Chapter I · orientation",
    oneLiner: "An embedded retrieval engine for vectors and documents. No server, no daemon.",
    blocks: [
      {
        kind: "lede",
        text: "ELIPS lives inside your process. It is to vector search what SQLite is to relational data.",
      },
      {
        kind: "p",
        text: "From docs/overview/introduction.md: ELIPS is designed for CLIs, desktop apps, notebooks, edge workloads, single-process APIs, and tests — anything that wants local, in-process retrieval without operating a separate service.",
      },
      {
        kind: "diagram",
        name: "WhyVectorDB",
        caption:
          "Why a vector DB exists: text → embedding → nearest-neighbor lookup over a billion-row corpus in milliseconds.",
      },
      { kind: "h", text: "What you actually get" },
      {
        kind: "ul",
        items: [
          "ANN and exact vector search over the same vault",
          "First-class documents with chunk coordinates and embedding lineage",
          "Native text-first and hybrid query APIs",
          "Metadata filters with planner-side candidate narrowing",
          "WAL recovery plus segmented persistence",
          "Shared read-only mode for multi-reader serving",
          "Python and C++ SDKs over the same C++23 core",
        ],
      },
      { kind: "h", text: "What ELIPS is not" },
      {
        kind: "ul",
        items: [
          "Not a distributed database. No replication protocol, no sharding service.",
          "Not a SaaS. Nothing to deploy. Your process opens a directory.",
          "Not a vector-only store. Documents, payloads, lineage are first-class.",
        ],
      },
    ],
  },
  {
    slug: "02-installation",
    number: 2,
    level: "Beginner",
    title: "Install ELIPS",
    eyebrow: "Chapter I · orientation",
    oneLiner: "Pip-install the Python wheel, or CMake-build the C++ core.",
    blocks: [
      { kind: "lede", text: "Five minutes. One terminal. No ports, no auth tokens." },
      { kind: "h", text: "Python" },
      { kind: "code", lang: "bash", code: "pip install elips" },
      {
        kind: "p",
        text: "The wheel ships the C++23 core as a compiled pybind11 extension (see docs/python/bindings/pybind11-architecture.md).",
      },
      { kind: "h", text: "C++" },
      {
        kind: "code",
        lang: "bash",
        code: `git clone https://github.com/axiomchronicles/ellips.git
cd ellips && cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j`,
      },
      {
        kind: "ul",
        items: [
          "Python 3.10+",
          "C++23 toolchain: clang 17 / gcc 13 / MSVC 19.38",
          "CMake ≥ 3.24",
          "Optional GPU: CUDA 12 / HIP 6 / Metal / SYCL / Vulkan — selected at build time",
        ],
      },
      { kind: "h", text: "Verify" },
      {
        kind: "code",
        lang: "python",
        code: `import elips
db = elips.open(":memory:", dimension=4, metric="cosine")
print(db.vaults())   # ["default"]`,
      },
      {
        kind: "note",
        text: `":memory:" databases never touch the disk. Perfect for tests and tutorials.`,
      },
    ],
  },
  {
    slug: "03-first-database",
    number: 3,
    level: "Beginner",
    title: "Your first database",
    eyebrow: "Chapter II · first contact",
    oneLiner: "Open a directory, write a vector, read it back, close the process, reopen.",
    blocks: [
      {
        kind: "lede",
        text: "An ELIPS database is a directory. Same way a SQLite database is a file.",
      },
      {
        kind: "diagram",
        name: "OnDiskLayout",
        caption:
          "What ELIPS writes on disk. Each file is independently meaningful and atomically replaced.",
      },
      {
        kind: "code",
        lang: "python",
        filename: "first_db.py",
        code: `import elips

db = elips.open("/tmp/my_first_elips", dimension=4, metric="cosine")
notes = db.vault("notes")

rid = notes.place([0.1, 0.2, 0.3, 0.4], {"tag": "hello"})
print(rid)

hit = notes.seek([0.1, 0.2, 0.3, 0.4], top=1)[0]
print(hit.id, hit.distance, hit.payload)`,
      },
      {
        kind: "p",
        text: "Kill the process and run it again. The record is still there — ELIPS replayed the WAL on open.",
      },
    ],
  },
  {
    slug: "04-records-vectors-payloads",
    number: 4,
    level: "Beginner",
    title: "Records, vectors, payloads",
    eyebrow: "Chapter II · first contact",
    oneLiner: "The Record is the atom. Everything else hangs off it.",
    blocks: [
      {
        kind: "lede",
        text: "A Record is id + vector + optional payload + optional document attachment + optional lineage.",
      },
      {
        kind: "diagram",
        name: "ObjectModel",
        caption: "Instance → vaults → records. Records are the only objects the index sees.",
      },
      {
        kind: "code",
        lang: "python",
        code: `notes.place(
    vector=[0.1] * 128,
    payload={"author": "ann", "kind": "design"},
)`,
      },
      {
        kind: "ul",
        items: [
          "vector — fixed dimension per database; set at open(), immutable after",
          "payload — arbitrary dict-shaped metadata, JSON-compatible values",
          "id — UUIDv7 RecordID, auto-generated unless you pass one",
          "document, chunk, lineage — optional, populated by place_document()",
        ],
      },
      {
        kind: "note",
        text: "Payloads are read back with every hit. Don't put megabytes there; that's what document attachments are for.",
      },
    ],
  },
  {
    slug: "05-place-and-erase",
    number: 5,
    level: "Beginner",
    title: "place and erase",
    eyebrow: "Chapter II · first contact",
    oneLiner: "Two mutation verbs. Both go through the WAL before touching memory.",
    blocks: [
      { kind: "lede", text: "Two verbs. Everything else is a query." },
      {
        kind: "diagram",
        name: "Persistence",
        caption: "place / erase append to the WAL first, then mutate the in-memory store.",
      },
      {
        kind: "code",
        lang: "python",
        code: `rid = notes.place([0.1, 0.2, 0.3, 0.4], {"tag": "a"})
notes.erase(rid)   # marks tombstone; checkpoint/compact reclaims space`,
      },
      {
        kind: "p",
        text: "From docs/storage.md: the WAL supports insert, erase, and insert_ex (for records carrying DocumentAttachment, ChunkInfo, or EmbeddingLineage). Crash recovery restores full record state, not only vectors.",
      },
      { kind: "note", text: "Updates are place(id=existing, …). Same id, new vector or payload." },
    ],
  },
  {
    slug: "06-seek-basics",
    number: 6,
    level: "Beginner",
    title: "seek — nearest neighbors",
    eyebrow: "Chapter III · retrieval",
    oneLiner: "Top-k nearest vectors, optionally with a distance threshold.",
    blocks: [
      { kind: "lede", text: "seek() is the bread-and-butter call. Vector in, ranked hits out." },
      {
        kind: "diagram",
        name: "QueryPath",
        caption: "App → vault → planner → metadata → index → ranked hits.",
      },
      {
        kind: "code",
        lang: "python",
        code: `hits = notes.seek(
    query=[0.1, 0.2, 0.3, 0.4],
    top=5,
    threshold=0.3,
)
for h in hits:
    print(h.id, h.distance, h.payload)`,
      },
      {
        kind: "p",
        text: "From docs/cpp/api-reference/vault.md: SearchResult carries id, distance, data, document, chunk, lineage — all hydrated from the authoritative record store.",
      },
    ],
  },
  {
    slug: "07-filters",
    number: 7,
    level: "Beginner",
    title: "Filters and payload search",
    eyebrow: "Chapter III · retrieval",
    oneLiner: "Narrow candidates with equality, ranges, IN, contains, AND/OR/NOT.",
    blocks: [
      {
        kind: "lede",
        text: "If the filter is selective, the planner short-circuits through MetadataIndex before ever touching the vector index.",
      },
      {
        kind: "code",
        lang: "python",
        code: `hits = docs.seek(
    query=embed("alpha"),
    top=10,
    filter={
        "kind": "design",
        "year": {"$gte": 2023},
        "tags": {"$in": ["draft", "review"]},
    },
)`,
      },
      { kind: "p", text: "EQL exposes the same filter grammar — see lesson 11." },
    ],
  },
  {
    slug: "08-documents-and-chunks",
    number: 8,
    level: "Beginner",
    title: "Documents, chunks, lineage",
    eyebrow: "Chapter III · retrieval",
    oneLiner: "Text-first ingestion with full lineage that survives restarts.",
    blocks: [
      {
        kind: "lede",
        text: "place_document() embeds, chunks, and stores in one call — and records exactly which model produced each vector.",
      },
      {
        kind: "code",
        lang: "python",
        code: `docs = db.vault("documents")
docs.place_document(
    "Long markdown article ...",
    {"kind": "design"},
)

for hit in docs.seek_text("alpha", top=3):
    print(hit.document.text[:80])
    print(hit.chunk.start, hit.chunk.end)
    print(hit.lineage.provider, hit.lineage.model, hit.lineage.revision)`,
      },
      {
        kind: "p",
        text: "Lineage is what lets you re-embed only the records produced by a stale model. It's also what powers exact citations.",
      },
    ],
  },
  {
    slug: "09-hybrid-search",
    number: 9,
    level: "Beginner",
    title: "Hybrid search",
    eyebrow: "Chapter III · retrieval",
    oneLiner: "Fuse vector distance with lexical overlap from attached documents.",
    blocks: [
      {
        kind: "lede",
        text: "Pure ANN misses exact keywords. Pure lexical misses paraphrases. Fuse both.",
      },
      {
        kind: "code",
        lang: "python",
        code: `hits = docs.seek_hybrid(
    query=embed("alpha design note"),
    text="alpha design note",
    top=5,
    lexical_weight=0.25,   # default per docs/cpp/api-reference/vault.md
)`,
      },
      {
        kind: "p",
        text: "lexical_weight ∈ [0, 1]. 0 means pure vector; 1 means pure lexical. The fused score is on each SearchResult.",
      },
    ],
  },
  {
    slug: "10-cli-tour",
    number: 10,
    level: "Beginner",
    title: "Command line tour",
    eyebrow: "Chapter IV · operations",
    oneLiner: "Inspect, checkpoint, verify, import, export — without writing any code.",
    blocks: [
      {
        kind: "lede",
        text: "The elips binary is your maintenance toolbox. Built from src/ alongside the library.",
      },
      {
        kind: "code",
        lang: "bash",
        code: `elips info /my_db
elips vaults /my_db
elips stats /my_db
elips verify /my_db                # replays WAL, prints OK / CORRUPT
elips checkpoint /my_db
elips export /my_db --vault docs --output docs.jsonl
elips import /my_db --vault docs --input docs.jsonl --dimension 768
elips bench /tmp/bench_db --count 100000 --dim 768`,
      },
      {
        kind: "p",
        text: "Output of elips query is one JSON object per result row — pipe it through jq, store it in files, diff between runs.",
      },
    ],
  },
];

// ---------------- INTERMEDIATE ----------------

const intermediate: Lesson[] = [
  {
    slug: "11-eql-deep-dive",
    number: 11,
    level: "Intermediate",
    title: "EQL — the ELIPS Query Language",
    eyebrow: "Chapter V · the query layer",
    oneLiner: "A small expression-oriented language with a real lexer, parser, AST, and executor.",
    blocks: [
      {
        kind: "lede",
        text: "Reach for EQL when filter dicts stop being expressive enough. It compiles to the same QueryPlan the SDKs use.",
      },
      { kind: "h", text: "Statement grammar" },
      {
        kind: "code",
        lang: "text",
        code: `seek in <vault>
    nearest <[literal] | $binding>
    [top <int>]
    [threshold <number>]
    [where <filter>]
    [rank_by <distance | field>]
    [project <* | f1, f2, ...>]
    yield

fetch from <vault> id "<uuid>" yield
scan in <vault> [where ...] [offset N] [limit N] yield
place in <vault> vector [...] [data {...}]
erase from <vault> id "<uuid>"`,
      },
      { kind: "h", text: "Filter grammar" },
      {
        kind: "code",
        lang: "text",
        code: `filter   := term (("and" | "or") term)* | "not" filter | "(" filter ")"
term     := field <comparator> value
          | field "in" [ value, ... ]
          | field "contains" "substring"
comparator := = | != | < | <= | > | >=`,
      },
      { kind: "h", text: "Run from Python" },
      {
        kind: "code",
        lang: "python",
        code: `db.query("""
  seek in articles
    nearest $embedding
    top 10
    where category = "technology" and published_year >= 2023
    project title, author
    yield
""", bindings={"embedding": embed(query)})`,
      },
      { kind: "h", text: "Run from the CLI" },
      {
        kind: "code",
        lang: "bash",
        code: `elips query /my_db --eql 'seek in docs nearest [0.1,0.2,0.3] top 5 yield'
elips query /my_db --file ranking.eql`,
      },
    ],
  },
  {
    slug: "12-transactions",
    number: 12,
    level: "Intermediate",
    title: "Transactions",
    eyebrow: "Chapter V · the query layer",
    oneLiner: "Buffered atomic batches. Validation is eager; commit cannot fail mid-batch.",
    blocks: [
      {
        kind: "lede",
        text: "ELIPS validates every operation as you enqueue it, so commit() is a pure write.",
      },
      {
        kind: "diagram",
        name: "TransactionLifecycle",
        caption:
          "Three terminal paths: commit, rollback, or destructor-rollback. done_ flips once.",
      },
      {
        kind: "p",
        text: "From docs/cpp/api-reference/transaction.md: TransactionVault::place() validates dimension and finiteness at enqueue, throwing DimensionMismatch or InvalidVector. If enqueue succeeds for every op, commit() cannot fail validation.",
      },
      {
        kind: "code",
        lang: "python",
        code: `with db.begin_transaction() as txn:
    v = txn.vault("notes")
    v.place(vector1, {"tag": "a"})
    v.place(vector2, {"tag": "b"})
    # clean exit -> commit
    # exception  -> rollback (destructor)`,
      },
      {
        kind: "code",
        lang: "cpp",
        code: `auto txn = db->begin_transaction();
auto& tv = txn.vault("notes");
tv.place(vector1, {{"tag", std::string{"a"}}});
tv.place(vector2, {{"tag", std::string{"b"}}});
txn.commit();   // or txn.rollback();`,
      },
      {
        kind: "note",
        text: "Unlike Vault::place(), TransactionVault::place() does not normalize the vector at enqueue — normalization happens inside the per-op place() during commit.",
      },
    ],
  },
  {
    slug: "13-configuration",
    number: 13,
    level: "Intermediate",
    title: "Configuration & durability",
    eyebrow: "Chapter VI · persistence",
    oneLiner: "dimension, metric, index, durability — set once, recorded forever in IDENTITY.",
    blocks: [
      {
        kind: "lede",
        text: "IDENTITY is the durable source of truth. Reopening with a conflicting dimension is a typed error, not a footgun.",
      },
      { kind: "h", text: "The four core enums (docs/cpp/api-reference/config.md)" },
      {
        kind: "code",
        lang: "cpp",
        code: `enum class Metric      { cosine, euclidean, dot_product };
enum class IndexType   { graph, exact };
enum class Durability  { paranoid, standard, relaxed, ephemeral };
enum class AccessMode  { read_write, read_only };`,
      },
      { kind: "h", text: "Durability tradeoffs (docs/storage.md)" },
      {
        kind: "ul",
        items: [
          "paranoid / standard — flush each write to disk before returning",
          "relaxed — buffer in the WAL until checkpoint or close",
          "ephemeral — no WAL attachment at all (in-process only)",
        ],
      },
      { kind: "h", text: "Full C++ builder" },
      {
        kind: "code",
        lang: "cpp",
        code: `elips::Config{}
    .dimension(768)
    .metric(elips::Metric::cosine)
    .index(elips::IndexType::graph)
    .graph_params({.max_connections = 32, .ef_construction = 400, .ef_search = 100})
    .durability(elips::Durability::standard)
    .access_mode(elips::AccessMode::read_write)
    .segmented_storage(true)
    .metadata_acceleration(true)
    .auto_text_embedder(true);`,
      },
    ],
  },
  {
    slug: "14-storage-layout",
    number: 14,
    level: "Intermediate",
    title: "On-disk layout, from the outside",
    eyebrow: "Chapter VI · persistence",
    oneLiner: "Eight files, two modes, one lock.",
    blocks: [
      {
        kind: "lede",
        text: "Open any ELIPS directory and you can tell what state it's in without running a single command.",
      },
      {
        kind: "diagram",
        name: "OnDiskLayout",
        caption:
          "Snapshot mode vs. segmented mode share LOCK, IDENTITY, wal.log, and the optional TEXT_EMBEDDER manifest.",
      },
      {
        kind: "code",
        lang: "text",
        code: `/my_db/
├── LOCK                              # advisory file lock (flock)
├── IDENTITY                          # dimension, metric, index type
├── TEXT_EMBEDDER.manifest            # embedder identity (provider/model/revision)
├── wal.log                           # append-only write-ahead log
├── elips.manifest                    # segmented mode: per-vault segment list
├── text_embedder/
│   └── default_v1_<dim>.localembed   # built-in local embedder artifact
├── segments/
│   └── vault_<n>_<epoch>.segment     # one fresh segment per vault per checkpoint
└── elips.snapshot                    # snapshot mode (older / opt-in)`,
      },
      {
        kind: "p",
        text: "From docs/storage.md: segmented mode is the default. Snapshot mode is preserved for compatibility and atomic rename (elips.snapshot.tmp → elips.snapshot).",
      },
    ],
  },
  {
    slug: "15-checkpoint-compact",
    number: 15,
    level: "Intermediate",
    title: "checkpoint() and compact()",
    eyebrow: "Chapter VI · persistence",
    oneLiner: "checkpoint flushes; compact rebuilds.",
    blocks: [
      {
        kind: "lede",
        text: "These are the two maintenance verbs. Schedule them; don't pray for them.",
      },
      {
        kind: "ul",
        items: [
          "checkpoint() — writes the current logical state to disk and truncates the WAL",
          "compact() — rebuilds each vault index from the record store, then checkpoints",
        ],
      },
      {
        kind: "p",
        text: "Segmented mode writes one fresh segment per vault, rewrites elips.manifest, and removes obsolete segment files. Snapshot mode writes elips.snapshot.tmp, atomic-renames to elips.snapshot, removes segmented artifacts.",
      },
      {
        kind: "code",
        lang: "python",
        code: `db.checkpoint()   # flush + truncate WAL
db.compact()      # rebuild indexes + checkpoint`,
      },
      {
        kind: "note",
        text: "Run checkpoint() from a job runner on a cadence that matches your durability target. Run compact() after large delete bursts.",
      },
    ],
  },
  {
    slug: "16-read-only-replicas",
    number: 16,
    level: "Intermediate",
    title: "Read-only mode & shared readers",
    eyebrow: "Chapter VII · concurrency",
    oneLiner: "One writer, many readers, all coordinated through a single advisory lock.",
    blocks: [
      {
        kind: "lede",
        text: "Open the same database as read_only from any number of processes. They take a shared flock; mutations are rejected at the API boundary.",
      },
      {
        kind: "code",
        lang: "python",
        code: `replica = elips.open("/var/data/elips", read_only=True)
for hit in replica.vault("docs").seek(q, top=10):
    ...`,
      },
      {
        kind: "ul",
        items: [
          "One read_write opener via exclusive lock",
          "Many read_only openers via shared lock",
          "Read-only vaults reject place/erase/checkpoint with StorageError",
        ],
      },
      {
        kind: "p",
        text: "This is the supported mode for shared-reader analytics and read-fanout serving. Writers run elsewhere; replicas pick up new segments on the next open or rotation.",
      },
    ],
  },
  {
    slug: "17-embedders",
    number: 17,
    level: "Intermediate",
    title: "Embedders — built-in, custom, rehydratable",
    eyebrow: "Chapter VIII · text",
    oneLiner: "ELIPS ships a local embedder. Bring your own when you need to.",
    blocks: [
      {
        kind: "lede",
        text: "An embedder is whatever implements TextEmbedderPort. ELIPS records its identity in TEXT_EMBEDDER.manifest so reopens are deterministic.",
      },
      { kind: "h", text: "Use the built-in local embedder" },
      {
        kind: "code",
        lang: "python",
        code: `db = elips.open("/data/db", dimension=128, auto_text_embedder=True)`,
      },
      { kind: "h", text: "Custom embedder" },
      {
        kind: "code",
        lang: "python",
        code: `class MyEmbedder(elips.TextEmbedder):
    dimension = 768
    name = "my-embedder/v1"
    revision = "2026-06"

    def embed(self, texts):
        return model.encode(texts)

db = elips.open("/data/db", dimension=768, embedder=MyEmbedder())`,
      },
      {
        kind: "p",
        text: "From docs/cpp/api-reference/config.md: TextEmbedderPort exposes provider_name, model_name, revision_name, backend_name, output_dimension. ELIPS uses these to fingerprint the manifest.",
      },
      {
        kind: "note",
        text: "Non-rehydratable external embedders store metadata only. Reopening without the same embedder fails seek_text() with an actionable ConfigError.",
      },
    ],
  },
  {
    slug: "18-planner-explain",
    number: 18,
    level: "Intermediate",
    title: "The planner and explain_seek()",
    eyebrow: "Chapter IX · observability",
    oneLiner: "Ask the engine what it's about to do before it does it.",
    blocks: [
      { kind: "lede", text: "explain_seek() is your single best window into runtime behavior." },
      {
        kind: "code",
        lang: "cpp",
        code: `auto plan = vault.explain_seek(query, /*top=*/10, filter);
// plan.strategy           — QueryStrategy enum
// plan.candidate_count    — narrowed candidate set size
// plan.metadata_accelerated
// plan.gpu_index
// plan.index_type_name`,
      },
      {
        kind: "ul",
        items: [
          "Selective filter + accelerated metadata → candidate_count drops by orders of magnitude",
          "GPU built in → gpu_index = true; the dispatch goes through GpuPort",
          "No accelerator, exact index → planner picks brute-force; expect linear-in-N latency",
        ],
      },
      {
        kind: "p",
        text: "Treat strategy + candidate_count as the two telemetry numbers you actually emit per query.",
      },
    ],
  },
  {
    slug: "19-error-model",
    number: 19,
    level: "Intermediate",
    title: "Error model",
    eyebrow: "Chapter IX · observability",
    oneLiner: "Six typed C++ exceptions with a parallel Python hierarchy.",
    blocks: [
      { kind: "lede", text: "Errors are typed. Catch them with intent." },
      {
        kind: "code",
        lang: "text",
        code: `std::runtime_error
└── elips::ElipsError
    ├── ConfigError          // identity / dimension / embedder mismatch
    ├── StorageError         // WAL / segments / lock / read-only writes
    ├── DimensionMismatch    // wrong-shape vector
    ├── InvalidVector        // NaN / Inf
    ├── NotFound             // unknown id, missing vault
    └── EmbedderError        // text embedding failure`,
      },
      {
        kind: "code",
        lang: "python",
        code: `try:
    vault.place([float("nan")] * 4)
except elips.InvalidVector as e:
    log.warning("rejected: %s", e)`,
      },
    ],
  },
  {
    slug: "20-python-bindings",
    number: 20,
    level: "Intermediate",
    title: "How the Python bindings work",
    eyebrow: "Chapter X · interop",
    oneLiner: "pybind11 plus a thin Pythonic veneer. Same engine, two surfaces.",
    blocks: [
      {
        kind: "lede",
        text: "Python doesn't reimplement anything. It binds the same ElipsInstance and Vault that C++ users hold.",
      },
      {
        kind: "p",
        text: "From docs/python/bindings/pybind11-architecture.md: there is a low-level surface (elips.open, Database, Vault — direct binding mirrors) and a modern surface (connect(), Engine, Arena — opinionated Pythonic wrappers).",
      },
      {
        kind: "code",
        lang: "python",
        code: `# low-level — mirrors C++ shapes
db = elips.open("/data/db", dimension=384)
docs = db.vault("documents")

# modern — opinionated
eng = elips.connect("/data/db", dimension=384)
arena = eng.arena("documents")`,
      },
      {
        kind: "ul",
        items: [
          "Vectors cross the boundary as zero-copy float buffers when possible",
          "Payload dicts are converted via pybind11's std::variant binding",
          "Errors are translated to elips.ConfigError / StorageError / … on the Python side",
        ],
      },
    ],
  },
];

// ---------------- ADVANCED ----------------

const advanced: Lesson[] = [
  {
    slug: "21-architecture-overview",
    number: 21,
    level: "Advanced",
    title: "Architecture — the whole stack",
    eyebrow: "Chapter XI · internals",
    oneLiner: "SDKs, instance, vaults, ports, engines, storage. Five layers.",
    blocks: [
      {
        kind: "lede",
        text: "Every ELIPS feature lives on exactly one of five layers. Knowing which one is the difference between a one-line patch and a refactor.",
      },
      {
        kind: "diagram",
        name: "SystemShape",
        caption:
          "SDKs → ElipsInstance → Vault → planner + ports (index/metadata/text/GPU) → storage (WAL + segments).",
      },
      {
        kind: "p",
        text: "From docs/architecture/system-overview.md: each layer talks down to the next via abstract ports. Nothing in the planner depends on a concrete index. Nothing in the index depends on a concrete storage layout.",
      },
      {
        kind: "diagram",
        name: "AlgorithmPorts",
        caption:
          "Ports & adapters: the planner sees IndexPort, MetadataIndex, TextEmbedderPort, GpuPort — not concrete classes.",
      },
    ],
  },
  {
    slug: "22-storage-engine-internals",
    number: 22,
    level: "Advanced",
    title: "Storage engine internals",
    eyebrow: "Chapter XI · internals",
    oneLiner: "WAL frame format, segment files, manifest, atomic swap.",
    blocks: [
      { kind: "lede", text: "Every byte in /my_db has a job. This is what it does." },
      { kind: "h", text: "WAL frames (docs/storage.md, include/elips/storage/WAL.hpp)" },
      {
        kind: "ul",
        items: [
          "insert        — id + vector + payload",
          "erase         — id only (tombstone)",
          "insert_ex     — id + vector + payload + DocumentAttachment + ChunkInfo + EmbeddingLineage",
        ],
      },
      {
        kind: "p",
        text: "Every frame is length-prefixed and hashed. Recovery replays the longest valid prefix and truncates a corrupt tail without aborting.",
      },
      { kind: "h", text: "Segments" },
      {
        kind: "p",
        text: "checkpoint() writes one segment per vault: vault_<n>_<epoch>.segment. The new elips.manifest references the new epoch. Old segment files are unlinked only after the manifest has been atomically replaced.",
      },
      { kind: "h", text: "Snapshot mode (legacy / opt-in)" },
      {
        kind: "p",
        text: "Writes to elips.snapshot.tmp, fsync, rename to elips.snapshot. Single-file, atomic, simpler — slower for large databases.",
      },
    ],
  },
  {
    slug: "23-recovery-flow",
    number: 23,
    level: "Advanced",
    title: "Open & recovery flow",
    eyebrow: "Chapter XI · internals",
    oneLiner: "Six steps from open() to a queryable instance.",
    blocks: [
      {
        kind: "lede",
        text: "open() is the most important code path in any storage engine. ELIPS' is six steps long.",
      },
      {
        kind: "diagram",
        name: "RecoveryFlow",
        caption:
          "lock → identity → embedder manifest → segments-or-snapshot → WAL replay → attach live WAL (skipped for read-only).",
      },
      {
        kind: "ul",
        items: [
          "1. Acquire advisory lock (exclusive for RW, shared for RO)",
          "2. Read IDENTITY (dimension, metric, index)",
          "3. Resolve TEXT_EMBEDDER.manifest, rehydrate built-in embedder if applicable",
          "4. Load elips.manifest + segments if present, else elips.snapshot",
          "5. Replay the valid WAL prefix; truncate at first invalid frame",
          "6. Attach live WAL unless read-only or ephemeral",
        ],
      },
      {
        kind: "p",
        text: "Corrupt WAL tails are tolerated by design — the database is the longest valid prefix that hashed cleanly.",
      },
    ],
  },
  {
    slug: "24-hnsw",
    number: 24,
    level: "Advanced",
    title: "HierarchicalGraphIndex — HNSW in ELIPS",
    eyebrow: "Chapter XII · algorithms",
    oneLiner:
      "Multi-layer skip-list-shaped proximity graph. Log-time approximate nearest neighbors.",
    blocks: [
      {
        kind: "lede",
        text: "ANN is mostly graph traversal in disguise. HNSW makes the disguise efficient.",
      },
      {
        kind: "diagram",
        name: "HnswLayers",
        caption:
          "Top layer: long-range hubs. Bottom layer: every point. Search descends one layer at a time.",
      },
      { kind: "h", text: "Parameters that matter (docs/cpp/api-reference/config.md)" },
      {
        kind: "ul",
        items: [
          "max_connections (M) — outgoing edges per node per layer; default 16",
          "ef_construction — candidate-list size when inserting; default 200",
          "ef_search — candidate-list size at query time; default 50",
        ],
      },
      {
        kind: "p",
        text: "Increase ef_search to trade latency for recall at query time without rebuilding. Increase ef_construction or max_connections only at build time — and only if your recall plateau is below target.",
      },
    ],
  },
  {
    slug: "25-lock-manager",
    number: 25,
    level: "Advanced",
    title: "LockManager internals",
    eyebrow: "Chapter XIII · concurrency",
    oneLiner: "A small, deterministic, deadlock-free hierarchical lock manager.",
    blocks: [
      {
        kind: "lede",
        text: "Concurrency control is in-process. ELIPS does not coordinate across processes for writes — that is what the file lock is for.",
      },
      {
        kind: "p",
        text: "From docs/internals/lock-manager.md: locks are acquired in a fixed hierarchical order (database → vault → record). Acquiring out of order is a programmer error and is asserted in debug builds.",
      },
      {
        kind: "ul",
        items: [
          "Shared (read) and exclusive (write) modes per resource",
          "Deadlock prevention via fixed acquisition order — no cycle detector needed",
          "Lock-free read paths through the index when the writer is not active",
        ],
      },
    ],
  },
  {
    slug: "26-transaction-engine",
    number: 26,
    level: "Advanced",
    title: "Transaction engine internals",
    eyebrow: "Chapter XIII · concurrency",
    oneLiner: "Eager validation, deferred apply, single done_ flag, destructor as the safety net.",
    blocks: [
      {
        kind: "lede",
        text: "The transaction is the simplest object in the engine. That is by design.",
      },
      {
        kind: "diagram",
        name: "TransactionLifecycle",
        caption:
          "Begin → enqueue (validated) → commit OR rollback OR destructor-rollback. done_ flips exactly once.",
      },
      {
        kind: "p",
        text: "From docs/internals/transaction-engine.md: TransactionVault holds a non-owning pointer to its parent Transaction. Validation runs at enqueue time. commit() walks the op list under the database write lock and applies via per-vault place() / erase(). The destructor rolls back if neither commit nor rollback was called.",
      },
      {
        kind: "code",
        lang: "cpp",
        code: `class Transaction {
public:
    TransactionVault vault(std::string name);
    void commit();
    void rollback();
    ~Transaction();   // rollback() if !done_
private:
    bool done_{false};
    std::vector<Op> ops_;
};`,
      },
    ],
  },
  {
    slug: "27-query-executor",
    number: 27,
    level: "Advanced",
    title: "EQL — lexer, parser, AST, executor",
    eyebrow: "Chapter XIV · the query engine",
    oneLiner: "Four small files in src/. Hand-written recursive descent.",
    blocks: [
      {
        kind: "lede",
        text: "EQL is not a toy. It is a real, four-stage compiler embedded in the engine.",
      },
      {
        kind: "ul",
        items: [
          "EQLLexer.cpp — char stream → tokens",
          "EQLParser.cpp — tokens → AST (include/elips/query_engine/AST.hpp)",
          "QueryExecutor.cpp — AST → QueryPlan → Vault calls",
          "MetadataIndex collapses equality and IN-list terms into candidate sets",
        ],
      },
      {
        kind: "p",
        text: "Bindings ($name) are typed at runtime — the executor checks that a $vector binding is the right dimension before the plan ever reaches a Vault.",
      },
    ],
  },
  {
    slug: "28-gpu-engine",
    number: 28,
    level: "Advanced",
    title: "GPU engine architecture",
    eyebrow: "Chapter XV · GPU",
    oneLiner: "Ports & adapters at the device boundary. Five backends, one selector.",
    blocks: [
      {
        kind: "lede",
        text: "GPU support is optional, compile-time-selectable, and entirely behind a port. Nothing in the planner mentions CUDA.",
      },
      {
        kind: "diagram",
        name: "GpuEngine",
        caption:
          "GpuPort + GpuMemoryPort + GpuKernelPort + GpuStreamPort + GpuIndexPort. Backends fulfill the contract.",
      },
      { kind: "h", text: "The 13-method GpuPort (include/elips/gpu_engine/GpuPort.hpp)" },
      {
        kind: "code",
        lang: "cpp",
        code: `class GpuPort {
public:
    [[nodiscard]] virtual std::expected<void, GpuError>
        initialize(const GpuConfig&) = 0;
    virtual void shutdown() noexcept = 0;

    [[nodiscard]] virtual GpuDeviceInfo device_info() const noexcept = 0;
    [[nodiscard]] virtual bool is_available() const noexcept = 0;

    [[nodiscard]] virtual std::expected<GpuBuffer, GpuError>
        allocate_device(size_t bytes) = 0;
    virtual void free_device(GpuBuffer&&) noexcept = 0;

    [[nodiscard]] virtual std::expected<void, GpuError>
        upload(const void*, GpuBuffer&, size_t) = 0;
    [[nodiscard]] virtual std::expected<void, GpuError>
        download(const GpuBuffer&, void*, size_t) = 0;

    [[nodiscard]] virtual std::expected<void, GpuError>
        compute_distances_batch(
            std::span<const float> queries, std::span<const float> db,
            std::span<float> out, size_t nq, size_t nb, size_t dim,
            elips::Metric) = 0;

    [[nodiscard]] virtual std::expected<void, GpuError>
        top_k(std::span<const float> distances,
              std::span<uint32_t> idx, std::span<float> val,
              size_t nq, size_t nb, size_t k) = 0;

    virtual void synchronize() = 0;
    [[nodiscard]] virtual bool is_idle() const noexcept = 0;
};`,
      },
      {
        kind: "p",
        text: "All operations return std::expected<T, GpuError>. The error enum: DeviceNotFound, InsufficientMemory, KernelLaunchFailed, TransferFailed, IndexBuildFailed, UnsupportedMetric, InitializationFailed, BackendUnavailable.",
      },
    ],
  },
  {
    slug: "29-gpu-backends-selection",
    number: 29,
    level: "Advanced",
    title: "Backends, device probe, selection",
    eyebrow: "Chapter XV · GPU",
    oneLiner: "Metal, CUDA, HIP, SYCL, Vulkan — each behind a compile-time flag.",
    blocks: [
      {
        kind: "lede",
        text: "GpuDeviceManager probes whatever was compiled in. GpuSelector picks the best device that matches policy.",
      },
      {
        kind: "p",
        text: "Probe order (docs/internals/gpu-engine.md): Metal (Apple), CUDA, HIP, SYCL, Vulkan. Each probe init+inspect+shutdowns the backend inside a try/catch. Devices are then sorted by CAGRA support, then by total device memory descending.",
      },
      {
        kind: "ul",
        items: [
          "GpuPolicy::CpuOnly — selector returns nullopt; never touches GPU",
          "GpuPolicy::PreferGpu — uses best probed device or falls back to CPU",
          "GpuPolicy::RequireGpu — returns nullopt (fatal) if no enabled backend matches",
        ],
      },
      {
        kind: "diagram",
        name: "GpuAccelGlance",
        caption: "Probe → sort → select → initialize. Falls back per GpuPolicy.",
      },
    ],
  },
  {
    slug: "30-dynamic-batcher-pipelines",
    number: 30,
    level: "Advanced",
    title: "DynamicBatcher and the GPU pipelines",
    eyebrow: "Chapter XV · GPU",
    oneLiner: "Coalesce concurrent queries into one launch; ingest and quantize on the device.",
    blocks: [
      {
        kind: "lede",
        text: "GPUs hate small launches. The DynamicBatcher exists to make them stop being small.",
      },
      {
        kind: "diagram",
        name: "DynamicBatcher",
        caption:
          "Concurrent queries wait inside a tiny window, then ride one kernel launch as a batch.",
      },
      { kind: "h", text: "Configurable knobs (include/elips/gpu_engine/GpuConfig.hpp)" },
      {
        kind: "ul",
        items: [
          "window_us — how long to wait for stragglers (microseconds)",
          "max_batch — hard cap on coalesced queries per launch",
          "algorithm — brute-force, IVF-Flat, IVF-PQ, graph (CAGRA), hybrid",
        ],
      },
      {
        kind: "p",
        text: "GpuIngestionPipeline streams record vectors onto the device. GpuQuantizationPipeline trains IVF-PQ centroids and pre-encodes residuals. GpuSearchPipeline orchestrates the batched distance + top-k path. GpuProfiler instruments every stage.",
      },
      {
        kind: "code",
        lang: "cpp",
        code: `auto cfg = elips::Config{}
    .dimension(768)
    .metric(elips::Metric::cosine)
    .gpu(elips::gpu::GpuConfig{}
        .policy(elips::gpu::GpuPolicy::PreferGpu)
        .algorithm(elips::gpu::Algorithm::cagra)
        .window_us(500)
        .max_batch(64));`,
      },
    ],
  },
  {
    slug: "31-performance",
    number: 31,
    level: "Advanced",
    title: "Performance — what to measure, what to tune",
    eyebrow: "Chapter XVI · production",
    oneLiner: "Three knobs cover 90% of recall-vs-latency tradeoffs.",
    blocks: [
      { kind: "lede", text: "Tune what you measure. Measure what you tune." },
      {
        kind: "ul",
        items: [
          "Query latency — p50 / p95 / p99 per vault, per index type",
          "Recall@k — measured against an exact-search ground truth on a sampled subset",
          "Candidate-set size — emit from explain_seek(); spikes here predict tail latency",
          "WAL flush time — paranoid/standard durability shifts cost into commit",
        ],
      },
      { kind: "h", text: "The three knobs" },
      {
        kind: "ul",
        items: [
          "ef_search — query-time recall vs. latency (no rebuild)",
          "ef_construction — build-time recall ceiling (rebuild required)",
          "max_connections — graph density vs. memory (rebuild required)",
        ],
      },
      {
        kind: "p",
        text: "The elips bench command builds a synthetic database at a given dimension and count, then runs warm-cache queries — use it as a sanity floor when comparing hardware or builds.",
      },
    ],
  },
  {
    slug: "32-production-checklist",
    number: 32,
    level: "Advanced",
    title: "Going to production",
    eyebrow: "Chapter XVI · production",
    oneLiner: "Checkpoint cadence, replicas, telemetry, backups. The four boxes you must tick.",
    blocks: [
      {
        kind: "lede",
        text: "ELIPS is embedded — production is your binary, on a server, with disks you trust.",
      },
      {
        kind: "diagram",
        name: "ElipsAgenticSystem",
        caption: "What the layered stack looks like when an agent runtime owns its own retrieval.",
      },
      {
        kind: "ul",
        items: [
          "Durability: paranoid on durable storage, fsync_on_checkpoint on ephemeral disks",
          "Replicas: open with read_only=True from N readers; shared flock coordinates",
          "Checkpoint: schedule elips checkpoint via cron/systemd-timer; tune cadence to recovery time",
          "Backups: snapshot the directory while it is closed, or while only read-only openers hold it",
          "Telemetry: emit explain_seek().strategy and candidate_count per query",
          "Compaction: run elips compact after large delete bursts to reclaim tombstones",
        ],
      },
      {
        kind: "p",
        text: "You've now seen every concept in the engine. Each one has its own page in /docs and its own chapter here. The roadmap (docs/roadmap.md) is what is coming next.",
      },
    ],
  },
];

export const lessons: Lesson[] = [...beginner, ...intermediate, ...advanced];

export function lessonBySlug(slug: string): Lesson | undefined {
  return lessons.find((l) => l.slug === slug);
}

export function neighbors(slug: string) {
  const i = lessons.findIndex((l) => l.slug === slug);
  return {
    prev: i > 0 ? lessons[i - 1] : undefined,
    next: i >= 0 && i < lessons.length - 1 ? lessons[i + 1] : undefined,
  };
}

export function lessonsByLevel(level: Level) {
  return lessons.filter((l) => l.level === level);
}
