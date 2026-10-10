// ELIPS tutorial — 30 lessons, three levels.
// Every factual claim traces to docs/ in github.com/axiomchronicles/elips.
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
          "ANN and exact vector search over the same collection",
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
          "Not a vector-only store. Documents, metadata, lineage are first-class.",
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
        code: `git clone https://github.com/axiomchronicles/elips.git
cd elips && cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j`,
      },
      {
        kind: "ul",
        items: [
          "Python 3.10+",
          "C++23 toolchain: clang 17 / gcc 13 / MSVC 19.38",
          "CMake ≥ 3.24",
          "Optional GPU: Metal / CUDA 12 / HIP 6 — selected at build time (CUDA/HIP experimental)",
        ],
      },
      { kind: "h", text: "Verify" },
      {
        kind: "code",
        lang: "python",
        code: `import elips
client = elips.Client(dimension=4, metric="cosine")
print(client.list_collections())   # []`,
      },
      {
        kind: "note",
        text: `elips.Client() is an in-memory database — it never touches the disk. Perfect for tests and tutorials. Use elips.PersistentClient(path, ...) to persist to a directory.`,
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

client = elips.PersistentClient("/tmp/my_first_elips", dimension=4, metric="cosine")
notes = client.get_or_create_collection("notes")

key = notes.add(embedding=[0.1, 0.2, 0.3, 0.4], metadata={"tag": "hello"})
print(key)

hit = notes.query(query_embeddings=[0.1, 0.2, 0.3, 0.4], n_results=1)[0]
print(hit.key, hit.distance, hit.meta)`,
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
    title: "Rows, vectors, metadata",
    eyebrow: "Chapter II · first contact",
    oneLiner: "The Row is the atom. Everything else hangs off it.",
    blocks: [
      {
        kind: "lede",
        text: "A Row is key + vector + optional metadata + optional document attachment + optional lineage.",
      },
      {
        kind: "diagram",
        name: "ObjectModel",
        caption: "Client → collections → rows. Rows are the only objects the index sees.",
      },
      {
        kind: "code",
        lang: "python",
        code: `notes.add(
    embedding=[0.1] * 128,
    metadata={"author": "ann", "kind": "design"},
)`,
      },
      {
        kind: "ul",
        items: [
          "embedding — fixed dimension per database; set when you create the Client, immutable after",
          "metadata — arbitrary dict-shaped values, JSON-compatible",
          "key — string id, auto-generated unless you pass id=",
          "document, chunk, lineage — optional, populated by add(documents=...)",
        ],
      },
      {
        kind: "note",
        text: "Metadata is read back with every hit. Don't put megabytes there; that's what document attachments are for.",
      },
    ],
  },
  {
    slug: "05-place-and-erase",
    number: 5,
    level: "Beginner",
    title: "add and delete",
    eyebrow: "Chapter II · first contact",
    oneLiner: "Two mutation verbs. Both go through the WAL before touching memory.",
    blocks: [
      { kind: "lede", text: "Two verbs. Everything else is a query." },
      {
        kind: "diagram",
        name: "Persistence",
        caption: "add / delete append to the WAL first, then mutate the in-memory store.",
      },
      {
        kind: "code",
        lang: "python",
        code: `key = notes.add(embedding=[0.1, 0.2, 0.3, 0.4], metadata={"tag": "a"})
notes.delete(ids=key)   # marks tombstone; checkpoint/compact reclaims space`,
      },
      {
        kind: "p",
        text: "From docs/storage.md: the WAL supports insert, erase, and insert_ex (for records carrying DocumentAttachment, ChunkInfo, or EmbeddingLineage). Crash recovery restores full record state, not only vectors.",
      },
      { kind: "note", text: "Updates are upsert(...) or update(ids=..., embeddings=...). Re-adding the same id overwrites it." },
    ],
  },
  {
    slug: "06-seek-basics",
    number: 6,
    level: "Beginner",
    title: "query — nearest neighbors",
    eyebrow: "Chapter III · retrieval",
    oneLiner: "Top-k nearest vectors, optionally with a distance cutoff.",
    blocks: [
      { kind: "lede", text: "query() is the bread-and-butter call. Vector in, ranked hits out." },
      {
        kind: "diagram",
        name: "QueryPath",
        caption: "App → collection → planner → metadata → index → ranked hits.",
      },
      {
        kind: "code",
        lang: "python",
        code: `hits = notes.query(
    query_embeddings=[0.1, 0.2, 0.3, 0.4],
    n_results=5,
    max_distance=0.3,
)
for h in hits:
    print(h.key, h.distance, h.meta)`,
      },
      {
        kind: "p",
        text: "From docs/cpp/api-reference/collection.md: each Hit carries key, distance, meta, document, vector — all hydrated from the authoritative record store.",
      },
    ],
  },
  {
    slug: "07-filters",
    number: 7,
    level: "Beginner",
    title: "Filters and metadata search",
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
        code: `hits = docs.query(
    query_embeddings=embed("alpha"),
    n_results=10,
    where={
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
        text: "add(text=...) embeds, chunks, and stores in one call — and records exactly which model produced each vector. Column form uses documents=/metadatas= for batches.",
      },
      {
        kind: "code",
        lang: "python",
        code: `docs = client.get_or_create_collection("documents")
docs.add(
    text="Long markdown article ...",
    metadata={"kind": "design"},
)

for hit in docs.query(query_texts="alpha", n_results=3):
    print(hit.text[:80])
    print(hit.meta)
    print(hit.document.uri if hit.document else None)`,      },
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
        code: `hits = docs.query(
    query_embeddings=embed("alpha design note"),
    query_texts="alpha design note",
    n_results=5,
    lexical_weight=0.25,   # default; pass both to fuse vector + lexical
)`,
      },
      {
        kind: "p",
        text: "lexical_weight ∈ [0, 1]. 0 means pure vector; 1 means pure lexical. The fused score is on each Hit's distance.",
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
elips collections /my_db
elips stats /my_db
elips verify /my_db                # replays WAL, prints OK / CORRUPT
elips checkpoint /my_db
elips export /my_db --collection docs --output docs.jsonl
elips import /my_db --collection docs --input docs.jsonl --dimension 768
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
        code: `search in <collection>
    nearest <[literal] | $binding>
    [k <int>]
    [threshold <number>]
    [where <filter>]
    [rank_by <distance | field>]
    [project <* | f1, f2, ...>]
    yield

get from <collection> id "<string-id>" yield
scan in <collection> [where ...] [offset N] [limit N] yield
add in <collection> vector [...] [metadata {...}]
delete from <collection> id "<string-id>"`,
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
        code: `client.query_eql("""
  search in articles
    nearest $embedding
    k 10
    where category = "technology" and published_year >= 2023
    project title, author
    yield
""", {"embedding": embed(query)})`,
      },
      { kind: "h", text: "Run from the CLI" },
      {
        kind: "code",
        lang: "bash",
        code: `elips query /my_db --eql 'search in docs nearest [0.1,0.2,0.3] k 5 yield'
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
        text: "From docs/cpp/api-reference/transaction.md: the transactional collection's add() validates dimension and finiteness at enqueue, throwing DimensionMismatch or InvalidVector. If enqueue succeeds for every op, commit() cannot fail validation.",
      },
      {
        kind: "code",
        lang: "python",
        code: `with client.begin_transaction() as txn:
    c = txn.collection("notes")
    c.add(embedding=vector1, metadata={"tag": "a"})
    c.add(embedding=vector2, metadata={"tag": "b"})
    # clean exit -> commit
    # exception  -> rollback (destructor)`,
      },
      {
        kind: "code",
        lang: "cpp",
        code: `auto txn = db->begin_transaction();
auto& tc = txn.collection("notes");
tc.add(vector1, {{"tag", std::string{"a"}}});
tc.add(vector2, {{"tag", std::string{"b"}}});
txn.commit();   // or txn.rollback();`,
      },
      {
        kind: "note",
        text: "Unlike Collection::add(), the transactional add() does not normalize the vector at enqueue — normalization happens inside the per-op add() during commit.",
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
enum class IndexType   { hnsw, flat };
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
    .index(elips::IndexType::hnsw)
    .hnsw_params({.max_connections = 32, .ef_construction = 400, .ef_search = 100})
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
├── elips.manifest                    # segmented mode: per-collection segment list
├── text_embedder/
│   └── default_v1_<dim>.localembed   # built-in local embedder artifact
├── segments/
│   └── collection_<n>_<epoch>.segment # one fresh segment per collection per checkpoint
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
          "compact() — rebuilds each collection index from the record store, then checkpoints",
        ],
      },
      {
        kind: "p",
        text: "Segmented mode writes one fresh segment per collection, rewrites elips.manifest, and removes obsolete segment files. Snapshot mode writes elips.snapshot.tmp, atomic-renames to elips.snapshot, removes segmented artifacts.",
      },
      {
        kind: "code",
        lang: "python",
        code: `client.checkpoint()   # flush + truncate WAL
client.compact()      # rebuild indexes + checkpoint`,
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
        code: `replica = elips.PersistentClient("/var/data/elips", access_mode="read_only")
for hit in replica.get_collection("docs").query(query_embeddings=q, n_results=10):
    ...`,
      },
      {
        kind: "ul",
        items: [
          "One read_write opener via exclusive lock",
          "Many read_only openers via shared lock",
          "Read-only collections reject add/delete/checkpoint with StorageError",
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
        code: `client = elips.PersistentClient("/data/db", dimension=128, use_default_text_embedder=True)`,
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

client = elips.PersistentClient("/data/db", dimension=768, embedder=MyEmbedder())`,
      },
      {
        kind: "p",
        text: "From docs/cpp/api-reference/config.md: TextEmbedderPort exposes provider_name, model_name, revision_name, backend_name, output_dimension. ELIPS uses these to fingerprint the manifest.",
      },
      {
        kind: "note",
        text: "Non-rehydratable external embedders store metadata only. Reopening without the same embedder fails query(query_texts=...) with an actionable ConfigError.",
      },
    ],
  },
  {
    slug: "18-planner-explain",
    number: 18,
    level: "Intermediate",
    title: "The planner and explain()",
    eyebrow: "Chapter IX · observability",
    oneLiner: "Ask the engine what it's about to do before it does it.",
    blocks: [
      { kind: "lede", text: "explain() is your single best window into runtime behavior." },
      {
        kind: "code",
        lang: "cpp",
        code: `auto plan = collection.explain_query(query, /*k=*/10, filter);
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
          "GPU built in → gpu_index = true; the dispatch goes through GpuBackend",
          "No accelerator, flat index → planner picks brute-force; expect linear-in-N latency",
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
    oneLiner: "Typed C++ exceptions with a parallel Python hierarchy.",
    blocks: [
      { kind: "lede", text: "Errors are typed. Catch them with intent." },
      {
        kind: "code",
        lang: "text",
        code: `std::runtime_error
└── elips::ElipsError
    ├── ConfigError              // identity / dimension / embedder mismatch
    ├── StorageError             // WAL / segments / lock / read-only writes
    ├── DimensionMismatch        // wrong-shape vector
    ├── InvalidVector            // NaN / Inf
    ├── NotFound                 // unknown id
    ├── CollectionNotFoundError  // missing collection
    ├── LockConflict             // writer-lock contention
    ├── NotTrainedError          // query before train / compress
    ├── BackendUnavailableError  // requested GPU backend absent
    └── ParseError               // malformed EQL`,
      },
      {
        kind: "code",
        lang: "python",
        code: `try:
    notes.add(embedding=[float("nan")] * 4)
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
        text: "Python doesn't reimplement anything. It binds the same Database and Collection that C++ users hold.",
      },
      {
        kind: "p",
        text: "From docs/python/bindings/pybind11-architecture.md: there is a low-level surface (elips.open, Database, Collection — direct binding mirrors, kept as legacy shims) and the modern Chroma-style surface (Client / PersistentClient / Collection) that most code should use.",
      },
      {
        kind: "code",
        lang: "python",
        code: `# legacy shim — mirrors C++ shapes
db = elips.open("/data/db", dimension=384)
docs = db.collection("documents")

# modern — Chroma-style, recommended
client = elips.PersistentClient("/data/db", dimension=384)
docs = client.get_or_create_collection("documents")`,
      },
      {
        kind: "ul",
        items: [
          "Vectors cross the boundary as zero-copy float buffers when possible",
          "Metadata dicts are converted via pybind11's std::variant binding",
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
    oneLiner: "SDKs, database, collections, ports, engines, storage. Five layers.",
    blocks: [
      {
        kind: "lede",
        text: "Every ELIPS feature lives on exactly one of five layers. Knowing which one is the difference between a one-line patch and a refactor.",
      },
      {
        kind: "diagram",
        name: "SystemShape",
        caption:
          "SDKs → Database → Collection → planner + ports (index/metadata/text/GPU) → storage (WAL + segments).",
      },
      {
        kind: "p",
        text: "From docs/architecture/system-overview.md: each layer talks down to the next via abstract ports. Nothing in the planner depends on a concrete index. Nothing in the index depends on a concrete storage layout.",
      },
      {
        kind: "diagram",
        name: "AlgorithmPorts",
        caption:
          "Ports & adapters: the planner sees Index, MetadataIndex, TextEmbedderPort, GpuBackend — not concrete classes.",
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
        text: "checkpoint() writes one segment per collection: collection_<n>_<epoch>.segment. The new elips.manifest references the new epoch. Old segment files are unlinked only after the manifest has been atomically replaced.",
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
    title: "IndexHNSW — HNSW in ELIPS",
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
      { kind: "h", text: "The FAISS-style native index API" },
      {
        kind: "p",
        text: "Below the collection layer, ELIPS exposes a FAISS-shaped index family: IndexFlat, IndexIVFFlat, IndexIVFPQ, IndexIVFScalarQuantizer, IndexHNSW, plus index_factory. Metric strings are FAISS-style ('l2' for euclidean).",
      },
      {
        kind: "code",
        lang: "python",
        code: `import numpy as np, elips
xb = np.random.rand(100000, 128).astype("float32")
xq = np.random.rand(1000, 128).astype("float32")

index = elips.IndexHNSW(128, 32, "l2")   # d, M, metric
index.add(xb)
D, I = index.search(xq, 10)
print(index.ntotal, index.d, index.is_trained)

# or an IVF index via the factory, with a probe sweep:
ivf = elips.index_factory(128, "IVF316,Flat", "l2")
ivf.train(xb); ivf.add(xb); ivf.nprobe = 128
D, I = ivf.search(xq, 10)`,
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
        text: "From docs/internals/lock-manager.md: locks are acquired in a fixed hierarchical order (database → collection → record). Acquiring out of order is a programmer error and is asserted in debug builds.",
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
        text: "From docs/internals/transaction-engine.md: the transactional collection holds a non-owning pointer to its parent Transaction. Validation runs at enqueue time. commit() walks the op list under the database write lock and applies via per-collection add() / remove(). The destructor rolls back if neither commit nor rollback was called.",
      },
      {
        kind: "code",
        lang: "cpp",
        code: `class Transaction {
public:
    TransactionCollection collection(std::string name);
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
        text: "Bindings ($name) are typed at runtime — the executor checks that a $vector binding is the right dimension before the plan ever reaches a Collection.",
      },
    ],
  },
  {
    slug: "28-gpu-engine",
    number: 28,
    level: "Advanced",
    title: "GPU engine architecture",
    eyebrow: "Chapter XV · GPU",
    oneLiner: "Ports & adapters at the device boundary. Metal, CUDA, HIP, CPU — one backend contract.",
    blocks: [
      {
        kind: "lede",
        text: "GPU support is optional, compile-time-selectable, and entirely behind a port. Nothing in the planner mentions CUDA.",
      },
      {
        kind: "diagram",
        name: "GpuEngine",
        caption:
          "GpuBackend + GpuMemoryPort + GpuKernelPort + GpuStreamPort + GpuIndexPort. Backends fulfill the contract.",
      },
      { kind: "h", text: "The 13-method GpuBackend (include/elips/gpu_engine/GpuBackend.hpp)" },
      {
        kind: "code",
        lang: "cpp",
        code: `class GpuBackend {
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
    oneLiner: "Metal, CUDA, HIP, CPU — each behind a compile-time flag.",
    blocks: [
      {
        kind: "lede",
        text: "GpuDeviceManager probes whatever was compiled in. BackendSelector picks the best device that matches policy.",
      },
      {
        kind: "p",
        text: "Probe order (docs/internals/gpu-engine.md): Metal (Apple), CUDA, HIP. Each probe init+inspect+shutdowns the backend inside a try/catch. Devices are then sorted by total device memory descending.",
      },
      { kind: "h", text: "Backend maturity" },
      {
        kind: "ul",
        items: [
          "Metal — production, default ON on Apple; distance + top-k kernels",
          "CUDA — experimental",
          "HIP / ROCm — experimental",
          "CPU — always available; the fallback path",
        ],
      },
      { kind: "h", text: "Select a device from Python" },
      {
        kind: "code",
        lang: "python",
        code: `import elips

for acc in elips.accelerators():
    print(acc.name, acc.backend, acc.memory_bytes, acc.unified_memory)

# device=: "cpu" | "auto" | "metal" | "cuda:N" | "hip:N"  (NOT gpu=True)
client = elips.Client(dimension=128, device="metal", fallback=True)
print(client.native.gpu_info())`,
      },
      {
        kind: "ul",
        items: [
          "device='auto' — best probed device, else CPU",
          "device='cpu' — never touches the GPU",
          "device='cuda:0'/'hip:0'/'metal' — a named backend; raises BackendUnavailableError if absent unless fallback=True",
        ],
      },
      {
        kind: "diagram",
        name: "GpuAccelGlance",
        caption: "Probe → sort → select → initialize. Falls back when fallback=True.",
      },
    ],
  },
  {
    slug: "30-dynamic-batcher-pipelines",
    number: 30,
    level: "Advanced",
    title: "QueryBatcher and the GPU pipelines",
    eyebrow: "Chapter XV · GPU",
    oneLiner: "Coalesce concurrent queries into one launch; ingest and quantize on the device.",
    blocks: [
      {
        kind: "lede",
        text: "GPUs hate small launches. The QueryBatcher exists to make them stop being small.",
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
          "algorithm — brute-force, IVF-Flat, IVF-PQ, HNSW, hybrid",
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
        .algorithm(elips::gpu::Algorithm::ivf_flat)
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
          "Candidate-set size — emit from explain(); spikes here predict tail latency",
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
      {
        kind: "note",
        text: "Real numbers (ELIPS 1.1.0, Apple M4, dim=128, 100k vectors, 1k queries, k=10, L2): IndexFlat on CPU reaches recall 1.000 at ~489 QPS; bf_knn on Metal (batched) recall 1.000 at ~914 QPS (and ~24 QPS for one-query-at-a-time); IVFFlat nlist=316 nprobe=128 recall 0.801 at ~1616 QPS, trading down to recall 0.152 at ~14123 QPS with nprobe=8.",
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
          "Durability: paranoid on durable storage, relaxed with scheduled checkpoints on ephemeral disks",
          "Replicas: open with access_mode='read_only' from N readers; shared flock coordinates",
          "Checkpoint: schedule elips checkpoint via cron/systemd-timer; tune cadence to recovery time",
          "Backups: snapshot the directory while it is closed, or while only read-only openers hold it",
          "Telemetry: emit explain().strategy and candidate_count per query",
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
