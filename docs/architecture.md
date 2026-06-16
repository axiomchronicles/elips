# Architecture

ELIPS is organized around a narrow core API and a small set of storage and
query subsystems. The current architecture is document-aware rather than
vector-only: every record may carry metadata, a source document attachment,
chunk coordinates, and embedding lineage.

```text
Python modern API / Python core bindings / C++ SDK / CLI
                         |
                    ElipsInstance
         lifecycle · config · vault registry · WAL attachment
                         |
                      Vault
  record store · planner · metadata index · ANN / exact / GPU index
                         |
      ----------------------------------------------------------
      |                     |                  |               |
   Query path           Text path         Persistence      Locking
 seek / scan / fetch  place_document     manifest/WAL     shared RO /
 seek_text / hybrid   seek_text/hybrid   segments/snap    exclusive RW
```

## Core Objects

- `ElipsInstance`: top-level database handle. Owns vaults, persistence, locks,
  and optional GPU backend state.
- `Vault`: authoritative record store plus search planner and index.
- `Record`: `id`, `vector`, `payload`, and optional `document`, `chunk`,
  `lineage`.
- `MetadataIndex`: exact-match accelerator used by the planner for equality and
  set-membership filters.
- `TextEmbedderPort`: optional core interface for native text ingestion and
  text-first querying.

## Query Path

Every vector or hybrid query goes through `Vault::plan_seek()` first.

- `ann_index`: use the configured ANN or GPU index directly.
- `exact_candidates`: restrict the search domain using `MetadataIndex`.
- `full_scan`: fall back when the candidate set is small or the query shape
  makes a scan cheaper.
- `text_probe`: used for text-only search when no embedding stage is needed.
- `hybrid_fusion`: combine vector distance with lexical overlap from attached
  documents.

The planner emits `QueryPlan`, which is exposed in both C++ and Python for
inspection and testing.

## Persistence Path

Persistent databases can run in two layouts:

- Segmented mode, enabled by default: root `elips.manifest` plus one segment
  file per vault under `segments/`.
- Snapshot mode: a single `elips.snapshot` file for compatibility or when
  segmented storage is disabled.
- Text embedder identity: `TEXT_EMBEDDER.manifest` plus optional
  `text_embedder/*.localembed` artifacts for the built-in local embedder.

All mutations are WAL-appended before in-memory mutation. WAL replay rebuilds
the record store on open, including document attachments, chunk info, and
embedding lineage.

## Concurrency Model

- One read-write opener at a time via an exclusive advisory lock.
- Multiple read-only openers via shared advisory locks.
- Read-only vaults reject `place`, `place_document`, `erase`, `rebuild_index`,
  and `compact`-driven mutation paths with `StorageError`.

ELIPS is an embedded engine, not a server. Process-level coordination is done
through file locks rather than background daemons.

## Python Surface

There are two layers:

- Low-level bindings: `open()`, `open_with_config()`, `Vault`, `Config`.
- Modern wrapper: `connect()`, `Engine`, and `Arena` for typed text-first
  ingestion and retrieval.

The modern wrapper prefers native core text APIs when the database config
resolves a text embedder, including the default local embedder attached for new
databases. When the config has no native embedder but the wrapper is given a
Python callable embedder, it falls back to Python-side embedding plus hybrid
querying. If neither exists, text-first APIs raise `ConfigError` / `ValueError`
rather than silently switching to lexical-only behavior.

For the detailed FAISS reverse-engineering and the corresponding Elips GPU
index design and implementation plan, see
[FAISS GPU Acceleration Design](/Users/kuroyami/ellips/docs/architecture/faiss-gpu-acceleration-design.md).
