# Local Embedding System

ELIPS now has a first-class local text embedding workflow. New databases attach
the built-in local embedder automatically unless `Config.auto_text_embedder(false)`
or `use_default_text_embedder=False` is set. The built-in embedder is
deterministic, offline-safe, lazily loaded, and rehydrated from local storage
on reopen.

## Design Summary

- **Default behavior:** new databases auto-provision `elips-local/default@v1`
  against the database dimension.
- **Persistent identity:** `TEXT_EMBEDDER.manifest` records the database-level
  embedding identity and whether it can be restored automatically.
- **Local artifacts:** built-in local embedders store their deterministic
  artifact under `text_embedder/` inside the database by default.
- **Explicit external embedders:** Python callables and custom C++ embedders
  still work, but persistent reopen requires the caller to supply the same
  embedder again because ELIPS only stores metadata for them.
- **Vector-first compatibility:** `place()` remains unchanged and can still
  attach documents manually. Text-first search is only meaningful when the
  stored vectors and the configured text embedder are the same model.

## Python And C++ Usage

Python default local workflow:

```python
import elips

db = elips.open("/tmp/elips-local", dimension=128, metric="cosine")
docs = db.vault("documents")
docs.place_document("alpha design note", {"kind": "design"})
print(docs.seek_text("alpha", top=1)[0].document.text)
```

Python explicit local workflow:

```python
local = elips.LocalEmbedderConfig(
    model="compact",
    revision="v1",
    storage_path="/tmp/elips-models/compact.localembed",
    dimension=128,
)

db = elips.open(
    "/tmp/elips-local-explicit",
    dimension=128,
    metric="cosine",
    embedder=local,
    use_default_text_embedder=False,
)
```

C++ explicit local workflow:

```cpp
auto db = elips::open(
    "/tmp/elips-cpp-local",
    elips::Config{}
        .dimension(128)
        .metric(elips::Metric::cosine)
        .local_text_embedder(elips::LocalTextEmbedderOptions{
            .model = "compact",
            .revision = "v1",
            .storage_path = "/tmp/elips-models/compact.localembed",
            .dimension = 128,
        }));
```

## Resolver Flow

```mermaid
flowchart LR
    A[Python API or C++ SDK] --> B[Config]
    B --> C[open() resolver]

    C -->|explicit callable| D[PythonTextEmbedder or custom TextEmbedderPort]
    C -->|explicit local or default local| E[LocalTextEmbedder]

    C --> F[TEXT_EMBEDDER.manifest]
    E --> G[text_embedder/*.localembed]

    D --> H[metadata-only identity]
    E --> I[local artifact load on first text call]

    D --> J[place_document / seek_text]
    I --> J

    J --> K[Embedding vector]
    K --> L[Vault record store + ANN index]
    J --> M[EmbeddingLineage]
```

## Reopen Semantics

- Built-in local embedders are restored automatically from
  `TEXT_EMBEDDER.manifest` and the referenced artifact.
- Python callables and custom `TextEmbedderPort` implementations persist
  metadata only. Reopen with the same embedder if you want `place_document()`
  and `seek_text()` to work again.
- Disabling automatic text embedding is explicit: use
  `Config.auto_text_embedder(false)` in C++ or
  `use_default_text_embedder=False` / `Config.auto_text_embedder(False)` in
  Python.

## Compared With Chroma-Style Defaults

- ELIPS attaches the default embedder at database open, then persists that
  identity in `TEXT_EMBEDDER.manifest`.
- ELIPS keeps vector-first APIs first-class: `place()` and `seek()` never
  depend on text embedding.
- ELIPS does not silently fall back to lexical-only text search when no text
  embedder is available. Text-first APIs fail with an actionable `ConfigError`
  so the embedding model remains explicit.

## Failure Modes

- If a database expects an external non-rehydratable embedder and one is not
  supplied on reopen, `place_document()` and `seek_text()` raise `ConfigError`
  that explains how to reopen with the matching embedder.
- If the local artifact referenced by `TEXT_EMBEDDER.manifest` is missing,
  reopen raises `StorageError`.
- If the local artifact is corrupt, the error is raised the first time the
  embedder is loaded, preserving lazy-load behavior while still failing
  explicitly.
