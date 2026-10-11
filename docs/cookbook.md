# Cookbook

Worked patterns on the ELIPS Python SDK. Every runnable block below executes
against the built module. Several patterns use your own embedding model; the
helper here is a stand-in so the examples run as-is:

```python
import hashlib
import elips


def embed(text: str) -> list[float]:
    """Toy deterministic embedder (replace with your real model)."""
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in digest[:8]]


client = elips.Client(dimension=8, metric="cosine")
```

## Semantic document search

With the built-in text embedder you can store and search raw documents
directly — no explicit vectors needed.

```python
corpus = client.get_or_create_collection("corpus")
corpus.add(
    ids=["vax", "ml", "db"],
    documents=[
        "how vaccines train the immune system",
        "gradient descent and neural networks",
        "write-ahead logging in databases",
    ],
    metadatas=[{"topic": "bio"}, {"topic": "ai"}, {"topic": "sys"}],
)

for hit in corpus.query(query_texts="how do vaccines work?", n_results=2):
    print(hit.meta["topic"], round(hit.distance, 3))
```

## RAG retrieval

```python
def retrieve(question, k=3):
    hits = corpus.query(query_texts=question, n_results=k)
    return [h.text for h in hits]


context = "\n\n".join(retrieve("how do neural networks learn?"))
# prompt = f"Answer using the context:\n{context}\n\nQ: ..."
print(len(context) > 0)
```

## Recommendation (user → item embeddings)

Supply explicit vectors when you compute embeddings yourself. `where` filters
candidates with a metadata predicate.

```python
items = client.get_or_create_collection("items")
items.add(
    ids=["i1", "i2"],
    embeddings=[embed("noise-cancelling headphones"), embed("running shoes")],
    metadatas=[{"in_stock": True}, {"in_stock": False}],
)

recs = items.query(
    query_embeddings=embed("headphones"),
    n_results=20,
    where=elips.Filter().field("in_stock").equals(True),
)
print([r.key for r in recs])
```

## Agent long-term memory

```python
mem = client.get_or_create_collection("memory")
mem.add(
    ids=["m1"],
    embeddings=[embed("user prefers dark mode")],
    metadatas=[{"session": "s1", "kind": "fact"}],
)

recent = mem.query(
    query_embeddings=embed("ui preferences"),
    n_results=10,
    where=elips.Filter().field("session").equals("s1").field("kind").equals("fact"),
)
print(len(recent))
```

## Time-decayed retrieval

Fetch a generous candidate set, then re-rank by similarity × recency in Python.

```python
import time

now = time.time()
cands = mem.query(query_embeddings=embed("ui preferences"), n_results=100)


def score(h):
    age_days = (now - h.meta.get("ts", now)) / 86400
    return (1 - h.distance) * (0.97 ** age_days)


ranked = sorted(cands, key=score, reverse=True)[:10]
print(len(ranked))
```

## Duplicate / near-neighbor detection

Cap the result set and inspect the closest hit's distance.

```python
near = corpus.query(query_texts="vaccines", n_results=1)
if near and near[0].distance < 0.2:
    print("likely duplicate of", near[0].key)
else:
    print("no close duplicate")
```

## Metadata-filtered search (EQL)

EQL is the declarative query language. Run it through `client.query_eql(...)`
(or `client.native.query(...)`); it returns native `Result` rows with `.id`,
`.distance`, and `.data`.

```python
articles = client.get_or_create_collection("articles")
articles.add(
    ids=["a1", "a2"],
    embeddings=[embed("ai systems paper"), embed("old biology paper")],
    metadatas=[{"category": "ai", "year": 2024}, {"category": "bio", "year": 2011}],
)

rows = client.query_eql(
    'search in articles nearest $q k 10 where year >= 2023 project category yield',
    {"q": embed("ai systems paper")},
)
print([r.data.get("category") for r in rows])
```

## Hybrid pre-filter then vector rank

Restrict to a partition with a filter, then rank by vector distance.

```python
flt = elips.Filter().field("category").equals("ai").field("year").gte(2020)
hits = articles.query(query_embeddings=embed("ai systems paper"), n_results=10, where=flt)
print([h.key for h in hits])
client.close()
```

## Raw vector indexing (FAISS-style)

When you want a bare vector index with no record store or metadata, use the
`Index` API with `float32` NumPy arrays.

```python
import numpy as np

d = 32
xb = np.random.rand(1000, d).astype("float32")
index = elips.index_factory(d, "IVF16,Flat", metric="l2")
index.train(xb)
index.add(xb)
index.nprobe = 4
distances, labels = index.search(xb[:5], 10)
print(distances.shape, labels.shape)
```

## GPU acceleration

GPU is opt-in via a device string on the client. Metal is Beta (validated on
Apple M4); CUDA/HIP are experimental and not hardware-validated. A named
backend that is absent raises `BackendUnavailableError` unless `fallback=True`.

```python
# no-exec — requires a GPU backend; shape only
gpu = elips.Client(dimension=384, device="metal")       # or "cuda:0", "auto"
col = gpu.get_or_create_collection("corpus")
hits = col.query(query_embeddings=[0.0] * 384, n_results=10)
gpu.close()
```

`elips.bf_knn(xq, xb, k, device="auto")` runs an exact brute-force k-NN on the
GPU when one is available, falling back to CPU otherwise.

These patterns compose: filters, thresholds (`max_distance=`), and EQL bindings
cover classification, clustering seeds, few-shot retrieval, code search, and
anomaly detection with the same primitives.
