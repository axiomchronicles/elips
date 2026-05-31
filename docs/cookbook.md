# Cookbook

Worked patterns built on the ELIPS Python SDK. Each assumes an embedding model
of your choice (`embed(text) -> list[float]`).

## Semantic document search

```python
import elips

db = elips.open("/data/search", dimension=384, metric="cosine")
docs = db.vault("corpus")
for doc in corpus:
    docs.place(embed(doc.text), {"title": doc.title, "url": doc.url})

for hit in docs.seek(embed("how do vaccines work?"), top=5):
    print(hit.data["title"], hit.distance)
```

## RAG retrieval

```python
def retrieve(question, k=5):
    hits = docs.seek(embed(question), top=k)
    return [h.data["text"] for h in hits]

context = "\n\n".join(retrieve(user_question))
answer = llm(f"Answer using the context:\n{context}\n\nQ: {user_question}")
```

## Recommendation (user → item embeddings)

```python
items = db.vault("items")
# items placed with their content/collaborative embedding + metadata
recs = items.seek(user_embedding, top=20,
                  where=elips.Filter().field("in_stock").equals(True))
```

## Agent long-term memory

```python
mem = db.vault("memory")
mem.place(embed(observation), {"ts": now, "session": sid, "kind": "fact"})

recent_facts = mem.seek(
    embed(current_goal), top=10,
    where=elips.Filter().field("session").equals(sid).field("kind").equals("fact"))
```

## Time-decayed retrieval

```python
# fetch a generous candidate set, then re-rank by similarity * recency in Python
cands = mem.seek(embed(query), top=100)
def score(h):
    age_days = (now - h.data["ts"]) / 86400
    return (1 - h.distance) * (0.97 ** age_days)
ranked = sorted(cands, key=score, reverse=True)[:10]
```

## Duplicate detection (range search)

```python
near = docs.seek(embed(candidate), top=1000, threshold=0.05)  # cosine distance
if near:
    print("likely duplicate of", near[0].id)
```

## Metadata-filtered search (EQL)

```python
results = db.query("""
    seek in articles nearest $q top 10
    where category in ["ai", "systems"] and year >= 2023
    project title, author yield
""", bindings={"q": embed(query)})
```

## Hybrid pre-filter then vector rank

```python
# Restrict to a tenant/partition with a filter, then rank by vector distance.
flt = elips.Filter().field("tenant").equals(tenant_id).field("lang").equals("en")
hits = docs.seek(embed(query), top=10, where=flt)
```

These patterns compose: filters, thresholds, projections, and EQL bindings cover
classification, clustering seeds, few-shot example retrieval, code search, image
similarity, and anomaly detection (large nearest-neighbor distance) with the same
primitives.
