# Python SDK

Build and import:

```bash
cmake -S . -B build -G Ninja -DELIPS_BUILD_PYTHON=ON
cmake --build build --target elips_pymodule
export PYTHONPATH=$PWD/bindings/python
```

```python
import elips
```

## Open

```python
db = elips.open("/data/vectors", dimension=1536, metric="cosine")  # or "euclidean", "dot_product"
db = elips.open(":memory:", dimension=768)        # ephemeral
db = elips.open("/data/vectors", index="exact")   # brute-force backend
```

Use it as a context manager to checkpoint + release the lock on exit:

```python
with elips.open("/data/vectors", dimension=384) as db:
    ...
```

## Vaults & writes

```python
docs = db.vault("documents")
rid = docs.place([0.1, 0.2, ...], {"title": "Doc", "year": 2024})
docs.place([0.3, 0.4, ...], {"title": "Other"}, id="custom-uuid")
```

## Search

```python
for hit in docs.seek([0.1, 0.2, ...], top=10):
    print(hit.id, hit.distance, hit.data)

# filtered
flt = elips.Filter().field("year").gte(2023).field("category").equals("tech")
docs.seek(query, top=10, where=flt)

# range search
docs.seek(query, top=1000, where=flt, threshold=0.25)
```

## Filters

```python
f = (elips.Filter()
     .field("category").equals("tech")
     .field("score").gte(0.8)
     .field("country").one_of(["US", "GB"])
     .field("title").contains("intro"))

either = elips.Filter().field("tier").equals("pro").or_(
         elips.Filter().field("year").gte(2023))
negated = elips.Filter.not_(elips.Filter().field("draft").equals(True))
```

## EQL

```python
results = db.query("""
    seek in documents nearest $q top 20
    where category = "finance" and year >= 2022
    project title, author yield
""", bindings={"q": query_vector})
```

## Fetch, scan, erase, maintenance

```python
record = docs.fetch(rid)             # dict | None
rows = docs.scan(where=flt)          # list of {"id", "data"}
docs.erase(rid)
db.checkpoint()
db.list_vaults()
```

All public APIs are typed (`py.typed` + `_core.pyi`). Errors raise
`elips.ElipsError` (and `elips.LockConflict` for writer contention).
