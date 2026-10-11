# FAQ

### Why do I have to pass `dimension=`?

ELIPS is **dimension-first**. `Client(...)` and `PersistentClient(...)` require
`dimension=` (or a dimension-fixing `embedder=`). Unlike Chroma it does not infer
the dimension from a default embedding function — a mismatched vector is an error,
not a silent reshape. On reopen, the persisted `IDENTITY` is authoritative and a
conflicting `dimension` raises `ConfigError`.

### Can I use my own string ids?

Yes. The Collection API accepts arbitrary string `ids`; ELIPS maps each to an
internal `RecordID`, persists the original, and returns it from `get`/`query`.
Omit `ids` and ELIPS generates them.

### Which backend am I using, and is my GPU being used?

CPU is always available and is the reference path. `elips.has_gpu` reports whether
the build includes a GPU backend, and `elips.accelerators()` lists the devices it
can drive:

```python
import elips

print(elips.has_gpu)
for spec in elips.accelerators():
    print(spec.name, spec.backend)
```

On a CPU-only machine `accelerators()` returns an empty list — that is normal, not
an error. `bf_knn(..., device="auto")` uses Metal when available and falls back to
CPU; pass `device="cpu"` or `device="metal"` to force one.

### Which GPU backends actually work?

| Backend | Status |
|---|---|
| CPU | Stable — all indexes and metrics. |
| Metal (Apple Silicon) | Beta — validated on Apple M4, with real GPU kernels. |
| CUDA (NVIDIA) | **Experimental** — compiles against the interfaces, not hardware-validated. |
| HIP / ROCm (AMD) | **Experimental** — compiles against the interfaces, not hardware-validated. |

Treat CUDA and HIP as unvalidated: the code is interface-complete and compiles,
but no test has run a real CUDA/HIP kernel on hardware. Do not rely on them in
production yet.

### Why do my `euclidean` distances differ from FAISS?

ELIPS returns the **true L2 distance** (`sqrt(sum((a-b)^2))`), whereas FAISS's
`METRIC_L2` returns the **squared** L2 distance. Same ordering, different scale —
square ELIPS distances (or compare thresholds accordingly) if you need FAISS
parity. `cosine` is inner-product on L2-normalized inputs.

```python
import numpy as np
import elips

index = elips.IndexFlat(2, "l2")
index.add(np.array([[0, 0]], dtype=np.float32))
D, _ = index.search(np.array([[3, 4]], dtype=np.float32), 1)
assert float(D[0, 0]) == 5.0          # true L2 (FAISS would report 25.0)
```

### What's the difference between the Collection API and the Index API?

The Collection API (Chroma-style) is the high-level surface: documents, metadata,
string ids, filters, text embedding, persistence. The Index API (FAISS-style) is
the low-level numpy surface: raw float32 vectors, `train`/`add`/`search`, and
`index_factory`. Use the Collection API for applications; reach for the Index API
when you want direct control over the ANN index.

### Do the old names still work?

Yes, for this release. The pre-rename names (`connect`/`open`, `Vault`,
`place`/`seek`/`fetch`/`erase`, `arena`/`ingest`/`probe`, and the EQL `place`/`seek`
verbs) are kept as deprecation shims and emit warnings. They are scheduled for
removal in the next minor version — see [MIGRATION.md](../MIGRATION.md).
