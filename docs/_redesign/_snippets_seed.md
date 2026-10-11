<!-- Phase 8B docs-snippet harness seed. Phase 9 replaces/extends this with the
     real documentation tree; the CI job runs scripts/run_doc_snippets.py over
     these blocks today so the harness itself is exercised and kept green.
     Every executable block here must run offline against the built `elips`
     module (no embedder downloads, no filesystem writes). -->

# Executable Snippet Seed

Fixed-seed exact k-NN via the FAISS-style Index API — pure CPU, offline:

```python
import numpy as np
import elips

rng = np.random.RandomState(0)
xb = rng.randn(200, 8).astype("float32")
xq = xb[:3]

index = elips.IndexFlat(8, metric="l2")
index.add(xb)
_, labels = index.search(xq, 5)

# Each query's nearest neighbour is itself (distance 0).
assert [int(labels[i][0]) for i in range(3)] == [0, 1, 2]
```

The following block is intentionally **not** executed — it is marked `no-exec`
because it describes an environment-specific step rather than runnable code:

```python no-exec
# Requires an NVIDIA GPU + the CUDA build; pseudocode for illustration only.
db = elips.connect(":memory:", dimension=128, device="cuda:0")
```
