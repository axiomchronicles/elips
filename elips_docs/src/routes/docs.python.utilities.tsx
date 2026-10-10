import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/python/utilities")({
  head: () => ({
    meta: [
      { title: "Utilities & Vector Helpers — Python API" },
      {
        name: "description",
        content:
          "Complete reference for ELIPS Python top-level utilities: distance, magnitude, normalize, generate_id, is_valid_id, NumPy interoperability, and batch embedder helpers.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference · Python"
      title="Utilities & Vector Helpers"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "vector-math", label: "Distance & Magnitude" },
        { id: "normalization", label: "Normalization" },
        { id: "ids", label: "Record IDs" },
        { id: "numpy-interop", label: "NumPy Interoperability" },
        { id: "batch-helpers", label: "Batch Embedder Helpers" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS exports a small set of vector-math and id helpers at the top level
        of the <code>elips</code> namespace. They operate on plain Python
        sequences or NumPy arrays and share the same C++ math core the engine
        uses internally.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        The helpers below are available directly as{" "}
        <code>elips.distance</code>, <code>elips.magnitude</code>,{" "}
        <code>elips.normalize</code>, <code>elips.generate_id</code>, and{" "}
        <code>elips.is_valid_id</code>. They let you compute metric-consistent
        distances, prepare vectors, and mint record ids without opening a
        database.
      </p>

      <h2 id="vector-math">Distance &amp; Magnitude</h2>
      <CodeBlock lang="python">{`import elips

vec_a = [0.1, 0.5, 0.9, 0.4]
vec_b = [0.2, 0.4, 0.8, 0.5]

# Ordering-normalized distance for a given metric
# metric ∈ {"cosine", "euclidean", "dot_product"}
d_cos = elips.distance("cosine", vec_a, vec_b)
d_l2  = elips.distance("euclidean", vec_a, vec_b)
d_dot = elips.distance("dot_product", vec_a, vec_b)

print(f"cosine distance:    {d_cos:.4f}")
print(f"euclidean distance: {d_l2:.4f}")
print(f"dot distance:       {d_dot:.4f}")

# Euclidean L2 norm of a single vector
print(elips.magnitude([3.0, 4.0]))  # 5.0`}</CodeBlock>
      <p>
        <code>distance(metric, a, b)</code> returns the same ordering-normalized
        value the engine ranks on, so a smaller number always means "closer"
        regardless of metric. This is the value exposed as{" "}
        <code>Hit.distance</code> on search results.
      </p>

      <h2 id="normalization">Normalization</h2>
      <p>
        <code>normalize(vector)</code> L2-normalizes a vector to unit length and
        returns a <code>tuple[float, ...]</code>. Zero vectors are returned
        unchanged (no division by zero).
      </p>
      <CodeBlock lang="python">{`norm_vec = elips.normalize([3.0, 4.0])
print(norm_vec)  # (0.6, 0.8)

print(elips.normalize([0.0, 0.0]))  # (0.0, 0.0) — unchanged`}</CodeBlock>

      <h2 id="ids">Record IDs</h2>
      <p>
        ELIPS record ids are UUIDv7 hex strings. Generate one ahead of a write
        when the caller must know the id before the record exists (e.g. to
        publish it to an event queue), or validate an externally supplied id.
      </p>
      <ul>
        <li>
          <code>generate_id() -&gt; str</code> — a fresh UUIDv7 hex string.
        </li>
        <li>
          <code>is_valid_id(id: str) -&gt; bool</code> — validates the id format.
        </li>
      </ul>
      <CodeBlock lang="python">{`rid = elips.generate_id()
print(rid)                      # e.g. "018f...e7a1"
print(elips.is_valid_id(rid))   # True
print(elips.is_valid_id("nope"))  # False

# Use a caller-assigned id at write time
with elips.Client(dimension=2) as client:
    docs = client.get_or_create_collection("docs")
    docs.add(vector=[1.0, 0.0], id=rid)`}</CodeBlock>

      <h2 id="numpy-interop">NumPy Interoperability</h2>
      <p>
        All vector arguments accept Python lists or NumPy arrays
        interchangeably. Collection methods take 1-D arrays for single records
        and 2-D arrays for column-form batches; returned vectors are plain
        tuples you can wrap back into NumPy.
      </p>
      <CodeBlock lang="python">{`import numpy as np
import elips

# A batch of 1000 x 1536 float32 embeddings
matrix = np.random.randn(1000, 1536).astype(np.float32)

with elips.Client(dimension=1536) as client:
    docs = client.get_or_create_collection("docs")

    # Column form accepts a 2-D NumPy array directly
    ids = docs.add(embeddings=matrix)

    # A single NumPy row works as a query vector
    hits = docs.query(query_embeddings=matrix[0], n_results=5)

    # Returned vectors are tuples — wrap them back into NumPy
    rows = docs.get(ids[:3], include=["embeddings"])
    back = np.array([row.vector for row in rows], dtype=np.float32)
    print(back.shape)  # (3, 1536)`}</CodeBlock>

      <h2 id="batch-helpers">Batch Embedder Helpers</h2>
      <p>
        An embedder is any callable taking a batch of strings and returning one
        vector per string. The snippet below wraps a single-text model into an
        efficient batched callable you can pass as{" "}
        <code>embedder=</code> when opening a client or creating a collection.
      </p>
      <CodeBlock lang="python">{`def create_batch_embedder(model_func, batch_size=64):
    """Wrap a single-batch embedder into a chunked batched embedder."""
    def embed_batch(texts: list[str]) -> list[list[float]]:
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            results.extend(model_func(chunk))
        return results
    return embed_batch`}</CodeBlock>
    </DocsShell>
  );
}
