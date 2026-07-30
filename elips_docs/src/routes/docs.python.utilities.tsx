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
          "Complete reference for ELIPS Python vector utilities: cosine similarity, L2 distance, normalization, numpy interoperability, and batch embedder helpers.",
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
        { id: "vector-math", label: "Vector Distance & Math" },
        { id: "normalization", label: "Vector Normalization" },
        { id: "numpy-interop", label: "NumPy Interoperability" },
        { id: "batch-helpers", label: "Batch Embedder Helpers" },
        { id: "code-examples", label: "Code Examples" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS exports optimized C++ vector math functions and NumPy interoperability helpers in <code>elips.utils</code>.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        The <code>elips.utils</code> module provides pure vector operations, metric distance calculators, and zero-copy conversion routines to bridge Python numerical computing libraries (like NumPy and PyTorch) with the ELIPS C++ core.
      </p>

      <h2 id="vector-math">Vector Distance & Math</h2>
      <CodeBlock lang="python">{`import elips.utils as utils

vec_a = [0.1, 0.5, 0.9, 0.4]
vec_b = [0.2, 0.4, 0.8, 0.5]

# Distance and similarity calculators
cos_sim = utils.cosine_similarity(vec_a, vec_b)
l2_dist = utils.l2_distance(vec_a, vec_b)
dot_prod = utils.dot_product(vec_a, vec_b)

print(f"Cosine Similarity: {cos_sim:.4f}")
print(f"L2 Distance:        {l2_dist:.4f}")
print(f"Dot Product:        {dot_prod:.4f}")`}</CodeBlock>

      <h2 id="normalization">Vector Normalization</h2>
      <p>
        Normalizes a floating point vector to unit length ($L_2$ norm = 1.0):
      </p>
      <CodeBlock lang="python">{`norm_vec = utils.normalize([3.0, 4.0])
print(norm_vec)  # [0.6, 0.8]`}</CodeBlock>

      <h2 id="numpy-interop">NumPy Interoperability</h2>
      <p>
        Convert seamlessly between Python lists, NumPy 2D matrices, and ELIPS vectors:
      </p>
      <CodeBlock lang="python">{`import numpy as np
import elips.utils as utils

# Convert 2D NumPy array of shape (1000, 1536) to list of floats
matrix = np.random.randn(1000, 1536).astype(np.float32)

# Convert single row to ELIPS vector
vec = utils.from_numpy(matrix[0])

# Convert ELIPS search result vectors back to NumPy array
matrix_back = utils.to_numpy([hit.record.vector for hit in hits])`}</CodeBlock>

      <h2 id="batch-helpers">Batch Embedder Helpers</h2>
      <CodeBlock lang="python">{`def create_batch_embedder(model_func, batch_size=64):
    """
    Wraps a single-text embedder function into an efficient batched embedder.
    """
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
