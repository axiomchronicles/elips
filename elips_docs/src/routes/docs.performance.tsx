import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/performance")({
  head: () => ({
    meta: [
      { title: "Performance — ELIPS Docs" },
      {
        name: "description",
        content:
          "Tuning HNSW, choosing exact vs graph, GPU batching, and benchmarking with elips bench.",
      },
      { property: "og:title", content: "Performance — ELIPS" },
      { property: "og:description", content: "Tuning and benchmarking ELIPS." },
      { property: "og:url", content: "/docs/performance" },
    ],
    links: [{ rel: "canonical", href: "/docs/performance" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Practice"
      title="Performance"
      toc={[
        { id: "metrics", label: "What to measure" },
        { id: "tune", label: "Tuning HNSW" },
        { id: "metadata", label: "Plan for filters" },
        { id: "io", label: "Durability cost" },
        { id: "gpu", label: "GPU batching" },
      ]}
    >
      <h2 id="metrics">What to measure</h2>
      <p>
        Three numbers carry most decisions: <strong>recall@k</strong> against an exact baseline,{" "}
        <strong>p95 query latency</strong> at your real top-k, and <strong>write throughput</strong>{" "}
        at your real batch size. Anything else is downstream of these.
      </p>
      <CodeBlock lang="bash">{`elips bench /tmp/bench_db --count 100000 --dim 768`}</CodeBlock>

      <h2 id="tune">Tuning HNSW</h2>
      <table>
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Default</th>
            <th>Trade-off</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <code>M</code>
            </td>
            <td>
              <code>16</code>
            </td>
            <td>Higher → better recall, more memory.</td>
          </tr>
          <tr>
            <td>
              <code>ef_construction</code>
            </td>
            <td>
              <code>200</code>
            </td>
            <td>Higher → better graph, slower writes.</td>
          </tr>
          <tr>
            <td>
              <code>ef_search</code>
            </td>
            <td>
              <code>64</code>
            </td>
            <td>Higher → better recall, slower reads.</td>
          </tr>
        </tbody>
      </table>
      <p>
        Use <code>ExactIndex</code> at the same query set as a ground-truth baseline when sweeping{" "}
        <code>ef_search</code>.
      </p>

      <h2 id="metadata">Plan for filters</h2>
      <p>
        Equality and set-membership predicates accelerate through <code>MetadataIndex</code>. When
        the candidate set is small enough the planner switches to <code>exact_candidates</code> —
        often faster than running ANN over the full population. Inspect with{" "}
        <code>explain_seek</code>.
      </p>

      <h2 id="io">Durability cost</h2>
      <p>
        <code>paranoid</code> calls <code>fsync</code> per write; throughput is bounded by your
        storage's sync rate. <code>standard</code> drops the fsync but keeps the flush.{" "}
        <code>relaxed</code> buffers and amortises at checkpoint — pick it when crash recovery to
        the most recent checkpoint is acceptable.
      </p>

      <h2 id="gpu">GPU batching</h2>
      <p>
        With <code>ELIPS_GPU_ENABLED</code>, the dynamic batcher coalesces concurrent queries into a
        single GPU launch. Increase the in-flight query window in your serving layer to give the
        batcher room to work; very low concurrency leaves the GPU under-fed.
      </p>
    </DocsShell>
  );
}
