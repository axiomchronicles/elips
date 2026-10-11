import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/benchmarks")({
  head: () => ({
    meta: [
      { title: "Benchmarks & Performance Suite — ELIPS Documentation" },
      {
        name: "description",
        content:
          "Reproducible ELIPS ANN benchmarks — recall@10, QPS, p50/p99 latency — measured against faiss on Apple M4, plus the Metal bf_knn exact path and the elips bench CLI.",
      },
    ],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell
      eyebrow="Reference"
      title="Benchmarks & Performance Suite"
      toc={[
        { id: "overview", label: "Overview" },
        { id: "metrics", label: "Key metrics" },
        { id: "ann-results", label: "ANN results (M4)" },
        { id: "ivf-tradeoffs", label: "IVF nprobe trade-offs" },
        { id: "cpu-vs-gpu", label: "CPU vs. Metal GPU" },
        { id: "cli-benchmarking", label: "Running benchmarks" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS ships a reproducible benchmark harness (<code>benchmarks/bench_ann.py</code>) that
        measures recall@k and QPS against an exact-Flat sanity gate, with faiss as the reference. The
        numbers below are from that harness — reported as measured, wins or losses.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        Environment: Apple M4 (arm64), macOS, Python 3.13, ELIPS 1.1.0 (GPU bindings on),
        faiss-cpu 1.15.1. Settings: <code>dim=128</code>, <code>nb=100000</code>,{" "}
        <code>nq=1000</code>, <code>k=10</code>, metric euclidean (L2). Ground truth is exact numpy
        L2 brute force; the exact-Flat sanity gate passes. QPS and p50/p99 are from single-query
        (non-batched) search unless noted.
      </p>

      <h2 id="metrics">Key metrics</h2>
      <ul>
        <li>
          <strong>Recall@10:</strong> fraction of the top-10 approximate results that match the
          exact L2 ground truth.
        </li>
        <li>
          <strong>QPS:</strong> queries per second (single-query search unless a row says batched).
        </li>
        <li>
          <strong>p50 / p99:</strong> per-query latency percentiles in milliseconds.
        </li>
        <li>
          <strong>Build s:</strong> index construction time.
        </li>
      </ul>

      <h2 id="ann-results">ANN results (Apple M4)</h2>
      <table>
        <thead>
          <tr>
            <th>Config</th>
            <th>Recall@10</th>
            <th>QPS</th>
            <th>p50 ms</th>
            <th>p99 ms</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>ELIPS IndexFlat (cpu)</td>
            <td>1.000</td>
            <td>489</td>
            <td>1.981</td>
            <td>3.336</td>
          </tr>
          <tr>
            <td>ELIPS bf_knn (metal, 1-query)</td>
            <td>1.000</td>
            <td>24</td>
            <td>41.315</td>
            <td>69.498</td>
          </tr>
          <tr>
            <td>ELIPS bf_knn (metal, batched)</td>
            <td>1.000</td>
            <td>914</td>
            <td>1.094</td>
            <td>1.094</td>
          </tr>
          <tr>
            <td>ELIPS IVFFlat nlist=316 nprobe=128 (cpu)</td>
            <td>0.801</td>
            <td>1,616</td>
            <td>0.617</td>
            <td>0.650</td>
          </tr>
          <tr>
            <td>faiss IndexFlatL2 (cpu)</td>
            <td>1.000</td>
            <td>989</td>
            <td>0.926</td>
            <td>1.432</td>
          </tr>
          <tr>
            <td>faiss IVFFlat nlist=316 nprobe=128 (cpu)</td>
            <td>0.793</td>
            <td>2,587</td>
            <td>0.385</td>
            <td>0.416</td>
          </tr>
        </tbody>
      </table>
      <p>
        ELIPS and faiss both use the FAISS-style Index API (<code>IndexFlat</code> /{" "}
        <code>IndexIVFFlat</code>, <code>metric="l2"</code>); labels are row indices, so recall is
        measured against the same ground truth.
      </p>

      <h2 id="ivf-tradeoffs">IVF nprobe trade-offs</h2>
      <p>
        For <code>IndexIVFFlat</code> with <code>nlist=316</code>, raising <code>nprobe</code> trades
        QPS for recall (ELIPS, cpu, M4):
      </p>
      <table>
        <thead>
          <tr>
            <th>nprobe</th>
            <th>Recall@10</th>
            <th>QPS</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>1</td>
            <td>0.030</td>
            <td>46,614</td>
          </tr>
          <tr>
            <td>8</td>
            <td>0.152</td>
            <td>14,123</td>
          </tr>
          <tr>
            <td>32</td>
            <td>0.381</td>
            <td>6,273</td>
          </tr>
          <tr>
            <td>128</td>
            <td>0.801</td>
            <td>1,616</td>
          </tr>
        </tbody>
      </table>

      <h2 id="cpu-vs-gpu">CPU vs. Metal GPU</h2>
      <p>
        The only implemented GPU backend is Metal (Apple silicon); it accelerates the exact
        brute-force path, <code>elips.bf_knn(..., device="metal")</code>. Batching matters: a single
        query over Metal is latency-bound (24 QPS), but amortised over one batched call it reaches
        914 QPS at recall 1.000 — comparable to the CPU exact path (<code>IndexFlat</code>, 489 QPS)
        while leaving the CPU free. CUDA and HIP backends are experimental and not benchmarked here.
      </p>

      <h2 id="cli-benchmarking">Running benchmarks (elips bench)</h2>
      <p>Run the built-in benchmarking tool directly from the command line:</p>
      <CodeBlock lang="bash">{`# Reproduce the ANN suite
python benchmarks/bench_ann.py

# Quick engine micro-benchmark via the CLI
elips bench /path/to/db --count 100000 --dim 128`}</CodeBlock>
    </DocsShell>
  );
}
