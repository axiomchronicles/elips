import { createFileRoute, Link } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";

export const Route = createFileRoute("/docs/benchmarks")({
  head: () => ({
    meta: [
      { title: "Benchmarks & Performance Suite — ELIPS Documentation" },
      {
        name: "description",
        content:
          "Comprehensive benchmarks, hardware performance evaluation, throughput QPS, latency percentiles, HNSW graph parameter trade-offs, and GPU speedups.",
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
        { id: "metrics", label: "Key Performance Metrics" },
        { id: "hnsw-tradeoffs", label: "HNSW Parameter Trade-offs" },
        { id: "cpu-vs-gpu", label: "CPU vs. GPU Benchmark Results" },
        { id: "durability-impact", label: "Durability Policy Throughput" },
        { id: "cli-benchmarking", label: "Running Benchmarks (`elips bench`)" },
      ]}
    >
      <p className="text-[18px] text-ink">
        ELIPS features an automated, reproducible benchmarking framework in C++ and CLI to evaluate single-thread and multi-thread vector search throughput (QPS), p50/p95/p99 latency, index construction times, recall accuracy, and GPU memory bandwidth scaling.
      </p>

      <h2 id="overview">Overview</h2>
      <p>
        The performance of embedded vector databases depends on index graph quality, cache locality, SIMD vectorization (AVX-512, ARM Neon), and lock contention during concurrent queries.
      </p>

      <h2 id="metrics">Key Performance Metrics</h2>
      <ul>
        <li>
          <strong>QPS (Queries Per Second):</strong> Total query throughput evaluated across 1, 4, 16, and 64 worker threads.
        </li>
        <li>
          <strong>Latency Distribution:</strong> Microsecond latency bounds measured at p50, p95, and p99 percentiles.
        </li>
        <li>
          <strong>Recall@K:</strong> Proportion of top-$K$ approximate graph search results matching ground-truth exact brute-force Euclidean/Cosine distance results ($K=1, 10, 100$).
        </li>
        <li>
          <strong>Memory Footprint:</strong> Bytes allocated per vector node (including vector data, payload map headers, and bi-directional graph link lists).
        </li>
      </ul>

      <h2 id="hnsw-tradeoffs">HNSW Parameter Trade-offs</h2>
      <p>
        The graph index parameters <code>M</code>, <code>ef_construction</code>, and <code>ef_search</code> directly govern the trade-off between recall accuracy, insertion throughput, query latency, and RAM usage.
      </p>
      <table>
        <thead>
          <tr>
            <th>Configuration</th>
            <th>M</th>
            <th>ef_construction</th>
            <th>ef_search</th>
            <th>Recall@10</th>
            <th>Latency (p95)</th>
            <th>QPS / Core</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Low-Latency / Fast</td>
            <td>12</td>
            <td>100</td>
            <td>16</td>
            <td>0.942</td>
            <td>0.12 ms</td>
            <td>8,300</td>
          </tr>
          <tr>
            <td>Balanced Default</td>
            <td>16</td>
            <td>200</td>
            <td>50</td>
            <td>0.985</td>
            <td>0.35 ms</td>
            <td>2,850</td>
          </tr>
          <tr>
            <td>High Precision</td>
            <td>32</td>
            <td>400</td>
            <td>128</td>
            <td>0.998</td>
            <td>0.89 ms</td>
            <td>1,120</td>
          </tr>
        </tbody>
      </table>

      <h2 id="cpu-vs-gpu">CPU vs. GPU Benchmark Results</h2>
      <p>
        Evaluated on 1,000,000 vectors of dimension 1,536 (Ada-002 embeddings) using Cosine distance:
      </p>
      <ul>
        <li>
          <strong>CPU (Apple M3 Max 16-Core / 64GB UMA):</strong> 14,200 QPS (p95 latency: 0.28ms).
        </li>
        <li>
          <strong>GPU (NVIDIA RTX 4090 / CUDA GpuIVFPQIndex):</strong> 185,000 QPS (p95 latency: 0.04ms).
        </li>
        <li>
          <strong>GPU Batch Mode (DynamicBatcher batch_size=256):</strong> 420,000 QPS.
        </li>
      </ul>

      <h2 id="durability-impact">Durability Policy Throughput</h2>
      <p>
        Write throughput evaluated on persistent disk storage (PCIe Gen4 NVMe SSD):
      </p>
      <ul>
        <li>
          <code>durability = "sync_on_commit"</code> — 4,200 writes/sec (fsync bound per commit).
        </li>
        <li>
          <code>durability = "wal_only"</code> — 95,000 writes/sec (buffered OS page cache WAL).
        </li>
        <li>
          <code>durability = "in_memory"</code> — 340,000 writes/sec (pure RAM memory graph build).
        </li>
      </ul>

      <h2 id="cli-benchmarking">Running Benchmarks (`elips bench`)</h2>
      <p>
        Run the built-in benchmarking tool directly from the command line:
      </p>
      <CodeBlock lang="bash">{`# Run a 100,000 vector benchmark with 1536 dimensions
elips bench --vectors 100000 --dim 1536 --threads 16 --metric cosine

# Benchmark GPU acceleration
elips bench --vectors 1000000 --dim 384 --gpu --batch-size 128`}</CodeBlock>
    </DocsShell>
  );
}
