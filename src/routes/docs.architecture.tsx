import { createFileRoute } from "@tanstack/react-router";
import { DocsShell } from "../components/Chrome";
import { CodeBlock } from "../components/Code";
import { SketchCard, SystemShapeDiagram, QueryPathDiagram, PersistenceDiagram } from "../components/SketchDiagram";

export const Route = createFileRoute("/docs/architecture")({
  head: () => ({
    meta: [
      { title: "Architecture — ELIPS Docs" },
      { name: "description", content: "How ElipsInstance, Vault, the planner, and the persistence layer fit together in ELIPS." },
      { property: "og:title", content: "Architecture — ELIPS" },
      { property: "og:description", content: "A tour of the ELIPS runtime — engines, vaults, planner, indexes, persistence, and locking." },
      { property: "og:url", content: "/docs/architecture" },
    ],
    links: [{ rel: "canonical", href: "/docs/architecture" }],
  }),
  component: Page,
});

function Page() {
  return (
    <DocsShell eyebrow="Concepts · Architecture" title="Architecture" toc={[
      { id: "shape", label: "System shape" },
      { id: "ports", label: "Hexagonal ports" },
      { id: "query", label: "Query path" },
      { id: "persistence", label: "Persistence path" },
      { id: "concurrency", label: "Concurrency" },
      { id: "surfaces", label: "Surfaces" },
    ]}>
      <p className="text-[18px] text-ink">
        ELIPS is a small core with sharp seams. Domain code talks to
        abstractions — <code>IndexPort</code>, <code>GpuPort</code>,{" "}
        <code>TextEmbedderPort</code>, <code>WAL</code> — never to concrete
        engines. This is the lever that lets HNSW, exact search, GPU
        indexes, and future quantized variants compose behind one runtime.
      </p>

      <h2 id="shape">System shape</h2>
      <SketchCard caption="Vertical slice. SDK surfaces sit above the instance; storage and locking sit beneath every vault.">
        <SystemShapeDiagram />
      </SketchCard>

      <h2 id="ports">Hexagonal ports</h2>
      <p>
        Every replaceable subsystem hides behind a pure-virtual port.
        Concrete engines plug in; nothing outside <code>src/gpu_engine/</code>{" "}
        ever includes a backend header. The composition root is
        <code> IndexFactory::make_index()</code>.
      </p>
      <CodeBlock lang="cpp">
{`class Vault {
  std::unique_ptr<IndexPort> index_;   // abstract
  WAL*                       wal_;     // non-owning
};

auto index = make_index(config, dimension);   // returns unique_ptr<IndexPort>`}
      </CodeBlock>

      <h2 id="query">Query path</h2>
      <SketchCard caption="A vector or hybrid query always passes through the planner before touching an index.">
        <QueryPathDiagram />
      </SketchCard>
      <p>
        Strategies emitted by the planner: <code>ann_index</code>,{" "}
        <code>exact_candidates</code>, <code>full_scan</code>,{" "}
        <code>text_probe</code>, <code>hybrid_fusion</code>. The strategy is
        the single source of truth for "what actually ran" — exposed
        through <code>explain_seek</code> and through EQL.
      </p>

      <h2 id="persistence">Persistence path</h2>
      <SketchCard caption="Every mutation appends to the WAL before the in-memory store is touched.">
        <PersistenceDiagram />
      </SketchCard>
      <p>
        See <a href="/docs/storage">Storage &amp; recovery</a> for the
        on-disk layout, the WAL operation set, and recovery rules.
      </p>

      <h2 id="concurrency">Concurrency</h2>
      <p>
        ELIPS is single-writer, multi-reader. The writer takes an
        exclusive <code>flock</code> on the database's <code>LOCK</code>{" "}
        file; readers take a shared lock. Locks are RAII-bound to{" "}
        <code>LockManager</code>: dropping the handle releases the lock,
        even on exception unwinding. There is no background thread, no
        daemon, no cross-process coordination beyond the lock.
      </p>

      <h2 id="surfaces">Surfaces</h2>
      <p>
        Two Python layers live on top of the same core:
      </p>
      <ul>
        <li><strong>Low-level bindings</strong> mirror the C++ surface: <code>open</code>, <code>Database</code>, <code>Vault</code>, <code>Config</code>, <code>Filter</code>, <code>QueryPlan</code>.</li>
        <li><strong>Modern wrapper</strong> adds typed text-first ergonomics: <code>connect</code>, <code>Engine</code>, <code>Arena</code>, <code>RecordInput</code>, <code>Row</code>, <code>Hit</code>.</li>
      </ul>
      <p>
        The wrapper prefers native core text APIs when the database has a
        resolved embedder. If only a Python callable is provided, it falls
        back to Python-side embedding plus <code>seek_hybrid</code>. If
        neither exists, text-first calls raise <code>ConfigError</code> —
        ELIPS never silently degrades to lexical-only behaviour.
      </p>
    </DocsShell>
  );
}
