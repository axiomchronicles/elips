import { createFileRoute, Link } from "@tanstack/react-router";
import { Lightbulb } from "lucide-react";
import { CodeBlock } from "../components/Code";
import {
  SketchCard,
  WhyVectorDBDiagram,
  AgenticFlowDiagram,
  ElipsAgenticSystemDiagram,
  GpuAccelGlanceDiagram,
  AlgorithmPortsDiagram,
} from "../components/SketchDiagram";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "ELIPS — Embedded vector & document retrieval, in your process" },
      {
        name: "description",
        content:
          "ELIPS is an in-process vector and document engine in C++23: HNSW + exact ANN, hybrid retrieval, document lineage, WAL recovery, and Python + C++ SDKs.",
      },
      { property: "og:title", content: "ELIPS — Embedded vector & document retrieval" },
      {
        property: "og:description",
        content:
          "Local-first ANN, hybrid retrieval, document lineage, and segmented persistence — embedded in your process.",
      },
      { property: "og:url", content: "/" },
    ],
    links: [{ rel: "canonical", href: "/" }],
  }),
  component: Home,
});

function Home() {
  return (
    <div>
      {/* HERO */}
      <section className="max-w-[1200px] mx-auto px-6 pt-6 md:pt-10 pb-16">
        <div className="max-w-[860px]">
          <h1
            className="handwritten text-ink fade-up-2"
            style={{
              fontSize: "clamp(54px, 8.4vw, 104px)",
              lineHeight: 1.02,
              letterSpacing: "-0.005em",
            }}
          >
            An embedded retrieval engine
            <br />
            for vectors <span className="text-primary">&amp;</span> documents.
          </h1>
          <p
            className="text-[18px] md:text-[20px] text-body mt-7 max-w-[640px] fade-up-3"
            style={{ lineHeight: 1.55 }}
          >
            ELIPS is the local, in-process layer beneath your application. ANN and exact indexes,
            first-class document lineage, hybrid retrieval, WAL recovery, segmented persistence —
            without running a separate service.
          </p>
          <div className="mt-9 flex flex-wrap items-center gap-3 fade-up-3">
            <Link to="/docs" className="btn btn-ink">
              Read the docs
            </Link>
            <Link to="/chat" className="btn btn-ghost">
              <Lightbulb size={14} aria-hidden /> Ask AI
            </Link>
            <a
              href="https://github.com/axiomchronicles/elips"
              target="_blank"
              rel="noreferrer"
              className="btn btn-text"
            >
              View on GitHub →
            </a>
          </div>
        </div>

        {/* "IDE mockup" — editorial code surface */}
        <div className="mt-16 fade-up-3">
          <CodeBlock lang="python" filename="quickstart.py">
            {`import elips

engine = elips.connect(":memory:", dimension=128)
arena = engine.arena("documents")

arena.ingest(
    texts=["alpha design note", "beta incident runbook"],
    meta=[{"kind": "design"}, {"kind": "ops"}],
)

for hit in arena.probe_text("alpha", top=2):
    print(hit.key, hit.distance, hit.text, hit.meta)`}
          </CodeBlock>
        </div>
      </section>

      {/* DESIGN NOTEBOOK — hand-drawn architecture inspiration */}
      <section className="max-w-[1200px] mx-auto px-6 py-20 hairline-t">
        <div className="flex items-end justify-between flex-wrap gap-4 mb-10">
          <div>
            <div className="eyebrow mb-3">From the notebook</div>
            <h2 className="display-lg text-ink max-w-[640px]">
              The shape of ELIPS, <span className="handwritten text-primary">sketched out</span>.
            </h2>
          </div>
          <p className="text-body text-[15px] max-w-[380px]">
            Architecture decisions worth illustrating. Every diagram in the docs is rendered the
            same way — hand-drawn, editorial, never stock.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {/* Notebook card 1 — Hexagonal ports */}
          <figure className="sketch-frame">
            <span className="sketch-corner sketch-corner-tl" aria-hidden />
            <span className="sketch-corner sketch-corner-tr" aria-hidden />
            <span className="sketch-corner sketch-corner-bl" aria-hidden />
            <span className="sketch-corner sketch-corner-br" aria-hidden />
            <svg
              viewBox="0 0 360 240"
              className="w-full h-auto sketch-svg"
              role="img"
              aria-label="Hexagonal ports diagram"
            >
              <g fill="none" strokeLinecap="round" strokeLinejoin="round">
                <polygon
                  points="180,40 250,80 250,160 180,200 110,160 110,80"
                  className="edge"
                  strokeWidth="2"
                  fill="var(--color-surface)"
                />
                <text
                  x="180"
                  y="118"
                  textAnchor="middle"
                  className="handwritten"
                  fontSize="26"
                  fill="var(--color-ink)"
                >
                  Vault
                </text>
                <text
                  x="180"
                  y="138"
                  textAnchor="middle"
                  fontSize="10"
                  fill="var(--color-muted)"
                  style={{ letterSpacing: "0.08em", textTransform: "uppercase" }}
                >
                  core domain
                </text>
                {/* ports */}
                <circle cx="110" cy="80" r="6" className="accent" />
                <circle cx="250" cy="80" r="6" className="accent" />
                <circle cx="110" cy="160" r="6" className="accent" />
                <circle cx="250" cy="160" r="6" className="accent" />
                {/* port tick lines */}
                <path d="M 100 70 L 80 56" className="edge" />
                <path d="M 260 70 L 280 56" className="edge" />
                <path d="M 100 170 L 80 184" className="edge" />
                <path d="M 260 170 L 280 184" className="edge" />
                {/* labels — pushed clear of the hex vertices */}
                <text
                  x="78"
                  y="50"
                  textAnchor="end"
                  className="handwritten"
                  fontSize="17"
                  fill="var(--color-ink)"
                >
                  IndexPort
                </text>
                <text
                  x="282"
                  y="50"
                  textAnchor="start"
                  className="handwritten"
                  fontSize="17"
                  fill="var(--color-ink)"
                >
                  WAL
                </text>
                <text
                  x="78"
                  y="200"
                  textAnchor="end"
                  className="handwritten"
                  fontSize="17"
                  fill="var(--color-ink)"
                >
                  GpuPort
                </text>
                <text
                  x="282"
                  y="200"
                  textAnchor="start"
                  className="handwritten"
                  fontSize="17"
                  fill="var(--color-ink)"
                >
                  Embedder
                </text>
              </g>
            </svg>
            <figcaption className="sketch-caption">
              <span className="sketch-caption-mark" aria-hidden>
                ✎
              </span>
              Hexagonal ports: nothing imports an engine
            </figcaption>
          </figure>

          {/* Notebook card 2 — WAL → segments */}
          <figure className="sketch-frame">
            <span className="sketch-corner sketch-corner-tl" aria-hidden />
            <span className="sketch-corner sketch-corner-tr" aria-hidden />
            <span className="sketch-corner sketch-corner-bl" aria-hidden />
            <span className="sketch-corner sketch-corner-br" aria-hidden />
            <svg
              viewBox="0 0 320 220"
              className="w-full h-auto"
              role="img"
              aria-label="Write-ahead log to segments"
            >
              <g
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-ink"
              >
                <rect x="20" y="40" width="80" height="50" rx="8" />
                <text
                  x="60"
                  y="68"
                  textAnchor="middle"
                  className="handwritten"
                  fontSize="16"
                  fill="currentColor"
                  stroke="none"
                >
                  write
                </text>
                <text
                  x="60"
                  y="82"
                  textAnchor="middle"
                  fontSize="9"
                  fill="currentColor"
                  stroke="none"
                  opacity=".7"
                >
                  place / erase
                </text>
                <path d="M 105 65 C 130 65, 130 65, 145 65" markerEnd="url(#arrow1)" />
                <rect x="150" y="40" width="80" height="50" rx="8" />
                <text
                  x="190"
                  y="62"
                  textAnchor="middle"
                  className="handwritten"
                  fontSize="14"
                  fill="currentColor"
                  stroke="none"
                >
                  WAL
                </text>
                <text
                  x="190"
                  y="80"
                  textAnchor="middle"
                  fontSize="9"
                  fill="currentColor"
                  stroke="none"
                  opacity=".7"
                >
                  CRC32C frames
                </text>
                <path d="M 235 65 C 260 65, 260 65, 275 65" markerEnd="url(#arrow1)" />
                <circle cx="295" cy="65" r="10" fill="#f54e00" stroke="none" />
                <text x="295" y="69" textAnchor="middle" fontSize="11" fill="white" stroke="none">
                  M
                </text>
                {/* segments */}
                <text
                  x="20"
                  y="135"
                  className="handwritten"
                  fontSize="14"
                  fill="currentColor"
                  stroke="none"
                >
                  checkpoint ↓
                </text>
                <g transform="translate(20 150)">
                  {[0, 38, 76, 114, 152, 190, 228].map((x, i) => (
                    <rect key={i} x={x} y="0" width="32" height="46" rx="4" />
                  ))}
                </g>
                <text
                  x="170"
                  y="208"
                  textAnchor="middle"
                  fontSize="11"
                  fill="currentColor"
                  stroke="none"
                  opacity=".7"
                >
                  segments — atomic rename
                </text>
                <defs>
                  <marker
                    id="arrow1"
                    markerWidth="10"
                    markerHeight="10"
                    refX="8"
                    refY="5"
                    orient="auto"
                  >
                    <path d="M0 0 L10 5 L0 10 z" fill="currentColor" />
                  </marker>
                </defs>
              </g>
            </svg>
            <figcaption className="sketch-caption">
              <span className="sketch-caption-mark" aria-hidden>
                ✎
              </span>
              WAL first, segments later, recovery always
            </figcaption>
          </figure>

          {/* Notebook card 3 — hybrid fusion */}
          <figure className="sketch-frame">
            <span className="sketch-corner sketch-corner-tl" aria-hidden />
            <span className="sketch-corner sketch-corner-tr" aria-hidden />
            <span className="sketch-corner sketch-corner-bl" aria-hidden />
            <span className="sketch-corner sketch-corner-br" aria-hidden />
            <svg
              viewBox="0 0 320 220"
              className="w-full h-auto"
              role="img"
              aria-label="Hybrid fusion"
            >
              <g
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-ink"
              >
                <ellipse cx="90" cy="80" rx="60" ry="40" />
                <text
                  x="90"
                  y="78"
                  textAnchor="middle"
                  className="handwritten"
                  fontSize="17"
                  fill="currentColor"
                  stroke="none"
                >
                  vector
                </text>
                <text
                  x="90"
                  y="95"
                  textAnchor="middle"
                  fontSize="10"
                  fill="currentColor"
                  stroke="none"
                  opacity=".7"
                >
                  ANN distance
                </text>
                <ellipse cx="200" cy="80" rx="60" ry="40" />
                <text
                  x="200"
                  y="78"
                  textAnchor="middle"
                  className="handwritten"
                  fontSize="17"
                  fill="currentColor"
                  stroke="none"
                >
                  lexical
                </text>
                <text
                  x="200"
                  y="95"
                  textAnchor="middle"
                  fontSize="10"
                  fill="currentColor"
                  stroke="none"
                  opacity=".7"
                >
                  overlap score
                </text>
                <path d="M 145 80 Q 145 145 145 175" />
                <path d="M 145 175 C 145 188, 155 188, 165 188 L 235 188" />
                <rect x="235" y="165" width="70" height="42" rx="8" fill="#f54e00" stroke="none" />
                <text
                  x="270"
                  y="183"
                  textAnchor="middle"
                  className="handwritten"
                  fontSize="14"
                  fill="white"
                  stroke="none"
                >
                  fused
                </text>
                <text
                  x="270"
                  y="200"
                  textAnchor="middle"
                  fontSize="10"
                  fill="white"
                  stroke="none"
                  opacity=".9"
                >
                  SearchResult[]
                </text>
              </g>
            </svg>
            <figcaption className="sketch-caption">
              <span className="sketch-caption-mark" aria-hidden>
                ✎
              </span>
              Two scores, one planner, one ranking
            </figcaption>
          </figure>
        </div>
      </section>

      {/* TIMELINE — signature pastel pills */}
      <section className="max-w-[1200px] mx-auto px-6 py-20 hairline-t">
        <div className="eyebrow mb-4">Inside a hybrid query</div>
        <h2 className="display-lg text-ink max-w-[760px]">
          Every retrieval walks a small, inspectable pipeline.
        </h2>
        <p className="text-body text-[17px] mt-5 max-w-[680px]">
          The planner emits a <code className="font-mono text-[14px]">QueryPlan</code> with a
          strategy, candidate set, metadata acceleration flags, and any text component. You can read
          it directly — in Python or C++.
        </p>

        <div className="mt-10 grid md:grid-cols-5 gap-3">
          {[
            { p: "pill-thinking", l: "Plan", d: "Resolve filters, choose strategy." },
            { p: "pill-grep", l: "Filter", d: "Narrow via MetadataIndex equality sets." },
            { p: "pill-read", l: "Probe", d: "ANN, exact, or hybrid fusion." },
            { p: "pill-edit", l: "Rank", d: "Re-sort by distance or metadata." },
            { p: "pill-done", l: "Yield", d: "Project requested fields and return." },
          ].map((s) => (
            <div key={s.l} className="rounded-lg border border-hairline p-5 bg-surface">
              <span className={`pill ${s.p}`}>{s.l}</span>
              <p className="text-[14px] text-body mt-3 leading-relaxed">{s.d}</p>
            </div>
          ))}
        </div>
      </section>

      {/* FEATURES */}
      <section className="max-w-[1200px] mx-auto px-6 py-24">
        <div className="grid md:grid-cols-12 gap-12">
          <div className="md:col-span-4">
            <div className="eyebrow mb-3">Why ELIPS</div>
            <h2 className="display-lg text-ink">
              A retrieval primitive,
              <br /> not a service.
            </h2>
          </div>
          <div className="md:col-span-8 grid sm:grid-cols-2 gap-10">
            {[
              {
                t: "Embedded by design",
                d: "One process. Advisory file locks coordinate readers and writers. No daemons, no sidecars.",
              },
              {
                t: "Document-aware records",
                d: "Every record may carry text, chunk coordinates, and embedding lineage — restored across restarts.",
              },
              {
                t: "ANN and exact, behind one port",
                d: "HNSW and an exact index plug into the same IndexPort. GPU indexes follow the same contract.",
              },
              {
                t: "Hybrid retrieval",
                d: "seek_text, seek_hybrid, and EQL share one planner. Lexical overlap fuses with vector distance.",
              },
              {
                t: "Crash-safe WAL",
                d: "Every mutation appends with CRC32C before the in-memory store changes. Corrupt tails truncate cleanly.",
              },
              {
                t: "Inspectable planner",
                d: "explain_seek returns the strategy, candidate set, and acceleration flags used by the query.",
              },
            ].map((f) => (
              <div key={f.t}>
                <h3 className="text-[17px] font-semibold text-ink">{f.t}</h3>
                <p className="text-[15px] text-body mt-2 leading-relaxed">{f.d}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* DUAL SDK BAND */}
      <section className="bg-canvas-soft hairline-t hairline-b py-24">
        <div className="max-w-[1200px] mx-auto px-6">
          <div className="eyebrow mb-3">Two SDKs, one core</div>
          <h2 className="display-lg text-ink max-w-[760px]">
            The same runtime, in the language you reach for.
          </h2>
          <div className="grid md:grid-cols-2 gap-6 mt-10">
            <div>
              <div className="eyebrow mb-3">Python</div>
              <CodeBlock lang="python" filename="python_sdk.py">
                {`import elips

db = elips.open("/tmp/elips", dimension=128, metric="cosine")
docs = db.vault("documents")
docs.place_document("alpha design note", {"kind": "design"})
hit = docs.seek_text("alpha", top=1)[0]
print(hit.document.text, hit.distance)`}
              </CodeBlock>
            </div>
            <div>
              <div className="eyebrow mb-3">C++23</div>
              <CodeBlock lang="cpp" filename="quickstart.cpp">
                {`#include "elips/elips.hpp"

auto db = elips::open(
    ":memory:",
    elips::Config{}.dimension(128).metric(elips::Metric::cosine));

auto& docs = db->vault("documents");
docs.place_document("alpha design note",
                    {{"kind", std::string{"design"}}});

auto hits = docs.seek_text("alpha", 1);`}
              </CodeBlock>
            </div>
          </div>
        </div>
      </section>

      {/* WHY A VECTOR DB */}
      <section className="max-w-[1200px] mx-auto px-6 py-24 hairline-t">
        <div className="grid md:grid-cols-12 gap-12 items-start">
          <div className="md:col-span-5">
            <div className="eyebrow mb-3">Why a vector database</div>
            <h2 className="display-lg text-ink">
              SQL asks <span className="handwritten text-primary">"equals?"</span>
              <br />
              Vectors ask <span className="handwritten text-primary">"close to?"</span>
            </h2>
            <p className="handwritten-lede mt-5 text-body">
              Meaning lives in geometry. Embeddings turn language, code, and user behavior into
              points in ℝᵈ, and the only useful question becomes <em>"what's near this point?"</em>{" "}
              — the question relational databases were never built to answer.
            </p>
            <p className="text-body text-[15px] mt-4 leading-relaxed">
              A vector database is the missing primitive between raw embeddings and the agents,
              search bars, and recommenders that consume them. ELIPS makes it small enough to embed
              and durable enough to trust.
            </p>
          </div>
          <div className="md:col-span-7">
            <SketchCard caption="Embed once, search by proximity forever. The query is just another point in the same space.">
              <WhyVectorDBDiagram />
            </SketchCard>
          </div>
        </div>
      </section>

      {/* AGENTIC FLOW */}
      <section className="bg-canvas-soft hairline-t hairline-b py-24">
        <div className="max-w-[1200px] mx-auto px-6">
          <div className="eyebrow mb-3">Agentic flow</div>
          <h2 className="display-lg text-ink max-w-[820px]">
            Agents need memory. ELIPS{" "}
            <span className="handwritten text-primary">is the memory</span>.
          </h2>
          <p className="handwritten-lede mt-5 text-body max-w-[760px]">
            Every useful agent loop ends the same way — retrieve, reason, respond, remember. ELIPS
            lives inside the loop, not across a network boundary, so the retrieval step costs
            microseconds and the write-back is just another function call.
          </p>

          <div className="mt-10 grid md:grid-cols-12 gap-6">
            <div className="md:col-span-7">
              <SketchCard caption="Retrieve → reason → respond → remember. The whole cycle inside one process.">
                <AgenticFlowDiagram />
              </SketchCard>
            </div>
            <div className="md:col-span-5 grid gap-4">
              {[
                {
                  t: "Episodic memory",
                  d: "Every turn is embedded and placed back into a vault for the next session.",
                },
                {
                  t: "Semantic memory",
                  d: "Entity cards and stable facts live in their own vault and survive restarts.",
                },
                {
                  t: "Document corpus",
                  d: "PDFs, code, tickets — chunked with lineage so citations are exact.",
                },
                {
                  t: "Tool traces",
                  d: "Past tool outputs are searchable, so the agent learns from its own runs.",
                },
              ].map((m) => (
                <div key={m.t} className="rounded-lg border border-hairline p-5 bg-surface">
                  <h3 className="text-[16px] font-semibold text-ink">{m.t}</h3>
                  <p className="text-[14px] text-body mt-2 leading-relaxed">{m.d}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* LOW-LEVEL SYSTEM DESIGN */}
      <section className="max-w-[1200px] mx-auto px-6 py-24">
        <div className="eyebrow mb-3">System design</div>
        <h2 className="display-lg text-ink max-w-[820px]">
          A low-level look at an <span className="handwritten text-primary">agent stack</span> on
          ELIPS.
        </h2>
        <p className="handwritten-lede mt-5 text-body max-w-[760px]">
          Five layers, no sidecars. The agent runtime calls one client; the client talks to vaults;
          vaults route through the planner and the ports; everything terminates in a WAL frame and a
          segment on disk.
        </p>
        <div className="mt-10">
          <SketchCard caption="Five layers from orchestration to persistence — the agent owns the bytes the whole way down.">
            <ElipsAgenticSystemDiagram />
          </SketchCard>
        </div>
      </section>

      {/* ALGORITHMS + GPU */}
      <section className="bg-canvas-soft hairline-t hairline-b py-24">
        <div className="max-w-[1200px] mx-auto px-6">
          <div className="grid md:grid-cols-2 gap-10">
            <div>
              <div className="eyebrow mb-3">Algorithms</div>
              <h2 className="display-lg text-ink">
                One <span className="handwritten text-primary">contract</span>, three engines.
              </h2>
              <p className="handwritten-lede mt-5 text-body">
                HNSW for scale, exact for ground truth, GPU for throughput — all behind{" "}
                <code>IndexPort</code>. The planner never branches on which one is mounted.
              </p>
              <div className="mt-6">
                <SketchCard caption="The planner sees one shape — IndexPort. Recall/latency trade-offs are a config switch, not a rewrite.">
                  <AlgorithmPortsDiagram />
                </SketchCard>
              </div>
            </div>
            <div>
              <div className="eyebrow mb-3">GPU acceleration</div>
              <h2 className="display-lg text-ink">
                Coalesce, launch <span className="handwritten text-primary">once</span>, ship.
              </h2>
              <p className="handwritten-lede mt-5 text-body">
                A dynamic batcher gathers concurrent queries inside a tiny window and fires a single
                kernel. One HBM trip, saturated SMs, std::expected on every fallible call.
              </p>
              <div className="mt-6">
                <SketchCard caption="DynamicBatcher turns N CPU-side queries into one GPU launch — the only honest way to amortise PCIe.">
                  <GpuAccelGlanceDiagram />
                </SketchCard>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* TUTORIAL TEASER */}
      <section className="max-w-[1200px] mx-auto px-6 py-24">
        <div className="rounded-2xl border border-hairline-strong bg-surface p-10 md:p-14 relative overflow-hidden">
          <div className="eyebrow mb-3">Learn ELIPS, end to end</div>
          <h2 className="display-lg text-ink max-w-[760px]">
            The <span className="handwritten text-primary">sixteen-lesson</span> tutorial.
          </h2>
          <p className="handwritten-lede mt-5 text-body max-w-[680px]">
            From <code>pip install</code> to GPU-accelerated production serving, with sketched
            diagrams and runnable Python + C++ on every page.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link to="/docs/tutorial" className="btn btn-ink">
              Start the tutorial
            </Link>
            <Link
              to="/docs/tutorial/$lesson"
              params={{ lesson: "01-installation" }}
              className="btn btn-ghost"
            >
              Lesson 1 →
            </Link>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-28">
        <div className="max-w-[1200px] mx-auto px-6 text-center">
          <h2 className="display-lg text-ink max-w-[720px] mx-auto">
            Embed it once. Forget it ships with your binary.
          </h2>
          <div className="mt-8 flex justify-center gap-3">
            <Link to="/docs/installation" className="btn btn-primary">
              Install ELIPS
            </Link>
            <Link to="/docs/architecture" className="btn btn-ghost">
              Read the architecture
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
