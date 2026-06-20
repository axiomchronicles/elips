import type { ReactNode } from "react";

/**
 * Hand-drawn, editorial SVG diagrams. No Mermaid, no stock look — every
 * line is composed by hand and rendered with rough, slightly wobbly
 * geometry so each diagram reads as if pulled from a designer's
 * notebook. Wrap with <SketchCard> for the corner-tick paper frame.
 */
export function SketchCard({
  children,
  caption,
  className = "",
}: {
  children: ReactNode;
  caption?: ReactNode;
  className?: string;
}) {
  return (
    <figure className={`sketch-figure ${className}`}>
      <div className="sketch-frame">
        <span className="sketch-corner sketch-corner-tl" aria-hidden />
        <span className="sketch-corner sketch-corner-tr" aria-hidden />
        <span className="sketch-corner sketch-corner-bl" aria-hidden />
        <span className="sketch-corner sketch-corner-br" aria-hidden />
        {children}
      </div>
      {caption && (
        <figcaption className="sketch-caption">
          <span className="sketch-caption-mark" aria-hidden>✎</span>
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

/* ---------- atoms ---------- */
function RoughRect({
  x, y, w, h, label, sub, accent = false,
}: { x: number; y: number; w: number; h: number; label: string; sub?: string; accent?: boolean }) {
  // slightly wobbly rectangle made from 4 hand-drawn paths
  const r = 10;
  const wobble = (a: number) => a + (Math.sin(a * 1.3) * 0.6);
  const top = `M ${x + r} ${y} L ${x + w - r} ${wobble(y)} Q ${x + w} ${y} ${x + w} ${y + r}`;
  const right = `L ${x + w + 0.5} ${y + h - r} Q ${x + w} ${y + h} ${x + w - r} ${y + h}`;
  const bottom = `L ${x + r} ${y + h - 0.5} Q ${x} ${y + h} ${x} ${y + h - r}`;
  const left = `L ${wobble(x)} ${y + r} Q ${x} ${y} ${x + r} ${y}`;
  return (
    <g>
      <path d={`${top} ${right} ${bottom} ${left}`} className={accent ? "edge accent-stroke" : "edge"} fill={accent ? "var(--color-primary)" : "var(--color-surface)"} fillOpacity={accent ? 1 : 0.65} strokeWidth={accent ? 2 : 1.6} />
      <text x={x + w / 2} y={y + h / 2 + (sub ? -4 : 6)} textAnchor="middle" className="label" style={{ fontSize: 20 }} fill={accent ? "var(--color-on-primary)" : "var(--color-ink)"}>{label}</text>
      {sub && (
        <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" className="sub" fill={accent ? "var(--color-on-primary)" : "var(--color-muted)"} opacity={accent ? 0.9 : 1}>{sub}</text>
      )}
    </g>
  );
}

function Arrow({ d, dashed = false }: { d: string; dashed?: boolean }) {
  return (
    <path d={d} className="edge" strokeDasharray={dashed ? "4 5" : undefined} markerEnd="url(#sk-arrow)" />
  );
}

function Defs() {
  return (
    <defs>
      <marker id="sk-arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-ink)" />
      </marker>
      <marker id="sk-arrow-accent" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-primary)" />
      </marker>
      <pattern id="sk-grid" width="22" height="22" patternUnits="userSpaceOnUse">
        <circle cx="1" cy="1" r="0.6" fill="var(--color-hairline-strong)" opacity="0.4" />
      </pattern>
    </defs>
  );
}

/* ---------- SYSTEM SHAPE: surfaces → instance → vault → branches ---------- */
export function SystemShapeDiagram() {
  return (
    <svg viewBox="0 0 880 460" className="sketch-svg w-full h-auto" role="img" aria-label="ELIPS system shape">
      <Defs />
      <rect width="880" height="460" fill="url(#sk-grid)" opacity="0.5" />

      {/* surface row */}
      <text x="40" y="40" className="sub">surfaces</text>
      <RoughRect x={40}  y={56} w={170} h={62} label="Python · modern" sub="Engine · Arena" />
      <RoughRect x={230} y={56} w={170} h={62} label="Python · core" sub="open · Vault" />
      <RoughRect x={420} y={56} w={170} h={62} label="C++23 SDK" sub="elips::open" />
      <RoughRect x={610} y={56} w={170} h={62} label="elips CLI" sub="ops & repl" />

      {/* converge into instance */}
      <Arrow d="M 125 120 C 125 150, 400 150, 440 180" />
      <Arrow d="M 315 120 C 315 150, 420 150, 440 180" />
      <Arrow d="M 505 120 C 505 150, 460 150, 460 180" />
      <Arrow d="M 695 120 C 695 150, 480 150, 460 180" />

      <RoughRect x={300} y={184} w={280} h={68} label="ElipsInstance" sub="lifecycle · config · WAL · vault registry" accent />

      {/* fan out to vault */}
      <Arrow d="M 440 258 L 440 296" />
      <RoughRect x={300} y={300} w={280} h={64} label="Vault" sub="records · planner · index · metadata" />

      {/* four pillars */}
      <Arrow d="M 340 366 L 200 410" />
      <Arrow d="M 410 366 L 360 410" />
      <Arrow d="M 470 366 L 520 410" />
      <Arrow d="M 540 366 L 680 410" />

      <text x="160" y="430" className="label" style={{ fontSize: 17 }}>query path</text>
      <text x="160" y="446" className="sub">seek / scan / hybrid</text>
      <text x="320" y="430" className="label" style={{ fontSize: 17 }}>text path</text>
      <text x="320" y="446" className="sub">place_document</text>
      <text x="480" y="430" className="label" style={{ fontSize: 17 }}>persistence</text>
      <text x="480" y="446" className="sub">WAL · segments</text>
      <text x="640" y="430" className="label" style={{ fontSize: 17 }}>locking</text>
      <text x="640" y="446" className="sub">flock RO / RW</text>

      {/* handwritten annotation */}
      <g transform="translate(620 210)">
        <path d="M 0 0 C 20 -10, 50 -5, 70 -18" className="edge accent-stroke" strokeWidth={1.6} fill="none" />
        <text x="78" y="-14" className="label" fill="var(--color-primary)" style={{ fontSize: 17 }}>← composition root</text>
      </g>
    </svg>
  );
}

/* ---------- QUERY PATH sequence ---------- */
export function QueryPathDiagram() {
  const lanes = [
    { x: 80,  label: "App" },
    { x: 240, label: "Vault" },
    { x: 400, label: "Planner" },
    { x: 560, label: "MetaIndex" },
    { x: 720, label: "IndexPort" },
  ];
  return (
    <svg viewBox="0 0 820 420" className="sketch-svg w-full h-auto" role="img" aria-label="Query path sequence">
      <Defs />
      {/* lane headers */}
      {lanes.map((l) => (
        <g key={l.label}>
          <rect x={l.x - 50} y={20} width={100} height={36} rx={8} className="node" />
          <text x={l.x} y={43} textAnchor="middle" className="label" style={{ fontSize: 17 }}>{l.label}</text>
          <line x1={l.x} y1={62} x2={l.x} y2={400} className="edge" strokeDasharray="3 5" strokeWidth={1} />
        </g>
      ))}
      {/* messages */}
      <g>
        {[
          { y: 100, from: 0, to: 1, txt: "seek_hybrid(vec, text, filter)" },
          { y: 145, from: 1, to: 2, txt: "plan_seek(...)" },
          { y: 190, from: 2, to: 3, txt: "candidates(filter)" },
          { y: 230, from: 3, to: 2, txt: "id set", dashed: true },
          { y: 275, from: 2, to: 1, txt: "QueryPlan", dashed: true },
          { y: 320, from: 1, to: 4, txt: "search(query, top, ids)" },
          { y: 365, from: 4, to: 1, txt: "ranked hits", dashed: true },
        ].map((m, i) => {
          const a = lanes[m.from].x;
          const b = lanes[m.to].x;
          const dir = b > a ? 1 : -1;
          return (
            <g key={i}>
              <path d={`M ${a + 6 * dir} ${m.y} L ${b - 6 * dir} ${m.y + (i % 2 ? 1.2 : -0.8)}`} className="edge" strokeDasharray={m.dashed ? "5 5" : undefined} markerEnd="url(#sk-arrow)" />
              <text x={(a + b) / 2} y={m.y - 6} textAnchor="middle" className="label" style={{ fontSize: 15 }}>{m.txt}</text>
            </g>
          );
        })}
      </g>
      {/* margin note */}
      <g transform="translate(620 100)">
        <path d="M 0 0 C 30 -8, 70 -12, 95 -6" className="edge accent-stroke" />
        <text x="100" y="-2" className="label" fill="var(--color-primary)" style={{ fontSize: 16 }}>strategy lives here</text>
      </g>
    </svg>
  );
}

/* ---------- PERSISTENCE flow ---------- */
export function PersistenceDiagram() {
  return (
    <svg viewBox="0 0 880 360" className="sketch-svg w-full h-auto" role="img" aria-label="Persistence path">
      <Defs />
      <RoughRect x={30}  y={130} w={150} h={66} label="write" sub="place / erase" />
      <RoughRect x={220} y={130} w={150} h={66} label="WAL append" sub="CRC32C frames" />
      <RoughRect x={410} y={70}  w={170} h={62} label="record store" sub="in-memory" />
      <RoughRect x={410} y={196} w={170} h={62} label="MetadataIndex" sub="equality sets" />
      <RoughRect x={620} y={133} w={140} h={60} label="checkpoint?" sub="threshold" />

      <Arrow d="M 180 163 L 218 163" />
      <Arrow d="M 370 163 C 392 163, 392 100, 408 100" />
      <Arrow d="M 370 163 C 392 163, 392 226, 408 226" />
      <Arrow d="M 580 100 C 605 100, 605 158, 618 158" />
      <Arrow d="M 580 226 C 605 226, 605 168, 618 168" />

      {/* segments row */}
      <text x="40" y="290" className="label" style={{ fontSize: 18 }}>segments</text>
      <g transform="translate(40 300)">
        {[0, 36, 72, 108, 144, 180, 216, 252].map((x, i) => (
          <rect key={i} x={x} y={0} width={30} height={42} rx={4} className="node" />
        ))}
      </g>
      <g transform="translate(360 300)">
        <RoughRect x={0} y={0} w={170} h={42} label="atomic rename" sub="segments.new → segments" accent />
      </g>
      <Arrow d="M 690 195 C 690 260, 540 280, 460 305" />

      <text x="700" y="290" className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 22 }}>truncate WAL ↻</text>
    </svg>
  );
}

/* ---------- CORE-CONCEPTS object model ---------- */
export function ObjectModelDiagram() {
  return (
    <svg viewBox="0 0 880 380" className="sketch-svg w-full h-auto" role="img" aria-label="Object model">
      <Defs />
      <RoughRect x={40} y={150} w={180} h={70} label="ElipsInstance" sub="lifecycle · config · WAL" accent />

      <RoughRect x={300} y={70}  w={160} h={60} label="Vault: documents" />
      <RoughRect x={300} y={230} w={160} h={60} label="Vault: faces" />

      <Arrow d="M 220 175 C 260 175, 260 100, 298 100" />
      <Arrow d="M 220 195 C 260 195, 260 260, 298 260" />

      {/* vault internals */}
      <RoughRect x={520} y={20}  w={150} h={48} label="IndexPort" sub="graph / exact / gpu" />
      <RoughRect x={520} y={80}  w={150} h={48} label="record store" />
      <RoughRect x={520} y={140} w={150} h={48} label="MetadataIndex" />
      <RoughRect x={520} y={200} w={150} h={48} label="Planner" />

      <Arrow d="M 460 100 L 518 44" />
      <Arrow d="M 460 100 L 518 104" />
      <Arrow d="M 460 100 L 518 164" />
      <Arrow d="M 460 100 L 518 224" />

      {/* Record */}
      <RoughRect x={710} y={80} w={150} h={120} label="Record" />
      <g transform="translate(720 130)">
        {["id · vector", "payload", "document?", "chunk?", "lineage?"].map((t, i) => (
          <text key={t} x={0} y={i * 14} className="sub" fill="var(--color-body)" style={{ textTransform: "none", letterSpacing: 0 }}>{t}</text>
        ))}
      </g>
      <Arrow d="M 670 104 L 708 130" />

      {/* annotation */}
      <g transform="translate(70 280)">
        <path d="M 0 0 C 30 10, 90 18, 180 12" className="edge accent-stroke" />
        <text x="0" y="-6" className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 22 }}>one process · one writer</text>
      </g>
    </svg>
  );
}

/* ---------- HNSW LAYERS ---------- */
export function HnswLayersDiagram() {
  const dot = (cx: number, cy: number, label?: string, accent = false) => (
    <g key={`${cx}-${cy}-${label ?? ""}`}>
      <circle cx={cx} cy={cy} r={9} className="node" stroke={accent ? "var(--color-primary)" : undefined} strokeWidth={accent ? 2 : 1.6} />
      {label && <text x={cx} y={cy + 4} textAnchor="middle" className="sub" style={{ fontSize: 10, letterSpacing: 0 }}>{label}</text>}
    </g>
  );
  const link = (x1: number, y1: number, x2: number, y2: number, accent = false) => (
    <path d={`M ${x1} ${y1} Q ${(x1 + x2) / 2} ${(y1 + y2) / 2 - 8} ${x2} ${y2}`} className={accent ? "edge accent-stroke" : "edge"} strokeWidth={accent ? 1.8 : 1} fill="none" />
  );
  return (
    <svg viewBox="0 0 820 460" className="sketch-svg w-full h-auto" role="img" aria-label="HNSW multi-layer proximity graph">
      <Defs />
      <rect width="820" height="460" fill="url(#sk-grid)" opacity="0.5" />

      {/* Layer 2 — sparse */}
      <text x="40" y="60" className="sub">layer 2 · sparse hubs</text>
      {link(180, 95, 540, 95)}
      {dot(180, 95, "a", true)}
      {dot(540, 95, "h", true)}

      {/* Layer 1 */}
      <text x="40" y="180" className="sub">layer 1 · regional</text>
      {link(140, 215, 280, 215)}
      {link(280, 215, 420, 215)}
      {link(420, 215, 560, 215)}
      {link(140, 215, 420, 215)}
      {dot(140, 215, "a")}
      {dot(280, 215, "c")}
      {dot(420, 215, "e")}
      {dot(560, 215, "h")}

      {/* Layer 0 — full */}
      <text x="40" y="310" className="sub">layer 0 · full population</text>
      {[80, 160, 240, 320, 400, 480, 560, 640, 720].map((x, i, arr) => (
        <g key={x}>
          {i < arr.length - 1 && link(x, 345, arr[i + 1], 345)}
          {dot(x, 345)}
        </g>
      ))}
      {link(80, 345, 240, 345)}
      {link(160, 345, 400, 345)}
      {link(320, 345, 560, 345)}
      {link(480, 345, 720, 345)}

      {/* descent arrows */}
      <Arrow d="M 180 110 C 180 150, 140 170, 140 198" />
      <Arrow d="M 140 232 C 140 280, 80 310, 80 332" />

      <g transform="translate(580 360)">
        <path d="M 0 0 C 30 -14, 80 -10, 110 -22" className="edge accent-stroke" strokeWidth={1.6} fill="none" />
        <text x="118" y="-18" className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 22 }}>descend until ef_search converges</text>
      </g>

      <g transform="translate(40 420)">
        <text className="label handwritten" style={{ fontSize: 22 }}>greedy search → climb down → refine</text>
      </g>
    </svg>
  );
}

/* ---------- RECOVERY FLOW ---------- */
export function RecoveryFlowDiagram() {
  return (
    <svg viewBox="0 0 880 360" className="sketch-svg w-full h-auto" role="img" aria-label="Open-time recovery flow">
      <Defs />
      <RoughRect x={20}  y={140} w={120} h={62} label="open()" sub="path · cfg" accent />
      <RoughRect x={170} y={140} w={140} h={62} label="flock LOCK" sub="RW excl · RO shared" />
      <RoughRect x={340} y={60}  w={150} h={56} label="IDENTITY" sub="dim · metric · idx" />
      <RoughRect x={340} y={150} w={150} h={56} label="EMBEDDER manifest" sub="rehydrate" />
      <RoughRect x={340} y={240} w={150} h={56} label="manifest + segs?" sub="vs snapshot" />
      <RoughRect x={520} y={104} w={140} h={56} label="load segments" sub="elips.manifest" />
      <RoughRect x={520} y={196} w={140} h={56} label="load snapshot" sub="elips.snapshot" />
      <RoughRect x={690} y={150} w={150} h={56} label="WAL replay" sub="valid prefix only" />

      <Arrow d="M 140 171 L 168 171" />
      <Arrow d="M 310 171 C 325 171, 325 90, 338 90" />
      <Arrow d="M 310 171 C 325 171, 325 178, 338 178" />
      <Arrow d="M 310 171 C 325 171, 325 268, 338 268" />
      <Arrow d="M 490 268 C 505 268, 505 130, 518 130" />
      <Arrow d="M 490 268 C 505 268, 505 220, 518 220" />
      <Arrow d="M 660 132 C 678 132, 678 168, 690 168" />
      <Arrow d="M 660 224 C 678 224, 678 184, 690 184" />

      <g transform="translate(520 320)">
        <path d="M 0 0 C 40 -8, 110 -6, 170 -16" className="edge accent-stroke" strokeWidth={1.6} fill="none" />
        <text x="180" y="-12" className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 22 }}>RO ↛ attaches no WAL writer</text>
      </g>
    </svg>
  );
}

/* ---------- ON-DISK LAYOUT ---------- */
export function OnDiskLayoutDiagram() {
  const Row = ({ y, name, note, accent }: { y: number; name: string; note: string; accent?: boolean }) => (
    <g transform={`translate(0 ${y})`}>
      <line x1={80} y1={12} x2={110} y2={12} className="edge" strokeWidth={1.2} />
      <rect x={110} y={-2} width={150} height={28} rx={6} className="node" stroke={accent ? "var(--color-primary)" : undefined} strokeWidth={accent ? 1.8 : 1.4} />
      <text x={120} y={16} className="label" style={{ fontSize: 14, fontFamily: "var(--font-mono)" }}>{name}</text>
      <text x={280} y={16} className="sub" style={{ letterSpacing: 0.5, textTransform: "none" }}>{note}</text>
    </g>
  );
  return (
    <svg viewBox="0 0 820 380" className="sketch-svg w-full h-auto" role="img" aria-label="On-disk database layout">
      <Defs />
      <rect x={40} y={30} width={50} height={310} rx={6} fill="none" className="edge accent-stroke" strokeWidth={1.6} strokeDasharray="3 4" />
      <text x="48" y="20" className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 22 }}>/my_db</text>

      <Row y={50}  name="LOCK"                  note="advisory flock — single writer" accent />
      <Row y={88}  name="IDENTITY"              note="dimension · metric · index" />
      <Row y={126} name="TEXT_EMBEDDER.manifest" note="provider · model · fingerprint" />
      <Row y={164} name="wal.log"               note="CRC32C framed mutations" accent />
      <Row y={202} name="elips.manifest"        note="segmented mode root" />
      <Row y={240} name="text_embedder/"        note="rehydratable artifact" />
      <Row y={278} name="segments/"             note="atomically renamed segment files" />
      <Row y={316} name="elips.snapshot"        note="compat single-file layout" />

      <g transform="translate(520 80)">
        <path d="M 0 0 C 40 -16, 110 -12, 170 -28" className="edge accent-stroke" strokeWidth={1.6} fill="none" />
        <text x="180" y="-24" className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 22 }}>one directory = one database</text>
      </g>
    </svg>
  );
}

/* ---------- GPU ENGINE LAYERS ---------- */
export function GpuEngineDiagram() {
  return (
    <svg viewBox="0 0 880 460" className="sketch-svg w-full h-auto" role="img" aria-label="GPU engine layered architecture">
      <Defs />
      <rect width="880" height="460" fill="url(#sk-grid)" opacity="0.4" />

      <text x="40" y="34" className="sub">surface</text>
      <RoughRect x={40} y={50} w={800} h={56} label="Vault · seek / seek_hybrid" sub="domain code is backend-agnostic" accent />

      <text x="40" y="140" className="sub">orchestration</text>
      <RoughRect x={40}  y={156} w={180} h={62} label="GpuSelector" sub="rank backends" />
      <RoughRect x={240} y={156} w={180} h={62} label="GpuDeviceManager" sub="enumerate devices" />
      <RoughRect x={440} y={156} w={200} h={62} label="DynamicBatcher" sub="window_us · max_batch" />
      <RoughRect x={660} y={156} w={180} h={62} label="GpuMemoryManager" sub="pool · pinned" />

      <text x="40" y="252" className="sub">interface segregation</text>
      <RoughRect x={40}  y={268} w={150} h={56} label="GpuPort" sub="compute · top_k" />
      <RoughRect x={210} y={268} w={150} h={56} label="GpuMemoryPort" sub="alloc · pinned" />
      <RoughRect x={380} y={268} w={150} h={56} label="GpuKernelPort" sub="cos · l2 · dot" />
      <RoughRect x={550} y={268} w={150} h={56} label="GpuStreamPort" sub="synchronise" />
      <RoughRect x={720} y={268} w={150} h={56} label="GpuIndexPort" sub="build · search" />

      <text x="40" y="360" className="sub">backends · indexes</text>
      <RoughRect x={40}  y={376} w={130} h={56} label="CUDA" sub="NVIDIA" />
      <RoughRect x={190} y={376} w={130} h={56} label="HIP / ROCm" sub="AMD" />
      <RoughRect x={340} y={376} w={130} h={56} label="Metal" sub="Apple unified" />
      <RoughRect x={490} y={376} w={150} h={56} label="BruteForce · IVF" sub="Flat · PQ" />
      <RoughRect x={660} y={376} w={180} h={56} label="Graph · Hybrid · Dist" sub="CAGRA · multi-GPU" />

      {/* arrows */}
      <Arrow d="M 440 108 L 440 152" />
      <Arrow d="M 130 218 L 115 264" />
      <Arrow d="M 330 218 L 285 264" />
      <Arrow d="M 540 218 L 455 264" />
      <Arrow d="M 540 218 L 625 264" />
      <Arrow d="M 750 218 L 795 264" />
      <Arrow d="M 115 324 L 105 372" />
      <Arrow d="M 285 324 L 255 372" />
      <Arrow d="M 455 324 L 405 372" />
      <Arrow d="M 625 324 L 565 372" />
      <Arrow d="M 795 324 L 750 372" />

      <g transform="translate(580 130)">
        <path d="M 0 0 C 30 -10, 80 -6, 110 -20" className="edge accent-stroke" strokeWidth={1.6} fill="none" />
        <text x="118" y="-16" className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 22 }}>std::expected — failure is expected</text>
      </g>
    </svg>
  );
}

/* ---------- DYNAMIC BATCHER timeline ---------- */
export function DynamicBatcherDiagram() {
  const slots = [
    { t: 30,  q: "Q1" },
    { t: 95,  q: "Q2" },
    { t: 170, q: "Q3" },
    { t: 230, q: "Q4" },
    { t: 320, q: "Q5" },
  ];
  return (
    <svg viewBox="0 0 820 300" className="sketch-svg w-full h-auto" role="img" aria-label="Dynamic batcher coalescing">
      <Defs />
      <text x="40" y="34" className="sub">request stream</text>
      <line x1="40" y1="70" x2="780" y2="70" className="edge" strokeDasharray="4 5" />
      {slots.map((s) => (
        <g key={s.q}>
          <circle cx={60 + s.t * 2} cy={70} r={7} className="node accent-stroke" stroke="var(--color-primary)" strokeWidth={1.8} />
          <text x={60 + s.t * 2} y={56} textAnchor="middle" className="sub" style={{ fontSize: 11 }}>{s.q}</text>
        </g>
      ))}

      {/* batching window */}
      <rect x="80"  y="100" width="260" height="50" rx="10" className="node" stroke="var(--color-ink)" strokeDasharray="4 4" fill="var(--color-surface-strong)" fillOpacity={0.6} />
      <text x="210" y="130" textAnchor="middle" className="label handwritten" style={{ fontSize: 22 }}>window_us (≤500µs)</text>
      <rect x="380" y="100" width="180" height="50" rx="10" className="node" stroke="var(--color-ink)" strokeDasharray="4 4" fill="var(--color-surface-strong)" fillOpacity={0.6} />
      <text x="470" y="130" textAnchor="middle" className="label handwritten" style={{ fontSize: 22 }}>next window</text>

      <Arrow d="M 210 152 L 210 200" />
      <Arrow d="M 470 152 L 470 200" />

      <RoughRect x={130} y={210} w={160} h={56} label="batched launch" sub="Q1+Q2+Q3" accent />
      <RoughRect x={390} y={210} w={160} h={56} label="batched launch" sub="Q4+Q5" accent />

      <g transform="translate(600 230)">
        <text className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 22 }}>1 kernel ≫ N kernels</text>
      </g>
    </svg>
  );
}

/* ---------- TRANSACTION LIFECYCLE state machine ---------- */
export function TransactionLifecycleDiagram() {
  return (
    <svg viewBox="0 0 880 380" className="sketch-svg w-full h-auto" role="img" aria-label="Transaction lifecycle state machine">
      <Defs />
      <rect width="880" height="380" fill="url(#sk-grid)" opacity="0.4" />

      <text x="40" y="36" className="sub">entry</text>
      <RoughRect x={40} y={50} w={180} h={60} label="begin_transaction()" sub="db.begin_transaction()" accent />

      <Arrow d="M 230 80 L 290 80" />

      <RoughRect x={300} y={50} w={180} h={60} label="ACTIVE" sub="done_=false · ops_=[]" />

      {/* self-loop enqueue */}
      <path d="M 390 50 C 360 10, 420 10, 405 48" className="edge accent-stroke" markerEnd="url(#sk-arrow-accent)" fill="none" />
      <text x="395" y="14" textAnchor="middle" className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 18 }}>enqueue place/erase</text>

      {/* fan to three outcomes */}
      <Arrow d="M 480 80 C 540 80, 560 80, 620 80" />
      <Arrow d="M 390 110 C 390 170, 390 180, 390 210" />
      <Arrow d="M 480 90 C 540 130, 600 170, 660 210" dashed />

      <RoughRect x={620} y={50} w={200} h={60} label="commit()" sub="for op : ops_ → WAL+apply" accent />
      <RoughRect x={300} y={216} w={180} h={60} label="ROLLED BACK" sub="ops_.clear() · done_=T" />
      <RoughRect x={620} y={216} w={200} h={60} label="~Transaction()" sub="auto-rollback if !done_" />

      <Arrow d="M 720 110 C 720 150, 720 170, 720 210" />

      {/* terminal */}
      <RoughRect x={300} y={310} w={520} h={56} label="TERMINAL · done_=true" sub="no further ops · safe to destroy" />
      <Arrow d="M 390 276 L 420 308" />
      <Arrow d="M 720 276 L 700 308" />
      <Arrow d="M 720 110 C 760 200, 760 280, 700 310" dashed />

      <g transform="translate(40 320)">
        <text className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 22 }}>RAII: no done_ → rollback ✓</text>
      </g>
    </svg>
  );
}

/* ---------- ERROR HIERARCHY ---------- */
export function ErrorHierarchyDiagram({
  root = "std::runtime_error",
  base = "elips::ElipsError",
  leaves,
  asides = [],
}: { root?: string; base?: string; leaves: string[]; asides?: string[] }) {
  const cx = 440;
  const baseY = 130;
  const leafY = 240;
  const cols = leaves.length;
  const colW = 760 / Math.max(cols, 1);
  return (
    <svg viewBox="0 0 880 360" className="sketch-svg w-full h-auto" role="img" aria-label="Error hierarchy">
      <Defs />
      <rect width="880" height="360" fill="url(#sk-grid)" opacity="0.4" />
      <RoughRect x={cx - 110} y={30} w={220} h={56} label={root} sub="std" />
      <Arrow d={`M ${cx} 88 L ${cx} ${baseY - 4}`} />
      <RoughRect x={cx - 130} y={baseY} w={260} h={56} label={base} sub="every elips throw inherits this" accent />

      {leaves.map((l, i) => {
        const x = 60 + i * colW + colW / 2;
        return (
          <g key={l}>
            <path d={`M ${cx} ${baseY + 60} C ${cx} ${(baseY + leafY) / 2}, ${x} ${(baseY + leafY) / 2 - 10}, ${x} ${leafY}`} className="edge" markerEnd="url(#sk-arrow)" fill="none" />
            <g transform={`translate(${x - 70} ${leafY})`}>
              <rect x={0} y={0} width={140} height={44} rx={8} className="node" />
              <text x={70} y={26} textAnchor="middle" className="label" style={{ fontSize: 14, fontFamily: "var(--font-mono)" }}>{l}</text>
            </g>
          </g>
        );
      })}

      {asides.length > 0 && (
        <g transform="translate(40 308)">
          <text className="sub">parallel surfaces</text>
          {asides.map((a, i) => (
            <text key={a} x={0} y={20 + i * 16} className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 18 }}>↳ {a}</text>
          ))}
        </g>
      )}
    </svg>
  );
}

/* ---------- WHY VECTOR DB ---------- */
export function WhyVectorDBDiagram() {
  return (
    <svg viewBox="0 0 880 460" className="sketch-svg w-full h-auto" role="img" aria-label="Why a vector database">
      <Defs />
      <rect width="880" height="460" fill="url(#sk-grid)" opacity="0.4" />

      {/* left: raw world */}
      <text x="40" y="34" className="sub">your data</text>
      <RoughRect x={40}  y={50}  w={180} h={48} label="documents" sub="md · pdf · code" />
      <RoughRect x={40}  y={108} w={180} h={48} label="chats" sub="memory · logs" />
      <RoughRect x={40}  y={166} w={180} h={48} label="images / audio" sub="multimodal" />
      <RoughRect x={40}  y={224} w={180} h={48} label="user actions" sub="events · traces" />

      {/* embedder */}
      <Arrow d="M 220 74  C 260 74, 260 240, 300 240" />
      <Arrow d="M 220 132 C 260 132, 260 240, 300 240" />
      <Arrow d="M 220 190 C 260 190, 260 240, 300 240" />
      <Arrow d="M 220 248 L 300 248" />

      <RoughRect x={300} y={220} w={180} h={60} label="embedder" sub="text → vector ∈ ℝᵈ" accent />

      <Arrow d="M 480 250 L 540 250" />

      {/* vector space */}
      <g transform="translate(540 130)">
        <rect x={0} y={0} width={300} height={240} rx={14} className="node" />
        <text x={150} y={22} textAnchor="middle" className="sub">vector space · ℝᵈ</text>
        {[
          [60, 70], [85, 95], [110, 60], [140, 110], [70, 140],
          [210, 80], [240, 110], [225, 60], [260, 90],
          [120, 180], [150, 200], [95, 200], [180, 175],
        ].map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={5} className="node accent-stroke" stroke={i < 5 ? "var(--color-primary)" : "var(--color-ink)"} strokeWidth={1.6} />
        ))}
        {/* query */}
        <circle cx={95} cy={100} r={9} fill="var(--color-primary)" />
        <circle cx={95} cy={100} r={28} className="edge accent-stroke" fill="none" strokeDasharray="3 4" />
        <text x={102} y={92} className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 18 }}>query</text>
        <text x={130} y={140} className="sub" style={{ fontSize: 11 }}>top-k nearest = "meaning-similar"</text>
      </g>

      {/* bottom annotation */}
      <g transform="translate(40 330)">
        <path d="M 0 0 C 100 -20, 320 -10, 500 -22" className="edge accent-stroke" strokeWidth={1.6} fill="none" />
        <text x="0" y="20" className="label handwritten" style={{ fontSize: 22 }}>
          SQL asks "equals?" — vectors ask "close to?"
        </text>
        <text x="0" y="42" className="sub" style={{ fontSize: 12, letterSpacing: 0.5 }}>
          retrieval by similarity is the only way agents recall prior context, code, tickets, memory.
        </text>
      </g>
    </svg>
  );
}

/* ---------- AGENTIC FLOW with ELIPS ---------- */
export function AgenticFlowDiagram() {
  return (
    <svg viewBox="0 0 880 460" className="sketch-svg w-full h-auto" role="img" aria-label="Agentic flow with ELIPS">
      <Defs />
      <rect width="880" height="460" fill="url(#sk-grid)" opacity="0.4" />

      <RoughRect x={40}  y={40}  w={160} h={60} label="user turn" sub="prompt · tool call" />
      <RoughRect x={240} y={40}  w={180} h={60} label="agent loop" sub="LLM · planner" accent />
      <RoughRect x={460} y={40}  w={180} h={60} label="retrieval step" sub="seek_hybrid(q, k)" />
      <RoughRect x={680} y={40}  w={160} h={60} label="LLM call" sub="grounded reply" />

      <Arrow d="M 200 70 L 238 70" />
      <Arrow d="M 420 70 L 458 70" />
      <Arrow d="M 640 70 L 678 70" />

      {/* down to ELIPS */}
      <Arrow d="M 550 100 L 550 160" />

      <RoughRect x={300} y={160} w={460} h={70} label="ELIPS · in-process retrieval" sub="vault · planner · index · WAL" accent />

      {/* fan out memory types */}
      <Arrow d="M 360 230 L 240 290" />
      <Arrow d="M 460 230 L 400 290" />
      <Arrow d="M 560 230 L 560 290" />
      <Arrow d="M 660 230 L 720 290" />

      <RoughRect x={140} y={290} w={200} h={56} label="episodic memory" sub="past turns · summaries" />
      <RoughRect x={320} y={290} w={180} h={56} label="semantic memory" sub="facts · entities" />
      <RoughRect x={520} y={290} w={180} h={56} label="document corpus" sub="docs · chunks · lineage" />
      <RoughRect x={720} y={290} w={140} h={56} label="tool traces" sub="exec results" />

      {/* writeback loop */}
      <path d="M 720 70 C 800 200, 820 380, 720 400" className="edge accent-stroke" fill="none" markerEnd="url(#sk-arrow-accent)" strokeDasharray="5 5" />
      <text x="820" y="240" className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 18 }} transform="rotate(90 820 240)">write back · place_document</text>

      <g transform="translate(40 410)">
        <text className="label handwritten" style={{ fontSize: 22 }}>retrieve → reason → respond → remember</text>
      </g>
    </svg>
  );
}

/* ---------- LOW-LEVEL AGENTIC SYSTEM with ELIPS ---------- */
export function ElipsAgenticSystemDiagram() {
  return (
    <svg viewBox="0 0 900 540" className="sketch-svg w-full h-auto" role="img" aria-label="Low-level system design: agentic flow on ELIPS">
      <Defs />
      <rect width="900" height="540" fill="url(#sk-grid)" opacity="0.4" />

      {/* row 1: orchestrator */}
      <text x="30" y="30" className="sub">layer 1 · orchestration</text>
      <RoughRect x={30}  y={46} w={200} h={56} label="Agent runtime" sub="loop · tool router" accent />
      <RoughRect x={250} y={46} w={200} h={56} label="Tool registry" sub="search · code · fs · http" />
      <RoughRect x={470} y={46} w={200} h={56} label="Policy / guard" sub="quota · safety" />
      <RoughRect x={690} y={46} w={180} h={56} label="Telemetry" sub="spans · evals" />

      {/* row 2: retrieval boundary */}
      <text x="30" y="130" className="sub">layer 2 · retrieval boundary (RAG bus)</text>
      <RoughRect x={30}  y={146} w={840} h={56} label="ElipsClient — seek · seek_text · seek_hybrid · explain_seek" sub="one process · no network hop" accent />

      <Arrow d="M 130 102 L 130 144" />
      <Arrow d="M 350 102 L 350 144" />
      <Arrow d="M 570 102 L 570 144" />
      <Arrow d="M 780 102 L 780 144" />

      {/* row 3: vaults */}
      <text x="30" y="230" className="sub">layer 3 · vaults (per memory class)</text>
      <RoughRect x={30}  y={246} w={200} h={70} label="vault: episodic" sub="turn embeddings + meta" />
      <RoughRect x={250} y={246} w={200} h={70} label="vault: semantic" sub="entity cards" />
      <RoughRect x={470} y={246} w={200} h={70} label="vault: corpus" sub="docs · chunks" />
      <RoughRect x={690} y={246} w={180} h={70} label="vault: tools" sub="trace embeddings" />

      <Arrow d="M 130 202 L 130 244" />
      <Arrow d="M 350 202 L 350 244" />
      <Arrow d="M 570 202 L 570 244" />
      <Arrow d="M 780 202 L 780 244" />

      {/* row 4: engine */}
      <text x="30" y="340" className="sub">layer 4 · engine</text>
      <RoughRect x={30}  y={356} w={160} h={60} label="Planner" sub="strategy · filters" />
      <RoughRect x={210} y={356} w={160} h={60} label="IndexPort" sub="hnsw · exact · gpu" />
      <RoughRect x={390} y={356} w={170} h={60} label="MetadataIndex" sub="equality sets" />
      <RoughRect x={580} y={356} w={140} h={60} label="Embedder" sub="local · hosted" />
      <RoughRect x={740} y={356} w={130} h={60} label="GpuPort" sub="batch · stream" />

      <Arrow d="M 130 316 L 110 354" />
      <Arrow d="M 350 316 L 290 354" />
      <Arrow d="M 570 316 L 470 354" />
      <Arrow d="M 780 316 L 650 354" />
      <Arrow d="M 780 316 L 800 354" />

      {/* row 5: persistence */}
      <text x="30" y="450" className="sub">layer 5 · persistence</text>
      <RoughRect x={30}  y={466} w={200} h={56} label="WAL (CRC32C)" sub="every mutation" accent />
      <RoughRect x={250} y={466} w={200} h={56} label="segments/ (atomic rename)" sub="checkpoint" />
      <RoughRect x={470} y={466} w={200} h={56} label="LOCK (flock)" sub="single writer" />
      <RoughRect x={690} y={466} w={180} h={56} label="text_embedder/" sub="rehydratable" />

      <Arrow d="M 110 416 L 130 464" />
      <Arrow d="M 290 416 L 350 464" />
      <Arrow d="M 470 416 L 570 464" />
      <Arrow d="M 800 416 L 780 464" />

      {/* annotation */}
      <g transform="translate(580 520)">
        <text className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 20 }}>no daemons · no sidecars · agent owns the bytes</text>
      </g>
    </svg>
  );
}

/* ---------- GPU ACCELERATION at-a-glance ---------- */
export function GpuAccelGlanceDiagram() {
  return (
    <svg viewBox="0 0 880 320" className="sketch-svg w-full h-auto" role="img" aria-label="GPU acceleration at a glance">
      <Defs />
      <rect width="880" height="320" fill="url(#sk-grid)" opacity="0.4" />

      <RoughRect x={30}  y={120} w={170} h={70} label="N CPU queries" sub="latency-bound" />
      <Arrow d="M 200 155 L 240 155" />
      <RoughRect x={240} y={120} w={200} h={70} label="DynamicBatcher" sub="window_us · max_batch" accent />
      <Arrow d="M 440 155 L 480 155" />
      <RoughRect x={480} y={120} w={180} h={70} label="GpuPort kernel" sub="cos · l2 · dot" />
      <Arrow d="M 660 155 L 700 155" />
      <RoughRect x={700} y={120} w={150} h={70} label="top-k results" sub="merged" accent />

      <g transform="translate(40 40)">
        <text className="sub">why this wins</text>
        <text x="0" y="22" className="label handwritten" style={{ fontSize: 20 }}>1 launch ≫ N launches · one HBM trip · saturated SMs</text>
      </g>

      <g transform="translate(240 220)">
        <text className="sub">backends</text>
        <text x="0" y="22" className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 20 }}>CUDA · HIP / ROCm · Metal — selected by GpuSelector</text>
      </g>
    </svg>
  );
}

/* ---------- ALGORITHM PORTS overview ---------- */
export function AlgorithmPortsDiagram() {
  return (
    <svg viewBox="0 0 880 360" className="sketch-svg w-full h-auto" role="img" aria-label="Algorithm ports">
      <Defs />
      <rect width="880" height="360" fill="url(#sk-grid)" opacity="0.4" />
      <RoughRect x={300} y={30} w={280} h={56} label="IndexPort" sub="build · upsert · search · erase" accent />
      <Arrow d="M 380 86 L 220 150" />
      <Arrow d="M 440 86 L 440 150" />
      <Arrow d="M 500 86 L 660 150" />
      <RoughRect x={120} y={150} w={200} h={70} label="HNSW (graph)" sub="M · ef_construction · ef_search" />
      <RoughRect x={340} y={150} w={200} h={70} label="Exact (flat)" sub="brute force · deterministic" />
      <RoughRect x={560} y={150} w={200} h={70} label="GPU family" sub="brute · ivf · pq · cagra" />
      <text x="220" y="250" textAnchor="middle" className="sub">log-ish recall@k</text>
      <text x="440" y="250" textAnchor="middle" className="sub">100% recall, O(N) per query</text>
      <text x="660" y="250" textAnchor="middle" className="sub">batched, async, GpuPort-bound</text>
      <g transform="translate(80 300)">
        <text className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 20 }}>one contract — the planner doesn't care which</text>
      </g>
    </svg>
  );
}

/* ---------- TUTORIAL ROADMAP (used on tutorial hub) ---------- */
export function TutorialRoadmapDiagram() {
  const stops = [
    "install", "first DB", "records", "place / erase",
    "seek + filters", "documents", "hybrid", "EQL",
    "txns", "recovery", "config", "C++ hello",
    "C++ vault", "embedder", "GPU", "ship it",
  ];
  return (
    <svg viewBox="0 0 880 360" className="sketch-svg w-full h-auto" role="img" aria-label="Tutorial roadmap">
      <Defs />
      <rect width="880" height="360" fill="url(#sk-grid)" opacity="0.4" />
      {/* winding path */}
      <path d="M 40 90 C 200 30, 360 160, 520 80 S 820 200, 840 130" className="edge accent-stroke" strokeWidth={2} fill="none" strokeDasharray="2 6" />
      <path d="M 40 230 C 200 170, 360 300, 520 220 S 820 340, 840 270" className="edge accent-stroke" strokeWidth={2} fill="none" strokeDasharray="2 6" />

      {stops.map((s, i) => {
        const row = i < 8 ? 0 : 1;
        const col = i % 8;
        const x = 50 + col * 100;
        const y = row === 0 ? 90 + (col % 2 === 0 ? 0 : 30) : 230 + (col % 2 === 0 ? 0 : 30);
        return (
          <g key={s}>
            <circle cx={x} cy={y} r={14} className="node accent-stroke" stroke="var(--color-primary)" strokeWidth={1.8} />
            <text x={x} y={y + 4} textAnchor="middle" className="label" style={{ fontSize: 11 }}>{i + 1}</text>
            <text x={x} y={y + 32} textAnchor="middle" className="sub" style={{ fontSize: 10, letterSpacing: 0.4, textTransform: "none" }}>{s}</text>
          </g>
        );
      })}
      <g transform="translate(40 330)">
        <text className="label handwritten" fill="var(--color-primary)" style={{ fontSize: 20 }}>16 lessons · Python first, C++ where it matters</text>
      </g>
    </svg>
  );
}

