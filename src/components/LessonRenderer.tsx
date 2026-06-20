import type { ReactNode } from "react";
import type { LessonBlock, DiagramName } from "../lib/tutorial";
import { CodeBlock } from "./Code";
import {
  SketchCard,
  SystemShapeDiagram,
  QueryPathDiagram,
  PersistenceDiagram,
  ObjectModelDiagram,
  HnswLayersDiagram,
  RecoveryFlowDiagram,
  OnDiskLayoutDiagram,
  GpuEngineDiagram,
  DynamicBatcherDiagram,
  TransactionLifecycleDiagram,
  WhyVectorDBDiagram,
  AgenticFlowDiagram,
  ElipsAgenticSystemDiagram,
  GpuAccelGlanceDiagram,
  AlgorithmPortsDiagram,
  TutorialRoadmapDiagram,
} from "./SketchDiagram";

const DIAGRAMS: Record<DiagramName, () => ReactNode> = {
  SystemShape: () => <SystemShapeDiagram />,
  QueryPath: () => <QueryPathDiagram />,
  Persistence: () => <PersistenceDiagram />,
  ObjectModel: () => <ObjectModelDiagram />,
  HnswLayers: () => <HnswLayersDiagram />,
  RecoveryFlow: () => <RecoveryFlowDiagram />,
  OnDiskLayout: () => <OnDiskLayoutDiagram />,
  GpuEngine: () => <GpuEngineDiagram />,
  DynamicBatcher: () => <DynamicBatcherDiagram />,
  TransactionLifecycle: () => <TransactionLifecycleDiagram />,
  WhyVectorDB: () => <WhyVectorDBDiagram />,
  AgenticFlow: () => <AgenticFlowDiagram />,
  ElipsAgenticSystem: () => <ElipsAgenticSystemDiagram />,
  GpuAccelGlance: () => <GpuAccelGlanceDiagram />,
  AlgorithmPorts: () => <AlgorithmPortsDiagram />,
  TutorialRoadmap: () => <TutorialRoadmapDiagram />,
};

export function LessonBlocks({ blocks }: { blocks: LessonBlock[] }) {
  return (
    <>
      {blocks.map((b, i) => {
        switch (b.kind) {
          case "lede":
            return (
              <p key={i} className="handwritten-lede text-body my-6">
                {b.text}
              </p>
            );
          case "p":
            return (
              <p key={i} className="text-[16px] text-ink leading-relaxed my-5">
                {b.text}
              </p>
            );
          case "h":
            return (
              <h2 key={i} className="display-md text-ink mt-12 mb-4">
                {b.text}
              </h2>
            );
          case "code":
            return (
              <div key={i} className="my-6">
                <CodeBlock lang={b.lang} filename={b.filename}>
                  {b.code}
                </CodeBlock>
              </div>
            );
          case "diagram": {
            const D = DIAGRAMS[b.name];
            return (
              <div key={i} className="my-8">
                <SketchCard caption={b.caption}>
                  <D />
                </SketchCard>
              </div>
            );
          }
          case "note":
            return (
              <aside
                key={i}
                className="my-6 rounded-lg border border-hairline-strong bg-canvas-soft px-5 py-4 text-[15px] text-body"
              >
                <span className="handwritten text-primary mr-2" style={{ fontSize: 20 }}>
                  ✎ note
                </span>
                {b.text}
              </aside>
            );
          case "ul":
            return (
              <ul
                key={i}
                className="my-5 list-disc pl-6 text-[16px] text-ink leading-relaxed space-y-1.5"
              >
                {b.items.map((it) => (
                  <li key={it}>{it}</li>
                ))}
              </ul>
            );
          default:
            return null;
        }
      })}
    </>
  );
}
