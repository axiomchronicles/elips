import type { ReactNode } from "react";

export function StandalonePage({
  eyebrow,
  title,
  lede,
  children,
}: {
  eyebrow?: string;
  title: string;
  lede?: string;
  children: ReactNode;
}) {
  return (
    <div className="max-w-[1200px] mx-auto px-6 pt-16 pb-24 fade-up">
      <div className="max-w-[820px]">
        {eyebrow && <div className="eyebrow mb-4">{eyebrow}</div>}
        <h1 className="display-lg text-ink">{title}</h1>
        {lede && <p className="text-[18px] text-body mt-5 leading-relaxed">{lede}</p>}
      </div>
      <div className="mt-12 max-w-[820px] prose">{children}</div>
    </div>
  );
}
