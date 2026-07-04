import type { ReactNode } from "react";

export function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="panel">
      <h2 className="panel-title">{title}</h2>
      {children}
    </section>
  );
}
