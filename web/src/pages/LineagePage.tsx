import { useEffect, useMemo, useReducer, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { openRunEvents } from "../api";
import { DiffView } from "../components/DiffView";
import { Panel } from "../components/Panel";
import {
  GAP_X,
  layoutLineage,
  NODE_H,
  NODE_W,
} from "../state/lineage";
import { initialRunState, runReducer } from "../state/runReducer";

const PAD = 16;

export function LineagePage() {
  const { runId } = useParams<{ runId: string }>();
  const [state, dispatch] = useReducer(runReducer, initialRunState);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return;
    const source = openRunEvents(runId, dispatch);
    return () => source.close();
  }, [runId]);

  const layout = useMemo(
    () => layoutLineage(state.nodes, state.edges),
    [state.nodes, state.edges],
  );
  const pos = useMemo(
    () => Object.fromEntries(layout.nodes.map((n) => [n.id, n])),
    [layout],
  );

  const node = selected ? state.nodes[selected] : null;
  const parent =
    node && node.parents.length > 0 ? state.nodes[node.parents[0]] : null;

  return (
    <div className="split">
      <div className="grow">
        <Panel title={`Lineage — Run ${runId ?? ""}`}>
          <div className="dim" style={{ marginBottom: 8 }}>
            <Link to={`/runs/${runId}`}>← BACK TO RUN</Link>
            {"  ·  "}
            {layout.nodes.length} candidates, {layout.layers} layers
          </div>
          {layout.nodes.length === 0 ? (
            <div className="dim">no candidates yet</div>
          ) : (
            <div style={{ overflow: "auto" }}>
              <svg
                width={layout.width + 2 * PAD}
                height={layout.height + 2 * PAD}
                data-testid="lineage-svg"
              >
                <g transform={`translate(${PAD},${PAD})`}>
                  {layout.edges.map((e, i) => {
                    const a = pos[e.from];
                    const b = pos[e.to];
                    if (!a || !b) return null;
                    const x1 = a.x + NODE_W;
                    const y1 = a.y + NODE_H / 2;
                    const x2 = b.x;
                    const y2 = b.y + NODE_H / 2;
                    const mx = x1 + GAP_X / 2;
                    return (
                      <path
                        key={i}
                        d={`M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`}
                        fill="none"
                        stroke="var(--border)"
                        strokeWidth={1}
                      />
                    );
                  })}
                  {layout.nodes.map((n) => {
                    const data = state.nodes[n.id];
                    return (
                      <g
                        key={n.id}
                        className={`lineage-node${selected === n.id ? " selected" : ""}`}
                        transform={`translate(${n.x},${n.y})`}
                        onClick={() => setSelected(n.id)}
                      >
                        <rect width={NODE_W} height={NODE_H} strokeWidth={1} />
                        <text x={8} y={17}>
                          {n.id.slice(0, 14)}
                        </text>
                        <text x={8} y={33} className="score">
                          {data.score !== null ? data.score.toFixed(4) : "·"}
                          {data.source ? `  ${data.source}` : ""}
                        </text>
                      </g>
                    );
                  })}
                </g>
              </svg>
            </div>
          )}
        </Panel>
      </div>

      <div className="side">
        <Panel title="Candidate">
          {!node ? (
            <div className="dim">select a node</div>
          ) : (
            <>
              <div className="stat-cards" style={{ marginBottom: 12 }}>
                <div className="stat-card">
                  <div className="k">ID</div>
                  <div className="v" style={{ fontSize: 12 }}>
                    {node.id}
                  </div>
                </div>
                <div className="stat-card">
                  <div className="k">Score</div>
                  <div className="v">
                    {node.score !== null ? node.score.toFixed(4) : "—"}
                  </div>
                </div>
                {node.source && (
                  <div className="stat-card">
                    <div className="k">Source</div>
                    <div className="v" style={{ fontSize: 12 }}>
                      {node.source}
                    </div>
                  </div>
                )}
              </div>
              <h2 className="panel-title" style={{ margin: "0 -12px 8px" }}>
                Instruction
              </h2>
              <pre className="mono-block">
                {node.instruction ?? "(instruction not in event payload)"}
              </pre>
              {parent && node.instruction !== null && (
                <>
                  <h2 className="panel-title" style={{ margin: "8px -12px 8px" }}>
                    Diff vs parent {parent.id.slice(0, 12)}
                  </h2>
                  <DiffView
                    before={parent.instruction ?? ""}
                    after={node.instruction}
                  />
                </>
              )}
            </>
          )}
        </Panel>
      </div>
    </div>
  );
}
