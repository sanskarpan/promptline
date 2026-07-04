import { useEffect, useReducer, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, openRunEvents, type RunSummary } from "../api";
import { Panel } from "../components/Panel";
import { ScoreCurve } from "../components/ScoreCurve";
import { initialRunState, runReducer, type BudgetState } from "../state/runReducer";

function Meter({
  label,
  used,
  max,
  fmt,
}: {
  label: string;
  used: number;
  max: number | null;
  fmt: (n: number) => string;
}) {
  const pct = max && max > 0 ? Math.min(100, (used / max) * 100) : 0;
  return (
    <div className="meter-row">
      <span className="label">{label}</span>
      <div className="meter">
        <div className={`fill${pct > 85 ? " warn" : ""}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="value">
        {fmt(used)} / {max !== null ? fmt(max) : "∞"}
      </span>
    </div>
  );
}

function BudgetMeters({ budget }: { budget: BudgetState }) {
  return (
    <>
      <Meter
        label="Rollouts"
        used={budget.rolloutsUsed}
        max={budget.maxRollouts}
        fmt={(n) => String(Math.round(n))}
      />
      <Meter
        label="Cost USD"
        used={budget.costUsed}
        max={budget.maxCostUsd}
        fmt={(n) => `$${n.toFixed(4)}`}
      />
    </>
  );
}

export function RunDetail() {
  const { id } = useParams<{ id: string }>();
  const [state, dispatch] = useReducer(runReducer, initialRunState);
  const [run, setRun] = useState<RunSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const feedRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (!id) return;
    api
      .getRun(id)
      .then(setRun)
      .catch((e: Error) => setError(e.message));
    const source = openRunEvents(id, dispatch);
    return () => source.close();
  }, [id]);

  // Auto-scroll the event feed.
  useEffect(() => {
    const el = feedRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [state.eventCount]);

  const status = state.status !== "idle" ? state.status : run?.status ?? "…";

  return (
    <>
      <Panel title={`Run ${id ?? ""}`}>
        <div className="stat-cards">
          <div className="stat-card">
            <div className="k">Status</div>
            <div className={`v status-${status}`}>{String(status).toUpperCase()}</div>
          </div>
          <div className="stat-card">
            <div className="k">Best</div>
            <div className="v">
              {state.bestScore !== null ? state.bestScore.toFixed(4) : "—"}
            </div>
          </div>
          <div className="stat-card">
            <div className="k">Best ID</div>
            <div className="v">{state.bestId ?? "—"}</div>
          </div>
          <div className="stat-card">
            <div className="k">Events</div>
            <div className="v">{state.eventCount}</div>
          </div>
        </div>
        <div style={{ marginTop: 8 }}>
          <Link to={`/lineage/${id}`}>VIEW LINEAGE →</Link>
        </div>
        {error && <div className="error-text">{error}</div>}
      </Panel>

      <Panel title="Score Curve">
        <ScoreCurve fullEvals={state.fullEvals} minibatches={state.minibatches} />
      </Panel>

      <Panel title="Budget">
        <BudgetMeters budget={state.budget} />
      </Panel>

      <Panel title="Event Feed">
        <pre className="event-feed" ref={feedRef} data-testid="event-feed">
          {state.eventLog
            .map(
              (e) =>
                `${new Date(e.ts * 1000).toISOString().slice(11, 19)} ${e.type.padEnd(
                  18,
                )} ${JSON.stringify(e.payload)}`,
            )
            .join("\n")}
        </pre>
      </Panel>
    </>
  );
}
