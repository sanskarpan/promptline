import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type RunSummary } from "../api";
import { Panel } from "../components/Panel";

const OPTIMIZERS = ["gepa", "protegi", "opro", "mipro", "bootstrap", "bootstrap-rs"];

function statusClass(status: string): string {
  return `status-${status}`;
}

function bestScore(run: RunSummary): string {
  const s = run.summary as Record<string, unknown> | null | undefined;
  const v = s?.["best_score"];
  return typeof v === "number" ? v.toFixed(4) : "—";
}

export function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [optimizer, setOptimizer] = useState(OPTIMIZERS[0]);
  const [dataPath, setDataPath] = useState("");
  const [budget, setBudget] = useState("");
  const [starting, setStarting] = useState(false);
  const navigate = useNavigate();

  const refresh = () => {
    api
      .listRuns()
      .then(setRuns)
      .catch((e: Error) => setError(e.message));
  };

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, []);

  const startRun = async () => {
    setStarting(true);
    setError(null);
    try {
      const { run_id } = await api.startRun({
        optimizer,
        data_path: dataPath,
        budget: budget ? Number(budget) : null,
      });
      navigate(`/ui/runs/${run_id}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setStarting(false);
    }
  };

  return (
    <>
      <Panel title="New Run">
        <div className="form-row">
          <div className="field">
            <label>Optimizer</label>
            <select value={optimizer} onChange={(e) => setOptimizer(e.target.value)}>
              {OPTIMIZERS.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Data path</label>
            <input
              value={dataPath}
              onChange={(e) => setDataPath(e.target.value)}
              placeholder="data/train.jsonl"
              size={32}
            />
          </div>
          <div className="field">
            <label>Budget</label>
            <input
              value={budget}
              onChange={(e) => setBudget(e.target.value)}
              placeholder="rollouts"
              size={10}
              inputMode="numeric"
            />
          </div>
          <button onClick={startRun} disabled={starting}>
            {starting ? "Starting…" : "New Run"}
          </button>
        </div>
        {error && <div className="error-text">{error}</div>}
      </Panel>

      <Panel title="Runs">
        <table>
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Status</th>
              <th>Best Score</th>
            </tr>
          </thead>
          <tbody>
            {runs.length === 0 && (
              <tr>
                <td colSpan={3} className="dim">
                  no runs
                </td>
              </tr>
            )}
            {runs.map((r) => (
              <tr
                key={r.run_id}
                className="clickable"
                onClick={() => navigate(`/ui/runs/${r.run_id}`)}
              >
                <td>{r.run_id}</td>
                <td className={statusClass(r.status)}>{r.status.toUpperCase()}</td>
                <td>{bestScore(r)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </>
  );
}
