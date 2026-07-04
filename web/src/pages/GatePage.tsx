import { useState } from "react";
import { api, type GateReport } from "../api";
import { Panel } from "../components/Panel";

function fmt(n: number | null | undefined, digits = 4): string {
  return typeof n === "number" && Number.isFinite(n) ? n.toFixed(digits) : "—";
}

/** SVG histogram of per-candidate mean deltas. */
function DeltaHistogram({ deltas }: { deltas: number[] }) {
  if (deltas.length === 0) return null;
  const W = 480;
  const H = 120;
  const PAD = 10;
  const barW = Math.max(8, Math.min(48, (W - 2 * PAD) / deltas.length - 4));
  const maxAbs = Math.max(...deltas.map((d) => Math.abs(d)), 1e-9);
  const zeroY = H / 2;
  return (
    <svg
      width={W}
      height={H}
      style={{ background: "var(--bg)", border: "1px solid var(--border)" }}
      data-testid="delta-histogram"
    >
      <line x1={0} y1={zeroY} x2={W} y2={zeroY} stroke="var(--border)" />
      {deltas.map((d, i) => {
        const h = (Math.abs(d) / maxAbs) * (H / 2 - PAD);
        const x = PAD + i * (barW + 4);
        const y = d >= 0 ? zeroY - h : zeroY;
        return (
          <rect
            key={i}
            x={x}
            y={y}
            width={barW}
            height={Math.max(1, h)}
            fill={d >= 0 ? "var(--accent)" : "var(--red)"}
          />
        );
      })}
    </svg>
  );
}

export function GatePage() {
  const [program, setProgram] = useState("");
  const [candidateIds, setCandidateIds] = useState("");
  const [incumbentId, setIncumbentId] = useState("");
  const [devPath, setDevPath] = useState("");
  const [valPath, setValPath] = useState("");
  const [report, setReport] = useState<GateReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [actionMsg, setActionMsg] = useState<string | null>(null);

  const runGate = async () => {
    setBusy(true);
    setError(null);
    setReport(null);
    setActionMsg(null);
    try {
      const ids = candidateIds
        .split(/[\s,]+/)
        .map((s) => s.trim())
        .filter(Boolean);
      const rep = await api.gate({
        program,
        incumbent_id: incumbentId,
        candidate_ids: ids,
        dev_path: devPath,
        val_path: valPath,
      });
      setReport(rep);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const promote = async () => {
    if (!report?.winner_id) return;
    if (!window.confirm(`Activate ${report.winner_id} for ${report.program}?`)) {
      return;
    }
    try {
      await api.registryActivate(report.program, report.winner_id, report);
      setActionMsg(`activated ${report.winner_id}`);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const rollback = async () => {
    const prog = report?.program || program;
    if (!prog) return;
    if (!window.confirm(`Rollback active prompt for ${prog}?`)) return;
    try {
      const res = await api.registryRollback(prog);
      setActionMsg(`rolled back to ${res.prompt_id}`);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <>
      <Panel title="Gate">
        <div className="form-row">
          <div className="field">
            <label>Program</label>
            <input value={program} onChange={(e) => setProgram(e.target.value)} size={16} />
          </div>
          <div className="field">
            <label>Incumbent ID (optional)</label>
            <input
              value={incumbentId}
              onChange={(e) => setIncumbentId(e.target.value)}
              size={16}
            />
          </div>
          <div className="field">
            <label>Candidate IDs (comma/space sep)</label>
            <input
              value={candidateIds}
              onChange={(e) => setCandidateIds(e.target.value)}
              size={40}
            />
          </div>
          <div className="field">
            <label>Dev path</label>
            <input value={devPath} onChange={(e) => setDevPath(e.target.value)} size={22} />
          </div>
          <div className="field">
            <label>Val path</label>
            <input value={valPath} onChange={(e) => setValPath(e.target.value)} size={22} />
          </div>
          <button onClick={runGate} disabled={busy}>
            {busy ? "Gating…" : "Run Gate"}
          </button>
        </div>
        {error && <div className="error-text">{error}</div>}
        {actionMsg && <div className="ok">{actionMsg}</div>}
      </Panel>

      {report && (
        <>
          <div className={`verdict ${report.verdict}`} data-testid="verdict">
            {report.verdict === "promote" ? "PROMOTE" : "REJECT"}
            {report.winner_id ? ` — ${report.winner_id}` : ""}
          </div>

          <Panel title="Candidates">
            <table>
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Δ mean</th>
                  <th>95% CI</th>
                  <th>p</th>
                  <th>Holm</th>
                  <th>Dev mean</th>
                  <th>Incumbent</th>
                </tr>
              </thead>
              <tbody>
                {report.results.map((r) => (
                  <tr key={r.candidate_id}>
                    <td>{r.candidate_id}</td>
                    <td className={r.mean_delta >= 0 ? "ok" : "bad"}>
                      {fmt(r.mean_delta)}
                    </td>
                    <td>
                      [{fmt(r.ci_low)}, {fmt(r.ci_high)}]
                    </td>
                    <td>{fmt(r.p_value)}</td>
                    <td className={r.holm_significant ? "ok" : "bad"}>
                      {r.holm_significant ? "✓" : "✗"}
                    </td>
                    <td>{fmt(r.dev_mean)}</td>
                    <td>{fmt(r.incumbent_dev_mean)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {report.val_mean_delta !== null && (
              <div style={{ marginTop: 8, fontSize: 12 }}>
                VAL Δ {fmt(report.val_mean_delta)} CI [{fmt(report.val_ci_low)},{" "}
                {fmt(report.val_ci_high)}]
              </div>
            )}
          </Panel>

          <Panel title="Delta Histogram">
            <DeltaHistogram deltas={report.results.map((r) => r.mean_delta)} />
          </Panel>

          {(report.warnings.length > 0 || report.flags.length > 0) && (
            <Panel title="Warnings / Flags">
              <ul className="warn-list">
                {report.flags.map((f, i) => (
                  <li key={`f${i}`}>FLAG: {f}</li>
                ))}
                {report.warnings.map((w, i) => (
                  <li key={`w${i}`}>{w}</li>
                ))}
              </ul>
            </Panel>
          )}

          {report.spot_samples.length > 0 && (
            <Panel title="Spot Samples">
              {report.spot_samples.map((s, i) => (
                <details key={i}>
                  <summary>sample {i + 1}</summary>
                  <pre className="mono-block">{JSON.stringify(s, null, 2)}</pre>
                </details>
              ))}
            </Panel>
          )}

          <Panel title="Actions">
            <div className="form-row">
              <button
                onClick={promote}
                disabled={report.verdict !== "promote" || !report.winner_id}
              >
                Promote → Activate
              </button>
              <button className="danger" onClick={rollback}>
                Rollback
              </button>
            </div>
          </Panel>
        </>
      )}
    </>
  );
}
