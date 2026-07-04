import { useEffect, useState } from "react";
import { api, type CalibrationCertificate } from "../api";
import { Panel } from "../components/Panel";

function CertBadge({ cert }: { cert: CalibrationCertificate }) {
  if (cert.degenerate) {
    return <span className="badge status-degenerate">DEGENERATE</span>;
  }
  return cert.passed ? (
    <span className="badge status-pass">PASS</span>
  ) : (
    <span className="badge status-fail">FAIL</span>
  );
}

function Confusion({ matrix }: { matrix: number[][] }) {
  const n = matrix.length;
  if (n === 0) return <div className="dim">empty confusion matrix</div>;
  const maxCount = Math.max(1, ...matrix.flat());
  return (
    <div
      className="confusion"
      style={{ gridTemplateColumns: `repeat(${n + 1}, auto)` }}
      data-testid="confusion"
    >
      <div className="cell head">h\j</div>
      {matrix[0].map((_, j) => (
        <div key={`h${j}`} className="cell head">
          {j}
        </div>
      ))}
      {matrix.map((row, i) => (
        <>
          <div key={`r${i}`} className="cell head">
            {i}
          </div>
          {row.map((count, j) => (
            <div
              key={`${i}.${j}`}
              className="cell"
              style={{
                background: `rgba(74, 246, 195, ${(0.35 * count) / maxCount})`,
                color: count === 0 ? "var(--dim)" : "var(--text)",
              }}
            >
              {count}
            </div>
          ))}
        </>
      ))}
    </div>
  );
}

export function JudgePage() {
  const [certs, setCerts] = useState<CalibrationCertificate[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .certificates()
      .then(setCerts)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <>
      <Panel title="Judge Calibration Certificates">
        {error && <div className="error-text">{error}</div>}
        {certs.length === 0 && !error && (
          <div className="dim">no certificates</div>
        )}
      </Panel>
      {certs.map((cert, i) => (
        <Panel
          key={`${cert.judge_name}-${cert.criterion}-${i}`}
          title={`${cert.judge_name} / ${cert.criterion}`}
        >
          <div style={{ marginBottom: 10 }}>
            <CertBadge cert={cert} />
            <span className="dim" style={{ marginLeft: 12, fontSize: 11 }}>
              {cert.created_at} · labels [{cert.label_min}, {cert.label_max}] ·
              threshold κ ≥ {cert.threshold}
            </span>
          </div>
          <div className="stat-cards" style={{ marginBottom: 12 }}>
            <div className="stat-card">
              <div className="k">Kappa</div>
              <div className={`v ${cert.kappa >= cert.threshold ? "ok" : "bad"}`}>
                {cert.kappa.toFixed(3)}
              </div>
            </div>
            <div className="stat-card">
              <div className="k">Spearman</div>
              <div className="v">{cert.spearman.toFixed(3)}</div>
            </div>
            <div className="stat-card">
              <div className="k">N Holdout</div>
              <div className="v">{cert.n_holdout}</div>
            </div>
          </div>
          <div className="dim" style={{ fontSize: 11, marginBottom: 4 }}>
            CONFUSION (human rows × judge cols)
          </div>
          <Confusion matrix={cert.confusion} />
        </Panel>
      ))}
    </>
  );
}
