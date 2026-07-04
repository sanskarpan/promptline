import type { ScorePoint } from "../state/runReducer";

const W = 640;
const H = 180;
const PAD = 24;

function polyline(points: ScorePoint[], lo: number, hi: number): string {
  const n = points.length;
  const span = hi - lo || 1;
  return points
    .map((p, i) => {
      const x = PAD + (n === 1 ? 0 : (i / (n - 1)) * (W - 2 * PAD));
      const y = H - PAD - ((p.score - lo) / span) * (H - 2 * PAD);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

/** Hand-rolled SVG score curve: full evals (accent) + minibatches (dim). */
export function ScoreCurve({
  fullEvals,
  minibatches,
}: {
  fullEvals: ScorePoint[];
  minibatches: ScorePoint[];
}) {
  const all = [...fullEvals, ...minibatches];
  if (all.length === 0) {
    return <div className="dim">no scores yet</div>;
  }
  const scores = all.map((p) => p.score);
  const lo = Math.min(...scores);
  const hi = Math.max(...scores);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      style={{ maxWidth: W, display: "block", background: "var(--bg)", border: "1px solid var(--border)" }}
      data-testid="score-curve"
    >
      <text x={4} y={12} fill="var(--dim)" fontSize={9} fontFamily="var(--mono)">
        {hi.toFixed(3)}
      </text>
      <text x={4} y={H - 4} fill="var(--dim)" fontSize={9} fontFamily="var(--mono)">
        {lo.toFixed(3)}
      </text>
      {minibatches.length > 1 && (
        <polyline
          points={polyline(minibatches, lo, hi)}
          fill="none"
          stroke="var(--dim)"
          strokeWidth={1}
          opacity={0.6}
        />
      )}
      {fullEvals.length > 1 && (
        <polyline
          points={polyline(fullEvals, lo, hi)}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={1.5}
        />
      )}
      {fullEvals.map((p, i) => {
        const n = fullEvals.length;
        const span = hi - lo || 1;
        const x = PAD + (n === 1 ? 0 : (i / (n - 1)) * (W - 2 * PAD));
        const y = H - PAD - ((p.score - lo) / span) * (H - 2 * PAD);
        return <circle key={i} cx={x} cy={y} r={2.5} fill="var(--accent)" />;
      })}
    </svg>
  );
}
