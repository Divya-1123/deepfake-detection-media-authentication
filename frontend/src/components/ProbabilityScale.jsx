/** Ruler-style scale: shaded zones either side of the decision line, a marker for this result. */
export default function ProbabilityScale({ p, t }) {
  const pct = (v) => `${Math.min(100, Math.max(0, v * 100))}%`;
  return (
    <div className="scale" role="img" aria-label={`Manipulation probability ${(p * 100).toFixed(1)} percent; decision line at ${(t * 100).toFixed(0)} percent`}>
      <div className="track">
        <span className="zone ok" style={{ width: pct(t) }} />
        <span className="zone flag" style={{ left: pct(t), right: 0 }} />
        <span className="line" style={{ left: pct(t) }} />
        <span className="marker" style={{ left: pct(p) }} />
      </div>
      <div className="ticks" aria-hidden="true">{[0, 25, 50, 75, 100].map((n) => <span key={n}>{n}%</span>)}</div>
    </div>
  );
}
