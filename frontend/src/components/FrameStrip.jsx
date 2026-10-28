/** One bar per sampled frame (height = manipulation probability); the horizontal line is the decision threshold. */
export default function FrameStrip({ result }) {
  const frames = result.frame_results || [];
  const t = result.threshold ?? 0.5;
  const flagged = result.suspicious_frames || [];
  const top = [...flagged].sort((a, b) => b.fake_probability - a.fake_probability).slice(0, 5);
  return (
    <div className="strip-wrap">
      <h3>Frame by frame</h3>
      <p className="lede">{flagged.length === 0 ? 'No sampled frame crossed the decision line.' : `${flagged.length} of ${frames.length} sampled frames crossed the decision line.`}</p>
      <div className="strip" role="img" aria-label="Manipulation probability for each sampled frame">
        <span className="tline" style={{ bottom: `${t * 100}%` }} />
        {frames.map((f) => (
          <span key={f.frame} className={`fbar ${f.fake_probability >= t ? 'flag' : ''}`} style={{ height: `${Math.max(2, f.fake_probability * 100)}%` }}
            title={`Frame ${f.frame} at ${f.timestamp_seconds}s: ${(f.fake_probability * 100).toFixed(1)}%`} />
        ))}
      </div>
      {top.length > 0 && (
        <table className="plain"><caption>Most suspicious frames</caption>
          <thead><tr><th>Frame</th><th>Time</th><th>Probability</th></tr></thead>
          <tbody>{top.map((f) => <tr key={f.frame}><td>{f.frame}</td><td>{f.timestamp_seconds} s</td><td>{(f.fake_probability * 100).toFixed(1)}%</td></tr>)}</tbody>
        </table>
      )}
    </div>
  );
}
