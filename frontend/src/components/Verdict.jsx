import ProbabilityScale from './ProbabilityScale';

function faceLine(r) {
  if (r.frames_analyzed !== undefined) {
    return `A face was found in ${r.frames_with_face} of ${r.frames_analyzed} sampled frames.${r.frames_with_face === 0 ? ' Whole frames were used instead, so treat this with extra caution.' : ''}`;
  }
  if (!r.face_detected) return 'No face was found, so the whole image was used. This makes the result less reliable.';
  const many = r.faces_found > 1 ? ` (${r.faces_found} found; the largest was used)` : '';
  return r.model_mode === 'DEMO'
    ? `A face was found${many}. Demo mode scores the whole image; a trained model would examine the cropped face.`
    : `A face was found${many} and examined after normalising its size and compression.`;
}

export default function Verdict({ result }) {
  const flagged = result.prediction !== 'REAL';
  const demo = result.model_mode === 'DEMO';
  const p = result.fake_probability;
  const t = result.threshold ?? 0.5;
  const head = flagged ? 'Looks manipulated' : 'Looks authentic';
  return (
    <div className={`verdict ${flagged ? 'flag' : 'ok'}`}>
      <h2>{head}{demo && <small>demo estimate</small>}</h2>
      <p className="lede">
        {demo ? 'The statistical indicators' : 'The classifier'} put the chance of manipulation at <b>{(p * 100).toFixed(1)}%</b>.
        The decision line is at {(t * 100).toFixed(0)}%, so this is called {flagged ? 'manipulated' : 'authentic'}.
      </p>
      <ProbabilityScale p={p} t={t} />
      <ul className="facts">
        <li>{faceLine(result)}</li>
        <li>{result.frames_analyzed !== undefined ? `${result.frames_analyzed} frames were sampled and averaged.` : 'One image was examined.'} Processing took {Math.round(result.processing_time_ms)} ms.</li>
        {!demo && <li>A high score is a reason to look closer, not proof. Heavy compression and unusual lighting can also raise it.</li>}
      </ul>
    </div>
  );
}
