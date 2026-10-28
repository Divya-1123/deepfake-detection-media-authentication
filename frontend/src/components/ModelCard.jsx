const pct = (v) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(1)}%`);
const LABELS = {
  test: 'Held-out test, all fake types',
  test_stylegan: 'Test: GAN-generated faces',
  test_faceapp: 'Test: FaceApp edits',
  unseen_celebdf: 'Never seen: Celeb-DF face swaps',
  unseen_faceapp: 'Never seen: FaceApp edits',
};

/** What is actually loaded, and what was measured on held-out data. Numbers come from the server, never hard-coded. */
export default function ModelCard({ health }) {
  if (!health) return null;
  const meta = health.model_meta || {};
  const splits = health.evaluation?.splits || {};
  const rows = Object.entries(splits);
  return (
    <details className="about">
      <summary>About the model and how well it works</summary>
      {health.demo_mode ? (
        <p>No trained checkpoint is loaded. Train one with <code>python -m ml.train</code> (see the README) and restart the server.</p>
      ) : (
        <>
          <p>
            {meta.arch === 'efficientnet_b0' ? 'EfficientNet-B0, pretrained on ImageNet and fine-tuned' : `${meta.arch || 'Custom CNN'}, trained`} on <b>{meta.dataset_name}</b>
            {meta.train_counts ? ` (${meta.train_counts[0]} real and ${meta.train_counts[1]} fake training images)` : ''}. Decision line: {pct(meta.threshold)}.
          </p>
          {rows.length ? (
            <div className="scroll">
              <table className="plain">
                <caption>Measured on images the model was not trained on</caption>
                <thead><tr><th>Group</th><th>Images</th><th>Accuracy</th><th>Fakes caught</th><th>Reals kept</th><th>ROC-AUC</th></tr></thead>
                <tbody>
                  {rows.map(([k, s]) => {
                    const fakeOnly = s.class_counts && s.class_counts.real === 0;
                    return (
                      <tr key={k}>
                        <td>{LABELS[k] || k}</td><td>{s.n_samples}</td>
                        <td>{fakeOnly ? '—' : pct(s.accuracy)}</td><td>{pct(s.recall)}</td>
                        <td>{fakeOnly ? '—' : pct(s.specificity)}</td><td>{s.roc_auc === null || s.roc_auc === undefined ? '—' : s.roc_auc.toFixed(3)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : <p className="fine">No evaluation results found for this checkpoint. Run <code>python -m ml.evaluate</code>.</p>}
          <p className="fine">“Fakes caught” is the share of fake images flagged; “Reals kept” is the share of genuine images correctly passed. Scores on groups the model never saw are the better guide to real-world use, and they are much lower. Treat a verdict as a lead to check, not a ruling.</p>
        </>
      )}
    </details>
  );
}
