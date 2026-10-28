import { useRef, useState } from 'react';

const ACCEPT = '.jpg,.jpeg,.png,.webp,.mp4,.avi,.mov,.mkv';

/** The "light table": drop zone before a file is chosen, then the specimen with the detected face outlined. */
export default function Stage({ file, preview, video, result, loading, onFile }) {
  const input = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [view, setView] = useState('original');
  const open = () => !loading && input.current?.click();
  const choose = (f) => { if (f && !loading) { setView('original'); onFile(f); } };

  const res = result?.forensic_analysis?.resolution;
  const box = result?.face_box;
  const heat = result?.heatmap_png_b64;
  const showHeat = view === 'heatmap' && heat;

  if (!file) {
    return (
      <div
        className={`stage drop ${dragging ? 'over' : ''}`}
        role="button" tabIndex={0} aria-label="Add an image or video"
        onClick={open}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } }}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files[0]); }}
      >
        <input ref={input} hidden type="file" accept={ACCEPT} data-testid="file-input" onChange={(e) => { choose(e.target.files[0]); e.target.value = ''; }} />
        <h2>Place a photo or video on the table</h2>
        <p>Drop it here, or press Enter to browse.</p>
        <p className="fine">JPG, PNG, WEBP up to 10 MB. MP4, AVI, MOV, MKV up to 100 MB.</p>
      </div>
    );
  }

  return (
    <div className="stage">
      {/* keep the input mounted so a new file can be chosen later */}
      <input ref={input} hidden type="file" accept={ACCEPT} data-testid="file-input" onChange={(e) => { choose(e.target.files[0]); e.target.value = ''; }} />
      <div className={`specimen ${loading ? 'scanning' : ''}`}>
        {video ? (
          <video controls src={preview} />
        ) : (
          <div className="frame">
            {showHeat
              ? <img src={`data:image/png;base64,${heat}`} alt="Grad-CAM heat-map over the analysed face" />
              : <img src={preview} alt="Selected specimen" />}
            {!showHeat && box && res && (
              <span className="facebox" aria-label="Detected face"
                style={{ left: `${(box[0] / res.width) * 100}%`, top: `${(box[1] / res.height) * 100}%`, width: `${(box[2] / res.width) * 100}%`, height: `${(box[3] / res.height) * 100}%` }} />
            )}
          </div>
        )}
      </div>
      <div className="caption">
        <span className="fname">{file.name}</span>
        <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
        {heat && !video && (
          <span className="seg" role="group" aria-label="Image view">
            <button type="button" aria-pressed={view === 'original'} onClick={() => setView('original')}>Original</button>
            <button type="button" aria-pressed={view === 'heatmap'} onClick={() => setView('heatmap')}>Heat-map</button>
          </span>
        )}
      </div>
      {heat && video && <p className="fine">Heat-map of the most suspicious sampled frame:</p>}
      {heat && video && <img className="vheat" src={`data:image/png;base64,${heat}`} alt="Grad-CAM heat-map for the most suspicious frame" />}
    </div>
  );
}
