const fmt = (v, d = 1) => (v === undefined || v === null ? '—' : typeof v === 'number' ? v.toFixed(d) : v);
const mb = (n) => (n === undefined ? '—' : `${(n / 1024 / 1024).toFixed(2)} MB`);

export default function Readings({ forensic, video }) {
  if (!forensic) return null;
  const rows = video
    ? [['Resolution', `${forensic.resolution?.width} × ${forensic.resolution?.height}`], ['Frame rate', `${fmt(forensic.fps)} fps`], ['Duration', `${fmt(forensic.duration_seconds)} s`], ['Frames', forensic.frame_count], ['Codec', forensic.codec], ['File size', mb(forensic.file?.size_bytes)]]
    : [['Resolution', `${forensic.resolution?.width} × ${forensic.resolution?.height}`], ['Format', forensic.file?.format], ['File size', mb(forensic.file?.size_bytes)],
       ['Sharpness', fmt(forensic.blur?.variance)], ['Noise', fmt(forensic.noise?.stddev)], ['Brightness', fmt(forensic.brightness_contrast?.brightness)],
       ['Contrast', fmt(forensic.brightness_contrast?.contrast)], ['Edge density', fmt(forensic.edges?.density, 3)], ['Block artefacts', fmt(forensic.compression?.blockiness)]];
  return (
    <div className="readings">
      <h3>{video ? 'Video details' : 'Instrument readings'}</h3>
      <dl>{rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
      {!video && Object.keys(forensic.exif || {}).length === 0 && <p className="fine">No camera metadata (EXIF) is present. Many genuine photos lose it when shared online, so its absence proves nothing.</p>}
    </div>
  );
}
