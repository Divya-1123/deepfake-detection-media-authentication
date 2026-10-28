import { useEffect, useMemo, useState } from 'react';
import Stage from './components/Stage';
import Verdict from './components/Verdict';
import FrameStrip from './components/FrameStrip';
import Readings from './components/Readings';
import ModelCard from './components/ModelCard';
import { detectImage, detectVideo, healthCheck } from './services/api';
import { isVideoFile, validateFile } from './validation';
import './App.css';

const STATUS = {
  connecting: ['Connecting to server…', 'wait'],
  offline: ['Server offline', 'bad'],
  DEMO: ['Demo mode: no trained model', 'wait'],
  TRAINED: ['Trained model active', 'ok'],
};

function Dashboard({ onLogout, account }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState(null);       // null = connecting, false = offline, object = ok

  const video = useMemo(() => isVideoFile(file), [file]);
  const state = health === null ? 'connecting' : health === false ? 'offline' : health.model_mode;
  const [statusText, statusTone] = STATUS[state] || STATUS.connecting;

  useEffect(() => { healthCheck().then(setHealth).catch(() => setHealth(false)); }, []);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const select = (f) => {
    setResult(null);
    const problem = validateFile(f);
    if (problem) { setError(problem); setFile(null); setPreview(''); return; }
    setError('');
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    try {
      setResult(video ? await detectVideo(file) : await detectImage(file));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => { setFile(null); setPreview(''); setResult(null); setError(''); };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true"><circle cx="11" cy="11" r="7.5" fill="none" stroke="currentColor" strokeWidth="2" /><path d="M16.5 16.5 23 23" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" /></svg>
          <div><strong>Media Authentication Platform</strong><span>Deepfake detection for images and video</span></div>
        </div>
        <div className="header-right"><div className={`status ${statusTone}`} role="status"><i />{statusText}</div><div className="account-chip"><span className="avatar">{(account?.name || "U").slice(0,1).toUpperCase()}</span><span className="account-name">{account?.name || "Guest"}</span><button className="quiet small-button" onClick={onLogout}>Log out</button></div></div>
      </header>

      <main>
        <section className="welcome"><div><p className="eyebrow">MEDIA INTEGRITY WORKSPACE</p><h1>Know what you are looking at.</h1><p>Screen images and videos for manipulation signals, then review the evidence behind each result.</p></div><div className="welcome-stat"><span>01</span><b>Upload</b><small>Choose image or video</small></div><div className="welcome-stat"><span>02</span><b>Analyze</b><small>Inspect forensic signals</small></div><div className="welcome-stat"><span>03</span><b>Review</b><small>Read the model verdict</small></div></section>
        {health === false && (
          <p className="alert" role="alert">The analysis server is not reachable. Start it with <code>uvicorn app:app --reload</code> in the backend folder, then reload this page.</p>
        )}
        {health && health.demo_mode && (
          <p className="alert soft"><b>Demo mode.</b> No trained model is loaded, so results come from simple image statistics (sharpness, noise, compression). They are not a validated deepfake test.</p>
        )}

        <div className="desk">
          <section className="specimen-col" aria-label="Specimen">
            <Stage file={file} preview={preview} video={video} result={result} loading={loading} onFile={select} />
            <div className="actions">
              {file && !result && <button className="primary" disabled={loading || health === false} onClick={analyze}>{loading ? 'Examining…' : 'Examine this file'}</button>}
              {(file || result) && <button className="quiet" disabled={loading} onClick={reset}>Start over</button>}
            </div>
            {error && <p className="alert" role="alert">{error}</p>}
          </section>

          <section className="findings" aria-label="Findings" aria-live="polite">
            {!result && !loading && (
              <div className="idle">
                <h2>Nothing examined yet</h2>
                <p>Add a photo or video. The server finds the face, normalises its size and compression, and asks the classifier how likely it is to be manipulated.</p>
              </div>
            )}
            {loading && <div className="idle"><h2>Examining…</h2><p>Detecting faces, sampling frames and running the classifier.</p></div>}
            {result && (
              <>
                <Verdict result={result} />
                {video && <FrameStrip result={result} />}
                <Readings forensic={result.forensic_analysis || result.metadata} video={video} />
              </>
            )}
          </section>
        </div>

        <ModelCard health={health} />
      </main>
    </div>
  );
}


function AuthPage({ onLogin }) {
  const [mode, setMode] = useState('login');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [notice, setNotice] = useState('');
  const submit = (event) => {
    event.preventDefault();
    if (password.length < 6) { setNotice('Use at least 6 characters for this demo session.'); return; }
    if (!email.includes('@')) { setNotice('Enter a valid email address.'); return; }
    if (mode === 'register' && !name.trim()) { setNotice('Add your name to create a demo profile.'); return; }
    const account = { name: (name.trim() || email.split('@')[0] || 'Researcher'), email };
    sessionStorage.setItem('deepfake-demo-account', JSON.stringify(account));
    onLogin(account);
  };
  return (
    <div className="auth-shell">
      <div className="auth-visual">
        <div className="auth-brand"><span className="brand-mark">⌕</span><span>VERITY<span className="brand-sub">MEDIA AUTHENTICATION</span></span></div>
        <div className="auth-message"><p className="eyebrow">TRUST, WITH EVIDENCE</p><h1>Look closer.<br/>Question what you see.</h1><p>One workspace for image and video integrity checks, forensic clues, and model-assisted review.</p><div className="auth-orbit"><div className="orbit-ring ring-one"></div><div className="orbit-ring ring-two"></div><div className="orbit-core">V</div><span className="orbit-label label-a">PIXEL SIGNALS</span><span className="orbit-label label-b">MODEL REVIEW</span><span className="orbit-label label-c">MEDIA FORENSICS</span></div></div>
        <p className="auth-footnote">A research prototype · Results are indicators, not proof.</p>
      </div>
      <div className="auth-panel">
        <div className="auth-mobile-brand">⌕ VERITY <span>MEDIA AUTHENTICATION</span></div>
        <div className="auth-form-wrap">
          <p className="eyebrow">{mode === 'login' ? 'WELCOME BACK' : 'GET STARTED'}</p>
          <h2>{mode === 'login' ? 'Sign in to your workspace' : 'Create your workspace'}</h2>
          <p className="auth-intro">Review media checks in one clear, focused dashboard.</p>
          <div className="auth-tabs"><button className={mode === 'login' ? 'active' : ''} onClick={() => {setMode('login');setNotice('');}}>Sign in</button><button className={mode === 'register' ? 'active' : ''} onClick={() => {setMode('register');setNotice('');}}>Create account</button></div>
          <form onSubmit={submit}>
            {mode === 'register' && <label>Full name<input autoComplete="name" value={name} onChange={e=>setName(e.target.value)} placeholder="Your name" /></label>}
            <label>Email address<input type="email" autoComplete="email" value={email} onChange={e=>setEmail(e.target.value)} placeholder="you@example.com" required /></label>
            <label>Password<input type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} value={password} onChange={e=>setPassword(e.target.value)} placeholder="At least 6 characters" minLength={6} required /></label>
            {notice && <p className="auth-notice" role="alert">{notice}</p>}
            <button className="auth-submit" type="submit">{mode === 'login' ? 'Sign in to workspace' : 'Create demo account'} <span>↗</span></button>
          </form>
          <p className="auth-demo-note"><strong>Prototype sign-in</strong><br/>This is a front-end demo session. It does not verify credentials or create a secure server-side account. Connect a backend authentication service before public deployment.</p>
        </div>
        <p className="auth-legal">Built for responsible media analysis · Do not treat model output as definitive proof.</p>
      </div>
    </div>
  );
}

export default function App() {
  const [account, setAccount] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem('deepfake-demo-account') || 'null'); } catch { return null; }
  });
  const logout = () => { sessionStorage.removeItem('deepfake-demo-account'); setAccount(null); };
  return account ? <Dashboard account={account} onLogout={logout} /> : <AuthPage onLogin={setAccount} />;
}
