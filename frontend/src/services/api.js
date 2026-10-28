const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

async function request(path, file) {
  const body = new FormData();
  body.append('file', file);
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, { method: 'POST', body });
  } catch {
    throw new Error('Cannot reach the analysis server. Make sure the backend is running (uvicorn app:app --reload).');
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    // FastAPI validation errors may return a list in `detail`
    const detail = Array.isArray(data.detail) ? data.detail.map(d => d.msg).join('; ') : data.detail;
    throw new Error(detail || 'The server could not process this file.');
  }
  return data;
}

export const detectImage = (file) => request('/api/detect/image', file);
export const detectVideo = (file) => request('/api/detect/video', file);
export const analyzeMetadata = (file) => request('/api/analyze/metadata', file);
export const healthCheck = async () => {
  const response = await fetch(`${API_BASE}/api/health`);
  if (!response.ok) throw new Error('Health check failed');
  return response.json();
};
