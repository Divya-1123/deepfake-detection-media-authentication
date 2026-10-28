import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, beforeEach, describe, it, expect } from 'vitest';
import App from './App';

const imageResult = {
  media_type: 'image', filename: 'a.png', prediction: 'REAL', confidence: 0.82, fake_probability: 0.18, threshold: 0.5, model_mode: 'DEMO',
  face_detected: true, faces_found: 1, face_box: [10, 10, 40, 40], face_detector: 'dnn', processing_time_ms: 12.3,
  forensic_analysis: { resolution: { width: 100, height: 100 }, file: { size_bytes: 1000, format: 'PNG' }, exif: {}, blur: { variance: 50 }, noise: { stddev: 5 }, brightness_contrast: { brightness: 100, contrast: 40 }, edges: { density: 0.05 }, compression: { blockiness: 3 } },
};
const videoResult = {
  media_type: 'video', prediction: 'POTENTIALLY MANIPULATED', confidence: 0.7, fake_probability: 0.7, threshold: 0.5, model_mode: 'TRAINED',
  frames_analyzed: 3, frames_with_face: 2, processing_time_ms: 90,
  frame_results: [{ frame: 0, timestamp_seconds: 0, fake_probability: 0.2 }, { frame: 10, timestamp_seconds: 1, fake_probability: 0.9 }, { frame: 20, timestamp_seconds: 2, fake_probability: 0.95 }],
  suspicious_frames: [{ frame: 10, timestamp_seconds: 1, fake_probability: 0.9 }, { frame: 20, timestamp_seconds: 2, fake_probability: 0.95 }],
  metadata: { resolution: { width: 640, height: 480 }, file: { size_bytes: 5000 }, fps: 10, duration_seconds: 3, frame_count: 30, codec: 'mp4v' },
};
const trainedHealth = {
  model_mode: 'TRAINED', demo_mode: false,
  model_meta: { arch: 'efficientnet_b0', dataset_name: 'Test set', train_counts: [100, 60], threshold: 0.41 },
  evaluation: { splits: { test: { n_samples: 80, accuracy: 0.9, precision: 0.88, recall: 0.86, specificity: 0.93, f1: 0.87, roc_auc: 0.95, class_counts: { real: 50, fake: 30 } }, unseen_celebdf: { n_samples: 50, recall: 0.4, specificity: null, roc_auc: null, class_counts: { real: 0, fake: 50 } } } },
};

function mockFetch({ health = { model_mode: 'DEMO', demo_mode: true, model_meta: {}, evaluation: null }, detect }) {
  global.fetch = vi.fn((url) => (String(url).endsWith('/api/health') ? Promise.resolve({ ok: true, json: () => Promise.resolve(health) }) : detect(url)));
}
const ok = (body) => () => Promise.resolve({ ok: true, json: () => Promise.resolve(body) });
const png = () => new File([new Uint8Array([137, 80, 78, 71, 1, 2, 3])], 'face.png', { type: 'image/png' });
const mp4 = () => new File([new Uint8Array(10)], 'clip.mp4', { type: 'video/mp4' });
const status = (text) => waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(text));
const examine = async (file) => {
  await userEvent.upload(screen.getByTestId('file-input'), file);
  await userEvent.click(screen.getByRole('button', { name: /examine this file/i }));
};

beforeEach(() => {
  vi.restoreAllMocks();
  sessionStorage.setItem('deepfake-demo-account', JSON.stringify({ name: 'Test Researcher', email: 'test@example.com' }));
});

describe('server state is read from the backend, never assumed', () => {
  it('demo mode: badge and warning banner', async () => {
    mockFetch({ detect: () => Promise.reject() });
    render(<App />);
    await status(/Demo mode: no trained model/);
    expect(screen.getByText(/not a validated deepfake test/)).toBeInTheDocument();
  });
  it('trained mode: badge, and no demo banner', async () => {
    mockFetch({ health: trainedHealth, detect: () => Promise.reject() });
    render(<App />);
    await status(/Trained model active/);
    expect(screen.queryByText(/not a validated deepfake test/)).toBeNull();
  });
  it('offline: clear alert and disabled analysis', async () => {
    global.fetch = vi.fn(() => Promise.reject(new TypeError('failed')));
    render(<App />);
    await status(/Server offline/);
    expect(screen.getByRole('alert')).toHaveTextContent(/not reachable/);
  });
});

describe('image workflow', () => {
  it('uploads, calls the API and shows verdict, scale, face note', async () => {
    mockFetch({ detect: ok(imageResult) });
    render(<App />);
    await status(/Demo mode/);
    await examine(png());
    expect(await screen.findByText(/Looks authentic/)).toBeInTheDocument();
    expect(screen.getByRole('img', { name: /Manipulation probability 18.0 percent/ })).toBeInTheDocument();
    expect(screen.getByText(/A face was found/)).toBeInTheDocument();
    expect(screen.getByLabelText('Detected face')).toBeInTheDocument();
    const call = global.fetch.mock.calls.find((c) => String(c[0]).includes('/api/detect/image'));
    expect(call[1].method).toBe('POST');
  });
  it('does not claim the cropped face was examined in demo mode', async () => {
    mockFetch({ detect: ok(imageResult) });
    render(<App />);
    await status(/Demo mode/);
    await examine(png());
    expect(await screen.findByText(/Demo mode scores the whole image/)).toBeInTheDocument();
  });
  it('flags a manipulated verdict', async () => {
    mockFetch({ detect: ok({ ...imageResult, prediction: 'POTENTIALLY MANIPULATED', fake_probability: 0.9 }) });
    render(<App />);
    await status(/Demo mode/);
    await examine(png());
    expect(await screen.findByText(/Looks manipulated/)).toBeInTheDocument();
  });
  it('shows the server error message when analysis fails', async () => {
    mockFetch({ detect: () => Promise.resolve({ ok: false, json: () => Promise.resolve({ detail: 'Image is too large (12000x10000).' }) }) });
    render(<App />);
    await status(/Demo mode/);
    await examine(png());
    expect(await screen.findByText(/Image is too large/)).toBeInTheDocument();
  });
  it('rejects unsupported types without calling the API', async () => {
    mockFetch({ detect: () => Promise.reject() });
    render(<App />);
    await status(/Demo mode/);
    await userEvent.upload(screen.getByTestId('file-input'), new File(['x'], 'notes.txt', { type: 'text/plain' }), { applyAccept: false });
    expect(await screen.findByText(/Unsupported file type/)).toBeInTheDocument();
    expect(global.fetch.mock.calls.some((c) => String(c[0]).includes('/api/detect'))).toBe(false);
  });
  it('start over returns to the drop zone', async () => {
    mockFetch({ detect: ok(imageResult) });
    render(<App />);
    await status(/Demo mode/);
    await examine(png());
    await screen.findByText(/Looks authentic/);
    await userEvent.click(screen.getByRole('button', { name: /start over/i }));
    expect(screen.getByText(/Place a photo or video on the table/)).toBeInTheDocument();
  });
});

describe('trained model features', () => {
  it('offers the heat-map toggle only when a heat-map is returned', async () => {
    mockFetch({ health: trainedHealth, detect: ok({ ...imageResult, model_mode: 'TRAINED', heatmap_png_b64: 'AAAA' }) });
    render(<App />);
    await status(/Trained model active/);
    await examine(png());
    await userEvent.click(await screen.findByRole('button', { name: 'Heat-map' }));
    expect(screen.getByAltText(/Grad-CAM heat-map/)).toBeInTheDocument();
  });
  it('model card shows measured numbers from the server, including unseen sources', async () => {
    mockFetch({ health: trainedHealth, detect: () => Promise.reject() });
    render(<App />);
    await status(/Trained model active/);
    const card = screen.getByText(/About the model/).closest('details');
    expect(within(card).getByText('90.0%')).toBeInTheDocument();
    const unseen = within(card).getByText(/Never seen: Celeb-DF face swaps/).closest('tr');
    expect(unseen).toHaveTextContent('40.0%');                 // recall shown
    expect(within(unseen).getAllByText('—').length).toBeGreaterThanOrEqual(2);   // accuracy/specificity not shown for fake-only group
    expect(card).toHaveTextContent(/much lower/);
  });
  it('video: frame strip lists suspicious frames', async () => {
    mockFetch({ health: trainedHealth, detect: ok(videoResult) });
    render(<App />);
    await status(/Trained model active/);
    await userEvent.upload(screen.getByTestId('file-input'), mp4());
    await userEvent.click(screen.getByRole('button', { name: /examine this file/i }));
    expect(await screen.findByText(/2 of 3 sampled frames crossed the decision line/)).toBeInTheDocument();
    expect(screen.getByText('Most suspicious frames')).toBeInTheDocument();
    expect(screen.getByText(/A face was found in 2 of 3 sampled frames/)).toBeInTheDocument();
  });
});
