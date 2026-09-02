const BASE = '/api';

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export async function uploadVideo(file) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`${BASE}/videos`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(`upload failed: ${res.status}`);
  return res.json();
}

export const createSession = (videoId) => post('/sessions', { video_id: videoId });
export const getProgress = (videoId) => get(`/progress/${videoId}`);
export const ask = (sessionId, question) => post(`/sessions/${sessionId}/ask`, { question });
export const getMemory = (sessionId) => get(`/sessions/${sessionId}/memory`);
export const getTrace = (sessionId) => get(`/sessions/${sessionId}/trace`);
export const resetSession = (sessionId) => post(`/sessions/${sessionId}/reset`, {});
export const videoUrl = (videoId) => `${BASE}/videos/${videoId}`;
export const getModels = () => get('/models');
export const getBackendStatus = () => get('/backend/status');

export function formatTime(seconds) {
  if (seconds == null) return '--:--';
  const s = Math.floor(seconds);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`;
}
