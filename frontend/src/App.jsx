import React, { useEffect, useRef, useState } from 'react';
import * as api from './api.js';
import VideoPlayer from './components/VideoPlayer.jsx';
import Chat from './components/Chat.jsx';
import Inspector from './components/Inspector.jsx';

export default function App() {
  const [videoId, setVideoId] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [duration, setDuration] = useState(null);
  const [messages, setMessages] = useState([]);
  const [memory, setMemory] = useState(null);
  const [trace, setTrace] = useState([]);
  const [activeTab, setActiveTab] = useState('evidence');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [stage, setStage] = useState(null);
  const [models, setModels] = useState([]);
  const [backendStatus, setBackendStatus] = useState(null);
  const [selectedModel, setSelectedModel] = useState('qwen3-vl-8b-local');
  const videoRef = useRef(null);

  useEffect(() => {
    api.getModels().then((r) => setModels(r.models || [])).catch(() => {});
    api.getBackendStatus().then((r) => setBackendStatus(r)).catch(() => {});
  }, []);

  async function handleUpload(file) {
    setError(null);
    setStage('uploading');
    try {
      const v = await api.uploadVideo(file);
      setVideoId(v.video_id);
      setStage('building');
      const s = await api.createSession(v.video_id);
      setSessionId(s.session_id);
      setDuration(s.duration);
      setStage(null);
      setMessages([]);
      setMemory(null);
      setTrace([]);
    } catch (e) {
      setError(e.message);
      setStage(null);
    }
  }

  async function handleAsk(question) {
    if (!sessionId) return;
    setError(null);
    setBusy(true);
    setMessages((m) => [...m, { role: 'user', content: question }]);
    try {
      const r = await api.ask(sessionId, question);
      setMessages((m) => [
        ...m,
        { role: 'agent', content: r.answer, evidence: r.evidence, trace: r.trace, timestamp: r.timestamp },
      ]);
      setTrace(r.trace || []);
      setActiveTab('evidence');
      const mem = await api.getMemory(sessionId);
      setMemory(mem);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function seek(seconds) {
    if (videoRef.current && seconds != null) {
      videoRef.current.currentTime = seconds;
      videoRef.current.play?.().catch(() => {});
    }
  }

  async function handleReset() {
    if (!sessionId) return;
    await api.resetSession(sessionId);
    setMessages([]);
    setMemory(null);
    setTrace([]);
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-dot" />
          <span>LongVideoAgent</span>
        </div>
        <div className="header-right">
          <select
            className="model-select"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id} disabled={!m.available}>
                {m.name}
                {m.available ? '' : '（未配置）'}
              </option>
            ))}
          </select>
          {sessionId && <span className="session-id">session {sessionId}</span>}
          {sessionId && (
            <button className="ghost" onClick={handleReset}>
              Reset
            </button>
          )}
        </div>
      </header>

      <div className="app-body">
        <section className="video-pane">
          <VideoPlayer
            videoRef={videoRef}
            videoId={videoId}
            onUpload={handleUpload}
            stage={stage}
            duration={duration}
          />
        </section>

        <section className="chat-pane">
          <Chat messages={messages} busy={busy} onAsk={handleAsk} onSeek={seek} disabled={!sessionId} />
        </section>

        <section className="inspector-pane">
          <Inspector
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            messages={messages}
            trace={trace}
            memory={memory}
            backendStatus={backendStatus}
            sessionId={sessionId}
            onSeek={seek}
          />
        </section>
      </div>

      {error && <div className="error-banner">{error}</div>}
    </div>
  );
}
