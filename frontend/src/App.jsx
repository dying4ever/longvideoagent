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
  const [activeTab, setActiveTab] = useState('memory');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [stage, setStage] = useState(null);
  const [buildProgress, setBuildProgress] = useState(null);
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
    let poller = null;
    try {
      const v = await api.uploadVideo(file);
      setVideoId(v.video_id);
      setStage('building');
      poller = setInterval(() => {
        api.getProgress(v.video_id).then(setBuildProgress).catch(() => {});
      }, 800);
      const s = await api.createSession(v.video_id);
      setSessionId(s.session_id);
      setDuration(s.duration);
      const mem = await api.getMemory(s.session_id);
      setStage(null);
      setMessages([]);
      setMemory(mem);
      setTrace([]);
    } catch (e) {
      setError(e.message);
      setStage(null);
    } finally {
      if (poller) clearInterval(poller);
      setBuildProgress(null);
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
        {
          role: 'agent',
          content: r.answer,
          evidence: r.evidence,
          trace: r.trace,
          timestamp: r.timestamp,
          originalQuestion: question,
          resolvedQuestion: r.resolved_question,
          temporalType: r.temporal_type,
          referenceTimestamp: r.reference_timestamp,
          workingMemory: r.working_memory,
        },
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
          <span className="brand-mark">LVA</span>
          <div>
            <strong>LongVideoAgent</strong>
            <span className="brand-subtitle">交互式长视频理解工作台</span>
          </div>
        </div>
        <div className="header-right">
          <select
            className="model-select"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {models.length === 0 && (
              <option value="qwen3-vl-8b-local">Qwen3-VL 8B（本地）</option>
            )}
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

      <nav className="module-tabs" aria-label="Agent modules">
        {[
          ['memory', 'Memory 记忆'],
          ['manage', 'Backend'],
          ['process', 'Trace 轨迹'],
          ['evidence', 'Evidence 证据'],
        ].map(([id, label]) => (
          <button
            key={id}
            className={`module-tab ${activeTab === id ? 'active' : ''}`}
            onClick={() => setActiveTab(id)}
          >
            {label}
          </button>
        ))}
        <span className={`system-pill ${backendStatus?.model_loaded ? 'online' : ''}`}>
          <i /> {backendStatus?.model_loaded ? '模型已就绪' : '等待模型'}
        </span>
      </nav>

      <section className="overview-card">
        <div className="overview-head">
          <div>
            <span className="eyebrow">LONG VIDEO UNDERSTANDING</span>
            <h1>{videoId ? '视频记忆与多轮推理空间' : '上传长视频，开始交互式理解'}</h1>
          </div>
          <span className="json-badge">⌁ Plan · Ground · Reason · Critic</span>
        </div>
        <p className="overview-summary">
          {memory?.video?.global_summary ||
            '系统会先建立视频记忆，再由规划、视觉定位、视觉推理与批判检查模块协同回答问题。点击证据时间可直接跳转到对应画面。'}
        </p>
        <div className="topic-chips">
          <span>长视频理解</span><span>多轮对话</span><span>时间定位</span>
          <span>视觉推理</span><span>证据回溯</span><span>Critic 闭环</span>
        </div>
        <div className="overview-meta">
          <span>时长 <b>{duration != null ? api.formatTime(duration) : '--:--'}</b></span>
          <span>会话 <b>{sessionId ? '已建立' : '未开始'}</b></span>
          <span>记忆片段 <b>{memory?.video?.chapters?.length || 0}</b></span>
          <span>推理步骤 <b>{trace.length}</b></span>
        </div>
        <div className="overview-timeline">
          <i /><i /><i /><i /><i /><i /><i /><i /><i /><i />
        </div>
      </section>

      <div className="app-body">
        <section className="video-pane">
          <div className="pane-title"><span>01</span><b>视频工作区</b><em>Video Grounding</em></div>
          <VideoPlayer
            videoRef={videoRef}
            videoId={videoId}
            onUpload={handleUpload}
            stage={stage}
            duration={duration}
            buildProgress={buildProgress}
          />
        </section>

        <section className="chat-pane">
          <div className="pane-title"><span>02</span><b>交互问答</b><em>Multi-turn Dialogue</em></div>
          <Chat messages={messages} busy={busy} onAsk={handleAsk} onSeek={seek} disabled={!sessionId} />
        </section>

        <section className="inspector-pane">
          <div className="pane-title"><span>03</span><b>Agent 推理闭环</b><em>Plan → Ground → Reason → Critic</em></div>
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
