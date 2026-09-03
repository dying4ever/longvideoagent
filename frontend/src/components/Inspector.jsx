import React from 'react';
import * as api from '../api.js';

const TABS = [
  { id: 'evidence', label: 'Evidence' },
  { id: 'process', label: 'Agent Process' },
  { id: 'memory', label: 'Memory' },
  { id: 'manage', label: 'Manage' },
];

export default function Inspector({ activeTab, setActiveTab, messages, trace, memory, backendStatus, sessionId, onSeek }) {
  const lastAgent = [...messages].reverse().find((m) => m.role === 'agent');
  return (
    <div className="inspector">
      <div className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab ${activeTab === t.id ? 'active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="inspector-body">
        {activeTab === 'evidence' && (
          <EvidencePane evidence={lastAgent?.evidence} onSeek={onSeek} />
        )}
        {activeTab === 'process' && <ProcessPane trace={trace} onSeek={onSeek} />}
        {activeTab === 'memory' && <MemoryPane memory={memory} onSeek={onSeek} />}
        {activeTab === 'manage' && (
          <ManagePane backendStatus={backendStatus} sessionId={sessionId} />
        )}
      </div>
    </div>
  );
}

function ManagePane({ backendStatus, sessionId }) {
  return (
    <div className="memory">
      <Section title="Backend">
        <Kv label="runtime" value={backendStatus?.runtime} />
        <Kv label="model" value={backendStatus?.model} />
        <Kv label="model loaded" value={backendStatus?.model_loaded ? 'yes' : 'no'} />
      </Section>
      <Section title="Cache">
        <Kv label="memory dir" value={backendStatus?.memory_cache?.dir} />
        <Kv label="cached memories" value={backendStatus?.memory_cache?.n_cached} />
      </Section>
      <Section title="Session">
        <Kv label="session id" value={sessionId} />
      </Section>
    </div>
  );
}

function EvidencePane({ evidence, onSeek }) {
  if (!evidence || evidence.length === 0) {
    return <Empty text="No evidence yet." />;
  }
  return (
    <div className="evidence-list">
      {evidence.map((ev, i) => (
        <button key={i} className="evidence-row" onClick={() => onSeek(ev.timestamp)}>
          <span className="evidence-time mono">{api.formatTime(ev.timestamp)}</span>
          <span className="evidence-desc">{ev.description}</span>
        </button>
      ))}
    </div>
  );
}

function ProcessPane({ trace, onSeek }) {
  if (!trace || trace.length === 0) {
    return <Empty text="No trace yet." />;
  }
  const { context, rounds } = groupTrace(trace);
  return (
    <div className="trace-workflow">
      {context.length > 0 && (
        <div className="trace-context">
          <span className="trace-context-label">QUERY UNDERSTANDING</span>
          {context.map((entry, i) => (
            <span className="trace-context-item" key={i}>{summarize(entry, onSeek)}</span>
          ))}
        </div>
      )}
      {rounds.map((round, roundIndex) => {
        const failed = round.entries.some((e) => isChecker(e) && e.sufficient === false);
        const passed = round.entries.some((e) => isChecker(e) && e.sufficient === true);
        const hasNext = roundIndex < rounds.length - 1;
        const state = failed && hasNext ? 'replan' : failed ? 'review' : passed ? 'passed' : 'running';
        return (
          <React.Fragment key={round.number}>
            <section className={`trace-round ${state}`}>
              <header className="trace-round-head">
                <div><span>ROUND {round.number}</span><b>{roundGoal(round.entries)}</b></div>
                <em>{roundStatusLabel(state)}</em>
              </header>
              <div className="trace-agent-flow">
                {round.entries.map((entry, i) => (
                  <React.Fragment key={`${entry.step}-${i}`}>
                    {i > 0 && <span className="trace-arrow">→</span>}
                    <div className={`trace-agent-card ${entry.agent} ${statusOf(entry)}`}>
                      <div className="trace-agent-top">
                        <span>{agentShortName(entry.agent)}</span>
                        <i>STEP {entry.step || i + 1}</i>
                      </div>
                      <b>{agentDisplayName(entry.agent)}</b>
                      <div className="trace-agent-summary">{summarize(entry, onSeek)}</div>
                    </div>
                  </React.Fragment>
                ))}
              </div>
            </section>
            {state === 'replan' && (
              <div className="replan-bridge">
                <span>证据不足</span><b>REPLAN · 扩大范围并重新观察</b><i>↓</i>
              </div>
            )}
          </React.Fragment>
        );
      })}
      <div className={`trace-outcome ${tracePassed(trace) ? 'passed' : 'pending'}`}>
        <span>{tracePassed(trace) ? '✓' : '…'}</span>
        <div>
          <small>FINAL CHECK</small>
          <b>{tracePassed(trace) ? '证据验证通过，可以生成答案' : '已完成当前轮次，保留可追溯证据'}</b>
        </div>
      </div>
    </div>
  );
}

function groupTrace(trace) {
  const context = [];
  const rounds = [];
  let inferredRound = 0;
  let current = null;
  trace.forEach((entry) => {
    if (entry.agent === 'temporal_parser' || entry.agent === 'context_resolver') {
      context.push(entry);
      return;
    }
    let number = Number(entry.iteration || 0);
    if (!number) {
      if (entry.agent === 'planner' || !current) inferredRound += 1;
      number = inferredRound || 1;
    }
    if (!current || current.number !== number) {
      current = { number, entries: [] };
      rounds.push(current);
    }
    current.entries.push(entry);
  });
  return { context, rounds };
}

function isChecker(entry) {
  return entry.agent === 'temporal_verifier' || entry.agent === 'critic';
}

function tracePassed(trace) {
  return [...trace].reverse().some((entry) => isChecker(entry) && entry.sufficient === true);
}

function roundGoal(entries) {
  const plan = entries.find((entry) => entry.agent === 'planner');
  return ({
    ground_video: '在视频记忆中定位相关区间',
    inspect_interval: '回看指定区间的原始画面',
    verify_answer: '检查当前证据是否足够',
    finish: '整理证据并输出答案',
  })[plan?.action] || '基于上下文完成视觉观察';
}

function roundStatusLabel(state) {
  return ({ replan: '需要重规划', review: '证据待补充', passed: '验证通过', running: '观察完成' })[state];
}

function agentShortName(agent) {
  return ({ planner: 'PLAN', grounding: 'GROUND', reasoning: 'REASON', temporal_verifier: 'VERIFY', critic: 'CRITIC' })[agent] || 'AGENT';
}

function agentDisplayName(agent) {
  return ({
    planner: 'Planner 规划',
    grounding: 'Visual Grounding',
    reasoning: 'Visual Reasoning',
    temporal_verifier: 'Temporal Verifier',
    critic: 'Visual Critic',
  })[agent] || agent;
}

function statusOf(entry) {
  if (entry.agent === 'temporal_verifier' || entry.agent === 'critic') {
    return entry.sufficient ? 'ok' : 'warn';
  }
  if (entry.agent === 'temporal_parser') return 'ok';
  return 'neutral';
}

function summarize(entry, onSeek) {
  if (entry.agent === 'planner') return entry.reason || `执行 ${entry.action}`;
  if (entry.agent === 'grounding') {
    const candidates = entry.candidates || [];
    if (!candidates.length) return '未找到候选区间';
    const top = candidates[0];
    return `${candidates.length} 个候选 · 首选 ${api.formatTime(top.start)}–${api.formatTime(top.end)}`;
  }
  if (entry.agent === 'reasoning') {
    const iv = entry.interval;
    return iv ? (
      <button className="chip" onClick={() => onSeek(iv[0])}>
        {api.formatTime(iv[0])}–{api.formatTime(iv[1])}
      </button>
    ) : '观察候选画面';
  }
  if (entry.agent === 'temporal_verifier') return `${entry.sufficient ? '时间覆盖充分' : '时间覆盖不足'} · ${entry.reason || ''}`;
  if (entry.agent === 'critic') return `${entry.sufficient ? '视觉证据可信' : '视觉证据不足'} · ${entry.reason || ''}`;
  if (entry.agent === 'temporal_parser') return `时间意图：${entry.type || 'NORMAL'}`;
  if (entry.agent === 'context_resolver') return entry.reference_timestamp != null
    ? `复用参考时间：${api.formatTime(entry.reference_timestamp)}` : '已解析对话上下文';
  return '';
}

function MemoryPane({ memory, onSeek }) {
  if (!memory) return <Empty text="No memory yet." />;
  const wm = memory.working_memory || {};
  const conv = memory.conversation || {};
  const video = memory.video || {};
  return (
    <div className="memory">
      <Section title="Working Memory">
        <Kv label="current subject" value={wm.current_subject} />
        <Kv label="reference event" value={wm.reference_event ? JSON.stringify(wm.reference_event) : null} />
        <Kv label="active entities" value={(wm.active_entities || []).join(', ') || null} />
      </Section>
      <Section title="Conversation">
        <Kv label="entities" value={Object.keys(conv.entities || {}).join(', ') || null} />
        {[...(conv.confirmed_events || [])].reverse().map((e, i) => (
          <button key={i} className="memory-event" onClick={() => onSeek(e.timestamp)}>
            <span className="mono">{api.formatTime(e.timestamp)}</span>
            <span>
              {e.subject}
              {e.predicate} · {e.verification_status}
            </span>
          </button>
        ))}
      </Section>
      <Section title="Video Memory">
        <p className="memory-summary">{video.global_summary}</p>
        {(video.chapters || []).map((c, i) => (
          <div key={i} className="chapter">
            <div className="mono">
              {api.formatTime(c.start)}–{api.formatTime(c.end)}
            </div>
            <div className="chapter-summary">{c.summary}</div>
          </div>
        ))}
      </Section>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div className="memory-section">
      <div className="memory-section-title">{title}</div>
      {children}
    </div>
  );
}

function Kv({ label, value }) {
  if (!value) return null;
  return (
    <div className="kv">
      <span className="kv-label">{label}</span>
      <span className="kv-value">{value}</span>
    </div>
  );
}

function Empty({ text }) {
  return <div className="empty">{text}</div>;
}
