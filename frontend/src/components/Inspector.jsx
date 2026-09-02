import React from 'react';
import * as api from '../api.js';

const TABS = [
  { id: 'evidence', label: 'Evidence' },
  { id: 'trace', label: 'Trace' },
  { id: 'memory', label: 'Memory' },
];

export default function Inspector({ activeTab, setActiveTab, messages, trace, memory, onSeek }) {
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
        {activeTab === 'trace' && <TracePane trace={trace} onSeek={onSeek} />}
        {activeTab === 'memory' && <MemoryPane memory={memory} onSeek={onSeek} />}
      </div>
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

function TracePane({ trace, onSeek }) {
  if (!trace || trace.length === 0) {
    return <Empty text="No trace yet." />;
  }
  return (
    <div className="trace-list">
      {trace.map((entry, i) => (
        <div key={i} className="trace-row">
          <span className="trace-step mono">{String(entry.step).padStart(2, '0')}</span>
          <span className="trace-agent">{entry.agent}</span>
          <span className="trace-summary">
            {summarize(entry, onSeek)}
          </span>
        </div>
      ))}
    </div>
  );
}

function summarize(entry, onSeek) {
  if (entry.agent === 'planner') return `${entry.action} ${entry.reason || ''}`;
  if (entry.agent === 'grounding') return `${(entry.candidates || []).length} candidate(s)`;
  if (entry.agent === 'reasoning') {
    const iv = entry.interval;
    return iv ? (
      <button className="chip" onClick={() => onSeek(iv[0])}>
        {api.formatTime(iv[0])}–{api.formatTime(iv[1])}
      </button>
    ) : 'reasoning';
  }
  if (entry.agent === 'temporal_verifier') return `${entry.sufficient ? '✓' : '…'} ${entry.reason || ''}`;
  if (entry.agent === 'critic') return `${entry.sufficient ? '✓' : '✗'} ${entry.reason || ''}`;
  if (entry.agent === 'temporal_parser') return `type=${entry.type}`;
  if (entry.agent === 'context_resolver') return 'resolve';
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
