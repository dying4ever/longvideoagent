import React, { useEffect, useRef, useState } from 'react';
import * as api from '../api.js';

const TEMPORAL_LABELS = {
  NORMAL: '普通内容理解',
  FIRST: '第一次 · FIRST',
  LAST: '最后一次 · LAST',
  REPEAT: '再次出现 · REPEAT',
  BEFORE: '事件之前 · BEFORE',
  AFTER: '事件之后 · AFTER',
  ALWAYS: '全程状态 · ALWAYS',
};

export default function Chat({ messages, busy, onAsk, onSeek, disabled }) {
  const [input, setInput] = useState('');
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, busy]);

  function submit(e) {
    e.preventDefault();
    if (!input.trim() || busy || disabled) return;
    onAsk(input.trim());
    setInput('');
  }

  return (
    <div className="chat">
      <div className="chat-scroll">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>针对视频发起问题</p>
            <p className="chat-hint">例如：“这个人第一次出现是什么时候？”</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="msg-label">{m.role === 'user' ? '你的问题' : 'Agent 回答'}</div>
            <div className="msg-content">{m.content || '…'}</div>
            {m.role === 'agent' && (
              <UnderstandingCard message={m} onSeek={onSeek} />
            )}
            {m.role === 'agent' && m.timestamp != null && (
              <div className="msg-timestamp">
                <button className="chip" onClick={() => onSeek(m.timestamp)}>
                  ⏱ {api.formatTime(m.timestamp)}
                </button>
              </div>
            )}
            {m.role === 'agent' && m.evidence && m.evidence.length > 0 && (
              <div className="msg-evidence">
                {m.evidence.slice(0, 3).map((ev, j) => (
                  <button key={j} className="chip" onClick={() => onSeek(ev.timestamp)}>
                    {api.formatTime(ev.timestamp)} · {ev.description}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
        {busy && (
          <div className="msg agent">
            <div className="msg-label">Agent 正在推理</div>
            <div className="thinking">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form className="chat-input" onSubmit={submit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={disabled ? '请先上传视频' : '输入关于视频的问题…'}
          disabled={disabled || busy}
        />
        <button className="primary" type="submit" disabled={disabled || busy || !input.trim()}>
          发送
        </button>
      </form>
    </div>
  );
}

function UnderstandingCard({ message, onSeek }) {
  const resolved = message.resolvedQuestion || message.originalQuestion;
  const changed = resolved && message.originalQuestion && resolved !== message.originalQuestion;
  const subject = message.workingMemory?.current_subject;
  const referenceEvent = message.workingMemory?.reference_event?.event;
  const referenceTimestamp = message.referenceTimestamp;

  if (!resolved && !message.temporalType && referenceTimestamp == null) return null;

  return (
    <div className={`understanding-card ${changed ? 'has-resolution' : ''}`}>
      <div className="understanding-title">
        <span>CONTEXT RESOLVER</span>
        <b>{changed ? '已理解上下文' : '语义解析完成'}</b>
      </div>
      <div className="understanding-grid">
        {changed && (
          <div className="understanding-row wide">
            <small>指代解析</small>
            <span className="question-transform">
              <del>{message.originalQuestion}</del><i>→</i><strong>{resolved}</strong>
            </span>
          </div>
        )}
        {subject && (
          <div className="understanding-row">
            <small>当前主体</small><strong>{subject}</strong>
          </div>
        )}
        {message.temporalType && (
          <div className="understanding-row">
            <small>时间类型</small>
            <strong>{TEMPORAL_LABELS[message.temporalType] || message.temporalType}</strong>
          </div>
        )}
        {referenceEvent && (
          <div className="understanding-row">
            <small>复用事件</small><strong>{referenceEvent}</strong>
          </div>
        )}
        {referenceTimestamp != null && (
          <button className="understanding-row reference-time" onClick={() => onSeek(referenceTimestamp)}>
            <small>参考时间</small><strong>{api.formatTime(referenceTimestamp)} ↗</strong>
          </button>
        )}
      </div>
    </div>
  );
}
