import React, { useEffect, useRef, useState } from 'react';
import * as api from '../api.js';

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
            <p>Ask about the video.</p>
            <p className="chat-hint">e.g. “乔治第一次什么时候出现？”</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="msg-label">{m.role === 'user' ? 'You' : 'Agent'}</div>
            <div className="msg-content">{m.content || '…'}</div>
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
            <div className="msg-label">Agent</div>
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
          placeholder={disabled ? 'Upload a video first' : 'Ask about the video…'}
          disabled={disabled || busy}
        />
        <button className="primary" type="submit" disabled={disabled || busy || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
