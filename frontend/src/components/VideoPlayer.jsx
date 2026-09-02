import React, { useRef } from 'react';
import * as api from '../api.js';

function buildLabel(p) {
  if (p?.phase === 'loading') return '正在加载模型…';
  if (p?.phase === 'segmenting') return `正在切分事件：第 ${p.window + 1} / ${p.total} 个窗口`;
  if (p?.phase === 'hierarchy') return '正在生成摘要与章节…';
  return '正在构建视频记忆…';
}

export default function VideoPlayer({ videoRef, videoId, onUpload, stage, duration, buildProgress }) {
  const fileRef = useRef(null);

  if (!videoId) {
    return (
      <div className="video-empty">
        <div className="video-empty-inner">
          <div className="video-empty-icon">▶</div>
          <p className="video-empty-title">上传一个长视频开始分析</p>
          <p className="video-empty-sub">支持 .mp4 · .mov · .mkv · .webm</p>
          <button className="primary" onClick={() => fileRef.current?.click()}>
            选择视频
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".mp4,.mov,.mkv,.webm"
            hidden
            onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="video-wrap">
      <video ref={videoRef} src={api.videoUrl(videoId)} controls className="video" />
      {stage && (
        <div className="video-overlay">
          <div className="spinner" />
          <p>{stage === 'uploading' ? '正在上传视频…' : buildLabel(buildProgress)}</p>
          {stage === 'building' && buildProgress?.phase === 'segmenting' && (
            <div className="build-progress">
              <div
                className="build-progress-bar"
                style={{ width: `${Math.round((buildProgress.window / buildProgress.total) * 100)}%` }}
              />
            </div>
          )}
        </div>
      )}
      {duration != null && <div className="video-duration">{api.formatTime(duration)}</div>}
    </div>
  );
}
