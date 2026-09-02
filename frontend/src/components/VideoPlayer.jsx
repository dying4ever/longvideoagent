import React, { useRef } from 'react';
import * as api from '../api.js';

export default function VideoPlayer({ videoRef, videoId, onUpload, stage, duration }) {
  const fileRef = useRef(null);

  if (!videoId) {
    return (
      <div className="video-empty">
        <div className="video-empty-inner">
          <p className="video-empty-title">Upload a video to start</p>
          <p className="video-empty-sub">.mp4 · .mov · .mkv · .webm</p>
          <button className="primary" onClick={() => fileRef.current?.click()}>
            Choose video
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
          <p>{stage === 'uploading' ? 'Uploading…' : 'Preparing video memory…'}</p>
        </div>
      )}
      {duration != null && <div className="video-duration">{api.formatTime(duration)}</div>}
    </div>
  );
}
