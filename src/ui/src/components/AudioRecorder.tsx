/**
 * Meowsformer — AudioRecorder Component
 * ========================================
 * Real-time audio recording UI that streams audio chunks via WebSocket,
 * displays partial transcription, analysis previews, and the final
 * cat-sound result.
 *
 * Replaces the old file-upload UI with live microphone input.
 */

import React, { useCallback, useState } from "react";
import {
  useStreamingTranslation,
  type StreamingState,
} from "../hooks/useStreamingTranslation";
import { useAudioPreview } from "../hooks/useAudioPreview";

// ── Sub-components ───────────────────────────────────────────────────────

function StateIndicator({ state }: { state: StreamingState }) {
  const labels: Record<StreamingState, string> = {
    idle: "就绪",
    connecting: "连接中...",
    connected: "已连接",
    recording: "录音中...",
    processing: "分析中...",
    result: "完成",
    error: "错误",
  };

  const colors: Record<StreamingState, string> = {
    idle: "#888",
    connecting: "#f0ad4e",
    connected: "#5bc0de",
    recording: "#d9534f",
    processing: "#f0ad4e",
    result: "#5cb85c",
    error: "#d9534f",
  };

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        color: colors[state],
        fontWeight: 600,
      }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          backgroundColor: colors[state],
          animation: state === "recording" ? "pulse 1s infinite" : "none",
        }}
      />
      {labels[state]}
    </span>
  );
}

// ── Main Component ───────────────────────────────────────────────────────

export const AudioRecorder: React.FC = () => {
  const [breedPreference, setBreedPreference] = useState<string>("");
  const streaming = useStreamingTranslation();
  const audioPlayer = useAudioPreview();

  const handleConnect = useCallback(() => {
    streaming.connect(breedPreference || undefined);
  }, [streaming, breedPreference]);

  const handleStartRecording = useCallback(async () => {
    if (streaming.state === "idle") {
      streaming.connect(breedPreference || undefined);
      // Wait a bit for connection
      await new Promise((r) => setTimeout(r, 500));
    }
    await streaming.startRecording();
  }, [streaming, breedPreference]);

  const handleStopRecording = useCallback(() => {
    streaming.stopRecording();
  }, [streaming]);

  // Load audio when result arrives
  React.useEffect(() => {
    if (streaming.result?.audioBase64) {
      audioPlayer.loadBase64(streaming.result.audioBase64);
    }
  }, [streaming.result?.audioBase64]);

  return (
    <div
      style={{
        maxWidth: 600,
        margin: "0 auto",
        padding: 24,
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <h2 style={{ marginBottom: 16 }}>Meowsformer — 实时猫语翻译</h2>

      {/* Status */}
      <div style={{ marginBottom: 16 }}>
        <StateIndicator state={streaming.state} />
      </div>

      {/* Breed preference */}
      <div style={{ marginBottom: 16 }}>
        <label style={{ display: "block", marginBottom: 4, fontSize: 14 }}>
          品种偏好 (可选):
        </label>
        <select
          value={breedPreference}
          onChange={(e) => setBreedPreference(e.target.value)}
          style={{ padding: "6px 12px", borderRadius: 4, border: "1px solid #ccc" }}
          disabled={streaming.state === "recording" || streaming.state === "processing"}
        >
          <option value="">自动选择</option>
          <option value="Maine Coon">Maine Coon 缅因猫</option>
          <option value="European Shorthair">European Shorthair 欧洲短毛猫</option>
        </select>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
        {streaming.state !== "recording" ? (
          <button
            onClick={handleStartRecording}
            disabled={
              streaming.state === "connecting" ||
              streaming.state === "processing"
            }
            style={{
              padding: "10px 24px",
              borderRadius: 8,
              border: "none",
              backgroundColor: "#d9534f",
              color: "#fff",
              fontSize: 16,
              cursor: "pointer",
            }}
          >
            开始录音
          </button>
        ) : (
          <button
            onClick={handleStopRecording}
            style={{
              padding: "10px 24px",
              borderRadius: 8,
              border: "none",
              backgroundColor: "#5cb85c",
              color: "#fff",
              fontSize: 16,
              cursor: "pointer",
            }}
          >
            停止录音
          </button>
        )}

        <button
          onClick={streaming.reset}
          style={{
            padding: "10px 24px",
            borderRadius: 8,
            border: "1px solid #ccc",
            backgroundColor: "#fff",
            fontSize: 16,
            cursor: "pointer",
          }}
        >
          重置
        </button>
      </div>

      {/* Partial transcription */}
      {streaming.partialText && (
        <div
          style={{
            padding: 12,
            backgroundColor: "#f7f7f7",
            borderRadius: 8,
            marginBottom: 12,
          }}
        >
          <strong>转录:</strong> {streaming.partialText}
        </div>
      )}

      {/* Analysis preview */}
      {streaming.preview && (
        <div
          style={{
            padding: 12,
            backgroundColor: "#eef6ff",
            borderRadius: 8,
            marginBottom: 12,
          }}
        >
          <strong>分析预览:</strong> 情绪: {streaming.preview.emotion},
          意图: {streaming.preview.intent}
        </div>
      )}

      {/* Error */}
      {streaming.error && (
        <div
          style={{
            padding: 12,
            backgroundColor: "#fff0f0",
            borderRadius: 8,
            marginBottom: 12,
            color: "#d9534f",
          }}
        >
          <strong>错误:</strong> {streaming.error}
        </div>
      )}

      {/* Result */}
      {streaming.result && (
        <div
          style={{
            padding: 16,
            backgroundColor: "#f0fff0",
            borderRadius: 8,
            marginBottom: 12,
          }}
        >
          <h3 style={{ marginTop: 0 }}>翻译结果</h3>

          <p>
            <strong>转录文字:</strong> {streaming.result.transcription}
          </p>

          <p>
            <strong>LLM推理:</strong> {streaming.result.reasoning}
          </p>

          {streaming.result.selectedSample && (
            <div style={{ fontSize: 14, color: "#555" }}>
              <p>
                <strong>匹配样本:</strong>{" "}
                {streaming.result.selectedSample.sample_id} (
                {streaming.result.selectedSample.breed},{" "}
                {streaming.result.selectedSample.context})
              </p>
              <p>
                <strong>匹配分数:</strong>{" "}
                {(streaming.result.selectedSample.match_score * 100).toFixed(1)}%
              </p>
            </div>
          )}

          {/* Audio playback */}
          {streaming.result.audioBase64 && (
            <div style={{ marginTop: 12 }}>
              <button
                onClick={
                  audioPlayer.state === "playing"
                    ? audioPlayer.pause
                    : audioPlayer.play
                }
                style={{
                  padding: "8px 20px",
                  borderRadius: 6,
                  border: "none",
                  backgroundColor: "#5bc0de",
                  color: "#fff",
                  fontSize: 14,
                  cursor: "pointer",
                }}
              >
                {audioPlayer.state === "playing" ? "暂停" : "播放猫语"}
              </button>
              <span style={{ marginLeft: 8, fontSize: 14, color: "#888" }}>
                {audioPlayer.currentTime.toFixed(1)}s / {audioPlayer.duration.toFixed(1)}s
              </span>
            </div>
          )}
        </div>
      )}

      {/* Pulse animation */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </div>
  );
};

export default AudioRecorder;
