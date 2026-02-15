/**
 * Meowsformer — MeowPreviewPlayer Component
 * ============================================
 * Audio preview player with playback controls, confidence description
 * display, and a mandatory "Confirm & Send" button.
 *
 * Design contract:
 * - The user MUST be able to hear the synthesised audio before sending.
 * - The "Send" button is disabled until the user has listened at least once.
 * - The confidence description (NatureLM-audio style) is shown alongside
 *   the player to build user trust in the system's output.
 *
 * Props:
 * - audioBase64: base64-encoded WAV from the API
 * - previewDescription: structured description from the backend
 * - synthesisMetadata: technical metadata
 * - onConfirm: callback when user confirms the audio
 * - onReject: callback when user wants to re-generate
 */

import React, { useEffect, useMemo } from "react";
import { useAudioPreview, type PlaybackState } from "../hooks/useAudioPreview";
import type { PreviewDescription, SynthesisMetadata } from "../types/api";

// ── Props ────────────────────────────────────────────────────────────────

export interface MeowPreviewPlayerProps {
  /** Base64-encoded WAV audio from the synthesis API */
  audioBase64: string | null;
  /** NatureLM-audio-style preview description */
  previewDescription: PreviewDescription | null;
  /** Technical synthesis metadata */
  synthesisMetadata: SynthesisMetadata | null;
  /** Called when user confirms the synthesised audio for sending */
  onConfirm: () => void;
  /** Called when user rejects and wants to re-generate */
  onReject: () => void;
  /** Whether the component is in a loading state */
  isLoading?: boolean;
}

// ── Sub-components ───────────────────────────────────────────────────────

/** Playback control button */
function PlayButton({
  state,
  onPlay,
  onPause,
}: {
  state: PlaybackState;
  onPlay: () => void;
  onPause: () => void;
}) {
  const isPlaying = state === "playing";

  return (
    <button
      type="button"
      onClick={isPlaying ? onPause : onPlay}
      disabled={state === "loading"}
      className="meow-preview__play-btn"
      aria-label={isPlaying ? "暂停预览" : "播放预览"}
    >
      {state === "loading" ? "⏳" : isPlaying ? "⏸ 暂停" : "▶ 播放预览"}
    </button>
  );
}

/** Progress bar showing playback position */
function ProgressBar({
  currentTime,
  duration,
}: {
  currentTime: number;
  duration: number;
}) {
  const pct = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="meow-preview__progress" role="progressbar" aria-valuenow={pct}>
      <div
        className="meow-preview__progress-fill"
        style={{ width: `${pct}%` }}
      />
      <span className="meow-preview__progress-time">
        {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
      </span>
    </div>
  );
}

/** Confidence description panel */
function DescriptionPanel({
  description,
}: {
  description: PreviewDescription;
}) {
  return (
    <div className="meow-preview__description">
      {/* Summary line */}
      <p className="meow-preview__summary">{description.summary}</p>

      {/* Confidence badge */}
      <div className="meow-preview__confidence">
        <span className="meow-preview__badge">
          置信度: {description.confidence_level} (
          {(description.confidence_score * 100).toFixed(0)}%)
        </span>
        <span className="meow-preview__badge">
          VA距离: {description.va_distance.toFixed(3)}
        </span>
        <span className="meow-preview__badge">
          品种: {description.breed}
        </span>
      </div>

      {/* Detailed breakdown (expandable) */}
      <details className="meow-preview__details">
        <summary>查看详细分析</summary>
        <pre className="meow-preview__detail-text">{description.detail}</pre>
      </details>
    </div>
  );
}

/** Metadata display */
function MetadataPanel({
  metadata,
}: {
  metadata: SynthesisMetadata;
}) {
  return (
    <div className="meow-preview__metadata">
      <small>
        样本: {metadata.matched_sample_id} | 品种: {metadata.matched_breed} |
        场景: {metadata.matched_context} | 时长: {metadata.duration_seconds.toFixed(2)}s |
        采样率: {metadata.sample_rate} Hz
      </small>
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────

/**
 * MeowPreviewPlayer — Preview & confirm synthesised cat vocalisation.
 *
 * The user must listen to the audio at least once before the "Confirm"
 * button becomes active. This prevents accidental sends of unexpected audio.
 */
export const MeowPreviewPlayer: React.FC<MeowPreviewPlayerProps> = ({
  audioBase64,
  previewDescription,
  synthesisMetadata,
  onConfirm,
  onReject,
  isLoading = false,
}) => {
  const audio = useAudioPreview();

  // Load audio when base64 data arrives
  useEffect(() => {
    if (audioBase64) {
      audio.loadBase64(audioBase64);
    } else {
      audio.reset();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioBase64]);

  // Confirm button is only enabled after the user has listened
  const canConfirm = audio.hasListened && !isLoading;

  // If no audio data, show nothing or a placeholder
  if (!audioBase64 && !isLoading) {
    return null;
  }

  return (
    <div className="meow-preview" role="region" aria-label="猫语合成预览">
      {/* Loading overlay */}
      {isLoading && (
        <div className="meow-preview__loading">
          <span>正在合成猫语...</span>
        </div>
      )}

      {/* Description panel */}
      {previewDescription && (
        <DescriptionPanel description={previewDescription} />
      )}

      {/* Audio player controls */}
      {audioBase64 && (
        <div className="meow-preview__player">
          <PlayButton
            state={audio.state}
            onPlay={audio.play}
            onPause={audio.pause}
          />
          <ProgressBar
            currentTime={audio.currentTime}
            duration={audio.duration}
          />
        </div>
      )}

      {/* Metadata */}
      {synthesisMetadata && <MetadataPanel metadata={synthesisMetadata} />}

      {/* Confirm / Reject actions */}
      <div className="meow-preview__actions">
        <button
          type="button"
          onClick={onConfirm}
          disabled={!canConfirm}
          className="meow-preview__confirm-btn"
          title={
            canConfirm
              ? "确认并发送此猫语"
              : "请先播放预览音频"
          }
        >
          ✓ 确认发送
        </button>
        <button
          type="button"
          onClick={onReject}
          disabled={isLoading}
          className="meow-preview__reject-btn"
        >
          ✗ 重新生成
        </button>
      </div>

      {/* Listen-first hint */}
      {!audio.hasListened && audioBase64 && (
        <p className="meow-preview__hint">
          请先播放预览音频，确认满意后再点击"确认发送"。
        </p>
      )}
    </div>
  );
};

export default MeowPreviewPlayer;
