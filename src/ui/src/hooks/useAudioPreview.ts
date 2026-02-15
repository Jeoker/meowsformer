/**
 * Meowsformer — useAudioPreview Hook
 * =====================================
 * Manages the lifecycle of audio preview playback from a base64-encoded WAV.
 *
 * Responsibilities:
 * - Decode base64 → Blob → Object URL
 * - Control play / pause / stop
 * - Track playback state for the confirm button
 * - Clean up Object URLs on unmount to prevent memory leaks
 */

import { useCallback, useEffect, useRef, useState } from "react";

export type PlaybackState = "idle" | "loading" | "playing" | "paused" | "ended";

export interface AudioPreviewControls {
  /** Current playback state */
  state: PlaybackState;
  /** Duration of the loaded audio in seconds (0 if not loaded) */
  duration: number;
  /** Current playback position in seconds */
  currentTime: number;
  /** Whether the user has listened to at least part of the audio */
  hasListened: boolean;
  /** Start or resume playback */
  play: () => void;
  /** Pause playback */
  pause: () => void;
  /** Stop and reset to beginning */
  stop: () => void;
  /** Load new audio from base64 */
  loadBase64: (base64Wav: string) => void;
  /** Reset all state */
  reset: () => void;
}

/**
 * Hook for managing audio preview playback from base64-encoded WAV data.
 *
 * @example
 * ```tsx
 * const audio = useAudioPreview();
 *
 * useEffect(() => {
 *   if (response?.audio_base64) {
 *     audio.loadBase64(response.audio_base64);
 *   }
 * }, [response]);
 *
 * return (
 *   <button onClick={audio.play} disabled={audio.state === "playing"}>
 *     Play Preview
 *   </button>
 * );
 * ```
 */
export function useAudioPreview(): AudioPreviewControls {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const [state, setState] = useState<PlaybackState>("idle");
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [hasListened, setHasListened] = useState(false);

  // ── Cleanup helper ────────────────────────────────────────────────
  const revokeUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);

  // ── Load base64 WAV ───────────────────────────────────────────────
  const loadBase64 = useCallback(
    (base64Wav: string) => {
      setState("loading");
      revokeUrl();

      try {
        // Decode base64 → binary → Blob
        const binaryString = atob(base64Wav);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        objectUrlRef.current = url;

        // Create or reuse audio element
        if (!audioRef.current) {
          audioRef.current = new Audio();
        }
        const audio = audioRef.current;
        audio.src = url;
        audio.load();

        // Event listeners
        audio.onloadedmetadata = () => {
          setDuration(audio.duration);
          setState("idle");
        };
        audio.ontimeupdate = () => {
          setCurrentTime(audio.currentTime);
        };
        audio.onplay = () => {
          setState("playing");
          setHasListened(true);
        };
        audio.onpause = () => {
          if (audio.currentTime < audio.duration) {
            setState("paused");
          }
        };
        audio.onended = () => {
          setState("ended");
          setCurrentTime(0);
        };
        audio.onerror = () => {
          setState("idle");
          console.error("Audio playback error");
        };
      } catch (err) {
        console.error("Failed to decode base64 audio:", err);
        setState("idle");
      }
    },
    [revokeUrl]
  );

  // ── Playback controls ─────────────────────────────────────────────
  const play = useCallback(() => {
    audioRef.current?.play().catch(console.error);
  }, []);

  const pause = useCallback(() => {
    audioRef.current?.pause();
  }, []);

  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setCurrentTime(0);
    setState("idle");
  }, []);

  const reset = useCallback(() => {
    stop();
    revokeUrl();
    if (audioRef.current) {
      audioRef.current.src = "";
    }
    setDuration(0);
    setHasListened(false);
    setState("idle");
  }, [stop, revokeUrl]);

  // ── Cleanup on unmount ────────────────────────────────────────────
  useEffect(() => {
    return () => {
      audioRef.current?.pause();
      revokeUrl();
    };
  }, [revokeUrl]);

  return {
    state,
    duration,
    currentTime,
    hasListened,
    play,
    pause,
    stop,
    loadBase64,
    reset,
  };
}
