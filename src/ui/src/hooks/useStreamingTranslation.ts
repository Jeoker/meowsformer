/**
 * Meowsformer — useStreamingTranslation Hook
 * =============================================
 * Manages WebSocket connection lifecycle, audio chunk streaming,
 * and progressive result handling for real-time cat-sound translation.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  WSServerMessage,
  TaggedSampleInfo,
} from "../types/api";

// ── State types ─────────────────────────────────────────────────────────

export type StreamingState =
  | "idle"
  | "connecting"
  | "connected"
  | "recording"
  | "processing"
  | "result"
  | "error";

export interface StreamingResult {
  transcription: string;
  selectedSample: TaggedSampleInfo | null;
  audioBase64: string | null;
  reasoning: string;
}

export interface StreamingTranslationControls {
  /** Current connection/recording state */
  state: StreamingState;
  /** Partial transcription text (updated in real-time) */
  partialText: string;
  /** Analysis preview (emotion + intent, before final result) */
  preview: { emotion: string; intent: string } | null;
  /** Final result after processing completes */
  result: StreamingResult | null;
  /** Error message if something went wrong */
  error: string | null;
  /** Connect to the WebSocket server */
  connect: (breedPreference?: string) => void;
  /** Start recording and streaming audio */
  startRecording: () => Promise<void>;
  /** Stop recording and trigger final processing */
  stopRecording: () => void;
  /** Disconnect from the WebSocket server */
  disconnect: () => void;
  /** Reset all state for a new session */
  reset: () => void;
}

// ── Configuration ───────────────────────────────────────────────────────

const CHUNK_INTERVAL_MS = 200;
const SAMPLE_RATE = 16000;

function getWsUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws/translate`;
}

// ── Hook ────────────────────────────────────────────────────────────────

export function useStreamingTranslation(): StreamingTranslationControls {
  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const contextRef = useRef<AudioContext | null>(null);

  const [state, setState] = useState<StreamingState>("idle");
  const [partialText, setPartialText] = useState("");
  const [preview, setPreview] = useState<{
    emotion: string;
    intent: string;
  } | null>(null);
  const [result, setResult] = useState<StreamingResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // ── WebSocket message handler ───────────────────────────────────────

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const msg: WSServerMessage = JSON.parse(event.data);

      switch (msg.type) {
        case "transcription":
          setPartialText(msg.text);
          if (msg.is_final) {
            setState("processing");
          }
          break;

        case "analysis_preview":
          setPreview({ emotion: msg.emotion, intent: msg.intent });
          break;

        case "result":
          setResult({
            transcription: msg.transcription,
            selectedSample: msg.selected_category,
            audioBase64: msg.audio_base64,
            reasoning: msg.reasoning,
          });
          setState("result");
          break;

        case "error":
          setError(msg.detail);
          setState("error");
          break;
      }
    } catch (e) {
      console.error("Failed to parse WS message:", e);
    }
  }, []);

  // ── Connect ─────────────────────────────────────────────────────────

  const connect = useCallback(
    (breedPreference?: string) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) return;

      setState("connecting");
      const ws = new WebSocket(getWsUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        setState("connected");
        if (breedPreference) {
          ws.send(
            JSON.stringify({ type: "config", breed_preference: breedPreference })
          );
        }
      };

      ws.onmessage = handleMessage;

      ws.onerror = () => {
        setError("WebSocket connection error");
        setState("error");
      };

      ws.onclose = () => {
        if (state !== "error" && state !== "result") {
          setState("idle");
        }
      };
    },
    [handleMessage, state]
  );

  // ── Start Recording ─────────────────────────────────────────────────

  const startRecording = useCallback(async () => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError("WebSocket not connected");
      setState("error");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: SAMPLE_RATE,
        },
      });
      mediaStreamRef.current = stream;

      const audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
      contextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      // Buffer size 4096 at 16kHz ≈ 256ms
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (ws.readyState !== WebSocket.OPEN) return;

        const float32 = e.inputBuffer.getChannelData(0);
        // Convert float32 → int16 PCM
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]));
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        ws.send(int16.buffer);
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      setState("recording");
      setPartialText("");
      setPreview(null);
      setResult(null);
      setError(null);
    } catch (e) {
      setError(`Microphone access denied: ${e}`);
      setState("error");
    }
  }, []);

  // ── Stop Recording ──────────────────────────────────────────────────

  const stopRecording = useCallback(() => {
    // Stop media stream
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    // Disconnect processor
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    // Close audio context
    if (contextRef.current) {
      contextRef.current.close();
      contextRef.current = null;
    }

    // Send stop signal
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
      setState("processing");
    }
  }, []);

  // ── Disconnect ──────────────────────────────────────────────────────

  const disconnect = useCallback(() => {
    stopRecording();
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setState("idle");
  }, [stopRecording]);

  // ── Reset ───────────────────────────────────────────────────────────

  const reset = useCallback(() => {
    disconnect();
    setPartialText("");
    setPreview(null);
    setResult(null);
    setError(null);
    setState("idle");
  }, [disconnect]);

  // ── Cleanup on unmount ──────────────────────────────────────────────

  useEffect(() => {
    return () => {
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
      processorRef.current?.disconnect();
      contextRef.current?.close();
      wsRef.current?.close();
    };
  }, []);

  return {
    state,
    partialText,
    preview,
    result,
    error,
    connect,
    startRecording,
    stopRecording,
    disconnect,
    reset,
  };
}
