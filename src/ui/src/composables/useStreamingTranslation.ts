/**
 * Meowsformer — useStreamingTranslation Composable
 * ===================================================
 * Manages WebSocket connection lifecycle, audio chunk streaming,
 * and progressive result handling for real-time cat-sound translation.
 */

import { ref, shallowRef, onUnmounted } from "vue";
import type { WSServerMessage, TaggedSampleInfo } from "../types/api";

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

// ── Configuration ───────────────────────────────────────────────────────

const SAMPLE_RATE = 16000;

function getWsUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws/translate`;
}

// ── Composable ──────────────────────────────────────────────────────────

export function useStreamingTranslation() {
  const ws = shallowRef<WebSocket | null>(null);
  const mediaStream = shallowRef<MediaStream | null>(null);
  const processor = shallowRef<ScriptProcessorNode | null>(null);
  const audioCtx = shallowRef<AudioContext | null>(null);

  const state = ref<StreamingState>("idle");
  const partialText = ref("");
  const preview = ref<{ emotion: string; intent: string } | null>(null);
  const result = ref<StreamingResult | null>(null);
  const error = ref<string | null>(null);

  // ── WebSocket message handler ───────────────────────────────────────

  function handleMessage(event: MessageEvent) {
    try {
      const msg: WSServerMessage = JSON.parse(event.data);

      switch (msg.type) {
        case "transcription":
          partialText.value = msg.text;
          if (msg.is_final) {
            state.value = "processing";
          }
          break;

        case "analysis_preview":
          preview.value = { emotion: msg.emotion, intent: msg.intent };
          break;

        case "result":
          result.value = {
            transcription: msg.transcription,
            selectedSample: msg.selected_category,
            audioBase64: msg.audio_base64,
            reasoning: msg.reasoning,
          };
          state.value = "result";
          break;

        case "error":
          error.value = msg.detail;
          state.value = "error";
          break;
      }
    } catch (e) {
      console.error("Failed to parse WS message:", e);
    }
  }

  // ── Connect ─────────────────────────────────────────────────────────

  function connect(breedPreference?: string) {
    if (ws.value?.readyState === WebSocket.OPEN) return;

    state.value = "connecting";
    const socket = new WebSocket(getWsUrl());
    ws.value = socket;

    socket.onopen = () => {
      state.value = "connected";
      if (breedPreference) {
        socket.send(
          JSON.stringify({ type: "config", breed_preference: breedPreference })
        );
      }
    };

    socket.onmessage = handleMessage;

    socket.onerror = () => {
      error.value = "WebSocket connection error";
      state.value = "error";
    };

    socket.onclose = () => {
      if (state.value !== "error" && state.value !== "result") {
        state.value = "idle";
      }
    };
  }

  // ── Start Recording ─────────────────────────────────────────────────

  async function startRecording() {
    const socket = ws.value;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      error.value = "WebSocket not connected";
      state.value = "error";
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: SAMPLE_RATE,
        },
      });
      mediaStream.value = stream;

      const context = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioCtx.value = context;

      const source = context.createMediaStreamSource(stream);
      // Buffer size 4096 at 16kHz ≈ 256ms
      const proc = context.createScriptProcessor(4096, 1, 1);
      processor.value = proc;

      proc.onaudioprocess = (e) => {
        if (socket.readyState !== WebSocket.OPEN) return;

        const float32 = e.inputBuffer.getChannelData(0);
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]));
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        ws.value?.send(int16.buffer);
      };

      source.connect(proc);
      proc.connect(context.destination);

      state.value = "recording";
      partialText.value = "";
      preview.value = null;
      result.value = null;
      error.value = null;
    } catch (e) {
      error.value = `Microphone access denied: ${e}`;
      state.value = "error";
    }
  }

  // ── Stop Recording ──────────────────────────────────────────────────

  function stopRecording() {
    if (mediaStream.value) {
      mediaStream.value.getTracks().forEach((t) => t.stop());
      mediaStream.value = null;
    }
    if (processor.value) {
      processor.value.disconnect();
      processor.value = null;
    }
    if (audioCtx.value) {
      audioCtx.value.close();
      audioCtx.value = null;
    }

    const socket = ws.value;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "stop" }));
      state.value = "processing";
    }
  }

  // ── Disconnect ──────────────────────────────────────────────────────

  function disconnect() {
    stopRecording();
    if (ws.value) {
      ws.value.close();
      ws.value = null;
    }
    state.value = "idle";
  }

  // ── Reset ───────────────────────────────────────────────────────────

  function reset() {
    disconnect();
    partialText.value = "";
    preview.value = null;
    result.value = null;
    error.value = null;
    state.value = "idle";
  }

  // ── Cleanup on unmount ──────────────────────────────────────────────

  onUnmounted(() => {
    mediaStream.value?.getTracks().forEach((t) => t.stop());
    processor.value?.disconnect();
    audioCtx.value?.close();
    ws.value?.close();
  });

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
