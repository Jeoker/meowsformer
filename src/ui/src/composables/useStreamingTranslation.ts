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
const WS_CONNECT_TIMEOUT_MS = 25_000;

/**
 * Browsers often ignore AudioContext({ sampleRate: 16000 }) and run at 44100/48000 Hz.
 * The backend assumes 16 kHz PCM; without resampling, Whisper sees wrong-speed audio
 * and returns unrelated or hallucinated text.
 */
function resampleFloat32Linear(
  input: Float32Array,
  sampleRateIn: number,
  sampleRateOut: number
): Float32Array {
  if (sampleRateIn === sampleRateOut) {
    return input;
  }
  const ratio = sampleRateIn / sampleRateOut;
  const outLen = Math.max(1, Math.floor(input.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcPos = i * ratio;
    const i0 = Math.floor(srcPos);
    const frac = srcPos - i0;
    const a = input[i0] ?? 0;
    const b = input[i0 + 1] ?? a;
    out[i] = a + frac * (b - a);
  }
  return out;
}

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
  let pendingConnect: Promise<void> | null = null;

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

  function connect(breedPreference?: string): Promise<void> {
    if (ws.value?.readyState === WebSocket.OPEN) {
      return Promise.resolve();
    }
    if (pendingConnect) {
      return pendingConnect;
    }

    state.value = "connecting";
    error.value = null;
    const socket = new WebSocket(getWsUrl());
    ws.value = socket;

    socket.onmessage = handleMessage;

    pendingConnect = new Promise<void>((resolve, reject) => {
      const timeoutId = window.setTimeout(() => {
        error.value = "WebSocket connection timed out. Please try again.";
        state.value = "error";
        socket.close();
        reject(new Error("WebSocket connection timed out"));
      }, WS_CONNECT_TIMEOUT_MS);

      socket.onopen = () => {
        window.clearTimeout(timeoutId);
        state.value = "connected";
        if (breedPreference) {
          socket.send(
            JSON.stringify({ type: "config", breed_preference: breedPreference })
          );
        }
        resolve();
      };

      socket.onerror = () => {
        window.clearTimeout(timeoutId);
        error.value = "WebSocket connection error. Please try again.";
        state.value = "error";
        reject(new Error("WebSocket connection error"));
      };

      socket.onclose = () => {
        window.clearTimeout(timeoutId);
        if (ws.value === socket) {
          ws.value = null;
        }
        if (state.value !== "error" && state.value !== "result") {
          state.value = "idle";
        }
        reject(new Error("WebSocket closed before connecting"));
      };
    }).finally(() => {
      pendingConnect = null;
    });

    return pendingConnect;
  }

  /** Stop mic / processor / AudioContext without telling the server (used when closing WS). */
  function cleanupAudioCapture() {
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
      await context.resume();

      const source = context.createMediaStreamSource(stream);
      // Buffer size 4096 at 16kHz ≈ 256ms
      const proc = context.createScriptProcessor(4096, 1, 1);
      processor.value = proc;

      proc.onaudioprocess = (e) => {
        if (socket.readyState !== WebSocket.OPEN) return;

        const raw = e.inputBuffer.getChannelData(0);
        const inRate = e.inputBuffer.sampleRate;
        const float32 =
          inRate === SAMPLE_RATE
            ? raw
            : resampleFloat32Linear(raw, inRate, SAMPLE_RATE);

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
    cleanupAudioCapture();

    const socket = ws.value;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "stop" }));
      state.value = "processing";
    }
  }

  // ── Disconnect ──────────────────────────────────────────────────────

  function disconnect() {
    const wasRecording = state.value === "recording";

    cleanupAudioCapture();

    const socket = ws.value;
    if (socket && socket.readyState === WebSocket.OPEN) {
      // Only notify the server if we were still recording. Sending "stop" while idle
      // (e.g. user hit Reset after a result) makes the server run the pipeline on an
      // empty buffer and can confuse the next session on some timing paths.
      if (wasRecording) {
        socket.send(JSON.stringify({ type: "stop" }));
      }
      socket.close();
      ws.value = null;
    } else if (ws.value) {
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
    cleanupAudioCapture();
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
