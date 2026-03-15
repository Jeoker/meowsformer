/**
 * Meowsformer — useAudioPreview Composable
 * ==========================================
 * Manages the lifecycle of audio preview playback from a base64-encoded WAV.
 *
 * Responsibilities:
 * - Decode base64 → Blob → Object URL
 * - Control play / pause / stop
 * - Track playback state for the confirm button
 * - Clean up Object URLs on unmount to prevent memory leaks
 */

import { ref, shallowRef, onUnmounted } from "vue";

export type PlaybackState = "idle" | "loading" | "playing" | "paused" | "ended";

/**
 * Composable for managing audio preview playback from base64-encoded WAV data.
 *
 * @example
 * ```vue
 * <script setup>
 * import { watch } from 'vue'
 * import { useAudioPreview } from '../composables/useAudioPreview'
 *
 * const { state, play, loadBase64 } = useAudioPreview()
 *
 * watch(() => response.value?.audio_base64, (b64) => {
 *   if (b64) loadBase64(b64)
 * })
 * </script>
 * <template>
 *   <button @click="play" :disabled="state === 'playing'">Play Preview</button>
 * </template>
 * ```
 */
export function useAudioPreview() {
  const audioEl = shallowRef<HTMLAudioElement | null>(null);
  let objectUrl: string | null = null;

  const state = ref<PlaybackState>("idle");
  const duration = ref(0);
  const currentTime = ref(0);
  const hasListened = ref(false);

  // ── Cleanup helper ────────────────────────────────────────────────
  function revokeUrl() {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
  }

  // ── Load base64 WAV ───────────────────────────────────────────────
  function loadBase64(base64Wav: string) {
    state.value = "loading";
    revokeUrl();

    try {
      const binaryString = atob(base64Wav);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      objectUrl = url;

      if (!audioEl.value) {
        audioEl.value = new Audio();
      }
      const audio = audioEl.value;
      audio.src = url;
      audio.load();

      audio.onloadedmetadata = () => {
        duration.value = audio.duration;
        state.value = "idle";
      };
      audio.ontimeupdate = () => {
        currentTime.value = audio.currentTime;
      };
      audio.onplay = () => {
        state.value = "playing";
        hasListened.value = true;
      };
      audio.onpause = () => {
        if (audio.currentTime < audio.duration) {
          state.value = "paused";
        }
      };
      audio.onended = () => {
        state.value = "ended";
        currentTime.value = 0;
      };
      audio.onerror = () => {
        state.value = "idle";
        console.error("Audio playback error");
      };
    } catch (err) {
      console.error("Failed to decode base64 audio:", err);
      state.value = "idle";
    }
  }

  // ── Playback controls ─────────────────────────────────────────────
  function play() {
    audioEl.value?.play().catch(console.error);
  }

  function pause() {
    audioEl.value?.pause();
  }

  function stop() {
    if (audioEl.value) {
      audioEl.value.pause();
      audioEl.value.currentTime = 0;
    }
    currentTime.value = 0;
    state.value = "idle";
  }

  function reset() {
    stop();
    revokeUrl();
    if (audioEl.value) {
      audioEl.value.src = "";
    }
    duration.value = 0;
    hasListened.value = false;
    state.value = "idle";
  }

  // ── Cleanup on unmount ────────────────────────────────────────────
  onUnmounted(() => {
    audioEl.value?.pause();
    revokeUrl();
  });

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
