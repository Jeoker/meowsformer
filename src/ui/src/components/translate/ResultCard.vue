<!--
  Meowsformer — ResultCard Component
  ====================================
  Tailwind dark-theme result card with emotion/intent/acoustic tag badges,
  match score display, and audio playback button.
-->

<script setup lang="ts">
import { watch } from "vue";
import { useAudioPreview } from "../../composables/useAudioPreview";
import type { TaggedSampleInfo } from "../../types/api";

const props = defineProps<{
  transcription: string;
  sample: TaggedSampleInfo | null;
  audioBase64: string | null;
  reasoning: string;
}>();

const {
  state: playbackState,
  play,
  pause,
  loadBase64,
} = useAudioPreview();

watch(
  () => props.audioBase64,
  (base64) => {
    if (base64) loadBase64(base64);
  }
);

function toggle() {
  if (playbackState.value === "playing") {
    pause();
  } else {
    play();
  }
}

const colorMap = {
  purple: "bg-purple-900/40 text-purple-300 border-purple-800",
  blue: "bg-blue-900/40 text-blue-300 border-blue-800",
  green: "bg-green-900/40 text-green-300 border-green-800",
} as const;
</script>

<template>
  <div class="bg-gray-900 border border-gray-800 rounded-2xl p-6 space-y-4">
    <!-- Transcription -->
    <div>
      <p class="text-xs uppercase tracking-widest text-gray-500 mb-1">
        You said
      </p>
      <p class="text-gray-200 italic">"{{ transcription }}"</p>
    </div>

    <!-- Tags -->
    <div v-if="sample" class="flex flex-wrap gap-2">
      <span
        v-for="t in sample.tags.emotion"
        :key="`emotion-${t}`"
        :class="[
          'text-xs px-2.5 py-1 rounded-full border font-medium',
          colorMap.purple,
        ]"
      >
        {{ t }}
      </span>
      <span
        v-for="t in sample.tags.intent"
        :key="`intent-${t}`"
        :class="[
          'text-xs px-2.5 py-1 rounded-full border font-medium',
          colorMap.blue,
        ]"
      >
        {{ t }}
      </span>
      <span
        v-for="t in sample.tags.acoustic?.slice(0, 3)"
        :key="`acoustic-${t}`"
        :class="[
          'text-xs px-2.5 py-1 rounded-full border font-medium',
          colorMap.green,
        ]"
      >
        {{ t }}
      </span>
    </div>

    <!-- Reasoning -->
    <p
      v-if="reasoning"
      class="text-sm text-gray-400 leading-relaxed"
    >
      {{ reasoning }}
    </p>

    <!-- Audio player -->
    <button
      v-if="audioBase64"
      @click="toggle"
      :disabled="playbackState === 'loading'"
      class="flex items-center gap-3 bg-meow-600/20 hover:bg-meow-600/30 border border-meow-700 rounded-xl px-4 py-3 w-full transition group disabled:opacity-50"
    >
      <span class="text-2xl">{{
        playbackState === "playing" ? "⏸" : "▶️"
      }}</span>
      <div class="text-left">
        <p
          class="text-sm font-medium text-meow-300 group-hover:text-meow-200"
        >
          {{ playbackState === "playing" ? "Pause meow" : "Play meow" }}
        </p>
        <p v-if="sample" class="text-xs text-gray-500">
          Match score: {{ (sample.match_score * 100).toFixed(0) }}%
        </p>
      </div>
    </button>
  </div>
</template>
