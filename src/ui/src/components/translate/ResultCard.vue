<!--
  Meowsformer — streaming pipeline result card (playback + tags)
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
  },
  { immediate: true }
);

function toggle() {
  if (playbackState.value === "playing") {
    pause();
  } else {
    play();
  }
}

const colorMap = {
  purple: "bg-purple-950/55 text-purple-200 border-purple-800/60",
  blue: "bg-blue-950/55 text-blue-200 border-blue-800/60",
  green: "bg-emerald-950/50 text-emerald-200 border-emerald-800/60",
} as const;
</script>

<template>
  <div
    class="rounded-2xl border border-white/10 bg-gray-900/60 p-5 sm:p-6 backdrop-blur-md shadow-xl shadow-black/20 space-y-5"
  >
    <div>
      <p class="text-[10px] font-semibold uppercase tracking-widest text-gray-500 mb-1.5">
        You said
      </p>
      <p class="text-lg text-gray-100 leading-snug">
        “{{ transcription }}”
      </p>
    </div>

    <div
      v-if="sample"
      class="flex flex-wrap gap-2"
    >
      <span
        v-for="t in sample.tags.emotion ?? []"
        :key="`emotion-${t}`"
        :class="[
          'text-xs px-2.5 py-1 rounded-lg border font-medium',
          colorMap.purple,
        ]"
      >
        {{ t }}
      </span>
      <span
        v-for="t in sample.tags.intent ?? []"
        :key="`intent-${t}`"
        :class="[
          'text-xs px-2.5 py-1 rounded-lg border font-medium',
          colorMap.blue,
        ]"
      >
        {{ t }}
      </span>
      <span
        v-for="t in sample.tags.acoustic?.slice(0, 3) ?? []"
        :key="`acoustic-${t}`"
        :class="[
          'text-xs px-2.5 py-1 rounded-lg border font-medium',
          colorMap.green,
        ]"
      >
        {{ t }}
      </span>
    </div>

    <p
      v-if="reasoning"
      class="text-sm text-gray-400 leading-relaxed border-t border-white/5 pt-4"
    >
      {{ reasoning }}
    </p>

    <p
      v-if="sample"
      class="text-xs text-gray-500 tabular-nums"
    >
      Sample {{ sample.sample_id }} · {{ sample.breed }} · {{ sample.context }}
    </p>

    <button
      v-if="audioBase64"
      type="button"
      :disabled="playbackState === 'loading'"
      class="flex w-full items-center gap-4 rounded-2xl border border-meow-700/50 bg-gradient-to-r from-meow-900/40 to-meow-800/30 px-4 py-4 text-left transition hover:border-meow-600/70 hover:from-meow-900/60 disabled:opacity-50"
      @click="toggle"
    >
      <span
        class="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-meow-600/30 text-xl text-meow-100"
        aria-hidden="true"
      >
        {{ playbackState === "playing" ? "⏸" : "▶" }}
      </span>
      <div class="min-w-0 flex-1">
        <p class="text-sm font-semibold text-meow-200">
          {{ playbackState === "playing" ? "Pause meow" : "Play meow" }}
        </p>
        <p
          v-if="sample"
          class="text-xs text-gray-500 mt-0.5"
        >
          Match {{ (sample.match_score * 100).toFixed(1) }}%
        </p>
      </div>
    </button>
  </div>
</template>
