<script setup lang="ts">
import { computed, watch } from "vue";
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

function humanize(value: string): string {
  return value.replace(/_/g, " ");
}

const emotion = computed(() => props.sample?.tags.emotion?.[0] ?? "expressive");
const intent = computed(() => props.sample?.tags.intent?.[0] ?? "communicative");

const colorMap = {
  purple: "bg-purple-950/55 text-purple-200 border-purple-800/60",
  blue: "bg-blue-950/55 text-blue-200 border-blue-800/60",
  green: "bg-emerald-950/50 text-emerald-200 border-emerald-800/60",
} as const;
</script>

<template>
  <div
    class="overflow-hidden rounded-3xl border border-white/10 bg-gray-950/65 shadow-xl shadow-black/20"
  >
    <div class="border-b border-white/5 p-5 sm:p-6">
      <p class="text-[10px] font-semibold uppercase tracking-widest text-gray-500">You said</p>
      <p class="mt-2 text-lg leading-snug text-gray-100">“{{ transcription }}”</p>
    </div>

    <div class="space-y-5 p-5 sm:p-6">
      <div v-if="sample">
        <p class="text-xs font-medium text-meow-400">Closest cat voice</p>
        <h3 class="mt-1 text-xl font-semibold capitalize tracking-tight text-white">
          A {{ humanize(emotion) }} meow with {{ humanize(intent) }} intent
        </h3>
      </div>

      <button
        v-if="audioBase64"
        type="button"
        :disabled="playbackState === 'loading'"
        class="flex w-full items-center gap-4 rounded-2xl border border-meow-700/50 bg-gradient-to-r from-meow-900/50 to-meow-800/25 px-4 py-4 text-left transition hover:border-meow-500/70 disabled:opacity-50"
        @click="toggle"
      >
        <span class="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-meow-500/25 text-xl text-meow-100" aria-hidden="true">
          {{ playbackState === "playing" ? "Ⅱ" : "▶" }}
        </span>
        <span class="min-w-0 flex-1">
          <span class="block text-sm font-semibold text-meow-100">
            {{ playbackState === "playing" ? "Pause cat voice" : "Hear your cat voice" }}
          </span>
          <span v-if="sample" class="mt-0.5 block text-xs text-gray-500">
            {{ (sample.match_score * 100).toFixed(1) }}% weighted tag match
          </span>
        </span>
      </button>

      <div v-if="sample" class="flex flex-wrap gap-2">
        <span
          v-for="tag in sample.tags.emotion?.slice(0, 2) ?? []"
          :key="`emotion-${tag}`"
          :class="['rounded-lg border px-2.5 py-1 text-xs font-medium capitalize', colorMap.purple]"
        >
          {{ humanize(tag) }}
        </span>
        <span
          v-for="tag in sample.tags.intent?.slice(0, 2) ?? []"
          :key="`intent-${tag}`"
          :class="['rounded-lg border px-2.5 py-1 text-xs font-medium capitalize', colorMap.blue]"
        >
          {{ humanize(tag) }}
        </span>
        <span
          v-for="tag in sample.tags.acoustic?.slice(0, 3) ?? []"
          :key="`acoustic-${tag}`"
          :class="['rounded-lg border px-2.5 py-1 text-xs font-medium capitalize', colorMap.green]"
        >
          {{ humanize(tag) }}
        </span>
      </div>

      <details v-if="sample || reasoning" class="group rounded-2xl border border-white/5 bg-white/[0.025] px-4 py-3">
        <summary class="flex cursor-pointer list-none items-center justify-between gap-4 text-sm font-medium text-gray-300">
          Why this match?
          <span class="text-gray-600 transition group-open:rotate-180" aria-hidden="true">⌄</span>
        </summary>
        <div class="space-y-4 pt-4 text-sm leading-relaxed text-gray-400">
          <p v-if="reasoning">{{ reasoning }}</p>
          <p v-if="sample" class="border-t border-white/5 pt-3 text-xs text-gray-600">
            Sample {{ sample.sample_id }} · {{ sample.breed }} · {{ sample.context }} context
          </p>
        </div>
      </details>
    </div>
  </div>
</template>
