<script setup lang="ts">
import { computed } from "vue";
import type { StreamingState } from "../../composables/useStreamingTranslation";

const props = defineProps<{
  state: StreamingState;
}>();

const emit = defineEmits<{
  start: [];
  stop: [];
  reset: [];
}>();

const showStop = computed(() => props.state === "recording");

const startDisabled = computed(
  () =>
    props.state === "connecting" ||
    props.state === "processing" ||
    props.state === "recording"
);
</script>

<template>
  <div class="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
    <button
      v-if="!showStop"
      type="button"
      :disabled="startDisabled"
      class="group relative flex-1 overflow-hidden rounded-2xl bg-gradient-to-br from-meow-500 to-meow-700
             px-8 py-5 text-base font-semibold text-white shadow-lg shadow-meow-900/40
             transition hover:from-meow-400 hover:to-meow-600
             disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
      @click="emit('start')"
    >
      <span
        class="absolute inset-0 bg-white/10 opacity-0 transition group-hover:opacity-100"
        aria-hidden="true"
      />
      <span class="relative">开始录音</span>
    </button>

    <button
      v-else
      type="button"
      class="relative flex-1 rounded-2xl border-2 border-rose-400/60 bg-rose-600/20 px-8 py-5 text-base font-semibold
             text-rose-100 shadow-[0_0_40px_-8px_rgba(244,63,94,0.5)] transition hover:bg-rose-600/30"
      @click="emit('stop')"
    >
      <span class="inline-flex items-center justify-center gap-2">
        <span
          class="h-3 w-3 rounded-full bg-rose-400 ring-4 ring-rose-500/30 animate-pulse"
          aria-hidden="true"
        />
        停止录音
      </span>
    </button>

    <button
      type="button"
      class="rounded-2xl border border-gray-600 bg-gray-900/60 px-6 py-4 text-sm font-medium text-gray-300
             transition hover:border-gray-500 hover:bg-gray-800/80"
      @click="emit('reset')"
    >
      重置
    </button>
  </div>
</template>
