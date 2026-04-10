<script setup lang="ts">
import type { StreamingState } from "../../composables/useStreamingTranslation";

const props = defineProps<{
  state: StreamingState;
}>();

const labels: Record<StreamingState, string> = {
  idle: "Ready",
  connecting: "Connecting",
  connected: "Connected",
  recording: "Listening",
  processing: "Matching",
  result: "Done",
  error: "Error",
};

const styles: Record<StreamingState, string> = {
  idle: "bg-gray-800/80 text-gray-300 border-gray-700",
  connecting: "bg-amber-950/50 text-amber-200 border-amber-800/60",
  connected: "bg-sky-950/40 text-sky-200 border-sky-800/50",
  recording: "bg-rose-950/40 text-rose-200 border-rose-700/60",
  processing: "bg-amber-950/50 text-amber-200 border-amber-800/60",
  result: "bg-emerald-950/35 text-emerald-200 border-emerald-800/50",
  error: "bg-red-950/40 text-red-200 border-red-800/50",
};

const dotStyles: Record<StreamingState, string> = {
  idle: "bg-gray-500",
  connecting: "bg-amber-400 animate-pulse",
  connected: "bg-sky-400",
  recording: "bg-rose-400 animate-ping",
  processing: "bg-amber-400 animate-pulse",
  result: "bg-emerald-400",
  error: "bg-red-400",
};
</script>

<template>
  <div
    class="inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium tabular-nums"
    :class="styles[props.state]"
    role="status"
    :aria-live="state === 'recording' ? 'polite' : 'off'"
  >
    <span
      class="relative flex h-2 w-2 shrink-0"
      aria-hidden="true"
    >
      <span
        v-if="state === 'recording'"
        class="absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"
      />
      <span
        class="relative inline-flex h-2 w-2 rounded-full"
        :class="dotStyles[props.state]"
      />
    </span>
    {{ labels[props.state] }}
  </div>
</template>
