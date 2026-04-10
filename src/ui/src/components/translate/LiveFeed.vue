<script setup lang="ts">
defineProps<{
  partialText: string;
  preview: { emotion: string; intent: string } | null;
}>();
</script>

<template>
  <div class="space-y-3">
    <div
      v-if="partialText"
      class="rounded-2xl border border-white/5 bg-gray-900/50 p-4 backdrop-blur-sm"
    >
      <p class="text-[10px] font-semibold uppercase tracking-widest text-gray-500 mb-2">
        Live transcript
      </p>
      <p class="text-gray-100 text-base leading-relaxed font-light">
        {{ partialText }}
      </p>
    </div>

    <div
      v-if="preview"
      class="rounded-2xl border-l-4 border-meow-500/80 bg-meow-950/25 p-4 backdrop-blur-sm"
    >
      <p class="text-[10px] font-semibold uppercase tracking-widest text-meow-400/90 mb-2">
        Analysis preview
      </p>
      <div class="flex flex-wrap gap-2 text-sm">
        <span
          class="rounded-lg bg-purple-950/60 px-2.5 py-1 text-purple-200 border border-purple-800/50"
        >
          Emotion {{ preview.emotion }}
        </span>
        <span
          class="rounded-lg bg-blue-950/60 px-2.5 py-1 text-blue-200 border border-blue-800/50"
        >
          Intent {{ preview.intent }}
        </span>
      </div>
    </div>
  </div>
</template>
