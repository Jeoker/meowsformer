<script setup lang="ts">
import ResultCard from "./ResultCard.vue";
import type { StreamingResult } from "../../composables/useStreamingTranslation";

defineProps<{
  result: StreamingResult;
}>();
</script>

<template>
  <section class="space-y-3">
    <h2 class="text-[10px] font-semibold uppercase tracking-widest text-gray-500">
      This round
    </h2>
    <ResultCard
      :transcription="result.transcription"
      :sample="result.selectedSample"
      :audio-base64="result.audioBase64"
      :reasoning="result.reasoning"
    />
  </section>
</template>
