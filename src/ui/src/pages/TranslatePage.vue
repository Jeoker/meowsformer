<script setup lang="ts">
import { ref } from "vue";
import { useStreamingTranslation } from "../composables/useStreamingTranslation";
import DemoHero from "../components/translate/DemoHero.vue";
import ConnectionStatus from "../components/translate/ConnectionStatus.vue";
import BreedPreference from "../components/translate/BreedPreference.vue";
import RecordingDeck from "../components/translate/RecordingDeck.vue";
import LiveFeed from "../components/translate/LiveFeed.vue";
import ErrorBanner from "../components/translate/ErrorBanner.vue";
import ResultSection from "../components/translate/ResultSection.vue";
import AdUnit from "../components/ads/AdUnit.vue";
import PageSideAds from "../components/ads/PageSideAds.vue";
import { ADSENSE_SLOT_RESULT } from "../config/adsense";

const breedPreference = ref("");

const {
  state,
  partialText,
  preview,
  result,
  error,
  connect,
  startRecording,
  stopRecording,
  reset,
} = useStreamingTranslation();

const breedLocked = () =>
  state.value === "recording" || state.value === "processing";

async function handleStart() {
  try {
    await connect(breedPreference.value || undefined);
    await startRecording();
  } catch {
    // connect() already exposes a user-facing error and leaves Start enabled
    // so the user can retry after a cold-start or transient network failure.
  }
}
</script>

<template>
  <div
    class="min-h-screen bg-app-gradient text-gray-100 selection:bg-meow-500/30 selection:text-meow-50"
  >
    <PageSideAds>
      <div
        class="mx-auto flex min-h-screen w-full max-w-lg flex-col gap-8 px-4 py-10 sm:px-6 sm:py-14"
      >
        <DemoHero />

        <div class="flex justify-center">
          <ConnectionStatus :state="state" />
        </div>

        <BreedPreference
          v-model="breedPreference"
          :disabled="breedLocked()"
        />

        <RecordingDeck
          :state="state"
          @start="handleStart"
          @stop="stopRecording"
          @reset="reset"
        />

        <LiveFeed
          :partial-text="partialText"
          :preview="preview"
        />

        <ErrorBanner
          v-if="error"
          :message="error"
        />

        <ResultSection
          v-if="result"
          :result="result"
        />

        <AdUnit
          v-if="result"
          :slot-id="ADSENSE_SLOT_RESULT"
          class="mt-4"
        />

        <footer class="pt-4 text-center text-[11px] text-gray-600 leading-relaxed">
          Matches come from the scientific corpus and multi-dimensional tag engine—never random playback.
        </footer>
      </div>
    </PageSideAds>
  </div>
</template>
