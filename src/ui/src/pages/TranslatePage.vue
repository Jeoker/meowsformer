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
import HowItWorks from "../components/translate/HowItWorks.vue";
import TrustAndLimits from "../components/translate/TrustAndLimits.vue";
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
      <main class="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 sm:py-10 lg:px-8">
        <DemoHero />

        <section id="translator" class="scroll-mt-24 py-14 sm:py-20">
          <div class="mx-auto max-w-2xl space-y-6">
            <header class="space-y-3 text-center">
              <p class="text-xs font-semibold uppercase tracking-[0.24em] text-meow-400">
                Your translation lab
              </p>
              <h2 class="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                What would you like to say?
              </h2>
              <p class="mx-auto max-w-xl text-sm leading-relaxed text-gray-400 sm:text-base">
                Speak naturally for 5–10 seconds. A clear, complete thought gives
                the matcher more emotion and intent to work with.
              </p>
            </header>

            <div class="rounded-[2rem] border border-white/10 bg-gray-900/60 p-5 shadow-2xl shadow-black/20 backdrop-blur-xl sm:p-7">
              <div v-if="state !== 'idle'" class="mb-5 flex justify-center">
                <ConnectionStatus :state="state" />
              </div>

              <div class="space-y-5">
                <RecordingDeck
                  :state="state"
                  @start="handleStart"
                  @stop="stopRecording"
                  @reset="reset"
                />

                <BreedPreference
                  v-model="breedPreference"
                  :disabled="breedLocked()"
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
              </div>
            </div>

            <div class="flex flex-wrap justify-center gap-x-5 gap-y-2 text-xs text-gray-600">
              <span>Try “Dinner time”</span>
              <span>“I missed you”</span>
              <span>“Please stop scratching that”</span>
            </div>

            <AdUnit
              v-if="result"
              :slot-id="ADSENSE_SLOT_RESULT"
              class="mt-4"
            />
          </div>
        </section>

        <HowItWorks />
        <TrustAndLimits />

        <section class="mb-10 rounded-[2rem] border border-meow-800/40 bg-gradient-to-r from-meow-950/50 to-gray-900/70 px-6 py-10 text-center sm:px-10">
          <h2 class="text-2xl font-semibold tracking-tight text-white sm:text-3xl">Ready to say it in cat?</h2>
          <p class="mx-auto mt-3 max-w-xl text-sm leading-relaxed text-gray-400">Record one clear thought and hear the closest voice in the research corpus.</p>
          <a href="#translator" class="mt-6 inline-flex rounded-2xl bg-meow-600 px-6 py-3 text-sm font-semibold text-white transition hover:bg-meow-500">Start a translation</a>
        </section>
      </main>
    </PageSideAds>
  </div>
</template>
