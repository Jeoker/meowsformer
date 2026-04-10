<script setup lang="ts">
import { useConsent } from "../../composables/useConsent";

const { consent, accept, decline } = useConsent();
</script>

<template>
  <div
    v-if="consent === 'pending'"
    class="fixed bottom-0 left-0 right-0 z-[100] border-t border-white/10 bg-gray-950/95 px-4 py-4 shadow-[0_-8px_32px_rgba(0,0,0,0.45)] backdrop-blur-md"
    role="dialog"
    aria-label="Cookie and advertising consent"
  >
    <div
      class="mx-auto flex max-w-3xl flex-col gap-4 sm:flex-row sm:items-center sm:justify-between sm:gap-6"
    >
      <p class="text-sm text-gray-300 leading-relaxed">
        We use cookies and Google AdSense to show ads. By continuing, you agree
        to how Google may process data for advertising.
      </p>
      <div class="flex shrink-0 flex-wrap items-center gap-2 sm:justify-end">
        <button
          type="button"
          class="rounded-lg bg-meow-600 px-4 py-2 text-sm font-medium text-white hover:bg-meow-500 transition-colors"
          @click="accept"
        >
          Accept
        </button>
        <button
          type="button"
          class="rounded-lg border border-white/15 bg-transparent px-4 py-2 text-sm text-gray-400 hover:bg-white/5 hover:text-gray-200 transition-colors"
          @click="decline"
        >
          Decline
        </button>
      </div>
    </div>
  </div>
</template>
