<script setup lang="ts">
import { watch, nextTick, onMounted } from "vue";
import { useConsent } from "../../composables/useConsent";
import { ADSENSE_CLIENT_ID } from "../../config/adsense";

withDefaults(
  defineProps<{
    slotId: string;
    format?: string;
    layout?: string;
  }>(),
  {
    format: "auto",
  },
);

const { consent, adsenseReady } = useConsent();

let pushed = false;

function pushOnce() {
  if (pushed) return;
  pushed = true;
  try {
    (window.adsbygoogle = window.adsbygoogle || []).push({});
  } catch {
    pushed = false;
  }
}

watch(
  [consent, adsenseReady],
  async () => {
    if (consent.value !== "accepted" || !adsenseReady.value) return;
    await nextTick();
    pushOnce();
  },
  { immediate: true },
);

onMounted(() => {
  void nextTick(() => {
    if (consent.value !== "accepted" || !adsenseReady.value) return;
    pushOnce();
  });
});
</script>

<template>
  <div
    v-if="consent === 'accepted'"
    class="w-full"
  >
    <p class="mb-1 text-center text-[10px] uppercase tracking-wide text-gray-500">
      Advertisement
    </p>
    <div
      class="flex min-h-[120px] w-full items-start justify-center overflow-hidden"
    >
      <ins
        class="adsbygoogle"
        style="display: block"
        :data-ad-client="ADSENSE_CLIENT_ID"
        :data-ad-slot="slotId"
        :data-ad-format="format"
        data-full-width-responsive="true"
        v-bind="layout ? { 'data-ad-layout': layout } : {}"
      />
    </div>
  </div>
</template>
