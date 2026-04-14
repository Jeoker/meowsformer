<script setup lang="ts">
import AdUnit from "./AdUnit.vue";
import { ADSENSE_SLOT_ABOUT_MID } from "../../config/adsense";

withDefaults(
  defineProps<{
    /** Defaults to `ADSENSE_SLOT_ABOUT_MID` (`VITE_ADSENSE_SLOT_ABOUT_MID`). */
    railSlotId?: string;
  }>(),
  {
    railSlotId: ADSENSE_SLOT_ABOUT_MID,
  },
);
</script>

<template>
  <div
    class="mx-auto flex w-full max-w-[90rem] justify-center gap-6 xl:gap-8"
  >
    <aside
      class="hidden w-[160px] shrink-0 xl:block"
      aria-label="Advertisement rail"
    >
      <div class="sticky top-20 space-y-2 pt-6">
        <AdUnit
          :slot-id="railSlotId"
          format="vertical"
          :full-width-responsive="false"
        />
      </div>
    </aside>

    <div class="min-w-0 flex-1">
      <slot />
    </div>

    <aside
      class="hidden w-[160px] shrink-0 xl:block"
      aria-label="Advertisement rail"
    >
      <div class="sticky top-20 space-y-2 pt-6">
        <AdUnit
          :slot-id="railSlotId"
          format="vertical"
          :full-width-responsive="false"
        />
      </div>
    </aside>
  </div>
</template>
