<script setup lang="ts">
defineProps<{
  title?: string;
  subtitle?: string;
}>();
</script>

<template>
  <header class="text-center space-y-3 pb-2">
    <p
      class="text-xs font-medium uppercase tracking-[0.25em] text-meow-400/90"
    >
      Bioacoustic lab
    </p>
    <h1 class="text-3xl sm:text-4xl font-semibold tracking-tight text-white">
      {{ title ?? "Meowsformer" }}
    </h1>
    <p class="text-sm sm:text-base text-gray-400 max-w-md mx-auto leading-relaxed">
      {{
        subtitle ??
        "对着麦克风说话，系统会把你的语义变成一帧有依据的猫叫回应。"
      }}
    </p>
  </header>
</template>
