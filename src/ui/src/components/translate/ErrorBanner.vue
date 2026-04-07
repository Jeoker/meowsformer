<script setup lang="ts">
defineProps<{
  message: string;
}>();
</script>

<template>
  <div
    class="rounded-2xl border border-red-900/60 bg-red-950/30 px-4 py-3 text-sm text-red-100"
    role="alert"
  >
    <span class="font-semibold text-red-300">错误 · </span>
    {{ message }}
  </div>
</template>
