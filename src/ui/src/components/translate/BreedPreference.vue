<script setup lang="ts">
const model = defineModel<string>({ default: "" });

defineProps<{
  disabled?: boolean;
}>();
</script>

<template>
  <div class="rounded-2xl border border-white/5 bg-gray-900/40 p-4 backdrop-blur-sm">
    <label class="block text-xs font-medium uppercase tracking-wider text-gray-500 mb-2">
      品种偏好（可选）
    </label>
    <select
      v-model="model"
      :disabled="disabled"
      class="w-full rounded-xl border border-gray-700 bg-gray-950/80 px-3 py-2.5 text-sm text-gray-200
             focus:border-meow-600 focus:outline-none focus:ring-1 focus:ring-meow-500
             disabled:cursor-not-allowed disabled:opacity-50"
    >
      <option value="">自动</option>
      <option value="Maine Coon">Maine Coon · 缅因</option>
      <option value="European Shorthair">
        European Shorthair · 欧洲短毛
      </option>
    </select>
  </div>
</template>
