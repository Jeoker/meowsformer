<script setup lang="ts">
const model = defineModel<string>({ default: "" });

defineProps<{
  disabled?: boolean;
}>();
</script>

<template>
  <details class="group rounded-2xl border border-white/5 bg-gray-950/35 px-4 py-3">
    <summary class="flex cursor-pointer list-none items-center justify-between gap-4 text-sm text-gray-400">
      <span>Voice options <span class="text-gray-600">· {{ model || "Auto match" }}</span></span>
      <span class="text-xs text-gray-600 transition group-open:rotate-180" aria-hidden="true">⌄</span>
    </summary>
    <div class="pt-4">
      <label class="mb-2 block text-xs font-medium uppercase tracking-wider text-gray-500">
        Breed preference (optional)
      </label>
      <select
        v-model="model"
        :disabled="disabled"
        class="w-full rounded-xl border border-gray-700 bg-gray-950/80 px-3 py-2.5 text-sm text-gray-200
               focus:border-meow-600 focus:outline-none focus:ring-1 focus:ring-meow-500
               disabled:cursor-not-allowed disabled:opacity-50"
      >
        <option value="">Auto</option>
        <option value="Maine Coon">Maine Coon</option>
        <option value="European Shorthair">European Shorthair</option>
      </select>
      <p class="mt-2 text-xs leading-relaxed text-gray-600">A preference nudges the match; meaning and intent still come first.</p>
    </div>
  </details>
</template>
