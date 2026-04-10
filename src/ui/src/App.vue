<script setup lang="ts">
import { RouterLink, RouterView } from "vue-router";
</script>

<template>
  <div>
    <nav
      class="sticky top-0 z-50 border-b border-white/10 bg-gray-950/85 backdrop-blur-md"
      aria-label="Main"
    >
      <div
        class="mx-auto flex max-w-3xl items-center justify-between gap-4 px-4 py-3 sm:px-6"
      >
        <RouterLink
          to="/"
          class="text-sm font-semibold tracking-tight text-white hover:text-meow-300 transition-colors"
        >
          Meowsformer
        </RouterLink>
        <div class="flex items-center gap-5 text-sm">
          <RouterLink
            to="/"
            class="text-gray-400 hover:text-white transition-colors"
            active-class="text-meow-400 font-medium"
          >
            Home
          </RouterLink>
          <RouterLink
            to="/about"
            class="text-gray-400 hover:text-white transition-colors"
            active-class="text-meow-400 font-medium"
          >
            Science
          </RouterLink>
        </div>
      </div>
    </nav>
    <RouterView />
  </div>
</template>
