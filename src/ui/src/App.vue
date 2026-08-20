<script setup lang="ts">
import { RouterLink, RouterView } from "vue-router";
import ConsentBanner from "./components/ads/ConsentBanner.vue";
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
        <div class="flex items-center gap-4 sm:gap-5 text-sm">
          <RouterLink
            to="/"
            class="text-gray-400 hover:text-white transition-colors"
            active-class="text-meow-400 font-medium"
          >
            Translate
          </RouterLink>
          <RouterLink
            :to="{ path: '/', hash: '#how-it-works' }"
            class="hidden text-gray-400 transition-colors hover:text-white sm:inline"
          >
            How it works
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
    <footer
      class="border-t border-white/5 px-4 py-4 text-center text-xs text-gray-600"
    >
      <RouterLink
        to="/privacy"
        class="text-gray-500 hover:text-gray-400 transition-colors"
      >
        Privacy Policy
      </RouterLink>
      <span class="mx-2 text-gray-700" aria-hidden="true">·</span>
      <a
        href="https://ko-fi.com/jeoker0880"
        target="_blank"
        rel="noopener noreferrer"
        class="text-gray-500 transition-colors hover:text-pink-300"
      >
        Support the project
      </a>
      <span class="mx-2 text-gray-700" aria-hidden="true">·</span>
      <span>© 2026 Meowsformer</span>
    </footer>
    <ConsentBanner />
  </div>
</template>
