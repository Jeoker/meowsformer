<script setup lang="ts">
import { ref } from "vue";

const exampleAudio = ref<HTMLAudioElement | null>(null);
const examplePlaying = ref(false);

async function toggleExample() {
  const audio = exampleAudio.value;
  if (!audio) return;

  if (audio.paused) {
    await audio.play();
  } else {
    audio.pause();
  }
}
</script>

<template>
  <section
    class="grid items-center gap-10 border-b border-white/10 pb-14 pt-4 lg:grid-cols-[1.08fr_0.92fr] lg:gap-14 lg:pb-20 lg:pt-10"
  >
    <div class="space-y-7">
      <div class="space-y-4">
        <p class="text-xs font-semibold uppercase tracking-[0.28em] text-meow-400">
          A playful bioacoustic experiment
        </p>
        <h1
          class="max-w-3xl text-5xl font-semibold leading-[0.98] tracking-[-0.04em] text-white sm:text-6xl lg:text-7xl"
        >
          Say it in cat.
        </h1>
        <p class="max-w-2xl text-base leading-relaxed text-gray-300 sm:text-lg">
          Record a short message. Meowsformer identifies its emotion and
          intent, matches it with a real cat vocalisation from open research
          datasets, and shows you why it chose that voice.
        </p>
      </div>

      <div class="flex flex-col gap-3 sm:flex-row">
        <a
          href="#translator"
          class="inline-flex items-center justify-center rounded-2xl bg-gradient-to-r from-meow-500 to-meow-700 px-6 py-3.5 text-sm font-semibold text-white shadow-lg shadow-meow-950/40 transition hover:from-meow-400 hover:to-meow-600"
        >
          Try a 5-second message
        </a>
        <button
          type="button"
          class="inline-flex items-center justify-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-6 py-3.5 text-sm font-semibold text-gray-100 transition hover:border-meow-600/60 hover:bg-white/10"
          @click="toggleExample"
        >
          <span aria-hidden="true">{{ examplePlaying ? "Pause" : "▶" }}</span>
          {{ examplePlaying ? "Pause example" : "Hear an example" }}
        </button>
      </div>

      <div class="flex flex-wrap gap-x-5 gap-y-2 text-xs text-gray-500">
        <span>Real research recordings</span>
        <span>Transparent matching</span>
        <span>No random sound effects</span>
      </div>
    </div>

    <div
      class="relative overflow-hidden rounded-[2rem] border border-white/10 bg-gray-900/70 p-5 shadow-2xl shadow-black/30 backdrop-blur-xl sm:p-7"
    >
      <div
        class="pointer-events-none absolute -right-16 -top-20 h-48 w-48 rounded-full bg-meow-500/15 blur-3xl"
        aria-hidden="true"
      />

      <div class="relative space-y-5">
        <div class="flex items-center justify-between gap-4">
          <p class="text-[10px] font-semibold uppercase tracking-[0.22em] text-meow-400">
            Example output
          </p>
          <span class="rounded-full border border-emerald-800/60 bg-emerald-950/40 px-2.5 py-1 text-[10px] font-medium text-emerald-300">
            Explainable match
          </span>
        </div>

        <div>
          <p class="mb-1.5 text-xs uppercase tracking-wider text-gray-500">You say</p>
          <p class="text-2xl font-medium tracking-tight text-white">“Dinner time!”</p>
        </div>

        <div class="rounded-2xl border border-meow-800/40 bg-meow-950/25 p-4">
          <p class="mb-1 text-xs font-medium text-meow-400">Closest cat voice</p>
          <p class="text-base leading-relaxed text-gray-100">
            An eager, attention-seeking meow with a bright, rising contour.
          </p>
        </div>

        <div class="flex flex-wrap gap-2">
          <span class="rounded-lg border border-purple-800/60 bg-purple-950/55 px-2.5 py-1 text-xs text-purple-200">Eager</span>
          <span class="rounded-lg border border-blue-800/60 bg-blue-950/55 px-2.5 py-1 text-xs text-blue-200">Requesting</span>
          <span class="rounded-lg border border-emerald-800/60 bg-emerald-950/50 px-2.5 py-1 text-xs text-emerald-200">Rising pitch</span>
        </div>

        <button
          type="button"
          class="flex w-full items-center gap-4 rounded-2xl border border-meow-700/50 bg-gradient-to-r from-meow-900/50 to-meow-800/25 px-4 py-4 text-left transition hover:border-meow-500/70"
          @click="toggleExample"
        >
          <span class="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-meow-500/25 text-lg text-meow-100" aria-hidden="true">
            {{ examplePlaying ? "Ⅱ" : "▶" }}
          </span>
          <span>
            <span class="block text-sm font-semibold text-meow-100">
              {{ examplePlaying ? "Pause example" : "Play example meow" }}
            </span>
            <span class="mt-0.5 block text-xs text-gray-500">A sample output from the matching pipeline</span>
          </span>
        </button>

        <audio
          ref="exampleAudio"
          src="/examples/example-meow.wav"
          preload="metadata"
          @play="examplePlaying = true"
          @pause="examplePlaying = false"
          @ended="examplePlaying = false"
        />
      </div>
    </div>
  </section>
</template>
