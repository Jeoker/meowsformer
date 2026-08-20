<template>
  <section id="how-it-works" class="scroll-mt-24 space-y-8 border-t border-white/10 py-14 sm:py-20">
    <div class="max-w-2xl space-y-3">
      <p class="text-xs font-semibold uppercase tracking-[0.24em] text-meow-400">How it works</p>
      <h2 class="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
        From human meaning to the closest cat voice
      </h2>
      <p class="leading-relaxed text-gray-400">
        The language model describes the message. A separate matching engine
        chooses the recording, so every result can be inspected.
      </p>
    </div>

    <div class="grid gap-4 md:grid-cols-3">
      <article class="rounded-3xl border border-white/10 bg-gray-900/45 p-6">
        <span class="mb-5 flex h-9 w-9 items-center justify-center rounded-full bg-meow-500/20 text-sm font-semibold text-meow-200">1</span>
        <h3 class="text-lg font-semibold text-white">Understand</h3>
        <p class="mt-2 text-sm leading-relaxed text-gray-400">
          Your short recording is transcribed and mapped to emotion,
          communicative intent, and likely acoustic character.
        </p>
      </article>
      <article class="rounded-3xl border border-white/10 bg-gray-900/45 p-6">
        <span class="mb-5 flex h-9 w-9 items-center justify-center rounded-full bg-sky-500/20 text-sm font-semibold text-sky-200">2</span>
        <h3 class="text-lg font-semibold text-white">Match</h3>
        <p class="mt-2 text-sm leading-relaxed text-gray-400">
          A deterministic scorer compares those tags with indexed recordings
          from open cat-vocalisation datasets.
        </p>
      </article>
      <article class="rounded-3xl border border-white/10 bg-gray-900/45 p-6">
        <span class="mb-5 flex h-9 w-9 items-center justify-center rounded-full bg-emerald-500/20 text-sm font-semibold text-emerald-200">3</span>
        <h3 class="text-lg font-semibold text-white">Explain</h3>
        <p class="mt-2 text-sm leading-relaxed text-gray-400">
          You hear the closest cat voice and see its tags, source context,
          breed, match score, and the reasoning behind the choice.
        </p>
      </article>
    </div>
  </section>
</template>
