<template>
  <section class="grid gap-6 pb-14 sm:pb-20 lg:grid-cols-[1.1fr_0.9fr]">
    <div class="rounded-3xl border border-white/10 bg-gray-900/45 p-6 sm:p-8">
      <p class="text-xs font-semibold uppercase tracking-[0.22em] text-meow-400">Why this is different</p>
      <h2 class="mt-3 text-2xl font-semibold tracking-tight text-white">Built to be playful and inspectable</h2>
      <div class="mt-6 grid gap-5 sm:grid-cols-3 lg:grid-cols-1 xl:grid-cols-3">
        <div>
          <p class="font-medium text-gray-100">Real recordings</p>
          <p class="mt-1 text-sm leading-relaxed text-gray-500">Matches come from public bioacoustic corpora, not a folder of anonymous sound effects.</p>
        </div>
        <div>
          <p class="font-medium text-gray-100">Visible reasoning</p>
          <p class="mt-1 text-sm leading-relaxed text-gray-500">Emotion, intent, acoustic tags, source context, and match score remain visible.</p>
        </div>
        <div>
          <p class="font-medium text-gray-100">Reproducible sources</p>
          <p class="mt-1 text-sm leading-relaxed text-gray-500">The Science page links to the original CatMeows and Meowsic materials.</p>
        </div>
      </div>
      <RouterLink to="/about" class="mt-7 inline-flex text-sm font-semibold text-meow-300 transition hover:text-meow-200">
        Explore the science →
      </RouterLink>
    </div>

    <aside class="rounded-3xl border border-amber-900/40 bg-amber-950/15 p-6 sm:p-8">
      <p class="text-xs font-semibold uppercase tracking-[0.22em] text-amber-400">Read results responsibly</p>
      <h2 class="mt-3 text-2xl font-semibold tracking-tight text-white">A translation experiment, not a cat dictionary</h2>
      <ul class="mt-5 space-y-3 text-sm leading-relaxed text-gray-400">
        <li>Cat vocalisations do not map word-for-word to human language.</li>
        <li>Individual cats, relationships, body language, and context still matter.</li>
        <li>Results are educational and playful, never veterinary or welfare advice.</li>
      </ul>
    </aside>
  </section>
</template>
