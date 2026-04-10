<script setup lang="ts">
const googleAdsPolicies = "https://policies.google.com/technologies/ads";
</script>

<template>
  <div
    class="min-h-screen bg-app-gradient text-gray-100 selection:bg-meow-500/30 selection:text-meow-50"
  >
    <article
      class="mx-auto max-w-3xl px-4 py-10 sm:px-6 sm:py-14 pb-24 space-y-10"
    >
      <header class="space-y-4 border-b border-white/10 pb-8">
        <p
          class="text-xs font-medium uppercase tracking-[0.25em] text-meow-400/90"
        >
          Legal
        </p>
        <h1 class="text-3xl sm:text-4xl font-semibold tracking-tight text-white">
          Privacy policy
        </h1>
        <p class="text-base text-gray-400 leading-relaxed">
          How Meowsformer handles information when you use this site.
        </p>
      </header>

      <section class="space-y-4">
        <h2 class="text-xl font-semibold text-white">
          What data we collect
        </h2>
        <p class="text-gray-300 leading-relaxed">
          This website does not operate user accounts and does not ask you to
          submit a name, email address, or similar identifiers to use the demo.
          Audio you record is sent to our backend for transcription and
          matching during your session and is not stored as a permanent personal
          archive on our servers for later retrieval tied to your identity.
        </p>
      </section>

      <section class="space-y-4">
        <h2 class="text-xl font-semibold text-white">
          Cookies and advertising
        </h2>
        <p class="text-gray-300 leading-relaxed">
          We may use Google AdSense to display advertisements. Google and its
          partners may use cookies or similar technologies to serve and
          personalize ads, including based on your prior visits to this or other
          websites. You can control whether ads personalization applies via the
          consent banner shown on your first visit. For more detail on how
          Google uses data in advertising, see Google’s
          <a
            :href="googleAdsPolicies"
            class="text-meow-400 underline decoration-meow-500/50 hover:text-meow-300"
            target="_blank"
            rel="noopener noreferrer"
            >Advertising &amp; Privacy</a
          >
          resources.
        </p>
      </section>

      <section class="space-y-4">
        <h2 class="text-xl font-semibold text-white">
          Third-party services
        </h2>
        <ul
          class="list-disc pl-5 space-y-2 text-gray-300 leading-relaxed marker:text-meow-500"
        >
          <li>
            <strong class="text-gray-200">OpenAI</strong> — Whisper API for
            speech-to-text transcription of audio you send while using the
            translation feature.
          </li>
          <li>
            <strong class="text-gray-200">Google AdSense</strong> — advertising
            when you have accepted cookies for ads in the consent banner.
          </li>
        </ul>
      </section>

      <section class="space-y-4">
        <h2 class="text-xl font-semibold text-white">
          Your rights
        </h2>
        <p class="text-gray-300 leading-relaxed">
          You can withdraw consent for advertising-related storage at any time by
          clearing site data for this domain in your browser (including cookies
          and <code class="rounded bg-white/5 px-1 py-0.5 text-[0.9em]">localStorage</code>),
          which resets the consent prompt on your next visit.
        </p>
      </section>

      <section class="space-y-4">
        <h2 class="text-xl font-semibold text-white">
          Contact
        </h2>
        <p class="text-gray-300 leading-relaxed">
          Questions about this policy:
          <a
            href="mailto:privacy@meowsformer.com"
            class="text-meow-400 underline decoration-meow-500/50 hover:text-meow-300"
          >privacy@meowsformer.com</a>
        </p>
      </section>

      <footer class="pt-4 border-t border-white/10 text-xs text-gray-600">
        Last updated: April 2026
      </footer>
    </article>
  </div>
</template>
