<script setup lang="ts">
import AdUnit from "../components/ads/AdUnit.vue";
import PageSideAds from "../components/ads/PageSideAds.vue";
import { ADSENSE_SLOT_ABOUT_BOTTOM } from "../config/adsense";

const zenodoCatMeows = "https://doi.org/10.5281/zenodo.4007940";
const zenodoMeowsic = "https://doi.org/10.5281/zenodo.3245999";
const meowsicPortal =
  "https://portal.research.lu.se/en/publications/phonetic-methods-in-cat-vocalisation-studies-a-report-from-the-me/";
</script>

<template>
  <div
    class="min-h-screen bg-app-gradient text-gray-100 selection:bg-meow-500/30 selection:text-meow-50"
  >
    <PageSideAds>
      <article
        class="mx-auto max-w-3xl px-4 py-10 sm:px-6 sm:py-14 pb-24 space-y-10"
      >
        <header class="space-y-4 border-b border-white/10 pb-8">
          <p
            class="text-xs font-medium uppercase tracking-[0.25em] text-meow-400/90"
          >
            Evidence-based bioacoustics
          </p>
          <h1
            class="text-3xl sm:text-4xl font-semibold tracking-tight text-white"
          >
            The science behind our cat sound library
          </h1>
          <p class="text-base text-gray-400 leading-relaxed">
            Meowsformer does not play random clips. Every translated
            vocalisation is chosen by a deterministic matcher from a curated
            registry of real felid vocalisations and multidimensional tags
            derived from published animal behaviour and acoustics research.
          </p>
        </header>

        <section class="space-y-4">
          <h2 class="text-xl font-semibold text-white">
            CatMeows: controlled contexts and open data
          </h2>
          <p class="text-gray-300 leading-relaxed">
            The core public corpus we build on is
            <strong class="text-gray-200">CatMeows</strong>, a dataset of
            domestic cat meows recorded in three everyday situations: waiting
            for food, social grooming (brushing), and short-term isolation in an
            unfamiliar space. Recordings come from pedigreed European Shorthairs
            and Maine Coons, so breed and sex metadata travel with the
            waveforms. The dataset is released under a Creative Commons licence
            on Zenodo (<a
              :href="zenodoCatMeows"
              class="text-meow-400 underline decoration-meow-500/50 hover:text-meow-300"
              target="_blank"
              rel="noopener noreferrer"
              >DOI 10.5281/zenodo.4007940</a
            >), which makes it possible for tools like ours to ship
            reproducible, citeable behaviour instead of anonymous sound effects.
          </p>
          <p class="text-gray-300 leading-relaxed">
            Companion work on automatic context classification from the same
            research line shows that humans and models can often tell these
            situations apart from the sound alone—supporting the idea that “what
            the meow is for” is partly encoded in the vocal signal, not only in
            the cat’s body language. Meowsformer uses that intuition: recording
            context and continuous
            <strong class="text-gray-200">valence</strong> (pleasantness) and
            <strong class="text-gray-200">arousal</strong> (activation)
            coordinates anchor our tags in an affective space related to
            Russell’s circumplex model, adapted for downstream audio matching.
          </p>
        </section>

        <section class="space-y-4">
          <h2 class="text-xl font-semibold text-white">
            Meowsic (Lund): phonetics and human–cat communication
          </h2>
          <p class="text-gray-300 leading-relaxed">
            Independently, the
            <strong class="text-gray-200">Meowsic</strong> project at Lund
            University (Sweden) applies phonetic and multimodal methods to how
            cats and people vocalise during interaction—“melody in human–cat
            communication.” Their report and associated materials document
            rigorous recording and annotation practice for real-world cat sounds
            beyond a single lab paradigm (<a
              :href="meowsicPortal"
              class="text-meow-400 underline decoration-meow-500/50 hover:text-meow-300"
              target="_blank"
              rel="noopener noreferrer"
              >Lund University Research Portal</a
            >;
            <a
              :href="zenodoMeowsic"
              class="text-meow-400 underline decoration-meow-500/50 hover:text-meow-300"
              target="_blank"
              rel="noopener noreferrer"
              >Zenodo DOI 10.5281/zenodo.3245999</a
            >). We cite this line of work because it shares Meowsformer’s
            premise: cat vocalisations are structured signals worth measuring,
            not generic “animal noise.”
          </p>
        </section>

        <section class="space-y-4">
          <h2 class="text-xl font-semibold text-white">
            Five tag dimensions in Meowsformer
          </h2>
          <p class="text-gray-300 leading-relaxed">
            Our streaming pipeline asks a language model for a
            <strong class="text-gray-200">target tag set</strong>, not for a
            clip ID. A dedicated matcher then scores every indexed sample with a
            weighted similarity measure across five independent dimensions:
          </p>
          <ul
            class="list-disc pl-5 space-y-2 text-gray-300 leading-relaxed marker:text-meow-500"
          >
            <li>
              <strong class="text-gray-200">Emotion</strong> — affective state
              inferred from scenario and valence–arousal rules (e.g.
              anticipation vs distress).
            </li>
            <li>
              <strong class="text-gray-200">Intent</strong> — communicative goal
              such as solicitation, protest, or greeting-like signalling.
            </li>
            <li>
              <strong class="text-gray-200">Acoustic</strong> — measurable cues
              (pitch contour, duration, energy) derived from signal processing.
            </li>
            <li>
              <strong class="text-gray-200">Social context</strong> — situation
              tags aligned with how the underlying recordings were collected
              (feeding, isolation, grooming, etc.).
            </li>
            <li>
              <strong class="text-gray-200">Breed voice</strong> — timbre and
              breed metadata so preferences can nudge the match toward a
              plausible voice type.
            </li>
          </ul>
          <p class="text-gray-300 leading-relaxed">
            Emotion and intent currently carry the largest weights in the
            overall score; acoustic and social context are secondary; breed
            voice adds a smaller, tunable bias. The outcome is a transparent
            ranking: you can inspect tags and scores instead of trusting a black
            box that might secretly roll dice.
          </p>
        </section>

        <section class="space-y-4">
          <h2 class="text-xl font-semibold text-white">
            FAQ: how should I read a “translation”?
          </h2>
          <dl class="space-y-5">
            <div>
              <dt class="font-medium text-meow-300">
                Food bowl / “it’s dinner time” vibes
              </dt>
              <dd class="mt-1 text-gray-300 leading-relaxed">
                In CatMeows, food-related sessions tend to be high-arousal and
                positively valenced—think eager chirps and insistent meows. Our
                tags often map this to demanding or eager affect with
                solicitation intent.
              </dd>
            </div>
            <div>
              <dt class="font-medium text-meow-300">
                Alone in a new room (isolation)
              </dt>
              <dd class="mt-1 text-gray-300 leading-relaxed">
                Isolation recordings are associated with more negative valence
                and vocalisations people describe as uneasy or complaining. That
                aligns with anxious or distressed emotional tags rather than
                relaxed grooming talk.
              </dd>
            </div>
            <div>
              <dt class="font-medium text-meow-300">
                Brushing or calm handling
              </dt>
              <dd class="mt-1 text-gray-300 leading-relaxed">
                Grooming contexts span polite greetings, relaxed commentary, and
                occasional protest if the cat objects. The model distinguishes
                comfort vs annoyance using valence and arousal-shaped tag sets,
                so the matched meow should reflect that nuance when the
                transcript is clear.
              </dd>
            </div>
            <div>
              <dt class="font-medium text-meow-300">
                Is this veterinary or welfare advice?
              </dt>
              <dd class="mt-1 text-gray-300 leading-relaxed">
                No. Meowsformer is an educational demo grounded in open datasets
                and engineering choices described on this page. If your cat
                shows sudden vocal or behaviour changes, ask a qualified
                veterinarian.
              </dd>
            </div>
          </dl>
        </section>

        <div class="mt-6">
          <AdUnit :slot-id="ADSENSE_SLOT_ABOUT_BOTTOM" />
        </div>

        <footer
          class="pt-4 border-t border-white/10 text-sm text-gray-500 leading-relaxed"
        >
          Meowsformer is an independent project that indexes and matches public
          research audio; it is not endorsed by Lund University or the CatMeows
          authors. Always credit the original datasets when publishing derived
          work.
        </footer>
      </article>
    </PageSideAds>
  </div>
</template>
