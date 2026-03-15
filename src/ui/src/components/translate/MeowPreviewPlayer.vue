<!--
  Meowsformer — MeowPreviewPlayer Component
  ============================================
  Audio preview player with playback controls, confidence description
  display, and a mandatory "Confirm & Send" button.

  The user MUST be able to hear the synthesised audio before sending.
  The "Send" button is disabled until the user has listened at least once.
-->

<script setup lang="ts">
import { watch, computed } from "vue";
import { useAudioPreview } from "../../composables/useAudioPreview";
import type { PreviewDescription, SynthesisMetadata } from "../../types/api";
import "./MeowPreviewPlayer.css";

const props = withDefaults(
  defineProps<{
    audioBase64: string | null;
    previewDescription: PreviewDescription | null;
    synthesisMetadata: SynthesisMetadata | null;
    isLoading?: boolean;
  }>(),
  { isLoading: false }
);

const emit = defineEmits<{
  confirm: [];
  reject: [];
}>();

const {
  state: playbackState,
  play,
  pause,
  currentTime,
  duration,
  hasListened,
  loadBase64: audioLoadBase64,
  reset: audioReset,
} = useAudioPreview();

watch(
  () => props.audioBase64,
  (base64) => {
    if (base64) {
      audioLoadBase64(base64);
    } else {
      audioReset();
    }
  }
);

const canConfirm = computed(
  () => hasListened.value && !props.isLoading
);
const progressPct = computed(() =>
  duration.value > 0
    ? (currentTime.value / duration.value) * 100
    : 0
);
const shouldShow = computed(
  () => props.audioBase64 || props.isLoading
);
</script>

<template>
  <div
    v-if="shouldShow"
    class="meow-preview"
    role="region"
    aria-label="猫语合成预览"
  >
    <!-- Loading overlay -->
    <div v-if="isLoading" class="meow-preview__loading">
      <span>正在合成猫语...</span>
    </div>

    <!-- Description panel -->
    <div v-if="previewDescription" class="meow-preview__description">
      <p class="meow-preview__summary">
        {{ previewDescription.summary }}
      </p>

      <div class="meow-preview__confidence">
        <span class="meow-preview__badge">
          置信度: {{ previewDescription.confidence_level }} ({{
            (previewDescription.confidence_score * 100).toFixed(0)
          }}%)
        </span>
        <span class="meow-preview__badge">
          VA距离: {{ previewDescription.va_distance.toFixed(3) }}
        </span>
        <span class="meow-preview__badge">
          品种: {{ previewDescription.breed }}
        </span>
      </div>

      <details class="meow-preview__details">
        <summary>查看详细分析</summary>
        <pre class="meow-preview__detail-text">{{
          previewDescription.detail
        }}</pre>
      </details>
    </div>

    <!-- Audio player controls -->
    <div v-if="audioBase64" class="meow-preview__player">
      <button
        type="button"
        @click="playbackState === 'playing' ? pause() : play()"
        :disabled="playbackState === 'loading'"
        class="meow-preview__play-btn"
        :aria-label="
          playbackState === 'playing' ? '暂停预览' : '播放预览'
        "
      >
        {{
          playbackState === "loading"
            ? "⏳"
            : playbackState === "playing"
              ? "⏸ 暂停"
              : "▶ 播放预览"
        }}
      </button>
      <div
        class="meow-preview__progress"
        role="progressbar"
        :aria-valuenow="progressPct"
      >
        <div
          class="meow-preview__progress-fill"
          :style="{ width: `${progressPct}%` }"
        />
        <span class="meow-preview__progress-time">
          {{ currentTime.toFixed(1) }}s /
          {{ duration.toFixed(1) }}s
        </span>
      </div>
    </div>

    <!-- Metadata -->
    <div v-if="synthesisMetadata" class="meow-preview__metadata">
      <small>
        样本: {{ synthesisMetadata.matched_sample_id }} | 品种:
        {{ synthesisMetadata.matched_breed }} | 场景:
        {{ synthesisMetadata.matched_context }} | 时长:
        {{ synthesisMetadata.duration_seconds.toFixed(2) }}s | 采样率:
        {{ synthesisMetadata.sample_rate }} Hz
      </small>
    </div>

    <!-- Confirm / Reject actions -->
    <div class="meow-preview__actions">
      <button
        type="button"
        @click="emit('confirm')"
        :disabled="!canConfirm"
        class="meow-preview__confirm-btn"
        :title="
          canConfirm ? '确认并发送此猫语' : '请先播放预览音频'
        "
      >
        ✓ 确认发送
      </button>
      <button
        type="button"
        @click="emit('reject')"
        :disabled="isLoading"
        class="meow-preview__reject-btn"
      >
        ✗ 重新生成
      </button>
    </div>

    <!-- Listen-first hint -->
    <p
      v-if="!hasListened && audioBase64"
      class="meow-preview__hint"
    >
      请先播放预览音频，确认满意后再点击"确认发送"。
    </p>
  </div>
</template>
