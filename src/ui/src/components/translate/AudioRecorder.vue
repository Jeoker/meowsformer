<!--
  Meowsformer — AudioRecorder Component
  ========================================
  Real-time audio recording UI that streams audio chunks via WebSocket,
  displays partial transcription, analysis previews, and the final
  cat-sound result.
-->

<script setup lang="ts">
import { ref, watch } from "vue";
import {
  useStreamingTranslation,
  type StreamingState,
} from "../../composables/useStreamingTranslation";
import { useAudioPreview } from "../../composables/useAudioPreview";

const breedPreference = ref("");

const {
  state,
  partialText,
  preview,
  result,
  error,
  connect,
  startRecording,
  stopRecording,
  reset,
} = useStreamingTranslation();

const {
  state: playbackState,
  play,
  pause,
  currentTime,
  duration,
  loadBase64,
} = useAudioPreview();

async function handleStartRecording() {
  if (state.value === "idle") {
    connect(breedPreference.value || undefined);
    await new Promise((r) => setTimeout(r, 500));
  }
  await startRecording();
}

watch(
  () => result.value?.audioBase64,
  (base64) => {
    if (base64) loadBase64(base64);
  }
);

const stateLabels: Record<StreamingState, string> = {
  idle: "就绪",
  connecting: "连接中...",
  connected: "已连接",
  recording: "录音中...",
  processing: "分析中...",
  result: "完成",
  error: "错误",
};

const stateColors: Record<StreamingState, string> = {
  idle: "#888",
  connecting: "#f0ad4e",
  connected: "#5bc0de",
  recording: "#d9534f",
  processing: "#f0ad4e",
  result: "#5cb85c",
  error: "#d9534f",
};
</script>

<template>
  <div
    style="
      max-width: 600px;
      margin: 0 auto;
      padding: 24px;
      font-family: system-ui, sans-serif;
    "
  >
    <h2 style="margin-bottom: 16px">Meowsformer — 实时猫语翻译</h2>

    <!-- Status -->
    <div style="margin-bottom: 16px">
      <span
        :style="{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '6px',
          color: stateColors[state],
          fontWeight: 600,
        }"
      >
        <span
          :style="{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: stateColors[state],
            animation:
              state === 'recording' ? 'pulse 1s infinite' : 'none',
          }"
        />
        {{ stateLabels[state] }}
      </span>
    </div>

    <!-- Breed preference -->
    <div style="margin-bottom: 16px">
      <label style="display: block; margin-bottom: 4px; font-size: 14px">
        品种偏好 (可选):
      </label>
      <select
        v-model="breedPreference"
        style="
          padding: 6px 12px;
          border-radius: 4px;
          border: 1px solid #ccc;
        "
        :disabled="state === 'recording' || state === 'processing'"
      >
        <option value="">自动选择</option>
        <option value="Maine Coon">Maine Coon 缅因猫</option>
        <option value="European Shorthair">
          European Shorthair 欧洲短毛猫
        </option>
      </select>
    </div>

    <!-- Controls -->
    <div style="display: flex; gap: 8px; margin-bottom: 20px">
      <button
        v-if="state !== 'recording'"
        @click="handleStartRecording"
        :disabled="state === 'connecting' || state === 'processing'"
        style="
          padding: 10px 24px;
          border-radius: 8px;
          border: none;
          background-color: #d9534f;
          color: #fff;
          font-size: 16px;
          cursor: pointer;
        "
      >
        开始录音
      </button>
      <button
        v-else
        @click="stopRecording"
        style="
          padding: 10px 24px;
          border-radius: 8px;
          border: none;
          background-color: #5cb85c;
          color: #fff;
          font-size: 16px;
          cursor: pointer;
        "
      >
        停止录音
      </button>

      <button
        @click="reset"
        style="
          padding: 10px 24px;
          border-radius: 8px;
          border: 1px solid #ccc;
          background-color: #fff;
          font-size: 16px;
          cursor: pointer;
        "
      >
        重置
      </button>
    </div>

    <!-- Partial transcription -->
    <div
      v-if="partialText"
      style="
        padding: 12px;
        background-color: #f7f7f7;
        border-radius: 8px;
        margin-bottom: 12px;
      "
    >
      <strong>转录:</strong> {{ partialText }}
    </div>

    <!-- Analysis preview -->
    <div
      v-if="preview"
      style="
        padding: 12px;
        background-color: #eef6ff;
        border-radius: 8px;
        margin-bottom: 12px;
      "
    >
      <strong>分析预览:</strong> 情绪: {{ preview.emotion }}, 意图:
      {{ preview.intent }}
    </div>

    <!-- Error -->
    <div
      v-if="error"
      style="
        padding: 12px;
        background-color: #fff0f0;
        border-radius: 8px;
        margin-bottom: 12px;
        color: #d9534f;
      "
    >
      <strong>错误:</strong> {{ error }}
    </div>

    <!-- Result -->
    <div
      v-if="result"
      style="
        padding: 16px;
        background-color: #f0fff0;
        border-radius: 8px;
        margin-bottom: 12px;
      "
    >
      <h3 style="margin-top: 0">翻译结果</h3>

      <p>
        <strong>转录文字:</strong> {{ result.transcription }}
      </p>

      <p>
        <strong>LLM推理:</strong> {{ result.reasoning }}
      </p>

      <div
        v-if="result.selectedSample"
        style="font-size: 14px; color: #555"
      >
        <p>
          <strong>匹配样本:</strong>
          {{ result.selectedSample.sample_id }} ({{
            result.selectedSample.breed
          }}, {{ result.selectedSample.context }})
        </p>
        <p>
          <strong>匹配分数:</strong>
          {{
            (result.selectedSample.match_score * 100).toFixed(1)
          }}%
        </p>
      </div>

      <!-- Audio playback -->
      <div v-if="result.audioBase64" style="margin-top: 12px">
        <button
          @click="playbackState === 'playing' ? pause() : play()"
          style="
            padding: 8px 20px;
            border-radius: 6px;
            border: none;
            background-color: #5bc0de;
            color: #fff;
            font-size: 14px;
            cursor: pointer;
          "
        >
          {{ playbackState === "playing" ? "暂停" : "播放猫语" }}
        </button>
        <span style="margin-left: 8px; font-size: 14px; color: #888">
          {{ currentTime.toFixed(1) }}s /
          {{ duration.toFixed(1) }}s
        </span>
      </div>
    </div>
  </div>
</template>

<style>
@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.3;
  }
}
</style>
