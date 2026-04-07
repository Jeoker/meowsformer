# Phase 8 — Web 单页 UI 与流式前端修复（2026-04-06）

> Sprint Mode 备忘：与 [phase8-sprint-plan.md](./phase8-sprint-plan.md) Batch 1 对齐；本节记录**已实现**范围与根因级修复，便于检索。

---

## 1. 背景与目标

- **路线：** Web-First 单页（`App.vue` → `TranslatePage`），暗色 + `meow` 色板，信息分区：状态条 → 品种 → 录音 → 实时转写/预览 → 结果。
- **范围：** 仅 `src/ui/` 与为修复行为而改动的少量后端/配置（见下表）；composable **业务协议**未改（仍对接既有 WS 消息）。

---

## 2. 行为变更摘要

| 主题 | 根因 | 处理 |
|------|------|------|
| 浏览器识别胡编 | 实际采样率常为 44.1k/48k，后端按 16kHz 写 WAV | 前端按 `inputBuffer.sampleRate` **线性重采样到 16kHz** 再发 PCM |
| Whisper 语言 | 短句易被误判语种 | 可选 `.env` **`WHISPER_LANGUAGE`**（如 `zh`）传给 Whisper |
| WS 日志报错 | 收到 `websocket.disconnect` 后仍 `receive()` | `ws_endpoints` 遇 `type==websocket.disconnect` **break** |
| 结果区播放无声 | `watch(audioBase64)` 无 **immediate**，挂载时未 `loadBase64` | `ResultCard` / `MeowPreviewPlayer`：`immediate: true`；`useAudioPreview` 加固 |
| 重置后第二轮识别乱 | 非录音态点「重置」仍发 **`stop`**，服务端对空缓冲再跑一轮 | `disconnect`：**仅 `state==recording` 时**发 `stop`；抽出 `cleanupAudioCapture()` |
| 第二轮 AudioContext | 部分环境第二次上下文为 suspended | `startRecording` 内 **`await context.resume()`** |

---

## 3. 文件清单

| 区域 | 路径 | 说明 |
|------|------|------|
| 页面 | `src/ui/src/pages/TranslatePage.vue` | 单页编排；`handleStart` 在 `idle` 时 `connect` + 延迟后 `startRecording` |
| 组件 | `src/ui/src/components/translate/*.vue` | `DemoHero`, `ConnectionStatus`, `BreedPreference`, `RecordingDeck`, `LiveFeed`, `ErrorBanner`, `ResultSection`, `ResultCard` |
| 入口 | `src/ui/src/App.vue` | 仅挂载 `TranslatePage` |
| 样式 | `src/ui/src/index.css` | `.bg-app-gradient` |
| Composable | `src/ui/src/composables/useStreamingTranslation.ts` | 重采样、`cleanupAudioCapture`、`disconnect` 条件 `stop`、`resume` |
| Composable | `src/ui/src/composables/useAudioPreview.ts` | base64 去 data-URL 前缀、`playsinline`、play 前检查 |
| 后端 | `app/api/ws_endpoints.py` | 断开消息处理 |
| 配置 | `app/core/config.py` | `WHISPER_LANGUAGE` |
| 转写 | `app/services/streaming_transcription_service.py`, `transcription_service.py` | 非空时传入 `language` |

**移除：** `src/ui/src/components/translate/AudioRecorder.vue`、根目录 `src/ui/src/components/AudioRecorder.vue`（逻辑并入 `TranslatePage` + composable）。

---

## 4. 配置与验收

| 项 | 说明 |
|----|------|
| `WHISPER_LANGUAGE` | 可选；空则 Whisper 自动检测 |
| 本地联调 | `python main.py` + `cd src/ui && npm run dev` → `http://localhost:5173` |
| 测试 | 本批次**无**新增自动化测试；手动验证录音→转写→结果→播放、重置后再录 |

---

## 5. 已知边界

- 采集仍用 **ScriptProcessorNode**（已废弃 API）；后续可换 **AudioWorklet** 降低延迟与浏览器差异。
- Phase 8 **Batch 2/3**（静态文件、Docker、云端 URL）见 sprint 计划，**未**在本批次完成。
