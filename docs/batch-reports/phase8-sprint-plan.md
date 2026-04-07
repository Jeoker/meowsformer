# Phase 8 — Web Demo Sprint 计划

> **决策日期：** 2026-03-28
> **Sprint Mode：** 本阶段使用轻量工作流，跳过 reviewer/test-engineer 循环。详见 `.cursor/rules/dev-workflow.mdc` 的 Sprint Mode 章节。

---

## 1. 技术路线转向：为什么做这个变更

### 1.1 问题诊断

Phase 0-5 建立了高质量的后端：Whisper 流式转录、GPT-4o 结构化标签生成、5 维加权 Jaccard 匹配引擎、推测性执行、483 个科学语料库样本索引、330 个测试函数。这些都是 Done。

但整个系统**没有一个可以分享的 URL**。

Phase 7 的 Flet 客户端经历了 6 个 Batch（每个走完 developer → reviewer → test-engineer → reviewer → 文档更新的完整循环），产出是一个只能在 WSL2 桌面模式下本地运行的客户端。与此同时，`src/ui/` 中的 Vue 前端在 Phase 5 期间就已编写了完整的 WebSocket composable、音频播放 composable、录音组件和结果展示组件——全部闲置。

### 1.2 工作流瓶颈

标准工作流的每个 Batch 需要 4-8 次 subagent 调用（developer → reviewer 多轮 → test-engineer → reviewer 多轮）。Phase 7 的 6 个 Batch ≈ 30-50 次 subagent 来回。这套流程模拟的是 4 人企业团队的协作规范，对 UI/前端工作的 ceremony-to-code 比率过高。

### 1.3 决策

| 维度 | 之前 | 之后 |
|------|------|------|
| 首要客户端 | Flet 移动端（本地运行） | Vue Web（可部署、可分享 URL） |
| 部署策略 | Phase 4 暂缓 (Cloud Run) | Phase 8 包含一体化部署 (Railway/Render) |
| 开发工作流 | 标准流程（所有代码都走审核循环） | Sprint Mode（UI/部署跳过审核，后端仍用标准流程） |
| 文档更新 | 每 Batch 更新 4 份文档 + batch report | Sprint 结束后统一更新 |

### 1.4 不变的部分

- 后端代码（Phase 0-5）：**零修改**
- 科学匹配引擎、标签体系、DSP 管线：**不动**
- Flet 客户端：**保留**在 `src/flet_mobile/`，不删除
- 330 个测试：**继续有效**，全部可运行

---

## 2. 已有 Vue 资产盘点

以下文件是 Phase 8 的起点。**developer 应使用而非重写这些文件**（除非特别标注需要修改）：

### 2.1 完整可用、无需修改

| 文件 | 职责 |
|------|------|
| `src/ui/src/composables/useStreamingTranslation.ts` | WebSocket 连接管理 + 状态机 (7 states)；PCM 16kHz 音频采集 (ScriptProcessorNode, buffer=4096 ≈ 256ms/帧)；Float32→Int16 转换；4 种 Server 消息 (transcription/analysis_preview/result/error) 解析分发；connect/startRecording/stopRecording/disconnect/reset 控制接口；Vue 3 Composition API (ref/shallowRef/onUnmounted) |
| `src/ui/src/composables/useAudioPreview.ts` | base64 WAV → Blob → ObjectURL → HTMLAudioElement；play/pause/stop/loadBase64/reset 控制；播放状态追踪 (idle/loading/playing/paused/ended)；hasListened 标记；unmount 自动清理 ObjectURL；Vue 3 Composition API |
| `src/ui/src/types/api.ts` | TypeScript 类型完全镜像后端 Pydantic 模型：Phase 0-3 (CatTranslationResponse, MeowSynthesisResponse) + Phase 5 (TargetTagSet, TaggedSampleInfo, StreamingTranslationResult) + WebSocket (WSTranscriptionMessage, WSAnalysisPreviewMessage, WSResultMessage, WSErrorMessage, WSServerMessage) |
| `src/ui/src/components/translate/ResultCard.vue` | Tailwind 暗色主题结果卡片；emotion (purple)、intent (blue)、acoustic (green) 标签徽章；匹配分数百分比展示；内嵌 useAudioPreview 播放控制 |
| `src/ui/vite.config.ts` | Vite 开发服务器 :5173；proxy `/api` → `http://localhost:8000`，`/ws` → `ws://localhost:8000` |
| `src/ui/tailwind.config.js` | 自定义 meow 色板 (meow-50 `#fff9f0` ~ meow-900 `#7c3f16`)；content 扫描 `./src/**/*.{js,ts,vue}` |
| `src/ui/src/index.css` | Tailwind base/components/utilities；body 暗色主题 `bg-gray-950 text-gray-100` |
| `src/ui/package.json` | Vue 3 + Vite 5 + Tailwind 3.4 + TypeScript 5.3 |

### 2.2 需要修改

| 文件 | 当前状态 | 需要做什么 |
|------|---------|-----------|
| `src/ui/src/App.vue` (占位) | 占位页面，仅渲染 `<h1>Meowformer</h1>` | 替换为 `<TranslatePage />`；单页应用无需路由 |
| `src/ui/src/components/translate/AudioRecorder.vue` (功能完整) | **功能完整**（已接入两个 composable，品种选择、录音、转录、预览、结果播放全有），但使用 inline style | 用 Tailwind 重写 UI 样式（功能逻辑和 composable 接线保持不变） |

### 2.3 关键代码片段

`AudioRecorder.vue` 中已完成的 composable 接线（无需改动逻辑，只需重写样式）：

```vue
<script setup>
// 这些接线已经完成，developer 不需要重新实现
const {
  state, partialText, preview, result, error,
  connect, startRecording, stopRecording, reset,
} = useStreamingTranslation();

const {
  state: playbackState, play, pause,
  currentTime, duration, loadBase64,
} = useAudioPreview();

// 结果到达时自动加载音频
watch(() => result.value?.audioBase64, (base64) => {
  if (base64) loadBase64(base64);
});
</script>
```

`useStreamingTranslation` 的 `getWsUrl()` 基于 `window.location` 动态构建 WebSocket URL：

```ts
function getWsUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws/translate`;
}
```

这意味着一体化部署后（FastAPI serve 前端静态文件 + WebSocket），无需任何 URL 配置，自动就能连上。

---

## 3. Batch 计划

### Batch 1：前端激活 + UI 打磨

**目标：** `npm run dev` 打开浏览器能看到完整的录音→转录→标签预览→猫叫播放流程。

**改动范围：** 仅 `src/ui/src/` 内的前端文件。后端零改动。

**具体任务：**

1. **`App.vue`** — 替换占位内容为实际的翻译页面
   - 单页应用，无需路由
   - 直接渲染 TranslatePage 组件

2. **`AudioRecorder.vue`** — 用 Tailwind 重写 UI
   - 保持全部功能逻辑和 composable 接线不变
   - inline style → Tailwind classes
   - 设计方向：暗色主题（与 index.css 一致）、meow 色板（Tailwind config 已定义）、移动端响应式
   - UI 层次：hero 区域（项目名 + 一句话介绍）→ 中央录音按钮（大、醒目）→ 实时反馈区（转录文本 + 标签预览）→ 结果区（复用 ResultCard 或内嵌展示）

3. **可选：新建 `TranslatePage.vue`** — 如果 AudioRecorder 职责过重，可拆出一个 page 级组件做布局编排

**验收标准：**
- `cd src/ui && npm install && npm run dev` 启动 Vite 开发服务器
- 同时 `python main.py` 启动 FastAPI 后端
- 浏览器打开 `http://localhost:5173`
- 能录音、看到实时转录、看到标签预览、听到猫叫声

**落地备忘（2026-04-06）：** 以 `TranslatePage` + 子组件替代原 `AudioRecorder` 单文件方案；并含流式采样率、WS 断开、播放与重置等根因修复 — 详见 [phase8-batch-ui-2026-04-06.md](./phase8-batch-ui-2026-04-06.md)（实际改动含少量后端/配置，与上表「仅前端」不完全一致，以备忘为准）。

### Batch 2：静态文件服务 + 部署配置

**目标：** `docker build && docker run` 启动一个自包含的服务，在单个端口上同时提供 API + WebSocket + 前端页面。

**改动范围：** `main.py`（加 StaticFiles）、新增 `Dockerfile`、新增 `.dockerignore`。

**部署架构：**

```
Browser ──────────────────► Railway/Render (单进程)
                               │
                            FastAPI + Uvicorn
                               │
                    ┌──────────┼──────────────┐
                    │          │              │
              /api/*      /ws/translate    /* (SPA)
           REST 端点     WebSocket 流式    Vue 静态文件
                                          (index.html fallback)
```

**具体任务：**

1. **`main.py`** — 加入静态文件服务
   ```python
   from fastapi.staticfiles import StaticFiles
   from starlette.responses import FileResponse
   import os

   STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

   # 所有 API/WS 路由注册完毕后：
   if os.path.isdir(STATIC_DIR):
       app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="static-assets")

       @app.get("/{full_path:path}")
       async def serve_spa(full_path: str):
           file_path = os.path.join(STATIC_DIR, full_path)
           if os.path.isfile(file_path):
               return FileResponse(file_path)
           return FileResponse(os.path.join(STATIC_DIR, "index.html"))
   ```
   关键：API 和 WS 路由必须在 SPA fallback 之前注册，否则会被拦截。

2. **`Dockerfile`** — 双阶段构建
   - 第一阶段 (Node)：`cd src/ui && npm ci && npm run build`，产出 `dist/`
   - 第二阶段 (Python)：安装 Python 依赖、复制后端代码、复制前端 `dist/` → `static/`、复制音频数据
   - 入口：`uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **`.dockerignore`** — 排除 venv、node_modules、.git、__pycache__、.env

4. **可选：`requirements-deploy.txt`** — 精简版依赖（移除 flet、flet-audio、sounddevice 等仅本地开发需要的包）

**音频数据处理：**
- 483 个 WAV 文件（`assets/raw_data/catmeows/`）和 `assets/audio_db/tagged_samples.json` 必须在运行时可访问
- MVP 方案：Dockerfile 中 `COPY assets/ ./assets/` 直接打包进镜像
- 如果镜像过大（>1GB），备选：Dockerfile 的 build stage 运行 `python -m tools.download_datasets` 下载数据；或使用 cloud storage

**验收标准：**
- `docker build -t meowsformer . && docker run -p 8000:8000 -e OPENAI_API_KEY=sk-xxx meowsformer`
- 浏览器打开 `http://localhost:8000` 看到前端页面
- 录音→转录→猫叫播放完整流程可用
- `http://localhost:8000/docs` FastAPI 文档仍可访问

### Batch 3：云端部署

**目标：** 生成一个公开 URL，任何人可打开使用。

**推荐平台：Railway**（备选：Render、Fly.io）
- 原生 Docker 支持
- WebSocket 支持开箱即用
- 免费额度（每月 $5 credit，hobby plan $5/月）
- 环境变量通过 Dashboard 设置

**具体任务：**

1. Railway 项目初始化（`railway init` 或 Dashboard 操作）
2. 设置环境变量：`OPENAI_API_KEY`、`CHROMA_DB_PATH`、`PORT`
3. 部署：`railway up` 或 git push 触发
4. 验证公开 URL 端到端可用

**验收标准：**
- 一个 `https://meowsformer-xxx.up.railway.app` 格式的公开 URL
- 全流程可用：录音 → 实时转录 → 标签预览 → 猫叫播放
- WebSocket 连接稳定（wss:// 自动适配）

---

## 4. Developer Subagent 沟通模板

以下是 PM 调用 developer subagent 时应包含的 prompt 模板。每次调用都应包含"通用上下文"加上对应 Batch 的"具体任务"。

### 4.1 通用上下文（每次 developer 调用都包含）

```
## 通用上下文

战略背景：我们正在从 Flet-First 转向 Web-First。Vue 前端在 src/ui/ 中
已有完整的 composable 和组件（useStreamingTranslation、useAudioPreview、
AudioRecorder、ResultCard），在 Phase 5 期间编写但从未接入 App.vue。
Phase 8 的目标是激活这些组件、打磨 UI、部署到云端，产出可分享的公开 URL。

工作流：Sprint Mode — 无 reviewer/test-engineer 循环。你完成后直接交给 PM
手动验证。

关键约束：
- 后端代码（app/、main.py）在 Batch 1 中不修改
- 不要修改 useStreamingTranslation.ts 和 useAudioPreview.ts 的逻辑
- 不要修改 types/api.ts
- 不要添加认证、登录、注册等功能
- 不要添加 vue-router 多页面路由
```

### 4.2 Batch 1 Prompt

```
## 任务：Phase 8 Batch 1 — 前端激活 + UI 打磨

[插入通用上下文]

## 具体任务

1. 修改 src/ui/src/App.vue：
   - 单页应用，直接渲染翻译页面组件

2. 重写 src/ui/src/components/translate/AudioRecorder.vue 的 UI：
   - 功能逻辑和 composable 接线（useStreamingTranslation + useAudioPreview）全部保留
   - 将所有 inline style 替换为 Tailwind CSS classes
   - 设计方向：暗色主题（bg-gray-950 已设定）、使用 meow 色板（tailwind.config.js
     已定义 meow-50 到 meow-900）
   - 布局：顶部 hero 区（项目名 + 一句话介绍）→ 中央录音按钮（大、醒目、
     录音中有脉动动画）→ 实时反馈区（转录文本 + 情绪/意图标签预览）→
     结果区（可复用 ResultCard 组件）
   - 移动端响应式

3. 可选：如果觉得 AudioRecorder 组件职责过重，可以拆出一个 TranslatePage.vue
   做布局编排

## 已有资产（直接使用，不要重写）

- src/ui/src/composables/useStreamingTranslation.ts — WebSocket 状态机 + PCM 采集
- src/ui/src/composables/useAudioPreview.ts — base64 音频播放
- src/ui/src/types/api.ts — TypeScript 类型定义
- src/ui/src/components/translate/ResultCard.vue — Tailwind 结果卡片
- src/ui/tailwind.config.js — meow 色板已定义
- src/ui/src/index.css — 暗色主题基础

## 验收标准

- cd src/ui && npm install && npm run dev（Vite :5173）
- 同时 python main.py（FastAPI :8000）
- 浏览器 http://localhost:5173 能完成：录音 → 实时转录 → 标签预览 → 猫叫播放
```

### 4.3 Batch 2 Prompt

```
## 任务：Phase 8 Batch 2 — 静态文件服务 + Dockerfile

[插入通用上下文]

## 具体任务

1. 修改 main.py：在所有 API/WS 路由注册之后，添加 Vue 静态文件服务
   - 静态文件目录：项目根目录下的 static/（由 Docker 构建时从 src/ui/dist/ 复制）
   - /assets/* 走 StaticFiles
   - 其余所有非 API 路径 fallback 到 static/index.html（SPA 路由）
   - 仅当 static/ 目录存在时才挂载（本地开发时可以不存在，仍用 Vite 代理）

2. 新建 Dockerfile（项目根目录）：
   - Stage 1 (node:20-slim)：cd src/ui && npm ci && npm run build → 产出 dist/
   - Stage 2 (python:3.12-slim)：
     - 安装系统依赖（ffmpeg）
     - pip install -r requirements.txt（或 requirements-deploy.txt 精简版）
     - COPY 后端代码（app/、src/engine/、tools/、main.py）
     - COPY --from=stage1 dist/ → static/
     - COPY assets/ ./assets/（音频数据和标签索引）
     - EXPOSE $PORT
     - CMD uvicorn main:app --host 0.0.0.0 --port $PORT

3. 新建 .dockerignore：排除 venv/、node_modules/、.git/、__pycache__/、
   .env、*.pyc、db/

4. 可选：新建 requirements-deploy.txt，移除仅本地开发需要的包
   （flet、flet-audio、sounddevice）

## 验收标准

- docker build -t meowsformer .
- docker run -p 8000:8000 -e OPENAI_API_KEY=sk-xxx meowsformer
- http://localhost:8000 → Vue 前端
- http://localhost:8000/docs → FastAPI 文档
- 录音→转录→猫叫播放完整流程可用
```

### 4.4 Batch 3 Prompt

```
## 任务：Phase 8 Batch 3 — 云端部署

[插入通用上下文]

## 具体任务

1. 选择部署平台（推荐 Railway，备选 Render/Fly.io）
2. 配置部署：
   - 项目初始化
   - 环境变量设置：OPENAI_API_KEY、CHROMA_DB_PATH=./db/chroma_db、PORT
   - Docker 部署（使用 Batch 2 的 Dockerfile）
3. 验证公开 URL 端到端可用

## 验收标准

- 一个 https:// 开头的公开 URL
- 全流程可用：录音 → 实时转录 → 标签预览 → 猫叫播放
- WebSocket (wss://) 连接稳定
```

---

## 5. 风险与降级方案

| 风险 | 影响 | 降级方案 |
|------|------|---------|
| Docker 镜像过大（音频数据 > 500MB） | 部署平台可能拒绝或速度极慢 | 容器启动时下载数据（`tools/download_datasets.py` + `tools/build_tags.py`）而非打包进镜像 |
| Railway 免费额度不足 | 服务被暂停 | 切换到 Render 免费 tier 或 Fly.io |
| 浏览器麦克风权限被拒绝 | 无法录音 | 前端展示友好错误提示（AudioRecorder 已有 error 状态处理） |
| Whisper/GPT-4o API 延迟高 | 用户体验差 | 推测性执行已内建（Phase 5）；前端展示 loading 状态 |
| HTTPS 要求（麦克风 API 需要 secure context） | 部署后麦克风不可用 | Railway/Render 默认提供 HTTPS；若自定义域名需确保 SSL |

---

## 6. 完成标志

Phase 8 Sprint 完成时，应满足以下全部条件：

- [ ] 存在一个公开的 HTTPS URL
- [ ] 任何人（不需要安装任何东西）可以在该 URL 上：录音 → 看到实时转录 → 看到情绪/意图标签 → 听到科学匹配的猫叫声
- [ ] FastAPI 文档 (`/docs`) 仍可访问
- [ ] 330 个既有测试全部通过（后端未修改）
- [ ] `docs/development-overview.md` 和 `docs/technical-reference.md` 已更新
