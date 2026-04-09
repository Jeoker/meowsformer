# Meowsformer — 开发概览

> 模块详解、函数签名、算法细节见 [`docs/technical-reference.md`](./technical-reference.md)  
> 测试统计与规范见 [`docs/project-testing.md`](./project-testing.md)  
> 每轮 Batch 开发详细报告见 [`docs/batch-reports/`](./batch-reports/)

---

## 文档与实际差异（最近核对 2026-03-15）

| 项目 | 文档描述 | 实际状况 |
|------|----------|----------|
| `app/auth/` | Phase 6 计划 | **不存在**（暂缓），Batch 1 已移除所有 auth 导入，服务器正常启动 |
| `test_auth.py` | 计划中 | **不存在**（暂缓） |
| 前端 auth 组件 | AuthContext, Navbar, ProtectedRoute, LoginPage 等 | **不存在**（暂缓），Batch 1 已简化 App.vue 为占位页面 |
| Dockerfile / `fly.toml` | Phase 4 曾暂缓 | **已存在**（多阶段 Vue→`static/ui` + FastAPI；Fly 见 [`deployment-fly.md`](./deployment-fly.md)） |
| `db/meowsformer_auth.db` | — | **残留文件** — auth 模块不存在，但空 SQLite 数据库文件仍留存于 `db/` 目录 |
| 测试数量 | — | **330 个测试函数**，全部可运行（Batch 1: +24, Batch 2: +31, Batch 3: +25, Batch 4: +37, Phase 7 后 API 供应方切换: +42+31, 既有测试修复: +15） |

---

## ⚠️ 已知未解决问题（下一轮开发需处理）

### Flet 版本兼容性（已解决，2026-03-14 / 2026-03-15）

**已解决：** Flet 已升级至 `0.82.2`，新增 `flet-audio==0.82.2` 扩展包。音频播放已从 `page.launch_url()` 临时 workaround 恢复为 `fta.Audio` 原生应用内播放。版本已精确锁定（`flet==0.82.2`、`flet-audio==0.82.2`）。

**2026-03-15 补充修复：** Flet 0.80.0 起小写模块级辅助函数已废弃，`app.py` 中 3 处调用已迁移至类方法：`ft.padding.symmetric()` → `ft.Padding.symmetric()`、`ft.border.all()` → `ft.Border.all()`。功能不变，消除运行时 DeprecationWarning。

### 遗留问题 — Mock 测试对第三方库 API 兼容性无保护（P1，测试架构缺陷）

- **现象：** `test_batch2/3/4` 将整个 `flet` 模块替换为自定义 mock，mock 不验证真实 API 签名，导致 API 变更可能逃过测试。
- **修复方向：**
  - 为 Flet 控件补充一组"冒烟测试"，使用真实 `flet` 而非 mock，仅做实例化验证
  - 或在 CI 中增加一步：`python -c "from src.flet_mobile.app import meowsformer_ui"` 用真实 flet 做导入检查
- **影响文件：** `tests/test_batch2_ws_streaming.py`、`tests/test_batch4_ux_enhancements.py`

---

## 1. 项目概览

Meowsformer 是基于 FastAPI 的猫语翻译后端服务，将人类语音翻译为具有生物学意义的真实猫叫声。核心基于瑞典 Lund 大学 Meowsic 研究的科学猫语库。

系统支持两条并行管线：

1. **Legacy Pipeline (Phase 0–3):** 文件上传 → Whisper 转录 → RAG 科学上下文 → LLM 情绪分析 → DSP (VA 映射 + PSOLA 韵律变换) → 合成音频
2. **Streaming Pipeline (Phase 5):** WebSocket 实时音频 → 分块 Whisper → LLM 目标标签生成 → 5 维加权匹配 → 真实猫叫录音播放

**当前状态：** Phase 0–3, 5 后端完成。Phase 6 (Auth) 与 Phase 4 (CI/CD 生产化) 暂缓。Phase 7 (Flet 移动端流式翻译) 已完成。**Phase 8 (Web Demo Sprint) 进行中**；**Phase 9 (Streaming Pipeline 延迟优化) 已完成** — Batch 1（AsyncOpenAI 迁移）+ Batch 2（滚动推测 + 并行 Stop）。详见 [Phase 9 章节](#phase-9--streaming-pipeline-延迟优化)。331 个测试函数全部可运行。

---

## 2. 开发阶段

| Phase | 描述 | 状态 |
|-------|------|------|
| **Phase 0** | 核心 API — FastAPI 端点、Whisper 转录、LLM 分析、RAG | Done |
| **Phase 1** | 数据获取 — Zenodo 语料库下载、元数据解析、registry 索引 | Done |
| **Phase 2** | DSP 引擎 — VA 映射、音频检索、PSOLA 韵律变换 | Done |
| **Phase 3** | 集成 — DSP 接入 API 管线、端到端流程、UI 预览 | Done |
| **Phase 4** | 部署 — CI/CD、生产化加固 | 暂缓（仓库已有 `Dockerfile` / `fly.toml`，见 [`deployment-fly.md`](./deployment-fly.md)） |
| **Phase 5** | 流式管线 — WebSocket streaming、多维标签体系、LLM 目标标签、加权匹配 | Done |
| **Phase 6** | 全栈前端 + JWT 认证 | 暂缓 |
| **Phase 7** | **Flet 移动端流式翻译** — 解除阻塞 → WS 接入 → 音频播放 → UX | **Done** (Batch 1-4 ✅) |
| **Phase 8** | **Web Demo Sprint** — 激活 Vue 前端 → Tailwind 打磨 → 一体化云端部署 → 可分享 URL | **进行中**（UI 批次见 batch 报告） |
| **Phase 9** | **Streaming Pipeline 延迟优化** — AsyncOpenAI + 滚动推测 + 并行 Stop | **Done** (Batch 1-2 ✅) |

---

## 3. 技术栈

| 类别 | 技术 |
|------|------|
| **语言** | Python 3.10+ (推荐 3.12) |
| **Web 框架** | FastAPI + Uvicorn |
| **数据校验** | Pydantic V2 |
| **AI / LLM** | OpenAI API (GPT-4o, Whisper `whisper-1`), `instructor` (结构化输出) |
| **向量数据库** | ChromaDB (Phase 0 RAG, 本地持久存储) |
| **音频处理** | FFmpeg (subprocess), `python-multipart` |
| **音频 DSP** | `librosa` (f0/pYIN, I/O), `pytsmod` (WSOLA), `soundfile`, `scipy`, `numpy` |
| **声学特征提取** | `librosa` (pYIN f0, RMS 能量, 时长 — 用于标签构建) |
| **WebSocket** | Starlette/FastAPI 内置 WS |
| **数据获取** | `zenodo-get` |
| **环境管理** | `python-dotenv`, `pydantic-settings` |
| **日志** | `loguru` |
| **测试** | `unittest` |
| **前端** | Vue 3 + TypeScript + Vite + Tailwind CSS |
| **移动端** | Flet (Python, Material Design 3) |
| **认证** | `python-jose`, `passlib`, SQLAlchemy, Alembic（依赖已安装，`app/auth/` 模块未实现） |

---

## 4. 系统架构（高层）

```
┌──────────────────────────────────────────────────────────────────┐
│                    Meowsformer 系统架构                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─ Legacy Pipeline (REST, Phase 0–3) ─────────────────────────┐ │
│  │  POST /api/translate  →  Whisper → RAG → LLM → JSON         │ │
│  │  POST /api/v1/translate →  ... + DSP(PSOLA) → base64 WAV    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ Streaming Pipeline (WebSocket, Phase 5) ───────────────────┐ │
│  │  WS /ws/translate → chunked Whisper → LLM target tags       │ │
│  │                   → catalog matching → real WAV playback    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ Shared Infrastructure ─────────────────────────────────────┐ │
│  │  OpenAI API (Whisper + GPT-4o)  │  ChromaDB  │  librosa     │ │
│  │  483 CatMeows samples  │  tagged_samples.json (5维标签)      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. 当前进度概览

### Phase 7 — Flet 移动端流式翻译

> **决策记录：** Auth (Phase 6) 暂缓，先移除 auth 导入解除阻塞，优先推进 Flet 移动端与 Phase 5 流式管线的集成。

Phase 7 分 4 个 Batch 串行推进，每个 Batch 独立走完 developer → reviewer → test-engineer → reviewer → 文档更新 的完整流程。

| Batch | 目标 | 状态 | 详细报告 |
|-------|------|------|---------|
| Batch 1 | 解除服务器启动阻塞 | ✅ 已完成 | [phase7-batch1.md](./batch-reports/phase7-batch1.md) |
| Batch 2 | 接入 WebSocket 流式管线 | ✅ 已完成 | [phase7-batch2.md](./batch-reports/phase7-batch2.md) |
| Batch 3 | 修复音频播放 | ✅ 已完成 | [phase7-batch3.md](./batch-reports/phase7-batch3.md) |
| Batch 4 | UX 完善 | ✅ 已完成 | [phase7-batch4.md](./batch-reports/phase7-batch4.md) |
| **Batch 5** | **flet-audio 升级：Flet 0.82.2 + 原生应用内播放恢复** | ✅ **已完成** | [phase7-batch5.md](./batch-reports/phase7-batch5.md) |
| Batch 6 | Web 浏览器兼容 | ❌ **已取消** | 改用 WSLg 桌面模式，见 [wsl-dev-setup-2026-03-15.md](./batch-reports/wsl-dev-setup-2026-03-15.md) |

---

### Phase 8 — Web Demo Sprint（进行中）

> **决策记录 (2026-03-28)：技术路线转向 — 从 Flet-First 到 Web-First**
>
> **背景：** Phase 7 完成了 Flet 移动端流式翻译（6 个 Batch），但产出只能在本地 WSL2 桌面模式运行，无法生成可分享的 URL。同时 `src/ui/` 中的 Vue 前端已包含完整的功能组件（WebSocket composable、音频播放 composable、录音组件、结果卡片），全部在 Phase 5 期间编写但从未接入 `App.vue`。
>
> **核心问题：** 后端 Phase 0-5 已是生产质量（330 测试、双管线、5 维匹配引擎），但项目缺少一个别人可以打开的网页。大量精力投在用户看不到的地方（后端完整性、测试覆盖率），零投入在用户唯一能看到的地方（可访问的界面）。
>
> **决策：**
> 1. 从 Flet-First 转向 Web-First — 激活已有 Vue 前端，而非继续投入 Flet
> 2. 一体化部署 — FastAPI 同时 serve Vue 构建产物 + 音频文件，一个服务搞定
> 3. 采用 Sprint Mode 工作流 — 跳过 reviewer/test-engineer 多轮循环，快速迭代
>
> **不变的部分：** 后端代码（Phase 0-5）零修改；Flet 客户端保留在 `src/flet_mobile/` 不删除；330 个测试继续有效。

#### Vue 前端资产（2026-04-06 更新）

| 文件 / 目录 | 说明 |
|-------------|------|
| `src/ui/src/pages/TranslatePage.vue` | 单页编排：composable 接线 + 子组件组合 |
| `src/ui/src/components/translate/` | `DemoHero`, `ConnectionStatus`, `BreedPreference`, `RecordingDeck`, `LiveFeed`, `ErrorBanner`, `ResultSection`, `ResultCard`；Legacy `MeowPreviewPlayer` 仍保留 |
| `src/ui/src/composables/useStreamingTranslation.ts` | WS 状态机；ScriptProcessor + **按真实采样率重采样至 16kHz**；`disconnect` 仅在录音中发 `stop`；`AudioContext.resume()` |
| `src/ui/src/composables/useAudioPreview.ts` | base64→播放；调用方对 `audioBase64` 使用 `watch(..., { immediate: true })` |
| `src/ui/src/types/api.ts` | 镜像后端 Pydantic / WS 消息 |
| `src/ui/src/App.vue` | 挂载 `TranslatePage` |
| `src/ui/vite.config.ts` | 开发代理 `/api`、`/ws` → :8000 |

**批次备忘：** [phase8-batch-ui-2026-04-06.md](./batch-reports/phase8-batch-ui-2026-04-06.md)（根因修复与文件表）。

#### Sprint 计划

详细的 Batch 计划、部署架构、developer subagent 沟通模板见 [phase8-sprint-plan.md](./batch-reports/phase8-sprint-plan.md)。

概要：

| Batch | 目标 | 关键产出 |
|-------|------|---------|
| Batch 1 | 前端激活 + UI 打磨 | ✅ `TranslatePage` + 子组件 + 流式前端/WS/播放根因修复 — 见 [phase8-batch-ui-2026-04-06.md](./batch-reports/phase8-batch-ui-2026-04-06.md) |
| Batch 2 | 静态文件服务 + 部署配置 | ✅ `main.py` `/` + `/assets`；`Dockerfile` 多阶段 — 见 [`deployment-fly.md`](./deployment-fly.md) |
| Batch 3 | 云端部署 | Fly.io（`fly deploy`）；公开 URL 端到端验证 |

**工作流：** 采用 Sprint Mode（见 `dev-workflow.mdc`），跳过 reviewer/test-engineer 循环，PM 通过浏览器手动验证。

---

### Phase 9 — Streaming Pipeline 延迟优化（进行中）

> **决策记录 (2026-04-08)：** 用户松开按钮到收到猫叫音频典型延迟 3–6s、最差 11s。三个根因：sync OpenAI 客户端阻塞 event loop、stop 处理完全串行、推测性执行只触发一次。目标：典型 → 1–2s，最差 → 3–4s。

| Batch | 目标 | 状态 | 详细报告 |
|-------|------|------|---------|
| Batch 1 | AsyncOpenAI 基础设施迁移（sync → async transport） | ✅ 已完成 | [phase9-batch1.md](./batch-reports/phase9-batch1.md) |
| Batch 2 | 滚动推测 + 并行 Stop | ✅ 已完成 | [phase9-batch2.md](./batch-reports/phase9-batch2.md) |

**优化方案文档：** [phase9-latency-optimization-plan.md](./batch-reports/phase9-latency-optimization-plan.md)

---

### Phase 6 — Auth & Frontend ⏸ 暂缓

- Batch 1 已移除所有 auth 导入，`app/auth/` 模块推迟到后续阶段实现。
- **残留：** `db/meowsformer_auth.db` 空 SQLite 文件、`requirements.txt` 中的 auth 相关依赖。
- **待实现：** `app/auth/` 完整模块、前端认证组件、`tests/test_auth.py`。

---

## 6. 搭建与运行

### 前提

- Python 3.10+ (推荐 3.12)
- FFmpeg (系统 PATH)
- OpenAI API Key

### 安装

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-key-here
CHROMA_DB_PATH=./db/chroma_db
DEBUG_MODE=True
EOF

python -m tools.download_datasets          # 下载语料库
python -m tools.build_tags                 # 构建标签索引 (~2 min)
python -m tools.build_tags --skip-audio    # 仅元数据标签 (秒级)
```

### 启动服务

```bash
python main.py
# 或: uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**可用端点：**
- API 文档: `http://localhost:8000/docs`
- 健康检查: `GET /health`
- Legacy 翻译: `POST /api/translate`, `POST /api/v1/translate`
- 流式翻译: `WS ws://localhost:8000/ws/translate`
- 认证: `POST /auth/register`, `POST /auth/login`, `GET /auth/me`（待实现）

### 启动 Flet 移动端

```bash
# 终端 1
python main.py

# 终端 2（WSL2 需先完成 [wsl2-audio-setup.md](./wsl2-audio-setup.md)）
MEOWSFORMER_FLET_VIEW=desktop flet run -m src.flet_mobile.app
```

必须用 `flet run -m` 而非 `python -m`，否则预构建客户端不包含 flet-audio，会显示 "Unknown control: Audio"。

### 可选：Docker 与 Fly.io

- 镜像说明、`FETCH_DATA`、Chroma 打入镜像、`fly secrets`、计费文档链接：**[`deployment-fly.md`](./deployment-fly.md)**。
- 部署：`fly auth login` → `fly secrets set OPENAI_API_KEY=...` → `fly deploy`（`fly.toml` 中 `internal_port` 与容器 `PORT` 一致，默认 8080）。

---

## 7. 后续阶段

| 阶段 | 内容 | 备注 |
|------|------|------|
| **Phase 8 — Web Demo Sprint** | **Vue 前端激活 + 一体化云端部署** | **准备中，下一个要做的阶段**；详见 [sprint 计划](./batch-reports/phase8-sprint-plan.md) |
| Phase 6 — Auth | JWT 认证 (`app/auth/`)、前端认证组件 | 暂缓，Phase 8 后视需求决定 |
| Phase 4 — 部署加固 | CI/CD、生产化加固 | Phase 8 包含基础 Dockerfile + 云部署；CI/CD 后续补充 |
| Phase 5 补充测试 | sample_matcher, streaming_transcription, sound_selection, ws_endpoints | 可穿插完成 |
| 高级功能 | 用户反馈循环 (RLHF), 多语言, 标签权重调优, 多品种声线偏好 | 长期 |

---

## 8. 文档索引

| 文档 | 内容 |
|------|------|
| **本文件** (`development-overview.md`) | 项目概览、开发阶段、技术栈、架构、进度、搭建指南、路线图 |
| [`technical-reference.md`](./technical-reference.md) | 完整目录结构、模块详解、函数签名与职责、数据流与调用链、算法细节 |
| [`project-testing.md`](./project-testing.md) | 测试总览、运行方式、用例统计、测试规范 |
| [`wsl2-audio-setup.md`](./wsl2-audio-setup.md) | WSL2 麦克风录音配置 |
| [`batch-reports/`](./batch-reports/) | 每轮 Batch 的详细开发报告（修改文件、技术方案、验收结果） |
| [`batch-reports/phase8-sprint-plan.md`](./batch-reports/phase8-sprint-plan.md) | **Phase 8 Web Demo Sprint 详细计划**（技术路线转向决策、Batch 拆分、部署架构、developer 沟通模板） |
| [`deployment-fly.md`](./deployment-fly.md) | Docker 镜像、`fly.toml`、Fly.io 命令、计费与官方文档链接 |
