# Meowsformer — 技术参考手册

本文档详细记录每个文件、主要 functions 的作用及代码运行流程。  
项目概览、开发进度与计划见 [`docs/development-overview.md`](./development-overview.md)。  
每轮 Batch 的详细开发报告见 [`docs/batch-reports/`](./batch-reports/)。

---

## 1. 完整目录结构

```text
/
├── app/                                  # 核心后端应用
│   ├── __init__.py
│   ├── api/                              # API 路由层
│   │   ├── __init__.py                   # 空包（Batch 1 清空，断开连锁导入链）
│   │   ├── endpoints.py                  # REST API 端点 (POST /api/translate, POST /api/v1/translate)
│   │   └── ws_endpoints.py              ★ WebSocket 流式端点 (WS /ws/translate) — Phase 5
│   ├── auth/                            ★ 认证模块 — Phase 6（暂缓，目录不存在，所有导入已移除）
│   ├── core/                             # 核心配置
│   │   ├── __init__.py
│   │   ├── config.py                     # Pydantic Settings (.env 读取; JWT 字段已移除)
│   │   └── api_client.py               ★ API 供应方工厂 — 边缘功能 (openai / ai_builders 单点切换)
│   ├── data/                            ★ 数据定义层 — Phase 5
│   │   ├── __init__.py
│   │   └── meow_catalog.py             ★ 5 维标签分类体系 + 规则引擎
│   ├── db/                               # 数据库层
│   │   ├── __init__.py
│   │   └── vector_store.py               # ChromaDB 客户端初始化 (RAG 向量存储)
│   ├── schemas/                          # Pydantic 数据模型
│   │   ├── __init__.py
│   │   ├── translation.py                # Phase 0–3 响应模型 (CatTranslationResponse, MeowSynthesisResponse)
│   │   └── ws_messages.py              ★ WebSocket 消息协议模型 — Phase 5
│   │                                      (TargetTagSet, TaggedSampleInfo, StreamingTranslationResult,
│   │                                       WSConfigMessage, WSStopMessage, WSTranscriptionMessage,
│   │                                       WSAnalysisPreviewMessage, WSResultMessage, WSErrorMessage)
│   └── services/                         # 业务逻辑服务层
│       ├── __init__.py
│       ├── audio_processor.py            # FFmpeg 音频转换/特征提取 (Phase 0)
│       ├── llm_service.py                # OpenAI GPT-4o 意图分析 (Phase 0, instructor 结构化输出)
│       ├── rag_service.py                # ChromaDB 知识检索 (Phase 0 RAG)
│       ├── transcription_service.py      # OpenAI Whisper 文件转录 (Phase 0)
│       ├── synthesis_service.py          # DSP 合成桥接服务 (Phase 3, emotion→intent→VA→PSOLA)
│       ├── streaming_transcription_service.py ★ 流式 Whisper 转录 — Phase 5
│       ├── sound_selection_service.py   ★ LLM 目标标签生成 + 推测性执行 — Phase 5
│       └── sample_matcher.py            ★ 多维加权标签匹配引擎 — Phase 5
│
├── src/                                  # DSP 引擎 & 前端
│   ├── __init__.py
│   ├── engine/                           # Phase 2 — DSP 音频合成引擎
│   │   ├── __init__.py
│   │   ├── dsp_processor.py              # VA 映射、最近邻检索、PSOLA 韵律变换
│   │   └── description_generator.py      # NatureLM-audio 风格置信描述生成 (Phase 3)
│   ├── flet_mobile/                      # Flet 移动端原型 (API-First)
│   │   ├── __init__.py
│   │   ├── app.py                        # 主页面编排 (Batch 2: REST/Streaming 模式切换, Phase 5 标签适配)
│   │   ├── audio_recorder.py             # 16kHz PCM 采集 + 波形快照
│   │   ├── translation_client.py         # httpx REST + websockets WS 并发客户端 (Batch 2: asyncio.TaskGroup)
│   │   ├── bioacoustic_player.py         # sound_id 映射本地样本 (flet-audio 0.82.2: fta.Audio 应用内播放, release→update→play 顺序)
│   │   └── theme.py                      # 视觉 Token
│   └── ui/                               # 前端 Vue SPA
│       ├── index.html
│       ├── package.json                  # Vue 3, Tailwind CSS
│       ├── package-lock.json
│       ├── vite.config.ts                # Vite 开发服务器 (代理 /api + /ws 到 FastAPI :8000)
│       ├── postcss.config.js             # PostCSS + Tailwind 插件
│       ├── tailwind.config.js            # 自定义 meow 色板
│       ├── tsconfig.json, tsconfig.node.json
│       └── src/
│           ├── main.ts                   # Vue 入口
│           ├── App.vue                   # 占位页面（Batch 1 移除 auth 导入，简化为 TranslatePlaceholder）
│           ├── index.css                 # Tailwind base styles
│           ├── env.d.ts                  # Vue 模块类型声明
│           ├── context/                  # 未实现 (auth 状态管理)，暂缓
│           ├── composables/
│           │   ├── useAudioPreview.ts    # base64→Blob→ObjectURL 播放
│           │   └── useStreamingTranslation.ts ★ WebSocket 流式翻译 composable (Phase 5)
│           │   # useAuth.ts 未实现
│           ├── types/
│           │   └── api.ts                # TypeScript 类型 (镜像后端 Pydantic 模型)
│           ├── components/
│           │   ├── AudioRecorder.vue     # Legacy 录音组件
│           │   ├── MeowPreviewPlayer.vue # Legacy 预览播放器
│           │   ├── MeowPreviewPlayer.css
│           │   └── translate/
│           │       ├── ResultCard.vue     # 翻译结果展示
│           │       ├── AudioRecorder.vue ★ 实时录音 (Phase 5)
│           │       ├── MeowPreviewPlayer.vue
│           │       └── MeowPreviewPlayer.css
│           │   # auth/, layout/ 未实现
│           └── pages/                    # 未实现 (LoginPage, RegisterPage, TranslatePage)
│
├── tools/                                # 工具脚本
│   ├── __init__.py
│   ├── download_datasets.py              # Zenodo 数据集下载/解析/registry 构建 (Phase 1)
│   ├── build_tags.py                    ★ 标签构建脚本 — Phase 5 (registry.json + librosa → tagged_samples.json)
│   └── play_audio.py                     # 音频试听调试工具
│
├── assets/                               # 静态资源
│   ├── audio_db/
│   │   ├── registry.json                 # 元数据索引 (483 CatMeows 样本, VA 标注)
│   │   └── tagged_samples.json          ★ 带多维标签的样本索引 (由 build_tags.py 生成)
│   └── raw_data/                         # 音频语料库 (git-ignored)
│       ├── catmeows/                     #   CatMeows dataset (Zenodo 10.5281/zenodo.4007940)
│       └── meowsic/                      #   Meowsic dataset  (Zenodo 10.5281/zenodo.3245999)
│
├── tests/                                # 单元 & 集成测试 (详见 docs/project-testing.md)
│   ├── __init__.py
│   ├── shared_params.py                ★ 跨文件共享测试常量 (URL / 模型名 / 样本 ID / 音频 stub 等)
│   ├── flet_mocks.py                   ★ flet 控件 mock 基础设施 (_Ctrl / install_flet_mock / BaseMockPage)
│   ├── ws_stubs.py                     ★ WebSocket 测试桩 (MockWebSocket / ws_connect_coro / async_chunks)
│   ├── test_api_endpoints.py           # API 端点集成测试
│   ├── test_audio_services.py          # 音频服务测试
│   ├── test_llm_service.py             # LLM 服务测试
│   ├── test_rag_service.py             # RAG 服务测试
│   ├── test_download_datasets.py       # 数据集下载/解析测试
│   ├── test_dsp_processor.py           # DSP 引擎测试
│   ├── test_description_generator.py   # 描述生成测试
│   ├── test_synthesis_service.py       # 合成服务测试
│   ├── test_batch1_auth_removal.py     # Phase 7 Batch 1 验证
│   ├── test_batch2_ws_streaming.py     # Phase 7 Batch 2 验证
│   ├── test_batch3_audio_playback.py   # Phase 7 Batch 3 验证
│   ├── test_batch4_ux_enhancements.py  # Phase 7 Batch 4 验证
│   ├── test_api_client.py              # API provider 切换测试
│   └── test_api_provider_switch.py     # API provider 端到端切换验证
│
├── docs/                                 # 项目文档
│   ├── development-overview.md           # 项目概览、开发阶段、进度、路线图
│   ├── technical-reference.md            # 本文件
│   ├── project-testing.md                # 测试总览与规范
│   ├── wsl2-audio-setup.md                # WSL2 麦克风录音配置 (2026-03-15)
│   └── batch-reports/                    # 每轮 Batch 详细开发报告
│       ├── phase7-batch1.md
│       ├── phase7-batch2.md
│       ├── phase7-batch3.md
│       ├── phase7-batch4.md
│       ├── phase7-batch5.md
│       ├── test-param-refactor.md        ★ 测试参数重构报告 (shared_params / flet_mocks / ws_stubs)
│       └── wsl-dev-setup-2026-03-15.md    # WSL2 开发环境验证
│
├── scripts/
│   └── run_e2e_test.sh                   # E2E 上线测试 (Docker / 本地)
├── main.py                               # FastAPI app 创建, 路由注册, startup 事件
├── requirements.txt
├── .env                                  # 环境变量 (不提交)
├── .gitignore
└── LICENSE                               # Apache 2.0
```

> 标注 ★ 的文件为 Phase 5 或后续批次新增。

---

## 2. 服务间依赖关系

```
main.py (Batch 1 后，auth 导入已全部移除)
  │
  ├── [LINE 6] from app.api.endpoints import router as api_router
  │     │
  │     ├── app/api/__init__.py  (空包，不再触发连锁导入)
  │     │
  │     └── app/api/endpoints.py  (无 auth 依赖)
  │           ├── app/services/audio_processor.py
  │           ├── app/services/transcription_service.py
  │           │     └── OpenAI Whisper API (whisper-1)
  │           ├── app/services/rag_service.py
  │           │     └── app/db/vector_store.py → ChromaDB
  │           ├── app/services/llm_service.py
  │           │     └── OpenAI GPT-4o (via instructor)
  │           └── app/services/synthesis_service.py
  │                 ├── src/engine/dsp_processor.py
  │                 │     └── assets/audio_db/registry.json
  │                 └── src/engine/description_generator.py
  │
  ├── [LINE 7] app/api/ws_endpoints.py (WebSocket 路由)
  │     ├── app/services/streaming_transcription_service.py
  │     │     └── OpenAI Whisper API (whisper-1)
  │     ├── app/services/sound_selection_service.py
  │     │     ├── OpenAI GPT-4o (via instructor)
  │     │     ├── app/data/meow_catalog.py (标签词汇表)
  │     │     └── app/services/sample_matcher.py
  │     │           └── assets/audio_db/tagged_samples.json
  │     └── app/schemas/ws_messages.py (协议定义)
  │
  └── (auth 相关导入已全部移除，无连锁失败风险)
```

---

## 2.5 API 供应方切换

### `app/core/api_client.py`（新增）

唯一的供应方工厂，所有服务从此获取 OpenAI 客户端实例。

| 函数 | 签名 | 说明 |
|------|------|------|
| `get_openai_client` | `() -> OpenAI` | 按 `settings.API_PROVIDER` 返回配置好的 `OpenAI` 实例；不内置缓存，调用方应自行在模块级持有缓存引用 |
| `get_instructor_client` | `() -> instructor.Instructor` | 返回 `instructor.from_openai(get_openai_client())` 封装 |

### `app/core/config.py`（新增字段）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `API_PROVIDER` | `Literal["openai","ai_builders"]` | `"openai"` | 供应方选择，日常无需修改 |
| `AI_BUILDER_TOKEN` | `str` | `""` | ai-builders Bearer token |
| `AI_BUILDER_BASE_URL` | `str` | `"https://space.ai-builders.com/backend/v1"` | ai-builders API 基础 URL（MCP 指定，含 /v1） |
| `LLM_MODEL` | `str` | `""` | 空时按 `API_PROVIDER` 自动解析：openai→gpt-4o，ai_builders→deepseek；可在 .env 中覆盖 |

### 受影响的服务文件

所有文件改动均向下兼容，默认配置下行为不变：

| 文件 | 改动 |
|------|------|
| `transcription_service.py` | 增加模块级 `_client` 懒加载缓存，`_get_client()` 内调用 `get_openai_client()` |
| `streaming_transcription_service.py` | `_get_client()` 内部改用 `get_openai_client()` |
| `llm_service.py` | 增加模块级 `_client` 懒加载缓存；`model` 硬编码改为 `settings.LLM_MODEL` |
| `sound_selection_service.py` | `_get_client()` 改用 `get_instructor_client()`；模型名参数化 |
| `vector_store.py` | `_embedding_fn` 按 `API_PROVIDER` 条件初始化，ai-builders 分支传入 `api_key` 与 `api_base` |

---

## 3. 模块详解 — 5 维标签分类体系

### 3.1. 标签词汇表与规则 (`app/data/meow_catalog.py`)

5 个独立维度，每个样本可在多维度同时携带多个标签：

#### 维度 1 — emotion (猫的情绪)

| 标签 | 分配规则 |
|------|---------|
| `hungry` | context=Food |
| `eager` | context=Food 且 arousal > 0.8 |
| `demanding` | context=Food 且 arousal > 0.8 |
| `anxious` | context=Isolation 且 arousal > 0.6 |
| `lonely` | context=Isolation |
| `distressed` | context=Isolation 且 valence < -0.5 |
| `content` | context=Brushing 且 valence > 0 |
| `relaxed` | context=Brushing 且 valence > 0 且 arousal < 0.5 |
| `annoyed` | context=Brushing 且 valence < 0 |
| `agitated` | valence < 0 且 arousal > 0.6 (与上下文无关) |
| `calm` | arousal < 0.4 (与上下文无关) |

#### 维度 2 — intent (沟通目的)

| 标签 | 分配规则 |
|------|---------|
| `requesting_food` | context=Food |
| `demanding_attention` | context=Food 或 Isolation |
| `seeking_companionship` | context=Isolation |
| `expressing_comfort` | context=Brushing 且 valence > 0 |
| `protesting` | context=Brushing 且 valence < 0 |
| `greeting` | context=Brushing 且 valence > 0.2 且 0.3 ≤ arousal ≤ 0.6 |

#### 维度 3 — acoustic (声学特征，librosa 提取)

| 标签 | 分配规则 |
|------|---------|
| `high_pitch` | 中位 f0 > 600 Hz |
| `low_pitch` | 中位 f0 < 400 Hz |
| `mid_pitch` | 400 ≤ f0 ≤ 600 Hz |
| `short_burst` | 时长 < 0.5s |
| `medium_length` | 0.5s ≤ 时长 ≤ 1.5s |
| `prolonged` | 时长 > 1.5s |
| `loud` | RMS > P75 (全局) |
| `soft` | RMS < P25 (全局) |
| `rising_tone` | f0 线性斜率 > 0 |
| `falling_tone` | f0 线性斜率 < 0 |
| `trembling` | f0 标准差 > 80 Hz |

**声学特征提取流程** (`tools/build_tags.py`):

1. `librosa.load()` 加载 WAV
2. `librosa.pyin()` 估算基频 (fmin=60Hz, fmax=1500Hz)
3. 计算 voiced f0 的 median、std、linear slope
4. `np.sqrt(np.mean(y**2))` 计算 RMS 能量
5. 所有样本 RMS 排序后计算 P25/P75 分位线

#### 维度 4 — social_context (社交场景)

| 标签 | 分配规则 |
|------|---------|
| `feeding_time` | context=Food |
| `alone_at_home` | context=Isolation |
| `separation` | context=Isolation |
| `being_petted` | context=Brushing |
| `physical_contact` | context=Brushing |
| `near_owner` | context=Brushing 或 Food |

#### 维度 5 — breed_voice (品种声线)

| 标签 | 分配规则 |
|------|---------|
| `deep_voice` | breed=Maine Coon |
| `bright_voice` | breed=European Shorthair |

**主要函数:** `tag_emotion()`, `tag_intent()`, `tag_acoustic()`, `tag_social_context()`, `tag_breed_voice()`

### 3.2. 标签构建管线 (`tools/build_tags.py`)

一次性脚本，处理全部 483 样本：

```
registry.json (483 samples)
      │
      ├──► 维度 1/2/4/5: 基于 context/VA/breed 的规则标签
      │
      ├──► 维度 3: librosa 声学特征提取
      │      pYIN f0 → median, std, slope
      │      RMS → 全局百分位排名
      │      duration → 直接计算
      │
      └──► 输出 tagged_samples.json (平均每样本 12.1 个标签)
```

### 3.3. 加权标签匹配引擎 (`app/services/sample_matcher.py`)

**核心算法 — 加权 Jaccard 相似度：**

```
score(target, sample) = Σ_dim  weight[dim] × |target[dim] ∩ sample[dim]|
                                              ─────────────────────────────
                                              |target[dim] ∪ sample[dim]|
```

**维度权重：**

| 维度 | 权重 | 理由 |
|------|------|------|
| emotion | 0.30 | 情绪匹配决定声音主观感受 |
| intent | 0.30 | 沟通意图与情绪同等重要 |
| acoustic | 0.15 | 声学特征为辅助匹配 |
| social_context | 0.15 | 场景匹配确保语义合理 |
| breed_voice | 0.10 | 品种声线为次要偏好 |

品种偏好 boost: 匹配用户指定品种的样本额外 +0.05 分。

**主要函数：**
- `score_sample(target_tags, sample)` — 单样本 5 维评分，跳过目标标签为空的维度
- `find_best_match(target_tags, breed_preference=None, top_k=1)` — 内部调用 `get_samples()` 加载样本，遍历 483 样本取 top-K

### 3.4. LLM 目标标签生成 (`app/services/sound_selection_service.py`)

**核心设计：** LLM 不选具体样本，只输出目标标签 (TargetTagSet)。样本由匹配引擎确定性选出。

LLM 系统提示词将完整标签词汇表注入，约束 LLM 只在有效范围内选择。

**输出示例：**

```json
{
  "emotion": ["lonely", "anxious"],
  "intent": ["seeking_companionship"],
  "acoustic": ["prolonged", "soft", "falling_tone"],
  "social_context": ["alone_at_home", "separation"],
  "reasoning": "用户表达了对猫咪的思念，猫咪应以孤独渴望陪伴的方式回应"
}
```

**推测性执行 (Speculative Execution)：**

```
            ┌─── 部分转录 (≥5 词) ───┐
            │                         ▼
  录音中... │    异步 LLM 调用 → SpeculativeCache.store(text₁, tags₁)
            │
            └─── 用户停止 ──────────► 最终转录 text₂
                                        │
                                   cache.is_similar(text₂) — SequenceMatcher ratio
                                        │
                              ┌─── ratio ≥ 0.7 ────┐─── ratio < 0.7 ────┐
                              │                     │                     │
                        cache.get() → tags₁     新 LLM 调用 → tags₂
                           (零延迟)
```

多数场景可省去 2–3 秒 LLM 延迟。

**主要函数：**
- `async generate_target_tags(text)` — GPT-4o → TargetTagSet
- `SpeculativeCache.store(text, tags)` — 存储推测结果
- `SpeculativeCache.is_similar(final_text, threshold=0.7)` — SequenceMatcher 检查文本相似度
- `SpeculativeCache.get()` — 返回缓存的 TargetTagSet（无参数）
- `SpeculativeCache.clear()` — 清空缓存
- `async select_and_encode(target_tags, breed_preference)` — 匹配 + 读取 WAV + base64

---

## 4. 模块详解 — 流式转录服务

### `app/services/streaming_transcription_service.py`

维护一个持续增长的 PCM 音频缓冲区：

| 参数 | 值 | 说明 |
|------|-----|------|
| `MIN_TRANSCRIPTION_INTERVAL` | 2.5s | 中间转录最小间隔 |
| `MIN_BUFFER_SIZE` | 32000 bytes | ≈ 1 秒 16kHz 16-bit mono |
| 采样率 | 16000 Hz | PCM 16-bit mono |

**缓冲区 → WAV 流程：**

```
[chunk₁] + [chunk₂] + ... → bytes 拼接
→ np.frombuffer(dtype=int16) → float32 / 32768.0
→ soundfile.write(tmp.wav, PCM_16)
→ OpenAI Whisper API → 文本
```

**主要函数：**
- `StreamingTranscriptionSession.add_chunk(data: bytes)` — 累积 PCM
- `should_transcribe()` — 判断是否满足中间转录条件（缓冲区 ≥ MIN_BUFFER_SIZE 且距上次 ≥ 2.5s）
- `transcribe_intermediate()` — 中间转录 (每 ~2.5s)，调用 Whisper `whisper-1`
- `transcribe_final()` — 最终转录 (用户停止时)
- `get_buffer_as_wav_bytes()` — 返回完整缓冲区的 WAV bytes（调试用）
- `reset()` — 清空缓冲区和状态

---

## 5. 模块详解 — WebSocket 端点

### `app/api/ws_endpoints.py`

#### 协议定义

**Client → Server:**

| 消息类型 | 格式 | 说明 |
|----------|------|------|
| `config` | JSON `{"type":"config","breed_preference":"Maine Coon"}` | 连接配置 (可选) |
| 音频块 | Binary (PCM 16-bit 16kHz) | 每 ~256ms 一帧 (buffer=4096 @ 16kHz) |
| `stop` | JSON `{"type":"stop"}` | 停止录音 |

**Server → Client:**

| 消息类型 | 触发时机 | 关键字段 |
|----------|---------|----------|
| `transcription` | 每 ~2.5s 及最终 | `text`, `is_final` |
| `analysis_preview` | 推测性 LLM 完成 | `emotion`, `intent` |
| `result` | 最终结果 | `transcription`, `selected_category`, `audio_base64`, `reasoning` |
| `error` | 任何错误 | `detail` |

#### 会话状态 (StreamingSession)

每个 WebSocket 连接维护独立的 `StreamingSession`：
- `breed_preference` — 用户品种偏好
- `StreamingTranscriptionSession` — 音频缓冲区 + 转录状态
- `SpeculativeCache` — 推测性 LLM 结果缓存
- `_speculative_task` — asyncio Task 句柄

**主要函数：** `ws_translate(websocket)` — 主处理循环

---

## 6. 模块详解 — Legacy Pipeline (Phase 0–3)

### 6.1. 音频处理 (`app/services/audio_processor.py`)

- `convert_to_wav(input_path, output_path)` — 异步 FFmpeg → 16kHz mono WAV
- `get_audio_duration(file_path)` — ffprobe 获取音频时长
- `get_audio_volume(file_path)` — ffmpeg volumedetect 获取均值分贝 + RMS 振幅
- `extract_basic_features(file_path)` — 提取 `duration_seconds`, `mean_volume_db`, `rms_amplitude`

### 6.2. 转录 (`app/services/transcription_service.py`)

- `transcribe_audio(file_path)` — 文件 → FFmpeg WAV → OpenAI Whisper (`whisper-1`) → 文本

### 6.3. RAG (`app/services/rag_service.py`)

- `initialize_knowledge_base()` — 向 ChromaDB 填充猫声学科学文献
- `retrieve_context(query, n=3)` — 检索 top-3 相关上下文

### 6.4. LLM 分析 (`app/services/llm_service.py`)

- `analyze_intention(text, features, rag_ctx)` — 拼接提示词 → GPT-4o → `CatTranslationResponse`

### 6.5. DSP 引擎 (`src/engine/dsp_processor.py`)

**Intent → VA 映射 (Russell 情绪环状模型)：**

| Intent | Valence | Arousal |
|--------|---------|---------|
| Affiliative | +0.70 | 0.35 |
| Contentment | +0.80 | 0.15 |
| Play | +0.60 | 0.85 |
| Requesting | +0.30 | 0.75 |
| Solicitation | +0.40 | 0.60 |
| Agonistic | −0.80 | 0.90 |
| Distress | −0.70 | 0.85 |
| Frustration | −0.50 | 0.70 |
| Alert | 0.00 | 0.65 |
| Neutral | 0.00 | 0.40 |

**PSOLA 韵律变换流程：**

1. pYIN f0 估计
2. 品种基频混合 (8 品种基线, 50% blend)
3. Arousal → 时长调制 (高 arousal 压缩, 低 arousal 拉伸)
4. WSOLA 时域拉伸 (pytsmod)
5. 重采样音高偏移
6. Arousal 包络整形
7. 峰值归一化 (0.95)

**主要函数：**
- `map_intent_to_va(intent)` — Intent → (valence, arousal)
- `get_best_match(target_v, target_a, registry)` — 最近邻样本检索
- `apply_prosody_transform(audio, sr, target_a, breed)` — PSOLA 变换

### 6.6. 合成桥接 (`app/services/synthesis_service.py`)

**Emotion → Intent 映射：** Hungry→Requesting, Angry→Agonistic, Happy→Affiliative, Alert→Alert

**完整流程：** emotion → intent → VA → 最近邻 → PSOLA → base64 WAV → NatureLM 描述 → `MeowSynthesisResponse`

- `synthesize_and_describe(emotion, breed, ...)` — 主入口函数

### 6.7. 描述生成器 (`src/engine/description_generator.py`)

- Intent → 中文语义标签
- VA 距离 → 指数衰减置信分数 `exp(-d)` → 五级中文评价
- `generate_description_from_synthesis(...)` — 拼装结构化中文描述

---

## 7. 模块详解 — 认证 (Phase 6) — 暂缓

### `app/auth/` — 暂缓实现

**Batch 1 已完成的清理工作：**
- `main.py` — 移除 `auth_router` 导入、`create_tables()` 调用、路由注册
- `app/api/endpoints.py` — 移除 `get_optional_user` 导入
- `app/api/__init__.py` — 清空为空包，断开连锁导入链
- `app/core/config.py` — 移除 `JWT_SECRET_KEY`、`JWT_ALGORITHM`、`JWT_ACCESS_TOKEN_EXPIRE_MINUTES`
- `src/ui/vite.config.ts` — 移除 `/auth` proxy
- `src/ui/src/App.vue` — 移除 auth 组件导入，简化为占位页面

**残留文件：** `db/meowsformer_auth.db` — 空 SQLite 数据库文件。

**计划实现（暂缓至后续阶段）：**

| 文件 | 主要函数/类 | 职责 |
|------|-------------|------|
| `models.py` | `User` (SQLAlchemy) | 用户表: id, email, hashed_password, created_at |
| `database.py` | `get_db()` | Session 工厂, 数据库路径 `./db/meowsformer_auth.db` |
| `schemas.py` | `UserRegister`, `UserLogin`, `UserResponse`, `TokenResponse` | 请求/响应校验 |
| `service.py` | `hash_password()`, `verify_password()`, `create_access_token()`, `decode_token()` | 密码哈希 + JWT |
| `dependencies.py` | `get_current_user()`, `get_optional_user()` | FastAPI 依赖注入 |
| `router.py` | `POST /auth/register`, `POST /auth/login`, `GET /auth/me` | 认证端点 |

---

## 8. 模块详解 — 前端组件

### 8.1. TypeScript 类型 (`src/ui/src/types/api.ts`)

镜像后端 Pydantic 模型：
- Phase 0–3: `CatTranslationResponse`, `MeowSynthesisResponse`
- Phase 5: `TargetTagSet`, `TaggedSampleInfo`, `StreamingTranslationResult`
- WebSocket: `WSTranscriptionMessage`, `WSAnalysisPreviewMessage`, `WSResultMessage`, `WSErrorMessage`

### 8.2. Composables

| Composable | 状态 | 职责 |
|------|------|------|
| `useAudioPreview.ts` | 已实现 | base64 WAV → Blob → ObjectURL → HTMLAudioElement 生命周期；Vue 3 Composition API (ref/shallowRef/onUnmounted) |
| `useStreamingTranslation.ts` | 已实现 | WebSocket 连接管理 (状态机, 无自动重连), ScriptProcessorNode 采集 PCM 16kHz (buffer=4096, ~256ms/帧), Float32→Int16 转换, 接收分发 4 类服务端消息；Vue 3 Composition API |
| `useAuth.ts` | **未实现** | 计划: `apiRegister()`, `apiLogin()` — 调用 `/auth/*` 端点 |

### 8.3. 认证状态管理 — 未实现

计划：全局认证状态管理（Pinia store 或 provide/inject），localStorage 存储 token，页面刷新时调用 `/auth/me` 校验有效性。

### 8.4. 页面 — 未实现

| 页面 | 计划功能 |
|------|------|
| `LoginPage.vue` | email + password, 错误展示, 注册成功 banner |
| `RegisterPage.vue` | email + password + confirm, 客户端验证 |
| `TranslatePage.vue` | 模式切换 (streaming / file upload), 结果卡片 |

### 8.5. Flet 移动端原型 (`src/flet_mobile/`)

API-First 架构，所有逻辑由 FastAPI 承担：
- `app.py` — 主页面编排 (The Bridge / The Lab / The Output / The Library)；Batch 2 新增 REST/Streaming 模式切换、`on_ws_event()` 4 种消息处理、Phase 5 TargetTagSet 5 维标签展示（兼容 Legacy fallback）、`_chunk_generator()` async generator 桥接音频线程；Batch 4 新增 `ws_status_chip` WS 连接状态指示器 + `_update_ws_status()` 状态映射、`_show_snackbar()` 统一 SnackBar 错误通知 (3s 自动消失)、`recording_timer_loop()` 录音时长计时器 (wall-clock 锚定 MM:SS)、`_fallback_to_rest()` WS 失败自动降级 REST、`append_history()` 增强为 ExpansionTile 展示完整 5 维标签；已完成 Flet 0.80+ 废弃 API 迁移 (`ft.padding.symmetric` → `ft.Padding.symmetric`、`ft.border.all` → `ft.Border.all`)
- `audio_recorder.py` — `AudioRecorder` (16kHz PCM + 实时波形快照)
- `translation_client.py` — `TranslationClient`：REST 模式 (httpx → `/api/v1/translate`) + Streaming 模式 (websockets → `/ws/translate`，Batch 2 重写为 `asyncio.TaskGroup` 并发 `_sender`/`_receiver`，含 `JSONDecodeError` 容错)；Batch 4 新增 `WebSocketConnectionError` 异常类、`on_state_change` 回调参数（fire connecting/connected/disconnected）、`WS_CONNECT_TIMEOUT` 5s 连接超时检测
- `bioacoustic_player.py` — `BioacousticPlayer`：flet-audio 0.82.2 升级，使用 `fta.Audio` 原生应用内播放；`__init__` 延迟导入 `flet_audio`，创建 `fta.Audio` 并注册到 `page.overlay`；`_play_wav_bytes()`：`await release()` → `src=bytes` → `update()` → `await play()`；`play_from_base64()` base64 解码后委托 `_play_wav_bytes`；`dispose()` 从 overlay 移除控件并 `page.update()`；librosa DSP (pitch/tempo) + `_build_index()` / `_resolve_sound()` 样本检索不变
- `theme.py` — 视觉 Token (奶油底色, 琥珀色主色, 森林绿科学引用)

**Flet 启动：** 用 `flet run -m` 而非 `python -m`，否则 fta.Audio 不可用。WSL2 见 [wsl2-audio-setup.md](./wsl2-audio-setup.md)。

---

## 9. 代码运行流程

### 9.1. Legacy Pipeline (POST /api/v1/translate)

```
用户上传音频文件
      │
      ▼
endpoints.translate_v1()
      │
      ├──► audio_processor.extract_basic_features()   → {duration, rms}
      ├──► transcription_service.transcribe_audio()    → "用户说的话"
      ├──► rag_service.retrieve_context()              → 科学上下文
      ├──► llm_service.analyze_intention()             → CatTranslationResponse
      │         │                                           (emotion_category, sound_id, pitch_adjust...)
      │         ▼
      └──► synthesis_service.synthesize_and_describe()
                │
                ├── emotion→intent 映射 (Hungry→Requesting)
                ├── dsp_processor.map_intent_to_va()       → VA 坐标
                ├── dsp_processor.get_best_match()         → 最近邻样本
                ├── dsp_processor.apply_prosody_transform() → PSOLA 变换后音频
                ├── description_generator.generate_description_from_synthesis()
                └── base64 编码 → MeowSynthesisResponse
```

### 9.2. Streaming Pipeline (WS /ws/translate)

```
用户对着麦克风说话
      │
      ▼
ws_endpoints.ws_translate()
      │
      │  ┌─────────────────────── 连接建立 ───────────────────────┐
      │  │  1. websocket.accept()                                  │
      │  │  2. load_tagged_samples() — 加载 483 个带标签样本到内存  │
      │  │  3. 创建 StreamingSession (会话状态容器)                 │
      │  └─────────────────────────────────────────────────────────┘
      │
      │  ┌─────────────────────── 录音阶段 ───────────────────────┐
      │  │  每 ~256ms (buffer=4096 @ 16kHz)：                        │
      │  │    Client → Binary frame (PCM 16-bit 16kHz)             │
      │  │    → StreamingTranscriptionSession.add_chunk()          │
      │  │                                                         │
      │  │  每 ~2.5s (缓冲区达到阈值)：                             │
      │  │    → transcribe_intermediate() → Whisper API            │
      │  │    → Server → {"type":"transcription", "is_final":false}│
      │  │                                                         │
      │  │  文本达到 5 词以上 (首次)：                               │
      │  │    → 异步启动 speculative LLM 分析                       │
      │  │    → generate_target_tags() → SpeculativeCache.store()  │
      │  │    → Server → {"type":"analysis_preview"}               │
      │  └─────────────────────────────────────────────────────────┘
      │
      │  ┌─────────────────── 停止 & 出结果 ──────────────────────┐
      │  │  Client → {"type": "stop"}                              │
      │  │                                                         │
      │  │  1. 等待推测性 LLM 任务完成 (最多 5s)                    │
      │  │  2. transcribe_final() → Whisper API 最终转录            │
      │  │  3. Server → {"type":"transcription", "is_final":true}  │
      │  │                                                         │
      │  │  4. 判断是否复用缓存：                                    │
      │  │     cache.is_similar(final_text) → ratio ≥ 0.7?         │
      │  │       → cache.get() 直接复用 target_tags (零延迟)        │
      │  │     else → generate_target_tags(final_text) (新调用)     │
      │  │                                                         │
      │  │  5. sample_matcher.find_best_match(target_tags)         │
      │  │     → 遍历 483 样本, 加权 Jaccard 评分, 选出最高分       │
      │  │                                                         │
      │  │  6. 读取 WAV → base64 编码                               │
      │  │  7. Server → {"type":"result", audio_base64, ...}       │
      │  └─────────────────────────────────────────────────────────┘
```

**关键区别：** Phase 5 管线直接播放真实录音，不做 DSP 合成 / PSOLA 变换。

### 9.3. 端到端选择流程

LLM 返回目标标签后：

1. `sample_matcher.find_best_match()` 对全部 483 样本评分
2. 应用品种偏好 boost (如有)
3. 选出最高分样本
4. 读取对应 WAV (`assets/raw_data/catmeows/dataset/*.wav`)
5. base64 编码
6. 返回 `WSResultMessage` (含匹配标签、分数、推理说明)

---

## 10. 更新规则

每次 developer 或 test-engineer 完成审核循环后，由 PM 同步更新本文档：
- 新增 / 修改 / 删除的文件及其主要函数
- 变更的调用链或数据流
- 新增的算法或协议细节
