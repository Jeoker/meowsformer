# Meowsformer - Backend Project Documentation

## 1. Project Overview

Meowsformer is a FastAPI-based backend service that translates human speech into realistic cat vocalisations. The system supports two parallel pipelines:

1. **Legacy pipeline (Phase 0–3):** File upload → Whisper → RAG → LLM emotion analysis → DSP (VA mapping + PSOLA) → synthesised audio.
2. **Streaming pipeline (Phase 5):** Real-time WebSocket audio → chunked Whisper → LLM target-tag generation → multi-dimensional catalog matching → real recording playback.

**Current Status:** Phase 0–3 complete (legacy pipeline). Phase 5 (Streaming + Catalog Redesign) complete. All 140 legacy unit tests passing. Streaming WebSocket endpoint verified end-to-end.

### Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Core API — FastAPI endpoints, Whisper transcription, LLM analysis, RAG | Done |
| **Phase 1** | Data Acquisition — Zenodo corpus download, metadata parsing, registry index | Done |
| **Phase 2** | DSP Engine — VA mapping, audio retrieval, PSOLA prosody transform | Done |
| **Phase 3** | Integration — Wire DSP engine into API pipeline, end-to-end flow, UI preview | Done |
| **Phase 4** | Deployment — Dockerise, CI/CD, production hardening | Pending |
| **Phase 5** | Streaming + Catalog Redesign — WebSocket streaming, multi-dimensional tag system, LLM target-tag generation, weighted sample matching | Done |

---

## 2. Tech Stack & Dependencies

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.10+ (recommended 3.12) |
| **Web Framework** | FastAPI + Uvicorn |
| **Data Validation** | Pydantic V2 |
| **AI / LLM** | OpenAI API (GPT-4o, Whisper V3), `instructor` (structured outputs) |
| **Vector Database** | ChromaDB (Phase 0 RAG, local persistent storage) |
| **Audio Processing** | FFmpeg (via subprocess), `python-multipart` |
| **Audio DSP Engine** | `librosa` (f0/pYIN, audio I/O), `pytsmod` (WSOLA), `soundfile`, `scipy`, `numpy` |
| **Audio Feature Extraction** | `librosa` (pYIN f0, RMS energy, duration — used by tag builder) |
| **WebSocket Transport** | `websockets` (Starlette/FastAPI built-in WS support) |
| **Data Acquisition** | `zenodo-get` (Zenodo corpus download) |
| **Environment Management** | `python-dotenv`, `pydantic-settings` |
| **Logging** | `loguru` |
| **Testing** | `unittest` |
| **Frontend** | React + TypeScript + Vite (development preview) |

---

## 3. Directory Structure

```text
/
├── app/                                  # 核心后端应用
│   ├── __init__.py
│   ├── api/                              # API 路由层
│   │   ├── __init__.py
│   │   ├── endpoints.py                  # REST API 端点 (POST /api/translate, POST /api/v1/translate)
│   │   └── ws_endpoints.py              ★ WebSocket 流式端点 (WS /ws/translate) — Phase 5
│   ├── core/                             # 核心配置
│   │   └── config.py                     # Pydantic Settings 管理 (.env 读取)
│   ├── data/                            ★ 数据定义层 — Phase 5
│   │   ├── __init__.py
│   │   └── meow_catalog.py             ★ 5维标签分类体系 + 规则引擎 (tag_emotion, tag_intent, tag_acoustic 等)
│   ├── db/                               # 数据库层
│   │   └── vector_store.py               # ChromaDB 客户端初始化 (RAG 向量存储)
│   ├── schemas/                          # Pydantic 数据模型
│   │   ├── translation.py                # Phase 0–3 响应模型 (CatTranslationResponse, MeowSynthesisResponse)
│   │   └── ws_messages.py              ★ WebSocket 消息协议模型 — Phase 5
│   │                                      (TargetTagSet, TaggedSampleInfo, StreamingTranslationResult,
│   │                                       WSConfigMessage, WSStopMessage, WSTranscriptionMessage,
│   │                                       WSAnalysisPreviewMessage, WSResultMessage, WSErrorMessage)
│   └── services/                         # 业务逻辑服务层
│       ├── audio_processor.py            # FFmpeg 音频转换/特征提取 (Phase 0)
│       ├── llm_service.py                # OpenAI GPT-4o 意图分析 (Phase 0, instructor 结构化输出)
│       ├── rag_service.py                # ChromaDB 知识检索 (Phase 0 RAG)
│       ├── transcription_service.py      # OpenAI Whisper 文件转录 (Phase 0)
│       ├── synthesis_service.py          # DSP 合成桥接服务 (Phase 3, emotion→intent→VA→PSOLA)
│       ├── streaming_transcription_service.py ★ 流式 Whisper 转录 — Phase 5
│       │                                      (音频缓冲区管理, 增量/最终转录)
│       ├── sound_selection_service.py   ★ LLM 目标标签生成 + 推测性执行 — Phase 5
│       │                                      (generate_target_tags, SpeculativeCache, select_and_encode)
│       └── sample_matcher.py            ★ 多维加权标签匹配引擎 — Phase 5
│                                              (score_sample, find_best_match, Jaccard 相似度)
│
├── src/                                  # DSP 引擎 & 前端
│   ├── __init__.py
│   ├── engine/                           # Phase 2 — DSP 音频合成引擎
│   │   ├── __init__.py
│   │   ├── dsp_processor.py              # VA 映射、最近邻检索、PSOLA 韵律变换
│   │   └── description_generator.py      # NatureLM-audio 风格置信描述生成 (Phase 3)
│   └── ui/                               # 前端 React 预览组件
│       ├── vite.config.ts                # Vite 开发服务器配置 (代理 /api + /ws 到 FastAPI)
│       └── src/
│           ├── types/
│           │   └── api.ts                # TypeScript 类型定义 (镜像后端 Pydantic 模型, 含 Phase 5 WS 类型)
│           ├── hooks/
│           │   ├── useAudioPreview.ts     # 音频预览播放 Hook (base64→Blob→ObjectURL)
│           │   └── useStreamingTranslation.ts ★ WebSocket 流式翻译 Hook — Phase 5
│           │                                    (连接管理, PCM 音频流, 渐进式结果)
│           └── components/
│               ├── MeowPreviewPlayer.tsx  # 预览播放器 + 确认 UI (Phase 3)
│               ├── MeowPreviewPlayer.css  # 播放器样式
│               └── AudioRecorder.tsx     ★ 实时录音组件 — Phase 5
│                                              (麦克风录制, 实时转录显示, 结果播放)
│
├── tools/                                # 工具脚本
│   ├── __init__.py
│   ├── download_datasets.py              # Zenodo 数据集下载/解析/registry 构建 (Phase 1)
│   ├── build_tags.py                    ★ 一次性标签构建脚本 — Phase 5
│   │                                        (registry.json + librosa 特征提取 → tagged_samples.json)
│   └── play_audio.py                     # 音频试听调试工具
│
├── assets/                               # 静态资源
│   ├── audio_db/
│   │   ├── registry.json                 # 元数据索引 (483 CatMeows 样本, VA 标注)
│   │   └── tagged_samples.json          ★ 带多维标签的样本索引 — Phase 5 (由 build_tags.py 生成)
│   └── raw_data/                         # 已下载的音频语料库 (git-ignored)
│       ├── catmeows/                     #   CatMeows dataset (Zenodo 10.5281/zenodo.4007940)
│       └── meowsic/                      #   Meowsic dataset  (Zenodo 10.5281/zenodo.3245999)
│
├── tests/                                # 单元 & 集成测试
│   ├── __init__.py
│   ├── test_api_endpoints.py             # API 端点测试 (POST /translate, mock services)
│   ├── test_audio_services.py            # 音频服务测试 (FFmpeg)
│   ├── test_download_datasets.py         # 数据获取管线测试
│   ├── test_dsp_processor.py             # DSP 引擎测试 (45 cases)
│   ├── test_description_generator.py     # 描述生成器测试 (31 cases)
│   ├── test_synthesis_service.py         # 合成集成测试 (15 cases)
│   ├── test_llm_service.py               # LLM 服务测试
│   └── test_rag_service.py               # RAG 服务测试
│
├── scripts/
│   └── run_e2e_test.sh                   # 端到端测试脚本
├── .env                                  # 环境变量 (不提交)
├── .gitignore
├── LICENSE
├── main.py                               # 应用入口 (FastAPI app 创建, 路由注册, 启动事件)
├── requirements.txt                      # Python 依赖
└── PROJECT_STATUS.md                     # 本文件
```

> 标注 ★ 的文件为 Phase 5 (Streaming + Catalog Redesign) 新增文件。

---

## 4. 系统架构与 API 调用关系

### 4.1. 总览：两条并行管线

```
┌──────────────────────────────────────────────────────────────────┐
│                    Meowsformer 系统架构                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─ Legacy Pipeline (REST, Phase 0–3) ─────────────────────────┐ │
│  │  POST /api/translate  →  Whisper → RAG → LLM → JSON        │ │
│  │  POST /api/v1/translate →  ... + DSP(PSOLA) → base64 WAV   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ Streaming Pipeline (WebSocket, Phase 5) ───────────────────┐ │
│  │  WS /ws/translate → chunked Whisper → LLM target tags      │ │
│  │                   → catalog matching → real WAV playback    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ Shared Infrastructure ─────────────────────────────────────┐ │
│  │  OpenAI API (Whisper + GPT-4o)  │  ChromaDB  │  librosa    │ │
│  │  483 CatMeows samples  │  tagged_samples.json (5维标签)     │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2. Legacy Pipeline 调用链

```
用户上传音频文件
      │
      ▼
POST /api/v1/translate (app/api/endpoints.py)
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
                ├── dsp_processor.map_intent_to_va()       → VA坐标
                ├── dsp_processor.get_best_match()         → 最近邻样本
                ├── dsp_processor.apply_prosody_transform() → PSOLA 变换后音频
                ├── description_generator.generate_description_from_synthesis()
                └── base64 编码 → MeowSynthesisResponse
```

### 4.3. Streaming Pipeline 调用链 (Phase 5)

```
用户对着麦克风说话
      │
      ▼
WS /ws/translate (app/api/ws_endpoints.py)
      │
      │  ┌─────────────────────── 连接建立 ───────────────────────┐
      │  │  1. websocket.accept()                                  │
      │  │  2. load_tagged_samples() — 加载 483 个带标签样本到内存  │
      │  │  3. 创建 StreamingSession (会话状态容器)                 │
      │  └─────────────────────────────────────────────────────────┘
      │
      │  ┌─────────────────────── 录音阶段 ───────────────────────┐
      │  │  每 ~200ms：                                             │
      │  │    Client → Binary frame (PCM 16-bit 16kHz)             │
      │  │    → StreamingTranscriptionSession.add_chunk()          │
      │  │                                                         │
      │  │  每 ~2.5s (缓冲区达到阈值)：                             │
      │  │    → transcribe_intermediate() → Whisper API            │
      │  │    → Server → {"type":"transcription", "is_final":false}│
      │  │                                                         │
      │  │  文本达到 5 词以上 (首次)：                               │
      │  │    → 异步启动 speculative LLM 分析                       │
      │  │    → sound_selection_service.generate_target_tags()     │
      │  │    → SpeculativeCache.store(text, tags)                 │
      │  │    → Server → {"type":"analysis_preview"}               │
      │  └─────────────────────────────────────────────────────────┘
      │
      │  ┌─────────────────── 停止 & 出结果 ──────────────────────┐
      │  │  Client → {"type": "stop"}                              │
      │  │                                                         │
      │  │  1. 等待推测性 LLM 任务完成 (最多5s)                     │
      │  │  2. transcribe_final() → Whisper API 最终转录            │
      │  │  3. Server → {"type":"transcription", "is_final":true}  │
      │  │                                                         │
      │  │  4. 判断是否复用缓存：                                    │
      │  │     if SequenceMatcher(cached_text, final_text) ≥ 0.7:  │
      │  │       → 直接复用 cached target_tags (零延迟)             │
      │  │     else:                                                │
      │  │       → generate_target_tags(final_text) (新 LLM 调用)  │
      │  │                                                         │
      │  │  5. sample_matcher.find_best_match(target_tags)         │
      │  │     → 遍历 483 样本, 加权 Jaccard 评分                   │
      │  │     → 选出最高分样本                                      │
      │  │                                                         │
      │  │  6. 读取 WAV → base64 编码                               │
      │  │  7. Server → {"type":"result", audio_base64, ...}       │
      │  └─────────────────────────────────────────────────────────┘
```

### 4.4. 服务间依赖关系图

```
main.py
  ├── app/api/endpoints.py (REST 路由)
  │     ├── app/services/audio_processor.py
  │     ├── app/services/transcription_service.py
  │     │     └── OpenAI Whisper API
  │     ├── app/services/rag_service.py
  │     │     └── app/db/vector_store.py → ChromaDB
  │     ├── app/services/llm_service.py
  │     │     └── OpenAI GPT-4o (via instructor)
  │     └── app/services/synthesis_service.py
  │           ├── src/engine/dsp_processor.py
  │           │     └── assets/audio_db/registry.json
  │           └── src/engine/description_generator.py
  │
  └── app/api/ws_endpoints.py (WebSocket 路由)
        ├── app/services/streaming_transcription_service.py
        │     └── OpenAI Whisper API
        ├── app/services/sound_selection_service.py
        │     ├── OpenAI GPT-4o (via instructor)
        │     ├── app/data/meow_catalog.py (标签分类体系)
        │     └── app/services/sample_matcher.py
        │           └── assets/audio_db/tagged_samples.json
        └── app/schemas/ws_messages.py (协议定义)
```

---

## 5. 核心模块与算法详解

### 5.1. 多维标签分类体系 (`app/data/meow_catalog.py`)

定义了 5 个独立维度的标签词汇表和基于规则的标签分配逻辑。每个样本可在多个维度上同时携带多个标签，从而实现细粒度匹配。

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

#### 维度 3 — acoustic (声学特征，由 librosa 提取)

| 标签 | 分配规则 |
|------|---------|
| `high_pitch` | 中位 f0 > 600 Hz |
| `low_pitch` | 中位 f0 < 400 Hz |
| `mid_pitch` | 400 Hz ≤ 中位 f0 ≤ 600 Hz |
| `short_burst` | 时长 < 0.5s |
| `medium_length` | 0.5s ≤ 时长 ≤ 1.5s |
| `prolonged` | 时长 > 1.5s |
| `loud` | RMS 能量 > 第 75 百分位 (全局计算) |
| `soft` | RMS 能量 < 第 25 百分位 (全局计算) |
| `rising_tone` | f0 斜率 > 0 (线性回归) |
| `falling_tone` | f0 斜率 < 0 |
| `trembling` | f0 标准差 > 80 Hz |

**声学特征提取流程** (`tools/build_tags.py`):
1. `librosa.load()` 加载 WAV
2. `librosa.pyin()` 估算基频 f0 (fmin=60Hz, fmax=1500Hz)
3. 计算 voiced f0 的 median、std、linear slope
4. `np.sqrt(np.mean(y**2))` 计算 RMS 能量
5. 所有样本 RMS 排序后计算 P25/P75 百分位线，分配 high/low/mid

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

### 5.2. 标签构建管线 (`tools/build_tags.py`)

一次性运行的脚本，读取 `registry.json`，对全部 483 个样本执行：

```
registry.json (483 samples)
      │
      ├──► 维度 1/2/4/5: 基于 context/VA/breed 的规则标签
      │      (tag_emotion, tag_intent, tag_social_context, tag_breed_voice)
      │
      ├──► 维度 3: librosa 声学特征提取
      │      pYIN f0 → median_f0, f0_std, f0_slope
      │      RMS energy → 全局百分位排名
      │      duration → 直接从采样点计算
      │      → tag_acoustic()
      │
      └──► 输出 tagged_samples.json
             483 个样本，平均每个样本 12.1 个标签
```

**用法:**

```bash
python -m tools.build_tags                # 完整运行 (含 librosa 声学特征提取，约 2 分钟)
python -m tools.build_tags --skip-audio   # 仅元数据标签 (跳过声学特征，秒级完成)
```

### 5.3. 加权标签匹配引擎 (`app/services/sample_matcher.py`)

#### 核心算法：加权 Jaccard 相似度

对每个维度独立计算 Jaccard 系数，然后按维度权重加权求和：

```
score(target, sample) = Σ_dim  weight[dim] × |target[dim] ∩ sample[dim]|
                                              ─────────────────────────────
                                              |target[dim] ∪ sample[dim]|
```

#### 维度权重 (可调参)

| 维度 | 权重 | 理由 |
|------|------|------|
| emotion | 0.30 | 情绪匹配最重要，决定声音的主观感受 |
| intent | 0.30 | 沟通意图与情绪同等重要 |
| acoustic | 0.15 | 声学特征为辅助匹配 |
| social_context | 0.15 | 场景匹配确保语义合理 |
| breed_voice | 0.10 | 品种声线为次要偏好 |

#### 品种偏好提升

如果用户指定了 `breed_preference`，匹配该品种的样本额外加 0.05 分。

#### 匹配流程

1. 遍历内存中全部 483 个 `TaggedSample`
2. 对每个样本计算 5 维加权 Jaccard 分数
3. 可选：应用品种偏好 boost
4. 按分数降序排序，返回 top-K

### 5.4. LLM 目标标签生成 (`app/services/sound_selection_service.py`)

#### 核心设计

**LLM 不选择具体样本。** 它只输出一组「目标标签」(TargetTagSet)，描述理想猫叫声应该具有的特征。具体样本由匹配引擎确定性选出。

#### LLM 系统提示词结构

```
你是一位猫咪生物声学专家和情感分析师。
分析用户说的话，判断猫应发出什么样的声音回应。
输出目标标签，分为 5 个维度：
  emotion: [有效标签列表]
  intent: [有效标签列表]
  acoustic: [有效标签列表]
  social_context: [有效标签列表]
  breed_voice: [有效标签列表]
```

完整标签词汇表注入提示词，确保 LLM 只在有效范围内选择。

#### TargetTagSet 输出示例

```json
{
  "emotion": ["lonely", "anxious"],
  "intent": ["seeking_companionship"],
  "acoustic": ["prolonged", "soft", "falling_tone"],
  "social_context": ["alone_at_home", "separation"],
  "reasoning": "用户表达了对猫咪的思念，猫咪应以孤独渴望陪伴的方式回应"
}
```

#### 推测性执行 (Speculative Execution)

```
            ┌─── 部分转录 (≥5 词) ───┐
            │                         ▼
  录音中... │    异步 LLM 调用 → cache(text₁, tags₁)
            │
            └─── 用户停止 ──────────► 最终转录 text₂
                                        │
                                   SequenceMatcher(text₁, text₂)
                                        │
                              ┌─── ratio ≥ 0.7 ────┐─── ratio < 0.7 ────┐
                              │                     │                     │
                        复用 tags₁ (零延迟)     新 LLM 调用 → tags₂
```

**延迟节省:** 在多数场景下，用户在最后 2-3 秒说的话与之前相似，LLM 结果可直接复用，省去 2-3 秒 LLM 延迟。

### 5.5. 流式转录服务 (`app/services/streaming_transcription_service.py`)

#### 设计

- 维护一个持续增长的 PCM 音频缓冲区
- 每 ~2.5 秒 (且缓冲区 ≥ 32KB ≈ 1 秒音频) 向 Whisper 发送完整累积音频
- Whisper 每次重新处理完整缓冲区 (对 <60s 语句可接受)
- `stop` 信号触发最终转录

#### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `MIN_TRANSCRIPTION_INTERVAL` | 2.5s | 中间转录最小间隔 |
| `MIN_BUFFER_SIZE` | 32000 bytes | 约 1 秒 16kHz 16-bit 单声道 |
| 采样率 | 16000 Hz | PCM 16-bit mono |

#### 缓冲区 → WAV 流程

```
[chunk₁] + [chunk₂] + ... → bytes 拼接
→ np.frombuffer(dtype=int16) → float32 / 32768.0
→ soundfile.write(tmp.wav, PCM_16)
→ OpenAI Whisper API
→ 文本
```

### 5.6. WebSocket 端点 (`app/api/ws_endpoints.py`)

#### 协议定义

**Client → Server:**

| 消息类型 | 格式 | 说明 |
|----------|------|------|
| `config` | JSON `{"type":"config","breed_preference":"Maine Coon"}` | 连接配置 (可选) |
| 音频块 | Binary (PCM 16-bit 16kHz) | 原始音频数据 |
| `stop` | JSON `{"type":"stop"}` | 用户停止录音 |

**Server → Client:**

| 消息类型 | 触发时机 | 字段 |
|----------|---------|------|
| `transcription` | 每 ~2.5s 及最终 | `text`, `is_final` |
| `analysis_preview` | 推测性 LLM 完成 | `emotion`, `intent` |
| `result` | 最终结果 | `transcription`, `selected_category`, `audio_base64`, `reasoning` |
| `error` | 任何错误 | `detail` |

#### 会话状态 (StreamingSession)

每个 WebSocket 连接维护独立的 `StreamingSession` 对象：
- `breed_preference` — 用户品种偏好
- `StreamingTranscriptionSession` — 音频缓冲区 + 转录状态
- `SpeculativeCache` — 推测性 LLM 结果缓存
- `_speculative_task` — 异步 LLM 任务句柄

### 5.7. 端到端选择流程

当 LLM 返回目标标签后：

1. `sample_matcher.find_best_match()` 对全部 483 样本评分
2. 应用品种偏好 boost (如有)
3. 选出最高分样本
4. 读取对应 WAV 文件 (`assets/raw_data/catmeows/dataset/*.wav`)
5. base64 编码
6. 返回 `WSResultMessage` (含匹配标签、分数、推理说明)

**关键区别：** Phase 5 管线直接播放真实录音，不做 DSP 合成/PSOLA 变换。

---

## 6. Legacy Pipeline 模块详解 (Phase 0–3)

### 6.1. 音频处理 (`app/services/audio_processor.py`)
- **`convert_to_wav(input, output)`**: 异步调用 FFmpeg 转换为 16kHz 单声道 WAV。
- **`extract_basic_features(path)`**: 提取 `duration_seconds` 和 `rms_amplitude`。

### 6.2. 转录 (`app/services/transcription_service.py`)
- **`transcribe_audio(path)`**: 文件→WAV 转换→OpenAI Whisper API→文本。

### 6.3. RAG (`app/services/rag_service.py`)
- **`initialize_knowledge_base()`**: 向 ChromaDB 填充猫声学科学文献。
- **`retrieve_context(query, n=3)`**: 检索 top-3 相关上下文。

### 6.4. LLM 分析 (`app/services/llm_service.py`)
- **`analyze_intention(text, features, rag_ctx)`**: 拼接提示词→GPT-4o→`CatTranslationResponse`。

### 6.5. DSP 引擎 (`src/engine/dsp_processor.py`)

#### Intent → VA 映射 (Russell 情绪环状模型)

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

#### PSOLA 韵律变换流程
1. pYIN f0 估计
2. 品种基频混合 (8 品种基线，50% blend)
3. Arousal→时长调制 (高 arousal 压缩，低 arousal 拉伸)
4. WSOLA 时域拉伸 (pytsmod)
5. 重采样音高偏移
6. Arousal 包络整形
7. 峰值归一化 (0.95)

### 6.6. 合成桥接 (`app/services/synthesis_service.py`)
- **Emotion→Intent 映射:** Hungry→Requesting, Angry→Agonistic, Happy→Affiliative, Alert→Alert
- **完整流程:** emotion→intent→VA→最近邻→PSOLA→base64 WAV→NatureLM 描述→`MeowSynthesisResponse`

### 6.7. 描述生成器 (`src/engine/description_generator.py`)
- Intent→中文语义标签
- VA 距离→指数衰减置信分数 `exp(-d)`→五级中文评价
- 拼装结构化中文描述

---

## 7. 前端组件

### 7.1. TypeScript 类型 (`src/ui/src/types/api.ts`)
镜像后端 Pydantic 模型，包含：
- Phase 0–3 响应类型 (`CatTranslationResponse`, `MeowSynthesisResponse`)
- Phase 5 流式类型 (`TargetTagSet`, `TaggedSampleInfo`, `StreamingTranslationResult`)
- WebSocket 消息类型 (`WSTranscriptionMessage`, `WSAnalysisPreviewMessage`, `WSResultMessage`, `WSErrorMessage`)

### 7.2. 音频预览 Hook (`useAudioPreview.ts`)
base64 WAV→Blob→ObjectURL→HTMLAudioElement 生命周期管理。

### 7.3. 流式翻译 Hook (`useStreamingTranslation.ts`)
- WebSocket 连接管理 (自动重连、状态机)
- `MediaRecorder` / `ScriptProcessorNode` 采集 PCM 16kHz
- Float32→Int16 转换后发送二进制帧
- 接收并分发 4 类服务端消息

### 7.4. 预览播放器 (`MeowPreviewPlayer.tsx`)
Phase 3 播放器：播放/暂停、进度条、置信描述、先听后发。

### 7.5. 录音组件 (`AudioRecorder.tsx`)
Phase 5 实时录音 UI：品种选择、状态指示灯、实时转录、分析预览、结果展示、音频播放。

### 7.6. Flet 移动端原型 (`src/flet_mobile/`)
新增 API-First 的 Python/Flet 移动 UI 骨架，逻辑仍完全由 FastAPI 承担：
- `app.py`: 主页面编排（The Bridge / The Lab / The Output / The Library）
- `audio_recorder.py`: `AudioRecorder`（16kHz PCM 采集 + 实时波形快照）
- `translation_client.py`: `TranslationClient`（`httpx` 调用 `/api/v1/translate`）
- `bioacoustic_player.py`: `BioacousticPlayer`（`sound_id` 映射本地样本 + pitch/tempo DSP 调整）
- `theme.py`: 视觉 Token（奶油底色、琥珀色主色、森林绿科学引用）

---

## 8. Setup & Running

### Prerequisites
- Python 3.10+ (推荐 3.12)
- FFmpeg (系统 PATH)
- OpenAI API Key

### Installation

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-key-here
CHROMA_DB_PATH=./db/chroma_db
DEBUG_MODE=True
EOF

# 4. 下载音频语料库
python -m tools.download_datasets
# 或仅索引 (已有音频):
python -m tools.download_datasets --skip-download

# 5. 构建多维标签索引 (Phase 5 必须)
python -m tools.build_tags
# 或跳过声学特征 (快速):
python -m tools.build_tags --skip-audio
```

### Running the Server

```bash
python main.py
# 或:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Running Flet Mobile UI (Prototype)

```bash
python -m src.flet_mobile.app
```

**可用端点:**
- REST API 文档: `http://localhost:8000/docs`
- 健康检查: `GET http://localhost:8000/health`
- Legacy 翻译: `POST http://localhost:8000/api/translate`
- Legacy 合成: `POST http://localhost:8000/api/v1/translate`
- **流式翻译: `WS ws://localhost:8000/ws/translate`**

---

## 9. Testing

```bash
export PYTHONPATH=$PYTHONPATH:.
python -m unittest discover tests
```

### Test Summary

| Test File | Module | Cases | Description |
|-----------|--------|-------|-------------|
| `test_api_endpoints.py` | API | — | `POST /translate` with mocked services |
| `test_audio_services.py` | Audio | — | `extract_basic_features`, `convert_to_wav` |
| `test_llm_service.py` | LLM | — | `analyze_intention` with mocked OpenAI |
| `test_rag_service.py` | RAG | — | `initialize_knowledge_base`, `retrieve_context` |
| `test_download_datasets.py` | Data | — | Filename parsing, registry building |
| `test_dsp_processor.py` | DSP | 45 | VA mapping, audio retrieval, f0, PSOLA, envelope |
| `test_description_generator.py` | Descriptions | 31 | Intent labels, confidence scoring, preview generation |
| `test_synthesis_service.py` | Synthesis | 15 | Emotion→intent, base64, schema, pipeline, degradation |

**Total: 140 tests, all passing.**

**Note on Compatibility:** 推荐使用 Python 3.12。Python 3.14 与 `chromadb` 的 Pydantic V1 依赖存在兼容性问题。

---

## 10. Next Steps

- **Phase 5 补充:** 为 Phase 5 新模块编写单元测试 (sample_matcher, streaming_transcription, sound_selection, ws_endpoints)。
- **Phase 4 — Deployment:** Dockerise (确保 FFmpeg + librosa 在 Dockerfile 中)。
- **Phase 4 — Frontend Build:** 完善 `src/ui/` React 应用 (npm install, 路由, 完整 UI)。
- **Phase 4 — Database:** 翻译历史持久化 (SQLite/PostgreSQL)。
- **Future — 高级功能:** 用户账号, 反馈循环 (RLHF), 多语言支持, 标签权重动态调优。
