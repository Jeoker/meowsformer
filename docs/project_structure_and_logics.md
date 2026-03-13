# Meowformer — 项目结构与代码逻辑

本文档详细记录每个文件、主要 functions 的作用及代码运行流程。  
宏观设计、进度与计划见 [`docs/PROJECT_STATUS.md`](./PROJECT_STATUS.md)。

---

## 1. 完整目录结构

```text
/
├── app/                                  # 核心后端应用
│   ├── __init__.py
│   ├── api/                              # API 路由层
│   │   ├── __init__.py
│   │   ├── endpoints.py                  # REST API 端点 (POST /api/translate, POST /api/v1/translate)
│   │   └── ws_endpoints.py              ★ WebSocket 流式端点 (WS /ws/translate) — Phase 5
│   ├── auth/                            ★ 认证模块 — Phase 6
│   │   ├── __init__.py
│   │   ├── models.py                     # SQLAlchemy User 模型 (id, email, hashed_password, created_at)
│   │   ├── database.py                   # SQLAlchemy engine + session factory (SQLite: ./db/meowsformer_auth.db)
│   │   ├── schemas.py                    # Pydantic V2 schemas (UserRegister, UserLogin, UserResponse, TokenResponse)
│   │   ├── service.py                    # bcrypt 密码哈希 (passlib) + JWT 签发/解码 (python-jose)
│   │   ├── dependencies.py              # get_current_user (强制认证) + get_optional_user (可选认证)
│   │   └── router.py                     # /auth/register, /auth/login, /auth/me
│   ├── core/                             # 核心配置
│   │   └── config.py                     # Pydantic Settings (.env 读取, JWT 配置项)
│   ├── data/                            ★ 数据定义层 — Phase 5
│   │   ├── __init__.py
│   │   └── meow_catalog.py             ★ 5 维标签分类体系 + 规则引擎
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
│   │   ├── app.py                        # 主页面编排
│   │   ├── audio_recorder.py             # 16kHz PCM 采集 + 波形快照
│   │   ├── translation_client.py         # httpx → /api/v1/translate
│   │   ├── bioacoustic_player.py         # sound_id 映射本地样本
│   │   └── theme.py                      # 视觉 Token
│   └── ui/                               # 前端 React SPA
│       ├── index.html
│       ├── package.json                  # React 18, React Router v6, Tailwind CSS
│       ├── vite.config.ts                # Vite 开发服务器 (代理 /api + /ws 到 FastAPI :8000)
│       ├── postcss.config.js             # PostCSS + Tailwind 插件
│       ├── tailwind.config.js            # 自定义 meow 色板
│       ├── tsconfig.json, tsconfig.node.json
│       └── src/
│           ├── main.tsx                  # React 入口
│           ├── App.tsx                   # React Router v6 + AuthProvider
│           ├── index.css                 # Tailwind base styles
│           ├── context/
│           │   └── AuthContext.tsx        # 全局认证状态, localStorage 持久化, /auth/me 校验
│           ├── hooks/
│           │   ├── useAuth.ts            # apiRegister, apiLogin helpers
│           │   ├── useAudioPreview.ts    # base64→Blob→ObjectURL 播放
│           │   └── useStreamingTranslation.ts ★ WebSocket 流式翻译 Hook (Phase 5)
│           ├── types/
│           │   └── api.ts                # TypeScript 类型 (镜像后端 Pydantic 模型)
│           ├── components/
│           │   ├── auth/
│           │   │   ├── LoginForm.tsx
│           │   │   └── RegisterForm.tsx
│           │   ├── layout/
│           │   │   ├── Navbar.tsx         # 响应式导航, 登录/登出状态
│           │   │   └── ProtectedRoute.tsx # 未认证 → /login 重定向
│           │   └── translate/
│           │       ├── ResultCard.tsx     # 翻译结果展示 (转录, 标签, 音频播放)
│           │       ├── AudioRecorder.tsx ★ 实时录音 (Phase 5, 品种选择, 实时转录)
│           │       ├── MeowPreviewPlayer.tsx  # Phase 3 预览播放器
│           │       └── MeowPreviewPlayer.css
│           └── pages/
│               ├── LoginPage.tsx
│               ├── RegisterPage.tsx
│               └── TranslatePage.tsx      # 模式切换 (streaming / file upload)
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
├── tests/                                # 单元 & 集成测试
│   ├── __init__.py
│   ├── test_api_endpoints.py
│   ├── test_audio_services.py
│   ├── test_download_datasets.py
│   ├── test_dsp_processor.py             # 45 cases
│   ├── test_description_generator.py     # 31 cases
│   ├── test_synthesis_service.py         # 15 cases
│   ├── test_llm_service.py
│   ├── test_rag_service.py
│   └── test_auth.py                     ★ 10 cases (Phase 6)
│
├── docs/                                 # 项目文档
│   ├── PROJECT_STATUS.md                 # 宏观设计、进度、计划
│   └── project_structure_and_logics.md   # 本文件
│
├── scripts/
│   └── run_e2e_test.sh
├── main.py                               # FastAPI app 创建, 路由注册, startup 事件
├── requirements.txt
├── .env                                  # 环境变量 (不提交)
├── .gitignore
└── LICENSE
```

> 标注 ★ 的文件为 Phase 5 或 Phase 6 新增。

---

## 2. 服务间依赖关系

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
  ├── app/api/ws_endpoints.py (WebSocket 路由)
  │     ├── app/services/streaming_transcription_service.py
  │     │     └── OpenAI Whisper API
  │     ├── app/services/sound_selection_service.py
  │     │     ├── OpenAI GPT-4o (via instructor)
  │     │     ├── app/data/meow_catalog.py (标签词汇表)
  │     │     └── app/services/sample_matcher.py
  │     │           └── assets/audio_db/tagged_samples.json
  │     └── app/schemas/ws_messages.py (协议定义)
  │
  └── app/auth/router.py (认证路由)
        ├── app/auth/service.py (JWT + bcrypt)
        ├── app/auth/dependencies.py (get_current_user)
        └── app/auth/database.py → SQLite
```

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
- `score_sample(target, sample)` — 单样本 5 维评分
- `find_best_match(target_tags, samples, top_k, breed_preference)` — 遍历 483 样本取 top-K

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

多数场景可省去 2–3 秒 LLM 延迟。

**主要函数：**
- `generate_target_tags(text)` — GPT-4o → TargetTagSet
- `SpeculativeCache.store(text, tags)` / `.get(text, final_text)` — 缓存管理
- `select_and_encode(target_tags, breed_preference)` — 匹配 + 读取 WAV + base64

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
- `StreamingTranscriptionSession.add_chunk(bytes)` — 累积 PCM
- `transcribe_intermediate()` — 中间转录 (每 ~2.5s)
- `transcribe_final()` — 最终转录 (用户停止时)

---

## 5. 模块详解 — WebSocket 端点

### `app/api/ws_endpoints.py`

#### 协议定义

**Client → Server:**

| 消息类型 | 格式 | 说明 |
|----------|------|------|
| `config` | JSON `{"type":"config","breed_preference":"Maine Coon"}` | 连接配置 (可选) |
| 音频块 | Binary (PCM 16-bit 16kHz) | 每 ~200ms 一帧 |
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

**主要函数：** `websocket_translate(websocket)` — 主处理循环

---

## 6. 模块详解 — Legacy Pipeline (Phase 0–3)

### 6.1. 音频处理 (`app/services/audio_processor.py`)

- `convert_to_wav(input, output)` — 异步 FFmpeg → 16kHz mono WAV
- `extract_basic_features(path)` — 提取 `duration_seconds`, `rms_amplitude`

### 6.2. 转录 (`app/services/transcription_service.py`)

- `transcribe_audio(path)` — 文件 → WAV → OpenAI Whisper API → 文本

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

## 7. 模块详解 — 认证 (Phase 6)

### `app/auth/`

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

### 8.2. Hooks

| Hook | 职责 |
|------|------|
| `useAudioPreview.ts` | base64 WAV → Blob → ObjectURL → HTMLAudioElement 生命周期 |
| `useStreamingTranslation.ts` | WebSocket 连接管理 (自动重连, 状态机), MediaRecorder/ScriptProcessorNode 采集 PCM 16kHz, Float32→Int16 转换, 接收分发 4 类服务端消息 |
| `useAuth.ts` | `apiRegister()`, `apiLogin()` — 调用 `/auth/*` 端点 |

### 8.3. 认证上下文 (`AuthContext.tsx`)

全局认证状态管理：localStorage 存储 token, 页面刷新时调用 `/auth/me` 校验有效性。

### 8.4. 页面

| 页面 | 功能 |
|------|------|
| `LoginPage.tsx` | email + password, 错误展示, 注册成功 banner, 跳转链接 |
| `RegisterPage.tsx` | email + password + confirm, 客户端验证 (格式/强度/匹配) |
| `TranslatePage.tsx` | 模式切换 (streaming / file upload), streaming 复用 AudioRecorder, file upload 支持拖拽, 结果卡片 |

### 8.5. Flet 移动端原型 (`src/flet_mobile/`)

API-First 架构，所有逻辑由 FastAPI 承担：
- `app.py` — 主页面编排 (The Bridge / The Lab / The Output / The Library)
- `audio_recorder.py` — `AudioRecorder` (16kHz PCM + 实时波形快照)
- `translation_client.py` — `TranslationClient` (httpx 调用 `/api/v1/translate`)
- `bioacoustic_player.py` — `BioacousticPlayer` (sound_id 映射本地样本 + DSP 调整)
- `theme.py` — 视觉 Token (奶油底色, 琥珀色主色, 森林绿科学引用)

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
ws_endpoints.websocket_translate()
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
      │  │     SequenceMatcher(cached_text, final_text) ≥ 0.7      │
      │  │       → 直接复用 cached target_tags (零延迟)             │
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
