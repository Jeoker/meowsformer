# Meowsformer — 项目宏观设计与进度

> 文件/函数/算法/协议的详细说明见 [`docs/project_structure_and_logics.md`](./project_structure_and_logics.md)

---

## 1. 项目概览

Meowsformer 是基于 FastAPI 的猫语翻译后端服务，将人类语音翻译为具有生物学意义的真实猫叫声。核心基于瑞典 Lund 大学 Meowsic 研究的科学猫语库。

系统支持两条并行管线：

1. **Legacy Pipeline (Phase 0–3):** 文件上传 → Whisper 转录 → RAG 科学上下文 → LLM 情绪分析 → DSP (VA 映射 + PSOLA 韵律变换) → 合成音频
2. **Streaming Pipeline (Phase 5):** WebSocket 实时音频 → 分块 Whisper → LLM 目标标签生成 → 5 维加权匹配 → 真实猫叫录音播放

**当前状态：** Phase 0–3, 5 完成。Phase 6 (Full Frontend + Auth) 进行中。150 单元测试通过。

---

## 2. 开发阶段

| Phase | 描述 | 状态 |
|-------|------|------|
| **Phase 0** | 核心 API — FastAPI 端点、Whisper 转录、LLM 分析、RAG | Done |
| **Phase 1** | 数据获取 — Zenodo 语料库下载、元数据解析、registry 索引 | Done |
| **Phase 2** | DSP 引擎 — VA 映射、音频检索、PSOLA 韵律变换 | Done |
| **Phase 3** | 集成 — DSP 接入 API 管线、端到端流程、UI 预览 | Done |
| **Phase 4** | 部署 — Docker 容器化、CI/CD、生产化 | Pending |
| **Phase 5** | 流式管线 — WebSocket streaming、多维标签体系、LLM 目标标签、加权匹配 | Done |
| **Phase 6** | 全栈前端 + JWT 认证 | In Progress |

---

## 3. 技术栈

| 类别 | 技术 |
|------|------|
| **语言** | Python 3.10+ (推荐 3.12) |
| **Web 框架** | FastAPI + Uvicorn |
| **数据校验** | Pydantic V2 |
| **AI / LLM** | OpenAI API (GPT-4o, Whisper V3), `instructor` (结构化输出) |
| **向量数据库** | ChromaDB (Phase 0 RAG, 本地持久存储) |
| **音频处理** | FFmpeg (subprocess), `python-multipart` |
| **音频 DSP** | `librosa` (f0/pYIN, I/O), `pytsmod` (WSOLA), `soundfile`, `scipy`, `numpy` |
| **声学特征提取** | `librosa` (pYIN f0, RMS 能量, 时长 — 用于标签构建) |
| **WebSocket** | Starlette/FastAPI 内置 WS |
| **数据获取** | `zenodo-get` |
| **环境管理** | `python-dotenv`, `pydantic-settings` |
| **日志** | `loguru` |
| **测试** | `unittest` |
| **前端** | React 18 + TypeScript + Vite + Tailwind CSS |
| **移动端** | Flet (Python, Material Design 3) |
| **认证** | `python-jose` (JWT), `passlib` (bcrypt), SQLAlchemy + SQLite |

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

## 5. 当前进度详情

### Phase 6A — Backend JWT Auth ✅ COMPLETE

- `app/auth/` 模块 — User 模型 (SQLAlchemy + SQLite), bcrypt 密码哈希, JWT 签发/验证
- 端点: `POST /auth/register`, `POST /auth/login`, `GET /auth/me`
- 依赖注入: `get_current_user` (强制认证) + `get_optional_user` (可选认证)
- Alembic 迁移已配置并应用 (`add_users_table`)
- `app/core/config.py` 扩展: `JWT_SECRET_KEY`, `JWT_ALGORITHM`, `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`
- `tests/test_auth.py` — 10 tests, all passing

### Phase 6B — Frontend SPA Structure ✅ COMPLETE

- React Router v6, `AuthContext` (全局认证状态 + localStorage 持久化 + `/auth/me` 验证)
- `LoginForm`, `RegisterForm`, `Navbar`, `ProtectedRoute`
- Tailwind CSS + 自定义 `meow` 色板
- TypeScript 编译通过, Vite 生产构建通过 (185KB JS)

### Phase 6C — Core Page UI ✅ COMPLETE

- **LoginPage**: email + password, 错误展示, 注册成功 banner
- **RegisterPage**: email + password + confirm, 客户端验证 (格式/强度/匹配)
- **TranslatePage**: 模式切换 (streaming / file upload), 结果卡片 (标签 + 音频播放)

---

## 6. 测试总览

```bash
export PYTHONPATH=$PYTHONPATH:.
python -m unittest discover tests
```

| 测试文件 | 模块 | 用例数 | 描述 |
|----------|------|--------|------|
| `test_api_endpoints.py` | API | — | `POST /translate` 端点, mock services |
| `test_audio_services.py` | Audio | — | FFmpeg 转换与特征提取 |
| `test_llm_service.py` | LLM | — | `analyze_intention`, mock OpenAI |
| `test_rag_service.py` | RAG | — | 知识库初始化, 上下文检索 |
| `test_download_datasets.py` | Data | — | 文件名解析, registry 构建 |
| `test_dsp_processor.py` | DSP | 45 | VA 映射, 音频检索, f0, PSOLA, 包络 |
| `test_description_generator.py` | Descriptions | 31 | Intent 标签, 置信评分, 预览生成 |
| `test_synthesis_service.py` | Synthesis | 15 | emotion→intent, base64, 管线, 降级 |
| `test_auth.py` | Auth | 10 | 密码哈希, JWT, 注册/登录/me |

**总计: 150 tests, all passing.**

> **兼容性:** 推荐 Python 3.12。Python 3.14 与 `chromadb` 的 Pydantic V1 依赖存在兼容性问题。

---

## 7. 搭建与运行

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

**可用端点:**
- API 文档: `http://localhost:8000/docs`
- 健康检查: `GET /health`
- Legacy 翻译: `POST /api/translate`, `POST /api/v1/translate`
- 流式翻译: `WS ws://localhost:8000/ws/translate`
- 认证: `POST /auth/register`, `POST /auth/login`, `GET /auth/me`

---

## 8. 未来开发计划

- **Phase 5 补充测试:** sample_matcher, streaming_transcription, sound_selection, ws_endpoints 单元测试
- **Phase 4 — 部署:** Dockerfile (FFmpeg + librosa), Cloud Run, CI/CD
- **Phase 4 — 持久化:** 翻译历史 (SQLite/PostgreSQL), ChromaDB 迁移至云端
- **前端完善:** `npm run dev` 联调, 路由完善, Flet 移动端 UI
- **高级功能:** 用户反馈循环 (RLHF), 多语言, 标签权重动态调优, 多品种声线偏好
