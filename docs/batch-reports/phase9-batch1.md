# Phase 9 Batch 1 — AsyncOpenAI 基础设施迁移

> **完成日期：** 2026-04-08
> **工作流：** 标准流程（后端核心逻辑）

---

## 目标

将 Streaming Pipeline 的 OpenAI 调用从 sync 切换为 async，解除 event loop 阻塞。行为不变，Legacy Pipeline 不改。

## 修改文件

| 文件 | 改动 |
|------|------|
| `app/core/api_client.py` | 新增 `get_async_openai_client()` → `AsyncOpenAI`，`get_async_instructor_client()` → `instructor.AsyncInstructor`。原有 sync 函数不变。 |
| `app/services/streaming_transcription_service.py` | import 从 `get_openai_client` 换为 `get_async_openai_client`；`_client` 类型 `OpenAI` → `AsyncOpenAI`；`_call_whisper()` 中 Whisper API 调用添加 `await`。 |
| `app/services/sound_selection_service.py` | import 从 `get_instructor_client` 换为 `get_async_instructor_client`；`_client` 类型 `Instructor` → `AsyncInstructor`；`generate_target_tags()` 中 LLM API 调用添加 `await`。 |
| `tests/test_api_client.py` | 新增 29 个测试方法（6 个测试类），覆盖 async 工厂函数、lazy-load 缓存、await 路径、错误降级。 |

## 未改动文件

| 文件 | 原因 |
|------|------|
| `app/services/transcription_service.py` | Legacy Pipeline |
| `app/services/llm_service.py` | Legacy Pipeline |
| `app/services/rag_service.py` | Legacy Pipeline |
| `app/services/synthesis_service.py` | Legacy Pipeline |
| `app/db/vector_store.py` | Legacy Pipeline |
| `app/services/sample_matcher.py` | 纯内存计算，无 I/O |
| `app/api/ws_endpoints.py` | Batch 2 范围 |

## 技术方案

### 核心变更

`_call_whisper()` 和 `generate_target_tags()` 函数签名早在 Phase 5 已声明为 `async`，但内部使用 sync `OpenAI` / `instructor.Instructor` 客户端，实际阻塞 event loop。Batch 1 将 transport 层替换为 `AsyncOpenAI` / `instructor.AsyncInstructor`，使 `await` 真正异步。

### async 工厂模式

```
api_client.py
├── get_openai_client()           → OpenAI (sync, Legacy)
├── get_instructor_client()       → Instructor (sync, Legacy)
├── get_async_openai_client()     → AsyncOpenAI (async, Streaming)    ← NEW
└── get_async_instructor_client() → AsyncInstructor (async, Streaming) ← NEW
```

各服务的 `_get_client()` 懒加载模式不变，仅替换底层工厂函数。

## 审核记录

### 产品代码审核
- **Round 1:** 0 critical, 0 warnings, 2 suggestions
  - S1: `get_async_instructor_client` docstring 缺少 caching 提示 → PM 采纳修复
  - S2: `_call_whisper` 中 `open()` 为同步 I/O → pre-existing，不阻塞本批次

### 测试代码审核
- **Round 1:** 0 critical, 2 warnings, 3 suggestions
  - W1+W2: 真实 `AsyncOpenAI` 实例未关闭导致 ResourceWarning → 删除（mock 版已覆盖）
  - S3: fallback 测试只断言 2/5 维度 → 补齐全部 5 维
  - S4: 缺少 `file` kwarg 断言 → 添加
  - S5: 重复结构 → 拒绝（测试不同路径，显式更清晰）
- **Round 2:** 0 issues, code is ready

## 验收结果

- `python main.py` 启动无报错
- 321 个测试全部通过（292 existing + 29 new）
- Legacy REST 翻译不受影响
- `ws_endpoints.py` 未修改（Batch 2 范围）
