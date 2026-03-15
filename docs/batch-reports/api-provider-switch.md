# API 供应方切换 — 开发报告

**完成日期：** 2026-03-14  
**性质：** 边缘功能（不依附于特定 Phase，独立交付）  
**测试数量：** +42（`test_api_client.py`）；全项目共 330 个测试全部通过

---

## 1. 需求背景

项目此前全量硬编码使用 OpenAI 官方 API。需求：在保持日常使用零改动的前提下，支持在 OpenAI 与 ai-builders 兼容平台之间手动切换。

ai-builders 是 OpenAI 兼容接口（相同 SDK、相同请求/响应格式），区别仅在 `api_key` 和 `base_url` 两个参数。

---

## 2. 技术方案

利用 OpenAI SDK 的 `base_url` 参数实现供应方切换，无需引入任何新依赖。

### 核心架构

```
.env (API_PROVIDER / AI_BUILDER_TOKEN / LLM_MODEL)
  │
  └─► app/core/config.py (Settings 新增 4 个字段)
        │
        └─► app/core/api_client.py (新文件：供应方工厂)
              │
              ├─► transcription_service.py   (懒加载 _get_client)
              ├─► streaming_transcription_service.py (同上)
              ├─► llm_service.py             (懒加载 _get_client)
              ├─► sound_selection_service.py (同上)
              └─► vector_store.py            (_embedding_fn 条件初始化)
```

---

## 3. 修改文件清单

### 新增

| 文件 | 说明 |
|------|------|
| `app/core/api_client.py` | 供应方工厂（~45 行），提供 `get_openai_client()` 和 `get_instructor_client()` |
| `tests/test_api_client.py` | 42 个单元测试，覆盖工厂逻辑、config 默认值、懒加载隔离、模型名传播 |
| `docs/batch-reports/api-provider-switch.md` | 本报告 |

### 修改

| 文件 | 改动内容 |
|------|---------|
| `app/core/config.py` | 新增 `API_PROVIDER`、`AI_BUILDER_TOKEN`、`AI_BUILDER_BASE_URL`、`LLM_MODEL` 四字段 |
| `app/services/transcription_service.py` | 模块级懒加载缓存 + `_get_client()` 调用工厂 |
| `app/services/streaming_transcription_service.py` | `_get_client()` 内改用 `get_openai_client()` |
| `app/services/llm_service.py` | 模块级懒加载缓存；`model` 参数化为 `settings.LLM_MODEL` |
| `app/services/sound_selection_service.py` | `_get_client()` 改用 `get_instructor_client()`；模型名参数化 |
| `app/db/vector_store.py` | `_embedding_fn` 按 `API_PROVIDER` 条件传入 `api_key` / `api_base` |

---

## 4. 切换方式

```bash
# 切换到 ai-builders（在 .env 中修改）
API_PROVIDER=ai_builders
AI_BUILDER_TOKEN=sk_c...
AI_BUILDER_BASE_URL=https://space.ai-builders.com/backend/v1   # 默认已正确，MCP 指定
LLM_MODEL=deepseek   # 可选；空时自动为 deepseek，可覆盖为 deepseek-chat 等

# 切回 OpenAI（默认）
API_PROVIDER=openai
```

---

## 5. 审核循环记录

### 产品代码（2 轮）

**第 1 轮：**
- Critical × 2：`transcription_service` 和 `llm_service` 将 client 实例化放在函数体内，丢失连接池复用
- Warning × 1：`streaming_transcription_service` 直接导入 `OpenAI` 未用 `TYPE_CHECKING` 保护
- Suggestion × 3：model_validator（拒绝）、变量名 `openai_ef`（采纳）、docstring 指导（采纳）

**第 2 轮：** No new issues. Code is ready.

### 测试代码（3 轮）

**第 1 轮：**
- Warning × 3：`TestLLMServiceLazyLoad` 缺少重置缓存测试、`call_args` 拆包风格不一致、`assertRaises` 过于宽松
- Suggestion × 2：`lambda **_` 命名、辅助方法缺少类型注解

**第 2 轮：** 全部采纳，发现 1 处漏改 + 1 处未使用导入

**第 3 轮：** No issues found. Code is ready.

---

## 6. 验收结果

- ✅ 默认配置（`API_PROVIDER=openai`）下所有现有功能行为不变
- ✅ 切换到 `ai_builders` 只需改 3 个 `.env` 变量
- ✅ 零新增外部依赖
- ✅ 330 个测试全部通过
