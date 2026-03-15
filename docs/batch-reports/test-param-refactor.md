# 测试参数重构报告

**日期：** 2026-03-15  
**类型：** 测试基础设施重构（非功能变更，产品代码零修改）  
**状态：** 完成，190 个测试全部通过

---

## 背景

随着 Phase 7 各 Batch 测试文件的积累，测试套件中出现了大量跨文件重复的硬编码字符串（如 `"https://space.ai-builders.com/backend/v1"`、`"gpt-4o"`、`"cat_001"` 等），以及在 batch2 / batch3 / batch4 三个文件中完全重复定义的 flet mock 基础设施类（约 120 行逐字重复）。本次重构解决这两个问题。

---

## 变更文件清单

### 新建文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `tests/shared_params.py` | 53 | 纯常量模块，覆盖所有跨文件重复的硬编码值 |
| `tests/flet_mocks.py` | 140 | flet mock 基础设施：控件类 + `install_flet_mock()` + `BaseMockPage` |
| `tests/ws_stubs.py` | 48 | WebSocket 测试桩：`MockWebSocket` + `ws_connect_coro()` + `async_chunks()` |

### 修改文件

| 文件 | 主要变更 |
|------|----------|
| `tests/test_api_client.py` | 导入 4 个共享常量；新增 `_DUMMY_SK_KEY` 文件内常量；替换 `"openai"` ×10、`"ai_builders"` ×5、`"https://space.ai-builders.com/backend/v1"` ×5、`"gpt-4o"` ×3、`"sk-key"` ×2 |
| `tests/test_api_provider_switch.py` | 导入 9 个共享常量；新增 5 个 SK key 文件内常量（`_SK_BUILDER_TOKEN` ×3、`_SK_BUILDER_SIMPLE` ×2 等）；替换 `AI_BUILDER_BASE_URL` ×8、`EMBEDDING_MODEL` ×2、`MODEL_OPENAI_DEFAULT` ×3、`MODEL_AI_BUILDERS_DEFAULT` ×1、`CHROMA_PATH_*` 各路径 |
| `tests/test_batch2_ws_streaming.py` | 删除 5 个重复定义（`_Ctrl`、`_TextCtrl`、`_ListCtrl`、`_install_flet_mock`、`MockWebSocket`、`_ws_connect_coro`、`_async_chunks`、`_MockPage`）；从新模块 import；替换 `AUDIO_B64_STUB` ×4、`SAMPLE_ID_*`、`MATCH_SCORE_*`、`STREAMING_SETTLE_SECS` 等 |
| `tests/test_batch3_audio_playback.py` | 删除同上重复定义；新增文件内常量 `_FAKE_WAV_PATH` ×4、`_NONEXISTENT_CATALOG` ×3；替换 `AUDIO_B64_DECODED` ×3、`DUMMY_PCM_BYTES` ×2、`SAMPLE_ID_PRIMARY` ×5 等 |
| `tests/test_batch4_ux_enhancements.py` | 删除同上重复定义；替换 `AUDIO_B64_STUB`、`MATCH_SCORE_HIGH`、`SOUND_ID_LEGACY` ×2、`BREED_DEFAULT`、`STREAMING_SETTLE_SECS` ×2 等 |

---

## 核心设计决策

### 1. 三文件职责分离，不合并成单一 helpers

| 文件 | 只包含 |
|------|--------|
| `shared_params.py` | 不含 app 导入；以纯值常量为主，含 `json.dumps` 等标准库调用 |
| `flet_mocks.py` | flet 控件类 + page mock，无断言逻辑 |
| `ws_stubs.py` | WS 测试桩类 + 工具函数，无断言逻辑 |

### 2. `BaseMockPage` 合并三种变体

batch2 / batch3 / batch4 各自定义了略有不同的 `_MockPage`。差异分析：

| 属性/行为 | batch2 | batch3 | batch4 | 合并策略 |
|---|---|---|---|---|
| `on_disconnect` | ✗ | ✓ | ✓ | 统一加入（batch2 不使用，无副作用） |
| `_opened` + `open()` 记录 | ✓ | ✗（no-op） | ✓ | 统一记录（batch3 不断言 `_opened`，无副作用） |
| `_run_task_calls` + `run_task()` 记录 | ✗（no-op） | ✗（no-op） | ✓ | 统一记录（batch2/3 不断言，无副作用） |

`BaseMockPage` 为 batch4 超集，其附加行为对 batch2/3 无影响。

### 3. LLM 模型常量不合并

`MODEL_OPENAI_DEFAULT = "gpt-4o"` 与 `MODEL_AI_BUILDERS_DEFAULT = "deepseek"` 是两个独立常量，分别对应两个 provider 的 Settings 真实默认值，语义不同，不可合并为单一常量。

### 4. `_patch_ws` 保留在各文件内

batch2 的 `_patch_ws` patch 目标为 **class**（`TranslationClient`），batch4 的 patch 目标为 **instance**（`client`），行为不同，不提取到 `ws_stubs.py`。

---

## 附带修复

重构过程中发现并修复了一个预存在的 flet mock 缺陷：

- **问题：** `app.py` 调用 `ft.Padding.symmetric(...)` 和 `ft.Border.all(...)`，但原有 `_install_flet_mock()` 中未设置大写的 `ft.Padding` 和 `ft.Border`（仅设置了小写的 `ft.padding` 和 `ft.border`）。
- **影响：** batch4 所有 46 个依赖 `asyncSetUp` 的测试在 `meowsformer_ui` 调用时抛出 `AttributeError`，静默失败（因错误在 setUp 阶段，非断言阶段）。
- **修复：** 在 `flet_mocks.py` 的 `install_flet_mock()` 中将 `"Padding"` 和 `"Border"` 加入 MagicMock 属性列表。

---

## 验证结果

```
Ran 190 tests in 4.533s

OK
```

190 个测试（`test_api_client` + `test_api_provider_switch` + `test_batch1` + `test_batch2` + `test_batch3` + `test_batch4`）全部通过，无回归。
