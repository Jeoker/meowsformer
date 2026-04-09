# Phase 9 Batch 2 — 滚动推测 + 并行 Stop

> **完成日期：** 2026-04-08
> **工作流：** 标准流程（后端核心逻辑）
> **前提：** Batch 1（AsyncOpenAI 迁移）已完成，`asyncio.create_task()` 可真正并行。

---

## 目标

1. **滚动推测性执行：** 每次中间转录（≥5 词）都触发新推测，取消旧任务，提高缓存命中率
2. **并行 Stop 序列：** 最终转录与推测等待并行执行，减少 stop 延迟

## 修改文件

| 文件 | 改动 |
|------|------|
| `app/api/ws_endpoints.py` | 滚动推测（移除 `is None` 条件，添加 `cancel()`）+ `CancelledError` 处理 + 并行 `_handle_stop`（`create_task` + 并行 await） |
| `tests/test_phase9_streaming_optimization.py` | 新增 10 个测试方法（2 个测试类） |

## 技术方案

### 改动 C — 滚动推测

```
优化前: 仅首次触发推测 (is_speculative_task is None)
优化后: 每次中间转录 → cancel 旧任务 → 启动新推测

效果: 推测基于最近一次中间转录，cache.is_similar() 命中率大幅提升
```

`_speculative_analysis()` 添加 `except asyncio.CancelledError` 处理，日志级别为 `debug`（常规取消）。

### 改动 B — 并行 Stop

```
优化前 (串行):
|── wait speculative ──|── Whisper final ──|   总耗时 = sum

优化后 (并行):
|── wait speculative ──|                       总耗时 = max
|── Whisper final ─────|
```

`_handle_stop()` 先 `asyncio.create_task(transcribe_final())`，然后 `await wait_for(speculative, 5s)`，最后 `await final_transcription_task`。

`CancelledError` 在 `wait_for` 中正确传播：取消 `final_transcription_task` 后 re-raise。

转录失败降级为 `logger.warning`（区分于内部 `logger.error`），使用 `latest_text`。

## 优化效果预期

| 场景 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| 短语句（< 5s）、推测命中 | ~1.5s | ~1.5s | — |
| 中等语句（5-10s）、推测曾命中但文本偏移 | ~4.5s | ~2s | **-2.5s** |
| 长语句（10-15s）、推测首次触发后文本大幅变化 | ~6s | ~2s | **-4s** |
| 最差情况（推测超时 + cache miss） | ~11s | ~3-4s | **-7s** |

## 审核记录

### 产品代码审核
- **Round 1:** 0 critical, 1 warning, 2 suggestions
  - W1: `CancelledError` 被静默吞掉 → 采纳（cancel + re-raise）
  - S2: 双重 error log → 采纳（外层改 `logger.warning`）
  - S3: timeout 后未 await 确认 → 拒绝（实际无影响）
- **Round 2:** 0 issues, code is ready

### 测试代码审核
- **Round 1:** 0 critical, 2 warnings, 3 suggestions
  - W1: 并行耗时测试 flaky → 采纳（改为结构性验证）
  - W2: sleep(0) 竞态 → 采纳（sleep(0.05) + 注释）
  - S3: sleep(0.15) 意图 → 采纳（加注释）
  - S4: 嵌套 with 风格 → 采纳（统一多行 `with`）
  - S5: wait_for mock 脆弱 → 采纳（新增 helper + docstring）
- **Round 2:** 0 issues, code is ready

## 验收结果

- `python main.py` 启动无报错
- 331 个测试全部通过（321 existing + 10 new）
- 滚动推测取消/重启逻辑正确
- `CancelledError` 不逃逸
- 并行 stop 中 Whisper 与推测等待重叠执行
- 仅修改 `ws_endpoints.py` + 新增测试文件
