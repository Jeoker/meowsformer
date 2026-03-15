# Phase 7 Batch 4 — UX 完善

| 属性 | 值 |
|------|-----|
| **状态** | ✅ 已完成 |
| **目标** | 提升 Flet 移动端整体使用体验：WS 状态指示、自动降级、Snackbar 通知、录音计时、历史记录增强 |

---

## 完成范围

- `src/flet_mobile/translation_client.py` — `WebSocketConnectionError` 异常类；`on_state_change` 回调参数；`WS_CONNECT_TIMEOUT` 5s 连接超时
- `src/flet_mobile/app.py` — `ws_status_chip` 连接状态指示器 + `_update_ws_status()`；`_show_snackbar()` 统一 SnackBar 通知；`recording_timer_loop()` wall-clock 锚定计时器；`_fallback_to_rest()` WS 失败自动降级；`append_history()` ExpansionTile 5 维标签展示
- `tests/test_batch4_ux_enhancements.py` — 37 个测试用例

---

## 修改文件清单

| 文件 | 操作 | 具体改动 |
|------|------|---------|
| `src/flet_mobile/translation_client.py` | 修改 | 新增 `WebSocketConnectionError` 异常类；`stream_translate()` 增加 `on_state_change` 回调参数，fire connecting/connected/disconnected；`asyncio.wait_for()` 5s 超时检测；`TimeoutError`/`OSError`/`WebSocketException` 统一包装为 `WebSocketConnectionError` |
| `src/flet_mobile/app.py` | 修改 | 5 项 UX 功能（详见技术方案）；6 处错误文本替换为 Snackbar |
| `tests/test_batch2_ws_streaming.py` | 修改 | 适配新 API：SnackBar mock、`page.open()` 方法、WS 连接模型变更、布局断言更新 |
| `tests/test_batch3_audio_playback.py` | 修改 | 适配新 API：SnackBar mock、`page.open()` 方法、布局断言更新 |
| `tests/test_batch4_ux_enhancements.py` | 新增 | 37 个测试用例（13 个测试类）|

---

## 技术方案

### 1. WebSocket 连接状态指示器

Bridge 卡片标题行新增 `ws_status_chip` (Chip widget)，通过 `_update_ws_status(state)` 映射状态到 UI：

| 状态 | 标签 | 颜色 | 图标 |
|------|------|------|------|
| `connecting` | 连接中... | AMBER | WIFI_ROUNDED |
| `connected` | 已连接 | FOREST_GREEN | WIFI_ROUNDED |
| `disconnected` | 已断开 | PAW_PINK | WIFI_OFF_ROUNDED |
| `reconnecting` | 重连中... | AMBER | WIFI_ROUNDED (reserved) |

`translation_client.py` 的 `stream_translate()` 新增 `on_state_change: Callable[[str], None] | None` 参数，在 WS 生命周期各阶段回调。

### 2. 网络错误自动降级

```
stream_translate() → asyncio.wait_for(websockets.connect(), timeout=5.0)
  ├─ TimeoutError  ──► WebSocketConnectionError
  ├─ OSError       ──► WebSocketConnectionError
  └─ WebSocketException ──► WebSocketConnectionError

on_record_toggle() catches WebSocketConnectionError:
  → _fallback_to_rest()
    → translate_mode = "rest"
    → mode_selector.selected = {"rest"}
    → Snackbar "WebSocket 不可用，已切换为文件上传模式" (amber)
  → 若有 raw_pcm → request_translation(raw_pcm) via REST
```

### 3. Snackbar 错误通知

`_show_snackbar(message, is_error=True)` 统一入口：
- `is_error=True` → RED_700 背景
- `is_error=False` → AMBER 背景
- `duration=3000` (3 秒自动消失)

替换了 6 处原有的 `analysis_status.value = f"...错误..."` 文本赋值。

### 4. 录音时长实时显示

`recording_timer_text` (Text widget, MM:SS, AMBER, W_600) 放置在波形区旁边。

`recording_timer_loop()` 使用 `asyncio.get_event_loop().time()` 锚定 wall-clock 时间，避免 sleep-based 累积漂移。录音开始时显示，停止时隐藏。

### 5. 历史记录增强

`append_history()` 每条记录包含：
- 转录文本
- 摘要行：主要 emotion/intent 标签 + 匹配分数 (百分比) + 时间戳
- `ft.ExpansionTile`（"查看完整 5 维标签"）：5 个维度行，各显示维度名 + 标签列表（空维度显示 "-"）

---

## 审核历程

### 产品代码审核
- Round 1：0 Critical, 2 Warnings, 4 Suggestions
- Developer 采纳 5 条（W1 移除死代码、W2 wall-clock 计时器、S1 添加注释、S2 移除不可达代码、S4 变量重命名），拒绝 1 条（S3 mock 提取推迟）
- Round 2：Reviewer 确认所有修改正确，无新问题

### 测试代码审核
- Round 1：0 Critical, 3 Warnings, 5 Suggestions
- Test-engineer 采纳 7 条（W1 import 位置、W2 wall-clock 测试增强、W3 函数名守卫、S2 调用参数验证、S3 deepcopy 模板、S4 注释、S5 排序测试），拒绝 1 条（S1 mock 提取推迟）
- Round 2：Reviewer 确认所有修改正确，无新问题

---

## 验收结果

- WebSocket 连接状态指示器实时更新 ✅
- 断网时自动切换到 REST 模式并 Snackbar 提示用户 ✅
- 所有错误以 Snackbar 展示，3 秒后自动消失 ✅
- 录音时显示实时时长 (MM:SS, wall-clock 锚定) ✅
- 历史记录条目含标签与分数，可展开查看完整 5 维标签 ✅
- 257 个测试全部通过 ✅

---

## 测试新增

| 测试文件 | 用例数 | 描述 |
|----------|--------|------|
| `test_batch4_ux_enhancements.py` | 37 | WebSocketConnectionError、on_state_change 回调序列、超时/OSError 包装、WS 状态指示器 4 态、Snackbar 颜色/时长/内容、录音计时器格式/可见性/wall-clock 锚定、自动降级 REST (模式切换+Snackbar+REST 重放)、ExpansionTile 5 维标签 (维度标签/空标签/转录/时间戳/分数/排序)、Bridge 卡片结构、Legacy fallback |
