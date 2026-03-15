# Phase 7 Batch 2 — 接入 WebSocket 流式管线

| 属性 | 值 |
|------|-----|
| **状态** | ✅ 已完成 |
| **目标** | Flet 移动端接入 Phase 5 流式翻译，录音过程中实时显示转录文本与推测性分析 |

---

## 完成范围

- `src/flet_mobile/translation_client.py` — 重写 `stream_translate()` 为 `asyncio.TaskGroup` 并发（`_sender` + `_receiver`）；添加 `JSONDecodeError` 容错
- `src/flet_mobile/app.py` — 添加 REST/Streaming `SegmentedButton` 模式切换；`on_ws_event()` 处理 4 种消息类型；`update_tags()` 适配 Phase 5 TargetTagSet 5 维标签（兼容 Legacy fallback）；`append_history()` 适配 Phase 5 格式；`_chunk_generator()` 使用 `asyncio.Queue` 桥接音频线程与事件循环；`asyncio.wait_for` 15s 超时保护
- `tests/test_batch2_ws_streaming.py` — 31 个测试用例（15 个测试类）

---

## 修改文件清单

| 文件 | 操作 | 具体改动 |
|------|------|---------|
| `src/flet_mobile/translation_client.py` | 重写 | `stream_translate()` 改为 `asyncio.TaskGroup` 并发 (`_sender` + `_receiver`)；`JSONDecodeError` 容错 |
| `src/flet_mobile/app.py` | 修改 | REST/Streaming `SegmentedButton` 模式切换；`on_ws_event()` 处理 4 种消息；`update_tags()` Phase 5 + Legacy 兼容；`_chunk_generator()` Queue 桥接；15s 超时保护 |
| `tests/test_batch2_ws_streaming.py` | 新增 | 31 个测试用例（15 个测试类）|

---

## 技术方案

### `stream_translate()` 并发实现

```python
async with asyncio.TaskGroup() as tg:
    tg.create_task(self._sender(ws, chunks))
    tg.create_task(self._receiver(ws, on_event))
```

### 模式切换

- `ft.SegmentedButton` 切换 `mode = "rest" | "streaming"`
- REST 模式：保留 `request_translation()` 逻辑
- Streaming 模式：`asyncio.Queue` + `_chunk_generator()` 桥接 → `stream_translate()` 并发通信
- `on_ws_event()` 处理: `transcription` → 实时转录, `analysis_preview` → 标签预览, `result` → 完整结果, `error` → 错误提示

### Phase 5 标签适配

```python
for dim in TAG_DIMENSIONS:
    for tag in tags.get(dim, []):
        chips.append(ft.Chip(label=ft.Text(f"{dim}: {tag}", size=12)))
```

无 Phase 5 标签时自动回退到 Legacy 格式。

---

## Flet 移动端流式翻译数据流

```
用户对着手机说话
      │
      ▼
AudioRecorder.on_chunk(pcm_bytes)
      │
      ├──► translation_client.stream_translate()
      │         │
      │     ┌── _sender() ──────────────────────────────────────────┐
      │     │  每 ~256ms: ws.send(pcm_bytes)                        │
      │     │  用户停止: ws.send({"type":"stop"})                    │
      │     └───────────────────────────────────────────────────────┘
      │         │
      │     ┌── _receiver() (并发) ─────────────────────────────────┐
      │     │  收到 transcription → on_event → 更新 live_transcription│
      │     │  收到 analysis_preview → on_event → 更新 tags_wrap     │
      │     │  收到 result → on_event → 播放音频 + 展示结果           │
      │     │  收到 error → on_event → Snackbar 提示                 │
      │     └───────────────────────────────────────────────────────┘
      │
      ▼
BioacousticPlayer.play_from_base64(audio_base64)
      │
      └── ft.Audio(src=tmp_wav_path, autoplay=True)
```

---

## 验收结果

- REST 模式行为不变 ✅
- Streaming 模式并发发送/接收 ✅
- `update_tags()` 兼容 Legacy 和 Phase 5 格式 ✅
- 195 个测试全部通过 ✅

---

## 测试新增

| 测试文件 | 用例数 | 描述 |
|----------|--------|------|
| `test_batch2_ws_streaming.py` | 31 | 并发通信、Config/Stop、TaskGroup 异常、JSONDecodeError、Phase 5 标签、WS 事件回调、chunk generator |
