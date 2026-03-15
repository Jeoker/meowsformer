# Phase 7 Batch 3 — 修复音频播放

| 属性 | 值 |
|------|-----|
| **状态** | ✅ 已完成 |
| **目标** | 用 Flet 原生 `ft.Audio` 替代 `page.launch_url` hack，实现应用内音频播放 |

---

## 完成范围

- `src/flet_mobile/bioacoustic_player.py` — 重构: `_play_wav_bytes()` async + `asyncio.to_thread` 文件 I/O；`_ensure_audio_overlay()` 懒注册 `ft.Audio`；`play_from_base64()` 直接播放 base64；`_write_temp_file()` 异常安全；`dispose()` 清理资源；`_cleanup_temp()` 管理临时文件生命周期
- `src/flet_mobile/app.py` — REST 自动播放（`request_translation` 中调用 `play_from_base64`）；Streaming 自动播放（`on_ws_event` result 中调用 `play_from_base64`）；`page.on_disconnect` 调用 `player.dispose()`
- `tests/test_batch3_audio_playback.py` — 25 个测试用例（14 个测试类）

---

## 修改文件清单

| 文件 | 操作 | 具体改动 |
|------|------|---------|
| `src/flet_mobile/bioacoustic_player.py` | 重构 | `_play_wav_bytes()` async + `asyncio.to_thread`；`_ensure_audio_overlay()` 懒注册；`play_from_base64()` 直接播放；`_write_temp_file()` 异常安全；`dispose()` 清理；空输入防御 |
| `src/flet_mobile/app.py` | 修改 | REST/Streaming 自动播放 (`play_from_base64`)；`page.on_disconnect` 调用 `dispose()` |
| `tests/test_batch3_audio_playback.py` | 新增 | 25 个测试用例（14 个测试类）|

---

## 技术方案

### 播放架构

```python
async def _play_wav_bytes(self, wav_bytes: bytes) -> None:
    self._cleanup_temp()
    tmp_path = await asyncio.to_thread(self._write_temp_file, wav_bytes)
    self._temp_file = tmp_path
    self._ensure_audio_overlay()
    self._audio_control.src = tmp_path
    self._audio_control.autoplay = True
    self.page.update()
```

保留 librosa DSP 处理（pitch/tempo），输出通过 `ft.Audio` 应用内播放。

---

## 验收结果

- 翻译完成后自动应用内播放（无跳转浏览器）✅
- REST 和 Streaming 均自动播放 ✅
- pitch/tempo 调节后播放 DSP 处理后音频 ✅
- 220 个测试全部通过 ✅

---

## 测试新增

| 测试文件 | 用例数 | 描述 |
|----------|--------|------|
| `test_batch3_audio_playback.py` | 25 | ft.Audio overlay、临时文件管理、DSP 处理、base64 播放、REST/Streaming 自动播放、dispose |
