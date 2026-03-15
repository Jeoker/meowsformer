# Phase 7 — Batch 5：flet-audio 应用内音频播放恢复

| 字段 | 值 |
|------|-----|
| **状态** | ✅ 已完成 |
| **目标** | 将 Flet 从 0.80.5 升级至 0.82.2，引入 `flet-audio` 扩展包，恢复应用内原生音频播放（替代 `page.launch_url()` + FastAPI `/audio/latest` 临时 workaround） |
| **日期** | 2026-03-14 |

---

## 背景

Batch 3 实现音频播放时，因 Flet 0.80.x 将 `ft.Audio` 拆分为独立扩展包，当时采用 `page.launch_url()` + FastAPI `/audio/latest` 端点作为临时方案——播放时在浏览器中打开新标签页，体验严重退化。

本 Batch 通过研究 Flet 0.82.2 的官方 `flet-audio` 包，发现完整的原生解决方案，并完成从 workaround 到原生播放的迁移。

---

## 技术方案

### 旧机制（已移除）
```
_play_wav_bytes():
  asyncio.to_thread(_write_cache, wav_bytes)  →  临时文件
  page.launch_url("http://127.0.0.1:8000/audio/latest")  →  浏览器新标签播放
```
FastAPI 新增了 `/audio/latest` GET 端点专门为此服务。

### 新机制（fta.Audio）
```
__init__():
  self._audio = fta.Audio(src=None, volume=1.0, release_mode=ReleaseMode.STOP)
  page.overlay.append(self._audio)
  page.update()

_play_wav_bytes(wav_bytes: bytes):
  await self._audio.release()   # 停止旧播放，释放资源
  self._audio.src = wav_bytes   # src 接受 str|bytes|None
  self._audio.update()          # 推送到 Flutter 层（必须在 play() 之前）
  await self._audio.play()

dispose():
  if self._audio in page.overlay:
      page.overlay.remove(self._audio)
      page.update()
```

### 关键设计决策

| 决策 | 原因 |
|------|------|
| `flet_audio` 延迟导入（在 `__init__` 内部） | `flet_audio/types.py` 模块级代码引用 `ft.Event`，而测试 mock 没有该属性；延迟导入使 test_batch4 正常运行 |
| `src = bytes`（非 base64 字符串） | `fta.Audio.src` 原生支持 `str \| bytes \| None`，无需二次 base64 编码 |
| `await release()` 先于 `src` 赋值 | 防止重入播放（用户快速连续翻译）时行为未定义 |
| `self._audio.update()` 在 `play()` 之前 | Flet 属性赋值只修改 Python 侧状态，必须 `update()` 才能推送到 Flutter 层 |
| 版本精确锁定 `==0.82.2` | `flet-audio` 历史上每个小版本都有不兼容变更；精确锁定防止悄默升级 |

---

## 修改文件

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `requirements.txt` | 修改 | `flet>=0.80.5` → `flet==0.82.2`；新增 `flet-audio==0.82.2` |
| `src/flet_mobile/bioacoustic_player.py` | 重构 | 移除 `_AUDIO_CACHE`、`_AUDIO_SERVE_URL`、`_write_cache`；引入 `fta.Audio` 播放机制 |
| `app/api/endpoints.py` | 删除 | 移除 `/audio/latest` GET 端点及相关模块变量 |
| `tests/test_batch3_audio_playback.py` | 重写 | 全面重写，覆盖新的 `fta.Audio` 播放机制（25 个测试用例保持不变） |

---

## 验收结果

- **329/329 测试通过**（全量测试套件）
- `bioacoustic_player.py` 导入正常，无 `_AUDIO_CACHE` 残留
- `play_from_base64()` / `play_sound_id()` 公共 API 签名不变，`app.py` 无需任何修改
- `app/api/endpoints.py` 无 `/audio/latest` 路由
- 版本精确锁定，`requirements.txt` 可重复安装

---

## 后续补丁（2026-03-15）

### Flet 0.80+ 废弃 API 迁移

Flet 0.80.0 起将小写模块级辅助函数标记为 deprecated，`app.py` 中 3 处调用已迁移至对应类方法：

| 位置 | 旧 API | 新 API |
|------|--------|--------|
| L74 `ws_status_chip` | `ft.padding.symmetric(...)` | `ft.Padding.symmetric(...)` |
| L120 `waveform` | `ft.padding.symmetric(...)` | `ft.Padding.symmetric(...)` |
| L144 `rag_bubble` | `ft.border.all(...)` | `ft.Border.all(...)` |

功能与参数不变，消除运行时 DeprecationWarning。

**Edge 浏览器注意事项：** Microsoft Edge 的 Tracking Prevention 可能拦截 `cdn.jsdelivr.net` 上的 Flutter Rive WASM 依赖（`rive_native.js`），导致页面空白。解决方法：在 Edge 地址栏左侧图标中为 localhost 关闭 Tracking Prevention。

---

## 测试覆盖（test_batch3，重写后）

| TestCase | 用例数 | 覆盖内容 |
|----------|--------|----------|
| `TestBioacousticPlayerInit` | 3 | `fta.Audio` 参数、overlay 注册、`page.update()` 次数 |
| `TestPlayWavBytes` | 4 | 调用顺序 `release→update→play`、`src` 赋值、两个 async 调用均被 await |
| `TestPlayFromBase64` | 3 | 有效 base64 解码播放、空字符串守卫、无效 base64 异常传播 |
| `TestDispose` | 2 | 在 overlay 中时 remove + update、不在 overlay 时安全退出 |
| `TestBuildIndex` | 2 | catalog 不存在 → `{}`、有效 catalog → 正确映射 |
| `TestResolveSound` | 3 | 索引命中、未命中→fallback、文件缺失→fallback |
| `TestPlaySoundId` | 1 | 完整 DSP 链路验证 |
| `TestProcessToWavBytes` | 2 | WAV 输出、极值 pitch/tempo 裁剪 |
| `TestRESTAutoplay` | 2 | audio_base64 触发播放、缺失 key 跳过 |
| `TestStreamingAutoplay` | 1 | Streaming result 触发播放 |
| `TestManualPlay` | 1 | 滑块值正确转发（不断言 sound_id 默认值） |
| `TestOnDisconnect` | 1 | page.on_disconnect 调用 dispose |
