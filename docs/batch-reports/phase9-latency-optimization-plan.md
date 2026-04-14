# Phase 9 — Streaming Pipeline 延迟优化

> **创建日期：** 2026-04-08
> **工作流：** 标准流程（后端核心逻辑）
> **范围：** 仅 Streaming Pipeline（`/ws/translate`）；Legacy Pipeline（`/api/v1/translate`）不改

---

## 1. 问题诊断

用户松开按钮到收到猫叫音频，典型延迟 **3–6s**，最差 **11s**。

### 1.1 当前 stop 处理时间线（串行）

```
用户松开按钮
├─ [步骤1] await wait_for(speculative_task, 5s)     ⏱ 0–5s
├─ [步骤2] await transcribe_final()                 ⏱ 1–3s  （sync HTTP，阻塞 event loop）
├─ [步骤3] 检查推测缓存
│   ├─ 命中 → 用缓存标签                            ⏱ ~0s
│   └─ 未命中 → await generate_target_tags()        ⏱ 2–3s  （sync HTTP，阻塞 event loop）
├─ [步骤4] find_best_match() 遍历 483 样本           ⏱ ~10ms
├─ [步骤5] 读磁盘 WAV + base64 编码                  ⏱ ~10ms
└─ [步骤6] WebSocket 发送结果                        ⏱ ~5ms
```

最差 11s（5+3+3）/ 典型 4.5s（0+2+2.5）/ 最佳 1.5s（0+1.5+0）

### 1.2 三个根因

| # | 根因 | 影响 |
|---|------|------|
| **A** | **sync OpenAI 客户端阻塞 event loop** | `api_client.py` 返回 `OpenAI`（sync），所有 Whisper/LLM 调用阻塞 asyncio loop，`asyncio.create_task()` 无法真正并行 |
| **B** | **stop 处理完全串行** | `_handle_stop()` 顺序等待：推测完成 → 最终转录 → 缓存判断 → （可能）新 LLM 调用 |
| **C** | **推测性执行只触发一次** | `_speculative_task is None` 条件导致首次触发后不再更新，用户继续说话后文本大幅变化，相似度 < 0.7，推测结果被丢弃 |

### 1.3 优化后目标

```
用户松开按钮
├─ 并行启动：transcribe_final()（async）+ 等待推测完成（滚动更新，大概率已完成）
├─ 总耗时 = max(Whisper, 推测等待) ≈ 1–2s
├─ cache.is_similar() → 高概率命中（推测基于最近一次中间转录）
└─ match + encode ~20ms → 总延迟 ~1–2s
```

**目标：典型场景 3–6s → 1–2s，最差场景 11s → 3–4s。**

---

## 2. Provider 说明

> **Batch 0 已移除 ai-builders 兼容层。** 项目现在仅使用 OpenAI API，无需 provider 分支逻辑。
> async 工厂函数为单路径：`AsyncOpenAI(api_key=settings.OPENAI_API_KEY)`，`instructor.from_openai(client)` 默认 `Mode.TOOLS`。

---

## 3. 实现方案

### Batch 1 — AsyncOpenAI 基础设施迁移

**目标：** transport 层 sync → async，行为不变，解除 event loop 阻塞。
**不改：** Legacy Pipeline 全部服务文件、`ws_endpoints.py`（Batch 2 改）。

#### `app/core/api_client.py`

新增两个 async 工厂函数，保留原有 sync 函数：

```python
from openai import OpenAI, AsyncOpenAI
import instructor

# 现有（不变）
def get_openai_client() -> OpenAI: ...
def get_instructor_client() -> instructor.Instructor: ...

# 新增
def get_async_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

def get_async_instructor_client() -> instructor.AsyncInstructor:
    client = get_async_openai_client()
    return instructor.from_openai(client)
```

#### `app/services/streaming_transcription_service.py`

```python
# 修改前
from app.core.api_client import get_openai_client
_client: Optional["OpenAI"] = None
def _get_client() -> "OpenAI": ...
transcription = client.audio.transcriptions.create(**kw)  # sync，阻塞 event loop

# 修改后
from app.core.api_client import get_async_openai_client
_client: Optional["AsyncOpenAI"] = None
def _get_client() -> "AsyncOpenAI": ...
transcription = await client.audio.transcriptions.create(**kw)  # async
```

`_call_whisper()` 完整实现：

```python
async def _call_whisper(self) -> str:
    combined = b"".join(self._chunks)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        audio_array = np.frombuffer(combined, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(str(tmp_path), audio_array, self.sample_rate, format="WAV", subtype="PCM_16")

        client = _get_client()
        with open(tmp_path, "rb") as audio_file:
            kw: dict = {
                "model": "whisper-1",
                "file": audio_file,
                "response_format": "text",
            }
            lang = (settings.WHISPER_LANGUAGE or "").strip()
            if lang:
                kw["language"] = lang
            transcription = await client.audio.transcriptions.create(**kw)

        return transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
```

#### `app/services/sound_selection_service.py`

```python
# 修改前
from app.core.api_client import get_instructor_client
_client: Optional[instructor.Instructor] = None
response = client.chat.completions.create(...)  # sync

# 修改后
from app.core.api_client import get_async_instructor_client
_client: Optional[instructor.AsyncInstructor] = None
response = await client.chat.completions.create(...)  # async
```

`generate_target_tags()` 完整实现：

```python
async def generate_target_tags(text: str) -> TargetTagSet:
    client = _get_client()
    user_prompt = f"主人说的话（转录）: \"{text}\"\n\n请做语义翻译并输出目标标签。"

    try:
        response = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            response_model=TargetTagSet,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        logger.debug("LLM target tags: {}", response.model_dump())
        return response
    except Exception as e:
        logger.error("LLM target-tag generation failed: {}", e)
        return TargetTagSet(
            emotion=["calm"],
            intent=["expressing_comfort"],
            acoustic=["mid_pitch", "medium_length"],
            social_context=["near_owner"],
            reasoning=f"LLM调用失败，使用默认标签: {e}",
        )
```

#### 文件不变清单

| 文件 | 原因 |
|------|------|
| `app/services/transcription_service.py` | Legacy Pipeline |
| `app/services/llm_service.py` | Legacy Pipeline |
| `app/services/rag_service.py` | Legacy Pipeline |
| `app/services/synthesis_service.py` | Legacy Pipeline |
| `app/db/vector_store.py` | Legacy Pipeline |
| `app/services/sample_matcher.py` | 纯内存计算，无 I/O |
| `app/api/ws_endpoints.py` | Batch 2 改 |
| 所有前端文件 | 无需改动 |

#### Batch 1 验收标准

- `python main.py` 启动无报错
- WebSocket 流式翻译端到端可用（录音 → 中间转录 → 推测预览 → 最终结果）
- Legacy REST 翻译端到端可用
- 330 个现有测试全部通过
- 新增测试覆盖 `get_async_openai_client`、`get_async_instructor_client`

---

### Batch 2 — 滚动推测 + 并行 Stop

**前提：** Batch 1 已完成（async client 就位，`asyncio.create_task()` 可真正并行）。

#### 改动 C — 滚动推测性执行

**文件：** `app/api/ws_endpoints.py`

```python
# 修改前：仅首次触发
if _word_count(text) >= 5 and session._speculative_task is None:
    session._speculative_task = asyncio.create_task(
        _speculative_analysis(session, text, websocket)
    )

# 修改后：每次中间转录更新，取消旧推测、启动新推测
if _word_count(text) >= 5:
    if session._speculative_task and not session._speculative_task.done():
        session._speculative_task.cancel()
    session._speculative_task = asyncio.create_task(
        _speculative_analysis(session, text, websocket)
    )
```

`_speculative_analysis()` 需优雅处理 `CancelledError`：

```python
async def _speculative_analysis(
    session: StreamingSession,
    text: str,
    websocket: WebSocket,
) -> None:
    try:
        tags = await generate_target_tags(text)
        session.speculative_cache.store(text, tags)

        primary_emotion = tags.emotion[0] if tags.emotion else "unknown"
        primary_intent = tags.intent[0] if tags.intent else "unknown"

        await websocket.send_json(
            WSAnalysisPreviewMessage(
                emotion=primary_emotion,
                intent=primary_intent,
            ).model_dump()
        )
    except asyncio.CancelledError:
        logger.debug("Speculative analysis cancelled (superseded by newer text)")
    except Exception as e:
        logger.error("Speculative analysis failed: {}", e)
```

#### 改动 B — 并行 Stop 序列

**文件：** `app/api/ws_endpoints.py` `_handle_stop()`

```python
async def _handle_stop(websocket, session):
    # 第一阶段：并行执行最终转录 + 等待推测完成
    final_transcription_task = asyncio.create_task(
        session.transcription.transcribe_final()
    )

    if session._speculative_task and not session._speculative_task.done():
        try:
            await asyncio.wait_for(session._speculative_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Speculative task timed out")
            session._speculative_task.cancel()
        except asyncio.CancelledError:
            pass

    try:
        final_text = await final_transcription_task
    except Exception as e:
        logger.error("Final transcription failed: {}", e)
        final_text = session.transcription.latest_text

    await websocket.send_json(
        WSTranscriptionMessage(text=final_text, is_final=True).model_dump()
    )

    if not final_text:
        await _send_error(websocket, "No speech detected")
        return

    # 第二阶段：决定标签来源
    if session.speculative_cache.is_similar(final_text):
        logger.info("Reusing cached LLM result (text similar)")
        target_tags = session.speculative_cache.get()
    else:
        logger.info("Final text differs — calling LLM again")
        target_tags = await generate_target_tags(final_text)

    if target_tags is None:
        await _send_error(websocket, "Failed to generate target tags")
        return

    # 第三阶段：匹配 + 编码 + 发送
    result = await select_and_encode(
        target_tags=target_tags,
        breed_preference=session.breed_preference,
    )

    if result is None:
        await _send_error(websocket, "No matching cat sound found")
        return

    result.transcription = final_text

    await websocket.send_json(
        WSResultMessage(
            transcription=final_text,
            selected_category=result.selected_sample,
            audio_base64=result.audio_base64,
            reasoning=result.reasoning,
        ).model_dump()
    )

    logger.success(
        "Result sent: sample={}, score={:.3f}",
        result.selected_sample.sample_id,
        result.selected_sample.match_score,
    )
```

并行化效果：

```
串行：|── wait speculative ──|── Whisper final ──|   总耗时 = sum
并行：|── wait speculative ──|                       总耗时 = max
      |── Whisper final ─────|

推测未完成（推测还要 1.5s、Whisper 2s）：串行 3.5s → 并行 2s（-1.5s）
推测超时（推测还要 5s、Whisper 2s）：    串行 7s  → 并行 5s（-2s）
```

#### Batch 2 验收标准

- WebSocket 流式翻译端到端可用
- **延迟验证：** `_handle_stop` 入口/出口打时间戳，3 次测试取平均值
- 日志可见多次 "Firing speculative LLM" + "cancelled (superseded)"
- "Reusing cached LLM result" 频率显著高于优化前
- 330 个现有测试全部通过
- 新增测试：滚动推测取消/重启、`CancelledError` 优雅处理、并行 stop 耗时验证、缓存命中/未命中分支

---

## 4. 修改文件矩阵

| 文件 | Batch 1 | Batch 2 | 改动性质 |
|------|---------|---------|---------|
| `app/core/api_client.py` | ✅ 新增 2 个 async 工厂函数 | — | 新增函数，不改已有函数 |
| `app/services/streaming_transcription_service.py` | ✅ sync→async client | — | transport 层替换 |
| `app/services/sound_selection_service.py` | ✅ sync→async client | — | transport 层替换 |
| `app/api/ws_endpoints.py` | — | ✅ 滚动推测 + 并行 stop | 逻辑重构 |
| `tests/test_api_client.py` | ✅ 新增 async 工厂测试 | — | 新增测试方法 |
| `tests/test_phase9_*.py` | — | ✅ 新增测试文件 | 覆盖滚动推测 + 并行 stop |

不改动：Legacy Pipeline 全部服务文件（5 个）、`app/services/sample_matcher.py`（纯内存计算）、`app/core/config.py`、所有前端文件、所有现有测试文件。

---

## 5. 测试策略

### Batch 1

| 测试范围 | 方法 |
|---------|------|
| `get_async_openai_client()` | mock `settings.OPENAI_API_KEY`，断言返回 `AsyncOpenAI` 且 `api_key` 正确 |
| `get_async_instructor_client()` | 断言返回 `AsyncInstructor`，默认 `Mode.TOOLS` |
| `_call_whisper()` | mock `AsyncOpenAI.audio.transcriptions.create` 为 async，验证 `await` 调用 |
| `generate_target_tags()` | mock async instructor `create`，验证 `await` 调用 + TargetTagSet 解析 |
| 回归 | 330 个现有测试全部通过 |

### Batch 2

| 测试范围 | 方法 |
|---------|------|
| 滚动推测 — 旧任务取消 | 两次中间转录（均 ≥5 词），断言第一个 task 被 cancel |
| 滚动推测 — 新任务启动 | 断言第二次中间转录触发新的 `asyncio.create_task()` |
| `CancelledError` 处理 | 模拟 cancel，断言无异常泄漏、不发 error 消息 |
| 并行 stop — 耗时 | mock `transcribe_final` sleep(1s) + speculative_task sleep(2s)，断言总耗时 ≈ 2s 而非 3s |
| 并行 stop — 推测命中 | 相似度 > 0.7，断言复用缓存 |
| 并行 stop — 推测未命中 | 不相似文本，断言调用新的 `generate_target_tags()` |
| 回归 | 330 个现有测试 + Batch 1 新增测试全部通过 |

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| `instructor.from_openai(AsyncOpenAI)` 与 sync 版行为微妙不同 | 低 | LLM 返回解析失败 | Batch 1 集成测试覆盖；`generate_target_tags` 有 fallback 默认标签 |
| 滚动推测导致过多 LLM API 调用（每 2.5s 一次） | 中 | API 费用增加 | 可加去抖：仅在 SequenceMatcher 对比上次推测文本差异 > 0.3 时重新触发 |
| `asyncio.CancelledError` 在 `instructor` 内部被捕获方式不符预期 | 低 | 推测任务 cancel 失败 | 在 `_speculative_analysis` 外层显式 catch，确保不逃逸 |
| 现有 mock 了 `get_openai_client` 的测试受影响 | 低 | 测试误报 | 新增 async 函数为独立符号，不替换已有函数，现有 mock 不受影响 |

---

## 7. 优化效果预期

| 场景 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| 短语句（< 5s）、推测命中 | ~1.5s | ~1.5s | — |
| 中等语句（5-10s）、推测曾命中但文本偏移 | ~4.5s | ~2s | **-2.5s** |
| 长语句（10-15s）、推测首次触发后文本大幅变化 | ~6s | ~2s | **-4s** |
| 最差情况（推测超时 + cache miss） | ~11s | ~3-4s | **-7s** |

核心改善来源：
1. async 释放 event loop → 并行执行成为现实（-0 ~ -2s）
2. 滚动推测 → cache 命中率大幅提升（-2 ~ -3s）
3. 并行 stop → Whisper 与推测等待重叠（-0 ~ -2s）
