# Meowsformer — 测试文档

---

## 运行方式

```bash
export PYTHONPATH=$PYTHONPATH:.
python -m unittest discover tests
```

> **兼容性:** 推荐 Python 3.12。Python 3.14 与 `chromadb` 的 Pydantic V1 依赖存在兼容性问题。

**Phase 8 Web UI（`src/ui/`）：** 当前无独立前端单元测试；Sprint 验收以浏览器手动联调为准（`npm run dev` + `python main.py`）。详见 [phase8-batch-ui-2026-04-06.md](./batch-reports/phase8-batch-ui-2026-04-06.md)。

---

## 测试基础设施文件

`tests/` 目录下有三个非测试文件，供各 Batch 测试文件共享使用：

| 文件 | 职责 |
|------|------|
| `tests/shared_params.py` | 纯常量：API provider 字符串、LLM 模型名、AI Builders URL、ChromaDB 路径、样本 ID、音频 stub、匹配分数、路由路径、品种偏好、异步等待时长等 |
| `tests/flet_mocks.py` | flet 控件 mock 类（`_Ctrl` / `_TextCtrl` / `_ListCtrl`）、`install_flet_mock()` 安装函数、统一的 `BaseMockPage`（合并 batch2/3/4 三种 `_MockPage` 变体，超集兼容） |
| `tests/ws_stubs.py` | WebSocket 测试桩：`MockWebSocket`、`ws_connect_coro()`、`async_chunks()` |

**设计约束：**
- `shared_params.py` 不含 app 模块导入，以纯值常量为主，含 `json.dumps` 等标准库调用
- `flet_mocks.py` 与 `ws_stubs.py` 不包含断言逻辑，仅提供基础设施
- 各测试文件内部仍可定义只在本文件出现 2+ 次的私有常量（如 `_FAKE_WAV_PATH`、SK key stubs）

---

## 测试总览

| 测试文件 | 模块 | 用例数 | 描述 |
|----------|------|--------|------|
| `test_api_endpoints.py` | API | 1 | 可运行（Batch 1 修复了连锁导入失败） |
| `test_audio_services.py` | Audio | 2 | FFmpeg 转换与特征提取（独立函数，非 unittest.TestCase） |
| `test_llm_service.py` | LLM | 1 | `analyze_intention`, mock OpenAI |
| `test_rag_service.py` | RAG | 2 | 知识库初始化, 上下文检索 |
| `test_download_datasets.py` | Data | 46 | 文件名解析, registry 构建（9 个测试类） |
| `test_dsp_processor.py` | DSP | 45 | VA 映射, 音频检索, f0, PSOLA, 包络 |
| `test_description_generator.py` | Descriptions | 31 | Intent 标签, 置信评分, 预览生成 |
| `test_synthesis_service.py` | Synthesis | 15 | emotion→intent, base64, 管线, 降级 |
| `test_auth.py` | Auth | — | 未实现（模块不存在，暂缓） |
| `test_batch1_auth_removal.py` | Auth Removal | 24 | Batch 1 验证: 导入完整性、路由注册、Health 端点、配置无 JWT、源码无 auth 残留、Vite proxy |
| `test_batch2_ws_streaming.py` | WS Streaming | 31 | Batch 2 验证: 并发通信、Config/Stop、TaskGroup 异常、JSONDecodeError、Phase 5 标签、WS 事件回调、chunk generator |
| `test_batch3_audio_playback.py` | Audio Playback | 25 | Batch 3 + flet-audio 升级验证: fta.Audio overlay 注册、release→update→play 调用顺序、DSP 处理、base64 播放、REST/Streaming 自动播放、dispose 清理 |
| `test_batch4_ux_enhancements.py` | UX Enhancements | 37 | Batch 4 验证: WS 连接状态指示器、WebSocketConnectionError + on_state_change 回调、Snackbar 错误通知、录音计时器 (wall-clock 锚定)、自动降级 REST、历史记录 ExpansionTile 5 维标签、多条记录排序 |
| `test_api_client.py` | API Provider Switch | 42 | 供应方工厂 (`get_openai_client`、`get_instructor_client`)、config 默认值、懒加载缓存隔离、LLM 模型名传播 |
| `test_api_provider_switch.py` | API Provider E2E | 31 | 供应方端到端切换验证、config 条件分支、懒加载隔离、模型名传播 |

**总计: 330 个测试函数，全部可运行。** `unittest discover` 实际运行 330 个。`test_audio_services` 的 2 个为独立函数，非 `TestCase` 子类，不被 discover 发现（需单独运行该文件）。

---

## 测试规范

- **框架：** `unittest`（项目标准，所有测试统一使用）
- **位置：** `tests/test_<module_name>.py`
- **外部服务 mock：** OpenAI (Whisper, GPT-4o)、ChromaDB — 绝不发起真实 API 调用
- **覆盖范围：** 正常路径、空/非法输入、schema 校验、错误处理、async 行为
- **算法测试：** 确定性 fixture、数值容差 (`assertAlmostEqual`)
- **FastAPI 端点：** 使用 `TestClient`，mock 所有 service 依赖
- **硬编码值：** 跨文件重复的常量统一定义在 `tests/shared_params.py`；flet mock 基础设施来自 `tests/flet_mocks.py`；WS 测试桩来自 `tests/ws_stubs.py`
- **Provider 区分：** `MODEL_OPENAI_DEFAULT = "gpt-4o"` 与 `MODEL_AI_BUILDERS_DEFAULT = "deepseek"` 是两个独立常量，不可合并——二者分别对应 openai / ai_builders 两个 provider 的真实 Settings 默认值

---

## 更新规则

每次 test-engineer 完成审核循环后，由 PM 同步更新本文档：
- 新增的测试文件与用例数
- 总计用例数更新
- 覆盖模块变更
