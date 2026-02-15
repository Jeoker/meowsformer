# Meowsformer 运行与接口验证指南

本文档说明如何运行 Meowsformer 系统，以及如何验证当前上线接口是否满足计划功能。

---

## 一、系统运行步骤

### 1.1 环境准备

| 依赖项 | 要求 | 验证命令 |
|--------|------|----------|
| Python | 3.10 或 3.11（推荐） | `python --version` |
| FFmpeg | 已安装且在 PATH | `ffmpeg -version` |
| OpenAI API Key | 有效密钥 | 见下方 .env 配置 |

### 1.2 安装与配置

```bash
# 1. 进入项目目录
cd /home/jeoker/code/meowsformer

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate   # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量（.env 已存在则检查内容）
# 确保包含：
# OPENAI_API_KEY=sk-your-key-here
# CHROMA_DB_PATH=./db/chroma_db
# DEBUG_MODE=True

# 5. 下载音频语料库（Phase 1 必需，用于 DSP 引擎）
python -m tools.download_datasets
# 若已下载，仅重建索引：
# python -m tools.download_datasets --skip-download
```

### 1.3 启动服务

```bash
# 方式一：直接运行
python main.py

# 方式二：使用 uvicorn（支持热重载）
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

服务启动后：
- **API 文档（Swagger）**：http://localhost:8000/docs
- **ReDoc**：http://localhost:8000/redoc
- **健康检查**：http://localhost:8000/health

---

## 二、当前上线接口清单

| 接口 | 方法 | 路径 | 功能 |
|------|------|------|------|
| 健康检查 | GET | `/health` | 服务存活检测 |
| 猫叫翻译 | POST | `/api/translate` | 音频上传 → 转录 → RAG → LLM 分析 → 返回结构化文本 |

---

## 三、接口功能验证方案

### 3.1 健康检查

```bash
curl http://localhost:8000/health
```

**预期响应：**
```json
{"status": "ok", "app": "MeowTranslator"}
```

### 3.2 主接口 `POST /api/translate` 验证

#### 方式 A：使用 Swagger UI（推荐）

1. 打开 http://localhost:8000/docs
2. 找到 `POST /api/translate`
3. 点击 "Try it out"
4. 上传一个音频文件（支持 WAV、MP3、M4A 等，FFmpeg 可解码格式）
5. 点击 "Execute"

#### 方式 B：使用 curl

```bash
curl -X POST "http://localhost:8000/api/translate" \
  -H "accept: application/json" \
  -F "file=@/path/to/your/audio.wav"
```

#### 方式 C：使用 Python 脚本

```python
import requests

url = "http://localhost:8000/api/translate"
with open("test_audio.wav", "rb") as f:
    response = requests.post(url, files={"file": ("test.wav", f, "audio/wav")})

print(response.status_code)
print(response.json())
```

**预期响应结构（200 OK）：**
```json
{
  "sound_id": "purr_happy_01",
  "pitch_adjust": 1.0,
  "human_interpretation": "I'm hungry!",
  "emotion_category": "Hungry",
  "behavior_note": "Short meow indicating demand."
}
```

### 3.3 验证检查清单

| 检查项 | 预期结果 | 验证方法 |
|--------|----------|----------|
| 服务启动无报错 | 日志显示 "Knowledge base initialized successfully" | 查看终端输出 |
| `/health` 返回 200 | `{"status": "ok"}` | curl 或浏览器 |
| 上传有效音频 | 200 + JSON 含 `sound_id`, `human_interpretation`, `emotion_category`, `behavior_note` | Swagger 或 curl |
| 上传无效/空文件 | 4xx 或 5xx，有明确错误信息 | 故意上传错误格式 |
| 字段类型正确 | `pitch_adjust` 在 0.8–1.5，`emotion_category` 为 Hungry/Angry/Happy/Alert 之一 | 检查 JSON 结构 |

---

## 四、计划功能 vs 当前实现（差距分析）

### 4.1 项目目标（来自 PROJECT_STATUS.md）

> 将人类语音输入，通过语义识别、情绪分析及生物声学 RAG，**翻译成对应的真实猫叫声**。

### 4.2 当前实现 vs 计划

| 功能模块 | 计划 | 当前状态 | 差距 |
|----------|------|----------|------|
| 语音转文字 | Whisper 转录 | ✅ 已实现 | 无 |
| 意图分析 | LLM 分析情绪/意图 | ✅ 已实现 | 无 |
| 生物声学 RAG | ChromaDB 检索上下文 | ✅ 已实现 | 无 |
| **猫叫合成** | DSP 引擎生成 WAV | ❌ **未接入 API** | **Phase 3 待完成** |
| **返回音频** | 响应中包含合成猫叫 | ❌ **仅返回 JSON 文本** | **Phase 3 待完成** |

### 4.3 结论

- **当前接口能实现：** 人类语音 → 转录 → 意图/情绪分析 → 文本解释（`human_interpretation`、`emotion_category`、`behavior_note`）
- **当前接口不能实现：** 返回合成的猫叫音频文件

DSP 引擎（`src/engine/dsp_processor.synthesize_meow()`）已开发完成并通过 45 个单元测试，但尚未接入 `POST /api/translate`。要达成“人类语音 → 真实猫叫声”的完整目标，需要完成 Phase 3 集成。

---

## 五、自动化验证脚本（可选）

可将以下内容保存为 `scripts/verify_api.sh` 便于快速验证：

```bash
#!/bin/bash
BASE_URL="${1:-http://localhost:8000}"

echo "=== 1. Health Check ==="
curl -s "$BASE_URL/health" | jq .

echo -e "\n=== 2. API Docs Available ==="
curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/docs" 
echo " (expect 200)"

echo -e "\n=== 3. Translate Endpoint (requires audio file) ==="
echo "Use: curl -X POST $BASE_URL/api/translate -F 'file=@your_audio.wav'"
```

---

## 六、单元测试（回归验证）

在修改代码后，建议运行完整测试套件确保无回归：

```bash
export PYTHONPATH=$PYTHONPATH:.
python -m unittest discover tests
```

预期：约 90 个测试全部通过。
