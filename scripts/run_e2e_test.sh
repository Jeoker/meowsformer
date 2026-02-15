#!/bin/bash
# Meowsformer 端到端上线测试脚本
# 使用 Docker 确保 Python 3.11 环境（chromadb 与 Python 3.14 不兼容）

set -e
cd "$(dirname "$0")/.."

echo "=== Meowsformer E2E 上线测试 ==="

# 检查 .env
if [ ! -f .env ] || ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo "❌ 请确保 .env 中存在有效的 OPENAI_API_KEY"
    exit 1
fi

# 创建测试音频（若不存在）
TEST_AUDIO="/tmp/test_meow_input.wav"
if [ ! -f "$TEST_AUDIO" ]; then
    echo "创建测试音频..."
    ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -ar 16000 -ac 1 -c:a pcm_s16le "$TEST_AUDIO" 2>/dev/null
fi

# 使用 Docker 运行（若可用）
if command -v docker &>/dev/null; then
    echo "使用 Docker 启动服务..."
    docker build -t meowsformer:test . 2>/dev/null
    docker run -d --name meowsformer-test -p 8000:8000 --env-file .env meowsformer:test
    trap "docker stop meowsformer-test; docker rm meowsformer-test" EXIT
    echo "等待服务启动..."
    sleep 8
else
    echo "Docker 未安装，尝试使用本地 Python..."
    if ! python -c "import chromadb" 2>/dev/null; then
        echo "❌ chromadb 无法在当前 Python 加载（需 Python 3.10/3.11）"
        echo "   请安装 Docker 或使用 conda: conda create -n meows python=3.11"
        exit 1
    fi
    python main.py &
    SERVER_PID=$!
    trap "kill $SERVER_PID 2>/dev/null" EXIT
    sleep 8
fi

# 1. 健康检查
echo ""
echo "--- 1. 健康检查 GET /health ---"
HEALTH=$(curl -s http://localhost:8000/health)
echo "$HEALTH"
if echo "$HEALTH" | grep -q '"status":"ok"'; then
    echo "✅ 健康检查通过"
else
    echo "❌ 健康检查失败"
    exit 1
fi

# 2. 翻译接口
echo ""
echo "--- 2. 翻译接口 POST /api/translate ---"
RESP=$(curl -s -w "\n%{http_code}" -X POST "http://localhost:8000/api/translate" \
  -F "file=@$TEST_AUDIO")

HTTP_CODE=$(echo "$RESP" | tail -1)
BODY=$(echo "$RESP" | sed '$d')

echo "HTTP $HTTP_CODE"
echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"

if [ "$HTTP_CODE" = "200" ]; then
    if echo "$BODY" | grep -q '"human_interpretation"'; then
        echo "✅ 翻译接口通过"
    else
        echo "⚠️ 返回 200 但结构异常"
    fi
else
    echo "❌ 翻译接口失败 (HTTP $HTTP_CODE)"
    exit 1
fi

echo ""
echo "=== 上线测试完成 ==="
