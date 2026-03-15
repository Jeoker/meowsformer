#!/usr/bin/env python3
"""
验证 API Key 与 ai-builders 配置是否正确。

用法:
  python tools/test_api_key.py

会依次检查:
  1. 当前 API 供应方与 token 配置
  2. Chat 接口连通性（LLM）
  3. 可选：Whisper 转录接口（通过 TEST_WAV=/path/to/audio.wav 指定）

ai-builders 配置要点:
  - .env 中需设置: API_PROVIDER=ai_builders, AI_BUILDER_TOKEN=sk_c...
  - AI_BUILDER_BASE_URL 必须指向正确的 API 地址（若返回 HTML 错误页则说明 URL 错误）
  - 若 Chat 成功但 Whisper 失败，可考虑 LLM 用 ai-builders、Whisper 用 OpenAI
"""

from __future__ import annotations

import os
import sys

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    from app.core.config import settings

    print("=" * 60)
    print("Meowsformer API Key 配置检查")
    print("=" * 60)

    # 1. 配置摘要
    print("\n[1] 当前配置:")
    print(f"    API_PROVIDER     = {settings.API_PROVIDER!r}")
    if settings.API_PROVIDER == "ai_builders":
        token = settings.AI_BUILDER_TOKEN
        token_display = f"{token[:12]}...{token[-4:]}" if len(token) > 16 else "(未设置或过短)"
        print(f"    AI_BUILDER_TOKEN = {token_display}")
        print(f"    AI_BUILDER_BASE_URL = {settings.AI_BUILDER_BASE_URL!r}")
        print(f"    LLM_MODEL       = {settings.LLM_MODEL!r}")
        if not token or token == "" or token == "sk-placeholder":
            print("\n    ⚠️  AI_BUILDER_TOKEN 未设置或无效，请在 .env 中配置")
            sys.exit(1)
    else:
        token = settings.OPENAI_API_KEY
        token_display = f"{token[:12]}...{token[-4:]}" if len(token) > 16 else "(未设置或过短)"
        print(f"    OPENAI_API_KEY  = {token_display}")
        if token == "sk-placeholder" or not token:
            print("\n    ⚠️  OPENAI_API_KEY 未设置或为占位符，请在 .env 中配置")
            sys.exit(1)

    # 2. 测试 Chat 接口（LLM）
    print("\n[2] 测试 Chat 接口 (LLM)...")
    try:
        from app.core.api_client import get_openai_client

        client = get_openai_client()
        resp = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": "请用一句长句回复：哟呵，是你啊"}],
            max_tokens=50,
        )
        content = resp.choices[0].message.content if resp.choices else ""
        print(f"    ✓ 成功: {content[:80]!r}...")
    except Exception as e:
        print(f"    ✗ 失败: {e}")
        print("\n    可能原因:")
        if settings.API_PROVIDER == "ai_builders":
            print("    - AI_BUILDER_TOKEN 无效或已过期")
            print("    - AI_BUILDER_BASE_URL 不正确")
            print("    - LLM_MODEL 不在 ai-builders 支持列表中")
        else:
            print("    - OPENAI_API_KEY 无效或已过期")
        sys.exit(1)

    # 3. 可选：测试 Whisper（通过环境变量 TEST_WAV=/path/to/audio.wav 指定）
    test_wav = os.environ.get("TEST_WAV", "").strip()
    if not test_wav:
        test_wav = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "tools", "data", "test-audio.wav",
        )
    print(f"test_wav: {test_wav}")
    if os.path.exists(test_wav) and os.path.getsize(test_wav) > 0:
        print("\n[3] 测试 Whisper 转录接口...")
        try:
            from app.core.api_client import get_openai_client

            client = get_openai_client()
            with open(test_wav, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text",
                )
            text = result if isinstance(result, str) else getattr(result, "text", str(result))
            print(f"    ✓ 成功: 转录结果 = {text[:80]!r}...")
        except Exception as e:
            print(f"    ✗ 失败: {e}")
            if settings.API_PROVIDER == "ai_builders":
                print("\n    注意: ai-builders 的 /v1/audio/transcriptions 可能与 OpenAI 略有差异")
                print("    若 Chat 成功但 Whisper 失败，可考虑临时切回 API_PROVIDER=openai 做转录")
    else:
        print("\n[3] 跳过 Whisper 测试（无测试音频）")

    print("\n" + "=" * 60)
    print("配置检查完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
