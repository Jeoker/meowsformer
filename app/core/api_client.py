"""
API 供应方切换说明
==================
默认使用 OpenAI 官方 API，日常使用无需任何修改。

切换到 ai-builders 平台：
  1. 在 .env 中设置 API_PROVIDER=ai_builders
  2. 设置 AI_BUILDER_TOKEN=sk_c...（你的 ai-builders token）
  3. LLM_MODEL 会按供应方自动切换：openai→gpt-4o，ai_builders→deepseek
     若需覆盖，可在 .env 中设置 LLM_MODEL=gemini-2.5-pro 等

ai-builders 是 OpenAI 兼容接口，使用相同的 SDK，只需不同的 api_key 和 base_url。
"""

import instructor
from openai import OpenAI

from app.core.config import settings


def get_openai_client() -> OpenAI:
    """Return an OpenAI-compatible client for the configured API provider.

    This factory creates a new instance on every call (no internal cache).
    Callers that serve repeated requests **should** hold a module-level
    cached reference (e.g. via a ``_get_client()`` lazy pattern) to avoid
    re-initialising the HTTP connection pool on every request.
    """
    if settings.API_PROVIDER == "ai_builders":
        return OpenAI(
            api_key=settings.AI_BUILDER_TOKEN,
            base_url=settings.AI_BUILDER_BASE_URL,
        )
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def get_instructor_client() -> instructor.Instructor:
    """Return an instructor-patched OpenAI client for the configured provider.

    This factory creates a new instance on every call (no internal cache).
    Callers that serve repeated requests **should** hold a module-level
    cached reference to avoid reconstructing the instructor wrapper on every
    call.

    When using ai_builders (DeepSeek etc.), the model returns plain JSON in
    content rather than tool calls. We use Mode.JSON so instructor parses
    content directly; otherwise the default Mode.TOOLS triggers
    "Instructor does not support multiple tool calls" when tool_calls is empty.
    """
    client = get_openai_client()
    if settings.API_PROVIDER == "ai_builders":
        return instructor.from_openai(client, mode=instructor.Mode.JSON)
    return instructor.from_openai(client)
