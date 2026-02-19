"""
Meowsformer — Sound Selection Service
=======================================
LLM-based target-tag generation + speculative execution.

Key design:
- LLM does NOT select a sample.  It outputs a ``TargetTagSet`` describing
  what kind of cat sound would be the ideal response.
- The matching engine (``sample_matcher``) finds the best real sample.
- Speculative execution: first LLM call fires on partial text; if the
  final text is similar, the cached result is reused instantly.
"""

from __future__ import annotations

import base64
import io
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import instructor
import soundfile as sf
from loguru import logger
from openai import OpenAI

from app.core.config import settings
from app.data.meow_catalog import TAG_TAXONOMY
from app.schemas.ws_messages import (
    StreamingTranslationResult,
    TaggedSampleInfo,
    TargetTagSet,
)
from app.services.sample_matcher import (
    MatchResult,
    find_best_match,
    get_samples,
)

# ── Assets directory ─────────────────────────────────────────────────────

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"

# ── OpenAI client (instructor-patched) ───────────────────────────────────

_client: Optional[instructor.Instructor] = None


def _get_client() -> instructor.Instructor:
    global _client
    if _client is None:
        _client = instructor.from_openai(OpenAI(api_key=settings.OPENAI_API_KEY))
    return _client


# ── System prompt with full tag taxonomy ─────────────────────────────────

_SYSTEM_PROMPT = """你是一位猫咪生物声学专家和情感分析师。
你的任务是分析用户说的话（已转录为文字），然后判断一只猫听到这段话后应该发出什么样的声音来回应。

你需要输出一组"目标标签"（target tags），描述理想的猫叫声应该具有的特征。
标签分为5个维度，每个维度的有效标签如下：

**emotion** (猫的情绪): {emotion_tags}

**intent** (沟通目的): {intent_tags}

**acoustic** (声学特征): {acoustic_tags}

**social_context** (社交场景): {social_context_tags}

**breed_voice** (品种声线，可选): {breed_voice_tags}

规则：
1. 每个维度选择1-3个最相关的标签
2. acoustic维度：根据语境选择合适的音高、时长、音量、音调特征
3. breed_voice维度可以留空，除非用户明确提到品种偏好
4. reasoning字段用中文解释你的判断逻辑
5. 你的目标是让猫的回应在语义上合理、在声学上自然
""".format(
    emotion_tags=", ".join(TAG_TAXONOMY["emotion"]),
    intent_tags=", ".join(TAG_TAXONOMY["intent"]),
    acoustic_tags=", ".join(TAG_TAXONOMY["acoustic"]),
    social_context_tags=", ".join(TAG_TAXONOMY["social_context"]),
    breed_voice_tags=", ".join(TAG_TAXONOMY["breed_voice"]),
)


# ── LLM Target-Tag Generation ───────────────────────────────────────────


async def generate_target_tags(text: str) -> TargetTagSet:
    """Call LLM to generate target tags for the given transcription.

    Parameters
    ----------
    text : str
        User's transcribed speech.

    Returns
    -------
    TargetTagSet
        Multi-dimensional target tags describing the ideal cat response.
    """
    client = _get_client()

    user_prompt = f"用户对猫说的话: \"{text}\"\n\n请分析并输出目标标签。"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
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
        # Return a sensible default
        return TargetTagSet(
            emotion=["calm"],
            intent=["expressing_comfort"],
            acoustic=["mid_pitch", "medium_length"],
            social_context=["near_owner"],
            reasoning=f"LLM调用失败，使用默认标签: {e}",
        )


# ── Speculative Execution Cache ─────────────────────────────────────────


class SpeculativeCache:
    """Caches a speculative LLM result tied to the text it was based on."""

    def __init__(self) -> None:
        self.cached_text: Optional[str] = None
        self.cached_tags: Optional[TargetTagSet] = None

    def store(self, text: str, tags: TargetTagSet) -> None:
        self.cached_text = text
        self.cached_tags = tags

    def is_similar(self, final_text: str, threshold: float = 0.7) -> bool:
        """Check if the final text is similar enough to reuse the cache.

        Uses SequenceMatcher ratio.  A ratio > threshold means we can
        reuse; otherwise, we need a new LLM call.
        """
        if self.cached_text is None:
            return False
        ratio = SequenceMatcher(None, self.cached_text, final_text).ratio()
        logger.debug(
            "Text similarity: {:.2f} (threshold: {:.2f})",
            ratio,
            threshold,
        )
        return ratio >= threshold

    def get(self) -> Optional[TargetTagSet]:
        return self.cached_tags

    def clear(self) -> None:
        self.cached_text = None
        self.cached_tags = None


# ── End-to-end Selection Flow ────────────────────────────────────────────


def _encode_wav_base64(wav_path: Path) -> str:
    """Read a WAV file and return it as base64."""
    with open(wav_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


async def select_and_encode(
    target_tags: TargetTagSet,
    breed_preference: Optional[str] = None,
) -> Optional[StreamingTranslationResult]:
    """Score all samples, pick the best, encode as base64.

    Parameters
    ----------
    target_tags : TargetTagSet
        LLM-generated target tags.
    breed_preference : str | None
        Optional breed preference.

    Returns
    -------
    StreamingTranslationResult | None
        Full result with audio, or None if no match found.
    """
    matches = find_best_match(
        target_tags=target_tags,
        breed_preference=breed_preference,
        top_k=1,
    )

    if not matches:
        logger.warning("No match found for target tags")
        return None

    best: MatchResult = matches[0]
    sample = best.sample

    # Read the WAV file
    wav_path = ASSETS_DIR / sample.file_path
    if not wav_path.exists():
        logger.warning("WAV file not found: {}", wav_path)
        return None

    audio_b64 = _encode_wav_base64(wav_path)

    return StreamingTranslationResult(
        transcription="",  # Will be filled by the caller
        target_tags=target_tags,
        selected_sample=TaggedSampleInfo(
            sample_id=sample.id,
            breed=sample.breed,
            context=sample.context,
            tags=sample.tags,
            match_score=round(best.score, 4),
            matched_tags=best.matched_tags,
        ),
        audio_base64=audio_b64,
        reasoning=target_tags.reasoning,
    )
