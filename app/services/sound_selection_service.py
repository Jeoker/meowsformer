"""
Meowsformer — Sound Selection Service
=======================================
LLM-based target-tag generation + speculative execution.

Key design:
- LLM does NOT select a sample.  It outputs a ``TargetTagSet`` describing
  the cat vocalisation that best **translates** the owner's utterance.
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

from app.core.api_client import get_async_instructor_client
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

# ── Instructor client (lazy-initialized) ─────────────────────────────────

_client: Optional[instructor.AsyncInstructor] = None


def _get_client() -> instructor.AsyncInstructor:
    global _client
    if _client is None:
        _client = get_async_instructor_client()
    return _client


# ── System prompt with full tag taxonomy ─────────────────────────────────

_SYSTEM_PROMPT = """You are a cat bioacoustics specialist and translation analyst.
Translate the owner's transcribed words into the multidimensional traits a cat
vocalisation would need to express a similar core emotion and communicative
intent. This is semantic translation, not a prediction of how a cat would react.

Return target tags for an equivalent feline expression. Valid tags are:

**emotion**: {emotion_tags}

**intent**: {intent_tags}

**acoustic**: {acoustic_tags}

**social_context**: {social_context_tags}

**breed_voice** (optional): {breed_voice_tags}

Rules:
1. Choose 1–3 of the most relevant tags for each applicable dimension.
2. For acoustic tags, infer a plausible pitch, duration, energy, and contour.
3. Leave breed_voice empty unless the owner explicitly requests a breed.
4. Write the reasoning field in concise, natural English for an end user.
5. Preserve the owner's central emotion and intent while keeping the proposed
   vocal character plausible within feline social signalling.
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
        Owner's transcribed speech (source utterance to translate).

    Returns
    -------
    TargetTagSet
        Multi-dimensional target tags for the cat vocalisation that best
        expresses the same meaning as ``text`` (translation, not reply).
    """
    client = _get_client()

    user_prompt = (
        f'Owner transcript: "{text}"\n\n'
        "Perform the semantic translation and return the target tags."
    )

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
        # Return a sensible default
        return TargetTagSet(
            emotion=["calm"],
            intent=["expressing_comfort"],
            acoustic=["mid_pitch", "medium_length"],
            social_context=["near_owner"],
            reasoning=f"The language analysis was unavailable, so a calm default match was used: {e}",
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
        top_k=5,
    )

    if not matches:
        logger.warning("No match found for target tags")
        return None

    for best in matches:
        sample = best.sample
        wav_path = ASSETS_DIR / sample.file_path
        if not wav_path.exists():
            logger.warning("WAV file not found: {}", wav_path)
            continue
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

    logger.warning(
        "No on-disk WAV for any of top {} scoring samples — corpus may be missing",
        len(matches),
    )
    return None
