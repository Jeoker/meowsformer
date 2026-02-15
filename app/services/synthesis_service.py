"""
Meowsformer — Synthesis Integration Service (Phase 3)
======================================================
Bridges the Phase 0 LLM analysis with the Phase 2 DSP engine.

Responsibilities:
- Map the LLM ``emotion_category`` to a bioacoustic intent label.
- Invoke ``synthesize_meow`` from the DSP engine.
- Generate the NatureLM-audio-style preview description.
- Encode the output audio as base64 WAV for frontend playback.

This module follows the Service Layer Pattern: all business logic lives
here, keeping the API endpoint thin.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger

from app.schemas.translation import (
    CatTranslationResponse,
    MeowSynthesisResponse,
    PreviewDescriptionSchema,
    SynthesisMetadata,
)
from src.engine.description_generator import (
    PreviewDescription,
    generate_description_from_synthesis,
)
from src.engine.dsp_processor import (
    ASSETS_DIR,
    SampleMatch,
    VAPoint,
    apply_prosody_transform,
    get_best_match,
    map_intent_to_va,
)

# ── Emotion → Intent mapping ─────────────────────────────────────────────
# The Phase 0 LLM returns a coarse emotion_category; we map it to the
# finer-grained bioacoustic intents used by the DSP engine's VA space.
EMOTION_TO_INTENT: dict[str, str] = {
    "Hungry": "Requesting",
    "Angry": "Agonistic",
    "Happy": "Affiliative",
    "Alert": "Alert",
}

# ── Supported output sample rates ────────────────────────────────────────
VALID_SAMPLE_RATES = {16000, 44100}
DEFAULT_OUTPUT_SR = 16000  # Web / telephony standard


def _emotion_to_intent(emotion_category: str) -> str:
    """Map an LLM emotion category to a DSP intent label.

    Falls back to ``"Neutral"`` for unknown categories.
    """
    intent = EMOTION_TO_INTENT.get(emotion_category, "Neutral")
    logger.debug("Emotion '{}' → Intent '{}'", emotion_category, intent)
    return intent


def _encode_audio_base64(
    audio: np.ndarray,
    sr: int,
    *,
    target_sr: int = DEFAULT_OUTPUT_SR,
) -> str:
    """Encode audio array as a base64 WAV string.

    Resamples to ``target_sr`` if the source rate differs, ensuring
    consistent 16 kHz or 44.1 kHz output per the project spec.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio signal (float32).
    sr : int
        Source sample rate.
    target_sr : int
        Desired output sample rate (must be 16000 or 44100).

    Returns
    -------
    str
        Base64-encoded WAV bytes.
    """
    if target_sr not in VALID_SAMPLE_RATES:
        logger.warning(
            "Requested sr={} not in {}; defaulting to {}",
            target_sr,
            VALID_SAMPLE_RATES,
            DEFAULT_OUTPUT_SR,
        )
        target_sr = DEFAULT_OUTPUT_SR

    # Resample if needed
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Write to in-memory WAV buffer
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)

    encoded = base64.b64encode(buf.read()).decode("ascii")
    logger.debug("Encoded audio: {} bytes base64 @ {} Hz", len(encoded), sr)
    return encoded


def _preview_to_schema(desc: PreviewDescription) -> PreviewDescriptionSchema:
    """Convert the engine's dataclass to the Pydantic schema."""
    return PreviewDescriptionSchema(
        summary=desc.summary,
        intent_label=desc.intent_label,
        vocalisation_type=desc.vocalisation_type,
        confidence_score=desc.confidence_score,
        confidence_level=desc.confidence_level,
        va_distance=desc.va_distance,
        pitch_description=desc.pitch_description,
        tempo_description=desc.tempo_description,
        breed=desc.breed,
        source_context=desc.source_context,
        detail=desc.detail,
    )


async def synthesize_and_describe(
    llm_result: CatTranslationResponse,
    *,
    breed: str = "Default",
    output_sr: int = DEFAULT_OUTPUT_SR,
) -> MeowSynthesisResponse:
    """Run the full Phase 3 synthesis pipeline.

    1. Map LLM emotion → bioacoustic intent.
    2. Intent → VA coordinates → nearest-neighbour retrieval.
    3. PSOLA prosody transform.
    4. Generate NatureLM-audio-style preview description.
    5. Encode audio as base64 WAV.

    If synthesis fails (e.g. missing audio files), the response still
    includes the Phase 0 LLM fields with ``synthesis_ok=False``.

    Parameters
    ----------
    llm_result : CatTranslationResponse
        The Phase 0 LLM analysis result.
    breed : str
        Target cat breed for synthesis.
    output_sr : int
        Output sample rate (16000 or 44100).

    Returns
    -------
    MeowSynthesisResponse
        Combined response with LLM analysis + audio + description.
    """
    # Start with the base LLM fields (always present)
    base_fields = {
        "sound_id": llm_result.sound_id,
        "pitch_adjust": llm_result.pitch_adjust,
        "human_interpretation": llm_result.human_interpretation,
        "emotion_category": llm_result.emotion_category,
        "behavior_note": llm_result.behavior_note,
    }

    try:
        # ── 1. Map emotion → intent ──────────────────────────────────
        intent = _emotion_to_intent(llm_result.emotion_category)

        # ── 2. Intent → VA → retrieval ──────────────────────────────
        target_va = map_intent_to_va(intent)
        matches = get_best_match(
            target_va.valence,
            target_va.arousal,
            top_k=1,
        )

        if not matches:
            logger.warning("No matching samples found in registry")
            return MeowSynthesisResponse(**base_fields, synthesis_ok=False)

        best = matches[0]
        wav_path = ASSETS_DIR / best.file_path

        if not wav_path.exists():
            logger.warning("Audio file not found: {}", wav_path)
            return MeowSynthesisResponse(**base_fields, synthesis_ok=False)

        # ── 3. Compute prosody deltas (same as synthesize_meow) ──────
        va_pitch_hint = (target_va.valence - best.valence) * 2.0
        duration_factor = 1.0 + (best.arousal - target_va.arousal) * 0.5

        # ── 4. PSOLA transform ───────────────────────────────────────
        audio, sr = apply_prosody_transform(
            audio_path=wav_path,
            target_pitch_shift=va_pitch_hint,
            duration_factor=duration_factor,
            breed=breed,
            arousal=target_va.arousal,
        )

        # ── 5. Generate preview description ──────────────────────────
        preview_desc = generate_description_from_synthesis(
            intent=intent,
            match=best,
            breed=breed,
        )

        # ── 6. Encode as base64 WAV ──────────────────────────────────
        audio_b64 = _encode_audio_base64(audio, sr, target_sr=output_sr)

        # ── 7. Build response ────────────────────────────────────────
        duration_seconds = round(len(audio) / sr, 3) if sr > 0 else 0.0

        return MeowSynthesisResponse(
            **base_fields,
            audio_base64=audio_b64,
            preview_description=_preview_to_schema(preview_desc),
            synthesis_metadata=SynthesisMetadata(
                matched_sample_id=best.sample_id,
                matched_breed=best.breed,
                matched_context=best.context,
                target_valence=round(target_va.valence, 4),
                target_arousal=round(target_va.arousal, 4),
                duration_seconds=duration_seconds,
                sample_rate=output_sr,
            ),
            synthesis_ok=True,
        )

    except Exception as exc:
        logger.error("Synthesis pipeline failed: {}", exc)
        return MeowSynthesisResponse(**base_fields, synthesis_ok=False)
