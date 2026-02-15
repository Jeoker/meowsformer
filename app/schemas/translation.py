from typing import Literal, Optional
from pydantic import BaseModel, Field


class CatTranslationResponse(BaseModel):
    """
    Structured output for Cat Translation (Phase 0 — LLM analysis).
    """

    sound_id: str = Field(
        ...,
        description="The ID of the corresponding sound effect (e.g., 'purr_happy_01').",
    )
    pitch_adjust: float = Field(
        ...,
        ge=0.8,
        le=1.5,
        description="Pitch adjustment factor (0.8 - 1.5).",
    )
    human_interpretation: str = Field(
        ...,
        description="Translation text intended for humans.",
    )
    emotion_category: Literal["Hungry", "Angry", "Happy", "Alert"] = Field(
        ...,
        description="Emotion category: Hungry, Angry, Happy, Alert.",
    )
    behavior_note: str = Field(
        ...,
        description="A short biological explanation of the cat's current behavior.",
    )


# ── Phase 3 — Synthesis preview metadata ─────────────────────────────────


class PreviewDescriptionSchema(BaseModel):
    """Human-readable description of the synthesised audio for preview."""

    summary: str = Field(
        ...,
        description="One-sentence preview summary (Chinese) describing the matched vocalisation.",
    )
    intent_label: str = Field(
        ...,
        description="Localised intent label (e.g. '积极求食').",
    )
    vocalisation_type: str = Field(
        ...,
        description="Inferred vocalisation type (e.g. '高频上升调鸣叫').",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Match confidence score [0, 1] from VA-space distance.",
    )
    confidence_level: str = Field(
        ...,
        description="Qualitative confidence level (e.g. '高', '中等').",
    )
    va_distance: float = Field(
        ...,
        ge=0.0,
        description="Raw Euclidean distance in VA space.",
    )
    pitch_description: str = Field(
        ...,
        description="Description of the pitch transformation applied.",
    )
    tempo_description: str = Field(
        ...,
        description="Description of the temporal transformation applied.",
    )
    breed: str = Field(
        ...,
        description="Target cat breed used for synthesis.",
    )
    source_context: str = Field(
        ...,
        description="Recording context of the matched sample (e.g. 'Food').",
    )
    detail: str = Field(
        ...,
        description="Multi-line detailed description of the synthesis pipeline.",
    )


class SynthesisMetadata(BaseModel):
    """Metadata about the DSP synthesis result."""

    matched_sample_id: str = Field(
        ...,
        description="ID of the matched audio sample from the registry.",
    )
    matched_breed: str = Field(
        ...,
        description="Breed of the matched sample.",
    )
    matched_context: str = Field(
        ...,
        description="Recording context of the matched sample.",
    )
    target_valence: float = Field(
        ...,
        description="Target Valence coordinate in VA space.",
    )
    target_arousal: float = Field(
        ...,
        description="Target Arousal coordinate in VA space.",
    )
    duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Duration of the synthesised audio in seconds.",
    )
    sample_rate: int = Field(
        ...,
        description="Sample rate of the output audio (Hz).",
    )


class MeowSynthesisResponse(BaseModel):
    """
    Full Phase 3 response: LLM analysis + synthesised audio + preview description.

    The ``audio_base64`` field contains the WAV audio encoded in base64 for
    direct playback in the frontend preview component. The user should listen
    and confirm before the audio is "sent".
    """

    # ── Phase 0 fields (LLM analysis) ────────────────────────────────
    sound_id: str = Field(
        ...,
        description="LLM-assigned sound effect ID.",
    )
    pitch_adjust: float = Field(
        ...,
        ge=0.8,
        le=1.5,
        description="LLM-suggested pitch adjustment factor.",
    )
    human_interpretation: str = Field(
        ...,
        description="Human-readable translation of the input speech.",
    )
    emotion_category: Literal["Hungry", "Angry", "Happy", "Alert"] = Field(
        ...,
        description="Emotion category from LLM analysis.",
    )
    behavior_note: str = Field(
        ...,
        description="Biological explanation of the cat's behavior.",
    )

    # ── Phase 3 fields (synthesis + preview) ─────────────────────────
    audio_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded WAV audio for preview playback. "
        "None if synthesis was skipped or failed.",
    )
    preview_description: Optional[PreviewDescriptionSchema] = Field(
        default=None,
        description="Human-readable preview description of the synthesised meow.",
    )
    synthesis_metadata: Optional[SynthesisMetadata] = Field(
        default=None,
        description="Technical metadata about the synthesis result.",
    )
    synthesis_ok: bool = Field(
        default=False,
        description="Whether DSP synthesis completed successfully.",
    )
