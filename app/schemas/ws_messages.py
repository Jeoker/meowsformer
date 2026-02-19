"""
Meowsformer — WebSocket Message Schemas
=========================================
Pydantic models for the WebSocket streaming translation protocol.

Includes:
- Client → Server message types (config, stop)
- Server → Client message types (transcription, analysis_preview, result, error)
- TargetTagSet — LLM output describing the ideal cat sound
- TaggedSampleInfo — matched sample details returned to the client
- StreamingTranslationResult — final result payload
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── LLM Target-Tag Output ────────────────────────────────────────────────


class TargetTagSet(BaseModel):
    """LLM outputs: what tags should the ideal response sound have?

    The LLM does NOT select a sample — it describes the *ideal* cat sound
    in terms of multi-dimensional tags.  The matching engine then finds
    the best real sample.
    """

    emotion: list[str] = Field(
        default_factory=list,
        description="Desired emotion tags (e.g. ['lonely', 'anxious']).",
    )
    intent: list[str] = Field(
        default_factory=list,
        description="Desired intent tags (e.g. ['seeking_companionship']).",
    )
    acoustic: list[str] = Field(
        default_factory=list,
        description="Desired acoustic tags (e.g. ['prolonged', 'soft', 'falling_tone']).",
    )
    social_context: list[str] = Field(
        default_factory=list,
        description="Desired social context tags (e.g. ['alone_at_home']).",
    )
    reasoning: str = Field(
        default="",
        description="LLM reasoning in Chinese explaining the tag selection.",
    )


# ── Matched Sample Info ──────────────────────────────────────────────────


class TaggedSampleInfo(BaseModel):
    """Information about the matched sample returned to the client."""

    sample_id: str = Field(..., description="Sample ID from the registry.")
    breed: str = Field(..., description="Breed of the sample cat.")
    context: str = Field(..., description="Recording context (Food/Isolation/Brushing).")
    tags: dict[str, list[str]] = Field(
        default_factory=dict,
        description="All tags on this sample, keyed by dimension.",
    )
    match_score: float = Field(
        ..., ge=0.0, le=1.0, description="Weighted match score [0, 1]."
    )
    matched_tags: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Tags that overlapped with the target per dimension.",
    )


# ── Streaming Translation Result ─────────────────────────────────────────


class StreamingTranslationResult(BaseModel):
    """Final result payload sent over the WebSocket."""

    transcription: str = Field(..., description="Final transcribed text.")
    target_tags: TargetTagSet = Field(
        ..., description="Target tags the LLM requested."
    )
    selected_sample: TaggedSampleInfo = Field(
        ..., description="The sample that best matched the target tags."
    )
    audio_base64: str = Field(
        ..., description="Base64-encoded WAV of the selected cat sound."
    )
    reasoning: str = Field(
        ..., description="LLM reasoning for the tag selection."
    )


# ── WebSocket Protocol Messages ──────────────────────────────────────────


class WSConfigMessage(BaseModel):
    """Client → Server: optional configuration on connect."""

    type: Literal["config"] = "config"
    breed_preference: Optional[str] = Field(
        default=None,
        description="Optional breed preference (e.g. 'Maine Coon').",
    )


class WSStopMessage(BaseModel):
    """Client → Server: user finished speaking."""

    type: Literal["stop"] = "stop"


class WSTranscriptionMessage(BaseModel):
    """Server → Client: partial or final transcription."""

    type: Literal["transcription"] = "transcription"
    text: str = Field(..., description="Transcription text so far.")
    is_final: bool = Field(
        default=False, description="True if this is the final transcription."
    )


class WSAnalysisPreviewMessage(BaseModel):
    """Server → Client: speculative LLM analysis preview."""

    type: Literal["analysis_preview"] = "analysis_preview"
    emotion: str = Field(..., description="Predicted primary emotion.")
    intent: str = Field(..., description="Predicted primary intent.")


class WSResultMessage(BaseModel):
    """Server → Client: final result with audio."""

    type: Literal["result"] = "result"
    transcription: str
    selected_category: TaggedSampleInfo
    audio_base64: str
    reasoning: str


class WSErrorMessage(BaseModel):
    """Server → Client: error message."""

    type: Literal["error"] = "error"
    detail: str
