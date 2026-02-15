"""
Tests for the Meowsformer Synthesis Integration Service (Phase 3)
==================================================================
Covers:

- Emotion → Intent mapping
- Base64 audio encoding / decoding round-trip
- Preview description → Pydantic schema conversion
- Full synthesize_and_describe pipeline (mocked file system)
- Graceful degradation when synthesis fails
- Sample rate validation and resampling
"""

from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import soundfile as sf

# ── Mock chromadb before importing app modules ────────────────────────────
mock_chromadb = MagicMock()
sys.modules.setdefault("chromadb", mock_chromadb)
sys.modules.setdefault("chromadb.utils", MagicMock())
sys.modules.setdefault("chromadb.utils.embedding_functions", MagicMock())

from app.schemas.translation import (
    CatTranslationResponse,
    MeowSynthesisResponse,
    PreviewDescriptionSchema,
    SynthesisMetadata,
)
from app.services.synthesis_service import (
    DEFAULT_OUTPUT_SR,
    EMOTION_TO_INTENT,
    _emotion_to_intent,
    _encode_audio_base64,
    _preview_to_schema,
    synthesize_and_describe,
)
from src.engine.description_generator import PreviewDescription
from src.engine.dsp_processor import SampleMatch, VAPoint


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_llm_result(**overrides) -> CatTranslationResponse:
    """Create a mock CatTranslationResponse."""
    defaults = {
        "sound_id": "purr_happy_01",
        "pitch_adjust": 1.0,
        "human_interpretation": "I'm hungry!",
        "emotion_category": "Hungry",
        "behavior_note": "Short meow indicating demand.",
    }
    defaults.update(overrides)
    return CatTranslationResponse(**defaults)


def _make_sine_audio(
    freq: float = 440.0,
    duration: float = 0.3,
    sr: int = 22050,
) -> tuple[np.ndarray, int]:
    """Generate a short sine wave for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.8 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    return y, sr


def _make_mock_sample_match(**overrides) -> SampleMatch:
    defaults = {
        "sample_id": "TEST_001",
        "file_path": "test/test_001.wav",
        "distance": 0.12,
        "valence": 0.5,
        "arousal": 0.9,
        "breed": "Maine Coon",
        "context": "Food",
        "metadata": {"id": "TEST_001"},
    }
    defaults.update(overrides)
    return SampleMatch(**defaults)


# ════════════════════════════════════════════════════════════════════════
#  1. Emotion → Intent Mapping Tests
# ════════════════════════════════════════════════════════════════════════


class TestEmotionToIntent(unittest.TestCase):
    """Tests for _emotion_to_intent."""

    def test_known_mappings(self) -> None:
        self.assertEqual(_emotion_to_intent("Hungry"), "Requesting")
        self.assertEqual(_emotion_to_intent("Angry"), "Agonistic")
        self.assertEqual(_emotion_to_intent("Happy"), "Affiliative")
        self.assertEqual(_emotion_to_intent("Alert"), "Alert")

    def test_unknown_emotion_falls_back(self) -> None:
        self.assertEqual(_emotion_to_intent("Confused"), "Neutral")
        self.assertEqual(_emotion_to_intent(""), "Neutral")

    def test_all_mapped_emotions_exist(self) -> None:
        """All emotion categories should map to valid DSP intents."""
        from src.engine.dsp_processor import INTENT_VA_MAP

        for emotion, intent in EMOTION_TO_INTENT.items():
            self.assertIn(
                intent,
                INTENT_VA_MAP,
                f"Emotion '{emotion}' maps to unknown intent '{intent}'",
            )


# ════════════════════════════════════════════════════════════════════════
#  2. Base64 Audio Encoding Tests
# ════════════════════════════════════════════════════════════════════════


class TestEncodeAudioBase64(unittest.TestCase):
    """Tests for _encode_audio_base64."""

    def test_round_trip(self) -> None:
        """Encode → decode should recover valid WAV."""
        audio, sr = _make_sine_audio(sr=16000)
        encoded = _encode_audio_base64(audio, sr, target_sr=16000)

        # Decode
        wav_bytes = base64.b64decode(encoded)
        buf = io.BytesIO(wav_bytes)
        decoded_audio, decoded_sr = sf.read(buf)

        self.assertEqual(decoded_sr, 16000)
        self.assertGreater(len(decoded_audio), 0)

    def test_resampling_to_16k(self) -> None:
        """Audio at 22050 Hz should be resampled to 16000 Hz."""
        audio, sr = _make_sine_audio(sr=22050)
        encoded = _encode_audio_base64(audio, sr, target_sr=16000)

        wav_bytes = base64.b64decode(encoded)
        buf = io.BytesIO(wav_bytes)
        _, decoded_sr = sf.read(buf)
        self.assertEqual(decoded_sr, 16000)

    def test_resampling_to_44100(self) -> None:
        """Should support 44.1kHz output."""
        audio, sr = _make_sine_audio(sr=22050)
        encoded = _encode_audio_base64(audio, sr, target_sr=44100)

        wav_bytes = base64.b64decode(encoded)
        buf = io.BytesIO(wav_bytes)
        _, decoded_sr = sf.read(buf)
        self.assertEqual(decoded_sr, 44100)

    def test_invalid_sr_falls_back(self) -> None:
        """Invalid sample rate should fall back to default."""
        audio, sr = _make_sine_audio(sr=16000)
        encoded = _encode_audio_base64(audio, sr, target_sr=48000)

        wav_bytes = base64.b64decode(encoded)
        buf = io.BytesIO(wav_bytes)
        _, decoded_sr = sf.read(buf)
        self.assertEqual(decoded_sr, DEFAULT_OUTPUT_SR)

    def test_output_is_valid_base64(self) -> None:
        """The output should be valid base64."""
        audio, sr = _make_sine_audio(sr=16000)
        encoded = _encode_audio_base64(audio, sr, target_sr=16000)

        # Should not raise
        decoded = base64.b64decode(encoded)
        self.assertGreater(len(decoded), 0)


# ════════════════════════════════════════════════════════════════════════
#  3. Preview → Schema Conversion Tests
# ════════════════════════════════════════════════════════════════════════


class TestPreviewToSchema(unittest.TestCase):
    """Tests for _preview_to_schema."""

    def test_converts_all_fields(self) -> None:
        desc = PreviewDescription(
            summary="Test summary",
            intent_label="积极求食",
            vocalisation_type="高频上升调鸣叫",
            confidence_score=0.85,
            confidence_level="高",
            va_distance=0.12,
            pitch_description="微幅升调",
            tempo_description="节奏略加紧凑",
            breed="Maine Coon",
            source_context="Food",
            detail="Test detail",
        )

        schema = _preview_to_schema(desc)

        self.assertIsInstance(schema, PreviewDescriptionSchema)
        self.assertEqual(schema.summary, "Test summary")
        self.assertEqual(schema.intent_label, "积极求食")
        self.assertEqual(schema.confidence_score, 0.85)
        self.assertEqual(schema.breed, "Maine Coon")


# ════════════════════════════════════════════════════════════════════════
#  4. Full Pipeline Tests (Mocked)
# ════════════════════════════════════════════════════════════════════════


class TestSynthesizeAndDescribe(unittest.IsolatedAsyncioTestCase):
    """Tests for the full synthesize_and_describe pipeline."""

    async def test_successful_synthesis(self) -> None:
        """Full pipeline with mocked DSP engine should succeed."""
        llm_result = _make_llm_result(emotion_category="Hungry")
        mock_audio, mock_sr = _make_sine_audio(sr=22050)
        mock_match = _make_mock_sample_match()

        with (
            patch("app.services.synthesis_service.get_best_match") as mock_get_match,
            patch("app.services.synthesis_service.apply_prosody_transform") as mock_transform,
            patch("app.services.synthesis_service.ASSETS_DIR") as mock_assets,
        ):
            mock_get_match.return_value = [mock_match]
            mock_transform.return_value = (mock_audio, mock_sr)

            # Mock the file existence check
            mock_wav_path = MagicMock()
            mock_wav_path.exists.return_value = True
            mock_assets.__truediv__ = MagicMock(return_value=mock_wav_path)

            result = await synthesize_and_describe(llm_result, breed="Maine Coon")

        self.assertIsInstance(result, MeowSynthesisResponse)
        self.assertTrue(result.synthesis_ok)
        self.assertIsNotNone(result.audio_base64)
        self.assertIsNotNone(result.preview_description)
        self.assertIsNotNone(result.synthesis_metadata)

        # Phase 0 fields preserved
        self.assertEqual(result.sound_id, "purr_happy_01")
        self.assertEqual(result.emotion_category, "Hungry")

        # Metadata
        self.assertEqual(result.synthesis_metadata.matched_sample_id, "TEST_001")
        self.assertEqual(result.synthesis_metadata.sample_rate, DEFAULT_OUTPUT_SR)

    async def test_no_matching_samples(self) -> None:
        """Should degrade gracefully when no samples match."""
        llm_result = _make_llm_result()

        with patch("app.services.synthesis_service.get_best_match") as mock_get:
            mock_get.return_value = []

            result = await synthesize_and_describe(llm_result)

        self.assertFalse(result.synthesis_ok)
        self.assertIsNone(result.audio_base64)
        # Phase 0 fields should still be present
        self.assertEqual(result.sound_id, "purr_happy_01")

    async def test_missing_audio_file(self) -> None:
        """Should degrade gracefully when the audio file doesn't exist."""
        llm_result = _make_llm_result()
        mock_match = _make_mock_sample_match()

        with (
            patch("app.services.synthesis_service.get_best_match") as mock_get,
            patch("app.services.synthesis_service.ASSETS_DIR") as mock_assets,
        ):
            mock_get.return_value = [mock_match]
            mock_wav_path = MagicMock()
            mock_wav_path.exists.return_value = False
            mock_assets.__truediv__ = MagicMock(return_value=mock_wav_path)

            result = await synthesize_and_describe(llm_result)

        self.assertFalse(result.synthesis_ok)

    async def test_dsp_exception_caught(self) -> None:
        """DSP exceptions should be caught, not propagated."""
        llm_result = _make_llm_result()

        with (
            patch("app.services.synthesis_service.get_best_match") as mock_get,
            patch("app.services.synthesis_service.ASSETS_DIR") as mock_assets,
            patch("app.services.synthesis_service.apply_prosody_transform") as mock_transform,
        ):
            mock_match = _make_mock_sample_match()
            mock_get.return_value = [mock_match]
            mock_wav_path = MagicMock()
            mock_wav_path.exists.return_value = True
            mock_assets.__truediv__ = MagicMock(return_value=mock_wav_path)
            mock_transform.side_effect = RuntimeError("PSOLA failed")

            result = await synthesize_and_describe(llm_result)

        self.assertFalse(result.synthesis_ok)
        self.assertEqual(result.human_interpretation, "I'm hungry!")

    async def test_different_emotions(self) -> None:
        """All emotion categories should produce valid responses."""
        mock_audio, mock_sr = _make_sine_audio()
        mock_match = _make_mock_sample_match()

        for emotion in ["Hungry", "Angry", "Happy", "Alert"]:
            llm_result = _make_llm_result(emotion_category=emotion)

            with (
                patch("app.services.synthesis_service.get_best_match") as mock_get,
                patch("app.services.synthesis_service.apply_prosody_transform") as mock_t,
                patch("app.services.synthesis_service.ASSETS_DIR") as mock_a,
            ):
                mock_get.return_value = [mock_match]
                mock_t.return_value = (mock_audio, mock_sr)
                mock_path = MagicMock()
                mock_path.exists.return_value = True
                mock_a.__truediv__ = MagicMock(return_value=mock_path)

                result = await synthesize_and_describe(llm_result)

            self.assertTrue(result.synthesis_ok, f"Failed for emotion: {emotion}")
            self.assertEqual(result.emotion_category, emotion)

    async def test_custom_output_sr(self) -> None:
        """Should respect the output_sr parameter."""
        llm_result = _make_llm_result()
        mock_audio, mock_sr = _make_sine_audio(sr=22050)
        mock_match = _make_mock_sample_match()

        with (
            patch("app.services.synthesis_service.get_best_match") as mock_get,
            patch("app.services.synthesis_service.apply_prosody_transform") as mock_t,
            patch("app.services.synthesis_service.ASSETS_DIR") as mock_a,
        ):
            mock_get.return_value = [mock_match]
            mock_t.return_value = (mock_audio, mock_sr)
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_a.__truediv__ = MagicMock(return_value=mock_path)

            result = await synthesize_and_describe(
                llm_result, output_sr=44100
            )

        self.assertTrue(result.synthesis_ok)
        self.assertEqual(result.synthesis_metadata.sample_rate, 44100)


# ════════════════════════════════════════════════════════════════════════
#  Run
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
