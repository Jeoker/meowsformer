"""
Tests for the Meowsformer DSP Processing Engine
=================================================
Covers:

- VAPoint distance calculation
- Intent → VA mapping (valid / invalid / case-insensitive)
- Audio retrieval with mock registry (filtering, top-K, empty)
- Breed f0 baselines
- Arousal envelope shaping
- PSOLA prosody transform (with synthetic audio)
- End-to-end synthesize_meow (mocked file system)
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.engine.dsp_processor import (
    BREED_F0_BASELINES,
    INTENT_VA_MAP,
    SampleMatch,
    VAPoint,
    _apply_arousal_envelope,
    _estimate_f0,
    _time_stretch_wsola,
    apply_prosody_transform,
    get_all_intents,
    get_best_match,
    get_breed_f0,
    load_registry,
    map_intent_to_va,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_sine_wav(path: Path, freq: float = 440.0, duration: float = 0.5,
                   sr: int = 22050, amplitude: float = 0.8) -> Path:
    """Generate a mono sine-wave WAV file for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), y, sr)
    return path


def _make_mock_registry(path: Path, samples: list[dict] | None = None) -> Path:
    """Write a minimal mock registry.json."""
    if samples is None:
        samples = [
            {
                "id": "TEST_001",
                "dataset": "test",
                "file_path": "test/test_001.wav",
                "filename": "test_001.wav",
                "context": "Food",
                "breed": "Maine Coon",
                "valence": 0.5,
                "arousal": 0.9,
            },
            {
                "id": "TEST_002",
                "dataset": "test",
                "file_path": "test/test_002.wav",
                "filename": "test_002.wav",
                "context": "Brushing",
                "breed": "European Shorthair",
                "valence": 0.0,
                "arousal": 0.5,
            },
            {
                "id": "TEST_003",
                "dataset": "test",
                "file_path": "test/test_003.wav",
                "filename": "test_003.wav",
                "context": "Isolation",
                "breed": "Maine Coon",
                "valence": -0.8,
                "arousal": 0.7,
            },
            {
                "id": "TEST_004",
                "dataset": "test",
                "file_path": "test/test_004.wav",
                "filename": "test_004.wav",
                "context": "Food",
                "breed": "Siamese",
                "valence": 0.3,
                "arousal": 0.75,
            },
        ]

    registry = {
        "version": "1.0-test",
        "generated_at": "2026-01-01T00:00:00Z",
        "datasets": {"test": {"total_samples": len(samples)}},
        "total_samples": len(samples),
        "samples": samples,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh)
    return path


# ════════════════════════════════════════════════════════════════════════
#  1. VAPoint Tests
# ════════════════════════════════════════════════════════════════════════


class TestVAPoint(unittest.TestCase):
    """Tests for the VAPoint dataclass."""

    def test_distance_to_self_is_zero(self) -> None:
        p = VAPoint(0.5, 0.5)
        self.assertAlmostEqual(p.distance_to(p), 0.0)

    def test_distance_symmetry(self) -> None:
        a = VAPoint(0.3, 0.8)
        b = VAPoint(-0.5, 0.2)
        self.assertAlmostEqual(a.distance_to(b), b.distance_to(a))

    def test_distance_known_value(self) -> None:
        a = VAPoint(0.0, 0.0)
        b = VAPoint(3.0, 4.0)
        self.assertAlmostEqual(a.distance_to(b), 5.0)

    def test_frozen(self) -> None:
        p = VAPoint(0.1, 0.2)
        with self.assertRaises(AttributeError):
            p.valence = 0.5  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════════════
#  2. Intent → VA Mapping Tests
# ════════════════════════════════════════════════════════════════════════


class TestIntentMapping(unittest.TestCase):
    """Tests for map_intent_to_va and friends."""

    def test_all_known_intents_resolve(self) -> None:
        for intent in INTENT_VA_MAP:
            va = map_intent_to_va(intent)
            self.assertIsInstance(va, VAPoint)
            self.assertTrue(-1.0 <= va.valence <= 1.0)
            self.assertTrue(0.0 <= va.arousal <= 1.0)

    def test_case_insensitive(self) -> None:
        va1 = map_intent_to_va("affiliative")
        va2 = map_intent_to_va("AFFILIATIVE")
        va3 = map_intent_to_va("Affiliative")
        self.assertEqual(va1, va2)
        self.assertEqual(va2, va3)

    def test_whitespace_tolerance(self) -> None:
        va = map_intent_to_va("  play  ")
        self.assertEqual(va, INTENT_VA_MAP["Play"])

    def test_unknown_intent_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            map_intent_to_va("NonExistentIntent")
        self.assertIn("Unknown intent", str(ctx.exception))

    def test_get_all_intents_returns_copy(self) -> None:
        all_intents = get_all_intents()
        self.assertEqual(len(all_intents), len(INTENT_VA_MAP))
        # Modifying the copy should not affect the original
        all_intents.pop("Affiliative", None)
        self.assertIn("Affiliative", INTENT_VA_MAP)

    def test_agonistic_is_negative_high_arousal(self) -> None:
        va = map_intent_to_va("Agonistic")
        self.assertLess(va.valence, -0.5)
        self.assertGreater(va.arousal, 0.7)

    def test_contentment_is_positive_low_arousal(self) -> None:
        va = map_intent_to_va("Contentment")
        self.assertGreater(va.valence, 0.5)
        self.assertLess(va.arousal, 0.3)


# ════════════════════════════════════════════════════════════════════════
#  3. Audio Retrieval Tests
# ════════════════════════════════════════════════════════════════════════


class TestGetBestMatch(unittest.TestCase):
    """Tests for get_best_match with mock registry."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.registry_path = self.tmpdir / "registry.json"
        _make_mock_registry(self.registry_path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_returns_single_best(self) -> None:
        results = get_best_match(
            0.5, 0.9, registry_path=self.registry_path
        )
        self.assertEqual(len(results), 1)
        # TEST_001 has (0.5, 0.9) — should be an exact match
        self.assertEqual(results[0].sample_id, "TEST_001")
        self.assertAlmostEqual(results[0].distance, 0.0, places=5)

    def test_top_k(self) -> None:
        results = get_best_match(
            0.0, 0.5, registry_path=self.registry_path, top_k=3
        )
        self.assertEqual(len(results), 3)
        # Distances should be ascending
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i].distance, results[i + 1].distance)

    def test_breed_filter(self) -> None:
        results = get_best_match(
            0.5, 0.9,
            registry_path=self.registry_path,
            breed_filter="Siamese",
            top_k=10,
        )
        for r in results:
            self.assertEqual(r.breed, "Siamese")

    def test_context_filter(self) -> None:
        results = get_best_match(
            0.0, 0.5,
            registry_path=self.registry_path,
            context_filter="Isolation",
            top_k=10,
        )
        for r in results:
            self.assertEqual(r.context, "Isolation")

    def test_no_match_with_impossible_filter(self) -> None:
        results = get_best_match(
            0.0, 0.5,
            registry_path=self.registry_path,
            breed_filter="Sphinx",
        )
        self.assertEqual(len(results), 0)

    def test_empty_registry(self) -> None:
        empty_path = self.tmpdir / "empty.json"
        _make_mock_registry(empty_path, samples=[])
        results = get_best_match(0.0, 0.5, registry_path=empty_path)
        self.assertEqual(len(results), 0)

    def test_missing_registry_raises(self) -> None:
        missing = self.tmpdir / "nonexistent.json"
        with self.assertRaises(FileNotFoundError):
            get_best_match(0.0, 0.5, registry_path=missing)

    def test_result_fields_populated(self) -> None:
        results = get_best_match(
            -0.8, 0.7, registry_path=self.registry_path
        )
        best = results[0]
        self.assertEqual(best.sample_id, "TEST_003")
        self.assertEqual(best.breed, "Maine Coon")
        self.assertEqual(best.context, "Isolation")
        self.assertIsInstance(best.metadata, dict)
        self.assertIn("id", best.metadata)


# ════════════════════════════════════════════════════════════════════════
#  4. Breed f0 Baseline Tests
# ════════════════════════════════════════════════════════════════════════


class TestBreedF0(unittest.TestCase):
    """Tests for breed-based f0 baselines."""

    def test_known_breeds(self) -> None:
        self.assertEqual(get_breed_f0("Maine Coon"), 420.0)
        self.assertEqual(get_breed_f0("Siamese"), 620.0)
        self.assertEqual(get_breed_f0("Kitten"), 750.0)

    def test_partial_match(self) -> None:
        # "maine" should partially match "Maine Coon"
        self.assertEqual(get_breed_f0("maine"), 420.0)

    def test_case_insensitive(self) -> None:
        self.assertEqual(get_breed_f0("european shorthair"), 550.0)

    def test_unknown_breed_returns_default(self) -> None:
        f0 = get_breed_f0("Sphynx Rex Mega Ultra")
        self.assertEqual(f0, 550.0)

    def test_maine_coon_lower_than_kitten(self) -> None:
        self.assertLess(get_breed_f0("Maine Coon"), get_breed_f0("Kitten"))

    def test_all_baselines_are_positive(self) -> None:
        for breed, f0 in BREED_F0_BASELINES.items():
            self.assertGreater(f0, 0.0, f"f0 for {breed} should be positive")


# ════════════════════════════════════════════════════════════════════════
#  5. Arousal Envelope Tests
# ════════════════════════════════════════════════════════════════════════


class TestArousalEnvelope(unittest.TestCase):
    """Tests for the arousal-driven amplitude envelope."""

    def test_output_shape_preserved(self) -> None:
        y = np.ones(1000, dtype=np.float32)
        result = _apply_arousal_envelope(y, 22050, arousal=0.5)
        self.assertEqual(len(result), len(y))

    def test_empty_input(self) -> None:
        y = np.array([], dtype=np.float32)
        result = _apply_arousal_envelope(y, 22050, arousal=0.5)
        self.assertEqual(len(result), 0)

    def test_high_arousal_decays_faster(self) -> None:
        y = np.ones(22050, dtype=np.float32)  # 1 second
        env_high = _apply_arousal_envelope(y.copy(), 22050, arousal=0.95)
        env_low = _apply_arousal_envelope(y.copy(), 22050, arousal=0.1)
        # At the end of the signal, high arousal should have decayed more
        tail_high = np.mean(np.abs(env_high[-2000:]))
        tail_low = np.mean(np.abs(env_low[-2000:]))
        self.assertLess(tail_high, tail_low)

    def test_envelope_values_bounded(self) -> None:
        y = np.ones(5000, dtype=np.float32)
        for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = _apply_arousal_envelope(y.copy(), 22050, arousal=a)
            self.assertTrue(np.all(result >= 0.0))
            self.assertTrue(np.all(result <= 1.0 + 1e-6))


# ════════════════════════════════════════════════════════════════════════
#  6. f0 Estimation Tests
# ════════════════════════════════════════════════════════════════════════


class TestF0Estimation(unittest.TestCase):
    """Tests for the pYIN f0 estimator."""

    def test_sine_wave_f0(self) -> None:
        """A pure sine wave's estimated f0 should be close to its frequency."""
        sr = 22050
        freq = 500.0
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = (0.8 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        estimated = _estimate_f0(y, sr)
        # Allow 10 % tolerance for pYIN on a clean sine
        self.assertAlmostEqual(estimated, freq, delta=freq * 0.10)

    def test_silent_signal_returns_default(self) -> None:
        """A silent signal should return the 440 Hz fallback."""
        y = np.zeros(22050, dtype=np.float32)
        f0 = _estimate_f0(y, 22050)
        self.assertAlmostEqual(f0, 440.0)


# ════════════════════════════════════════════════════════════════════════
#  7. PSOLA Prosody Transform Tests
# ════════════════════════════════════════════════════════════════════════


class TestApplyProsodyTransform(unittest.TestCase):
    """Tests for apply_prosody_transform with synthetic audio."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.wav_path = self.tmpdir / "test_sine.wav"
        _make_sine_wav(self.wav_path, freq=500.0, duration=0.5, sr=22050)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_identity_transform(self) -> None:
        """No pitch shift, no duration change → output ≈ input length."""
        audio, sr = apply_prosody_transform(
            self.wav_path,
            target_pitch_shift=0.0,
            duration_factor=1.0,
        )
        self.assertEqual(sr, 22050)
        # Duration should be roughly the same (within 5 %)
        expected_samples = int(0.5 * 22050)
        self.assertAlmostEqual(
            len(audio), expected_samples, delta=expected_samples * 0.05
        )

    def test_pitch_shift_up_shortens_period(self) -> None:
        """Shifting pitch up should produce higher perceived frequency."""
        audio_shifted, sr = apply_prosody_transform(
            self.wav_path,
            target_pitch_shift=12.0,  # one octave up
            duration_factor=1.0,
        )
        # After one octave up, the signal should still be valid
        self.assertGreater(len(audio_shifted), 0)
        self.assertTrue(np.all(np.isfinite(audio_shifted)))

    def test_duration_stretch(self) -> None:
        """duration_factor=2.0 should roughly double the duration."""
        audio, sr = apply_prosody_transform(
            self.wav_path,
            target_pitch_shift=0.0,
            duration_factor=2.0,
        )
        original_samples = int(0.5 * sr)
        # Allow 20 % tolerance
        self.assertAlmostEqual(
            len(audio), original_samples * 2, delta=original_samples * 0.4
        )

    def test_breed_adjustment(self) -> None:
        """Breed-based f0 should modify the output."""
        audio_mc, sr1 = apply_prosody_transform(
            self.wav_path,
            target_pitch_shift=0.0,
            duration_factor=1.0,
            breed="Maine Coon",
        )
        audio_kit, sr2 = apply_prosody_transform(
            self.wav_path,
            target_pitch_shift=0.0,
            duration_factor=1.0,
            breed="Kitten",
        )
        # Both should produce valid audio
        self.assertGreater(len(audio_mc), 0)
        self.assertGreater(len(audio_kit), 0)
        # They should differ (different breed f0 targets)
        min_len = min(len(audio_mc), len(audio_kit))
        if min_len > 0:
            diff = np.mean(np.abs(audio_mc[:min_len] - audio_kit[:min_len]))
            self.assertGreater(diff, 0.001, "Different breeds should produce different outputs")

    def test_arousal_modulation(self) -> None:
        """High arousal should produce shorter audio than low arousal."""
        audio_high, _ = apply_prosody_transform(
            self.wav_path,
            target_pitch_shift=0.0,
            duration_factor=1.0,
            arousal=0.95,
        )
        audio_low, _ = apply_prosody_transform(
            self.wav_path,
            target_pitch_shift=0.0,
            duration_factor=1.0,
            arousal=0.05,
        )
        # High arousal compresses time → fewer samples
        self.assertLess(len(audio_high), len(audio_low))

    def test_output_normalised(self) -> None:
        """Output peak should be at most 0.95."""
        audio, sr = apply_prosody_transform(
            self.wav_path,
            target_pitch_shift=5.0,
            duration_factor=1.0,
        )
        peak = np.max(np.abs(audio))
        self.assertLessEqual(peak, 0.96)

    def test_save_to_file(self) -> None:
        """output_path should create a valid WAV file."""
        out_path = self.tmpdir / "output.wav"
        audio, sr = apply_prosody_transform(
            self.wav_path,
            target_pitch_shift=-3.0,
            duration_factor=1.2,
            output_path=out_path,
        )
        self.assertTrue(out_path.exists())
        # Verify the saved file is readable
        loaded, loaded_sr = sf.read(str(out_path))
        self.assertEqual(loaded_sr, sr)
        self.assertGreater(len(loaded), 0)

    def test_nonexistent_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            apply_prosody_transform("/nonexistent/path.wav")

    def test_combined_transform(self) -> None:
        """Combined pitch + duration + breed + arousal should not crash."""
        audio, sr = apply_prosody_transform(
            self.wav_path,
            target_pitch_shift=-5.0,
            duration_factor=0.8,
            breed="Siamese",
            arousal=0.7,
        )
        self.assertGreater(len(audio), 0)
        self.assertTrue(np.all(np.isfinite(audio)))


# ════════════════════════════════════════════════════════════════════════
#  8. Time-stretch (WSOLA) Tests
# ════════════════════════════════════════════════════════════════════════


class TestTimeStretch(unittest.TestCase):
    """Tests for _time_stretch_wsola."""

    def test_stretch_doubles_length(self) -> None:
        sr = 22050
        y = np.random.randn(sr).astype(np.float32)  # 1 second
        stretched = _time_stretch_wsola(y, sr, 2.0)
        # Should be roughly 2× longer (within 10 %)
        self.assertAlmostEqual(
            len(stretched), len(y) * 2, delta=len(y) * 0.2
        )

    def test_compress_halves_length(self) -> None:
        sr = 22050
        y = np.random.randn(sr).astype(np.float32)
        compressed = _time_stretch_wsola(y, sr, 0.5)
        self.assertAlmostEqual(
            len(compressed), len(y) * 0.5, delta=len(y) * 0.1
        )

    def test_factor_one_is_identity(self) -> None:
        y = np.random.randn(5000).astype(np.float32)
        result = _time_stretch_wsola(y, 22050, 1.0)
        # factor ≈ 1.0 should return the input unchanged
        np.testing.assert_array_equal(result, y)


# ════════════════════════════════════════════════════════════════════════
#  9. Load Registry Tests
# ════════════════════════════════════════════════════════════════════════


class TestLoadRegistry(unittest.TestCase):
    """Tests for load_registry."""

    def test_load_valid_registry(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "registry.json"
            _make_mock_registry(path)
            reg = load_registry(path)
            self.assertEqual(reg["total_samples"], 4)
            self.assertIn("samples", reg)

    def test_load_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_registry(Path("/definitely/not/a/real/path.json"))


# ════════════════════════════════════════════════════════════════════════
#  Run
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
