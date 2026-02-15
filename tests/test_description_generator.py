"""
Tests for the Meowsformer Description Generator (Phase 3)
==========================================================
Covers:

- Intent → Chinese label mapping
- Confidence score computation (exponential decay)
- Confidence level classification
- Descriptor lookup (arousal, valence, pitch)
- Full preview description generation
- Convenience wrapper generate_description_from_synthesis
- Edge cases (unknown intent, extreme VA values)
"""

from __future__ import annotations

import math
import unittest

from src.engine.description_generator import (
    CONTEXT_CN_LABELS,
    INTENT_CN_LABELS,
    PreviewDescription,
    _compute_confidence_score,
    _confidence_level_cn,
    _lookup_descriptor,
    _AROUSAL_DESCRIPTORS,
    _PITCH_DESCRIPTORS,
    _VALENCE_DESCRIPTORS,
    _VOCALISATION_TYPES,
    generate_description_from_synthesis,
    generate_preview_description,
)
from src.engine.dsp_processor import INTENT_VA_MAP, SampleMatch, VAPoint


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_sample_match(
    distance: float = 0.15,
    valence: float = 0.3,
    arousal: float = 0.75,
    breed: str = "Maine Coon",
    context: str = "Food",
    sample_id: str = "TEST_001",
) -> SampleMatch:
    """Create a mock SampleMatch for testing."""
    return SampleMatch(
        sample_id=sample_id,
        file_path="test/test_001.wav",
        distance=distance,
        valence=valence,
        arousal=arousal,
        breed=breed,
        context=context,
        metadata={"id": sample_id},
    )


# ════════════════════════════════════════════════════════════════════════
#  1. Label Mapping Tests
# ════════════════════════════════════════════════════════════════════════


class TestIntentLabels(unittest.TestCase):
    """Tests for intent → Chinese label mapping."""

    def test_all_intents_have_cn_labels(self) -> None:
        """Every intent in the VA map should have a Chinese label."""
        for intent in INTENT_VA_MAP:
            self.assertIn(
                intent,
                INTENT_CN_LABELS,
                f"Missing Chinese label for intent '{intent}'",
            )

    def test_all_intents_have_vocalisation_types(self) -> None:
        """Every intent should have a vocalisation type description."""
        for intent in INTENT_VA_MAP:
            self.assertIn(
                intent,
                _VOCALISATION_TYPES,
                f"Missing vocalisation type for intent '{intent}'",
            )

    def test_context_labels_exist(self) -> None:
        """Known contexts should have Chinese labels."""
        for ctx in ["Food", "Isolation", "Brushing"]:
            self.assertIn(ctx, CONTEXT_CN_LABELS)


# ════════════════════════════════════════════════════════════════════════
#  2. Confidence Score Tests
# ════════════════════════════════════════════════════════════════════════


class TestConfidenceScore(unittest.TestCase):
    """Tests for _compute_confidence_score."""

    def test_perfect_match(self) -> None:
        """Distance = 0 → confidence = 1.0."""
        self.assertAlmostEqual(_compute_confidence_score(0.0), 1.0)

    def test_distance_one(self) -> None:
        """Distance = 1.0 → confidence ≈ e^(-1) ≈ 0.368."""
        expected = math.exp(-1.0)
        self.assertAlmostEqual(_compute_confidence_score(1.0), expected, places=3)

    def test_large_distance(self) -> None:
        """Large distance → confidence close to 0."""
        score = _compute_confidence_score(10.0)
        self.assertLess(score, 0.001)

    def test_monotonically_decreasing(self) -> None:
        """Confidence should decrease as distance increases."""
        prev = 1.0
        for d in [0.1, 0.5, 1.0, 2.0, 5.0]:
            score = _compute_confidence_score(d)
            self.assertLess(score, prev)
            prev = score


# ════════════════════════════════════════════════════════════════════════
#  3. Confidence Level Tests
# ════════════════════════════════════════════════════════════════════════


class TestConfidenceLevel(unittest.TestCase):
    """Tests for _confidence_level_cn."""

    def test_very_high(self) -> None:
        self.assertEqual(_confidence_level_cn(0.95), "极高")

    def test_high(self) -> None:
        self.assertEqual(_confidence_level_cn(0.75), "高")

    def test_medium(self) -> None:
        self.assertEqual(_confidence_level_cn(0.55), "中等")

    def test_low(self) -> None:
        self.assertEqual(_confidence_level_cn(0.35), "较低")

    def test_very_low(self) -> None:
        self.assertEqual(_confidence_level_cn(0.1), "低")

    def test_boundary_values(self) -> None:
        """Test exact boundary thresholds."""
        self.assertEqual(_confidence_level_cn(0.90), "极高")
        self.assertEqual(_confidence_level_cn(0.70), "高")
        self.assertEqual(_confidence_level_cn(0.50), "中等")
        self.assertEqual(_confidence_level_cn(0.30), "较低")


# ════════════════════════════════════════════════════════════════════════
#  4. Descriptor Lookup Tests
# ════════════════════════════════════════════════════════════════════════


class TestDescriptorLookup(unittest.TestCase):
    """Tests for _lookup_descriptor."""

    def test_arousal_extreme_low(self) -> None:
        desc = _lookup_descriptor(0.0, _AROUSAL_DESCRIPTORS)
        self.assertEqual(desc, "极平静的")

    def test_arousal_extreme_high(self) -> None:
        desc = _lookup_descriptor(1.0, _AROUSAL_DESCRIPTORS)
        self.assertEqual(desc, "极度紧迫的")

    def test_arousal_mid(self) -> None:
        desc = _lookup_descriptor(0.5, _AROUSAL_DESCRIPTORS)
        # Should match 0.4 or 0.6 — closest by abs distance
        self.assertIn(desc, ["平稳的", "中等活跃的"])

    def test_valence_negative(self) -> None:
        desc = _lookup_descriptor(-0.9, _VALENCE_DESCRIPTORS)
        self.assertEqual(desc, "强烈消极")

    def test_valence_positive(self) -> None:
        desc = _lookup_descriptor(0.8, _VALENCE_DESCRIPTORS)
        self.assertIn(desc, ["积极", "强烈积极"])

    def test_pitch_no_shift(self) -> None:
        desc = _lookup_descriptor(0.0, _PITCH_DESCRIPTORS)
        self.assertEqual(desc, "原始音调")

    def test_pitch_large_up(self) -> None:
        desc = _lookup_descriptor(10.0, _PITCH_DESCRIPTORS)
        self.assertEqual(desc, "大幅升调")


# ════════════════════════════════════════════════════════════════════════
#  5. Full Preview Description Tests
# ════════════════════════════════════════════════════════════════════════


class TestGeneratePreviewDescription(unittest.TestCase):
    """Tests for generate_preview_description."""

    def test_basic_generation(self) -> None:
        """Should produce a non-empty PreviewDescription."""
        match = _make_sample_match()
        target_va = VAPoint(valence=0.30, arousal=0.75)

        desc = generate_preview_description(
            intent="Requesting",
            match=match,
            target_va=target_va,
            breed="Maine Coon",
            pitch_shift_st=+1.5,
            duration_factor=0.9,
        )

        self.assertIsInstance(desc, PreviewDescription)
        self.assertTrue(len(desc.summary) > 10)
        self.assertTrue(len(desc.detail) > 20)
        self.assertEqual(desc.intent_label, "积极求食")
        self.assertEqual(desc.breed, "Maine Coon")

    def test_summary_contains_key_info(self) -> None:
        """Summary should mention intent label, breed, and confidence."""
        match = _make_sample_match(distance=0.05)
        target_va = VAPoint(valence=0.70, arousal=0.35)

        desc = generate_preview_description(
            intent="Affiliative",
            match=match,
            target_va=target_va,
            breed="Siamese",
        )

        self.assertIn("友好问候", desc.summary)
        self.assertIn("Siamese", desc.summary)
        self.assertIn("置信度", desc.summary)

    def test_confidence_score_in_range(self) -> None:
        """Confidence score should be in [0, 1]."""
        match = _make_sample_match(distance=0.5)
        target_va = VAPoint(valence=0.0, arousal=0.5)

        desc = generate_preview_description(
            intent="Neutral",
            match=match,
            target_va=target_va,
        )

        self.assertGreaterEqual(desc.confidence_score, 0.0)
        self.assertLessEqual(desc.confidence_score, 1.0)

    def test_detail_has_all_sections(self) -> None:
        """Detail text should contain all analysis sections."""
        match = _make_sample_match()
        target_va = VAPoint(valence=-0.80, arousal=0.90)

        desc = generate_preview_description(
            intent="Agonistic",
            match=match,
            target_va=target_va,
        )

        for section in ["意图映射", "情感空间", "匹配样本", "VA 距离",
                        "音高调整", "时长因子", "目标品种", "发声类型"]:
            self.assertIn(section, desc.detail, f"Missing section: {section}")

    def test_all_intents_generate_descriptions(self) -> None:
        """Every known intent should produce a valid description."""
        match = _make_sample_match()

        for intent, va in INTENT_VA_MAP.items():
            desc = generate_preview_description(
                intent=intent,
                match=match,
                target_va=va,
            )
            self.assertIsInstance(desc, PreviewDescription)
            self.assertTrue(len(desc.summary) > 0)
            self.assertTrue(len(desc.intent_label) > 0)

    def test_tempo_descriptions(self) -> None:
        """Different duration factors should produce appropriate tempo descriptions."""
        match = _make_sample_match()
        target_va = VAPoint(valence=0.0, arousal=0.5)

        # Fast tempo
        desc_fast = generate_preview_description(
            "Neutral", match, target_va, duration_factor=0.7
        )
        self.assertIn("加快", desc_fast.tempo_description)

        # Slow tempo
        desc_slow = generate_preview_description(
            "Neutral", match, target_va, duration_factor=1.3
        )
        self.assertIn("放缓", desc_slow.tempo_description)

        # Normal tempo
        desc_norm = generate_preview_description(
            "Neutral", match, target_va, duration_factor=1.0
        )
        self.assertIn("原始", desc_norm.tempo_description)

    def test_zero_distance_gives_max_confidence(self) -> None:
        """A perfect VA match should give confidence ≈ 1.0."""
        match = _make_sample_match(distance=0.0)
        target_va = VAPoint(valence=0.3, arousal=0.75)

        desc = generate_preview_description(
            "Requesting", match, target_va
        )

        self.assertAlmostEqual(desc.confidence_score, 1.0, places=3)
        self.assertEqual(desc.confidence_level, "极高")


# ════════════════════════════════════════════════════════════════════════
#  6. Convenience Wrapper Tests
# ════════════════════════════════════════════════════════════════════════


class TestGenerateDescriptionFromSynthesis(unittest.TestCase):
    """Tests for generate_description_from_synthesis."""

    def test_requesting_intent(self) -> None:
        """Should derive correct transform params for 'Requesting'."""
        match = _make_sample_match(valence=0.5, arousal=0.9)

        desc = generate_description_from_synthesis(
            intent="Requesting",
            match=match,
            breed="Kitten",
        )

        self.assertIsInstance(desc, PreviewDescription)
        self.assertEqual(desc.intent_label, "积极求食")
        self.assertEqual(desc.breed, "Kitten")

    def test_agonistic_intent(self) -> None:
        """Should handle negative valence intents correctly."""
        match = _make_sample_match(valence=-0.8, arousal=0.7)

        desc = generate_description_from_synthesis(
            intent="Agonistic",
            match=match,
        )

        self.assertEqual(desc.intent_label, "威胁警告")
        self.assertIn("嘶吼", desc.vocalisation_type)

    def test_unknown_intent_falls_back(self) -> None:
        """Unknown intent should fall back to neutral VA."""
        match = _make_sample_match()

        desc = generate_description_from_synthesis(
            intent="CompletelyUnknown",
            match=match,
        )

        # Should not crash, should use default
        self.assertIsInstance(desc, PreviewDescription)
        self.assertTrue(len(desc.summary) > 0)

    def test_all_known_intents(self) -> None:
        """All known intents should work through the convenience wrapper."""
        match = _make_sample_match()

        for intent in INTENT_VA_MAP:
            desc = generate_description_from_synthesis(intent=intent, match=match)
            self.assertIsInstance(desc, PreviewDescription)
            self.assertIn(desc.intent_label, INTENT_CN_LABELS.values())


# ════════════════════════════════════════════════════════════════════════
#  Run
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()
