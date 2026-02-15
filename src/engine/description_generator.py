"""
Meowsformer — Confidence Description Generator
================================================
Lightweight wrapper inspired by NatureLM-audio's descriptive captioning
approach. Generates human-readable preview descriptions for synthesised
cat vocalisations by combining:

- **Matched sample metadata** (context, breed, VA coordinates)
- **Intent semantics** (mapped from the VA space)
- **Acoustic characteristics** inferred from the DSP transform parameters

The output is a localised Chinese-language description suitable for
display in the frontend preview panel, giving the user confidence about
what they are about to send before confirmation.

References
----------
- NatureLM-audio (2024): Language model-based audio captioning.
- Russell's Circumplex Model of Affect for VA-space interpretation.

Usage::

    from src.engine.description_generator import generate_preview_description

    desc = generate_preview_description(
        intent="Requesting",
        match=sample_match,
        target_va=va_point,
        breed="Maine Coon",
        pitch_shift_st=+2.3,
        duration_factor=0.85,
    )
    # → "系统匹配了一段代表'积极求食'的高频上升调鸣叫（Maine Coon 品种，
    #    置信距离 0.12），音高上调 2.3 半音，节奏略加紧凑以体现较高的唤醒度。"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.engine.dsp_processor import INTENT_VA_MAP, SampleMatch, VAPoint


# ── Intent → 中文语义标签 ───────────────────────────────────────────────
INTENT_CN_LABELS: dict[str, str] = {
    "Affiliative": "友好问候",
    "Contentment": "满足放松",
    "Play": "嬉戏邀请",
    "Requesting": "积极求食",
    "Solicitation": "温柔请求",
    "Agonistic": "威胁警告",
    "Distress": "痛苦求助",
    "Frustration": "烦躁不满",
    "Alert": "警觉信号",
    "Neutral": "平静基线",
}

# ── Context → 中文描述 ──────────────────────────────────────────────────
CONTEXT_CN_LABELS: dict[str, str] = {
    "Food": "进食场景",
    "Isolation": "隔离场景",
    "Brushing": "梳理场景",
}

# ── Arousal → 描述性词汇 ────────────────────────────────────────────────
_AROUSAL_DESCRIPTORS: list[tuple[float, str]] = [
    (0.0, "极平静的"),
    (0.2, "舒缓低沉的"),
    (0.4, "平稳的"),
    (0.6, "中等活跃的"),
    (0.8, "高亢激昂的"),
    (0.9, "极度紧迫的"),
]

# ── Valence → 情感色彩 ──────────────────────────────────────────────────
_VALENCE_DESCRIPTORS: list[tuple[float, str]] = [
    (-1.0, "强烈消极"),
    (-0.5, "消极"),
    (-0.2, "略带消极"),
    (0.0, "中性"),
    (0.2, "略带积极"),
    (0.5, "积极"),
    (1.0, "强烈积极"),
]

# ── Pitch shift → 音调描述 ──────────────────────────────────────────────
_PITCH_DESCRIPTORS: list[tuple[float, str]] = [
    (-12.0, "大幅降调"),
    (-6.0, "明显降调"),
    (-2.0, "微幅降调"),
    (0.0, "原始音调"),
    (2.0, "微幅升调"),
    (6.0, "明显升调"),
    (12.0, "大幅升调"),
]

# ── Vocalisation type heuristics ────────────────────────────────────────
_VOCALISATION_TYPES: dict[str, str] = {
    "Affiliative": "短促喵叫",
    "Contentment": "呼噜/低频共鸣",
    "Play": "啁啾高频颤音",
    "Requesting": "高频上升调鸣叫",
    "Solicitation": "中频渐进式喵叫",
    "Agonistic": "嘶吼/低频喉音",
    "Distress": "尖锐高频哀鸣",
    "Frustration": "断续低沉喵叫",
    "Alert": "短促中频警示音",
    "Neutral": "标准喵叫",
}


# ══════════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════════


def _lookup_descriptor(value: float, table: list[tuple[float, str]]) -> str:
    """Find the closest descriptor from a sorted (threshold, label) table."""
    best_label = table[0][1]
    best_dist = abs(value - table[0][0])
    for threshold, label in table[1:]:
        dist = abs(value - threshold)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def _compute_confidence_score(distance: float) -> float:
    """Convert Euclidean VA distance to a [0, 1] confidence score.

    Uses an exponential decay: confidence = exp(-k * distance).
    A perfect match (distance=0) yields 1.0;
    distance=1.0 yields ~0.37; distance=2.0 yields ~0.14.
    """
    k = 1.0  # decay rate
    return math.exp(-k * distance)


def _confidence_level_cn(score: float) -> str:
    """Map a confidence score [0, 1] to a Chinese confidence level."""
    if score >= 0.90:
        return "极高"
    elif score >= 0.70:
        return "高"
    elif score >= 0.50:
        return "中等"
    elif score >= 0.30:
        return "较低"
    else:
        return "低"


# ══════════════════════════════════════════════════════════════════════════
#  Primary Description Data
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class PreviewDescription:
    """Structured preview description for frontend display.

    Attributes
    ----------
    summary : str
        One-sentence human-readable summary (Chinese).
    intent_label : str
        Localised intent label.
    vocalisation_type : str
        Inferred vocalisation type description.
    confidence_score : float
        Match confidence in [0, 1].
    confidence_level : str
        Qualitative confidence level (Chinese).
    va_distance : float
        Raw Euclidean distance in VA space.
    pitch_description : str
        Description of pitch transformation applied.
    tempo_description : str
        Description of temporal transformation applied.
    breed : str
        Target breed.
    source_context : str
        Recording context of the matched sample.
    detail : str
        Multi-line detailed description (Chinese).
    """

    summary: str
    intent_label: str
    vocalisation_type: str
    confidence_score: float
    confidence_level: str
    va_distance: float
    pitch_description: str
    tempo_description: str
    breed: str
    source_context: str
    detail: str


# ══════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════


def generate_preview_description(
    intent: str,
    match: SampleMatch,
    target_va: VAPoint,
    *,
    breed: str = "Default",
    pitch_shift_st: float = 0.0,
    duration_factor: float = 1.0,
    arousal: Optional[float] = None,
) -> PreviewDescription:
    """Generate a human-readable preview description for the synthesised meow.

    Mimics NatureLM-audio's descriptive captioning: combines acoustic tags,
    intent semantics, and transform parameters into a natural-language
    summary that helps the user understand *what* the system matched and
    *how* the audio has been processed before they confirm playback.

    Parameters
    ----------
    intent : str
        The mapped communicative intent (e.g. ``"Requesting"``).
    match : SampleMatch
        The nearest-neighbour sample returned by ``get_best_match``.
    target_va : VAPoint
        The target VA coordinates for this synthesis.
    breed : str
        Target cat breed.
    pitch_shift_st : float
        Total pitch shift applied (in semitones).
    duration_factor : float
        Effective duration factor after arousal modulation.
    arousal : float, optional
        Arousal level [0, 1]; if None, derived from target_va.

    Returns
    -------
    PreviewDescription
        Structured description with summary, detail, and metadata.
    """
    normalised_intent = intent.strip().title()

    # ── Resolve labels ────────────────────────────────────────────────
    intent_label = INTENT_CN_LABELS.get(normalised_intent, intent)
    vocalisation_type = _VOCALISATION_TYPES.get(normalised_intent, "喵叫")
    context_label = CONTEXT_CN_LABELS.get(match.context, match.context or "未知场景")

    # ── Confidence ────────────────────────────────────────────────────
    confidence = _compute_confidence_score(match.distance)
    confidence_level = _confidence_level_cn(confidence)

    # ── Acoustic descriptors ──────────────────────────────────────────
    effective_arousal = arousal if arousal is not None else target_va.arousal
    arousal_desc = _lookup_descriptor(effective_arousal, _AROUSAL_DESCRIPTORS)
    valence_desc = _lookup_descriptor(target_va.valence, _VALENCE_DESCRIPTORS)
    pitch_desc = _lookup_descriptor(pitch_shift_st, _PITCH_DESCRIPTORS)

    # ── Tempo description ─────────────────────────────────────────────
    if duration_factor < 0.85:
        tempo_desc = "节奏明显加快以体现紧迫感"
    elif duration_factor < 0.95:
        tempo_desc = "节奏略加紧凑"
    elif duration_factor <= 1.05:
        tempo_desc = "保持原始节奏"
    elif duration_factor <= 1.15:
        tempo_desc = "节奏略有舒缓"
    else:
        tempo_desc = "节奏明显放缓以体现从容感"

    # ── Build summary (one-sentence) ──────────────────────────────────
    pitch_shift_str = f"{pitch_shift_st:+.1f}" if abs(pitch_shift_st) > 0.1 else ""
    pitch_clause = f"，音高{pitch_desc}（{pitch_shift_str}半音）" if pitch_shift_str else ""

    summary = (
        f"系统匹配了一段代表'{intent_label}'的{arousal_desc}{vocalisation_type}"
        f"（{breed} 品种，置信度{confidence_level}"
        f"，VA距离 {match.distance:.3f}）"
        f"{pitch_clause}，{tempo_desc}。"
    )

    # ── Build detail (multi-line) ─────────────────────────────────────
    detail_lines = [
        f"【意图映射】{intent} → {intent_label}",
        f"【情感空间】Valence={target_va.valence:+.2f}（{valence_desc}），"
        f"Arousal={effective_arousal:.2f}（{arousal_desc}）",
        f"【匹配样本】{match.sample_id}（{context_label} / {match.breed}）",
        f"【VA 距离】{match.distance:.4f}（置信度 {confidence:.1%} — {confidence_level}）",
        f"【音高调整】{pitch_desc}"
        + (f"（{pitch_shift_st:+.1f} 半音）" if abs(pitch_shift_st) > 0.1 else ""),
        f"【时长因子】×{duration_factor:.2f} — {tempo_desc}",
        f"【目标品种】{breed}",
        f"【发声类型】{vocalisation_type}",
    ]
    detail = "\n".join(detail_lines)

    result = PreviewDescription(
        summary=summary,
        intent_label=intent_label,
        vocalisation_type=vocalisation_type,
        confidence_score=round(confidence, 4),
        confidence_level=confidence_level,
        va_distance=round(match.distance, 4),
        pitch_description=pitch_desc,
        tempo_description=tempo_desc,
        breed=breed,
        source_context=match.context,
        detail=detail,
    )

    logger.info(
        "Generated preview: intent='{}' confidence={:.1%} breed='{}'",
        intent,
        confidence,
        breed,
    )
    return result


def generate_description_from_synthesis(
    intent: str,
    match: SampleMatch,
    *,
    breed: str = "Default",
) -> PreviewDescription:
    """Convenience wrapper that derives transform parameters from the match.

    Re-computes the same pitch/duration deltas that ``synthesize_meow``
    uses internally, so the description accurately reflects what was
    actually applied to the audio.

    Parameters
    ----------
    intent : str
        The communicative intent label.
    match : SampleMatch
        The matched sample from ``get_best_match``.
    breed : str
        Target cat breed.

    Returns
    -------
    PreviewDescription
    """
    normalised = intent.strip().title()
    target_va = INTENT_VA_MAP.get(normalised)
    if target_va is None:
        target_va = VAPoint(valence=0.0, arousal=0.4)

    # Reproduce the same deltas as synthesize_meow
    va_pitch_hint = (target_va.valence - match.valence) * 2.0
    duration_factor = 1.0 + (match.arousal - target_va.arousal) * 0.5

    # Arousal-based time modulation (same formula as dsp_processor)
    arousal_time_mod = 1.0 - (target_va.arousal - 0.5) * 0.3
    effective_duration = duration_factor * arousal_time_mod

    return generate_preview_description(
        intent=intent,
        match=match,
        target_va=target_va,
        breed=breed,
        pitch_shift_st=va_pitch_hint,
        duration_factor=effective_duration,
        arousal=target_va.arousal,
    )
