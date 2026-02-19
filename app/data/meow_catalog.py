"""
Meowsformer — Multi-dimensional Tag Taxonomy & Rule Engine
============================================================
Defines the 5-dimension tag vocabulary and rule-based assignment logic.

Dimensions:
    1. emotion      — what the cat is feeling (from context + VA)
    2. intent       — communicative purpose (from context)
    3. acoustic     — sound character (from librosa features)
    4. social_context — when the sound is appropriate (from context)
    5. breed_voice  — vocal character (from breed)

Rule functions accept a sample dict (from registry.json) or extracted
acoustic features and return the set of tags for that dimension.
"""

from __future__ import annotations

from typing import Any

# ══════════════════════════════════════════════════════════════════════════
# Tag vocabulary — every valid tag across all 5 dimensions
# ══════════════════════════════════════════════════════════════════════════

TAG_TAXONOMY: dict[str, list[str]] = {
    "emotion": [
        "hungry",
        "eager",
        "demanding",
        "anxious",
        "lonely",
        "distressed",
        "content",
        "relaxed",
        "annoyed",
        "agitated",
        "calm",
    ],
    "intent": [
        "requesting_food",
        "demanding_attention",
        "seeking_companionship",
        "expressing_comfort",
        "protesting",
        "greeting",
    ],
    "acoustic": [
        "high_pitch",
        "low_pitch",
        "mid_pitch",
        "short_burst",
        "medium_length",
        "prolonged",
        "loud",
        "soft",
        "rising_tone",
        "falling_tone",
        "trembling",
    ],
    "social_context": [
        "feeding_time",
        "alone_at_home",
        "separation",
        "being_petted",
        "physical_contact",
        "near_owner",
    ],
    "breed_voice": [
        "deep_voice",
        "bright_voice",
    ],
}


# ══════════════════════════════════════════════════════════════════════════
# Dimension 1 — emotion (context + VA coordinates)
# ══════════════════════════════════════════════════════════════════════════


def tag_emotion(sample: dict[str, Any]) -> list[str]:
    """Assign emotion tags based on context, valence (V), and arousal (A)."""
    ctx = sample.get("context", "")
    v = float(sample.get("valence", 0.0))
    a = float(sample.get("arousal", 0.0))
    tags: list[str] = []

    if ctx == "Food":
        tags.append("hungry")
        if a > 0.8:
            tags.append("eager")
            tags.append("demanding")
    elif ctx == "Isolation":
        tags.append("lonely")
        if a > 0.6:
            tags.append("anxious")
        if v < -0.5:
            tags.append("distressed")
    elif ctx == "Brushing":
        if v > 0:
            tags.append("content")
            if a < 0.5:
                tags.append("relaxed")
        if v < 0:
            tags.append("annoyed")

    # Context-independent rules
    if v < 0 and a > 0.6:
        if "agitated" not in tags:
            tags.append("agitated")
    if a < 0.4:
        if "calm" not in tags:
            tags.append("calm")

    return tags


# ══════════════════════════════════════════════════════════════════════════
# Dimension 2 — intent (context + VA)
# ══════════════════════════════════════════════════════════════════════════


def tag_intent(sample: dict[str, Any]) -> list[str]:
    """Assign intent tags based on context and VA."""
    ctx = sample.get("context", "")
    v = float(sample.get("valence", 0.0))
    a = float(sample.get("arousal", 0.0))
    tags: list[str] = []

    if ctx == "Food":
        tags.append("requesting_food")
        tags.append("demanding_attention")
    elif ctx == "Isolation":
        tags.append("seeking_companionship")
        tags.append("demanding_attention")
    elif ctx == "Brushing":
        if v > 0:
            tags.append("expressing_comfort")
            if 0.2 < v and 0.3 <= a <= 0.6:
                tags.append("greeting")
        if v < 0:
            tags.append("protesting")

    return tags


# ══════════════════════════════════════════════════════════════════════════
# Dimension 3 — acoustic (librosa-extracted features)
# ══════════════════════════════════════════════════════════════════════════


def tag_acoustic(features: dict[str, Any]) -> list[str]:
    """Assign acoustic tags from pre-extracted librosa features.

    Expected keys in ``features``:
        - ``median_f0``     : float | None (Hz)
        - ``duration``      : float (seconds)
        - ``rms_energy``    : float
        - ``f0_slope``      : float (Hz/s)
        - ``f0_std``        : float (Hz)
        - ``rms_percentile``: str ("high" | "low" | "mid")
    """
    tags: list[str] = []

    # Pitch
    median_f0 = features.get("median_f0")
    if median_f0 is not None:
        if median_f0 > 600:
            tags.append("high_pitch")
        elif median_f0 < 400:
            tags.append("low_pitch")
        else:
            tags.append("mid_pitch")

    # Duration
    dur = features.get("duration", 0.0)
    if dur < 0.5:
        tags.append("short_burst")
    elif dur <= 1.5:
        tags.append("medium_length")
    else:
        tags.append("prolonged")

    # Loudness
    rms_pct = features.get("rms_percentile", "mid")
    if rms_pct == "high":
        tags.append("loud")
    elif rms_pct == "low":
        tags.append("soft")

    # F0 slope
    f0_slope = features.get("f0_slope", 0.0)
    if f0_slope is not None:
        if f0_slope > 0:
            tags.append("rising_tone")
        elif f0_slope < 0:
            tags.append("falling_tone")

    # F0 variability
    f0_std = features.get("f0_std", 0.0)
    if f0_std is not None and f0_std > 80:
        tags.append("trembling")

    return tags


# ══════════════════════════════════════════════════════════════════════════
# Dimension 4 — social_context (from recording context)
# ══════════════════════════════════════════════════════════════════════════


def tag_social_context(sample: dict[str, Any]) -> list[str]:
    """Assign social context tags from the recording context."""
    ctx = sample.get("context", "")
    tags: list[str] = []

    if ctx == "Food":
        tags.append("feeding_time")
        tags.append("near_owner")
    elif ctx == "Isolation":
        tags.append("alone_at_home")
        tags.append("separation")
    elif ctx == "Brushing":
        tags.append("being_petted")
        tags.append("physical_contact")
        tags.append("near_owner")

    return tags


# ══════════════════════════════════════════════════════════════════════════
# Dimension 5 — breed_voice (from breed metadata)
# ══════════════════════════════════════════════════════════════════════════


def tag_breed_voice(sample: dict[str, Any]) -> list[str]:
    """Assign breed voice tags from breed metadata."""
    breed = sample.get("breed", "")
    tags: list[str] = []

    if breed == "Maine Coon":
        tags.append("deep_voice")
    elif breed == "European Shorthair":
        tags.append("bright_voice")

    return tags


# ══════════════════════════════════════════════════════════════════════════
# Convenience: apply all non-acoustic tags at once
# ══════════════════════════════════════════════════════════════════════════


def tag_sample_metadata(sample: dict[str, Any]) -> dict[str, list[str]]:
    """Apply all metadata-based (non-acoustic) tagging rules.

    Returns a dict keyed by dimension with the assigned tag lists.
    Acoustic tags must be added separately after librosa feature extraction.
    """
    return {
        "emotion": tag_emotion(sample),
        "intent": tag_intent(sample),
        "social_context": tag_social_context(sample),
        "breed_voice": tag_breed_voice(sample),
    }
