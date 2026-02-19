"""
Meowsformer — Multi-dimensional Tag Matching Engine
=====================================================
Loads ``tagged_samples.json`` at startup and provides weighted Jaccard-like
scoring to find the best cat sound for a given set of target tags.

This replaces the old VA-space nearest-neighbour + PSOLA pipeline entirely.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, Field

from app.schemas.ws_messages import TargetTagSet

# ── Paths ─────────────────────────────────────────────────────────────────

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
TAGGED_SAMPLES_PATH = ASSETS_DIR / "audio_db" / "tagged_samples.json"

# ── Dimension weights (tunable) ─────────────────────────────────────────

DIMENSION_WEIGHTS: dict[str, float] = {
    "emotion": 0.30,
    "intent": 0.30,
    "acoustic": 0.15,
    "social_context": 0.15,
    "breed_voice": 0.10,
}

# Breed preference boost
BREED_BOOST = 0.05


# ── Data models ──────────────────────────────────────────────────────────


class TaggedSample(BaseModel):
    """A single audio sample with multi-dimensional tags."""

    id: str
    file_path: str
    breed: str
    valence: float
    arousal: float
    context: str
    tags: dict[str, list[str]] = Field(default_factory=dict)


class MatchResult(BaseModel):
    """Result of matching a sample against target tags."""

    sample: TaggedSample
    score: float
    matched_tags: dict[str, list[str]] = Field(default_factory=dict)


# ── In-memory sample store ───────────────────────────────────────────────

_samples: list[TaggedSample] = []
_loaded: bool = False


def load_tagged_samples(force_reload: bool = False) -> list[TaggedSample]:
    """Load tagged samples from disk into memory.

    Called once at startup. Returns the loaded sample list.
    """
    global _samples, _loaded

    if _loaded and not force_reload:
        return _samples

    if not TAGGED_SAMPLES_PATH.exists():
        logger.warning(
            "tagged_samples.json not found at {}. "
            "Run `python -m tools.build_tags` first.",
            TAGGED_SAMPLES_PATH,
        )
        _samples = []
        _loaded = True
        return _samples

    with open(TAGGED_SAMPLES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_samples = data.get("samples", [])
    _samples = [TaggedSample(**s) for s in raw_samples]
    _loaded = True
    logger.info("Loaded {} tagged samples from {}", len(_samples), TAGGED_SAMPLES_PATH)
    return _samples


def get_samples() -> list[TaggedSample]:
    """Get the loaded samples (auto-loads if needed)."""
    if not _loaded:
        load_tagged_samples()
    return _samples


# ── Scoring ──────────────────────────────────────────────────────────────


def score_sample(target_tags: TargetTagSet, sample: TaggedSample) -> tuple[float, dict[str, list[str]]]:
    """Compute weighted Jaccard-like overlap score across all dimensions.

    Returns
    -------
    tuple[float, dict]
        (total_score, matched_tags_per_dimension)
    """
    total = 0.0
    matched: dict[str, list[str]] = {}

    for dim, weight in DIMENSION_WEIGHTS.items():
        target = set(getattr(target_tags, dim, []))
        sample_tags = set(sample.tags.get(dim, []))

        if not target:
            continue

        overlap = target & sample_tags
        union = target | sample_tags
        jaccard = len(overlap) / len(union) if union else 0.0

        total += weight * jaccard
        if overlap:
            matched[dim] = sorted(overlap)

    return total, matched


def find_best_match(
    target_tags: TargetTagSet,
    breed_preference: Optional[str] = None,
    top_k: int = 1,
) -> list[MatchResult]:
    """Find the best-matching samples for the given target tags.

    Parameters
    ----------
    target_tags : TargetTagSet
        LLM-generated target tag set.
    breed_preference : str | None
        Optional breed preference for a small score boost.
    top_k : int
        Number of top matches to return.

    Returns
    -------
    list[MatchResult]
        Top-K matches sorted by score descending.
    """
    samples = get_samples()
    if not samples:
        logger.warning("No tagged samples loaded — cannot match.")
        return []

    results: list[MatchResult] = []

    for sample in samples:
        score, matched_tags = score_sample(target_tags, sample)

        # Apply breed preference boost
        if breed_preference and sample.breed == breed_preference:
            score += BREED_BOOST

        results.append(
            MatchResult(
                sample=sample,
                score=score,
                matched_tags=matched_tags,
            )
        )

    # Sort by score descending, then by sample id for determinism
    results.sort(key=lambda r: (-r.score, r.sample.id))

    return results[:top_k]
