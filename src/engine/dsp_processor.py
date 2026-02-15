"""
Meowsformer — DSP Processing Engine
=====================================
Core audio processing module implementing:

- **Intent → Valence/Arousal (VA) space mapping** based on Russell's
  Circumplex Model of Affect, adapted for feline bioacoustics.
- **Dynamic audio retrieval** via Euclidean distance nearest-neighbour
  search over the sample registry's VA annotations.
- **PSOLA-based prosody transformation** (Pitch-Synchronous Overlap-and-Add)
  decomposed into WSOLA time-stretching + resampling, with breed-specific
  f0 adjustment and arousal-driven envelope shaping.

References
----------
- Russell, J. A. (1980). A circumplex model of affect.
- Moulines, E. & Charpentier, F. (1990). Pitch-synchronous waveform
  processing techniques for text-to-speech synthesis using diphones.
- Latos, M. et al. (2020). CatMeows dataset (Zenodo 10.5281/zenodo.4007940).

Usage::

    from src.engine.dsp_processor import (
        map_intent_to_va,
        get_best_match,
        apply_prosody_transform,
        synthesize_meow,
    )

    # Quick synthesis
    audio, sr, match = synthesize_meow("Affiliative", breed="Maine Coon")

    # Step-by-step
    va = map_intent_to_va("Requesting")
    matches = get_best_match(va.valence, va.arousal, top_k=3)
    audio, sr = apply_prosody_transform("sample.wav", +2.0, 0.9,
                                         breed="Kitten", arousal=0.8)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import soundfile as sf
from loguru import logger

# ── Optional: pytsmod for high-quality WSOLA time-stretching ─────────
try:
    import pytsmod

    HAS_PYTSMOD = True
except ImportError:
    HAS_PYTSMOD = False
    logger.warning(
        "pytsmod not installed; falling back to librosa for time-stretching. "
        "Install with: pip install pytsmod"
    )


# ──────────────────────────────── paths ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
REGISTRY_PATH = ASSETS_DIR / "audio_db" / "registry.json"


# ════════════════════════════════════════════════════════════════════════
#  1. Intent → VA Space Mapping
# ════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class VAPoint:
    """A point in the two-dimensional Valence–Arousal affective space.

    Attributes
    ----------
    valence : float
        Emotional polarity in **[-1, 1]**.
        −1 = maximally negative (fear, anger),
        +1 = maximally positive (joy, affection).
    arousal : float
        Activation level in **[0, 1]**.
        0 = very calm / sleepy,
        1 = very excited / agitated.
    """

    valence: float
    arousal: float

    def distance_to(self, other: VAPoint) -> float:
        """Euclidean distance to another point in VA space."""
        return math.sqrt(
            (self.valence - other.valence) ** 2
            + (self.arousal - other.arousal) ** 2
        )


# ── Intent mapping matrix ────────────────────────────────────────────
# Maps human communicative intents to target VA coordinates.
# Values are empirically calibrated from cat behaviour studies and the
# CatMeows dataset annotations (Brushing≈neutral, Food≈positive-high,
# Isolation≈negative-high).
INTENT_VA_MAP: dict[str, VAPoint] = {
    # Positive / social
    "Affiliative": VAPoint(valence=0.70, arousal=0.35),  # friendly greeting
    "Contentment": VAPoint(valence=0.80, arousal=0.15),  # purring / relaxed
    "Play": VAPoint(valence=0.60, arousal=0.85),  # playful chirp

    # Requesting / need-based
    "Requesting": VAPoint(valence=0.30, arousal=0.75),  # food / attention demand
    "Solicitation": VAPoint(valence=0.40, arousal=0.60),  # gentle asking

    # Negative / defensive
    "Agonistic": VAPoint(valence=-0.80, arousal=0.90),  # hiss / growl threat
    "Distress": VAPoint(valence=-0.70, arousal=0.85),  # pain / fear cry
    "Frustration": VAPoint(valence=-0.50, arousal=0.70),  # irritation

    # Neutral / informational
    "Alert": VAPoint(valence=0.00, arousal=0.65),  # attention signal
    "Neutral": VAPoint(valence=0.00, arousal=0.40),  # baseline vocalisation
}


def map_intent_to_va(intent: str) -> VAPoint:
    """Map a human communicative intent label to VA coordinates.

    Parameters
    ----------
    intent : str
        One of the recognised intent labels (case-insensitive).
        See :data:`INTENT_VA_MAP` for the full list.

    Returns
    -------
    VAPoint
        Target (Valence, Arousal) coordinates.

    Raises
    ------
    ValueError
        If *intent* is not in the mapping.
    """
    normalised = intent.strip().title()
    if normalised not in INTENT_VA_MAP:
        available = ", ".join(sorted(INTENT_VA_MAP.keys()))
        raise ValueError(
            f"Unknown intent '{intent}'. Available intents: {available}"
        )
    va = INTENT_VA_MAP[normalised]
    logger.debug("Intent '{}' → VA({:.2f}, {:.2f})", intent, va.valence, va.arousal)
    return va


def get_all_intents() -> dict[str, VAPoint]:
    """Return a copy of the full intent → VA mapping dictionary."""
    return dict(INTENT_VA_MAP)


# ════════════════════════════════════════════════════════════════════════
#  2. Dynamic Audio Retrieval (VA-space nearest-neighbour)
# ════════════════════════════════════════════════════════════════════════


@dataclass
class SampleMatch:
    """Result of a VA-space nearest-neighbour lookup."""

    sample_id: str
    file_path: str
    distance: float
    valence: float
    arousal: float
    breed: str = ""
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def load_registry(registry_path: Optional[Path] = None) -> dict[str, Any]:
    """Load the audio sample registry from its JSON file.

    Parameters
    ----------
    registry_path : Path, optional
        Override for the default ``assets/audio_db/registry.json``.

    Raises
    ------
    FileNotFoundError
        If the registry file does not exist.
    """
    path = registry_path or REGISTRY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Registry not found at {path}. "
            "Run the data acquisition pipeline first: "
            "python -m tools.download_datasets"
        )
    with open(path, "r", encoding="utf-8") as fh:
        registry = json.load(fh)
    logger.debug(
        "Loaded registry v{} with {} samples",
        registry.get("version", "?"),
        registry.get("total_samples", 0),
    )
    return registry


def get_best_match(
    target_v: float,
    target_a: float,
    *,
    registry_path: Optional[Path] = None,
    breed_filter: Optional[str] = None,
    context_filter: Optional[str] = None,
    top_k: int = 1,
) -> list[SampleMatch]:
    """Find the *top_k* closest audio sample(s) in VA space.

    Computes the Euclidean distance between the target vector
    ``(target_v, target_a)`` and every sample's annotated
    ``(valence, arousal)`` in ``registry.json``.

    Parameters
    ----------
    target_v : float
        Target Valence in ``[-1, 1]``.
    target_a : float
        Target Arousal in ``[0, 1]``.
    registry_path : Path, optional
        Override path to the registry JSON.
    breed_filter : str, optional
        If set, only consider samples from this breed
        (e.g. ``"Maine Coon"``).
    context_filter : str, optional
        If set, only consider samples recorded in this context
        (e.g. ``"Food"``).
    top_k : int
        Number of nearest matches to return (default 1).

    Returns
    -------
    list[SampleMatch]
        Matches sorted by ascending Euclidean distance.
    """
    registry = load_registry(registry_path)
    samples = registry.get("samples", [])

    if not samples:
        logger.warning("Registry contains no samples")
        return []

    target = VAPoint(valence=target_v, arousal=target_a)
    scored: list[SampleMatch] = []

    for s in samples:
        # Optional filters
        if breed_filter and s.get("breed", "") != breed_filter:
            continue
        if context_filter and s.get("context", "") != context_filter:
            continue

        sv = float(s.get("valence", 0.0))
        sa = float(s.get("arousal", 0.5))
        dist = target.distance_to(VAPoint(valence=sv, arousal=sa))

        scored.append(
            SampleMatch(
                sample_id=s.get("id", ""),
                file_path=s.get("file_path", ""),
                distance=dist,
                valence=sv,
                arousal=sa,
                breed=s.get("breed", ""),
                context=s.get("context", ""),
                metadata=s,
            )
        )

    scored.sort(key=lambda m: m.distance)
    results = scored[:top_k]

    if results:
        best = results[0]
        logger.info(
            "Best VA match for ({:.2f}, {:.2f}): id={} dist={:.4f} breed={}",
            target_v,
            target_a,
            best.sample_id,
            best.distance,
            best.breed,
        )
    else:
        logger.warning(
            "No samples matched filters (breed={}, context={})",
            breed_filter,
            context_filter,
        )

    return results


# ════════════════════════════════════════════════════════════════════════
#  3. Breed Physiological f0 Baselines
# ════════════════════════════════════════════════════════════════════════

# Fundamental frequency baselines (Hz) grounded in feline bioacoustics.
# Larger cats have longer vocal folds → lower f0; kittens have short
# vocal tracts → higher f0.
BREED_F0_BASELINES: dict[str, float] = {
    "Maine Coon": 420.0,  # large breed, deep voice
    "European Shorthair": 550.0,  # standard medium build
    "Siamese": 620.0,  # notably vocal, higher pitch
    "Persian": 480.0,  # medium-large, moderate pitch
    "Bengal": 560.0,  # athletic medium build
    "British Shorthair": 500.0,  # stocky build
    "Kitten": 750.0,  # juvenile, very short vocal tract
    "Default": 550.0,  # fallback
}


def get_breed_f0(breed: str) -> float:
    """Return the physiological f0 baseline (Hz) for a cat breed.

    Supports exact matches and case-insensitive partial matches.
    Falls back to **550 Hz** for unknown breeds.

    Parameters
    ----------
    breed : str
        Breed name (e.g. ``"Maine Coon"``, ``"Kitten"``).
    """
    # Exact match
    if breed in BREED_F0_BASELINES:
        return BREED_F0_BASELINES[breed]

    # Case-insensitive partial match
    breed_lower = breed.lower()
    for name, f0 in BREED_F0_BASELINES.items():
        if name.lower() in breed_lower or breed_lower in name.lower():
            return f0

    logger.debug("Unknown breed '{}'; using default f0 = 550 Hz", breed)
    return BREED_F0_BASELINES["Default"]


# ════════════════════════════════════════════════════════════════════════
#  4. PSOLA Prosody Transform
# ════════════════════════════════════════════════════════════════════════


def _estimate_f0(y: np.ndarray, sr: int) -> float:
    """Estimate the median fundamental frequency of a signal.

    Uses librosa's **pYIN** algorithm (probabilistic YIN) for robust
    multi-pitch estimation, returning the median of voiced frames.
    """
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
    )
    # Keep only voiced frames
    if voiced_flag is not None:
        voiced_f0 = f0[voiced_flag]
    else:
        voiced_f0 = f0[~np.isnan(f0)]

    if len(voiced_f0) == 0:
        logger.warning("No voiced frames detected; assuming f0 = 440 Hz")
        return 440.0

    median_f0 = float(np.median(voiced_f0))
    logger.debug("Estimated median f0 = {:.1f} Hz", median_f0)
    return median_f0


def _time_stretch_wsola(y: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Time-stretch audio by *factor* using WSOLA (or librosa fallback).

    Parameters
    ----------
    y : np.ndarray
        Mono audio signal (1-D).
    sr : int
        Sample rate (used only by the librosa fallback).
    factor : float
        Stretch factor. ``>1`` = output is *factor*× longer (slower),
        ``<1`` = shorter (faster).

    Returns
    -------
    np.ndarray
        Time-stretched audio.
    """
    if abs(factor - 1.0) < 0.01:
        return y  # No meaningful change

    if HAS_PYTSMOD:
        try:
            # pytsmod expects shape (channels, samples)
            y_2d = y[np.newaxis, :] if y.ndim == 1 else y
            stretched = pytsmod.wsola(y_2d, factor)
            # pytsmod may return 1-D or 2-D depending on version
            if stretched.ndim == 2:
                stretched = stretched[0]
            return stretched
        except Exception as exc:
            logger.warning(
                "pytsmod.wsola failed ({}); falling back to librosa", exc
            )

    # Fallback: librosa  (rate > 1 = faster — inverse of our convention)
    return librosa.effects.time_stretch(y, rate=1.0 / factor)


def _apply_arousal_envelope(
    y: np.ndarray,
    sr: int,
    arousal: float,
) -> np.ndarray:
    """Shape the amplitude envelope based on arousal intensity.

    High arousal  → sharper attack, rapid decay  (urgent / staccato)
    Low  arousal  → gentle onset,  sustained body (calm / legato)

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    sr : int
        Sample rate.
    arousal : float
        Arousal level in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Envelope-shaped audio (same length).
    """
    n = len(y)
    if n == 0:
        return y

    # Clamp arousal to valid range
    arousal = max(0.0, min(1.0, arousal))

    t = np.linspace(0.0, 1.0, n)

    # Arousal-dependent envelope parameters
    attack_speed = 2.0 + arousal * 8.0  # 2 (calm) … 10 (urgent)
    decay_speed = 1.0 + arousal * 4.0  # 1 (calm) …  5 (urgent)
    peak_pos = 0.15 + (1.0 - arousal) * 0.25  # 0.15 (urgent) … 0.40 (calm)

    envelope = np.ones(n, dtype=np.float64)

    # Attack phase (rising)
    attack_mask = t < peak_pos
    if np.any(attack_mask):
        t_attack = t[attack_mask] / peak_pos
        envelope[attack_mask] = 1.0 - np.exp(-attack_speed * t_attack)

    # Decay phase (falling)
    decay_mask = t >= peak_pos
    if np.any(decay_mask):
        t_decay = (t[decay_mask] - peak_pos) / (1.0 - peak_pos + 1e-9)
        envelope[decay_mask] = np.exp(-decay_speed * t_decay)

    return y * envelope


def apply_prosody_transform(
    audio_path: str | Path,
    target_pitch_shift: float = 0.0,
    duration_factor: float = 1.0,
    *,
    breed: Optional[str] = None,
    arousal: Optional[float] = None,
    output_path: Optional[str | Path] = None,
    sr: int = 22050,
) -> tuple[np.ndarray, int]:
    """Apply PSOLA-based prosody transformation to a cat vocalisation.

    The algorithm is implemented as a **time-stretch + resample**
    decomposition — mathematically equivalent to pitch-synchronous
    overlap-and-add (PSOLA):

    1. **Analyse** — estimate source f0 via pYIN.
    2. **Breed adjustment** — compute additional semitone offset to
       match the target breed's physiological f0 baseline.
    3. **Arousal modulation** — compress / stretch time according to
       the arousal parameter (high arousal ⇒ faster, more urgent).
    4. **Time-stretch** — WSOLA preserves pitch while changing duration
       by the combined factor ``pitch_ratio × effective_duration``.
    5. **Resample** — change sample count to shift pitch while restoring
       the target duration.
    6. **Envelope shaping** — apply arousal-driven attack / decay curve.
    7. **Normalise** — prevent clipping.

    Parameters
    ----------
    audio_path : str | Path
        Path to the source ``.wav`` file.
    target_pitch_shift : float
        Pitch shift in **semitones** (positive = higher, negative = lower).
    duration_factor : float
        Explicit time-stretch factor (``>1`` = slower, ``<1`` = faster).
    breed : str, optional
        Target cat breed for physiological f0 correction.
    arousal : float, optional
        Arousal level ``[0, 1]`` for envelope & tempo shaping.
    output_path : str | Path, optional
        Save the processed audio to this path.
    sr : int
        Target sample rate (default 22 050 Hz).

    Returns
    -------
    tuple[np.ndarray, int]
        ``(processed_audio, sample_rate)``

    Raises
    ------
    FileNotFoundError
        If *audio_path* does not exist.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(
        "PSOLA transform: {} | pitch={:+.1f}st dur={:.2f}x breed={} arousal={}",
        audio_path.name,
        target_pitch_shift,
        duration_factor,
        breed or "auto",
        arousal,
    )

    # ── 1. Load ──────────────────────────────────────────────────────
    y, orig_sr = librosa.load(str(audio_path), sr=sr, mono=True)

    if len(y) == 0:
        logger.warning("Empty audio file: {}", audio_path)
        return y, sr

    # ── 2. Estimate source f0 & compute breed-adjusted shift ─────────
    source_f0 = _estimate_f0(y, sr)
    total_shift_st = target_pitch_shift

    if breed:
        target_f0 = get_breed_f0(breed)
        breed_shift_st = 12.0 * math.log2(target_f0 / (source_f0 + 1e-9))
        # 50 % breed adaptation blended with the explicit shift
        total_shift_st += breed_shift_st * 0.5
        logger.debug(
            "Breed adjustment: src_f0={:.1f}Hz target_f0={:.1f}Hz "
            "breed_shift={:+.2f}st → total={:+.2f}st",
            source_f0,
            target_f0,
            breed_shift_st,
            total_shift_st,
        )

    # ── 3. Arousal-based duration modulation ─────────────────────────
    effective_dur = duration_factor
    if arousal is not None:
        # High arousal → compress (×0.85), low → expand (×1.15)
        arousal_time_mod = 1.0 - (arousal - 0.5) * 0.3
        effective_dur *= arousal_time_mod
        logger.debug(
            "Arousal time mod: a={:.2f} mod={:.3f} eff_dur={:.3f}",
            arousal,
            arousal_time_mod,
            effective_dur,
        )

    # ── 4. PSOLA = time-stretch (WSOLA) + resample ──────────────────
    pitch_ratio = 2.0 ** (total_shift_st / 12.0)

    # stretch_factor: how much longer the intermediate signal is.
    # After resampling by 1/pitch_ratio the final duration equals
    # effective_dur × original_duration.
    stretch_factor = pitch_ratio * effective_dur

    # 4a. Time-stretch with WSOLA (preserves pitch, changes length)
    y = _time_stretch_wsola(y, sr, stretch_factor)

    # 4b. Resample to shift pitch (changes pitch, corrects length)
    if abs(pitch_ratio - 1.0) > 0.01:
        original_len = len(y)
        target_len = int(original_len / pitch_ratio)
        if target_len > 0:
            y = np.interp(
                np.linspace(0, original_len - 1, target_len),
                np.arange(original_len),
                y,
            ).astype(np.float32)

    # ── 5. Arousal envelope ──────────────────────────────────────────
    if arousal is not None:
        y = _apply_arousal_envelope(y, sr, arousal)

    # ── 6. Normalise ─────────────────────────────────────────────────
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.95

    y = y.astype(np.float32)

    # ── 7. Optionally save ───────────────────────────────────────────
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), y, sr)
        logger.success("Saved processed audio → {}", output_path)

    return y, sr


# ════════════════════════════════════════════════════════════════════════
#  5. High-level Pipeline
# ════════════════════════════════════════════════════════════════════════


def synthesize_meow(
    intent: str,
    *,
    breed: str = "Default",
    registry_path: Optional[Path] = None,
    output_path: Optional[str | Path] = None,
) -> tuple[np.ndarray, int, SampleMatch]:
    """End-to-end meow synthesis: intent → VA → retrieve → transform.

    Parameters
    ----------
    intent : str
        Human communicative intent (e.g. ``"Affiliative"``).
    breed : str
        Target cat breed for physiological pitch tuning.
    registry_path : Path, optional
        Override path to the sample registry.
    output_path : str | Path, optional
        Save the final synthesised audio.

    Returns
    -------
    tuple[np.ndarray, int, SampleMatch]
        ``(audio_array, sample_rate, matched_sample_info)``
    """
    # 1 — Intent → VA
    target = map_intent_to_va(intent)

    # 2 — Find best matching sample
    matches = get_best_match(
        target.valence,
        target.arousal,
        registry_path=registry_path,
        top_k=1,
    )

    if not matches:
        raise RuntimeError("No matching samples found in registry")

    best = matches[0]

    # Resolve path relative to assets
    wav_path = ASSETS_DIR / best.file_path
    if not wav_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {wav_path}. "
            "Ensure the raw audio has been downloaded "
            "(python -m tools.download_datasets)."
        )

    # 3 — Compute prosody deltas from VA mismatch
    va_pitch_hint = (target.valence - best.valence) * 2.0  # semitones
    duration_factor = 1.0 + (best.arousal - target.arousal) * 0.5

    # 4 — PSOLA transform
    audio, sr = apply_prosody_transform(
        audio_path=wav_path,
        target_pitch_shift=va_pitch_hint,
        duration_factor=duration_factor,
        breed=breed,
        arousal=target.arousal,
        output_path=output_path,
    )

    logger.success(
        "Synthesised meow: intent='{}' breed='{}' match='{}' dur={:.2f}s",
        intent,
        breed,
        best.sample_id,
        len(audio) / sr,
    )
    return audio, sr, best
