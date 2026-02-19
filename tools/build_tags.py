"""
Meowsformer — Build Tagged Samples
====================================
One-time build script that reads ``assets/audio_db/registry.json``,
extracts acoustic features from each WAV via librosa, applies rule-based
tags from all 5 dimensions, and writes ``assets/audio_db/tagged_samples.json``.

Usage::

    python -m tools.build_tags                # full run with acoustic features
    python -m tools.build_tags --skip-audio   # metadata tags only (no librosa)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
REGISTRY_PATH = ASSETS_DIR / "audio_db" / "registry.json"
OUTPUT_PATH = ASSETS_DIR / "audio_db" / "tagged_samples.json"

# Import tagging rules
sys.path.insert(0, str(PROJECT_ROOT))
from app.data.meow_catalog import tag_acoustic, tag_sample_metadata  # noqa: E402


def extract_acoustic_features(wav_path: Path) -> dict[str, Any]:
    """Extract acoustic features from a WAV file using librosa.

    Returns a dict with keys expected by ``tag_acoustic()``:
        - median_f0, duration, rms_energy, f0_slope, f0_std, rms_percentile
    """
    import librosa

    try:
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    except Exception as e:
        logger.warning("Failed to load {}: {}", wav_path.name, e)
        return {}

    duration = float(len(y) / sr) if sr > 0 else 0.0

    # RMS energy
    rms = np.sqrt(np.mean(y ** 2))

    # F0 via pYIN
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=60,
        fmax=1500,
        sr=sr,
    )
    voiced_f0 = f0[voiced_flag] if f0 is not None else np.array([])

    median_f0: float | None = None
    f0_slope: float | None = None
    f0_std: float | None = None

    if len(voiced_f0) > 0:
        median_f0 = float(np.median(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        # Slope: linear regression of f0 over time
        if len(voiced_f0) > 1:
            x = np.arange(len(voiced_f0), dtype=np.float64)
            coeffs = np.polyfit(x, voiced_f0, 1)
            f0_slope = float(coeffs[0])  # Hz per frame
        else:
            f0_slope = 0.0

    return {
        "median_f0": median_f0,
        "duration": duration,
        "rms_energy": float(rms),
        "f0_slope": f0_slope,
        "f0_std": f0_std,
    }


def compute_rms_percentiles(
    samples_with_features: list[dict[str, Any]],
) -> None:
    """Compute global RMS percentiles and set ``rms_percentile`` in place."""
    rms_values = [
        s["_features"]["rms_energy"]
        for s in samples_with_features
        if s.get("_features", {}).get("rms_energy") is not None
    ]
    if not rms_values:
        return

    arr = np.array(rms_values)
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))

    for s in samples_with_features:
        feat = s.get("_features", {})
        rms = feat.get("rms_energy")
        if rms is None:
            feat["rms_percentile"] = "mid"
        elif rms > p75:
            feat["rms_percentile"] = "high"
        elif rms < p25:
            feat["rms_percentile"] = "low"
        else:
            feat["rms_percentile"] = "mid"


def build(skip_audio: bool = False) -> None:
    """Main build pipeline."""
    logger.info("Loading registry from {}", REGISTRY_PATH)

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    samples: list[dict[str, Any]] = registry.get("samples", [])
    logger.info("Found {} samples", len(samples))

    # ── Phase 1: Extract acoustic features (optional) ────────────────
    if not skip_audio:
        logger.info("Extracting acoustic features (this may take a while)...")
        for i, sample in enumerate(samples):
            wav_path = ASSETS_DIR / sample["file_path"]
            if wav_path.exists():
                features = extract_acoustic_features(wav_path)
                sample["_features"] = features
            else:
                logger.debug("WAV not found: {}", wav_path)
                sample["_features"] = {}

            if (i + 1) % 50 == 0:
                logger.info("  Processed {}/{} samples", i + 1, len(samples))

        # Compute global RMS percentiles
        compute_rms_percentiles(samples)
    else:
        logger.info("Skipping audio feature extraction (--skip-audio)")
        for sample in samples:
            sample["_features"] = {}

    # ── Phase 2: Apply all tags ──────────────────────────────────────
    tagged_samples: list[dict[str, Any]] = []

    for sample in samples:
        # Metadata-based tags (dimensions 1, 2, 4, 5)
        tags = tag_sample_metadata(sample)

        # Acoustic tags (dimension 3)
        features = sample.get("_features", {})
        tags["acoustic"] = tag_acoustic(features)

        # Build output entry
        entry = {
            "id": sample["id"],
            "file_path": sample["file_path"],
            "breed": sample.get("breed", "Unknown"),
            "valence": sample.get("valence", 0.0),
            "arousal": sample.get("arousal", 0.0),
            "context": sample.get("context", "Unknown"),
            "tags": tags,
        }
        tagged_samples.append(entry)

    # ── Phase 3: Write output ────────────────────────────────────────
    output = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(tagged_samples),
        "skip_audio": skip_audio,
        "samples": tagged_samples,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.success("Wrote {} tagged samples to {}", len(tagged_samples), OUTPUT_PATH)

    # Quick stats
    total_tags = sum(
        sum(len(v) for v in s["tags"].values()) for s in tagged_samples
    )
    logger.info("Total tags assigned: {} (avg {:.1f} per sample)", total_tags, total_tags / len(tagged_samples) if tagged_samples else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build tagged_samples.json")
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip acoustic feature extraction (metadata tags only).",
    )
    args = parser.parse_args()
    build(skip_audio=args.skip_audio)
