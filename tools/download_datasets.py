"""
Meowsformer — Data Acquisition & Metadata Indexing Pipeline
============================================================
Downloads cat vocalisation corpora from Zenodo, extracts audio assets,
parses the CatMeows naming convention, pre-assigns Valence/Arousal
values, and writes a consolidated ``registry.json``.

Usage::

    python -m tools.download_datasets            # full pipeline
    python -m tools.download_datasets --skip-download  # index only
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# ──────────────────────────────── paths ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
RAW_DATA_DIR = ASSETS_DIR / "raw_data"
AUDIO_DB_DIR = ASSETS_DIR / "audio_db"

CATMEOWS_DIR = RAW_DATA_DIR / "catmeows"
MEOWSIC_DIR = RAW_DATA_DIR / "meowsic"

REGISTRY_PATH = AUDIO_DB_DIR / "registry.json"

# ──────────────────────────────── DOIs ──────────────────────────────────
CATMEOWS_DOI = "10.5281/zenodo.4007940"
MEOWSIC_DOI = "10.5281/zenodo.3245999"

# ───────────────────── Valence / Arousal presets ────────────────────────
#   Food     → positive anticipation, high arousal
#   Isolation → negative distress, moderately high arousal
#   Brushing  → neutral baseline (per-individual variation injected later)
CONTEXT_VA_PRESETS: dict[str, dict[str, float]] = {
    "F": {"valence": 0.5, "arousal": 0.9},
    "I": {"valence": -0.8, "arousal": 0.7},
    "B": {"valence": 0.0, "arousal": 0.5},   # overridden per individual
}

# ───────────────────── code → label mappings ────────────────────────────
CONTEXT_LABELS: dict[str, str] = {
    "B": "Brushing",
    "F": "Food",
    "I": "Isolation",
}

BREED_LABELS: dict[str, str] = {
    "MC": "Maine Coon",
    "EU": "European Shorthair",
}

SEX_LABELS: dict[str, str] = {
    "FN": "Female Neutered",
    "FI": "Female Intact",
    "MN": "Male Neutered",
    "MI": "Male Intact",
}

# ───────────── CatMeows filename regex ──────────────────────────────────
# Documented convention: C_NNNNN_BB_SS_OOOOO_RXX
# Actual files use alphanumeric IDs and plain-numeric recordings, e.g.:
#   B_ANI01_MC_FN_SIM01_101.wav      (standard)
#   I_BLE01_EU_FN_DEL01_1SEQ1.wav    (sequence variant)
CATMEOWS_PATTERN = re.compile(
    r"^(?P<context>[BFI])_"
    r"(?P<cat_id>[A-Za-z]+\d+)_"
    r"(?P<breed>[A-Z]{2})_"
    r"(?P<sex>[A-Z]{2})_"
    r"(?P<name>[A-Za-z]+\d+)_"
    r"(?P<recording>\d+(?:SEQ\d+)?)$"
)

# ════════════════════════════════════════════════════════════════════════
#  Helper utilities
# ════════════════════════════════════════════════════════════════════════


def ensure_directories() -> None:
    """Create the full directory tree required by the pipeline."""
    for directory in (CATMEOWS_DIR, MEOWSIC_DIR, AUDIO_DB_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory: {}", directory)


def brushing_va_for_individual(cat_id: str) -> dict[str, float]:
    """Return a deterministic but per-cat Valence/Arousal for *Brushing*.

    Uses a stable hash so values look "random" yet are reproducible
    for the same ``cat_id`` string (e.g. ``"ANI01"``).
    """
    # deterministic hash: sum of ord values × Knuth constant
    KNUTH_CONST = 2654435761
    raw = sum(ord(c) for c in cat_id)
    hashed = (raw * KNUTH_CONST) % (2**32)
    t = (hashed % 1000) / 1000.0          # normalise → [0, 1)
    return {
        "valence": round(-0.2 + t * 0.6, 2),   # range −0.20 … +0.40
        "arousal": round(0.3 + t * 0.4, 2),     # range  0.30 …  0.70
    }


# ════════════════════════════════════════════════════════════════════════
#  Download
# ════════════════════════════════════════════════════════════════════════


def download_zenodo_dataset(doi: str, output_dir: Path) -> bool:
    """Download a Zenodo record via ``zenodo_get``.

    Returns ``True`` on success, ``False`` otherwise.
    """
    logger.info("⬇  Downloading DOI {} → {}", doi, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["zenodo_get", "-d", doi, "-o", str(output_dir)],
            capture_output=True,
            text=True,
            timeout=1200,  # 20-minute safety net for large files
        )
        if result.returncode != 0:
            logger.error("zenodo_get stderr:\n{}", result.stderr)
            return False
        logger.success("Download complete for DOI {}", doi)
        return True

    except FileNotFoundError:
        logger.error(
            "zenodo_get not found. Install with: pip install zenodo-get"
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("Download timed out for DOI {}", doi)
        return False


# ════════════════════════════════════════════════════════════════════════
#  Archive extraction
# ════════════════════════════════════════════════════════════════════════


def extract_archives(target_dir: Path) -> int:
    """Extract all ``.zip`` and ``.tar.gz`` archives found in *target_dir*.

    Returns the number of archives successfully extracted.
    """
    extracted = 0

    for archive in sorted(target_dir.iterdir()):
        if archive.suffix == ".zip":
            logger.info("Extracting ZIP: {}", archive.name)
            try:
                with zipfile.ZipFile(archive, "r") as zf:
                    zf.extractall(target_dir)
                extracted += 1
            except zipfile.BadZipFile:
                logger.warning("Bad ZIP — skipping {}", archive.name)

        elif archive.name.endswith(".tar.gz") or archive.name.endswith(".tgz"):
            logger.info("Extracting TAR: {}", archive.name)
            try:
                with tarfile.open(archive, "r:gz") as tf:
                    tf.extractall(target_dir)
                extracted += 1
            except tarfile.TarError:
                logger.warning("Bad TAR — skipping {}", archive.name)

    logger.info("Extracted {} archive(s) in {}", extracted, target_dir)
    return extracted


# ════════════════════════════════════════════════════════════════════════
#  WAV collection
# ════════════════════════════════════════════════════════════════════════


def collect_wav_files(source_dir: Path) -> list[Path]:
    """Recursively gather all ``.wav`` files under *source_dir*."""
    if not source_dir.exists():
        logger.warning("Source directory does not exist: {}", source_dir)
        return []
    wavs = sorted(source_dir.rglob("*.wav"))
    logger.info("Found {} .wav file(s) in {}", len(wavs), source_dir)
    return wavs


# ════════════════════════════════════════════════════════════════════════
#  Filename parsing
# ════════════════════════════════════════════════════════════════════════


def parse_catmeows_filename(filename: str) -> Optional[dict[str, Any]]:
    """Parse a CatMeows filename and return structured metadata.

    Naming convention ``C_NNNNN_BB_SS_OOOOO_RXX``:

    * **C**       — Context: ``B`` (Brushing), ``F`` (Food), ``I`` (Isolation)
    * **NNNNN**   — Numeric cat identifier
    * **BB**      — Breed: ``MC`` (Maine Coon), ``EU`` (European Shorthair)
    * **SS**      — Sex / neuter status (e.g. ``FN``, ``MI``)
    * **OOOOO**   — Individual cat name / code
    * **RXX**     — Recording index

    Returns ``None`` when the filename does not match the pattern.
    """
    stem = Path(filename).stem
    match = CATMEOWS_PATTERN.match(stem)
    if not match:
        logger.debug("No CatMeows match for '{}'", filename)
        return None

    g = match.groupdict()
    ctx = g["context"]

    # Valence / Arousal assignment
    if ctx == "B":
        va = brushing_va_for_individual(g["cat_id"])
    else:
        va = CONTEXT_VA_PRESETS[ctx]

    return {
        "context_code": ctx,
        "context": CONTEXT_LABELS.get(ctx, "Unknown"),
        "cat_id": g["cat_id"],
        "breed_code": g["breed"],
        "breed": BREED_LABELS.get(g["breed"], g["breed"]),
        "sex_code": g["sex"],
        "sex": SEX_LABELS.get(g["sex"], g["sex"]),
        "cat_name": g["name"],
        "recording": g["recording"],
        "valence": va["valence"],
        "arousal": va["arousal"],
    }


# ════════════════════════════════════════════════════════════════════════
#  Registry builder
# ════════════════════════════════════════════════════════════════════════


def build_registry(
    catmeows_dir: Path,
    meowsic_dir: Path,
) -> dict[str, Any]:
    """Scan dataset directories and build a complete metadata registry.

    Returns a dict ready for JSON serialisation::

        {
            "version": "1.0",
            "generated_at": "<ISO timestamp>",
            "datasets": { ... summary ... },
            "total_samples": N,
            "samples": [ ... ]
        }
    """
    samples: list[dict[str, Any]] = []

    # ── CatMeows ──────────────────────────────────────────────────────
    catmeow_wavs = collect_wav_files(catmeows_dir)
    for wav in catmeow_wavs:
        try:
            rel = wav.relative_to(ASSETS_DIR)
        except ValueError:
            rel = wav
        entry: dict[str, Any] = {
            "id": wav.stem,
            "dataset": "catmeows",
            "file_path": str(rel),
            "filename": wav.name,
        }
        parsed = parse_catmeows_filename(wav.name)
        if parsed:
            entry.update(parsed)
        else:
            entry.update({"context": "Unknown", "valence": 0.0, "arousal": 0.5})
        samples.append(entry)

    # ── Meowsic ───────────────────────────────────────────────────────
    meowsic_wavs = collect_wav_files(meowsic_dir)
    for wav in meowsic_wavs:
        try:
            rel = wav.relative_to(ASSETS_DIR)
        except ValueError:
            rel = wav
        entry = {
            "id": wav.stem,
            "dataset": "meowsic",
            "file_path": str(rel),
            "filename": wav.name,
            "context": "Meowsic",
            "valence": 0.0,
            "arousal": 0.5,
        }
        samples.append(entry)

    registry: dict[str, Any] = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "catmeows": {
                "doi": CATMEOWS_DOI,
                "total_samples": len(catmeow_wavs),
            },
            "meowsic": {
                "doi": MEOWSIC_DOI,
                "total_samples": len(meowsic_wavs),
            },
        },
        "total_samples": len(samples),
        "samples": samples,
    }

    logger.info(
        "Registry built — {} CatMeows + {} Meowsic = {} total",
        len(catmeow_wavs),
        len(meowsic_wavs),
        len(samples),
    )
    return registry


def save_registry(registry: dict[str, Any], output_path: Path) -> None:
    """Persist the registry dict to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, ensure_ascii=False)
    logger.success(
        "Registry saved → {} ({} samples)",
        output_path,
        registry.get("total_samples", "?"),
    )


# ════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ════════════════════════════════════════════════════════════════════════


def run_pipeline(*, skip_download: bool = False) -> bool:
    """Execute the full data-acquisition pipeline.

    Parameters
    ----------
    skip_download : bool
        When ``True`` skip the ``zenodo_get`` step and only
        (re-)build the metadata index from existing files.
    """
    logger.info("=== Meowsformer Data Acquisition Pipeline ===")

    # 1. Directory scaffold
    ensure_directories()

    # 2. Download (optional)
    if not skip_download:
        cat_ok = download_zenodo_dataset(CATMEOWS_DOI, CATMEOWS_DIR)
        meo_ok = download_zenodo_dataset(MEOWSIC_DOI, MEOWSIC_DIR)
        if not (cat_ok or meo_ok):
            logger.error("Both downloads failed — aborting.")
            return False

        # 3. Extract
        extract_archives(CATMEOWS_DIR)
        extract_archives(MEOWSIC_DIR)
    else:
        logger.info("--skip-download active; working with existing files")

    # 4. Build & save registry
    registry = build_registry(CATMEOWS_DIR, MEOWSIC_DIR)
    save_registry(registry, REGISTRY_PATH)

    logger.success("=== Pipeline complete ===")
    return True


# ════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meowsformer — download & index cat vocalisation datasets",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Zenodo download; only rebuild the metadata index",
    )
    args = parser.parse_args()
    success = run_pipeline(skip_download=args.skip_download)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
