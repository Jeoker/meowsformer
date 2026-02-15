"""
Tests for tools.download_datasets
==================================
Covers filename parsing, VA value assignment, directory scaffolding,
WAV collection, archive extraction, registry building / saving, and
the download function (mocked).

All test filenames follow the **real** CatMeows naming convention
observed in the Zenodo dataset (DOI 10.5281/zenodo.4007940):

    C_AAANN_BB_SS_OOONN_NNN.wav          (standard)
    C_AAANN_BB_SS_OOONN_NSEQN.wav        (sequence variant)

e.g.  B_ANI01_MC_FN_SIM01_101.wav
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.download_datasets import (
    CONTEXT_VA_PRESETS,
    CATMEOWS_PATTERN,
    brushing_va_for_individual,
    build_registry,
    collect_wav_files,
    download_zenodo_dataset,
    ensure_directories,
    extract_archives,
    parse_catmeows_filename,
    save_registry,
)


class TestParseCatMeowsFilename(unittest.TestCase):
    """Validate CatMeows naming convention parser."""

    # ── Food context ──────────────────────────────────────────────────

    def test_food_context_basic(self):
        result = parse_catmeows_filename("F_BAC01_MC_MN_SIM01_101.wav")
        self.assertIsNotNone(result)
        self.assertEqual(result["context_code"], "F")
        self.assertEqual(result["context"], "Food")
        self.assertEqual(result["cat_id"], "BAC01")
        self.assertEqual(result["breed_code"], "MC")
        self.assertEqual(result["breed"], "Maine Coon")
        self.assertEqual(result["sex_code"], "MN")
        self.assertEqual(result["sex"], "Male Neutered")
        self.assertEqual(result["cat_name"], "SIM01")
        self.assertEqual(result["recording"], "101")

    def test_food_va_values(self):
        result = parse_catmeows_filename("F_MAG01_EU_FN_FED01_201.wav")
        self.assertAlmostEqual(result["valence"], 0.5)
        self.assertAlmostEqual(result["arousal"], 0.9)

    def test_food_eu_breed(self):
        result = parse_catmeows_filename("F_BLE01_EU_FN_DEL01_103.wav")
        self.assertIsNotNone(result)
        self.assertEqual(result["breed"], "European Shorthair")
        self.assertEqual(result["sex"], "Female Neutered")

    # ── Isolation context ─────────────────────────────────────────────

    def test_isolation_context_basic(self):
        result = parse_catmeows_filename("I_DAK01_MC_FN_SIM01_116.wav")
        self.assertIsNotNone(result)
        self.assertEqual(result["context_code"], "I")
        self.assertEqual(result["context"], "Isolation")
        self.assertEqual(result["breed"], "Maine Coon")
        self.assertEqual(result["cat_id"], "DAK01")

    def test_isolation_va_values(self):
        result = parse_catmeows_filename("I_BLE01_EU_FN_DEL01_205.wav")
        self.assertAlmostEqual(result["valence"], -0.8)
        self.assertAlmostEqual(result["arousal"], 0.7)

    # ── Brushing context ──────────────────────────────────────────────

    def test_brushing_context_basic(self):
        result = parse_catmeows_filename("B_ANI01_MC_FN_SIM01_101.wav")
        self.assertIsNotNone(result)
        self.assertEqual(result["context_code"], "B")
        self.assertEqual(result["context"], "Brushing")

    def test_brushing_va_varies_per_individual(self):
        """Two different cats in Brushing context must yield different VA."""
        r1 = parse_catmeows_filename("B_ANI01_MC_FN_SIM01_101.wav")
        r2 = parse_catmeows_filename("B_CAN01_EU_FN_GIA01_101.wav")
        # Both must be in the valid range
        for r in (r1, r2):
            self.assertGreaterEqual(r["valence"], -0.2)
            self.assertLessEqual(r["valence"], 0.4)
            self.assertGreaterEqual(r["arousal"], 0.3)
            self.assertLessEqual(r["arousal"], 0.7)
        # Different cats → at least one VA axis should differ
        self.assertTrue(
            r1["valence"] != r2["valence"] or r1["arousal"] != r2["arousal"],
            "Brushing VA should vary per individual",
        )

    def test_brushing_va_deterministic(self):
        """Same cat_id must always produce the same VA across recordings."""
        a = parse_catmeows_filename("B_ANI01_MC_FN_SIM01_101.wav")
        b = parse_catmeows_filename("B_ANI01_MC_FN_SIM01_302.wav")
        self.assertEqual(a["valence"], b["valence"])
        self.assertEqual(a["arousal"], b["arousal"])

    # ── Sequence variant ──────────────────────────────────────────────

    def test_seq_variant(self):
        result = parse_catmeows_filename("I_BLE01_EU_FN_DEL01_1SEQ1.wav")
        self.assertIsNotNone(result)
        self.assertEqual(result["context_code"], "I")
        self.assertEqual(result["recording"], "1SEQ1")

    def test_seq_variant_brushing(self):
        result = parse_catmeows_filename("B_CAN01_EU_FN_GIA01_2SEQ2.wav")
        self.assertIsNotNone(result)
        self.assertEqual(result["recording"], "2SEQ2")

    # ── Invalid / edge-case filenames ─────────────────────────────────

    def test_invalid_filename_returns_none(self):
        self.assertIsNone(parse_catmeows_filename("random_noise.wav"))
        self.assertIsNone(parse_catmeows_filename("meow.wav"))
        self.assertIsNone(parse_catmeows_filename(""))

    def test_wrong_context_letter(self):
        self.assertIsNone(parse_catmeows_filename("X_ANI01_MC_FN_SIM01_101.wav"))

    def test_missing_fields(self):
        self.assertIsNone(parse_catmeows_filename("F_BAC01_MC.wav"))

    def test_stem_only_without_extension(self):
        """Parser should also work on bare stems (no .wav)."""
        result = parse_catmeows_filename("F_MAG01_EU_FN_FED01_301")
        self.assertIsNotNone(result)
        self.assertEqual(result["context_code"], "F")

    def test_female_intact(self):
        result = parse_catmeows_filename("B_BRI01_MC_FI_SIM01_201.wav")
        self.assertIsNotNone(result)
        self.assertEqual(result["sex_code"], "FI")
        self.assertEqual(result["sex"], "Female Intact")

    def test_male_intact(self):
        result = parse_catmeows_filename("I_NUL01_MC_MI_SIM01_301.wav")
        self.assertIsNotNone(result)
        self.assertEqual(result["sex_code"], "MI")
        self.assertEqual(result["sex"], "Male Intact")


class TestBrushingVA(unittest.TestCase):
    """Dedicated tests for the brushing VA helper."""

    def test_range(self):
        cat_ids = [
            "ANI01", "BAC01", "BRA01", "BRI01", "CAN01", "DAK01",
            "IND01", "JJX01", "MAG01", "MAT01", "MIN01", "NIG01",
            "NUL01", "REG01", "SPI01", "TIG01", "WHO01",
        ]
        for cid in cat_ids:
            va = brushing_va_for_individual(cid)
            self.assertGreaterEqual(va["valence"], -0.2)
            self.assertLessEqual(va["valence"], 0.4)
            self.assertGreaterEqual(va["arousal"], 0.3)
            self.assertLessEqual(va["arousal"], 0.7)

    def test_reproducibility(self):
        self.assertEqual(
            brushing_va_for_individual("ANI01"),
            brushing_va_for_individual("ANI01"),
        )

    def test_diversity(self):
        """At least *some* variation across real cat IDs."""
        cat_ids = [
            "ANI01", "BAC01", "BRA01", "BRI01", "CAN01", "DAK01",
            "IND01", "JJX01", "MAG01", "MAT01", "MIN01", "NIG01",
            "NUL01", "REG01", "SPI01", "TIG01", "WHO01",
        ]
        vals = {brushing_va_for_individual(c)["valence"] for c in cat_ids}
        self.assertGreater(len(vals), 5, "Expected more diversity in valence")


class TestEnsureDirectories(unittest.TestCase):
    def test_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = [Path(tmp) / "a" / "b", Path(tmp) / "c"]
            with patch("tools.download_datasets.CATMEOWS_DIR", dirs[0]), \
                 patch("tools.download_datasets.MEOWSIC_DIR", dirs[1]), \
                 patch("tools.download_datasets.AUDIO_DB_DIR", Path(tmp) / "d"):
                ensure_directories()
            for d in dirs:
                self.assertTrue(d.exists())

    def test_idempotent(self):
        """Calling twice should not raise."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "x"
            with patch("tools.download_datasets.CATMEOWS_DIR", d), \
                 patch("tools.download_datasets.MEOWSIC_DIR", d), \
                 patch("tools.download_datasets.AUDIO_DB_DIR", d):
                ensure_directories()
                ensure_directories()  # second call should be fine
            self.assertTrue(d.exists())


class TestCollectWavFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_finds_wav_recursively(self):
        base = Path(self.tmp)
        sub = base / "sub1" / "sub2"
        sub.mkdir(parents=True)
        (base / "top.wav").touch()
        (sub / "deep.wav").touch()
        (base / "readme.txt").touch()  # non-wav

        wavs = collect_wav_files(base)
        self.assertEqual(len(wavs), 2)
        names = {w.name for w in wavs}
        self.assertIn("top.wav", names)
        self.assertIn("deep.wav", names)

    def test_empty_directory(self):
        self.assertEqual(collect_wav_files(Path(self.tmp)), [])

    def test_nonexistent_directory(self):
        self.assertEqual(collect_wav_files(Path("/nonexistent_dir_xyz")), [])


class TestExtractArchives(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_extract_zip(self):
        zip_path = self.tmp / "test_data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sample.wav", b"\x00" * 100)

        count = extract_archives(self.tmp)
        self.assertEqual(count, 1)
        self.assertTrue((self.tmp / "sample.wav").exists())

    def test_skip_bad_zip(self):
        bad_zip = self.tmp / "corrupt.zip"
        bad_zip.write_bytes(b"this is not a zip file")
        count = extract_archives(self.tmp)
        self.assertEqual(count, 0)

    def test_empty_directory(self):
        count = extract_archives(self.tmp)
        self.assertEqual(count, 0)


class TestDownloadZenodoDataset(unittest.TestCase):
    """Test the download wrapper with mocked subprocess."""

    @patch("tools.download_datasets.subprocess.run")
    def test_success(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            ok = download_zenodo_dataset("10.5281/zenodo.0000000", Path(tmp))
        self.assertTrue(ok)
        mock_run.assert_called_once()

    @patch("tools.download_datasets.subprocess.run")
    def test_failure_nonzero_exit(self, mock_run: MagicMock):
        mock_run.return_value = MagicMock(returncode=1, stderr="error msg")
        with tempfile.TemporaryDirectory() as tmp:
            ok = download_zenodo_dataset("10.5281/zenodo.0000000", Path(tmp))
        self.assertFalse(ok)

    @patch(
        "tools.download_datasets.subprocess.run",
        side_effect=FileNotFoundError,
    )
    def test_zenodo_get_not_installed(self, _mock):
        with tempfile.TemporaryDirectory() as tmp:
            ok = download_zenodo_dataset("10.5281/zenodo.0000000", Path(tmp))
        self.assertFalse(ok)

    @patch(
        "tools.download_datasets.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="zenodo_get", timeout=10),
    )
    def test_timeout(self, _mock):
        with tempfile.TemporaryDirectory() as tmp:
            ok = download_zenodo_dataset("10.5281/zenodo.0000000", Path(tmp))
        self.assertFalse(ok)


class TestBuildRegistry(unittest.TestCase):
    """Build a registry from synthetic mock WAV files using real names."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.catmeows = self.tmp / "catmeows"
        self.meowsic = self.tmp / "meowsic"
        self.catmeows.mkdir()
        self.meowsic.mkdir()

        # Create mock CatMeows files (real naming convention)
        self.catmeow_names = [
            "F_BAC01_MC_MN_SIM01_101.wav",       # Food
            "I_BLE01_EU_FN_DEL01_205.wav",        # Isolation
            "B_ANI01_MC_FN_SIM01_101.wav",        # Brushing rec 1
            "B_ANI01_MC_FN_SIM01_302.wav",        # Brushing rec 2 (same cat)
            "I_CAN01_EU_FN_GIA01_1SEQ1.wav",      # Isolation SEQ variant
        ]
        for name in self.catmeow_names:
            (self.catmeows / name).touch()

        # Create mock Meowsic files
        self.meowsic_names = ["meow_sample_01.wav", "purr_clip_02.wav"]
        for name in self.meowsic_names:
            (self.meowsic / name).touch()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    @patch("tools.download_datasets.ASSETS_DIR")
    def test_total_samples(self, mock_assets):
        mock_assets.__class__ = Path
        mock_assets.return_value = self.tmp
        registry = build_registry(self.catmeows, self.meowsic)
        self.assertEqual(registry["total_samples"], 7)
        self.assertEqual(registry["datasets"]["catmeows"]["total_samples"], 5)
        self.assertEqual(registry["datasets"]["meowsic"]["total_samples"], 2)

    @patch("tools.download_datasets.ASSETS_DIR")
    def test_catmeows_food_metadata(self, mock_assets):
        mock_assets.__class__ = Path
        mock_assets.return_value = self.tmp
        registry = build_registry(self.catmeows, self.meowsic)
        food = [
            s for s in registry["samples"]
            if s.get("context_code") == "F"
        ]
        self.assertEqual(len(food), 1)
        self.assertEqual(food[0]["breed"], "Maine Coon")
        self.assertEqual(food[0]["cat_id"], "BAC01")
        self.assertAlmostEqual(food[0]["valence"], 0.5)
        self.assertAlmostEqual(food[0]["arousal"], 0.9)

    @patch("tools.download_datasets.ASSETS_DIR")
    def test_isolation_va(self, mock_assets):
        mock_assets.__class__ = Path
        mock_assets.return_value = self.tmp
        registry = build_registry(self.catmeows, self.meowsic)
        iso = [
            s for s in registry["samples"]
            if s.get("context_code") == "I"
        ]
        self.assertEqual(len(iso), 2)  # standard + SEQ
        for entry in iso:
            self.assertAlmostEqual(entry["valence"], -0.8)
            self.assertAlmostEqual(entry["arousal"], 0.7)

    @patch("tools.download_datasets.ASSETS_DIR")
    def test_meowsic_defaults(self, mock_assets):
        mock_assets.__class__ = Path
        mock_assets.return_value = self.tmp
        registry = build_registry(self.catmeows, self.meowsic)
        meo = [s for s in registry["samples"] if s["dataset"] == "meowsic"]
        self.assertEqual(len(meo), 2)
        for entry in meo:
            self.assertEqual(entry["context"], "Meowsic")
            self.assertAlmostEqual(entry["valence"], 0.0)
            self.assertAlmostEqual(entry["arousal"], 0.5)

    @patch("tools.download_datasets.ASSETS_DIR")
    def test_registry_has_version(self, mock_assets):
        mock_assets.__class__ = Path
        mock_assets.return_value = self.tmp
        registry = build_registry(self.catmeows, self.meowsic)
        self.assertEqual(registry["version"], "1.0")
        self.assertIn("generated_at", registry)

    @patch("tools.download_datasets.ASSETS_DIR")
    def test_brushing_same_cat_same_va(self, mock_assets):
        """Two recordings from the same cat should share VA values."""
        mock_assets.__class__ = Path
        mock_assets.return_value = self.tmp
        registry = build_registry(self.catmeows, self.meowsic)
        brushing = [
            s for s in registry["samples"]
            if s.get("context_code") == "B"
        ]
        self.assertEqual(len(brushing), 2)
        self.assertEqual(brushing[0]["valence"], brushing[1]["valence"])
        self.assertEqual(brushing[0]["arousal"], brushing[1]["arousal"])

    @patch("tools.download_datasets.ASSETS_DIR")
    def test_seq_variant_parsed(self, mock_assets):
        """SEQ-variant filenames should still be parsed correctly."""
        mock_assets.__class__ = Path
        mock_assets.return_value = self.tmp
        registry = build_registry(self.catmeows, self.meowsic)
        seq = [
            s for s in registry["samples"]
            if s.get("recording") == "1SEQ1"
        ]
        self.assertEqual(len(seq), 1)
        self.assertEqual(seq[0]["cat_id"], "CAN01")


class TestSaveRegistry(unittest.TestCase):
    def test_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub" / "registry.json"
            registry = {
                "version": "1.0",
                "total_samples": 1,
                "samples": [{"id": "test"}],
            }
            save_registry(registry, out)
            self.assertTrue(out.exists())

            with open(out, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["total_samples"], 1)
            self.assertEqual(loaded["samples"][0]["id"], "test")

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a" / "b" / "c" / "reg.json"
            save_registry({"version": "1.0", "total_samples": 0, "samples": []}, out)
            self.assertTrue(out.exists())


class TestCatMeowsPatternRegex(unittest.TestCase):
    """Direct regex tests — ensures the pattern compiles and matches."""

    def test_standard_match(self):
        m = CATMEOWS_PATTERN.match("F_BAC01_MC_MN_SIM01_101")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("context"), "F")
        self.assertEqual(m.group("cat_id"), "BAC01")

    def test_isolation_match(self):
        m = CATMEOWS_PATTERN.match("I_DAK01_MC_FN_SIM01_316")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("cat_id"), "DAK01")
        self.assertEqual(m.group("recording"), "316")

    def test_seq_variant_match(self):
        m = CATMEOWS_PATTERN.match("I_BLE01_EU_FN_DEL01_3SEQ2")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("recording"), "3SEQ2")

    def test_brushing_match(self):
        m = CATMEOWS_PATTERN.match("B_WHO01_MC_FI_SIM01_305")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("sex"), "FI")

    def test_no_match_on_garbage(self):
        self.assertIsNone(CATMEOWS_PATTERN.match("hello_world"))
        self.assertIsNone(CATMEOWS_PATTERN.match(""))

    def test_no_match_pure_numeric_catid(self):
        """Pure numeric cat_id (no alpha prefix) should not match."""
        self.assertIsNone(CATMEOWS_PATTERN.match("F_123_MC_FN_SIM01_101"))


if __name__ == "__main__":
    unittest.main()
