"""
Tests for Batch 3 — Audio Playback (flet-audio native playback)
===============================================================
Covers BioacousticPlayer refactoring to flet-audio (fta.Audio) for in-app
playback, replacing the old page.launch_url() + temp-file workaround.

Tests:
  BioacousticPlayer (bioacoustic_player.py):
    1.  TestBioacousticPlayerInit: fta.Audio params, overlay.append, page.update
    2.  TestPlayWavBytes: release→src→update→play order, src assignment, empty bytes
    3.  TestPlayFromBase64: decode→play, empty skips, invalid raises
    4.  TestDispose: removes from overlay, safe when not in overlay
    5.  TestBuildIndex: missing catalog → {}, valid catalog parse
    6.  TestResolveSound: indexed+exists, not-indexed, indexed+missing
    7.  TestPlaySoundId: resolve → DSP → play chain
    8.  TestProcessToWavBytes: valid bytes, extreme clip values

  App integration (app.py):
    9.  TestRESTAutoplay: audio_base64 triggers play_from_base64
    10. TestStreamingAutoplay: streaming result with audio_base64
    11. TestManualPlay: pitch/tempo sliders passed to play_sound_id
    12. TestOnDisconnect: dispose() called on page.on_disconnect
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# ── Mock chromadb before any app imports (project convention) ──────
sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("chromadb.utils", MagicMock())
sys.modules.setdefault("chromadb.utils.embedding_functions", MagicMock())

# ── Mock flet_audio before BioacousticPlayer import ────────────────
# The deferred `import flet_audio as fta` inside __init__ picks up
# our mock from sys.modules instead of the real package.
_fta_mock = MagicMock()
_fta_mock.ReleaseMode = MagicMock()
_fta_mock.ReleaseMode.STOP = "stop"
sys.modules["flet_audio"] = _fta_mock

# ── Test parameters ────────────────────────────────────────────────
from tests.shared_params import (  # noqa: E402
    AUDIO_B64_DECODED,
    DUMMY_PCM_BYTES,
    SAMPLE_ID_PRIMARY,
    STREAMING_SETTLE_SECS,
    WAVEFORM_MOCK_RETURN,
)
from tests.flet_mocks import BaseMockPage, _Ctrl, install_flet_mock  # noqa: E402

# File-internal constants
_FAKE_WAV_PATH       = Path("/fake/sample.wav")
_NONEXISTENT_CATALOG = "nonexistent/catalog.json"

install_flet_mock()

# ── Import BioacousticPlayer after mocks are in place ─────────────
from src.flet_mobile.bioacoustic_player import BioacousticPlayer  # noqa: E402


# ══════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════

def _make_page() -> MagicMock:
    """Minimal ft.Page mock: real overlay list + tracked update calls."""
    page = MagicMock()
    page.overlay = []
    page.update = MagicMock()
    return page


def _make_audio_mock() -> MagicMock:
    """Create a fresh fta.Audio instance mock with async play/release."""
    audio = MagicMock()
    audio.play = AsyncMock()
    audio.release = AsyncMock()
    audio.update = MagicMock()
    return audio


def _reset_fta_mock(audio_mock: MagicMock | None = None) -> MagicMock:
    """Reset _fta_mock.Audio call history and set a new return_value."""
    _fta_mock.Audio.reset_mock()
    if audio_mock is None:
        audio_mock = _make_audio_mock()
    _fta_mock.Audio.return_value = audio_mock
    return audio_mock


def _make_player(
    page: MagicMock | None = None,
    catalog_path: str = _NONEXISTENT_CATALOG,
) -> BioacousticPlayer:
    """Create a BioacousticPlayer with the given (non-existent by default) catalog."""
    if page is None:
        page = _make_page()
    return BioacousticPlayer(page=page, catalog_path=catalog_path)


# ══════════════════════════════════════════════════════════════════
#  1. TestBioacousticPlayerInit
# ══════════════════════════════════════════════════════════════════

class TestBioacousticPlayerInit(unittest.TestCase):
    """BioacousticPlayer.__init__ backend selection and fta.Audio registration."""

    def setUp(self) -> None:
        self._mock_audio = _reset_fta_mock()
        self._page = _make_page()

    @patch.dict(os.environ, {"MEOWSFORMER_AUDIO_BACKEND": "flet_audio"})
    def test_fta_audio_created_with_correct_params(self) -> None:
        """fta.Audio is called with src=None, autoplay=False, volume=1.0, STOP mode."""
        BioacousticPlayer(page=self._page, catalog_path=_NONEXISTENT_CATALOG)
        _fta_mock.Audio.assert_called_once_with(
            src=None,
            autoplay=False,
            volume=1.0,
            release_mode=_fta_mock.ReleaseMode.STOP,
        )

    @patch.dict(os.environ, {"MEOWSFORMER_AUDIO_BACKEND": "flet_audio"})
    def test_audio_instance_appended_to_page_overlay(self) -> None:
        """The Audio instance returned by fta.Audio() is added to page.overlay."""
        BioacousticPlayer(page=self._page, catalog_path=_NONEXISTENT_CATALOG)
        self.assertIn(self._mock_audio, self._page.overlay)

    @patch.dict(os.environ, {"MEOWSFORMER_AUDIO_BACKEND": "flet_audio"})
    def test_page_update_called_after_overlay_append(self) -> None:
        """page.update() is called to push the Audio overlay to the Flutter layer."""
        BioacousticPlayer(page=self._page, catalog_path=_NONEXISTENT_CATALOG)
        self._page.update.assert_called_once()

    @patch.dict(os.environ, {"MEOWSFORMER_AUDIO_BACKEND": "sounddevice"})
    def test_sounddevice_backend_skips_fta_audio(self) -> None:
        """When using sounddevice backend, fta.Audio is NOT created or added."""
        _fta_mock.Audio.reset_mock()
        player = BioacousticPlayer(page=self._page, catalog_path=_NONEXISTENT_CATALOG)
        _fta_mock.Audio.assert_not_called()
        self.assertIsNone(player._audio)
        self.assertEqual(len(self._page.overlay), 0)

    def test_default_backend_is_sounddevice(self) -> None:
        """Without MEOWSFORMER_AUDIO_BACKEND env var, defaults to sounddevice."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEOWSFORMER_AUDIO_BACKEND", None)
            _fta_mock.Audio.reset_mock()
            player = BioacousticPlayer(page=self._page, catalog_path=_NONEXISTENT_CATALOG)
            self.assertFalse(player._use_native)
            _fta_mock.Audio.assert_not_called()


# ══════════════════════════════════════════════════════════════════
#  2. TestPlayWavBytes
# ══════════════════════════════════════════════════════════════════

class TestPlayWavBytesNative(unittest.IsolatedAsyncioTestCase):
    """_play_wav_bytes with flet_audio backend: release → set src → update → play."""

    def setUp(self) -> None:
        self._mock_audio = _reset_fta_mock()
        self._env_patch = patch.dict(os.environ, {"MEOWSFORMER_AUDIO_BACKEND": "flet_audio"})
        self._env_patch.start()
        self._player = _make_player()

    def tearDown(self) -> None:
        self._env_patch.stop()

    async def test_correct_call_order_release_update_play(self) -> None:
        """release() fires first, then update(), then play() — never out of order."""
        call_order: list[str] = []
        self._mock_audio.release = AsyncMock(
            side_effect=lambda: call_order.append("release")
        )
        self._mock_audio.update = MagicMock(
            side_effect=lambda: call_order.append("update")
        )
        self._mock_audio.play = AsyncMock(
            side_effect=lambda: call_order.append("play")
        )

        await self._player._play_wav_bytes(b"wav bytes")

        self.assertEqual(call_order, ["release", "update", "play"])

    async def test_src_set_to_passed_bytes(self) -> None:
        """self._audio.src is assigned the exact bytes passed to _play_wav_bytes."""
        wav_data = b"RIFF\x00\x00\x00\x00WAVEfmt test"
        await self._player._play_wav_bytes(wav_data)
        self.assertEqual(self._player._audio.src, wav_data)

    async def test_release_and_play_both_awaited(self) -> None:
        """Both release() and play() are awaited exactly once per call."""
        await self._player._play_wav_bytes(b"wav bytes")
        self._mock_audio.release.assert_awaited_once()
        self._mock_audio.play.assert_awaited_once()

    async def test_empty_bytes_executes_full_pipeline(self) -> None:
        """Empty bytes bypass no guard in _play_wav_bytes; all steps still run."""
        await self._player._play_wav_bytes(b"")
        self._mock_audio.release.assert_awaited_once()
        self._mock_audio.update.assert_called_once()
        self._mock_audio.play.assert_awaited_once()


class TestPlayWavBytesSounddevice(unittest.IsolatedAsyncioTestCase):
    """_play_wav_bytes with sounddevice backend: sf.read → sd.play → sd.wait."""

    def setUp(self) -> None:
        _reset_fta_mock()
        self._env_patch = patch.dict(os.environ, {"MEOWSFORMER_AUDIO_BACKEND": "sounddevice"})
        self._env_patch.start()
        self._player = _make_player()

    def tearDown(self) -> None:
        self._env_patch.stop()

    @patch("src.flet_mobile.bioacoustic_player.sf.read")
    @patch("sounddevice.wait")
    @patch("sounddevice.play")
    async def test_sounddevice_play_and_wait_called(
        self,
        mock_sd_play: MagicMock,
        mock_sd_wait: MagicMock,
        mock_sf_read: MagicMock,
    ) -> None:
        import numpy as np

        fake_data = np.zeros(16000, dtype=np.float32)
        mock_sf_read.return_value = (fake_data, 16000)

        await self._player._play_wav_bytes(b"RIFF fake wav")

        mock_sf_read.assert_called_once()
        mock_sd_play.assert_called_once_with(fake_data, 16000)
        mock_sd_wait.assert_called_once()


# ══════════════════════════════════════════════════════════════════
#  3. TestPlayFromBase64
# ══════════════════════════════════════════════════════════════════

class TestPlayFromBase64(unittest.IsolatedAsyncioTestCase):
    """play_from_base64: decode valid b64, skip empty, propagate invalid."""

    def setUp(self) -> None:
        _reset_fta_mock()
        self._player = _make_player()

    async def test_valid_base64_decoded_and_played(self) -> None:
        """Valid base64 string is decoded to bytes and forwarded to _play_wav_bytes."""
        raw = base64.b64decode(AUDIO_B64_DECODED)
        b64 = AUDIO_B64_DECODED

        with patch.object(self._player, "_play_wav_bytes", new_callable=AsyncMock) as mock_play:
            await self._player.play_from_base64(b64)

        mock_play.assert_called_once_with(raw)

    async def test_empty_string_skips_play(self) -> None:
        """Empty string returns immediately; _play_wav_bytes is never called."""
        with patch.object(self._player, "_play_wav_bytes", new_callable=AsyncMock) as mock_play:
            await self._player.play_from_base64("")

        mock_play.assert_not_called()

    async def test_invalid_base64_raises_binascii_error(self) -> None:
        """Malformed base64 propagates binascii.Error to the caller."""
        import binascii

        with self.assertRaises(binascii.Error):
            await self._player.play_from_base64("!!!not-base64!!!")


# ══════════════════════════════════════════════════════════════════
#  4. TestDispose
# ══════════════════════════════════════════════════════════════════

class TestDispose(unittest.TestCase):
    """dispose() removes the Audio control from page.overlay and updates the page."""

    def setUp(self) -> None:
        _reset_fta_mock()

    @patch.dict(os.environ, {"MEOWSFORMER_AUDIO_BACKEND": "flet_audio"})
    def test_removes_audio_from_overlay_and_calls_update(self) -> None:
        """When _audio is in overlay, dispose() removes it and calls page.update()."""
        page = _make_page()
        player = _make_player(page=page)

        self.assertIn(player._audio, page.overlay, "setUp: audio should be in overlay")

        page.update.reset_mock()
        player.dispose()

        self.assertNotIn(player._audio, page.overlay)
        page.update.assert_called_once()

    @patch.dict(os.environ, {"MEOWSFORMER_AUDIO_BACKEND": "flet_audio"})
    def test_safe_when_audio_not_in_overlay(self) -> None:
        """dispose() is a no-op (no exception) if _audio is already absent."""
        page = _make_page()
        player = _make_player(page=page)

        page.overlay.remove(player._audio)
        self.assertNotIn(player._audio, page.overlay)

        try:
            player.dispose()
        except Exception as exc:  # pragma: no cover
            self.fail(f"dispose() raised unexpectedly: {exc}")

    def test_dispose_safe_when_sounddevice_backend(self) -> None:
        """dispose() is a no-op when _audio is None (sounddevice backend)."""
        with patch.dict(os.environ, {"MEOWSFORMER_AUDIO_BACKEND": "sounddevice"}):
            page = _make_page()
            player = _make_player(page=page)
            self.assertIsNone(player._audio)
            try:
                player.dispose()
            except Exception as exc:  # pragma: no cover
                self.fail(f"dispose() raised unexpectedly: {exc}")


# ══════════════════════════════════════════════════════════════════
#  5. TestBuildIndex
# ══════════════════════════════════════════════════════════════════

class TestBuildIndex(unittest.TestCase):
    """_build_index: {} for missing catalog, correct mapping for valid catalog."""

    def setUp(self) -> None:
        _reset_fta_mock()

    def test_missing_catalog_returns_empty_dict(self) -> None:
        player = _make_player(catalog_path="no/such/catalog.json")
        self.assertEqual(player._sample_index, {})

    def test_parses_valid_catalog_into_id_path_mapping(self) -> None:
        catalog = {
            "samples": [
                {"id": "cat_001", "file_path": "assets/raw_data/001.wav"},
                {"id": "cat_002", "file_path": "assets/raw_data/002.wav"},
                {"id": None, "file_path": "skip_null_id"},
                {"id": "cat_003"},  # missing file_path — skipped
            ],
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(catalog, tmp)
        tmp.close()

        try:
            player = _make_player()
            player.catalog_path = Path(tmp.name)
            index = player._build_index()

            self.assertEqual(len(index), 2)
            self.assertIn("cat_001", index)
            self.assertIn("cat_002", index)
            self.assertNotIn("cat_003", index)
            self.assertEqual(
                index["cat_001"],
                player.repo_root / "assets/raw_data/001.wav",
            )
        finally:
            os.unlink(tmp.name)


# ══════════════════════════════════════════════════════════════════
#  6. TestResolveSound
# ══════════════════════════════════════════════════════════════════

class TestResolveSound(unittest.TestCase):
    """_resolve_sound returns the indexed path when it exists, else fallback."""

    def setUp(self) -> None:
        _reset_fta_mock()

    def test_indexed_and_file_exists_returns_path(self) -> None:
        player = _make_player()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            player._sample_index = {SAMPLE_ID_PRIMARY: Path(tmp.name)}
            self.assertEqual(player._resolve_sound(SAMPLE_ID_PRIMARY), Path(tmp.name))
        finally:
            os.unlink(tmp.name)

    def test_not_indexed_returns_fallback(self) -> None:
        player = _make_player()
        player._sample_index = {}
        self.assertEqual(player._resolve_sound("missing_id"), player._fallback_file)

    def test_indexed_but_file_missing_returns_fallback(self) -> None:
        player = _make_player()
        player._sample_index = {SAMPLE_ID_PRIMARY: Path("/nonexistent/path.wav")}
        self.assertEqual(player._resolve_sound(SAMPLE_ID_PRIMARY), player._fallback_file)


# ══════════════════════════════════════════════════════════════════
#  7. TestPlaySoundId
# ══════════════════════════════════════════════════════════════════

class TestPlaySoundId(unittest.IsolatedAsyncioTestCase):
    """play_sound_id: resolve → asyncio.to_thread(DSP) → _play_wav_bytes."""

    def setUp(self) -> None:
        _reset_fta_mock()

    async def test_resolve_process_play_chain(self) -> None:
        """Full chain is wired: _resolve_sound → _process_to_wav_bytes → _play_wav_bytes."""
        player = _make_player()
        fake_source = _FAKE_WAV_PATH
        fake_wav = b"processed wav bytes"

        with patch.object(player, "_resolve_sound", return_value=fake_source) as mock_resolve, \
             patch.object(player, "_process_to_wav_bytes", return_value=fake_wav) as mock_dsp, \
             patch.object(player, "_play_wav_bytes", new_callable=AsyncMock) as mock_play:
            result = await player.play_sound_id(SAMPLE_ID_PRIMARY, pitch_factor=1.2, tempo_factor=0.9)

        self.assertIsNone(result)
        mock_resolve.assert_called_once_with(SAMPLE_ID_PRIMARY)
        mock_dsp.assert_called_once_with(fake_source, 1.2, 0.9)
        mock_play.assert_awaited_once_with(fake_wav)


# ══════════════════════════════════════════════════════════════════
#  8. TestProcessToWavBytes
# ══════════════════════════════════════════════════════════════════

class TestProcessToWavBytes(unittest.TestCase):
    """_process_to_wav_bytes: load → stretch → shift → WAV bytes; clamps extremes."""

    @patch("src.flet_mobile.bioacoustic_player.sf.write")
    @patch("src.flet_mobile.bioacoustic_player.librosa.effects.pitch_shift")
    @patch("src.flet_mobile.bioacoustic_player.librosa.effects.time_stretch")
    @patch("src.flet_mobile.bioacoustic_player.librosa.load")
    def test_returns_valid_wav_bytes(
        self,
        mock_load: MagicMock,
        mock_stretch: MagicMock,
        mock_shift: MagicMock,
        mock_sf_write: MagicMock,
    ) -> None:
        import numpy as np

        audio = np.zeros(16000, dtype=np.float32)
        mock_load.return_value = (audio, 16000)
        mock_stretch.return_value = audio
        mock_shift.return_value = audio
        mock_sf_write.side_effect = lambda buf, *_a, **_kw: buf.write(b"FAKEWAV")

        result = BioacousticPlayer._process_to_wav_bytes(
            _FAKE_WAV_PATH, 1.0, 1.0,
        )

        self.assertIsInstance(result, bytes)
        self.assertEqual(result, b"FAKEWAV")
        mock_load.assert_called_once_with(_FAKE_WAV_PATH, sr=None, mono=True)
        mock_stretch.assert_called_once()
        mock_shift.assert_called_once()

    @patch("src.flet_mobile.bioacoustic_player.sf.write")
    @patch("src.flet_mobile.bioacoustic_player.librosa.effects.pitch_shift")
    @patch("src.flet_mobile.bioacoustic_player.librosa.effects.time_stretch")
    @patch("src.flet_mobile.bioacoustic_player.librosa.load")
    def test_extreme_values_are_clipped(
        self,
        mock_load: MagicMock,
        mock_stretch: MagicMock,
        mock_shift: MagicMock,
        mock_sf_write: MagicMock,
    ) -> None:
        import numpy as np

        audio = np.zeros(16000, dtype=np.float32)
        mock_load.return_value = (audio, 16000)
        mock_stretch.return_value = audio
        mock_shift.return_value = audio
        mock_sf_write.side_effect = lambda buf, *_a, **_kw: buf.write(b"WAV")

        BioacousticPlayer._process_to_wav_bytes(
            _FAKE_WAV_PATH, pitch_factor=99.0, tempo_factor=0.01,
        )

        actual_rate = mock_stretch.call_args.kwargs["rate"]
        self.assertAlmostEqual(actual_rate, 0.6)

        actual_n_steps = mock_shift.call_args.kwargs["n_steps"]
        expected_semitones = (1.5 - 1.0) * 12.0
        self.assertAlmostEqual(actual_n_steps, expected_semitones)


# ══════════════════════════════════════════════════════════════════
# App integration tests — Base class
# ══════════════════════════════════════════════════════════════════

# _MockPage replaced by BaseMockPage from tests.flet_mocks


class _Batch3AppTestBase(unittest.IsolatedAsyncioTestCase):
    """
    Shared setUp: run meowsformer_ui with mocked dependencies and
    extract key controls / callbacks for assertions.
    """

    async def asyncSetUp(self) -> None:
        self._patchers: list = []

        def _start(target: str):
            p = patch(target)
            self._patchers.append(p)
            return p.start()

        MockTC = _start("src.flet_mobile.app.TranslationClient")
        MockAR = _start("src.flet_mobile.app.AudioRecorder")
        MockBP = _start("src.flet_mobile.app.BioacousticPlayer")

        self.mock_player = MockBP.return_value
        self.mock_player.play_from_base64 = AsyncMock()
        self.mock_player.play_sound_id = AsyncMock()
        self.mock_player.dispose = MagicMock()

        self.mock_client = MockTC.return_value
        self.mock_client.stream_translate = AsyncMock()
        self.mock_client.translate_file = AsyncMock(return_value={})

        self.mock_recorder = MockAR.return_value
        self.mock_recorder.is_recording = False
        self.mock_recorder.on_chunk = None

        def _start_rec() -> None:
            self.mock_recorder.is_recording = True

        self.mock_recorder.start = MagicMock(side_effect=_start_rec)
        self.mock_recorder.stop = MagicMock(return_value=b"")
        self.mock_recorder.snapshot_waveform = MagicMock(return_value=WAVEFORM_MOCK_RETURN)

        self.page = BaseMockPage()

        from src.flet_mobile.app import meowsformer_ui

        await meowsformer_ui(self.page)

        # ── Navigate control tree ─────────────────────────────────
        main_col = self.page._added[0]
        self.assertEqual(
            len(main_col.controls), 5,
            "Expected 5 top-level cards (bridge, record, lab, output, library); "
            "layout changed — update indices",
        )

        bridge_card = main_col.controls[0]
        bridge_col = bridge_card.content
        header_row = bridge_col.controls[0]
        self.assertEqual(
            header_row.controls[0].value, "The Bridge",
            "bridge_card header mismatch — layout changed",
        )
        # bridge_col: [0]=header_row, [1]=cat_avatar, [2]=mode_selector, ...
        self.mode_selector = bridge_col.controls[2]

        record_container = main_col.controls[1]
        self.record_button = record_container.content
        self.on_record_toggle = self.record_button.on_click

        output_card = main_col.controls[3]
        output_col = output_card.content
        self.assertEqual(
            output_col.controls[0].value, "The Output",
            "output_card header mismatch — layout changed",
        )
        # output_col: [0]=header, [1]=subtitle, [2]=tempo_slider, [3]=pitch_slider, [4]=play_btn
        self.tempo_slider = output_col.controls[2]
        self.pitch_slider = output_col.controls[3]
        play_btn = output_col.controls[4]
        self.on_play_processed = play_btn.on_click

        self.on_disconnect = self.page.on_disconnect

    async def asyncTearDown(self) -> None:
        for p in self._patchers:
            p.stop()


# ══════════════════════════════════════════════════════════════════
#  9. TestRESTAutoplay
# ══════════════════════════════════════════════════════════════════

class TestRESTAutoplay(_Batch3AppTestBase):
    """REST response with audio_base64 must trigger play_from_base64."""

    async def test_plays_audio_base64_from_response(self) -> None:
        self.mock_recorder.stop = MagicMock(return_value=DUMMY_PCM_BYTES)
        self.mock_client.translate_file = AsyncMock(return_value={
            "human_interpretation": "你好",
            "audio_base64": AUDIO_B64_DECODED,
            "sound_id": SAMPLE_ID_PRIMARY,
        })

        await self.on_record_toggle(MagicMock())  # start
        await self.on_record_toggle(MagicMock())  # stop

        self.mock_player.play_from_base64.assert_called_once_with(AUDIO_B64_DECODED)

    async def test_no_play_when_audio_base64_absent(self) -> None:
        self.mock_recorder.stop = MagicMock(return_value=DUMMY_PCM_BYTES)
        self.mock_client.translate_file = AsyncMock(return_value={
            "human_interpretation": "你好",
            "sound_id": SAMPLE_ID_PRIMARY,
        })

        await self.on_record_toggle(MagicMock())  # start
        await self.on_record_toggle(MagicMock())  # stop

        self.mock_player.play_from_base64.assert_not_called()


# ══════════════════════════════════════════════════════════════════
#  10. TestStreamingAutoplay
# ══════════════════════════════════════════════════════════════════

class TestStreamingAutoplay(_Batch3AppTestBase):
    """Streaming result with audio_base64 must trigger play_from_base64."""

    async def test_plays_audio_base64_from_streaming_result(self) -> None:
        mode_evt = MagicMock()
        mode_evt.control.selected = {"streaming"}
        self.mode_selector.on_change(mode_evt)

        self.mock_recorder.is_recording = False
        await self.on_record_toggle(MagicMock())
        # stream_translate is an AsyncMock with no real I/O; a single event-loop
        # yield (0.05 s) is sufficient for the coroutine to be scheduled and the
        # call_args to be populated before we inspect them.
        await asyncio.sleep(STREAMING_SETTLE_SECS)

        self.assertTrue(
            self.mock_client.stream_translate.called,
            "stream_translate should have been invoked",
        )
        on_ws_event = self.mock_client.stream_translate.call_args.kwargs["on_event"]

        await on_ws_event({
            "type": "result",
            "transcription": "你好小猫",
            "selected_category": {
                "tags": {"emotion": ["happy"]},
                "sample_id": "cat_042",
                "match_score": 0.9,
            },
            "audio_base64": AUDIO_B64_DECODED,
            "reasoning": "test",
        })

        self.mock_player.play_from_base64.assert_called_once_with(AUDIO_B64_DECODED)


# ══════════════════════════════════════════════════════════════════
#  11. TestManualPlay
# ══════════════════════════════════════════════════════════════════

class TestManualPlay(_Batch3AppTestBase):
    """on_play_processed must call play_sound_id with current slider values."""

    async def test_play_with_pitch_tempo(self) -> None:
        self.tempo_slider.value = 1.2
        self.pitch_slider.value = 0.9

        await self.on_play_processed(MagicMock())

        actual_call = self.mock_player.play_sound_id.call_args
        self.mock_player.play_sound_id.assert_called_once()
        self.assertEqual(actual_call.kwargs["pitch_factor"], 0.9)
        self.assertEqual(actual_call.kwargs["tempo_factor"], 1.2)


# ══════════════════════════════════════════════════════════════════
#  12. TestOnDisconnect
# ══════════════════════════════════════════════════════════════════

class TestOnDisconnect(_Batch3AppTestBase):
    """page.on_disconnect handler must call player.dispose()."""

    async def test_disconnect_calls_dispose(self) -> None:
        self.assertIsNotNone(self.on_disconnect)
        self.on_disconnect(MagicMock())
        self.mock_player.dispose.assert_called_once()


if __name__ == "__main__":
    unittest.main()
