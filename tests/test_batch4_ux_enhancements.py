"""
Tests for Batch 4 — UX 完善 (UX Enhancements)
==============================================
Covers 5 UX features for the Flet mobile client:
  1. WebSocket connection status indicator (ws_status_chip, _update_ws_status)
  2. Network error auto-fallback to REST (WebSocketConnectionError, _fallback_to_rest)
  3. Snackbar error notifications (_show_snackbar)
  4. Recording duration timer (recording_timer_loop, recording_timer_text)
  5. Enhanced history with ExpansionTile (append_history with 5-dim tags)

Tests:
  TranslationClient (translation_client.py):
    1.  WebSocketConnectionError inherits from Exception
    2.  on_state_change fires connecting → connected → disconnected on success
    3.  Connection timeout (>5s) raises WebSocketConnectionError
    4.  OSError / connection refused raises WebSocketConnectionError
    5.  on_state_change fires connecting → disconnected on failure
    6.  on_state_change=None (default) works without error

  App UI (app.py):
    7.  _update_ws_status: connecting / connected / disconnected / unknown states
    8.  _show_snackbar: error (red) vs non-error (amber), duration 3000ms
    9.  recording_timer_loop: initial "00:00" format, MM:SS pattern, visibility
    10. _fallback_to_rest: mode switch to REST, amber snackbar, REST replay
    11. append_history: ExpansionTile with 5-dim tags, transcription, timestamp, score
    12. Bridge card structure: ws_status_chip and recording_timer_text
    13. History legacy fallback without streaming tags
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import types
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# ── Mock chromadb before any app imports (project convention) ──────
sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("chromadb.utils", MagicMock())
sys.modules.setdefault("chromadb.utils.embedding_functions", MagicMock())

# ── Test parameters ────────────────────────────────────────────────
from tests.shared_params import (  # noqa: E402
    AUDIO_B64_STUB,
    BREED_DEFAULT,
    DUMMY_PCM_BYTES,
    MATCH_SCORE_HIGH,
    MINIMAL_RESULT_MSG,
    SAMPLE_ID_PRIMARY,
    SAMPLE_ID_SECONDARY,
    SOUND_ID_LEGACY,
    STREAMING_SETTLE_SECS,
    WAVEFORM_MOCK_RETURN,
)
from tests.flet_mocks import (  # noqa: E402
    BaseMockPage,
    _Ctrl,
    _ListCtrl,
    _TextCtrl,
    install_flet_mock,
)
from tests.ws_stubs import MockWebSocket, async_chunks, ws_connect_coro  # noqa: E402

install_flet_mock()

import flet as ft  # W1: module-level import after mock is installed

# ── Import modules under test ─────────────────────────────────────
from src.flet_mobile.translation_client import (
    TranslationClient,
    WebSocketConnectionError,
)
from src.flet_mobile.theme import AMBER, FOREST_GREEN, PAW_PINK


# ══════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════

class MockWebSocket:
    """In-memory WS stub: records sent frames and yields server messages."""

    def __init__(self, server_messages: list | None = None) -> None:
        self.sent: list = []
        self._server_messages: list = server_messages or []

    async def send(self, data: Any) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        pass

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for msg in self._server_messages:
            yield msg


def _ws_connect_coro(mock_ws: MockWebSocket):
    """Return a callable that mimics ``websockets.connect()`` as a coroutine."""

    async def _connect(*_args: Any, **_kwargs: Any):
        return mock_ws

    return _connect


async def _async_chunks(*chunks: bytes):
    """Yield *chunks* as an async iterable."""
    for c in chunks:
        yield c


def _patch_ws(client: TranslationClient, mock_ws: MockWebSocket):
    """Return combined patches for websockets.connect + _build_ws_url."""
    return (
        patch.object(client, "_build_ws_url", return_value="ws://mock"),
        patch(
            "src.flet_mobile.translation_client.websockets.connect",
            ws_connect_coro(mock_ws),
        ),
    )


# ══════════════════════════════════════════════════════════════════
# PART 1: translation_client.py tests
# ══════════════════════════════════════════════════════════════════


# ── 1. WebSocketConnectionError ───────────────────────────────────

class TestWebSocketConnectionError(unittest.TestCase):
    """WebSocketConnectionError must be a standard Exception subclass."""

    def test_inherits_from_exception(self) -> None:
        self.assertTrue(issubclass(WebSocketConnectionError, Exception))

    def test_instantiation_preserves_message(self) -> None:
        exc = WebSocketConnectionError("connection failed")
        self.assertEqual(str(exc), "connection failed")


# ── 2. on_state_change callback order on success ──────────────────

class TestOnStateChangeSuccess(unittest.IsolatedAsyncioTestCase):
    """Successful stream_translate fires connecting → connected → disconnected."""

    async def test_fires_connecting_connected_disconnected(self) -> None:
        mock_ws = MockWebSocket(server_messages=[MINIMAL_RESULT_MSG])
        states: list[str] = []

        client = TranslationClient()
        p1, p2 = _patch_ws(client, mock_ws)
        with p1, p2:
            await client.stream_translate(
                chunks=async_chunks(),
                on_event=AsyncMock(),
                on_state_change=lambda s: states.append(s),
            )

        self.assertEqual(states, ["connecting", "connected", "disconnected"])


# ── 3. Connection timeout ─────────────────────────────────────────

class TestConnectionTimeout(unittest.IsolatedAsyncioTestCase):
    """Connection hanging beyond WS_CONNECT_TIMEOUT raises WebSocketConnectionError."""

    async def test_timeout_raises_ws_connection_error(self) -> None:
        async def _hang(*_a: Any, **_kw: Any):
            await asyncio.sleep(999)

        client = TranslationClient()
        with patch.object(client, "_build_ws_url", return_value="ws://mock"), \
             patch("src.flet_mobile.translation_client.websockets.connect", _hang), \
             patch("src.flet_mobile.translation_client.WS_CONNECT_TIMEOUT", 0.01):
            with self.assertRaises(WebSocketConnectionError) as ctx:
                await client.stream_translate(
                    chunks=async_chunks(),
                    on_event=AsyncMock(),
                )

        self.assertIn("TimeoutError", str(ctx.exception))


# ── 4. OSError / connection refused ───────────────────────────────

class TestConnectionOSError(unittest.IsolatedAsyncioTestCase):
    """Network-level failures (OSError) raise WebSocketConnectionError."""

    async def test_os_error_raises_ws_connection_error(self) -> None:
        async def _refuse(*_a: Any, **_kw: Any):
            raise OSError("Connection refused")

        client = TranslationClient()
        with patch.object(client, "_build_ws_url", return_value="ws://mock"), \
             patch("src.flet_mobile.translation_client.websockets.connect", _refuse):
            with self.assertRaises(WebSocketConnectionError) as ctx:
                await client.stream_translate(
                    chunks=async_chunks(),
                    on_event=AsyncMock(),
                )

        self.assertIn("OSError", str(ctx.exception))


# ── 5. on_state_change fires connecting → disconnected on failure ─

class TestOnStateChangeFailure(unittest.IsolatedAsyncioTestCase):
    """Connection failure fires connecting then disconnected (skipping connected)."""

    async def test_timeout_fires_connecting_disconnected(self) -> None:
        async def _hang(*_a: Any, **_kw: Any):
            await asyncio.sleep(999)

        states: list[str] = []
        client = TranslationClient()
        with patch.object(client, "_build_ws_url", return_value="ws://mock"), \
             patch("src.flet_mobile.translation_client.websockets.connect", _hang), \
             patch("src.flet_mobile.translation_client.WS_CONNECT_TIMEOUT", 0.01):
            with self.assertRaises(WebSocketConnectionError):
                await client.stream_translate(
                    chunks=async_chunks(),
                    on_event=AsyncMock(),
                    on_state_change=lambda s: states.append(s),
                )

        self.assertEqual(states, ["connecting", "disconnected"])

    async def test_os_error_fires_connecting_disconnected(self) -> None:
        async def _refuse(*_a: Any, **_kw: Any):
            raise OSError("Connection refused")

        states: list[str] = []
        client = TranslationClient()
        with patch.object(client, "_build_ws_url", return_value="ws://mock"), \
             patch("src.flet_mobile.translation_client.websockets.connect", _refuse):
            with self.assertRaises(WebSocketConnectionError):
                await client.stream_translate(
                    chunks=async_chunks(),
                    on_event=AsyncMock(),
                    on_state_change=lambda s: states.append(s),
                )

        self.assertEqual(states, ["connecting", "disconnected"])


# ── 6. on_state_change=None (default) ─────────────────────────────

class TestOnStateChangeNone(unittest.IsolatedAsyncioTestCase):
    """Passing None or omitting on_state_change must not raise."""

    async def test_explicit_none_no_error(self) -> None:
        mock_ws = MockWebSocket(server_messages=[MINIMAL_RESULT_MSG])
        client = TranslationClient()
        p1, p2 = _patch_ws(client, mock_ws)
        with p1, p2:
            await client.stream_translate(
                chunks=async_chunks(),
                on_event=AsyncMock(),
                on_state_change=None,
            )

    async def test_default_omitted_no_error(self) -> None:
        mock_ws = MockWebSocket(server_messages=[MINIMAL_RESULT_MSG])
        client = TranslationClient()
        p1, p2 = _patch_ws(client, mock_ws)
        with p1, p2:
            await client.stream_translate(
                chunks=async_chunks(),
                on_event=AsyncMock(),
            )


# ══════════════════════════════════════════════════════════════════
# PART 2: app.py tests — Base class
# ══════════════════════════════════════════════════════════════════

class _Batch4AppTestBase(unittest.IsolatedAsyncioTestCase):
    """
    Shared setUp: run meowsformer_ui with mocked dependencies and
    extract key controls from the control tree.  Does NOT start
    recording — subclasses opt-in via _start_streaming_recording().
    """

    async def asyncSetUp(self) -> None:
        self._patchers: list = []

        MockTC = self._start_patch("src.flet_mobile.app.TranslationClient")
        MockAR = self._start_patch("src.flet_mobile.app.AudioRecorder")
        MockBP = self._start_patch("src.flet_mobile.app.BioacousticPlayer")

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

        def _start_rec():
            self.mock_recorder.is_recording = True

        self.mock_recorder.start = MagicMock(side_effect=_start_rec)
        self.mock_recorder.stop = MagicMock(return_value=b"")
        self.mock_recorder.snapshot_waveform = MagicMock(
            return_value=WAVEFORM_MOCK_RETURN,
        )

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

        self.ws_status_chip = header_row.controls[1]
        self.mode_selector = bridge_col.controls[2]

        waveform_row = bridge_col.controls[4]
        self.recording_timer_text = waveform_row.controls[1]
        self.live_transcription = bridge_col.controls[5]

        record_container = main_col.controls[1]
        self.record_button = record_container.content
        self.on_record_toggle = self.record_button.on_click

        lab_card = main_col.controls[2]
        lab_col = lab_card.content
        self.assertEqual(
            lab_col.controls[0].value, "The Lab",
            "lab_card header mismatch — layout changed",
        )
        self.analysis_status = lab_col.controls[1]
        self.speculative_bar = lab_col.controls[2]
        self.tags_wrap = lab_col.controls[3]

        library_card = main_col.controls[4]
        lib_col = library_card.content
        self.assertEqual(
            lib_col.controls[0].value, "The Library",
            "library_card header mismatch — layout changed",
        )
        self.history_view = lib_col.controls[2]

    def _start_patch(self, target: str):
        p = patch(target)
        self._patchers.append(p)
        return p.start()

    async def _start_streaming_recording(self) -> None:
        """Switch to streaming mode and start recording."""
        mode_evt = MagicMock()
        mode_evt.control.selected = {"streaming"}
        self.mode_selector.on_change(mode_evt)

        self.mock_recorder.is_recording = False
        await self.on_record_toggle(MagicMock())
        # Mocked stream_translate completes (or raises) instantly;
        # 50ms lets the created asyncio.Task resolve before we proceed.
        await asyncio.sleep(STREAMING_SETTLE_SECS)

    async def asyncTearDown(self) -> None:
        for p in self._patchers:
            p.stop()


# ══════════════════════════════════════════════════════════════════
#  7. _update_ws_status
# ══════════════════════════════════════════════════════════════════

class TestUpdateWsStatus(_Batch4AppTestBase):
    """_update_ws_status maps state strings to label/bgcolor on ws_status_chip."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        await self._start_streaming_recording()
        self.on_state_change = (
            self.mock_client.stream_translate.call_args.kwargs["on_state_change"]
        )

    async def test_connecting_sets_label_amber(self) -> None:
        self.on_state_change("connecting")
        self.assertEqual(self.ws_status_chip.label.value, "连接中...")
        self.assertEqual(self.ws_status_chip.bgcolor, AMBER)

    async def test_connected_sets_label_green(self) -> None:
        self.on_state_change("connected")
        self.assertEqual(self.ws_status_chip.label.value, "已连接")
        self.assertEqual(self.ws_status_chip.bgcolor, FOREST_GREEN)

    async def test_disconnected_sets_label_pink(self) -> None:
        self.on_state_change("disconnected")
        self.assertEqual(self.ws_status_chip.label.value, "已断开")
        self.assertEqual(self.ws_status_chip.bgcolor, PAW_PINK)

    async def test_unknown_falls_back_to_disconnected(self) -> None:
        self.on_state_change("totally_unknown_state")
        self.assertEqual(self.ws_status_chip.label.value, "已断开")
        self.assertEqual(self.ws_status_chip.bgcolor, PAW_PINK)


# ══════════════════════════════════════════════════════════════════
#  8. _show_snackbar
# ══════════════════════════════════════════════════════════════════

class TestShowSnackbar(_Batch4AppTestBase):
    """_show_snackbar opens SnackBar with correct color and duration."""

    async def test_error_snackbar_red_bgcolor(self) -> None:
        await self._start_streaming_recording()
        on_ws_event = (
            self.mock_client.stream_translate.call_args.kwargs["on_event"]
        )

        await on_ws_event({"type": "error", "detail": "test error"})

        self.assertTrue(len(self.page.overlay) > 0)
        snackbar = self.page.overlay[-1]
        self.assertEqual(snackbar.bgcolor, ft.Colors.RED_700)

    async def test_non_error_snackbar_amber_bgcolor(self) -> None:
        await self.on_record_toggle(MagicMock())  # start (REST)
        await self.on_record_toggle(MagicMock())  # stop with empty PCM

        self.assertTrue(len(self.page.overlay) > 0)
        snackbar = self.page.overlay[-1]
        self.assertEqual(snackbar.bgcolor, AMBER)

    async def test_snackbar_duration_3000ms(self) -> None:
        await self._start_streaming_recording()
        on_ws_event = (
            self.mock_client.stream_translate.call_args.kwargs["on_event"]
        )

        await on_ws_event({"type": "error", "detail": "timeout"})

        snackbar = self.page.overlay[-1]
        self.assertEqual(snackbar.duration, 3000)

    async def test_snackbar_content_includes_message(self) -> None:
        await self._start_streaming_recording()
        on_ws_event = (
            self.mock_client.stream_translate.call_args.kwargs["on_event"]
        )

        await on_ws_event({"type": "error", "detail": "Whisper API timeout"})

        snackbar = self.page.overlay[-1]
        self.assertIn("Whisper API timeout", snackbar.content.value)


# ══════════════════════════════════════════════════════════════════
#  9. Recording timer
# ══════════════════════════════════════════════════════════════════

class TestRecordingTimer(_Batch4AppTestBase):
    """recording_timer_loop displays MM:SS, anchored to wall-clock time."""

    async def _get_timer_fn(self):
        """Start recording, extract recording_timer_loop, and guard with __name__."""
        await self.on_record_toggle(MagicMock())
        self.assertGreaterEqual(len(self.page._run_task_calls), 3)
        timer_fn = self.page._run_task_calls[2]
        self.assertEqual(
            timer_fn.__name__, "recording_timer_loop",
            "run_task call order changed — timer_fn index needs updating",
        )
        return timer_fn

    async def test_initial_format_zero(self) -> None:
        """Timer loop sets visible=True and value='00:00' before entering while."""
        timer_fn = await self._get_timer_fn()

        self.mock_recorder.is_recording = False
        await timer_fn()

        self.assertTrue(self.recording_timer_text.visible)
        self.assertEqual(self.recording_timer_text.value, "00:00")

    async def test_mm_ss_format_pattern(self) -> None:
        """Timer value always matches the MM:SS regex pattern."""
        timer_fn = await self._get_timer_fn()

        self.mock_recorder.is_recording = False
        await timer_fn()

        self.assertRegex(self.recording_timer_text.value, r"^\d{2}:\d{2}$")

    async def test_timer_hidden_after_stop(self) -> None:
        """on_record_toggle stop sets recording_timer_text.visible = False."""
        await self.on_record_toggle(MagicMock())  # start
        await self.on_record_toggle(MagicMock())  # stop

        self.assertFalse(self.recording_timer_text.visible)

    async def test_timer_anchored_to_wall_clock_time(self) -> None:
        """Timer computes elapsed from asyncio.get_event_loop().time(), not
        a counter.  Verified by injecting controlled time values and asserting
        the resulting MM:SS string matches the expected difference."""
        timer_fn = await self._get_timer_fn()

        mock_loop = MagicMock()
        mock_loop.time = MagicMock(side_effect=[100.0, 165.0])

        async def _stop_after_one(_duration: float) -> None:
            self.mock_recorder.is_recording = False

        with patch("asyncio.get_event_loop", return_value=mock_loop), \
             patch("asyncio.sleep", side_effect=_stop_after_one):
            await timer_fn()

        self.assertEqual(mock_loop.time.call_count, 2)
        self.assertEqual(self.recording_timer_text.value, "01:05")


# ══════════════════════════════════════════════════════════════════
#  10. _fallback_to_rest
# ══════════════════════════════════════════════════════════════════

class TestFallbackToRest(_Batch4AppTestBase):
    """WebSocketConnectionError triggers automatic fallback to REST mode."""

    async def _setup_streaming_failure(self, raw_pcm: bytes = b"") -> None:
        mode_evt = MagicMock()
        mode_evt.control.selected = {"streaming"}
        self.mode_selector.on_change(mode_evt)

        self.mock_client.stream_translate = AsyncMock(
            side_effect=WebSocketConnectionError("connection failed"),
        )

        self.mock_recorder.is_recording = False
        await self.on_record_toggle(MagicMock())  # start streaming
        # Mocked stream_translate raises immediately; 50ms lets the Task resolve.
        await asyncio.sleep(STREAMING_SETTLE_SECS)

        self.mock_recorder.stop.return_value = raw_pcm
        await self.on_record_toggle(MagicMock())  # stop → fallback

    async def test_switches_mode_to_rest(self) -> None:
        await self._setup_streaming_failure()
        self.assertEqual(self.mode_selector.selected, ["rest"])

    async def test_shows_amber_snackbar_with_degradation_message(self) -> None:
        await self._setup_streaming_failure()

        fallback_sb = None
        for sb in self.page.overlay:
            if hasattr(sb, "content") and hasattr(sb.content, "value"):
                if "WebSocket 不可用" in sb.content.value:
                    fallback_sb = sb
                    break

        self.assertIsNotNone(fallback_sb, "Fallback snackbar not found")
        self.assertEqual(fallback_sb.bgcolor, AMBER)

    async def test_replays_audio_via_rest_when_pcm_available(self) -> None:
        await self._setup_streaming_failure(raw_pcm=b"\x00" * 3200)
        self.mock_client.translate_file.assert_called_once()
        call_kwargs = self.mock_client.translate_file.call_args.kwargs
        self.assertEqual(call_kwargs["file_name"], "recording.wav")
        self.assertEqual(call_kwargs["breed"], BREED_DEFAULT)
        self.assertTrue(call_kwargs["audio_bytes"].startswith(b"RIFF"))


# ══════════════════════════════════════════════════════════════════
#  11. append_history — ExpansionTile with 5-dim tags
# ══════════════════════════════════════════════════════════════════

class TestAppendHistoryExpansionTile(_Batch4AppTestBase):
    """append_history creates ExpansionTile with full 5-dimension tag detail."""

    _RESULT_PAYLOAD_TEMPLATE: dict[str, Any] = {
        "type": "result",
        "transcription": "你好小猫",
        "selected_category": {
            "tags": {
                "emotion": ["lonely", "anxious"],
                "intent": ["seeking_companionship"],
                "acoustic": ["prolonged", "soft"],
                "social_context": ["alone_at_home"],
                "breed_voice": ["deep_voice"],
            },
            "sample_id": SAMPLE_ID_SECONDARY,
            "match_score": MATCH_SCORE_HIGH,
        },
        "audio_base64": AUDIO_B64_STUB,
        "reasoning": "test",
    }

    @classmethod
    def _make_result_payload(cls) -> dict[str, Any]:
        return copy.deepcopy(cls._RESULT_PAYLOAD_TEMPLATE)

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        await self._start_streaming_recording()
        self.on_ws_event = (
            self.mock_client.stream_translate.call_args.kwargs["on_event"]
        )

    async def test_creates_expansion_tile_with_5_rows(self) -> None:
        await self.on_ws_event(self._make_result_payload())

        self.assertEqual(len(self.history_view.controls), 1)
        item = self.history_view.controls[0]
        detail_panel = item.content.controls[2]

        self.assertIsInstance(detail_panel, _ListCtrl)
        self.assertEqual(len(detail_panel.controls), 5)

    async def test_expansion_tile_dimension_labels(self) -> None:
        await self.on_ws_event(self._make_result_payload())

        detail_panel = self.history_view.controls[0].content.controls[2]
        dims = ("emotion", "intent", "acoustic", "social_context", "breed_voice")
        for i, dim in enumerate(dims):
            row = detail_panel.controls[i]
            self.assertEqual(row.controls[0].value, dim)

    async def test_expansion_tile_tag_values(self) -> None:
        await self.on_ws_event(self._make_result_payload())

        detail_panel = self.history_view.controls[0].content.controls[2]
        self.assertEqual(detail_panel.controls[0].controls[1].value, "lonely, anxious")
        self.assertEqual(detail_panel.controls[1].controls[1].value, "seeking_companionship")
        self.assertEqual(detail_panel.controls[2].controls[1].value, "prolonged, soft")
        self.assertEqual(detail_panel.controls[3].controls[1].value, "alone_at_home")
        self.assertEqual(detail_panel.controls[4].controls[1].value, "deep_voice")

    async def test_empty_tags_show_dash(self) -> None:
        payload = {
            "type": "result",
            "transcription": "你好",
            "selected_category": {
                "tags": {
                    "emotion": ["happy"],
                    "intent": [],
                    "acoustic": [],
                    "social_context": [],
                    "breed_voice": [],
                },
                "sample_id": "cat_001",
                "match_score": 0.5,
            },
            "audio_base64": AUDIO_B64_STUB,
        }
        await self.on_ws_event(payload)

        detail_panel = self.history_view.controls[0].content.controls[2]
        self.assertEqual(detail_panel.controls[0].controls[1].value, "happy")
        self.assertEqual(detail_panel.controls[1].controls[1].value, "-")
        self.assertEqual(detail_panel.controls[2].controls[1].value, "-")

    async def test_shows_transcription_text(self) -> None:
        await self.on_ws_event(self._make_result_payload())

        item = self.history_view.controls[0]
        self.assertEqual(item.content.controls[0].value, "你好小猫")

    async def test_shows_timestamp_in_subtitle(self) -> None:
        await self.on_ws_event(self._make_result_payload())

        subtitle = self.history_view.controls[0].content.controls[1].value
        self.assertRegex(subtitle, r"\d{2}:\d{2}:\d{2}")

    async def test_score_shows_match_percentage(self) -> None:
        await self.on_ws_event(self._make_result_payload())

        subtitle = self.history_view.controls[0].content.controls[1].value
        self.assertIn("85%", subtitle)

    async def test_no_score_omits_match_percentage(self) -> None:
        payload = {
            "type": "result",
            "transcription": "你好",
            "selected_category": {
                "tags": {"emotion": ["happy"]},
                "sample_id": "cat_001",
            },
        }
        await self.on_ws_event(payload)

        subtitle = self.history_view.controls[0].content.controls[1].value
        self.assertNotIn("匹配:", subtitle)

    async def test_multiple_entries_newest_first(self) -> None:
        """append_history inserts at index 0, so newest entries appear first."""
        payload1 = {
            "type": "result",
            "transcription": "第一条",
            "selected_category": {
                "tags": {"emotion": ["happy"]},
                "sample_id": "cat_001",
            },
        }
        payload2 = {
            "type": "result",
            "transcription": "第二条",
            "selected_category": {
                "tags": {"emotion": ["lonely"]},
                "sample_id": "cat_002",
            },
        }
        await self.on_ws_event(payload1)
        await self.on_ws_event(payload2)

        self.assertEqual(len(self.history_view.controls), 2)
        self.assertEqual(
            self.history_view.controls[0].content.controls[0].value, "第二条",
        )
        self.assertEqual(
            self.history_view.controls[1].content.controls[0].value, "第一条",
        )


# ══════════════════════════════════════════════════════════════════
#  12. Bridge card structure
# ══════════════════════════════════════════════════════════════════

class TestBridgeCardStructure(_Batch4AppTestBase):
    """Bridge card must contain ws_status_chip and recording_timer_text."""

    async def test_ws_status_chip_initial_state(self) -> None:
        self.assertIsInstance(self.ws_status_chip, _Ctrl)
        self.assertEqual(self.ws_status_chip.label.value, "已断开")
        self.assertEqual(self.ws_status_chip.bgcolor, PAW_PINK)

    async def test_recording_timer_initial_state(self) -> None:
        self.assertIsInstance(self.recording_timer_text, _TextCtrl)
        self.assertEqual(self.recording_timer_text.value, "00:00")
        self.assertFalse(self.recording_timer_text.visible)


# ══════════════════════════════════════════════════════════════════
#  13. History legacy fallback
# ══════════════════════════════════════════════════════════════════

class TestHistoryLegacyFallback(_Batch4AppTestBase):
    """History entries without streaming tags show Legacy fallback."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        await self._start_streaming_recording()
        self.on_ws_event = (
            self.mock_client.stream_translate.call_args.kwargs["on_event"]
        )

    async def test_legacy_uses_emotion_category_in_subtitle(self) -> None:
        payload = {
            "type": "result",
            "emotion_category": "happy",
            "sound_id": SOUND_ID_LEGACY,
        }
        await self.on_ws_event(payload)

        subtitle = self.history_view.controls[0].content.controls[1].value
        self.assertIn("happy", subtitle)

    async def test_legacy_expansion_tile_shows_all_dashes(self) -> None:
        payload = {
            "type": "result",
            "emotion_category": "happy",
            "sound_id": SOUND_ID_LEGACY,
        }
        await self.on_ws_event(payload)

        detail_panel = self.history_view.controls[0].content.controls[2]
        self.assertEqual(len(detail_panel.controls), 5)
        for row in detail_panel.controls:
            self.assertEqual(row.controls[1].value, "-")


if __name__ == "__main__":
    unittest.main()
