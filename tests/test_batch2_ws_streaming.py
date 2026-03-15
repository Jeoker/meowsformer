"""
Tests for Batch 2 — WebSocket Streaming Pipeline Integration
=============================================================
Covers TranslationClient concurrent WS communication and
app.py UI logic for streaming mode, 5-dim tags, and event handling.

Tests:
  TranslationClient (translation_client.py):
    1. Concurrent send/receive via TaskGroup
    2. Config message sent first (with breed_preference)
    3. Sender sends stop after chunks exhausted
    4. Receiver terminates on result/error; skips binary
    5. TaskGroup exception propagation (sender↔receiver)
    6. JSONDecodeError → synthetic error event
    7. translate_file REST method + _build_ws_url

  App UI logic (app.py closures):
    8.  update_tags Phase 5 TargetTagSet → 5-dim chips
    9.  update_tags Legacy fallback
    10. on_ws_event transcription → live_transcription update
    11. on_ws_event result → tags + history + player_status
    12. on_ws_event error → analysis_status error display
    12b. on_ws_event analysis_preview → preview chips + speculative bar
    13. append_history Phase 5 adaptation (transcription + score)
    14. _chunk_generator RuntimeError when queue is None (specification test)
    15. _chunk_generator normal flow until None sentinel (specification test)
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
import unittest

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# ── Mock chromadb before any app imports (project convention) ──────
sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("chromadb.utils", MagicMock())
sys.modules.setdefault("chromadb.utils.embedding_functions", MagicMock())

# ── Test parameters ───────────────────────────────────────────────
from tests.shared_params import (  # noqa: E402
    AUDIO_B64_STUB,
    BREED_ALT,
    BREED_DEFAULT,
    MATCH_SCORE_HIGH,
    MATCH_SCORE_PERFECT,
    MINIMAL_RESULT_MSG,
    REST_PATH_V1,
    SAMPLE_ID_PRIMARY,
    SAMPLE_ID_SECONDARY,
    SERVER_BASE_URL,
    SOUND_ID_LEGACY,
    STREAMING_SETTLE_SECS,
    WAVEFORM_MOCK_RETURN,
    WS_PATH,
)
from tests.flet_mocks import BaseMockPage, _Ctrl, _TextCtrl, install_flet_mock  # noqa: E402
from tests.ws_stubs import MockWebSocket, async_chunks, ws_connect_coro  # noqa: E402

install_flet_mock()

# ── Import TranslationClient (no flet dependency) ─────────────────
from src.flet_mobile.translation_client import TranslationClient  # noqa: E402


async def _chunk_generator_spec(
    queue: asyncio.Queue[bytes | None] | None,
) -> AsyncGenerator[bytes, None]:
    """Specification-level replica of the ``_chunk_generator`` closure defined
    inside ``meowsformer_ui``.  Because the production function is a closure
    that captures ``_chunk_queue`` from the enclosing scope, it cannot be
    imported directly.  This helper mirrors the *contract* (raise on None
    queue, yield until sentinel) so tests verify the algorithm specification
    rather than acting as regression tests against the real closure.  Any
    behavioural change in the production closure must be reflected here."""
    if queue is None:
        raise RuntimeError(
            "_chunk_queue must be initialised before streaming"
        )
    while True:
        chunk = await queue.get()
        if chunk is None:
            return
        yield chunk


def _patch_ws(mock_ws: MockWebSocket):
    """Return combined patches for websockets.connect + _build_ws_url."""
    return (
        patch.object(TranslationClient, "_build_ws_url", return_value="ws://mock"),
        patch(
            "src.flet_mobile.translation_client.websockets.connect",
            ws_connect_coro(mock_ws),
        ),
    )


# ══════════════════════════════════════════════════════════════════
#  1. Concurrent Communication (Test 1)
# ══════════════════════════════════════════════════════════════════

class TestConcurrentCommunication(unittest.IsolatedAsyncioTestCase):
    """stream_translate() must send PCM + receive events concurrently."""

    async def test_concurrent_send_and_receive(self) -> None:
        transcription = json.dumps(
            {"type": "transcription", "text": "你好", "is_final": False}
        )
        result = json.dumps(
            {
                "type": "result",
                "transcription": "你好小猫",
                "selected_category": {
                    "tags": {},
                    "sample_id": SAMPLE_ID_PRIMARY,
                    "match_score": MATCH_SCORE_PERFECT,
                },
                "audio_base64": AUDIO_B64_STUB,
                "reasoning": "test",
            }
        )
        mock_ws = MockWebSocket(server_messages=[transcription, result])
        events: list[dict] = []

        async def on_event(payload: dict) -> None:
            events.append(payload)

        client = TranslationClient()
        p1, p2 = _patch_ws(mock_ws)
        with p1, p2:
            await client.stream_translate(
                chunks=async_chunks(b"\x00" * 4096, b"\x01" * 4096),
                on_event=on_event,
                breed_preference=BREED_DEFAULT,
            )

        self.assertEqual(len(mock_ws.sent), 4, "config + 2 chunks + stop")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["type"], "transcription")
        self.assertEqual(events[1]["type"], "result")


# ══════════════════════════════════════════════════════════════════
#  2. Config Message Sent First (Test 2)
# ══════════════════════════════════════════════════════════════════

class TestConfigMessage(unittest.IsolatedAsyncioTestCase):
    """First message on the WS must be config JSON with breed_preference."""

    async def test_config_sent_with_breed(self) -> None:
        mock_ws = MockWebSocket(server_messages=[MINIMAL_RESULT_MSG])
        client = TranslationClient()
        p1, p2 = _patch_ws(mock_ws)
        with p1, p2:
            await client.stream_translate(
                chunks=async_chunks(),
                on_event=AsyncMock(),
                breed_preference=BREED_ALT,
            )

        config = json.loads(mock_ws.sent[0])
        self.assertEqual(config["type"], "config")
        self.assertEqual(config["breed_preference"], BREED_ALT)

    async def test_config_defaults_to_default(self) -> None:
        mock_ws = MockWebSocket(server_messages=[MINIMAL_RESULT_MSG])
        client = TranslationClient()
        p1, p2 = _patch_ws(mock_ws)
        with p1, p2:
            await client.stream_translate(
                chunks=async_chunks(),
                on_event=AsyncMock(),
            )

        config = json.loads(mock_ws.sent[0])
        self.assertEqual(config["breed_preference"], "Default")


# ══════════════════════════════════════════════════════════════════
#  3. Sender Stop Semantic (Test 3)
# ══════════════════════════════════════════════════════════════════

class TestSenderStop(unittest.IsolatedAsyncioTestCase):
    """After chunk iteration ends, sender must send {"type": "stop"}."""

    async def test_stop_after_chunks(self) -> None:
        mock_ws = MockWebSocket(server_messages=[MINIMAL_RESULT_MSG])
        client = TranslationClient()
        p1, p2 = _patch_ws(mock_ws)
        with p1, p2:
            await client.stream_translate(
                chunks=async_chunks(b"\xAA", b"\xBB"),
                on_event=AsyncMock(),
            )

        stop_msg = json.loads(mock_ws.sent[-1])
        self.assertEqual(stop_msg, {"type": "stop"})

    async def test_stop_sent_with_no_chunks(self) -> None:
        mock_ws = MockWebSocket(server_messages=[MINIMAL_RESULT_MSG])
        client = TranslationClient()
        p1, p2 = _patch_ws(mock_ws)
        with p1, p2:
            await client.stream_translate(
                chunks=async_chunks(),
                on_event=AsyncMock(),
            )

        self.assertEqual(len(mock_ws.sent), 2, "config + stop only")
        stop_msg = json.loads(mock_ws.sent[-1])
        self.assertEqual(stop_msg, {"type": "stop"})


# ══════════════════════════════════════════════════════════════════
#  4. Receiver Termination (Test 4)
# ══════════════════════════════════════════════════════════════════

class TestReceiverTermination(unittest.IsolatedAsyncioTestCase):
    """Receiver must stop processing after result or error."""

    async def test_stops_on_result(self) -> None:
        messages = [
            json.dumps({"type": "transcription", "text": "a"}),
            json.dumps({"type": "result", "transcription": "b"}),
            json.dumps({"type": "transcription", "text": "unreachable"}),
        ]
        mock_ws = MockWebSocket(server_messages=messages)
        events: list[dict] = []

        async def on_event(p: dict) -> None:
            events.append(p)

        client = TranslationClient()
        p1, p2 = _patch_ws(mock_ws)
        with p1, p2:
            await client.stream_translate(
                chunks=async_chunks(), on_event=on_event,
            )

        self.assertEqual(len(events), 2)
        self.assertEqual(events[-1]["type"], "result")

    async def test_stops_on_error(self) -> None:
        messages = [
            json.dumps({"type": "error", "detail": "timeout"}),
            json.dumps({"type": "transcription", "text": "unreachable"}),
        ]
        mock_ws = MockWebSocket(server_messages=messages)
        events: list[dict] = []

        async def on_event(p: dict) -> None:
            events.append(p)

        client = TranslationClient()
        p1, p2 = _patch_ws(mock_ws)
        with p1, p2:
            await client.stream_translate(
                chunks=async_chunks(), on_event=on_event,
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")

    async def test_skips_binary_messages(self) -> None:
        messages: list = [
            b"\x00\x01\x02",
            MINIMAL_RESULT_MSG,
        ]
        mock_ws = MockWebSocket(server_messages=messages)
        events: list[dict] = []

        async def on_event(p: dict) -> None:
            events.append(p)

        client = TranslationClient()
        p1, p2 = _patch_ws(mock_ws)
        with p1, p2:
            await client.stream_translate(
                chunks=async_chunks(), on_event=on_event,
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "result")


# ══════════════════════════════════════════════════════════════════
#  5. TaskGroup Exception Propagation (Test 5)
# ══════════════════════════════════════════════════════════════════

class TestTaskGroupExceptionPropagation(unittest.IsolatedAsyncioTestCase):
    """Errors in one coroutine must propagate via ExceptionGroup."""

    async def test_sender_error_propagates(self) -> None:
        async def bad_chunks():
            yield b"\x00"
            raise ValueError("sender boom")

        mock_ws = MockWebSocket()

        async def _block_forever():
            while True:
                await asyncio.sleep(100)
                yield "never"  # pragma: no cover

        mock_ws._iter = _block_forever  # type: ignore[assignment]

        client = TranslationClient()
        p1, p2 = _patch_ws(mock_ws)
        with p1, p2:
            with self.assertRaises(ExceptionGroup) as ctx:
                await client.stream_translate(
                    chunks=bad_chunks(), on_event=AsyncMock(),
                )

        self.assertEqual(len(ctx.exception.exceptions), 1)
        self.assertIsInstance(ctx.exception.exceptions[0], ValueError)

    async def test_receiver_error_propagates(self) -> None:
        messages = [json.dumps({"type": "transcription", "text": "x"})]
        mock_ws = MockWebSocket(server_messages=messages)

        async def exploding_event(_payload: dict) -> None:
            raise RuntimeError("handler boom")

        async def slow_chunks():
            await asyncio.sleep(100)
            yield b"\x00"  # pragma: no cover

        client = TranslationClient()
        p1, p2 = _patch_ws(mock_ws)
        with p1, p2:
            with self.assertRaises(ExceptionGroup) as ctx:
                await client.stream_translate(
                    chunks=slow_chunks(), on_event=exploding_event,
                )

        self.assertEqual(len(ctx.exception.exceptions), 1)
        self.assertIsInstance(ctx.exception.exceptions[0], RuntimeError)


# ══════════════════════════════════════════════════════════════════
#  6. JSONDecodeError Handling (Test 6)
# ══════════════════════════════════════════════════════════════════

class TestJSONDecodeError(unittest.IsolatedAsyncioTestCase):
    """Malformed JSON from server must produce a synthetic error event."""

    async def test_malformed_json_triggers_error(self) -> None:
        mock_ws = MockWebSocket(server_messages=["not valid json {{{{"])
        events: list[dict] = []

        async def on_event(p: dict) -> None:
            events.append(p)

        client = TranslationClient()
        p1, p2 = _patch_ws(mock_ws)
        with p1, p2:
            await client.stream_translate(
                chunks=async_chunks(), on_event=on_event,
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertIn("畸形 JSON", events[0]["detail"])

    async def test_malformed_json_truncates_long_content(self) -> None:
        long_garbage = "x" * 250
        mock_ws = MockWebSocket(server_messages=[long_garbage])
        events: list[dict] = []

        async def on_event(p: dict) -> None:
            events.append(p)

        client = TranslationClient()
        p1, p2 = _patch_ws(mock_ws)
        with p1, p2:
            await client.stream_translate(
                chunks=async_chunks(), on_event=on_event,
            )

        detail = events[0]["detail"]
        raw_content = detail.split(": ", 1)[1]
        self.assertLessEqual(len(raw_content), 120)


# ══════════════════════════════════════════════════════════════════
#  7. translate_file REST + _build_ws_url (Test 7)
# ══════════════════════════════════════════════════════════════════

class TestTranslateFileREST(unittest.IsolatedAsyncioTestCase):
    """REST translate_file() must POST to /api/v1/translate correctly."""

    async def test_translate_file_calls_api(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "emotion_category": "happy",
            "sound_id": SOUND_ID_LEGACY,
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)

        with patch("src.flet_mobile.translation_client.httpx.AsyncClient") as MockAC:
            MockAC.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            MockAC.return_value.__aexit__ = AsyncMock(return_value=False)

            client = TranslationClient(SERVER_BASE_URL)
            result = await client.translate_file(
                file_name="test.wav",
                audio_bytes=b"\x00" * 100,
                breed=BREED_DEFAULT,
                output_sr=22050,
            )

        self.assertEqual(result["emotion_category"], "happy")
        post_url = mock_http.post.call_args.args[0]
        self.assertIn(REST_PATH_V1, post_url)

    def test_build_ws_url_http(self) -> None:
        client = TranslationClient(SERVER_BASE_URL)
        self.assertEqual(
            client._build_ws_url(WS_PATH),
            "ws://localhost:8000/ws/translate",
        )

    def test_build_ws_url_https(self) -> None:
        client = TranslationClient("https://api.example.com")
        self.assertEqual(
            client._build_ws_url(WS_PATH),
            "wss://api.example.com/ws/translate",
        )


# ══════════════════════════════════════════════════════════════════
#  App UI logic tests — Base class
# ══════════════════════════════════════════════════════════════════

class _AppTestBase(unittest.IsolatedAsyncioTestCase):
    """
    Shared setUp: run meowsformer_ui with mocked dependencies, switch to
    streaming mode, start recording, and capture the on_ws_event closure.
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
            return_value=WAVEFORM_MOCK_RETURN
        )

        self.page = BaseMockPage()

        from src.flet_mobile.app import meowsformer_ui

        await meowsformer_ui(self.page)

        # ── Navigate control tree ─────────────────────────────────
        # Guard assertions detect UI layout changes that would
        # silently break index-based navigation.
        main_col = self.page._added[0]
        self.assertEqual(
            len(main_col.controls), 5,
            "Expected 5 top-level cards (bridge, record, lab, output, library); "
            "layout has changed — update indices below",
        )

        bridge_card = main_col.controls[0]
        bridge_col = bridge_card.content
        header_row = bridge_col.controls[0]
        self.assertEqual(
            header_row.controls[0].value, "The Bridge",
            "bridge_card section header mismatch — layout changed",
        )
        self.mode_selector = bridge_col.controls[2]
        self.live_transcription = bridge_col.controls[5]

        record_container = main_col.controls[1]
        self.record_button = record_container.content
        self.on_record_toggle = self.record_button.on_click

        lab_card = main_col.controls[2]
        lab_col = lab_card.content
        self.assertEqual(
            lab_col.controls[0].value, "The Lab",
            "lab_card section header mismatch — layout changed",
        )
        self.analysis_status = lab_col.controls[1]
        self.speculative_bar = lab_col.controls[2]
        self.tags_wrap = lab_col.controls[3]

        output_card = main_col.controls[3]
        output_col = output_card.content
        self.assertEqual(
            output_col.controls[0].value, "The Output",
            "output_card section header mismatch — layout changed",
        )
        self.player_status = output_col.controls[5]

        library_card = main_col.controls[4]
        lib_col = library_card.content
        self.assertEqual(
            lib_col.controls[0].value, "The Library",
            "library_card section header mismatch — layout changed",
        )
        self.history_view = lib_col.controls[2]

        # ── Switch to streaming + start recording ─────────────────
        mode_evt = MagicMock()
        mode_evt.control.selected = {"streaming"}
        self.mode_selector.on_change(mode_evt)

        self.mock_recorder.is_recording = False
        await self.on_record_toggle(MagicMock())
        await asyncio.sleep(STREAMING_SETTLE_SECS)

        self.assertTrue(
            self.mock_client.stream_translate.called,
            "stream_translate should have been invoked by _run_streaming_session",
        )
        self.on_ws_event = (
            self.mock_client.stream_translate.call_args.kwargs["on_event"]
        )

    async def asyncTearDown(self) -> None:
        for p in self._patchers:
            p.stop()


# ══════════════════════════════════════════════════════════════════
#  8. update_tags — Phase 5 TargetTagSet (Test 8)
# ══════════════════════════════════════════════════════════════════

class TestUpdateTagsPhase5(_AppTestBase):
    """Phase 5 result should produce one Chip per dimension:tag pair."""

    async def test_phase5_creates_5dim_chips(self) -> None:
        payload = {
            "type": "result",
            "transcription": "我很想你",
            "selected_category": {
                "tags": {
                    "emotion": ["lonely", "anxious"],
                    "intent": ["seeking_companionship"],
                    "acoustic": ["prolonged"],
                    "social_context": ["alone_at_home"],
                    "breed_voice": [],
                },
                "sample_id": SAMPLE_ID_SECONDARY,
                    "match_score": MATCH_SCORE_HIGH,
                },
                "audio_base64": AUDIO_B64_STUB,
                "reasoning": "用户表达思念",
            }
        await self.on_ws_event(payload)

        chip_labels = [c.label.value for c in self.tags_wrap.controls]
        expected = [
            "emotion: lonely",
            "emotion: anxious",
            "intent: seeking_companionship",
            "acoustic: prolonged",
            "social_context: alone_at_home",
        ]
        self.assertEqual(chip_labels, expected)
        self.assertEqual(len(self.tags_wrap.controls), 5)


# ══════════════════════════════════════════════════════════════════
#  9. update_tags — Legacy Fallback (Test 9)
# ══════════════════════════════════════════════════════════════════

class TestUpdateTagsLegacy(_AppTestBase):
    """Result without Phase 5 tags should fall back to legacy chip format."""

    async def test_legacy_fallback_chips(self) -> None:
        payload = {
            "type": "result",
            "emotion_category": "happy",
            "sound_id": SOUND_ID_LEGACY,
            "pitch_adjust": 1.2,
        }
        await self.on_ws_event(payload)

        chip_labels = [c.label.value for c in self.tags_wrap.controls]
        self.assertEqual(len(chip_labels), 5)
        self.assertIn("Emotion: happy", chip_labels)
        self.assertIn(f"Intent: {SOUND_ID_LEGACY}", chip_labels)
        self.assertIn("Acoustic: pitch 1.2", chip_labels)
        self.assertIn("Social: owner_present", chip_labels)
        self.assertTrue(
            any(lbl.startswith("Breed:") for lbl in chip_labels),
            "Legacy fallback should include a Breed chip",
        )


# ══════════════════════════════════════════════════════════════════
#  10. on_ws_event — transcription (Test 10)
# ══════════════════════════════════════════════════════════════════

class TestOnWsEventTranscription(_AppTestBase):
    """Transcription event should update live_transcription text."""

    async def test_transcription_updates_text(self) -> None:
        await self.on_ws_event(
            {"type": "transcription", "text": "你好小猫", "is_final": False}
        )
        self.assertEqual(self.live_transcription.value, "你好小猫")

    async def test_transcription_empty_text(self) -> None:
        await self.on_ws_event({"type": "transcription"})
        self.assertEqual(self.live_transcription.value, "")


# ══════════════════════════════════════════════════════════════════
#  11. on_ws_event — result (Test 11)
# ══════════════════════════════════════════════════════════════════

class TestOnWsEventResult(_AppTestBase):
    """Result event should update tags, history, and player status."""

    async def test_result_updates_all(self) -> None:
        payload = {
            "type": "result",
            "transcription": "我很想你",
            "selected_category": {
                "tags": {
                    "emotion": ["lonely"],
                    "intent": ["seeking_companionship"],
                    "acoustic": [],
                    "social_context": [],
                    "breed_voice": [],
                },
                "sample_id": SAMPLE_ID_SECONDARY,
                "match_score": MATCH_SCORE_HIGH,
            },
            "audio_base64": AUDIO_B64_STUB,
            "reasoning": "用户表达思念",
        }
        await self.on_ws_event(payload)

        self.assertEqual(self.speculative_bar.value, 1.0)
        self.assertIn("翻译完成", self.analysis_status.value)
        self.assertEqual(self.live_transcription.value, "我很想你")
        self.assertIn(SAMPLE_ID_SECONDARY, self.player_status.value)
        self.assertGreater(len(self.tags_wrap.controls), 0)
        self.assertGreater(len(self.history_view.controls), 0)


# ══════════════════════════════════════════════════════════════════
#  12. on_ws_event — error (Test 12)
# ══════════════════════════════════════════════════════════════════

class TestOnWsEventError(_AppTestBase):
    """Error event should display error via Snackbar."""

    async def test_error_shows_detail(self) -> None:
        await self.on_ws_event(
            {"type": "error", "detail": "Whisper API timeout"}
        )
        self.assertTrue(
            len(self.page.overlay) > 0,
            "Snackbar should have been opened on error",
        )
        snackbar = self.page.overlay[-1]
        self.assertIn("Whisper API timeout", snackbar.content.value)
        self.assertEqual(self.speculative_bar.value, 0.0)

    async def test_error_unknown_detail(self) -> None:
        await self.on_ws_event({"type": "error"})
        self.assertTrue(
            len(self.page.overlay) > 0,
            "Snackbar should have been opened on error",
        )
        snackbar = self.page.overlay[-1]
        self.assertIn("未知错误", snackbar.content.value)


# ══════════════════════════════════════════════════════════════════
#  12b. on_ws_event — analysis_preview (Test 12b)
# ══════════════════════════════════════════════════════════════════

class TestOnWsEventAnalysisPreview(_AppTestBase):
    """analysis_preview event should update preview chips, speculative bar,
    and analysis status text."""

    async def test_preview_updates_emotion_intent_chips(self) -> None:
        await self.on_ws_event({
            "type": "analysis_preview",
            "emotion": ["lonely", "anxious"],
            "intent": ["seeking_companionship"],
        })

        chip_labels = [c.label.value for c in self.tags_wrap.controls]
        self.assertEqual(chip_labels, [
            "emotion: lonely",
            "emotion: anxious",
            "intent: seeking_companionship",
        ])

    async def test_preview_sets_speculative_bar(self) -> None:
        await self.on_ws_event({
            "type": "analysis_preview",
            "emotion": ["happy"],
            "intent": [],
        })
        self.assertAlmostEqual(self.speculative_bar.value, 0.65)

    async def test_preview_updates_analysis_status(self) -> None:
        await self.on_ws_event({
            "type": "analysis_preview",
            "emotion": ["content"],
            "intent": ["expressing_comfort"],
        })
        self.assertIn("推测性分析就绪", self.analysis_status.value)

    async def test_preview_empty_tags_preserves_existing_chips(self) -> None:
        """When both emotion and intent are empty, tags_wrap should keep its
        previous controls (the branch guards with ``if preview_chips:``)."""
        sentinel_chip = _Ctrl(label=_TextCtrl("sentinel"))
        self.tags_wrap.controls = [sentinel_chip]

        await self.on_ws_event({
            "type": "analysis_preview",
            "emotion": [],
            "intent": [],
        })

        self.assertEqual(len(self.tags_wrap.controls), 1)
        self.assertEqual(self.tags_wrap.controls[0].label.value, "sentinel")
        self.assertAlmostEqual(self.speculative_bar.value, 0.65)


# ══════════════════════════════════════════════════════════════════
#  13. append_history — Phase 5 Adaptation (Test 13)
# ══════════════════════════════════════════════════════════════════

class TestAppendHistoryPhase5(_AppTestBase):
    """History entry should extract Phase 5 transcription, tags, and score."""

    async def test_history_entry_content(self) -> None:
        payload = {
            "type": "result",
            "transcription": "我很想你",
            "selected_category": {
                "tags": {
                    "emotion": ["lonely", "anxious"],
                    "intent": ["seeking_companionship"],
                    "acoustic": [],
                    "social_context": [],
                    "breed_voice": [],
                },
                "sample_id": SAMPLE_ID_SECONDARY,
                "match_score": MATCH_SCORE_HIGH,
            },
            "audio_base64": AUDIO_B64_STUB,
            "reasoning": "用户表达思念",
        }
        await self.on_ws_event(payload)

        self.assertEqual(len(self.history_view.controls), 1)
        item = self.history_view.controls[0]
        col = item.content

        transcription_text = col.controls[0].value
        self.assertEqual(transcription_text, "我很想你")

        subtitle = col.controls[1].value
        self.assertIn("lonely", subtitle)
        self.assertIn("seeking_companionship", subtitle)
        self.assertIn("85%", subtitle)

    async def test_history_fallback_transcription(self) -> None:
        """When 'transcription' is absent, fall back to human_interpretation."""
        payload = {
            "type": "result",
            "human_interpretation": "回退文本",
        }
        await self.on_ws_event(payload)

        item = self.history_view.controls[0]
        transcription_text = item.content.controls[0].value
        self.assertEqual(transcription_text, "回退文本")


# ══════════════════════════════════════════════════════════════════
#  14. _chunk_generator — RuntimeError (Test 14)
#
#  SPECIFICATION TEST — _chunk_generator is a closure inside
#  meowsformer_ui and cannot be imported.  These tests verify the
#  *specified contract* (raise when queue is None) using a faithful
#  replica (_chunk_generator_spec).  They do NOT guard against
#  regressions in the production closure; if the closure's logic
#  changes, _chunk_generator_spec must be updated in lockstep.
# ══════════════════════════════════════════════════════════════════

class TestChunkGeneratorRuntimeError(unittest.IsolatedAsyncioTestCase):
    """_chunk_generator must raise RuntimeError when _chunk_queue is None.

    Specification test — see _chunk_generator_spec docstring."""

    async def test_raises_when_queue_is_none(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            async for _ in _chunk_generator_spec(None):
                pass  # pragma: no cover

        self.assertIn("_chunk_queue must be initialised", str(ctx.exception))


# ══════════════════════════════════════════════════════════════════
#  15. _chunk_generator — Normal Flow (Test 15)
#
#  SPECIFICATION TEST — see note on Test 14 above.
# ══════════════════════════════════════════════════════════════════

class TestChunkGeneratorNormalFlow(unittest.IsolatedAsyncioTestCase):
    """_chunk_generator must yield chunks from queue until None sentinel.

    Specification test — see _chunk_generator_spec docstring."""

    async def test_yields_until_sentinel(self) -> None:
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        queue.put_nowait(b"\xAA" * 100)
        queue.put_nowait(b"\xBB" * 200)
        queue.put_nowait(None)

        collected: list[bytes] = []
        async for chunk in _chunk_generator_spec(queue):
            collected.append(chunk)

        self.assertEqual(len(collected), 2)
        self.assertEqual(collected[0], b"\xAA" * 100)
        self.assertEqual(collected[1], b"\xBB" * 200)

    async def test_empty_immediately_stops(self) -> None:
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        queue.put_nowait(None)

        collected: list[bytes] = []
        async for chunk in _chunk_generator_spec(queue):
            collected.append(chunk)  # pragma: no cover

        self.assertEqual(len(collected), 0)


if __name__ == "__main__":
    unittest.main()
