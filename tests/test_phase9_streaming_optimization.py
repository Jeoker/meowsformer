"""
Tests for Phase 9 Batch 2 — Rolling Speculative Execution + Parallel Stop
==========================================================================
Covers changes in ``app/api/ws_endpoints.py``:

Change C — Rolling speculative execution:
    1. Old speculative task cancelled when new intermediate arrives (≥5 words)
    2. New asyncio task created for each rolling intermediate
    3. <5 words does NOT trigger speculative task
    4. CancelledError caught gracefully, no error message leaked

Change B — Parallel stop sequence:
    5. Final transcription + speculative wait run in parallel (timing)
    6. Speculative cache hit → reuse cached tags (no new LLM call)
    7. Cache miss → fresh generate_target_tags call
    8. Speculative timeout → task cancelled, flow continues
    9. CancelledError propagation (final_transcription_task also cancelled)
   10. Transcription failure → falls back to latest_text
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ── Mock chromadb before any app imports (project convention) ─────────────
sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("chromadb.utils", MagicMock())
sys.modules.setdefault("chromadb.utils.embedding_functions", MagicMock())

from app.api.ws_endpoints import (  # noqa: E402
    StreamingSession,
    _handle_stop,
    _speculative_analysis,
    ws_translate,
)
from app.schemas.ws_messages import (  # noqa: E402
    StreamingTranslationResult,
    TaggedSampleInfo,
    TargetTagSet,
)

_MOD = "app.api.ws_endpoints"


def _patch_ws_wait_for(side_effect):  # type: ignore[no-untyped-def]
    """Patch ``wait_for`` as bound in ``ws_endpoints`` (``asyncio.wait_for``).

    If production code switches to ``from asyncio import wait_for``, also add
    ``patch(f\"{_MOD}.wait_for\", ...)`` alongside this helper.
    """
    return patch(f"{_MOD}.asyncio.wait_for", side_effect=side_effect)


# ── Text fixtures ─────────────────────────────────────────────────────────
_5W = "hello world how are you"
_6W = "hello world how are you today"
_SHORT = "hi there"
_DIFF = "completely unrelated sentence about dogs and weather patterns"


# ── Object factories ─────────────────────────────────────────────────────


def _tags(**kw: object) -> TargetTagSet:
    d: dict = dict(
        emotion=["lonely"],
        intent=["seeking_companionship"],
        acoustic=["prolonged"],
        social_context=["alone_at_home"],
        reasoning="test",
    )
    d.update(kw)
    return TargetTagSet(**d)


def _result() -> StreamingTranslationResult:
    return StreamingTranslationResult(
        transcription="",
        target_tags=_tags(),
        selected_sample=TaggedSampleInfo(
            sample_id="cat_001",
            breed="Maine Coon",
            context="Food",
            tags={"emotion": ["hungry"]},
            match_score=0.85,
            matched_tags={"emotion": ["hungry"]},
        ),
        audio_base64="AAAA",
        reasoning="test",
    )


class _FakeWS:
    """Minimal async WebSocket double for endpoint-level tests."""

    def __init__(self, messages: list[dict] | None = None) -> None:
        self._messages = list(messages or [])
        self._idx = 0
        self.sent: list[dict] = []

    async def accept(self) -> None:
        pass

    async def receive(self) -> dict:
        if self._idx >= len(self._messages):
            return {"type": "websocket.disconnect", "code": 1000}
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)


def _mock_ts(
    intermediates: list[str | None],
    final: str = "final",
    should_flags: list[bool] | None = None,
) -> MagicMock:
    """Build a mock ``StreamingTranscriptionSession``."""
    m = MagicMock()
    m.add_chunk = MagicMock()
    m.should_transcribe = MagicMock(
        side_effect=should_flags or [True] * len(intermediates),
    )
    m.transcribe_intermediate = AsyncMock(side_effect=intermediates)
    m.transcribe_final = AsyncMock(return_value=final)
    m.latest_text = final
    m._latest_text = final
    m.reset = MagicMock()
    return m


# ══════════════════════════════════════════════════════════════════════════
#  1. Rolling speculative — old task cancelled
# ══════════════════════════════════════════════════════════════════════════


class TestRollingSpecCancelOld(unittest.IsolatedAsyncioTestCase):
    """Two ≥5-word intermediates → first speculative task must be cancelled."""

    async def test_first_task_cancelled(self) -> None:
        async def gen(text: str) -> TargetTagSet:
            # Block only the first intermediate's LLM; key by text so the second task
            # never takes the slow path even if it runs before the first calls gen().
            if text == _5W:
                try:
                    await asyncio.sleep(100)
                except asyncio.CancelledError:
                    raise
            return _tags()

        ts = _mock_ts([_5W, _6W], final=_6W)
        ws = _FakeWS([
            {"bytes": b"\x00"},
            {"bytes": b"\x00"},
            {"text": '{"type":"stop"}'},
        ])

        orig_create_task = asyncio.create_task
        speculative_tasks: list[asyncio.Task] = []

        def track_create(coro, *, name=None):  # type: ignore[no-untyped-def]
            t = orig_create_task(coro, name=name)
            speculative_tasks.append(t)
            return t

        with (
            patch(f"{_MOD}.load_tagged_samples"),
            patch(f"{_MOD}.StreamingTranscriptionSession", return_value=ts),
            patch(f"{_MOD}.generate_target_tags", side_effect=gen),
            patch(f"{_MOD}.select_and_encode", new_callable=AsyncMock, return_value=_result()),
            patch(f"{_MOD}.asyncio.create_task", side_effect=track_create),
        ):
            await ws_translate(ws)

        # Let the loop finish unwinding speculative tasks after ws_translate returns.
        await asyncio.sleep(0.15)

        self.assertGreaterEqual(len(speculative_tasks), 2)
        self.assertTrue(
            speculative_tasks[0].cancelled(),
            "First speculative asyncio.Task should be cancelled by the second",
        )


# ══════════════════════════════════════════════════════════════════════════
#  2. Rolling speculative — new task started
# ══════════════════════════════════════════════════════════════════════════


class TestRollingSpecNewTask(unittest.IsolatedAsyncioTestCase):
    """Second ≥5-word intermediate fires a new speculative task (analysis_preview sent)."""

    async def test_second_intermediate_sends_preview(self) -> None:
        async def gen(text: str) -> TargetTagSet:
            if text == _5W:
                try:
                    await asyncio.sleep(100)
                except asyncio.CancelledError:
                    raise
            return _tags()

        ts = _mock_ts([_5W, _6W], final=_6W)
        ws = _FakeWS([
            {"bytes": b"\x00"},
            {"bytes": b"\x00"},
            {"text": '{"type":"stop"}'},
        ])

        with (
            patch(f"{_MOD}.load_tagged_samples"),
            patch(f"{_MOD}.StreamingTranscriptionSession", return_value=ts),
            patch(f"{_MOD}.generate_target_tags", side_effect=gen),
            patch(f"{_MOD}.select_and_encode", new_callable=AsyncMock, return_value=_result()),
        ):
            await ws_translate(ws)

        previews = [m for m in ws.sent if m.get("type") == "analysis_preview"]
        self.assertEqual(len(previews), 1, "Exactly one preview from the second (surviving) task")


# ══════════════════════════════════════════════════════════════════════════
#  3. Rolling speculative — <5 words does NOT trigger
# ══════════════════════════════════════════════════════════════════════════


class TestRollingSpecShortText(unittest.IsolatedAsyncioTestCase):
    """Intermediate with <5 words must NOT create a speculative task."""

    async def test_no_speculative_under_five_words(self) -> None:
        ts = _mock_ts([_SHORT], final=_SHORT)
        ws = _FakeWS([
            {"bytes": b"\x00"},
            {"text": '{"type":"stop"}'},
        ])

        with (
            patch(f"{_MOD}.load_tagged_samples"),
            patch(f"{_MOD}.StreamingTranscriptionSession", return_value=ts),
            patch(f"{_MOD}.generate_target_tags", new_callable=AsyncMock, return_value=_tags()),
            patch(f"{_MOD}.select_and_encode", new_callable=AsyncMock, return_value=_result()),
        ):
            await ws_translate(ws)

        previews = [m for m in ws.sent if m.get("type") == "analysis_preview"]
        self.assertEqual(len(previews), 0, "No speculative preview for <5 words")


# ══════════════════════════════════════════════════════════════════════════
#  4. CancelledError in _speculative_analysis — no leak
# ══════════════════════════════════════════════════════════════════════════


class TestSpecCancelledErrorHandled(unittest.IsolatedAsyncioTestCase):
    """CancelledError in _speculative_analysis is caught; no error message sent."""

    async def test_no_error_on_cancel(self) -> None:
        session = StreamingSession()
        ws = _FakeWS()

        async def block_gen(text: str) -> TargetTagSet:
            await asyncio.sleep(100)
            return _tags()  # unreachable when cancelled

        with patch(f"{_MOD}.generate_target_tags", side_effect=block_gen):
            task = asyncio.create_task(
                _speculative_analysis(session, _5W, ws),
            )
            # Yield so the speculative task enters await generate_target_tags before cancel.
            await asyncio.sleep(0.01)
            task.cancel()
            # Task should complete normally (CancelledError caught internally)
            await task

        errors = [m for m in ws.sent if m.get("type") == "error"]
        self.assertEqual(len(errors), 0, "No error message should be sent on cancel")
        self.assertIsNone(
            session.speculative_cache.cached_tags,
            "Cache should NOT be populated after cancellation",
        )


# ══════════════════════════════════════════════════════════════════════════
#  5. Parallel stop — timing verification
# ══════════════════════════════════════════════════════════════════════════


class TestParallelStopTiming(unittest.IsolatedAsyncioTestCase):
    """transcribe_final and speculative wait_for overlap (structural, not wall-clock)."""

    async def test_parallel_faster_than_serial(self) -> None:
        marks: dict[str, float] = {}
        real_wait_for = asyncio.wait_for

        async def slow_final() -> str:
            marks["t_final"] = time.monotonic()
            await asyncio.sleep(0.5)
            return _5W

        async def track_wait_for(aw, timeout=5.0):  # type: ignore[no-untyped-def]
            marks["t_wait"] = time.monotonic()
            return await real_wait_for(aw, timeout=timeout)

        session = StreamingSession()
        session.transcription.transcribe_final = slow_final  # type: ignore[assignment]
        session.transcription._latest_text = _5W

        async def slow_spec() -> None:
            await asyncio.sleep(1.0)

        session._speculative_task = asyncio.create_task(slow_spec())
        session.speculative_cache.store(_5W, _tags())

        ws = _FakeWS()
        with _patch_ws_wait_for(track_wait_for):
            with (
                patch(f"{_MOD}.generate_target_tags", new_callable=AsyncMock, return_value=_tags()),
                patch(f"{_MOD}.select_and_encode", new_callable=AsyncMock, return_value=_result()),
            ):
                await _handle_stop(ws, session)

        self.assertIn("t_final", marks)
        self.assertIn("t_wait", marks)
        self.assertLess(
            abs(marks["t_final"] - marks["t_wait"]),
            0.25,
            "transcribe_final and wait_for(speculative) should start overlapped, "
            "not strictly serial",
        )


# ══════════════════════════════════════════════════════════════════════════
#  6. Parallel stop — speculative cache hit
# ══════════════════════════════════════════════════════════════════════════


class TestParallelStopCacheHit(unittest.IsolatedAsyncioTestCase):
    """Cache hit (similar text) → reuse cached tags, skip new LLM call."""

    async def test_no_new_llm_call(self) -> None:
        session = StreamingSession()
        session.transcription.transcribe_final = AsyncMock(return_value=_5W)  # type: ignore[assignment]
        session.speculative_cache.store(_5W, _tags())

        ws = _FakeWS()
        with (
            patch(f"{_MOD}.generate_target_tags", new_callable=AsyncMock) as mock_gen,
            patch(f"{_MOD}.select_and_encode", new_callable=AsyncMock, return_value=_result()),
        ):
            await _handle_stop(ws, session)

        mock_gen.assert_not_called()
        results = [m for m in ws.sent if m.get("type") == "result"]
        self.assertEqual(len(results), 1)


# ══════════════════════════════════════════════════════════════════════════
#  7. Parallel stop — cache miss
# ══════════════════════════════════════════════════════════════════════════


class TestParallelStopCacheMiss(unittest.IsolatedAsyncioTestCase):
    """Cache miss (dissimilar text) → new generate_target_tags call."""

    async def test_new_llm_call_on_miss(self) -> None:
        session = StreamingSession()
        session.transcription.transcribe_final = AsyncMock(return_value=_DIFF)  # type: ignore[assignment]
        session.speculative_cache.store(_5W, _tags(emotion=["happy"]))

        ws = _FakeWS()
        new_tags = _tags(emotion=["lonely"])
        with (
            patch(f"{_MOD}.generate_target_tags", new_callable=AsyncMock, return_value=new_tags) as mock_gen,
            patch(f"{_MOD}.select_and_encode", new_callable=AsyncMock, return_value=_result()),
        ):
            await _handle_stop(ws, session)

        mock_gen.assert_called_once_with(_DIFF)


# ══════════════════════════════════════════════════════════════════════════
#  8. Parallel stop — speculative timeout
# ══════════════════════════════════════════════════════════════════════════


class TestParallelStopSpecTimeout(unittest.IsolatedAsyncioTestCase):
    """Speculative task exceeds 5s timeout → cancelled, flow continues normally."""

    async def test_timeout_cancels_and_continues(self) -> None:
        session = StreamingSession()
        session.transcription.transcribe_final = AsyncMock(return_value=_5W)  # type: ignore[assignment]

        spec_task = MagicMock()
        spec_task.done.return_value = False
        spec_task.cancel = MagicMock()
        session._speculative_task = spec_task

        async def raise_timeout(aw, timeout=5.0):  # type: ignore[no-untyped-def]
            raise asyncio.TimeoutError()

        ws = _FakeWS()
        with (
            _patch_ws_wait_for(raise_timeout),
            patch(f"{_MOD}.generate_target_tags", new_callable=AsyncMock, return_value=_tags()),
            patch(f"{_MOD}.select_and_encode", new_callable=AsyncMock, return_value=_result()),
        ):
            await _handle_stop(ws, session)

        spec_task.cancel.assert_called_once()
        results = [m for m in ws.sent if m.get("type") == "result"]
        self.assertEqual(len(results), 1, "Flow should produce a result after timeout")


# ══════════════════════════════════════════════════════════════════════════
#  9. Parallel stop — CancelledError propagation
# ══════════════════════════════════════════════════════════════════════════


class TestParallelStopCancelPropagation(unittest.IsolatedAsyncioTestCase):
    """CancelledError from wait_for → final_transcription_task also cancelled."""

    async def test_cancel_propagates_to_final_task(self) -> None:
        session = StreamingSession()

        final_cancelled = asyncio.Event()

        async def tracked_final() -> str:
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                final_cancelled.set()
                raise
            return "unreachable"

        async def slow_spec() -> None:
            await asyncio.sleep(100)

        session.transcription.transcribe_final = tracked_final  # type: ignore[assignment]
        session._speculative_task = asyncio.create_task(slow_spec())

        ws = _FakeWS()
        with (
            patch(f"{_MOD}.generate_target_tags", new_callable=AsyncMock),
            patch(f"{_MOD}.select_and_encode", new_callable=AsyncMock),
        ):
            handle = asyncio.create_task(_handle_stop(ws, session))
            # Yield until _handle_stop has started final transcription and entered
            # ``await asyncio.wait_for(speculative)`` (avoid cancelling too early).
            await asyncio.sleep(0.05)
            handle.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await handle

        await asyncio.sleep(0.05)
        self.assertTrue(
            final_cancelled.is_set(),
            "final_transcription_task must be cancelled when _handle_stop is cancelled",
        )

        # Cleanup dangling tasks
        session._speculative_task.cancel()
        try:
            await session._speculative_task
        except asyncio.CancelledError:
            pass


# ══════════════════════════════════════════════════════════════════════════
#  10. Parallel stop — transcription failure fallback
# ══════════════════════════════════════════════════════════════════════════


class TestParallelStopTranscriptionFallback(unittest.IsolatedAsyncioTestCase):
    """transcribe_final raises → falls back to latest_text."""

    async def test_fallback_to_latest_text(self) -> None:
        session = StreamingSession()
        session.transcription.transcribe_final = AsyncMock(  # type: ignore[assignment]
            side_effect=RuntimeError("Whisper API down"),
        )
        session.transcription._latest_text = "fallback text works here"

        ws = _FakeWS()
        with (
            patch(f"{_MOD}.generate_target_tags", new_callable=AsyncMock, return_value=_tags()),
            patch(f"{_MOD}.select_and_encode", new_callable=AsyncMock, return_value=_result()),
        ):
            await _handle_stop(ws, session)

        final_transcriptions = [
            m for m in ws.sent
            if m.get("type") == "transcription" and m.get("is_final") is True
        ]
        self.assertEqual(len(final_transcriptions), 1)
        self.assertEqual(
            final_transcriptions[0]["text"],
            "fallback text works here",
        )


if __name__ == "__main__":
    unittest.main()
