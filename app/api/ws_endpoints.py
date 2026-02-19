"""
Meowsformer — WebSocket Streaming Endpoint
============================================
Real-time audio input → parallel transcription → LLM target-tag
generation → sample matching → cat sound playback.

Endpoint: ``ws://{host}/ws/translate``

Protocol (see ``app/schemas/ws_messages.py`` for full schema):

Client → Server:
    - JSON ``{"type": "config", "breed_preference": "Maine Coon"}``
    - Binary frames — raw PCM 16-bit 16 kHz audio chunks
    - JSON ``{"type": "stop"}``

Server → Client:
    - ``{"type": "transcription", "text": "...", "is_final": false}``
    - ``{"type": "analysis_preview", "emotion": "...", "intent": "..."}``
    - ``{"type": "result", ...}``
    - ``{"type": "error", "detail": "..."}``
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.schemas.ws_messages import (
    StreamingTranslationResult,
    WSAnalysisPreviewMessage,
    WSErrorMessage,
    WSResultMessage,
    WSTranscriptionMessage,
)
from app.services.sample_matcher import load_tagged_samples
from app.services.sound_selection_service import (
    SpeculativeCache,
    generate_target_tags,
    select_and_encode,
)
from app.services.streaming_transcription_service import (
    StreamingTranscriptionSession,
)

router = APIRouter()

# ── Session state ────────────────────────────────────────────────────────


class StreamingSession:
    """Per-connection session holding all state."""

    def __init__(self) -> None:
        self.breed_preference: Optional[str] = None
        self.transcription = StreamingTranscriptionSession()
        self.speculative_cache = SpeculativeCache()
        self._speculative_task: Optional[asyncio.Task] = None

    def reset(self) -> None:
        self.transcription.reset()
        self.speculative_cache.clear()
        if self._speculative_task and not self._speculative_task.done():
            self._speculative_task.cancel()
        self._speculative_task = None


# ── Helpers ──────────────────────────────────────────────────────────────


def _word_count(text: str) -> int:
    """Rough word count for Chinese + English mixed text."""
    # Chinese characters count as individual "words" for this purpose
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_words = len(text.split())
    return chinese_chars + ascii_words


# ── WebSocket endpoint ───────────────────────────────────────────────────


@router.websocket("/ws/translate")
async def ws_translate(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming cat-sound translation."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    # Ensure samples are loaded
    load_tagged_samples()

    session = StreamingSession()

    try:
        while True:
            # Receive next message (text or binary)
            message = await websocket.receive()

            if "text" in message:
                # JSON control message
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await _send_error(websocket, "Invalid JSON")
                    continue

                msg_type = data.get("type", "")

                if msg_type == "config":
                    session.breed_preference = data.get("breed_preference")
                    logger.info("Config: breed_preference={}", session.breed_preference)

                elif msg_type == "stop":
                    logger.info("Stop signal received — processing final result")
                    await _handle_stop(websocket, session)
                    # Reset for next utterance (keep connection open)
                    session.reset()

                else:
                    await _send_error(websocket, f"Unknown message type: {msg_type}")

            elif "bytes" in message:
                # Binary audio chunk
                audio_data = message["bytes"]
                session.transcription.add_chunk(audio_data)

                # Check if we should do an intermediate transcription
                if session.transcription.should_transcribe():
                    text = await session.transcription.transcribe_intermediate()
                    if text:
                        # Send partial transcription
                        await websocket.send_json(
                            WSTranscriptionMessage(
                                text=text, is_final=False
                            ).model_dump()
                        )

                        # Speculative LLM call if enough words
                        if _word_count(text) >= 5 and session._speculative_task is None:
                            logger.info(
                                "Firing speculative LLM (text: '{}'...)",
                                text[:30],
                            )
                            session._speculative_task = asyncio.create_task(
                                _speculative_analysis(session, text, websocket)
                            )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error("WebSocket error: {}", e)
        try:
            await _send_error(websocket, str(e))
        except Exception:
            pass


async def _speculative_analysis(
    session: StreamingSession,
    text: str,
    websocket: WebSocket,
) -> None:
    """Run speculative LLM analysis on partial text."""
    try:
        tags = await generate_target_tags(text)
        session.speculative_cache.store(text, tags)

        # Send analysis preview
        primary_emotion = tags.emotion[0] if tags.emotion else "unknown"
        primary_intent = tags.intent[0] if tags.intent else "unknown"

        await websocket.send_json(
            WSAnalysisPreviewMessage(
                emotion=primary_emotion,
                intent=primary_intent,
            ).model_dump()
        )
    except Exception as e:
        logger.error("Speculative analysis failed: {}", e)


async def _handle_stop(
    websocket: WebSocket,
    session: StreamingSession,
) -> None:
    """Handle the stop signal: final transcription → result."""
    try:
        # Wait for any running speculative task
        if session._speculative_task and not session._speculative_task.done():
            try:
                await asyncio.wait_for(session._speculative_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Speculative task timed out; proceeding with fresh LLM call")
                session._speculative_task.cancel()

        # Final transcription
        final_text = await session.transcription.transcribe_final()

        # Send final transcription
        await websocket.send_json(
            WSTranscriptionMessage(text=final_text, is_final=True).model_dump()
        )

        if not final_text:
            await _send_error(websocket, "No speech detected")
            return

        # Decide whether to reuse cached tags or call LLM again
        if session.speculative_cache.is_similar(final_text):
            logger.info("Reusing cached LLM result (text similar)")
            target_tags = session.speculative_cache.get()
        else:
            logger.info("Final text differs — calling LLM again")
            target_tags = await generate_target_tags(final_text)

        if target_tags is None:
            await _send_error(websocket, "Failed to generate target tags")
            return

        # Find best match and encode
        result = await select_and_encode(
            target_tags=target_tags,
            breed_preference=session.breed_preference,
        )

        if result is None:
            await _send_error(websocket, "No matching cat sound found")
            return

        # Fill in the transcription
        result.transcription = final_text

        # Send final result
        await websocket.send_json(
            WSResultMessage(
                transcription=final_text,
                selected_category=result.selected_sample,
                audio_base64=result.audio_base64,
                reasoning=result.reasoning,
            ).model_dump()
        )

        logger.success(
            "Result sent: sample={}, score={:.3f}",
            result.selected_sample.sample_id,
            result.selected_sample.match_score,
        )

    except Exception as e:
        logger.error("Error in stop handler: {}", e)
        await _send_error(websocket, f"Processing error: {e}")


async def _send_error(websocket: WebSocket, detail: str) -> None:
    """Send an error message to the client."""
    try:
        await websocket.send_json(WSErrorMessage(detail=detail).model_dump())
    except Exception:
        pass
