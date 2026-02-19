"""
Meowsformer — Streaming Transcription Service
===============================================
Accumulates audio chunks into a growing buffer and periodically sends
them to the OpenAI Whisper API for transcription.

Design:
- Audio chunks arrive ~200ms apart over WebSocket.
- Every ~2-3s (or on significant buffer growth), we re-transcribe
  the accumulated buffer (Whisper re-processes the whole thing).
- On ``stop``, one final Whisper call produces the definitive text.
- Acceptable for utterances < 60s.
"""

from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger
from openai import OpenAI

from app.core.config import settings

# OpenAI client
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


# ── Streaming Session ────────────────────────────────────────────────────


class StreamingTranscriptionSession:
    """Manages audio buffer and incremental Whisper transcription."""

    # Minimum interval between intermediate transcription calls (seconds)
    MIN_TRANSCRIPTION_INTERVAL = 2.5
    # Minimum buffer size before first transcription attempt (bytes)
    MIN_BUFFER_SIZE = 16000 * 2 * 1  # ~1s of 16kHz 16-bit mono

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._chunks: list[bytes] = []
        self._total_bytes = 0
        self._last_transcription_time = 0.0
        self._latest_text = ""

    @property
    def buffer_size(self) -> int:
        return self._total_bytes

    @property
    def latest_text(self) -> str:
        return self._latest_text

    def add_chunk(self, data: bytes) -> None:
        """Append an audio chunk to the buffer."""
        self._chunks.append(data)
        self._total_bytes += len(data)

    def should_transcribe(self) -> bool:
        """Check if we should fire an intermediate transcription."""
        if self._total_bytes < self.MIN_BUFFER_SIZE:
            return False
        elapsed = time.monotonic() - self._last_transcription_time
        return elapsed >= self.MIN_TRANSCRIPTION_INTERVAL

    async def transcribe_intermediate(self) -> Optional[str]:
        """Transcribe the accumulated buffer (intermediate call).

        Returns the transcription text, or None if skipped/failed.
        """
        if not self._chunks:
            return None

        try:
            text = await self._call_whisper()
            self._latest_text = text
            self._last_transcription_time = time.monotonic()
            return text
        except Exception as e:
            logger.error("Intermediate transcription failed: {}", e)
            return None

    async def transcribe_final(self) -> str:
        """Transcribe the complete buffer (final call after user stops).

        Returns the final transcription text.
        """
        if not self._chunks:
            return ""

        try:
            text = await self._call_whisper()
            self._latest_text = text
            return text
        except Exception as e:
            logger.error("Final transcription failed: {}", e)
            return self._latest_text  # Fall back to last intermediate result

    async def _call_whisper(self) -> str:
        """Send the full buffer to Whisper API."""
        # Combine all chunks into a single buffer
        combined = b"".join(self._chunks)

        # Write to temporary WAV file (Whisper needs a file)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Interpret as 16-bit PCM
            audio_array = np.frombuffer(combined, dtype=np.int16).astype(np.float32) / 32768.0

            sf.write(
                str(tmp_path),
                audio_array,
                self.sample_rate,
                format="WAV",
                subtype="PCM_16",
            )

            client = _get_client()
            with open(tmp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                )

            return transcription.strip() if isinstance(transcription, str) else str(transcription).strip()

        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def get_buffer_as_wav_bytes(self) -> bytes:
        """Return the full buffer as WAV bytes (for debug / fallback)."""
        combined = b"".join(self._chunks)
        audio_array = np.frombuffer(combined, dtype=np.int16).astype(np.float32) / 32768.0

        buf = io.BytesIO()
        sf.write(buf, audio_array, self.sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()

    def reset(self) -> None:
        """Clear the buffer and state."""
        self._chunks.clear()
        self._total_bytes = 0
        self._last_transcription_time = 0.0
        self._latest_text = ""
