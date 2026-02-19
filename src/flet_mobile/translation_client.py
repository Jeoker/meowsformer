"""HTTP/WebSocket client for the FastAPI translation backend."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

import httpx
import websockets


class TranslationClient:
    """API-first client wrapper used by the Flet presentation layer."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(30.0)

    async def translate_file(
        self,
        file_name: str,
        audio_bytes: bytes,
        breed: str = "Default",
        output_sr: int = 16000,
    ) -> dict[str, Any]:
        """Call /api/v1/translate with an audio file."""
        url = f"{self.base_url}/api/v1/translate"
        files = {"file": (file_name, audio_bytes, "audio/wav")}
        params = {"breed": breed, "output_sr": output_sr}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, params=params, files=files)
            response.raise_for_status()
            return response.json()

    async def stream_translate(
        self,
        chunks: AsyncIterable[bytes],
        on_event: Callable[[dict[str, Any]], Awaitable[None]],
        breed_preference: str | None = None,
    ) -> None:
        """
        Send PCM chunks to /ws/translate and forward events to callback.

        The callback receives parsed JSON messages from the server, including
        transcription, analysis_preview, result, and error payloads.
        """
        ws_url = self._build_ws_url("/ws/translate")
        async with websockets.connect(ws_url, max_size=5 * 1024 * 1024) as ws:
            await ws.send(
                json.dumps(
                    {"type": "config", "breed_preference": breed_preference or "Default"}
                )
            )

            async for chunk in chunks:
                await ws.send(chunk)

            await ws.send(json.dumps({"type": "stop"}))
            while True:
                message = await ws.recv()
                if not isinstance(message, str):
                    continue
                payload = json.loads(message)
                await on_event(payload)
                if payload.get("type") in {"result", "error"}:
                    break

    def _build_ws_url(self, endpoint: str) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{scheme}://{parsed.netloc}{endpoint}"

