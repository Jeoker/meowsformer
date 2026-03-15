"""HTTP/WebSocket client for the FastAPI translation backend."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

import httpx
import websockets


WS_CONNECT_TIMEOUT = 5.0


class WebSocketConnectionError(Exception):
    """Raised when the WebSocket connection cannot be established."""


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
        on_state_change: Callable[[str], None] | None = None,
    ) -> None:
        """
        Send PCM chunks to /ws/translate and forward server events concurrently.

        ``_sender`` pushes PCM frames as they arrive from the audio recorder,
        while ``_receiver`` forwards every server message to *on_event* in
        real-time.  Both coroutines run simultaneously via `asyncio.TaskGroup`.

        *on_state_change* fires with "connecting", "connected", or
        "disconnected" to let the UI track the WebSocket lifecycle.
        """
        ws_url = self._build_ws_url("/ws/translate")

        if on_state_change:
            on_state_change("connecting")

        try:
            ws = await asyncio.wait_for(
                websockets.connect(ws_url, max_size=5 * 1024 * 1024),
                timeout=WS_CONNECT_TIMEOUT,
            )
        except (asyncio.TimeoutError, OSError, websockets.WebSocketException) as exc:
            if on_state_change:
                on_state_change("disconnected")
            raise WebSocketConnectionError(
                f"无法建立 WebSocket 连接 ({type(exc).__name__}): {exc}"
            ) from exc

        try:
            if on_state_change:
                on_state_change("connected")

            await ws.send(
                json.dumps(
                    {"type": "config", "breed_preference": breed_preference or "Default"}
                )
            )

            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._sender(ws, chunks))
                tg.create_task(self._receiver(ws, on_event))
        finally:
            if on_state_change:
                on_state_change("disconnected")
            await ws.close()

    # ------------------------------------------------------------------
    # Internal coroutines
    # ------------------------------------------------------------------

    @staticmethod
    async def _sender(
        ws: websockets.WebSocketClientProtocol,
        chunks: AsyncIterable[bytes],
    ) -> None:
        """Push PCM chunks then send the stop sentinel."""
        async for chunk in chunks:
            await ws.send(chunk)
        await ws.send(json.dumps({"type": "stop"}))

    @staticmethod
    async def _receiver(
        ws: websockets.WebSocketClientProtocol,
        on_event: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Forward server JSON messages until a terminal event arrives."""
        async for message in ws:
            if not isinstance(message, str):
                continue
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                await on_event({"type": "error", "detail": f"畸形 JSON: {message[:120]}"})
                break
            await on_event(payload)
            if payload.get("type") in {"result", "error"}:
                break

    def _build_ws_url(self, endpoint: str) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{scheme}://{parsed.netloc}{endpoint}"

