"""
In-memory WebSocket stubs for TranslationClient unit tests.

Provides the three primitives shared between test_batch2_ws_streaming and
test_batch4_ux_enhancements.  Each test file defines its own ``_patch_ws``
helper (they differ in whether the patch targets the class or an instance),
so that function is intentionally NOT included here.
"""

from __future__ import annotations

from typing import Any


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


def ws_connect_coro(mock_ws: MockWebSocket):
    """Return an async callable that mimics ``websockets.connect()``."""

    async def _connect(*_args: Any, **_kwargs: Any):
        return mock_ws

    return _connect


async def async_chunks(*chunks: bytes):
    """Yield *chunks* as an async iterable (mirrors the app's chunk stream)."""
    for c in chunks:
        yield c
