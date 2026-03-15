"""
Lightweight flet control mocks and ft.Page stub for Meowsformer unit tests.

Usage
-----
Call ``install_flet_mock()`` **before** importing any ``src.flet_mobile``
module so that ``import flet as ft`` inside those modules picks up the mock.
Use ``BaseMockPage`` as the ``page`` argument when exercising ``meowsformer_ui``.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock


# ══════════════════════════════════════════════════════════════════════════════
# Fake flet controls
# ══════════════════════════════════════════════════════════════════════════════

class _Ctrl:
    """Generic mock flet control: stores all kwargs as attributes."""

    def __init__(self, *_args: Any, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update(self) -> None:
        pass


class _TextCtrl(_Ctrl):
    """Mock ft.Text — tracks .value for assertion."""

    def __init__(self, value: str = "", **kwargs: Any) -> None:
        self.value = value
        super().__init__(**kwargs)


class _ListCtrl(_Ctrl):
    """Mock ft.Column / ft.Row / ft.ListView — tracks .controls list."""

    def __init__(self, controls: list | None = None, **kwargs: Any) -> None:
        self.controls: list = list(controls) if controls else []
        super().__init__(**kwargs)


def install_flet_mock() -> types.ModuleType:
    """Build a unified fake ``flet`` module and install it in sys.modules.

    Superset of all per-file ``_install_flet_mock`` variants across
    test_batch2, test_batch3, and test_batch4 — safe to use for all three.
    """
    ft = types.ModuleType("flet")

    ft.Text = _TextCtrl  # type: ignore[attr-defined]
    ft.Column = _ListCtrl  # type: ignore[attr-defined]
    ft.Row = _ListCtrl  # type: ignore[attr-defined]
    ft.ListView = _ListCtrl  # type: ignore[attr-defined]
    ft.ExpansionTile = _ListCtrl  # type: ignore[attr-defined]
    ft.Container = _Ctrl  # type: ignore[attr-defined]
    ft.Chip = _Ctrl  # type: ignore[attr-defined]
    ft.Icon = _Ctrl  # type: ignore[attr-defined]
    ft.ProgressBar = _Ctrl  # type: ignore[attr-defined]
    ft.Slider = _Ctrl  # type: ignore[attr-defined]
    ft.SegmentedButton = _Ctrl  # type: ignore[attr-defined]
    ft.Segment = _Ctrl  # type: ignore[attr-defined]
    ft.TextField = _Ctrl  # type: ignore[attr-defined]
    ft.FilledButton = _Ctrl  # type: ignore[attr-defined]
    ft.Theme = _Ctrl  # type: ignore[attr-defined]
    ft.Dropdown = _Ctrl  # type: ignore[attr-defined]
    ft.LinearGradient = _Ctrl  # type: ignore[attr-defined]
    ft.ButtonStyle = lambda **_kw: MagicMock()  # type: ignore[attr-defined]
    ft.SnackBar = _Ctrl  # type: ignore[attr-defined]
    ft.Audio = _Ctrl  # type: ignore[attr-defined]
    ft.TileAffinity = MagicMock()  # type: ignore[attr-defined]
    ft.dropdown = MagicMock()  # type: ignore[attr-defined]
    ft.dropdown.Option = _Ctrl
    ft.run = MagicMock()  # type: ignore[attr-defined]

    for attr in (
        "Colors", "Icons", "FontWeight", "ScrollMode",
        "MainAxisAlignment", "CrossAxisAlignment", "AppView",
        "ControlEvent", "border", "padding", "Padding", "Border",
    ):
        setattr(ft, attr, MagicMock())

    ft.Alignment = lambda *_a, **_kw: MagicMock()  # type: ignore[attr-defined]
    ft.Scale = lambda *_a, **_kw: MagicMock()  # type: ignore[attr-defined]
    ft.BoxShadow = lambda **_kw: MagicMock()  # type: ignore[attr-defined]
    ft.Offset = lambda *_a, **_kw: MagicMock()  # type: ignore[attr-defined]

    sys.modules["flet"] = ft
    return ft


# ══════════════════════════════════════════════════════════════════════════════
# Unified ft.Page mock
# ══════════════════════════════════════════════════════════════════════════════

class BaseMockPage:
    """Unified ft.Page mock for all Meowsformer unit test suites.

    Superset of the three per-file ``_MockPage`` variants:

    * ``_added``          — controls passed to ``page.add()``
    * ``_opened``         — controls passed to ``page.open()``   (batch2 / batch4)
    * ``_run_task_calls`` — coroutines passed to ``page.run_task()``  (batch4)
    * ``on_disconnect``   — writable attribute for disconnect handler (batch3 / batch4)

    The richer behaviours are additive: recording ``_opened`` and
    ``_run_task_calls`` does not affect test suites that do not assert on them.
    """

    def __init__(self) -> None:
        self.title = ""
        self.bgcolor = ""
        self.padding = 0
        self.scroll = None
        self.theme = None
        self.on_disconnect = None
        self.overlay: list = []
        self._added: list = []
        self._opened: list = []
        self._run_task_calls: list = []

    def add(self, *controls: Any) -> None:
        self._added.extend(controls)

    def open(self, control: Any) -> None:
        self._opened.append(control)

    def update(self) -> None:
        pass

    def run_task(self, coro_fn: Any) -> None:
        self._run_task_calls.append(coro_fn)
