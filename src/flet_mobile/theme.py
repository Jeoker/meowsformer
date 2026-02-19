"""Design tokens for the Meowsformer Flet mobile UI."""

from __future__ import annotations

import flet as ft

CREAM_BG = "#FFFDD0"
OAT_BG = "#FAF3E0"
AMBER = "#FFBF00"
PAW_PINK = "#FFD1DC"
FOREST_GREEN = "#228B22"
TEXT_DARK = "#3B2F2F"
TEXT_MUTED = "#6E6A65"


def soft_card_style() -> dict:
    return {
        "border_radius": 28,
        "padding": 16,
        "shadow": ft.BoxShadow(
            spread_radius=0,
            blur_radius=24,
            color=ft.Colors.with_opacity(0.16, ft.Colors.BLACK),
            offset=ft.Offset(0, 8),
        ),
    }

