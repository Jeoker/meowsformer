"""Meowsformer mobile UI built with Flet (API-first)."""

from __future__ import annotations

import asyncio
import datetime as dt
import io
import os
import platform
import wave
from typing import Any

import flet as ft

from .audio_recorder import AudioRecorder
from .bioacoustic_player import BioacousticPlayer
from .theme import AMBER, CREAM_BG, FOREST_GREEN, OAT_BG, PAW_PINK, TEXT_DARK, TEXT_MUTED
from .theme import soft_card_style
from .translation_client import TranslationClient

BREEDS = ["Maine Coon", "Ragdoll", "Domestic Shorthair"]


def pcm16_to_wav_bytes(raw_pcm: bytes, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw_pcm)
    return buffer.getvalue()


async def meowsformer_ui(page: ft.Page) -> None:
    page.title = "Meowsformer Mobile"
    page.bgcolor = CREAM_BG
    page.padding = 14
    page.scroll = ft.ScrollMode.AUTO
    page.theme = ft.Theme(color_scheme_seed=AMBER, use_material3=True)

    client = TranslationClient()
    recorder = AudioRecorder()
    player = BioacousticPlayer(page=page)

    selected_breed = BREEDS[0]
    current_sound_id = "purr_happy_01"

    analysis_status = ft.Text("就绪，点击按钮开始录音。", color=TEXT_MUTED, size=13)
    speculative_bar = ft.ProgressBar(width=320, value=0.0, color=AMBER, bgcolor=PAW_PINK)

    cat_avatar = ft.Container(
        width=160,
        height=160,
        border_radius=80,
        bgcolor=ft.Colors.with_opacity(0.8, ft.Colors.WHITE),
        alignment=ft.Alignment(0, 0),
        content=ft.Text("😺", size=76),
        animate_scale=300,
        animate_opacity=300,
    )

    waveform_bars = [
        ft.Container(
            width=4,
            height=10,
            bgcolor=ft.Colors.with_opacity(0.45, AMBER),
            border_radius=6,
        )
        for _ in range(48)
    ]
    waveform = ft.Container(
        height=86,
        bgcolor=ft.Colors.with_opacity(0.35, ft.Colors.WHITE),
        border_radius=16,
        padding=ft.padding.symmetric(horizontal=8, vertical=8),
        content=ft.Row(
            controls=waveform_bars,
            spacing=2,
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.END,
        ),
    )
    live_transcription = ft.Text(
        "实时转录内容将显示在这里...",
        color=TEXT_MUTED,
        size=14,
        italic=True,
    )
    tags_wrap = ft.Row(
        controls=[],
        spacing=8,
        run_spacing=8,
        wrap=True,
    )

    rag_bubble = ft.Container(
        visible=False,
        bgcolor=ft.Colors.with_opacity(0.75, ft.Colors.WHITE),
        border=ft.border.all(1, FOREST_GREEN),
        border_radius=20,
        padding=12,
        content=ft.Row(
            [
                ft.Icon(ft.Icons.MENU_BOOK_ROUNDED, color=FOREST_GREEN, size=18),
                ft.Text("Lund University 研究背书将在翻译完成后显示。", color=FOREST_GREEN),
            ]
        ),
    )

    tempo_slider = ft.Slider(min=0.8, max=1.4, value=1.0, divisions=12, label="Tempo: {value}x")
    pitch_slider = ft.Slider(min=0.8, max=1.4, value=1.0, divisions=12, label="Pitch: {value}x")
    player_status = ft.Text("等待翻译结果...", size=13, color=TEXT_MUTED)
    history = ft.ListView(spacing=10, auto_scroll=False, height=210)
    cat_profile_panel = ft.ExpansionTile(
        title=ft.Text("我的猫咪"),
        leading=ft.Icon(ft.Icons.PETS_ROUNDED, color=AMBER),
        controls=[
            ft.TextField(label="昵称", hint_text="比如：Milo"),
            ft.TextField(label="年龄", hint_text="比如：2"),
            ft.Dropdown(
                label="品种偏好",
                options=[ft.dropdown.Option(name) for name in BREEDS],
                value=selected_breed,
                on_change=lambda e: _set_breed(e.data),
            ),
        ],
    )

    async def update_waveform_loop() -> None:
        while recorder.is_recording:
            points = recorder.snapshot_waveform(len(waveform_bars))
            for i, value in enumerate(points):
                bar_height = max(6, min(64, int(6 + value * 58)))
                waveform_bars[i].height = bar_height
                waveform_bars[i].bgcolor = ft.Colors.with_opacity(
                    min(0.9, 0.25 + value * 0.75),
                    AMBER,
                )
            waveform.update()
            await asyncio.sleep(0.08)

    async def breathing_glow_loop() -> None:
        while recorder.is_recording:
            record_button.scale = ft.Scale(1.04)
            record_button.shadow = ft.BoxShadow(
                blur_radius=30,
                color=ft.Colors.with_opacity(0.42, AMBER),
                offset=ft.Offset(0, 0),
            )
            record_button.update()
            await asyncio.sleep(0.55)
            record_button.scale = ft.Scale(1.0)
            record_button.shadow = ft.BoxShadow(
                blur_radius=12,
                color=ft.Colors.with_opacity(0.22, AMBER),
                offset=ft.Offset(0, 2),
            )
            record_button.update()
            await asyncio.sleep(0.55)

    def _set_breed(value: str | None) -> None:
        nonlocal selected_breed
        if value:
            selected_breed = value

    def update_tags(response: dict[str, Any]) -> None:
        chips = [
            ft.Chip(label=ft.Text(f"Emotion: {response.get('emotion_category', '-')}", size=12)),
            ft.Chip(label=ft.Text(f"Intent: {response.get('sound_id', '-')}", size=12)),
            ft.Chip(label=ft.Text(f"Acoustic: pitch {response.get('pitch_adjust', 1.0)}", size=12)),
            ft.Chip(label=ft.Text(f"Social: owner_present", size=12)),
            ft.Chip(label=ft.Text(f"Breed: {selected_breed}", size=12)),
        ]
        tags_wrap.controls = chips
        tags_wrap.update()

    def append_history(response: dict[str, Any]) -> None:
        now = dt.datetime.now().strftime("%H:%M:%S")
        item = ft.Container(
            bgcolor=ft.Colors.WHITE,
            border_radius=20,
            padding=12,
            content=ft.Column(
                [
                    ft.Text(response.get("human_interpretation", "无转录"), color=TEXT_DARK),
                    ft.Text(
                        f"意图: {response.get('emotion_category', '-')}, 时间: {now}",
                        color=TEXT_MUTED,
                        size=12,
                    ),
                ],
                spacing=3,
            ),
        )
        history.controls.insert(0, item)
        history.update()

    async def request_translation(raw_pcm: bytes) -> None:
        nonlocal current_sound_id
        analysis_status.value = "意图分析中..."
        speculative_bar.value = 0.35
        page.update()

        wav_bytes = pcm16_to_wav_bytes(raw_pcm, sample_rate=16000)
        response = await client.translate_file(
            file_name="recording.wav",
            audio_bytes=wav_bytes,
            breed=selected_breed,
            output_sr=16000,
        )

        speculative_bar.value = 1.0
        live_transcription.value = response.get("human_interpretation", "未识别文本")
        analysis_status.value = "翻译完成，已生成猫语音频。"
        update_tags(response)
        append_history(response)

        rag_text = (
            response.get("preview_description", {}) or {}
        ).get("summary", "匹配到高置信度生物声学样本。")
        rag_bubble.content = ft.Row(
            [
                ft.Icon(ft.Icons.MENU_BOOK_ROUNDED, color=FOREST_GREEN, size=18),
                ft.Text(rag_text, color=FOREST_GREEN),
            ]
        )
        rag_bubble.visible = True

        metadata = response.get("synthesis_metadata") or {}
        current_sound_id = metadata.get("matched_sample_id") or response.get("sound_id", current_sound_id)
        player_status.value = f"已就绪: {current_sound_id}"
        page.update()

    async def on_record_toggle(_e: ft.ControlEvent) -> None:
        if not recorder.is_recording:
            analysis_status.value = "录音中..."
            speculative_bar.value = None
            live_transcription.value = "正在监听，请说话..."
            cat_avatar.scale = ft.Scale(1.06)
            recorder.start()
            page.run_task(update_waveform_loop)
            page.run_task(breathing_glow_loop)
            page.update()
            return

        raw_pcm = recorder.stop()
        cat_avatar.scale = ft.Scale(1.0)
        speculative_bar.value = 0.0
        record_button.scale = ft.Scale(1.0)
        record_button.shadow = None
        page.update()

        if not raw_pcm:
            analysis_status.value = "未采集到音频，请重试。"
            page.update()
            return

        try:
            await request_translation(raw_pcm)
        except Exception as exc:  # pragma: no cover - runtime/network boundary
            analysis_status.value = f"请求失败: {exc}"
            speculative_bar.value = 0.0
            page.update()

    async def on_play_processed(_e: ft.ControlEvent) -> None:
        try:
            await player.play_sound_id(
                sound_id=current_sound_id,
                pitch_factor=float(pitch_slider.value),
                tempo_factor=float(tempo_slider.value),
            )
            player_status.value = f"播放中: {current_sound_id}"
        except Exception as exc:  # pragma: no cover - runtime/audio boundary
            player_status.value = f"播放失败: {exc}"
        page.update()

    breed_selector = ft.SegmentedButton(
        segments=[ft.Segment(text=name, value=name) for name in BREEDS],
        selected={selected_breed},
        show_selected_icon=False,
        style=ft.ButtonStyle(bgcolor=ft.Colors.with_opacity(0.55, ft.Colors.WHITE)),
        on_change=lambda e: _set_breed(next(iter(e.control.selected), selected_breed)),
    )

    record_button = ft.Container(
        width=96,
        height=96,
        border_radius=48,
        bgcolor=AMBER,
        ink=True,
        alignment=ft.Alignment(0, 0),
        content=ft.Icon(ft.Icons.MIC_ROUNDED, size=44, color=ft.Colors.WHITE),
        on_click=on_record_toggle,
        animate_scale=300,
    )

    bridge_card = ft.Container(
        gradient=ft.LinearGradient(
            colors=[OAT_BG, ft.Colors.WHITE],
            begin=ft.Alignment(-1, -1),
            end=ft.Alignment(1, 1),
        ),
        content=ft.Column(
            [
                ft.Text("The Bridge", color=TEXT_DARK, size=18, weight=ft.FontWeight.W_600),
                cat_avatar,
                breed_selector,
                waveform,
                live_transcription,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
        ),
        **soft_card_style(),
    )

    lab_card = ft.Container(
        bgcolor=ft.Colors.WHITE,
        content=ft.Column(
            [
                ft.Text("The Lab", color=TEXT_DARK, size=18, weight=ft.FontWeight.W_600),
                analysis_status,
                speculative_bar,
                tags_wrap,
                rag_bubble,
            ],
            spacing=10,
        ),
        **soft_card_style(),
    )

    output_card = ft.Container(
        bgcolor=ft.Colors.WHITE,
        content=ft.Column(
            [
                ft.Text("The Output", color=TEXT_DARK, size=18, weight=ft.FontWeight.W_600),
                ft.Text("韵律控制器", color=TEXT_DARK, weight=ft.FontWeight.W_500),
                tempo_slider,
                pitch_slider,
                ft.FilledButton("播放处理后猫语", icon=ft.Icons.VOLUME_UP, on_click=on_play_processed),
                player_status,
            ],
            spacing=8,
        ),
        **soft_card_style(),
    )

    library_card = ft.Container(
        bgcolor=ft.Colors.WHITE,
        content=ft.Column(
            [
                ft.Text("The Library", color=TEXT_DARK, size=18, weight=ft.FontWeight.W_600),
                cat_profile_panel,
                history,
            ],
            spacing=10,
        ),
        **soft_card_style(),
    )

    page.add(
        ft.Column(
            controls=[
                bridge_card,
                ft.Container(content=record_button, alignment=ft.Alignment(0, 0)),
                lab_card,
                output_card,
                library_card,
            ],
            spacing=12,
        )
    )


def main() -> None:
    # WSL/Linux often lacks desktop runtime libs (e.g. libsecret),
    # so default to browser mode for better out-of-box startup.
    explicit_view = os.getenv("MEOWSFORMER_FLET_VIEW", "").strip().lower()
    if explicit_view == "desktop":
        view = ft.AppView.FLET_APP
    elif explicit_view == "browser":
        view = ft.AppView.WEB_BROWSER
    else:
        is_wsl = "microsoft" in platform.release().lower()
        view = ft.AppView.WEB_BROWSER if is_wsl else ft.AppView.FLET_APP

    host = os.getenv("MEOWSFORMER_FLET_HOST", "0.0.0.0")
    port = int(os.getenv("MEOWSFORMER_FLET_PORT", "8550"))

    if view == ft.AppView.WEB_BROWSER:
        print(f"Meowsformer Flet UI starting at: http://127.0.0.1:{port}")
        print("If browser auto-open fails in WSL, open the URL manually.")

    ft.run(meowsformer_ui, view=view, host=host, port=port)


if __name__ == "__main__":
    main()

