"""Meowsformer mobile UI built with Flet (API-first)."""

from __future__ import annotations

import asyncio
import datetime as dt
import io
import os
import platform
import wave
from collections.abc import AsyncGenerator
from typing import Any

import flet as ft

from .audio_recorder import AudioRecorder
from .bioacoustic_player import BioacousticPlayer
from .theme import AMBER, CREAM_BG, FOREST_GREEN, OAT_BG, PAW_PINK, TEXT_DARK, TEXT_MUTED
from .theme import soft_card_style
from .translation_client import TranslationClient, WebSocketConnectionError

BREEDS = ["Maine Coon", "Ragdoll", "Domestic Shorthair"]
TAG_DIMENSIONS = ("emotion", "intent", "acoustic", "social_context", "breed_voice")


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

    def _on_page_error(e: ft.ControlEvent) -> None:
        page.overlay[:] = [c for c in page.overlay if not isinstance(c, ft.SnackBar)]
        page.overlay.append(
            ft.SnackBar(
                content=ft.Text(str(e.data), color=ft.Colors.WHITE, size=13),
                bgcolor=ft.Colors.RED_700,
                duration=6000,
                open=True,
            )
        )
        page.update()

    page.on_error = _on_page_error

    client = TranslationClient()
    recorder = AudioRecorder()
    player = BioacousticPlayer(page=page)

    def _on_disconnect(_e: ft.ControlEvent) -> None:
        player.dispose()

    page.on_disconnect = _on_disconnect

    selected_breed = BREEDS[0]
    current_sound_id = "purr_happy_01"
    translate_mode: str = "rest"
    _streaming_task: asyncio.Task[None] | None = None
    _chunk_queue: asyncio.Queue[bytes | None] | None = None

    def _show_snackbar(message: str, *, is_error: bool = True) -> None:
        color = ft.Colors.RED_700 if is_error else AMBER
        # Remove stale SnackBars to prevent overlay buildup
        page.overlay[:] = [c for c in page.overlay if not isinstance(c, ft.SnackBar)]
        page.overlay.append(
            ft.SnackBar(
                content=ft.Text(message, color=ft.Colors.WHITE),
                bgcolor=color,
                duration=3000,
                open=True,
            )
        )

    ws_status_chip = ft.Chip(
        label=ft.Text("已断开", size=11),
        bgcolor=PAW_PINK,
        padding=ft.Padding.symmetric(horizontal=4, vertical=0),
        leading=ft.Icon(ft.Icons.WIFI_OFF_ROUNDED, size=14),
    )

    def _update_ws_status(state: str) -> None:
        label_map = {
            "connecting": ("连接中...", AMBER, ft.Icons.WIFI_ROUNDED),
            "connected": ("已连接", FOREST_GREEN, ft.Icons.WIFI_ROUNDED),
            "disconnected": ("已断开", PAW_PINK, ft.Icons.WIFI_OFF_ROUNDED),
            "reconnecting": ("重连中...", AMBER, ft.Icons.WIFI_ROUNDED),  # reserved for future auto-reconnect
        }
        text, color, icon = label_map.get(state, ("已断开", PAW_PINK, ft.Icons.WIFI_OFF_ROUNDED))
        ws_status_chip.label = ft.Text(text, size=11)
        ws_status_chip.bgcolor = color
        ws_status_chip.leading = ft.Icon(icon, size=14)
        ws_status_chip.update()

    recording_timer_text = ft.Text("00:00", size=14, color=AMBER, weight=ft.FontWeight.W_600, visible=False)

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
        padding=ft.Padding.symmetric(horizontal=8, vertical=8),
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
        border=ft.Border.all(1, FOREST_GREEN),
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
                on_select=lambda e: _set_breed(e.control.value),
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

    async def recording_timer_loop() -> None:
        recording_timer_text.visible = True
        recording_timer_text.value = "00:00"
        recording_timer_text.update()
        start = asyncio.get_event_loop().time()
        while recorder.is_recording:
            await asyncio.sleep(1.0)
            elapsed = int(asyncio.get_event_loop().time() - start)
            mins, secs = divmod(elapsed, 60)
            recording_timer_text.value = f"{mins:02d}:{secs:02d}"
            recording_timer_text.update()

    def _set_breed(value: str | None) -> None:
        nonlocal selected_breed
        if value:
            selected_breed = value

    def update_tags(response: dict[str, Any]) -> None:
        chips: list[ft.Chip] = []
        selected = response.get("selected_category") or {}
        tags = selected.get("tags") or {}
        for dim in TAG_DIMENSIONS:
            for tag in tags.get(dim, []):
                chips.append(ft.Chip(label=ft.Text(f"{dim}: {tag}", size=12)))
        if not chips:
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
        selected = response.get("selected_category") or {}
        tags = selected.get("tags") or {}
        score = selected.get("match_score")

        transcription = (
            response.get("transcription")
            or response.get("human_interpretation")
            or "无转录"
        )

        tag_parts: list[str] = []
        for dim in ("emotion", "intent"):
            for tag in tags.get(dim, []):
                tag_parts.append(tag)
        if not tag_parts:
            tag_parts.append(response.get("emotion_category", "-"))

        subtitle = f"标签: {', '.join(tag_parts)}"
        if score is not None:
            subtitle += f" | 匹配: {score:.0%}"
        subtitle += f" | {now}"

        dim_rows: list[ft.Control] = []
        for dim in TAG_DIMENSIONS:
            dim_tags = tags.get(dim, [])
            dim_rows.append(
                ft.Row(
                    [
                        ft.Text(dim, weight=ft.FontWeight.W_600, size=12, color=TEXT_DARK, width=110),
                        ft.Text(", ".join(dim_tags) if dim_tags else "-", size=12, color=TEXT_MUTED),
                    ],
                    spacing=6,
                )
            )

        detail_panel = ft.ExpansionTile(
            title=ft.Text("查看完整 5 维标签", size=12, color=FOREST_GREEN),
            affinity=ft.TileAffinity.LEADING,
            expanded=False,
            controls=dim_rows,
        )

        item = ft.Container(
            bgcolor=ft.Colors.WHITE,
            border_radius=20,
            padding=12,
            content=ft.Column(
                [
                    ft.Text(transcription, color=TEXT_DARK),
                    ft.Text(subtitle, color=TEXT_MUTED, size=12),
                    detail_panel,
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

        audio_b64 = response.get("audio_base64")
        if audio_b64:
            await player.play_from_base64(audio_b64)

        metadata = response.get("synthesis_metadata") or {}
        current_sound_id = metadata.get("matched_sample_id") or response.get("sound_id", current_sound_id)
        player_status.value = f"已就绪: {current_sound_id}"
        page.update()

    async def _chunk_generator() -> AsyncGenerator[bytes, None]:
        """Yield PCM chunks from the queue until a None sentinel is received."""
        queue = _chunk_queue
        if queue is None:
            raise RuntimeError("_chunk_queue must be initialised before streaming")
        while True:
            chunk = await queue.get()
            if chunk is None:
                return
            yield chunk

    async def on_ws_event(payload: dict[str, Any]) -> None:
        """Handle incoming WebSocket server messages during streaming."""
        nonlocal current_sound_id
        msg_type = payload.get("type")

        if msg_type == "transcription":
            live_transcription.value = payload.get("text", "")
            live_transcription.update()

        elif msg_type == "analysis_preview":
            preview_chips: list[ft.Chip] = []
            for dim in ("emotion", "intent"):
                for tag in payload.get(dim, []):
                    preview_chips.append(ft.Chip(label=ft.Text(f"{dim}: {tag}", size=12)))
            if preview_chips:
                tags_wrap.controls = preview_chips
                tags_wrap.update()
            speculative_bar.value = 0.65
            analysis_status.value = "推测性分析就绪..."
            speculative_bar.update()
            analysis_status.update()

        elif msg_type == "result":
            speculative_bar.value = 1.0
            analysis_status.value = "翻译完成，已生成猫语音频。"
            live_transcription.value = payload.get("transcription", live_transcription.value)
            update_tags(payload)
            append_history(payload)

            audio_b64 = payload.get("audio_base64")
            if audio_b64:
                await player.play_from_base64(audio_b64)

            selected = payload.get("selected_category") or {}
            current_sound_id = selected.get("sample_id", current_sound_id)
            player_status.value = f"已就绪: {current_sound_id}"
            reasoning = payload.get("reasoning", "")
            if reasoning:
                rag_bubble.content = ft.Row(
                    [
                        ft.Icon(ft.Icons.MENU_BOOK_ROUNDED, color=FOREST_GREEN, size=18),
                        ft.Text(reasoning, color=FOREST_GREEN),
                    ]
                )
                rag_bubble.visible = True
            page.update()

        elif msg_type == "error":
            _show_snackbar(f"服务端错误: {payload.get('detail', '未知错误')}")
            speculative_bar.value = 0.0
            page.update()

    async def _run_streaming_session() -> None:
        """Run the full streaming translate lifecycle."""
        try:
            await client.stream_translate(
                chunks=_chunk_generator(),
                on_event=on_ws_event,
                breed_preference=selected_breed,
                on_state_change=_update_ws_status,
            )
        except WebSocketConnectionError:
            raise
        except Exception as exc:  # pragma: no cover - runtime/network boundary
            _show_snackbar(f"Streaming 失败: {exc}")
            speculative_bar.value = 0.0
            page.update()

    def _fallback_to_rest() -> None:
        """Switch to REST mode and update the UI selector."""
        nonlocal translate_mode
        translate_mode = "rest"
        mode_selector.selected = ["rest"]
        mode_selector.update()
        _show_snackbar("WebSocket 不可用，已切换为文件上传模式", is_error=False)

    async def on_record_toggle(_e: ft.ControlEvent) -> None:
        nonlocal _streaming_task, _chunk_queue

        if not recorder.is_recording:
            recording_mode = translate_mode  # snapshot before recording starts

            analysis_status.value = "录音中..."
            speculative_bar.value = None
            live_transcription.value = "正在监听，请说话..."
            cat_avatar.scale = ft.Scale(1.06)

            if recording_mode == "streaming":
                _chunk_queue = asyncio.Queue()
                loop = asyncio.get_running_loop()
                recorder.on_chunk = lambda data: loop.call_soon_threadsafe(
                    _chunk_queue.put_nowait, data
                )
                recorder.start()
                _streaming_task = asyncio.create_task(_run_streaming_session())
            else:
                recorder.on_chunk = None
                recorder.start()

            page.run_task(update_waveform_loop)
            page.run_task(breathing_glow_loop)
            page.run_task(recording_timer_loop)
            page.update()
            return

        raw_pcm = recorder.stop()
        recording_timer_text.visible = False
        recording_timer_text.update()
        cat_avatar.scale = ft.Scale(1.0)
        speculative_bar.value = 0.0
        record_button.scale = ft.Scale(1.0)
        record_button.shadow = None
        page.update()

        if _streaming_task is not None:
            if _chunk_queue is not None:
                _chunk_queue.put_nowait(None)
            try:
                await asyncio.wait_for(_streaming_task, timeout=15.0)
            except asyncio.TimeoutError:
                _streaming_task.cancel()
                _show_snackbar("Streaming 超时，请重试。")
                page.update()
            except WebSocketConnectionError:
                _fallback_to_rest()
                if raw_pcm:
                    try:
                        await request_translation(raw_pcm)
                    except Exception as rest_exc:  # pragma: no cover
                        _show_snackbar(f"REST 请求失败: {rest_exc}")
                        speculative_bar.value = 0.0
                        page.update()
            except Exception as exc:  # pragma: no cover
                _show_snackbar(f"Streaming 失败: {exc}")
                page.update()
            finally:
                _streaming_task = None
                _chunk_queue = None
            return

        if not raw_pcm:
            _show_snackbar("未采集到音频，请重试。", is_error=False)
            page.update()
            return

        try:
            await request_translation(raw_pcm)
        except Exception as exc:  # pragma: no cover - runtime/network boundary
            _show_snackbar(f"请求失败: {exc}")
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
            _show_snackbar(f"播放失败: {exc}")
        page.update()

    def _set_mode(e: ft.ControlEvent) -> None:
        nonlocal translate_mode
        translate_mode = next(iter(e.control.selected), "rest")

    mode_selector = ft.SegmentedButton(
        segments=[
            ft.Segment(value="rest", label=ft.Text("REST"), icon=ft.Icon(ft.Icons.UPLOAD_FILE)),
            ft.Segment(value="streaming", label=ft.Text("Streaming"), icon=ft.Icon(ft.Icons.STREAM)),
        ],
        selected=["rest"],
        show_selected_icon=False,
        style=ft.ButtonStyle(bgcolor=ft.Colors.with_opacity(0.55, ft.Colors.WHITE)),
        on_change=_set_mode,
    )

    breed_selector = ft.SegmentedButton(
        segments=[ft.Segment(label=ft.Text(name), value=name) for name in BREEDS],
        selected=[selected_breed],
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
                ft.Row(
                    [
                        ft.Text("The Bridge", color=TEXT_DARK, size=18, weight=ft.FontWeight.W_600),
                        ws_status_chip,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                cat_avatar,
                mode_selector,
                breed_selector,
                ft.Row(
                    [waveform, recording_timer_text],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=8,
                    expand=True,
                ),
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

    ft.run(meowsformer_ui, view=view, host=host, port=port, web_renderer="canvaskit", no_cdn=True)


if __name__ == "__main__":
    main()

