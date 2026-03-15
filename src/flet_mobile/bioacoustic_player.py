"""Local sound-id player with realtime pitch/tempo processing."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import flet as ft
import librosa
import numpy as np
import soundfile as sf


def _should_use_native_audio() -> bool:
    """Return True when flet_audio's Flutter widget is available.

    The flet-desktop-light pre-built client does NOT include the
    audioplayers plugin, so fta.Audio is reported as "Unknown control".
    We fall back to sounddevice for desktop/dev sessions and reserve
    fta.Audio for production mobile builds (``flet build``).
    """
    explicit = os.getenv("MEOWSFORMER_AUDIO_BACKEND", "").strip().lower()
    if explicit == "flet_audio":
        return True
    if explicit == "sounddevice":
        return False
    # Auto-detect: native audio only works in full flet builds.
    # flet run / ft.run uses flet-desktop-light which lacks plugins.
    return False


class BioacousticPlayer:
    """Resolve sound_id to local sample and play with DSP tweaks.

    Supports two playback backends:
    - **sounddevice** (default in dev/desktop): plays WAV via system audio.
    - **flet_audio** (production mobile builds): uses Flutter audioplayers
      plugin registered on ``page.overlay``.
    """

    def __init__(
        self,
        page: ft.Page,
        catalog_path: str = "assets/audio_db/tagged_samples.json",
    ) -> None:
        self.page = page
        self.repo_root = Path(__file__).resolve().parents[2]
        self.catalog_path = self.repo_root / catalog_path
        self._sample_index = self._build_index()
        self._fallback_file = self.repo_root / "meow_output.wav"

        self._use_native = _should_use_native_audio()
        self._audio: Any = None

        if self._use_native:
            import flet_audio as fta  # noqa: PLC0415

            self._audio = fta.Audio(
                src=None,
                autoplay=False,
                volume=1.0,
                release_mode=fta.ReleaseMode.STOP,
            )
            page.overlay.append(self._audio)
            page.update()

    async def _play_wav_bytes(self, wav_bytes: bytes) -> None:
        if self._use_native and self._audio is not None:
            await self._audio.release()
            self._audio.src = wav_bytes
            self._audio.update()
            await self._audio.play()
        else:
            await self._play_via_sounddevice(wav_bytes)

    @staticmethod
    async def _play_via_sounddevice(wav_bytes: bytes) -> None:
        """Play WAV bytes through the system audio device."""
        import sounddevice as sd  # noqa: PLC0415

        buf = io.BytesIO(wav_bytes)
        data, samplerate = sf.read(buf, dtype="float32")
        await asyncio.to_thread(sd.play, data, samplerate)
        await asyncio.to_thread(sd.wait)

    def dispose(self) -> None:
        """Remove the audio Service control from the page. Call on page disconnect."""
        if self._audio is not None and self._audio in self.page.overlay:
            self.page.overlay.remove(self._audio)
            self.page.update()

    async def play_sound_id(
        self,
        sound_id: str,
        pitch_factor: float = 1.0,
        tempo_factor: float = 1.0,
    ) -> None:
        source = self._resolve_sound(sound_id)
        wav_bytes = await asyncio.to_thread(
            self._process_to_wav_bytes,
            source,
            pitch_factor,
            tempo_factor,
        )
        await self._play_wav_bytes(wav_bytes)

    async def play_from_base64(self, audio_base64: str) -> None:
        """Play base64-encoded WAV audio directly (from REST/streaming results)."""
        if not audio_base64:
            return
        wav_bytes = base64.b64decode(audio_base64)
        await self._play_wav_bytes(wav_bytes)

    def _build_index(self) -> dict[str, Path]:
        if not self.catalog_path.exists():
            return {}
        with self.catalog_path.open("r", encoding="utf-8") as f:
            payload: dict[str, Any] = json.load(f)
        index: dict[str, Path] = {}
        for item in payload.get("samples", []):
            sample_id = item.get("id")
            file_path = item.get("file_path")
            if not sample_id or not file_path:
                continue
            resolved = self.repo_root / str(file_path)
            index[str(sample_id)] = resolved
        return index

    def _resolve_sound(self, sound_id: str) -> Path:
        candidate = self._sample_index.get(sound_id)
        if candidate and candidate.exists():
            return candidate
        return self._fallback_file

    @staticmethod
    def _process_to_wav_bytes(
        source: Path,
        pitch_factor: float,
        tempo_factor: float,
    ) -> bytes:
        y, sr = librosa.load(source, sr=None, mono=True)
        tempo_factor = float(np.clip(tempo_factor, 0.6, 1.8))
        pitch_factor = float(np.clip(pitch_factor, 0.7, 1.5))

        y_stretched = librosa.effects.time_stretch(y, rate=tempo_factor)
        semitones = (pitch_factor - 1.0) * 12.0
        y_shifted = librosa.effects.pitch_shift(y_stretched, sr=sr, n_steps=semitones)

        buffer = io.BytesIO()
        sf.write(buffer, y_shifted, sr, format="WAV")
        return buffer.getvalue()
