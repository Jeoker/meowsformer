"""Local sound-id player with realtime pitch/tempo processing."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
from typing import Any

import flet as ft
import librosa
import numpy as np
import soundfile as sf


class BioacousticPlayer:
    """Resolve sound_id to local sample and play with DSP tweaks."""

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

    async def play_sound_id(
        self,
        sound_id: str,
        pitch_factor: float = 1.0,
        tempo_factor: float = 1.0,
    ) -> str:
        source = self._resolve_sound(sound_id)
        wav_bytes = await asyncio.to_thread(
            self._process_to_wav_bytes,
            source,
            pitch_factor,
            tempo_factor,
        )
        data_url = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
        self.page.launch_url(data_url)
        return data_url

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

