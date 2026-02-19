"""Audio capture utility for realtime waveform rendering."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional runtime dependency
    sd = None


@dataclass(slots=True)
class RecorderConfig:
    sample_rate: int = 16000
    channels: int = 1
    block_size: int = 2048
    waveform_window: int = 3200


class AudioRecorder:
    """Capture PCM audio and provide waveform snapshots for UI."""

    def __init__(
        self,
        on_chunk: Callable[[bytes], None] | None = None,
        config: RecorderConfig | None = None,
    ) -> None:
        self.config = config or RecorderConfig()
        self.on_chunk = on_chunk
        self.is_recording = False
        self._stream = None
        self._lock = threading.Lock()
        self._waveform = deque(maxlen=self.config.waveform_window)
        self._recorded = bytearray()
        self._fallback_thread: threading.Thread | None = None
        self._fallback_stop = threading.Event()

    def start(self) -> None:
        self.is_recording = True
        self._recorded.clear()
        if sd is None:
            self._start_fallback_waveform()
            return
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="int16",
            blocksize=self.config.block_size,
            callback=self._on_audio_frame,
        )
        self._stream.start()

    def stop(self) -> bytes:
        self.is_recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._fallback_stop.set()
        return bytes(self._recorded)

    def snapshot_waveform(self, points: int = 64) -> list[float]:
        with self._lock:
            values = list(self._waveform)
        if not values:
            return [0.0] * points
        array = np.array(values, dtype=np.float32)
        buckets = np.array_split(array, points)
        return [float(np.mean(np.abs(bucket))) if len(bucket) else 0.0 for bucket in buckets]

    def _on_audio_frame(self, indata, frames, _time_info, _status) -> None:
        chunk = indata[:, 0].copy()
        pcm_bytes = chunk.tobytes()
        with self._lock:
            self._recorded.extend(pcm_bytes)
            self._waveform.extend((chunk.astype(np.float32) / 32768.0).tolist())
        if self.on_chunk:
            self.on_chunk(pcm_bytes)

    def _start_fallback_waveform(self) -> None:
        self._fallback_stop.clear()
        if self._fallback_thread and self._fallback_thread.is_alive():
            return

        def _simulate() -> None:
            t = 0.0
            while not self._fallback_stop.is_set():
                simulated = math.sin(t) * 0.8
                with self._lock:
                    self._waveform.extend([simulated] * 64)
                t += 0.25
                time.sleep(0.05)

        self._fallback_thread = threading.Thread(target=_simulate, daemon=True)
        self._fallback_thread.start()

