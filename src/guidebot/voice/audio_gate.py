"""Client-side audio conditioning for realtime microphone streams."""

from __future__ import annotations

import math
from array import array

from .interfaces import AudioSource
from .models import AudioFrame


def pcm16_rms(pcm: bytes) -> float:
    """Return root-mean-square amplitude for little-endian PCM16 audio."""

    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


class NoiseGateAudioSource:
    """Replace low-energy microphone frames with digital silence.

    The realtime provider still receives a continuous audio stream, which helps
    server-side turn detection observe pauses. The content of frames below the
    gate threshold is zeroed so fan noise, motor whine, and speaker bleed are
    less likely to be transcribed as speech.
    """

    def __init__(
        self,
        source: AudioSource,
        *,
        rms_threshold: float,
        hangover_ms: int = 200,
    ) -> None:
        if rms_threshold < 0:
            raise ValueError("rms_threshold must be non-negative")
        if hangover_ms < 0:
            raise ValueError("hangover_ms must be non-negative")
        self.source = source
        self.rms_threshold = rms_threshold
        self.hangover_ms = hangover_ms
        self._hangover_remaining_ms = 0

    async def read(self) -> AudioFrame | None:
        frame = await self.source.read()
        if frame is None or self.rms_threshold == 0:
            return frame
        if frame.sample_width_bytes != 2:
            raise ValueError("NoiseGateAudioSource currently supports PCM16 only")

        if pcm16_rms(frame.pcm) >= self.rms_threshold:
            self._hangover_remaining_ms = self.hangover_ms
            return frame
        if self._hangover_remaining_ms > 0:
            self._hangover_remaining_ms = max(
                0,
                self._hangover_remaining_ms - frame.duration_ms,
            )
            return frame
        return AudioFrame(
            bytes(len(frame.pcm)),
            frame.sample_rate,
            frame.channels,
            frame.sample_width_bytes,
            frame.duration_ms,
            frame.timestamp,
        )

    async def close(self) -> None:
        close = getattr(self.source, "close", None)
        if close is not None:
            await close()
