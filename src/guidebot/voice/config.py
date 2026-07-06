"""Dependency-free voice runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    sample_rate: int = 16_000
    channels: int = 1
    sample_width_bytes: int = 2
    chunk_ms: int = 20
    pre_roll_ms: int = 300
    min_speech_ms: int = 250
    silence_hangover_ms: int = 500
    max_turn_ms: int = 20_000

    def __post_init__(self) -> None:
        positive = (
            self.sample_rate,
            self.channels,
            self.sample_width_bytes,
            self.chunk_ms,
            self.min_speech_ms,
            self.silence_hangover_ms,
            self.max_turn_ms,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("voice timing and audio format values must be positive")
        if self.pre_roll_ms < 0:
            raise ValueError("pre_roll_ms must be non-negative")

    @property
    def frame_bytes(self) -> int:
        samples = self.sample_rate * self.chunk_ms // 1000
        return samples * self.channels * self.sample_width_bytes

