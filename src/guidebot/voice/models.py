"""Voice domain models shared by hardware and cloud/local backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from guidebot.models import utc_now


@dataclass(frozen=True, slots=True)
class AudioFrame:
    pcm: bytes
    sample_rate: int = 16_000
    channels: int = 1
    sample_width_bytes: int = 2
    duration_ms: int = 20
    timestamp: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.channels <= 0 or self.sample_width_bytes <= 0:
            raise ValueError("invalid audio format")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")


@dataclass(frozen=True, slots=True)
class CapturedTurn:
    pcm: bytes
    sample_rate: int
    channels: int
    sample_width_bytes: int
    speech_ms: int
    total_ms: int


@dataclass(frozen=True, slots=True)
class VoiceTurnResult:
    transcript: str
    response_text: str
    audio_chunks_played: int
    interrupted: bool
    first_audio_latency_ms: float | None

