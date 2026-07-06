"""Streaming speech endpoint state machine."""

from __future__ import annotations

from enum import Enum


class TurnEvent(str, Enum):
    START = "start"
    END = "end"


class TurnDetector:
    def __init__(self, min_speech_ms: int = 250, silence_hangover_ms: int = 500) -> None:
        if min_speech_ms <= 0 or silence_hangover_ms <= 0:
            raise ValueError("turn detector durations must be positive")
        self.min_speech_ms = min_speech_ms
        self.silence_hangover_ms = silence_hangover_ms
        self.speaking = False
        self.speech_ms = 0
        self.silence_ms = 0

    def update(self, is_speech: bool, chunk_ms: int = 20) -> TurnEvent | None:
        if chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if not self.speaking and self.speech_ms >= self.min_speech_ms:
                self.speaking = True
                return TurnEvent.START
            return None

        if not self.speaking:
            self.speech_ms = 0
            return None
        self.silence_ms += chunk_ms
        if self.silence_ms >= self.silence_hangover_ms:
            self.speaking = False
            self.speech_ms = 0
            self.silence_ms = 0
            return TurnEvent.END
        return None

    def reset(self) -> None:
        self.speaking = False
        self.speech_ms = 0
        self.silence_ms = 0

