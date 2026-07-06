"""Lightweight reference backends for tests and dependency-free demos."""

from __future__ import annotations

import asyncio
import math
from array import array
from collections import deque
from collections.abc import AsyncIterator, Iterable

from .models import AudioFrame, CapturedTurn


class EnergyVAD:
    """PCM16 RMS detector; useful as a fallback, not a replacement for Silero in noise."""

    def __init__(self, rms_threshold: float = 500.0) -> None:
        if rms_threshold < 0:
            raise ValueError("rms_threshold must be non-negative")
        self.rms_threshold = rms_threshold

    def is_speech(self, frame: AudioFrame) -> bool:
        if frame.sample_width_bytes != 2:
            raise ValueError("EnergyVAD currently supports 16-bit PCM only")
        samples = array("h")
        samples.frombytes(frame.pcm)
        if not samples:
            return False
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        return rms >= self.rms_threshold


class ScriptedAudioSource:
    def __init__(self, frames: Iterable[AudioFrame]) -> None:
        self.frames = deque(frames)

    async def read(self) -> AudioFrame | None:
        await asyncio.sleep(0)
        return self.frames.popleft() if self.frames else None


class ScriptedSTT:
    def __init__(self, transcripts: Iterable[str]) -> None:
        self.transcripts = deque(transcripts)
        self.turns: list[CapturedTurn] = []

    async def transcribe(self, turn: CapturedTurn) -> str:
        self.turns.append(turn)
        return self.transcripts.popleft() if self.transcripts else ""


class EchoDialogue:
    async def respond(self, text: str) -> str:
        return f"我听到了：{text}"


class StreamingEchoDialogue:
    def __init__(self, delay_seconds: float = 0.0) -> None:
        self.delay_seconds = delay_seconds

    async def stream(self, text: str) -> AsyncIterator[str]:
        for token in ("我听到了。", "你说的是：", text):
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            yield token


class Utf8ChunkTTS:
    """Encodes text chunks as bytes so the full pipeline runs without a TTS model."""

    def __init__(self, characters_per_chunk: int = 12, delay_seconds: float = 0.0) -> None:
        if characters_per_chunk < 1 or delay_seconds < 0:
            raise ValueError("invalid TTS chunk configuration")
        self.characters_per_chunk = characters_per_chunk
        self.delay_seconds = delay_seconds

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        for start in range(0, len(text), self.characters_per_chunk):
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            yield text[start : start + self.characters_per_chunk].encode("utf-8")


class MemoryAudioPlayer:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.stopped = False

    async def play(self, audio: bytes) -> None:
        self.chunks.append(audio)

    async def stop(self) -> None:
        self.stopped = True
