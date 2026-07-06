"""Ports that isolate Guidebot from microphone, model, and speaker vendors."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from .models import AudioFrame, CapturedTurn


class AudioSource(Protocol):
    async def read(self) -> AudioFrame | None: ...


class VoiceActivityDetector(Protocol):
    def is_speech(self, frame: AudioFrame) -> bool: ...


class SpeechToText(Protocol):
    async def transcribe(self, turn: CapturedTurn) -> str: ...


class DialogueBackend(Protocol):
    async def respond(self, text: str) -> str: ...


class StreamingDialogueBackend(Protocol):
    def stream(self, text: str) -> AsyncIterator[str]: ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...


class AudioPlayer(Protocol):
    async def play(self, audio: bytes) -> None: ...

    async def stop(self) -> None: ...


class NativeSpeechSession(Protocol):
    """Port for direct speech-to-speech services such as realtime audio models."""

    async def send_audio(self, frame: AudioFrame) -> None: ...

    def receive_audio(self) -> AsyncIterator[bytes]: ...

    def receive_events(self) -> AsyncIterator[object]: ...

    async def interrupt(self) -> None: ...

    async def close(self) -> None: ...
