"""Playback-time VAD monitor for user interruption."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from enum import Enum

from .interfaces import AudioSource, VoiceActivityDetector
from .pipeline import VoicePipeline
from .providers.dashscope_realtime import RealtimeEvent
from .turn_detector import TurnDetector, TurnEvent


class BargeInMode(str, Enum):
    OFF = "off"
    TRANSCRIPT = "transcript"
    VAD = "vad"


@dataclass(frozen=True, slots=True)
class BargeInPolicy:
    """Decides whether user input should interrupt model playback.

    ``VAD`` is fastest but coughs and speaker echo can trigger it. ``TRANSCRIPT``
    waits for a meaningful transcription signal, which is slightly slower but
    much better for open microphones on a robot chassis.
    """

    mode: BargeInMode = BargeInMode.TRANSCRIPT
    min_transcript_chars: int = 4
    stop_playback_on_speech_start: bool = True
    speech_start_hold_ms: int = 1_200
    interrupt_phrases: tuple[str, ...] = (
        "停",
        "停一下",
        "暂停",
        "等下",
        "等一下",
        "先别说",
        "别说了",
        "不对",
        "不是",
        "打断一下",
    )

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            object.__setattr__(self, "mode", BargeInMode(self.mode))
        if self.min_transcript_chars < 1:
            raise ValueError("min_transcript_chars must be positive")
        if self.speech_start_hold_ms < 0:
            raise ValueError("speech_start_hold_ms must be non-negative")

    def should_hold_playback(self, event: RealtimeEvent, *, responding: bool) -> bool:
        """Return whether local playback should pause while speech is verified."""

        if (
            not responding
            or self.mode is BargeInMode.OFF
            or not self.stop_playback_on_speech_start
        ):
            return False
        return event.type in {
            "input_audio_buffer.speech_started",
            "conversation.item.input_audio_transcription.delta",
        }

    def should_interrupt(self, event: RealtimeEvent, *, responding: bool) -> bool:
        if not responding or self.mode is BargeInMode.OFF:
            return False
        if self.mode is BargeInMode.VAD:
            return event.type == "input_audio_buffer.speech_started"
        if event.type not in {
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.completed",
        }:
            return False
        text = _normalize(event.text)
        if any(_normalize(phrase) in text for phrase in self.interrupt_phrases):
            return True
        return len(text) >= self.min_transcript_chars


class BargeInMonitor:
    def __init__(
        self,
        source: AudioSource,
        vad: VoiceActivityDetector,
        pipeline: VoicePipeline,
        *,
        min_speech_ms: int = 200,
    ) -> None:
        self.source = source
        self.vad = vad
        self.pipeline = pipeline
        self.min_speech_ms = min_speech_ms

    async def watch(self, response_task: asyncio.Task) -> bool:
        detector = TurnDetector(self.min_speech_ms, silence_hangover_ms=500)
        while not response_task.done():
            frame = await self.source.read()
            if frame is None:
                return False
            event = detector.update(self.vad.is_speech(frame), frame.duration_ms)
            if event is TurnEvent.START:
                await self.pipeline.interrupt()
                return True
        return False


def _normalize(text: str) -> str:
    return "".join(re.findall(r"[0-9a-zA-Z\u4e00-\u9fff]+", text.casefold()))
