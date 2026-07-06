"""Playback-time VAD monitor for user interruption."""

from __future__ import annotations

import asyncio

from .interfaces import AudioSource, VoiceActivityDetector
from .pipeline import VoicePipeline
from .turn_detector import TurnDetector, TurnEvent


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
