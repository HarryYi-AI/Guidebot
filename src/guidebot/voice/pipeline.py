"""End-to-end async voice turn orchestration with interrupt support."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .capture import TurnCapture
from .interfaces import AudioPlayer, DialogueBackend, SpeechToText, TextToSpeech
from .models import VoiceTurnResult
from .segmentation import TextSegmenter


@dataclass(slots=True)
class _StreamState:
    tokens: list[str] = field(default_factory=list)
    chunks_played: int = 0
    first_audio_latency_ms: float | None = None


class VoicePipeline:
    def __init__(
        self,
        capture: TurnCapture,
        stt: SpeechToText,
        dialogue: DialogueBackend,
        tts: TextToSpeech,
        player: AudioPlayer,
        segmenter_factory=TextSegmenter,
    ) -> None:
        self.capture = capture
        self.stt = stt
        self.dialogue = dialogue
        self.tts = tts
        self.player = player
        self.segmenter_factory = segmenter_factory
        self._playback_task: asyncio.Task[None] | None = None
        self._stream_state: _StreamState | None = None
        self._interrupted = False

    async def run_once(self) -> VoiceTurnResult | None:
        turn = await self.capture.capture()
        if turn is None:
            return None
        transcript = (await self.stt.transcribe(turn)).strip()
        if not transcript:
            return VoiceTurnResult("", "", 0, False, None)
        return await self.handle_text(transcript)

    async def handle_text(self, transcript: str) -> VoiceTurnResult:
        self._interrupted = False
        state = _StreamState()
        self._stream_state = state
        started = time.monotonic()
        self._playback_task = asyncio.create_task(
            self._stream_and_speak(transcript, state, started)
        )
        try:
            await self._playback_task
        except asyncio.CancelledError:
            pass
        finally:
            self._playback_task = None
            self._stream_state = None
        return VoiceTurnResult(
            transcript,
            "".join(state.tokens).strip(),
            state.chunks_played,
            self._interrupted,
            state.first_audio_latency_ms,
        )

    async def interrupt(self) -> None:
        """Stop current TTS immediately when a separate VAD listener detects barge-in."""

        self._interrupted = True
        if self._playback_task is not None and not self._playback_task.done():
            self._playback_task.cancel()
        await self.player.stop()

    async def _stream_and_speak(
        self,
        transcript: str,
        state: _StreamState,
        started: float,
    ) -> None:
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=4)

        async def produce() -> None:
            segmenter = self.segmenter_factory()
            try:
                async for token in self._response_tokens(transcript):
                    state.tokens.append(token)
                    for segment in segmenter.feed(token):
                        await queue.put(segment)
                final = segmenter.flush()
                if final:
                    await queue.put(final)
            finally:
                await queue.put(None)

        async def consume() -> None:
            while True:
                segment = await queue.get()
                if segment is None:
                    return
                async for audio in self.tts.synthesize(segment):
                    if state.first_audio_latency_ms is None:
                        state.first_audio_latency_ms = (time.monotonic() - started) * 1000.0
                    await self.player.play(audio)
                    state.chunks_played += 1

        await asyncio.gather(produce(), consume())

    async def _response_tokens(self, transcript: str):
        stream = getattr(self.dialogue, "stream", None)
        if stream is not None:
            async for token in stream(transcript):
                yield token
            return
        yield await self.dialogue.respond(transcript)
