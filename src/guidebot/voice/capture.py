"""VAD-gated turn capture with pre-roll and a hard duration bound."""

from __future__ import annotations

from collections import deque

from .config import VoiceConfig
from .interfaces import AudioSource, VoiceActivityDetector
from .models import CapturedTurn
from .turn_detector import TurnDetector, TurnEvent


class TurnCapture:
    def __init__(
        self,
        source: AudioSource,
        vad: VoiceActivityDetector,
        config: VoiceConfig | None = None,
    ) -> None:
        self.source = source
        self.vad = vad
        self.config = config or VoiceConfig()

    async def capture(self) -> CapturedTurn | None:
        config = self.config
        detector = TurnDetector(config.min_speech_ms, config.silence_hangover_ms)
        pre_roll_frames = max(1, config.pre_roll_ms // config.chunk_ms)
        pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames)
        captured: list[bytes] | None = None
        total_ms = 0
        speech_ms = 0

        while total_ms < config.max_turn_ms:
            frame = await self.source.read()
            if frame is None:
                return self._finish(captured, speech_ms, total_ms)
            if (
                frame.sample_rate != config.sample_rate
                or frame.channels != config.channels
                or frame.sample_width_bytes != config.sample_width_bytes
            ):
                raise ValueError("audio source format does not match VoiceConfig")

            total_ms += frame.duration_ms
            speech = self.vad.is_speech(frame)
            if speech:
                speech_ms += frame.duration_ms
            if captured is None:
                pre_roll.append(frame.pcm)
            event = detector.update(speech, frame.duration_ms)
            if event is TurnEvent.START:
                captured = list(pre_roll)
            elif captured is not None:
                captured.append(frame.pcm)
            if event is TurnEvent.END:
                return self._finish(captured, speech_ms, total_ms)

        return self._finish(captured, speech_ms, total_ms)

    def _finish(
        self,
        frames: list[bytes] | None,
        speech_ms: int,
        total_ms: int,
    ) -> CapturedTurn | None:
        if not frames:
            return None
        config = self.config
        return CapturedTurn(
            b"".join(frames),
            config.sample_rate,
            config.channels,
            config.sample_width_bytes,
            speech_ms,
            total_ms,
        )

