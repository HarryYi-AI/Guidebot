from __future__ import annotations

from guidebot.voice.backends import EnergyVAD, ScriptedAudioSource
from guidebot.voice.capture import TurnCapture
from guidebot.voice.config import VoiceConfig
from guidebot.voice.models import AudioFrame
from guidebot.voice.turn_detector import TurnDetector, TurnEvent


def _frame(value: int, config: VoiceConfig) -> AudioFrame:
    sample = value.to_bytes(2, "little", signed=True)
    return AudioFrame(sample * (config.frame_bytes // 2), duration_ms=config.chunk_ms)


def test_turn_detector_requires_continuous_speech_and_hangover() -> None:
    detector = TurnDetector(min_speech_ms=40, silence_hangover_ms=40)

    assert detector.update(True, 20) is None
    assert detector.update(False, 20) is None
    assert detector.update(True, 20) is None
    assert detector.update(True, 20) is TurnEvent.START
    assert detector.update(False, 20) is None
    assert detector.update(False, 20) is TurnEvent.END


async def test_turn_capture_keeps_pre_roll_and_ends_on_silence() -> None:
    config = VoiceConfig(
        chunk_ms=20,
        pre_roll_ms=40,
        min_speech_ms=40,
        silence_hangover_ms=40,
    )
    silence = _frame(0, config)
    speech = _frame(1_000, config)
    source = ScriptedAudioSource((silence, speech, speech, silence, silence))

    turn = await TurnCapture(source, EnergyVAD(), config).capture()

    assert turn is not None
    assert turn.speech_ms == 40
    assert turn.total_ms == 100
    assert len(turn.pcm) == config.frame_bytes * 4


async def test_capture_returns_none_when_no_speech_exists() -> None:
    config = VoiceConfig(min_speech_ms=40, silence_hangover_ms=40)
    source = ScriptedAudioSource((_frame(0, config), _frame(0, config)))

    assert await TurnCapture(source, EnergyVAD(), config).capture() is None

