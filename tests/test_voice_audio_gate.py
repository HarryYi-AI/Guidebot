from __future__ import annotations

import pytest

from guidebot.voice.audio_gate import NoiseGateAudioSource, pcm16_rms
from guidebot.voice.backends import ScriptedAudioSource
from guidebot.voice.models import AudioFrame


def _frame(sample: int, *, duration_ms: int = 20) -> AudioFrame:
    return AudioFrame(
        sample.to_bytes(2, "little", signed=True) * 160,
        duration_ms=duration_ms,
    )


@pytest.mark.asyncio
async def test_noise_gate_zeroes_low_energy_frames() -> None:
    source = ScriptedAudioSource((_frame(100),))
    gate = NoiseGateAudioSource(source, rms_threshold=500)

    frame = await gate.read()

    assert frame is not None
    assert pcm16_rms(frame.pcm) == 0


@pytest.mark.asyncio
async def test_noise_gate_preserves_speech_and_hangover() -> None:
    speech = _frame(1_000)
    soft_tail = _frame(100)
    late_noise = _frame(100)
    source = ScriptedAudioSource((speech, soft_tail, late_noise))
    gate = NoiseGateAudioSource(source, rms_threshold=500, hangover_ms=20)

    first = await gate.read()
    second = await gate.read()
    third = await gate.read()

    assert first is not None
    assert second is not None
    assert third is not None
    assert pcm16_rms(first.pcm) >= 500
    assert pcm16_rms(second.pcm) > 0
    assert pcm16_rms(third.pcm) == 0
