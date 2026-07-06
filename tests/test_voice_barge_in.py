from __future__ import annotations

import asyncio

from guidebot.voice.backends import (
    EchoDialogue,
    EnergyVAD,
    MemoryAudioPlayer,
    ScriptedAudioSource,
    ScriptedSTT,
    Utf8ChunkTTS,
)
from guidebot.voice.barge_in import BargeInMonitor
from guidebot.voice.capture import TurnCapture
from guidebot.voice.config import VoiceConfig
from guidebot.voice.models import AudioFrame
from guidebot.voice.pipeline import VoicePipeline


async def test_barge_in_monitor_interrupts_active_response() -> None:
    config = VoiceConfig(min_speech_ms=20)
    sample = (1_000).to_bytes(2, "little", signed=True)
    speech = AudioFrame(sample * (config.frame_bytes // 2))
    player = MemoryAudioPlayer()
    empty_capture = TurnCapture(ScriptedAudioSource(()), EnergyVAD(), config)
    pipeline = VoicePipeline(
        empty_capture,
        ScriptedSTT(()),
        EchoDialogue(),
        Utf8ChunkTTS(characters_per_chunk=1, delay_seconds=0.05),
        player,
    )
    response_task = asyncio.create_task(pipeline.handle_text("讲一个很长的故事"))
    monitor = BargeInMonitor(
        ScriptedAudioSource((speech,)),
        EnergyVAD(),
        pipeline,
        min_speech_ms=20,
    )

    interrupted = await monitor.watch(response_task)
    result = await response_task

    assert interrupted
    assert result.interrupted
    assert player.stopped
