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
from guidebot.voice.capture import TurnCapture
from guidebot.voice.config import VoiceConfig
from guidebot.voice.models import AudioFrame
from guidebot.voice.pipeline import VoicePipeline


def _pipeline(*, tts_delay: float = 0.0) -> tuple[VoicePipeline, MemoryAudioPlayer]:
    config = VoiceConfig(
        chunk_ms=20,
        pre_roll_ms=20,
        min_speech_ms=20,
        silence_hangover_ms=20,
    )
    sample = (1_000).to_bytes(2, "little", signed=True)
    speech = AudioFrame(sample * (config.frame_bytes // 2), duration_ms=20)
    silence = AudioFrame(bytes(config.frame_bytes), duration_ms=20)
    player = MemoryAudioPlayer()
    pipeline = VoicePipeline(
        TurnCapture(ScriptedAudioSource((speech, silence)), EnergyVAD(), config),
        ScriptedSTT(("你好",)),
        EchoDialogue(),
        Utf8ChunkTTS(characters_per_chunk=2, delay_seconds=tts_delay),
        player,
    )
    return pipeline, player


async def test_pipeline_runs_vad_stt_dialogue_tts_and_playback() -> None:
    pipeline, player = _pipeline()

    result = await pipeline.run_once()

    assert result is not None
    assert result.transcript == "你好"
    assert result.response_text == "我听到了：你好"
    assert result.audio_chunks_played > 0
    assert b"".join(player.chunks).decode() == result.response_text
    assert result.first_audio_latency_ms is not None


async def test_pipeline_interrupt_cancels_streaming_tts_and_stops_player() -> None:
    pipeline, player = _pipeline(tts_delay=0.05)
    task = asyncio.create_task(pipeline.handle_text("请讲一个很长很长的故事"))
    await asyncio.sleep(0.01)

    await pipeline.interrupt()
    result = await task

    assert result.interrupted
    assert player.stopped


async def test_llm_generation_and_tts_playback_run_concurrently() -> None:
    release_second_token = asyncio.Event()

    class StreamingDialogue:
        async def stream(self, text: str):
            yield "这是第一句话。"
            await release_second_token.wait()
            yield "这是稍后生成的第二句话。"

    class SignalingPlayer(MemoryAudioPlayer):
        async def play(self, audio: bytes) -> None:
            await super().play(audio)
            release_second_token.set()

    pipeline, _ = _pipeline()
    player = SignalingPlayer()
    pipeline.dialogue = StreamingDialogue()
    pipeline.player = player

    result = await asyncio.wait_for(pipeline.handle_text("开始"), timeout=1)

    assert result.response_text == "这是第一句话。这是稍后生成的第二句话。"
    assert b"".join(player.chunks).decode() == result.response_text
    assert result.audio_chunks_played > 1
