"""Developer-facing simulation command."""

from __future__ import annotations

import argparse
import asyncio
import os

from .devices import SimulatedDevice
from .hub import GuidebotHub
from .models import Reading, SensorKind
from .self_evolving import build_default_library
from .simulation import SimulationSuite
from .voice.backends import (
    EnergyVAD,
    MemoryAudioPlayer,
    ScriptedAudioSource,
    ScriptedSTT,
    StreamingEchoDialogue,
    Utf8ChunkTTS,
)
from .voice.capture import TurnCapture
from .voice.config import VoiceConfig
from .voice.models import AudioFrame
from .voice.pipeline import VoicePipeline


def _print_realtime_event(event: object) -> None:
    from .voice.providers import RealtimeEvent

    if not isinstance(event, RealtimeEvent):
        return
    if event.type == "conversation.item.input_audio_transcription.completed":
        print(f"[User] {event.text}")
    elif event.type == "response.audio_transcript.done":
        print(f"[Guidebot] {event.text}")
    elif event.type == "session.open":
        print("已连接 Qwen Realtime，可以开始说话（Ctrl+C 退出）")


async def run_voice_qwen(args: argparse.Namespace) -> None:
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("请先设置 DASHSCOPE_API_KEY 环境变量")
    from .voice.native_runtime import NativeVoiceRuntime
    from .voice.providers import DashScopeRealtimeConfig, DashScopeRealtimeSession
    from .voice.system_audio import AplayAudioPlayer, ArecordAudioSource

    input_config = VoiceConfig(sample_rate=16_000, channels=1, sample_width_bytes=2)
    output_config = VoiceConfig(sample_rate=24_000, channels=1, sample_width_bytes=2)
    provider_config = DashScopeRealtimeConfig(
        voice=args.voice,
        enable_search=args.search and not args.no_search,
    )
    runtime = NativeVoiceRuntime(
        ArecordAudioSource(input_config, args.input_device),
        AplayAudioPlayer(output_config, args.output_device),
        DashScopeRealtimeSession(provider_config),
        _print_realtime_event,
    )
    try:
        await runtime.run()
    except KeyboardInterrupt:
        pass


async def run_demo() -> None:
    device = SimulatedDevice()
    hub = GuidebotHub(device)
    await hub.start()
    samples = (
        Reading(SensorKind.TEMPERATURE, 29.2, "°C", "simulator"),
        Reading(SensorKind.TOUCH, True, source="simulator"),
        Reading(SensorKind.AIR_QUALITY, 126, "AQI", "simulator"),
    )
    for sample in samples:
        trajectory = await hub.ingest(sample)
        print(f"[{sample.kind}] {trajectory.decision.response or '已记录，无需动作'}")
    await hub.stop()


def run_simulation() -> None:
    report = SimulationSuite().run(build_default_library())
    print(
        f"simulation score={report.score:.4f} "
        f"comfort_error={report.metrics.comfort_error:.3f} "
        f"safety_violations={report.metrics.unsafe_action_count}"
    )


def run_evolve_dry() -> None:
    report = SimulationSuite().run(build_default_library())
    print(
        "dry-run: no skill mutation; "
        f"baseline_score={report.score:.4f}; scenarios={len(report.scenarios)}"
    )

async def run_voice_demo() -> None:
    config = VoiceConfig(
        chunk_ms=20,
        pre_roll_ms=20,
        min_speech_ms=40,
        silence_hangover_ms=40,
    )
    silence = AudioFrame(bytes(config.frame_bytes), duration_ms=config.chunk_ms)
    sample = (1_200).to_bytes(2, "little", signed=True)
    speech = AudioFrame(sample * (config.frame_bytes // 2), duration_ms=config.chunk_ms)
    source = ScriptedAudioSource((silence, speech, speech, silence, silence))
    player = MemoryAudioPlayer()
    pipeline = VoicePipeline(
        TurnCapture(source, EnergyVAD(), config),
        ScriptedSTT(("你好 Guidebot",)),
        StreamingEchoDialogue(),
        Utf8ChunkTTS(),
        player,
    )
    result = await pipeline.run_once()
    if result is None:
        print("voice-demo: no speech detected")
        return
    print(
        f"voice-demo transcript={result.transcript!r} response={result.response_text!r} "
        f"audio_chunks={result.audio_chunks_played}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Guidebot development runtime")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("demo")
    subparsers.add_parser("simulate")
    subparsers.add_parser("voice-demo")
    qwen = subparsers.add_parser("voice-qwen", help="run Qwen Omni realtime speech")
    qwen.add_argument("--voice", default="Tina")
    qwen.add_argument("--input-device")
    qwen.add_argument("--output-device")
    qwen.add_argument("--search", action="store_true", help="enable web search; slower")
    qwen.add_argument("--no-search", action="store_true", help=argparse.SUPPRESS)
    evolve = subparsers.add_parser("evolve")
    evolve.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args()
    if args.command in (None, "demo"):
        asyncio.run(run_demo())
    elif args.command == "simulate":
        run_simulation()
    elif args.command == "evolve":
        run_evolve_dry()
    elif args.command == "voice-demo":
        asyncio.run(run_voice_demo())
    elif args.command == "voice-qwen":
        try:
            asyncio.run(run_voice_qwen(args))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
