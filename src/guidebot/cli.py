"""Developer-facing simulation command."""

from __future__ import annotations

import argparse
import asyncio
import json
import locale
import os
import sys

from .devices import SimulatedDevice
from .events import Event
from .hub import GuidebotHub
from .intent_analyzer import IntentAnalyzer
from .logbook import RuntimeLogger, to_jsonable
from .model_serving import (
    DeterministicBackend,
    ModelRequest,
    OpenAICompatibleBackend,
    benchmark_backend,
)
from .models import Reading, SensorKind
from .runtime import GuidebotRuntime, RuntimeTrace
from .runtime_skills import build_default_runtime_skills
from .self_evolving import build_default_library
from .service import CommandPoller, CommandStream, GuidebotService, GuidebotServiceConfig
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


def _configure_stdio() -> None:
    """Respect locale by default; allow explicit stdio encoding override."""
    encoding = os.getenv("GUIDEBOT_STDIO_ENCODING")
    if not encoding:
        preferred = locale.getpreferredencoding(False)
        encoding = preferred if preferred.casefold() == "ascii" else ""
    if not encoding:
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding=encoding, errors="replace")


def _ascii_logs() -> bool:
    return os.getenv("GUIDEBOT_ASCII_LOGS") == "1"


def _runtime_text(chinese: str, english: str) -> str:
    return english if _ascii_logs() else chinese


def _print_realtime_event(event: object) -> None:
    from .voice.providers import RealtimeEvent

    if not isinstance(event, RealtimeEvent):
        return
    if event.type == "conversation.item.input_audio_transcription.completed":
        print(f"[User] {event.text if not _ascii_logs() else '<transcript>'}")
    elif event.type == "response.audio_transcript.done":
        print(f"[Guidebot] {event.text if not _ascii_logs() else '<response>'}")
    elif event.type == "session.open":
        print(_runtime_text("已连接 Qwen Realtime，可以开始说话（Ctrl+C 退出）", "Qwen Realtime connected. Press Ctrl+C to exit."))
    elif event.type in {"guidebot.session.awake", "guidebot.session.sleep"}:
        print(f"[Guidebot] {event.text}")
    elif event.type.startswith("guidebot.command."):
        print(f"[Guidebot] {event.text}")


async def run_voice_qwen(args: argparse.Namespace) -> None:
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise SystemExit("请先设置 DASHSCOPE_API_KEY 环境变量")
    from .voice.audio_gate import NoiseGateAudioSource
    from .voice.barge_in import BargeInMode, BargeInPolicy
    from .voice.commands import AlsaVolumeController, VoiceIntentRouter
    from .voice.native_runtime import NativeVoiceRuntime
    from .voice.providers import DashScopeRealtimeConfig, DashScopeRealtimeSession
    from .voice.session_control import WakeSleepController
    from .voice.system_audio import AplayAudioPlayer, ArecordAudioSource

    input_config = VoiceConfig(sample_rate=16_000, channels=1, sample_width_bytes=2)
    output_config = VoiceConfig(sample_rate=24_000, channels=1, sample_width_bytes=2)
    provider_config = DashScopeRealtimeConfig(
        voice=args.voice,
        enable_search=args.search and not args.no_search,
        connect_retries=args.connect_retries,
        turn_detection_threshold=args.vad_threshold,
        turn_detection_silence_duration_ms=args.vad_silence_ms,
    )
    source = ArecordAudioSource(input_config, args.input_device)
    if args.input_gate_rms > 0:
        source = NoiseGateAudioSource(
            source,
            rms_threshold=args.input_gate_rms,
            hangover_ms=args.input_gate_hangover_ms,
        )
    session_controller = None
    if args.require_wake:
        session_controller = WakeSleepController(
            wake_phrases=tuple(args.wake_phrase),
            sleep_phrases=tuple(args.sleep_phrase),
            require_wake=True,
            debug_inactive_transcripts=args.debug_inactive_transcripts,
        )
    command_router = None
    if not args.disable_voice_commands:
        command_router = VoiceIntentRouter(
            AlsaVolumeController(device=args.volume_device, mixer=args.volume_mixer),
            volume_step=args.volume_step,
        )
    runtime = NativeVoiceRuntime(
        source,
        AplayAudioPlayer(output_config, args.output_device),
        DashScopeRealtimeSession(provider_config),
        _print_realtime_event,
        session_controller,
        command_router,
        BargeInPolicy(
            BargeInMode(args.barge_in_mode),
            min_transcript_chars=args.barge_in_min_chars,
            stop_playback_on_speech_start=args.barge_in_early_stop_ms > 0,
            speech_start_hold_ms=args.barge_in_early_stop_ms,
        ),
    )
    try:
        await runtime.run()
    except KeyboardInterrupt:
        pass


async def run_serve(args: argparse.Namespace) -> None:
    """Run the resident multimodal robot loop."""
    from .voice.providers import RealtimeEvent
    from .voice.session_control import TRANSCRIPT_DONE

    command_pollers = []
    if args.scene_command:
        command_pollers.append(
            CommandPoller("scene_command", args.scene_command, args.scene_interval, "scene")
        )
    if args.health_command:
        command_pollers.append(
            CommandPoller("health_command", args.health_command, args.health_interval, "health")
        )
    if args.climate_command:
        command_pollers.append(
            CommandPoller(
                "climate_command",
                args.climate_command,
                args.climate_interval,
                "climate",
            )
        )
    if args.ultrasonic_command:
        command_pollers.append(
            CommandPoller(
                "ultrasonic_command",
                args.ultrasonic_command,
                args.ultrasonic_interval,
                "ultrasonic",
            )
        )
    command_streams = []
    for index, command in enumerate(args.scene_stream_command or (), start=1):
        command_streams.append(CommandStream(f"scene_stream_{index}", command, "scene"))
    for index, command in enumerate(args.health_stream_command or (), start=1):
        command_streams.append(CommandStream(f"health_stream_{index}", command, "health"))
    for index, command in enumerate(args.climate_stream_command or (), start=1):
        command_streams.append(CommandStream(f"climate_stream_{index}", command, "climate"))
    for index, command in enumerate(args.ultrasonic_stream_command or (), start=1):
        command_streams.append(
            CommandStream(f"ultrasonic_stream_{index}", command, "ultrasonic")
        )

    voice_runtime = None

    async def interrupt_voice() -> None:
        if voice_runtime is not None:
            await voice_runtime.interrupt_now()

    service = GuidebotService(
        GuidebotServiceConfig(
            log_dir=args.log_dir,
            notify_command=args.notify_command,
            notify_min_priority=args.notify_min_priority,
            command_pollers=command_pollers,
            command_streams=command_streams,
            mock_sensors=args.mock_sensors,
        ),
        on_preempt=interrupt_voice,
    )

    def on_realtime_event(event: object) -> None:
        _print_realtime_event(event)
        if isinstance(event, RealtimeEvent) and event.type == TRANSCRIPT_DONE and event.text:
            service.emit(
                Event(
                    "user.text",
                    "voice_asr",
                    {"text": event.text},
                    confidence=1.0,
                    priority_hint=10,
                )
            )

    tasks: list[asyncio.Task[None]] = [
        asyncio.create_task(service.run_forever(), name="guidebot-service")
    ]

    if args.stdin_text:
        tasks.append(asyncio.create_task(_stdin_text_loop(service), name="stdin-text"))

    if not args.no_voice:
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise SystemExit("请先设置 DASHSCOPE_API_KEY 环境变量，或加 --no-voice 只跑传感器链路")
        from .voice.audio_gate import NoiseGateAudioSource
        from .voice.barge_in import BargeInMode, BargeInPolicy
        from .voice.commands import AlsaVolumeController, VoiceIntentRouter
        from .voice.native_runtime import NativeVoiceRuntime
        from .voice.providers import DashScopeRealtimeConfig, DashScopeRealtimeSession
        from .voice.session_control import WakeSleepController
        from .voice.system_audio import AplayAudioPlayer, ArecordAudioSource

        input_config = VoiceConfig(sample_rate=16_000, channels=1, sample_width_bytes=2)
        output_config = VoiceConfig(sample_rate=24_000, channels=1, sample_width_bytes=2)
        source = ArecordAudioSource(input_config, args.input_device)
        if args.input_gate_rms > 0:
            source = NoiseGateAudioSource(
                source,
                rms_threshold=args.input_gate_rms,
                hangover_ms=args.input_gate_hangover_ms,
            )
        session_controller = WakeSleepController(
            wake_phrases=tuple(args.wake_phrase),
            sleep_phrases=tuple(args.sleep_phrase),
            require_wake=True,
            debug_inactive_transcripts=args.debug_inactive_transcripts,
        )
        command_router = None
        if not args.disable_voice_commands:
            command_router = VoiceIntentRouter(
                AlsaVolumeController(device=args.volume_device, mixer=args.volume_mixer),
                volume_step=args.volume_step,
            )
        voice_runtime = NativeVoiceRuntime(
            source,
            AplayAudioPlayer(output_config, args.output_device),
            DashScopeRealtimeSession(
                DashScopeRealtimeConfig(
                    voice=args.voice,
                    enable_search=args.search and not args.no_search,
                    connect_retries=args.connect_retries,
                    turn_detection_threshold=args.vad_threshold,
                    turn_detection_silence_duration_ms=args.vad_silence_ms,
                )
            ),
            on_realtime_event,
            session_controller,
            command_router,
            BargeInPolicy(
                BargeInMode(args.barge_in_mode),
                min_transcript_chars=args.barge_in_min_chars,
                stop_playback_on_speech_start=args.barge_in_early_stop_ms > 0,
                speech_start_hold_ms=args.barge_in_early_stop_ms,
            ),
        )
        tasks.append(asyncio.create_task(voice_runtime.run(), name="voice-qwen"))

    print(
        _runtime_text(
            "Guidebot 常驻服务已启动。语音等待唤醒；传感器异常会自动通知。Ctrl+C 退出。",
            "Guidebot resident service started. Voice waits for wake phrase; sensor alerts are automatic. Ctrl+C exits.",
        )
    )
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await service.stop()


async def _stdin_text_loop(service: GuidebotService) -> None:
    while True:
        text = await asyncio.to_thread(input, "> ")
        if text.strip():
            service.emit(Event("user.text", "stdin", {"text": text.strip()}))


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


def _print_json(value: object) -> None:
    print(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2))


def _print_trace(trace: RuntimeTrace, *, as_json: bool) -> None:
    if as_json:
        _print_json(trace)
        return
    print(trace.human_summary())
    if trace.action is not None:
        print(trace.action.get("message", trace.action))
    else:
        print(f"intent={trace.intent.intent_type.value}; reason={trace.reason}")


def _runtime() -> GuidebotRuntime:
    return GuidebotRuntime(logger=RuntimeLogger())


def run_runtime_command(args: argparse.Namespace) -> None:
    runtime = _runtime()
    trace = runtime.ingest(Event("runtime.started", "cli", {"message": "Guidebot runtime ready"}))
    _print_trace(trace, as_json=getattr(args, "json", False))


def run_scene_scan(args: argparse.Namespace) -> None:
    runtime = _runtime()
    event = runtime.modules["scene_monitor"].scan_once(
        label=args.label,
        summary=args.summary,
        confidence=args.confidence,
    )
    trace = runtime.ingest(event)
    _print_trace(trace, as_json=args.json)


def run_health_check(args: argparse.Namespace) -> None:
    runtime = _runtime()
    event = runtime.modules["health_monitor"].check_once(
        sedentary=args.sedentary,
        fatigue=args.fatigue,
    )
    trace = runtime.ingest(event)
    _print_trace(trace, as_json=args.json)


def run_alarm_set(args: argparse.Namespace) -> None:
    runtime = _runtime()
    text = f"设置闹钟 {args.time}"
    trace = runtime.ingest(Event("user.text", "cli", {"text": text, "time": args.time}))
    _print_trace(trace, as_json=args.json)


def run_intent_parse(args: argparse.Namespace) -> None:
    event = Event("user.text", "cli", {"text": args.text})
    intent = IntentAnalyzer().analyze(event)
    if args.json:
        _print_json(intent)
    else:
        print(intent.intent_type.value)


def run_schedule_demo(args: argparse.Namespace) -> None:
    runtime = _runtime()
    if args.scenario == "fire":
        event = Event("scene.detected", "demo", {"label": "fire", "summary": "发现明火"})
    elif args.scenario == "obstacle":
        event = Event("ultrasonic.obstacle", "demo", {"obstacle": True, "distance_mm": 120})
    elif args.scenario == "sedentary":
        event = Event("health.detected", "demo", {"label": "sedentary", "sedentary": True})
    else:
        event = Event("user.text", "demo", {"text": "你好"})
    trace = runtime.ingest(event)
    _print_trace(trace, as_json=getattr(args, "json", False))


def run_climate_status(args: argparse.Namespace) -> None:
    runtime = _runtime()
    status = runtime.modules["climate_control"].status()
    if args.json:
        _print_json(status)
    else:
        print(
            f"temperature={status['temperature_c']}°C humidity={status['humidity']}% "
            "adapter=mock"
        )


def run_climate_check(args: argparse.Namespace) -> None:
    runtime = _runtime()
    event = runtime.modules["climate_control"].observe(
        temperature_c=args.temperature_c,
        humidity=args.humidity,
        ac_on=args.ac_on,
        occupied=args.occupied,
        confidence=args.confidence,
    )
    trace = runtime.ingest(event)
    _print_trace(trace, as_json=args.json)


def run_ad_demo(args: argparse.Namespace) -> None:
    runtime = _runtime()
    trace = runtime.ingest(
        Event(
            "user.text",
            "cli",
            {
                "text": f"为 {args.product} 生成广告海报",
                "product": args.product,
                "audience": args.audience,
                "goal": args.goal,
            },
        )
    )
    _print_trace(trace, as_json=args.json)


def run_eval_suite(args: argparse.Namespace) -> None:
    from .eval import RuntimeEvalRunner, core_eval_cases

    report = RuntimeEvalRunner().run(core_eval_cases(), suite=args.suite)
    if args.json:
        _print_json(report)
    else:
        print(
            f"eval suite={report.suite} score={report.score:.3f} "
            f"passed={report.passed} failed={report.failed}"
        )
    if report.failed:
        raise SystemExit(1)


def run_eval_replay(args: argparse.Namespace) -> None:
    from .eval import RuntimeEvalRunner

    report = RuntimeEvalRunner().replay(args.path)
    if args.json:
        _print_json(report)
    else:
        print(
            f"replay events={report.events} tasks={report.scheduled_tasks} "
            f"blocked={report.blocked_tasks} intents={report.intent_counts}"
        )


def run_model_benchmark(args: argparse.Namespace) -> None:
    if args.backend == "openai-compatible":
        if not args.base_url or not args.model:
            raise SystemExit("openai-compatible backend requires --base-url and --model")
        backend = OpenAICompatibleBackend(
            args.base_url,
            args.model,
            api_key=os.getenv(args.api_key_env),
        )
    else:
        backend = DeterministicBackend()
    report = benchmark_backend(
        backend,
        ModelRequest("benchmark", {"prompt": args.prompt}),
        iterations=args.iterations,
    )
    if args.json:
        _print_json(report)
    else:
        print(
            f"backend={report.backend} rps={report.requests_per_second:.2f} "
            f"p50={report.p50_latency_ms:.2f}ms p95={report.p95_latency_ms:.2f}ms "
            f"errors={report.errors}"
        )


def run_tools_list(args: argparse.Namespace) -> None:
    registry = build_default_runtime_skills()
    skills = registry.all()
    if args.json:
        _print_json(skills)
        return
    for skill in skills:
        permission = skill.contract.permission.value if skill.contract else "unspecified"
        print(f"{skill.skill_id}: {skill.target_module}.{skill.action} [{permission}]")


def run_tool_describe(args: argparse.Namespace) -> None:
    try:
        skill = build_default_runtime_skills().get(args.skill_id)
    except KeyError as exc:
        raise SystemExit(f"unknown skill: {args.skill_id}") from exc
    _print_json(skill) if args.json else print(skill)


def run_planner_demo(args: argparse.Namespace) -> None:
    from .agents import EmbodiedPlannerAgent, ScriptedPlannerClient

    if args.scenario == "approach":
        response = {
            "response": "我靠近一点看看。",
            "rationale": "target is too far",
            "steps": [
                {
                    "option": "move_closer",
                    "parameters": {"distance_m": 0.4, "speed": 0.99},
                }
            ],
        }
    elif args.scenario == "hot":
        response = {
            "response": "我建议调到 25 度。",
            "rationale": "room is hot",
            "steps": [{"option": "turn_ac", "parameters": {"target_c": 25}}],
        }
    else:
        response = {
            "response": "你希望我具体做什么呢？",
            "rationale": "request is ambiguous",
            "steps": [
                {"option": "ask_user", "parameters": {"question": "请告诉我更多信息。"}}
            ],
        }

    async def demo() -> tuple[object, object]:
        planner = EmbodiedPlannerAgent(ScriptedPlannerClient((response,)))
        device = SimulatedDevice()
        hub = GuidebotHub(device, agent=planner)
        await hub.start()
        if args.scenario == "approach":
            hub.state.update(Reading(SensorKind.DISTANCE, 1.0, "m", "planner-demo"))
        trajectory = await hub.say(args.text or "请根据当前情况行动")
        await hub.stop()
        assert planner.last_plan is not None
        return planner.last_plan, trajectory

    plan, trajectory = asyncio.run(demo())
    payload = {"high_level_plan": plan, "trajectory": trajectory}
    if args.json:
        _print_json(payload)
    else:
        print(payload)


def run_agent_demo(args: argparse.Namespace) -> None:
    """Run deterministic, hardware-free end-to-end scenarios."""
    runtime = GuidebotRuntime()
    if args.scenario == "fire-confirmation":
        session_id = "demo-fire"
        events = (
            Event(
                "scene.detected",
                "camera",
                {"label": "fire", "summary": "远处存在疑似火光", "ground_truth": "fire"},
                confidence=0.65,
                session_id=session_id,
            ),
            Event(
                "scene.detected",
                "camera",
                {"label": "fire", "summary": "二次观察确认明火", "ground_truth": "fire"},
                confidence=0.93,
                session_id=session_id,
            ),
        )
    elif args.scenario == "sedentary":
        events = (
            Event(
                "health.detected",
                "posture_detector",
                {"label": "sedentary", "sedentary": True},
                confidence=0.92,
                session_id="demo-health",
            ),
            Event(
                "health.detected",
                "posture_detector",
                {"label": "sedentary", "sedentary": True},
                confidence=0.94,
                session_id="demo-health",
            ),
        )
    else:
        events = (
            Event(
                "user.text",
                "voice_asr",
                {"text": "设置闹钟 +1m", "time": "+1m"},
                session_id="demo-alarm",
            ),
            Event(
                "alarm.triggered",
                "alarm_timer",
                {"alarm_id": "demo", "text": "起床时间到了"},
                session_id="demo-alarm",
            ),
            Event(
                "mobility.command",
                "alarm_timer",
                {
                    "action": "move_forward",
                    "speed": 0.25,
                    "obstacle": False,
                    "confirmed": True,
                },
                session_id="demo-alarm",
            ),
            Event(
                "ultrasonic.obstacle",
                "ultrasonic",
                {"obstacle": True, "distance_mm": 120},
                priority_hint=100,
                session_id="demo-alarm",
            ),
        )
    traces = tuple(runtime.ingest(event) for event in events)
    payload = {"scenario": args.scenario, "trajectories": traces}
    if args.json:
        _print_json(payload)
    else:
        for trace in traces:
            print(trace.human_summary())


def run_interview_demo(args: argparse.Namespace) -> None:
    if args.scenario == "break-reminder":
        from .runtime.demos import run_break_reminder_demo

        result = asyncio.run(run_break_reminder_demo())
    else:
        from .replay import DEFAULT_FIRE_REPLAY, replay_fire_observations

        result = asyncio.run(replay_fire_observations(DEFAULT_FIRE_REPLAY))
    _print_json(result)


def run_replay(args: argparse.Namespace) -> None:
    from .replay import load_replay, replay_fire_observations

    result = asyncio.run(replay_fire_observations(load_replay(args.path)))
    _print_json(result)


def add_qwen_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--voice", default="Tina")
    parser.add_argument("--input-device")
    parser.add_argument("--output-device")
    parser.add_argument("--search", action="store_true", help="enable web search; slower")
    parser.add_argument("--no-search", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--connect-retries", type=int, default=3)
    parser.add_argument(
        "--vad-threshold",
        type=float,
        help="raise this, e.g. 0.75-0.9, if background noise triggers speech",
    )
    parser.add_argument(
        "--vad-silence-ms",
        type=int,
        help="silence duration before ending a turn; e.g. 700-1000",
    )
    parser.add_argument(
        "--input-gate-rms",
        type=float,
        default=0,
        help="zero microphone frames below this RMS amplitude; try 400-1200",
    )
    parser.add_argument(
        "--input-gate-hangover-ms",
        type=int,
        default=250,
        help="keep sending speech briefly after RMS drops below the local gate",
    )
    parser.add_argument(
        "--require-wake",
        action="store_true",
        help="wait for a wake phrase before playing model replies",
    )
    parser.add_argument(
        "--wake-phrase",
        action="append",
        default=["你好guidebot", "你好小盖", "小盖同学"],
    )
    parser.add_argument(
        "--sleep-phrase",
        action="append",
        default=["今天聊到这里", "结束对话", "先这样", "不用聊了", "休眠"],
    )
    parser.add_argument("--debug-inactive-transcripts", action="store_true")
    parser.add_argument("--disable-voice-commands", action="store_true")
    parser.add_argument("--volume-device", default="default")
    parser.add_argument("--volume-mixer", default="Master")
    parser.add_argument("--volume-step", type=int, default=10)
    parser.add_argument(
        "--barge-in-mode",
        choices=("off", "transcript", "vad"),
        default="transcript",
        help="interrupt policy while Guidebot is speaking; vad is fastest but noisy",
    )
    parser.add_argument(
        "--barge-in-min-chars",
        type=int,
        default=4,
        help="minimum transcribed characters required for transcript-mode interruption",
    )
    parser.add_argument(
        "--barge-in-early-stop-ms",
        type=int,
        default=1_200,
        help="immediately stop local playback for this long after speech starts; 0 disables",
    )


def main(argv: list[str] | None = None) -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Guidebot development runtime")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--json", action="store_true")
    run_parser.set_defaults(handler=run_runtime_command)
    serve = subparsers.add_parser("serve", help="run resident voice/sensor runtime")
    add_qwen_args(serve)
    serve.add_argument("--no-voice", action="store_true")
    serve.add_argument("--stdin-text", action="store_true")
    serve.add_argument("--log-dir", default="logs")
    serve.add_argument("--notify-command")
    serve.add_argument("--notify-min-priority", type=int, default=50)
    serve.add_argument("--scene-command")
    serve.add_argument("--scene-stream-command", action="append")
    serve.add_argument("--scene-interval", type=float, default=10.0)
    serve.add_argument("--health-command")
    serve.add_argument("--health-stream-command", action="append")
    serve.add_argument("--health-interval", type=float, default=30.0)
    serve.add_argument("--climate-command")
    serve.add_argument("--climate-stream-command", action="append")
    serve.add_argument("--climate-interval", type=float, default=30.0)
    serve.add_argument("--ultrasonic-command")
    serve.add_argument("--ultrasonic-stream-command", action="append")
    serve.add_argument("--ultrasonic-interval", type=float, default=0.3)
    serve.add_argument("--mock-sensors", action="store_true")
    chat = subparsers.add_parser("chat")
    add_qwen_args(chat)
    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("scenario", nargs="?", choices=("break-reminder", "fire-verify"))
    subparsers.add_parser("simulate")
    subparsers.add_parser("voice-demo")
    qwen = subparsers.add_parser("voice-qwen", help="run Qwen Omni realtime speech")
    add_qwen_args(qwen)
    scene = subparsers.add_parser("scene")
    scene_sub = scene.add_subparsers(dest="scene_command", required=True)
    scene_scan = scene_sub.add_parser("scan")
    scene_scan.add_argument("--json", action="store_true")
    scene_scan.add_argument("--label", default="normal")
    scene_scan.add_argument("--summary", default="未发现明显异常。")
    scene_scan.add_argument("--confidence", type=float, default=0.9)
    scene_scan.set_defaults(handler=run_scene_scan)
    health = subparsers.add_parser("health")
    health_sub = health.add_subparsers(dest="health_command", required=True)
    health_check = health_sub.add_parser("check")
    health_check.add_argument("--json", action="store_true")
    health_check.add_argument("--sedentary", action="store_true")
    health_check.add_argument("--fatigue", action="store_true")
    health_check.set_defaults(handler=run_health_check)
    alarm = subparsers.add_parser("alarm")
    alarm_sub = alarm.add_subparsers(dest="alarm_command", required=True)
    alarm_set = alarm_sub.add_parser("set")
    alarm_set.add_argument("--time", required=True)
    alarm_set.add_argument("--json", action="store_true")
    alarm_set.set_defaults(handler=run_alarm_set)
    intent = subparsers.add_parser("intent")
    intent_sub = intent.add_subparsers(dest="intent_command", required=True)
    intent_parse = intent_sub.add_parser("parse")
    intent_parse.add_argument("text")
    intent_parse.add_argument("--json", action="store_true")
    intent_parse.set_defaults(handler=run_intent_parse)
    schedule = subparsers.add_parser("schedule")
    schedule_sub = schedule.add_subparsers(dest="schedule_command", required=True)
    schedule_demo = schedule_sub.add_parser("demo")
    schedule_demo.add_argument("--scenario", choices=("fire", "obstacle", "sedentary", "chat"), required=True)
    schedule_demo.add_argument("--json", action="store_true")
    schedule_demo.set_defaults(handler=run_schedule_demo)
    climate = subparsers.add_parser("climate")
    climate_sub = climate.add_subparsers(dest="climate_command", required=True)
    climate_status = climate_sub.add_parser("status")
    climate_status.add_argument("--json", action="store_true")
    climate_status.set_defaults(handler=run_climate_status)
    climate_check = climate_sub.add_parser("check")
    climate_check.add_argument("--temperature-c", type=float)
    climate_check.add_argument("--humidity", type=float)
    climate_check.add_argument("--ac-on", action="store_true", default=None)
    climate_check.add_argument("--occupied", dest="occupied", action="store_true", default=None)
    climate_check.add_argument("--unoccupied", dest="occupied", action="store_false")
    climate_check.add_argument("--confidence", type=float, default=0.9)
    climate_check.add_argument("--json", action="store_true")
    climate_check.set_defaults(handler=run_climate_check)
    ad = subparsers.add_parser("ad", help="advertising creative workflow demo")
    ad_sub = ad.add_subparsers(dest="ad_command", required=True)
    ad_demo = ad_sub.add_parser("demo")
    ad_demo.add_argument("--product", default="Guidebot")
    ad_demo.add_argument("--audience", default="年轻家庭")
    ad_demo.add_argument("--goal", default="提升点击率")
    ad_demo.add_argument("--json", action="store_true")
    ad_demo.set_defaults(handler=run_ad_demo)
    eval_parser = subparsers.add_parser("eval", help="run held-out evals or replay logs")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_sub.add_parser("run")
    eval_run.add_argument("--suite", choices=("core",), default="core")
    eval_run.add_argument("--json", action="store_true")
    eval_run.set_defaults(handler=run_eval_suite)
    eval_replay = eval_sub.add_parser("replay")
    eval_replay.add_argument("path")
    eval_replay.add_argument("--json", action="store_true")
    eval_replay.set_defaults(handler=run_eval_replay)
    model = subparsers.add_parser("model", help="model-serving tools")
    model_sub = model.add_subparsers(dest="model_command", required=True)
    model_bench = model_sub.add_parser("benchmark")
    model_bench.add_argument(
        "--backend", choices=("deterministic", "openai-compatible"), default="deterministic"
    )
    model_bench.add_argument("--base-url")
    model_bench.add_argument("--model")
    model_bench.add_argument("--api-key-env", default="MODEL_API_KEY")
    model_bench.add_argument("--iterations", type=int, default=10)
    model_bench.add_argument("--prompt", default="用一句话介绍 Guidebot")
    model_bench.add_argument("--json", action="store_true")
    model_bench.set_defaults(handler=run_model_benchmark)
    tools_parser = subparsers.add_parser("tools", help="inspect auditable runtime tool contracts")
    tools_sub = tools_parser.add_subparsers(dest="tools_command", required=True)
    tools_list = tools_sub.add_parser("list")
    tools_list.add_argument("--json", action="store_true")
    tools_list.set_defaults(handler=run_tools_list)
    tools_describe = tools_sub.add_parser("describe")
    tools_describe.add_argument("skill_id")
    tools_describe.add_argument("--json", action="store_true")
    tools_describe.set_defaults(handler=run_tool_describe)
    planner_parser = subparsers.add_parser("planner", help="high-level skill planning demo")
    planner_sub = planner_parser.add_subparsers(dest="planner_command", required=True)
    planner_demo = planner_sub.add_parser("demo")
    planner_demo.add_argument("--scenario", choices=("approach", "hot", "ambiguous"), default="approach")
    planner_demo.add_argument("--text")
    planner_demo.add_argument("--json", action="store_true")
    planner_demo.set_defaults(handler=run_planner_demo)
    agent_parser = subparsers.add_parser("agent", help="multimodal embodied-agent demos")
    agent_sub = agent_parser.add_subparsers(dest="agent_command", required=True)
    agent_demo = agent_sub.add_parser("demo")
    agent_demo.add_argument(
        "--scenario",
        choices=("fire-confirmation", "sedentary", "alarm-obstacle"),
        required=True,
    )
    agent_demo.add_argument("--json", action="store_true")
    agent_demo.set_defaults(handler=run_agent_demo)
    replay_parser = subparsers.add_parser("replay", help="replay software observations from JSON")
    replay_parser.add_argument("path")
    replay_parser.set_defaults(handler=run_replay)
    evolve = subparsers.add_parser("evolve")
    evolve.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args(argv)
    if hasattr(args, "handler"):
        args.handler(args)
        return
    if args.command is None or (args.command == "demo" and not args.scenario):
        asyncio.run(run_demo())
    elif args.command == "demo" and args.scenario:
        run_interview_demo(args)
    elif args.command == "simulate":
        run_simulation()
    elif args.command == "evolve":
        run_evolve_dry()
    elif args.command == "voice-demo":
        asyncio.run(run_voice_demo())
    elif args.command in {"voice-qwen", "chat"}:
        try:
            asyncio.run(run_voice_qwen(args))
        except KeyboardInterrupt:
            pass
    elif args.command == "serve":
        try:
            asyncio.run(run_serve(args))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
