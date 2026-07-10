"""Long-running Guidebot service runtime.

This module turns Guidebot from a set of one-shot CLI demos into a resident
robot loop:

Voice / Sensors / Timers -> EventBus -> IntentAnalyzer -> Scheduler ->
SafetyGate -> Modules -> notification/logging.

Heavy model code remains outside this package. Background command sources can
call the original camera, YOLO, ultrasonic, or vendor scripts and normalize
their JSON result into Guidebot events.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .events import Event
from .logbook import RuntimeLogger
from .runtime import GuidebotRuntime, RuntimeTrace
from .scheduler import Scheduler


PreemptCallback = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class CommandPoller:
    """Polls an external command that prints JSON and converts it to events."""

    name: str
    command: str
    interval_seconds: float
    kind: str
    timeout_seconds: float = 20.0


@dataclass(frozen=True, slots=True)
class CommandStream:
    """Runs a long-lived command that prints one JSON object per line."""

    name: str
    command: str
    kind: str
    restart_delay_seconds: float = 2.0


@dataclass(slots=True)
class GuidebotServiceConfig:
    log_dir: str = "logs"
    notify_command: str | None = None
    notify_min_priority: int = 50
    command_pollers: list[CommandPoller] = field(default_factory=list)
    command_streams: list[CommandStream] = field(default_factory=list)
    mock_sensors: bool = False


class GuidebotService:
    """Resident service that consumes events from voice and background sensors."""

    def __init__(
        self,
        config: GuidebotServiceConfig | None = None,
        *,
        runtime: GuidebotRuntime | None = None,
        on_preempt: PreemptCallback | None = None,
    ) -> None:
        self.config = config or GuidebotServiceConfig()
        self.runtime = runtime or GuidebotRuntime(logger=RuntimeLogger(self.config.log_dir))
        self.on_preempt = on_preempt
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []
        self._alarm_tasks: list[asyncio.Task[None]] = []
        self._closed = False

    def emit(self, event: Event) -> None:
        """Thread-safe enough entrypoint for adapters running in the same loop."""
        if not self._closed:
            self._queue.put_nowait(event)

    async def handle_event(self, event: Event) -> RuntimeTrace:
        trace = self.runtime.ingest(event)
        if trace.task is not None and Scheduler.can_preempt(trace.task):
            if self.on_preempt is not None:
                try:
                    await self.on_preempt()
                except Exception as exc:  # noqa: BLE001 - preemption is best-effort.
                    print(f"[Guidebot] 语音抢占未完成，继续执行安全任务：{exc}", flush=True)
        await self._notify(trace)
        self._maybe_schedule_alarm(trace)
        return trace

    async def run_forever(self) -> None:
        self._tasks = [asyncio.create_task(self._event_loop(), name="guidebot-events")]
        for poller in self.config.command_pollers:
            self._tasks.append(
                asyncio.create_task(self._command_loop(poller), name=f"poll-{poller.name}")
            )
        for stream in self.config.command_streams:
            self._tasks.append(
                asyncio.create_task(self._stream_loop(stream), name=f"stream-{stream.name}")
            )
        if self.config.mock_sensors:
            self._tasks.append(asyncio.create_task(self._mock_sensor_loop(), name="mock-sensors"))
        try:
            await asyncio.gather(*self._tasks)
        finally:
            await self.stop()

    async def run_once(self, *, timeout_seconds: float = 0.1) -> RuntimeTrace | None:
        try:
            event = await asyncio.wait_for(self._queue.get(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return None
        return await self.handle_event(event)

    async def stop(self) -> None:
        self._closed = True
        for task in self._tasks:
            task.cancel()
        for task in self._alarm_tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._alarm_tasks:
            await asyncio.gather(*self._alarm_tasks, return_exceptions=True)
        self.runtime.stop()

    async def _event_loop(self) -> None:
        while True:
            event = await self._queue.get()
            await self.handle_event(event)

    async def _command_loop(self, poller: CommandPoller) -> None:
        while True:
            events = await _events_from_command(poller)
            for event in events:
                self.emit(event)
            await asyncio.sleep(max(0.1, poller.interval_seconds))

    async def _stream_loop(self, stream: CommandStream) -> None:
        while True:
            try:
                process = await asyncio.create_subprocess_shell(
                    stream.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                if process.stdout is not None:
                    while True:
                        line = await process.stdout.readline()
                        if not line:
                            break
                        event = _event_from_json_line(stream, line)
                        if event is not None:
                            self.emit(event)
                await process.wait()
            except OSError as exc:
                self.emit(
                    Event(
                        "source.error",
                        stream.name,
                        {"error": str(exc), "command": stream.command},
                        confidence=0.0,
                    )
                )
            await asyncio.sleep(max(0.1, stream.restart_delay_seconds))

    async def _mock_sensor_loop(self) -> None:
        """Small resident demo loop; real deployments should use command pollers."""
        fired = False
        while True:
            if not fired:
                self.emit(
                    Event(
                        "scene.detected",
                        "mock_scene",
                        {"label": "fire", "summary": "模拟：检测到疑似明火。"},
                        confidence=0.95,
                        priority_hint=100,
                    )
                )
                fired = True
            await asyncio.sleep(3600)

    async def _notify(self, trace: RuntimeTrace) -> None:
        task = trace.task
        action = trace.action
        if task is None or action is None:
            return
        message = str(action.get("message") or action.get("reason") or "")
        if not message:
            return
        print(f"[Guidebot] {message}", flush=True)
        if self.config.notify_command is None or task.priority < self.config.notify_min_priority:
            return
        await _run_notify_command(self.config.notify_command, message)

    def _maybe_schedule_alarm(self, trace: RuntimeTrace) -> None:
        if trace.task is None or trace.task.target_module != "alarm_timer":
            return
        if trace.task.action != "set_alarm" or trace.action is None:
            return
        alarm_id = str(trace.action.get("alarm_id") or trace.task.task_id)
        alarm_time = str(trace.action.get("time") or trace.task.payload.get("time") or "")
        delay_seconds = _alarm_delay_seconds(alarm_time)
        if delay_seconds is None:
            return
        task = asyncio.create_task(
            self._alarm_after(delay_seconds, alarm_id, alarm_time),
            name=f"alarm-{alarm_id}",
        )
        self._alarm_tasks.append(task)

    async def _alarm_after(self, delay_seconds: float, alarm_id: str, alarm_time: str) -> None:
        await asyncio.sleep(max(0.0, delay_seconds))
        self.emit(
            Event(
                "alarm.triggered",
                "alarm_timer",
                {"alarm_id": alarm_id, "text": f"提醒时间到了：{alarm_time}"},
                confidence=1.0,
                priority_hint=80,
            )
        )


async def _events_from_command(poller: CommandPoller) -> list[Event]:
    try:
        process = await asyncio.create_subprocess_shell(
            poller.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=poller.timeout_seconds,
        )
    except (OSError, asyncio.TimeoutError) as exc:
        return [
            Event(
                "source.error",
                poller.name,
                {"error": str(exc), "command": poller.command},
                confidence=0.0,
            )
        ]
    if process.returncode != 0:
        return [
            Event(
                "source.error",
                poller.name,
                {
                    "returncode": process.returncode,
                    "stderr": stderr.decode("utf-8", errors="replace").strip(),
                },
                confidence=0.0,
            )
        ]
    text = stdout.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return [
            Event(
                "source.error",
                poller.name,
                {"error": f"invalid json: {exc}", "stdout": text[:500]},
                confidence=0.0,
            )
        ]
    items = payload if isinstance(payload, list) else [payload]
    return [_event_from_payload(poller, item) for item in items if isinstance(item, dict)]


def _event_from_payload(poller: CommandPoller, payload: dict[str, Any]) -> Event:
    if "event_type" in payload:
        return Event(
            str(payload["event_type"]),
            str(payload.get("source", poller.name)),
            dict(payload.get("payload") or {}),
            confidence=float(payload.get("confidence", 1.0)),
            priority_hint=int(payload.get("priority_hint", 0)),
        )

    if poller.kind == "scene":
        label = str(payload.get("label", "normal"))
        priority = 100 if label.casefold() in {"fire", "smoke", "fall"} else 20
        return Event(
            "scene.detected",
            poller.name,
            dict(payload),
            confidence=float(payload.get("confidence", 0.9)),
            priority_hint=int(payload.get("priority_hint", priority)),
        )
    if poller.kind == "health":
        return Event(
            "health.detected",
            poller.name,
            dict(payload),
            confidence=float(payload.get("confidence", 0.9)),
            priority_hint=int(payload.get("priority_hint", 50)),
        )
    if poller.kind == "ultrasonic":
        return Event(
            "ultrasonic.obstacle",
            poller.name,
            dict(payload),
            confidence=float(payload.get("confidence", 1.0)),
            priority_hint=int(payload.get("priority_hint", 100)),
        )
    return Event(
        "sensor.detected",
        poller.name,
        dict(payload),
        confidence=float(payload.get("confidence", 1.0)),
        priority_hint=int(payload.get("priority_hint", 0)),
    )


def _event_from_json_line(stream: CommandStream, line: bytes) -> Event | None:
    text = line.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return Event(
            "source.error",
            stream.name,
            {"error": f"invalid json: {exc}", "stdout": text[:500]},
            confidence=0.0,
        )
    if not isinstance(payload, dict):
        return None
    return _event_from_payload(
        CommandPoller(stream.name, stream.command, 0.0, stream.kind),
        payload,
    )


async def _run_notify_command(command: str, message: str) -> None:
    env = dict(os.environ)
    env["GUIDEBOT_MESSAGE"] = message
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
    )
    await process.communicate()


def _alarm_delay_seconds(value: str, *, now: datetime | None = None) -> float | None:
    text = value.strip()
    if not text:
        return None
    now = now or datetime.now().astimezone()
    if text.startswith("+") and text.endswith("m") and text[1:-1].isdigit():
        return int(text[1:-1]) * 60.0
    if text.startswith("+") and text.endswith("h") and text[1:-1].isdigit():
        return int(text[1:-1]) * 3600.0
    if text.startswith("+") and text.endswith("s") and text[1:-1].isdigit():
        return float(text[1:-1])
    if text == "tomorrow_morning":
        target = (now + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)
        return max(0.0, (target - now).total_seconds())
    try:
        hour_text, minute_text = text.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError:
        return None
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()
