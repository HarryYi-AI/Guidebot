import asyncio
import json
import sys

import pytest

from guidebot.events import Event
from guidebot.service import (
    CommandPoller,
    CommandStream,
    GuidebotService,
    _event_from_json_line,
    _events_from_command,
)


@pytest.mark.asyncio
async def test_service_fire_event_preempts_and_notifies(tmp_path, capsys) -> None:
    preempted = False

    async def on_preempt() -> None:
        nonlocal preempted
        preempted = True

    service = GuidebotService(on_preempt=on_preempt)

    trace = await service.handle_event(
        Event("scene.detected", "camera", {"label": "fire", "summary": "发现明火"})
    )

    assert trace.intent.intent_type.value == "safety_fire_alert"
    assert preempted is True
    assert "明火" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_preempt_callback_failure_does_not_crash_service(capsys) -> None:
    async def broken_preempt() -> None:
        raise ConnectionError("closed websocket")

    service = GuidebotService(on_preempt=broken_preempt)

    trace = await service.handle_event(
        Event("scene.detected", "camera", {"label": "fire", "summary": "发现明火"})
    )

    assert trace.intent.intent_type.value == "safety_fire_alert"
    assert "语音抢占未完成" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_command_poller_converts_scene_json(tmp_path) -> None:
    script = tmp_path / "scene.py"
    script.write_text(
        "import json\n"
        "print(json.dumps({'label': 'fire', 'summary': '发现明火', 'confidence': 0.97}))\n",
        encoding="utf-8",
    )

    events = await _events_from_command(
        CommandPoller("scene_command", f"{sys.executable} {script}", 1.0, "scene")
    )

    assert len(events) == 1
    assert events[0].event_type == "scene.detected"
    assert events[0].payload["label"] == "fire"
    assert events[0].priority_hint == 100


@pytest.mark.asyncio
async def test_service_stream_reads_json_with_noisy_stderr(tmp_path) -> None:
    script = tmp_path / "stream.py"
    script.write_text(
        "import sys, time\n"
        "print('mediapipe warning', file=sys.stderr)\n"
        "print('{\"label\":\"fatigue\",\"fatigue\":true,\"confidence\":0.9}', flush=True)\n"
        "time.sleep(0.1)\n",
        encoding="utf-8",
    )
    service = GuidebotService()
    service.config.command_streams.append(
        CommandStream("health", f"{sys.executable} {script}", "health")
    )
    task = asyncio.create_task(service.run_forever())
    try:
        trace = await service.run_once(timeout_seconds=1.0)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await service.stop()

    assert trace is not None
    assert trace.intent.intent_type.value == "health_fatigue"


@pytest.mark.asyncio
async def test_service_alarm_text_schedules_trigger() -> None:
    service = GuidebotService()

    await service.handle_event(Event("user.text", "test", {"text": "0秒后提醒我"}))
    trace = await service.run_once(timeout_seconds=0.5)

    assert trace is not None
    assert trace.event.event_type == "alarm.triggered"
    assert trace.intent.intent_type.value == "timer_reminder"


def test_command_poller_accepts_direct_event_json(tmp_path) -> None:
    payload = {
        "event_type": "ultrasonic.obstacle",
        "source": "ultrasonic",
        "payload": {"obstacle": True, "distance_mm": 100},
        "confidence": 1.0,
        "priority_hint": 100,
    }
    assert json.dumps(payload, ensure_ascii=False)


def test_stream_ignores_non_json_logs() -> None:
    assert _event_from_json_line(CommandStream("health", "cmd", "health"), b"[STATE] ok\n") is None


def test_stream_converts_health_json_line() -> None:
    event = _event_from_json_line(
        CommandStream("health", "cmd", "health"),
        b'{"label":"fatigue","fatigue":true,"confidence":0.9}\n',
    )

    assert event is not None
    assert event.event_type == "health.detected"
    assert event.payload["fatigue"] is True
