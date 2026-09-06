import json

from guidebot.cli import main
from guidebot.events import Event
from guidebot.runtime import GuidebotRuntime
from guidebot.runtime_skills import RiskLevel, build_default_runtime_skills


def test_uncertain_fire_gathers_information_then_confirms_alert() -> None:
    runtime = GuidebotRuntime()
    first = runtime.ingest(
        Event(
            "scene.detected",
            "camera",
            {"label": "fire", "ground_truth": "fire"},
            confidence=0.65,
            session_id="room-1",
        )
    )
    second = runtime.ingest(
        Event(
            "scene.detected",
            "camera",
            {"label": "fire", "ground_truth": "fire"},
            confidence=0.93,
            session_id="room-1",
        )
    )

    assert first.task is not None
    assert first.task.skill_id == "scene.inspect_again"
    assert first.trajectory is not None
    assert first.trajectory.belief.before.probability == 0.5
    assert first.trajectory.belief.after.probability == 0.65
    assert second.task is not None
    assert second.task.skill_id == "scene.fire_alert"
    assert second.trajectory is not None
    assert second.trajectory.belief.after.probability > 0.85


def test_uncertain_fire_can_ask_user_but_cannot_directly_alert() -> None:
    runtime = GuidebotRuntime()
    trace = runtime.ingest(
        Event(
            "scene.detected",
            "camera",
            {"label": "smoke", "can_reinspect": False, "user_available": True},
            confidence=0.6,
        )
    )

    assert trace.task is not None
    assert trace.task.skill_id == "scene.ask_user"
    assert trace.task.skill_id != "scene.fire_alert"


def test_information_gathering_motion_still_requires_safety_confirmation() -> None:
    runtime = GuidebotRuntime()
    trace = runtime.ingest(
        Event(
            "scene.detected",
            "camera",
            {
                "label": "fire",
                "information_action": "move_closer",
                "obstacle": False,
            },
            confidence=0.6,
        )
    )

    assert trace.task is not None
    assert trace.task.skill_id == "mobility.move_closer"
    assert trace.final_status == "safety_rejected"


def test_unified_skill_exposes_contract_result_and_risk() -> None:
    skill = build_default_runtime_skills().get("scene.fire_alert")

    assert skill.name == "scene.fire_alert"
    assert skill.description
    assert skill.precondition
    assert skill.input_schema == {}
    assert "module" in skill.result_schema
    assert skill.risk_level is RiskLevel.CRITICAL
    assert callable(skill.execute)


def test_trajectory_contains_reward_and_end_to_end_fields() -> None:
    trace = GuidebotRuntime().ingest(Event("user.text", "voice", {"text": "你好"}))

    trajectory = trace.trajectory
    assert trajectory is not None
    assert trajectory.observation.event_type == "user.text"
    assert trajectory.selected_skill == "voice.chat"
    assert trajectory.arguments == {"text": "你好"}
    assert trajectory.safety_decision is not None
    assert trajectory.result is not None
    assert trajectory.success is True
    assert trajectory.latency_ms >= 0


def test_false_high_confidence_alert_has_risk_dominated_reward() -> None:
    trace = GuidebotRuntime().ingest(
        Event(
            "scene.detected",
            "camera",
            {"label": "fire", "ground_truth": "normal"},
            confidence=0.95,
        )
    )

    assert trace.trajectory is not None
    reward = trace.trajectory.reward
    assert reward.risk_penalty == 1.0
    assert reward.total < 0


def test_alarm_navigation_demo_stops_for_obstacle(capsys) -> None:
    main(["agent", "demo", "--scenario", "alarm-obstacle", "--json"])

    payload = json.loads(capsys.readouterr().out)
    traces = payload["trajectories"]
    assert [trace["trajectory"]["selected_skill"] for trace in traces] == [
        "alarm.set",
        "alarm.remind",
        "mobility.forward",
        "mobility.stop",
    ]
    assert traces[-1]["action"]["stopped"] is True


def test_sedentary_demo_records_cooldown(capsys) -> None:
    main(["agent", "demo", "--scenario", "sedentary", "--json"])

    payload = json.loads(capsys.readouterr().out)
    traces = payload["trajectories"]
    assert traces[0]["trajectory"]["selected_skill"] == "health.sedentary"
    repeated = traces[1]
    assert repeated["trajectory"]["selected_skill"] == "health.sedentary"
    assert repeated["final_status"] == "suppressed"
    assert repeated["outcome_type"] == "suppressed"
    assert repeated["reason"] == "cooldown_or_dedup"
    assert repeated["trajectory"]["success"] is True
    assert repeated["trajectory"]["reward"]["total"] == 0.0
