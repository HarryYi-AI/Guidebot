from guidebot.events import Event
from guidebot.intent import Intent, IntentType
from guidebot.runtime import GuidebotRuntime
from guidebot.runtime_skills import RuntimeSkill, RuntimeSkillRegistry
from guidebot.scheduler import Scheduler


def test_scheduler_uses_runtime_skill_registry() -> None:
    registry = RuntimeSkillRegistry(
        (
            RuntimeSkill(
                "custom.chat",
                IntentType.CHAT,
                "voice_chat",
                "custom_chat",
                "custom test chat",
            ),
        )
    )
    scheduler = Scheduler(skill_registry=registry)

    task = scheduler.schedule(Intent(IntentType.CHAT, Event("user.text", "test", {"text": "hi"})))

    assert task is not None
    assert task.skill_id == "custom.chat"
    assert task.target_module == "voice_chat"
    assert task.action == "custom_chat"


def test_generic_scene_abnormal_routes_to_scene_alert_skill() -> None:
    runtime = GuidebotRuntime()

    trace = runtime.ingest(
        Event(
            "scene.detected",
            "camera",
            {
                "label": "abnormal",
                "abnormal": True,
                "risk_level": "medium",
                "feedback": "检测到门窗异常，请检查。",
            },
        )
    )

    assert trace.intent.intent_type is IntentType.SAFETY_SCENE_ALERT
    assert trace.task is not None
    assert trace.task.skill_id == "scene.abnormal_alert"
    assert trace.action is not None
    assert trace.action["message"] == "检测到门窗异常，请检查。"


def test_alarm_reminder_uses_runtime_skill_id() -> None:
    runtime = GuidebotRuntime()

    trace = runtime.ingest(Event("alarm.triggered", "alarm", {"text": "起床了"}))

    assert trace.task is not None
    assert trace.task.skill_id == "alarm.remind"
