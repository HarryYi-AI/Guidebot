from datetime import timedelta

from guidebot.events import Event
from guidebot.intent import Intent, IntentType
from guidebot.models import utc_now
from guidebot.scheduler import Scheduler


def _intent(kind: IntentType, priority: int) -> Intent:
    return Intent(kind, Event("test", "test"), priority=priority)


def test_fire_preempts_chat() -> None:
    scheduler = Scheduler()
    chat = scheduler.schedule(_intent(IntentType.CHAT, 10))
    fire = scheduler.schedule(_intent(IntentType.SAFETY_FIRE_ALERT, 100))

    assert chat is not None
    assert fire is not None
    assert scheduler.next_task() is fire
    assert Scheduler.can_preempt(fire)


def test_sedentary_cooldown_blocks_repeated_reminder() -> None:
    now = utc_now()
    scheduler = Scheduler()
    first = scheduler.schedule(_intent(IntentType.HEALTH_SEDENTARY, 50), now)
    second = scheduler.schedule(_intent(IntentType.HEALTH_SEDENTARY, 50), now + timedelta(minutes=5))

    assert first is not None
    assert second is None
