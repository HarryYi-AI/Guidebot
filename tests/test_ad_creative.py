from guidebot.events import Event
from guidebot.intent import IntentType
from guidebot.runtime import GuidebotRuntime


def test_ad_creative_is_routed_to_reviewable_skill() -> None:
    runtime = GuidebotRuntime()

    trace = runtime.ingest(
        Event(
            "user.text",
            "test",
            {
                "text": "帮我生成一张 Guidebot 广告海报",
                "product": "Guidebot",
                "audience": "年轻家庭",
            },
        )
    )

    assert trace.intent.intent_type is IntentType.GENERATE_AD_CREATIVE
    assert trace.task is not None
    assert trace.task.skill_id == "ad.generate_brief"
    assert trace.action is not None
    assert trace.action["requires_human_review"] is True
    assert trace.action["published"] is False
