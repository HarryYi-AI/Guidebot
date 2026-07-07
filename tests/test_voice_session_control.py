from guidebot.voice.providers import RealtimeEvent
from guidebot.voice.session_control import WakeSleepController


def _transcript(text: str) -> RealtimeEvent:
    return RealtimeEvent(
        "conversation.item.input_audio_transcription.completed",
        text,
    )


def test_wake_controller_suppresses_inactive_transcripts_until_wake() -> None:
    controller = WakeSleepController(require_wake=True, wake_phrases=("你好小云",))

    ignored = controller.update(_transcript("窗外有一点噪声"))

    assert controller.active is False
    assert ignored.emit_event is False
    assert ignored.interrupt_response is True

    awakened = controller.update(_transcript("你好 小云"))

    assert controller.active is True
    assert awakened.emit_event is True
    assert awakened.generated_events[0].type == "guidebot.session.awake"


def test_wake_controller_enters_sleep_on_end_phrase() -> None:
    controller = WakeSleepController(
        require_wake=True,
        active=True,
        sleep_phrases=("今天聊到这里",),
    )

    decision = controller.update(_transcript("今天聊到这里吧"))

    assert controller.active is False
    assert decision.interrupt_response is True
    assert decision.stop_playback is True
    assert decision.generated_events[0].type == "guidebot.session.sleep"


def test_wake_controller_defaults_to_active_without_required_wake() -> None:
    controller = WakeSleepController(require_wake=False)

    assert controller.allow_playback is True
    assert controller.update(RealtimeEvent("response.created")).emit_event is True
