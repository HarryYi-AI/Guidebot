from guidebot.events import Event
from guidebot.intent import IntentType
from guidebot.intent_analyzer import IntentAnalyzer


def test_scene_fire_routes_to_safety_alert() -> None:
    intent = IntentAnalyzer().analyze(
        Event("scene.detected", "test", {"label": "fire", "summary": "明火"})
    )

    assert intent.intent_type is IntentType.SAFETY_FIRE_ALERT
    assert intent.priority == 100


def test_ultrasonic_obstacle_routes_to_stop() -> None:
    intent = IntentAnalyzer().analyze(
        Event("ultrasonic.obstacle", "test", {"obstacle": True})
    )

    assert intent.intent_type is IntentType.MOBILITY_STOP
    assert intent.priority == 100


def test_user_text_sets_alarm() -> None:
    intent = IntentAnalyzer().analyze(Event("user.text", "test", {"text": "明早七点叫我"}))

    assert intent.intent_type is IntentType.SET_ALARM
    assert intent.requires_confirmation is True


def test_unknown_text_routes_to_chat() -> None:
    intent = IntentAnalyzer().analyze(Event("user.text", "test", {"text": "讲个故事"}))

    assert intent.intent_type is IntentType.CHAT


def test_climate_sensor_routes_to_comfort_skill() -> None:
    intent = IntentAnalyzer().analyze(
        Event("climate.detected", "test", {"temperature_c": 29.2, "humidity": 76})
    )

    assert intent.intent_type is IntentType.CLIMATE_COMFORT
    assert intent.priority == 40


def test_ac_left_on_routes_to_alert() -> None:
    intent = IntentAnalyzer().analyze(
        Event("climate.detected", "test", {"ac_on": True, "occupied": False})
    )

    assert intent.intent_type is IntentType.AC_LEFT_ON_ALERT
    assert intent.priority == 70
