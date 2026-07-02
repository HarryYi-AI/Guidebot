from __future__ import annotations

from guidebot.failure_attribution import FailureAttributor, FailureType
from guidebot.models import Action, ActionKind, Decision, Event, Trajectory
from guidebot.observation import Observation
from guidebot.reflection import EnvironmentFeedback


def _trajectory(*, rejected: bool = False) -> Trajectory:
    action = Action(ActionKind.SET_HVAC, {"target_c": 24}, "test")
    return Trajectory(
        Event("sensor.reading", None),
        Decision((action,)),
        () if rejected else (action,),
        (action,) if rejected else (),
    )


def test_only_skill_error_and_high_confidence_preference_shift_evolve() -> None:
    observation = Observation(30, 50, event_kind="temperature")
    failed = EnvironmentFeedback(-1, False, observation)
    attributor = FailureAttributor()

    skill_error = attributor.attribute(_trajectory(), device_feedback=failed)
    preference = attributor.attribute(
        _trajectory(),
        device_feedback=failed,
        user_feedback={"preference_shift": True, "confidence": 0.9},
    )

    assert skill_error.failure_type is FailureType.SKILL_ERROR
    assert skill_error.should_evolve_skill
    assert preference.failure_type is FailureType.USER_PREFERENCE_SHIFT
    assert preference.should_evolve_skill


def test_safety_rejection_and_execution_lapse_do_not_evolve() -> None:
    observation = Observation(30, 50, event_kind="temperature")
    attributor = FailureAttributor()

    safety = attributor.attribute(_trajectory(rejected=True))
    lapse = attributor.attribute(
        _trajectory(),
        device_feedback=EnvironmentFeedback(
            -1,
            False,
            observation,
            details={"execution_lapse": True},
        ),
    )

    assert safety.failure_type is FailureType.SAFETY_REJECTION
    assert lapse.failure_type is FailureType.EXECUTION_LAPSE
    assert not safety.should_evolve_skill
    assert not lapse.should_evolve_skill

