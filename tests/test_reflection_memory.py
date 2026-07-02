from __future__ import annotations

from datetime import datetime, timedelta, timezone

from guidebot.memory import Experience, MemoryStream
from guidebot.models import Action, ActionKind, Decision, Event, Trajectory
from guidebot.observation import Observation
from guidebot.reflection import Critique, EnvironmentFeedback, ReflectionEngine


def _trajectory(*, rejected: bool = False) -> Trajectory:
    action = Action(ActionKind.SET_HVAC, {"target_c": 25}, "temperature")
    return Trajectory(
        Event("sensor.reading", None),
        Decision((action,)),
        () if rejected else (action,),
        (action,) if rejected else (),
    )


def test_reflection_emits_structured_safety_critique() -> None:
    observation = Observation(30, 50, 0, 0)
    feedback = EnvironmentFeedback(-1, False, observation)

    critique = ReflectionEngine().reflect(observation, _trajectory(rejected=True), feedback)

    assert critique.failure_mode == "safety_rejection"
    assert critique.severity == 1.0
    assert critique.entropy < 0.1


def _experience(observation: Observation, timestamp: datetime, mode: str = "none") -> Experience:
    failed = mode != "none"
    critique = Critique("cause", mode, 0.8 if failed else 0, "fix", 0.9, 0.1)
    feedback = EnvironmentFeedback(-1 if failed else 1, not failed, observation)
    return Experience(observation, "skill", Decision(), feedback, critique, timestamp)


def test_memory_retrieval_combines_similarity_and_time_decay() -> None:
    now = datetime(2026, 1, 2, tzinfo=timezone.utc)
    query = Observation(30, 50, 0, 0)
    old_match = _experience(query, now - timedelta(days=1))
    recent_match = _experience(query, now - timedelta(seconds=1))
    memory = MemoryStream(decay_lambda=1 / 86_400)
    memory.add(old_match)
    memory.add(recent_match)

    results = memory.retrieve(query, now=now)

    assert results[0].experience is recent_match
    assert results[0].similarity == 1.0
    assert results[0].decay > results[1].decay


def test_memory_filters_failure_modes() -> None:
    now = datetime.now(timezone.utc)
    memory = MemoryStream()
    memory.add(_experience(Observation(), now, "overheat"))
    memory.add(_experience(Observation(), now, "none"))

    assert len(memory.failures()) == 1
    assert len(memory.failures("overheat")) == 1
    assert memory.failures("unknown") == ()

